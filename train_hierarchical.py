#!/usr/bin/env python3
"""
Train the two stages of the hierarchical detector from the CSV files written by
ddos_hierarchical_controller.py in data-collection mode (APP_TYPE = 0).

Input rows:  time, FER, SIGS, FSR, PLV, BLV, type      (switch_<dpid>_data.csv)
         or: FER, SIGS, FSR, PLV, BLV, type            (result.csv)
type: 0 = normal, 1 = volumetric attack, 2 = low-rate attack

Each file argument is PATH or PATH:LABEL. PATH:LABEL forces every row of the
file to LABEL (useful for older files whose type column is 0/1 only).

Stage 1  volumetric (1) against everything else (normal + low-rate), per interval.
Stage 2  low-rate (1) against normal (0), on stride-1 windows of 10 consecutive
         intervals, built only from rows of one uninterrupted run.

Evaluation is leak-free: by default the LAST (1 - train_frac) of every run is
held out, and windows never cross the train/test boundary. For a stricter test
use separate runs: --test-files run_a.csv run_b.csv  (then all rows of the
positional files are used for training).

Examples
    python train_hierarchical.py switch_1_data.csv
    python train_hierarchical.py train/*.csv --test-files test/*.csv
    python train_hierarchical.py switch_1_data.csv --stage2 knn --k 7 --weights distance
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import csv

from hierarchical_ml import (FEATURE_NAMES, LABEL_NAMES, LOW_RATE, NORMAL, VOLUMETRIC,
                             WINDOW_SIZE, HierarchicalDetector, build_windows,
                             save_models)

TIME_FMT = "%m/%d/%Y, %H:%M:%S"


# --------------------------------------------------------------------------
# Reading data
# --------------------------------------------------------------------------
class Segment(object):
    """A run of consecutive intervals with one label."""

    def __init__(self, label, source):
        self.label = label
        self.source = source
        self.rows = []


def split_label(arg):
    """'file.csv:2' -> ('file.csv', 2);  'file.csv' -> ('file.csv', None)."""
    head, sep, tail = arg.rpartition(":")
    if sep and tail in ("0", "1", "2"):
        return head, int(tail)
    return arg, None


def parse_row(row):
    """Return (time or None, [5 features], label) or None for a header/bad row."""
    if len(row) == 7:
        raw_time, vals, lab = row[0], row[1:6], row[6]
    elif len(row) == 6:
        raw_time, vals, lab = None, row[0:5], row[5]
    else:
        return None
    try:
        feats = [float(v) for v in vals]
        label = int(float(lab))
    except ValueError:
        return None                     # header line or garbage
    if label not in LABEL_NAMES or not all(np.isfinite(feats)):
        return None
    t = None
    if raw_time is not None:
        try:
            t = datetime.strptime(raw_time.strip(), TIME_FMT)
        except ValueError:
            t = None
    return t, feats, label


def read_segments(path, forced_label, interval, gap_factor):
    """
    Split a CSV into segments of consecutive same-label rows. A new segment also
    starts at a header line (a new run appended to the same file) or when the
    timestamps jump by more than gap_factor * interval seconds, so a window can
    never span two separate runs.
    """
    segments, cur, last_t = [], None, None
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            if row[0].strip().lower() == "time" or row[0].strip().upper() == "FER":
                cur, last_t = None, None
                continue
            parsed = parse_row(row)
            if parsed is None:
                continue
            t, feats, label = parsed
            if forced_label is not None:
                label = forced_label
            new = cur is None or label != cur.label
            if (not new and t is not None and last_t is not None
                    and (t - last_t).total_seconds() > gap_factor * interval):
                new = True
            if new:
                cur = Segment(label, path)
                segments.append(cur)
            cur.rows.append(feats)
            last_t = t
    return segments


def load_files(args_list, interval, gap_factor):
    segs = []
    for a in args_list:
        path, forced = split_label(a)
        if not os.path.isfile(path):
            sys.exit("file not found: %s" % path)
        segs.extend(read_segments(path, forced, interval, gap_factor))
    return segs


def make_parts(segments, skip_first, train_frac):
    """
    Drop the warm-up rows of every segment, then split it chronologically.
    Returns [(label, ndarray)] for train and for test.
    """
    train, test = [], []
    for s in segments:
        rows = s.rows[skip_first:]
        if not rows:
            continue
        n_tr = int(round(len(rows) * train_frac))
        if n_tr > 0:
            train.append((s.label, np.array(rows[:n_tr], dtype=float)))
        if len(rows) - n_tr > 0:
            test.append((s.label, np.array(rows[n_tr:], dtype=float)))
    return train, test


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(kind, n_samples, k, weights, seed):
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                     random_state=seed, n_jobs=1)
    elif kind == "dt":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=max(1, min(k, n_samples)), weights=weights)
    elif kind == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def report(y_true, y_pred, labels, names):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                zero_division=0, output_dict=True)
    text = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                 zero_division=0, digits=4)
    return cm, rep, text


def print_cm(cm, names):
    label_w = max(len(n) for n in names) + 7
    col_w = max(len(n) for n in names) + 8
    print(" " * label_w + "".join("%*s" % (col_w, "pred_" + n) for n in names))
    for n, r in zip(names, cm):
        print("%-*s" % (label_w, "true_" + n) + "".join("%*d" % (col_w, v) for v in r))


def evaluate_end_to_end(detector, test_parts):
    """Run the real hierarchical logic (window reset, warm-up) over the test runs."""
    y_true, y_pred, warm = [], [], []
    for i, (label, X) in enumerate(test_parts):
        key = ("test", i)
        for row in X:
            v = detector.step(key, row)
            y_true.append(label)
            y_pred.append(v.label)
            warm.append(v.warming_up)
        detector.reset(key)
    return np.array(y_true), np.array(y_pred), np.array(warm)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="training CSV files (PATH or PATH:LABEL)")
    ap.add_argument("--test-files", nargs="*", default=[],
                    help="CSV files used only for testing (separate runs)")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--window", type=int, default=WINDOW_SIZE,
                    help="Stage-2 window length in intervals (default %(default)s)")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="monitoring interval used when the data was collected (s)")
    ap.add_argument("--gap-factor", type=float, default=2.5,
                    help="timestamp gap (x interval) that starts a new run")
    ap.add_argument("--skip-first", type=int, default=1,
                    help="warm-up rows dropped at the start of every run")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="share of each run used for training when --test-files is empty")
    ap.add_argument("--stage1", choices=["rf", "dt", "knn", "svm"], default="rf")
    ap.add_argument("--stage2", choices=["rf", "dt", "knn", "svm"], default="knn")
    ap.add_argument("--k", type=int, default=5, help="neighbours for knn")
    ap.add_argument("--weights", choices=["uniform", "distance"], default="distance",
                    help="knn weighting")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    train_segs = load_files(a.files, a.interval, a.gap_factor)
    if a.test_files:
        train_parts, _ = make_parts(train_segs, a.skip_first, 1.0)
        _, test_parts = make_parts(load_files(a.test_files, a.interval, a.gap_factor),
                                   a.skip_first, 0.0)
    else:
        train_parts, test_parts = make_parts(train_segs, a.skip_first, a.train_frac)

    counts = {n: sum(len(X) for l, X in train_parts if l == k) for k, n in LABEL_NAMES.items()}
    print("training rows per class:", counts)
    print("test rows per class    :", {n: sum(len(X) for l, X in test_parts if l == k)
                                       for k, n in LABEL_NAMES.items()})

    # ---------------- Stage 1: volumetric vs rest, per interval ----------------
    X1 = np.vstack([X for _, X in train_parts])
    y1 = np.concatenate([np.full(len(X), 1 if l == VOLUMETRIC else 0) for l, X in train_parts])
    if len(set(y1)) < 2:
        sys.exit("Stage 1 needs volumetric rows AND non-volumetric (normal/low-rate) rows.")
    stage1 = make_model(a.stage1, len(X1), a.k, a.weights, a.seed).fit(X1, y1)

    # ---------------- Stage 2: low-rate vs normal, windows ---------------------
    W, y2 = [], []
    for label, X in train_parts:
        if label == VOLUMETRIC:
            continue                    # Stage 2 never sees volumetric intervals
        wins = build_windows(X, a.window)
        W.extend(wins)
        y2.extend([1 if label == LOW_RATE else 0] * len(wins))
    W, y2 = np.array(W), np.array(y2)
    if len(W) == 0 or len(set(y2)) < 2:
        sys.exit("Stage 2 needs normal AND low-rate runs with at least %d consecutive "
                 "intervals each (after skipping %d warm-up rows)." % (a.window, a.skip_first))
    stage2 = make_model(a.stage2, len(W), a.k, a.weights, a.seed).fit(W, y2)
    print("stage 2 training windows: %d normal, %d low-rate" % ((y2 == 0).sum(), (y2 == 1).sum()))

    metrics = {"train_rows": counts}

    # ---------------- Evaluation ----------------
    if test_parts:
        Xt = np.vstack([X for _, X in test_parts])
        yt = np.concatenate([np.full(len(X), 1 if l == VOLUMETRIC else 0) for l, X in test_parts])
        cm, rep, txt = report(yt, stage1.predict(Xt), [0, 1], ["not_volumetric", "volumetric"])
        print("\n=== Stage 1 (per interval, volumetric vs rest) ===\n" + txt)
        print_cm(cm, ["not_volumetric", "volumetric"])
        metrics["stage1"] = {"report": rep, "confusion_matrix": cm.tolist()}

        Wt, yt2 = [], []
        for label, X in test_parts:
            if label == VOLUMETRIC:
                continue
            wins = build_windows(X, a.window)
            Wt.extend(wins)
            yt2.extend([1 if label == LOW_RATE else 0] * len(wins))
        if Wt and len(set(yt2)) == 2:
            cm, rep, txt = report(np.array(yt2), stage2.predict(np.array(Wt)),
                                  [0, 1], ["normal", "low_rate"])
            print("\n=== Stage 2 (sliding window of %d, low-rate vs normal) ===\n%s" % (a.window, txt))
            print_cm(cm, ["normal", "low_rate"])
            metrics["stage2"] = {"report": rep, "confusion_matrix": cm.tolist()}
        else:
            print("\n(Stage 2 not evaluated: the test data needs normal and low-rate runs of "
                  ">= %d intervals)" % a.window)

        det = HierarchicalDetector(stage1, stage2, a.window)
        names = [LABEL_NAMES[i] for i in (NORMAL, VOLUMETRIC, LOW_RATE)]
        y_true, y_pred, warm = evaluate_end_to_end(det, test_parts)
        cm, rep, txt = report(y_true, y_pred, [0, 1, 2], names)
        print("\n=== End to end (hierarchical, every interval, incl. Stage-2 warm-up) ===\n" + txt)
        print_cm(cm, names)
        metrics["end_to_end"] = {"report": rep, "confusion_matrix": cm.tolist(),
                                 "warmup_intervals": int(warm.sum())}
        steady = ~warm
        if steady.any():
            cm, rep, txt = report(y_true[steady], y_pred[steady], [0, 1, 2], names)
            print("\n=== End to end, steady state (warm-up intervals excluded: %d) ===\n%s"
                  % (int(warm.sum()), txt))
            print_cm(cm, names)
            metrics["end_to_end_steady_state"] = {"report": rep, "confusion_matrix": cm.tolist()}
    else:
        print("\nNo test data (train_frac = 1.0 and no --test-files): skipping evaluation.")

    save_models(a.model_dir, stage1, stage2, a.window,
                extra={"stage1_model": a.stage1, "stage2_model": a.stage2,
                       "knn_k": a.k, "knn_weights": a.weights,
                       "interval_seconds": a.interval, "skip_first": a.skip_first,
                       "trained_at": datetime.now().isoformat(timespec="seconds")})
    with open(os.path.join(a.model_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("\nsaved models + metadata.json + metrics.json to %s/" % a.model_dir)


if __name__ == "__main__":
    main()
