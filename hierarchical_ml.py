"""
Two-stage hierarchical DDoS detector (shared by the Ryu controller and the trainer).

Stage 1 - volumetric attacks (per-interval, instantaneous)
    Input : the 5 flow-table features of ONE monitoring interval
            [FER, SIGS, FSR, PLV, BLV]
    Output: volumetric / not volumetric

Stage 2 - low-rate attacks (temporal, sliding window)
    Input : the last WINDOW_SIZE (=10) intervals that Stage 1 passed as
            non-volumetric, summarised as mean/std/min/max/slope of every
            feature (5 features x 5 statistics = 25 values)
    Output: low-rate / normal
    The window advances by one interval at a time, so after the first
    WINDOW_SIZE intervals Stage 2 produces a decision at every interval.

Hierarchy rule
    1. Stage 1 runs on every interval.
    2. If Stage 1 says "volumetric", the interval is reported as volumetric and
       the Stage 2 window is cleared (the window must only contain intervals
       that Stage 1 considered non-volumetric - exactly the data Stage 2 is
       trained on).
    3. Otherwise the interval is pushed into the window. Once the window holds
       WINDOW_SIZE intervals, Stage 2 classifies it.

Labels: 0 = normal, 1 = volumetric, 2 = low-rate.
"""
import json
import os
from collections import deque

import joblib
import numpy as np

NORMAL = 0
VOLUMETRIC = 1
LOW_RATE = 2
LABEL_NAMES = {NORMAL: "normal", VOLUMETRIC: "volumetric", LOW_RATE: "low_rate"}

# FER = Flow Entry Rate, SIGS = Source-IP Generation Speed, FSR = Flow Success
# Ratio, PLV = Packet Load Variance, BLV = Byte Load Variance
FEATURE_NAMES = ["FER", "SIGS", "FSR", "PLV", "BLV"]

WINDOW_SIZE = 10
WINDOW_STATS = ["mean", "std", "min", "max", "slope"]

STAGE1_FILE = "stage1_volumetric.joblib"
STAGE2_FILE = "stage2_lowrate.joblib"
META_FILE = "metadata.json"


# --------------------------------------------------------------------------
# Window features (Stage 2)
# --------------------------------------------------------------------------
def window_feature_names():
    """Names of the 25 Stage-2 inputs, in the order window_features() emits them."""
    return ["%s_%s" % (f, s) for s in WINDOW_STATS for f in FEATURE_NAMES]


def window_features(window):
    """
    Summarise a (T, 5) block of consecutive intervals into one Stage-2 vector.

    mean/std/min/max describe the level and spread of each feature over the
    window; slope is the least-squares trend per interval, which captures the
    slow, sustained drift that characterises low-rate attacks.
    """
    w = np.asarray(window, dtype=float)
    if w.ndim != 2 or w.shape[1] != len(FEATURE_NAMES) or w.shape[0] < 2:
        raise ValueError("window must have shape (T>=2, %d), got %s"
                         % (len(FEATURE_NAMES), w.shape))
    t = np.arange(w.shape[0], dtype=float)
    t -= t.mean()
    slope = (t[:, None] * w).sum(axis=0) / float((t * t).sum())
    return np.concatenate([w.mean(axis=0), w.std(axis=0),
                           w.min(axis=0), w.max(axis=0), slope])


def build_windows(rows, window_size=WINDOW_SIZE):
    """All stride-1 windows of a chronologically ordered (N, 5) array."""
    rows = np.asarray(rows, dtype=float)
    return [window_features(rows[i:i + window_size])
            for i in range(0, len(rows) - window_size + 1)]


class SlidingWindow(object):
    """Fixed-length FIFO of the most recent intervals' feature vectors."""

    def __init__(self, size=WINDOW_SIZE):
        self.size = size
        self._buf = deque(maxlen=size)

    def push(self, features):
        self._buf.append([float(v) for v in features])

    def clear(self):
        self._buf.clear()

    def __len__(self):
        return len(self._buf)

    @property
    def full(self):
        return len(self._buf) == self.size

    def features(self):
        return window_features(list(self._buf))


# --------------------------------------------------------------------------
# Hierarchical detector
# --------------------------------------------------------------------------
class Verdict(object):
    """
    label       : NORMAL / VOLUMETRIC / LOW_RATE (final decision)
    stage1      : Stage-1 output (0/1)
    stage2      : Stage-2 output (0/1), or None when Stage 2 did not run
    window_fill : intervals currently in the Stage-2 window
    warming_up  : True while Stage 2 cannot decide yet (window not full)
    """
    __slots__ = ("label", "stage1", "stage2", "window_fill", "warming_up")

    def __init__(self, label, stage1, stage2, window_fill, warming_up):
        self.label = label
        self.stage1 = stage1
        self.stage2 = stage2
        self.window_fill = window_fill
        self.warming_up = warming_up

    def __repr__(self):
        return ("Verdict(%s, stage1=%s, stage2=%s, window=%d, warming_up=%s)"
                % (LABEL_NAMES[self.label], self.stage1, self.stage2,
                   self.window_fill, self.warming_up))


class HierarchicalDetector(object):
    def __init__(self, stage1_model, stage2_model, window_size=WINDOW_SIZE):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.window_size = int(window_size)
        self._windows = {}          # one independent window per switch / stream

    @classmethod
    def load(cls, model_dir):
        """Load the models written by train_hierarchical.py."""
        p1 = os.path.join(model_dir, STAGE1_FILE)
        p2 = os.path.join(model_dir, STAGE2_FILE)
        for p in (p1, p2):
            if not os.path.isfile(p):
                raise IOError("model file not found: %s  (run train_hierarchical.py first)" % p)
        window_size = WINDOW_SIZE
        meta_path = os.path.join(model_dir, META_FILE)
        if os.path.isfile(meta_path):
            with open(meta_path) as fh:
                window_size = json.load(fh).get("window_size", WINDOW_SIZE)
        return cls(joblib.load(p1), joblib.load(p2), window_size)

    def reset(self, key=None):
        """Forget the window of one stream (or of all streams)."""
        if key is None:
            self._windows.clear()
        else:
            self._windows.pop(key, None)

    def step(self, key, features):
        """Process one monitoring interval of stream `key` (e.g. a switch id)."""
        x = np.nan_to_num(np.asarray(features, dtype=float), posinf=0.0, neginf=0.0)
        win = self._windows.get(key)
        if win is None:
            win = self._windows[key] = SlidingWindow(self.window_size)

        # ---- Stage 1: volumetric, one interval ----
        s1 = int(self.stage1.predict(x.reshape(1, -1))[0])
        if s1 == 1:
            win.clear()
            return Verdict(VOLUMETRIC, s1, None, 0, False)

        # ---- Stage 2: low-rate, sliding window of Stage-1-passed intervals ----
        win.push(x)
        if not win.full:
            return Verdict(NORMAL, s1, None, len(win), True)
        s2 = int(self.stage2.predict(win.features().reshape(1, -1))[0])
        return Verdict(LOW_RATE if s2 == 1 else NORMAL, s1, s2, len(win), False)


def save_models(model_dir, stage1, stage2, window_size, extra=None):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(stage1, os.path.join(model_dir, STAGE1_FILE))
    joblib.dump(stage2, os.path.join(model_dir, STAGE2_FILE))
    meta = {
        "window_size": int(window_size),
        "feature_names": FEATURE_NAMES,
        "window_feature_names": window_feature_names(),
        "labels": {str(k): v for k, v in LABEL_NAMES.items()},
    }
    meta.update(extra or {})
    with open(os.path.join(model_dir, META_FILE), "w") as fh:
        json.dump(meta, fh, indent=2)
