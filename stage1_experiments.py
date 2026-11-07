"""
stage1_experiments.py -- examiner-proofing experiments for the STAGE-1 (volumetric / peak-load) dataset.

    python3 -I stage1_experiments.py <csv without header: FER,SIGS,FSR,PLV,BLV,type>

Reproduces the thesis protocol first (75/25 random split, random_state=0, RAW features, soft vote NB+KNN5),
then tests what the examiners ask: seed variability, scaling, chronological protocols, de-duplicated split,
operating points, ablation / exact Shapley, simple-rule baselines, McNemar tests, inference time.
Writes stage1_results.json
"""
import sys, json, time
import numpy as np, pandas as pd
from math import factorial
from scipy.stats import binomtest, beta, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

PATH = sys.argv[1]
F = ['FER', 'SIGS', 'FSR', 'PLV', 'BLV']
OUT = {}
T0 = time.time()
log = lambda *a: print('[%5.0fs]' % (time.time() - T0), *a, flush=True)

df = pd.read_csv(PATH, header=None, names=F + ['type'])
X, y, N = df[F].values, df['type'].values.astype(int), len(df)
idle = (np.abs(X[:, [0, 1, 3, 4]]).sum(1) == 0)
chg = np.flatnonzero(np.diff(y)) + 1
starts, ends = np.r_[0, chg], np.r_[chg, N]
BLOCKS = [(int(s), int(e)) for s, e in zip(starts, ends)]


def m1(yt, p):
    yb, pb = np.asarray(yt).astype(bool), np.asarray(p).astype(bool)
    tp, fp, fn, tn = int((pb & yb).sum()), int((pb & ~yb).sum()), int((~pb & yb).sum()), int((~pb & ~yb).sum())
    return dict(acc=(tp + tn) / len(yt) * 100, prec=tp / max(tp + fp, 1) * 100, rec=tp / max(tp + fn, 1) * 100,
                f1=2 * tp / max(2 * tp + fp + fn, 1) * 100, fpr=fp / max(fp + tn, 1) * 100, tn=tn, fp=fp, fn=fn, tp=tp)


def cp(k, n, a=.05):
    """exact Clopper-Pearson interval in %"""
    lo = 0. if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1. if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [lo * 100, hi * 100]


def vote_proba(Xa, ya, Xb, scaled=False, knn_k=5):
    if scaled:
        s = MinMaxScaler().fit(Xa); Xa, Xb = s.transform(Xa), s.transform(Xb)
    nb = GaussianNB().fit(Xa, ya); kn = KNeighborsClassifier(knn_k, n_jobs=-1).fit(Xa, ya)
    return nb.predict_proba(Xb)[:, 1], kn.predict_proba(Xb)[:, 1]


# ------------------------------------------------------------------ 0. dataset facts
g = df.groupby(F)['type'].agg(['sum', 'count'])
OUT['dataset'] = dict(
    rows=N, normal=int((y == 0).sum()), attack=int((y == 1).sum()), attack_pct=float(y.mean() * 100),
    label_blocks=[dict(label=int(y[s]), start=s, end=e, rows=e - s, idle_pct=float(idle[s:e].mean() * 100)) for s, e in BLOCKS],
    idle_windows=int(idle.sum()), idle_attack=int((idle & (y == 1)).sum()), idle_normal=int((idle & (y == 0)).sum()),
    distinct_vectors=int(len(g)), duplicate_rows=int(N - len(g)), duplicate_pct=float((N - len(g)) / N * 100),
    mixed_label_vectors=int(((g['sum'] > 0) & (g['sum'] < g['count'])).sum()),
    rows_in_mixed_vectors=int(g.loc[(g['sum'] > 0) & (g['sum'] < g['count']), 'count'].sum()),
    ceiling_acc_pct=float(np.maximum(g['sum'], g['count'] - g['sum']).sum() / N * 100),
    raw_ranges={f: [float(df[f].min()), float(df[f].median()), float(df[f].max())] for f in F},
    per_class_active_quartiles={str(lab): df[(y == lab) & ~idle][F].quantile([.25, .5, .75]).round(3).to_dict() for lab in (0, 1)},
    per_block_medians=[{f: float(df[f].iloc[s:e].median()) for f in F} for s, e in BLOCKS])
log('dataset', {k: v for k, v in OUT['dataset'].items() if not isinstance(v, (dict, list))})

# ------------------------------------------------------------------ 1. thesis protocol (75/25, rs=0, raw)
idx = np.arange(N)
itr, ite = train_test_split(idx, random_state=0, test_size=.25)
Xtr, ytr, Xte, yte = X[itr], y[itr], X[ite], y[ite]
pn, pk = vote_proba(Xtr, ytr, Xte)
vote_pred = ((pn + pk) / 2 > .5).astype(int)
thesis = {'vote': m1(yte, vote_pred), 'knn5': m1(yte, pk > .5), 'nb': m1(yte, pn > .5)}
ref = VotingClassifier([('a', GaussianNB()), ('b', KNeighborsClassifier(5, n_jobs=-1))], voting='soft').fit(Xtr, ytr).predict(Xte)
thesis['vote_equals_sklearn_VotingClassifier'] = bool((ref == vote_pred).all())
thesis['clopper_pearson_95'] = dict(acc=cp(int((vote_pred == yte).sum()), len(yte)), fpr=cp(thesis['vote']['fp'], thesis['vote']['fp'] + thesis['vote']['tn']),
                                    miss_rate=cp(thesis['vote']['fn'], thesis['vote']['fn'] + thesis['vote']['tp']))
OUT['thesis_protocol'] = thesis
log('thesis protocol vote', {k: round(v, 2) for k, v in thesis['vote'].items() if isinstance(v, float)})

# where the errors are
blk = np.digitize(ite, [e for s, e in BLOCKS[:-1]])
fp = (vote_pred == 1) & (yte == 0); fn = (vote_pred == 0) & (yte == 1); idl = idle[ite]
OUT['error_anatomy'] = dict(
    fp=int(fp.sum()), fn=int(fn.sum()), fp_idle_pct=float((fp & idl).sum() / max(fp.sum(), 1) * 100),
    fp_by_block={int(b): int((fp & (blk == b)).sum()) for b in np.unique(blk)},
    fpr_idle_normal=float(fp[(yte == 0) & idl].mean() * 100), fpr_active_normal=float(fp[(yte == 0) & ~idl].mean() * 100),
    acc_idle=float((vote_pred[idl] == yte[idl]).mean() * 100), acc_active=float((vote_pred[~idl] == yte[~idl]).mean() * 100),
    idle_share_of_test=float(idl.mean() * 100), attack_share_idle=float(yte[idl].mean() * 100), attack_share_active=float(yte[~idl].mean() * 100),
    active_only_vote=m1(yte[~idl], vote_pred[~idl]),
    fp_fsr_zero_pct=float((Xte[fp, 2] == 0).mean() * 100))
seen = set(map(tuple, np.round(Xtr, 9))); inside = np.array([tuple(r) in seen for r in np.round(Xte, 9)])
OUT['leakage'] = dict(test_vectors_seen_in_train_pct=float(inside.mean() * 100), acc_seen=float((vote_pred[inside] == yte[inside]).mean() * 100),
                      acc_unseen=float((vote_pred[~inside] == yte[~inside]).mean() * 100), n_unseen=int((~inside).sum()))
log('error anatomy', OUT['error_anatomy']['fp_idle_pct'], OUT['leakage'])

# ------------------------------------------------------------------ 2. scaling: raw vs MinMax (fit on train)
pns, pks = vote_proba(Xtr, ytr, Xte, scaled=True)
OUT['scaling'] = {'vote_raw': thesis['vote'], 'vote_minmax': m1(yte, (pns + pks) / 2 > .5), 'knn5_raw': thesis['knn5'], 'knn5_minmax': m1(yte, pks > .5),
                  'nb_raw': thesis['nb'], 'nb_minmax': m1(yte, pns > .5)}
log('scaling done', {k: round(v['acc'], 2) for k, v in OUT['scaling'].items()})

# ------------------------------------------------------------------ 3. seed variability (10 random 75/25 splits)
rows = []
for s in range(10):
    a, b = train_test_split(idx, random_state=s, test_size=.25)
    n_, k_ = vote_proba(X[a], y[a], X[b])
    d = DecisionTreeClassifier(random_state=0).fit(X[a], y[a]).predict(X[b])
    rows.append(dict(seed=s, vote=m1(y[b], (n_ + k_) / 2 > .5), knn5=m1(y[b], k_ > .5), nb=m1(y[b], n_ > .5), dt=m1(y[b], d)))
    log('seed', s, round(rows[-1]['vote']['acc'], 2))
OUT['seed_variability'] = {mdl: {m: dict(mean=float(np.mean([r[mdl][m] for r in rows])), std=float(np.std([r[mdl][m] for r in rows])),
                                         min=float(np.min([r[mdl][m] for r in rows])), max=float(np.max([r[mdl][m] for r in rows])))
                                 for m in ('acc', 'prec', 'rec', 'f1', 'fpr')} for mdl in ('vote', 'knn5', 'nb', 'dt')}
OUT['seed_variability']['vote_acc_per_seed'] = [r['vote']['acc'] for r in rows]

# ------------------------------------------------------------------ 4. baselines on the thesis split + McNemar
preds = {'Vote NB+KNN5 (thesis)': vote_pred, 'KNN k=5': (pk > .5).astype(int), 'Naive Bayes': (pn > .5).astype(int)}
base = {}
def add(name, p):
    preds[name] = np.asarray(p).astype(int)
add('Decision tree', DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).predict(Xte))
add('Random forest (100)', RandomForestClassifier(100, n_jobs=2, random_state=0).fit(Xtr, ytr).predict(Xte))
add('Extra trees (100)', ExtraTreesClassifier(100, n_jobs=2, random_state=0).fit(Xtr, ytr).predict(Xte))
add('Gradient boosting (hist)', HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr).predict(Xte))
sc = MinMaxScaler().fit(Xtr)
add('Logistic regression (MinMax)', LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte)))
add('KNN k=5 (MinMax)', pks > .5)
add('Vote NB+KNN5 (MinMax)', (pns + pks) / 2 > .5)
for d_ in (1, 2, 3):
    add(f'Decision tree depth {d_}', DecisionTreeClassifier(max_depth=d_, random_state=0).fit(Xtr, ytr).predict(Xte))
add('Rule: FSR < 0.5', (Xte[:, 2] < .5).astype(int))
add('Rule: FER >= 1', (Xte[:, 0] >= 1).astype(int))
base = {k: m1(yte, v) for k, v in preds.items()}
OUT['baselines'] = base


def mcnemar(a, b):
    ca, cb = preds[a] == yte, preds[b] == yte
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return dict(vote_only_correct=n01, other_only_correct=n10, p_value=float(binomtest(n01, n01 + n10, .5).pvalue) if n01 + n10 else 1.0)


OUT['mcnemar_vs_vote'] = {k: mcnemar('Vote NB+KNN5 (thesis)', k) for k in preds if k != 'Vote NB+KNN5 (thesis)'}
log('baselines', {k: round(v['acc'], 2) for k, v in base.items()})

# ------------------------------------------------------------------ 5. operating points of the vote (no re-training)
pv = (pn + pk) / 2
OUT['curves'] = dict(
    roc_auc_vote=float(roc_auc_score(yte, pv)), pr_auc_vote=float(average_precision_score(yte, pv)),
    roc_auc_knn=float(roc_auc_score(yte, pk)), pr_auc_knn=float(average_precision_score(yte, pk)),
    roc_auc_nb=float(roc_auc_score(yte, pn)), pr_auc_nb=float(average_precision_score(yte, pn)),
    nb_prob_extremes_pct=float(np.mean((pn < .01) | (pn > .99)) * 100), knn_prob_extremes_pct=float(np.mean((pk < .01) | (pk > .99)) * 100))
OUT['threshold_sweep'] = {f't={t}': m1(yte, pv > t) for t in (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)}
OUT['weight_sweep'] = {f'w_nb={w}': m1(yte, (w * pn + (1 - w) * pk) > .5) for w in (0, .25, .5, .75, 1)}
cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=3).fit(Xtr, ytr); pc = cal.predict_proba(Xte)[:, 1]
OUT['vote_calibrated_nb'] = m1(yte, (pc + pk) / 2 > .5)
log('curves', OUT['curves']['roc_auc_vote'], OUT['curves']['pr_auc_vote'])

# ------------------------------------------------------------------ 6. de-duplicated split (every distinct vector lives on one side only)
gid = df.groupby(F).ngroup().values
rng = np.random.default_rng(0); gperm = rng.permutation(gid.max() + 1)
te_g = np.zeros(gid.max() + 1, bool); te_g[gperm[:int(.25 * len(gperm))]] = True
mte = te_g[gid]; a, b = np.flatnonzero(~mte), np.flatnonzero(mte)
n_, k_ = vote_proba(X[a], y[a], X[b])
OUT['dedup_split'] = dict(train=int(len(a)), test=int(len(b)), test_attack_pct=float(y[b].mean() * 100),
                          vote=m1(y[b], (n_ + k_) / 2 > .5), knn5=m1(y[b], k_ > .5), nb=m1(y[b], n_ > .5),
                          dt=m1(y[b], DecisionTreeClassifier(random_state=0).fit(X[a], y[a]).predict(X[b])),
                          rf=m1(y[b], RandomForestClassifier(100, n_jobs=2, random_state=0).fit(X[a], y[a]).predict(X[b])))
log('dedup split', {k: round(v['acc'], 2) for k, v in OUT['dedup_split'].items() if isinstance(v, dict)})

# ------------------------------------------------------------------ 7. chronological protocols
def chrono_eval(tr_idx, te_idx):
    res = {}
    for scaled in (False, True):
        n_, k_ = vote_proba(X[tr_idx], y[tr_idx], X[te_idx], scaled=scaled)
        tag = 'minmax' if scaled else 'raw'
        res[f'vote_{tag}'] = m1(y[te_idx], (n_ + k_) / 2 > .5); res[f'knn5_{tag}'] = m1(y[te_idx], k_ > .5)
        if not scaled: res['nb'] = m1(y[te_idx], n_ > .5)
    res['dt'] = m1(y[te_idx], DecisionTreeClassifier(random_state=0).fit(X[tr_idx], y[tr_idx]).predict(X[te_idx]))
    res['rf'] = m1(y[te_idx], RandomForestClassifier(100, n_jobs=2, random_state=0).fit(X[tr_idx], y[tr_idx]).predict(X[te_idx]))
    return res


tr_l, te_l = [], []
for s, e in BLOCKS:
    c = s + int(.7 * (e - s)); tr_l += range(s, c); te_l += range(c, e)
tr_l, te_l = np.array(tr_l), np.array(te_l)
OUT['chrono_70_30'] = dict(train=int(len(tr_l)), test=int(len(te_l)), test_attack_pct=float(y[te_l].mean() * 100), results=chrono_eval(tr_l, te_l))
log('chrono 70/30', {k: round(v['acc'], 2) for k, v in OUT['chrono_70_30']['results'].items()})

# contiguous blocked 5-fold CV with a purge gap of 100 windows around every held-out chunk
PURGE = 100
fold = np.zeros(N, int)
for s, e in BLOCKS:
    n = e - s; fold[s:e] = np.minimum((np.arange(n) * 5) // n, 4)
cvres = []
for f in range(5):
    te_i = np.flatnonzero(fold == f)
    near = np.zeros(N, bool)
    for s, e in BLOCKS:
        m = np.flatnonzero(fold[s:e] == f) + s
        if len(m): near[max(s, m.min() - PURGE):min(e, m.max() + PURGE + 1)] = True
    tr_i = np.flatnonzero(~near)
    r = chrono_eval(tr_i, te_i); r['fold'] = f; r['test_attack_pct'] = float(y[te_i].mean() * 100); cvres.append(r)
    log('blocked fold', f, {k: round(v['acc'], 2) for k, v in r.items() if isinstance(v, dict)})
OUT['blocked_cv5'] = dict(purge=PURGE, folds=cvres, summary={mdl: {m: dict(mean=float(np.mean([r[mdl][m] for r in cvres])), std=float(np.std([r[mdl][m] for r in cvres])))
                                                                    for m in ('acc', 'prec', 'rec', 'f1', 'fpr')} for mdl in cvres[0] if isinstance(cvres[0][mdl], dict) and mdl != 'fold'})

# ------------------------------------------------------------------ 8. ablation + exact Shapley of the vote (raw features, thesis split)
ts = np.arange(0, len(ite), 9)


def vote_cols(cols, q):
    n_ = GaussianNB().fit(Xtr[:, cols], ytr); k_ = KNeighborsClassifier(5, n_jobs=-1).fit(Xtr[:, cols], ytr)
    return m1(yte[q], (n_.predict_proba(Xte[q][:, cols])[:, 1] + k_.predict_proba(Xte[q][:, cols])[:, 1]) / 2 > .5)


allc = list(range(5))
OUT['ablation'] = dict(all=vote_cols(allc, ts), leave_one_out={F[j]: vote_cols([c for c in allc if c != j], ts) for j in range(5)},
                       single={F[j]: vote_cols([j], ts) for j in range(5)}, test_subsample=int(len(ts)))
log('ablation done')

nbm = GaussianNB().fit(Xtr, ytr); knm = KNeighborsClassifier(5, n_jobs=-1).fit(Xtr, ytr)
f_vote = lambda Z: (nbm.predict_proba(Z)[:, 1] + knm.predict_proba(Z)[:, 1]) / 2
rng = np.random.default_rng(0)
Bg = Xtr[rng.choice(len(Xtr), 40, replace=False)]; Sx = rng.choice(len(Xte), 120, replace=False)
M = 5; masks = np.array([[(s >> j) & 1 for j in range(M)] for s in range(1 << M)], bool)
wt = {n: factorial(n) * factorial(M - n - 1) / factorial(M) for n in range(M)}
phi = np.zeros((len(Sx), M))
for r, i in enumerate(Sx):
    Z = np.vstack([np.where(m[None, :], Xte[i][None, :], Bg) for m in masks])
    v = f_vote(Z).reshape(1 << M, len(Bg)).mean(1)
    for j in range(M):
        for s in range(1 << M):
            if not masks[s, j]:
                phi[r, j] += wt[masks[s].sum()] * (v[s | (1 << j)] - v[s])
ma = np.abs(phi).mean(0)
OUT['shap_exact_vote'] = dict(explainer='exact interventional Shapley, 32 coalitions, output = vote P(attack)', background_rows=len(Bg), explained_rows=len(Sx),
                              mean_abs_pct={F[j]: float(ma[j] / ma.sum() * 100) for j in range(M)},
                              direction_spearman={F[j]: float(spearmanr(Xte[Sx, j], phi[:, j])[0]) for j in range(M)})
log('shap', OUT['shap_exact_vote']['mean_abs_pct'])

# ------------------------------------------------------------------ 9. inference time (one 2-second window at a time)
vt = VotingClassifier([('a', GaussianNB()), ('b', KNeighborsClassifier(5, n_jobs=1))], voting='soft').fit(Xtr, ytr)
lat = {'vote_sklearn_VotingClassifier': [], 'nb_only': [], 'knn5_only': [], 'vote_manual_nb_plus_knn': []}
nb1 = GaussianNB().fit(Xtr, ytr); kn1 = KNeighborsClassifier(5, n_jobs=1).fit(Xtr, ytr)
for i in np.random.default_rng(1).choice(len(Xte), 300, replace=False):
    z = Xte[i:i + 1]
    t = time.perf_counter(); vt.predict(z); lat['vote_sklearn_VotingClassifier'].append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter(); nb1.predict(z); lat['nb_only'].append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter(); kn1.predict(z); lat['knn5_only'].append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter(); (nb1.predict_proba(z) + kn1.predict_proba(z)); lat['vote_manual_nb_plus_knn'].append((time.perf_counter() - t) * 1e3)
OUT['inference_ms'] = {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in lat.items()}
t = time.perf_counter(); vt.predict(Xte[:5000]); OUT['inference_ms']['batch_5000_windows_total_s'] = time.perf_counter() - t

OUT['runtime_seconds'] = round(time.time() - T0)
json.dump(OUT, open('stage1_results.json', 'w'), indent=1, default=float)
log('DONE')
