"""
reviewer_experiments.py  --  reproducible answers to the examiners' methodology questions.

    python reviewer_experiments.py random      # 80/20 random split (the protocol of the three scripts)
    python reviewer_experiments.py temporal    # chronological 70/30 inside each contiguous label block

Input : result.csv  (no header: FER,SIGS,FSR,PLV,BLV,type)
Output: results_<protocol>.json  (+ printed tables)

Everything is fit on the training partition only (MinMax scaler, tuning, calibration).
Hyper-parameter search space = the thesis space: k in 1..150 x {uniform, distance} x {euclidean, manhattan}
(Minkowski p=2 is Euclidean, so it is not a separate option).  Fitness = mean F1(attack) over 5 folds on a 25% subsample.
The exhaustive grid is computed exactly and cheaply (one neighbour query per fold and metric, all k from the sorted
neighbour list), so GWO / PSO / random search can be compared against the true optimum over many seeds.
"""
import sys, json, time
import numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

PROTO = sys.argv[1] if len(sys.argv) > 1 else 'random'
F = ['FER', 'SIGS', 'FSR', 'PLV', 'BLV']
KMAX, METRICS, WEIGHTS = 150, ['euclidean', 'manhattan'], ['uniform', 'distance']
OUT = {'protocol': PROTO}
T0 = time.time()
log = lambda *a: print('[%5.0fs]' % (time.time() - T0), *a, flush=True)

df = pd.read_csv('result.csv', header=None, names=F + ['type'])
X, y, N = df[F].values, df['type'].values.astype(int), len(df)
BLOCKS = [(0, 135408), (135408, 292710), (292710, N)]


# ---------------------------------------------------------------- protocol
def make_split():
    if PROTO == 'random':
        rng = np.random.default_rng(42)
        perm = rng.permutation(N); c = int(.8 * N)
        return np.sort(perm[:c]), np.sort(perm[c:])          # sorted only for reproducibility; order is irrelevant here
    tr, te = [], []
    for s, e in BLOCKS:
        c = s + int(.7 * (e - s)); tr += range(s, c); te += range(c, e)
    return np.array(tr), np.array(te)


def make_folds(tr_sub):
    """fold id per subsample row. random: shuffled 5-fold. temporal: 5 contiguous chunks inside every block."""
    if PROTO == 'random':
        r = np.random.default_rng(7).permutation(len(tr_sub)); f = np.empty(len(tr_sub), int); f[r] = np.arange(len(tr_sub)) % 5
        return f
    f = np.zeros(len(tr_sub), int)
    for s, e in BLOCKS:
        m = (tr_sub >= s) & (tr_sub < e)
        pos = np.flatnonzero(m); f[pos] = np.minimum((np.arange(len(pos)) * 5) // len(pos), 4)
    return f


tr, te = make_split()
sc = MinMaxScaler().fit(X[tr])
Xtr, ytr, Xte, yte = sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te]
sub = np.arange(0, len(tr), 4)                                  # 25% subsample, stride keeps time order
Xs, ys, folds = Xtr[sub], ytr[sub], make_folds(tr[sub])
log(f'protocol={PROTO} train={len(tr)} test={len(te)} test_attack={yte.mean()*100:.1f}% subsample={len(sub)}')
OUT['sizes'] = dict(train=int(len(tr)), test=int(len(te)), test_attack_pct=float(yte.mean() * 100), tuning_subsample=int(len(sub)))


# ---------------------------------------------------------------- exact KNN for every k at once
def knn_all_k(Xa, ya, Xq, metric):
    nn = NearestNeighbors(n_neighbors=KMAX, metric=metric, n_jobs=-1).fit(Xa)
    d, idx = nn.kneighbors(Xq)
    L = ya[idx].astype(np.float64)
    ks = np.arange(1, KMAX + 1)
    out = {'uniform': (np.cumsum(L, 1) / ks) > .5}
    zero = d == 0
    w = np.where(zero, 0., 1. / np.where(zero, 1., d))
    h = zero.any(1); w[h] = zero[h].astype(float)              # sklearn: a zero-distance neighbour takes all the weight
    out['distance'] = (np.cumsum(w * L, 1) / np.cumsum(w, 1)) > .5
    return out


def cols_metrics(yt, P):
    """P: (n, KMAX) boolean predictions -> dict of arrays over k"""
    yb = yt.astype(bool)[:, None]
    tp = (P & yb).sum(0).astype(float); fp = (P & ~yb).sum(0).astype(float)
    fn = (~P & yb).sum(0).astype(float); tn = (~P & ~yb).sum(0).astype(float)
    return dict(acc=(tp + tn) / len(yt), prec=tp / np.maximum(tp + fp, 1), rec=tp / np.maximum(tp + fn, 1),
                f1=2 * tp / np.maximum(2 * tp + fp + fn, 1), fpr=fp / np.maximum(fp + tn, 1))


def m1(yt, p):
    yb, pb = yt.astype(bool), np.asarray(p).astype(bool)
    tp, fp, fn, tn = (pb & yb).sum(), (pb & ~yb).sum(), (~pb & yb).sum(), (~pb & ~yb).sum()
    return dict(acc=float((tp + tn) / len(yt)), prec=float(tp / max(tp + fp, 1)), rec=float(tp / max(tp + fn, 1)),
                f1=float(2 * tp / max(2 * tp + fp + fn, 1)), fpr=float(fp / max(fp + tn, 1)))


# ---------------------------------------------------------------- 1. exhaustive grid (CV table) -------------
t = time.time()
cv = np.zeros((2, 2, KMAX))
for f in range(5):
    a, b = folds != f, folds == f
    for mi, m in enumerate(METRICS):
        o = knn_all_k(Xs[a], ys[a], Xs[b], m)
        for wi, w in enumerate(WEIGHTS):
            cv[mi, wi] += cols_metrics(ys[b], o[w])['f1'] / 5
grid_seconds = time.time() - t
bi = np.unravel_index(np.argmax(cv), cv.shape)
OUT['grid'] = dict(seconds=round(grid_seconds, 1), n_configs=int(cv.size), best=dict(metric=METRICS[bi[0]], weights=WEIGHTS[bi[1]], k=int(bi[2]) + 1),
                   best_cv_f1=float(cv[bi]), default_cv_f1=float(cv[0, 0, 4]), cv_f1_min=float(cv.min()), cv_f1_median=float(np.median(cv)))
log('grid done', OUT['grid'])

# ---- verify the vectorised KNN against scikit-learn on a few configs (validation fold 0)
a, b = folds != 0, folds == 0
chk = {}
for (m, w, k) in [('euclidean', 'uniform', 5), ('euclidean', 'distance', 49), ('manhattan', 'distance', 7), ('manhattan', 'uniform', 100)]:
    ref = KNeighborsClassifier(k, weights=w, metric=m, n_jobs=-1).fit(Xs[a], ys[a]).predict(Xs[b])
    mine = knn_all_k(Xs[a], ys[a], Xs[b], m)[w][:, k - 1].astype(int)
    chk[f'{m}-{w}-k{k}'] = float((ref == mine).mean())
OUT['vectorised_vs_sklearn_agreement'] = chk
log('agreement with sklearn', chk)

# ---------------------------------------------------------------- 2. test landscape for EVERY config (never used for selection)
te_pred, te_m = {}, {}
for mi, m in enumerate(METRICS):
    o = knn_all_k(Xtr, ytr, Xte, m)
    for wi, w in enumerate(WEIGHTS):
        te_pred[(mi, wi)] = o[w]
        te_m[(mi, wi)] = cols_metrics(yte, o[w])
    log('test neighbour table', m)
test_f1 = np.array([[te_m[(mi, wi)]['f1'] for wi in range(2)] for mi in range(2)])  # (2,2,KMAX)
test_acc = np.array([[te_m[(mi, wi)]['acc'] for wi in range(2)] for mi in range(2)])
OUT['test_landscape'] = dict(f1_min=float(test_f1.min()), f1_median=float(np.median(test_f1)), f1_max=float(test_f1.max()),
                             acc_min=float(test_acc.min()), acc_median=float(np.median(test_acc)), acc_max=float(test_acc.max()),
                             oracle_best_test_f1_config=dict(zip(['metric', 'weights', 'k'], [METRICS[np.unravel_index(np.argmax(test_f1), test_f1.shape)[0]],
                                                                                               WEIGHTS[np.unravel_index(np.argmax(test_f1), test_f1.shape)[1]],
                                                                                               int(np.unravel_index(np.argmax(test_f1), test_f1.shape)[2]) + 1])))


def cfg_metrics(mi, wi, k):
    return {kk: float(v[k]) for kk, v in te_m[(mi, wi)].items()}


OUT['knn_default'] = cfg_metrics(0, 0, 4)
OUT['knn_grid_opt'] = cfg_metrics(*bi)
OUT['knn_grid_opt_cfg'] = OUT['grid']['best']


# ---------------------------------------------------------------- 3. GWO / PSO / random search vs the exact optimum
def decode(pos):
    return (int(np.clip(np.round(pos[1 + 1]), 0, 1)), int(np.clip(np.round(pos[1]), 0, 1)), int(np.clip(np.round(pos[0]), 1, KMAX)) - 1)


fit = lambda c: cv[c]
LB, UB = np.array([1., 0., 0.]), np.array([float(KMAX), 1., 1.])


def run_gwo(rng, pop=5, iters=10):
    P = LB + rng.random((pop, 3)) * (UB - LB); f = np.array([fit(decode(p)) for p in P]); ev = pop
    best = (f.max(), decode(P[f.argmax()])); seen = {decode(p) for p in P}
    for t in range(iters):
        o = np.argsort(-f); lead = [P[o[0]].copy(), P[o[1]].copy(), P[o[2]].copy()]; a = 2 * (1 - t / iters)
        for i in range(pop):
            acc = np.zeros(3)
            for L_ in lead:
                A = 2 * a * rng.random(3) - a; C = 2 * rng.random(3); acc += L_ - A * np.abs(C * L_ - P[i])
            P[i] = np.clip(acc / 3, LB, UB); c = decode(P[i]); f[i] = fit(c); ev += 1; seen.add(c)
            if f[i] > best[0]: best = (f[i], c)
    return best, ev, len(seen)


def run_pso(rng, pop=5, iters=10, w=.7, c1=1.5, c2=1.5):
    P = LB + rng.random((pop, 3)) * (UB - LB); V = np.zeros((pop, 3)); f = np.array([fit(decode(p)) for p in P]); ev = pop
    pb, pf = P.copy(), f.copy(); g = pb[pf.argmax()].copy(); best = (f.max(), decode(P[f.argmax()])); seen = {decode(p) for p in P}
    for _ in range(iters):
        for i in range(pop):
            V[i] = w * V[i] + c1 * rng.random(3) * (pb[i] - P[i]) + c2 * rng.random(3) * (g - P[i]); P[i] = np.clip(P[i] + V[i], LB, UB)
            c = decode(P[i]); fi = fit(c); ev += 1; seen.add(c)
            if fi > pf[i]: pf[i], pb[i] = fi, P[i].copy()
            if fi > best[0]: best = (fi, c); g = P[i].copy()
    return best, ev, len(seen)


def run_rand(rng, n=55):
    best = (-1, None)
    for _ in range(n):
        c = (int(rng.integers(2)), int(rng.integers(2)), int(rng.integers(KMAX)))
        if fit(c) > best[0]: best = (fit(c), c)
    return best, n, n


opt = float(cv.max()); S = 300
res = {}
for name, fn in [('GWO', run_gwo), ('PSO', run_pso), ('RandomSearch', run_rand)]:
    cvs, tf1, tacc, evs, uniq, hit = [], [], [], [], [], 0
    for s in range(S):
        (fv, c), ev, un = fn(np.random.default_rng(1000 + s))
        cvs.append(fv); tf1.append(float(test_f1[c])); tacc.append(float(test_acc[c])); evs.append(ev); uniq.append(un); hit += fv >= opt - 1e-12
    cvs, tf1, tacc = np.array(cvs), np.array(tf1), np.array(tacc)
    res[name] = dict(seeds=S, evaluations=int(np.mean(evs)), unique_configs_mean=float(np.mean(uniq)), pct_seeds_reaching_grid_optimum=float(hit / S * 100),
                     pct_seeds_within_0p002_cv_f1=float(np.mean(cvs >= opt - .002) * 100), cv_f1_mean=float(cvs.mean()), cv_f1_min=float(cvs.min()),
                     test_f1_mean=float(tf1.mean()), test_f1_std=float(tf1.std()), test_f1_min=float(tf1.min()), test_f1_max=float(tf1.max()),
                     test_acc_mean=float(tacc.mean()), test_acc_std=float(tacc.std()))
OUT['optimizers'] = res
OUT['search_space'] = dict(total_configs=int(cv.size), evaluations_per_run=int(res['GWO']['evaluations']), pct_of_space=float(res['GWO']['evaluations'] / cv.size * 100))
log('optimizers', json.dumps(res['GWO']))

# ---------------------------------------------------------------- 4. baselines (same split, same folds, same subsample for tuning)
def blocked_cv(make, grid):
    best, bs = None, -1
    for p in grid:
        s = 0
        for f in range(5):
            a, b = folds != f, folds == f
            s += m1(ys[b], make(**p).fit(Xs[a], ys[a]).predict(Xs[b]))['f1'] / 5
        if s > bs: best, bs = p, s
    return best


preds = {}
base = {}
def add(name, p, extra=None):
    preds[name] = np.asarray(p).astype(int); base[name] = m1(yte, p)
    if extra: base[name].update(extra)
    log('baseline', name, {k: round(v * 100, 2) if k != 'seeds' else v for k, v in base[name].items() if isinstance(v, float)})

# KNN
add('KNN default (k=5)', te_pred[(0, 0)][:, 4])
add('KNN grid-optimum', te_pred[(bi[0], bi[1])][:, bi[2]])
# trees
bp = blocked_cv(lambda **k: DecisionTreeClassifier(random_state=0, **k), [dict(max_depth=d, min_samples_leaf=l) for d in (None, 12) for l in (1, 5, 20, 50)])
add('Decision tree default', DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).predict(Xte))
add('Decision tree tuned', DecisionTreeClassifier(random_state=0, **bp).fit(Xtr, ytr).predict(Xte), dict(params=str(bp)))
bp = blocked_cv(lambda **k: RandomForestClassifier(100, n_jobs=2, random_state=0, **k), [dict(min_samples_leaf=l) for l in (1, 10)])
rf_runs = [m1(yte, RandomForestClassifier(100, n_jobs=2, random_state=s, **bp).fit(Xtr, ytr).predict(Xte)) for s in range(3)]
add('Random forest (100) tuned', RandomForestClassifier(100, n_jobs=2, random_state=0, **bp).fit(Xtr, ytr).predict(Xte),
    dict(params=str(bp), f1_std_3seeds=float(np.std([r['f1'] for r in rf_runs]))))
add('Extra trees (100)', ExtraTreesClassifier(100, n_jobs=2, random_state=0).fit(Xtr, ytr).predict(Xte))
add('Gradient boosting (hist)', HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr).predict(Xte))
add('Logistic regression', LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte))
nb = GaussianNB().fit(Xtr, ytr); add('Naive Bayes', nb.predict(Xte))
OUT['baselines'] = base

# ---------------------------------------------------------------- 5. McNemar (exact) on test predictions
def mcnemar(a, b):
    ca, cb = preds[a] == yte, preds[b] == yte
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())          # a right/b wrong, a wrong/b right
    p = binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0
    return dict(a_only_correct=n01, b_only_correct=n10, p_value=float(p))


ref = 'KNN grid-optimum'
OUT['mcnemar_vs_grid_opt_knn'] = {k: mcnemar(ref, k) for k in preds if k != ref}
OUT['mcnemar_default_vs_grid'] = mcnemar('KNN default (k=5)', ref)

# ---------------------------------------------------------------- 6. soft-voting: weights, calibration, threshold
kn5 = KNeighborsClassifier(5, n_jobs=-1).fit(Xtr, ytr)
pk, pn = kn5.predict_proba(Xte)[:, 1], nb.predict_proba(Xte)[:, 1]
cal = CalibratedClassifierCV(GaussianNB(), method='isotonic', cv=3).fit(Xtr, ytr); pc = cal.predict_proba(Xte)[:, 1]
vote = {}
for wn in (0, .25, .5, .75, 1):
    vote[f'w_nb={wn}'] = m1(yte, (wn * pn + (1 - wn) * pk) > .5)
OUT['vote_weight_sweep'] = vote
OUT['vote_equal'] = m1(yte, (pn + pk) / 2 > .5)
OUT['vote_equal_calibrated_nb'] = m1(yte, (pc + pk) / 2 > .5)
OUT['vote_threshold_sweep'] = {f't={t}': m1(yte, (pn + pk) / 2 > t) for t in (.2, .3, .4, .5, .6, .7)}
OUT['nb_prob_extremes_pct'] = float(np.mean((pn < .01) | (pn > .99)) * 100)
OUT['knn5_prob_extremes_pct'] = float(np.mean((pk < .01) | (pk > .99)) * 100)
preds['Vote NB+KNN5 equal'] = ((pn + pk) / 2 > .5).astype(int)
OUT['mcnemar_vote_vs_knn_default'] = mcnemar('Vote NB+KNN5 equal', 'KNN default (k=5)')
log('voting done')

# ---------------------------------------------------------------- 7. features: correlation, leave-one-out, single feature, permutation
OUT['spearman_corr'] = pd.DataFrame(sc.transform(X[tr]), columns=F).corr('spearman').round(3).to_dict()
mi, wi, kk = bi[0], bi[1], int(bi[2]) + 1
ts = np.arange(0, len(te), 3)                                          # test subsample for speed
def fitk(cols):
    m = KNeighborsClassifier(kk, weights=WEIGHTS[wi], metric=METRICS[mi], n_jobs=-1).fit(Xtr[:, cols], ytr)
    return m1(yte[ts], m.predict(Xte[ts][:, cols]))
allc = list(range(5)); full = fitk(allc)
OUT['ablation'] = dict(all=full, leave_one_out={F[j]: fitk([c for c in allc if c != j]) for j in range(5)}, single={F[j]: fitk([j]) for j in range(5)})
log('ablation done')

# ---------------------------------------------------------------- 8. exact Shapley values of the tuned KNN (5 features -> 32 coalitions)
model = KNeighborsClassifier(kk, weights=WEIGHTS[wi], metric=METRICS[mi], n_jobs=-1).fit(Xtr, ytr)
rng = np.random.default_rng(0)
B = Xtr[rng.choice(len(Xtr), 50, replace=False)]
S_idx = rng.choice(len(Xte), 150, replace=False)
from math import factorial
M = 5; masks = np.array([[(s >> j) & 1 for j in range(M)] for s in range(1 << M)], bool)
wt = {n: factorial(n) * factorial(M - n - 1) / factorial(M) for n in range(M)}
phi_all = np.zeros((len(S_idx), M))
for r, i in enumerate(S_idx):
    Z = np.vstack([np.where(m[None, :], Xte[i][None, :], B) for m in masks])
    v = model.predict_proba(Z)[:, 1].reshape(1 << M, len(B)).mean(1)
    for j in range(M):
        for s in range(1 << M):
            if not masks[s, j]:
                n = masks[s].sum(); phi_all[r, j] += wt[n] * (v[s | (1 << j)] - v[s])
mean_abs = np.abs(phi_all).mean(0)
OUT['shap_exact'] = dict(explainer='exact interventional Shapley (32 coalitions), output = predict_proba(attack)', model=OUT['grid']['best'],
                         background_rows=50, explained_rows=150,
                         mean_abs_pct={F[j]: float(mean_abs[j] / mean_abs.sum() * 100) for j in range(M)},
                         direction_spearman_value_vs_shap={F[j]: float(spearmanr(Xte[S_idx, j], phi_all[:, j])[0]) for j in range(M)})
log('shap done', OUT['shap_exact']['mean_abs_pct'])

# ---------------------------------------------------------------- 9. inference time of the chosen model, one window at a time
model1 = KNeighborsClassifier(kk, weights=WEIGHTS[wi], metric=METRICS[mi], n_jobs=1).fit(Xtr, ytr)
lat = []
for i in np.random.default_rng(1).choice(len(Xte), 300, replace=False):
    a = time.perf_counter(); model1.predict(Xte[i:i + 1]); lat.append((time.perf_counter() - a) * 1e3)
OUT['inference_ms_single_window'] = dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)), train_rows=int(len(Xtr)))

OUT['runtime_seconds'] = round(time.time() - T0)
json.dump(OUT, open(f'results_{PROTO}.json', 'w'), indent=1, default=float)
log('DONE')
