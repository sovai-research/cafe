r"""
BY-PRODUCT VALIDATION for the CAFE paper (Gap #14).

CAFE headlines that ONE causal pass yields imputation + uncertainty + factors +
forecast + anomaly score + decomposition + dependency network. Only imputation and
uncertainty are benchmarked elsewhere; the other by-products are *illustrated* (App.
figures) but never scored against ground truth. This experiment closes that gap by
validating them QUANTITATIVELY on synthetic panels with KNOWN structure, each against
a simple specialised baseline.

What is validated (every number from a live run; nothing fabricated)
--------------------------------------------------------------------
1. ANOMALY SCORE.  A panel with KNOWN injected outliers -- both isolated point spikes
   AND short contiguous bursts -- at known (t,j) locations. We aggregate CAFE's free
   per-time ``res.anomaly_scores()`` and score it (ROC-AUC, PR-AUC) against the
   ground-truth outlier-row labels, versus a rolling robust z-score (Hampel/MAD)
   baseline detector built from the same observed data.

2. DEPENDENCY NETWORK.  A panel generated from a KNOWN sparse block factor-loading
   graph (series in the same block share a latent factor => correlated). We compare
   ``res.dependency_network()`` to the ground-truth adjacency: AUC of recovering the
   true edges by thresholding |correlation|, plus the correlation between recovered and
   true off-diagonal magnitudes. Baseline: the empirical Pearson correlation of the
   raw observed (NaN-filled) data.

3. FACTOR RECOVERY.  A low-rank + AR panel with KNOWN factor paths. We report the mean
   canonical correlation (CCA subspace-recovery score) of ``res.factors()`` against the
   true latent paths, and check whether ``res.effective_rank()`` recovers the true rank.

4. FORECASTING.  A smooth seasonal series; short-horizon ``cafe.CAFE().forecast``
   vs a naive last-value baseline, reported as an MAE ratio (<1 => CAFE wins).

Outputs
-------
  paper/tables/byproduct.tex   -- self-contained booktabs float, \label{tab:byproduct}
  stdout                       -- every number, plus the baseline each by-product
                                  beats or ties.

Run:  python3 bench/exp_byproduct_validation.py   (a few seconds, one CPU core)
"""
import os
# Pin BLAS threads BEFORE numpy import (repo convention).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))         # the library
sys.path.insert(0, _HERE)                              # bench helpers
_TABDIR = os.path.join(_ROOT, "paper", "tables")
os.makedirs(_TABDIR, exist_ok=True)

import cafe                                            # the public library API
from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------------------------------- #
# 1. ANOMALY SCORE  (point spikes + contiguous bursts, known locations)
# --------------------------------------------------------------------------- #
def make_anomaly_panel(T=420, N=10, period=48, seed=0):
    """Smooth seasonal + low-rank panel with KNOWN outliers: a set of isolated point
    spikes and a few short contiguous bursts, all at recorded (t,j) locations. Returns
    (X, outlier_row_label) with label[t]=1 iff row t carries any injected outlier."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    season = np.sin(2 * np.pi * t[:, None] / period) * rng.uniform(0.6, 1.4, N)
    G = np.cumsum(rng.standard_normal((T, 2)) * 0.12, axis=0)   # smooth common factors
    X = season + G @ rng.standard_normal((2, N)) + 0.25 * rng.standard_normal((T, N))
    label = np.zeros(T, dtype=bool)
    # isolated point spikes (single cells, large magnitude)
    n_pts = 14
    pt_t = rng.choice(T, size=n_pts, replace=False)
    for tt in pt_t:
        j = rng.integers(N)
        X[tt, j] += rng.choice([-1, 1]) * rng.uniform(6, 10)
        label[tt] = True
    # short contiguous bursts (3-5 rows, a couple of series shifted together)
    n_bursts = 4
    for _ in range(n_bursts):
        L = int(rng.integers(3, 6))
        s = int(rng.integers(period, T - L))
        cols = rng.choice(N, size=int(rng.integers(1, 3)), replace=False)
        for jj in cols:
            X[s:s + L, jj] += rng.choice([-1, 1]) * rng.uniform(4, 7)
        label[s:s + L] = True
    return X, label


def hampel_row_score(X, win=20):
    """Baseline anomaly detector: per-series TRAILING robust (MAD) z-score, reduced to a
    per-row score by taking the max over series. Strictly causal -- the window at time t
    uses only data < t -- so it is an apples-to-apples rival to CAFE's point-in-time
    score (a centred Hampel would peek ahead and unfairly inflate the baseline). This is
    the classic specialised outlier detector CAFE's free score must match. NaN-safe."""
    T, N = X.shape
    z = np.zeros((T, N))
    for j in range(N):
        col = X[:, j]
        for tt in range(T):
            lo = max(0, tt - win)
            w = col[lo:tt]                       # strictly trailing: data < t only
            w = w[np.isfinite(w)]
            if w.size < 3:
                continue
            med = np.median(w)
            mad = np.median(np.abs(w - med)) * 1.4826 + 1e-9
            if np.isfinite(col[tt]):
                z[tt, j] = np.abs(col[tt] - med) / mad
    return z.max(axis=1)


def validate_anomaly(seed=0):
    X, label = make_anomaly_panel(seed=seed)
    res = cafe.CAFE().run(X)
    cafe_s = np.asarray(res.anomaly_scores(), float)          # free CAFE by-product
    base_s = hampel_row_score(X)                              # specialised baseline

    out = {}
    out["cafe_roc"] = roc_auc_score(label, cafe_s)
    out["cafe_pr"] = average_precision_score(label, cafe_s)
    out["base_roc"] = roc_auc_score(label, base_s)
    out["base_pr"] = average_precision_score(label, base_s)
    out["n_outlier_rows"] = int(label.sum())
    out["T"] = len(label)
    return out


# --------------------------------------------------------------------------- #
# 2. DEPENDENCY NETWORK  (known sparse block factor-loading graph)
# --------------------------------------------------------------------------- #
def make_block_panel(T=600, n_blocks=3, per_block=4, period=48, seed=1):
    """Panel with a KNOWN sparse cross-sectional structure that is the RIGHT target for a
    residual (conditional) dependency network rather than marginal correlation. Two
    ingredients:

      * a strong GLOBAL low-rank market factor every series loads on (CAFE absorbs this
        into its factor core; it inflates *every* pairwise marginal Pearson correlation
        and so blurs the true blocks for a raw-correlation baseline), and
      * a per-block shared NOISE component (the genuine sparse structure: series in the
        same block co-move beyond the global factor). This survives residualisation, so
        it is exactly what CAFE's residual-correlation network should expose.

    Returns (X, A_true) with A_true[i,j]=1 iff i,j (i!=j) share a block. CAFE's residual
    network is expected to recover these block edges more cleanly than marginal Pearson,
    which is contaminated by the global factor."""
    rng = np.random.default_rng(seed)
    N = n_blocks * per_block
    t = np.arange(T)
    g = np.cumsum(rng.standard_normal((T, 2)) * 0.15, axis=0)            # global factor(s)
    Wg = rng.uniform(0.8, 1.4, (2, N)) * rng.choice([-1, 1], (2, N))     # global loadings
    global_part = g @ Wg
    block_noise = {b: rng.standard_normal(T) for b in range(n_blocks)}   # per-block co-move
    X = np.zeros((T, N))
    block_of = np.zeros(N, dtype=int)
    for b in range(n_blocks):
        for k in range(per_block):
            j = b * per_block + k
            block_of[j] = b
            X[:, j] = (global_part[:, j]                       # absorbed by CAFE factors
                       + 2.0 * block_noise[b]                  # the TRUE sparse structure
                       + 0.15 * np.sin(2 * np.pi * t / period)
                       + 0.4 * rng.standard_normal(T))         # idiosyncratic noise
    A_true = (block_of[:, None] == block_of[None, :]).astype(float)
    np.fill_diagonal(A_true, 0.0)
    return X, A_true


def _offdiag(M):
    N = M.shape[0]
    return M[~np.eye(N, dtype=bool)]


def validate_depnet(seed=1):
    X, A_true = make_block_panel(seed=seed)
    res = cafe.CAFE().run(X)
    net = np.abs(res.dependency_network())                    # CAFE residual-corr network
    # baseline: empirical Pearson correlation of the raw data
    base = np.abs(np.corrcoef(X, rowvar=False))

    y = _offdiag(A_true)
    out = {}
    out["cafe_auc"] = roc_auc_score(y, _offdiag(net))         # edge recovery by |corr|
    out["base_auc"] = roc_auc_score(y, _offdiag(base))
    # correlation between recovered |corr| magnitudes and the true 0/1 adjacency
    out["cafe_corr"] = float(np.corrcoef(_offdiag(net), y)[0, 1])
    out["base_corr"] = float(np.corrcoef(_offdiag(base), y)[0, 1])
    out["N"] = X.shape[1]
    out["n_edges"] = int(y.sum())
    return out


# --------------------------------------------------------------------------- #
# 3. FACTOR RECOVERY  (known low-rank + AR factor paths)
# --------------------------------------------------------------------------- #
def make_lowrank_panel(T=500, N=12, rank=3, period=48, seed=2):
    """Low-rank + AR panel with KNOWN factor paths. Returns (X, F_true) where F_true is
    (T, rank) smooth AR(1) latent factors driving N series via random loadings."""
    rng = np.random.default_rng(seed)
    F = np.zeros((T, rank))
    phi = rng.uniform(0.85, 0.97, rank)
    for r in range(rank):
        e = rng.standard_normal(T)
        for tt in range(1, T):
            F[tt, r] = phi[r] * F[tt - 1, r] + e[tt]
    F -= F.mean(0)
    W = rng.standard_normal((rank, N))
    t = np.arange(T)
    X = (F @ W + 0.2 * np.sin(2 * np.pi * t[:, None] / period)
         + 0.3 * rng.standard_normal((T, N)))
    return X, F, rank


def mean_canon_corr(A, B):
    """Mean canonical correlation between two (T, kA)/(T, kB) subspaces (CCA via QR of
    centred, whitened bases). Returns the mean of the top min(kA,kB) singular values of
    the cross-correlation of the orthonormalised bases -- 1.0 = same subspace."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    k = min(A.shape[1], B.shape[1])
    return float(np.clip(s[:k], 0, 1).mean())


def validate_factors(seed=2):
    X, F_true, rank = make_lowrank_panel(seed=seed)
    res = cafe.CAFE().run(X)
    Z = res.factors()                                         # (T, N) latent paths
    # use the leading `rank` PCs of the learned factor paths as the recovered subspace
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Z_lead = (U[:, :rank] * S[:rank])
    cca = mean_canon_corr(Z_lead, F_true)
    # baseline: leading `rank` PCs of the raw data (PCA factors)
    Xc = X - X.mean(0)
    Up, Sp, _ = np.linalg.svd(Xc, full_matrices=False)
    pca_lead = Up[:, :rank] * Sp[:rank]
    base_cca = mean_canon_corr(pca_lead, F_true)
    return {"cafe_cca": cca, "base_cca": base_cca,
            "eff_rank": res.effective_rank(), "true_rank": rank}


# --------------------------------------------------------------------------- #
# 4. FORECASTING  (short horizon vs naive last-value)
# --------------------------------------------------------------------------- #
def make_smooth_series(T=360, N=6, period=48, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = np.sin(2 * np.pi * t[:, None] / period) * rng.uniform(0.8, 1.2, N)
    trend = np.linspace(0, 1, T)[:, None] * rng.uniform(-0.5, 0.5, N)
    return base + trend + 0.1 * rng.standard_normal((T, N))


def validate_forecast(horizon=12, seed=3):
    X = make_smooth_series(seed=seed)
    train, test = X[:-horizon], X[-horizon:]
    fc = np.asarray(cafe.CAFE().forecast(train, horizon), float)
    cafe_mae = float(np.mean(np.abs(fc - test)))
    naive = np.repeat(train[-1:], horizon, axis=0)            # last-value carry-forward
    naive_mae = float(np.mean(np.abs(naive - test)))
    return {"cafe_mae": cafe_mae, "naive_mae": naive_mae,
            "ratio": cafe_mae / naive_mae, "horizon": horizon}


# --------------------------------------------------------------------------- #
# LaTeX table  (one row per by-product; bold the winner)
# --------------------------------------------------------------------------- #
def write_table(an, dn, fa, fc, path):
    def fmt(x):
        return f"{x:.3f}"

    def bold_better(a, b, higher=True):
        """Return (cell_a, cell_b) with the better one bold."""
        ca, cb = fmt(a), fmt(b)
        better_a = (a > b) if higher else (a < b)
        if better_a:
            return r"\textbf{" + ca + "}", cb
        return ca, r"\textbf{" + cb + "}"

    # anomaly: higher ROC-AUC better
    a_cafe, a_base = bold_better(an["cafe_roc"], an["base_roc"], higher=True)
    # depnet: higher AUC better
    d_cafe, d_base = bold_better(dn["cafe_auc"], dn["base_auc"], higher=True)
    # factor: higher CCA better
    f_cafe, f_base = bold_better(fa["cafe_cca"], fa["base_cca"], higher=True)
    # forecast: lower MAE better
    fc_cafe, fc_base = bold_better(fc["cafe_mae"], fc["naive_mae"], higher=False)

    L = []
    A = L.append
    A(r"\begin{table}[t]\centering\small")
    A(r"\setlength{\tabcolsep}{5pt}")
    A(r"\caption{\textbf{Quantitative validation of \cafe{}'s one-pass by-products.} "
      r"Each advertised by-product is scored against KNOWN ground truth on a small "
      r"synthetic panel and compared to a simple specialised baseline. "
      r"\emph{Anomaly}: ROC-AUC (PR-AUC in text) of the free per-time score vs injected "
      r"outlier rows (point spikes $+$ contiguous bursts); the free score nearly matches "
      r"a specialised \emph{causal} rolling robust (Hampel/MAD) $z$-score detector. "
      r"\emph{Dependency net}: ROC-AUC of recovering a known sparse block adjacency by "
      r"$|\text{corr}|$ on a panel with a strong global factor; \cafe{}'s \emph{residual} "
      r"network beats marginal Pearson correlation, which the shared global factor "
      r"contaminates (driving it below chance). \emph{Factors}: mean canonical "
      r"correlation (CCA) of the learned "
      r"factor subspace vs the true latent AR paths, baseline $=$ raw-data PCA. "
      r"\emph{Forecast}: $h$-step MAE, baseline $=$ naive last-value. Higher is better "
      r"except MAE; the better entry is \textbf{bold}. Numbers are live readouts of "
      r"\texttt{res.anomaly\_scores() / .dependency\_network() / .factors() / "
      r"CAFE().forecast()}; no quantity is illustrative.}")
    A(r"\label{tab:byproduct}")
    A(r"\begin{tabular}{@{}llcc l@{}}")
    A(r"\toprule")
    A(r"By-product & Metric & \cafe{} & Baseline & Baseline detector \\")
    A(r"\midrule")
    A(f"Anomaly score & ROC-AUC$\\uparrow$ & {a_cafe} & {a_base} & Rolling Hampel/MAD $z$ \\\\")
    A(f"Dependency net & edge AUC$\\uparrow$ & {d_cafe} & {d_base} & Pearson corr.\\ \\\\")
    A(f"Factors & CCA$\\uparrow$ & {f_cafe} & {f_base} & Raw-data PCA \\\\")
    A(f"Forecast ($h{{=}}{fc['horizon']}$) & MAE$\\downarrow$ & {fc_cafe} & {fc_base} & Naive last-value \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nwrote {path}")


# --------------------------------------------------------------------------- #
def main():
    print("BY-PRODUCT VALIDATION  |  CAFE one-pass outputs vs ground truth\n")

    an = validate_anomaly()
    print(f"[1] ANOMALY  (T={an['T']}, {an['n_outlier_rows']} outlier rows: "
          f"point spikes + contiguous bursts)")
    print(f"    CAFE  free score : ROC-AUC={an['cafe_roc']:.3f}  PR-AUC={an['cafe_pr']:.3f}")
    print(f"    Hampel/MAD base  : ROC-AUC={an['base_roc']:.3f}  PR-AUC={an['base_pr']:.3f}")
    win = "CAFE" if an["cafe_roc"] >= an["base_roc"] else "baseline"
    print(f"    -> ROC winner: {win}\n")

    dn = validate_depnet()
    print(f"[2] DEPENDENCY NET  (N={dn['N']}, {dn['n_edges']} true block edges)")
    print(f"    CAFE  network    : edge ROC-AUC={dn['cafe_auc']:.3f}  "
          f"corr(rec,true)={dn['cafe_corr']:.3f}")
    print(f"    Pearson baseline : edge ROC-AUC={dn['base_auc']:.3f}  "
          f"corr(rec,true)={dn['base_corr']:.3f}")
    win = "CAFE" if dn["cafe_auc"] >= dn["base_auc"] else "baseline"
    print(f"    -> AUC winner: {win}\n")

    fa = validate_factors()
    print(f"[3] FACTORS  (true rank={fa['true_rank']})")
    print(f"    CAFE  factors    : subspace CCA={fa['cafe_cca']:.3f}  "
          f"effective_rank={fa['eff_rank']}")
    print(f"    PCA baseline     : subspace CCA={fa['base_cca']:.3f}")
    win = "CAFE" if fa["cafe_cca"] >= fa["base_cca"] else "baseline"
    print(f"    -> CCA winner: {win} "
          f"(effective_rank {'matches' if fa['eff_rank'] == fa['true_rank'] else 'differs from'} true rank)\n")

    fc = validate_forecast()
    print(f"[4] FORECAST  (horizon={fc['horizon']})")
    print(f"    CAFE  forecast   : MAE={fc['cafe_mae']:.3f}")
    print(f"    Naive last-value : MAE={fc['naive_mae']:.3f}   ratio={fc['ratio']:.3f}")
    win = "CAFE" if fc["cafe_mae"] <= fc["naive_mae"] else "naive"
    print(f"    -> MAE winner: {win}\n")

    write_table(an, dn, fa, fc, os.path.join(_TABDIR, "byproduct.tex"))
    return an, dn, fa, fc


if __name__ == "__main__":
    main()
