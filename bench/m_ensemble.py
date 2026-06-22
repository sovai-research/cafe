"""
LowRank+kNN Ensemble (Investigator 8).

Method:
  1) SoftImpute-ALS low-rank matrix completion (compact from-scratch numpy).
  2) k-NN imputation in the LEARNED FACTOR SPACE: rows are represented by the
     low-rank factor scores U (T,r); for each missing cell we find the nearest
     rows (in factor space) that observe that column and average their observed
     values. (Falls back to the low-rank reconstruction when no neighbor
     observes the column.)
  3) Blend the two predictions. The blend weight is tuned on a tiny held-out
     re-masked set (re-mask a few % of OBSERVED cells, score both predictors,
     pick the weight that minimizes held-out RMSE), defaulting to 0.5/0.5.

We also report the internal plain-SoftImpute numbers so the blend can be
compared against SoftImpute alone (printed; the harness only scores the blend).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from harness import run_method, summarize


# --------------------------------------------------------------------------- #
# SoftImpute-ALS  (Mazumder/Hastie/Tibshirani style soft-thresholded SVD)
# --------------------------------------------------------------------------- #
def softimpute(X, rank=None, lam=None, max_iter=100, tol=1e-4, seed=0):
    """
    Soft-thresholded SVD matrix completion on a standardized matrix.
    Returns (filled, U_scores) where U_scores is the (T,r) left-factor (row)
    representation used as the learned factor space.
    X has NaN at missing entries.
    """
    T, N = X.shape
    obs = ~np.isnan(X)
    # column standardize using observed entries
    mu = np.array([X[obs[:, j], j].mean() if obs[:, j].any() else 0.0
                   for j in range(N)])
    sd = np.array([X[obs[:, j], j].std() if obs[:, j].any() else 1.0
                   for j in range(N)])
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd

    if rank is None:
        rank = min(T, N, max(2, min(N, 30)))
    rank = int(min(rank, T, N))

    # init filled with zeros (mean in standardized space)
    Z = np.where(obs, Xs, 0.0)

    if lam is None:
        # set lambda from the initial spectrum (shrink moderately)
        s0 = np.linalg.svd(Z, compute_uv=False)
        lam = float(s0[min(rank, len(s0) - 1)]) if len(s0) > 1 else 0.0

    U = Vt = s = None
    prev = np.inf
    for _ in range(max_iter):
        # truncated SVD via full svd (matrices are small/medium)
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        U = U[:, :rank]; s = s[:rank]; Vt = Vt[:rank, :]
        s_th = np.maximum(s - lam, 0.0)
        low = (U * s_th) @ Vt
        Z_new = np.where(obs, Xs, low)
        change = np.linalg.norm(Z_new - Z) / (np.linalg.norm(Z) + 1e-12)
        Z = Z_new
        if change < tol:
            break
        prev = change

    # final low-rank reconstruction
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    keep = max(1, int(np.sum(np.maximum(s - lam, 0.0) > 1e-8)))
    keep = min(keep, rank)
    s_th = np.maximum(s[:keep] - lam, 0.0)
    Uf = U[:, :keep]
    low_std = (Uf * s_th) @ Vt[:keep, :]
    # de-standardize
    low = low_std * sd + mu
    filled = X.copy()
    filled[~obs] = low[~obs]
    # row factor scores (in standardized space) = U * sqrt(s_th) scaled
    U_scores = Uf * s_th[None, :]
    return filled, low, U_scores, obs


# --------------------------------------------------------------------------- #
# k-NN in learned factor space
# --------------------------------------------------------------------------- #
def knn_factor_impute(X, U_scores, low, obs, k=10):
    """
    Impute each missing cell by averaging observed values of that column from
    the k nearest rows in factor space. Fall back to `low` reconstruction if no
    neighbor observes the column.
    """
    T, N = X.shape
    out = low.copy()              # default = low-rank reconstruction
    # normalize factor scores for distance stability
    F = U_scores
    fn = np.linalg.norm(F, axis=1, keepdims=True)
    fn = np.where(fn < 1e-12, 1.0, fn)
    Fn = F / fn

    miss_rows = np.where(np.any(~obs, axis=1))[0]
    kk = min(k, T - 1)
    if kk < 1:
        return out

    for i in miss_rows:
        # cosine-ish distance: use euclidean on normalized factors
        d = np.sum((Fn - Fn[i]) ** 2, axis=1)
        d[i] = np.inf
        # nearest neighbors overall
        nn = np.argpartition(d, kk)[:kk]
        miss_cols = np.where(~obs[i])[0]
        for j in miss_cols:
            obs_nb = nn[obs[nn, j]]
            if obs_nb.size > 0:
                # distance-weighted average
                w = 1.0 / (d[obs_nb] + 1e-9)
                out[i, j] = np.sum(w * X[obs_nb, j]) / np.sum(w)
            # else keep low-rank value
    return out


# --------------------------------------------------------------------------- #
# Blend with tiny held-out tuning
# --------------------------------------------------------------------------- #
def _tune_weight(X, obs, seed=0):
    """
    Re-mask ~5% of observed cells, fit both predictors on the reduced matrix,
    and choose the blend weight alpha (low-rank weight) minimizing held-out RMSE.
    Returns alpha in [0,1].
    """
    rng = np.random.default_rng(123 + seed)
    obs_idx = np.argwhere(obs)
    if len(obs_idx) < 50:
        return 0.5
    n_hold = max(20, int(0.05 * len(obs_idx)))
    n_hold = min(n_hold, len(obs_idx) // 2)
    sel = rng.choice(len(obs_idx), size=n_hold, replace=False)
    hold = obs_idx[sel]
    Xtr = X.copy()
    Xtr[hold[:, 0], hold[:, 1]] = np.nan
    try:
        _, low_t, U_t, obs_t = softimpute(Xtr, max_iter=40)
        knn_t = knn_factor_impute(Xtr, U_t, low_t, obs_t, k=10)
    except Exception:
        return 0.5
    true_h = X[hold[:, 0], hold[:, 1]]
    low_h = low_t[hold[:, 0], hold[:, 1]]
    knn_h = knn_t[hold[:, 0], hold[:, 1]]
    best_a, best_e = 0.5, np.inf
    for a in np.linspace(0, 1, 11):
        pred = a * low_h + (1 - a) * knn_h
        e = np.sqrt(np.mean((pred - true_h) ** 2))
        if e < best_e:
            best_e, best_a = e, a
    return float(best_a)


# track internal SoftImpute-alone metrics for comparison
_SOFT_PRED = {}


def make_impute(tune=True):
    def impute(X, meta):
        T, N = X.shape
        filled, low, U_scores, obs = softimpute(X, max_iter=80)
        knn = knn_factor_impute(X, U_scores, low, obs, k=10)
        alpha = _tune_weight(X, obs) if tune else 0.5
        blend = alpha * low + (1 - alpha) * knn
        out = X.copy()
        out[~obs] = blend[~obs]
        # stash soft-only for side comparison
        soft_only = X.copy()
        soft_only[~obs] = low[~obs]
        _SOFT_PRED[(T, N, int(np.isnan(X).sum()))] = (soft_only, alpha)
        return out
    return impute


def softimpute_only(X, meta):
    filled, low, U_scores, obs = softimpute(X, max_iter=80)
    out = X.copy()
    out[~obs] = low[~obs]
    return out


if __name__ == "__main__":
    print("=== Plain SoftImpute (internal baseline) ===")
    rows_soft = run_method("SoftImpute(internal)", softimpute_only)
    summarize(rows_soft)

    print("\n=== LowRank+kNN Ensemble (tuned blend) ===")
    rows_ens = run_method("LowRank+kNN Ensemble", make_impute(tune=True))
    summarize(rows_ens)

    print("\n=== Blend vs SoftImpute (RMSE delta, negative = blend better) ===")
    soft_by_ds = {r["dataset"]: r for r in rows_soft}
    for r in rows_ens:
        s = soft_by_ds[r["dataset"]]
        d = r["rmse"] - s["rmse"]
        verdict = "blend wins" if d < 0 else "soft wins"
        print(f"{r['dataset']:18s} soft_rmse={s['rmse']:.4f}  "
              f"blend_rmse={r['rmse']:.4f}  delta={d:+.4f}  {verdict}")
