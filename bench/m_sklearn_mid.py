import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")

import numpy as np
from harness import run_method, summarize

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# --------------------------------------------------------------------------- #
# (a) MICE = IterativeImputer with BayesianRidge
# --------------------------------------------------------------------------- #
def mice_impute(X, meta):
    """IterativeImputer (MICE-style) with BayesianRidge.

    IterativeImputer cost scales with n_features^2 * n_samples * max_iter.
    The harness datasets are wide-ish in time (T up to 2000) but narrow in
    features (N up to 60), so the per-feature regression dominates. We cap
    max_iter modestly (10) to keep runtime sane. No feature explosion here,
    so the large 2D set (2000x60) is still tractable but is the slowest.
    """
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        sample_posterior=False,
        initial_strategy="mean",
        n_nearest_features=None,
        tol=1e-3,
        random_state=0,
    )
    out = imp.fit_transform(X)
    # Safety: if any column was fully missing, sklearn may drop it; guard NaNs.
    if np.isnan(out).any():
        col = np.nanmean(X, axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(col, idx[1])
    return out


# --------------------------------------------------------------------------- #
# (b) IterativeSVD = soft-impute-lite via rank-r truncated SVD loop
# --------------------------------------------------------------------------- #
def itersvd_impute(X, meta, rank=None, max_iter=50, tol=1e-4):
    """Iterative truncated-SVD imputation (a la Troyanskaya 2001 SVDimpute /
    soft-impute-lite). Fill missing with column means, then repeatedly take a
    rank-r truncated SVD and overwrite ONLY the missing entries with the
    low-rank reconstruction until convergence.
    """
    Xf = X.astype(float).copy()
    mask = np.isnan(Xf)
    if not mask.any():
        return Xf

    # initial fill with column means (fallback to global mean for empty cols)
    col_mean = np.nanmean(Xf, axis=0)
    gmean = np.nanmean(Xf)
    col_mean = np.where(np.isfinite(col_mean), col_mean, gmean if np.isfinite(gmean) else 0.0)
    fill_idx = np.where(mask)
    Xf[fill_idx] = np.take(col_mean, fill_idx[1])

    T, N = Xf.shape
    if rank is None:
        rank = min(T, N, 10)        # modest rank; data is genuinely low-rank
    rank = max(1, min(rank, min(T, N) - 1))

    prev = Xf.copy()
    for _ in range(max_iter):
        # truncated SVD reconstruction
        U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
        Xr = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
        Xf[fill_idx] = Xr[fill_idx]     # only overwrite missing entries
        delta = np.linalg.norm(Xf[fill_idx] - prev[fill_idx])
        denom = np.linalg.norm(prev[fill_idx]) + 1e-9
        prev = Xf.copy()
        if delta / denom < tol:
            break
    return Xf


if __name__ == "__main__":
    all_rows = []

    print("Running MICE (IterativeImputer / BayesianRidge, max_iter=10)...")
    mice_rows = run_method("MICE", mice_impute)
    for r in mice_rows:
        r["dataset"] = f"MICE:{r['dataset']}"
        r["method"] = "SklearnMid"
    all_rows.extend(mice_rows)

    print("Running IterativeSVD (rank<=10, max_iter=50, tol=1e-4)...")
    svd_rows = run_method("IterativeSVD", itersvd_impute)
    for r in svd_rows:
        r["dataset"] = f"IterativeSVD:{r['dataset']}"
        r["method"] = "SklearnMid"
    all_rows.extend(svd_rows)

    print()
    summarize(all_rows)
    print(f"\n{len(all_rows)} result rows (2 methods x 7 datasets).")
