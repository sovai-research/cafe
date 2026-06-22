import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from harness import run_method, summarize
from sklearn.impute import KNNImputer


def _col_mean_fill(X):
    """Fallback fill for any column that is entirely NaN."""
    col = np.nanmean(X, axis=0)
    g = np.nanmean(X) if np.isfinite(np.nanmean(X)) else 0.0
    col = np.where(np.isfinite(col), col, g)
    return col


def linear_interp(X, meta):
    """Per-column time-aware linear interpolation with ffill/bfill at edges."""
    T, N = X.shape
    out = X.copy()
    t = np.arange(T, dtype=float)
    colmean = _col_mean_fill(X)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        k = obs.sum()
        if k == 0:
            out[:, j] = colmean[j]
            continue
        if k == 1:
            col[~obs] = col[obs][0]
            out[:, j] = col
            continue
        # interp handles interior linearly and clamps edges to nearest obs (ffill/bfill)
        out[:, j] = np.interp(t, t[obs], col[obs])
    return out


def ffill_bfill(X, meta):
    """Forward-fill then backward-fill per column (carry last/next observation)."""
    T, N = X.shape
    out = X.copy()
    colmean = _col_mean_fill(X)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        if obs.sum() == 0:
            out[:, j] = colmean[j]
            continue
        idx = np.where(obs, np.arange(T), -1)
        # forward fill
        fwd = np.maximum.accumulate(idx)
        # backward fill index
        ridx = np.where(obs, np.arange(T), T)
        bwd = np.minimum.accumulate(ridx[::-1])[::-1]
        fill_idx = np.where(fwd >= 0, fwd, bwd)
        out[:, j] = col[fill_idx]
    return out


def knn_impute(X, meta):
    """sklearn KNNImputer(n_neighbors=5). Falls back to col-mean for all-NaN cols."""
    out = KNNImputer(n_neighbors=5).fit_transform(X)
    # KNNImputer drops all-NaN columns silently in some versions; guard shape.
    if out.shape[1] != X.shape[1]:
        # rebuild with col-mean for fully-missing columns
        colmean = _col_mean_fill(X)
        full = X.copy()
        nanmask = np.isnan(full)
        # column-mean placeholder then run KNN on rest
        for j in range(X.shape[1]):
            if np.all(np.isnan(X[:, j])):
                full[:, j] = colmean[j]
        sub = KNNImputer(n_neighbors=5).fit_transform(full)
        out = sub
    return out


if __name__ == "__main__":
    all_rows = []
    for name, fn in [("LinearInterp", linear_interp),
                     ("FFillBFill", ffill_bfill),
                     ("KNN", knn_impute)]:
        rows = run_method(name, fn)
        summarize(rows)
        print()
        all_rows.extend(rows)
