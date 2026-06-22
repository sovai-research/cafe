"""
CausalXSecReg -- pure CONTEMPORANEOUS cross-sectional ridge regression imputer.

Goal: isolate how much the cross-section "free lunch" alone buys, with NO temporal
modeling at all. For each target feature i we maintain a ridge regression that
predicts feature i from the OTHER features. Coefficients are estimated ONLY from
rows at times <= tau, using a running (expanding / warm-started) cross-product
matrix G = [1,x]^T [1,x] accumulated over fully-observed rows. We solve via
cho_solve (no inv/pinv).

Point-in-time discipline:
  - Iterate by TIME (group panel entities sharing a time together).
  - Parameters at time tau use ONLY complete rows with time <= tau, INCLUDING the
    contemporaneous complete rows AT tau (the cross-section is the allowed lifeline,
    and including tau's own complete cross-section is causal: removing future rows
    leaves tau's complete rows unchanged).
  - To impute (e,tau,i): regress on the OTHER features observed at the SAME (e,tau)
    using the current betas restricted to the co-observed columns. If too few
    co-observed features (or betas not yet warm), fall back to the expanding mean
    of feature i over observed values at time <= tau.

This is contemporaneous-only by construction: it never looks at past values of the
same series for the regression itself (only the running covariance pools past+present
COMPLETE rows to estimate cross-feature structure). It should shine on MCAR (where
each missing cell almost always has its co-features observed) and degrade on BLOCK
(where a feature can be dark for a whole stretch, killing the co-observation lifeline
and forcing the expanding-mean fallback).
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
from scipy.linalg import cho_factor, cho_solve

RIDGE = 1e-2          # ridge penalty on the (non-intercept) coefficients
MIN_COOBS = 2         # need at least this many co-observed OTHER features to regress
MIN_ROWS = 8          # need at least this many complete rows accumulated to trust betas


def online_impute(X, meta):
    X = np.ascontiguousarray(X, dtype=float)
    T, N = X.shape
    out = X.copy()

    tid = meta["time_ids"] if (meta and "time_ids" in meta) else np.arange(T)
    tid = np.asarray(tid)

    # Per-time row-index lists (rows of X sharing each time), in time order.
    order = np.argsort(tid, kind="stable")
    sorted_t = tid[order]
    # boundaries of contiguous equal-time groups in the sorted order
    uniq_times, starts = np.unique(sorted_t, return_index=True)
    starts = list(starts) + [T]
    time_groups = [order[starts[k]:starts[k + 1]] for k in range(len(uniq_times))]

    F = N
    # Running cross-product of [1, x] over COMPLETE rows seen so far (time <= tau).
    # G has shape (F+1, F+1): G[0,0]=count, G[0,1:]=sum x, G[1:,1:]=sum x x^T.
    G = np.zeros((F + 1, F + 1))
    # Expanding running mean of each feature over OBSERVED values (time <= tau).
    feat_sum = np.zeros(F)
    feat_cnt = np.zeros(F)
    global_sum = 0.0
    global_cnt = 0

    # Cached betas per target feature; recomputed once per time step (after we know
    # which complete rows are available up to and including tau). beta[i] has length
    # F+1: [intercept, coef_for_each_OTHER_feature] with coef at position i unused.
    betas = None
    betas_valid = False

    for grp in time_groups:
        # --- 1) refresh betas using complete rows accumulated through PREVIOUS times
        #         plus this time's complete rows (added below first). We add this
        #         time's complete cross-section to G BEFORE imputing, so the
        #         contemporaneous lifeline is used. This is causal. ---
        rows_obs = ~np.isnan(X[grp])                     # (g, F) observed mask
        complete = rows_obs.all(axis=1)
        if complete.any():
            Xc = X[grp][complete]                        # (c, F)
            c = Xc.shape[0]
            G[0, 0] += c
            sx = Xc.sum(axis=0)
            G[0, 1:] += sx
            G[1:, 0] += sx
            G[1:, 1:] += Xc.T @ Xc

        # update expanding per-feature means with this time's observed values
        Xg = X[grp]
        obs = rows_obs
        if obs.any():
            vals = np.where(obs, Xg, 0.0)
            feat_sum += vals.sum(axis=0)
            feat_cnt += obs.sum(axis=0)
            global_sum += Xg[obs].sum()
            global_cnt += int(obs.sum())

        # solve ridge betas for every target feature from current G
        n_rows = G[0, 0]
        if n_rows >= MIN_ROWS:
            betas = _solve_all_betas(G, F)
            betas_valid = True

        # --- 2) impute the missing cells AT this time using contemporaneous cross-
        #         section + current betas; fall back to expanding mean otherwise ---
        feat_mean = np.where(feat_cnt > 0, feat_sum / np.maximum(feat_cnt, 1),
                             global_sum / max(global_cnt, 1))
        for li in range(len(grp)):
            r = grp[li]
            row = Xg[li]
            ro = obs[li]
            miss_cols = np.where(~ro)[0]
            if miss_cols.size == 0:
                continue
            for i in miss_cols:
                pred = np.nan
                if betas_valid:
                    b = betas[i]                         # length F+1
                    # co-observed OTHER features
                    co = ro.copy()
                    co[i] = False                        # exclude target itself
                    n_co = int(co.sum())
                    if n_co >= MIN_COOBS:
                        # pred = intercept + sum_j b[1+j]*x_j over co-observed j
                        pred = b[0] + float(b[1:][co] @ row[co])
                if not np.isfinite(pred):
                    pred = feat_mean[i]
                out[r, i] = pred

    # safety: any residual NaN (e.g. all-cold start) -> global mean / 0
    if np.isnan(out).any():
        gm = global_sum / max(global_cnt, 1)
        out[np.isnan(out)] = gm
    return out


def _solve_all_betas(G, F):
    """
    For each target feature i, ridge-regress i on intercept + all OTHER features,
    using the running moment matrix G (over complete rows). Returns array (F, F+1):
    row i = [intercept, b_0, ..., b_{F-1}] where b_i is unused (the target column is
    dropped from the design). Solve via cho_solve on the (F)x(F) sub-system that
    keeps intercept + (F-1) predictors.

    G layout: index 0 = intercept (count/sums), indices 1..F = features 0..F-1.
    Design columns for target i = {0 (intercept)} U {1..F} minus {i+1}.
    Normal equations: (D^T D + lambda*P) beta = D^T y, where D^T D and D^T y are
    sub-blocks of G (with y = feature i column => moments in row/col i+1).
    """
    betas = np.zeros((F, F + 1))
    # ridge regularizer applied to non-intercept coefficients only
    for i in range(F):
        cols = [0] + [j + 1 for j in range(F) if j != i]      # design indices in G
        cols = np.asarray(cols)
        yidx = i + 1
        # A = D^T D  (len(cols) x len(cols)); rhs = D^T y
        A = G[np.ix_(cols, cols)].copy()
        rhs = G[cols, yidx].copy()
        # ridge on non-intercept diagonal entries (skip index 0 = intercept)
        d = np.diag_indices_from(A)
        reg = np.full(A.shape[0], RIDGE)
        reg[0] = 0.0                                          # no penalty on intercept
        A[d] += reg
        try:
            cf = cho_factor(A, lower=True, check_finite=False)
            sol = cho_solve(cf, rhs, check_finite=False)      # [intercept, coefs...]
        except Exception:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        # scatter sol back into length F+1 vector [intercept, per-feature coefs]
        b = np.zeros(F + 1)
        b[0] = sol[0]
        b[1:][[j for j in range(F) if j != i]] = sol[1:]
        betas[i] = b
    return betas


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    rows = run_causal("CausalXSecReg", online_impute)
    summarize_causal(rows)
    print()
    for r in rows:
        print(f"{r['dataset']:18s} causal={r['causal']} detail={r['causal_detail']}")
