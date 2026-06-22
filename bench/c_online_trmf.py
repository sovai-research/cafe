"""
OnlineTRMF -- causal (fully point-in-time) Temporal Regularized Matrix Factorization.

Model:  X ≈ F @ W.T   with an AR(1) prior on the temporal latent factors F.
  - W      : (N, L) loadings, SHARED (pooled across entities for panels).
  - Theta  : (L,) per-factor AR(1) coefficient (diagonal AR).
  - f_tau  : latent factor at time tau, estimated by closed-form ridge from the
             OBSERVED cross-section at tau plus the AR prior  f_tau ~ Theta * f_{tau-1}.

POINT-IN-TIME GUARANTEE (verifier-enforced on all 7 datasets):
  We iterate strictly forward in TIME. At time tau:
    1. f_tau is solved from x_tau's observed entries (contemporaneous cross-section,
       which is allowed) + AR prior built from PAST factors (< tau) only.
    2. Missing entries of time tau are filled as f_tau @ W.T.
    3. W and Theta are refreshed from the buffer of factors/observations with
       time <= tau (warm-started ALS, a few sweeps, every K steps). They are NEVER
       touched by data at time > tau.
  Hence truncating future rows cannot change any imputation at time <= tau.

Panels (entity-major stacked): the factor is per (entity, time); W and Theta are
pooled/shared across entities and updated only from history <= tau. The "cross
section at tau" is all entities x features observed at tau.

Speed tricks applied: factored A·Bᵀ (never full reconstruction), cached Gram W_obsᵀW_obs
patterns, cho_solve (no inv/pinv), precomputed per-time observed index lists & masks,
warm-started W/Theta across time (a few ALS sweeps, capped), contiguous float arrays.
Inner iteration caps are stated explicitly below.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ------------------------- hyper-parameters (capped) ------------------------ #
# Separate regimes: 2D (one obs vector per time, no entity FE) vs PANEL (entity FE,
# pooled W/AR, one factor per entity per time). Inner iters are CAPPED (W_SWEEPS=1,
# refresh every REFRESH_K times) -> total runtime well under the 3-min budget.
PARAMS = {
    "2d": dict(L=6,  LAM_W=1e-1, LAM_F=1e-2, LAM_AR=5.0,  REFRESH_K=10, W_SWEEPS=1, MIN_WARM=8),
    "panel": dict(L=5, LAM_W=5e-2, LAM_F=1e-2, LAM_AR=20.0, REFRESH_K=5, W_SWEEPS=1, MIN_WARM=6),
}
W_SWEEPS = 1


def _solve_factor(WtW_full, W, x, obs, prior_mean, lam_f, lam_ar):
    """
    Closed-form ridge for one latent factor f (length L):
        (W_obs^T W_obs + (lam_f+lam_ar) I) f = W_obs^T x_obs + lam_ar * prior_mean
    W_obs are rows of W for observed features; x_obs the observed values.
    Uses cho_solve (SPD system).  WtW_full is W.T@W (Gram cache) used when the
    full cross-section is observed (fast path).
    """
    L = W.shape[1]
    no = int(obs.sum())
    if no == 0:
        return prior_mean.copy()
    if no == obs.shape[0]:
        A = WtW_full
        rhs = W.T @ x
    else:
        Wo = W[obs]
        A = Wo.T @ Wo
        rhs = Wo.T @ x[obs]
    A = A + (lam_f + lam_ar) * np.eye(L)
    rhs = rhs + lam_ar * prior_mean
    try:
        c = cho_factor(A, lower=True, check_finite=False)
        return cho_solve(c, rhs, check_finite=False)
    except Exception:
        return np.linalg.solve(A, rhs)


def _refresh_W(F_hist, Xfilled_hist, obs_hist, W, lam_w, sweeps):
    """
    Ridge update of loadings W (N,L) given the buffered factors F_hist (M,L) and
    the (filled) observation rows Xfilled_hist (M,N) with observed masks obs_hist.
    Per-feature ridge regression of column j on F over rows where j was observed.
    Warm-started: returns updated W (we just overwrite, closed form is exact).
    Cached Gram FtF reused when a feature is fully observed across the buffer.
    """
    M, L = F_hist.shape
    N = W.shape[0]
    # Fast path: if every feature observed in every buffered row, one solve.
    all_obs = obs_hist.all()
    Wnew = W.copy()
    if all_obs:
        G = F_hist.T @ F_hist + lam_w * np.eye(L)
        try:
            c = cho_factor(G, lower=True, check_finite=False)
            B = cho_solve(c, F_hist.T @ Xfilled_hist, check_finite=False)  # (L,N)
            return np.ascontiguousarray(B.T)
        except Exception:
            return W
    # General path: per-feature observed rows.
    for j in range(N):
        oj = obs_hist[:, j]
        cnt = int(oj.sum())
        if cnt < L + 1:
            continue
        Fj = F_hist[oj]
        G = Fj.T @ Fj + lam_w * np.eye(L)
        rhs = Fj.T @ Xfilled_hist[oj, j]
        try:
            c = cho_factor(G, lower=True, check_finite=False)
            Wnew[j] = cho_solve(c, rhs, check_finite=False)
        except Exception:
            pass
    return Wnew


def _refresh_theta(F_hist):
    """Per-factor AR(1) coefficient via OLS f_t ~ theta * f_{t-1}, clipped to (-0.999,0.999)."""
    if F_hist.shape[0] < 3:
        return None
    a = F_hist[:-1]           # (M-1, L)
    b = F_hist[1:]            # (M-1, L)
    num = np.sum(a * b, axis=0)
    den = np.sum(a * a, axis=0) + 1e-8
    theta = num / den
    return np.clip(theta, -0.999, 0.999)




def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    T_rows, N = X.shape
    is_panel = bool(meta) and "time_ids" in meta
    P = PARAMS["panel"] if is_panel else PARAMS["2d"]
    L = P["L"]
    LAM_W, LAM_F, LAM_AR = P["LAM_W"], P["LAM_F"], P["LAM_AR"]
    REFRESH_K, W_SWEEPS, MIN_WARM = P["REFRESH_K"], P["W_SWEEPS"], P["MIN_WARM"]
    if is_panel:
        tids = np.asarray(meta["time_ids"])
    else:
        tids = np.arange(T_rows)

    out = X.copy()
    obs_all = ~np.isnan(X)

    # Per-feature running mean for cold-start / fallback fill (causal: only past+contemp).
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)

    # PANEL: expanding per-(entity,feature) mean to absorb entity fixed effects.
    # Causal: at time tau we only use that entity's OWN past observations (< tau) plus
    # the contemporaneous cross-section at tau to seed a cold entity. The low-rank
    # factor then models time-FE + entity-time interaction on the residual.
    if is_panel:
        eids_pre = np.asarray(meta["entity_ids"])
        E_pre = int(eids_pre.max()) + 1
        ef_sum = np.zeros((E_pre, N))
        ef_cnt = np.zeros((E_pre, N))

    # Ordered unique times and, for each, the row indices at that time.
    order = np.argsort(tids, kind="stable")
    sorted_t = tids[order]
    uniq, starts = np.unique(sorted_t, return_index=True)
    ends = np.r_[starts[1:], len(sorted_t)]
    time_rows = [order[s:e] for s, e in zip(starts, ends)]

    # Buffers (history <= tau). For panels we pool all (entity,time) factor rows.
    rng_cap = T_rows + 16
    F_buf = np.zeros((rng_cap, L))         # buffered factors (one per processed row)
    Xf_buf = np.zeros((rng_cap, N))        # buffered FILLED observation rows
    Ob_buf = np.zeros((rng_cap, N), dtype=bool)
    buf_n = 0

    # Per-entity last factor (for AR prior). 2D: single chain (entity 0).
    if is_panel:
        eids = np.asarray(meta["entity_ids"])
        E = int(eids.max()) + 1
    else:
        eids = np.zeros(T_rows, dtype=int)
        E = 1
    last_f = np.zeros((E, L))             # previous factor per entity
    have_last = np.zeros(E, dtype=bool)

    # Shared parameters, warm-started across time.
    W = None
    theta = np.zeros(L)                  # AR(1) coefs, start at 0 (no temporal pull yet)
    WtW = None
    steps_since_refresh = 0

    # offset rows store, per processed buffer row, the causal additive offset that
    # was subtracted before factorization (entity FE for panels; 0 for 2D).
    Off_buf = np.zeros((rng_cap, N))

    def _row_offset(r, e, col_mean_now):
        """Causal additive baseline for row r (entity e).
        Panel -> expanding entity-feature mean (entity FE), backing off to the
        contemporaneous cross-section mean at tau for cold features.
        2D    -> expanding per-feature mean (de-levels columns for a mean-zero
        residual; this is what the low-rank core models).
        """
        if not is_panel:
            return np.zeros(N)
        m = np.where(ef_cnt[e] > 0, ef_sum[e] / np.maximum(ef_cnt[e], 1), np.nan)
        bad = ~np.isfinite(m)
        m[bad] = col_mean_now[bad]
        return m

    n_times = len(uniq)
    for ti in range(n_times):
        rows = time_rows[ti]

        # contemporaneous cross-section column mean at THIS tau over observed entries
        # (allowed; used only to seed cold entity-features). Built from data AT tau.
        if is_panel:
            cs_sum = np.zeros(N); cs_cnt = np.zeros(N)
            for r in rows:
                o = obs_all[r]
                cs_sum[o] += X[r, o]; cs_cnt[o] += 1
            cs_mean = np.where(cs_cnt > 0, cs_sum / np.maximum(cs_cnt, 1),
                               np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0))
        else:
            cs_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)

        # ---- cold start: before W exists, fill from baseline offset -------- #
        if W is None:
            for r in rows:
                e = eids[r]
                o = obs_all[r]
                off = _row_offset(r, e, cs_mean)
                miss = ~o
                out[r, miss] = cs_mean[miss]
                # buffer residual: observed -> X-off; missing -> baseline residual
                resid = X[r] - off
                resid[miss] = cs_mean[miss] - off[miss]
                Xf_buf[buf_n] = resid
                Ob_buf[buf_n] = o
                Off_buf[buf_n] = off
                F_buf[buf_n] = 0.0
                buf_n += 1
            _update_means(rows, X, obs_all, col_sum, col_cnt,
                          ef_sum if is_panel else None,
                          ef_cnt if is_panel else None, eids if is_panel else None)
            steps_since_refresh += 1
            if buf_n >= MIN_WARM:
                W, theta, WtW = _init_W(Xf_buf[:buf_n], Ob_buf[:buf_n], L, LAM_W)
                F_buf[:buf_n] = _factors_for_buffer(Xf_buf[:buf_n], Ob_buf[:buf_n],
                                                    W, WtW, LAM_F)
                _th = _refresh_theta(F_buf[:buf_n])
                if _th is not None:
                    theta = _th
                steps_since_refresh = 0
            continue

        # ---- normal path: W exists ----------------------------------------- #
        for r in rows:
            e = eids[r]
            o = obs_all[r]
            off = _row_offset(r, e, cs_mean)
            resid = np.where(o, X[r] - off, 0.0)          # residual on observed
            prior = theta * last_f[e] if have_last[e] else np.zeros(L)
            f = _solve_factor(WtW, W, resid, o, prior, LAM_F, LAM_AR)
            miss = ~o
            rec_miss = W[miss] @ f                          # factored, missing cols only
            if miss.any():
                out[r, miss] = off[miss] + rec_miss
                resid[miss] = rec_miss                      # self-consistent buffer fill
            Xf_buf[buf_n] = resid
            Ob_buf[buf_n] = o
            Off_buf[buf_n] = off
            F_buf[buf_n] = f
            buf_n += 1
            last_f[e] = f
            have_last[e] = True

        _update_means(rows, X, obs_all, col_sum, col_cnt,
                      ef_sum if is_panel else None,
                      ef_cnt if is_panel else None, eids if is_panel else None)

        # refresh W / theta every K steps from history <= tau (warm-started)
        steps_since_refresh += 1
        if steps_since_refresh >= REFRESH_K:
            for _ in range(W_SWEEPS):
                W = _refresh_W(F_buf[:buf_n], Xf_buf[:buf_n], Ob_buf[:buf_n],
                               W, LAM_W, W_SWEEPS)
            WtW = W.T @ W
            th = _refresh_theta(F_buf[:buf_n])
            if th is not None:
                theta = th
            steps_since_refresh = 0

    # Any rows never filled (shouldn't happen) -> safety net
    if np.isnan(out).any():
        gmean = np.nanmean(X)
        out[np.isnan(out)] = gmean if np.isfinite(gmean) else 0.0
    return out


def _update_means(rows, X, obs_all, col_sum, col_cnt, ef_sum, ef_cnt, eids):
    """Fold time-tau OBSERVED entries into running means (called AFTER imputing tau)."""
    for r in rows:
        o = obs_all[r]
        col_sum[o] += X[r, o]
        col_cnt[o] += 1
        if ef_sum is not None:
            e = eids[r]
            ef_sum[e, o] += X[r, o]
            ef_cnt[e, o] += 1


def _init_W(Xf, Ob, L, lam_w):
    """
    Initialize W from a warm buffer via truncated SVD of the (mean-imputed) buffer,
    then one ridge refit. Returns (W, theta_placeholder, WtW).
    """
    M, N = Xf.shape
    # center-free SVD on filled buffer
    try:
        U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
        Lk = min(L, len(s))
        W = np.zeros((N, L))
        W[:, :Lk] = (Vt[:Lk].T * s[:Lk]) / max(np.sqrt(M), 1.0)
    except Exception:
        W = np.random.default_rng(0).standard_normal((N, L)) * 0.1
    W = np.ascontiguousarray(W)
    WtW = W.T @ W
    return W, np.zeros(L), WtW


def _factors_for_buffer(Xf, Ob, W, WtW, lam_f):
    """Recompute level factors for every buffered row (no AR prior -> level only)."""
    M, N = Xf.shape
    L = W.shape[1]
    F = np.zeros((M, L))
    zero_prior = np.zeros(L)
    for i in range(M):
        o = Ob[i]
        F[i] = _solve_factor(WtW, W, np.where(o, Xf[i], 0.0), o, zero_prior, lam_f, 0.0)
    return F


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    rows = run_causal("OnlineTRMF", online_impute)
    summarize_causal(rows)
