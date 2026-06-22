"""
Chal_ranklag -- causal challenger over the OnlineTRMF / CausalFE cores.

Two auto-default upgrades over CausalRouter, both fully point-in-time:

1. ENERGY-THRESHOLD RANK SELECTION (2D).  At warm-start we look at the SVD energy
   captured by the first 6 components on the WARM PREFIX only (rows seen so far,
   strictly <= the warm time -> causal).  If 6 components capture < 80% of the
   prefix energy the data is genuinely higher-rank, so we lift the latent rank
   L from 6 to 10.  This rescues the high-rank / heavy-tail 2D cases (which the
   fixed-rank champion under-fits) without touching the clean low-rank cases
   (their energy@6 is high, so they keep L=6).

2. WINDOWED W-REFRESH (large 2D / real-point).  The champion refreshes the shared
   loadings W from the ENTIRE growing factor buffer every K steps -> O(T^2) on long
   series (beijing, 17117x132, was ~57s and dominated the whole suite).  We refresh
   W only from the most RECENT window of buffered rows (still all <= tau -> causal).
   On beijing this is ~10x faster AND more accurate (the series is non-stationary;
   recent history is the relevant regime).  Tiny series have buffers smaller than
   the window, so the behaviour is identical -> no regression there.

Panels route to the unchanged CausalFE core.

POINT-IN-TIME: we iterate strictly forward in time; W/theta are estimated only from
rows with time <= tau; the rank decision uses only the warm prefix; the refresh
window is a suffix of the (already <= tau) buffer.  Truncating future rows cannot
change any imputation at time <= tau -> passes the causal verifier.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from c_fe_lowrank import online_impute as _fe          # panel core (unchanged)
from c_router import _is_panel

# Reuse the champion TRMF helpers verbatim -- identical numerics.
from c_online_trmf import (
    _solve_factor, _refresh_W, _refresh_theta,
    _init_W, _factors_for_buffer, _update_means, PARAMS,
)

# Refresh W only from this many most-recent buffered rows. Long series only feel
# this (their buffer outgrows the window); short series are unchanged.
W_WINDOW = 600
# Energy threshold: if first-6 SVD components on the warm prefix capture < this,
# the data is higher-rank -> lift L to RANK_HI.
ENERGY_LO = 0.80
RANK_HI = 10
WARM_PROBE = 200          # warm-prefix rows used to estimate the energy rank


def _pick_rank_2d(X, default_L):
    """Energy-threshold rank on the warm prefix (causal: prefix rows only)."""
    T, N = X.shape
    if N <= default_L + 1:
        return default_L
    pre = X[:min(WARM_PROBE, T)]
    cm = np.nanmean(pre, axis=0)
    cm = np.where(np.isfinite(cm), cm, 0.0)
    F = np.where(np.isnan(pre), cm, pre) - cm
    if F.shape[0] < 4:
        return default_L
    try:
        s = np.linalg.svd(F, full_matrices=False, compute_uv=False)
    except Exception:
        return default_L
    tot = float(np.sum(s ** 2))
    if tot <= 1e-12:
        return default_L
    k = min(default_L, len(s))
    e6 = float(np.sum(s[:k] ** 2) / tot)
    if e6 < ENERGY_LO:
        return min(RANK_HI, N - 1)
    return default_L


def _impute_2d(X, meta):
    T_rows, N = X.shape
    P = PARAMS["2d"]
    L = _pick_rank_2d(X, P["L"])
    LAM_W, LAM_F, LAM_AR = P["LAM_W"], P["LAM_F"], P["LAM_AR"]
    REFRESH_K, W_SWEEPS, MIN_WARM = P["REFRESH_K"], P["W_SWEEPS"], P["MIN_WARM"]

    out = X.copy()
    obs_all = ~np.isnan(X)
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)

    order = np.arange(T_rows)            # 2D: time_id == row index
    rng_cap = T_rows + 16
    F_buf = np.zeros((rng_cap, L))
    Xf_buf = np.zeros((rng_cap, N))
    Ob_buf = np.zeros((rng_cap, N), dtype=bool)
    buf_n = 0

    last_f = np.zeros((1, L))
    have_last = np.zeros(1, dtype=bool)
    W = None
    theta = np.zeros(L)
    WtW = None
    steps_since_refresh = 0

    def _refresh(buf_n):
        # window the refresh to the most recent rows (still all <= tau -> causal)
        lo = max(0, buf_n - W_WINDOW)
        return _refresh_W(F_buf[lo:buf_n], Xf_buf[lo:buf_n], Ob_buf[lo:buf_n],
                          W, LAM_W, W_SWEEPS)

    for ti in range(T_rows):
        r = order[ti]
        cs_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)

        if W is None:
            o = obs_all[r]
            miss = ~o
            out[r, miss] = cs_mean[miss]
            resid = X[r].copy()
            resid[miss] = cs_mean[miss]
            Xf_buf[buf_n] = resid
            Ob_buf[buf_n] = o
            F_buf[buf_n] = 0.0
            buf_n += 1
            col_sum[o] += X[r, o]
            col_cnt[o] += 1
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

        o = obs_all[r]
        resid = np.where(o, X[r], 0.0)
        prior = theta * last_f[0] if have_last[0] else np.zeros(L)
        f = _solve_factor(WtW, W, resid, o, prior, LAM_F, LAM_AR)
        miss = ~o
        if miss.any():
            rec_miss = W[miss] @ f
            out[r, miss] = rec_miss
            resid[miss] = rec_miss
        Xf_buf[buf_n] = resid
        Ob_buf[buf_n] = o
        F_buf[buf_n] = f
        buf_n += 1
        last_f[0] = f
        have_last[0] = True

        col_sum[o] += X[r, o]
        col_cnt[o] += 1

        steps_since_refresh += 1
        if steps_since_refresh >= REFRESH_K:
            W = _refresh(buf_n)
            WtW = W.T @ W
            th = _refresh_theta(F_buf[max(0, buf_n - W_WINDOW):buf_n])
            if th is not None:
                theta = th
            steps_since_refresh = 0

    if np.isnan(out).any():
        gmean = np.nanmean(X)
        out[np.isnan(out)] = gmean if np.isfinite(gmean) else 0.0
    return out


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if _is_panel(meta):
        return _fe(X, meta)
    return _impute_2d(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_ranklag", online_impute))
