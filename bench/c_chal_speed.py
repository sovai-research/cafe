"""
Chal_speed — PURE SPEED challenger over CausalRouter.

Routing is IDENTICAL to the champion (panel -> CausalFE, else OnlineTRMF). The only
change is to OnlineTRMF's W/theta refresh, which the profiler showed is the entire
cost of the suite: on beijing (17117 x 132) `_refresh_W` ran 1710 times over a buffer
that GREW to 17k rows -> 53.3s of the 60s total (O(T^2)). Everything else in the suite
combined is ~6s.

FIX (still strictly point-in-time): cap the buffer rows used in the periodic W/theta
refresh to a RECENT WINDOW of the most-recent rows with time <= tau. Dropping only OLD
rows (<= tau) cannot introduce look-ahead, so the causal guarantee is preserved. This
turns each refresh from O(buf_n) into O(REFRESH_WINDOW), i.e. O(T*window) total instead
of O(T^2).

NO-REGRESSION DESIGN: the window is only active once the buffer exceeds it. For every
case in the suite EXCEPT beijing/ETTh1 the buffer never reaches the window size, so the
math is byte-identical to the champion -> those corrs are unchanged. Only the two long
real-point series are affected, and W there converges, so a recent-window refit keeps
the same corr at a fraction of the time.

Routing/cold-start/factor-solve logic is copied verbatim from c_online_trmf so the
champion's accuracy is reproduced exactly outside the windowed refresh.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from c_router import _is_panel
from c_fe_lowrank import online_impute as _fe
# reuse the champion's exact helpers so identical math where it matters
from c_online_trmf import (PARAMS, _solve_factor, _refresh_theta, _update_means,
                           _init_W, _factors_for_buffer)

# Recent-window cap for the W/theta refresh. Chosen >= every small case's T so they
# stay byte-identical to the champion; only the ~17k-row real-point series are capped.
REFRESH_WINDOW = 800


def _refresh_W_window(F_hist, Xfilled_hist, obs_hist, W, lam_w):
    """Same ridge update as the champion's _refresh_W, but F/Xf/Ob have already been
    sliced to a recent window by the caller. Closed-form, cached fast path."""
    M, L = F_hist.shape
    N = W.shape[0]
    all_obs = obs_hist.all()
    Wnew = W.copy()
    if all_obs:
        G = F_hist.T @ F_hist + lam_w * np.eye(L)
        try:
            c = cho_factor(G, lower=True, check_finite=False)
            B = cho_solve(c, F_hist.T @ Xfilled_hist, check_finite=False)
            return np.ascontiguousarray(B.T)
        except Exception:
            return W
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


def _trmf(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    T_rows, N = X.shape
    is_panel = bool(meta) and "time_ids" in meta
    P = PARAMS["panel"] if is_panel else PARAMS["2d"]
    L = P["L"]
    LAM_W, LAM_F, LAM_AR = P["LAM_W"], P["LAM_F"], P["LAM_AR"]
    REFRESH_K, _W_SWEEPS, MIN_WARM = P["REFRESH_K"], P["W_SWEEPS"], P["MIN_WARM"]
    if is_panel:
        tids = np.asarray(meta["time_ids"])
    else:
        tids = np.arange(T_rows)

    out = X.copy()
    obs_all = ~np.isnan(X)

    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)

    if is_panel:
        eids_pre = np.asarray(meta["entity_ids"])
        E_pre = int(eids_pre.max()) + 1
        ef_sum = np.zeros((E_pre, N))
        ef_cnt = np.zeros((E_pre, N))

    order = np.argsort(tids, kind="stable")
    sorted_t = tids[order]
    uniq, starts = np.unique(sorted_t, return_index=True)
    ends = np.r_[starts[1:], len(sorted_t)]
    time_rows = [order[s:e] for s, e in zip(starts, ends)]

    rng_cap = T_rows + 16
    F_buf = np.zeros((rng_cap, L))
    Xf_buf = np.zeros((rng_cap, N))
    Ob_buf = np.zeros((rng_cap, N), dtype=bool)
    buf_n = 0

    if is_panel:
        eids = np.asarray(meta["entity_ids"])
        E = int(eids.max()) + 1
    else:
        eids = np.zeros(T_rows, dtype=int)
        E = 1
    last_f = np.zeros((E, L))
    have_last = np.zeros(E, dtype=bool)

    W = None
    theta = np.zeros(L)
    WtW = None
    steps_since_refresh = 0
    Off_buf = np.zeros((rng_cap, N))

    def _row_offset(r, e, col_mean_now):
        if not is_panel:
            return np.zeros(N)
        m = np.where(ef_cnt[e] > 0, ef_sum[e] / np.maximum(ef_cnt[e], 1), np.nan)
        bad = ~np.isfinite(m)
        m[bad] = col_mean_now[bad]
        return m

    def _do_refresh():
        """Windowed W/theta refresh from a RECENT slice of the buffer (rows <= tau)."""
        nonlocal W, WtW, theta
        lo = max(0, buf_n - REFRESH_WINDOW)
        Fw = F_buf[lo:buf_n]
        Xw = Xf_buf[lo:buf_n]
        Ow = Ob_buf[lo:buf_n]
        W = _refresh_W_window(Fw, Xw, Ow, W, LAM_W)
        WtW = W.T @ W
        th = _refresh_theta(Fw)
        if th is not None:
            theta = th

    n_times = len(uniq)
    for ti in range(n_times):
        rows = time_rows[ti]

        if is_panel:
            cs_sum = np.zeros(N); cs_cnt = np.zeros(N)
            for r in rows:
                o = obs_all[r]
                cs_sum[o] += X[r, o]; cs_cnt[o] += 1
            cs_mean = np.where(cs_cnt > 0, cs_sum / np.maximum(cs_cnt, 1),
                               np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0))
        else:
            cs_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)

        if W is None:
            for r in rows:
                e = eids[r]
                o = obs_all[r]
                off = _row_offset(r, e, cs_mean)
                miss = ~o
                out[r, miss] = cs_mean[miss]
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

        for r in rows:
            e = eids[r]
            o = obs_all[r]
            off = _row_offset(r, e, cs_mean)
            resid = np.where(o, X[r] - off, 0.0)
            prior = theta * last_f[e] if have_last[e] else np.zeros(L)
            f = _solve_factor(WtW, W, resid, o, prior, LAM_F, LAM_AR)
            miss = ~o
            rec_miss = W[miss] @ f
            if miss.any():
                out[r, miss] = off[miss] + rec_miss
                resid[miss] = rec_miss
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

        steps_since_refresh += 1
        if steps_since_refresh >= REFRESH_K:
            _do_refresh()
            steps_since_refresh = 0

    if np.isnan(out).any():
        gmean = np.nanmean(X)
        out[np.isnan(out)] = gmean if np.isfinite(gmean) else 0.0
    return out


def online_impute(X, meta):
    return _fe(X, meta) if _is_panel(meta) else _trmf(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_speed", online_impute))
