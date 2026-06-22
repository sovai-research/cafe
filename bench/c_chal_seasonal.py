"""
Chal_seasonal — causal seasonal component for univariate (1D) series.

Targets 1d_seasonal_block (champion CausalRouter: corr 0.4461). The 1D series in
the suite is a strong daily(24)+weekly(168) cycle on a slow linear drift with
contiguous block gaps. A low-rank MF over a single feature (N=1) is degenerate
and falls back to running-mean / weak AR, so it cannot reconstruct across a gap
that spans multiple seasonal cycles.

This config GATES on data type with a cheap up-front signal:
  - PANEL (entity_ids, >1 entity)            -> CausalFE  (unchanged from router)
  - 2D, N>1                                   -> OnlineTRMF (unchanged from router)
  - 1D univariate (N==1)                      -> causal Fourier + trend regressor
So every non-1D case is byte-identical to the champion (same speed, same corr);
only the two 1D cases are touched, and only the seasonal one changes materially.

CAUSAL (point-in-time) GUARANTEE for the 1D path:
  We iterate strictly forward in time. The Fourier/trend coefficients used to fill
  the missing entry at time t are fit by ridge OLS on OBSERVED samples with index
  < t only (an expanding window). The dominant period(s) are detected from the
  autocorrelation of observed history (< t) as well. Truncating future rows cannot
  change any fill at time <= t. We refit only every REFIT_K observed steps (and
  whenever a gap starts after the model is stale) to stay fast; between refits the
  last (past-only) coefficients are used. Observed entries are returned unchanged.

Falls back to the AR baseline (last observed value) before enough history exists,
and to OnlineTRMF if the seasonal fit degenerates.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel

# ---- hyper-parameters (capped, 1D path only) ---- #
MIN_WARM = 60          # need this many observed samples before trusting Fourier fit
REFIT_K = 24           # refit coefficients every K newly-observed samples
MAX_HARM = 4           # harmonics per detected period
RIDGE = 1e-3           # ridge on Fourier/trend coefficients
N_PERIODS = 2          # top-N autocorrelation periods to model


def _acf_candidates(y):
    """Rough candidate periods from autocorrelation local maxima of de-trended y.
    Returns a list of integer lags, strongest peak first."""
    m = len(y)
    if m < 8:
        return []
    yc = y - y.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * m)))
    F = np.fft.rfft(yc, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:m]
    if ac[0] <= 0:
        return []
    ac = ac / ac[0]
    lo = 3
    hi = min(m - 2, m // 2 + 1)
    if hi <= lo:
        return []
    peaks = []
    for i in range(lo + 1, hi - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.08:
            peaks.append((ac[i], i))
    peaks.sort(reverse=True)
    return [lag for _, lag in peaks]


def _residual(t_obs, y_obs, periods, max_harm, ridge):
    """In-sample RSS for a Fourier+trend fit over the given periods (causal:
    only observed history is passed in)."""
    A = _design(t_obs, periods, max_harm)
    G = A.T @ A
    reg = ridge * np.trace(G) / G.shape[0]
    G = G + reg * np.eye(G.shape[0])
    try:
        beta = np.linalg.solve(G, A.T @ y_obs)
    except Exception:
        beta = np.linalg.lstsq(A, y_obs, rcond=None)[0]
    r = y_obs - A @ beta
    return float(r @ r)


def _detect_periods(t_obs, y_obs, n=N_PERIODS):
    """Greedy, FIT-DRIVEN period selection. Fourier extrapolation is extremely
    sensitive to the exact integer period (P=23 vs 24 flips corr from -0.15 to
    +0.91), so we do NOT trust raw ACF lags — we use them only to seed integer
    neighborhood searches and keep a period only if it materially cuts the
    in-sample residual. Causal: uses observed history (< t) only."""
    # ACF candidates from linearly de-trended observed history
    if len(t_obs) >= 4:
        pc = np.polyfit(t_obs, y_obs, 1)
        det = y_obs - (pc[0] * t_obs + pc[1])
    else:
        det = y_obs
    cands = _acf_candidates(det)
    if not cands:
        return []
    chosen = []
    base0 = _residual(t_obs, y_obs, [], MAX_HARM, RIDGE)  # trend-only RSS
    base = base0
    tmax = t_obs[-1] - t_obs[0]
    for c in cands:
        if len(chosen) >= n:
            break
        # refine c over an integer neighborhood (wider for longer periods, since
        # ACF lag precision degrades). Also try multiples of already-chosen
        # fundamentals (e.g. weekly = 7*daily) which ACF often resolves poorly.
        win = max(2, int(round(0.06 * c)))
        cand_set = set(range(max(3, c - win), c + win + 1))
        for q in chosen:
            for k in range(2, 9):
                cand_set.add(q * k)
        best_p, best_rss = None, base
        for P in cand_set:
            if P < 3 or P > tmax:
                continue
            if any(abs(P - q) <= 2 for q in chosen):
                continue
            rss = _residual(t_obs, y_obs, chosen + [P], MAX_HARM, RIDGE)
            if rss < best_rss:
                best_rss, best_p = rss, P
        # accept only if it explains >= 5% of the current residual
        if best_p is not None and best_rss < base * 0.95:
            chosen.append(best_p)
            base = best_rss
    return chosen


def _design(t_idx, periods, max_harm):
    """Design matrix: [1, t_norm, sin/cos harmonics for each period]."""
    t_idx = np.asarray(t_idx, float)
    cols = [np.ones_like(t_idx), t_idx]
    for P in periods:
        K = min(max_harm, max(1, int(P // 2)))
        for k in range(1, K + 1):
            w = 2 * np.pi * k / P
            cols.append(np.sin(w * t_idx))
            cols.append(np.cos(w * t_idx))
    return np.column_stack(cols)


def _fit(t_obs, y_obs, periods, max_harm, ridge):
    A = _design(t_obs, periods, max_harm)
    # normalize trend column scale for conditioning
    G = A.T @ A
    reg = ridge * np.trace(G) / G.shape[0]
    G = G + reg * np.eye(G.shape[0])
    try:
        beta = np.linalg.solve(G, A.T @ y_obs)
    except Exception:
        beta = np.linalg.lstsq(A, y_obs, rcond=None)[0]
    return beta


def _impute_1d(X):
    """Causal Fourier+trend imputation for an (T,1) series. Returns (out, used)
    where `used` is True iff a seasonal model was ever fit (a real period was
    detected). If no period is ever found the column is NOT seasonal and the
    caller should fall back to the temporal core (TRMF). Per-column generalizes
    if N>1 but this path is only entered when N==1."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    used_any = False
    for j in range(N):
        col = X[:, j]
        obs = ~np.isnan(col)
        idx_all = np.arange(T)
        beta = None
        periods = []
        last_val = None
        run_sum = 0.0
        run_cnt = 0
        since_fit = 0
        # observed buffers (history)
        t_hist = []
        y_hist = []
        for t in range(T):
            if obs[t]:
                # observed: update history AFTER it is available (it's contemporaneous
                # for ITSELF, but we never use t to fill t; we use it for future t').
                last_val = col[t]
                run_sum += col[t]
                run_cnt += 1
                t_hist.append(t)
                y_hist.append(col[t])
                since_fit += 1
                # periodic refit from history <= t (used only for future fills)
                if run_cnt >= MIN_WARM and (beta is None or since_fit >= REFIT_K):
                    ya = np.array(y_hist)
                    ta = np.array(t_hist, float)
                    periods = _detect_periods(ta, ya)
                    if periods:
                        beta = _fit(ta, ya, periods, MAX_HARM, RIDGE)
                        used_any = True
                    since_fit = 0
            else:
                # missing at t: fill using PAST-ONLY model.
                if beta is not None and periods:
                    A = _design([t], periods, MAX_HARM)
                    out[t, j] = float((A @ beta)[0])
                elif last_val is not None:
                    out[t, j] = last_val          # causal carry-forward
                elif run_cnt > 0:
                    out[t, j] = run_sum / run_cnt
                else:
                    out[t, j] = 0.0
        # safety net
        bad = ~np.isfinite(out[:, j])
        if bad.any():
            gm = np.nanmean(col)
            out[bad, j] = gm if np.isfinite(gm) else 0.0
    return out, used_any


def online_impute(X, meta):
    if _is_panel(meta):
        return _fe(X, meta)
    Xn = np.asarray(X, float)
    if Xn.shape[1] == 1:
        try:
            out, used = _impute_1d(Xn)
            if used:                       # a genuine seasonal period was found
                return out
        except Exception:
            pass
        return _trmf(X, meta)              # non-seasonal 1D -> temporal core
    return _trmf(X, meta)


if __name__ == "__main__":
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_seasonal", online_impute))
