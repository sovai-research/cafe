"""
w3_kalman_seasonal -- Trigonometric STRUCTURAL state-space seasonal model with a
forward Kalman FILTER, for univariate (N==1) series. Research block #3.

Targets 1d_seasonal_block (corr 0.662 under c_chal_uni). The series is a strong
daily(24)+weekly(168) cycle on a slow linear drift with contiguous block gaps. A
multi-cycle gap cannot be reconstructed by carry-forward / weak AR; a structural
seasonal model that PROPAGATES the harmonic phase through the gap can.

MODEL (Harvey's trigonometric basic structural model):
    state a = [ mu, beta,  (gamma_1, gamma*_1), ..., (gamma_K, gamma*_K) ]
    transition T = blockdiag( [[1,1],[0,1]],  R_1, ..., R_K )
        R_j = [[cos l_j,  sin l_j],
               [-sin l_j, cos l_j]],   l_j = 2*pi*j/s
    observation  y_t = mu_t + sum_j gamma_j,t + eps_t,   Z = [1,0, 1,0, 1,0, ...]
The seasonal is the sum of K harmonic oscillators of the detected period s; the
level+slope is a local-linear-trend. The forward filter is strictly causal:

  predict:  a = T a;   P = T P T' + Q
  update (y_t OBSERVED):  v = y_t - Z a;  F = Z P Z' + H;  K = P Z'/F;
                          a += K v;  P = (I-K Z) P (I-K Z)' + K H K'  (Joseph form)
  MISSING y_t:            skip update; emit yhat = Z a;  a,P unchanged (the gap's
                          uncertainty grows through the predict step) -> causal.

PERIOD DETECTION (AUTOPERIOD, causal): from the OBSERVED history strictly before t
we (1) take a periodogram peak as a candidate s, then (2) VERIFY it is a local
maximum of the autocorrelation at lag s. Require >= 2 full cycles of history before
trusting s. Detection uses only data < t, so truncating the future cannot change it.

VARIANCES: set by fixed sensible ratios relative to the online estimate of the
observed-series scale (NOT tuned to the benchmark). Diffuse init P0 = 1e6 I.

Routing (truncation-invariant): N==1 -> this filter; everything else -> trmf.
Panels -> fe. Non-seasonal 1D (no period ever confirmed) -> trmf, so 1d_ar is
byte-identical to the temporal core.
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
from c_chal_uni import online_impute as _uni

# ---- caps (declared, not tuned to the suite) ---- #
MAX_HARM = 4            # harmonics per detected period
MIN_CYCLES = 3          # need >= this many full cycles of observed history to trust s
MIN_ACF = 0.30          # min gap-aware ACF at the fundamental to call it seasonal
                        # (genuine cycles sit ~0.6-0.9; AR/noise pseudo-peaks ~0.2)
REDETECT_K = 16         # re-run period detection every K newly-observed samples
DIFFUSE = 1e6           # diffuse prior variance on the state
EPS = 1e-9


def _gap_acf(vc, obs, lag):
    """Mean-lagged-product autocorrelation at `lag` on a GAP-AWARE, TIME-INDEXED
    series. vc is centred values (0 where missing), obs is the boolean mask. Only
    index pairs where BOTH endpoints are observed contribute, so the true period is
    preserved even with contiguous block gaps (compressing observed-only samples
    would shift the period -> off-by-one -> Fourier extrapolation fails)."""
    a = vc[lag:]
    b = vc[:-lag]
    o = obs[lag:] & obs[:-lag]
    c = int(o.sum())
    if c < 5:
        return None
    return float(np.sum(a[o] * b[o]) / c)


def _autoperiod(val, obs, min_p=3, max_frac=0.5):
    """Causal period detection on TIME-INDEXED observed history (strictly < t).
    AUTOPERIOD: a periodogram peak proposes candidate periods; each is REFINED to
    the exact integer by the gap-aware ACF local maximum in a small neighbourhood,
    and accepted only if it is a genuine positive ACF peak. Returns int s or None.

    `val` is the value array (NaN at missing), `obs` the observed mask, both indexed
    on the original (gapped) time axis up to but not including t."""
    m = len(val)
    if m < 2 * min_p or obs.sum() < 2 * min_p:
        return None
    mean = np.nanmean(val)
    if not np.isfinite(mean):
        return None
    vc = np.where(obs, val - mean, 0.0)
    v0 = float(np.sum(vc[obs] ** 2) / obs.sum())
    if v0 < EPS:
        return None
    hi = max(min_p + 2, int(max_frac * m))
    if hi <= min_p:
        return None
    # spectral candidates from the (zero-filled) periodogram -> rough peaks
    nfft = 1 << int(np.ceil(np.log2(2 * m)))
    F = np.fft.rfft(vc, nfft)
    power = (F * np.conj(F)).real
    freqs = np.fft.rfftfreq(nfft, d=1.0)
    order = np.argsort(power[1:])[::-1] + 1
    tried = set()
    for fi in order[:10]:
        f = freqs[fi]
        if f <= 0:
            continue
        p0 = int(round(1.0 / f))
        # refine p0 to the best exact integer ACF peak in a +/-2 neighbourhood
        best_p, best_v = None, -1.0
        for p in range(max(min_p, p0 - 2), min(hi, p0 + 2) + 1):
            if p in tried:
                continue
            tried.add(p)
            ac = _gap_acf(vc, obs, p)
            if ac is None:
                continue
            r = ac / v0
            if r > best_v:
                best_v, best_p = r, p
        if best_p is None:
            continue
        p = best_p
        if m < MIN_CYCLES * p or p + 1 >= m:
            continue
        am1 = _gap_acf(vc, obs, p - 1)
        ap1 = _gap_acf(vc, obs, p + 1)
        ac = _gap_acf(vc, obs, p)
        if ac is None:
            continue
        rc = ac / v0
        rm1 = (am1 / v0) if am1 is not None else -1.0
        rp1 = (ap1 / v0) if ap1 is not None else -1.0
        # AR FIREWALL: a smooth AR(1) has MONOTONE-decaying ACF -> the value at the
        # half-period (p//2) is HIGHER than at p (no trough between). A genuine
        # seasonal cycle dips at the half-period and rises back at p, so ac[p] must
        # exceed the half-period ACF by a real margin. This rejects AR's long-lag
        # pseudo-peaks (an intrinsic shape test, not a tuned constant).
        half = _gap_acf(vc, obs, max(1, p // 2))
        rh = (half / v0) if half is not None else 1.0
        # genuine periodicity: STRONG positive local maximum that stands above the
        # half-period trough. The strength bar (MIN_ACF) separates real seasonal
        # cycles (ACF ~0.6-0.9) from AR/noise long-lag pseudo-peaks (ACF ~0.2).
        if rc >= rm1 and rc >= rp1 and rc > MIN_ACF and rc > rh + 0.05:
            return p
    return None


def _build_transition(s, K):
    """Trigonometric BSM transition T and observation Z for period s, K harmonics."""
    dim = 2 + 2 * K
    T = np.zeros((dim, dim))
    # local linear trend block
    T[0, 0] = 1.0
    T[0, 1] = 1.0
    T[1, 1] = 1.0
    Z = np.zeros(dim)
    Z[0] = 1.0
    for j in range(1, K + 1):
        lam = 2.0 * np.pi * j / s
        c, sn = np.cos(lam), np.sin(lam)
        b = 2 + 2 * (j - 1)
        T[b, b] = c
        T[b, b + 1] = sn
        T[b + 1, b] = -sn
        T[b + 1, b + 1] = c
        Z[b] = 1.0
    return T, Z


def _filter_column(col):
    """Causal trigonometric-BSM Kalman filter on one (T,) series with NaNs.
    Returns (out, used) where used is True iff a seasonal model was ever active."""
    T = len(col)
    out = col.copy()
    obs = np.isfinite(col)

    # running fallback stats (strictly past)
    run_sum = 0.0
    run_cnt = 0
    last_val = None

    # online scale estimate of first-difference of observed series (for variances)
    diff_sq_sum = 0.0
    diff_cnt = 0
    prev_obs_val = None

    # state-space (lazily initialised once a period is confirmed)
    s = None
    K = 0
    Tm = None
    Z = None
    Q = None
    H = 1.0
    a = None
    P = None
    dim = 0

    # time-indexed history buffers for gap-aware detection / warm start
    hist_val = np.full(T, np.nan)     # values at original time index (NaN if missing)
    since_detect = 0
    used = False

    for t in range(T):
        # ---------- predict (only meaningful once filter initialised) ----------
        if a is not None:
            a = Tm @ a
            P = Tm @ P @ Tm.T + Q
            P = 0.5 * (P + P.T)

        if obs[t]:
            y = col[t]
            # ---- causal UPDATE with the contemporaneous observation ----
            if a is not None:
                v = y - Z @ a
                F = float(Z @ P @ Z + H)
                if F > EPS:
                    PZt = P @ Z
                    Kg = PZt / F
                    a = a + Kg * v
                    # Joseph form for numerical PSD stability
                    ImKZ = np.eye(dim) - np.outer(Kg, Z)
                    P = ImKZ @ P @ ImKZ.T + np.outer(Kg, Kg) * H
                    P = 0.5 * (P + P.T)
            # observed value is returned unchanged (out already == col)
            # ---- update running fallback + scale AFTER use (causal) ----
            last_val = y
            run_sum += y
            run_cnt += 1
            if prev_obs_val is not None:
                d = y - prev_obs_val
                diff_sq_sum += d * d
                diff_cnt += 1
            prev_obs_val = y
            hist_val[t] = y
            since_detect += 1

            # ---- causal (re)detection of period from history <= t ----
            # We detect using observed history (gap-aware, time-indexed); the result
            # is used to build/refresh the model that fills FUTURE missing cells.
            if (s is None and run_cnt >= 2 * 3) or since_detect >= REDETECT_K:
                since_detect = 0
                hv = hist_val[:t + 1]
                ob = np.isfinite(hv)
                cand = _autoperiod(hv, ob)
                if cand is not None and cand != s:
                    # (re)build the state-space for the new period, warm-start the
                    # seasonal phase from a gap-aware harmonic regression on history.
                    s = cand
                    K = min(MAX_HARM, max(1, s // 2))
                    Tm, Z = _build_transition(s, K)
                    dim = 2 + 2 * K
                    a_new, P_new, Q_new, H_new = _init_state(
                        hv, ob, t, s, K, dim, diff_sq_sum, diff_cnt)
                    a, P, Q, H = a_new, P_new, Q_new, H_new
                    used = True
        else:
            # ---- MISSING: skip update. Emit Z a (a,P already predicted) ----
            if a is not None:
                out[t] = float(Z @ a)
            elif last_val is not None:
                out[t] = last_val
            elif run_cnt > 0:
                out[t] = run_sum / run_cnt
            else:
                out[t] = 0.0

    # safety net
    bad = ~np.isfinite(out)
    if bad.any():
        gm = np.nanmean(col)
        out[bad] = gm if np.isfinite(gm) else 0.0
    return out, used


def _init_state(hv, ob, t_cur, s, K, dim, diff_sq_sum, diff_cnt):
    """Warm-start the BSM state by a ridge harmonic+trend regression on GAP-AWARE
    observed history (causal, past-only, true time axis), and set variances by fixed
    ratios off the online observed-difference scale (no benchmark tuning). The state
    is initialised at time index t_cur (the current, last-observed step)."""
    t_idx = np.nonzero(ob)[0].astype(float)   # true (gapped) time indices of obs
    y = hv[ob]
    # design on the TRUE time axis: [1, t, sin/cos harmonics]
    cols = [np.ones_like(t_idx), t_idx]
    for j in range(1, K + 1):
        w = 2.0 * np.pi * j / s
        cols.append(np.sin(w * t_idx))
        cols.append(np.cos(w * t_idx))
    A = np.column_stack(cols)
    G = A.T @ A
    G = G + 1e-3 * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    try:
        beta = np.linalg.solve(G, A.T @ y)
    except Exception:
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    # observation noise from regression residual scale
    H = max(float(np.var(resid)), 1e-4)
    # process scale from observed first-difference variance (online), fall back to H
    if diff_cnt > 0:
        dscale = max(diff_sq_sum / diff_cnt, 1e-4)
    else:
        dscale = H

    # initialise state at the current time index t_cur (phase aligned to true axis)
    a = np.zeros(dim)
    level = beta[0] + beta[1] * t_cur
    slope = beta[1]
    a[0] = level
    a[1] = slope
    # seasonal harmonic state from regression coeffs, rotated to index t_cur
    for j in range(1, K + 1):
        sj = beta[2 + 2 * (j - 1)]      # sin coeff
        cj = beta[2 + 2 * (j - 1) + 1]  # cos coeff
        lam = 2.0 * np.pi * j / s
        phi = lam * t_cur
        # value of harmonic = sj sin(w t) + cj cos(w t); represent as
        # gamma = amplitude pieces consistent with the rotation recursion:
        gamma = sj * np.sin(phi) + cj * np.cos(phi)
        gamma_star = sj * np.cos(phi) - cj * np.sin(phi)
        b = 2 + 2 * (j - 1)
        a[b] = gamma
        a[b + 1] = gamma_star

    # covariances: moderate prior (not fully diffuse, since we warm-started),
    # process noise as fixed ratios of the difference scale.
    P = np.eye(dim) * (DIFFUSE * 1e-6 + 1.0) * H   # mild prior around warm start
    # level random walk small, slope smaller, seasonal small
    q_level = 0.1 * dscale
    q_slope = 0.001 * dscale
    q_seas = 0.01 * dscale
    qdiag = np.zeros(dim)
    qdiag[0] = q_level
    qdiag[1] = q_slope
    qdiag[2:] = q_seas
    Q = np.diag(qdiag)
    return a, P, Q, H


def online_impute(X, meta):
    # All non-univariate cases delegate to the current best causal baseline so they
    # stay byte-identical to it (no regression); only N==1 is handled here.
    Xn = np.ascontiguousarray(np.asarray(X, float))
    if _is_panel(meta) or Xn.shape[1] != 1:
        return _uni(X, meta)
    # ---- N==1 univariate path: trigonometric BSM Kalman filter ----
    try:
        out, used = _filter_column(Xn[:, 0])
        if used:
            # Use the baseline 1D fill as the base (matches it on the non-seasonal
            # early region), then override MISSING cells with the Kalman seasonal
            # forecast. Both are point-in-time.
            base = np.asarray(_uni(X, meta), float).copy()
            miss = np.isnan(Xn[:, 0])
            base[miss, 0] = out[miss]
            if not np.all(np.isfinite(base)):
                base = np.where(np.isfinite(base), base, 0.0)
            return base
    except Exception:
        pass
    return _uni(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w3_kalman_seasonal", online_impute))
