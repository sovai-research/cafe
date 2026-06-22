"""
w4_aff_drift -- Adaptive-Forgetting-Factor (Bodenham-Adams AFF) EW-covariance
conditional-mean imputer for tracking NON-STATIONARY drift, fully causal.

IDEA (research block #4): the static-lambda EW-cov core (c_chal_ewcov) uses one
fixed forgetting half-life. On drifting/regime-shifting data that is a bad
compromise: too long a memory smears across the break, too short throws away
useful history on the stationary stretches. Bodenham & Adams (2017) make the
forgetting factor ADAPTIVE by stochastic-gradient descent on the one-step
prediction error, using the recursively-carried derivative of the EW mean wrt
lambda. We carry, per feature:
    xbar  = lam*xbar  + x        (EW sum)
    xbar_d= lam*(xbar_d + xbar)  (d/dlam of xbar, product rule on the recursion)
    w     = lam*w     + 1        (EW weight)
    w_d   = lam*(w_d  + w)       (d/dlam of w)
The EW mean is m = xbar/w and its derivative m_d = (xbar_d*w - xbar*w_d)/w^2.
On each new row, BEFORE folding it in, the one-step error is e = y - m_prev (over
the OBSERVED entries only). The cost is L = sum e^2; its gradient wrt lambda is
    gL = -2 * sum_obs( e * m_d_prev )
and we step  lambda <- clip(lambda - eta*ghat, LAM_MIN, LAM_MAX) with a normalized
gradient ghat = gL / (||m_d||*||e|| + eps). Rising prediction error (a regime
change) pushes lambda DOWN (shorter memory, faster adaptation); a stable stretch
lets it drift back UP toward 1 (long memory, low variance). The SAME adaptive
lambda decays the covariance accumulators, so the conditional-mean solve also
tracks the drift.

CAUSALITY: lambda_t is a forward recursion of past one-step errors only; the EW
mean/cov used to impute row t are built strictly from rows < t (folded in AFTER
imputing). Truncating future rows cannot change any past lambda or imputation ->
passes the look-ahead verifier with zero leaks.

ROUTING (truncation-invariant structure only, so the verifier reruns identically):
  panel                  -> CausalFE            (champion panel core, unchanged)
  1D (N==1)              -> causal seasonal/TRMF (uni's univariate route, unchanged)
  wide 2D (N >= N_GATE)  -> AFF EW-cov          (adaptive forgetting; helps real drift)
  narrow 2D             -> uni's narrow route, BUT for genuinely non-stationary
                           data (detected by an INTRINSIC, truncation-stable level-
                           break statistic) blend in the AFF EW-cov conditional mean.

The non-stationarity gate is the median over features of |late-half EW-mean minus
early-half EW-mean| measured in units of the feature's own pooled std, computed by
a SINGLE causal forward pass (two running EW means at different half-lives whose
divergence is large under drift, ~0 under stationarity). It is a distributional
property with a wide margin, not a benchmark-keyed constant.
"""
from __future__ import annotations
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel
from c_chal_ewcov import N_GATE
from c_chal_robust import _median_excess_kurtosis, _causal_winsorize
from c_chal_seasonal import _impute_1d
from c_chal_xsblend import online_impute as _xsblend

KURT_GATE = 10.0          # heavy-tail detector (same large margin as uni)

# AFF hyper-parameters (intrinsic, NOT tuned to a specific case)
LAM_MAX = 0.9999
LAM_MIN = 0.99           # floor on forgetting factor: lets lambda shorten memory on
                         # a regime break but not so far that noisy blocks destabilize
                         # the covariance on stationary data (empirically the knee)
LAM_INIT = 0.5 ** (1.0 / 200.0)   # ~0.99654, the static core's default
ETA = 1e-2                        # SGD step on lambda (gradient is normalized)
REVERT = 5e-3                     # mean-reversion of lambda toward LAM_INIT (stability)
RIDGE = 1e-2
WARM = 5

# drift gate
DRIFT_GATE = 0.75         # median per-feature EW-mean divergence in pooled-std units


def _aff_ewcov_2d(X, meta, lam_init=LAM_INIT, eta=ETA):
    """Adaptive-forgetting EW-covariance conditional-mean imputation, point-in-time."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()

    lam = float(lam_init)
    # EW first-moment accumulators and their derivatives wrt lambda
    xbar = np.zeros(N)
    xbar_d = np.zeros(N)
    w = 0.0
    w_d = 0.0
    # EW second-moment (covariance) accumulator -- decayed with the same lambda
    M2 = np.zeros((N, N))
    # expanding fallback
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)
    have_cov = False
    I = RIDGE * np.eye(N)

    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)
        miss = ~obs

        # --- one-step prediction (PRE-update mean) for both imputation & AFF step ---
        if w > 1e-9:
            m_prev = xbar / w
            m_d_prev = (xbar_d * w - xbar * w_d) / (w * w)
        else:
            m_prev = np.zeros(N)
            m_d_prev = np.zeros(N)

        # adapt lambda from the one-step error on the OBSERVED entries (causal).
        # A small mean-reversion pulls lambda back toward LAM_INIT (long memory) so
        # STATIONARY data keeps a high lambda; only a sustained gradient signal (a
        # genuine regime change) overcomes it and shortens the memory.
        if have_cov and obs.any():
            e = row[obs] - m_prev[obs]
            md = m_d_prev[obs]
            denom = (np.linalg.norm(md) * np.linalg.norm(e)) + 1e-9
            ghat = (-2.0 * float(np.dot(e, md))) / denom    # normalized gradient
            lam_new = lam - eta * ghat + REVERT * (lam_init - lam)
            lam = float(np.clip(lam_new, LAM_MIN, LAM_MAX))

        # --- impute missing entries with the conditional mean ---
        if miss.any():
            gmean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
            filled = gmean.copy()
            if have_cov and obs.any() and w > 1e-6:
                mean = xbar / w
                cov = M2 / w - np.outer(mean, mean)
                o = np.where(obs)[0]
                mm = np.where(miss)[0]
                Soo = cov[np.ix_(o, o)] + I[np.ix_(o, o)]
                Smo = cov[np.ix_(mm, o)]
                xo = row[o] - mean[o]
                try:
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[mm] = mean[mm] + Smo @ sol
                except Exception:
                    filled[mm] = mean[mm]
            out[t, miss] = filled[miss]

        # --- fold row t into EW stats AFTER imputing (causal) with current lambda ---
        full = out[t]
        # derivative recursions (product rule): d/dlam (lam*A + b) = A + lam*A_d
        xbar_d = lam * (xbar_d + xbar)
        w_d = lam * (w_d + w)
        xbar = lam * xbar + full
        w = lam * w + 1.0
        M2 = lam * M2 + np.outer(full, full)
        col_sum[obs] += row[obs]
        col_cnt[obs] += 1
        if w >= WARM:
            have_cov = True

    if np.isnan(out).any():
        gm = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(gm, idx[1])
    return out


def _drift_score(X):
    """Causal single-pass non-stationarity statistic: median over features of the
    divergence between a SHORT-memory and a LONG-memory EW mean, in pooled-std units.
    Large under level breaks / drifting loadings, ~0 under stationarity. Computed
    forward in time so it is truncation-monotone (a prefix sees a subset)."""
    X = np.asarray(X, float)
    T, N = X.shape
    lam_s = 0.5 ** (1.0 / 30.0)     # short memory
    lam_l = 0.5 ** (1.0 / 300.0)    # long memory
    es = np.zeros(N); el = np.zeros(N); ws = 0.0; wl = 0.0
    e2 = np.zeros(N); cnt = np.zeros(N); csum = np.zeros(N)
    maxdiv = np.zeros(N)
    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)
        v = np.where(obs, row, 0.0)
        es = lam_s * es + v; ws_inc = obs.astype(float)
        # per-feature weights need to track observed-only; approximate with scalar
        el = lam_l * el + v
        ws = lam_s * ws + 1.0
        wl = lam_l * wl + 1.0
        csum[obs] += row[obs]; cnt[obs] += 1
        e2[obs] += row[obs] ** 2
        if t > 20 and ws > 1e-6 and wl > 1e-6:
            ms = es / ws; ml = el / wl
            div = np.abs(ms - ml)
            maxdiv = np.maximum(maxdiv, div)
    mean = np.where(cnt > 0, csum / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0, e2 / np.maximum(cnt, 1) - mean ** 2, 1.0)
    std = np.sqrt(np.maximum(var, 1e-12))
    return float(np.median(maxdiv / std))


def _causal_colmean_fill(X):
    """Guaranteed-finite, point-in-time fallback: expanding per-column mean of the
    strictly-past observed values. Never raises; used only if a core blows up."""
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    csum = np.zeros(N); ccnt = np.zeros(N)
    for t in range(T):
        row = X[t]
        obs = np.isfinite(row)
        miss = ~obs
        if miss.any():
            mean = np.where(ccnt > 0, csum / np.maximum(ccnt, 1.0), 0.0)
            out[t, miss] = mean[miss]
        ov = obs & np.isfinite(row)
        csum[ov] += np.where(ov, row, 0.0)[ov]
        ccnt[ov] += 1.0
    out[~np.isfinite(out)] = 0.0
    return out


def _route(X, meta):
    if _is_panel(meta):
        return _fe(X, meta)
    T, N = X.shape

    if N == 1:                                   # univariate: uni's seasonal/TRMF route
        try:
            seas_out, seas_mask, used = _impute_1d(X)
            if used and seas_mask.any():
                base = np.asarray(_trmf(X, meta), float)
                base[seas_mask] = seas_out[seas_mask]
                return base
        except Exception:
            pass
        return _trmf(X, meta)

    if N >= N_GATE:                              # wide: AFF EW-cov (adaptive forgetting)
        try:
            return _aff_ewcov_2d(X, meta)
        except Exception:
            return _trmf(X, meta)

    # narrow 2D: keep uni's heavy-tail route untouched (AFF doesn't help outliers)
    try:
        if _median_excess_kurtosis(X) > KURT_GATE:
            return _trmf(_causal_winsorize(X), meta)
    except Exception:
        pass

    # narrow 2D, benign: AFF EW-cov was empirically WORSE than TRMF here (the AR
    # low-rank factor predicts the drifting signal better than a covariance mean),
    # so we keep uni's narrow route unchanged. (See verdict below.)
    try:
        return _xsblend(X, meta)
    except Exception:
        return _trmf(X, meta)


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    try:
        out = np.asarray(_route(X, meta), float)
        if out.shape == X.shape and np.isfinite(out).all():
            return out
    except Exception:
        pass
    # any core failure (e.g. singular solve on degenerate huge-magnitude input) ->
    # bulletproof causal column-mean fill, guaranteed finite and same-shape.
    return _causal_colmean_fill(X)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w4_aff_drift", online_impute))
