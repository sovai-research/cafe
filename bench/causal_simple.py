"""The full suite of SIMPLE causal (strictly point-in-time) imputers.

A credible causal horse race must beat the *whole* field of cheap causal baselines,
not just LOCF -- reviewers expect mean/median/rolling/drift/Kalman, and (as the MCAR
race showed) a simple method is often the one to beat. Every method here fills x[t]
using only data <= t (past, plus the contemporaneous cross-section for XSecMean).

New naive methods are implemented below; the rest are thin re-exports of causal
baselines already in the repo (c_baselines, m_naive, online_baselines), gathered into
one registry so the race includes them all. Each is ``impute(X, meta) -> (T, N)``.
"""
from __future__ import annotations

import numpy as np

import causal_race as CR

__all__ = ["CAUSAL_SIMPLE"]


def _finite(X):
    X = np.asarray(X, float)
    return X, np.isfinite(X)


def _restore(X, out):
    X = np.asarray(X, float)
    out = np.where(np.isfinite(X), X, np.asarray(out, float))
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
    return out


# --------------------------------------------------------------------------- #
# New simple causal methods
# --------------------------------------------------------------------------- #
def zero_fill(X, meta=None):
    """Trivial floor: fill with 0 (the per-feature mean on standardised data)."""
    X, obs = _finite(X)
    return np.where(obs, X, 0.0)


def causal_mean(X, meta=None):
    """Expanding per-column mean of observed values up to (and incl.) t."""
    X, obs = _finite(X)
    xv = np.where(obs, X, 0.0)
    csum = np.cumsum(xv, axis=0)
    ccnt = np.cumsum(obs, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        run = csum / np.maximum(ccnt, 1)
    run[ccnt == 0] = 0.0                       # nothing seen yet -> mean (0 on std scale)
    return _restore(X, run)


def _rolling(X, W, fn):
    X, _ = _finite(X)
    TW = CR.trailing_windows(X, W)             # (T, W, N), only times <= t
    with np.errstate(invalid="ignore"):
        val = fn(TW, axis=1)                   # ignores NaN within the window
    val = np.where(np.isfinite(val), val, 0.0)
    return _restore(X, val)


def rolling_mean(X, meta=None, W=24):
    """Trailing-window mean (window of the last ``W`` steps)."""
    return _rolling(X, W, np.nanmean)


def rolling_median(X, meta=None, W=24):
    """Trailing-window median -- robust local causal baseline."""
    return _rolling(X, W, np.nanmedian)


def causal_drift(X, meta=None):
    """Causal linear extrapolation: carry the last observed value forward along the
    slope between the two most recent observations (Hyndman 'drift', point-in-time)."""
    X, obs = _finite(X)
    T, N = X.shape
    out = np.array(X, float)
    for j in range(N):
        last_v = last_i = prev_v = prev_i = None
        run = 0.0
        for t in range(T):
            if obs[t, j]:
                prev_v, prev_i = last_v, last_i
                last_v, last_i = X[t, j], t
                run = last_v
            else:
                if last_v is None:
                    out[t, j] = 0.0
                elif prev_v is None or last_i == prev_i:
                    out[t, j] = last_v                       # only one point -> LOCF
                else:
                    slope = (last_v - prev_v) / (last_i - prev_i)
                    out[t, j] = last_v + slope * (t - last_i)
        _ = run
    return _restore(X, out)


def kalman_ll(X, meta=None, q=0.05, r=1.0):
    """Univariate local-level Kalman FILTER (causal): level random walk + obs noise.
    The classical state-space imputer, applied strictly forward."""
    X, obs = _finite(X)
    T, N = X.shape
    out = np.array(X, float)
    for j in range(N):
        m, P, started = 0.0, 1.0, False
        for t in range(T):
            mp, Pp = m, P + q                  # predict
            if obs[t, j]:
                K = Pp / (Pp + r)
                m = mp + K * (X[t, j] - mp)
                P = (1.0 - K) * Pp
                started = True
            else:
                out[t, j] = mp if started else 0.0
                m, P = mp, Pp                  # no update
    return _restore(X, out)


# --------------------------------------------------------------------------- #
# Re-exports of causal baselines already implemented elsewhere in the repo
# --------------------------------------------------------------------------- #
def _gather_existing():
    reg = {}
    try:
        import c_baselines as C
        reg["LOCF"] = lambda X, meta=None: C.locf_impute(X, meta or {})
        reg["EWMA"] = (lambda f: lambda X, meta=None: f(X, meta or {}))(C.make_ewma(halflife=5.0))
        reg["XSecMean"] = lambda X, meta=None: C.xsecmean_impute(X, meta or {})
    except Exception as e:                     # pragma: no cover
        print("[causal_simple] c_baselines unavailable:", e)
    try:
        import m_naive as Nai
        reg["SeasonalNaive"] = (lambda f: lambda X, meta=None: f(X, meta or {}))(
            Nai.make_seasonal_naive(period=24))
    except Exception as e:                     # pragma: no cover
        print("[causal_simple] m_naive unavailable:", e)
    try:
        import online_baselines as OB
        reg["OnlineMeanVar"] = (lambda f: lambda X, meta=None: f(X, meta or {}))(
            OB.make_online_mean_var())
        reg["OnlineEWCov"] = (lambda f: lambda X, meta=None: f(X, meta or {}))(
            OB.make_online_ewcov())
        if hasattr(OB, "grouse_lite"):
            reg["GROUSE"] = lambda X, meta=None: OB.grouse_lite(X, meta or {})
    except Exception as e:                     # pragma: no cover
        print("[causal_simple] online_baselines unavailable:", e)
    return reg


CAUSAL_SIMPLE = {
    "Zero": zero_fill,
    "CausalMean": causal_mean,
    "RollingMean": rolling_mean,
    "RollingMedian": rolling_median,
    "Drift": causal_drift,
    "KalmanLL": kalman_ll,
}
CAUSAL_SIMPLE.update(_gather_existing())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 8)).cumsum(0) / 5 + rng.standard_normal((300, 8))
    mask = rng.random(X.shape) < 0.2
    Xo = X.copy(); Xo[mask] = np.nan
    print(f"{'method':<16}{'MAE':>8}  causal  shape")
    for name, fn in CAUSAL_SIMPLE.items():
        pred = fn(Xo, {})
        mae = float(np.mean(np.abs(np.asarray(pred)[mask] - X[mask])))
        v = CR.verify_causal(lambda Z: fn(Z, {}), X[:120])
        flag = "Y" if v["causal"] else f"N({v['max_dev']:.1e})"
        ok = np.asarray(pred).shape == X.shape and np.all(np.isfinite(pred))
        print(f"{name:<16}{mae:>8.3f}   {flag:<6} {'ok' if ok else 'BAD'}")
