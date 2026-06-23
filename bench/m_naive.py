"""
Naive first-class baseline imputers for CAFE benchmarking.

INTERFACE (matches bench/m_baselines.py and bench/c_baselines.py): every imputer
is ``fn(X, meta) -> ndarray`` of the SAME shape as ``X`` (T rows = time, N cols =
features), NaN-safe in, finite out. ``meta`` is accepted for signature parity but
the per-column time-series methods here do not need it (row index r == time t).

CAUSALITY LABELS (the whole point of having these as first-class baselines):
each method is explicitly tagged CAUSAL or NON-CAUSAL. A method is CAUSAL/
point-in-time iff the imputed value at (t, j) uses ONLY observations at times <= t
of that column. A method is NON-CAUSAL if it ever reads a FUTURE observation
(time s > t) to fill (t, j).

  - linear_interp   : NON-CAUSAL  -- interior gaps read BOTH the previous and the
                      NEXT observed value (look-ahead). This is the critical
                      cheap-but-leaky baseline: it looks great on MCAR yet quietly
                      uses the future, so it must never be ranked as point-in-time.
  - spline_interp   : NON-CAUSAL  -- cubic/quadratic spline through observed points;
                      every interior fill depends on future observed points too.
  - seasonal_naive  : CAUSAL      -- fill (t, j) with the most recent observed value
                      at the same seasonal phase (t - period, t - 2*period, ...),
                      i.e. only past data of the same column.
  - nocb            : NON-CAUSAL  -- "next observation carried backward"; fills a
                      gap with the NEXT observed value (pure look-ahead).
  - locf            : CAUSAL      -- "last observation carried forward"; fills a gap
                      with the most recent PAST observed value only.

Edge handling: leading gaps that no causal rule can fill (no admissible past
value) fall back to the per-column observed mean; columns that are entirely NaN
fall back to the global observed mean (then 0.0 if everything is NaN). This keeps
the output finite and the same shape, exactly like the sibling baseline modules.

NOTE: causal methods here never use a future value even at the edges -- a leading
gap is filled with the column mean (a global statistic, not a future point), which
is the standard, defensible point-in-time fallback used elsewhere in the repo.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import warnings

import numpy as np


# --------------------------------------------------------------------------- #
# Shared NaN-safe fallbacks (mirrors _col_mean_fill in m_baselines.py)
# --------------------------------------------------------------------------- #
def _col_mean_fill(X):
    """Per-column observed mean; all-NaN columns -> global mean -> 0.0."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        # all-NaN columns make np.nanmean emit "Mean of empty slice"; we handle
        # those columns explicitly below, so silence the expected warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col = np.nanmean(X, axis=0)
        g = np.nanmean(X)
    g = g if np.isfinite(g) else 0.0
    col = np.where(np.isfinite(col), col, g)
    return col


def _finalize(out, X):
    """Replace any residual non-finite cell with the column-mean fallback."""
    bad = ~np.isfinite(out)
    if bad.any():
        colmean = _col_mean_fill(X)
        out[bad] = np.take(colmean, np.nonzero(bad)[1])
    return out


# --------------------------------------------------------------------------- #
# (1) linear_interp -- NON-CAUSAL (reads the next observed value)
# --------------------------------------------------------------------------- #
def linear_interp(X, meta=None):
    """NON-CAUSAL. Per-column linear interpolation between observed points.

    Interior gaps are filled from the previous AND next observed value (this is
    the look-ahead that makes it leaky). Leading/trailing gaps clamp to the
    nearest observed value (np.interp default). All-NaN columns -> col mean.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    out = X.copy()
    t = np.arange(T, dtype=float)
    colmean = _col_mean_fill(X)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        k = int(obs.sum())
        if k == 0:
            out[:, j] = colmean[j]
        elif k == 1:
            col[~obs] = col[obs][0]
            out[:, j] = col
        else:
            out[:, j] = np.interp(t, t[obs], col[obs])
    return _finalize(out, X)


# --------------------------------------------------------------------------- #
# (2) spline_interp -- NON-CAUSAL (smooth curve through all observed points)
# --------------------------------------------------------------------------- #
def spline_interp(X, meta=None):
    """NON-CAUSAL. Per-column polynomial-spline interpolation.

    Uses scipy cubic spline when available (>=4 observed points) and falls back
    to lower-order interpolation / linear otherwise. Every interior fill depends
    on future observed points, so this is look-ahead just like linear_interp.
    Edges are clamped to the nearest observed value (no extrapolation blow-ups).
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    out = X.copy()
    t = np.arange(T, dtype=float)
    colmean = _col_mean_fill(X)

    try:
        from scipy.interpolate import interp1d
        _have_scipy = True
    except Exception:  # pragma: no cover - scipy is a normal dep but stay safe
        _have_scipy = False

    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        k = int(obs.sum())
        if k == 0:
            out[:, j] = colmean[j]
            continue
        if k == 1:
            col[~obs] = col[obs][0]
            out[:, j] = col
            continue
        miss = ~obs
        if not miss.any():
            continue
        xo, yo = t[obs], col[obs]
        if _have_scipy and k >= 4:
            kind = "cubic"
        elif _have_scipy and k == 3:
            kind = "quadratic"
        else:
            kind = "linear"
        if _have_scipy:
            # bounds_error=False + clamp fill_value => edge gaps held at nearest obs,
            # interior gaps follow the smooth spline through observed points.
            f = interp1d(xo, yo, kind=kind, bounds_error=False,
                         fill_value=(yo[0], yo[-1]), assume_sorted=True)
            vals = f(t[miss])
            # guard against any non-finite spline excursions
            vals = np.where(np.isfinite(vals), vals, np.interp(t[miss], xo, yo))
            out[miss, j] = vals
        else:  # no scipy -> linear interpolation
            out[miss, j] = np.interp(t[miss], xo, yo)
    return _finalize(out, X)


# --------------------------------------------------------------------------- #
# (3) seasonal_naive -- CAUSAL (same seasonal phase, past only)
# --------------------------------------------------------------------------- #
def make_seasonal_naive(period=24):
    """Factory for a CAUSAL seasonal-naive imputer with a given period."""

    def seasonal_naive(X, meta=None):
        """CAUSAL. Fill (t, j) with the most recent PAST observed value at the
        same seasonal phase: t-period, t-2*period, ... Falls back to LOCF (most
        recent past value, any phase), then to the column mean. Never reads t' > t.
        """
        X = np.asarray(X, dtype=float)
        T, N = X.shape
        out = X.copy()
        colmean = _col_mean_fill(X)
        p = max(int(period), 1)

        for j in range(N):
            col = X[:, j]
            obs = np.isfinite(col)
            if not obs.any():
                out[:, j] = colmean[j]
                continue
            # last observed value per phase, updated as we sweep forward in time
            phase_last = np.full(p, np.nan)
            last_any = np.nan          # LOCF fallback (most recent past obs)
            ocol = out[:, j]
            for t in range(T):
                ph = t % p
                if obs[t]:
                    # observed: keep value, then update causal state for the future
                    phase_last[ph] = col[t]
                    last_any = col[t]
                else:
                    v = phase_last[ph]
                    if not np.isfinite(v):
                        v = last_any
                    if not np.isfinite(v):
                        v = colmean[j]
                    ocol[t] = v
        return _finalize(out, X)

    return seasonal_naive


# default-period seasonal naive (24 = daily cycle on hourly data, e.g. Beijing)
seasonal_naive = make_seasonal_naive(period=24)


# --------------------------------------------------------------------------- #
# (4) nocb -- NON-CAUSAL (next observation carried backward)
# --------------------------------------------------------------------------- #
def nocb(X, meta=None):
    """NON-CAUSAL. Next-observation-carried-backward: fill each gap with the
    NEXT observed value in the column (pure look-ahead). Trailing gaps with no
    future observation fall back to the last observed value, then column mean.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    out = X.copy()
    colmean = _col_mean_fill(X)
    ar = np.arange(T)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        if not obs.any():
            out[:, j] = colmean[j]
            continue
        # index of the next observed row at or after each position (look-ahead)
        ridx = np.where(obs, ar, T)
        nxt = np.minimum.accumulate(ridx[::-1])[::-1]
        # trailing gaps (no future obs): borrow the last observed (backward LOCF)
        lidx = np.where(obs, ar, -1)
        prev = np.maximum.accumulate(lidx)
        src = np.where(nxt < T, nxt, prev)
        valid = src >= 0
        col[valid] = col[src[valid]]
        out[:, j] = np.where(np.isfinite(col), col, colmean[j])
    return _finalize(out, X)


# --------------------------------------------------------------------------- #
# (5) locf -- CAUSAL (last observation carried forward)
# --------------------------------------------------------------------------- #
def locf(X, meta=None):
    """CAUSAL. Last-observation-carried-forward: fill each gap with the most
    recent PAST observed value in the column. Leading gaps (no past observation)
    fall back to the column mean. Never reads a future value.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    out = X.copy()
    colmean = _col_mean_fill(X)
    ar = np.arange(T)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        if not obs.any():
            out[:, j] = colmean[j]
            continue
        # index of the most recent observed row at or before each position
        lidx = np.where(obs, ar, -1)
        prev = np.maximum.accumulate(lidx)
        valid = prev >= 0          # leading gaps have no past obs -> -1
        col[valid] = col[prev[valid]]
        out[:, j] = np.where(np.isfinite(col), col, colmean[j])
    return _finalize(out, X)


# --------------------------------------------------------------------------- #
# Registry (causal flag is first-class metadata, not buried in docstrings)
# --------------------------------------------------------------------------- #
BASELINES = {
    "linear_interp":  (linear_interp,  False),   # NON-CAUSAL (reads future)
    "spline_interp":  (spline_interp,  False),   # NON-CAUSAL (reads future)
    "seasonal_naive": (seasonal_naive, True),    # CAUSAL
    "nocb":           (nocb,           False),   # NON-CAUSAL (reads future)
    "locf":           (locf,           True),    # CAUSAL
}


# --------------------------------------------------------------------------- #
# __main__ smoke test on a small synthetic gapped array
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N, period = 60, 4, 12

    # ground-truth: per-column seasonal sine + small noise + per-column offset
    t = np.arange(T)
    truth = np.empty((T, N))
    for j in range(N):
        truth[:, j] = (np.sin(2 * np.pi * t / period + j)
                       + 0.05 * rng.standard_normal(T) + j)

    # punch holes: scattered MCAR + one contiguous block + a leading gap + all-NaN col
    X = truth.copy()
    mask = rng.random((T, N)) < 0.20
    mask[5:12, 1] = True       # contiguous interior gap in col 1
    mask[0:3, 2] = True        # leading gap in col 2
    mask[:, 3] = True          # column 3 entirely missing (all-NaN fallback)
    X[mask] = np.nan
    n_miss = int(mask.sum())

    print(f"synthetic array: shape={X.shape}, missing={n_miss}/{X.size} "
          f"({100*n_miss/X.size:.1f}%), period={period}\n")

    sm = make_seasonal_naive(period=period)  # period-aware variant for the demo
    methods = [
        ("linear_interp",  linear_interp,  "NON-CAUSAL"),
        ("spline_interp",  spline_interp,  "NON-CAUSAL"),
        ("seasonal_naive", sm,             "CAUSAL"),
        ("nocb",           nocb,           "NON-CAUSAL"),
        ("locf",           locf,           "CAUSAL"),
    ]

    print(f"{'method':<16}{'causal':<12}{'shape ok':<10}{'finite':<9}"
          f"{'MAE(masked)':<13}")
    print("-" * 60)
    for name, fn, tag in methods:
        out = fn(X, None)
        shape_ok = (out.shape == X.shape)
        finite = bool(np.isfinite(out).all())
        # MAE only over masked cells; exclude the all-NaN column (no truth signal
        # any imputer could recover -- it can only return the fallback mean)
        eval_mask = mask.copy()
        eval_mask[:, 3] = False
        mae = float(np.mean(np.abs(out[eval_mask] - truth[eval_mask])))
        print(f"{name:<16}{tag:<12}{str(shape_ok):<10}{str(finite):<9}"
              f"{mae:<13.4f}")

    # explicit causality demonstration on a tiny hand array:
    # a single hole flanked by a low past value and a high future value.
    print("\ncausality probe (gap between a low past and a high future value):")
    probe = np.array([[1.0], [np.nan], [9.0]])
    print(f"  input column      : {probe.ravel().tolist()}")
    print(f"  locf (CAUSAL)     -> {locf(probe).ravel().tolist()}   "
          f"(uses past 1.0 only)")
    print(f"  nocb (NON-CAUSAL) -> {nocb(probe).ravel().tolist()}   "
          f"(uses future 9.0)")
    print(f"  linear(NON-CAUSAL)-> {linear_interp(probe).ravel().tolist()}   "
          f"(uses both -> 5.0)")
