"""Causal split-conformal recalibration of CAFE's predictive intervals.

CAFE emits a per-cell Gaussian predictive band ``mu +/- z*sigma``. The band is honest
about *shape* but its *width* is admitted over-conservative: observed coverage runs
~+0.09 above nominal (see ``bench/exp_calibration_crps.py``). This module learns a
multiplier ``q(level)`` so the recalibrated band ``mu +/- q(level)*sigma`` hits nominal
coverage, WITHOUT touching the point imputation (sigma-only). ``q(level)`` is the
``(1 - alpha)`` empirical quantile of recent conformity scores
``s = |y - mu| / sigma`` -- a textbook split-conformal recalibration of the *normalized
residual*.

Where the scores come from (the exchangeability fix)
----------------------------------------------------
Naively scoring the residuals at already-observed cells is NOT exchangeable with the
held-out (missing) cells we want to cover: an observed cell is easier to predict (it is
part of its own contemporaneous cross-section) and is scored against a different,
one-step sigma. Calibrating on those scores systematically UNDER-covers. So instead we
do genuine split conformal: a SECOND causal forward pass over the data with a fresh
random *calibration mask* that hides a small fraction of the observed cells; those cells
are then predicted exactly as missing cells are, and ``|y - mu|/sigma`` at them IS
exchangeable with the test missing cells. The original imputation (``res.imputed``) is
produced by the unmasked run and is never touched -- the calibration pass only supplies
sigma-scaling scores.

Causality / point-in-time
-------------------------
Both passes are CAFE's only mode -- strictly online, no look-ahead: a cell's
``(mu, sigma)`` at time ``t`` uses only data ``<= t``. The trailing-window multiplier
used to recalibrate row ``t`` is the empirical quantile of calibration scores from rows
with timestamp ``STRICTLY < t`` (a window of length ``window``). Hence appending future
rows cannot change an earlier row's band (truncation invariance), and the imputed point
values are untouched.

Distribution-free guarantee
---------------------------
Split conformal gives finite-sample marginal coverage under exchangeability. The
trailing-window variant trades the exact guarantee for adaptivity to drift and strict
causality (the standard online/ACI relaxation). We use the
``ceil((1-alpha)*(n+1))/n`` conformal quantile (Vovk), conservative at small ``n``.

Dependency-light: numpy only (plus a CAFE run for the calibration pass).
"""
from __future__ import annotations

import numpy as np

__all__ = ["ConformalCalibrator", "conformal_multipliers"]

# Default trailing window of past calibration scores. Long enough that the empirical
# quantile at the 95% level is stable, short enough to track non-stationarity.
DEFAULT_WINDOW = 4000
# Min scores before we trust the empirical quantile; below this, fall back to the raw
# Gaussian z for the level (the safe, over-conservative default).
MIN_SCORES = 30
# Fraction of observed cells hidden by the calibration mask (the conformal split).
DEFAULT_CAL_RATE = 0.15
# Seed for the calibration mask (deterministic recalibration; results are reproducible).
DEFAULT_CAL_SEED = 0


def _z_for(level: float) -> float:
    """Two-sided standard-normal quantile with central mass ``level`` (e.g. 0.90 ->
    ~1.645) -- the raw band multiplier, used as the conformal fallback at small ``n``."""
    from math import erf, sqrt
    target = 0.5 * (1.0 + float(level))
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + erf(mid / sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Split-conformal ``(1-alpha)`` quantile of conformity ``scores`` for the central
    interval at nominal ``level`` (alpha = 1 - level). Finite-sample rank
    ``ceil(level*(n+1))`` (Vovk); returns the max score when that rank exceeds ``n``
    (the distribution-free band would be infinite -- we cap at the largest score seen)."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return float("nan")
    rank = int(np.ceil(level * (n + 1)))
    if rank >= n:
        return float(np.max(s))
    s = np.partition(s, rank - 1)
    return float(s[rank - 1])


def conformal_multipliers(scores, levels) -> dict:
    """Static (whole-sample) conformal multipliers ``q(level)`` from a pool of conformity
    ``scores``. For the point-in-time streaming multipliers use
    :class:`ConformalCalibrator`."""
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    out = {}
    for lvl in levels:
        out[lvl] = (_conformal_quantile(s, lvl) if s.size >= MIN_SCORES
                    else _z_for(lvl))
    return out


def _scores_per_row(X_true, mask, mu, sigma):
    """Per-row conformity scores ``|y - mu|/sigma`` at the cells in ``mask`` (the
    calibration cells), as a length-T list of 1-D arrays. ``X_true`` is the original
    (pre-calibration-mask) data so we score against the genuine held-out value."""
    X_true = np.asarray(X_true, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    sel = (np.asarray(mask, bool) & np.isfinite(X_true) & np.isfinite(mu)
           & np.isfinite(sigma) & (sigma > 1e-9))
    S = np.full(X_true.shape, np.nan)
    S[sel] = np.abs(X_true[sel] - mu[sel]) / sigma[sel]
    T = X_true.shape[0]
    return [S[t][np.isfinite(S[t])] for t in range(T)]


class ConformalCalibrator:
    """Causal trailing-window split-conformal recalibrator for CAFE's per-cell sigma.

    Build it with :meth:`from_data` (runs the internal calibration pass) and query
    :meth:`grid` for the point-in-time per-row multiplier ``q[t]`` at a level, or
    :meth:`multipliers` for the whole-window quantiles. The multiplier at row ``t`` uses
    only calibration scores from rows ``< t`` (point-in-time).

    Parameters
    ----------
    window : int
        Trailing window length (number of most-recent calibration scores retained).
    min_scores : int
        Below this many accrued scores, fall back to the raw Gaussian z multiplier.
    """

    def __init__(self, row_scores, window: int = DEFAULT_WINDOW,
                 min_scores: int = MIN_SCORES):
        self._row_scores = list(row_scores)
        self.window = int(window)
        self.min_scores = int(min_scores)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_data(cls, X, cal_rate: float = DEFAULT_CAL_RATE,
                  cal_seed: int = DEFAULT_CAL_SEED, window: int = DEFAULT_WINDOW,
                  min_scores: int = MIN_SCORES) -> "ConformalCalibrator":
        """Run the split-conformal CALIBRATION pass on ``X`` (a 2-D float matrix with NaN
        for missing) and return a fitted calibrator. A fraction ``cal_rate`` of the
        OBSERVED cells is hidden (seed ``cal_seed``) and predicted as missing; the
        normalized residuals at those cells are the conformity scores. ``X`` is not
        mutated and the original imputation is unaffected (this pass is internal)."""
        from .model import _run_traced            # local import avoids a cycle
        X = np.ascontiguousarray(np.asarray(X, float))
        obs = np.isfinite(X)
        rng = np.random.default_rng(int(cal_seed))
        cal_mask = (rng.random(X.shape) < float(cal_rate)) & obs
        if not cal_mask.any():                    # degenerate (no observed cells)
            T = X.shape[0]
            return cls([np.empty(0) for _ in range(T)], window, min_scores)
        Xc = X.copy()
        Xc[cal_mask] = np.nan
        filled, trace, core = _run_traced(Xc)
        # pull per-cell sigma from the traced calibration run
        from .model import CafeResult
        from .io import Ctx
        res = CafeResult(Ctx("numpy", False), Xc, filled, trace, core)
        sigma = np.sqrt(res._comp()["cvar"])      # NaN at non-calibration cells
        row_scores = _scores_per_row(X, cal_mask, filled, sigma)
        return cls(row_scores, window=window, min_scores=min_scores)

    @classmethod
    def from_row_scores(cls, row_scores, window: int = DEFAULT_WINDOW,
                        min_scores: int = MIN_SCORES) -> "ConformalCalibrator":
        """Build directly from precomputed per-row conformity scores (a length-T list of
        1-D arrays). Used by tests / advanced callers that supply their own scores."""
        return cls(row_scores, window=window, min_scores=min_scores)

    # -- queries --------------------------------------------------------------
    def grid(self, level: float):
        """(T,) point-in-time multiplier ``q[t]`` for the central ``level`` interval.
        ``q[t]`` is the conformal quantile of calibration scores from rows ``< t`` within
        the trailing window (raw Gaussian z fallback below ``min_scores``)."""
        T = len(self._row_scores)
        q = np.empty(T, float)
        buf: list = []
        for t in range(T):
            if len(buf) >= self.min_scores:
                q[t] = _conformal_quantile(np.asarray(buf, float), level)
            else:
                q[t] = _z_for(level)
            sc = self._row_scores[t]
            if len(sc):
                buf.extend(np.asarray(sc, float).tolist())
                if len(buf) > self.window:
                    del buf[:-self.window]
        return q

    def multipliers(self, levels) -> dict:
        """Whole-window (non-streaming) ``{level: q}`` over ALL calibration scores --
        a summary multiplier, not the point-in-time grid."""
        allsc = np.concatenate([np.asarray(s, float).ravel()
                                for s in self._row_scores if len(s)]) \
            if any(len(s) for s in self._row_scores) else np.empty(0)
        return conformal_multipliers(allsc, levels)
