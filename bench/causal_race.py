"""Causal horse race: turn ANY windowed imputer into a strict point-in-time filter.

The entire time-series imputation field benchmarks in the *bidirectional* (smoothing)
setting: every method may use the future to fill the past. That is invalid for any
sequential decision (a backtest, an online controller, an early-warning monitor).

This module provides the machinery for the *first causal leaderboard*: run the same
trained model two ways and compare.

  * BIDIRECTIONAL  -- non-overlapping windows, trust the whole window (uses the future
                      within each window). This is the standard, optimistic setting.
  * CAUSAL         -- for every time ``t`` build the trailing window ``[t-L+1 .. t]`` and
                      trust ONLY the right-most position. That output depends on data
                      <= t only, so it is truncation-invariant (passes ``assert_causal``):
                      a bidirectional architecture applied *honestly* as a filter.

The look-ahead gap ``delta = causal_mae - bidirectional_mae >= 0`` is the accuracy a
method silently borrows from the future. A natively causal model (CAFE) has delta = 0.

Pure numpy; no torch. ``deep_baselines.py`` plugs PyPOTS models into the
``predict_fn`` callable below.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "make_windows", "from_windows", "trailing_windows", "right_edge",
    "bidir_fill", "causal_fill", "look_ahead_gap", "verify_causal",
]


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def make_windows(X: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    """``(T, N) -> (n_windows, L, N)`` non-overlapping. The final partial window is
    NaN-padded so no row is dropped. Returns ``(W, T)`` (T = original length)."""
    X = np.asarray(X, float)
    T, N = X.shape
    n = int(np.ceil(T / L))
    pad = n * L - T
    if pad:
        X = np.vstack([X, np.full((pad, N), np.nan)])
    return X.reshape(n, L, N), T


def from_windows(W: np.ndarray, T: int, N: int, L: int) -> np.ndarray:
    """Inverse of :func:`make_windows`: ``(n, L, N) -> (T, N)`` (drops the pad)."""
    return np.asarray(W, float).reshape(-1, N)[:T]


def trailing_windows(X: np.ndarray, L: int) -> np.ndarray:
    """For every ``t`` the causal window ``X[t-L+1 .. t]`` (left NaN-pad early on).

    Returns ``(T, L, N)``; position ``[t, -1, :]`` is row ``t`` itself, and the whole
    window contains only times ``<= t``. Vectorised."""
    X = np.asarray(X, float)
    T, N = X.shape
    pad = np.full((L - 1, N), np.nan)
    Xp = np.vstack([pad, X])                         # (T+L-1, N)
    idx = np.arange(T)[:, None] + np.arange(L)[None, :]   # (T, L)
    return Xp[idx]                                    # (T, L, N)


def right_edge(pred_trailing: np.ndarray) -> np.ndarray:
    """Causal readout: the right-most position of each trailing window. ``(T,L,N)->(T,N)``."""
    return np.asarray(pred_trailing, float)[:, -1, :]


# --------------------------------------------------------------------------- #
# Fillers: a ``predict_fn`` maps a 3D NaN-holed batch (k, L, N) -> imputed (k, L, N)
# (a trained model's forward pass). Both fillers keep observed cells exact.
# --------------------------------------------------------------------------- #
def _keep_observed(X: np.ndarray, filled: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    out = np.where(np.isfinite(X), X, np.asarray(filled, float))
    # never emit NaN/Inf
    if not np.all(np.isfinite(out)):
        col = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        bad = ~np.isfinite(out)
        out[bad] = np.take(col, np.where(bad)[1])
    return out


def bidir_fill(predict_fn, X: np.ndarray, L: int) -> np.ndarray:
    """Standard bidirectional application: non-overlapping windows, full readout."""
    T, N = np.asarray(X, float).shape
    W, T0 = make_windows(X, L)
    pred = predict_fn(W)
    filled = from_windows(pred, T0, N, L)
    return _keep_observed(X, filled)


def causal_fill(predict_fn, X: np.ndarray, L: int, chunk: int = 512) -> np.ndarray:
    """Strict point-in-time application: trailing windows + right-edge readout.

    ``chunk`` batches the (T, L, N) trailing tensor to bound memory."""
    T, N = np.asarray(X, float).shape
    TW = trailing_windows(X, L)                       # (T, L, N)
    edges = np.empty((T, N), float)
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        edges[s:e] = right_edge(predict_fn(TW[s:e]))
    return _keep_observed(X, edges)


def look_ahead_gap(bidir_mae: float, causal_mae: float) -> float:
    """Accuracy borrowed from the future: ``causal_mae - bidir_mae`` (>= 0 typically)."""
    return float(causal_mae) - float(bidir_mae)


# --------------------------------------------------------------------------- #
# Verification: prove a causal filler is truly point-in-time (truncation-invariant).
# --------------------------------------------------------------------------- #
def verify_causal(impute_fn, X: np.ndarray, n_prefixes: int = 6, tol: float = 1e-6) -> dict:
    """Re-impute on growing time-prefixes; a causal method leaves earlier fills
    unchanged. Returns ``{causal: bool, max_dev: float}``. Self-contained (does not
    depend on bench/causal.py) so it works on a raw matrix."""
    X = np.asarray(X, float)
    T, N = X.shape
    full = np.asarray(impute_fn(X.copy()), float)
    max_dev = 0.0
    cuts = np.linspace(T // 3, T - 1, n_prefixes, dtype=int)
    for k in np.unique(cuts):
        pref = np.asarray(impute_fn(X[: k + 1].copy()), float)
        d = np.abs(pref - full[: k + 1])
        d = d[np.isfinite(d)]
        if d.size:
            max_dev = max(max_dev, float(d.max()))
    return {"causal": max_dev <= tol, "max_dev": max_dev}


if __name__ == "__main__":
    # self-test on a toy linear "model" (identity-ish) -- shapes + causality of the
    # right-edge readout, independent of any deep model.
    rng = np.random.default_rng(0)
    T, N, L = 96, 5, 12
    X = rng.standard_normal((T, N))
    X[rng.random((T, N)) < 0.2] = np.nan

    def predict_fn(W):                # trivial filler: replace NaN with 0 (deterministic)
        return np.where(np.isfinite(W), W, 0.0)

    b = bidir_fill(predict_fn, X, L)
    c = causal_fill(predict_fn, X, L)
    assert b.shape == c.shape == (T, N)
    assert np.all(np.isfinite(b)) and np.all(np.isfinite(c))
    # causal_fill with a memoryless predict_fn must be truncation-invariant
    v = verify_causal(lambda Z: causal_fill(predict_fn, Z, L), X)
    print("shapes ok; causal right-edge verify:", v)
    assert v["causal"], v
    print("causal_race self-test PASSED")
