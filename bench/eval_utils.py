"""Leak-free evaluation utilities.

The standard imputation-benchmark trap (documented in arXiv:2405.17508): standardise
a column using ALL its values, THEN mask and score. The normalisation statistics then
peek at the held-out cells -- a small but real look-ahead leak, and exactly the
methodological flaw we criticise in others. The honest protocol standardises using
ONLY the cells an imputer is actually allowed to see (observed, post-mask).

``standardize_on_observed`` is the single shared primitive; every benchmark scores on
its output so normalisation never sees a held-out cell.
"""
from __future__ import annotations

import numpy as np

__all__ = ["standardize_on_observed", "mcar_mask", "score_masked"]


def standardize_on_observed(clean: np.ndarray, mask: np.ndarray, eps: float = 1e-9):
    """Per-column z-score whose mean/std use ONLY observed (non-held-out) cells.

    Parameters
    ----------
    clean : (T, N) complete ground-truth matrix.
    mask  : (T, N) bool, True = held out for scoring (NOT visible to the imputer).

    Returns ``Xstd`` (the leak-free standardised ground truth, finite). Score and
    impute on THIS scale; the imputer is given ``Xstd`` with ``mask`` cells set NaN.
    Stats are computed over ``~mask`` (and finite) cells only -- no held-out cell
    influences any column's mean or scale.
    """
    clean = np.asarray(clean, float)
    mask = np.asarray(mask, bool)
    visible = (~mask) & np.isfinite(clean)
    Xstd = np.empty_like(clean)
    for j in range(clean.shape[1]):
        col = clean[:, j]
        vis = visible[:, j]
        if vis.sum() >= 2:
            mu = col[vis].mean()
            sd = col[vis].std()
        elif vis.sum() == 1:
            mu, sd = col[vis][0], 1.0
        else:                                   # nothing visible -> neutral column
            mu, sd = 0.0, 1.0
        Xstd[:, j] = (col - mu) / (sd + eps)
    return Xstd


def mcar_mask(shape, rate: float, seed: int) -> np.ndarray:
    """Deterministic MCAR boolean mask (True = held out)."""
    return np.random.default_rng(seed).random(shape) < rate


def block_mask(shape, rate: float, seed: int, min_gap: int = 8, max_gap: int = 40) -> np.ndarray:
    """Per-column CONTIGUOUS gaps (True = held out) totalling ~rate of each column.

    This is the regime where last-value/local methods fail (no nearby observation) and
    a model with AR+season+factor extrapolation matters -- and the realistic shape of a
    sensor outage or a market closure in a backtest."""
    T, N = shape
    rng = np.random.default_rng(seed)
    M = np.zeros((T, N), bool)
    target = int(round(rate * T))
    for j in range(N):
        filled, guard = 0, 0
        while filled < target and guard < 200:
            g = int(rng.integers(min_gap, max_gap + 1))
            g = min(g, T)
            s = int(rng.integers(0, max(1, T - g)))
            if not M[s:s + g, j].any():
                M[s:s + g, j] = True
                filled += g
            guard += 1
    return M


def make_mask(pattern: str, shape, rate: float, seed: int) -> np.ndarray:
    """Dispatch by pattern name: 'mcar'/'point' or 'block'."""
    if pattern in ("block", "subseq", "subsequence"):
        return block_mask(shape, rate, seed)
    return mcar_mask(shape, rate, seed)


def score_masked(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """MAE/RMSE on held-out cells only; NaN/Inf in pred is penalised, not ignored."""
    mask = np.asarray(mask, bool)
    t = np.asarray(truth, float)[mask]
    p = np.asarray(pred, float)[mask]
    bad = ~np.isfinite(p)
    if bad.any():                               # penalise non-finite predictions
        p = p.copy(); p[bad] = t[bad] + 1e3
    err = p - t
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "n": int(mask.sum())}
