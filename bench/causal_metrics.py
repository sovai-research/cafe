r"""
Scoring machinery for recovered causal graphs (CAFE-CD). **numpy only.**

No sklearn: AUC is the rank-sum (Mann-Whitney) identity in ~10 lines, so this module
is dependency-free and can migrate into ``src/cafe/`` unchanged if the causal API is
promoted out of the bench.

ORIENTATION AND FLATTENING -- the one thing to get right
-------------------------------------------------------
A *lagged* graph is ``(K,P,P)`` with ``A[k-1,i,j]`` = the coefficient of ``e[t-k,j]``
on ``e[t,i]`` (column = cause, row = effect: the VAR convention used by
``gen_causal_panel``). ``flat_lagged`` ravels it **including the diagonal**, because a
self-loop ``i -> i`` at lag ``k`` is a legitimate edge -- and under this generator it
is usually the *strongest* true edge, since it is the AR term of the idiosyncratic
component. ``exp_byproduct_validation._offdiag`` is right for a *contemporaneous*
score and would silently delete every AR edge here; that is the single easiest way to
report a meaningless number.

A *contemporaneous* graph is ``(P,P)`` and acyclic, so the diagonal carries nothing:
``flat_contemp`` drops it.

Because the diagonal of a lagged graph is both dense and easy, ``summarize_graph``
reports ``auc`` (everything, self-loops included) **and** ``auc_cross`` (off-diagonal
only). ``auc_cross`` is the number that says whether a method found any *structure*;
a detector that only ever fires on the diagonal scores well on ``auc`` alone.

Convention for every function below
-----------------------------------
* ``A_true`` : the ground-truth graph. Nonzero == an edge.
* ``S``      : a continuous score/weight array of the same shape. Larger |value| ==
               more confident. Magnitudes are taken internally where a magnitude is
               what is meant.
* ``A_hat``  : a *thresholded* estimate. Nonzero == a claimed edge. NaNs count as
               "no edge" (a method must not be rewarded for returning NaN).
"""
from __future__ import annotations

import numpy as np

__all__ = ["flat_lagged", "flat_contemp", "edge_auc", "edge_prf", "shd",
           "skeleton_f1", "lag_accuracy", "summarize_graph"]


# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #
def flat_lagged(A):
    """(K,P,P) -> (K*P*P,) in C order (k, i, j). SELF-LOOPS INCLUDED."""
    A = np.asarray(A, float)
    if A.ndim != 3:
        raise ValueError(f"flat_lagged expects a (K,P,P) array; got {A.shape}")
    return A.reshape(-1)


def flat_contemp(A):
    """(P,P) -> (P*P-P,), off-diagonal only, row-major."""
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"flat_contemp expects a square (P,P) array; got {A.shape}")
    return A[~np.eye(A.shape[0], dtype=bool)]


def _flat(A):
    """Shape-aware flattening: lagged keeps the diagonal, contemporaneous drops it."""
    A = np.asarray(A, float)
    return flat_lagged(A) if A.ndim == 3 else flat_contemp(A)


def _cross_mask(shape):
    """Boolean selector (in ``_flat`` order) for the OFF-DIAGONAL entries."""
    if len(shape) == 3:
        K, P, _ = shape
        return np.tile((~np.eye(P, dtype=bool)).reshape(-1), K)
    return np.ones(shape[0] * shape[0] - shape[0], dtype=bool)   # already off-diag


def _is_edge(A):
    """Nonzero and finite -> True. NaN/inf are treated as 'no edge'."""
    A = np.asarray(A, float)
    return np.isfinite(A) & (A != 0.0)


# --------------------------------------------------------------------------- #
# AUC (rank-sum / Mann-Whitney; no sklearn)
# --------------------------------------------------------------------------- #
def edge_auc(y_true, score):
    """ROC-AUC via the Mann-Whitney identity, with proper mid-rank tie handling.

    ``AUC = (sum of ranks of the positives - n1(n1+1)/2) / (n1 * n0)``.

    Non-finite scores are mapped to -inf, i.e. ranked worst: a method that emits NaN
    where it is unsure must not be rewarded for it. Returns NaN when one class is
    absent (AUC is undefined, not 0.5).
    """
    y = np.asarray(y_true).reshape(-1) != 0
    s = np.asarray(score, float).reshape(-1)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs score {s.shape}")
    if y.size == 0:
        return float("nan")
    s = np.where(np.isfinite(s), s, -np.inf)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)                       # last 1-based rank of each tie group
    mid = (csum - cnt + 1 + csum) / 2.0         # mid-rank of each tie group
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = mid[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# --------------------------------------------------------------------------- #
# Binary edge metrics
# --------------------------------------------------------------------------- #
def edge_prf(A_true, A_hat):
    """{"precision","recall","f1","tp","fp","fn"} over the shape-appropriate cells
    (lagged: diagonal included; contemporaneous: diagonal excluded)."""
    A_true = np.asarray(A_true, float)
    A_hat = np.asarray(A_hat, float)
    if A_true.shape != A_hat.shape:
        raise ValueError(f"shape mismatch: {A_true.shape} vs {A_hat.shape}")
    t = _is_edge(_flat(A_true))
    h = _is_edge(_flat(A_hat))
    tp = int(np.sum(t & h))
    fp = int(np.sum(~t & h))
    fn = int(np.sum(t & ~h))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn}


def _shd_2d(T, H, directed):
    """SHD between two boolean (P,P) adjacencies."""
    if directed:
        # a reversal is counted once, at the position where the truth has the edge
        rev = T & H.T & ~T.T & ~H
        n_rev = int(rev.sum())
        missing = int(np.sum(T & ~H & ~rev))
        extra = int(np.sum(H & ~T & ~rev.T))
        return missing + extra + n_rev
    Ts = (T | T.T)
    Hs = (H | H.T)
    up = np.triu(np.ones_like(Ts, dtype=bool))      # diagonal included: a self-loop
    return int(np.sum((Ts != Hs) & up))             # is a real (lagged) edge


def shd(A_true, A_hat, directed=True):
    """Structural Hamming Distance = missing + extra + reversed (lower is better).

    ``directed=False`` scores the skeleton only (an edge in either direction counts).
    For a lagged ``(K,P,P)`` graph the distance is summed over lag slices and a
    "reversal" means ``i -> j`` at lag k recovered as ``j -> i`` at the SAME lag; a
    self-loop can never be a reversal, so AR terms are handled correctly.
    """
    T = _is_edge(A_true)
    H = _is_edge(A_hat)
    if T.shape != H.shape:
        raise ValueError(f"shape mismatch: {T.shape} vs {H.shape}")
    if T.ndim == 3:
        return int(sum(_shd_2d(T[k], H[k], directed) for k in range(T.shape[0])))
    if T.ndim != 2:
        raise ValueError(f"shd expects (P,P) or (K,P,P); got {T.shape}")
    if not directed:
        return _shd_2d(T, H, False)
    # contemporaneous: the diagonal is not part of a DAG, so ignore it
    T = T.copy()
    H = H.copy()
    np.fill_diagonal(T, False)
    np.fill_diagonal(H, False)
    return _shd_2d(T, H, True)


def skeleton_f1(A_true, A_hat):
    """F1 of the UNDIRECTED skeleton (direction-agnostic adjacency).

    Lagged graphs are symmetrised per lag and scored on the upper triangle
    *including* the diagonal (self-loops are real edges); contemporaneous graphs are
    scored on the strict upper triangle.
    """
    T = _is_edge(A_true)
    H = _is_edge(A_hat)
    if T.shape != H.shape:
        raise ValueError(f"shape mismatch: {T.shape} vs {H.shape}")
    if T.ndim == 3:
        P = T.shape[1]
        keep = np.triu(np.ones((P, P), dtype=bool), 0)
        t = np.concatenate([((T[k] | T[k].T)[keep]) for k in range(T.shape[0])])
        h = np.concatenate([((H[k] | H[k].T)[keep]) for k in range(H.shape[0])])
    elif T.ndim == 2:
        P = T.shape[0]
        keep = np.triu(np.ones((P, P), dtype=bool), 1)
        t = (T | T.T)[keep]
        h = (H | H.T)[keep]
    else:
        raise ValueError(f"skeleton_f1 expects (P,P) or (K,P,P); got {T.shape}")
    tp = int(np.sum(t & h))
    fp = int(np.sum(~t & h))
    fn = int(np.sum(t & ~h))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def lag_accuracy(A_true, S):
    """Fraction of true lagged edges whose lag is identified correctly.

    For every ordered pair (i, j) that is a true parent at one or more lags, the pair
    counts as correct when ``argmax_k |S[k,i,j]|`` is one of its true lags. (With
    ``deg`` parents drawn independently per lag a pair CAN be a parent at several
    lags; "any true lag counts" is the pair-level definition -- it reduces to the
    literal ``argmax_k == k_true`` whenever a pair has a single true lag, which is the
    common case.) Returns NaN if there are no true lagged edges.
    """
    A_true = np.asarray(A_true, float)
    S = np.asarray(S, float)
    if A_true.ndim != 3 or S.shape != A_true.shape:
        raise ValueError(f"lag_accuracy expects matching (K,P,P); got "
                         f"{A_true.shape} and {S.shape}")
    T = _is_edge(A_true)
    pairs = T.any(axis=0)                                     # (P,P)
    if not pairs.any():
        return float("nan")
    mag = np.where(np.isfinite(S), np.abs(S), -np.inf)
    khat = np.argmax(mag, axis=0)                             # (P,P)
    hit = np.take_along_axis(T, khat[None], axis=0)[0]        # true at the argmax lag?
    return float(hit[pairs].sum() / pairs.sum())


# --------------------------------------------------------------------------- #
# One-call summary
# --------------------------------------------------------------------------- #
def summarize_graph(A_true, S=None, A_hat=None):
    """Everything above in one dict.

    ``S`` (continuous scores) drives the threshold-free metrics; ``A_hat``
    (thresholded) drives the binary ones. Either may be None.

    Keys: n_true, n_cells, lagged, auc, auc_cross, n_true_cross, lag_accuracy,
    precision, recall, f1, tp, fp, fn, shd, shd_skeleton, skeleton_f1.
    """
    A_true = np.asarray(A_true, float)
    lagged = A_true.ndim == 3
    y = _is_edge(_flat(A_true))
    out = {"lagged": bool(lagged), "n_true": int(y.sum()), "n_cells": int(y.size)}
    if S is not None:
        s = np.abs(np.asarray(S, float))
        if s.shape != A_true.shape:
            raise ValueError(f"shape mismatch: S {s.shape} vs A_true {A_true.shape}")
        sf = _flat(s)
        out["auc"] = edge_auc(y, sf)
        cross = _cross_mask(A_true.shape)
        out["n_true_cross"] = int(y[cross].sum())
        out["auc_cross"] = edge_auc(y[cross], sf[cross])
        if lagged:
            out["lag_accuracy"] = lag_accuracy(A_true, s)
    if A_hat is not None:
        out.update(edge_prf(A_true, A_hat))
        out["shd"] = shd(A_true, A_hat, directed=True)
        out["shd_skeleton"] = shd(A_true, A_hat, directed=False)
        out["skeleton_f1"] = skeleton_f1(A_true, A_hat)
    return out
