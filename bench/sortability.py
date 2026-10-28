r"""
Sortability diagnostics: is the ground-truth generator LEAKING the causal order?

A simulated causal panel can be accidentally rigged. If each node's marginal variance
grows with its depth in the graph, then "sort the nodes by variance and orient edges
from low to high" recovers the DAG with no learning at all -- and any structure
learner that is even weakly scale-sensitive inherits that free win. This is exactly
how continuous-optimisation learners (NOTEARS / DYNOTEARS class) have been flattered
in the literature:

  * Reisach, Seiler & Weichwald, "Beware of the Simulated DAG!", NeurIPS 2021
    (introduces *varsortability*);
  * Reisach et al., "A Scale-Invariant Sorting Criterion...", NeurIPS 2023
    (*R^2-sortability*: standardising the data kills varsortability but NOT the
    fraction of the order that leaks through predictability);
  * Lohse & Wahl, TMLR 2024 (the same failure mode in the time-series/DYNOTEARS
    setting).

Reading
-------
    ~0.5  no artefact -- the metric is blind to the causal order (what we want).
    ~1.0  the generator leaks the order through scale; every downstream number about
          "discovering" this graph is suspect.
    ~0.0  leaks it just as badly, in reverse.

Both functions take the skeleton in **row = CAUSE, col = EFFECT** orientation.
``gen_causal_panel``'s ``A_lag`` is stored the other way round (VAR convention, row =
effect), so always build the input with ``CausalPanelTruth.skeleton()`` rather than
transposing by hand -- passing the transpose silently reports ``1 - sortability``.
A 3-D array is rejected for the same reason.
"""
from __future__ import annotations

import numpy as np

__all__ = ["var_sortability", "r2_sortability"]


def _as_2d(X):
    """(N,P) or (E,T,P) -> (N,P). The entity axis is pooled: node variance and
    cross-sectional predictability are per-feature quantities."""
    X = np.asarray(X, float)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[2])
    if X.ndim != 2:
        raise ValueError(f"expected (N,P) or (E,T,P); got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("sortability needs a complete matrix (no NaN/inf)")
    return X


def _edges(A_skel, P):
    """(i, j) index arrays for the off-diagonal nonzeros of a (P,P) skeleton."""
    A = np.asarray(A_skel)
    if A.ndim == 3:
        raise ValueError(
            "var/r2_sortability take a (P,P) cause->effect skeleton, not a (K,P,P) "
            "lagged array. Use CausalPanelTruth.skeleton(), which collapses the lags "
            "AND fixes the orientation.")
    if A.shape != (P, P):
        raise ValueError(f"skeleton must be ({P},{P}) to match the data; got {A.shape}")
    E = (np.asarray(A, float) != 0)
    E = E & ~np.eye(P, dtype=bool)                 # i -> i carries no ordering
    return np.nonzero(E)


def _frac_greater(a_child, a_parent):
    """Fraction of pairs with child > parent; exact ties count 1/2 (the standard
    convention -- a tie is no evidence either way)."""
    n = a_child.size
    if n == 0:
        return float("nan")
    return float((np.sum(a_child > a_parent) + 0.5 * np.sum(a_child == a_parent)) / n)


def var_sortability(X, A_skel):
    """Fraction of true edges ``i -> j`` for which ``var(x_j) > var(x_i)``.

    X : (N,P) or (E,T,P).  A_skel : (P,P), row = cause, col = effect.
    """
    X = _as_2d(X)
    i, j = _edges(A_skel, X.shape[1])
    v = X.var(axis=0)
    return _frac_greater(v[j], v[i])


def _r2_each(X):
    """R^2 of every column regressed on ALL the others (+ intercept), via lstsq."""
    n, P = X.shape
    Xc = X - X.mean(axis=0)
    r2 = np.zeros(P)
    ones = np.ones((n, 1))
    for p in range(P):
        others = np.delete(np.arange(P), p)
        A = np.concatenate([Xc[:, others], ones], axis=1)
        y = Xc[:, p]
        sst = float(y @ y)
        if sst <= 0.0:
            r2[p] = 0.0
            continue
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        r2[p] = 1.0 - float(resid @ resid) / sst
    return r2


def r2_sortability(X, A_skel):
    """Fraction of true edges ``i -> j`` for which ``R^2(x_j | rest) >
    R^2(x_i | rest)``.

    Scale-invariant, so -- unlike varsortability -- it is NOT fixed by standardising
    the data: it measures whether *predictability* alone leaks the causal order.
    Needs N > P for the per-node regressions to be identified.
    """
    X = _as_2d(X)
    i, j = _edges(A_skel, X.shape[1])
    r2 = _r2_each(X)
    return _frac_greater(r2[j], r2[i])
