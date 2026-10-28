"""CAFE-CD -- the public causal-discovery assembler.

This module is the one public surface of CAFE's causal-discovery stack.  Every
component underneath it (:mod:`cafe._screen`, :mod:`cafe._skeleton`,
:mod:`cafe._orient`, :mod:`cafe._diagnostics`) is already tested in isolation; this
file only decides *what to run, in what order, on what*, and packages the answer.

    import cafe
    g = cafe.discover(df)                                  # zero-config, batch
    g.edges;  g.adjacency();  g.scores();  g.summary()

    G = cafe.discover(df, mode="pit", checkpoints=[100, 200, 300])
    G[-1].adjacency()                                      # graph as of t = 300

    res = cafe.CAFE().run(df)                              # reuse an existing pass
    g   = res.discover()

The pipeline
------------
1. **Residualise.**  Run the causal one-pass CAFE model and take
   ``residuals(factor_adjusted=True, mask_imputed=True, robust=False)``.  A latent
   common factor is a textbook confounder, so it is projected out by the same
   point-in-time pass that did the imputation.  ``robust=False`` is load-bearing:
   the winsorised residual has had its tail clipped and the tail is exactly what
   identifies edge direction in every LiNGAM-family branch downstream.
2. **Engine routing (panels only).**  :func:`loading_homogeneity_ratio` compares the
   *between-entity* alignment of half-sample loading subspaces to the *within-entity*
   alignment of the same objects.  The within number is the sampling-noise ceiling
   that identical loadings would produce, so the ratio has a null of 1 by
   construction and ``>= 0.95`` routes to ``engine="pooled"``.  This is a property of
   the design matrix, never of an outcome, so it stays causal-clean.
3. **Screen lags.**  :func:`cafe._screen.screen_lags` -- backward-only encoding,
   permutation-calibrated per-target threshold -- cuts the ``P * P * K`` triple space
   down to a candidate set.
4. **Skeleton.**  :func:`cafe._skeleton.panel_group_lasso` fits shared coefficients
   plus sparse per-entity deviations on the candidates and selects the support.
4b. **Scoring routing.**  :func:`cafe._skeleton.conditioning_diagnostic` measures how
   much of each source's identifying variation is its own rather than shared with the
   other sources (a generalised-VIF reciprocal, null 1 by construction).  Both
   conditional evidence components are only readable as statements about one source
   while that number clears 1/2, so below it :func:`_scoring_router` moves weight onto
   :func:`cafe._skeleton.bivariate_granger` -- the same estimator conditioning on the
   target's own past and one candidate at a time.  Design matrix only, never an
   outcome.
5. **Diagnose + orient.**  Lagged edges are oriented by the arrow of time for free
   (``identification_method="time"``).  Only the contemporaneous ambiguous graph is
   searched, and :func:`cafe._orient.orient` does that per connected component, with
   explicit abstention when no branch's assumptions hold.
6. **Assemble** a :class:`CausalGraph`.

No look-ahead
-------------
Every stage reads rows ``<= t`` only, so ``mode="pit"`` is implemented by truncating
the panel first and running the identical pipeline -- which makes the reported graph
truncation-invariant *by construction*, not by convention.
:meth:`CausalGraph.max_edge_revision` (and the module-level
:func:`max_edge_revision`) turns that into a number via
:func:`cafe.audit.verify_causal_series`, for CAFE or for any rival method.

Dependencies: numpy only.  pandas / polars are soft-imported inside
:meth:`CausalGraph.to_pandas` / :meth:`CausalGraph.to_polars`.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ._diagnostics import Diagnostics, diagnose, route
from ._orient import Edge, orient
from ._screen import screen_lags
from ._skeleton import (
    _two_way_demean,
    bivariate_granger,
    build_moment_tensor,
    conditioning_diagnostic,
    panel_group_lasso,
)
from .audit import verify_causal_series
from .io import to_matrix
from .model import CAFE

__all__ = [
    "discover", "CausalGraph", "PitGraphs", "Edge",
    "loading_homogeneity_ratio", "choose_engine", "default_max_lag",
    "conditioning_diagnostic", "max_edge_revision",
]

_EPS = 1e-12
#: ``ratio >= HOMOGENEITY_POOL`` routes a panel to the pooled engine.  The null of the
#: ratio is 1 (see :func:`loading_homogeneity_ratio`), so this is a 5% tolerance below
#: the "loadings are identical" reference point, not a free parameter fitted to data.
HOMOGENEITY_POOL = 0.95
#: A source block is "separable" when at least HALF of its identifying variation is its
#: own rather than shared with the other sources -- i.e. when
#: :func:`cafe._skeleton.conditioning_diagnostic` reports ``tol_j >= 0.5``.  This is the
#: only anchor the scoring router uses and it is not a fitted number: ``tol`` is a
#: generalised-VIF reciprocal whose null is **1 by construction** (orthogonal source
#: blocks), and 1/2 is the unique scale-free dichotomy on it -- the point where a
#: source's coefficient stops being determined mainly by variation unique to that
#: source and starts being determined mainly by variation it shares with its
#: competitors.  Below that point a *conditional* coefficient is not a statement about
#: that source any more, so the evidence blend starts moving weight to a statistic that
#: does not depend on telling the sources apart.  See :func:`_scoring_router`.
SOURCE_SEPARABILITY = 0.5
#: Hard ceiling on the auto-selected lag order (cost is O(K P^2) in screening).
MAX_LAG_CEILING = 12
#: Cap on the ``(K+1)^2 P^2`` moment tensor the evidence score is built from.  Above
#: it the assembler degrades to the screening statistic and says so in ``notes``.
_MOMENT_BUDGET_BYTES = 1 << 30


# --------------------------------------------------------------------------- #
# small numpy-only helpers
# --------------------------------------------------------------------------- #
def _ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Max absolute error ~1.15e-9 on (0, 1).  Exists so the contemporaneous screening
    threshold can be calibrated without scipy.
    """
    p = float(min(max(p, 1e-15), 1.0 - 1e-15))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _as_cube(X, obs=None):
    """``(T, P)`` or ``(E, T, P)`` -> ``(X3, M3)`` float64 cube + observed mask."""
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim == 2:
        A = A[None]
    if A.ndim != 3:
        raise ValueError(f"expected (T,P) or (E,T,P); got ndim={A.ndim}")
    finite = np.isfinite(A)
    if obs is None:
        M = finite
    else:
        O = np.asarray(obs, dtype=bool)
        if O.ndim == 2:
            O = O[None]
        M = finite & O
    return np.ascontiguousarray(A), np.ascontiguousarray(M)


def _masked_corr(X3, M3):
    """Pairwise-complete correlation of a residual cube.  Zero where undefined."""
    G, T, P = X3.shape
    X2 = np.where(M3, X3, 0.0).reshape(G * T, P)
    W2 = M3.reshape(G * T, P).astype(float)
    n = W2.T @ W2                                   # co-observation counts
    s = W2.T @ X2                                   # sum of x_j over co-observed
    n_safe = np.maximum(n, 1.0)
    mu_j = s.T / n_safe                             # mu_j[i, j] = mean of j given i obs
    S = X2.T @ X2 / n_safe - mu_j.T * mu_j          # cross moments, pair-centred
    S = 0.5 * (S + S.T)
    d = np.sqrt(np.maximum(np.diag(S), 0.0))
    den = np.outer(d, d)
    C = np.where(den > _EPS, S / np.maximum(den, _EPS), 0.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0), n


def _partial_corr(C):
    """Partial correlations from a correlation matrix (ridge-stabilised)."""
    P = C.shape[0]
    try:
        Theta = np.linalg.solve(C + 1e-6 * np.eye(P), np.eye(P))
    except np.linalg.LinAlgError:                              # pragma: no cover
        return np.abs(C)
    d = np.sqrt(np.maximum(np.diag(Theta), _EPS))
    R = -Theta / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


def _subspace(A, r):
    """Top-``r`` right-singular (loading) subspace of an ``(n, P)`` block.

    Columns are centred on their observed values and missing cells zero-filled, so a
    NaN-holed block still yields a subspace.  Returns ``(P, r')`` orthonormal with
    ``r' <= r``, or ``None`` when the block cannot support one.
    """
    A = np.asarray(A, float)
    if A.ndim != 2 or A.shape[0] < 2 or A.shape[1] < 1:
        return None
    M = np.isfinite(A)
    cnt = M.sum(axis=0)
    if not np.any(cnt >= 2):
        return None
    Z = np.where(M, A, 0.0)
    mu = Z.sum(axis=0) / np.maximum(cnt, 1)
    Z = np.where(M, Z - mu, 0.0)
    sd = np.sqrt((Z * Z).sum(axis=0) / np.maximum(cnt, 1))
    Z = Z / np.where(sd > _EPS, sd, 1.0)
    if not np.all(np.isfinite(Z)):                             # pragma: no cover
        return None
    try:
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    except np.linalg.LinAlgError:                              # pragma: no cover
        return None
    keep = int(min(r, np.count_nonzero(s > max(s[0], _EPS) * 1e-8))) if s.size else 0
    if keep < 1:
        return None
    return np.ascontiguousarray(Vt[:keep].T)


def _align(U, V):
    """Mean cosine of the principal angles between two subspaces, in ``[0, 1]``."""
    if U is None or V is None:
        return np.nan
    k = min(U.shape[1], V.shape[1])
    if k < 1:
        return np.nan
    try:
        s = np.linalg.svd(U[:, :k].T @ V[:, :k], compute_uv=False)
    except np.linalg.LinAlgError:                              # pragma: no cover
        return np.nan
    return float(np.clip(s, 0.0, 1.0).mean())


def _auto_rank(X3, M3, cap=8):
    """Number of correlation eigenvalues above the Marchenko-Pastur bulk edge."""
    G, T, P = X3.shape
    if P < 2:
        return 1
    C, n = _masked_corr(X3, M3)
    n_eff = float(np.median(np.diag(n))) if n.size else 0.0
    if n_eff < 4:
        return 1
    try:
        ev = np.linalg.eigvalsh(C)
    except np.linalg.LinAlgError:                              # pragma: no cover
        return 1
    edge = (1.0 + np.sqrt(P / max(n_eff, 1.0))) ** 2
    r = int(np.count_nonzero(ev > edge))
    return int(np.clip(r, 1, min(cap, P - 1, max(1, int(T // 4)))))


# --------------------------------------------------------------------------- #
# engine routing -- structural, never outcome-based
# --------------------------------------------------------------------------- #
def loading_homogeneity_ratio(X3d, r=None, obs=None, max_pairs=400, seed=0,
                              min_rows=8, return_parts=False):
    """Do the entities of a panel share one loading structure?

    The question the panel engine actually turns on is "can a single ``W`` represent
    every entity's common component".  The naive test -- align the entities' rank-``r``
    principal subspaces and threshold the alignment -- **fails**: the measured
    alignments were 0.73 (heterogeneous) and 0.98 (homogeneous), both far above any
    absolute cutoff one would write down a priori, because raw alignment is dominated
    by ``P``, ``r`` and the sample size rather than by homogeneity.

    So normalise by the sampling-noise ceiling instead:

    ``within``
        Split each entity's OWN rows in half and align the two half-sample rank-``r``
        subspaces.  This is what alignment would look like if the loadings were
        *identical* and only sampling noise separated the two estimates.
    ``between``
        Align half-sample subspaces taken from DIFFERENT entities -- same rank, same
        block length, same noise level, so the only extra source of misalignment is
        genuine loading heterogeneity.
    ``ratio = mean(between) / mean(within)``
        Null is **1 by construction**: under identical loadings the two collections
        are draws from the same distribution.  Measured 1.067 / 1.030 on homogeneous
        fixtures and 0.786 / 0.762 on heterogeneous ones.

    Parameters
    ----------
    X3d : (E, T, P) array
        Panel.  NaN marks missing; a ``(T, P)`` array has no entity axis and returns
        ``nan``.
    r : int, optional
        Subspace rank.  Default: the number of correlation eigenvalues above the
        Marchenko-Pastur bulk edge, clipped to ``[1, min(8, P-1, T//4)]``.
    obs : bool array, optional
        Extra observed mask (True == observed), ANDed with ``isfinite``.
    max_pairs : int
        Cap on the number of between-entity subspace pairs actually evaluated; the
        pairs are subsampled deterministically from ``seed`` when the cap binds.
    seed : int
        Seed for that subsample.  The statistic is otherwise deterministic.
    min_rows : int
        Minimum rows per half-block for an entity to contribute.
    return_parts : bool
        Return the full ``dict`` instead of just the ratio.

    Returns
    -------
    float, or dict with ``{ratio, within, between, r, n_entities, n_pairs, reason}``
    """
    X3, M3 = _as_cube(X3d, obs)
    E, T, P = X3.shape
    parts = {"ratio": float("nan"), "within": float("nan"), "between": float("nan"),
             "r": 0, "n_entities": int(E), "n_pairs": 0, "reason": ""}
    if E < 2:
        parts["reason"] = "fewer than 2 entities: no entity axis to pool over"
        return parts if return_parts else parts["ratio"]
    if T < 2 * min_rows or P < 2:
        parts["reason"] = f"panel too small for a split-half subspace (T={T}, P={P})"
        return parts if return_parts else parts["ratio"]

    rr = int(r) if r is not None else _auto_rank(X3, M3)
    rr = int(np.clip(rr, 1, max(1, min(P - 1, T // 2 - 1))))
    parts["r"] = rr

    half = T // 2
    subs = []                                   # [(U_first, U_second), ...] per entity
    for e in range(E):
        Ae = np.where(M3[e], X3[e], np.nan)
        U0 = _subspace(Ae[:half], rr)
        U1 = _subspace(Ae[half:2 * half], rr)
        if U0 is not None and U1 is not None and U0.shape[1] == U1.shape[1] == rr:
            subs.append((U0, U1))
    if len(subs) < 2:
        parts["reason"] = "fewer than 2 entities produced a usable split-half subspace"
        return parts if return_parts else parts["ratio"]

    within = [_align(U0, U1) for U0, U1 in subs]
    within = [w for w in within if np.isfinite(w)]

    n = len(subs)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    combos = [(a, b, ha, hb) for (a, b) in pairs for ha in (0, 1) for hb in (0, 1)]
    if len(combos) > max_pairs:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(combos), size=int(max_pairs), replace=False)
        combos = [combos[i] for i in np.sort(idx)]
    between = [_align(subs[a][ha], subs[b][hb]) for (a, b, ha, hb) in combos]
    between = [x for x in between if np.isfinite(x)]

    if not within or not between:                              # pragma: no cover
        parts["reason"] = "no finite alignment could be computed"
        return parts if return_parts else parts["ratio"]
    w = float(np.mean(within))
    b = float(np.mean(between))
    parts.update(within=w, between=b, n_pairs=len(between),
                 ratio=(b / w if w > _EPS else float("nan")))
    return parts if return_parts else parts["ratio"]


def choose_engine(X3d, r=None, obs=None, threshold=HOMOGENEITY_POOL, **kw):
    """``('pooled' | 'per_entity', parts)`` for a panel, from the homogeneity ratio.

    ``ratio >= threshold`` (default 0.95, i.e. within 5% of the identical-loadings
    null of 1) pools; otherwise every entity keeps its own loadings.  A non-panel or
    a degenerate panel falls back to ``'per_entity'``, which is always valid.
    """
    parts = loading_homogeneity_ratio(X3d, r=r, obs=obs, return_parts=True, **kw)
    ratio = parts["ratio"]
    if not np.isfinite(ratio):
        parts["engine"] = "per_entity"
        if not parts["reason"]:                                # pragma: no cover
            parts["reason"] = "homogeneity ratio undefined"
        return "per_entity", parts
    eng = "pooled" if ratio >= float(threshold) else "per_entity"
    parts["engine"] = eng
    return eng, parts


# --------------------------------------------------------------------------- #
# lag-order default
# --------------------------------------------------------------------------- #
def default_max_lag(T, P=None, ceiling=MAX_LAG_CEILING):
    """Zero-config lag order from the sample length alone.

    ``K = floor(4 * (T / 100) ** (2/9))`` -- the Schwert rule that the time-series
    literature uses for unit-root and VAR lag selection -- then clipped to
    ``[1, min(ceiling, (T - 1) // 5)]`` so that (a) screening cost stays ``O(K P^2)``
    with a bounded ``K`` and (b) at least five usable rows back every lag.

    Depends on ``T`` only, so truncating the panel to ``t`` and re-deriving it gives
    the same answer as any other run on the same prefix -- which is what keeps
    ``mode="pit"`` truncation-invariant.
    """
    T = int(T)
    if T < 6:
        return 1
    k = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    return int(np.clip(k, 1, max(1, min(int(ceiling), (T - 1) // 5))))


def _default_n_perm(P, n_perm=None):
    """Permutation budget for the screening null.  One permutation costs one full
    statistic pass, so the budget tapers as ``P`` grows; 0 disables the test."""
    if n_perm is not None:
        return int(n_perm)
    if P <= 30:
        return 200
    if P <= 80:
        return 100
    return 50


# --------------------------------------------------------------------------- #
# the graph containers
# --------------------------------------------------------------------------- #
@dataclass
class CausalGraph:
    """A discovered window causal graph: edges, dense scores, and why it looks so.

    Attributes
    ----------
    edges : list of :class:`cafe._orient.Edge`
        Lagged edges first (``orientation='-->'``, ``identification_method='time'``),
        then one entry per contemporaneous candidate pair -- oriented, or explicitly
        abstained on with a ``reason``.
    n_vars, max_lag : int
        ``P`` and ``K``.
    diagnostics : :class:`cafe._diagnostics.Diagnostics` or None
        The statistics that drove per-component routing.  ``None`` only when the
        input was too degenerate to diagnose; :attr:`notes` then says why.
    notes : list of str
        Everything the pipeline decided to skip, degrade or fall back on.  A
        zero-config call on a degenerate input returns an empty-but-valid graph and
        the reason lands here (and in :meth:`summary`).
    names : list or None
        Variable labels recovered from the input container, when it had any.
    meta : dict
        Settings actually used plus per-stage timings.
    """

    edges: List[Edge]
    n_vars: int
    max_lag: int
    diagnostics: Optional[Diagnostics] = None
    notes: List[str] = field(default_factory=list)
    names: Optional[List[Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    # (K+1, P, P) tensors, axis 0 indexed by lag 0..K, entry [k, i, j] == "j -> i at k"
    _adj: Optional[np.ndarray] = None
    _scores: Optional[np.ndarray] = None
    _options: Dict[str, Any] = field(default_factory=dict)

    # ---- tensors -------------------------------------------------------- #
    def _tensor(self, which):
        P, K = int(self.n_vars), int(self.max_lag)
        A = self._adj if which == "adj" else self._scores
        if A is None:
            A = np.zeros((K + 1, P, P), dtype=bool if which == "adj" else float)
        return A

    @staticmethod
    def _slice(A, lag, exclude_self, directed_only_mask=None):
        K1 = A.shape[0]
        if lag is None:
            out = A.any(axis=0) if A.dtype == bool else A.max(axis=0)
        elif isinstance(lag, str):
            key = lag.lower()
            if key == "all":
                out = A.copy()
            elif key in ("lagged", "window"):
                out = A[1:].copy()
            elif key == "contemp":
                out = A[0].copy()
            else:
                raise ValueError("lag must be None, an int, 'all', 'lagged' or "
                                 f"'contemp'; got {lag!r}")
        else:
            k = int(lag)
            if not (0 <= k < K1):
                raise ValueError(f"lag {k} outside [0, {K1 - 1}]")
            out = A[k].copy()
        if exclude_self:
            if out.ndim == 3:
                idx = np.arange(out.shape[-1])
                out[:, idx, idx] = False if out.dtype == bool else 0.0
            else:
                np.fill_diagonal(out, False if out.dtype == bool else 0.0)
        return out

    def adjacency(self, lag=None, exclude_self=False):
        """Boolean adjacency, with the CAFE convention ``A[target, source]``.

        ``lag=None``      -- ``(P, P)``, the OR over every lag including 0 (the summary
                             graph: "does j drive i at any lag in the window").
        ``lag=k``         -- ``(P, P)`` at exactly lag ``k``; ``k=0`` is contemporaneous.
        ``lag="lagged"``  -- ``(K, P, P)`` window graph, axis 0 indexed by lag ``1..K``.
        ``lag="all"``     -- ``(K+1, P, P)``, axis 0 indexed by lag ``0..K``.

        Contemporaneous edges that were *not* oriented (``'o-o'``) or were judged
        latent-confounded (``'<->'``) fill BOTH entries: adjacency is supported, the
        direction is not claimed.  ``exclude_self`` drops the self-lag diagonal, which
        is a genuine AR term but is often excluded from a reported structure.
        """
        return self._slice(self._tensor("adj"), lag, exclude_self)

    def scores(self, lag=None, exclude_self=False):
        """Continuous edge strengths in ``[0, 1]``, same shapes as :meth:`adjacency`.

        Dense over every ordered pair, so it can be fed straight to an AUC/AUPRC
        scorer.  This is the EVIDENCE view; :meth:`adjacency` is the DECISION view.
        Keeping them separate matters: threshold-free metrics need a global ranking,
        while F1/SHD need a calibrated support, and the two want different operating
        points.

        The score blends the fitted coefficient magnitude (where the skeleton scored a
        pair) with the screening statistic (elsewhere), monotone in evidence and with
        **no tier boundary**.  Selection shifts the distribution strongly upward but does
        not partition the range, so a well-evidenced unselected pair can still outrank a
        weak selected one -- which is exactly what a ranking metric should reward.

        An earlier design hard-tiered selected edges into ``[0.5, 1]`` and the rest into
        ``[0, 0.5)``.  Measured on ground truth that cost ~0.34 of AUC (0.595 vs 0.936),
        because it buried every true-but-unselected edge beneath every false-but-selected
        one.  Do not reintroduce it.
        """
        return self._slice(self._tensor("scores"), lag, exclude_self)

    # ---- views ---------------------------------------------------------- #
    @property
    def n_edges(self):
        """Number of ordered adjacency entries in the summary (lag-OR) graph."""
        return int(self.adjacency().sum())

    def edges_at(self, lag):
        """The :class:`Edge` objects at one lag (``0`` == contemporaneous)."""
        return [e for e in self.edges if int(e.lag) == int(lag)]

    def label(self, i):
        """Variable label for index ``i`` (falls back to ``x{i}``)."""
        if self.names is not None and 0 <= i < len(self.names):
            return self.names[i]
        return f"x{i}"

    def _records(self):
        return [{
            "source": int(e.source), "target": int(e.target), "lag": int(e.lag),
            "source_name": str(self.label(int(e.source))),
            "target_name": str(self.label(int(e.target))),
            "orientation": str(e.orientation), "strength": float(e.strength),
            "confounding_prob": float(e.confounding_prob),
            "identification_method": str(e.identification_method),
            "confidence": float(e.confidence), "reason": str(getattr(e, "reason", "")),
        } for e in self.edges]

    _COLUMNS = ("source", "target", "lag", "source_name", "target_name", "orientation",
                "strength", "confounding_prob", "identification_method", "confidence",
                "reason")

    def to_pandas(self):
        """Edge list as a ``pandas.DataFrame`` (pandas is an optional dependency)."""
        try:
            import pandas as pd
        except ImportError as exc:                              # pragma: no cover
            raise ImportError("CausalGraph.to_pandas() needs pandas, which CAFE keeps "
                              "optional: pip install pandas") from exc
        return pd.DataFrame(self._records(), columns=list(self._COLUMNS))

    def to_polars(self):
        """Edge list as a ``polars.DataFrame`` (polars is an optional dependency)."""
        try:
            import polars as pl
        except ImportError as exc:                              # pragma: no cover
            raise ImportError("CausalGraph.to_polars() needs polars, which CAFE keeps "
                              "optional: pip install polars") from exc
        recs = self._records()
        if not recs:
            return pl.DataFrame({c: [] for c in self._COLUMNS})
        return pl.DataFrame({c: [r[c] for r in recs] for c in self._COLUMNS})

    # ---- reporting ------------------------------------------------------ #
    def summary(self):
        """A short human-readable report of what was found and how."""
        P, K = int(self.n_vars), int(self.max_lag)
        m = self.meta
        L = [f"CAFE-CD graph: P={P} vars, K={K} lags, "
             f"{len(self.edges)} edges ({self.n_edges} adjacency entries)"]
        shape = m.get("shape")
        if shape:
            L.append(f"  data      : {shape[0]} entities x {shape[1]} times x "
                     f"{shape[2]} vars, {m.get('observed_frac', float('nan')):.1%} observed")
        L.append(f"  engine    : {m.get('engine', '?')}"
                 + (f"  (loading homogeneity ratio {m['homogeneity_ratio']:.3f}"
                    f" vs null 1.0, pool at >= {HOMOGENEITY_POOL})"
                    if np.isfinite(m.get("homogeneity_ratio", np.nan)) else ""))
        if "n_candidates" in m:
            L.append(f"  screening : {m['n_candidates']} candidate triples "
                     f"({m.get('reduction', float('nan')):.2%} of {P}x{P}x{K})")
        if "n_support" in m:
            L.append(f"  skeleton  : {m['n_support']} selected lagged coefficients "
                     f"at lambda={m.get('lambda_selected', float('nan')):.4g}")
        by_lag = {}
        for e in self.edges:
            by_lag[int(e.lag)] = by_lag.get(int(e.lag), 0) + 1
        if by_lag:
            L.append("  by lag    : " + ", ".join(f"lag{k}={by_lag[k]}"
                                                  for k in sorted(by_lag)))
        by_m = {}
        for e in self.edges:
            by_m[e.identification_method] = by_m.get(e.identification_method, 0) + 1
        if by_m:
            L.append("  oriented  : " + ", ".join(f"{k}={v}" for k, v in
                                                  sorted(by_m.items())))
        d = self.diagnostics
        if d is not None:
            L.append(f"  diagnostics: |exc.kurt|={d.non_gaussianity_summary:.2f}, "
                     f"|skew|={d.skewness_summary:.2f}, "
                     f"nonlinearity={d.nonlinearity_summary:.3f}, "
                     f"latent={d.latent_evidence:.2f}, "
                     f"factor={d.factor_strength:.2f}, n_eff={d.n_eff:.0f}")
            L.append(f"  global route for a P-wide contemporaneous component: "
                     f"{route(d, P)}")
        if "t_total_s" in m:
            stages = ", ".join(f"{k[2:-2]}={m[k]:.2f}s" for k in
                               ("t_model_s", "t_screen_s", "t_skeleton_s",
                                "t_diagnose_s", "t_orient_s") if k in m)
            L.append(f"  timings   : total={m['t_total_s']:.2f}s ({stages})")
        for n in self.notes:
            L.append(f"  note      : {n}")
        return "\n".join(L)

    def __repr__(self):                                        # pragma: no cover
        return (f"CausalGraph(P={self.n_vars}, K={self.max_lag}, "
                f"edges={len(self.edges)}, engine={self.meta.get('engine')})")

    # ---- the causality certificate -------------------------------------- #
    def max_edge_revision(self, X, fn=None, **kw):
        """Largest revision to a REPORTED edge set when the future is taken away.

        Convenience wrapper around the module-level :func:`max_edge_revision`; with
        ``fn=None`` it re-runs *this* graph's own pipeline (same options) as the
        reporter, so ``g.max_edge_revision(X)["max_revision"]`` is CAFE-CD's own
        certificate.  Pass ``fn`` to audit a rival method with the identical protocol.

        The built-in reporter reads ``X`` as ``(T, P)`` or ``(E, T, P)`` and truncates
        the TIME axis; pass an explicit ``fn`` for any other layout.
        """
        if fn is None:
            opts = dict(self._options)
            opts.pop("mode", None)
            opts.pop("checkpoints", None)

            def fn(Xa, t):                                     # noqa: E306
                A = np.asarray(Xa)
                # truncate TIME: axis -2 for an (E, T, P) cube, axis 0 for (T, P)
                Xt = A[..., :int(t) + 1, :] if A.ndim == 3 else A[:int(t) + 1]
                return discover(Xt, **opts)
            kw.setdefault("as_of", True)
        return max_edge_revision(fn, X, **kw)


class PitGraphs:
    """A point-in-time sequence of :class:`CausalGraph`: ``G_t`` from rows ``<= t``.

    Indexable and iterable like a list (``G[0]``, ``G[-1]``, ``for g in G``), plus
    :meth:`at` to address a graph by its checkpoint time rather than its position.
    Each graph is produced by truncating the panel to ``t`` and running the identical
    batch pipeline, so nothing computed at ``t`` can see a row after ``t``.
    """

    def __init__(self, checkpoints, graphs, meta=None):
        self.checkpoints = [int(t) for t in checkpoints]
        self.graphs = list(graphs)
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return PitGraphs(self.checkpoints[i], self.graphs[i], self.meta)
        return self.graphs[i]

    def __iter__(self):
        return iter(self.graphs)

    def at(self, t):
        """The graph reported AT checkpoint time ``t`` (exact match required)."""
        t = int(t)
        if t not in self.checkpoints:
            raise KeyError(f"no checkpoint at t={t}; have {self.checkpoints}")
        return self.graphs[self.checkpoints.index(t)]

    @property
    def edges(self):
        """The edges of the LAST checkpoint (the graph as it stands now)."""
        return self.graphs[-1].edges if self.graphs else []

    @property
    def diagnostics(self):
        """The diagnostics of the last checkpoint."""
        return self.graphs[-1].diagnostics if self.graphs else None

    def adjacency(self, lag=None, exclude_self=False):
        """``(n_checkpoints, ...)`` stack of the per-checkpoint adjacencies."""
        if not self.graphs:
            return np.zeros((0, 0, 0), dtype=bool)
        return np.stack([g.adjacency(lag=lag, exclude_self=exclude_self)
                         for g in self.graphs])

    def scores(self, lag=None, exclude_self=False):
        """``(n_checkpoints, ...)`` stack of the per-checkpoint score tensors."""
        if not self.graphs:
            return np.zeros((0, 0, 0), dtype=float)
        return np.stack([g.scores(lag=lag, exclude_self=exclude_self)
                         for g in self.graphs])

    def max_edge_revision(self, X, fn=None, **kw):
        """This sequence's own truncation certificate (see :func:`max_edge_revision`)."""
        kw.setdefault("checkpoints", self.checkpoints)
        if fn is None and self.graphs:
            return self.graphs[-1].max_edge_revision(X, **kw)
        return max_edge_revision(fn, X, **kw)

    def summary(self):
        L = [f"CAFE-CD point-in-time sequence: {len(self.graphs)} checkpoints "
             f"{self.checkpoints}"]
        for t, g in zip(self.checkpoints, self.graphs):
            L.append(f"  t={t:>6}: {len(g.edges)} edges, "
                     f"{g.n_edges} adjacency entries, engine={g.meta.get('engine')}")
        return "\n".join(L)

    def __repr__(self):                                        # pragma: no cover
        return f"PitGraphs(n={len(self.graphs)}, checkpoints={self.checkpoints})"


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def _cube_from_input(data, panel=None):
    """Any supported container -> ``(cube (E,T,P), names, kind, ctx)``.

    2-D inputs become a single-entity cube.  Long-format DataFrames (``panel=(time_col,
    entity_col)``) and 3-D arrays are scattered into a dense ``(E, T, P)`` cube with
    ``np.nan`` at (entity, time) cells the input never carried -- which is exactly how
    CAFE already treats them.
    """
    if isinstance(data, np.ndarray) and panel is None:
        A = np.asarray(data, float)
        if A.ndim in (2, 3):
            cube = A[None] if A.ndim == 2 else A
            return np.ascontiguousarray(cube, dtype=float), None, "numpy", None
        if A.ndim == 1:
            return np.ascontiguousarray(A.reshape(1, -1, 1), dtype=float), None, \
                "numpy", None
    X, ctx = to_matrix(data, panel=panel)
    names = list(ctx.columns) if getattr(ctx, "columns", None) else None
    kind = getattr(ctx, "kind", "numpy")
    if getattr(ctx, "panel_shape", None) is not None:
        E, T, F = ctx.panel_shape
        return np.ascontiguousarray(X.reshape(E, T, F)), names, kind, ctx
    meta = getattr(ctx, "panel_meta", None)
    if meta is not None:
        eid = np.asarray(meta["entity_ids"], dtype=np.intp)
        tid = np.asarray(meta["time_ids"], dtype=np.intp)
        E = int(eid.max()) + 1 if eid.size else 1
        T = int(tid.max()) + 1 if tid.size else 1
        cube = np.full((E, T, X.shape[1]), np.nan, dtype=float)
        cube[eid, tid] = X
        return cube, names, kind, ctx
    return np.ascontiguousarray(X[None]), names, kind, ctx


def _residual_cube(res, cube_shape):
    """Residual cube ``(E, T, P)`` + observed mask from any CAFE result object.

    ``robust=False`` is deliberate and load-bearing (see the module docstring).
    Panel results are returned in the input's stacked row layout, which is C-order
    over ``(entity, time)``, so the reshape is exact.
    """
    E, T, P = cube_shape
    kw = dict(factor_adjusted=True, mask_imputed=True, robust=False)
    if hasattr(res, "_pe") or hasattr(res, "stream_result"):   # Panel / PooledPanel
        out = res.residuals(**kw)                              # always (E2, obs) stacked
        Es, obs_s = out
        Es = np.asarray(Es, float)
        obs_s = np.asarray(obs_s, bool)
        if Es.shape[0] == E * T:
            return Es.reshape(E, T, P), obs_s.reshape(E, T, P)
        return Es[None], obs_s[None]                           # pragma: no cover
    Es, obs_s = res.residuals(return_mask=True, **kw)
    Es = np.asarray(Es, float)
    obs_s = np.asarray(obs_s, bool)
    if Es.shape[0] == E * T and E > 1:                         # pragma: no cover
        return Es.reshape(E, T, P), obs_s.reshape(E, T, P)
    return Es[None], obs_s[None]


# --------------------------------------------------------------------------- #
# score assembly -- EVIDENCE, deliberately separate from the DECISION
# --------------------------------------------------------------------------- #
# ``adjacency()`` is the DECISION (the selected support: precision / recall / F1 /
# SHD read it) and ``scores()`` is the EVIDENCE (a dense continuous statistic:
# AUC / AUPRC read it).  Conflating them was a real defect: ``scores()`` used to
# partition its range, every selected edge landing in ``[0.5, 1]`` and every
# unselected pair in ``[0, 0.5)``.  That hard step ranked EVERY false positive above
# EVERY true edge the selector happened to miss, which threw away most of the AUC the
# pipeline's own components already had (measured on the ``gen_causal_panel`` fixture
# at P=24: 0.595 with the step, 0.974 without it).  The two consumers genuinely want
# different operating points -- AUC wants generous evidence, F1 wants a calibrated
# decision -- so the two outputs are now computed by different rules.
#
#: Blend weights for the lagged evidence score: (refit |t|, group-lasso path
#: evidence, screening statistic).  All three are rank-normalised first so they are
#: commensurable; the refit |t| carries the most weight because it is the strongest
#: single component (0.969 vs 0.939 for the screener and 0.699 for the path on the
#: reference cell), and the other two are kept because the blend beats every
#: component alone (0.974).
#: These are the weights of a SEPARABLE design.  Two of the three components are
#: fully conditional and :func:`_scoring_router` re-weights them structurally when the
#: design cannot tell its sources apart; at the router's null (orthogonal source
#: blocks) the weights are exactly these, so nothing here is conditional on the router
#: existing.
SCORE_WEIGHTS = (0.60, 0.20, 0.20)
#: Blend weights for the contemporaneous evidence score: (|partial correlation|,
#: oriented strength).
CONTEMP_WEIGHTS = (0.75, 0.25)
#: Largest multivariate design (``K * P`` columns) that is refitted in one solve for
#: the direct ``|t|``.  Above it the refit is done per target on that target's
#: candidate columns only, and the rest of the tensor falls back to the marginal
#: ``|t|`` -- which is dense, cheap and always defined.
DENSE_T_MAX_COLS = 1024
#: Per-target screening budget ``m_i = ceil(CAND_PER_TARGET * K * log(P) / sparsity)``.
#: See :func:`_augment_candidates` for why the budget is tied to ``P`` and not to
#: ``n``, which is what the screener's own ``sis_c`` floor uses.
CAND_PER_TARGET = 2.0


def _rank01(A):
    """Rank-normalise to ``(0, 1]``, mid-ranks for ties, non-finite ranked worst.

    Rank rather than min-max because the three evidence components live on
    incomparable scales (a t-statistic, a path occupancy fraction and a sign
    correlation).  A rank transform is strictly monotone, so it preserves each
    component's own ordering exactly while making the blend well posed.
    """
    a = np.asarray(A, dtype=float).reshape(-1)
    n = a.size
    if n == 0:
        return np.zeros(np.shape(A), dtype=float)
    a = np.where(np.isfinite(a), a, -np.inf)
    order = np.argsort(a, kind="mergesort")
    _, inv, cnt = np.unique(a[order], return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    mid = (csum - cnt + 1 + csum) / 2.0                # mid-rank of each tie group
    r = np.empty(n, dtype=float)
    r[order] = mid[inv]
    return (r / float(n)).reshape(np.shape(A))


def _design_index(K, P):
    """Column index arrays ``(kk, jj)`` of the stacked lag design, in ``(k, j)`` order."""
    return np.repeat(np.arange(1, K + 1), P), np.tile(np.arange(P), K)


def _marginal_t(Gt, n, K, P):
    """Dense marginal ``|t|`` of every ``(k, i, j)`` from the standardised Gram.

    ``Gt[k, 0, j, i] / n`` is the correlation between design column ``(j, k)`` and
    target ``i`` (both columns carry unit RMS after :meth:`MomentTensor.scaled`), so
    the usual ``|r| sqrt(n - 2) / sqrt(1 - r^2)`` applies.  Always available: it costs
    one slice and is defined for every pair, which is what makes it the floor of the
    fallback ladder.
    """
    r = np.clip(Gt[1:K + 1, 0, :, :].transpose(0, 2, 1) / max(n, 1.0), -0.999999,
                0.999999)
    return np.abs(r) * np.sqrt(max(n - 2.0, 1.0)) / np.sqrt(1.0 - r * r)


def _refit_t(mt, K, P, cand_mask=None, max_cols=DENSE_T_MAX_COLS):
    """Debiased REFIT ``|t|`` (and raw-unit coefficients) for every ``(k, i, j)``.

    This is the evidence the old two-tier score buried.  The group lasso shrinks its
    coefficients towards zero and reports nothing at all off its support; an ordinary
    least-squares refit on the same two-way-demeaned moments is unshrunk, is defined
    for every pair the design can carry, and measures each edge against its own
    standard error rather than against the other edges' magnitudes.

    Three tiers, in order of preference:

    1. ``K * P`` columns at once, when that design is affordable and identified.  The
       design is SHARED by every target, so one factorisation serves all ``P`` of them.
    2. per target, restricted to that target's candidate columns, when it is not.
    3. the marginal ``|t|`` for whatever cell neither tier reached -- see
       :func:`_marginal_t`.

    Returns ``(T, B_raw, tier)``: ``(K, P, P)`` statistics, ``(K, P, P)`` coefficients
    in the panel's own units, and the name of the tier that produced them.
    """
    Gt = mt.scaled(True)
    n = float(mt.n_tot)
    scale = mt.scale
    Tt = np.full((K, P, P), np.nan, dtype=float)
    Bt = np.zeros((K, P, P), dtype=float)
    kk, jj = _design_index(K, P)
    m = int(kk.size)

    def _solve(cols, targets):
        """OLS on ``cols`` (positions into the stacked design) for ``targets``."""
        kc, jc = kk[cols], jj[cols]
        G = Gt[kc[:, None], kc[None, :], jc[:, None], jc[None, :]]
        c = Gt[kc, 0, jc, :][:, targets]                       # (len(cols), n_tgt)
        A = G + 1e-8 * n * np.eye(G.shape[0])
        try:
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:                          # pragma: no cover
            Ai = np.linalg.pinv(A)
        B = Ai @ c
        # y'y == n exactly (unit-RMS target column), and at the OLS solution the
        # cross term collapses: rss = y'y - b'X'y.
        rss = np.maximum(n - (B * c).sum(axis=0), _EPS)
        dof = max(n - G.shape[0], 1.0)
        s2 = rss / dof
        vd = np.maximum(np.diag(Ai), 1e-300)
        return B, np.abs(B) / np.sqrt(np.outer(vd, s2))

    tier = "marginal"
    if m <= int(max_cols) and n >= m + 10:
        tgt = np.arange(P)
        B, Tst = _solve(np.arange(m), tgt)
        Tt = Tst.T.reshape(P, K, P).transpose(1, 0, 2)
        Bt = B.T.reshape(P, K, P).transpose(1, 0, 2)
        tier = "multivariate"
    elif cand_mask is not None:
        tier = "per_target"
        pos = np.arange(m).reshape(K, P)                       # pos[k-1, j]
        for i in range(P):
            sub = np.asarray(cand_mask[:, i, :], bool).copy()
            sub[:, i] = True                                   # own lags always in
            kidx, jidx = np.nonzero(sub)
            if kidx.size == 0 or n < kidx.size + 10:
                continue
            cols = pos[kidx, jidx]
            B, Tst = _solve(cols, np.array([i]))
            Tt[kidx, i, jidx] = Tst[:, 0]
            Bt[kidx, i, jidx] = B[:, 0]
    if tier != "marginal":
        # standardised -> the panel's own units (mirrors _skeleton's scale folding)
        den = np.maximum(scale[1:K + 1], _EPS)[:, None, :]     # (K, 1, P) over source j
        Bt = Bt * scale[0][None, :, None] / den
    fallback = _marginal_t(Gt, n, K, P)
    Tt = np.where(np.isfinite(Tt), Tt, fallback)
    return Tt, Bt, tier


def _path_evidence(skel, K, P):
    """The group lasso's OWN ranking: how much of the lambda path an edge survives.

    ``path_support[l]`` is the support at the ``l``-th lambda, so the number of path
    points that contain an edge is a direct measure of how early it entered and how
    long it stayed -- the classic regularisation-path notion of evidence strength, and
    the one quantity here that is genuinely the lasso's contribution rather than a
    refit's or the screener's.  Normalised to ``[0, 1]``; a pair the skeleton never
    scored gets 0 and is ranked by the other two components instead.
    """
    ps = getattr(skel, "path_support", None)
    if not ps:
        return None
    L = len(ps)
    cnt = np.zeros((K, P, P), dtype=float)
    for sup in ps:
        cnt += np.asarray(sup, bool)
    return cnt / float(L)


def _blend(components):
    """Weighted mean of rank-normalised evidence components -> ``(0, 1]``.

    Every component is rank-normalised first, so the result is continuous, dense, and
    monotone in each component separately.  Crucially it has NO step: a component that
    is zero off the selected support (the path evidence) can shift a score by at most
    its own weight, so a well-evidenced unselected edge still outranks a
    poorly-evidenced selected one.  That is the whole point of the rewrite.
    """
    comps = [(float(w), A) for (w, A) in components if A is not None and w > 0.0]
    if not comps:
        return None
    tot = sum(w for w, _ in comps)
    out = np.zeros(np.shape(comps[0][1]), dtype=float)
    for w, A in comps:
        out += (w / tot) * _rank01(A)
    return np.clip(out, 0.0, 1.0)


def _scoring_router(mt, K, P, notes=None):
    """Structural route for the lagged EVIDENCE blend.  Never sees an outcome.

    The three statistics the blend is built from do not need the same assumption.
    The refit ``|t|`` and the group lasso's path occupancy are both **fully
    conditional**: they estimate what one source contributes with every other source
    held fixed, and they are only readable as statements about that source while the
    design can tell the sources apart.  The screening statistic is not conditional and
    is unaffected.  So the router asks one structural question --

        *how much of each source's identifying variation is its own?*

    -- and answers it with :func:`cafe._skeleton.conditioning_diagnostic`, whose
    per-source tolerance ``tol_j`` is a generalised-VIF reciprocal in ``(0, 1]`` with a
    null of **1 by construction** (orthogonal source blocks) and no calibration step.
    Writing ``own_j = tol_j`` and ``shared_j = 1 - tol_j``:

    ``w_bivariate = mean_j max(0, shared_j - own_j)``
        the average amount by which what a source shares with its competitors exceeds
        what is its own.  It is exactly 0 while every source clears
        :data:`SOURCE_SEPARABILITY` -- so on a separable design the blend is bit-for-bit
        what it was before this router existed -- and rises to 1 as the sources become
        indistinguishable.  That weight is carved OUT of the conditional refit's share
        and given to :func:`cafe._skeleton.bivariate_granger`, the same estimator with
        the one assumption the design cannot support dropped.  It is a blend and not a
        switch because the bivariate branch buys identification with omitted-variable
        bias: it is better evidence only in proportion to how badly the conditional
        design is confounded with itself, which is exactly what the weight measures.

    ``path_weight = mean_j tol_j``
        the group lasso admits ONE member of a set of near-duplicate sources and
        suppresses the rest -- its selection is only consistent when the sources are
        separable (the irrepresentable condition) -- so the fraction of the path
        statistic that carries information about *which* source is the true one is the
        separable fraction itself.  Multiplying its weight by the mean tolerance is
        that statement and nothing more; at the null it is 1 and nothing changes.

    Both numbers are functions of the design's second moments alone, so a truncated
    prefix produces the router decision that prefix is entitled to and ``mode="pit"``
    stays truncation-invariant by construction.

    Returns ``(w_bivariate, path_weight, parts)``.
    """
    parts = {"source_tolerance": 1.0, "min_source_tolerance": 1.0,
             "n_sources_below_half": 0, "w_bivariate": 0.0, "path_weight": 1.0,
             "branch": "conditional", "ok": False, "reason": "no moment tensor"}
    if mt is None:
        return 0.0, 1.0, parts
    try:
        d = conditioning_diagnostic(mt, max_lag=K, max_cols=DENSE_T_MAX_COLS)
    except Exception as exc:                                   # pragma: no cover
        parts["reason"] = f"{type(exc).__name__}: {exc}"
        return 0.0, 1.0, parts
    parts["reason"] = d["reason"]
    parts["ok"] = bool(d["ok"])
    if not d["ok"]:
        # No diagnostic -> keep the fully conditional evidence.  That is the branch
        # whose assumption we cannot check, but it is also the incumbent, so an
        # unmeasurable design changes nothing rather than being routed on a guess.
        if notes is not None and d["reason"]:
            notes.append("scoring router abstained: " + d["reason"])
        return 0.0, 1.0, parts
    tol = np.asarray(d["block_tolerance"], float)
    w_biv = float(np.mean(np.clip(1.0 - tol / max(SOURCE_SEPARABILITY, _EPS), 0.0, 1.0)))
    parts.update(source_tolerance=float(d["tolerance"]),
                 min_source_tolerance=float(d["min_tolerance"]),
                 n_sources_below_half=int(d["n_below_half"]),
                 w_bivariate=w_biv, path_weight=float(d["tolerance"]),
                 branch=("conditional" if w_biv <= 0.0 else
                         ("bivariate" if w_biv >= 1.0 else "blended")))
    if notes is not None and w_biv > 0.0:
        notes.append(
            f"scoring router: {d['n_below_half']}/{P} source blocks keep less than "
            f"half their identifying variation to themselves (mean source tolerance "
            f"{d['tolerance']:.3f}); {w_biv:.2f} of the conditional evidence weight "
            f"moved to bivariate Granger")
    return w_biv, float(d["tolerance"]), parts


def _augment_candidates(S, cands, K, P, sparsity=1.0):
    """Union the permutation survivors with a per-target top-``m`` budget.

    The screener offers two selectors and neither is right on its own here:

    * the permutation threshold alone (``sis_c = 0``, what this assembler used to ask
      for) is a family-wise-error test on a MAX statistic.  It controls false
      positives beautifully and is far too aggressive for structure recovery on short
      panels -- measured recall of the true triples was 0.30 at ``T = 120``;
    * the screener's SIS floor ``m_i = ceil(sis_c * n / log(P*K))`` ties the budget to
      the SAMPLE SIZE, which is the wrong quantity.  In the ``n >> P*K`` regime this
      assembler mostly lives in it keeps EVERYTHING (reduction 1.00, i.e. no screening
      at all), and in the ``n ~ P*K`` regime it starves.

    The number of candidates worth keeping per target scales with the expected
    in-degree and the dimension, not with how long the series is, so the budget here
    is ``m_i = ceil(CAND_PER_TARGET * K * log(P) / sparsity)`` -- capped at ``P * K``,
    and unioned with (never replacing) the permutation survivors and the forced self
    lags, so strong evidence is never dropped to stay inside the budget.  Measured on
    the ``gen_causal_panel`` family at the default ``sparsity = 1``: recall
    0.94 / 0.92 / 0.95 / 0.99 at reduction 0.50 / 0.31 / 0.16 / 0.11 for
    ``P = 12 / 24 / 60 / 100``.

    Returns ``(cand_dict, cand_mask, m_i)``.
    """
    S = np.asarray(S, float)
    mask = np.zeros((K, P, P), dtype=bool)
    for i, pairs in (cands or {}).items():
        for (j, k) in pairs:
            if 1 <= int(k) <= K:
                mask[int(k) - 1, int(i), int(j)] = True
    s = max(float(sparsity), _EPS)
    m_i = int(np.ceil(CAND_PER_TARGET * K * np.log(max(P, 3)) / s))
    m_i = int(np.clip(m_i, 1, P * K))
    for i in range(P):
        Si = np.where(np.isfinite(S[:, i, :]), S[:, i, :], -np.inf).copy()
        Si[:, i] = -np.inf                              # self lags are forced, not ranked
        flat = Si.reshape(-1)
        take = min(m_i, flat.size)
        if take > 0:
            top = np.argpartition(flat, flat.size - take)[flat.size - take:]
            sel = np.zeros(flat.size, dtype=bool)
            sel[top] = True
            mask[:, i, :] |= sel.reshape(K, P)
        mask[:, i, i] = True
    out = {}
    for i in range(P):
        kidx, jidx = np.nonzero(mask[:, i, :])
        order = np.lexsort((kidx, jidx))
        out[i] = [(int(jidx[o]), int(kidx[o]) + 1) for o in order]
    return out, mask, m_i


def _empty_graph(P, K, notes, names=None, meta=None, options=None, diag=None):
    P, K = int(max(P, 0)), int(max(K, 1))
    meta = dict(meta or {})
    meta["notes"] = list(notes)                 # mirrored so meta-only readers see it
    return CausalGraph(edges=[], n_vars=P, max_lag=K, diagnostics=diag,
                       notes=list(notes), names=names, meta=meta,
                       _adj=np.zeros((K + 1, P, P), dtype=bool),
                       _scores=np.zeros((K + 1, P, P), dtype=float),
                       _options=dict(options or {}))


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def _pipeline(cube, names, options):
    """Residualise -> route -> screen -> skeleton -> diagnose -> orient -> assemble.

    Never raises on degenerate input: every stage is guarded and a failure is recorded
    in ``notes`` before an empty-but-valid graph is returned.
    """
    t0 = time.perf_counter()
    notes = []
    o = options
    cube = np.ascontiguousarray(np.asarray(cube, float))
    E, T, P = cube.shape
    obs_in = np.isfinite(cube)
    meta = {"shape": (int(E), int(T), int(P)),
            "observed_frac": float(obs_in.mean()) if obs_in.size else 0.0,
            "engine": "per_entity", "homogeneity_ratio": float("nan")}

    K = int(o["max_lag"]) if o.get("max_lag") is not None else default_max_lag(T, P)
    K = int(np.clip(K, 1, max(1, T - 2)))
    meta["max_lag"] = K

    # ---- 0. degeneracy gate -------------------------------------------------- #
    if P < 2:
        notes.append(f"P={P}: a graph needs at least 2 variables")
        return _empty_graph(P, K, notes, names, meta, o)
    if not obs_in.any():
        notes.append("no observed cells: nothing to estimate")
        return _empty_graph(P, K, notes, names, meta, o)
    if T < K + 4:
        notes.append(f"T={T} rows against a K={K} lag window: not enough usable "
                     "rows to estimate any edge")
        return _empty_graph(P, K, notes, names, meta, o)

    # ---- 1. engine routing (structural; panels only) ------------------------- #
    engine = o.get("engine")
    if engine is None:
        if E > 1:
            engine, parts = choose_engine(cube, r=o.get("rank"))
            meta["homogeneity_ratio"] = float(parts["ratio"])
            meta["homogeneity"] = {k: parts[k] for k in
                                   ("within", "between", "r", "n_pairs", "reason")}
            if parts["reason"]:
                notes.append("engine router: " + parts["reason"])
        else:
            engine = "per_entity"
    meta["engine"] = engine

    # ---- 2. residualise ------------------------------------------------------ #
    t = time.perf_counter()
    run_in = cube if E > 1 else cube[0]
    try:
        res = CAFE().run(run_in, engine=engine)
    except Exception as exc:                                   # degrade, never raise
        if engine == "pooled":
            notes.append(f"pooled engine failed ({type(exc).__name__}: {exc}); "
                         "fell back to per_entity")
            engine = meta["engine"] = "per_entity"
            try:
                res = CAFE().run(run_in, engine=engine)
            except Exception as exc2:                          # pragma: no cover
                notes.append(f"CAFE run failed ({type(exc2).__name__}: {exc2})")
                return _empty_graph(P, K, notes, names, meta, o)
        else:
            notes.append(f"CAFE run failed ({type(exc).__name__}: {exc})")
            return _empty_graph(P, K, notes, names, meta, o)
    try:
        Eres, obs = _residual_cube(res, (E, T, P))
    except Exception as exc:                                   # pragma: no cover
        notes.append(f"residual read-out failed ({type(exc).__name__}: {exc})")
        return _empty_graph(P, K, notes, names, meta, o)
    obs = obs & np.isfinite(Eres)
    meta["t_model_s"] = time.perf_counter() - t
    if obs.sum() < max(4 * P, 3 * (K + 1)):
        notes.append(f"only {int(obs.sum())} observed residual cells: too few to "
                     "estimate a graph")
        return _empty_graph(P, K, notes, names, meta, o)

    return _pipeline_post(Eres, obs, names, o, meta, notes, t0)


def _pipeline_post(Eres, obs, names, o, meta, notes, t0):
    """Everything after residualisation: screen -> skeleton -> diagnose ->
    orient -> assemble.  Split out so ``res.discover()`` can reuse an existing
    CAFE pass without re-running the model."""
    E, T, P = Eres.shape
    K = int(meta["max_lag"])
    sparsity = float(o.get("sparsity", 1.0))
    if not (sparsity > 0.0):
        raise ValueError(f"sparsity must be > 0; got {sparsity!r}")

    # ---- 3a. two-way within transform, ONCE, shared by everything downstream -- #
    # The skeleton has always demeaned internally (``demean="twoway"``); the screener
    # and the score assembly were reading the RAW residual, which still carries the
    # part of the common factor the one-pass model did not remove.  On the reference
    # panel that alone cost the screening statistic 0.605 vs 0.939 of AUC and dropped
    # its recall of the true triples to 0.30.  The transform is a projection, so
    # applying it here and letting the skeleton apply it again is idempotent; the
    # time mean is a contemporaneous cross-section and the entity mean is taken over
    # exactly the rows this call was handed, so ``mode="pit"`` stays truncation-
    # invariant by construction.
    Vf = obs.astype(float)
    mt = None
    if (K + 1) ** 2 * P * P * 8 <= _MOMENT_BUDGET_BYTES:
        try:
            mt = build_moment_tensor(Eres, obs=obs, max_lag=K, demean="twoway")
        except Exception as exc:                               # pragma: no cover
            notes.append(f"moment tensor unavailable ({type(exc).__name__}: {exc}); "
                         "edge evidence falls back to the screening statistic")
    if mt is not None:
        Zd = mt.Z
    else:
        Zd = _two_way_demean(np.where(obs, Eres, 0.0), Vf, "twoway")
        if mt is None and "moment tensor unavailable" not in " ".join(notes):
            notes.append(f"moment tensor would exceed {_MOMENT_BUDGET_BYTES / 1e9:.1f} "
                         f"GB at K={K}, P={P}; edge evidence falls back to the "
                         "screening statistic")

    # ---- 3b. screen lags ----------------------------------------------------- #
    t = time.perf_counter()
    n_perm = _default_n_perm(P, o.get("n_perm"))
    try:
        scr = screen_lags(Zd, obs=obs, max_lag=K, stat=o.get("stat", "phi"),
                          method=o.get("screen_method", "auto"),
                          alpha=float(o.get("alpha", 0.05)), n_perm=n_perm,
                          sis_c=float(o.get("sis_c", 0.0)),
                          seed=int(o.get("seed", 0)), force_self=True)
    except Exception as exc:
        notes.append(f"lag screening failed ({type(exc).__name__}: {exc})")
        return _empty_graph(P, K, notes, names, meta, o)
    cand_dict, cand_mask, m_i = _augment_candidates(scr.S, scr.candidates, K, P,
                                                    sparsity=sparsity)
    meta["t_screen_s"] = time.perf_counter() - t
    meta["n_candidates"] = int(cand_mask.sum())
    meta["n_candidates_threshold"] = int(scr.n_candidates)
    meta["reduction"] = float(cand_mask.sum()) / float(max(P * P * K, 1))
    meta["cand_budget_per_target"] = int(m_i)
    meta["sparsity"] = float(sparsity)
    meta["n_perm"] = int(n_perm)
    meta["screen_method"] = scr.meta.get("method")

    # ---- 4. panel group lasso ------------------------------------------------ #
    t = time.perf_counter()
    skel = None
    if meta["n_candidates"]:
        try:
            skel = panel_group_lasso(
                Eres, candidates=cand_dict, obs=obs, max_lag=K,
                kappa=float(o.get("kappa", 4.0)), select=o.get("select", "ebic"),
                screening=o.get("screening", "kkt"), fit_deviations=(E > 1),
                seed=int(o.get("seed", 0)))
        except Exception as exc:
            notes.append(f"skeleton fit failed ({type(exc).__name__}: {exc}); "
                         "falling back to the screened candidate set as the support")
    meta["t_skeleton_s"] = time.perf_counter() - t

    support = np.zeros((K, P, P), dtype=bool)
    mag = np.zeros((K, P, P), dtype=float)
    if skel is not None:
        support = np.asarray(skel.support, bool)
        mag = np.abs(np.asarray(skel.B0, float))
        if getattr(skel, "Delta", None):
            dev = [np.abs(np.asarray(D, float)) for D in skel.Delta.values()]
            if dev:
                mag = mag + np.mean(dev, axis=0)
        meta["n_support"] = int(support.sum())
        meta["lambda_selected"] = float(skel.lambda_selected)
        # The DECISION's own conditional recall: how much of the candidate set the
        # lasso keeps.  Reported so the screener's contribution and the lasso's can be
        # read apart instead of being blamed on each other.
        meta["skeleton_keep_rate"] = (float(support.sum()) /
                                      float(max(meta["n_candidates"], 1)))
    else:                                                      # screening-only fallback
        support = cand_mask.copy()
        mag = np.asarray(scr.S, float)
        meta["n_support"] = int(support.sum())

    # ---- 5. diagnostics ------------------------------------------------------ #
    t = time.perf_counter()
    diag = None
    try:
        diag = diagnose(Eres, obs=obs, skeleton=support.any(axis=0))
    except Exception as exc:                                   # pragma: no cover
        notes.append(f"diagnostics failed ({type(exc).__name__}: {exc})")
    meta["t_diagnose_s"] = time.perf_counter() - t

    # ---- 6. contemporaneous candidate graph ---------------------------------- #
    C, npairs = _masked_corr(Eres, obs)
    n_eff = float(diag.n_eff) if diag is not None else float(np.median(np.diag(npairs)))
    use_partial = bool(o.get("partial", True)) and n_eff > P + 10
    R = _partial_corr(C) if use_partial else C
    if not use_partial:
        notes.append(f"contemporaneous screening used marginal (not partial) "
                     f"correlation: n_eff={n_eff:.0f} <= P+10={P + 10}")
    bg0 = np.abs(R)
    np.fill_diagonal(bg0, 0.0)
    A0 = np.zeros((P, P), dtype=bool)
    if o.get("contemp", True) and P >= 2 and n_eff > 6:
        m_pairs = max(P * (P - 1) // 2, 1)
        dof = max(n_eff - (P if use_partial else 0) - 3.0, 1.0)
        z = _ppf(1.0 - float(o.get("alpha", 0.05)) / (2.0 * m_pairs))
        thr = float(np.tanh(z / np.sqrt(dof)))
        A0 = bg0 > thr
        A0 = A0 | A0.T
        np.fill_diagonal(A0, False)
        meta["contemp_threshold"] = thr
        meta["n_contemp_candidates"] = int(A0.sum() // 2)
        cap = int(o.get("max_contemp", 4 * P))
        if A0.sum() // 2 > cap:                # keep the strongest ``cap`` pairs only
            iu = np.triu_indices(P, 1)
            vals = np.where(A0[iu], bg0[iu], -np.inf)
            keep = np.argsort(vals)[::-1][:cap]
            B = np.zeros((P, P), dtype=bool)
            B[iu[0][keep], iu[1][keep]] = True
            A0 = (B | B.T) & A0
            notes.append(f"contemporaneous candidates capped at {cap} pairs "
                         f"(of {meta['n_contemp_candidates']}) by |partial corr|")
    elif not o.get("contemp", True):
        notes.append("contemporaneous stage disabled (contemp=False)")

    # ---- 7. orient ----------------------------------------------------------- #
    t = time.perf_counter()
    lag_spec = {int(k + 1): np.where(support[k], np.maximum(mag[k], 1e-12), 0.0)
                for k in range(K) if support[k].any()}
    edges = []
    try:
        edges = orient(Eres, contemp_candidates=A0, lagged_support=lag_spec,
                       diag=diag, obs=obs)
    except Exception as exc:                                   # pragma: no cover
        notes.append(f"orientation failed ({type(exc).__name__}: {exc}); "
                     "lagged edges only")
        for k in range(K):
            for i, j in zip(*np.nonzero(support[k])):
                edges.append(Edge(source=int(j), target=int(i), lag=int(k + 1),
                                  orientation="-->", strength=float(mag[k, i, j]),
                                  confounding_prob=0.0, identification_method="time",
                                  confidence=1.0, reason="oriented by the arrow of time"))
    meta["t_orient_s"] = time.perf_counter() - t

    # ---- 8. assemble --------------------------------------------------------- #
    # adjacency := the DECISION (the selected support).  scores := the EVIDENCE, a
    # dense continuous blend with no tier boundary anywhere in it.
    adj = np.zeros((K + 1, P, P), dtype=bool)
    sco = np.zeros((K + 1, P, P), dtype=float)
    adj[1:] = support

    t_direct, b_refit, t_tier = None, None, "unavailable"
    if mt is not None:
        try:
            t_direct, b_refit, t_tier = _refit_t(mt, K, P, cand_mask=cand_mask)
        except Exception as exc:                               # pragma: no cover
            notes.append(f"refit |t| unavailable ({type(exc).__name__}: {exc}); "
                         "edge evidence uses the screening statistic only")
    meta["score_tier"] = t_tier

    # ---- 8a. structural route of the evidence blend -------------------------- #
    # Both conditional components (the refit |t| and the lasso path) are only readable
    # as statements about ONE source while the design can tell the sources apart.
    # _scoring_router measures that and nothing else; at its null the weights below are
    # exactly SCORE_WEIGHTS, so a separable design is unaffected.
    w_biv, w_pathmul, cond_parts = _scoring_router(mt, K, P, notes=notes)
    meta["conditioning"] = dict(cond_parts)
    t_biv = None
    if w_biv > 0.0 and mt is not None:
        try:
            t_biv = bivariate_granger(mt, max_lag=K)
        except Exception as exc:                               # pragma: no cover
            notes.append(f"bivariate Granger evidence unavailable "
                         f"({type(exc).__name__}: {exc}); the blend stays conditional")
            w_biv, t_biv = 0.0, None
            meta["conditioning"]["w_bivariate"] = 0.0
            meta["conditioning"]["branch"] = "conditional"
    w_t, w_p, w_s = SCORE_WEIGHTS
    ev = _blend([(w_t * (1.0 - w_biv), t_direct),
                 (w_t * w_biv, t_biv),
                 (w_p * w_pathmul,
                  _path_evidence(skel, K, P) if skel is not None else None),
                 (w_s, np.abs(np.asarray(scr.S, float)))])
    sco[1:] = ev if ev is not None else 0.0

    sel0 = np.zeros((P, P), dtype=bool)
    str0 = np.zeros((P, P), dtype=float)
    for e in edges:
        if int(e.lag) != 0:
            continue
        s, tg = int(e.source), int(e.target)
        w = float(e.strength) if np.isfinite(e.strength) else 0.0
        if e.orientation in ("-->", "o->"):
            sel0[tg, s] = True
            str0[tg, s] = max(str0[tg, s], w)
        else:                                  # 'o-o' / '<->': adjacency, no direction
            sel0[tg, s] = sel0[s, tg] = True
            str0[tg, s] = max(str0[tg, s], w)
            str0[s, tg] = max(str0[s, tg], w)
    adj[0] = sel0
    w_c, w_o = CONTEMP_WEIGHTS
    ev0 = _blend([(w_c, bg0), (w_o, str0)])
    if ev0 is not None:
        np.fill_diagonal(ev0, 0.0)             # a contemporaneous self-loop is not an edge
        sco[0] = ev0

    meta["t_total_s"] = time.perf_counter() - t0
    meta["n_edges"] = len(edges)
    meta["notes"] = list(notes)
    return CausalGraph(edges=edges, n_vars=P, max_lag=K, diagnostics=diag, notes=notes,
                       names=names, meta=meta, _adj=adj, _scores=sco,
                       _options=dict(o))


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
_DEFAULTS = dict(max_lag=None, engine=None, alpha=0.05, n_perm=None, stat="phi",
                 screen_method="auto", select="ebic", screening="kkt", kappa=4.0,
                 contemp=True, partial=True, max_contemp=None, rank=None, seed=0,
                 sis_c=0.0)


def _resolve(options, kwargs):
    o = dict(_DEFAULTS)
    o.update(options or {})
    unknown = set(kwargs) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"discover() got unexpected keyword argument(s) "
                        f"{sorted(unknown)}; known: {sorted(_DEFAULTS)}")
    o.update(kwargs)
    if o.get("max_contemp") is None:
        o.pop("max_contemp")
    return o


def discover(data, mode="batch", checkpoints=None, panel=None, n_checkpoints=5,
             **kwargs):
    """Discover a window causal graph.  Zero-config: ``cafe.discover(df)`` is enough.

    Parameters
    ----------
    data : array / DataFrame
        2-D ``(T, P)``, 3-D ``(E, T, P)``, or a long-format pandas/polars frame with
        ``panel=(time_col, entity_col)``.  NaN marks missing.  Rows are time.
    mode : {"batch", "pit"}
        ``"batch"`` returns one :class:`CausalGraph` estimated on everything.
        ``"pit"`` returns a :class:`PitGraphs` sequence: ``G_t`` computed from rows
        ``<= t`` ONLY, at each checkpoint.  Both are point-in-time; ``"pit"`` exists so
        the *sequence* can be audited, and because a backtest wants the graph as it
        stood, not the graph as it ended up.
    checkpoints : sequence of int, optional
        ``mode="pit"`` only.  Default: ``n_checkpoints`` times spread over the second
        two-thirds of the sample (the first third is warm-up).
    panel : (time_col, entity_col), optional
        Marks a long-format DataFrame as panel data.
    n_checkpoints : int
        Number of default checkpoints when ``checkpoints`` is not given.

    Other Parameters
    ----------------
    max_lag : int, optional
        Window length ``K``.  Default: :func:`default_max_lag` (Schwert's rule on
        ``T``, capped at 12).
    engine : {"per_entity", "pooled"}, optional
        Panel residualisation engine.  Default: routed by
        :func:`loading_homogeneity_ratio`.
    alpha : float
        Family-wise error rate, shared by the screening permutation threshold and the
        contemporaneous Fisher-z threshold (Bonferroni over pairs).
    n_perm : int, optional
        Screening permutations.  Default 200 / 100 / 50 as ``P`` crosses 30 / 80.
    select : {"ebic", "forward"}
        Skeleton model selection.  ``"ebic"`` is the default because the SUPPORT is
        the deliverable here and EBIC is selection-consistent; ``"forward"`` minimises
        predictive risk and buys many near-zero false edges for it.
    contemp : bool
        Estimate and orient the contemporaneous layer at all.
    partial : bool
        Use partial (rather than marginal) correlation for the contemporaneous
        candidate graph.  Automatically disabled when ``n_eff <= P + 10``.
    sis_c : float
        Sure-independence-screening floor constant handed to
        :func:`cafe._screen.screen_lags`.  **Default 0** -- the floor keeps
        ``ceil(sis_c * n / log(P*K))`` triples per target, which in the ``n >> P*K``
        regime this assembler mostly lives in is every triple there is, so it would
        silently override the permutation threshold and hand the whole space
        downstream.  Raise it (0.05-1.0) in the genuinely high-dimensional
        ``P * K >> n`` regime, where it is what buys the sure-screening property.
    stat, screen_method, screening, kappa, rank, max_contemp, seed
        Passed through to the components; see their docstrings.

    Returns
    -------
    :class:`CausalGraph`, or :class:`PitGraphs` when ``mode="pit"``.

    Notes
    -----
    Degenerate input never raises: ``P < 2``, ``T`` shorter than the lag window, an
    all-NaN or constant column, or a component that fails to fit all return a valid
    graph whose :attr:`CausalGraph.notes` says what happened.
    """
    o = _resolve(None, kwargs)
    cube, names, _kind, _ctx = _cube_from_input(data, panel=panel)
    E, T, P = cube.shape
    mode = str(mode).lower()
    if mode not in ("batch", "pit"):
        raise ValueError(f"mode must be 'batch' or 'pit'; got {mode!r}")
    if mode == "batch":
        return _pipeline(cube, names, o)

    if checkpoints is None:
        lo = max(int(T // 3), 8)
        cps = np.unique(np.linspace(min(lo, T - 1), T - 1,
                                    max(int(n_checkpoints), 1)).astype(int))
    else:
        cps = np.unique(np.asarray(list(checkpoints), dtype=int))
    cps = [int(t) for t in cps if 0 <= t < T]
    graphs = [_pipeline(cube[:, :t + 1, :], names, o) for t in cps]
    return PitGraphs(cps, graphs,
                     meta={"shape": (int(E), int(T), int(P)), "options": dict(o)})


def _result_discover(self, **kwargs):
    """Discover a causal graph from an ALREADY-COMPUTED CAFE pass (no re-fit).

    Reuses this result's own point-in-time residuals, so it is exactly
    ``cafe.discover(same_data)`` minus the model run.  The engine is whatever this
    result was produced with; ``engine=`` is therefore not accepted here.
    """
    if "engine" in kwargs:
        raise TypeError("engine= is fixed by the run that produced this result; "
                        "call cafe.discover(data, engine=...) instead")
    o = _resolve(None, kwargs)
    t0 = time.perf_counter()
    ctx = getattr(self, "_ctx", None)
    names = list(ctx.columns) if getattr(ctx, "columns", None) else None
    shape = getattr(ctx, "panel_shape", None)
    X = np.asarray(self._X, float)
    if shape is None:
        meta = getattr(ctx, "panel_meta", None)
        if meta is not None:
            eid = np.asarray(meta["entity_ids"], np.intp)
            tid = np.asarray(meta["time_ids"], np.intp)
            shape = (int(eid.max()) + 1, int(tid.max()) + 1, X.shape[1])
        else:
            shape = (1, X.shape[0], X.shape[1])
    E, T, P = shape
    notes = []
    obs_in = np.isfinite(X)
    meta = {"shape": (int(E), int(T), int(P)), "reused_result": type(self).__name__,
            "observed_frac": float(obs_in.mean()) if obs_in.size else 0.0,
            "engine": ("pooled" if hasattr(self, "stream_result") else
                       ("per_entity" if hasattr(self, "_pe") else "per_entity")),
            "homogeneity_ratio": float("nan")}
    K = int(o["max_lag"]) if o.get("max_lag") is not None else default_max_lag(T, P)
    K = int(np.clip(K, 1, max(1, T - 2)))
    meta["max_lag"] = K
    if P < 2 or T < K + 4:
        notes.append(f"P={P}, T={T} with K={K}: not enough usable data")
        return _empty_graph(P, K, notes, names, meta, o)
    try:
        Eres, obs = _residual_cube(self, (E, T, P))
    except Exception as exc:                                   # pragma: no cover
        notes.append(f"residual read-out failed ({type(exc).__name__}: {exc})")
        return _empty_graph(P, K, notes, names, meta, o)
    obs = obs & np.isfinite(Eres)
    if obs.sum() < max(4 * P, 3 * (K + 1)):
        notes.append(f"only {int(obs.sum())} observed residual cells: too few to "
                     "estimate a graph")
        return _empty_graph(P, K, notes, names, meta, o)
    return _pipeline_post(Eres, obs, names, o, meta, notes, t0)


# --------------------------------------------------------------------------- #
# the certificate: does the reported edge set get rewritten by the future?
# --------------------------------------------------------------------------- #
def _to_indicator(g, lag, P_hint=None):
    """Coerce anything discover-like into a 0/1 float adjacency indicator."""
    if g is None:
        return np.zeros((int(P_hint or 0), int(P_hint or 0)), dtype=float)
    if hasattr(g, "adjacency"):
        A = g.adjacency(lag=lag)
    elif isinstance(g, dict) and "adjacency" in g:
        A = g["adjacency"]
    else:
        A = np.asarray(g)
        if A.ndim == 3 and lag is None:
            A = A.any(axis=0) if A.dtype == bool else (np.abs(A) > 0).any(axis=0)
    A = np.asarray(A)
    return (np.abs(np.asarray(A, float)) > 0).astype(float)


def max_edge_revision(fn, X, checkpoints=None, n_prefixes=6, tol=0.0, lag=None,
                      as_of=None, panel_time_axis=None):
    """Truncation certificate for a discovered edge set: ``max_revision``.

    The number this returns is the largest change to a graph a method **already
    reported at time t** once the rows after ``t`` are taken away.  ``0.0`` exactly
    means no reported edge was ever revised -- the reported object is strictly
    point-in-time.  Anything ``> 0`` means an edge flipped, i.e. the "graph as of
    ``t``" silently used data from after ``t``.  It is
    :func:`cafe.audit.verify_causal_series` applied to the 0/1 support indicator, so
    the same certificate CAFE publishes for its imputations.

    Parameters
    ----------
    fn : callable
        Either of two protocols, auto-detected from the arity (override with
        ``as_of``):

        ``fn(X, t) -> graph``  (**as-of protocol**, the informative one)
            The method is handed the WHOLE array and asked what it reports as of time
            ``t``.  This is what catches look-ahead: a method that fits on everything
            and then snapshots, or that centres its window on ``t``, reports something
            different once the future is removed.  Use this to audit a rival.

        ``fn(X) -> graph``  (**prefix protocol**)
            The report at ``t`` is *defined* as ``fn(X[:t + 1])``.  Any deterministic,
            shape-stable method passes this by construction, so it certifies
            determinism and shape stability rather than the absence of look-ahead.

        ``graph`` may be a :class:`CausalGraph` (or anything with ``.adjacency(lag=)``),
        a ``(P, P)`` / ``(K, P, P)`` array, or a dict with an ``"adjacency"`` key.
    X : ndarray
        ``(T, P)``, or a panel.  A 3-D array is assumed ``(E, T, P)`` and is moved to
        time-major internally so the prefixes cut TIME, not entities; pass
        ``panel_time_axis`` to say otherwise.
    checkpoints : sequence of int, optional
        Times at which the reported object is compared.  Default: the prefix cut points.
    n_prefixes : int
        How many growing prefixes to re-run on.
    tol : float
        Revision tolerance.  ``0.0`` (the default) demands an EXACT certificate, which
        is the right bar for a 0/1 indicator.
    lag : None | int | str
        Which slice of the graph to certify; forwarded to ``adjacency(lag=...)``.
        ``None`` (default) certifies the lag-OR summary graph.
    as_of : bool, optional
        Force the protocol instead of detecting it from ``fn``'s arity.

    Returns
    -------
    dict
        ``{"causal": bool, "max_revision": float, "n_comparisons": int,
        "protocol": "as_of" | "prefix", "checkpoints": [...]}``

    Examples
    --------
    >>> rep = cafe.discovery.max_edge_revision(lambda Z: cafe.discover(Z), X)  # doctest: +SKIP
    >>> rep["max_revision"]                                                    # doctest: +SKIP
    0.0
    """
    A = np.asarray(X, float)
    if A.ndim == 3:
        ax = 1 if panel_time_axis is None else int(panel_time_axis)
        A_time = np.moveaxis(A, ax, 0)
        def _restore(Z):
            return np.moveaxis(np.asarray(Z, float), 0, ax)
    else:
        A_time = A
        def _restore(Z):
            return np.asarray(Z, float)

    if as_of is None:
        as_of = True
        try:
            sig = inspect.signature(fn)
            pos = [p for p in sig.parameters.values()
                   if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
            as_of = has_var or len(pos) >= 2
        except (TypeError, ValueError):                        # pragma: no cover
            pass

    P_hint = A_time.shape[-1]

    def reporter(Z, t):
        Z = np.asarray(Z, float)
        g = fn(_restore(Z), int(t)) if as_of else fn(_restore(Z[:int(t) + 1]))
        return _to_indicator(g, lag, P_hint)

    T = A_time.shape[0]
    cuts = np.unique(np.linspace(T // 3, T - 1, int(n_prefixes)).astype(int))
    cps = list(cuts if checkpoints is None else
               np.unique(np.asarray(list(checkpoints), int)))
    rep = verify_causal_series(reporter, A_time, n_prefixes=int(n_prefixes),
                               tol=float(tol), checkpoints=cps)
    rep = dict(rep)
    rep["protocol"] = "as_of" if as_of else "prefix"
    rep["checkpoints"] = [int(t) for t in cps if 0 <= int(t) < T]
    return rep


# --------------------------------------------------------------------------- #
# ``res.discover()`` -- attached here so model.py stays untouched
# --------------------------------------------------------------------------- #
def _attach():
    """Give every CAFE result object a ``.discover()``.

    Done here, by assignment, rather than in ``model.py``: the discovery stack sits
    ON TOP of the model and the model must not grow an import of it.  Anything that
    already defines ``discover`` is left alone.
    """
    from .model import CafeResult, PanelResult, PooledPanelResult
    for cls in (CafeResult, PanelResult, PooledPanelResult):
        if getattr(cls, "discover", None) is None:
            cls.discover = _result_discover


_attach()
