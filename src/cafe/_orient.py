"""
CAFE-CD stage 3b -- edge orientation.

Upstream stages reduce a ``P x P x K`` problem to a tiny candidate graph.  This
module turns that candidate graph into oriented edges, and **only where an
identification result actually holds**.

Two ideas carry the whole module:

1. **Lagged edges are free.**  An edge from ``x_i(t-k)`` to ``x_j(t)`` with
   ``k >= 1`` is oriented by the arrow of time.  No search of any kind is run on
   it: :func:`orient` stamps ``'-->'`` / ``identification_method='time'`` and moves
   on.  All the machinery below exists for the *contemporaneous* ambiguous
   component only, which after screening is typically 5-30 nodes.
2. **Route to the cheapest valid identification result, and abstain otherwise.**
   :mod:`cafe._diagnostics` diagnoses the component and picks a branch; this
   module implements the branches.  Where none of them identifies the direction
   the edge is emitted as ``o-o`` or ``<->`` with the reason recorded.  Fabricating
   a DAG would be the actual failure mode -- the benchmark literature is
   consistent that no single method is optimal under all misspecification, so
   honest abstention is the design goal, not a shortcoming.

Branches
--------
``time``          arrow of time; zero computation.
``lingam``        approximate DirectLiNGAM causal order (Shimizu et al.) using the
                  precompute-and-lookup heuristic of arXiv:2409.05500, which drops
                  the iterative data-refinement step so every entropy can be
                  precomputed:  ``O(m^3 + m^2 n)`` instead of ``O(m^3 n)``.
``cumulant``      pairwise third/fourth-order cross-cumulant ratios; one set of
                  sufficient statistics is computed once and reused across pairs.
``pag``           covariance-submatrix rank (FOFC-style vanishing tetrads) says a
                  latent common cause is present: emit ``<->`` / ``o-o``, never an
                  arrow.
``dagma``         local *linear* DAGMA with the log-det acyclicity characterisation
                  (Bello, Aragam & Ravikumar).  Near-Gaussian data identifies a DAG
                  only up to its Markov equivalence class, so only the **compelled**
                  edges of the recovered CPDAG are emitted as arrows; the rest are
                  ``o-o``.
``unidentified``  nonlinear, too few samples, or too large: ``o-o``.

Conventions
-----------
* Coefficient matrices follow the LiNGAM convention ``x = B x + e``, so
  ``B[i, j] != 0`` means **j -> i** ("row is the target").  The same convention is
  used for lagged support matrices.
* Observation masks follow the CAFE convention: **True == observed**.

Implementation constraints (house style)
----------------------------------------
numpy only -- no scipy, no sklearn, no ``lingam``/``causal-learn``, no numba.
Linear systems go through ``np.linalg.solve`` (never ``inv``/``pinv``); the ragged
per-node regressions are padded into a single batched LAPACK call; the ``m^2``
pairwise residual entropies are one chunked broadcast rather than a loop.

References
----------
* Shimizu et al., "DirectLiNGAM: A direct method for learning a linear
  non-Gaussian structural equation model", JMLR 2011.
* "Scalable Time-Series Causal Discovery with Approximate Causal Ordering",
  arXiv:2409.05500 (reference implementation ``Zyang123123/acc_lingam``, MIT).
* Hyvarinen, "New approximations of differential entropy for independent
  component analysis and projection pursuit", NIPS 1998.
* Hyvarinen & Smith, "Pairwise likelihood ratios for estimation of non-Gaussian
  structural equation models", JMLR 2013.
* "Causal Effect Identification in lvLiNGAM from Higher-Order Cumulants",
  ICML 2025.
* Silva, Scheines, Glymour & Spirtes, "Learning the structure of linear latent
  variable models", JMLR 2006 (pure-tetrad / FOFC condition).
* Bello, Aragam & Ravikumar, "DAGMA: Learning DAGs via M-matrices and a
  Log-Determinant Acyclicity Characterization", NeurIPS 2022.
* Reisach, Seiler & Weichwald, "Beware of the Simulated DAG!", NeurIPS 2021
  (varsortability -- why everything below is standardised first).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._diagnostics import (
    DAGMA_MAX_NODES,
    EPS,
    LATENT_PAIR_TOPK,
    MIN_NEFF_ANY,
    NONLIN_STRONG,
    RANK_RHO_FLOOR,
    _corr_from_cov,
    _rank_rtol,
    diagnose,
    quad_purity,
    route,
)

__all__ = [
    "Edge",
    "orient",
    "approx_lingam_order",
    "approx_lingam_adjacency",
    "cumulant_orient",
    "rank_evidence",
    "dagma_local",
    "compelled_edges",
    "topological_order",
    "entropy_maxent",
]

# --------------------------------------------------------------------------- #
# Constants.  Named, justified; nothing tuned to a dataset.
# --------------------------------------------------------------------------- #
# Hyvarinen (1998) maximum-entropy approximation, exactly the constants the
# LiNGAM family uses.  See ``entropy_maxent`` for why the offset does not matter.
_K1 = 79.047
_K2 = 7.4129
# Centring constant of the G1 contrast: E[log cosh z] for z ~ N(0,1), evaluated by
# Gauss-Hermite quadrature to 0.374567207491.  It is NOT the Euler-Mascheroni
# constant -- see ``entropy_maxent`` for why using that instead inverts the sign of
# the negentropy on every sub-Gaussian law.
_G1_GAUSS = 0.37456720749

RHO_CLIP = 0.999          # keeps 1 - rho^2 strictly positive in the residual step
LINGAM_CHUNK_ELEMS = 8_000_000   # memory cap for the (b, P, n) residual broadcast
LINGAM_PRUNE = 0.05       # |standardised coefficient| below this is numerical dust
LINGAM_Z_HALF = 1.0       # half-saturation of the (heuristic) edge confidence
MIN_COMPLETE_FRAC = 0.5   # use complete cases while at least this fraction survive

CUM_CLIP = 10.0           # winsorise standardised data before 6th/8th moments
CUM_Z_HALF = 4.0          # half-saturation of the (heuristic) cumulant confidence
CUM_MIN_CONF = 0.35       # below this the cumulant branch abstains (|t| ~ 2.2)

RANK_MAX_QUADS = 400      # per-pair cap on witness quadruples
RANK_SEED = 20240919
PAG_BIDIRECTED = 0.50     # confounding_prob above this -> '<->' rather than 'o-o'

DAGMA_L1 = 0.02           # L1 on standardised coefficients; the support is already
                          # restricted to the candidate skeleton, so this is light
DAGMA_L1_DELTA = 1e-6     # smoothing of |w| so plain GD has a defined gradient
DAGMA_S_PATH = (1.0, 0.9, 0.8, 0.7, 0.6)
DAGMA_MU_INIT = 1.0
DAGMA_MU_FACTOR = 0.1
DAGMA_INNER = 400
DAGMA_LR0 = 0.05
DAGMA_LR_MAX = 1.0
DAGMA_BACKTRACK = 30
DAGMA_TOL = 1e-9
DAGMA_W_THRESH = 0.30     # the paper's post-hoc edge threshold


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    """One candidate edge and everything known about its direction.

    Attributes
    ----------
    source, target : int
        Variable indices.  For ``'-->'`` and ``'o->'`` the arrowhead is at
        ``target``.  For the symmetric marks ``'o-o'`` and ``'<->'`` the pair is
        canonicalised with ``source < target``.
    lag : int
        0 for a contemporaneous edge, ``k >= 1`` for ``source(t-k) -> target(t)``.
    orientation : str
        ``'-->'``  direction identified.
        ``'o->'``  arrowhead at ``target``, tail unknown (not currently emitted by
                   any branch; reserved for partially-oriented output).
        ``'o-o'``  adjacency supported, direction **not** identified.
        ``'<->'``  latent common cause: neither variable causes the other.
    strength : float
        Absolute standardised structural coefficient when a branch produced one
        (``lingam``, ``dagma``), otherwise the absolute partial correlation of the
        pair given the rest of its component.  Comparable within a branch.
    confounding_prob : float
        In ``[0, 1]``: :func:`rank_evidence` for this pair.  Heuristic, uncalibrated.
    identification_method : str
        ``'time' | 'lingam' | 'cumulant' | 'dagma' | 'pag' | 'unidentified'``.
    confidence : float
        In ``[0, 1]``.  Heuristic, monotone in the branch's own evidence statistic;
        it is **not** a calibrated posterior probability.
    reason : str
        Free text; always populated when the edge is not oriented, so an abstention
        can be audited.
    """

    source: int
    target: int
    lag: int
    orientation: str
    strength: float
    confounding_prob: float
    identification_method: str
    confidence: float
    reason: str = ""


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _as_2d(X, obs=None):
    """Return ``(X2, M2)``: ``(n, P)`` float values and a bool observed mask.

    Accepts ``(T, P)`` or ``(n_ent, T, P)``; a panel is stacked row-wise because
    every contemporaneous contrast in this module is cross-sectional.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim == 3:
        A = A.reshape(-1, A.shape[-1])
    if A.ndim != 2:
        raise ValueError(f"X must be (T,P) or (n_ent,T,P); got {np.shape(X)}")
    finite = np.isfinite(A)
    if obs is None:
        M = finite
    else:
        O = np.asarray(obs)
        if O.ndim == 3:
            O = O.reshape(-1, O.shape[-1])
        if O.shape != A.shape:
            raise ValueError(f"obs shape {O.shape} does not match X shape {A.shape}")
        M = finite & O.astype(bool)
    return np.where(finite, A, 0.0), M


def _prepare(X, obs=None):
    """Standardise to zero mean / unit variance; return ``(Z, complete_case)``.

    Rows that are complete are used when at least :data:`MIN_COMPLETE_FRAC` of them
    survive, because every contrast here (entropies, cumulants, Gram matrices)
    assumes a common sample.  Otherwise the columns are mean-filled, which biases
    the higher-order moments toward Gaussian -- i.e. toward *abstention*, the safe
    direction -- and the caller is told via the returned flag.

    Standardising is not cosmetic: it is the varsortability defence.  Marginal
    variance must not be able to leak a topological order into any branch.
    """
    X2, M2 = _as_2d(X, obs)
    n, P = X2.shape
    complete = M2.all(axis=1)
    n_c = int(complete.sum())
    if n_c >= max(4, int(MIN_COMPLETE_FRAC * n)) and n_c > 4 * P:
        Z = X2[complete]
        cc = True
    else:
        Mf = M2.astype(float)
        cnt = np.maximum(Mf.sum(0), 1.0)
        mu = (X2 * Mf).sum(0) / cnt
        Z = np.where(M2, X2, mu)
        cc = False
    Z = Z - Z.mean(0)
    sd = Z.std(0)
    Z = Z / np.where(sd > EPS, sd, 1.0)
    return np.ascontiguousarray(Z), cc


# --------------------------------------------------------------------------- #
# (a) Approximate DirectLiNGAM
# --------------------------------------------------------------------------- #
def _log_cosh(u):
    """Numerically stable ``log(cosh(u))`` -- ``cosh`` overflows past |u| ~ 710."""
    a = np.abs(u)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def entropy_maxent(U, axis=-1):
    """Hyvarinen's maximum-entropy approximation to differential entropy.

    For a **standardised** variable ``u`` (zero mean, unit variance)::

        H(u) ~ (1 + log 2pi)/2
               - k1 * (E[G1(u)] - E[G1(z)])^2
               - k2 * (E[G2(u)])^2

    with ``G1(u) = log cosh u`` (symmetric non-Gaussianity / tail weight),
    ``G2(u) = u exp(-u^2/2)`` (asymmetry), ``k1 = 79.047``, ``k2 = 7.4129`` and
    ``z ~ N(0,1)``.  This is the approximation the whole LiNGAM family uses.  It is
    preferred here to a kernel or histogram estimator because it is a closed form
    in two sample moments, which is exactly what makes the ``m^2`` pairwise
    residual entropies precomputable in one vectorised pass -- the whole point of
    the approximate ordering.

    **The centring constants are load-bearing, not cosmetic.**  Each term is the
    squared deviation of a contrast from *its Gaussian expectation*; that is what
    makes the whole expression a negentropy, i.e. non-negative and zero exactly at
    the Gaussian, which in turn is what makes ``H(u) <= H(z)`` hold as it must
    (the Gaussian maximises entropy at fixed variance).  ``E[G2(z)] = 0`` by
    symmetry, so the second term is already centred.  The first needs
    ``E[log cosh z] = 0.374567207491`` (:data:`_G1_GAUSS`, Gauss-Hermite
    quadrature); this is the value the published DirectLiNGAM reference uses.

    Substituting the Euler-Mascheroni constant 0.5772 here -- a plausible-looking
    slip, since the symbol ``gamma`` is used for both -- is **not** a harmless
    offset, and this module previously did exactly that.  Two things break:

    1. The penalty ``k1 (g1 - c)^2`` is quadratic in ``g1``, so moving the centre
       moves the *zero* of the negentropy off the Gaussian.  Every law whose
       ``g1`` lies between the two constants is then scored as having **more**
       entropy than a Gaussian of the same variance, which is impossible.  That
       band is the entire sub-Gaussian half of the space: uniform sits at
       ``g1 = 0.401``, a two-point law at ``0.434``, against the Gaussian's
       ``0.3746`` (super-Gaussian laws sit below: Laplace ``0.338``, t5 ``0.343``,
       exponential ``0.330``).
    2. It does not cancel out of the ordering statistic.  Expanding
       ``k1 (g1 - c)^2`` leaves a cross term ``-2 k1 c * g1`` that is *linear* in
       the per-variable ``g1``, so ``diff_mi`` in :func:`approx_lingam_order` --
       a difference of two entropy sums -- retains a dependence on ``c`` unless
       the two sums happen to have equal total ``g1``.  Only the additive
       ``k1 c^2`` cancels.

    Measured consequence (P=8, n=4000, 60 replicates, Kendall tau against the true
    order): with the wrong constant, uniform-noise DAGs came back at
    **tau = -0.93**, i.e. the causal order almost perfectly *reversed*, while
    Laplace/t3/exponential were unaffected.  With the correct constant uniform
    gives ``+0.91`` and the super-Gaussian laws are unchanged to within one
    standard error.  Sub-Gaussian noise is not an exotic corner -- uniform is a
    standard test law in the LiNGAM literature.

    On Gaussian input all four entropies in the contrast coincide and the statistic
    is pure sampling noise, so no order is recovered.  That is the negative control,
    and it passes with either constant -- which is precisely why the bug needed a
    sub-Gaussian fixture to surface.
    """
    U = np.asarray(U, dtype=float)
    g1 = _log_cosh(U).mean(axis=axis)
    g2 = (U * np.exp(-0.5 * U * U)).mean(axis=axis)
    return ((1.0 + np.log(2.0 * np.pi)) / 2.0
            - _K1 * (g1 - _G1_GAUSS) ** 2 - _K2 * g2 ** 2)


def _pairwise_entropy_matrices(Z):
    """Return ``(H_x, H_r)`` -- the precompute step of the approximate ordering.

    ``H_x[i]``     entropy of standardised column ``i``.
    ``H_r[i, j]``  entropy of the standardised residual of the simple regression of
                   ``i`` on ``j``.

    Every one of the ``P^2`` simple regressions is a single correlation-matrix
    expression: on standardised data the residual is
    ``r_{i|j} = (z_i - rho_ij z_j) / sqrt(1 - rho_ij^2)``.  The whole tensor is
    formed by one broadcast, chunked over ``i`` so the ``(b, P, n)`` intermediate
    stays under :data:`LINGAM_CHUNK_ELEMS` elements.
    """
    n, P = Z.shape
    Rho = (Z.T @ Z) / n
    np.fill_diagonal(Rho, 1.0)
    Rho = np.clip(Rho, -RHO_CLIP, RHO_CLIP)
    H_x = entropy_maxent(Z.T, axis=1)
    H_r = np.zeros((P, P))
    scale = np.sqrt(np.maximum(1.0 - Rho * Rho, EPS))
    Zt = Z.T                                              # (P, n)
    b = max(1, int(LINGAM_CHUNK_ELEMS // max(P * n, 1)))
    for lo in range(0, P, b):
        hi = min(lo + b, P)
        U = (Zt[lo:hi, None, :] - Rho[lo:hi, :, None] * Zt[None, :, :])
        U /= scale[lo:hi, :, None]
        H_r[lo:hi] = entropy_maxent(U, axis=2)
    np.fill_diagonal(H_r, 0.0)
    return H_x, H_r


def approx_lingam_order(X, obs=None, return_diff=False):
    """Approximate DirectLiNGAM causal order.  ``O(m^3 + m^2 n)``.

    Standard DirectLiNGAM re-estimates every entropy from scratch inside each of
    the ``m`` calls to ``_search_causal_order``, on data that has been residualised
    against the variables already chosen -- that iterative refinement is what makes
    it ``O(m^3 n)``.  The heuristic of arXiv:2409.05500 drops the refinement step,
    which makes the pairwise statistic independent of the ordering state and
    therefore precomputable exactly once::

        diff_mi(i, j) = ( H(x_j) + H(r_{i|j}) ) - ( H(x_i) + H(r_{j|i}) )
        M_i           = - sum_j [ min(0, diff_mi(i, j)) ]^2
        next          = argmax_{i in U} M_i

    ``diff_mi(i, j) > 0`` is evidence that ``i`` is upstream of ``j``, so the
    penalty ``sum_j min(0, .)^2`` counts the evidence *against* ``i`` being the next
    exogenous variable and the selected variable is the one with the least such
    evidence.  (The negation in ``M_i`` is the sign convention of the DirectLiNGAM
    reference implementation; without it the recovered order comes out reversed.)

    After the precompute, each of the ``m`` rounds is one masked row-sum over the
    ``P x P`` penalty matrix -- ``O(m^2)`` per round, ``O(m^3)`` in total, with no
    further passes over the ``n`` samples.

    Parameters
    ----------
    X : array
        ``(T, P)`` or ``(n_ent, T, P)``.  Standardised internally.
    obs : array of bool, optional
        True == observed.
    return_diff : bool
        Also return the ``P x P`` ``diff_mi`` matrix (used for per-edge confidence).

    Returns
    -------
    order : np.ndarray
        Variable indices, most exogenous first.
    diff : np.ndarray, optional
        The ``diff_mi`` matrix, when ``return_diff`` is set.
    """
    Z, _ = _prepare(X, obs)
    n, P = Z.shape
    if P == 0:
        out = np.zeros(0, dtype=np.intp)
        return (out, np.zeros((0, 0))) if return_diff else out
    if P == 1:
        out = np.zeros(1, dtype=np.intp)
        return (out, np.zeros((1, 1))) if return_diff else out

    H_x, H_r = _pairwise_entropy_matrices(Z)
    diff = (H_x[None, :] + H_r) - (H_x[:, None] + H_r.T)
    np.fill_diagonal(diff, 0.0)
    pen = np.minimum(diff, 0.0) ** 2                       # (P, P)

    order = np.empty(P, dtype=np.intp)
    remaining = np.ones(P, dtype=bool)
    for k in range(P):
        idx = np.flatnonzero(remaining)
        M = -pen[np.ix_(idx, idx)].sum(axis=1)
        pick = idx[int(np.argmax(M))]
        order[k] = pick
        remaining[pick] = False
    return (order, diff) if return_diff else order


def _padded_batched_ols(S, allowed):
    """Solve ``P`` OLS problems with ragged supports as one batched LAPACK call.

    ``S`` is the ``(P, P)`` Gram/covariance of standardised data and
    ``allowed[i, j]`` says variable ``j`` may be a regressor for target ``i``.  The
    ragged supports are padded to the widest one; padded slots get an identity row
    and a zero right-hand side, so their coefficients come out exactly 0 and never
    perturb the real ones.  Returns ``B`` with ``B[i, j]`` the coefficient of ``j``
    in the regression of ``i``.
    """
    P = S.shape[0]
    B = np.zeros((P, P))
    sizes = allowed.sum(axis=1)
    s_max = int(sizes.max()) if P else 0
    if s_max == 0:
        return B
    idx = np.zeros((P, s_max), dtype=np.intp)
    valid = np.zeros((P, s_max), dtype=bool)
    for i in range(P):
        pa = np.flatnonzero(allowed[i])
        idx[i, : pa.size] = pa
        valid[i, : pa.size] = True
    G = S[idx[:, :, None], idx[:, None, :]]                # (P, s, s)
    pairv = valid[:, :, None] & valid[:, None, :]
    G = np.where(pairv, G, 0.0)
    eye = np.eye(s_max, dtype=bool)[None]
    G = np.where(~pairv & eye, 1.0, G)                     # identity on pad diagonal
    rhs = np.where(valid, S[np.arange(P)[:, None], idx], 0.0)
    ridge = 1e-8 * max(float(np.trace(S)) / max(P, 1), EPS)
    G = G + ridge * np.eye(s_max)[None]
    coef = np.linalg.solve(G, rhs[..., None])[..., 0]      # (P, s)
    np.add.at(B, (np.repeat(np.arange(P), s_max), idx.ravel()),
              np.where(valid, coef, 0.0).ravel())
    return B


def approx_lingam_adjacency(X, order, candidates=None, obs=None, prune=LINGAM_PRUNE):
    """Coefficients of each variable on its predecessors in ``order``.

    Restricted to ``candidates`` (the skeleton), so the result is strictly
    triangular in ``order`` by construction and cannot contain a cycle.

    Parameters
    ----------
    X : array
        ``(T, P)`` or ``(n_ent, T, P)``.
    order : sequence of int
        Causal order, most exogenous first (e.g. from :func:`approx_lingam_order`).
    candidates : array, optional
        ``(P, P)`` boolean; ``candidates[i, j]`` allows ``j -> i``.  Symmetrised, so
        an undirected skeleton works directly.  ``None`` allows every predecessor.
    prune : float
        Absolute standardised-coefficient floor; below it the entry is set to 0.

    Returns
    -------
    np.ndarray
        ``(P, P)`` matrix ``B`` in the ``x = B x + e`` convention: ``B[i, j]`` is the
        effect of ``j`` on ``i``.
    """
    Z, _ = _prepare(X, obs)
    n, P = Z.shape
    order = np.asarray(order, dtype=np.intp)
    if order.size != P:
        raise ValueError(f"order has {order.size} entries but X has {P} columns")
    pos = np.empty(P, dtype=np.intp)
    pos[order] = np.arange(P)
    allowed = pos[None, :] < pos[:, None]                  # allowed[i, j]: j before i
    if candidates is not None:
        Cand = np.asarray(candidates) != 0
        Cand = Cand | Cand.T
        allowed &= Cand
    np.fill_diagonal(allowed, False)
    S = (Z.T @ Z) / n
    B = _padded_batched_ols(S, allowed)
    B[np.abs(B) < prune] = 0.0
    B[~allowed] = 0.0
    return B


# --------------------------------------------------------------------------- #
# (b) Cumulant orientation
# --------------------------------------------------------------------------- #
def _cumulant_stats(Z):
    """All reusable cross-cumulant sufficient statistics, in ~10 GEMMs.

    Every pair-level quantity the orientation rule needs -- the third- and
    fourth-order cross moments *and* their sampling variances -- is a single
    ``Z^k.T @ Z^m / n`` product, so the cost is paid once for the whole component
    rather than per pair.  That reuse is the entire point of using cumulants here.

    Standardised data is winsorised at :data:`CUM_CLIP` SD first: the variance of a
    third moment needs a sixth moment and the variance of a fourth moment an eighth,
    and a single extreme draw otherwise dominates them.  It biases the tails
    slightly *toward* Gaussian, i.e. toward abstention.
    """
    Z = np.clip(Z, -CUM_CLIP, CUM_CLIP)
    n = Z.shape[0]
    Z2 = Z * Z
    Z3 = Z2 * Z
    Z4 = Z2 * Z2
    Z6 = Z3 * Z3
    st = {
        "rho": (Z.T @ Z) / n,
        "m21": (Z2.T @ Z) / n,        # m21[i, j] = E[z_i^2 z_j]
        "m31": (Z3.T @ Z) / n,        # m31[i, j] = E[z_i^3 z_j]
        "m42": (Z4.T @ Z2) / n,       # E[z_i^4 z_j^2]
        "m33": (Z3.T @ Z3) / n,       # E[z_i^3 z_j^3]
        "m62": (Z6.T @ Z2) / n,       # E[z_i^6 z_j^2]
        "m44": (Z4.T @ Z4) / n,       # E[z_i^4 z_j^4]
        "n": n,
    }
    np.fill_diagonal(st["rho"], 1.0)
    return st


def _order_contrast(a, b, rho, var_a, var_b, cov_ab, power, n):
    """Shared algebra for the third- and fourth-order direction contrasts.

    With ``x = z_i`` and ``y = z_j`` standardised and ``y = rho x + e`` with
    ``e`` independent of ``x`` (i.e. ``i -> j``), multilinearity of cumulants gives

        third order:   ``cum(x,y,y)   = rho   * cum(x,x,y)``
        fourth order:  ``cum(x,y,y,y) = rho^2 * cum(x,x,x,y)``

    exactly, and in the reverse model the same identity holds with the two sides
    swapped.  So the direction is decided by which of the two residuals

        ``d = b - rho^power * a``   (zero if ``i -> j``)
        ``e = a - rho^power * b``   (zero if ``j -> i``)

    is closer to zero.  Both are compared *in their own standard errors*, so a
    contrast built from a tiny cumulant cannot outvote a well-determined one:

        ``t = |e| / se(e) - |d| / se(d)``,  positive means ``i -> j``.

    Under Gaussianity every cumulant above second order vanishes, both residuals
    are pure noise, and ``t`` is symmetric about 0 -- which is exactly the
    at-chance behaviour the negative control requires.  ``rho`` is treated as known
    when propagating the variances; its own ``O(1/sqrt(n))`` error is second order.
    """
    r = rho ** power
    d = b - r * a
    e = a - r * b
    var_d = np.maximum(var_b + r * r * var_a - 2.0 * r * cov_ab, 0.0)
    var_e = np.maximum(var_a + r * r * var_b - 2.0 * r * cov_ab, 0.0)
    se_d = np.sqrt(var_d / n) + EPS
    se_e = np.sqrt(var_e / n) + EPS
    t = np.abs(e) / se_e - np.abs(d) / se_d
    score = (np.abs(e) - np.abs(d)) / (np.abs(e) + np.abs(d) + EPS)
    return t, score


def cumulant_orient(X, pairs, obs=None, stats=None):
    """Orient candidate pairs from third/fourth-order cross-cumulants.

    Covariance is symmetric in ``(i, j)`` and can never break the direction.  The
    cross-cumulants can: for a linear non-Gaussian pair the identity
    ``cum(x,y,y) = rho * cum(x,x,y)`` holds in the causal direction and fails in the
    reverse one, and the same at fourth order with ``rho^2``.  See
    :func:`_order_contrast` for the algebra and the test statistic.

    The sufficient statistics (:func:`_cumulant_stats`) are computed **once** for the
    whole variable set and reused across every pair -- adding pairs is free.

    Whichever order carries the stronger evidence (larger ``|t|``) decides; the
    third order breaks ties, since asymmetry is the signal the router sends here.

    Parameters
    ----------
    X : array
        ``(T, P)`` or ``(n_ent, T, P)``.
    pairs : sequence of (int, int)
        Candidate undirected pairs.
    stats : dict, optional
        Precomputed :func:`_cumulant_stats` output, to reuse across calls.

    Returns
    -------
    scores : np.ndarray
        ``(len(pairs),)`` signed, in ``[-1, 1]``.  **Positive means ``i -> j``**
        for ``pairs[k] == (i, j)``; negative means ``j -> i``; ~0 means the
        cumulants carry no asymmetry (e.g. Gaussian).
    confidences : np.ndarray
        ``(len(pairs),)`` in ``[0, 1)``, ``|t| / (|t| + CUM_Z_HALF)``.  Monotone in
        the evidence but **not** a calibrated probability.
    """
    pairs = np.asarray(list(pairs), dtype=np.intp).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    if stats is None:
        Z, _ = _prepare(X, obs)
        stats = _cumulant_stats(Z)
    i, j = pairs[:, 0], pairs[:, 1]
    n = stats["n"]
    rho = stats["rho"][i, j]

    # ---- third order: a = E[x^2 y] = cum(x,x,y), b = E[x y^2] = cum(x,y,y) ----
    a3 = stats["m21"][i, j]
    b3 = stats["m21"][j, i]
    va3 = np.maximum(stats["m42"][i, j] - a3 * a3, 0.0)
    vb3 = np.maximum(stats["m42"][j, i] - b3 * b3, 0.0)
    cab3 = stats["m33"][i, j] - a3 * b3
    t3, s3 = _order_contrast(a3, b3, rho, va3, vb3, cab3, 1.0, n)

    # ---- fourth order: cum(x,x,x,y) = E[x^3 y] - 3 rho on standardised data ----
    a4 = stats["m31"][i, j] - 3.0 * rho
    b4 = stats["m31"][j, i] - 3.0 * rho
    va4 = np.maximum(stats["m62"][i, j] - stats["m31"][i, j] ** 2, 0.0)
    vb4 = np.maximum(stats["m62"][j, i] - stats["m31"][j, i] ** 2, 0.0)
    cab4 = stats["m44"][i, j] - stats["m31"][i, j] * stats["m31"][j, i]
    t4, s4 = _order_contrast(a4, b4, rho, va4, vb4, cab4, 2.0, n)

    use3 = np.abs(t3) >= np.abs(t4)
    t = np.where(use3, t3, t4)
    score = np.where(use3, s3, s4)
    score = np.sign(t) * np.abs(score)      # sign always from the t statistic
    conf = np.abs(t) / (np.abs(t) + CUM_Z_HALF)
    return np.clip(score, -1.0, 1.0), np.clip(conf, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# (c) Rank-based latent-confounding test
# --------------------------------------------------------------------------- #
def rank_evidence(S, i, j, cond, rtol=None, n_eff=None, max_quads=RANK_MAX_QUADS):
    """Evidence in ``[0, 1]`` that ``i`` and ``j`` share a latent common cause.

    A latent common cause of four observed variables makes the ``2 x 2``
    cross-blocks of their covariance rank 1 for **all three** ways of splitting the
    four into two pairs -- the classical "pure tetrad" condition.  A sparse DAG with
    generic parameters gives full-rank blocks.  The three-way simultaneity matters:
    a single vanishing tetrad is also produced by an observed mediator (in the chain
    ``a -> b -> c -> d`` two of the three tetrads vanish), so a one-tetrad test
    cannot tell a latent from a chain and this one can.

    Concretely: for every pair ``(k, l)`` drawn from ``cond`` the quadruple
    ``{i, j, k, l}`` is scored by
    :func:`cafe._diagnostics.quad_purity`, and the returned score is the mean of the
    best :data:`cafe._diagnostics.LATENT_PAIR_TOPK` witnesses -- "how strong is the
    best handful of rank-1 witnesses for this pair".

    **Honesty about the tolerance.**  The rank decision is a *relative*
    singular-value tolerance (``RANK_RTOL_Z / sqrt(n_eff)``, clipped), not a
    calibrated hypothesis test.  There is no null distribution behind it and no
    error rate is controlled.  It is deliberately biased toward firing, because its
    only consequence is abstention (``<->`` / ``o-o``) whereas a missed confounder
    would produce a fabricated arrow.  Two structural limits are worth stating:
    fewer than two conditioning variables makes the test impossible and it returns
    0.0, and a latent confounding *exactly* two observed variables is
    observationally indistinguishable from a direct edge at second order -- no
    covariance-based test can detect it.

    Parameters
    ----------
    S : array
        ``(P, P)`` covariance (or correlation) matrix.  Converted to correlation.
    i, j : int
        The candidate pair.
    cond : sequence of int
        Other observed variables to use as witnesses -- typically the rest of the
        component, or the skeleton neighbourhood of ``i`` and ``j``.
    rtol : float, optional
        Explicit relative singular-value tolerance; otherwise derived from ``n_eff``.
    n_eff : float, optional
        Effective sample size, used to pick ``rtol``.
    max_quads : int
        Cap on witness quadruples (deterministically subsampled above it).

    Returns
    -------
    float
    """
    S = np.asarray(S, dtype=float)
    C = _corr_from_cov(S) if not np.allclose(np.diag(S), 1.0, atol=1e-6) else S
    cond = [int(c) for c in cond if int(c) != int(i) and int(c) != int(j)]
    cond = sorted(set(cond))
    if len(cond) < 2:
        return 0.0
    ci = np.asarray(cond, dtype=np.intp)
    ka, kb = np.triu_indices(ci.size, 1)
    if ka.size > max_quads:
        sel = np.random.default_rng(RANK_SEED).choice(ka.size, size=max_quads,
                                                      replace=False)
        sel.sort()
        ka, kb = ka[sel], kb[sel]
    quads = np.empty((ka.size, 4), dtype=np.intp)
    quads[:, 0] = int(i)
    quads[:, 1] = int(j)
    quads[:, 2] = ci[ka]
    quads[:, 3] = ci[kb]
    pur = quad_purity(C, quads, _rank_rtol(n_eff, rtol), RANK_RHO_FLOOR)
    if pur.size == 0:
        return 0.0
    k = int(min(LATENT_PAIR_TOPK, pur.size))
    top = np.sort(pur)[-k:]
    return float(np.clip(top.mean(), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# (d) Local linear DAGMA
# --------------------------------------------------------------------------- #
def _dagma_h(W, s):
    """Log-det acyclicity value and gradient; ``None`` when ``sI - W*W`` is invalid.

    ``h_s(W) = -logdet(sI - W o W) + P log s`` where ``o`` is the elementwise square.
    It is finite exactly on the set where ``sI - W o W`` is a nonsingular M-matrix,
    and zero iff ``W o W`` is nilpotent, i.e. iff ``W`` is a DAG (Bello et al.).  The
    gradient is ``2 (sI - W o W)^{-T} o W``.

    The one place an explicit inverse is needed is that gradient, and it is taken
    with ``np.linalg.solve`` against the identity -- one LAPACK LU with ``P``
    right-hand sides -- never ``np.linalg.inv``/``pinv``.
    """
    P = W.shape[0]
    M = s * np.eye(P) - W * W
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0 or not np.isfinite(logdet):
        return None, None
    try:
        Minv = np.linalg.solve(M, np.eye(P))
    except np.linalg.LinAlgError:
        return None, None
    return -logdet + P * np.log(s), 2.0 * Minv.T * W


def dagma_local(X, candidates=None, max_nodes=DAGMA_MAX_NODES, obs=None,
                lambda1=DAGMA_L1, w_threshold=DAGMA_W_THRESH,
                s_path=DAGMA_S_PATH, n_inner=DAGMA_INNER):
    """Linear DAGMA on a small candidate support.  Deliberately local.

    Minimises, along a decreasing central path in ``mu``::

        F_mu(W) = mu * ( 1/2n ||X - X W||_F^2 + lambda1 ||W||_1 ) + h_s(W)
        h_s(W)  = -logdet(sI - W o W) + P log s

    The least-squares term needs only the ``P x P`` Gram: with ``G = X'X/n`` the
    score is ``1/2 tr((I-W)' G (I-W))`` and its gradient is ``G W - G``, so no inner
    pass over the ``n`` samples is ever repeated.  Optimisation is plain gradient
    descent with a backtracking step that rejects any move leaving the region where
    ``h_s`` is defined (checked by the sign of ``np.linalg.slogdet``), warm-started
    across the path -- no Adam, no scipy.

    ``X`` is standardised first.  That is deliberate: on raw simulated data the
    marginal variances alone often reveal a topological order (varsortability),
    and a least-squares score will happily exploit it.  Standardising removes that
    leak.  It also removes any equal-error-variance identifiability, so on
    near-Gaussian data the returned DAG is one member of a Markov equivalence
    class -- which is why :func:`orient` emits only its *compelled* edges as arrows.

    **Refuses above ``max_nodes``** by raising ``ValueError``.  This is a local tool
    for the ambiguous component left over after screening, not a global solver.

    Parameters
    ----------
    X : array
        ``(T, P)`` or ``(n_ent, T, P)`` with ``P <= max_nodes``.
    candidates : array, optional
        ``(P, P)`` boolean support; symmetrised, diagonal cleared.  ``None`` allows
        every off-diagonal entry.
    w_threshold : float
        Post-hoc absolute-coefficient threshold, as in the paper.

    Returns
    -------
    np.ndarray
        ``(P, P)`` matrix ``B`` in the ``x = B x + e`` convention (``B[i, j]``: the
        effect of ``j`` on ``i``), guaranteed acyclic.

    Raises
    ------
    ValueError
        If ``P > max_nodes``.
    """
    Z, _ = _prepare(X, obs)
    n, P = Z.shape
    if P > max_nodes:
        raise ValueError(
            f"dagma_local refuses {P} nodes (max_nodes={max_nodes}). It is a local "
            "tool for the ambiguous component, not a global solver; route the "
            "component to 'unidentified' instead."
        )
    if P < 2:
        return np.zeros((P, P))
    if candidates is None:
        mask = ~np.eye(P, dtype=bool)
    else:
        Cand = np.asarray(candidates) != 0
        mask = Cand | Cand.T
        np.fill_diagonal(mask, False)
    if not mask.any():
        return np.zeros((P, P))
    maskf = mask.astype(float)

    G = (Z.T @ Z) / n

    def _score_grad(W):
        D = np.eye(P) - W
        sc = 0.5 * float(np.sum(D * (G @ D)))
        gr = G @ W - G
        l1 = float(np.sum(np.sqrt(W * W + DAGMA_L1_DELTA ** 2)) - P * DAGMA_L1_DELTA)
        gl1 = W / np.sqrt(W * W + DAGMA_L1_DELTA ** 2)
        return sc + lambda1 * l1, gr + lambda1 * gl1

    W = np.zeros((P, P))
    for k, s in enumerate(s_path):
        mu = DAGMA_MU_INIT * (DAGMA_MU_FACTOR ** k)
        h, gh = _dagma_h(W, s)
        if h is None:                      # warm start invalid for this s: restart
            W = np.zeros((P, P))
            h, gh = _dagma_h(W, s)
        f, gf = _score_grad(W)
        obj = mu * f + h
        step = DAGMA_LR0
        for _ in range(n_inner):
            grad = (mu * gf + gh) * maskf
            gn = float(np.max(np.abs(grad)))
            if gn < DAGMA_TOL:
                break
            improved = False
            for _bt in range(DAGMA_BACKTRACK):
                Wn = (W - step * grad) * maskf
                hn, ghn = _dagma_h(Wn, s)
                if hn is not None:
                    fn, gfn = _score_grad(Wn)
                    objn = mu * fn + hn
                    if objn <= obj - DAGMA_TOL * (1.0 + abs(obj)):
                        W, h, gh, f, gf, obj = Wn, hn, ghn, fn, gfn, objn
                        step = min(step * 1.2, DAGMA_LR_MAX)
                        improved = True
                        break
                step *= 0.5
            if not improved:
                break

    B = W.T.copy()                          # DAGMA's W[j,i] (j->i) -> B[i,j]
    B[np.abs(B) < w_threshold] = 0.0
    B[~mask.T] = 0.0
    return _force_acyclic(B)


def topological_order(A):
    """Kahn topological order of ``A[i, j] != 0`` meaning ``j -> i``; ``None`` if cyclic."""
    A = np.asarray(A)
    adj = (A != 0)
    P = adj.shape[0]
    indeg = adj.sum(axis=1).astype(int)      # parents of each node
    alive = np.ones(P, dtype=bool)
    order = []
    for _ in range(P):
        ready = np.flatnonzero(alive & (indeg == 0))
        if ready.size == 0:
            return None
        v = int(ready[0])
        order.append(v)
        alive[v] = False
        indeg[v] = -1
        indeg[adj[:, v] & alive] -= 1
    return np.asarray(order, dtype=np.intp)


def _force_acyclic(B):
    """Drop the weakest edges until ``B`` is acyclic.  No-op when already acyclic."""
    B = np.array(B, dtype=float, copy=True)
    while topological_order(B) is None:
        nz = np.flatnonzero(B.ravel() != 0)
        if nz.size == 0:
            break
        weakest = nz[int(np.argmin(np.abs(B.ravel()[nz])))]
        B.ravel()[weakest] = 0.0
    return B


# --------------------------------------------------------------------------- #
# CPDAG: which arrows of a DAG are actually identified?
# --------------------------------------------------------------------------- #
def compelled_edges(B):
    """Boolean mask of the arrows of DAG ``B`` that every Markov-equivalent DAG shares.

    A linear-Gaussian DAG is identified only up to its Markov equivalence class:
    the skeleton and the v-structures are determined by the distribution, and
    Meek's rules R1-R3 propagate those to every further orientation that is forced.
    Everything left undirected can be reversed by some equivalent DAG, so claiming
    a direction for it would be fabrication.

    Parameters
    ----------
    B : array
        ``(P, P)``; ``B[i, j] != 0`` means ``j -> i``.  Must be acyclic.

    Returns
    -------
    np.ndarray
        ``(P, P)`` boolean, ``out[i, j]`` == "``j -> i`` is compelled".
    """
    A = np.asarray(B) != 0
    P = A.shape[0]
    skel = A | A.T
    arrow = np.zeros((P, P), dtype=bool)     # arrow[i, j]: j -> i is oriented

    # v-structures: j -> i <- k with j, k non-adjacent
    for i in range(P):
        pa = np.flatnonzero(A[i])
        for u in range(pa.size):
            for v in range(u + 1, pa.size):
                j, k = int(pa[u]), int(pa[v])
                if not skel[j, k]:
                    arrow[i, j] = True
                    arrow[i, k] = True

    def _undirected(a, b):
        return skel[a, b] and not arrow[a, b] and not arrow[b, a]

    changed = True
    while changed:
        changed = False
        for a in range(P):
            for b in range(P):
                if not skel[a, b] or arrow[b, a] or arrow[a, b]:
                    continue
                # R1: c -> a  and  a - b  with c, b non-adjacent  =>  a -> b
                for c in np.flatnonzero(arrow[a]):
                    if c != b and not skel[int(c), b]:
                        arrow[b, a] = True
                        changed = True
                        break
                if arrow[b, a]:
                    continue
                # R2: a -> c -> b  and  a - b  =>  a -> b
                for c in range(P):
                    if c != a and c != b and arrow[c, a] and arrow[b, c]:
                        arrow[b, a] = True
                        changed = True
                        break
                if arrow[b, a]:
                    continue
                # R3: a - c, a - d, c -> b, d -> b, c and d non-adjacent => a -> b
                cands = [c for c in range(P)
                         if c != a and c != b and arrow[b, c] and _undirected(a, c)]
                done = False
                for u in range(len(cands)):
                    for v in range(u + 1, len(cands)):
                        if not skel[cands[u], cands[v]]:
                            arrow[b, a] = True
                            changed = True
                            done = True
                            break
                    if done:
                        break
    return arrow & A


# --------------------------------------------------------------------------- #
# Graph plumbing
# --------------------------------------------------------------------------- #
def _symmetric_adjacency(spec, P):
    """Normalise a candidate spec into a symmetric ``(P, P)`` boolean adjacency."""
    A = np.zeros((P, P), dtype=bool)
    if spec is None:
        return A
    arr = np.asarray(spec)
    if arr.ndim == 2 and arr.shape == (P, P):
        A = (arr != 0)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        for a, b in arr.astype(int):
            A[int(a), int(b)] = True
    elif arr.size == 0:
        return A
    else:
        raise ValueError(f"contemp_candidates must be (P,P) or a list of pairs; got {arr.shape}")
    A = A | A.T
    np.fill_diagonal(A, False)
    return A


def _components(A):
    """Connected components of a symmetric boolean adjacency, as index arrays."""
    P = A.shape[0]
    seen = np.zeros(P, dtype=bool)
    out = []
    for v in range(P):
        if seen[v]:
            continue
        stack = [v]
        seen[v] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in np.flatnonzero(A[u] & ~seen):
                seen[w] = True
                stack.append(int(w))
        out.append(np.asarray(sorted(comp), dtype=np.intp))
    return out


def _parse_lagged(spec, P):
    """Normalise a lagged-support spec into ``(source, target, lag, weight)`` tuples.

    Accepted forms (all with the ``[target, source]`` row-is-target convention):

    * ``None`` / empty
    * ``(P, P, K)`` array -- last axis is lag ``1..K``
    * ``(K, P, P)`` array -- first axis is lag ``1..K`` (used when the *last* two
      axes are the square ones)
    * ``dict`` ``{lag: (P, P) array}``
    * iterable of ``(source, target, lag)`` or ``(source, target, lag, weight)``

    A float array supplies the edge weight directly, so no computation whatsoever is
    needed for a lagged edge; a boolean array leaves the weight at ``nan`` and
    :func:`orient` fills it with a single cross-covariance pass (never a search).
    """
    out = []
    if spec is None:
        return out
    if isinstance(spec, dict):
        for lag, mat in spec.items():
            m = np.asarray(mat)
            tgt, src = np.nonzero(m != 0)
            for t, s in zip(tgt, src):
                w = float(m[t, s]) if m.dtype.kind == "f" else np.nan
                out.append((int(s), int(t), int(lag), w))
        return out
    arr = np.asarray(spec)
    if arr.ndim == 3:
        if arr.shape[0] == arr.shape[1] == P:
            for k in range(arr.shape[2]):
                tgt, src = np.nonzero(arr[:, :, k] != 0)
                for t, s in zip(tgt, src):
                    w = float(arr[t, s, k]) if arr.dtype.kind == "f" else np.nan
                    out.append((int(s), int(t), k + 1, w))
        elif arr.shape[1] == arr.shape[2] == P:
            for k in range(arr.shape[0]):
                tgt, src = np.nonzero(arr[k] != 0)
                for t, s in zip(tgt, src):
                    w = float(arr[k, t, s]) if arr.dtype.kind == "f" else np.nan
                    out.append((int(s), int(t), k + 1, w))
        else:
            raise ValueError(f"lagged_support 3-D array has ambiguous shape {arr.shape}")
        return out
    if arr.ndim == 2 and arr.shape[1] in (3, 4):
        for row in arr:
            s, t, lag = int(row[0]), int(row[1]), int(row[2])
            w = float(row[3]) if arr.shape[1] == 4 else np.nan
            out.append((s, t, lag, w))
        return out
    if arr.size == 0:
        return out
    raise ValueError(f"unrecognised lagged_support of shape {arr.shape}")


def _lagged_strengths(X3, M3, triples):
    """One cross-covariance pass for the lagged edge weights that were not supplied.

    The *orientation* of a lagged edge costs nothing -- it is the arrow of time.
    This fills the reported ``strength`` field only, and is a single pass per
    distinct lag, never a search.  Skipped entirely when every weight was supplied.
    """
    need = [t for t in triples if not np.isfinite(t[3])]
    if not need:
        return {}
    G, T, P = X3.shape
    mu = np.array([X3[..., p][M3[..., p]].mean() if M3[..., p].any() else 0.0
                   for p in range(P)])
    sd = np.array([X3[..., p][M3[..., p]].std() if M3[..., p].any() else 1.0
                   for p in range(P)])
    sd = np.where(sd > EPS, sd, 1.0)
    Z = np.where(M3, (X3 - mu) / sd, 0.0)
    Mf = M3.astype(float)
    out = {}
    for lag in sorted({t[2] for t in need}):
        if lag >= T:
            continue
        A = Z[:, lag:, :].reshape(-1, P)
        Bm = Z[:, :T - lag, :].reshape(-1, P)
        Wa = Mf[:, lag:, :].reshape(-1, P)
        Wb = Mf[:, :T - lag, :].reshape(-1, P)
        num = Bm.T @ A                       # num[s, t] = sum_t z_s(t-lag) z_t(t)
        den = Wb.T @ Wa                      # co-observed counts
        out[lag] = np.abs(num) / np.maximum(den, 1.0)   # mean standardised cross-product
    return out


def _partial_corr(C):
    """Partial correlations from a correlation matrix (used as a generic strength)."""
    P = C.shape[0]
    ridge = 1e-6
    try:
        Theta = np.linalg.solve(C + ridge * np.eye(P), np.eye(P))
    except np.linalg.LinAlgError:
        return np.abs(C)
    d = np.sqrt(np.maximum(np.diag(Theta), EPS))
    R = -Theta / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# (e) Public entry point
# --------------------------------------------------------------------------- #
def orient(E, contemp_candidates=None, lagged_support=None, diag=None, obs=None):
    """Orient a candidate graph, per connected component, with honest abstention.

    Lagged edges are stamped ``'-->'`` / ``'time'`` with no search.  The
    contemporaneous candidate graph is split into connected components and each one
    is diagnosed and routed **independently**, so a small non-Gaussian component and
    a large near-Gaussian one in the same call get different treatment.

    Parameters
    ----------
    E : array
        Residual matrix, ``(T, P)`` or ``(n_ent, T, P)``.
    contemp_candidates : array, optional
        ``(P, P)`` boolean adjacency (symmetrised) or an ``(m, 2)`` list of pairs.
    lagged_support : optional
        See :func:`_parse_lagged`.  Float arrays supply the strength directly.
    diag : Diagnostics, optional
        Precomputed diagnostics.  Reused only when a component spans **all** ``P``
        variables; otherwise diagnostics are recomputed on the component's own
        columns, because routing the whole graph on global statistics is exactly
        what this design is trying to avoid.
    obs : array of bool, optional
        True == observed.

    Returns
    -------
    list of Edge
        One entry per candidate edge -- lagged edges first, then one per
        contemporaneous candidate pair, oriented or explicitly abstained on.
    """
    X = np.asarray(E, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X3 = X[None] if X.ndim == 2 else X
    finite = np.isfinite(X3)
    if obs is None:
        M3 = finite
    else:
        O = np.asarray(obs)
        O3 = O[None] if O.ndim == 2 else O
        M3 = finite & O3.astype(bool)
    X3 = np.where(finite, X3, 0.0)
    P = X3.shape[2]

    edges = []

    # ---------------- lagged: free ---------------- #
    triples = _parse_lagged(lagged_support, P)
    strengths = _lagged_strengths(X3, M3, triples)
    for s, t, lag, w in triples:
        if not np.isfinite(w):
            w = float(strengths.get(lag, np.zeros((P, P)))[s, t]) if lag in strengths else 0.0
        edges.append(Edge(source=int(s), target=int(t), lag=int(lag), orientation="-->",
                          strength=float(abs(w)), confounding_prob=0.0,
                          identification_method="time", confidence=1.0,
                          reason="oriented by the arrow of time; no search performed"))

    # ---------------- contemporaneous: routed per component ---------------- #
    A = _symmetric_adjacency(contemp_candidates, P)
    for nodes in _components(A):
        if nodes.size < 2:
            continue
        sub = A[np.ix_(nodes, nodes)]
        if not sub.any():
            continue
        edges.extend(_orient_component(X3, M3, nodes, sub, diag, P))
    return edges


def _orient_component(X3, M3, nodes, sub, diag_global, P_total):
    """Diagnose, route and orient one connected contemporaneous component."""
    m = nodes.size
    Xc = X3[:, :, nodes]
    Mc = M3[:, :, nodes]
    if diag_global is not None and m == P_total:
        dg = diag_global
    else:
        dg = diagnose(Xc, Mc, skeleton=sub)
    branch = route(dg, m)

    Z, _ = _prepare(Xc, Mc)
    n = Z.shape[0]
    S = (Z.T @ Z) / max(n, 1)
    C = _corr_from_cov(S)
    pc = _partial_corr(C)

    pu, pv = np.triu_indices(m, 1)
    keep = sub[pu, pv]
    pairs = list(zip(pu[keep].tolist(), pv[keep].tolist()))
    if not pairs:
        return []

    conf_prob = {}
    for (a, b) in pairs:
        cond = [c for c in range(m) if c != a and c != b]
        conf_prob[(a, b)] = rank_evidence(C, a, b, cond, n_eff=dg.n_eff)

    out = []

    def _emit(a, b, orientation, strength, method, confidence, reason):
        """``a``/``b`` are component-local; directed marks read a -> b."""
        if orientation in ("o-o", "<->") and a > b:
            a, b = b, a
        out.append(Edge(source=int(nodes[a]), target=int(nodes[b]), lag=0,
                        orientation=orientation, strength=float(strength),
                        confounding_prob=float(conf_prob[(min(a, b), max(a, b))]),
                        identification_method=method,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        reason=reason))

    if branch == "lingam":
        order, dmat = approx_lingam_order(Z, return_diff=True)
        B = approx_lingam_adjacency(Z, order, candidates=sub)
        off = dmat[~np.eye(m, dtype=bool)]
        scale = float(np.median(np.abs(off - np.median(off)))) * 1.4826 + EPS
        for (a, b) in pairs:
            if B[b, a] != 0.0:        # a -> b
                z = max(dmat[a, b], 0.0) / scale
                _emit(a, b, "-->", abs(B[b, a]), "lingam", z / (z + LINGAM_Z_HALF),
                      "non-Gaussian linear SEM: causal order identified")
            elif B[a, b] != 0.0:      # b -> a
                z = max(dmat[b, a], 0.0) / scale
                _emit(b, a, "-->", abs(B[a, b]), "lingam", z / (z + LINGAM_Z_HALF),
                      "non-Gaussian linear SEM: causal order identified")
            else:
                _emit(a, b, "o-o", abs(pc[a, b]), "unidentified", 0.0,
                      "LiNGAM order recovered but the coefficient was pruned to zero")

    elif branch == "cumulant":
        stats = _cumulant_stats(Z)
        sc, cf = cumulant_orient(Z, pairs, stats=stats)
        for (a, b), s_, c_ in zip(pairs, sc, cf):
            if c_ < CUM_MIN_CONF or s_ == 0.0:
                _emit(a, b, "o-o", abs(pc[a, b]), "unidentified", float(c_),
                      f"cumulant asymmetry below the confidence floor ({c_:.2f} < "
                      f"{CUM_MIN_CONF})")
            elif s_ > 0:
                _emit(a, b, "-->", abs(s_), "cumulant", float(c_),
                      "third/fourth-order cross-cumulant asymmetry")
            else:
                _emit(b, a, "-->", abs(s_), "cumulant", float(c_),
                      "third/fourth-order cross-cumulant asymmetry")

    elif branch == "pag":
        for (a, b) in pairs:
            cp = conf_prob[(a, b)]
            if cp > PAG_BIDIRECTED:
                _emit(a, b, "<->", abs(pc[a, b]), "pag", cp,
                      "covariance-submatrix rank test: latent common cause")
            else:
                _emit(a, b, "o-o", abs(pc[a, b]), "pag", cp,
                      "component fails causal sufficiency; direction not identified")

    elif branch == "dagma":
        try:
            B = dagma_local(Z, candidates=sub)
        except ValueError as exc:
            for (a, b) in pairs:
                _emit(a, b, "o-o", abs(pc[a, b]), "unidentified", 0.0, str(exc))
            return out
        comp = compelled_edges(B)
        for (a, b) in pairs:
            if comp[b, a]:
                _emit(a, b, "-->", abs(B[b, a]), "dagma", 0.5,
                      "compelled by a v-structure in the recovered CPDAG")
            elif comp[a, b]:
                _emit(b, a, "-->", abs(B[a, b]), "dagma", 0.5,
                      "compelled by a v-structure in the recovered CPDAG")
            elif B[b, a] != 0.0 or B[a, b] != 0.0:
                _emit(a, b, "o-o", max(abs(B[b, a]), abs(B[a, b])), "dagma", 0.0,
                      "edge present but reversible within the Markov equivalence "
                      "class: near-Gaussian data does not identify its direction")
            else:
                _emit(a, b, "o-o", abs(pc[a, b]), "unidentified", 0.0,
                      "DAGMA left this candidate pair unselected")

    else:  # 'unidentified'
        why = _why_unidentified(dg, m)
        for (a, b) in pairs:
            _emit(a, b, "o-o", abs(pc[a, b]), "unidentified", 0.0, why)

    return out


def _why_unidentified(dg, m):
    """The audit trail for an abstention: which routing row actually fired."""
    if float(dg.n_eff) < MIN_NEFF_ANY:
        return f"effective sample size {dg.n_eff:.0f} below {MIN_NEFF_ANY}"
    if float(dg.nonlinearity_summary) > NONLIN_STRONG:
        return (f"nonlinearity {dg.nonlinearity_summary:.3f} > {NONLIN_STRONG}: the "
                "linear identification results do not apply; score-matching deferred")
    if m > DAGMA_MAX_NODES:
        return (f"component of {m} nodes exceeds the local solver limit "
                f"{DAGMA_MAX_NODES}")
    return "no identification result applies to this component"
