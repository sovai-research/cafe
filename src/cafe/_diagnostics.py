"""
CAFE-CD stage 3a -- cheap data diagnostics and the *identification router*.

Upstream stages (screening + skeleton) reduce a ``P x P x K`` problem to a tiny
candidate graph.  Lagged edges are oriented by time for free.  Only the
*contemporaneous* ambiguous component -- typically 5-30 nodes -- needs an
identification argument, and different assumptions call for different algebra.

This module computes, in ONE cheap pass over a residual matrix, the statistics
that decide *which identification result is actually available*, and then routes
to the cheapest valid branch.  When no branch is valid it says so; it never
manufactures a DAG.  (The benchmark literature is unambiguous that no single
discovery method dominates under misspecification, so honest abstention is the
correct behaviour, not a shortcoming.)

Design notes
------------
* **numpy only.**  No scipy, no sklearn, no numba.  Linear systems go through
  ``np.linalg.solve`` (never ``inv``/``pinv``); every per-variable quantity is a
  batched/vectorised expression rather than a Python loop.
* **Every threshold is a named module constant** with a stated justification, so
  the routing table is auditable and nothing is tuned to a particular dataset.
* Missing data is handled by masked moments + available-case covariance with a
  PSD projection; the multivariate diagnostics fall back to mean-filling, which
  is fine because these are *diagnostics*, not estimates.

Public API
----------
``Diagnostics``      dataclass of the computed statistics
``diagnose``         compute them from a residual matrix ``E`` (+ mask, skeleton)
``route``            the decision table: ``(diag, component_size, lag) -> branch``
``numerical_rank``   relative-tolerance numerical rank via SVD (shared helper)
``soft_rank_deficiency``  smooth rank-deficiency score for a stack of blocks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "Diagnostics",
    "diagnose",
    "route",
    "numerical_rank",
    "soft_rank_deficiency",
    "quad_purity",
    "BRANCHES",
    "NG_STRONG",
    "NG_WEAK",
    "SKEW_STRONG",
    "KURT_HEAVY",
    "LATENT_STRONG",
    "NONLIN_STRONG",
    "DAGMA_MAX_NODES",
    "LINGAM_MIN_NEFF",
    "MIN_NEFF_ANY",
    "RANK_RHO_FLOOR",
    "LATENT_QUAD_PURE",
    "LATENT_PAIR_TOPK",
]

# --------------------------------------------------------------------------- #
# Routing thresholds.  Named, justified, dataset-independent.
# --------------------------------------------------------------------------- #
# Excess kurtosis of reference laws: Gaussian 0, uniform -1.2, Laplace +3,
# t(5) +6.  ">1" is the conventional "clearly not Gaussian" line and is the
# regime where the LiNGAM entropy contrast has real signal to work with.
NG_STRONG = 1.0
# Below this the fourth cumulant is indistinguishable from Gaussian sampling
# noise (SE of excess kurtosis is sqrt(24/n) ~ 0.11 at n=2000).
NG_WEAK = 0.30
# Bulmer's rule of thumb: |skew| > 0.5 == "moderately skewed", i.e. the third
# cumulant carries usable asymmetry.
SKEW_STRONG = 0.50
# Very heavy tails: past t(6)-ish.  Fourth-order cumulant contrasts are the
# reliable tool here even when the entropy contrast is noisy.
KURT_HEAVY = 5.0
# latent_evidence is the *share of the component's variables that sit in a
# rank-1 ("pure") cluster*.  >1/3 of the component implicated is the abstention
# line.  Measured operating point on the fixtures in src/tests/test_orient.py
# (n=3000, P in 5..20): false-fire rate on causally sufficient DAGs 0.00-0.20,
# detection rate for a >=6-variable latent cluster 0.80-1.00.  Heuristic.
LATENT_STRONG = 0.35
# Bias-corrected R^2 gain of a quadratic/cubic Hermite basis over the linear
# one.  A tenth of the variance is a large amount of curvature to leave on the
# table; linear structural methods are not trustworthy past it.
NONLIN_STRONG = 0.10
# DAGMA here is deliberately a *local* tool.  30 nodes keeps the O(P^3) log-det
# barrier iterations in the microsecond range; above it we abstain rather than
# pretend to be a global solver.
DAGMA_MAX_NODES = 30
# The maximum-entropy contrast is a two-moment plug-in estimator; below ~100
# effective samples its variance swamps the non-Gaussian signal.
LINGAM_MIN_NEFF = 100
# Below this nothing at all is identifiable at a useful confidence.
MIN_NEFF_ANY = 50

BRANCHES = ("time", "lingam", "cumulant", "pag", "dagma", "unidentified")

# --------------------------------------------------------------------------- #
# Internal numerical constants.
# --------------------------------------------------------------------------- #
EPS = 1e-12
# Nonlinearity probe: regress each variable on its k strongest partners.
NONLIN_TOPK = 5
NONLIN_MAX_ROWS = 4000        # subsample cap; the probe is a diagnostic, not a fit
NONLIN_CLIP = 6.0             # winsorise at 6 SD so a few heavy-tail outliers
                              # cannot masquerade as curvature
NONLIN_SEED = 20240917
# Latent-confounding probe: FOFC-style "all three tetrads vanish" evidence.
LATENT_MAX_QUADS = 6000       # exhaustive below this many quads, sampled above
LATENT_SEED = 20240918
LATENT_MIN_VARS = 4           # a rank test needs >= 4 observed variables
LATENT_QUAD_PURE = 0.85       # per-quad purity above which a quad is a "witness"
LATENT_PAIR_TOPK = 3          # rank_evidence averages its best few witnesses
RANK_RTOL_Z = 3.0             # tolerance ~ 3 sigma of a correlation at n_eff
RANK_RTOL_MIN = 0.02
RANK_RTOL_MAX = 0.35
RANK_RHO_FLOOR = 0.25         # all six correlations in a quad must be a real
                              # association, not a barely-significant one -- a
                              # near-zero row is trivially rank-1 and is
                              # *absence* of association, not latent evidence
# Missingness-dependence probe.
MD_SCALE = 4.0                # chi-square excess -> [0,1) squashing scale
MD_MAX_PAIRS = 4000
RHO_CLIP = 0.999              # keeps 1 - rho^2 strictly positive


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class Diagnostics:
    """Cheap sufficient statistics that decide which identification result holds.

    Attributes
    ----------
    non_gaussianity : np.ndarray
        ``(P,)`` per-variable **excess kurtosis** (0 for a Gaussian).  This is the
        quantity the LiNGAM branch condition is stated in.
    normality_stat : np.ndarray
        ``(P,)`` per-variable Jarque-Bera statistic ``n/6 * (S^2 + K^2/4)``, i.e. a
        joint skew+kurtosis normality statistic.  Large == far from Gaussian.  No
        p-value is reported: converting it would need a chi-square CDF (scipy) and
        the asymptotic calibration is poor for heavy tails anyway.
    non_gaussianity_summary : float
        Scalar summary: mean ``|excess kurtosis|`` over the variables.  This is what
        :func:`route` thresholds against :data:`NG_STRONG` / :data:`NG_WEAK`.
    skewness : np.ndarray
        ``(P,)`` per-variable skewness.
    skewness_summary : float
        Mean ``|skewness|``.
    autocorr : np.ndarray
        ``(P,)`` lag-1 autocorrelation, computed *within* entity for panels.
    factor_strength : float
        Share of total variance carried by the eigenvalues of the correlation
        matrix that exceed the Marchenko-Pastur bulk edge ``(1+sqrt(P/n_eff))^2``.
        0.0 == no eigenvalue escapes the noise bulk.
    var_sortability : float or None
        Sortability of the marginal variances w.r.t. the supplied ``skeleton``.
        ``None`` when no skeleton is supplied.  See :func:`diagnose` for the exact
        definition in the directed and undirected cases.
    r2_sortability : float or None
        Same, using each variable's ``R^2`` regressed on all others (the
        scale-invariant sorting criterion) instead of its variance.
    nonlinearity : np.ndarray
        ``(P,)`` bias-corrected ``R^2`` gain of a quadratic+cubic Hermite basis over
        the plain linear one, per variable, against its strongest partners.
    nonlinearity_summary : float
        Mean of ``nonlinearity``.
    missing_dependence : float
        In ``[0, 1)``.  0 under MCAR; grows when the missingness indicators are
        predictable from contemporaneous or lagged observed values (MAR/MNAR).
    latent_evidence : float
        In ``[0, 1]``.  Mean soft rank-deficiency of 2x2 covariance sub-blocks
        (vanishing tetrads) over sampled quadruples.  High == a latent common
        cause is a better explanation than the sparse-DAG model.
    n_eff : float
        Effective sample size: mean observed rows per variable, deflated by the
        AR(1) factor ``(1-rho)/(1+rho)`` for the mean lag-1 autocorrelation.
    n_vars : int
        ``P``.
    """

    non_gaussianity: np.ndarray
    normality_stat: np.ndarray
    non_gaussianity_summary: float
    skewness: np.ndarray
    skewness_summary: float
    autocorr: np.ndarray
    factor_strength: float
    nonlinearity: np.ndarray
    nonlinearity_summary: float
    missing_dependence: float
    latent_evidence: float
    n_eff: float
    n_vars: int = 0
    var_sortability: Optional[float] = None
    r2_sortability: Optional[float] = None


# --------------------------------------------------------------------------- #
# Shared numerical-rank helpers (also used by ``_orient.rank_evidence``)
# --------------------------------------------------------------------------- #
def numerical_rank(M, rtol=1e-3):
    """Numerical rank of ``M`` at *relative* singular-value tolerance ``rtol``.

    ``rank = #{k : s_k > rtol * s_0}``.  ``M`` may be a stack ``(..., m, n)``, in
    which case the result has shape ``M.shape[:-2]``.

    This is a heuristic threshold, not a calibrated hypothesis test.  There is no
    null distribution behind ``rtol``; it is a statement about how small a
    singular value has to be before we are willing to call it zero.
    """
    M = np.asarray(M, dtype=float)
    s = np.linalg.svd(M, compute_uv=False)
    lead = s[..., :1]
    return np.sum(s > rtol * np.maximum(lead, EPS), axis=-1)


def soft_rank_deficiency(blocks, rtol):
    """Smooth per-block rank-deficiency score in ``(0, 1)`` for a stack of blocks.

    For each block we take the two leading singular values ``s0 >= s1`` and return

        ``d = rtol * s0 / (rtol * s0 + s1)``

    which is 0.5 exactly at the tolerance (``s1 == rtol * s0``), tends to 1 when
    the block is far more deficient than tolerance, and to 0 when it is clearly
    full rank.  The smooth form avoids a hard 0/1 cliff at an arbitrary cut, but
    ``rtol`` is still a heuristic: it is *not* calibrated against a null
    distribution of tetrad statistics.
    """
    blocks = np.asarray(blocks, dtype=float)
    s = np.linalg.svd(blocks, compute_uv=False)
    s0 = s[..., 0]
    s1 = s[..., 1] if s.shape[-1] > 1 else np.zeros_like(s0)
    return (rtol * s0) / (rtol * s0 + s1 + EPS)


def _rank_rtol(n_eff, rtol=None):
    """Sample-size-aware relative tolerance for the rank tests.

    A tetrad built from correlations has a finite-sample noise floor of order
    ``1/sqrt(n)``; the default tolerance is ``RANK_RTOL_Z / sqrt(n_eff)`` clipped
    into ``[RANK_RTOL_MIN, RANK_RTOL_MAX]``.  Heuristic, not calibrated.
    """
    if rtol is not None:
        return float(rtol)
    if n_eff is None or not np.isfinite(n_eff) or n_eff <= 1.0:
        return RANK_RTOL_MAX
    return float(np.clip(RANK_RTOL_Z / np.sqrt(n_eff), RANK_RTOL_MIN, RANK_RTOL_MAX))


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _as_panel(E, obs=None):
    """Return ``(X3, M3)`` with ``X3`` float64 ``(G, T, P)`` and ``M3`` bool observed.

    ``E`` may be ``(T, P)`` (single entity, ``G == 1``) or ``(n_ent, T, P)``.  ``obs``
    follows the CAFE convention **True == observed**; non-finite entries of ``E``
    are always treated as missing regardless of ``obs``.
    """
    X = np.asarray(E, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim == 2:
        X = X[None, :, :]
    if X.ndim != 3:
        raise ValueError(f"E must be (T,P) or (n_ent,T,P); got shape {np.shape(E)}")
    finite = np.isfinite(X)
    if obs is None:
        M = finite
    else:
        O = np.asarray(obs)
        if O.ndim == 1:
            O = O[:, None]
        if O.ndim == 2:
            O = O[None, :, :]
        if O.shape != X.shape:
            raise ValueError(f"obs shape {O.shape} does not match E shape {X.shape}")
        M = finite & O.astype(bool)
    X = np.where(finite, X, 0.0)
    return np.ascontiguousarray(X), np.ascontiguousarray(M)


def _masked_moments(X2, M2):
    """Masked column mean / sd / skewness / excess kurtosis / count.

    ``X2`` is ``(n, P)`` with missing entries already zeroed, ``M2`` the bool mask.
    """
    Mf = M2.astype(float)
    cnt = Mf.sum(0)
    cnt_s = np.maximum(cnt, 1.0)
    mean = (X2 * Mf).sum(0) / cnt_s
    d = (X2 - mean) * Mf
    m2 = (d * d).sum(0) / cnt_s
    sd = np.sqrt(np.maximum(m2, 0.0))
    sd_s = np.where(sd > EPS, sd, 1.0)
    m3 = (d ** 3).sum(0) / cnt_s
    m4 = (d ** 4).sum(0) / cnt_s
    skew = np.where(sd > EPS, m3 / sd_s ** 3, 0.0)
    exk = np.where(sd > EPS, m4 / sd_s ** 4 - 3.0, 0.0)
    return mean, sd, skew, exk, cnt


def _masked_cov(X2, M2, mean):
    """Available-case covariance, projected to the PSD cone.

    Columns are centred by their masked means and the missing cells zero-filled;
    the cross-products are then divided by the *pairwise* observation counts, which
    removes the attenuation that plain zero-filling would introduce.  Pairwise
    normalisation can break positive semi-definiteness, so the result is projected
    back by clipping the eigenvalues of ``np.linalg.eigh`` at zero -- but only when
    there is actually missing data (with none, ``Z'Z/(n-1)`` is already PSD and the
    projection is a no-op we skip).
    """
    Mf = M2.astype(float)
    Z = (X2 - mean) * Mf
    C = Z.T @ Z
    N = Mf.T @ Mf
    S = C / np.maximum(N - 1.0, 1.0)
    S = 0.5 * (S + S.T)
    if not M2.all():
        w, V = np.linalg.eigh(S)
        w = np.maximum(w, 0.0)
        S = (V * w) @ V.T
        S = 0.5 * (S + S.T)
    return S


def _corr_from_cov(S):
    d = np.sqrt(np.maximum(np.diag(S), EPS))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -RHO_CLIP, RHO_CLIP) + np.diag(np.full(S.shape[0], 1.0 - RHO_CLIP))


def _lag1_autocorr(X3, M3, mean, sd):
    """Within-entity lag-1 Pearson autocorrelation per variable, mask-aware."""
    G, T, P = X3.shape
    if T < 2:
        return np.zeros(P)
    sd_s = np.where(sd > EPS, sd, 1.0)
    Z = (X3 - mean) / sd_s * M3
    a = Z[:, 1:, :]
    b = Z[:, :-1, :]
    m = (M3[:, 1:, :] & M3[:, :-1, :]).astype(float)
    num = (a * b * m).sum(axis=(0, 1))
    da = (a * a * m).sum(axis=(0, 1))
    db = (b * b * m).sum(axis=(0, 1))
    den = np.sqrt(np.maximum(da * db, EPS))
    ac = np.where(den > EPS, num / den, 0.0)
    return np.clip(ac, -RHO_CLIP, RHO_CLIP)


# --------------------------------------------------------------------------- #
# Individual diagnostics
# --------------------------------------------------------------------------- #
def _factor_strength(C, n_eff):
    """Share of correlation-matrix variance above the Marchenko-Pastur bulk edge.

    For a ``P``-variate pure-noise correlation matrix with aspect ratio
    ``q = P / n``, the bulk of the eigenvalues lies below ``(1 + sqrt(q))^2``.
    Anything above that edge is a genuine common factor, so the share of the trace
    carried by those eigenvalues is a scale-free "how factor-driven is this?"
    statistic that is 0 for independent columns.
    """
    P = C.shape[0]
    if P < 2:
        return 0.0
    w = np.linalg.eigvalsh(C)
    tot = float(w.sum())
    if tot <= EPS:
        return 0.0
    q = P / max(float(n_eff), 2.0)
    edge = (1.0 + np.sqrt(q)) ** 2
    lead = w[w > edge]
    if lead.size == 0:
        return 0.0
    return float(np.clip(lead.sum() / tot, 0.0, 1.0))


def _neighbour_index(C, skeleton, k):
    """``(P, k)`` int array of probe partners for each variable.

    Skeleton neighbours when a skeleton is given (padded with the strongest
    remaining correlations if a node has fewer than ``k``), otherwise simply the
    ``k`` strongest absolute correlations.
    """
    A = np.abs(C).copy()
    np.fill_diagonal(A, -np.inf)
    if skeleton is not None:
        S = np.asarray(skeleton)
        S = (S != 0)
        S = S | S.T
        np.fill_diagonal(S, False)
        # skeleton neighbours first (score boosted above every non-neighbour)
        A = A + 10.0 * S.astype(float)
        np.fill_diagonal(A, -np.inf)
    order = np.argsort(-A, axis=1, kind="stable")
    return order[:, :k]


def _nonlinearity(X2, M2, mean, sd, C, skeleton, rng):
    """Bias-corrected R^2 gain of a Hermite quadratic/cubic basis over linear.

    For each variable ``i`` we regress it on its ``k`` strongest partners, once with
    the linear design ``[1, u]`` and once with ``[1, u, He2(u), He3(u)]`` where
    ``He2(u) = u^2 - 1`` and ``He3(u) = u^3 - 3u`` are the probabilists' Hermite
    polynomials (orthogonal under ``N(0,1)``, so the extra columns are essentially
    uncorrelated with the linear ones and the design stays well conditioned).

    The reported gain subtracts ``2k / n``, the R^2 that ``2k`` *pure noise*
    regressors earn by chance, so the statistic is ~0 under a linear model.  All
    ``P`` regressions are solved as one batched ``np.linalg.solve``.
    """
    n_all, P = X2.shape
    if P < 2:
        return np.zeros(P)
    k = int(min(NONLIN_TOPK, P - 1))
    sd_s = np.where(sd > EPS, sd, 1.0)
    Z = (X2 - mean) / sd_s
    Z = np.where(M2, Z, 0.0)                      # mean-fill (diagnostic only)
    Z = np.clip(Z, -NONLIN_CLIP, NONLIN_CLIP)
    if n_all > NONLIN_MAX_ROWS:
        idx = rng.choice(n_all, size=NONLIN_MAX_ROWS, replace=False)
        Z = Z[np.sort(idx)]
    n = Z.shape[0]
    if n < 4 * (3 * k + 1):
        return np.zeros(P)
    nb = _neighbour_index(C, skeleton, k)         # (P, k)
    U = np.transpose(Z[:, nb], (1, 0, 2))         # (P, n, k)
    y = Z.T                                       # (P, n)
    one = np.ones((P, n, 1))
    D_lin = np.concatenate([one, U], axis=2)
    D_nl = np.concatenate([one, U, U * U - 1.0, U ** 3 - 3.0 * U], axis=2)
    tss = np.maximum((y * y).sum(1), EPS)
    r2_lin = 1.0 - _batched_rss(D_lin, y) / tss
    r2_nl = 1.0 - _batched_rss(D_nl, y) / tss
    gain = (r2_nl - r2_lin) - (2.0 * k) / float(n)
    return np.clip(gain, 0.0, 1.0)


def _batched_rss(D, y):
    """Residual sum of squares of ``P`` independent OLS fits, one batched solve.

    ``D`` is ``(P, n, m)``, ``y`` is ``(P, n)``.  Ridge-stabilised normal equations;
    the ridge is relative to the design scale so it never bites a well-posed fit.
    """
    P, n, m = D.shape
    G = np.einsum("pna,pnb->pab", D, D)
    rhs = np.einsum("pna,pn->pa", D, y)
    ridge = 1e-8 * np.maximum(np.einsum("paa->p", G) / m, EPS)
    G = G + ridge[:, None, None] * np.eye(m)[None]
    beta = np.linalg.solve(G, rhs[..., None])[..., 0]
    fit = np.einsum("pna,pa->pn", D, beta)
    r = y - fit
    return (r * r).sum(1)


def _missing_dependence(X2, M2, mean, sd, n_eff):
    """Is missingness predictable from observed values?  0 under MCAR.

    Point-biserial correlations between each missingness indicator and (a) the
    contemporaneous values of the *other* variables and (b) the lag-1 value of
    every variable including itself.  Under MCAR each squared correlation has
    expectation ``1/n``, so ``stat = n_eff * mean(r^2)`` is ~1 by construction; the
    excess over 1 is squashed into ``[0, 1)`` by ``1 - exp(-(stat-1)/MD_SCALE)``.
    """
    if M2.all() or M2.shape[0] < 4:
        return 0.0
    n, P = X2.shape
    sd_s = np.where(sd > EPS, sd, 1.0)
    Z = np.where(M2, (X2 - mean) / sd_s, 0.0)
    R = (~M2).astype(float)
    rvar = R.var(0)
    valid_r = rvar > EPS
    if not valid_r.any():
        return 0.0
    Rc = R - R.mean(0)
    Rs = Rc / np.sqrt(np.maximum(rvar, EPS) * n)
    Zs = Z / np.sqrt(np.maximum((Z * Z).sum(0), EPS))
    cross = Rs.T @ Zs                       # (P missing-ind, P values)
    np.fill_diagonal(cross, 0.0)            # self is degenerate by construction
    lag = Rs[1:].T @ (Zs[:-1] / max(np.sqrt((n - 1) / n), EPS))
    r2 = np.concatenate([(cross ** 2)[valid_r].ravel(), (lag ** 2)[valid_r].ravel()])
    r2 = r2[np.isfinite(r2)]
    r2 = r2[r2 > 0.0]
    if r2.size == 0:
        return 0.0
    if r2.size > MD_MAX_PAIRS:
        r2 = np.sort(r2)[-MD_MAX_PAIRS:]
    stat = float(max(n_eff, 2.0) * r2.mean())
    return float(1.0 - np.exp(-max(0.0, stat - 1.0) / MD_SCALE))


def _quad_indices(P, max_quads=LATENT_MAX_QUADS, rng=None):
    """All ``C(P,4)`` sorted quadruples, or a deduplicated random sample if too many."""
    if P < LATENT_MIN_VARS:
        return np.zeros((0, 4), dtype=np.intp)
    n_all = P * (P - 1) * (P - 2) * (P - 3) // 24
    if n_all <= max_quads:
        a, b, c, d = np.meshgrid(*([np.arange(P)] * 4), indexing="ij")
        q = np.stack([a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=1)
        q = q[(q[:, 0] < q[:, 1]) & (q[:, 1] < q[:, 2]) & (q[:, 2] < q[:, 3])]
        return q.astype(np.intp)
    rng = np.random.default_rng(LATENT_SEED) if rng is None else rng
    q = rng.integers(0, P, size=(4 * max_quads, 4))
    q = np.sort(q, axis=1)
    ok = (q[:, 0] < q[:, 1]) & (q[:, 1] < q[:, 2]) & (q[:, 2] < q[:, 3])
    q = np.unique(q[ok], axis=0)
    return q[:max_quads].astype(np.intp)


def quad_purity(C, quads, rtol, rho_floor=RANK_RHO_FLOOR):
    """FOFC-style purity in ``[0, 1]`` for each quadruple: do **all three** tetrads vanish?

    A quadruple ``{a,b,c,d}`` admits three ways of splitting into two pairs, and each
    split gives a 2x2 cross-block of the correlation matrix::

        [[C[a,c], C[a,d]],      [[C[a,b], C[a,d]],      [[C[a,b], C[a,c]],
         [C[b,c], C[b,d]]]       [C[c,b], C[c,d]]]       [C[d,b], C[d,c]]]

    A *single latent common cause* of all four variables makes every one of these
    rank 1 simultaneously (all three tetrads vanish).  That simultaneity is what
    separates a latent from an observed mediator: in the chain
    ``a -> b -> c -> d`` two of the three tetrads vanish and the third does not,
    so taking the **minimum** deficiency across the three splits rejects chains,
    which the single-tetrad test cannot do.  This is the classical "pure tetrad"
    / FOFC condition (Silva, Scheines, Glymour & Spirtes).

    Quadruples in which any of the six pairwise correlations falls below
    ``rho_floor`` score 0: a near-zero row makes a block trivially rank-1, but
    that is *absence of association*, not evidence of a common cause.

    Returns
    -------
    np.ndarray
        ``(len(quads),)`` purity scores; 0 for filtered-out quadruples.
    """
    quads = np.asarray(quads, dtype=np.intp)
    out = np.zeros(quads.shape[0], dtype=float)
    if quads.shape[0] == 0:
        return out
    a, b, c, d = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    keep = np.ones(quads.shape[0], dtype=bool)
    for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
        keep &= np.abs(C[u, v]) > rho_floor
    if not keep.any():
        return out
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]

    def _block(i, k, j, l):
        return np.stack(
            [np.stack([C[i, j], C[i, l]], axis=-1),
             np.stack([C[k, j], C[k, l]], axis=-1)],
            axis=-2,
        )

    dfc = np.stack([
        soft_rank_deficiency(_block(a, b, c, d), rtol),
        soft_rank_deficiency(_block(a, c, b, d), rtol),
        soft_rank_deficiency(_block(a, d, b, c), rtol),
    ], axis=0).min(axis=0)
    out[keep] = dfc
    return out


def _latent_evidence(C, n_eff, rng, rtol=None):
    """Share of variables implicated in a latent ("pure") cluster, in ``[0, 1]``.

    Every quadruple is scored by :func:`quad_purity`; those above
    :data:`LATENT_QUAD_PURE` are *witnesses* of a single latent common cause, and
    the statistic is the fraction of the variables that appear in at least one
    witness.  It reads directly as "this much of the component looks
    latent-driven", which is exactly what the router needs in order to decide
    whether to abstain for the whole component.

    Honest caveats.  (1) The tolerance behind ``soft_rank_deficiency`` is a
    heuristic, not a calibrated tetrad test -- there is no null distribution
    behind it.  (2) The test needs at least four variables *and* at least four
    indicators of the same latent; a latent confounding exactly two observed
    variables is observationally indistinguishable from a direct edge at second
    order and is undetectable here, by any method.  (3) It over-fires rather than
    under-fires, which is the safe direction: the only consequence is abstention.
    """
    P = C.shape[0]
    if P < LATENT_MIN_VARS:
        return 0.0
    rtol = _rank_rtol(n_eff, rtol)
    quads = _quad_indices(P, rng=rng)
    if quads.shape[0] == 0:
        return 0.0
    pur = quad_purity(C, quads, rtol)
    hit = pur > LATENT_QUAD_PURE
    if not hit.any():
        return 0.0
    involved = np.unique(quads[hit].ravel())
    return float(np.clip(involved.size / P, 0.0, 1.0))


def _r2_on_all_others(S):
    """``R^2`` of each variable regressed on all the others, from the precision.

    ``R_i^2 = 1 - 1 / (S_ii * (S^-1)_ii)``.  All ``P`` diagonal entries of the
    inverse are needed at once, so this is the one place we form an inverse -- via
    ``np.linalg.solve`` against the identity (one LAPACK LU with ``P`` right-hand
    sides), never ``np.linalg.inv``/``pinv``.
    """
    P = S.shape[0]
    if P < 2:
        return np.zeros(P)
    d = np.diag(S)
    ridge = 1e-8 * max(float(d.mean()), EPS)
    A = S + ridge * np.eye(P)
    try:
        Ainv = np.linalg.solve(A, np.eye(P))
    except np.linalg.LinAlgError:
        return np.zeros(P)
    denom = np.maximum(d * np.diag(Ainv), 1.0 + EPS)
    return np.clip(1.0 - 1.0 / denom, 0.0, 1.0)


def _sortability(values, skeleton):
    """Sortability of a per-variable criterion w.r.t. a skeleton.

    * **Directed skeleton** (asymmetric boolean matrix, ``A[target, source]``):
      the standard sortability of Reisach et al. over the direct edges -- the
      fraction of edges along which the criterion *increases* (ties count 1/2).
      1.0 means the criterion alone recovers a topological order, which is the
      well-known simulation artefact to be suspicious of.
    * **Undirected skeleton** (symmetric): directions do not exist, so a true
      sortability cannot be computed.  We report the criterion's *decisiveness*
      instead -- the mean absolute log-gap across skeleton edges, normalised by
      the overall log-spread and clipped to ``[0, 1]``.  High means the criterion
      would drive orientation if any method leaned on it; it says nothing about
      whether it would be right.
    """
    if skeleton is None:
        return None
    A = np.asarray(skeleton)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != values.shape[0]:
        return None
    A = (A != 0)
    A = A.copy()
    np.fill_diagonal(A, False)
    if not A.any():
        return None
    v = np.log(np.maximum(np.asarray(values, dtype=float), EPS))
    if np.array_equal(A, A.T):
        iu = np.triu_indices(A.shape[0], 1)
        sel = A[iu]
        gap = np.abs(v[iu[0]][sel] - v[iu[1]][sel])
        spread = float(v.max() - v.min())
        if spread <= EPS:
            return 0.0
        return float(np.clip(gap.mean() / spread, 0.0, 1.0))
    tgt, src = np.nonzero(A)
    inc = (v[tgt] > v[src]).astype(float) + 0.5 * (v[tgt] == v[src]).astype(float)
    return float(inc.mean())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def diagnose(E, obs=None, skeleton=None):
    """Compute every routing statistic in one cheap pass.

    Parameters
    ----------
    E : array
        Residual matrix, ``(T, P)`` or ``(n_ent, T, P)``.  Non-finite entries count
        as missing.
    obs : array of bool, optional
        Same shape as ``E``; **True == observed** (the CAFE convention).
    skeleton : array, optional
        ``(P, P)`` adjacency.  Symmetric == undirected skeleton; asymmetric is read
        as ``A[target, source]``.  Only the two sortability fields need it; they
        are ``None`` when it is not supplied.

    Returns
    -------
    Diagnostics
    """
    X3, M3 = _as_panel(E, obs)
    G, T, P = X3.shape
    X2 = X3.reshape(G * T, P)
    M2 = M3.reshape(G * T, P)

    mean, sd, skew, exk, cnt = _masked_moments(X2, M2)
    jb = cnt / 6.0 * (skew ** 2 + 0.25 * exk ** 2)
    ac = _lag1_autocorr(X3, M3, mean, sd)

    rho_bar = float(np.clip(np.mean(np.abs(ac)), 0.0, 0.95)) if P else 0.0
    n_bar = float(cnt.mean()) if P else 0.0
    n_eff = max(1.0, n_bar * (1.0 - rho_bar) / (1.0 + rho_bar))

    S = _masked_cov(X2, M2, mean)
    C = _corr_from_cov(S)

    rng = np.random.default_rng(NONLIN_SEED)
    nl = _nonlinearity(X2, M2, mean, sd, C, skeleton, rng)
    md = _missing_dependence(X2, M2, mean, sd, n_eff)
    le = _latent_evidence(C, n_eff, np.random.default_rng(LATENT_SEED))
    fs = _factor_strength(C, n_eff)

    var_sort = _sortability(np.maximum(np.diag(S), EPS), skeleton)
    r2_sort = _sortability(_r2_on_all_others(S) + EPS, skeleton)

    return Diagnostics(
        non_gaussianity=exk,
        normality_stat=jb,
        non_gaussianity_summary=float(np.mean(np.abs(exk))) if P else 0.0,
        skewness=skew,
        skewness_summary=float(np.mean(np.abs(skew))) if P else 0.0,
        autocorr=ac,
        factor_strength=fs,
        nonlinearity=nl,
        nonlinearity_summary=float(np.mean(nl)) if P else 0.0,
        missing_dependence=md,
        latent_evidence=le,
        n_eff=float(n_eff),
        n_vars=int(P),
        var_sortability=var_sort,
        r2_sortability=r2_sort,
    )


def route(diag, component_size, lag=0):
    """Pick the cheapest *valid* identification branch for one component.

    The table is evaluated **top-down**; the first matching row wins.  Ordering is
    not cosmetic -- each earlier row invalidates an assumption the later rows rely
    on, so the priority is exactly the dependency order of the assumptions.

    ======  ==================================================  ================
    order   condition                                           branch
    ======  ==================================================  ================
    0       ``lag != 0`` (the edge is lagged)                   ``'time'``
    1       ``n_eff < MIN_NEFF_ANY``                            ``'unidentified'``
    2       ``nonlinearity_summary > NONLIN_STRONG``            ``'unidentified'``
    3       ``latent_evidence > LATENT_STRONG``                 ``'pag'``
    4       ``|excess kurt| > NG_STRONG`` and enough samples    ``'lingam'``
    5       ``|skew| > SKEW_STRONG`` or ``kurt > KURT_HEAVY``   ``'cumulant'``
    6       ``component_size <= DAGMA_MAX_NODES``               ``'dagma'``
    7       otherwise                                           ``'unidentified'``
    ======  ==================================================  ================

    Row-by-row rationale:

    0. **time.**  A lagged edge is oriented by the arrow of time.  No search is
       ever run on it, so it never reaches the rest of the table.
    1. Below :data:`MIN_NEFF_ANY` effective samples every contrast below is noise.
    2. **Linearity first.**  Every branch underneath -- the entropy contrast, the
       cumulant ratios, the covariance rank test, the linear DAGMA score -- is
       stated for a *linear* SEM.  If the quadratic basis buys
       :data:`NONLIN_STRONG` of extra variance, none of them is valid, and
       score-matching style nonlinear identification is deliberately deferred, so
       we abstain (``o-o``) rather than run an invalid contrast.
    3. **Causal sufficiency second.**  LiNGAM, the cumulant ratios and DAGMA all
       assume no latent confounder.  When the covariance-submatrix rank test
       fires, that assumption is violated and the honest output is a PAG-style
       ``<->`` / ``o-o``, never an arrow.
    4. **lingam.**  Strong excess kurtosis is exactly the regime where the
       LiNGAM entropy contrast identifies a full causal order in one shot, so it
       is the cheapest thing that actually identifies.
    5. **cumulant.**  Asymmetric-but-not-kurtotic noise: the third-order cross
       cumulants break the direction symmetry pairwise, cheaply, and reuse one set
       of sufficient statistics across every pair.
    6. **dagma.**  Near-Gaussian and small: the log-det acyclicity program is the
       remaining tool.  **This row promises a search, not an identification
       result.**  A linear-Gaussian DAG is identified only up to its Markov
       equivalence class (the equal-error-variance escape hatch is unavailable
       here because ``_orient`` standardises the data first, deliberately, to
       close the varsortability leak).  So ``_orient`` runs DAGMA, reduces the
       recovered DAG to its CPDAG via ``compelled_edges``, and emits an arrow
       only for a *compelled* edge; every reversible edge comes back ``o-o``.
    7. Too large for the local solver, or nothing above applied: abstain.

    Parameters
    ----------
    diag : Diagnostics
        Ideally computed on the *component's* columns, not globally.
    component_size : int
        Number of nodes in the contemporaneous component being routed.
    lag : int, optional
        0 for a contemporaneous component (the default).  Any non-zero lag short
        circuits to ``'time'``.

    Returns
    -------
    str
        One of :data:`BRANCHES`.
    """
    if lag != 0:
        return "time"
    if diag is None:
        return "unidentified"
    if float(diag.n_eff) < MIN_NEFF_ANY:
        return "unidentified"
    if float(diag.nonlinearity_summary) > NONLIN_STRONG:
        return "unidentified"
    if float(diag.latent_evidence) > LATENT_STRONG:
        return "pag"
    if (float(diag.non_gaussianity_summary) > NG_STRONG
            and float(diag.n_eff) >= LINGAM_MIN_NEFF):
        return "lingam"
    if (float(diag.skewness_summary) > SKEW_STRONG
            or float(np.mean(diag.non_gaussianity)) > KURT_HEAVY):
        return "cumulant"
    if int(component_size) <= DAGMA_MAX_NODES:
        return "dagma"
    return "unidentified"
