"""
Panel group-sparse VAR *skeleton* solver -- the estimation core of CAFE-CD.

Given an idiosyncratic residual panel ``E`` of shape ``(n_entities, T, P)`` (NaN at
unobserved cells) and a per-target candidate set ``C_i = {(j, k)}`` from a screening
stage, fit the sparse panel VAR

    e[e, t, i] = sum_{(j,k) in C_i} ( b0[i,j,k] + d[e,i,j,k] ) * e[e, t-k, j] + eps

by minimising the SPARSE-GROUP objective

    0.5 * sum_e sum_t ( y - sum (b0 + d_e) z )^2
      + lam0 * sum_{i,j}   sqrt(K) ||b0[i,j,:]||_2      # shared skeleton, group = LAGS
      + lam1 * sum_{e,i,j} sqrt(K) ||d [e,i,j,:]||_2    # entity deviations
      + lam2 * sum |d|                                  # optional extra sparsity

with ``lam1 = kappa * lam0`` (``kappa`` default 4). The hierarchy "shared relationship
unless the data pays for an exception" is imposed STRUCTURALLY by the ``b0 + d_e``
split, not by nesting the penalties. Grouping over lags means "j Granger-causes i at
*some* lag" enters or leaves the model as a unit -- exactly the skeleton object the
discovery stage wants.

-----------------------------------------------------------------------------------
THE DESIGN MATRIX IS NEVER MATERIALISED
-----------------------------------------------------------------------------------
Every design column is a *lagged residual column*, shared across all P targets; only
``y`` changes from target to target. So we accumulate ONE moment tensor over zero-copy
slice views:

    M[k, k', j, j'] = sum_e ( E_e[Kmax-k : T-k, :] )^T ( E_e[Kmax-k' : T-k', :] )

of shape ``(K+1, K+1, P, P)``. Then

  * any block's ``X^T X`` is a *gather*: ``(X'X)[(j,k),(j',k')] = M[k,k',j,j']``
  * ``X^T y`` is a *slice of the same tensor*: ``X'y_i = M[k, 0, j, i]``

Cost is ``binom(K+1,2) + K+1`` batched ``(n,P)^T (n,P)`` matmuls on strided views;
the symmetry ``M[k,k',j,j'] == M[k',k,j',j]`` halves it. Memory is
``(K+1)^2 * P^2 * 8`` bytes -- 6.5 MB at K=8,P=100; 162 MB at P=500. When the full
tensor would exceed ``moment_budget_gb`` (default 1 GB, ~P<=1750 at K=8) the solver
falls back to the sub-panel of series the candidate sets actually reference (the
column-wise demeaning and the moments are both separable in the column index, so this
is exact) -- still one streaming pass over slice views, never an ``(n, A)`` design.

-----------------------------------------------------------------------------------
BLOCK UPDATE -- EXACT, NOT ISTA
-----------------------------------------------------------------------------------
For a group ``g`` with ``G_g = X_g'X_g`` (``K x K``, ``K <= 8``) and ``c_g = X_g'r``:

  1. ZERO TEST FIRST: ``||c_g||_2 <= lam*sqrt(K)`` => ``beta_g = 0``, skip. This KKT
     check is what makes the whole thing fast -- almost every candidate fails it.
  2. Otherwise the block minimiser of ``0.5 b'G b - c'b + lam*sqrt(K)*||b||`` obeys
     ``b = (G + (lam*sqrt(K)/||b||) I)^{-1} c``. With ONE cached ``np.linalg.eigh(G)``
     (``G = Q diag(w) Q'``, ``u = Q'c``) the fixed point in ``mu = lam*sqrt(K)/||b||``
     is the root of the strictly increasing scalar

         h(mu) = sum_m ( mu*u_m / (w_m + mu) )^2 - (lam*sqrt(K))^2

     with ``h(0+) = -(lam sqrt K)^2 < 0`` and ``h(inf) = ||c||^2 - (lam sqrt K)^2 > 0``
     once the zero test has failed: a guaranteed bracket, each step ``O(K)``. We run
     safeguarded NEWTON inside that bracket (3-6 steps, warm-started from the previous
     solve of the same block) rather than raw bisection (~50 steps) -- same guarantee,
     ~6x cheaper, and bisection is still the fallback whenever a Newton step would
     leave the bracket. This costs essentially nothing at ``K <= 8`` and converges in
     far fewer outer sweeps than a majorised ISTA update. The FISTA form is kept only
     for the ``lam2 > 0`` sparse-group case, where no closed fixed point exists.

-----------------------------------------------------------------------------------
WORKING SETS (exact, not a heuristic)
-----------------------------------------------------------------------------------
  1. ``lam_max = max_g ||X_g'y||_2 / sqrt(K)`` -- closed form, the smallest lambda
     whose solution is all-zero.
  2. Path ``lam_l = lam_max * lambda_ratio^(l/(L-1))`` (geometric, default 100x range).
  3. Per lambda ``W <- support(lam_{l-1}) union top-p_new`` by KKT violation
     ``||X_g'r||/sqrt(K) - lam``, with ``p_new = max(10, 2*|support|)``.
  4. Block coordinate descent to tolerance ON ``W`` ONLY.
  5. Global KKT sweep over every candidate straight from the precomputed statistics.
     Because ``c = X'y - G b`` is maintained for ALL columns, the sweep is O(A) and
     essentially free. Violators join ``W``; repeat. This GUARANTEES the exact
     solution of the full problem while only ever running descent on ``W``.
  6. Warm start both ``beta`` and ``W`` from ``lam_{l-1}``.
  7. ``screening="kkt"`` (default, zero correctness risk) or ``"gapsafe"`` (duality-gap
     sphere test ``||X_g'theta|| + r_gap ||X_g||_2 < w_g``, from the same statistics).
     GAP-safe only removes provably-zero groups; step 5 still certifies the answer.

-----------------------------------------------------------------------------------
THE b0 + d_e SPLIT WITHOUT AN (E+1)x-WIDE DESIGN
-----------------------------------------------------------------------------------
The augmented Gram is block structured -- shared/shared ``= sum_e G_e``, shared/e
``= G_e``, e/e' ``= G_e delta_{ee'}``. Per-entity moment tensors are too big to store,
so we run a HYBRID: the pooled tensor for the ``b0`` level, and per-entity residual
streams ``r_e`` (only ``O(E*T*P)`` -- the size of the residual itself) with
``X_g' r_e`` formed on the fly for the deviation level. Two-level block Gauss-Seidel:

    for lam in path:
        repeat:
            (1) BCD on b0  (d fixed)   -- pooled Gram gather
            (2) BCD on d_e (b0 fixed)  -- per-entity residual streams
            (3) global KKT check
        until no violation

The objective is convex, the blocks are separable within each level and each level is
solved exactly, so the alternation converges.

Both directions of the residual/statistic exchange are ONE batched GEMM per lag:
``X'v`` for every ``(k, j)`` at once is ``Z^T Rs`` with ``Rs[:, K-k:T-k, k] = v``, and
the fitted values ``X beta`` are ``sum_k Z[:, K-k:T-k, :] @ coef[k]``. Never a loop
over columns.

-----------------------------------------------------------------------------------
MISSINGNESS -- a real bias trap, handled explicitly
-----------------------------------------------------------------------------------
We set ``E_k <- E_k * V_k`` (NaN -> 0) and accumulate the co-observation counts
``N[k,k',j,j'] = V_k^T V_{k'}`` alongside ``M``. Zero-filling WITHOUT renormalisation
biases every coefficient toward zero by the co-observation rate: ``sum x y`` counts
co-observed pairs while ``sum x^2`` counts all x-observed rows, and each cross-column
entry counts yet another subset, so the normal equations mix incompatible sample
sizes. The fix is the available-case (pairwise-complete) rescale

    Gt[a,b] = M[a,b] * n_tot / N[a,b] / (scale_a * scale_b),
    scale_a = sqrt( M[a,a] / N[a,a] )        # RMS of column a over its OWN valid count

which (i) rescales every entry to a common effective sample size and (ii) renormalises
each design column to unit norm using its own valid count, so ``Gt[a,a] == n_tot``
EXACTLY for every design column and every target's ``y``. Under MCAR this is
consistent; with complete data every correction factor is exactly 1, so nothing
changes. The deviation level applies the same correction to its per-entity ``K x K``
Grams and correlations. Pass ``available_case=False`` to reproduce (and measure) the
biased naive zero-fill behaviour -- do not use it for estimation.

-----------------------------------------------------------------------------------
FWL DEMEANING -- once, before any moments
-----------------------------------------------------------------------------------
    Ed = E - E.mean(1, keepdims=True) - E.mean(0, keepdims=True) + E.mean((0,1), ...)

is the exact two-way within transform, ``O(ETP)``, applied once up front (with
count-safe means under missingness). It is separable in the column index, which is
what makes the sub-panel memory fallback exact. The time (cross-sectional) mean at
date t uses only date t, so it is contemporaneous and causal; the entity mean uses
only rows in the prefix being fitted, so a fit that claims to use rows ``<= t0`` is
invariant to deleting rows ``> t0``.

NOTE: under entity fixed effects the lagged regressors carry a Nickell bias of
``O(1/T)`` -- about 0.5% at T=200, shrinking as 1/T. We state it; we do NOT implement
Arellano-Bond. For skeleton *recovery* (which coefficients are nonzero) an O(1/T)
attenuation is immaterial; for coefficient inference at small T it is not.

-----------------------------------------------------------------------------------
MODEL SELECTION
-----------------------------------------------------------------------------------
``select="forward"`` (default): forward-chaining time-series validation -- fit on
``t <= t0``, score on ``(t0, t1]``, expanding. Strictly causal end to end: the lambda
grid is derived from the FIRST training prefix, a fold's entity means come from that
fold's training prefix, the standardisation comes from the training prefix, and a
validation block is only ever scored by a model fitted strictly before it. K-fold CV
over time is a look-ahead hazard and would fail this project's causal certificate --
it is deliberately NOT offered.

By default the forward selector applies the ONE-STANDARD-ERROR rule over folds: the
sparsest lambda within one fold-SE of the best validation score. Plain argmin of
predictive MSE buys dozens of near-zero false edges for a fraction of a percent of
accuracy -- correct for forecasting, wrong for a skeleton. ``select_rule="min"``
restores the argmin; both use only fold scores, so both are equally causal.

``select="ebic"`` is the fast (single-fit) option, summed over targets:
``n log(RSS/n) + df log(n) + 2 gamma log(comb(|C_i|, df))`` with ``gamma = 0.5``.

WHICH SELECTOR TO USE. They optimise different things and it shows. Forward chaining
minimises PREDICTIVE risk; eBIC is SELECTION consistent. On these panels the held-out
MSE curve is genuinely near-flat over a wide range -- at P=50 it improves 0.8% going
from the 12 true edges to 576 -- because the two-way within transform leaves weak
cross-sectional dependence that a dense VAR can exploit. Predictive CV happily buys
that; a skeleton must not. Measured on the same fixtures (10 entities, T=200, K=8):
eBIC recovers F1 = 1.00 at P = 20, 50 and 100, while forward chaining with the 1-SE
rule gets 0.55 / 0.21 / 0.13. **If the SUPPORT is the deliverable, use
``select="ebic"``.** ``select="forward"`` is the right default when the fitted
coefficients feed a forecast, and it is the only selector here that is validated out
of sample.

Strict numpy only: no scipy, no sklearn, no numba. SPD systems go through
``np.linalg.solve`` / ``np.linalg.eigh``, never ``inv`` / ``pinv``; every heavy
contraction is a batched BLAS call over preallocated buffers.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "panel_group_lasso",
    "build_moment_tensor",
    "MomentTensor",
    "SkeletonResult",
    "conditioning_diagnostic",
    "bivariate_granger",
]

_EPS = 1e-12
_GATHER_CHUNK = 256          # rows per chunk when gathering a Gram from the tensor
_NEWTON_MAX = 60             # safeguarded-Newton steps for the exact block fixed point
_NEWTON_TOL = 1e-14          # relative residual of the scalar fixed point
_EBIC_GAMMA = 0.5
_ZERO_REL = 1e-10            # relative slack reconciling the two KKT zero tests
_L2_KKT_ROUNDS = 8           # global KKT rounds inside one deviation-level solve


# --------------------------------------------------------------------------- #
#  FWL two-way demeaning (count-safe under missingness)
# --------------------------------------------------------------------------- #
def _two_way_demean(X, Vf, mode):
    """Exact two-way within transform of a zero-filled panel.

    ``X`` is (n_ent, T, P) with NaN already replaced by 0 and ``Vf`` the matching
    float 0/1 validity mask.  Returns the demeaned, re-zero-filled panel.

    ``mode``: ``"twoway"`` entity + time (cross-sectional) FE -- the standard within
    transform; ``"entity"`` entity FE only; ``"none"`` no demeaning.

    The time mean at date t is a *contemporaneous cross-section*: it uses only data at
    date t, so it is causal.  The entity mean uses the whole array handed in, which is
    why callers slice to the prefix they are entitled to BEFORE calling.

    With a single entity the cross-sectional mean would annihilate the panel, so the
    time term is skipped automatically (time FE is not identified at n_ent == 1).

    Column-separable by construction: every mean is taken within a fixed j, so
    demeaning a subset of columns gives the same answer as demeaning all of them and
    slicing.  The memory-budget fallback relies on this.
    """
    if mode == "none":
        return X * Vf
    ent_s = X.sum(1, keepdims=True)
    ent_c = Vf.sum(1, keepdims=True)
    out = (X - ent_s / np.maximum(ent_c, 1.0)) * Vf
    if mode == "twoway" and X.shape[0] >= 2:
        # RE-MASK before the second projection. Without the ``* Vf`` above, an
        # unobserved cell would carry ``-entity_mean`` into the cross-sectional sum
        # while the count only counted observed cells -- contaminating every time
        # effect with the entity fixed effects and inflating the residual scale by
        # (here) 10-15% at 20% MCAR. Silent, and it biases every downstream moment.
        tim_s = out.sum(0, keepdims=True)
        tim_c = Vf.sum(0, keepdims=True)
        out = (out - tim_s / np.maximum(tim_c, 1.0)) * Vf
    return out


# --------------------------------------------------------------------------- #
#  Moment tensor
# --------------------------------------------------------------------------- #
@dataclass
class MomentTensor:
    """Sufficient statistics for every panel-VAR regression on a residual panel.

    Attributes
    ----------
    M : (K+1, K+1, P, P)
        ``M[k,k',j,j'] = sum_e sum_t Z_e[K+t-k, j] * Z_e[K+t-k', j']`` over the
        zero-filled (and, by default, demeaned) panel.  Lag index 0 is the
        contemporaneous level -- the target row -- and 1..K are the design lags, so
        ``M[k, 0, j, i]`` is literally ``X'y`` for target ``i``.
    N : (K+1, K+1, P, P)
        matching co-observation counts ``V_k^T V_k'`` (float).
    Z, V : (n_ent, T, P)
        the demeaned zero-filled panel and its float 0/1 validity mask.
    scale : (K+1, P)
        ``sqrt(M[k,k,j,j] / N[k,k,j,j])`` -- the RMS of design column ``(j,k)`` over
        its OWN valid count.  Dividing the column by this is the renormalisation that
        removes the co-observation-rate shrinkage.
    """

    M: Optional[np.ndarray]
    N: Optional[np.ndarray]
    Z: np.ndarray
    V: np.ndarray
    scale: np.ndarray
    max_lag: int
    n_ent: int
    n_rows: int
    n_tot: int
    T: int
    P: int
    series: Optional[np.ndarray] = None      # original column ids when subset
    _gt: Optional[np.ndarray] = field(default=None, repr=False)
    _gt_ac: bool = field(default=True, repr=False)

    @property
    def nbytes(self) -> int:
        n = 0
        if self.M is not None:
            n += self.M.nbytes
        if self.N is not None:
            n += self.N.nbytes
        return int(n)

    def scaled(self, available_case: bool = True) -> np.ndarray:
        """The standardised, available-case-corrected Gram tensor ``Gt``.

        ``Gt[k,k',j,j'] = M * n_tot / N / (scale[k,j] * scale[k',j'])`` so every design
        column has ``Gt[a,a] == n_tot`` exactly (unit RMS) and every entry sits at the
        same effective sample size.  With complete data all correction factors are 1
        and this is nothing but the column standardisation.

        ``available_case=False`` drops the ``n_tot / N`` factor -- the naive zero-fill
        statistics, kept so the bias they induce can be measured rather than assumed.
        """
        if self._gt is not None and self._gt_ac == bool(available_case):
            return self._gt
        s = self.scale
        den = s[:, None, :, None] * s[None, :, None, :]
        gt = self.M / np.maximum(den, _EPS)
        if available_case:
            gt *= self.n_tot / np.maximum(self.N, 1.0)
        np.nan_to_num(gt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self._gt = gt
        self._gt_ac = bool(available_case)
        return gt


def _accumulate_moments(Z, Vf, K, M, N):
    """Add the (K+1, K+1, P, P) cross-moments of ``Z`` (and counts of ``Vf``) into M/N.

    Rows K..T-1 are the target rows, so lag ``k`` contributes the strided view
    ``Z[:, K-k:T-k, :]``.  For each of the ``binom(K+1,2)+K+1`` ordered pairs we issue
    ONE batched ``(n_ent, n_rows, P)^T (n_ent, n_rows, P)`` matmul and sum over
    entities; the symmetry ``M[k,k',j,j'] == M[k',k,j',j]`` fills the mirror for free.
    """
    T = Z.shape[1]
    for k in range(K + 1):
        At = Z[:, K - k:T - k, :].transpose(0, 2, 1)
        Avt = Vf[:, K - k:T - k, :].transpose(0, 2, 1)
        for kp in range(k, K + 1):
            m = np.matmul(At, Z[:, K - kp:T - kp, :]).sum(axis=0)
            n = np.matmul(Avt, Vf[:, K - kp:T - kp, :]).sum(axis=0)
            M[k, kp] += m
            N[k, kp] += n
            if kp != k:
                M[kp, k] += m.T
                N[kp, k] += n.T


def build_moment_tensor(E, obs=None, max_lag=8, demean="twoway", series=None):
    """Build the panel moment tensor -- the only pass that ever touches the data.

    Parameters
    ----------
    E : (n_ent, T, P) array
        Idiosyncratic residual panel; NaN marks unobserved cells.  A (T, P) array is
        promoted to a single-entity panel.
    obs : (n_ent, T, P) bool array, optional
        Extra observation mask, ANDed with ``isfinite(E)``.
    max_lag : int
        ``K``.  Lag index 0 in the tensor is the contemporaneous level (the target).
    demean : {"twoway", "entity", "none"}
        FWL within transform applied before any moment is taken.
    series : 1-D int array, optional
        Restrict to a subset of the P columns.  Tensor indices then refer to positions
        within the subset; the original ids are kept on ``MomentTensor.series``.

    Returns
    -------
    MomentTensor
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 2:
        E = E[None]
    if E.ndim != 3:
        raise ValueError("E must be (n_entities, T, P); got shape %r" % (E.shape,))
    if obs is not None:
        obs = np.asarray(obs, dtype=bool)
        if obs.ndim == 2:
            obs = obs[None]
        if obs.shape != E.shape:
            raise ValueError("obs must match E's shape")
    if series is not None:
        series = np.asarray(series, dtype=np.intp)
        E = E[:, :, series]
        if obs is not None:
            obs = obs[:, :, series]
    n_ent, T, P = E.shape
    K = int(max_lag)
    if K < 1:
        raise ValueError("max_lag must be >= 1")
    if T <= K:
        raise ValueError("need T > max_lag; got T=%d, max_lag=%d" % (T, K))

    V = np.isfinite(E)
    if obs is not None:
        V &= obs
    Vf = V.astype(np.float64)
    Z = _two_way_demean(np.where(V, E, 0.0), Vf, demean)

    M = np.zeros((K + 1, K + 1, P, P), dtype=np.float64)
    N = np.zeros((K + 1, K + 1, P, P), dtype=np.float64)
    _accumulate_moments(Z, Vf, K, M, N)

    diag_m = np.einsum("kkjj->kj", M)
    diag_n = np.einsum("kkjj->kj", N)
    scale = np.sqrt(diag_m / np.maximum(diag_n, 1.0))
    scale = np.where(scale > _EPS, scale, 1.0)

    return MomentTensor(M=M, N=N, Z=Z, V=Vf, scale=scale, max_lag=K, n_ent=n_ent,
                        n_rows=T - K, n_tot=n_ent * (T - K), T=T, P=P, series=series)


# --------------------------------------------------------------------------- #
#  Gram gathers -- index arithmetic on the tensor, never a design matrix
# --------------------------------------------------------------------------- #
def _gather_gram(Gt, jj, kk, chunk=_GATHER_CHUNK):
    """``out[a,b] = Gt[kk[a], kk[b], jj[a], jj[b]]``, chunked to bound temporaries."""
    A = jj.size
    out = np.empty((A, A), dtype=np.float64)
    for s in range(0, A, chunk):
        e = min(s + chunk, A)
        sub = Gt[kk[s:e], :, jj[s:e], :]        # (m, K+1, P): advanced axes 0 and 2
        out[s:e] = sub[:, kk, jj]
    return out


def _gather_xty(Gt, jj, kk, i):
    """``X'y_i`` -- literally a slice of the same tensor: ``Gt[k, 0, j, i]``."""
    return Gt[kk, 0, jj, i]


# --------------------------------------------------------------------------- #
#  Exact group-lasso block solve
# --------------------------------------------------------------------------- #
def _block_solve(evals, evecs, c, lam_w, mu0=0.0):
    """Exact minimiser of ``0.5 b'G b - c'b + lam_w ||b||_2`` for one lag group.

    ``evals, evecs = np.linalg.eigh(G)`` are cached by the caller, so every step of the
    scalar solve is O(K).  Returns ``(None, 0.0)`` when the block is zero -- the KKT
    zero test, done on ``||c||`` BEFORE the eigen-rotation so the (overwhelmingly
    common) zero case costs one dot product.

    The scalar equation is ``h(mu) = sum (mu u_m/(w_m+mu))^2 - lam_w^2 = 0``, strictly
    increasing with ``h(0+) < 0 < h(inf)``: a guaranteed bracket.  Plain bisection needs
    ~50 O(K) steps; we run NEWTON (``h'(mu) = 2 mu sum u^2 w/(w+mu)^3``) safeguarded by
    that bracket -- it lands in 3-6 steps and can never leave the bracket, so it keeps
    bisection's guarantee at a fraction of the cost.  ``mu0`` warm-starts from the
    previous solve of the same block; the closed form for equal eigenvalues seeds the
    cold start.  Returns ``(b, mu)`` so the caller can cache the root.
    """
    nc2 = float(c @ c)
    lam2 = lam_w * lam_w
    # A relative slack on the zero test. The working-set screen compares
    # ``||c||/w_g`` to ``lam`` (a cumsum + sqrt + divide) while this compares
    # ``c'c`` to ``(lam w_g)^2`` (a dot): algebraically identical, but they round
    # differently, so a group sitting exactly ON the boundary (every group at
    # lambda_max, by construction) could be screened out by one test and admitted by
    # the other. Rounding toward zero at the boundary is both the numerically correct
    # side and what makes the working-set and dense paths agree bit for bit.
    if nc2 <= lam2 * (1.0 + _ZERO_REL):
        return None, 0.0
    u = evecs.T @ c
    w = evals
    u2 = u * u
    u2w = u2 * w
    lo, hi = 0.0, math.inf
    if mu0 > 0.0:
        mu = mu0
    else:
        # exact root when all eigenvalues coincide -- an excellent seed in general
        w0 = float(w.mean())
        nu = math.sqrt(nc2)
        mu = lam_w * w0 / max(nu - lam_w, _EPS)
    if not (mu > 0.0) or not math.isfinite(mu):
        mu = 1.0
    for _ in range(_NEWTON_MAX):
        d = w + mu
        d2 = d * d
        h = mu * mu * float(u2 @ (1.0 / d2)) - lam2
        if h > 0.0:
            hi = mu
        else:
            lo = mu
        if abs(h) <= _NEWTON_TOL * lam2:
            break
        hp = 2.0 * mu * float(u2w @ (1.0 / (d2 * d)))
        nxt = mu - h / hp if hp > 0.0 else math.inf
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi) if math.isfinite(hi) else 2.0 * mu
        if nxt == mu:
            break
        mu = nxt
    return evecs @ (u / (w + mu)), mu


def _block_solve_sgl(G, c, lam_w, lam1, b_init, lmax, tol=1e-13, max_iter=1000):
    """Sparse-group block solve (``lam1 > 0``) by FISTA -- no closed fixed point.

    Minimises ``0.5 b'G b - c'b + lam_w||b||_2 + lam1||b||_1``.  Used only when the
    optional extra L1 on the deviations is switched on; the exact bisection above
    handles the (default) ``lam1 == 0`` case.
    """
    st = np.sign(c) * np.maximum(np.abs(c) - lam1, 0.0)
    if math.sqrt(float(st @ st)) <= lam_w:
        return None, 0.0
    L = max(float(lmax), _EPS)
    b = b_init.copy()
    z = b.copy()
    t = 1.0
    for _ in range(max_iter):
        v = z - (G @ z - c) / L
        v = np.sign(v) * np.maximum(np.abs(v) - lam1 / L, 0.0)
        nv = math.sqrt(float(v @ v))
        thr = lam_w / L
        bn = v * (1.0 - thr / nv) if nv > thr else np.zeros_like(v)
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = bn + ((t - 1.0) / tn) * (bn - b)
        d = float(np.max(np.abs(bn - b)))
        b = bn
        t = tn
        if d <= tol:
            break
    return (None if not np.any(b) else b), 0.0


# --------------------------------------------------------------------------- #
#  Candidate bookkeeping
# --------------------------------------------------------------------------- #
class _Candidates:
    """One target's candidate columns laid out group-by-group (group = one source j)."""

    __slots__ = ("jj", "kk", "gstart", "gsrc", "wg", "A", "n_groups")

    def __init__(self, pairs):
        pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
        if pairs.size:
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            keep = np.ones(pairs.shape[0], dtype=bool)
            if pairs.shape[0] > 1:
                keep[1:] = np.any(pairs[1:] != pairs[:-1], axis=1)
            pairs = pairs[keep]
            self.jj = np.ascontiguousarray(pairs[:, 0])
            self.kk = np.ascontiguousarray(pairs[:, 1])
        else:
            self.jj = np.zeros(0, dtype=np.intp)
            self.kk = np.zeros(0, dtype=np.intp)
        self.A = self.jj.size
        if self.A:
            newg = np.ones(self.A, dtype=bool)
            newg[1:] = self.jj[1:] != self.jj[:-1]
            starts = np.flatnonzero(newg)
            self.gstart = np.concatenate([starts, [self.A]]).astype(np.intp)
            self.gsrc = self.jj[starts]
        else:
            self.gstart = np.zeros(1, dtype=np.intp)
            self.gsrc = np.zeros(0, dtype=np.intp)
        self.n_groups = self.gsrc.size
        sizes = np.diff(self.gstart).astype(np.float64)
        self.wg = np.sqrt(sizes) if sizes.size else np.zeros(0)

    def key(self):
        return (self.jj.tobytes(), self.kk.tobytes())


def _normalise_candidates(candidates, P, K):
    """Accept None / dict / (K,P,P) or (P,P,K) bool mask / sequence -> _Candidates."""
    if candidates is None:
        base = np.stack(np.meshgrid(np.arange(P), np.arange(1, K + 1), indexing="ij"),
                        axis=-1).reshape(-1, 2)
        c = _Candidates(base)
        return [c] * P
    if isinstance(candidates, np.ndarray) and candidates.dtype == bool:
        out = []
        if candidates.shape == (K, P, P):               # [k-1, i, j]
            for i in range(P):
                kidx, jidx = np.nonzero(candidates[:, i, :])
                out.append(_Candidates(np.column_stack([jidx, kidx + 1])))
            return out
        if candidates.shape == (P, P, K):               # [i, j, k-1]
            for i in range(P):
                jidx, kidx = np.nonzero(candidates[i])
                out.append(_Candidates(np.column_stack([jidx, kidx + 1])))
            return out
        raise ValueError("bool candidate mask must be (K,P,P) or (P,P,K)")
    if isinstance(candidates, dict):
        return [_Candidates(np.asarray(list(candidates.get(i, ())),
                                       dtype=np.intp).reshape(-1, 2))
                for i in range(P)]
    if isinstance(candidates, (list, tuple)):
        if len(candidates) != P:
            raise ValueError("candidate sequence must have one entry per target")
        return [_Candidates(np.asarray(list(pr), dtype=np.intp).reshape(-1, 2))
                for pr in candidates]
    raise TypeError("unsupported `candidates` type: %r" % type(candidates))


# --------------------------------------------------------------------------- #
#  Batched residual <-> statistic exchange (one GEMM per level, never per column)
# --------------------------------------------------------------------------- #
def _shift_stack(v, T, K, buf=None):
    """(n_ent, n_rows) -> (n_ent, T, K+1) with ``Rs[:, K-k:T-k, k] = v``.

    ``Z^T Rs`` then gives ``sum_t Z[e, K+t-k, j] v[e,t]`` for every (j, k) at once.
    """
    if buf is None:
        buf = np.zeros((v.shape[0], T, K + 1), dtype=np.float64)
    else:
        buf[:] = 0.0
    for k in range(K + 1):
        buf[:, K - k:T - k, k] = v
    return buf


def _fitted_shared(Z, coef, K):
    """``fit[e,t] = sum_{k,j} coef[k,j] Z[e, K+t-k, j]`` -- one batched GEMV per lag."""
    n_ent, T, _ = Z.shape
    out = np.zeros((n_ent, T - K), dtype=np.float64)
    for k in range(1, K + 1):
        ck = coef[k]
        if np.any(ck):
            out += Z[:, K - k:T - k, :] @ ck
    return out


def _fitted_per_entity(Z, Cd, K, out=None):
    """``fit[e,t] = sum_{k,j} Cd[e,k,j] Z[e, K+t-k, j]`` -- one BATCHED GEMV per lag.

    ``Cd`` is (n_ent, K+1, P). Entity-specific coefficients would naively need a Python
    loop over entities; stacking them into one array turns the whole deviation fit into
    ``K`` batched matmuls, which is what keeps the two-level alternation cheap.
    """
    n_ent, T, _ = Z.shape
    if out is None:
        out = np.zeros((n_ent, T - K), dtype=np.float64)
    else:
        out[:] = 0.0
    for k in range(1, K + 1):
        ck = Cd[:, k, :]
        if np.any(ck):
            out += np.matmul(Z[:, K - k:T - k, :], ck[:, :, None])[:, :, 0]
    return out


def _dev_coef_tensor(d, cand, inv_scale, buf):
    """Scatter the sparse deviation vector into a dense (n_ent, K+1, P) coef tensor."""
    buf[:] = 0.0
    ee, aa = np.nonzero(d)
    if ee.size:
        buf[ee, cand.kk[aa], cand.jj[aa]] = d[ee, aa] * inv_scale[aa]
    return buf, ee.size


def _kkt_violates(cnorm, wg, lam):
    """``||X_g'r|| > lam * w_g`` -- the group-lasso KKT violation test.

    Written as a SQUARED comparison with exactly the slack ``_block_solve`` uses, so
    the screen and the block solve can never disagree about a group sitting on the
    boundary. If they could, a group could be admitted to the working set, solved back
    to zero, re-flagged as violating, and the certification loop would spin forever --
    and at large ``lam`` an absolute tolerance ``lam + tol`` is not enough separation
    to prevent it. Squaring removes the sqrt/divide entirely and makes the two tests
    the same arithmetic.
    """
    lw = lam * wg
    return cnorm * cnorm > lw * lw * (1.0 + _ZERO_REL)


def _group_norms(v, gstart):
    """``||v_g||_2`` for every group along the last axis -- a segment sum, no loop."""
    sq = np.cumsum(np.concatenate(
        [np.zeros(v.shape[:-1] + (1,), dtype=np.float64), v * v], axis=-1), axis=-1)
    return np.sqrt(np.maximum(sq[..., gstart[1:]] - sq[..., gstart[:-1]], 0.0))


# --------------------------------------------------------------------------- #
#  Level 1: pooled shared skeleton b0, working-set BCD
# --------------------------------------------------------------------------- #
class _L1State:
    """Warm-startable state for one target's shared-level group lasso."""

    __slots__ = ("b", "Gb", "inW", "eig", "gram", "mu")

    def __init__(self, A, n_groups):
        self.b = np.zeros(A, dtype=np.float64)
        self.Gb = np.zeros(A, dtype=np.float64)
        self.inW = np.zeros(n_groups, dtype=bool)
        self.mu = np.zeros(n_groups, dtype=np.float64)   # warm-started scalar roots
        self.eig: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.gram: Dict[int, np.ndarray] = {}


def _ensure_block(G, st, g, s, e):
    blk = st.gram.get(g)
    if blk is None:
        blk = np.ascontiguousarray(G[s:e, s:e])
        blk = 0.5 * (blk + blk.T)
        st.gram[g] = blk
        w, Q = np.linalg.eigh(blk)
        st.eig[g] = (np.maximum(w, 0.0), Q)
    return blk, st.eig[g]


def _bcd_level1(G, c0, cand, lam, st, tol, max_iter, order):
    """Block CD over the working set.  Returns the max coefficient move.

    Two-phase, glmnet style: ONE full pass over the working set, then convergence on
    the currently-nonzero blocks alone, repeated until a full pass moves nothing.  A
    block that is zero and stays zero must not be re-tested on every inner sweep --
    doing so was 2.1M redundant block solves on a 20-series panel, and the zero blocks
    are the overwhelming majority by design.  Exact: the full pass is the KKT check
    over W, and the caller's global sweep covers everything outside it.
    """
    if order.size == 0:
        return 0.0
    gstart, wg, b = cand.gstart, cand.wg, st.b

    def sweep(groups):
        maxd = 0.0
        for g in groups:
            g = int(g)
            s, e = int(gstart[g]), int(gstart[g + 1])
            blk, (w, Q) = _ensure_block(G, st, g, s, e)
            bg = b[s:e]
            cg = c0[s:e] - st.Gb[s:e] + blk @ bg
            nb, st.mu[g] = _block_solve(w, Q, cg, lam * wg[g], st.mu[g])
            if nb is None:
                if not np.any(bg):
                    continue
                delta = -bg.copy()
                b[s:e] = 0.0
            else:
                delta = nb - bg
                if not np.any(delta):
                    continue
                b[s:e] = nb
            st.Gb += G[:, s:e] @ delta
            dm = float(np.max(np.abs(delta)))
            if dm > maxd:
                maxd = dm
        return maxd

    worst = 0.0
    for _ in range(max_iter):
        d_full = sweep(order)
        worst = max(worst, d_full)
        act = np.flatnonzero(_group_norms(b, gstart) > 0.0)
        for _ in range(max_iter):
            if act.size == 0:
                break
            d = sweep(act)
            worst = max(worst, d)
            if d <= tol:
                break
            act = np.flatnonzero(_group_norms(b, gstart) > 0.0)
        if d_full <= tol:
            break
    return worst


def _spectral_norms(G, cand):
    """``||X_g||_2`` for every group, in ONE batched ``eigh`` when sizes are uniform."""
    sizes = np.diff(cand.gstart)
    ng = cand.n_groups
    if ng == 0:
        return np.zeros(0)
    if np.all(sizes == sizes[0]):
        m = int(sizes[0])
        blocks = G.reshape(ng, m, ng, m)[np.arange(ng), :, np.arange(ng), :]
        w = np.linalg.eigvalsh(0.5 * (blocks + blocks.transpose(0, 2, 1)))
        return np.sqrt(np.maximum(w[:, -1], 0.0))
    out = np.empty(ng, dtype=np.float64)
    for g in range(ng):
        s, e = int(cand.gstart[g]), int(cand.gstart[g + 1])
        w = np.linalg.eigvalsh(0.5 * (G[s:e, s:e] + G[s:e, s:e].T))
        out[g] = math.sqrt(max(float(w[-1]), 0.0))
    return out


def _gapsafe_reject(cand, cnorm, lam, gap, gnorm):
    """GAP-safe sphere test: groups that provably have ``beta_g = 0`` at this lambda.

    ``theta = r / max(lam, max_g ||X_g'r||/w_g)`` is dual feasible; with radius
    ``r_gap = sqrt(2*gap)/lam`` a group is certified zero when
    ``||X_g'theta|| + r_gap ||X_g||_2 < w_g``.  Only ever *removes* work -- the global
    KKT sweep still certifies the returned solution.
    """
    if gap <= 0.0 or lam <= 0.0 or cand.n_groups == 0:
        return None
    alpha = max(lam, float(np.max(cnorm / cand.wg)))
    r_gap = math.sqrt(2.0 * gap) / lam
    return (cnorm / alpha + r_gap * gnorm) < cand.wg


# --------------------------------------------------------------------------- #
#  Level 2: per-entity deviations from residual streams
# --------------------------------------------------------------------------- #
class _L2State:
    __slots__ = ("d", "inW", "mu")

    def __init__(self, n_ent, A, n_groups):
        self.d = np.zeros((n_ent, A), dtype=np.float64)
        self.inW = np.zeros((n_ent, n_groups), dtype=bool)
        self.mu = np.zeros((n_ent, n_groups), dtype=np.float64)


class _DevWorkspace:
    """Per-target scratch for the deviation level: residual streams + block cache.

    Blocks are pre-multiplied by the target's validity mask, so the per-entity Gram,
    the correlation ``X_g' r_e`` and the incremental residual update are all restricted
    to the rows where ``y`` is observed -- automatically and consistently.  The Gram
    and the correlation get the same available-case ``n/N`` rescale as the pooled
    tensor, so the two levels agree on what a coefficient means.

    Everything is cached and computed BATCHED OVER ENTITIES: one ``(n_ent, n_rows, m)``
    stack per group, one batched ``eigh`` for all ``n_ent`` Grams at once.  The
    deviation level is exactly separable in ``e`` (``r_e`` depends only on ``d_e``), so
    sweeping every entity's copy of a group together is the SAME Gauss-Seidel as one at
    a time -- it just replaces ``n_ent`` tiny numpy calls with one batched call, which
    is where this level's time actually went.
    """

    def __init__(self, Z, Vf, vy, scale, cand, K, available_case):
        self.Z = Z
        self.Vf = Vf
        self.vy = vy                       # (n_ent, n_rows) float 0/1, y observed
        self.scale = scale
        self.cand = cand
        self.K = K
        self.available_case = available_case
        self.n_ent, self.T, _ = Z.shape
        self.n_rows = self.T - K
        self.rows_e = np.maximum(vy.sum(1), 1.0)
        self.groups: Dict[int, Any] = {}
        self._colcnt = None

    def colcounts(self):
        """(n_ent, A) co-observation counts of each candidate column with the target."""
        if self._colcnt is None:
            cand, K, T = self.cand, self.K, self.T
            out = np.empty((self.n_ent, cand.A), dtype=np.float64)
            for a in range(cand.A):
                k, j = int(cand.kk[a]), int(cand.jj[a])
                out[:, a] = (self.Vf[:, K - k:T - k, j] * self.vy).sum(1)
            self._colcnt = np.maximum(out, 1.0)
        return self._colcnt

    def group(self, g):
        """``(Xg, Gg, w, Q, cfac, s, e)`` for group ``g``, stacked over entities."""
        hit = self.groups.get(g)
        if hit is not None:
            return hit
        cand, K, T = self.cand, self.K, self.T
        s, en = int(cand.gstart[g]), int(cand.gstart[g + 1])
        j = int(cand.gsrc[g])
        lags = cand.kk[s:en]
        m = lags.size
        Xg = np.empty((self.n_ent, self.n_rows, m), dtype=np.float64)
        Vb = np.empty_like(Xg)
        for c, k in enumerate(lags):
            k = int(k)
            Xg[:, :, c] = self.Z[:, K - k:T - k, j] * self.vy / self.scale[k, j]
            Vb[:, :, c] = self.Vf[:, K - k:T - k, j] * self.vy
        Xt = Xg.transpose(0, 2, 1)
        Gg = np.matmul(Xt, Xg)                                # (n_ent, m, m) batched
        if self.available_case:
            cnt = np.matmul(Vb.transpose(0, 2, 1), Vb)
            Gg = Gg * (self.rows_e[:, None, None] / np.maximum(cnt, 1.0))
            cfac = self.rows_e[:, None] / self.colcounts()[:, s:en]
        else:
            cfac = None
        Gg = 0.5 * (Gg + Gg.transpose(0, 2, 1))
        w, Q = np.linalg.eigh(Gg)                             # ONE batched LAPACK call
        hit = (Xg, Xt, Gg, np.maximum(w, 0.0), Q, cfac, s, en)
        self.groups[g] = hit
        return hit


def _bcd_level2(ws, st2, cand, r, lam1, lam2, tol, max_iter, inv_scale, Zt, T,
                shift_buf, dense):
    """Working-set block CD on the per-entity deviations, from residual streams only.

    Returns the largest coefficient move seen (0.0 => already at the KKT point).
    """
    K = ws.K
    worst = 0.0
    cfac_all = (ws.rows_e[:, None] / ws.colcounts()) if ws.available_case else 1.0
    gstart, wg = cand.gstart, cand.wg

    def sweep(groups):
        maxd = 0.0
        for g in groups:
            g = int(g)
            Xg, Xt, Gg, w, Q, cfac, s, en = ws.group(g)
            D = st2.d[:, s:en].copy()
            # c_g = X_g' r_e + G_g d_g, for every entity in ONE batched pair of GEMMs
            C = np.matmul(Xt, r[:, :, None])[:, :, 0]
            if cfac is not None:
                C *= cfac
            C += np.matmul(Gg, D[:, :, None])[:, :, 0]
            lam_w = lam1 * wg[g]
            lw2 = lam_w * lam_w * (1.0 + _ZERO_REL)
            # vectorised zero test over entities: only the survivors need a solve
            live = np.flatnonzero(np.einsum("em,em->e", C, C) > lw2)
            keep = np.zeros(D.shape[0], dtype=bool)
            keep[live] = True
            touched = np.flatnonzero(keep | np.any(D != 0.0, axis=1))
            if touched.size == 0:
                continue
            new = np.zeros_like(D)
            for e in touched:
                e = int(e)
                if not keep[e]:
                    continue
                if lam2 > 0.0:
                    nb, mu = _block_solve_sgl(Gg[e], C[e], lam_w, lam2, D[e],
                                              w[e, -1] if w.shape[1] else 1.0)
                else:
                    nb, mu = _block_solve(w[e], Q[e], C[e], lam_w, st2.mu[e, g])
                st2.mu[e, g] = mu
                if nb is not None:
                    new[e] = nb
            delta = new[touched] - D[touched]
            if not np.any(delta):
                continue
            st2.d[touched, s:en] = new[touched]
            r[touched] -= np.matmul(Xg[touched], delta[:, :, None])[:, :, 0]
            dm = float(np.max(np.abs(delta)))
            if dm > maxd:
                maxd = dm
        return maxd

    for _round in range(_L2_KKT_ROUNDS):
        # global KKT over every (entity, group): ONE batched GEMM, then a vectorised
        # group-norm test.  X_g'r_e for all (e, j, k) at once.
        C = np.matmul(Zt, _shift_stack(r, T, K, shift_buf))        # (n_ent, P, K+1)
        cc = C[:, cand.jj, cand.kk] * inv_scale * cfac_all         # (n_ent, A)
        gn = _group_norms(cc, gstart)                              # (n_ent, n_groups)
        nz = _group_norms(st2.d, gstart) > 0.0
        # Descend only on blocks that are already nonzero plus the ones this KKT round
        # just certified as violating.  Blocks that are zero and not violating are
        # already at their optimum given the current residual; the next KKT round
        # re-checks every one of them, so nothing is missed and the (dominant) zero
        # blocks stop paying for a matvec per inner sweep.
        bad = (~nz) if dense else (_kkt_violates(gn, wg, lam1) & (~nz))
        if _round > 0 and not bad.any():
            break
        st2.inW |= bad
        order = np.flatnonzero((nz | bad).any(axis=0))
        if order.size == 0:
            break
        d_full = sweep(order)
        worst = max(worst, d_full)
        for _ in range(max_iter):                 # converge on the nonzero set alone
            act = np.flatnonzero((_group_norms(st2.d, gstart) > 0.0).any(axis=0))
            if act.size == 0:
                break
            d = sweep(act)
            worst = max(worst, d)
            if d <= tol:
                break
        if dense and d_full <= tol:
            break
    return worst


# --------------------------------------------------------------------------- #
#  Sparse solution container
# --------------------------------------------------------------------------- #
@dataclass
class _Sol:
    """One lambda's solution in COO form, so a 40-point path stays small.

    Values are the *scale-folded standardised* coefficients ``Bsc = b_std / scale[k,j]``
    (multiply by ``scale[0, i]`` for raw units).
    """
    b0_k: np.ndarray
    b0_i: np.ndarray
    b0_j: np.ndarray
    b0_v: np.ndarray
    dv_e: np.ndarray
    dv_k: np.ndarray
    dv_i: np.ndarray
    dv_j: np.ndarray
    dv_v: np.ndarray
    df: int = 0

    def bsc(self, K, P):
        out = np.zeros((K + 1, P, P), dtype=np.float64)
        if self.b0_v.size:
            out[self.b0_k, self.b0_i, self.b0_j] = self.b0_v
        return out

    def dev_bsc(self, K, P):
        if self.dv_v.size == 0:
            return {}
        out: Dict[int, np.ndarray] = {}
        for e in np.unique(self.dv_e):
            m = self.dv_e == e
            arr = np.zeros((K + 1, P, P), dtype=np.float64)
            arr[self.dv_k[m], self.dv_i[m], self.dv_j[m]] = self.dv_v[m]
            out[int(e)] = arr
        return out


# --------------------------------------------------------------------------- #
#  Result
# --------------------------------------------------------------------------- #
@dataclass
class SkeletonResult:
    B0: np.ndarray                       # (K, P, P) shared coefficients, raw units
    Delta: Dict[int, np.ndarray]         # {eid: (K, P, P)} sparse entity deviations
    support: np.ndarray                  # (K, P, P) bool -- shared OR any deviation
    lambda_path: np.ndarray
    lambda_selected: float
    path_support: List[np.ndarray]
    n_kkt_sweeps: int
    meta: Dict[str, Any]

    def edges(self):
        """Recovered ``(lag, target, source)`` triples of the skeleton."""
        k, i, j = np.nonzero(self.support)
        return np.column_stack([k + 1, i, j])

    def skeleton(self):
        """The (P, P) group-level skeleton: ``[i, j]`` is "j Granger-causes i".

        Because the penalty groups over lags, a source enters or leaves a target as a
        UNIT -- so this, not the per-lag support, is the object the discovery stage
        consumes and the right granularity at which to score recovery.
        """
        return self.support.any(axis=0)


def _log_comb(n, k):
    if k <= 0 or k >= n:
        return 0.0
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


# --------------------------------------------------------------------------- #
#  Core: fit the whole lambda path on a prefix
# --------------------------------------------------------------------------- #
def _fit_path(E, obs, t_hi, cand_list, cfg, lam_path=None):
    """Fit the whole lambda path using ONLY rows ``[0, t_hi)`` of ``E``.

    Truncating the input at ``t_hi`` is the very first thing this does and nothing
    downstream ever sees the full array, so the result is bit-identical to passing an
    already-truncated panel.  That is the no-look-ahead certificate the forward-
    chaining selector rests on.
    """
    E = np.asarray(E, dtype=np.float64)[:, :t_hi, :]
    obs_c = None if obs is None else np.asarray(obs, dtype=bool)[:, :t_hi, :]
    K = cfg["max_lag"]
    P = E.shape[2]

    mt = build_moment_tensor(E, obs_c, max_lag=K, demean=cfg["demean"])
    Gt = mt.scaled(available_case=cfg["available_case"])
    Ncnt, scale, Z, Vf = mt.N, mt.scale, mt.Z, mt.V
    n_tot, n_rows, n_ent, Tc = mt.n_tot, mt.n_rows, mt.n_ent, mt.T

    shared_key = cand_list[0].key()
    shared = all(c.key() == shared_key for c in cand_list)
    G_shared = (_gather_gram(Gt, cand_list[0].jj, cand_list[0].kk)
                if shared and cand_list[0].A else None)
    gnorm_shared = None

    if lam_path is None:
        lam_max = _EPS
        for i in range(P):
            cand = cand_list[i]
            if cand.A == 0:
                continue
            gn = _group_norms(_gather_xty(Gt, cand.jj, cand.kk, i),
                              cand.gstart) / cand.wg
            if gn.size:
                lam_max = max(lam_max, float(gn.max()))
        L = int(cfg["n_lambda"])
        lam_path = (np.array([lam_max]) if L == 1 else
                    lam_max * cfg["lambda_ratio"] ** (np.arange(L) / (L - 1.0)))
    L = lam_path.size

    acc = [{"b0": [], "dev": [], "df": 0, "rss": 0.0, "n": 0.0, "ebic": 0.0}
           for _ in range(L)]
    n_sweeps = 0
    kappa, lam2, tol = cfg["kappa"], cfg["lam2"], cfg["tol"]
    max_iter, max_gs = cfg["max_iter"], cfg["max_gs"]
    fit_dev = cfg["fit_deviations"] and n_ent > 1
    dense = not cfg["use_working_set"]
    gapsafe = cfg["screening"] == "gapsafe" and not dense

    Zt = Z.transpose(0, 2, 1)
    shift_buf = np.zeros((n_ent, Tc, K + 1), dtype=np.float64)
    cbuf = np.zeros((K + 1, P), dtype=np.float64)
    dbuf = (np.zeros((n_ent, K + 1, P), dtype=np.float64)
            if cfg["fit_deviations"] and n_ent > 1 else None)
    fbuf = (np.zeros((n_ent, n_rows), dtype=np.float64)
            if cfg["fit_deviations"] and n_ent > 1 else None)

    for i in range(P):
        cand = cand_list[i]
        if cand.A == 0:
            continue
        G = G_shared if shared else _gather_gram(Gt, cand.jj, cand.kk)
        xty = _gather_xty(Gt, cand.jj, cand.kk, i)
        nfac = (n_tot / np.maximum(Ncnt[cand.kk, 0, cand.jj, i], 1.0)
                if cfg["available_case"] else np.ones(cand.A))
        inv_scale = 1.0 / scale[cand.kk, cand.jj]
        yy = float(Gt[0, 0, i, i])                 # == n_tot exactly by construction
        if not gapsafe:
            gnorm = None
        elif shared:
            if gnorm_shared is None:
                gnorm_shared = _spectral_norms(G, cand)
            gnorm = gnorm_shared
        else:
            gnorm = _spectral_norms(G, cand)

        st1 = _L1State(cand.A, cand.n_groups)
        if dense:
            st1.inW[:] = True
        st2 = ws = ystd = None
        dev_lm = 0.0
        vy = Vf[:, K:, i]
        if fit_dev:
            st2 = _L2State(n_ent, cand.A, cand.n_groups)
            ws = _DevWorkspace(Z, Vf, vy, scale, cand, K, cfg["available_case"])
            ystd = Z[:, K:, i] * (vy / scale[0, i])
            # lam1 above this and d == 0 is PROVABLY optimal -- skips the top of the
            # path (where b0 == 0) without a single deviation sweep.
            C0 = np.matmul(Zt, _shift_stack(ystd * vy, Tc, K, shift_buf))
            cc0 = C0[:, cand.jj, cand.kk] * inv_scale
            if cfg["available_case"]:
                cc0 = cc0 * (ws.rows_e[:, None] / ws.colcounts())
            g0 = _group_norms(cc0, cand.gstart) / cand.wg
            dev_lm = float(g0.max()) if g0.size else 0.0

        for l in range(L):
            lam = float(lam_path[l])
            lam1 = kappa * lam
            r = None

            for _gs in range(max_gs):
                # ---------- (1) shared level b0, d fixed ---------------------- #
                if fit_dev and np.any(st2.d):
                    Cd, _ = _dev_coef_tensor(st2.d, cand, inv_scale, dbuf)
                    fdev = _fitted_per_entity(Z, Cd, K, fbuf)
                    fdev *= vy
                    Xtf = np.matmul(Zt, _shift_stack(fdev, Tc, K, shift_buf)).sum(0)
                    c0 = xty - Xtf[cand.jj, cand.kk] * inv_scale * nfac
                else:
                    c0 = xty

                st1.Gb[:] = G @ st1.b if np.any(st1.b) else 0.0
                cfull = c0 - st1.Gb
                cnorm = _group_norms(cfull, cand.gstart)
                if not dense:
                    supp = _group_norms(st1.b, cand.gstart) > 0.0
                    st1.inW |= supp
                    viol = cnorm / cand.wg - lam
                    p_new = max(10, 2 * int(supp.sum()))
                    fresh = np.flatnonzero((~st1.inW)
                                           & _kkt_violates(cnorm, cand.wg, lam))
                    if fresh.size > p_new:
                        fresh = fresh[np.argpartition(-viol[fresh],
                                                      p_new - 1)[:p_new]]
                    st1.inW[fresh] = True

                for _cert in range(cand.n_groups + 8):
                    n_sweeps += 1
                    order = np.flatnonzero(st1.inW)
                    _bcd_level1(G, c0, cand, lam, st1, tol, max_iter, order)
                    cfull = c0 - st1.Gb
                    cnorm = _group_norms(cfull, cand.gstart)
                    bzero = _group_norms(st1.b, cand.gstart) == 0.0
                    bad = np.flatnonzero(bzero & _kkt_violates(cnorm, cand.wg, lam))
                    if bad.size and gapsafe and not (fit_dev and np.any(st2.d)):
                        rss_h = max(yy - 2.0 * float(st1.b @ c0)
                                    + float(st1.b @ st1.Gb), 0.0)
                        pen = lam * float(_group_norms(st1.b, cand.gstart) @ cand.wg)
                        alpha = max(lam, float(np.max(cnorm / cand.wg)))
                        th = 1.0 / alpha
                        dual = lam * th * (yy - float(st1.b @ c0)) \
                            - 0.5 * (lam * th) ** 2 * rss_h
                        rej = _gapsafe_reject(cand, cnorm, lam,
                                              max(0.5 * rss_h + pen - dual, 0.0),
                                              gnorm)
                        if rej is not None:
                            bad = bad[~rej[bad]]
                    if bad.size == 0:
                        break
                    st1.inW[bad] = True

                if not fit_dev:
                    break

                # ---------- (2) deviation level d_e, b0 fixed ----------------- #
                b_zero = not np.any(st1.b)
                d_zero = not np.any(st2.d)
                if b_zero and d_zero and lam1 >= dev_lm:
                    break                       # provably d == 0 at this lambda
                cbuf[:] = 0.0
                nzb = np.flatnonzero(st1.b)
                if nzb.size:
                    cbuf[cand.kk[nzb], cand.jj[nzb]] = st1.b[nzb] * inv_scale[nzb]
                r = _fitted_shared(Z, cbuf, K)
                if not d_zero:
                    Cd, _ = _dev_coef_tensor(st2.d, cand, inv_scale, dbuf)
                    r += _fitted_per_entity(Z, Cd, K, fbuf)
                np.subtract(ystd, r, out=r)
                r *= vy
                moved = _bcd_level2(ws, st2, cand, r, lam1, lam2, tol,
                                    cfg["max_iter_dev"], inv_scale, Zt, Tc,
                                    shift_buf, dense)
                n_sweeps += 1
                if moved <= tol:
                    break

            # ---------- record this lambda ---------------------------------- #
            nzb = np.flatnonzero(st1.b)
            df = int(nzb.size)
            if nzb.size:
                acc[l]["b0"].append((cand.kk[nzb], np.full(nzb.size, i, np.intp),
                                     cand.jj[nzb], st1.b[nzb] * inv_scale[nzb]))
            has_dev = fit_dev and np.any(st2.d)
            if has_dev:
                ee, aa = np.nonzero(st2.d)
                df += int(ee.size)
                acc[l]["dev"].append((ee, cand.kk[aa],
                                      np.full(ee.size, i, np.intp), cand.jj[aa],
                                      st2.d[ee, aa] * inv_scale[aa]))
            if has_dev and r is not None:
                rss = float(np.sum(r * r))
            else:
                rss = max(yy - 2.0 * float(st1.b @ xty)
                          + float(st1.b @ (G @ st1.b)), 0.0)
            n_i = max(float(vy.sum()), 1.0)
            acc[l]["df"] += df
            acc[l]["rss"] += rss
            acc[l]["n"] += n_i
            acc[l]["ebic"] += (n_i * math.log(max(rss, _EPS) / n_i + _EPS)
                               + df * math.log(n_i)
                               + 2.0 * _EBIC_GAMMA * _log_comb(float(cand.A),
                                                               float(df)))

    prep = {"Z": Z, "V": Vf, "scale": scale, "K": K, "T": Tc, "n_ent": n_ent,
            "P": P, "n_tot": n_tot, "moment_bytes": int(Gt.nbytes)}
    out = []
    for l in range(L):
        s = acc[l]
        z_i, z_f = np.zeros(0, dtype=np.intp), np.zeros(0)
        if s["b0"]:
            bk, bi, bj, bv = (np.concatenate([x[c] for x in s["b0"]])
                              for c in range(4))
        else:
            bk = bi = bj = z_i
            bv = z_f
        if s["dev"]:
            de, dk, di, dj, dvv = (np.concatenate([x[c] for x in s["dev"]])
                                   for c in range(5))
        else:
            de = dk = di = dj = z_i
            dvv = z_f
        out.append((_Sol(bk, bi, bj, bv, de, dk, di, dj, dvv, s["df"]),
                    s["rss"], s["n"], s["ebic"]))
    return lam_path, out, prep, n_sweeps


# --------------------------------------------------------------------------- #
#  Validation scoring on a held-out block (strictly after the training prefix)
# --------------------------------------------------------------------------- #
def _score_block(E, obs, prep, sol, t_lo, t_hi, cfg):
    """Standardised SSE of ``sol`` on target rows ``[t_lo, t_hi)``.

    The block is demeaned with the TRAINING prefix's entity means plus its own
    contemporaneous cross-sectional means, and standardised with the TRAINING scales:
    nothing inside the validation window informs the model or its normalisation.
    """
    K, P, scale, tr = prep["K"], prep["P"], prep["scale"], prep["Ttr"]
    E = np.asarray(E, dtype=np.float64)[:, :t_hi, :]
    obs_c = None if obs is None else np.asarray(obs, dtype=bool)[:, :t_hi, :]
    V = np.isfinite(E)
    if obs_c is not None:
        V &= obs_c
    Vf = V.astype(np.float64)
    Xz = np.where(V, E, 0.0)
    if cfg["demean"] == "none":
        Zb = Xz
    else:
        ent_s = Xz[:, :tr, :].sum(1, keepdims=True)
        ent_c = Vf[:, :tr, :].sum(1, keepdims=True)
        Zb = (Xz - ent_s / np.maximum(ent_c, 1.0)) * Vf     # re-mask, as in training
        if cfg["demean"] == "twoway" and Zb.shape[0] >= 2:
            Zb = Zb - Zb.sum(0, keepdims=True) / np.maximum(Vf.sum(0, keepdims=True),
                                                            1.0)
    Zb = Zb * Vf
    T = Zb.shape[1]
    lo = max(t_lo, K)
    if lo >= T:
        return 0.0, 0.0
    Bsc = sol.bsc(K, P)
    dev = sol.dev_bsc(K, P)
    Yv = Zb[:, lo:T, :] / scale[0][None, None, :]
    Mv = Vf[:, lo:T, :]
    fit = np.zeros_like(Yv)
    for k in range(1, K + 1):
        Zl = Zb[:, lo - k:T - k, :]
        if np.any(Bsc[k]):
            fit += Zl @ Bsc[k].T
        for e, De in dev.items():
            if np.any(De[k]):
                fit[e] += Zl[e] @ De[k].T
    resid = (Yv - fit) * Mv
    return float(np.sum(resid * resid)), float(Mv.sum())


def _make_cfg(max_lag=8, kappa=4.0, n_lambda=40, lambda_ratio=1e-2, tol=1e-6,
              max_iter=200, fit_deviations=True, lam2=0.0, demean="twoway",
              available_case=True, max_gs=4, max_iter_dev=25, screening="kkt",
              use_working_set=True):
    """The internal solver config, in one place so callers and tests cannot drift."""
    return {"max_lag": int(max_lag), "kappa": float(kappa),
            "n_lambda": int(n_lambda), "lambda_ratio": float(lambda_ratio),
            "tol": float(tol), "max_iter": int(max_iter),
            "fit_deviations": bool(fit_deviations), "lam2": float(lam2),
            "demean": demean, "available_case": bool(available_case),
            "max_gs": int(max_gs), "max_iter_dev": int(max_iter_dev),
            "screening": screening, "use_working_set": bool(use_working_set)}


def _fold_bounds(T, K, n_splits):
    """Expanding-window cut points: train on [0, b_f), score on [b_f, b_{f+1})."""
    if n_splits < 1:
        return None
    lo = max(2 * K + 10, int(0.5 * T))
    if lo >= T - K - 2:
        return None
    b = np.unique(np.linspace(lo, T, n_splits + 1).astype(int))
    return b if b.size >= 2 else None


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def panel_group_lasso(E, candidates=None, obs=None, max_lag=8, kappa=4.0,
                      n_lambda=40, lambda_ratio=1e-2, select="forward",
                      screening="kkt", tol=1e-6, max_iter=200,
                      fit_deviations=True, seed=0,
                      *, select_rule="1se", lam2=0.0, n_splits=3, demean="twoway",
                      available_case=True, lambda_grid=None, max_gs=4,
                      max_iter_dev=25, moment_budget_gb=1.0,
                      use_working_set=True, verbose=False) -> SkeletonResult:
    """Estimate a sparse panel-VAR skeleton with shared structure + entity deviations.

    Parameters
    ----------
    E : (n_entities, T, P) array
        Idiosyncratic residual panel; ``np.nan`` marks unobserved cells.  A (T, P)
        array is promoted to a single-entity panel (deviations are then disabled).
    candidates : None | dict | (K,P,P) bool | (P,P,K) bool | sequence
        Per-target candidate set ``C_i = {(j, k)}`` with ``k`` in ``1..K``.  ``None``
        uses every (source, lag) pair.  A dict maps target index -> iterable of
        ``(j, k)``.  A bool array is indexed ``[k-1, i, j]`` or ``[i, j, k-1]``.
    obs : (n_entities, T, P) bool array, optional
        Extra observation mask, ANDed with ``isfinite(E)``.
    max_lag : int
        ``K``, the number of lags; also the group size (groups run over lags), so
        ``j -> i`` enters or leaves the skeleton as a unit.
    kappa : float
        ``lam1 = kappa * lam0``.  Default 4 -- deviations are expensive on purpose.
    n_lambda, lambda_ratio : int, float
        Geometric path ``lam_max * lambda_ratio ** (l/(L-1))``.
    select : {"forward", "ebic"}
        ``"forward"``: forward-chaining time-series validation (default, causal),
        which minimises predictive risk.  ``"ebic"``: single fit, extended BIC with
        ``gamma = 0.5``, which is selection consistent -- USE THIS WHEN THE SUPPORT IS
        THE DELIVERABLE.  See the module docstring for the measured difference; it is
        large (F1 1.00 vs 0.21 at P=50 on the test fixtures).
        K-fold CV over time is a look-ahead hazard and is deliberately unavailable.
    select_rule : {"1se", "min"}
        Only for ``select="forward"``.  ``"1se"`` (default) takes the sparsest lambda
        within one fold-standard-error of the best validation score; ``"min"`` takes
        the plain argmin.  Predictive CV buys dozens of near-zero false edges for a
        fraction of a percent of MSE -- right for forecasting, wrong for a skeleton --
        so the default selects for structure.  Both use only fold scores, so both are
        equally causal.
    screening : {"kkt", "gapsafe"}
        Working-set screening.  Both return the exact solution; ``"gapsafe"`` adds a
        duality-gap sphere test that discards provably-zero groups earlier (applied
        only while the deviation level is inactive, where the gap is exact).
    tol, max_iter : float, int
        Block-CD tolerance (on the standardised scale, where columns and targets have
        unit RMS) and the max sweeps per working set.
    fit_deviations : bool
        Fit the per-entity deviation level ``d_e`` (ignored when ``n_entities == 1``).
    seed : int
        Recorded in ``meta``; the solver itself is deterministic.
    lam2 : float
        Optional extra L1 on the deviations (sparse-group).  0 keeps the exact block
        solve; ``> 0`` switches those blocks to FISTA.
    n_splits : int
        Forward-chaining validation folds.
    demean : {"twoway", "entity", "none"}
        FWL within transform.  Note the O(1/T) Nickell bias of lagged regressors under
        entity FE (~0.5% at T=200); we state it rather than implement Arellano-Bond.
    available_case : bool
        Pairwise-complete renormalisation of the moment statistics.  Leave ``True`` --
        ``False`` reproduces the biased naive zero-fill regression.
    lambda_grid : array, optional
        Explicit lambda grid, bypassing ``lam_max`` / ``n_lambda``.
    max_gs, max_iter_dev : int
        Outer two-level Gauss-Seidel rounds and inner deviation sweeps.
    moment_budget_gb : float
        Cap on the moment tensor.  Above it the solver falls back to the sub-panel of
        series the candidates reference (exact: the transform and the moments are both
        column-separable).
    use_working_set : bool
        ``False`` runs plain block CD over every candidate group -- the dense reference
        path.  Same solution, much slower; used to certify the working-set machinery.

    Returns
    -------
    SkeletonResult
    """
    t_start = time.perf_counter()
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 2:
        E = E[None]
    if E.ndim != 3:
        raise ValueError("E must be (n_entities, T, P)")
    if obs is not None:
        obs = np.asarray(obs, dtype=bool)
        if obs.ndim == 2:
            obs = obs[None]
    n_ent, T, P = E.shape
    K = int(max_lag)
    if select not in ("forward", "ebic"):
        raise ValueError("select must be 'forward' or 'ebic'; K-fold CV over time is "
                         "a look-ahead hazard and is not offered")
    if screening not in ("kkt", "gapsafe"):
        raise ValueError("screening must be 'kkt' or 'gapsafe'")
    if select_rule not in ("1se", "min"):
        raise ValueError("select_rule must be '1se' or 'min'")
    if demean not in ("twoway", "entity", "none"):
        raise ValueError("demean must be 'twoway', 'entity' or 'none'")

    cand_list = _normalise_candidates(candidates, P, K)

    # ---- moment-tensor memory budget ----------------------------------------- #
    full_bytes = (K + 1) ** 2 * P * P * 8
    budget = int(moment_budget_gb * (1 << 30))
    if full_bytes > budget:
        used = np.array(sorted(set(int(x) for c in cand_list for x in c.gsrc) |
                               set(i for i in range(P) if cand_list[i].A)),
                        dtype=np.intp)
        sub_bytes = (K + 1) ** 2 * max(used.size, 1) ** 2 * 8
        if used.size and used.size < P and sub_bytes <= budget:
            # Column-separable transform + moments => restricting to the referenced
            # sub-panel is EXACT, not an approximation.
            remap = -np.ones(P, dtype=np.intp)
            remap[used] = np.arange(used.size)
            sub_cand = {}
            for pos, i in enumerate(used):
                c = cand_list[i]
                if c.A:
                    sub_cand[pos] = np.column_stack([remap[c.jj], c.kk])
            sub = panel_group_lasso(
                E[:, :, used], sub_cand, None if obs is None else obs[:, :, used],
                max_lag=K, kappa=kappa, n_lambda=n_lambda,
                lambda_ratio=lambda_ratio, select=select, screening=screening,
                tol=tol, max_iter=max_iter, fit_deviations=fit_deviations, seed=seed,
                lam2=lam2, n_splits=n_splits, demean=demean,
                available_case=available_case, lambda_grid=lambda_grid, max_gs=max_gs,
                max_iter_dev=max_iter_dev, moment_budget_gb=moment_budget_gb,
                use_working_set=use_working_set, verbose=verbose)
            return _expand_subset(sub, used, P, K)
        raise MemoryError(
            "moment tensor would need %.2f GB > moment_budget_gb=%.2f GB at K=%d, "
            "P=%d (%d series referenced). Screen harder (smaller candidate sets), "
            "lower max_lag, or raise moment_budget_gb."
            % (full_bytes / (1 << 30), moment_budget_gb, K, P,
               int(used.size) if used is not None else P))

    cfg = _make_cfg(max_lag=K, kappa=kappa, n_lambda=n_lambda,
                    lambda_ratio=lambda_ratio, tol=tol, max_iter=max_iter,
                    fit_deviations=fit_deviations, lam2=lam2, demean=demean,
                    available_case=available_case, max_gs=max_gs,
                    max_iter_dev=max_iter_dev, screening=screening,
                    use_working_set=use_working_set)

    n_kkt = 0
    lam_path = None if lambda_grid is None else np.asarray(lambda_grid, float)
    folds: List[Dict[str, Any]] = []
    score = None
    l_star = None

    if select == "forward":
        bounds = _fold_bounds(T, K, int(n_splits))
        if bounds is None:
            select = "ebic"                     # too short to validate causally
        else:
            per_fold = []
            for f in range(bounds.size - 1):
                cut, nxt = int(bounds[f]), int(bounds[f + 1])
                # the grid itself is derived from the FIRST training prefix, so no
                # part of model selection ever sees the future.
                lam_path, sols, prep, ns = _fit_path(E, obs, cut, cand_list, cfg,
                                                     lam_path)
                n_kkt += ns
                prep["Ttr"] = cut
                row = np.empty(len(sols), dtype=np.float64)
                for l, (sol, _r, _n, _b) in enumerate(sols):
                    ss, cc = _score_block(E, obs, prep, sol, cut, nxt, cfg)
                    row[l] = ss / max(cc, 1.0)
                per_fold.append(row)
                folds.append({"train_hi": cut, "val_hi": nxt})
            fold_scores = np.vstack(per_fold)                  # (n_folds, L)
            score = fold_scores.mean(axis=0)
            l_min = int(np.argmin(score))
            if select_rule == "min" or fold_scores.shape[0] < 2:
                l_star = l_min
            else:
                # ONE-STANDARD-ERROR RULE (over folds).  Predictive CV minimises MSE,
                # which happily buys dozens of near-zero false edges for a fraction of
                # a percent of accuracy -- fine for forecasting, wrong for a SKELETON.
                # Taking the sparsest model within one fold-SE of the minimum keeps the
                # causal protocol untouched (only fold scores are used) while selecting
                # for structure. Pass select_rule="min" for the plain argmin.
                se = fold_scores.std(axis=0, ddof=1) / math.sqrt(fold_scores.shape[0])
                thresh = score[l_min] + se[l_min]
                ok = np.flatnonzero(score[:l_min + 1] <= thresh)
                l_star = int(ok[0]) if ok.size else l_min

    lam_path, sols, prep, ns = _fit_path(E, obs, T, cand_list, cfg, lam_path)
    n_kkt += ns

    if select == "ebic":
        score = np.array([b for (_s, _r, _n, b) in sols], dtype=np.float64)
        l_star = int(np.argmin(score))
    l_star = min(int(l_star), len(sols) - 1)

    scale = prep["scale"]
    path_support = []
    for sol, _r, _n, _b in sols:
        sup = np.zeros((K, P, P), dtype=bool)
        if sol.b0_v.size:
            sup[sol.b0_k - 1, sol.b0_i, sol.b0_j] = True
        if sol.dv_v.size:
            sup[sol.dv_k - 1, sol.dv_i, sol.dv_j] = True
        path_support.append(sup)

    sol = sols[l_star][0]
    B0 = np.zeros((K, P, P), dtype=np.float64)
    if sol.b0_v.size:
        B0[sol.b0_k - 1, sol.b0_i, sol.b0_j] = sol.b0_v * scale[0, sol.b0_i]
    Delta: Dict[int, np.ndarray] = {}
    if sol.dv_v.size:
        for e in np.unique(sol.dv_e):
            m = sol.dv_e == e
            arr = np.zeros((K, P, P), dtype=np.float64)
            arr[sol.dv_k[m] - 1, sol.dv_i[m], sol.dv_j[m]] = \
                sol.dv_v[m] * scale[0, sol.dv_i[m]]
            Delta[int(e)] = arr

    shared_support = np.zeros((K, P, P), dtype=bool)
    if sol.b0_v.size:
        shared_support[sol.b0_k - 1, sol.b0_i, sol.b0_j] = True

    meta = {
        "select": select, "select_rule": select_rule,
        "screening": screening, "kappa": float(kappa),
        "lam2": float(lam2), "n_entities": int(n_ent), "T": int(T), "P": int(P),
        "max_lag": K, "n_rows_per_entity": int(T - K),
        "n_design_rows": int(prep["n_tot"]),
        "moment_bytes": int(prep["moment_bytes"]),
        "moment_mb": prep["moment_bytes"] / 1e6,
        "n_candidates": int(sum(c.A for c in cand_list)),
        "n_groups": int(sum(c.n_groups for c in cand_list)),
        "df": int(sol.df), "n_shared_edges": int(shared_support.sum()),
        "n_deviation_entities": len(Delta),
        "support_shared": shared_support,
        "selection_score": score, "folds": folds, "demean": demean,
        "available_case": bool(available_case),
        "fit_deviations": bool(fit_deviations and n_ent > 1),
        "use_working_set": bool(use_working_set),
        "nickell_bias_order": 1.0 / max(T, 1),
        "seed": int(seed), "seconds": time.perf_counter() - t_start,
    }
    if verbose:
        print("[skeleton] P=%d K=%d lam*=%.4g edges=%d sweeps=%d %.2fs moment=%.1fMB"
              % (P, K, lam_path[l_star], int(path_support[l_star].sum()), n_kkt,
                 meta["seconds"], meta["moment_mb"]))
    return SkeletonResult(B0=B0, Delta=Delta, support=path_support[l_star],
                          lambda_path=lam_path,
                          lambda_selected=float(lam_path[l_star]),
                          path_support=path_support, n_kkt_sweeps=int(n_kkt),
                          meta=meta)


def _expand_subset(sub, used, P, K):
    """Map a sub-panel result back onto the full P-series index space."""
    def up(arr):
        out = np.zeros((K, P, P), dtype=arr.dtype)
        out[np.ix_(np.arange(K), used, used)] = arr
        return out
    res = SkeletonResult(
        B0=up(sub.B0),
        Delta={e: up(v) for e, v in sub.Delta.items()},
        support=up(sub.support),
        lambda_path=sub.lambda_path,
        lambda_selected=sub.lambda_selected,
        path_support=[up(s) for s in sub.path_support],
        n_kkt_sweeps=sub.n_kkt_sweeps,
        meta=dict(sub.meta),
    )
    res.meta["P"] = P
    res.meta["subset_series"] = used
    res.meta["support_shared"] = up(sub.meta["support_shared"])
    return res


# --------------------------------------------------------------------------- #
#  Conditioning diagnostic + the bivariate branch it routes to
# --------------------------------------------------------------------------- #
# The skeleton and the refit that scores it are FULLY CONDITIONAL: every candidate
# source is estimated with every other candidate source held fixed.  That is the right
# estimand -- a DIRECT effect -- but it is only *identified* when the sources are
# separable in the design.  When several sources carry nearly the same lagged
# information, the conditional fit has to split one piece of evidence between them and
# the group lasso admits an arbitrary member of the correlated set, so both the
# coefficient and the admission order stop meaning what they are read as meaning.
#
# The two functions below are the structural half of that problem: one MEASURES how
# separable the sources are, the other computes the evidence a design that cannot
# separate them can still support (own past + one candidate at a time).  Neither looks
# at anything but the design's own second moments, so routing on them cannot leak an
# outcome.

def _blocks(K, P):
    """Column positions of each source block in the stacked ``(k, j)`` design."""
    return np.arange(K * P).reshape(K, P).T          # [j] -> the K columns of source j


def conditioning_diagnostic(mt, max_lag=None, ridge=1e-8, max_cols=1024):
    """How separable are the SOURCES of the stacked lag design?  (Structural.)

    The design has one ``K``-column block per source series.  The quantity that decides
    whether a *conditional* fit can tell those blocks apart is the generalised variance
    inflation factor of Fox & Monette (1992),

        GVIF_j = det(R_jj) * det( (R^-1)_jj )

    for ``R`` the correlation matrix of the whole design and ``R_jj`` the block of
    source ``j``.  (The usual determinant form ``det(R_jj) det(R_-j-j) / det(R)`` is the
    same number: the Schur identity ``det(R_-j-j) = det(R) det((R^-1)_jj)`` cancels the
    global determinant, which is what makes it one small gather instead of ``P``
    factorisations of the complement.)  Its ``K``-th root ``GVIF_j^(1/K)`` is the
    per-coefficient-equivalent inflation, and its reciprocal

        tol_j = GVIF_j^(-1/K)  in (0, 1]

    is the share of source ``j``'s identifying variation that is ITS OWN rather than
    shared with the other sources.

    Two properties make this the right statistic here, and both are structural:

    * **Null 1 by construction.**  Blocks orthogonal to each other give ``R`` block
      diagonal, ``det(R_jj) det((R^-1)_jj) = 1`` exactly, so ``tol_j = 1``.  There is no
      calibration step and no fitted reference -- exactly the discipline
      :func:`cafe.discovery.loading_homogeneity_ratio` uses for the engine router.
    * **Blind to WITHIN-source collinearity.**  Consecutive lags of one persistent
      series are nearly collinear in every real design, and that costs the estimator
      nothing here: the group lasso groups over lags and the bivariate alternative
      carries the same whole block, so a diagnostic that fired on it would fire on
      everything.  ``det(R_jj)`` and ``det(R)`` shrink together under within-block
      collinearity and the ratio does not move -- that is precisely what the Fox-Monette
      form was designed for, and why a per-COLUMN VIF is the wrong statistic.  Measured:
      a CausalRivers 5-station panel scores 0.14 per column (its 6-hourly discharge lags
      are near-duplicates) against 0.74 per source block.

    The threshold that reads this is 1/2 -- see
    :data:`cafe.discovery.SOURCE_SEPARABILITY`.

    Parameters
    ----------
    mt : MomentTensor
        Built by :func:`build_moment_tensor` on the residual panel.
    max_lag : int, optional
        Lags to include (default: the tensor's own ``max_lag``).
    ridge : float
        Relative jitter on the Gram before inversion, matching the scoring refit.
    max_cols : int
        Give up (and say so) above this many design columns.  The cost is one
        ``(K P)^2`` inverse, the same object the multivariate refit already forms.

    Returns
    -------
    dict
        ``{tolerance, block_tolerance, min_tolerance, n_below_half, n_sources,
        n_cols, ok, reason}``.  ``ok=False`` (with ``tolerance=1.0``) whenever the
        diagnostic could not be formed -- the caller then keeps the fully conditional
        evidence, which is the behaviour that needs no diagnostic to justify it.
    """
    K = int(mt.max_lag if max_lag is None else max_lag)
    P = int(mt.P)
    out = {"tolerance": 1.0, "block_tolerance": None, "min_tolerance": 1.0,
           "n_below_half": 0, "n_sources": int(P), "n_cols": int(K * P),
           "ok": False, "reason": ""}
    if K < 1 or P < 2:
        out["reason"] = f"nothing to condition on (K={K}, P={P})"
        return out
    if K * P > int(max_cols):
        out["reason"] = (f"design has {K * P} columns (> {int(max_cols)}): the "
                         "separability diagnostic was not formed")
        return out
    n = float(mt.n_tot)
    if n <= K * P + 2:
        out["reason"] = f"n={n:.0f} rows against {K * P} design columns: not identified"
        return out
    Gt = mt.scaled(True)
    kk = np.repeat(np.arange(1, K + 1), P)
    jj = np.tile(np.arange(P), K)
    R = _gather_gram(Gt, jj, kk) / n
    R = 0.5 * (R + R.T)
    A = R.shape[0]
    try:
        Ri = np.linalg.inv(R + float(ridge) * np.eye(A))
    except np.linalg.LinAlgError:                              # pragma: no cover
        out["reason"] = "design Gram is singular"
        return out
    tol = np.ones(P, dtype=float)
    for j, g in enumerate(_blocks(K, P)):
        s1, ld1 = np.linalg.slogdet(R[np.ix_(g, g)])
        s2, ld2 = np.linalg.slogdet(Ri[np.ix_(g, g)])
        if s1 <= 0 or s2 <= 0 or not np.isfinite(ld1 + ld2):   # pragma: no cover
            continue
        tol[j] = 1.0 / max(np.exp((ld1 + ld2) / K), 1.0)       # GVIF >= 1 by definition
    out.update(tolerance=float(tol.mean()), block_tolerance=tol,
               min_tolerance=float(tol.min()),
               n_below_half=int((tol < 0.5).sum()), ok=True)
    return out


def bivariate_granger(mt, max_lag=None, ridge=1e-8, max_pairs=100000):
    """``|t|`` of every ``(j, k)`` in the SMALLEST design that still controls own past.

    For target ``i`` and source ``j`` the design is ``[ lags 1..K of i , lags 1..K of j ]``
    -- 2K columns -- fitted by OLS from the very moment tensor the fully conditional
    refit uses, so the two statistics differ in NOTHING but their conditioning set.
    That is the point: it is the same estimator, one assumption weaker, and it is what
    the design can still support when :func:`conditioning_diagnostic` says the sources
    are not separable enough for the conditional coefficients to mean anything.

    It buys identification with bias, not the other way round: anything ``j`` shares
    with a suppressed third source is credited to ``j``.  So it is evidence to be
    BLENDED under a measured weight, never a free upgrade -- see
    :data:`cafe.discovery.SOURCE_SEPARABILITY`.

    The ``j == i`` cells are the own-lag regression on its own (``K`` columns), which is
    the same statistic every off-diagonal cell conditions on.

    Cost is ``P`` batched inverses of ``(P-1, 2K, 2K)``; the whole thing is one gather
    plus ``O(P^2 K^3)`` flops and never materialises a design matrix.

    Returns
    -------
    (K, P, P) float array
        ``|t|``, NaN wherever the pair could not be fitted.
    """
    K = int(mt.max_lag if max_lag is None else max_lag)
    P = int(mt.P)
    T = np.full((K, P, P), np.nan, dtype=float)
    if K < 1 or P < 1:
        return T
    n = float(mt.n_tot)
    if n <= 2 * K + 2 or P * P > int(max_pairs):
        return T
    Gt = mt.scaled(True)
    lg = slice(1, K + 1)
    jit2 = float(ridge) * n * np.eye(2 * K)
    jitK = float(ridge) * n * np.eye(K)
    dof2, dof1 = max(n - 2 * K, 1.0), max(n - K, 1.0)
    idx = np.arange(P)
    for i in range(P):
        o = idx[idx != i]
        # --- own lags alone: the restricted model, and the j == i cell -------- #
        Goo = Gt[lg, lg, i, i]
        coo = Gt[lg, 0, i, i]
        try:
            Ao = np.linalg.inv(Goo + jitK)
        except np.linalg.LinAlgError:                          # pragma: no cover
            Ao = np.linalg.pinv(Goo + jitK)
        bo = Ao @ coo
        s2o = max(n - float(bo @ coo), _EPS) / dof1
        T[:, i, i] = np.abs(bo) / np.sqrt(np.maximum(np.diag(Ao), 1e-300) * s2o)
        if o.size == 0:
            continue
        # --- own lags + exactly one other source ----------------------------- #
        m = o.size
        G = np.empty((m, 2 * K, 2 * K), dtype=float)
        c = np.empty((m, 2 * K), dtype=float)
        G[:, :K, :K] = Goo
        G[:, :K, K:] = Gt[lg, lg, i, :][:, :, o].transpose(2, 0, 1)
        G[:, K:, :K] = np.swapaxes(G[:, :K, K:], 1, 2)
        G[:, K:, K:] = Gt[lg, lg, o, o].transpose(2, 0, 1)
        c[:, :K] = coo
        c[:, K:] = Gt[lg, 0, o, i].T
        G += jit2
        try:
            Ai = np.linalg.inv(G)
        except np.linalg.LinAlgError:                          # pragma: no cover
            Ai = np.linalg.pinv(G)
        B = np.einsum("aij,aj->ai", Ai, c)
        # y'y == n exactly (unit-RMS target column); at the OLS solution the cross
        # term collapses, so rss = y'y - b'X'y -- the same identity the refit uses.
        s2 = np.maximum(n - (B * c).sum(axis=1), _EPS) / dof2
        vd = np.maximum(np.diagonal(Ai, axis1=1, axis2=2), 1e-300)
        t = np.abs(B) / np.sqrt(vd * s2[:, None])
        T[:, i, o] = t[:, K:].T
    return T
