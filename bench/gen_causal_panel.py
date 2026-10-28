r"""
Ground-truth CAUSAL panel generator (CAFE-CD).

WHY THIS FILE EXISTS
--------------------
`harness.py` is the canonical source of benchmark data and its docstring forbids
forking data generation *in a method file*. This is not a method file: it is the
missing canonical generator family. Nothing in the repo produced a **directed** or
**lagged** ground-truth graph -- `exp_byproduct_validation.make_block_panel` is
undirected and purely contemporaneous -- so causal discovery on CAFE's idiosyncratic
residuals had no instrument to be scored against. This module is that instrument.
It IMPORTS `_ar_factor` and `MASKERS` from `harness` (harness never imports this
module, so there is no cycle) and reuses the repo's `RNG_SEED` convention.

THE DATA-GENERATING PROCESS
---------------------------
For entity e, time t, feature p:

    x[e,t,p] = mu[e,p] + tfe[t,p] + (Lam @ z[e,t])[p] + season[t,p] + eidio[e,t,p]

the observable panel, and the *idiosyncratic* component -- the arm CAFE is supposed
to hand to a discovery step -- follows a structural VAR with a known graph:

    (I - C) eidio[e,t] = sum_{k=1..K} B_e[k] eidio[e,t-k] + u[e,t]

  * `Lam @ z` is the pervasive common factor: shared across entities by default, so
    it injects *cross-sectional* correlation between every pair of features. This is
    the whole point of the benchmark -- a discovery method run on the RAW panel sees
    that spurious correlation, a method run on a good residual does not. `chi` sets
    how much of each feature's variance the factor owns.
  * `B_e[k]` is the lagged graph. `A_lag[k-1,i,j]` is the coefficient of
    `eidio[t-k, j]` on `eidio[t, i]`: **column j is the CAUSE, row i is the EFFECT**
    (the usual VAR convention, and the transpose of the "row = cause" adjacency that
    `sortability.py` wants -- use :meth:`CausalPanelTruth.skeleton` and never
    hand-transpose).
  * `C` is an optional contemporaneous DAG, strictly lower-triangular in a random
    topological `order`.
  * `u` are the structural shocks, drawn from a named distribution and standardised
    to unit variance. `t3`/`laplace`/`mix` are what LiNGAM-family orientation feeds
    on; **`gauss` is the negative control on which orientation MUST fail.**

TWO NON-NEGOTIABLES
-------------------
1. **Stability.** The `(K*P, K*P)` companion matrix of the *reduced form* is rescaled
   until its spectral radius equals `spec_radius` exactly (fixed point, bisection
   fallback). An unstable panel explodes and every downstream number is noise.
   Achieved radii are asserted and recorded in `truth.meta`.
2. **Equal marginal variance (`equal_variance=True`, the default).** Without it the
   marginal variance of a node grows with its depth in the graph, `varsortability`
   goes to 1.0, and "sort the nodes by variance" recovers the causal order for free.
   That artefact is exactly how continuous-optimisation structure learners
   (NOTEARS/DYNOTEARS-class) have been flattered in the literature -- Reisach,
   Seiler & Weichwald, "Beware of the Simulated DAG" (NeurIPS 2021); Lohse & Wahl
   (TMLR 2024). A fixed point on the EXACT stationary variance rescales every node to
   unit variance -- coefficients included, so the returned graph stays the true DGP of
   the returned data. Keep `equal_variance=False` ONLY as an explicitly labelled
   *sortable control*. Note (measured, reported by the self-validation): the artefact
   lives in the CONTEMPORANEOUS DAG. A stationary VAR with `deg` parents per node has
   no depth to accumulate along and is close to varsortability 0.5 with or without
   the fix; switch `contemp=True` on to exercise it.

USAGE
-----
    from gen_causal_panel import gen_causal_panel, mask_panel
    X3d, X2d, meta, truth = gen_causal_panel(E=20, T=200, P=12, noise="t3")

    res  = cafe.CAFE().run(X3d)          # 3-D on-ramp (src/cafe/io.py)
    fill = cafe.impute(X2d, meta)        # stacked (E*T, P), row e*T+t
    M3d  = mask_panel("block", X3d, 0.30)

    A_skel = truth.skeleton()            # (P,P) row = CAUSE, col = EFFECT
    from causal_metrics import summarize_graph
    summarize_graph(truth.A_lag, S_hat)

Self-validation:  python3 bench/gen_causal_panel.py
"""
from __future__ import annotations

import os

# Pin BLAS threads BEFORE numpy is imported (repo convention). ``setdefault`` so a
# caller that deliberately configured threads is never overridden.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys                                                       # noqa: E402
from dataclasses import dataclass                                # noqa: E402

import numpy as np                                               # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from harness import MASKERS, RNG_SEED, _ar_factor          # noqa: E402

__all__ = ["CausalPanelTruth", "gen_causal_panel", "mask_panel",
           "companion", "spectral_radius", "NOISE_KINDS", "CORE_FOURIER_PERIODS"]

NOISE_KINDS = ("gauss", "laplace", "t3", "uniform", "mix")

#: Mirrors ``src/cafe/_core.py::_fourier_periods``. A `season` period drawn from this
#: set is *correctly specified* for CAFE's seasonal design; anything else (e.g. 33) is
#: the misspecification variant. Mirrored rather than imported so this bench module
#: never depends on a private library internal.
CORE_FOURIER_PERIODS = (7.0, 12.0, 24.0, 48.0, 168.0, 365.0)


# --------------------------------------------------------------------------- #
# Ground-truth container
# --------------------------------------------------------------------------- #
def _common_component(Lam, Z):
    """(E,T,P) common component, for homogeneous (P,R) or entity-specific (E,P,R) loadings.

    A shared factor path with SHARED loadings is bit-identically the same series for every
    entity, so a two-way (entity, time) within-transform removes it EXACTLY -- which makes
    such a panel useless for testing factor residualisation (measured: raw == oracle to
    four decimals). Entity-specific loadings -- the realistic "one market factor,
    heterogeneous betas" case -- are NOT removable by a time dummy, which is why
    `beta_hetero > 0` is the configuration that can actually discriminate between
    residualisation strategies.
    """
    Lam = np.asarray(Lam, float)
    if Lam.ndim == 2:
        return np.einsum("pr,etr->etp", Lam, Z)
    if Lam.ndim == 3:
        return np.einsum("epr,etr->etp", Lam, Z)
    raise ValueError(f"Lam must be (P,R) or (E,P,R); got {Lam.shape}")


@dataclass(frozen=True)
class CausalPanelTruth:
    """Everything the generator knows. `E_idio` is the ORACLE arm: the true
    idiosyncratic component a perfect residualiser would hand to a discovery step."""

    A_lag: np.ndarray        # (K,P,P)  A_lag[k-1,i,j] = coeff of e[t-k,j] -> e[t,i]
    A_contemp: np.ndarray    # (P,P) strictly lower-tri in `order`; zeros if disabled
    order: np.ndarray        # (P,) topological order of the contemporaneous DAG
    Delta: dict              # {eid: (K,P,P)} entity deviations; B_e = A_lag + Delta[e]
    Lam: np.ndarray          # (P,R) loadings, or (E,P,R) when beta_hetero>0
    Z: np.ndarray            # (E,T,R) latent AR factors
    E_idio: np.ndarray       # (E,T,P) TRUE idiosyncratic component -- the ORACLE arm
    U: np.ndarray            # (E,T,P) structural shocks
    mu_e: np.ndarray         # (E,P) entity fixed effects
    tfe: np.ndarray          # (T,P) time fixed effects
    meta: dict

    # -- convenience -------------------------------------------------------- #
    def common(self):
        """(E,T,P) common factor component, whatever shape `Lam` has."""
        return _common_component(self.Lam, self.Z)

    def B_entity(self, e):
        """(K,P,P) lagged coefficients actually used for entity `e`."""
        return self.A_lag + self.Delta[int(e)]

    def skeleton(self, include_contemp=True, include_self=False):
        """(P,P) binary adjacency in **row = CAUSE, col = EFFECT** orientation.

        `A_lag` is stored in VAR orientation (row = effect), so this transposes it;
        lags are collapsed with OR. This is the matrix `sortability.var_sortability`
        and `sortability.r2_sortability` expect -- always build it here rather than
        transposing by hand at the call site.
        """
        A = (np.abs(self.A_lag).sum(axis=0) != 0)          # [i,j] = j -> i
        S = A.T.copy()                                     # [i,j] = i -> j
        if include_contemp:
            S |= (np.abs(self.A_contemp) != 0).T
        if not include_self:
            np.fill_diagonal(S, False)
        return S.astype(float)

    def lagged_skeleton(self, include_self=True):
        """(P,P) collapsed LAGGED-only skeleton in VAR orientation (row = effect)."""
        S = (np.abs(self.A_lag).sum(axis=0) != 0)
        if not include_self:
            S = S.copy()
            np.fill_diagonal(S, False)
        return S.astype(float)


# --------------------------------------------------------------------------- #
# Companion matrix / stability
# --------------------------------------------------------------------------- #
def companion(B):
    """(K,P,P) lag coefficients -> the (K*P, K*P) VAR(1) companion matrix."""
    B = np.asarray(B, float)
    K, P, _ = B.shape
    C = np.zeros((K * P, K * P))
    C[:P] = np.concatenate([B[k] for k in range(K)], axis=1)
    if K > 1:
        C[P:, :-P] = np.eye((K - 1) * P)
    return C


def spectral_radius(B):
    """max |eigenvalue| of the companion matrix. < 1 <=> stationary."""
    B = np.asarray(B, float)
    if B.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(companion(B)))))


def _stabilise_scale(B, target, tol=1e-12, max_iter=200):
    """Scalar c with rho(companion(c*B)) == target.

    A *uniform* scale is used (not the closed-form ``B_k -> c**k B_k``) so the
    relative magnitude of lag-1 vs lag-K edges is preserved -- otherwise deep lags
    are shrunk by c**k and become undetectable, quietly destroying `lag_accuracy`.
    Uniform scaling has no closed form for K>1, so: multiplicative fixed point
    (converges in a handful of steps), bisection fallback for a guarantee.
    """
    r0 = spectral_radius(B)
    if r0 <= 1e-12 or target <= 0:
        return 1.0
    c = float(target / r0)
    for _ in range(max_iter):
        r = spectral_radius(c * B)
        if abs(r - target) <= tol * max(target, 1e-12):
            return c
        if r <= 1e-300:
            break
        c *= target / r
    # bisection fallback (rho is continuous, rho(0)=0, monotone in practice)
    lo, hi = 0.0, max(c, 1.0)
    for _ in range(80):
        if spectral_radius(hi * B) >= target:
            break
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid * B) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Structural pieces
# --------------------------------------------------------------------------- #
def _draw_noise(kind, size, rng):
    """Shocks standardised to POPULATION unit variance (not sample-standardised:
    rescaling by a realised sd would distort the tail that orientation depends on)."""
    if kind == "gauss":
        return rng.standard_normal(size)
    if kind == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)      # var = 2 b^2 = 1
    if kind == "t3":
        return rng.standard_t(3.0, size) / np.sqrt(3.0)        # var(t3) = 3
    if kind == "uniform":
        s3 = np.sqrt(3.0)
        return rng.uniform(-s3, s3, size)
    if kind == "mix":                                          # 90/10 scale mixture
        base = rng.standard_normal(size)
        heavy = rng.random(size) < 0.10
        return base * np.where(heavy, 3.0, 1.0) / np.sqrt(0.9 + 0.1 * 9.0)
    raise ValueError(f"unknown noise kind {kind!r}; expected one of {NOISE_KINDS}")


def _draw_lagged(P, K, deg, coef_lo, coef_hi, ar_self, rng):
    """(K,P,P) with EXACTLY `deg` parents per (node, lag), signs random, magnitudes
    U(coef_lo, coef_hi). With `ar_self` the lag-1 self-loop i->i is guaranteed (and
    positive): the idiosyncratic component of a real panel is persistent, and the AR
    term is the archetypal strongest true edge -- which is why `causal_metrics`
    flattens lagged graphs *including* the diagonal."""
    A = np.zeros((K, P, P))
    for k in range(K):
        for i in range(P):
            if ar_self and k == 0:
                if deg > 1 and P > 1:
                    pool = np.delete(np.arange(P), i)
                    extra = rng.choice(pool, size=min(deg - 1, P - 1), replace=False)
                else:
                    extra = np.zeros(0, dtype=int)
                parents = np.concatenate([np.array([i]), extra]).astype(int)
                signs = np.concatenate([np.array([1.0]),
                                        rng.choice([-1.0, 1.0], extra.size)])
            else:
                parents = rng.choice(P, size=min(deg, P), replace=False)
                signs = rng.choice([-1.0, 1.0], parents.size)
            A[k, i, parents] = rng.uniform(coef_lo, coef_hi, parents.size) * signs
    return A


def _draw_contemp(P, contemp_deg, coef_lo, coef_hi, rng):
    """(C, order): C[i,j] != 0 only where j precedes i in `order` (strictly lower
    triangular in that order => an acyclic contemporaneous DAG)."""
    order = rng.permutation(P)
    pos = np.empty(P, dtype=int)
    pos[order] = np.arange(P)
    C = np.zeros((P, P))
    for i in range(P):
        preds = order[:pos[i]]
        if preds.size == 0:
            continue
        k = int(min(contemp_deg, preds.size))
        par = rng.choice(preds, size=k, replace=False)
        C[i, par] = (rng.uniform(coef_lo, coef_hi, k)
                     * rng.choice([-1.0, 1.0], k))
    return C, order


def _stationary_var(B_all, Minv, su):
    """EXACT stationary marginal variance of every node, pooled over entities.

    Solves the discrete Lyapunov equation ``S = A S A' + Q`` for the companion form
    by doubling (``S <- S + M S M'``, ``M <- M M``), which converges geometrically in
    the spectral radius and is exact to machine precision in ~8 rounds. Second
    moments of a linear VAR depend only on the shock *variance*, so this is valid for
    every `noise` family.

    Used instead of measuring the sd on a simulated sample: a sample estimate carries
    a per-node Jensen bias (E[1/var_hat] > 1/E[var_hat], and the estimator is noisier
    for the more persistent, more aggregated downstream nodes), which quietly leaves
    varsortability at ~0.57 on contemporaneous DAGs instead of 0.5 -- exactly the
    artefact this is here to remove.
    """
    E, K, P, _ = B_all.shape
    v = np.asarray(su, float) ** 2
    Q0 = np.diag(v) if Minv is None else (Minv * v) @ Minv.T
    homog = bool(np.all(B_all == B_all[0]))
    out = np.zeros(P)
    for e in (range(1) if homog else range(E)):
        Phi = B_all[e] if Minv is None else np.einsum("pi,kij->kpj", Minv, B_all[e])
        M = companion(Phi)
        S = np.zeros((K * P, K * P))
        S[:P, :P] = Q0
        for _ in range(64):
            S = S + M @ S @ M.T
            M = M @ M
            mx = np.max(np.abs(M))
            if not np.isfinite(mx) or mx < 1e-16:
                break
        out += np.diag(S)[:P]
    return out / (1 if homog else E)


def _simulate(B_all, Minv, U):
    """e[t] = Minv @ (sum_k B[k] e[t-k] + u[t]) for every entity at once.

    B_all : (E,K,P,P) per-entity lag coefficients (VAR orientation)
    Minv  : (P,P) inv(I - C), or None
    U     : (E, Tt, P) shocks (Tt = burn_in + T)
    """
    E, Tt, P = U.shape
    K = B_all.shape[1]
    Y = np.zeros((E, Tt, P))
    for t in range(Tt):
        kmax = min(K, t)
        if kmax > 0:
            prev = np.stack([Y[:, t - k - 1] for k in range(kmax)], axis=1)  # (E,k,P)
            drive = np.einsum("ekij,ekj->ei", B_all[:, :kmax], prev,
                              optimize=True)
        else:
            drive = np.zeros((E, P))
        e_t = drive + U[:, t]
        if Minv is not None:
            e_t = e_t @ Minv.T
        Y[:, t] = e_t
    return Y


# --------------------------------------------------------------------------- #
# THE GENERATOR
# --------------------------------------------------------------------------- #
def gen_causal_panel(E=20, T=200, P=12, R=3, n_lags=2, deg=2,
                     rho_factor=0.9, chi=0.60,
                     spec_radius=0.70, coef_lo=0.25, coef_hi=0.55,
                     contemp=False, contemp_deg=1,
                     noise="gauss",              # gauss|laplace|t3|uniform|mix
                     hetero=0.0, shared_factors=True,
                     entity_fe=True, time_fe=True, season=None,
                     equal_variance=True, burn_in=100, seed=0,
                     ar_self=True, ev_iters=4, beta_hetero=0.0):
    """Panel with a KNOWN lagged (and optionally contemporaneous) causal graph on the
    idiosyncratic component, hidden under a pervasive common factor.

    Parameters
    ----------
    E, T, P, R      entities, time steps, features, latent factors.
    n_lags (K)      lag order of the structural VAR.
    deg             parents per (node, lag) -- exactly, self-loops included.
    rho_factor      AR(1) coefficient of every latent factor.
    chi             common-variance share per feature:
                    Var(Lam z)_p / (Var(Lam z)_p + Var(e)_p) == chi, hit EXACTLY by
                    rescaling each row of `Lam` against the realised variances.
                    chi -> 0: residualisation is pointless. chi -> 1: nothing
                    survives it. 0.60 is the interesting middle.
    spec_radius     target spectral radius of the reduced-form companion matrix.
    coef_lo/hi      |coefficient| range AS DRAWN. The stabilisation scale and the
                    equal-variance rescale are applied afterwards, so the coefficients
                    actually stored in `A_lag` are smaller (and no longer share a
                    common bound) -- read them off `truth.A_lag`, never off these.
    contemp         add a contemporaneous DAG (see `contemp_deg`).
    noise           shock family; `gauss` is the NEGATIVE CONTROL for LiNGAM-style
                    orientation, which must fail on it.
    hetero          fraction of edges re-drawn per entity (0 = homogeneous panel).
    shared_factors  True: one factor path for all entities (spurious cross-sectional
                    correlation in the raw arm -- the hard, realistic case).
    entity_fe/time_fe   additive fixed effects (N(0,2^2) / N(0,1), matching
                    `harness.gen_panel` scales).
    season          None, a period, or (period, rel_amp). A period outside
                    `CORE_FOURIER_PERIODS` is the misspecification variant.
    equal_variance  per-node shock-scale fixed point that kills varsortability.
                    False = the labelled "sortable control".
    burn_in         discarded warm-up steps of the VAR recursion.
    ar_self         guarantee a (positive) lag-1 self-loop per node.

    Returns
    -------
    X3d   (E,T,P) float
    X2d   (E*T,P) float, C-order entity-major -- row e*T+t, matching the 3-D
          flattening in `src/cafe/io.py::to_matrix`, so `X3d.reshape(E*T,P)` is X2d
          and `cafe.impute(X2d, meta)` sees the same panel as `cafe.CAFE().run(X3d)`.
    meta  {"entity_ids","time_ids","E","T","F"} -- exactly what `harness.metrics`,
          `harness.evaluate` and the library's panel path expect.
    truth CausalPanelTruth
    """
    if not (0.0 <= chi < 1.0):
        raise ValueError(f"chi must be in [0,1); got {chi}")
    if noise not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {noise!r}; expected one of {NOISE_KINDS}")
    if not (0.0 <= hetero <= 1.0):
        raise ValueError(f"hetero must be in [0,1]; got {hetero}")
    K = int(n_lags)
    rng = np.random.default_rng(RNG_SEED + 9000 + int(seed))

    # -- 1. latent factors ------------------------------------------------- #
    if shared_factors:
        Z1 = _ar_factor(T, R, rho_factor, rng)                 # (T,R), unit variance
        Z = np.repeat(Z1[None, :, :], E, axis=0)               # (E,T,R)
    else:
        Z = np.stack([_ar_factor(T, R, rho_factor, rng) for _ in range(E)], axis=0)

    # -- 2. loadings (rescaled to hit `chi` once Var(e) is known) ----------- #
    Lam = rng.standard_normal((P, R))
    if beta_hetero > 0.0:
        # one shared factor path, entity-specific exposures: Lam_e = Lam * (1 + b*xi_e)
        xi = rng.standard_normal((E, P, R))
        Lam = Lam[None, :, :] * (1.0 + float(beta_hetero) * xi)   # (E,P,R)

    # -- 3./5. structural graph -------------------------------------------- #
    A_lag = _draw_lagged(P, K, deg, coef_lo, coef_hi, ar_self, rng)
    if contemp:
        C, order = _draw_contemp(P, contemp_deg, coef_lo, coef_hi, rng)
        Minv = np.linalg.inv(np.eye(P) - C)                    # cached once
    else:
        C = np.zeros((P, P))
        order = np.arange(P)
        Minv = None

    # -- 4. stability (on the REDUCED form, which is what can explode) ------ #
    # Scaling the structural B_k by c scales the reduced form Minv @ B_k by the same
    # c, so one scalar fixes both and the sparsity pattern is untouched.
    red = A_lag if Minv is None else np.einsum("pi,kij->kpj", Minv, A_lag)
    c_stab = _stabilise_scale(red, spec_radius)
    A_lag = c_stab * A_lag
    red = A_lag if Minv is None else np.einsum("pi,kij->kpj", Minv, A_lag)
    rho_red = spectral_radius(red)
    rho_struct = spectral_radius(A_lag)
    assert abs(rho_red - spec_radius) < 1e-6, (
        f"stabilisation failed: achieved rho={rho_red:.6f} target={spec_radius}")

    # -- 7. entity heterogeneity ------------------------------------------- #
    nz = np.argwhere(A_lag != 0)
    n_re = int(round(hetero * nz.shape[0]))
    Delta = {}
    B_all = np.empty((E, K, P, P))
    for e in range(E):
        if n_re > 0:
            B_e = A_lag.copy()
            pick = rng.choice(nz.shape[0], size=n_re, replace=False)
            idx = tuple(nz[pick].T)
            B_e[idx] = (rng.uniform(coef_lo, coef_hi, n_re)
                        * rng.choice([-1.0, 1.0], n_re))
            red_e = B_e if Minv is None else np.einsum("pi,kij->kpj", Minv, B_e)
            B_e = _stabilise_scale(red_e, spec_radius) * B_e   # re-stabilise
            Delta[e] = B_e - A_lag                             # exact invariant
            B_all[e] = B_e
        else:
            Delta[e] = np.zeros((K, P, P))
            B_all[e] = A_lag

    # -- 8. EQUAL-VARIANCE FIXED POINT (the sortability fix) ---------------- #
    # Measure each node's marginal sd -> rescale that node to unit variance -> repeat
    # (`ev_iters` rounds; the trace of max|sd-1| is kept in truth.meta so the reader
    # can see it converge rather than take it on trust).
    #
    # TWO deliberate departures from the naive recipe, both forced by measurement:
    #
    # (a) The rescale is a *system* rescale, e~ = D^-1 e with D = diag(sd):
    #         A_lag <- D^-1 A_lag D,   C <- D^-1 C D,   u <- D^-1 u
    #     -- the SAME graph (D is diagonal and positive, so every zero, sign and
    #     topological order survives) driving exactly the same path, rescaled. The
    #     returned coefficients therefore remain the true DGP of the returned data,
    #     and the reduced-form companion undergoes only a similarity transform, so
    #     the spectral radius is unchanged to the last bit.
    #     Rescaling ONLY the shocks provably cannot work once a contemporaneous DAG
    #     is present: a downstream node inherits variance from its parents, so its
    #     marginal variance has a floor no shock scale can push it below. Measured,
    #     P=24 contemp_deg=3: shock-only scaling drives su_min -> 0.01 and still
    #     stalls at max|sd-1| = 0.055 after 60 rounds.
    #
    # (b) The sd is the EXACT stationary sd (Lyapunov), not one measured on a
    #     simulated sample. A sample estimate is unbiased for the variance but not
    #     for the rescale, and the bias is bigger for the noisier (more persistent,
    #     more aggregated => further downstream) nodes -- which leaves a residual
    #     ordering. Measured, P=24 contemp_deg=3: normalising by a sample sd from an
    #     independent 4x-length pilot still left varsortability at 0.568; the exact
    #     rescale brings it to ~0.51. The returned sample's variances then differ
    #     from 1 only by its own sampling noise, which is what an honest benchmark
    #     leaves in place (self-normalising the returned sample would pin every
    #     variance to exactly 1.0 and the gate below would pass vacuously on ties).
    Tt = int(burn_in) + T
    su = np.ones(P)
    ev_iters = int(ev_iters) if equal_variance else 0
    ev_trace = []
    if ev_iters > 0:
        for _ in range(ev_iters):
            sd = np.sqrt(_stationary_var(B_all, Minv, su))
            ev_trace.append(float(np.max(np.abs(sd - 1.0))))
            d = np.where(sd > 1e-12, sd, 1.0)
            A_lag = A_lag / d[None, :, None] * d[None, None, :]      # D^-1 A D
            B_all = B_all / d[None, None, :, None] * d[None, None, None, :]
            for e in range(E):
                Delta[e] = Delta[e] / d[None, :, None] * d[None, None, :]
            if Minv is not None:
                C = C / d[:, None] * d[None, :]
                Minv = Minv / d[:, None] * d[None, :]                # = D^-1 Minv D
            su = su / d
        red = A_lag if Minv is None else np.einsum("pi,kij->kpj", Minv, A_lag)
        rho_red = spectral_radius(red)                 # similarity => unchanged
        rho_struct = spectral_radius(A_lag)
        assert abs(rho_red - spec_radius) < 1e-6, (
            f"equal-variance rescale broke stability: rho={rho_red:.6f}")
    U0 = _draw_noise(noise, (E, Tt, P), rng)
    U_full = U0 * su
    E_full = _simulate(B_all, Minv, U_full)
    E_idio = np.ascontiguousarray(E_full[:, burn_in:])          # (E,T,P)
    U = np.ascontiguousarray(U_full[:, burn_in:])               # (E,T,P)
    if not np.all(np.isfinite(E_idio)):
        raise FloatingPointError("idiosyncratic VAR diverged -- check spec_radius")

    # -- 2 (cont). hit `chi` EXACTLY using realised variances ---------------- #
    var_e = E_idio.reshape(-1, P).var(axis=0)                   # (P,)
    common = _common_component(Lam, Z)
    var_c = common.reshape(-1, P).var(axis=0)
    if chi <= 0.0:
        Lam = np.zeros_like(Lam)
    else:
        want = (chi / (1.0 - chi)) * var_e                      # target Var(Lam z)_p
        scale = np.where(var_c > 1e-14, np.sqrt(want / np.maximum(var_c, 1e-300)), 0.0)
        Lam = Lam * (scale[:, None] if Lam.ndim == 2 else scale[None, :, None])
    common = _common_component(Lam, Z)
    var_c = common.reshape(-1, P).var(axis=0)
    chi_real = var_c / np.maximum(var_c + var_e, 1e-300)

    # -- 9. assembly -------------------------------------------------------- #
    mu_e = rng.standard_normal((E, P)) * 2.0 if entity_fe else np.zeros((E, P))
    tfe = rng.standard_normal((T, P)) * 1.0 if time_fe else np.zeros((T, P))
    X3d = (mu_e[:, None, :] + tfe[None, :, :] + common + E_idio)
    season_period = season_amp = None
    if season is not None:
        if np.ndim(season) == 0:
            season_period, rel_amp = float(season), 0.5
        else:
            season_period, rel_amp = float(season[0]), float(season[1])
        tt = np.arange(T)
        phase = rng.uniform(0, 2 * np.pi, P)
        season_amp = rel_amp * np.sqrt(var_c + var_e)
        S = season_amp * np.sin(2 * np.pi * tt[:, None] / season_period + phase)
        X3d = X3d + S[None, :, :]
    X3d = np.ascontiguousarray(X3d)
    X2d = np.ascontiguousarray(X3d.reshape(E * T, P))

    meta = {"entity_ids": np.repeat(np.arange(E), T),
            "time_ids": np.tile(np.arange(T), E),
            "E": E, "T": T, "F": P}

    tmeta = dict(
        E=E, T=T, P=P, R=R, K=K, deg=deg, seed=seed,
        rho_factor=rho_factor, chi=chi, spec_radius=spec_radius,
        coef_lo=coef_lo, coef_hi=coef_hi, contemp=bool(contemp),
        contemp_deg=contemp_deg, noise=noise, hetero=hetero,
        shared_factors=bool(shared_factors), entity_fe=bool(entity_fe),
        time_fe=bool(time_fe), equal_variance=bool(equal_variance),
        burn_in=burn_in, ar_self=bool(ar_self), ev_iters=ev_iters,
        ev_trace=[float(v) for v in ev_trace],
        ev_resid=(float(ev_trace[-1]) if ev_trace else float("nan")),
        stab_scale=float(c_stab),
        rho_reduced=float(rho_red), rho_structural=float(rho_struct),
        rho_entity_max=float(max(
            spectral_radius(B_all[e] if Minv is None
                            else np.einsum("pi,kij->kpj", Minv, B_all[e]))
            for e in range(E))),
        chi_realised_mean=float(chi_real.mean()),
        chi_realised_min=float(chi_real.min()),
        chi_realised_max=float(chi_real.max()),
        idio_sd_mean=float(np.sqrt(var_e).mean()),
        idio_sd_min=float(np.sqrt(var_e).min()),
        idio_sd_max=float(np.sqrt(var_e).max()),
        idio_sd_dev=float(np.max(np.abs(np.sqrt(var_e) - 1.0))),
        n_edges_lagged=int((A_lag != 0).sum()),
        n_edges_lagged_cross=int((A_lag != 0).sum()
                                 - sum(int(np.count_nonzero(np.diag(A_lag[k])))
                                       for k in range(K))),
        n_edges_contemp=int((C != 0).sum()),
        season_period=season_period,
        season_rel_amp=(None if season_period is None else float(rel_amp)),
        season_misspecified=(None if season_period is None
                             else season_period not in CORE_FOURIER_PERIODS),
    )
    truth = CausalPanelTruth(A_lag=A_lag, A_contemp=C, order=order, Delta=Delta,
                             Lam=Lam, Z=Z, E_idio=E_idio, U=U, mu_e=mu_e, tfe=tfe,
                             meta=tmeta)
    return X3d, X2d, meta, truth


# --------------------------------------------------------------------------- #
# Per-entity masking
# --------------------------------------------------------------------------- #
def mask_panel(mask_name, X3d, rate, seed=0):
    """Apply harness.MASKERS[mask_name] to EACH entity slice independently, so a
    'block' gap is a real per-series outage and never straddles an entity boundary.
    (harness.mask_block on a stacked (E*T, P) panel produces blocks that cross
    entity edges -- the existing panel_block30 dataset has this wart.)

    Returns an (E,T,P) boolean mask; `M.reshape(E*T, P)` lines up with X2d.
    """
    if mask_name not in MASKERS:
        raise KeyError(f"unknown masker {mask_name!r}; have {sorted(MASKERS)}")
    X3d = np.asarray(X3d, float)
    if X3d.ndim != 3:
        raise ValueError(f"mask_panel expects (E,T,P); got shape {X3d.shape}")
    E = X3d.shape[0]
    fn = MASKERS[mask_name]
    M = np.empty(X3d.shape, dtype=bool)
    for e in range(E):
        M[e] = fn(X3d[e], rate, seed=int(seed) * 1000 + e)     # decorrelated streams
    return M


# =========================================================================== #
# SELF-VALIDATION
# =========================================================================== #
def _oracle_var_recovery(truth, ridge=0.0):
    """Recover A_lag from the ORACLE residual by pooled OLS-VAR.

    THE most important self-check: if the true idiosyncratic component cannot
    recover its own graph, the generator is broken and nothing downstream means
    anything. Returns (A_hat (K,P,P), t-stats (K,P,P), summary dict) -- the t-stats
    give a TRUTH-FREE way to threshold, so the binary metrics can be exercised at a
    realistic operating point instead of one picked with knowledge of the answer.
    """
    import causal_metrics as cm
    Eid = truth.E_idio
    E, T, P = Eid.shape
    K = truth.A_lag.shape[0]
    Xs, Ys = [], []
    for e in range(E):
        y = Eid[e, K:]                                          # (T-K, P)
        lags = [Eid[e, K - k - 1:T - k - 1] for k in range(K)]   # lag k+1
        Xs.append(np.concatenate(lags, axis=1))                 # (T-K, K*P)
        Ys.append(y)
    Xd = np.concatenate(Xs, axis=0)
    Yd = np.concatenate(Ys, axis=0)
    Xd = np.column_stack([Xd, np.ones(len(Xd))])
    n, m = Xd.shape
    G = Xd.T @ Xd + (ridge * np.eye(m) if ridge > 0 else 0.0)
    Ginv = np.linalg.inv(G)
    Bfull = Ginv @ (Xd.T @ Yd)                                  # (m, P)
    resid = Yd - Xd @ Bfull
    s2 = (resid ** 2).sum(axis=0) / max(n - m, 1)               # (P,)
    se = np.sqrt(np.outer(s2, np.diag(Ginv)))                   # (P, m)
    Bhat = Bfull[:-1].T                                         # (P, K*P)
    Tst = np.abs(Bhat) / np.maximum(se[:, :-1], 1e-300)
    A_hat = np.stack([Bhat[:, k * P:(k + 1) * P] for k in range(K)], axis=0)
    A_t = np.stack([Tst[:, k * P:(k + 1) * P] for k in range(K)], axis=0)
    return A_hat, A_t, cm.summarize_graph(truth.A_lag, A_hat)


def _kurtosis(x):
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    s = x.std()
    return float(np.mean((x / s) ** 4) - 3.0) if s > 0 else float("nan")


def _skew(x):
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    s = x.std()
    return float(np.mean((x / s) ** 3)) if s > 0 else float("nan")


def _main():
    import causal_metrics as cm
    from sortability import r2_sortability, var_sortability

    ok = True
    print("=" * 78)
    print("gen_causal_panel.py  SELF-VALIDATION")
    print("=" * 78)

    # ---------------------------------------------------------------- G0 --- #
    print("\n[G0] SCORING MACHINERY  (causal_metrics.py, hand-checkable cases)")
    T2 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)     # 0->1, 1->2
    H2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], float)     # 0->1 REVERSED, 1->2 ok
    checks = [("shd directed (1 reversal)", cm.shd(T2, H2), 1),
              ("shd skeleton (same skeleton)", cm.shd(T2, H2, directed=False), 0),
              ("shd vs empty graph", cm.shd(T2, np.zeros((3, 3))), 2),
              ("shd with 1 spurious edge", cm.shd(T2, T2 + np.eye(3, k=-2)), 1),
              ("skeleton_f1 (direction-blind)", cm.skeleton_f1(T2, H2), 1.0),
              ("edge_prf f1 (tp1 fp1 fn1)", cm.edge_prf(T2, H2)["f1"], 0.5)]
    A3 = np.zeros((2, 3, 3))
    A3[0, 0, 0] = 0.5                       # a lag-1 SELF-LOOP: must be scored
    A3[0, 1, 0] = 0.3
    A3[1, 2, 1] = -0.4
    S3 = np.zeros_like(A3)
    S3[0, 0, 0], S3[0, 1, 0], S3[1, 2, 1] = 9.0, 8.0, 7.0
    checks += [("flat_lagged keeps the diagonal",
                int((cm.flat_lagged(A3) != 0).sum()), 3),
               ("flat_contemp drops it", cm.flat_contemp(A3[0]).size, 6),
               ("lag_accuracy, perfect scores", cm.lag_accuracy(A3, S3), 1.0),
               ("auc, perfect scores", cm.edge_auc(cm.flat_lagged(A3) != 0,
                                                   np.abs(cm.flat_lagged(S3))), 1.0),
               ("auc, reversed scores", cm.edge_auc(cm.flat_lagged(A3) != 0,
                                                    -np.abs(cm.flat_lagged(S3))), 0.0)]
    for name, got, want in checks:
        good = abs(float(got) - float(want)) < 1e-12
        print(f"      {name:<36s} = {got!s:<6s} (expect {want})  "
              f"{'ok' if good else '!! WRONG'}")
        ok &= good
    try:                                    # cross-check the no-sklearn AUC
        from sklearn.metrics import roc_auc_score
        rr = np.random.default_rng(0)
        worst = 0.0
        for trial in range(200):
            n = int(rr.integers(10, 400))
            y = (rr.random(n) < rr.uniform(0.05, 0.95)).astype(int)
            if y.sum() in (0, n):
                continue
            if trial % 3 == 0:
                s = rr.integers(0, 3, n).astype(float)          # heavy ties
            elif trial % 3 == 1:
                s = np.round(rr.standard_normal(n), 1) + 0.4 * y
            else:
                s = rr.standard_normal(n)
                s[rr.random(n) < 0.1] = np.nan                  # NaN scores
            ref = roc_auc_score(y, np.where(np.isfinite(s), s, -1e18))
            worst = max(worst, abs(ref - cm.edge_auc(y, s)))
        print(f"      {'edge_auc vs sklearn (200 trials,':<36s} max|diff| = "
              f"{worst:.1e}  {'ok' if worst < 1e-9 else '!! WRONG'}")
        print(f"      {'  ties + NaNs included)':<36s}")
        ok &= worst < 1e-9
    except ImportError:
        print("      (sklearn absent -- skipped the AUC cross-check)")

    # ---------------------------------------------------------------- G1 --- #
    print("\n[G1] SORTABILITY GATE  (equal_variance=True; every cell must be in "
          "[0.40, 0.60])")
    print("     var_sortability / r2_sortability of the ORACLE residual "
          "truth.E_idio.")
    print("     Each cell is a fraction over that panel's edges, so it carries "
          "binomial noise;")
    print("     the per-P column pools the 5 seeds and is the statistically "
          "meaningful test.\n")
    Ps = [6, 10, 16, 24, 40]
    seeds = list(range(5))
    print("      " + "".join(f"{'seed ' + str(s):>15s}" for s in seeds)
          + f"{'pooled(+-95%)':>24s}")
    grid_bad, pooled_rows = [], []
    n_cells = len(Ps) * len(seeds) * 2
    for P in Ps:
        cells, vs_all, rs_all, n_all = [], [], [], []
        for s in seeds:
            _, _, _, tr = gen_causal_panel(P=P, seed=s)
            skel = tr.skeleton()
            n_e = int(skel.sum())
            vs = var_sortability(tr.E_idio, skel)
            rs = r2_sortability(tr.E_idio, skel)
            vs_all.append(vs)
            rs_all.append(rs)
            n_all.append(n_e)
            cells.append(f"{vs:>7.3f}/{rs:<7.3f}")
            for nm, v in (("var", vs), ("r2", rs)):
                if not (0.40 <= v <= 0.60):
                    grid_bad.append((P, s, nm, v, n_e))
        n_tot = sum(n_all)
        mv = float(np.average(vs_all, weights=n_all))
        mr = float(np.average(rs_all, weights=n_all))
        band = 1.96 * np.sqrt(0.25 / n_tot)
        pooled_rows.append((P, mv, mr, n_tot))
        print(f"P={P:<4d}" + "".join(cells)
              + f"  {mv:>6.3f}/{mr:<6.3f} +-{band:.3f}")
    N = sum(r[3] for r in pooled_rows)
    gv = float(np.average([r[1] for r in pooled_rows],
                          weights=[r[3] for r in pooled_rows]))
    gr = float(np.average([r[2] for r in pooled_rows],
                          weights=[r[3] for r in pooled_rows]))
    gband = 1.96 * np.sqrt(0.25 / N)
    print(f"\n     grand pooled over {N} edges: var={gv:.4f}  r2={gr:.4f}  "
          f"(95% band under H0=0.5: +-{gband:.4f})")
    # per-P pooled is the gate that is not swamped by binomial noise
    for P, mv, mr, _n in pooled_rows:
        for nm, v in (("var", mv), ("r2", mr)):
            if not (0.40 <= v <= 0.60):
                print(f"     !! pooled P={P} {nm}={v:.3f} OUT OF BAND")
                ok = False
    if grid_bad:
        # 50 cells are inspected, so judge a single excursion at a
        # multiplicity-corrected level (z = 3.02 ~ 0.05 family-wise over 50 tests)
        z = 3.02
        print(f"     {len(grid_bad)}/{n_cells} single-seed cells outside "
              f"[0.40,0.60] -- classified against their own edge count:")
        for P, s, nm, v, n_e in grid_bad:
            band = z * np.sqrt(0.25 / max(n_e, 1))
            note = ("binomial noise" if abs(v - 0.5) <= band
                    else "BIAS -- generator is rigged")
            print(f"        P={P:<3d} seed={s} {nm:>3s}={v:.3f}  ({n_e} edges, "
                  f"+-{band:.3f} family-wise band under H0)  -> {note}")
            if "BIAS" in note:
                ok = False
    else:
        print("     every single-seed cell also inside [0.40, 0.60]")

    # ------------------------------------------------- sortable control --- #
    print("\n[G1b] SORTABLE CONTROL  (equal_variance=False must be visibly "
          "sortable, or the fix is a placebo)")
    print("      MEASURED FINDING: marginal variance accumulates along "
          "CONTEMPORANEOUS depth only.")
    print("      A stationary VAR with exactly `deg` parents per (node,lag) has no "
          "depth to accumulate")
    print("      along -- every node is up- and downstream of every other through "
          "the lag cycles, and")
    print("      in-degree is constant by construction -- so the lagged-only panel "
          "is near 0.5 either way.")
    print("      The artefact, and the fix, live in the contemporaneous DAG.\n")
    print(f"      {'configuration':<34s} {'equal_var':>9s} {'sd spread':>10s} "
          f"{'var':>7s} {'r2':>7s}")
    ctrl = [("lagged only (defaults), P=24", dict(P=24), "skel"),
            ("contemp_deg=2, P=12", dict(P=12, contemp=True, contemp_deg=2), "cont"),
            ("contemp_deg=3, P=24", dict(P=24, contemp=True, contemp_deg=3), "cont")]
    ctrl_res = {}
    for tag, cfg, which in ctrl:
        for ev in (False, True):
            vs, rs, sp = [], [], []
            for s in range(8):
                _, _, _, tr = gen_causal_panel(seed=s, equal_variance=ev, **cfg)
                sk = (tr.skeleton() if which == "skel"
                      else (np.abs(tr.A_contemp) != 0).T.astype(float))
                vs.append(var_sortability(tr.E_idio, sk))
                rs.append(r2_sortability(tr.E_idio, sk))
                sp.append(tr.meta["idio_sd_max"] / max(tr.meta["idio_sd_min"], 1e-12))
            ctrl_res[(tag, ev)] = float(np.mean(vs))
            print(f"      {tag:<34s} {str(ev):>9s} {np.mean(sp):>9.2f}x "
                  f"{np.mean(vs):>7.3f} {np.mean(rs):>7.3f}")
    for tag, _cfg, _which in ctrl[1:]:
        off, on = ctrl_res[(tag, False)], ctrl_res[(tag, True)]
        if not (off > 0.60 and on < off - 0.10):
            print(f"      !! the equal-variance fix is not doing its job on {tag} "
                  f"({off:.3f} -> {on:.3f})")
            ok = False
    print("      (scored on the CONTEMPORANEOUS skeleton for the contemp rows -- "
          "that is the DAG\n       whose topological order variance can leak.)")
    print("\n      CAVEAT, and it is a real one: r2_sortability is invariant to any "
          "per-node rescaling,\n      so the equal-variance fix cannot touch it -- "
          "identical columns above, exactly as\n      Reisach et al. (2023) predict. "
          "The DEFAULT lagged-only panel is clean on both\n      metrics (G1); the "
          "contemp=True variant is clean on var but leaks the order through\n"
          "      R^2 (~0.25/0.35, i.e. reversed). Anything reported on contemp=True "
          "panels must say so.")

    # ------------------------------------------------------- stability --- #
    print("\n[G2] STABILITY  (achieved reduced-form companion radius vs target)")
    cfgs = [dict(), dict(P=40, n_lags=3, deg=3), dict(contemp=True, contemp_deg=2),
            dict(n_lags=1), dict(n_lags=4, spec_radius=0.9),
            dict(hetero=0.5, P=16), dict(spec_radius=0.3), dict(deg=1, P=6)]
    for cfg in cfgs:
        _, _, _, tr = gen_causal_panel(seed=1, **cfg)
        m = tr.meta
        tag = ", ".join(f"{k}={v}" for k, v in cfg.items()) or "defaults"
        d = abs(m["rho_reduced"] - m["spec_radius"])
        print(f"      {tag:<34s} rho_red={m['rho_reduced']:.6f} "
              f"(target {m['spec_radius']}, |err|={d:.2e})  "
              f"rho_struct={m['rho_structural']:.4f}  "
              f"rho_ent_max={m['rho_entity_max']:.4f}")
        if d > 1e-6 or m["rho_entity_max"] >= 1.0:
            print("      !! stability violated")
            ok = False

    # ------------------------------------------- equal-variance convergence -- #
    print("\n[G2b] EQUAL-VARIANCE FIXED POINT  (max|sd-1| per round; the rescale is "
          "exact, so\n      round 1 lands on the fixed point and the rest are "
          "no-ops at machine precision)")
    for cfg in [dict(), dict(P=40, n_lags=3, deg=3), dict(contemp=True, contemp_deg=3),
                dict(hetero=0.5, P=16)]:
        _, _, _, tr = gen_causal_panel(seed=1, **cfg)
        m = tr.meta
        tag = ", ".join(f"{k}={v}" for k, v in cfg.items()) or "defaults"
        print(f"      {tag:<34s} trace={[f'{v:.1e}' for v in m['ev_trace']]}  "
              f"returned-sample max|sd-1|={m['idio_sd_dev']:.4f}")
        if m["ev_resid"] > 1e-10:
            print("      !! the equal-variance fixed point did not converge")
            ok = False

    # ------------------------------------------------------------- chi --- #
    print("\n[G3] REALISED COMMON-VARIANCE SHARE  (Var(Lam z)/(Var(Lam z)+Var(e)))")
    for chi in (0.0, 0.2, 0.6, 0.9):
        _, _, _, tr = gen_causal_panel(chi=chi, seed=2)
        m = tr.meta
        print(f"      chi={chi:<4} realised mean={m['chi_realised_mean']:.6f}  "
              f"min={m['chi_realised_min']:.6f}  max={m['chi_realised_max']:.6f}")
        if abs(m["chi_realised_mean"] - chi) > 1e-6:
            print("      !! chi not hit")
            ok = False

    # ------------------------------------------- container round-trips --- #
    print("\n[G4] CONTAINER ROUND-TRIP + LIBRARY COMPATIBILITY")
    X3d, X2d, meta, tr = gen_causal_panel(E=8, T=60, P=6, seed=3)
    same = np.array_equal(X3d.reshape(X3d.shape[0] * X3d.shape[1], X3d.shape[2]), X2d)
    print(f"      X3d.reshape(E*T,P) == X2d exactly : {same}")
    ok &= bool(same)
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
    import cafe                                                  # noqa: E402
    res = cafe.CAFE().run(X3d)
    print(f"      cafe.CAFE().run(X3d)   -> imputed {np.asarray(res.imputed).shape}")
    Xm = X2d.copy()
    M = mask_panel("mcar", X3d, 0.2, seed=0)
    Xm[M.reshape(X2d.shape)] = np.nan
    filled = cafe.impute(Xm, dict(meta))
    from harness import metrics as hmetrics                       # noqa: E402
    mm = hmetrics(X2d, np.asarray(filled, float), M.reshape(X2d.shape))
    print(f"      cafe.impute(X2d, meta) -> MAE={mm['mae']:.4f} on "
          f"{mm['n']} masked cells (harness.metrics accepted meta)")
    ok &= bool(np.all(np.isfinite(np.asarray(filled, float))))

    print("\n[G5] PER-ENTITY MASKING  (blocks must never straddle an entity edge)")
    X3d, X2d, meta, tr = gen_causal_panel(E=10, T=80, P=6, seed=4)
    for mech in ("mcar", "block", "blackout", "mar", "mnar"):
        M = mask_panel(mech, X3d, 0.30, seed=0)
        per_ent = M.reshape(M.shape[0], -1).mean(axis=1)
        print(f"      {mech:<9s} rate={M.mean():.3f}  per-entity "
              f"[{per_ent.min():.3f}, {per_ent.max():.3f}]")
    # a straddling block would show up as a run crossing row e*T-1 -> e*T
    M = mask_panel("block", X3d, 0.30, seed=0)
    M2 = M.reshape(-1, M.shape[2])
    Tn = X3d.shape[1]
    edges = [e * Tn for e in range(1, X3d.shape[0])]
    straddle = int(sum(np.sum(M2[r - 1] & M2[r]) for r in edges))
    from harness import mask_block                                # noqa: E402
    Ms = mask_block(X2d, 0.30, seed=0)
    straddle_old = int(sum(np.sum(Ms[r - 1] & Ms[r]) for r in edges))
    print(f"      cells masked on BOTH sides of an entity boundary: "
          f"mask_panel={straddle}  harness.mask_block(stacked)={straddle_old}")

    # -------------------------------------------- ORACLE graph recovery --- #
    print("\n[G6] ORACLE RECOVERY  (pooled OLS-VAR on truth.E_idio -> A_lag)")
    print("     if the oracle residual cannot recover its own graph, the generator "
          "is broken")
    cases = [("defaults", dict()),
             ("t3 noise", dict(noise="t3")),
             ("P=24,K=3,deg=3", dict(P=24, n_lags=3, deg=3)),
             ("chi=0.9 (factor-dominated)", dict(chi=0.9)),
             ("hetero=0.3", dict(hetero=0.3)),
             ("contemp DAG (reduced form)", dict(contemp=True, contemp_deg=2)),
             ("E=5,T=100 (small)", dict(E=5, T=100)),
             ("equal_variance=False", dict(equal_variance=False))]
    for tag, cfg in cases:
        aucs, aucx, lag_ok = [], [], []
        for s in range(3):
            _, _, _, tr = gen_causal_panel(seed=s, **cfg)
            _, _, sm = _oracle_var_recovery(tr)
            aucs.append(sm["auc"])
            aucx.append(sm["auc_cross"])
            lag_ok.append(sm["lag_accuracy"])
        a, ax, la = np.mean(aucs), np.mean(aucx), np.mean(lag_ok)
        flag = "" if a > 0.95 else "   <-- BELOW 0.95"
        print(f"      {tag:<28s} AUC={a:.4f}  AUC(cross-only)={ax:.4f}  "
              f"lag_acc={la:.3f}{flag}")
        if a <= 0.95 and tag != "contemp DAG (reduced form)":
            ok = False

    # thresholded recovery, to exercise the binary metrics end to end at a
    # TRUTH-FREE operating point (|t| > 4 on the OLS coefficients)
    _, _, _, tr = gen_causal_panel(seed=0)
    A_hat, A_t, _ = _oracle_var_recovery(tr)
    sm = cm.summarize_graph(tr.A_lag, A_hat, (A_t > 4.0).astype(float))
    print(f"      thresholded at |t|>4 (truth-free): P={sm['precision']:.3f} "
          f"R={sm['recall']:.3f} F1={sm['f1']:.3f} SHD={sm['shd']} "
          f"skelF1={sm['skeleton_f1']:.3f} lag_acc={sm['lag_accuracy']:.3f} "
          f"(true edges={sm['n_true']}/{sm['n_cells']})")

    # ------------------------------------------------- noise diagnostics -- #
    print("\n[G7] SHOCK DISTRIBUTION  (excess kurtosis of truth.U; se ~ "
          "sqrt(24/n); gauss is the NEGATIVE CONTROL)")
    for kind in NOISE_KINDS:
        ks, sk, vr = [], [], []
        for s in range(3):
            _, _, _, tr = gen_causal_panel(noise=kind, seed=s)
            u = tr.U.reshape(-1, tr.U.shape[-1])
            ks.append(_kurtosis(u))
            sk.append(_skew(u))
            vr.append(float(u.var()))
        n = tr.U.size
        z = np.mean(ks) / np.sqrt(24.0 / n)
        verdict = ("gaussian (orientation MUST fail)" if abs(z) < 6
                   else "clearly non-gaussian")
        print(f"      {kind:<8s} exc.kurt={np.mean(ks):>9.3f}  skew={np.mean(sk):>7.3f}"
              f"  z={z:>10.1f}  -> {verdict}")
        if kind == "gauss" and abs(z) >= 6:
            print("      !! the gaussian negative control is not gaussian")
            ok = False
        if kind in ("t3", "laplace", "mix") and np.mean(ks) < 1.0:
            print("      !! heavy-tailed family shows no excess kurtosis")
            ok = False

    # ---------------------------------------------------- factor contam --- #
    print("\n[G8] WHY THE RAW ARM IS CONTAMINATED  (mean |off-diag corr| of the "
          "contemporaneous cross-section)")
    for chi in (0.2, 0.6, 0.9):
        X3d, X2d, meta, tr = gen_causal_panel(chi=chi, seed=0)
        def mabs(M):
            Pn = M.shape[0]
            return float(np.abs(M[~np.eye(Pn, dtype=bool)]).mean())
        raw = mabs(np.corrcoef(X2d, rowvar=False))
        ora = mabs(np.corrcoef(tr.E_idio.reshape(-1, tr.E_idio.shape[-1]),
                               rowvar=False))
        print(f"      chi={chi}: raw panel={raw:.3f}   oracle residual={ora:.3f}")

    # -------------------------------------------------- variant smoke test -- #
    print("\n[G9] VARIANT MATRIX  (every switch runs, stays stable and stays "
          "recoverable)")
    variants = [
        ("shared_factors=False", dict(shared_factors=False)),
        ("no fixed effects", dict(entity_fe=False, time_fe=False)),
        ("season=48 (in CAFE's Fourier set)", dict(season=48)),
        ("season=33 (MISSPECIFIED)", dict(season=33)),
        ("season=(33,1.5) big amplitude", dict(season=(33, 1.5))),
        ("R=1 factor", dict(R=1)),
        ("chi=0 (no common factor)", dict(chi=0.0)),
        ("ar_self=False", dict(ar_self=False)),
        ("uniform noise, K=1", dict(noise="uniform", n_lags=1)),
        ("E=1 single entity, T=1000", dict(E=1, T=1000)),
    ]
    for tag, cfg in variants:
        X3d, X2d, meta, tr = gen_causal_panel(seed=0, **cfg)
        _, _, sm = _oracle_var_recovery(tr)
        m = tr.meta
        sm_flag = "" if m["season_period"] is None else (
            "  season_misspecified=" + str(m["season_misspecified"]))
        good = (np.all(np.isfinite(X3d)) and abs(m["rho_reduced"] - 0.7) < 1e-6
                and sm["auc"] > 0.95)
        print(f"      {tag:<34s} rho={m['rho_reduced']:.4f} "
              f"oracle AUC={sm['auc']:.4f} chi={m['chi_realised_mean']:.3f}"
              f"{sm_flag}  {'ok' if good else '!! PROBLEM'}")
        ok &= bool(good)
    print("      (oracle recovery is sample-size bound, not generator bound: the same "
          "E=1 panel\n       scores AUC 0.939 at T=200, 1.000 at T=1000 -- K*P "
          "regressors need the rows.)")

    print("\n" + "=" * 78)
    print("SELF-VALIDATION: " + ("ALL GATES PASS" if ok else "FAILURE -- see !! above"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
