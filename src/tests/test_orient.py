"""Contract tests for the edge-orientation layer (``cafe._orient`` + ``cafe._diagnostics``).

The layer's promise is narrow and testable: **orient an edge only where an
identification result actually holds, and abstain loudly everywhere else.**  So the
tests come in matched pairs -- a positive fixture where the identification result
holds and recovery is demanded, and a *negative control* on data where the result
provably does not hold and the code must sit at chance and say so.

What is asserted
----------------
1.  **Lagged edges are free.**  ``orient`` stamps ``'-->'`` / ``'time'`` / confidence 1
    on every lagged candidate with zero search.
2.  **LiNGAM order recovery.**  On linear non-Gaussian DAGs (Laplace, t3) with
    ``P in {5, 10, 20}``, mean Kendall tau against the true order > 0.9 and mean
    adjacency F1 > 0.9 against a *noisy* candidate skeleton.
3.  **The Gaussian negative control -- the load-bearing test.**  On a linear
    *Gaussian* DAG the order must be at chance (tau ~ 0, per-edge orientation
    accuracy ~ 0.5) and ``route`` must never select ``lingam``.  Crucially the
    accuracy must **not improve with n**: a leak (varsortability, R^2-sortability,
    an entropy estimator that keys on something other than non-Gaussianity) would
    show up as accuracy climbing with sample size, which is the one signature a
    single-``n`` test cannot see.
4.  **Approximate vs exact.**  A slow textbook DirectLiNGAM lives in this file and
    the approximation must agree with it on >= 80% of ordered pairs at P=8, much
    faster.
5.  **Cumulant orientation** beats chance on skewed data and is at chance -- *and
    abstains* -- on Gaussian data.
6.  **Rank test** fires on an injected latent common cause and stays quiet on a
    causally sufficient DAG.
7.  **DAGMA** returns an acyclic ``W`` (verified by topological sort) and refuses
    above ``max_nodes`` rather than pretending to be a global solver.
8.  **CPDAG.**  ``compelled_edges`` reproduces Meek's rules on hand-checked graphs,
    so the ``dagma`` branch can only emit arrows that every Markov-equivalent DAG
    shares.
9.  **Router** picks the intended branch on four purpose-built fixtures.
10. **Never fabricate.**  No branch that abstains may emit a directed mark.
11. **numpy only.**  Both modules import and run with scipy/sklearn/lingam/
    causal-learn/numba unimportable.

On the fixtures.  Every generator here is self-contained (this file must not depend
on sibling modules or on ``bench/``), and every simulated variable is put at exactly
**unit population variance**, so no method can win by reading the marginal scale.
That is the varsortability defence of Reisach et al., built into the fixture rather
than assumed away.

On the numbers.  The thresholds asserted below are *measured*, not aspirational, and
they were sized so the fixed-seed statistic clears the bar by at least ~3 Monte-Carlo
standard errors.  Two honest limits the measurements exposed and the tests encode:

* At P=5 the mean tau plateaus at ~0.93 and **does not improve with n** (checked to
  n=16000).  That residual is the approximation error of the non-iterative heuristic
  -- on ~20% of random 5-node DAGs it picks the wrong exogenous variable and no
  amount of data repairs it.  It is the price of dropping the O(m^3 n) refinement.
* The router's ``lingam`` row keys on mean ``|excess kurtosis| > 1``.  Down a long
  causal chain with r^2 = 0.5 per node the excess kurtosis of Laplace-driven
  variables decays to exactly 1.0, so deep chains sit on the threshold and route to
  ``unidentified``.  That is the conservative direction and is asserted as such, not
  papered over.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cafe._diagnostics import (  # noqa: E402
    DAGMA_MAX_NODES,
    LATENT_STRONG,
    NG_STRONG,
    NONLIN_STRONG,
    SKEW_STRONG,
    Diagnostics,
    _corr_from_cov,
    diagnose,
    numerical_rank,
    route,
    soft_rank_deficiency,
)
from cafe._orient import (  # noqa: E402
    CUM_MIN_CONF,
    Edge,
    _padded_batched_ols,
    approx_lingam_adjacency,
    approx_lingam_order,
    compelled_edges,
    cumulant_orient,
    dagma_local,
    entropy_maxent,
    orient,
    rank_evidence,
    topological_order,
)

_ORIENT_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "cafe", "_orient.py"))
_DIAG_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "cafe", "_diagnostics.py"))


# --------------------------------------------------------------------------- #
# Self-contained fixtures
# --------------------------------------------------------------------------- #
def _noise(kind, size, rng):
    """Unit-variance, zero-mean draws from a named law.

    Reference cumulants (excess kurtosis / skewness):
    ``gauss`` 0 / 0, ``uniform`` -1.2 / 0, ``laplace`` +3 / 0, ``t3`` undefined
    (infinite) / 0, ``expo`` +6 / +2, ``skewnormal`` ~+0.87 / ~+0.99.
    """
    if kind == "gauss":
        return rng.standard_normal(size)
    if kind == "laplace":
        return rng.laplace(size=size) / np.sqrt(2.0)
    if kind == "t3":
        return rng.standard_t(3, size=size) / np.sqrt(3.0)
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, size=size) * np.sqrt(3.0)
    if kind == "expo":
        return rng.exponential(size=size) - 1.0
    if kind == "skewnormal":
        # X = d|U| + sqrt(1-d^2) V is skew-normal.  Chosen deliberately: the router
        # sends *skewed but not kurtotic* data to the cumulant branch, and this is
        # the textbook family that is strongly asymmetric while its fourth cumulant
        # stays under NG_STRONG (|skew| ~ 0.99, excess kurtosis ~ 0.87).
        d = 0.995
        u = np.abs(rng.standard_normal(size))
        v = rng.standard_normal(size)
        x = d * u + np.sqrt(max(1.0 - d * d, 0.0)) * v
        m = d * np.sqrt(2.0 / np.pi)
        s = np.sqrt(max(1.0 - 2.0 * d * d / np.pi, 1e-12))
        return (x - m) / s
    raise ValueError(kind)


def _hamiltonian_dag(P, rng, extra_prob=0.3, lo=0.4, hi=0.9):
    """Random DAG whose topological order is **unique** (a spanning chain + extras).

    Uniqueness is not decoration.  Kendall tau against "the" true order is only a
    meaningful score when exactly one topological order exists: on a sparse DAG the
    relative order of two unconnected variables is not identifiable *by anything*,
    and scoring it would be scoring noise.  Returns ``(B, order)`` with ``B[i, j]``
    the effect of ``j`` on ``i`` (the LiNGAM row-is-target convention).
    """
    order = rng.permutation(P)
    B = np.zeros((P, P))
    for k in range(1, P):
        B[order[k], order[k - 1]] = rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi)
    for k in range(P):
        for m in range(k + 2, P):
            if rng.random() < extra_prob:
                B[order[m], order[k]] = rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi)
    return B, order


def _simulate_sem(B, order, n, rng, kind="laplace", r2=0.5):
    """Simulate ``x = Bx + e`` with every variable at exactly **unit population variance**.

    Each node's parent contribution is rescaled to carry variance ``r2`` and its
    noise ``1 - r2``, using the exact recursively-propagated covariance rather than
    a sample estimate.  Consequences that matter for the tests:

    * marginal variance is identical for every variable, so **varsortability is
      neutral by construction** -- nothing can recover the order from the scale;
    * the signal-to-noise ratio is the same at every depth, so a deep node is not
      silently easier or harder than a shallow one.

    Returns ``(X, B_scaled)``.
    """
    P = B.shape[0]
    Bs = B.copy()
    Sigma = np.zeros((P, P))
    X = np.zeros((n, P))
    for i in order:
        pa = np.flatnonzero(Bs[i])
        if pa.size:
            b = Bs[i, pa]
            v = float(b @ Sigma[np.ix_(pa, pa)] @ b)
            if v <= 1e-12:
                Bs[i, pa] = 0.0
                pa = np.zeros(0, dtype=int)
            else:
                Bs[i, pa] = b * np.sqrt(r2 / v)
        if pa.size:
            X[:, i] = X[:, pa] @ Bs[i, pa] + np.sqrt(1.0 - r2) * _noise(kind, n, rng)
            Sigma[i, :] = Bs[i, pa] @ Sigma[pa, :]
            Sigma[:, i] = Sigma[i, :]
        else:
            X[:, i] = _noise(kind, n, rng)
            Sigma[i, :] = 0.0
            Sigma[:, i] = 0.0
        Sigma[i, i] = 1.0
    return X, Bs


def _latent_panel(n, rng, P=6, lam=0.8, kind="gauss"):
    """Pure one-factor measurement model: ONE latent causes all ``P`` observed vars.

    This is the structure the FOFC / vanishing-tetrad test is stated for.  Note the
    test needs at least four indicators of the same latent: a latent confounding
    exactly two observed variables is observationally indistinguishable from a
    direct edge at second order and is undetectable by any covariance-based method.
    """
    L = _noise(kind, n, rng)
    X = np.empty((n, P))
    for k in range(P):
        X[:, k] = lam * L + np.sqrt(1.0 - lam ** 2) * _noise(kind, n, rng)
    return X


def _nonlinear_chain(n, rng, P=5, alpha=1.0, r2=0.6):
    """Chain with genuinely curved edges: ``tanh(2p) + alpha*(p^2 - 1)``.

    Both terms are needed.  ``tanh`` is monotone and a linear fit absorbs most of
    it; the Hermite ``p^2 - 1`` term is orthogonal to the linear one under a
    Gaussian parent, which is exactly what the nonlinearity probe measures.
    """
    X = np.empty((n, P))
    X[:, 0] = rng.standard_normal(n)
    for k in range(1, P):
        p = X[:, k - 1]
        f = np.tanh(2.0 * p) + alpha * (p * p - 1.0)
        f = (f - f.mean()) / f.std()
        y = np.sqrt(r2) * f + np.sqrt(1.0 - r2) * rng.standard_normal(n)
        X[:, k] = (y - y.mean()) / y.std()
    return X


def _noisy_skeleton(Bs, rng, extra=0.5):
    """True skeleton plus ~``extra`` x as many decoy pairs.

    Handing the adjacency step the *true* skeleton would make F1 near-trivially 1;
    the decoys make the coefficient step do real work, which is the realistic case
    (upstream screening returns a superset, not the truth).
    """
    P = Bs.shape[0]
    S = (Bs != 0) | (Bs != 0).T
    n_true = int(S.sum()) // 2
    iu = np.triu_indices(P, 1)
    free = np.flatnonzero(~S[iu])
    k = min(free.size, int(round(extra * max(n_true, 1))))
    if k:
        pick = rng.choice(free, size=k, replace=False)
        S[iu[0][pick], iu[1][pick]] = True
        S[iu[1][pick], iu[0][pick]] = True
    return S


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _kendall_tau(a, b):
    """Kendall tau-a between two orderings, in ``[-1, 1]``; 0 == chance."""
    a, b = np.asarray(a), np.asarray(b)
    n = a.size
    pa = np.empty(n, dtype=int)
    pb = np.empty(n, dtype=int)
    pa[a] = np.arange(n)
    pb[b] = np.arange(n)
    iu = np.triu_indices(n, 1)
    s = np.sign(pa[iu[0]] - pa[iu[1]]) * np.sign(pb[iu[0]] - pb[iu[1]])
    return float(s.mean())


def _precedence_agreement(ref_order, est_order):
    """Fraction of variable pairs the two orderings put in the same relative order."""
    n = len(ref_order)
    pr = np.empty(n, dtype=int)
    pe = np.empty(n, dtype=int)
    pr[np.asarray(ref_order)] = np.arange(n)
    pe[np.asarray(est_order)] = np.arange(n)
    iu = np.triu_indices(n, 1)
    return float(np.mean(np.sign(pr[iu[0]] - pr[iu[1]])
                         == np.sign(pe[iu[0]] - pe[iu[1]])))


def _edge_f1(true_A, est_A):
    T = np.asarray(true_A) != 0
    E = np.asarray(est_A) != 0
    tp = int((T & E).sum())
    fp = int((~T & E).sum())
    fn = int((T & ~E).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _edge_order_accuracy(Bs, est_order):
    """For every TRUE edge ``j -> i``, did the estimated order put ``j`` first?

    This is the quantity that must be 0.5 on Gaussian data.  Unlike tau it is
    defined edge-by-edge, so it measures exactly what an orientation would claim.
    """
    P = Bs.shape[0]
    pos = np.empty(P, dtype=int)
    pos[np.asarray(est_order)] = np.arange(P)
    tgt, src = np.nonzero(Bs)
    if tgt.size == 0:
        return np.nan
    return float(np.mean(pos[src] < pos[tgt]))


# --------------------------------------------------------------------------- #
# The slow reference: textbook DirectLiNGAM
# --------------------------------------------------------------------------- #
def _exact_direct_lingam_order(X):
    """DirectLiNGAM exactly as published (Shimizu et al. 2011, Algorithm 1).

    The ONE difference from :func:`cafe._orient.approx_lingam_order` is the
    iterative data-refinement step: after each variable is appended to the order,
    every remaining variable is replaced by its residual against it and **all** the
    entropies are recomputed from the refined data.  That inner pass over the ``n``
    samples inside every one of the ``m`` rounds is the ``O(m^3 n)`` cost the
    approximation removes by precomputing the pairwise statistic once.

    Deliberately written as plain nested loops: it is a correctness oracle, not a
    second implementation to be optimised.
    """
    Z = np.asarray(X, dtype=float).copy()
    Z = Z - Z.mean(0)
    Z = Z / np.where(Z.std(0) > 1e-12, Z.std(0), 1.0)
    P = Z.shape[1]
    U = list(range(P))
    K = []
    while len(U) > 1:
        best, best_M = None, -np.inf
        for i in U:
            xi = Z[:, i]
            xi = (xi - xi.mean()) / max(xi.std(), 1e-12)
            M = 0.0
            for j in U:
                if j == i:
                    continue
                xj = Z[:, j]
                xj = (xj - xj.mean()) / max(xj.std(), 1e-12)
                rho = float(np.mean(xi * xj))
                rij = xi - rho * xj
                rji = xj - rho * xi
                rij = rij / max(rij.std(), 1e-12)
                rji = rji / max(rji.std(), 1e-12)
                diff = ((entropy_maxent(xj) + entropy_maxent(rij))
                        - (entropy_maxent(xi) + entropy_maxent(rji)))
                M += min(0.0, diff) ** 2
            if -M > best_M:
                best_M, best = -M, i
        K.append(best)
        U.remove(best)
        xb = Z[:, best]
        xb = (xb - xb.mean()) / max(xb.std(), 1e-12)
        for j in U:                       # <-- the refinement the approximation drops
            Z[:, j] = Z[:, j] - np.dot(Z[:, j], xb) / np.dot(xb, xb) * xb
    K.extend(U)
    return np.asarray(K, dtype=np.intp)


# --------------------------------------------------------------------------- #
# 1. Lagged edges are free
# --------------------------------------------------------------------------- #
def test_lagged_edges_are_oriented_by_time_with_no_search():
    """Every lagged candidate comes back ``'-->'`` / ``'time'`` / confidence 1."""
    rng = np.random.default_rng(0)
    B, order = _hamiltonian_dag(6, rng, extra_prob=0.2)
    X, _ = _simulate_sem(B, order, 1500, rng, kind="laplace")
    lag = np.zeros((6, 6, 2))
    lag[1, 0, 0] = 0.4       # 0 -> 1 at lag 1
    lag[3, 2, 1] = -0.3      # 2 -> 3 at lag 2
    edges = orient(X, lagged_support=lag)
    assert len(edges) == 2
    assert {(e.source, e.target, e.lag) for e in edges} == {(0, 1, 1), (2, 3, 2)}
    for e in edges:
        assert e.orientation == "-->"
        assert e.identification_method == "time"
        assert e.confidence == 1.0
        assert e.confounding_prob == 0.0
        assert "time" in e.reason
    # a float support supplies the strength directly -- no computation at all
    assert {round(e.strength, 6) for e in edges} == {0.4, 0.3}


def test_lagged_support_accepts_every_documented_form():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 4))
    want = {(0, 1, 1), (2, 3, 2)}
    a3 = np.zeros((4, 4, 2), dtype=bool)
    a3[1, 0, 0] = True
    a3[3, 2, 1] = True
    forms = [
        a3,
        {1: np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
         2: np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])},
        np.array([[0, 1, 1], [2, 3, 2]]),
        np.array([[0, 1, 1, 0.5], [2, 3, 2, -0.25]]),
    ]
    for spec in forms:
        edges = orient(X, lagged_support=spec)
        assert {(e.source, e.target, e.lag) for e in edges} == want
        assert all(e.orientation == "-->" and e.identification_method == "time"
                   for e in edges)
        assert all(np.isfinite(e.strength) for e in edges)


# --------------------------------------------------------------------------- #
# 2. LiNGAM order + adjacency recovery
# --------------------------------------------------------------------------- #
# (replicates, n) per P.  Sized so the fixed-seed mean clears the 0.9 bar by
# >= 3 Monte-Carlo standard errors; small P needs more replicates because tau on
# 5 variables only takes 11 distinct values.
_ORDER_CFG = {5: (150, 4000), 10: (60, 4000), 20: (30, 4000)}


def _order_recovery_run(P, kind, r2=0.5, seed0=40000):
    R, n = _ORDER_CFG[P]
    taus, f1s = [], []
    for rep in range(R):
        rng = np.random.default_rng(seed0 + 977 * rep + 13 * P
                                    + (0 if kind == "laplace" else 7))
        B, order = _hamiltonian_dag(P, rng, extra_prob=min(0.35, 1.5 / P))
        X, Bs = _simulate_sem(B, order, n, rng, kind=kind, r2=r2)
        est = approx_lingam_order(X)
        Bhat = approx_lingam_adjacency(X, est, candidates=_noisy_skeleton(Bs, rng))
        taus.append(_kendall_tau(order, est))
        f1s.append(_edge_f1(Bs, Bhat))
    return np.asarray(taus), np.asarray(f1s)


@pytest.mark.parametrize("kind", ["laplace", "t3"])
@pytest.mark.parametrize("P", [5, 10, 20])
def test_lingam_recovers_order_and_adjacency_on_nongaussian_dags(P, kind):
    """Mean tau > 0.9 and mean adjacency F1 > 0.9 on linear non-Gaussian DAGs."""
    taus, f1s = _order_recovery_run(P, kind)
    assert taus.mean() > 0.90, f"{kind} P={P}: tau {taus.mean():.4f}"
    assert f1s.mean() > 0.90, f"{kind} P={P}: F1 {f1s.mean():.4f}"


def test_lingam_order_is_permutation_and_consistent_with_adjacency():
    """The returned order is a valid permutation and ``B`` is triangular in it."""
    rng = np.random.default_rng(11)
    B, order = _hamiltonian_dag(12, rng, extra_prob=0.2)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace")
    est = approx_lingam_order(X)
    assert sorted(est.tolist()) == list(range(12))
    Bhat = approx_lingam_adjacency(X, est, candidates=(Bs != 0) | (Bs != 0).T)
    # triangular in `est` => acyclic by construction, not by luck
    pos = np.empty(12, dtype=int)
    pos[est] = np.arange(12)
    tgt, src = np.nonzero(Bhat)
    assert np.all(pos[src] < pos[tgt])
    assert topological_order(Bhat) is not None


def test_lingam_adjacency_rejects_a_mismatched_order():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((200, 5))
    with pytest.raises(ValueError, match="order has"):
        approx_lingam_adjacency(X, np.arange(4))


# --------------------------------------------------------------------------- #
# 3. THE GAUSSIAN NEGATIVE CONTROL
# --------------------------------------------------------------------------- #
def _gaussian_control_run(n, R=200, P=8, seed0=60000):
    taus, accs, routes = [], [], []
    for rep in range(R):
        rng = np.random.default_rng(seed0 + 13 * rep)
        B, order = _hamiltonian_dag(P, rng, extra_prob=0.25)
        X, Bs = _simulate_sem(B, order, n, rng, kind="gauss", r2=0.5)
        est = approx_lingam_order(X)
        taus.append(_kendall_tau(order, est))
        accs.append(_edge_order_accuracy(Bs, est))
        routes.append(route(diagnose(X, skeleton=(Bs != 0) | (Bs != 0).T), P))
    return np.asarray(taus), np.asarray(accs), routes


def test_gaussian_control_order_is_at_chance():
    """On a linear GAUSSIAN DAG the recovered order must carry no information.

    A linear-Gaussian SEM is identified only up to its Markov equivalence class, so
    a method that appeared to recover the *order* here would be reading something
    it must not: the marginal scale (varsortability), the R^2 ranking, or a bug.
    The fixture puts every variable at unit population variance precisely to close
    the first of those doors, so anything left would be a genuine leak.
    """
    taus, accs, _ = _gaussian_control_run(3000)
    se_tau = taus.std() / np.sqrt(taus.size)
    se_acc = accs.std() / np.sqrt(accs.size)
    assert abs(taus.mean()) < 4.0 * se_tau, f"tau {taus.mean():+.4f} +- {se_tau:.4f}"
    assert abs(taus.mean()) < 0.10, f"tau {taus.mean():+.4f} is not chance"
    assert abs(accs.mean() - 0.5) < 4.0 * se_acc, \
        f"edge orientation accuracy {accs.mean():.4f} +- {se_acc:.4f}, expected 0.5"
    assert abs(accs.mean() - 0.5) < 0.05, f"accuracy {accs.mean():.4f} is not chance"


def test_gaussian_control_does_not_improve_with_sample_size():
    """The decisive leak detector.

    Sampling noise is symmetric about chance and shrinks with ``n``; a *leak* is a
    real signal in the data and therefore gets **easier** to exploit as ``n`` grows.
    So the diagnostic is not "is accuracy near 0.5 at one n" but "does accuracy
    climb with n".  Here it must not: the trend from n=1000 to n=10000 has to stay
    inside Monte-Carlo error.
    """
    out = {}
    for n in (1000, 10000):
        _, accs, _ = _gaussian_control_run(n, R=150)
        out[n] = (accs.mean(), accs.std() / np.sqrt(accs.size))
    (a_lo, se_lo), (a_hi, se_hi) = out[1000], out[10000]
    drift = a_hi - a_lo
    se = np.sqrt(se_lo ** 2 + se_hi ** 2)
    assert abs(drift) < 4.0 * se, (
        f"accuracy moved {drift:+.4f} (+-{se:.4f}) from n=1000 ({a_lo:.4f}) to "
        f"n=10000 ({a_hi:.4f}); a leak would show up exactly like this"
    )
    assert a_hi < 0.55, f"accuracy {a_hi:.4f} at n=10000 is above chance"


def test_router_never_selects_lingam_on_gaussian_data():
    """``route`` must refuse the LiNGAM branch when there is no non-Gaussianity."""
    _, _, routes = _gaussian_control_run(3000, R=60)
    assert "lingam" not in routes, f"router chose lingam on Gaussian data: {set(routes)}"
    assert set(routes) <= {"dagma", "pag", "unidentified"}, set(routes)


def test_gaussian_control_has_teeth():
    """The control is not vacuous: the SAME harness recovers the order on Laplace.

    Without this, "accuracy is 0.5 on Gaussian data" would also be satisfied by a
    function that always returns the identity permutation.
    """
    taus, accs = [], []
    for rep in range(40):
        rng = np.random.default_rng(60000 + 13 * rep)
        B, order = _hamiltonian_dag(8, rng, extra_prob=0.25)
        X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
        est = approx_lingam_order(X)
        taus.append(_kendall_tau(order, est))
        accs.append(_edge_order_accuracy(Bs, est))
    assert np.mean(taus) > 0.85, np.mean(taus)
    assert np.mean(accs) > 0.90, np.mean(accs)


# --------------------------------------------------------------------------- #
# 4. Approximate vs exact DirectLiNGAM
# --------------------------------------------------------------------------- #
def test_approx_lingam_agrees_with_exact_direct_lingam():
    """>= 80% pairwise-precedence agreement with the textbook algorithm at P=8."""
    agree = []
    for rep in range(6):
        rng = np.random.default_rng(4000 + 13 * rep)
        B, order = _hamiltonian_dag(8, rng, extra_prob=0.25)
        X, _ = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
        agree.append(_precedence_agreement(_exact_direct_lingam_order(X),
                                           approx_lingam_order(X)))
    assert np.mean(agree) >= 0.80, f"agreement {np.mean(agree):.3f}"
    assert np.min(agree) >= 0.70, f"worst-case agreement {np.min(agree):.3f}"


def test_approx_lingam_is_faster_than_exact():
    """The approximation must actually buy the speed it costs accuracy for.

    Both are pure numpy in this process, so the ratio understates the asymptotic
    ``O(m^3 n) -> O(m^3 + m^2 n)`` gain (the exact reference is loop-bound), but a
    large margin at P=8 is enough to show the precompute-and-lookup structure is
    real and not an equivalent amount of work rearranged.
    """
    rng = np.random.default_rng(4321)
    B, order = _hamiltonian_dag(8, rng, extra_prob=0.25)
    X, _ = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
    approx_lingam_order(X)                                  # warm up
    t0 = time.perf_counter()
    for _ in range(3):
        approx_lingam_order(X)
    t_approx = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    _exact_direct_lingam_order(X)
    t_exact = time.perf_counter() - t0
    assert t_exact > 3.0 * t_approx, f"exact {t_exact:.4f}s vs approx {t_approx:.4f}s"


# --------------------------------------------------------------------------- #
# 5. Cumulant orientation
# --------------------------------------------------------------------------- #
def _cumulant_pair_run(kind, n, R=60, beta=0.8, seed0=7000):
    """Bivariate ``0 -> 1``; returns (sign accuracy, mean confidence, #abstentions)."""
    hits, confs, abst = [], [], 0
    for rep in range(R):
        rng = np.random.default_rng(seed0 + 11 * rep)
        x = _noise(kind, n, rng)
        y = beta * x + np.sqrt(1.0 - beta ** 2) * _noise(kind, n, rng)
        sc, cf = cumulant_orient(np.column_stack([x, y]), [(0, 1)])
        hits.append(1.0 if sc[0] > 0 else 0.0)
        confs.append(float(cf[0]))
        abst += int(cf[0] < CUM_MIN_CONF)
    return float(np.mean(hits)), float(np.mean(confs)), abst


@pytest.mark.parametrize("kind", ["expo", "skewnormal"])
def test_cumulant_beats_chance_on_skewed_data(kind):
    acc, conf, abst = _cumulant_pair_run(kind, 4000)
    assert acc > 0.90, f"{kind}: sign accuracy {acc:.3f}"
    assert conf > CUM_MIN_CONF, f"{kind}: mean confidence {conf:.3f}"
    assert abst <= 3, f"{kind}: abstained on {abst}/60 clearly skewed pairs"


def test_cumulant_is_at_chance_and_abstains_on_gaussian_data():
    """The matched negative control: no cross-cumulant asymmetry exists to read.

    Above second order every cumulant of a Gaussian vanishes, so both direction
    residuals are pure noise and the contrast is symmetric about zero.  The test
    demands both halves: the *sign* must be a coin flip AND the confidence must sit
    under the abstention floor on every single replicate, so the branch emits
    ``o-o`` rather than a coin-flip arrow.
    """
    acc, conf, abst = _cumulant_pair_run("gauss", 4000)
    assert abs(acc - 0.5) < 0.20, f"Gaussian sign accuracy {acc:.3f} is not chance"
    assert abst == 60, f"cumulant branch failed to abstain on {60 - abst}/60 pairs"
    assert conf < CUM_MIN_CONF, f"mean confidence {conf:.3f} above the floor"


def test_cumulant_confidence_grows_with_sample_size_on_skewed_data():
    """Evidence accumulates where there is signal -- the mirror of the leak test."""
    _, c_lo, _ = _cumulant_pair_run("expo", 1000)
    _, c_hi, _ = _cumulant_pair_run("expo", 8000)
    assert c_hi > c_lo + 0.1, f"confidence {c_lo:.3f} -> {c_hi:.3f}"


def test_cumulant_confidence_does_not_grow_on_gaussian_data():
    """...and does not accumulate where there is none."""
    _, c_lo, _ = _cumulant_pair_run("gauss", 1000)
    _, c_hi, _ = _cumulant_pair_run("gauss", 8000)
    assert c_hi < CUM_MIN_CONF and c_lo < CUM_MIN_CONF, (c_lo, c_hi)


def test_cumulant_orients_a_skewed_chain():
    accs = []
    for rep in range(20):
        rng = np.random.default_rng(7700 + 11 * rep)
        B, order = _hamiltonian_dag(5, rng, extra_prob=0.0)
        X, Bs = _simulate_sem(B, order, 4000, rng, kind="expo", r2=0.55)
        tgt, src = np.nonzero(Bs)
        sc, _ = cumulant_orient(X, list(zip(src.tolist(), tgt.tolist())))
        accs.append(float(np.mean(sc > 0)))
    assert np.mean(accs) > 0.95, np.mean(accs)


def test_cumulant_orient_handles_an_empty_pair_list():
    sc, cf = cumulant_orient(np.random.default_rng(0).standard_normal((100, 3)), [])
    assert sc.shape == (0,) and cf.shape == (0,)


# --------------------------------------------------------------------------- #
# 6. Rank / latent-confounding test
# --------------------------------------------------------------------------- #
def test_rank_evidence_fires_on_an_injected_latent_common_cause():
    scores, routes = [], []
    for rep in range(20):
        rng = np.random.default_rng(10000 + 7 * rep)
        X = _latent_panel(4000, rng, P=6, lam=0.8)
        C = _corr_from_cov(np.cov(X, rowvar=False))
        d = diagnose(X)
        scores.append(rank_evidence(C, 0, 1, list(range(2, 6)), n_eff=d.n_eff))
        routes.append(route(d, 6))
    assert np.mean(scores) > 0.80, f"rank_evidence {np.mean(scores):.3f}"
    assert np.min(scores) > 0.50, f"worst rank_evidence {np.min(scores):.3f}"
    assert all(r == "pag" for r in routes), set(routes)


def test_rank_evidence_stays_quiet_on_a_causally_sufficient_dag():
    """The matched control: a sparse DAG with generic parameters has full-rank blocks.

    The threshold is deliberately loose.  This test is *not* a calibrated false
    positive rate -- there is no null distribution behind the tolerance -- and the
    probe is designed to over-fire, because its only consequence is abstention
    whereas a missed confounder would produce a fabricated arrow.  Measured
    false-fire rate of ``latent_evidence > LATENT_STRONG`` on these fixtures is
    0.00-0.12, consistent with the 0.00-0.20 recorded next to the constant.
    """
    scores, fires = [], 0
    for rep in range(20):
        rng = np.random.default_rng(11000 + 7 * rep)
        B, order = _hamiltonian_dag(6, rng, extra_prob=0.15)
        X, Bs = _simulate_sem(B, order, 4000, rng, kind="gauss", r2=0.5)
        d = diagnose(X, skeleton=(Bs != 0) | (Bs != 0).T)
        fires += int(d.latent_evidence > LATENT_STRONG)
        C = _corr_from_cov(np.cov(X, rowvar=False))
        tgt, src = np.nonzero(Bs)
        for s, t in zip(src, tgt):
            cond = [c for c in range(6) if c not in (int(s), int(t))]
            scores.append(rank_evidence(C, int(s), int(t), cond, n_eff=d.n_eff))
    scores = np.asarray(scores)
    assert scores.mean() < 0.40, f"mean rank_evidence {scores.mean():.3f}"
    assert np.mean(scores > 0.5) < 0.10, \
        f"{np.mean(scores > 0.5):.2%} of sufficient-DAG edges flagged as confounded"
    assert fires <= 4, f"latent_evidence false-fired on {fires}/20 sufficient DAGs"


def test_rank_evidence_is_impossible_with_too_few_witnesses():
    """Fewer than two conditioning variables cannot support a tetrad -- returns 0."""
    C = np.eye(4) * 0.5 + 0.5
    assert rank_evidence(C, 0, 1, []) == 0.0
    assert rank_evidence(C, 0, 1, [2]) == 0.0
    assert rank_evidence(C, 0, 1, [0, 1]) == 0.0          # self-references dropped


def test_numerical_rank_and_soft_deficiency_agree_on_clear_cases():
    rank1 = np.outer([1.0, 2.0], [3.0, 4.0])
    full = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert int(numerical_rank(rank1, rtol=1e-6)) == 1
    assert int(numerical_rank(full, rtol=1e-6)) == 2
    d = soft_rank_deficiency(np.stack([rank1, full]), rtol=0.05)
    assert d[0] > 0.9 and d[1] < 0.1


# --------------------------------------------------------------------------- #
# 7. DAGMA
# --------------------------------------------------------------------------- #
def test_dagma_returns_an_acyclic_matrix():
    """Verified by topological sort, not by trusting the barrier."""
    for rep in range(6):
        rng = np.random.default_rng(12000 + 17 * rep)
        B, order = _hamiltonian_dag(8, rng, extra_prob=0.2)
        X, Bs = _simulate_sem(B, order, 3000, rng, kind="gauss", r2=0.5)
        W = dagma_local(X, candidates=(Bs != 0) | (Bs != 0).T)
        assert W.shape == (8, 8)
        assert np.all(np.isfinite(W))
        assert np.all(np.diag(W) == 0.0)
        assert topological_order(W) is not None, "DAGMA returned a cyclic W"


def test_dagma_respects_the_candidate_support():
    rng = np.random.default_rng(13000)
    B, order = _hamiltonian_dag(8, rng, extra_prob=0.2)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="gauss", r2=0.5)
    S = (Bs != 0) | (Bs != 0).T
    W = dagma_local(X, candidates=S)
    assert not np.any(W[~S]), "DAGMA placed an edge outside the candidate skeleton"


def test_dagma_refuses_above_max_nodes():
    """A local tool must say so rather than silently becoming a global solver."""
    rng = np.random.default_rng(14000)
    X = rng.standard_normal((400, DAGMA_MAX_NODES + 1))
    with pytest.raises(ValueError, match="refuses"):
        dagma_local(X)
    ok = dagma_local(rng.standard_normal((400, DAGMA_MAX_NODES)))
    assert ok.shape == (DAGMA_MAX_NODES, DAGMA_MAX_NODES)


def test_dagma_handles_degenerate_supports():
    rng = np.random.default_rng(15000)
    X = rng.standard_normal((200, 4))
    assert dagma_local(X, candidates=np.zeros((4, 4), bool)).shape == (4, 4)
    assert not dagma_local(X, candidates=np.zeros((4, 4), bool)).any()
    assert dagma_local(rng.standard_normal((200, 1))).shape == (1, 1)


# --------------------------------------------------------------------------- #
# 8. CPDAG / Meek's rules
# --------------------------------------------------------------------------- #
def _dag(edges, P):
    A = np.zeros((P, P))
    for s, t in edges:
        A[t, s] = 1.0
    return A


def _compelled_pairs(A):
    return sorted((int(j), int(i)) for i, j in zip(*np.nonzero(compelled_edges(A))))


def test_compelled_edges_matches_hand_checked_cpdags():
    """Meek's rules, verified against graphs whose CPDAG can be read off by hand."""
    # A chain has no v-structure: BOTH edges are reversible inside the class.
    assert _compelled_pairs(_dag([(0, 1), (1, 2)], 3)) == []
    # A fork 1 <- 0 -> 2 is Markov equivalent to the chains: nothing compelled.
    assert _compelled_pairs(_dag([(0, 1), (0, 2)], 3)) == []
    # A collider with non-adjacent parents IS identified -- both arrows compelled.
    assert _compelled_pairs(_dag([(0, 2), (1, 2)], 3)) == [(0, 2), (1, 2)]
    # A fully connected triangle has no v-structure: nothing compelled.
    assert _compelled_pairs(_dag([(0, 1), (1, 2), (0, 2)], 3)) == []
    # Meek R1: the v-structure at 2 propagates along 2 - 3 (0 and 3 non-adjacent).
    assert _compelled_pairs(_dag([(0, 2), (1, 2), (2, 3)], 4)) == \
        [(0, 2), (1, 2), (2, 3)]
    # Meek R2: v-structure at 0 seeds 0->1 (R1); then 0->1->2 with 0-2 forces 0->2.
    # 1->2 stays reversible: flipping it creates no new v-structure (0 and 2 are
    # adjacent), so it is genuinely not identified.
    assert _compelled_pairs(_dag([(3, 0), (4, 0), (0, 1), (1, 2), (0, 2)], 5)) == \
        [(0, 1), (0, 2), (3, 0), (4, 0)]


def test_compelled_edges_is_a_subset_of_the_dag():
    for rep in range(10):
        rng = np.random.default_rng(16000 + rep)
        B, order = _hamiltonian_dag(7, rng, extra_prob=0.25)
        A = (B != 0)
        comp = compelled_edges(A.astype(float))
        assert np.all(comp <= A), "compelled an edge the DAG does not contain"


def test_topological_order_detects_cycles():
    assert topological_order(np.array([[0.0, 1.0], [1.0, 0.0]])) is None
    three_cycle = _dag([(0, 1), (1, 2), (2, 0)], 3)
    assert topological_order(three_cycle) is None
    assert topological_order(_dag([(0, 1), (1, 2)], 3)).tolist() == [0, 1, 2]
    assert topological_order(np.zeros((4, 4))) is not None


# --------------------------------------------------------------------------- #
# 9. The router: four purpose-built fixtures
# --------------------------------------------------------------------------- #
def _router_fixture(name, rng):
    """Return ``(X, skeleton, component_size)`` for one deliberately-built regime."""
    if name == "nongaussian_linear":
        B, order = _hamiltonian_dag(6, rng, extra_prob=0.2)
        X, Bs = _simulate_sem(B, order, 4000, rng, kind="laplace", r2=0.5)
        return X, (Bs != 0) | (Bs != 0).T, 6
    if name == "skewed_linear":
        # skew-normal: |skew| clears SKEW_STRONG while |excess kurtosis| stays
        # under NG_STRONG, which is precisely the cumulant branch's remit.
        B, order = _hamiltonian_dag(6, rng, extra_prob=0.15)
        X, Bs = _simulate_sem(B, order, 4000, rng, kind="skewnormal", r2=0.30)
        return X, (Bs != 0) | (Bs != 0).T, 6
    if name == "gaussian_linear":
        B, order = _hamiltonian_dag(6, rng, extra_prob=0.2)
        X, Bs = _simulate_sem(B, order, 4000, rng, kind="gauss", r2=0.5)
        return X, (Bs != 0) | (Bs != 0).T, 6
    if name == "latent_confounded":
        X = _latent_panel(4000, rng, P=6, lam=0.8)
        S = np.ones((6, 6), bool)
        np.fill_diagonal(S, False)
        return X, S, 6
    if name == "nonlinear":
        X = _nonlinear_chain(4000, rng, P=5, alpha=1.0)
        S = np.zeros((5, 5), bool)
        for k in range(4):
            S[k + 1, k] = S[k, k + 1] = True
        return X, S, 5
    raise ValueError(name)


@pytest.mark.parametrize("fixture,expected", [
    ("nongaussian_linear", "lingam"),
    ("skewed_linear", "cumulant"),
    ("gaussian_linear", "dagma"),
    ("latent_confounded", "pag"),
    ("nonlinear", "unidentified"),
])
def test_router_selects_the_intended_branch(fixture, expected):
    """Each regime must reach the branch whose assumption it actually satisfies."""
    picks = []
    for rep in range(15):
        rng = np.random.default_rng(17000 + 7 * rep)
        X, S, m = _router_fixture(fixture, rng)
        picks.append(route(diagnose(X, skeleton=S), m))
    hit = picks.count(expected)
    assert hit >= 13, f"{fixture}: chose {expected} only {hit}/15 times ({set(picks)})"


def test_router_short_circuits_lagged_edges():
    d = diagnose(np.random.default_rng(0).standard_normal((500, 4)))
    for lag in (1, 2, 7):
        assert route(d, 4, lag=lag) == "time"


def test_router_abstains_below_minimum_sample_size():
    rng = np.random.default_rng(18000)
    B, order = _hamiltonian_dag(5, rng, extra_prob=0.2)
    X, _ = _simulate_sem(B, order, 40, rng, kind="laplace", r2=0.5)
    assert route(diagnose(X), 5) == "unidentified"
    assert route(None, 5) == "unidentified"


def test_router_abstains_on_components_too_large_for_the_local_solver():
    """A near-Gaussian component above DAGMA_MAX_NODES gets no search at all."""
    d = Diagnostics(
        non_gaussianity=np.zeros(40), normality_stat=np.zeros(40),
        non_gaussianity_summary=0.0, skewness=np.zeros(40), skewness_summary=0.0,
        autocorr=np.zeros(40), factor_strength=0.0, nonlinearity=np.zeros(40),
        nonlinearity_summary=0.0, missing_dependence=0.0, latent_evidence=0.0,
        n_eff=5000.0, n_vars=40,
    )
    assert route(d, DAGMA_MAX_NODES) == "dagma"
    assert route(d, DAGMA_MAX_NODES + 1) == "unidentified"


def test_routing_thresholds_are_ordered_as_documented():
    """Guard the constants themselves: the table only makes sense in this order."""
    assert 0.0 < NG_STRONG
    assert 0.0 < SKEW_STRONG < 1.0
    assert 0.0 < NONLIN_STRONG < 1.0
    assert 0.0 < LATENT_STRONG < 1.0
    assert DAGMA_MAX_NODES >= 4


# --------------------------------------------------------------------------- #
# 10. orient(): never fabricate
# --------------------------------------------------------------------------- #
def test_orient_returns_exactly_one_edge_per_candidate_pair():
    rng = np.random.default_rng(19000)
    B, order = _hamiltonian_dag(7, rng, extra_prob=0.25)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
    S = (Bs != 0) | (Bs != 0).T
    edges = orient(X, contemp_candidates=S)
    assert len(edges) == int(S.sum()) // 2
    seen = {frozenset((e.source, e.target)) for e in edges}
    assert len(seen) == len(edges), "duplicate or self-loop edge emitted"
    for e in edges:
        assert e.source != e.target
        assert e.lag == 0
        assert np.isfinite(e.strength) and e.strength >= 0.0
        assert 0.0 <= e.confidence <= 1.0
        assert 0.0 <= e.confounding_prob <= 1.0
        assert isinstance(e, Edge)


def test_orient_never_emits_an_arrow_from_an_abstaining_branch():
    """``pag`` and ``unidentified`` may only produce ``o-o`` / ``<->``.

    This is the module's central promise.  A fabricated arrow on confounded or
    nonlinear data is a worse failure than no answer at all, so it is asserted
    directly rather than inferred from accuracy.
    """
    for fixture in ("latent_confounded", "nonlinear"):
        for rep in range(8):
            rng = np.random.default_rng(20000 + 7 * rep)
            X, S, _ = _router_fixture(fixture, rng)
            edges = orient(X, contemp_candidates=S)
            for e in edges:
                if e.identification_method in ("pag", "unidentified"):
                    assert e.orientation in ("o-o", "<->"), (fixture, e)
                    assert e.reason, "an abstention must record why"
                    if e.orientation in ("o-o", "<->"):
                        assert e.source < e.target, "symmetric mark not canonicalised"


def test_orient_recovers_a_nongaussian_graph_end_to_end():
    rng = np.random.default_rng(4242)
    B, order = _hamiltonian_dag(6, rng, extra_prob=0.2)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
    S = (Bs != 0) | (Bs != 0).T
    edges = orient(X, contemp_candidates=S)
    tgt, src = np.nonzero(Bs)
    truth = {(int(s), int(t)) for s, t in zip(src, tgt)}
    arrows = [e for e in edges if e.orientation == "-->"]
    assert len(arrows) >= 0.8 * len(truth), f"{len(arrows)} arrows for {len(truth)} edges"
    correct = sum(1 for e in arrows if (e.source, e.target) in truth)
    assert correct >= 0.9 * len(arrows), f"{correct}/{len(arrows)} arrows correct"
    assert all(e.identification_method == "lingam" for e in arrows)


def test_orient_routes_components_independently():
    """Two disconnected components with different laws must get different branches.

    Routing the whole graph on global statistics is exactly what the per-component
    design exists to avoid, so a shared branch here would be a real regression.
    """
    rng = np.random.default_rng(21000)
    Ba, oa = _hamiltonian_dag(5, rng, extra_prob=0.2)
    Xa, Bsa = _simulate_sem(Ba, oa, 4000, rng, kind="laplace", r2=0.5)
    Xb = _latent_panel(4000, rng, P=6, lam=0.8)
    X = np.column_stack([Xa, Xb])
    S = np.zeros((11, 11), bool)
    S[:5, :5] = (Bsa != 0) | (Bsa != 0).T
    S[5:, 5:] = ~np.eye(6, dtype=bool)
    edges = orient(X, contemp_candidates=S)
    m_a = {e.identification_method for e in edges if e.source < 5 and e.target < 5}
    m_b = {e.identification_method for e in edges if e.source >= 5 and e.target >= 5}
    assert m_a == {"lingam"}, m_a
    assert m_b == {"pag"}, m_b


def test_orient_handles_degenerate_inputs():
    rng = np.random.default_rng(22000)
    X = rng.standard_normal((200, 4))
    assert orient(X) == []                                    # no candidates at all
    assert orient(X, contemp_candidates=np.zeros((4, 4), bool)) == []
    assert len(orient(X, contemp_candidates=[(0, 1)])) == 1    # isolated single pair
    assert approx_lingam_order(np.zeros((10, 0))).size == 0
    assert approx_lingam_order(rng.standard_normal((10, 1))).tolist() == [0]


def test_orient_survives_a_perfectly_collinear_column():
    """A duplicated column must not raise, and must not be given a direction."""
    rng = np.random.default_rng(23000)
    Z = rng.standard_normal((400, 3))
    Z[:, 2] = Z[:, 0]
    edges = orient(Z, contemp_candidates=np.ones((3, 3), bool))
    assert len(edges) == 3
    assert all(np.isfinite(e.strength) and np.isfinite(e.confidence) for e in edges)


@pytest.mark.parametrize("frac", [0.05, 0.30])
def test_orient_degrades_gracefully_with_missing_data(frac):
    """True == observed; missing cells must never leak a NaN into the output."""
    rng = np.random.default_rng(24000)
    B, order = _hamiltonian_dag(8, rng, extra_prob=0.2)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
    obs = rng.random(X.shape) > frac
    Xm = X.copy()
    Xm[~obs] = np.nan
    S = (Bs != 0) | (Bs != 0).T
    edges = orient(Xm, contemp_candidates=S, obs=obs)
    assert len(edges) == int(S.sum()) // 2
    assert all(np.isfinite(e.strength) and np.isfinite(e.confidence) for e in edges)
    tgt, src = np.nonzero(Bs)
    truth = {(int(s), int(t)) for s, t in zip(src, tgt)}
    arrows = [e for e in edges if e.orientation == "-->"]
    correct = sum(1 for e in arrows if (e.source, e.target) in truth)
    assert correct >= 0.8 * len(arrows), f"{correct}/{len(arrows)}"


def test_panel_input_is_accepted():
    rng = np.random.default_rng(25000)
    B, order = _hamiltonian_dag(5, rng, extra_prob=0.2)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
    panel = X.reshape(3, 1000, 5)
    S = (Bs != 0) | (Bs != 0).T
    flat = orient(X, contemp_candidates=S)
    pan = orient(panel, contemp_candidates=S)
    assert len(flat) == len(pan)
    assert [(e.source, e.target, e.orientation) for e in flat] == \
           [(e.source, e.target, e.orientation) for e in pan]


# --------------------------------------------------------------------------- #
# 11. Numerical internals
# --------------------------------------------------------------------------- #
def test_entropy_maxent_is_a_negentropy_on_both_sides_of_gaussian():
    """The Gaussian must be the MAXIMUM, for sub- and super-Gaussian laws alike.

    This is the regression test for a real bug found in this module: the G1
    contrast was centred on the Euler-Mascheroni constant 0.5772 instead of
    ``E[log cosh z] = 0.3746`` for ``z ~ N(0,1)``.  Because the penalty is
    *quadratic* in the contrast, that moves the zero of the negentropy off the
    Gaussian, and every law whose ``E[log cosh]`` falls between the two constants
    -- the whole sub-Gaussian half of the space -- was scored as having MORE
    entropy than a Gaussian of the same variance, which is impossible.

    A super-Gaussian-only fixture cannot see this (Laplace sits on the correct
    side of both constants), which is exactly why the check below tests uniform
    and two-point laws as well.
    """
    rng = np.random.default_rng(26000)
    g = rng.standard_normal(400000)
    h_g = float(entropy_maxent(g))
    # correctly centred, the estimate at a Gaussian IS the Gaussian entropy
    assert abs(h_g - 0.5 * np.log(2 * np.pi * np.e)) < 1e-3, h_g
    for name, v in [
        ("laplace", rng.laplace(size=400000) / np.sqrt(2.0)),          # super-G
        ("t5", rng.standard_t(5, 400000) / np.sqrt(5 / 3)),            # super-G
        ("uniform", rng.uniform(-1, 1, 400000) * np.sqrt(3.0)),        # SUB-G
        ("twopoint", (rng.integers(0, 2, 400000) * 2 - 1) * 1.0),      # SUB-G
        ("expo", rng.exponential(size=400000) - 1.0),                  # skewed
    ]:
        h = float(entropy_maxent((v - v.mean()) / v.std()))
        assert h < h_g, f"{name}: H={h:.4f} exceeds the Gaussian's {h_g:.4f}"
    assert abs(float(entropy_maxent(-g)) - h_g) < 5e-3       # sign symmetric
    assert entropy_maxent(np.stack([g, g])).shape == (2,)    # vectorises over rows


def test_lingam_does_not_reverse_the_order_on_subgaussian_noise():
    """Uniform noise is a linear non-Gaussian SEM and must be recovered FORWARD.

    The companion regression test to the one above, at the level that actually
    matters.  With the mis-centred contrast this fixture returned tau = -0.93:
    a confidently, almost perfectly *reversed* causal order -- every arrow
    backwards, reported with full confidence.  Uniform noise is a standard test
    law in the LiNGAM literature, not a corner case.
    """
    taus = []
    for rep in range(40):
        rng = np.random.default_rng(50000 + 13 * rep)
        B, order = _hamiltonian_dag(8, rng, extra_prob=0.25)
        X, _ = _simulate_sem(B, order, 4000, rng, kind="uniform", r2=0.5)
        taus.append(_kendall_tau(order, approx_lingam_order(X)))
    assert np.mean(taus) > 0.80, (
        f"tau {np.mean(taus):+.4f} on sub-Gaussian noise; a strongly NEGATIVE value "
        "means the entropy contrast is mis-centred and the order is inverted"
    )


def test_batched_ols_matches_lstsq_on_ragged_supports():
    """The padded batched solve must be exactly ordinary least squares, per row."""
    rng = np.random.default_rng(27000)
    n, P = 2000, 6
    Z = rng.standard_normal((n, P))
    Z = (Z - Z.mean(0)) / Z.std(0)
    S = (Z.T @ Z) / n
    allowed = np.zeros((P, P), bool)
    allowed[1, [0]] = True
    allowed[2, [0, 1]] = True
    allowed[3, [1]] = True
    allowed[5, [0, 2, 4]] = True
    B = _padded_batched_ols(S, allowed)
    for i in range(P):
        pa = np.flatnonzero(allowed[i])
        off = np.ones(P, bool)
        off[pa] = False
        assert np.allclose(B[i, off], 0.0), f"row {i} leaked outside its support"
        if pa.size:
            beta, *_ = np.linalg.lstsq(Z[:, pa], Z[:, i], rcond=None)
            assert np.allclose(B[i, pa], beta, atol=1e-7), i


def test_orient_is_deterministic():
    """Same input, same output -- no unseeded RNG anywhere on the path."""
    rng = np.random.default_rng(28000)
    B, order = _hamiltonian_dag(6, rng, extra_prob=0.25)
    X, Bs = _simulate_sem(B, order, 2000, rng, kind="laplace", r2=0.5)
    S = (Bs != 0) | (Bs != 0).T
    a = orient(X, contemp_candidates=S)
    b = orient(X, contemp_candidates=S)
    assert [(e.source, e.target, e.orientation, e.strength) for e in a] == \
           [(e.source, e.target, e.orientation, e.strength) for e in b]
    assert np.array_equal(approx_lingam_order(X), approx_lingam_order(X))


def test_standardisation_closes_the_varsortability_channel():
    """Blowing up the marginal scales must not move a single orientation.

    Reisach et al. showed simulated DAGs routinely leak their topological order
    through the marginal variances alone.  Both modules standardise first; this is
    the test that the standardisation is actually load-bearing.
    """
    rng = np.random.default_rng(29000)
    B, order = _hamiltonian_dag(6, rng, extra_prob=0.25)
    X, Bs = _simulate_sem(B, order, 3000, rng, kind="laplace", r2=0.5)
    S = (Bs != 0) | (Bs != 0).T
    scale = np.exp(np.linspace(-2.0, 2.0, 6))               # 55x spread of variances
    base = orient(X, contemp_candidates=S)
    scaled = orient(X * scale, contemp_candidates=S)
    assert [(e.source, e.target, e.orientation) for e in base] == \
           [(e.source, e.target, e.orientation) for e in scaled]
    assert np.array_equal(approx_lingam_order(X), approx_lingam_order(X * scale))


# --------------------------------------------------------------------------- #
# 12. numpy only
# --------------------------------------------------------------------------- #
def test_module_sources_have_no_forbidden_imports():
    for path in (_ORIENT_PY, _DIAG_PY):
        src = open(path).read()
        for bad in ("scipy", "sklearn", "numba", "numexpr", "pandas", "polars",
                    "torch", "lingam", "causallearn", "networkx"):
            assert f"import {bad}" not in src, (path, bad)
            assert f"from {bad}" not in src, (path, bad)


def test_imports_and_runs_with_numpy_alone():
    """Exercise both modules in a subprocess where every non-numpy dep is blocked."""
    code = (
        "import sys, importlib, importlib.abc\n"
        "BAD = {'scipy','sklearn','numba','numexpr','pandas','polars','torch',"
        "'matplotlib','lingam','causallearn','networkx'}\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BAD:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(_ORIENT_PY))!r})\n"
        "import numpy as np\n"
        "from cafe._orient import orient, approx_lingam_order, dagma_local\n"
        "from cafe._diagnostics import diagnose, route\n"
        "rng = np.random.default_rng(0)\n"
        "e = rng.laplace(size=(600, 5))\n"
        "X = np.zeros_like(e)\n"
        "X[:, 0] = e[:, 0]\n"
        "for k in range(1, 5):\n"
        "    X[:, k] = 0.7 * X[:, k - 1] + e[:, k]\n"
        "S = np.zeros((5, 5), bool)\n"
        "for k in range(4):\n"
        "    S[k + 1, k] = S[k, k + 1] = True\n"
        "assert sorted(approx_lingam_order(X).tolist()) == list(range(5))\n"
        "assert route(diagnose(X, skeleton=S), 5) in "
        "('lingam','cumulant','pag','dagma','unidentified')\n"
        "assert dagma_local(X, candidates=S).shape == (5, 5)\n"
        "assert len(orient(X, contemp_candidates=S)) == 4\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
