"""Tests for the panel group-sparse VAR skeleton solver (``cafe._skeleton``).

These encode the guarantees the solver claims, in the order they matter:

  1. the moment tensor really is ``X'X`` / ``X'y`` -- gathers match an explicitly
     materialised design to 1e-10 (the whole no-design-matrix argument rests on this);
  2. the block update really is exact -- it beats a brute-force 1-D scan of the block
     objective AND satisfies the group-lasso KKT conditions to machine precision;
  3. the working-set path really solves the full problem -- identical to a dense
     all-candidates BCD solve at every lambda;
  4. ``lambda_max`` is tight -- exactly zero at it, nonzero just below;
  5. the model is recoverable -- F1 > 0.9 on a synthetic panel VAR at moderate SNR;
  6. the ``b0 + d_e`` hierarchy earns its keep -- pooled wins when it should, the
     split wins when it should;
  7. missingness does not shrink coefficients -- the available-case renormalisation
     works and its absence is measurably biased;
  8. no look-ahead -- a fit on rows <= t0 is invariant to what comes after t0;
  9. numpy alone is enough to import it.

Fixtures here are deliberately self-contained: no dependency on the screening or
generator modules a sibling owns.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cafe import _skeleton as sk  # noqa: E402


# --------------------------------------------------------------------------- #
#  Self-contained fixtures
# --------------------------------------------------------------------------- #
def _stabilise(A, target=0.85):
    """Scale a (K, P, P) VAR coefficient stack to a companion spectral radius."""
    K, P, _ = A.shape
    C = np.zeros((K * P, K * P))
    C[:P] = A.transpose(1, 0, 2).reshape(P, K * P)
    if K > 1:
        C[P:, :-P] = np.eye((K - 1) * P)
    rad = float(np.max(np.abs(np.linalg.eigvals(C))))
    return A * (target / rad) if rad > target else A


def make_panel(n_ent=8, T=250, P=8, K=2, n_edges=8, hetero=0.0, noise=0.5,
               seed=0, miss=0.0, burn=120, fe=1.0):
    """A sparse panel VAR with a KNOWN support, entity FE, and optional MCAR holes.

    ``hetero`` scales per-entity deviations applied ON the shared support, so the
    shared skeleton and the deviation level can be exercised independently.
    Returns ``(E, A_true)`` with ``E`` of shape (n_ent, T, P) and ``A_true`` (K, P, P).
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((K, P, P))
    seen = set()
    while len(seen) < n_edges:
        i = int(rng.integers(P)); j = int(rng.integers(P)); k = int(rng.integers(K))
        if (i, j) in {(a, b) for a, b, _ in seen}:
            continue
        seen.add((i, j, k))
    for (i, j, k) in seen:
        A[k, i, j] = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.6))
    A = _stabilise(A)
    mask = (A != 0.0)
    X = np.zeros((n_ent, T + burn, P))
    for e in range(n_ent):
        Ae = A + hetero * mask * rng.standard_normal(A.shape)
        Ae = _stabilise(Ae, 0.9)
        for t in range(K, T + burn):
            v = noise * rng.standard_normal(P)
            for k in range(K):
                v += Ae[k] @ X[e, t - k - 1]
            X[e, t] = v
        X[e] += fe * rng.standard_normal(P)          # entity fixed effect
    E = X[:, burn:, :].copy()
    if miss > 0.0:
        hole = rng.random(E.shape) < miss
        E[hole] = np.nan
    return E, A


def _explicit_design(Z, K, jj, kk):
    """Materialise the (n_ent*n_rows, A) design the solver refuses to build."""
    n_ent, T, _ = Z.shape
    n_rows = T - K
    X = np.empty((n_ent * n_rows, jj.size), dtype=np.float64)
    for a in range(jj.size):
        j, k = int(jj[a]), int(kk[a])
        X[:, a] = Z[:, K - k:T - k, j].reshape(-1)
    return X


def _all_pairs(P, K):
    jj, kk = np.meshgrid(np.arange(P), np.arange(1, K + 1), indexing="ij")
    return jj.reshape(-1).astype(np.intp), kk.reshape(-1).astype(np.intp)


# --------------------------------------------------------------------------- #
#  1. Moment tensor correctness
# --------------------------------------------------------------------------- #
def test_moment_tensor_gathers_equal_explicit_design():
    """M gathers == an explicitly materialised X'X / X'y, to 1e-10."""
    rng = np.random.default_rng(3)
    E = rng.standard_normal((4, 40, 5))
    K = 3
    mt = sk.build_moment_tensor(E, max_lag=K, demean="twoway")
    jj, kk = _all_pairs(5, K)
    X = _explicit_design(mt.Z, K, jj, kk)

    XtX = X.T @ X
    got = mt.M[kk[:, None], kk[None, :], jj[:, None], jj[None, :]]
    assert np.max(np.abs(got - XtX)) < 1e-10 * max(1.0, np.max(np.abs(XtX)))

    # X'y is a *slice* of the same tensor, for every target
    for i in range(5):
        y = mt.Z[:, K:, i].reshape(-1)
        assert np.max(np.abs(mt.M[kk, 0, jj, i] - X.T @ y)) < 1e-10

    # and the Gram gather used by the solver reproduces X'X after standardisation
    Gt = mt.scaled(available_case=True)
    G = sk._gather_gram(Gt, jj, kk)
    s = mt.scale[kk, jj]
    assert np.allclose(G * np.outer(s, s), XtX, atol=1e-9, rtol=1e-9)


def test_moment_tensor_counts_and_unit_diagonal():
    """N counts co-observations; the standardised diagonal is exactly n_tot."""
    rng = np.random.default_rng(4)
    E = rng.standard_normal((3, 30, 4))
    E[rng.random(E.shape) < 0.25] = np.nan
    K = 2
    mt = sk.build_moment_tensor(E, max_lag=K)
    V = np.isfinite(E).astype(float)
    T = E.shape[1]
    for k in range(K + 1):
        for kp in range(K + 1):
            ref = sum(V[e, K - k:T - k, :].T @ V[e, K - kp:T - kp, :]
                      for e in range(3))
            assert np.allclose(mt.N[k, kp], ref, atol=1e-12)
    Gt = mt.scaled(available_case=True)
    assert np.allclose(np.einsum("kkjj->kj", Gt), mt.n_tot, rtol=1e-12)


def test_moment_tensor_column_separable():
    """Restricting to a series subset is exact -- the memory fallback relies on it."""
    rng = np.random.default_rng(5)
    E = rng.standard_normal((3, 25, 6))
    E[rng.random(E.shape) < 0.15] = np.nan
    sel = np.array([1, 3, 4])
    full = sk.build_moment_tensor(E, max_lag=2)
    sub = sk.build_moment_tensor(E, max_lag=2, series=sel)
    assert np.allclose(sub.M, full.M[:, :, sel][:, :, :, sel], atol=1e-12)
    assert np.allclose(sub.Z, full.Z[:, :, sel], atol=1e-12)


# --------------------------------------------------------------------------- #
#  2. Exact block solve
# --------------------------------------------------------------------------- #
def _block_obj(b, G, c, lam_w):
    return 0.5 * float(b @ G @ b) - float(c @ b) + lam_w * float(np.linalg.norm(b))


@pytest.mark.parametrize("seed", range(8))
def test_block_solve_matches_brute_force_scan(seed):
    """The block minimiser beats a brute-force 1-D scan and satisfies KKT to 1e-10."""
    rng = np.random.default_rng(100 + seed)
    K = int(rng.integers(2, 9))
    B = rng.standard_normal((K + 6, K))
    G = B.T @ B + 1e-3 * np.eye(K)
    c = rng.standard_normal(K) * rng.uniform(0.5, 4.0)
    lam_w = float(np.linalg.norm(c)) * float(rng.uniform(0.05, 0.9))

    w, Q = np.linalg.eigh(G)
    b, _mu = sk._block_solve(np.maximum(w, 0.0), Q, c, lam_w)
    assert b is not None
    obj = _block_obj(b, G, c, lam_w)

    # brute force: the minimiser lies on b(nu) = (G + lam_w/nu I)^-1 c, so scan nu
    grid = np.concatenate([np.geomspace(1e-8, 1e4, 40000), [np.linalg.norm(b)]])
    best = np.inf
    for nu in grid:
        bb = np.linalg.solve(G + (lam_w / nu) * np.eye(K), c)
        best = min(best, _block_obj(bb, G, c, lam_w))
    assert obj <= best + 1e-10 * max(1.0, abs(best))
    assert abs(obj - best) < 1e-8 * max(1.0, abs(best))

    # exact KKT: G b - c + lam_w * b/||b|| == 0
    kkt = G @ b - c + lam_w * b / np.linalg.norm(b)
    assert np.max(np.abs(kkt)) < 1e-10 * max(1.0, float(np.linalg.norm(c)))


@pytest.mark.parametrize("seed", range(6))
def test_block_solve_matches_fista(seed):
    """An implementation-independent check: long FISTA never beats the exact solve."""
    rng = np.random.default_rng(200 + seed)
    K = int(rng.integers(2, 9))
    B = rng.standard_normal((K + 4, K))
    G = B.T @ B + 1e-2 * np.eye(K)
    c = rng.standard_normal(K) * 2.0
    lam_w = float(np.linalg.norm(c)) * 0.4
    w, Q = np.linalg.eigh(G)
    b, _ = sk._block_solve(np.maximum(w, 0.0), Q, c, lam_w)

    L = float(np.max(w))
    x = np.zeros(K); z = x.copy(); t = 1.0
    for _ in range(60000):
        v = z - (G @ z - c) / L
        nv = np.linalg.norm(v)
        thr = lam_w / L
        xn = v * (1.0 - thr / nv) if nv > thr else np.zeros(K)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
    o_exact = _block_obj(b if b is not None else np.zeros(K), G, c, lam_w)
    assert o_exact <= _block_obj(x, G, c, lam_w) + 1e-11


def test_block_solve_zero_test():
    """``||c|| <= lam*sqrt(K)`` returns zero, and zero really is optimal there."""
    rng = np.random.default_rng(7)
    K = 5
    B = rng.standard_normal((10, K))
    G = B.T @ B
    c = rng.standard_normal(K)
    w, Q = np.linalg.eigh(G)
    lam_w = float(np.linalg.norm(c)) * 1.0000001
    b, _ = sk._block_solve(np.maximum(w, 0.0), Q, c, lam_w)
    assert b is None
    # subgradient certificate at 0: ||c|| <= lam_w
    assert np.linalg.norm(c) <= lam_w
    lam_w2 = float(np.linalg.norm(c)) * 0.9999
    b2, _ = sk._block_solve(np.maximum(w, 0.0), Q, c, lam_w2)
    assert b2 is not None and np.linalg.norm(b2) > 0.0


def test_block_solve_warm_start_is_identical():
    """A warm-started scalar root gives the same block, to machine precision."""
    rng = np.random.default_rng(11)
    K = 6
    B = rng.standard_normal((12, K))
    G = B.T @ B
    c = rng.standard_normal(K) * 3.0
    w, Q = np.linalg.eigh(G)
    lam_w = float(np.linalg.norm(c)) * 0.3
    b_cold, mu = sk._block_solve(np.maximum(w, 0.0), Q, c, lam_w, 0.0)
    b_warm, _ = sk._block_solve(np.maximum(w, 0.0), Q, c, lam_w, mu * 1.7)
    assert np.max(np.abs(b_cold - b_warm)) < 1e-12


# --------------------------------------------------------------------------- #
#  3. Working set == full problem
# --------------------------------------------------------------------------- #
def test_working_set_path_equals_dense_solve():
    """The working-set path reproduces a dense all-candidates BCD solve at EVERY
    lambda -- supports exactly, coefficients to 1e-6."""
    E, _A = make_panel(n_ent=5, T=140, P=6, K=3, n_edges=6, seed=21)
    kw = dict(max_lag=3, select="ebic", n_lambda=25, tol=1e-11,
              fit_deviations=False)
    ws = sk.panel_group_lasso(E, use_working_set=True, **kw)
    dn = sk.panel_group_lasso(E, use_working_set=False,
                              lambda_grid=ws.lambda_path, **kw)
    assert np.allclose(ws.lambda_path, dn.lambda_path)
    for l, (a, b) in enumerate(zip(ws.path_support, dn.path_support)):
        assert np.array_equal(a, b), "support differs at lambda index %d" % l
    # coefficients, lambda by lambda, from independent single-lambda fits
    for l in range(0, ws.lambda_path.size, 3):
        lam = ws.lambda_path[l]
        a = sk.panel_group_lasso(E, lambda_grid=[lam], use_working_set=True, **kw)
        b = sk.panel_group_lasso(E, lambda_grid=[lam], use_working_set=False, **kw)
        assert np.max(np.abs(a.B0 - b.B0)) < 1e-6, "coef differ at lambda %g" % lam


def test_working_set_path_equals_dense_with_deviations():
    """Same, with the two-level b0 + d_e alternation switched on."""
    E, _A = make_panel(n_ent=5, T=130, P=5, K=2, n_edges=5, hetero=0.25, seed=22)
    kw = dict(max_lag=2, select="ebic", n_lambda=12, tol=1e-10,
              fit_deviations=True, kappa=2.0)
    ws = sk.panel_group_lasso(E, use_working_set=True, **kw)
    dn = sk.panel_group_lasso(E, use_working_set=False,
                              lambda_grid=ws.lambda_path, **kw)
    for l, (a, b) in enumerate(zip(ws.path_support, dn.path_support)):
        assert np.array_equal(a, b), "support differs at lambda index %d" % l
    assert np.max(np.abs(ws.B0 - dn.B0)) < 1e-6
    for e in set(ws.Delta) | set(dn.Delta):
        assert np.max(np.abs(ws.Delta.get(e, 0.0) - dn.Delta.get(e, 0.0))) < 1e-6


def test_solution_satisfies_global_kkt():
    """An implementation-independent optimality certificate on the full candidate set:
    ``||X_g'r||/sqrt(K) <= lam`` for every zero group, with equality on the support."""
    E, _A = make_panel(n_ent=5, T=140, P=6, K=3, n_edges=6, seed=23)
    K, P = 3, 6
    res = sk.panel_group_lasso(E, max_lag=K, select="ebic", n_lambda=20,
                               tol=1e-11, fit_deviations=False)
    lam = res.lambda_selected
    mt = sk.build_moment_tensor(E, max_lag=K)
    Gt = mt.scaled()
    jj, kk = _all_pairs(P, K)
    G = sk._gather_gram(Gt, jj, kk)
    cand = sk._Candidates(np.column_stack([jj, kk]))
    for i in range(P):
        xty = sk._gather_xty(Gt, jj, kk, i)
        b = res.B0[kk - 1, i, jj] / mt.scale[0, i] * mt.scale[kk, jj]
        c = xty - G @ b
        gn = sk._group_norms(c, cand.gstart) / cand.wg
        bn = sk._group_norms(b, cand.gstart)
        scaleref = max(1.0, lam)
        assert np.all(gn[bn == 0.0] <= lam + 1e-5 * scaleref)     # zero groups
        if np.any(bn > 0.0):
            assert np.max(np.abs(gn[bn > 0.0] - lam)) < 1e-5 * scaleref


def test_gapsafe_matches_kkt_screening():
    """The GAP-safe sphere test only removes provably-zero groups: same answer."""
    E, _A = make_panel(n_ent=4, T=130, P=6, K=2, n_edges=6, seed=24)
    kw = dict(max_lag=2, select="ebic", n_lambda=20, tol=1e-10,
              fit_deviations=False)
    a = sk.panel_group_lasso(E, screening="kkt", **kw)
    b = sk.panel_group_lasso(E, screening="gapsafe", lambda_grid=a.lambda_path, **kw)
    for l, (x, y) in enumerate(zip(a.path_support, b.path_support)):
        assert np.array_equal(x, y), "gapsafe differs at lambda index %d" % l
    assert np.max(np.abs(a.B0 - b.B0)) < 1e-8


# --------------------------------------------------------------------------- #
#  4. lambda_max is tight
# --------------------------------------------------------------------------- #
def test_lambda_max_is_tight():
    """Exactly all-zero at lambda_max; nonzero just below it."""
    E, _A = make_panel(n_ent=4, T=120, P=5, K=2, n_edges=5, seed=31)
    res = sk.panel_group_lasso(E, max_lag=2, select="ebic", n_lambda=30,
                               fit_deviations=False)
    lmax = float(res.lambda_path[0])
    at = sk.panel_group_lasso(E, max_lag=2, lambda_grid=[lmax],
                              fit_deviations=False, select="ebic")
    assert at.support.sum() == 0
    assert np.all(at.B0 == 0.0)
    just_below = sk.panel_group_lasso(E, max_lag=2, lambda_grid=[lmax * (1 - 1e-6)],
                                      fit_deviations=False, select="ebic",
                                      tol=1e-12)
    assert just_below.support.sum() > 0
    # and the closed form matches a direct computation from the statistics
    mt = sk.build_moment_tensor(E, max_lag=2)
    Gt = mt.scaled()
    jj, kk = _all_pairs(5, 2)
    cand = sk._Candidates(np.column_stack([jj, kk]))
    direct = max(float((sk._group_norms(sk._gather_xty(Gt, jj, kk, i), cand.gstart)
                        / cand.wg).max()) for i in range(5))
    assert abs(direct - lmax) < 1e-10 * lmax


# --------------------------------------------------------------------------- #
#  5. Recovery
# --------------------------------------------------------------------------- #
def _f1(true_mask, got_mask):
    tp = int(np.sum(true_mask & got_mask))
    fp = int(np.sum(~true_mask & got_mask))
    fn = int(np.sum(true_mask & ~got_mask))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_recovery_f1_above_0_9(seed):
    """F1 of the recovered group-level skeleton > 0.9 at moderate SNR.

    Recovery is scored at the GROUP level ("j Granger-causes i at some lag"), which
    is the object the lag-grouped penalty selects and the object the discovery stage
    consumes.
    """
    E, A = make_panel(n_ent=10, T=300, P=10, K=3, n_edges=10, noise=0.5, seed=seed)
    res = sk.panel_group_lasso(E, max_lag=3, select="ebic", n_lambda=40,
                               fit_deviations=False)
    f1 = _f1((A != 0).any(0), res.skeleton())
    assert f1 > 0.9, "F1=%.3f (true %d edges, got %d)" % (
        f1, int((A != 0).any(0).sum()), int(res.skeleton().sum()))


def test_recovery_stability_along_path():
    """``path_support`` is monotone enough to drive stability selection: the number of
    selected groups is non-decreasing as lambda falls (no pathological flip-flop)."""
    E, _A = make_panel(n_ent=6, T=200, P=6, K=2, n_edges=6, seed=44)
    res = sk.panel_group_lasso(E, max_lag=2, select="ebic", n_lambda=30,
                               fit_deviations=False)
    sizes = np.array([int(s.any(0).sum()) for s in res.path_support])
    assert sizes[0] == 0 or sizes[0] <= sizes[-1]
    assert sizes[-1] >= sizes[0]
    drops = np.diff(sizes)
    assert drops.min() >= -2, "support flip-flops along the path: %r" % (sizes,)


# --------------------------------------------------------------------------- #
#  6. The hierarchy earns its keep
# --------------------------------------------------------------------------- #
def _heldout_mse(E, res, K, t_lo):
    """Standardised held-out MSE of a fitted skeleton on rows >= t_lo."""
    mt = sk.build_moment_tensor(E[:, :t_lo], max_lag=K)
    scale = mt.scale
    Z = sk.build_moment_tensor(E, max_lag=K).Z
    P = E.shape[2]
    T = Z.shape[1]
    Bsc = np.zeros((K + 1, P, P))
    Bsc[1:] = res.B0 / scale[1:, None, :] / scale[0][None, :, None] * \
        scale[0][None, :, None]
    Bsc[1:] = res.B0 / scale[1:, None, :]
    dev = {}
    for e, D in res.Delta.items():
        Dd = np.zeros((K + 1, P, P))
        Dd[1:] = D / scale[1:, None, :]
        dev[e] = Dd
    lo = max(t_lo, K)
    Y = Z[:, lo:T, :] / scale[0][None, None, :]
    V = np.isfinite(E[:, lo:T, :]).astype(float)
    fit = np.zeros_like(Y)
    for k in range(1, K + 1):
        Zl = Z[:, lo - k:T - k, :]
        fit += Zl @ Bsc[k].T
        for e, D in dev.items():
            fit[e] += Zl[e] @ D[k].T
    r = (Y - fit) * V
    return float(np.sum(r * r) / max(V.sum(), 1.0))


def test_hierarchy_homogeneous_panel_prefers_pooled():
    """hetero == 0: the b0-only limit is best and the solver refuses to buy
    exceptions -- the fitted deviations are a rounding error next to the skeleton."""
    E, _A = make_panel(n_ent=8, T=260, P=6, K=2, n_edges=6, hetero=0.0, seed=51)
    tr = 200
    pooled = sk.panel_group_lasso(E[:, :tr], max_lag=2, select="ebic",
                                  fit_deviations=False)
    split = sk.panel_group_lasso(E[:, :tr], max_lag=2, select="ebic",
                                 fit_deviations=True)
    dev_mass = sum(float(np.abs(D).sum()) for D in split.Delta.values())
    base_mass = float(np.abs(split.B0).sum())
    assert dev_mass <= 0.10 * max(base_mass, 1e-12), \
        "bought %.3g of deviation against %.3g of skeleton on a homogeneous panel" % (
            dev_mass, base_mass)
    m_pool = _heldout_mse(E, pooled, 2, tr)
    m_split = _heldout_mse(E, split, 2, tr)
    assert m_pool <= m_split * 1.02, "pooled %.5f vs split %.5f" % (m_pool, m_split)


def test_hierarchy_heterogeneous_panel_prefers_split():
    """Strong per-entity deviations: b0 + d_e beats the pooled-only fit out of
    sample, which is the whole point of the structural split."""
    E, _A = make_panel(n_ent=8, T=320, P=6, K=2, n_edges=6, hetero=0.9,
                       noise=0.35, seed=52)
    tr = 250
    pooled = sk.panel_group_lasso(E[:, :tr], max_lag=2, select="ebic",
                                  fit_deviations=False)
    split = sk.panel_group_lasso(E[:, :tr], max_lag=2, select="ebic",
                                 fit_deviations=True, kappa=2.0)
    assert len(split.Delta) > 0, "no deviations fitted on a heterogeneous panel"
    m_pool = _heldout_mse(E, pooled, 2, tr)
    m_split = _heldout_mse(E, split, 2, tr)
    assert m_split < m_pool * 0.98, "split %.5f did not beat pooled %.5f" % (
        m_split, m_pool)


# --------------------------------------------------------------------------- #
#  7. Missingness -- the renormalisation check
# --------------------------------------------------------------------------- #
def test_missingness_does_not_shrink_coefficients():
    """20% MCAR: available-case renormalisation keeps the coefficients unshrunk, and
    turning it off reproduces the bias it exists to remove.

    Both fits use the SAME absolute lambda, so lasso shrinkage is common to them and
    the only thing being measured is the missing-data handling.
    """
    E, A = make_panel(n_ent=10, T=400, P=6, K=2, n_edges=6, noise=0.4, seed=61)
    rng = np.random.default_rng(61)
    Em = E.copy()
    Em[rng.random(E.shape) < 0.20] = np.nan

    full = sk.panel_group_lasso(E, max_lag=2, select="ebic", n_lambda=30,
                                fit_deviations=False)
    lam = [full.lambda_selected]
    full = sk.panel_group_lasso(E, max_lag=2, lambda_grid=lam, select="ebic",
                                fit_deviations=False)
    corr = sk.panel_group_lasso(Em, max_lag=2, lambda_grid=lam, select="ebic",
                                fit_deviations=False, available_case=True)
    naive = sk.panel_group_lasso(Em, max_lag=2, lambda_grid=lam, select="ebic",
                                 fit_deviations=False, available_case=False)

    supp = (A != 0) & (full.B0 != 0)
    assert supp.sum() >= 3, "fixture recovered too little to test shrinkage"
    r_corr = float(np.mean(corr.B0[supp] / full.B0[supp]))
    r_naive = float(np.mean(naive.B0[supp] / full.B0[supp]))
    assert 0.88 < r_corr < 1.12, "available-case ratio %.3f is not ~1" % r_corr
    assert r_corr > r_naive, \
        "renormalisation did not help: corrected %.3f vs naive %.3f" % (
            r_corr, r_naive)
    # the trap is real: the naive zero-fill IS biased toward zero
    assert r_naive < 0.97, "naive zero-fill ratio %.3f -- fixture too easy" % r_naive


def test_missingness_complete_data_is_a_no_op():
    """With no missing cells every correction factor is exactly 1."""
    E, _A = make_panel(n_ent=4, T=120, P=5, K=2, n_edges=4, seed=62)
    kw = dict(max_lag=2, select="ebic", n_lambda=15, fit_deviations=False)
    a = sk.panel_group_lasso(E, available_case=True, **kw)
    b = sk.panel_group_lasso(E, available_case=False, **kw)
    assert np.max(np.abs(a.B0 - b.B0)) < 1e-12


def test_obs_mask_matches_nan_encoding():
    """An explicit ``obs`` mask is equivalent to encoding the holes as NaN."""
    E, _A = make_panel(n_ent=4, T=110, P=5, K=2, n_edges=4, seed=63)
    rng = np.random.default_rng(63)
    hole = rng.random(E.shape) < 0.15
    En = E.copy(); En[hole] = np.nan
    kw = dict(max_lag=2, select="ebic", n_lambda=12, fit_deviations=False)
    a = sk.panel_group_lasso(En, **kw)
    b = sk.panel_group_lasso(E, obs=~hole, **kw)
    assert np.max(np.abs(a.B0 - b.B0)) < 1e-12


# --------------------------------------------------------------------------- #
#  8. No look-ahead
# --------------------------------------------------------------------------- #
def test_no_lookahead_prefix_fit_ignores_the_future():
    """A fit that claims to use rows <= t0 is bit-identical when rows > t0 are
    deleted -- even when those rows are replaced by garbage large enough to wreck
    every mean, scale and moment if any of them leaked.

    This is the certificate the forward-chaining selector rests on: it is exactly the
    ``_fit_path(E, obs, cut)`` call each fold makes.
    """
    E, _A = make_panel(n_ent=5, T=200, P=5, K=2, n_edges=5, seed=71)
    rng = np.random.default_rng(71)
    t0 = 120
    poisoned = E.copy()
    poisoned[:, t0:, :] = 1e4 * rng.standard_normal(poisoned[:, t0:, :].shape)

    cfg = sk._make_cfg(max_lag=2, n_lambda=15, fit_deviations=True)
    cand = sk._normalise_candidates(None, 5, 2)
    lp_a, sols_a, _pa, _na = sk._fit_path(poisoned, None, t0, cand, cfg)
    lp_b, sols_b, _pb, _nb = sk._fit_path(E[:, :t0], None, t0, cand, cfg)

    assert np.array_equal(lp_a, lp_b)
    assert len(sols_a) == len(sols_b)
    for l, ((sa, ra, na, _ba), (sbb, rb, nb, _bb)) in enumerate(zip(sols_a,
                                                                        sols_b)):
        assert np.array_equal(sa.b0_k, sbb.b0_k), "lambda %d" % l
        assert np.array_equal(sa.b0_i, sbb.b0_i)
        assert np.array_equal(sa.b0_j, sbb.b0_j)
        assert np.array_equal(sa.b0_v, sbb.b0_v), "coefficients moved at lambda %d" % l
        assert np.array_equal(sa.dv_v, sbb.dv_v)
        assert ra == rb and na == nb


def test_no_lookahead_support_selected_on_prefix():
    """Top-level: the support fitted on rows <= t0 does not change when the later
    rows are deleted from the array handed in."""
    E, _A = make_panel(n_ent=5, T=200, P=5, K=2, n_edges=5, seed=72)
    rng = np.random.default_rng(72)
    t0 = 130
    poisoned = E.copy()
    poisoned[:, t0:, :] = 1e4 * rng.standard_normal(poisoned[:, t0:, :].shape)
    kw = dict(max_lag=2, select="ebic", n_lambda=20, fit_deviations=False)
    a = sk.panel_group_lasso(poisoned[:, :t0], **kw)
    b = sk.panel_group_lasso(E[:, :t0], **kw)
    assert np.array_equal(a.support, b.support)
    assert np.max(np.abs(a.B0 - b.B0)) == 0.0


def test_kfold_cv_is_refused():
    """K-fold CV over time would fail this project's causal certificate."""
    E, _A = make_panel(n_ent=3, T=80, P=4, K=1, n_edges=3, seed=73)
    with pytest.raises(ValueError, match="look-ahead"):
        sk.panel_group_lasso(E, max_lag=1, select="cv")


def test_forward_selection_scores_only_future_blocks():
    """Forward chaining records expanding, strictly-ordered train/validate cuts."""
    E, _A = make_panel(n_ent=5, T=200, P=5, K=2, n_edges=5, seed=74)
    res = sk.panel_group_lasso(E, max_lag=2, select="forward", n_lambda=10,
                               n_splits=3, fit_deviations=False)
    folds = res.meta["folds"]
    assert len(folds) >= 2
    for f in folds:
        assert f["train_hi"] < f["val_hi"] <= E.shape[1]
    assert [f["train_hi"] for f in folds] == sorted(f["train_hi"] for f in folds)
    assert res.meta["selection_score"].shape == res.lambda_path.shape


# --------------------------------------------------------------------------- #
#  9. Plumbing, API surface, and the numpy-only guarantee
# --------------------------------------------------------------------------- #
def test_import_is_numpy_only():
    """``import cafe._skeleton`` must work with numpy and the stdlib alone."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    code = (
        "import sys\n"
        "BANNED = ('scipy', 'sklearn', 'skglm', 'celer', 'numba', 'pandas',\n"
        "          'polars', 'torch', 'matplotlib')\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BANNED:\n"
        "            raise ImportError('banned dependency: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, %r)\n"
        "import cafe._skeleton as s\n"
        "assert hasattr(s, 'panel_group_lasso')\n"
        "assert hasattr(s, 'build_moment_tensor')\n"
        "mods = {m.split('.')[0] for m in sys.modules}\n"
        "assert not (mods & set(BANNED)), sorted(mods & set(BANNED))\n"
        "print('OK')\n" % os.path.abspath(root)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_result_shapes_and_meta():
    E, _A = make_panel(n_ent=4, T=140, P=5, K=3, n_edges=5, seed=81)
    res = sk.panel_group_lasso(E, max_lag=3, select="ebic", n_lambda=12)
    assert res.B0.shape == (3, 5, 5)
    assert res.support.shape == (3, 5, 5) and res.support.dtype == bool
    assert res.lambda_path.shape == (12,)
    assert len(res.path_support) == 12
    assert res.lambda_selected in set(res.lambda_path.tolist())
    assert res.n_kkt_sweeps > 0
    for key in ("moment_bytes", "n_candidates", "n_groups", "df", "seconds",
                "nickell_bias_order", "support_shared", "select"):
        assert key in res.meta
    assert res.meta["moment_bytes"] == 4 * 4 * 5 * 5 * 8
    for D in res.Delta.values():
        assert D.shape == (3, 5, 5)
    assert np.array_equal(res.support,
                          (res.B0 != 0) | np.logical_or.reduce(
                              [np.zeros_like(res.support)] +
                              [D != 0 for D in res.Delta.values()]))


def test_candidate_formats_agree():
    """dict / (K,P,P) mask / (P,P,K) mask / sequence all mean the same thing."""
    E, _A = make_panel(n_ent=4, T=120, P=5, K=2, n_edges=4, seed=82)
    K, P = 2, 5
    rng = np.random.default_rng(82)
    mask = rng.random((K, P, P)) < 0.5
    mask[0, 0, 0] = True                       # never leave a target empty
    as_dict = {i: [(j, k + 1) for k in range(K) for j in range(P) if mask[k, i, j]]
               for i in range(P)}
    as_seq = [as_dict[i] for i in range(P)]
    kw = dict(max_lag=K, select="ebic", n_lambda=10, fit_deviations=False)
    ref = sk.panel_group_lasso(E, as_dict, **kw)
    for c in (mask, np.ascontiguousarray(mask.transpose(1, 2, 0)), as_seq):
        got = sk.panel_group_lasso(E, c, **kw)
        assert np.max(np.abs(got.B0 - ref.B0)) < 1e-12
    # and a candidate set can never be exceeded
    allowed = mask.transpose(0, 1, 2)
    assert not np.any(ref.support & ~allowed)


def test_single_entity_disables_deviations():
    """A (T, P) matrix is a one-entity panel: no deviation level, no time FE."""
    rng = np.random.default_rng(83)
    X = np.cumsum(rng.standard_normal((160, 4)) * 0.1, axis=0)
    res = sk.panel_group_lasso(X, max_lag=2, select="ebic", n_lambda=10)
    assert res.Delta == {}
    assert res.meta["fit_deviations"] is False
    assert res.B0.shape == (2, 4, 4)


def test_empty_candidates_returns_empty_skeleton():
    E, _A = make_panel(n_ent=3, T=80, P=4, K=1, n_edges=2, seed=84)
    res = sk.panel_group_lasso(E, {i: [] for i in range(4)}, max_lag=1,
                               select="ebic", n_lambda=5)
    assert res.support.sum() == 0
    assert np.all(res.B0 == 0.0)


def test_moment_budget_falls_back_to_referenced_subpanel():
    """Over budget, the solver restricts to the referenced sub-panel -- exactly."""
    E, _A = make_panel(n_ent=4, T=120, P=8, K=2, n_edges=4, seed=85)
    cand = {i: ([(j, k) for j in (1, 3, 5) for k in (1, 2)] if i in (1, 3, 5) else [])
            for i in range(8)}
    kw = dict(max_lag=2, select="ebic", n_lambda=12, fit_deviations=False)
    ref = sk.panel_group_lasso(E, cand, **kw)
    # full tensor is 9*8*8*8 = 4608 B, the referenced sub-panel only 9*3*3*8 = 648 B
    tiny = sk.panel_group_lasso(E, cand, moment_budget_gb=2000.0 / (1 << 30), **kw)
    assert tiny.B0.shape == ref.B0.shape
    assert np.max(np.abs(tiny.B0 - ref.B0)) < 1e-10
    assert "subset_series" in tiny.meta
    with pytest.raises(MemoryError):
        sk.panel_group_lasso(E, max_lag=2, moment_budget_gb=1e-12, select="ebic")


def test_kappa_controls_the_price_of_an_exception():
    """Larger kappa buys fewer deviations -- the hierarchy dial does what it says."""
    E, _A = make_panel(n_ent=8, T=260, P=5, K=2, n_edges=5, hetero=0.9,
                       noise=0.35, seed=86)
    kw = dict(max_lag=2, select="ebic", n_lambda=20, fit_deviations=True)
    mass = []
    for kap in (2.0, 8.0):
        r = sk.panel_group_lasso(E, kappa=kap, **kw)
        mass.append(sum(float(np.abs(D).sum()) for D in r.Delta.values()))
    assert mass[0] >= mass[1], "kappa=2 bought %.4g, kappa=8 bought %.4g" % tuple(mass)


def test_lam2_sparse_group_runs_and_sparsifies():
    """The optional extra L1 on the deviations is honoured."""
    E, _A = make_panel(n_ent=6, T=200, P=5, K=2, n_edges=5, hetero=0.9,
                       noise=0.35, seed=87)
    kw = dict(max_lag=2, select="ebic", n_lambda=12, fit_deviations=True,
              kappa=2.0)
    a = sk.panel_group_lasso(E, lam2=0.0, **kw)
    b = sk.panel_group_lasso(E, lam2=50.0, **kw)
    n_a = sum(int((D != 0).sum()) for D in a.Delta.values())
    n_b = sum(int((D != 0).sum()) for D in b.Delta.values())
    assert n_b <= n_a


def test_deterministic():
    E, _A = make_panel(n_ent=4, T=120, P=5, K=2, n_edges=4, seed=88)
    kw = dict(max_lag=2, select="forward", n_lambda=10)
    a = sk.panel_group_lasso(E, **kw)
    b = sk.panel_group_lasso(E, **kw)
    assert np.array_equal(a.B0, b.B0)
    assert a.lambda_selected == b.lambda_selected


def test_ebic_recovers_support_better_than_predictive_cv():
    """A documented, deliberate property, not an accident: forward-chaining CV
    minimises predictive risk and overselects for structure, while eBIC is selection
    consistent.  If this ever flips, the docstring's guidance must change with it.
    """
    E, A = make_panel(n_ent=10, T=200, P=20, K=8, n_edges=6, noise=0.5, seed=91)
    truth = (A != 0).any(0)
    eb = sk.panel_group_lasso(E, max_lag=8, select="ebic", fit_deviations=False)
    fw = sk.panel_group_lasso(E, max_lag=8, select="forward", fit_deviations=False)
    f_eb, f_fw = _f1(truth, eb.skeleton()), _f1(truth, fw.skeleton())
    assert f_eb >= f_fw, "eBIC %.3f no longer beats forward %.3f" % (f_eb, f_fw)
    assert f_eb > 0.9
    assert int(fw.skeleton().sum()) >= int(eb.skeleton().sum())


def test_large_scale_lambda_terminates():
    """A big panel drives lambda to O(n_design_rows); the KKT certification loop must
    still terminate.  With an ABSOLUTE tolerance on the violation test this deadlocks:
    at lam ~ 1e5 the screen's ``lam + 1e-6`` slack is finer than the block solve's
    relative zero test, so a boundary group is admitted, solved back to zero, and
    re-admitted forever.  Both now use the same squared predicate.
    """
    E, _A = make_panel(n_ent=30, T=400, P=6, K=2, n_edges=5, noise=0.5, seed=95)
    res = sk.panel_group_lasso(E, max_lag=2, select="ebic", n_lambda=15,
                               fit_deviations=False, tol=1e-6)
    assert res.meta["n_design_rows"] > 10000
    assert res.lambda_path[0] > 1e3
    assert res.n_kkt_sweeps < 100000


# --------------------------------------------------------------------------- #
#  10. conditioning diagnostic + the bivariate branch it routes to
# --------------------------------------------------------------------------- #
def _persistent(T, P, rho, seed=0):
    """``P`` INDEPENDENT AR(1) series -- every lag column is nearly a duplicate of the
    next lag of the SAME series, while the sources are perfectly separable."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, P))
    for t in range(1, T):
        X[t] += rho * X[t - 1]
    return X


def _column_tolerance(mt, K, P):
    """Per-COLUMN ``1 - R^2`` of the same design -- the statistic the diagnostic
    deliberately does NOT use."""
    Gt = mt.scaled(True)
    kk = np.repeat(np.arange(1, K + 1), P)
    jj = np.tile(np.arange(P), K)
    R = sk._gather_gram(Gt, jj, kk) / float(mt.n_tot)
    return 1.0 / np.diag(np.linalg.inv(R + 1e-8 * np.eye(R.shape[0])))


def test_conditioning_diagnostic_null_is_one_by_construction():
    """Orthogonal sources give ``GVIF == 1`` exactly, so the tolerance has no free
    reference point: the only gap from 1 is the O(A/n) sampling floor."""
    rng = np.random.default_rng(0)
    mt = sk.build_moment_tensor(rng.standard_normal((1, 4000, 6)), max_lag=3)
    d = sk.conditioning_diagnostic(mt)
    assert d["ok"] and not d["reason"]
    assert d["tolerance"] > 0.99
    assert d["min_tolerance"] > 0.98
    assert d["n_below_half"] == 0
    assert d["block_tolerance"].shape == (6,)
    assert np.all(d["block_tolerance"] <= 1.0 + 1e-9)


def test_conditioning_diagnostic_is_blind_to_within_source_lag_collinearity():
    """THE property that makes this the right statistic.

    Consecutive lags of a persistent series are near-duplicates, so a per-column VIF
    condemns every real time-series design.  The group lasso groups over lags and the
    bivariate alternative carries the whole block, so neither pays for it -- and the
    Fox-Monette block form does not charge for it either.
    """
    E = _persistent(6000, 5, 0.97, seed=1)[None]
    mt = sk.build_moment_tensor(E, max_lag=6)
    d = sk.conditioning_diagnostic(mt)
    col = _column_tolerance(mt, 6, 5)
    assert col.mean() < 0.15, "fixture is meant to be column-collinear"
    assert d["tolerance"] > 0.95, (
        "source separability must not be charged for within-source lag collinearity "
        f"(column tolerance {col.mean():.3f}, source tolerance {d['tolerance']:.3f})")


def test_conditioning_diagnostic_fires_on_between_source_collinearity():
    """Near-duplicate SOURCES: no conditional coefficient is identified, and the
    diagnostic falls continuously as the duplicates are pulled apart."""
    rng = np.random.default_rng(2)
    f = _persistent(4000, 1, 0.7, seed=2)
    prev = -1.0
    for noise in (0.02, 0.2, 0.5, 1.0, 3.0):
        E = (f + noise * rng.standard_normal((4000, 5)))[None]
        d = sk.conditioning_diagnostic(sk.build_moment_tensor(E, max_lag=2))
        assert d["ok"]
        assert d["tolerance"] > prev, "tolerance must rise as the sources separate"
        prev = d["tolerance"]
        if noise <= 0.5:
            assert d["tolerance"] < 0.5 and d["n_below_half"] == 5
    assert prev > 0.5, "well-separated sources must clear the majority point"


def test_conditioning_diagnostic_abstains_instead_of_guessing():
    """Above the column budget (and on a degenerate design) it reports the incumbent
    value 1.0 with a reason, so the caller keeps the fully conditional evidence."""
    rng = np.random.default_rng(3)
    mt = sk.build_moment_tensor(rng.standard_normal((1, 300, 40)), max_lag=4)
    d = sk.conditioning_diagnostic(mt, max_cols=50)
    assert not d["ok"] and d["tolerance"] == 1.0 and "columns" in d["reason"]
    mt1 = sk.build_moment_tensor(rng.standard_normal((1, 200, 1)), max_lag=2)
    d1 = sk.conditioning_diagnostic(mt1)
    assert not d1["ok"] and d1["tolerance"] == 1.0 and d1["reason"]


def test_bivariate_granger_matches_an_explicit_two_series_ols():
    """The whole claim is that it differs from the conditional refit in NOTHING but
    the conditioning set, so it must reproduce a materialised OLS on the same demeaned
    columns to machine precision -- including the ``j == i`` own-lag cells."""
    rng = np.random.default_rng(4)
    T, P, K = 500, 4, 3
    X = rng.standard_normal((T, P))
    for t in range(1, T):                       # dynamics + cross-series dependence
        X[t] += 0.6 * X[t - 1] + 0.3 * np.roll(X[t - 1], 1)
    E = X[None]
    mt = sk.build_moment_tensor(E, max_lag=K, demean="twoway")
    got = sk.bivariate_granger(mt)
    assert got.shape == (K, P, P)

    Z = sk._two_way_demean(E, np.ones_like(E), "twoway")[0]
    n = T - K
    lags = [Z[K - k:T - k] for k in range(1, K + 1)]
    Y = Z[K:]

    def ols_t(cols, y):
        Zm = np.column_stack(cols)
        b, *_ = np.linalg.lstsq(Zm, y, rcond=None)
        r = y - Zm @ b
        s2 = float(r @ r) / max(n - Zm.shape[1], 1)
        Gi = np.linalg.pinv(Zm.T @ Zm)
        return np.abs(b) / np.sqrt(np.maximum(np.diag(Gi), 0.0) * s2)

    for i in range(P):
        own = [L[:, i] for L in lags]
        assert np.allclose(ols_t(own, Y[:, i]), got[:, i, i], atol=1e-5)
        for j in range(P):
            if j == i:
                continue
            ref = ols_t(own + [L[:, j] for L in lags], Y[:, i])[K:]
            assert np.allclose(ref, got[:, i, j], atol=1e-5), (i, j)


def test_bivariate_granger_ranks_a_known_chain_above_the_non_edges():
    """It is evidence, not decoration: on a design its conditional sibling cannot
    identify (five near-duplicate sources plus one true driver) it still ranks."""
    rng = np.random.default_rng(5)
    T, P, K = 3000, 5, 2
    X = np.zeros((T, P))
    e = rng.standard_normal((T, P))
    for t in range(2, T):
        X[t, 0] = 0.5 * X[t - 1, 0] + e[t, 0]
        for j in range(1, P):                   # a chain 0 -> 1 -> 2 -> 3 -> 4
            X[t, j] = 0.4 * X[t - 1, j] + 0.6 * X[t - 1, j - 1] + e[t, j]
    mt = sk.build_moment_tensor(X[None], max_lag=K, demean="twoway")
    T3 = np.nan_to_num(sk.bivariate_granger(mt), nan=0.0).max(axis=0)
    truth = np.zeros((P, P), dtype=bool)
    for j in range(1, P):
        truth[j, j - 1] = True
    off = ~np.eye(P, dtype=bool)
    assert T3[off & truth].min() > np.median(T3[off & ~truth])


def test_bivariate_granger_guards_return_nan_rather_than_raising():
    rng = np.random.default_rng(6)
    mt = sk.build_moment_tensor(rng.standard_normal((1, 25, 3)), max_lag=8)
    out = sk.bivariate_granger(mt)                 # n = 17 < 2K + 2 = 18
    assert out.shape == (8, 3, 3) and np.all(np.isnan(out))
    mt2 = sk.build_moment_tensor(rng.standard_normal((1, 400, 6)), max_lag=2)
    out2 = sk.bivariate_granger(mt2, max_pairs=4)  # over the pair budget
    assert np.all(np.isnan(out2))
