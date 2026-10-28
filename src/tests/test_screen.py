"""Contract tests for the backward-only lag screener (``cafe._screen``).

The properties asserted here are the ones the screener is *bought* for:

1. **No look-ahead.** Row ``t`` of the encoding is a function of input rows ``<= t``
   only, and the result reported at a checkpoint ``t0`` does not move when rows
   ``> t0`` are scrambled. Asserted EXACTLY (bit-for-bit), not to a tolerance. A
   control test ("has teeth") shows the same assertion FAILS for a deliberately
   forward-looking encoder, so the test is not vacuous.
2. **Recovery.** On a sparse VAR with known structure, the screener's candidate set
   is a superset of the truth at a large reduction factor.
3. **Null behaviour.** On iid noise the permutation threshold controls the family-wise
   error rate at the nominal ``alpha``.
4. **``force_self``.** ``(i, i, 1..K)`` survive unconditionally, whatever the data.
5. **Backend equivalence.** ``dense`` and ``bits`` agree on ``S``.
6. **Missingness.** No NaN/inf ever appears in ``S``; recall degrades gracefully.
7. **numpy only.** The module imports with scipy/sklearn/numba/numexpr/pandas
   unimportable.

The fixtures are self-contained on purpose: the full causal-panel generator is owned
by a sibling module and this file must not depend on it yet.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cafe._screen import (  # noqa: E402
    ScreenResult,
    encode_bits,
    popcount_bytes,
    screen_lags,
    screen_lags_online,
)

_SCREEN_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "cafe", "_screen.py"))


# --------------------------------------------------------------------------- #
# self-contained fixtures
# --------------------------------------------------------------------------- #
def var_panel(P=100, T=400, K_true=4, n_edges=2, coef=0.55, ar=0.30, burn=200,
              seed=0, rho_target=0.8):
    """Sparse stationary VAR with a KNOWN lag structure.

    Each target gets one own-lag AR(1) term plus ``n_edges`` cross parents, each at a
    distinct lag in ``1..K_true``. The companion matrix is rescaled so the spectral
    radius is ``rho_target`` -- without that a randomly drawn sparse VAR is routinely
    explosive, and rescaling only to the stationarity boundary leaves a near-unit-root
    panel whose first differences are almost pure innovation (we observed recall
    collapse on exactly such a draw; it is a property of the data, not of the
    screener). Returns ``(X, edges)`` with ``edges`` a set of ``(i, j, k)`` triples,
    ``k`` 1-based, self terms excluded.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((K_true + 1, P, P))
    edges = set()
    for i in range(P):
        A[1, i, i] = ar
        js = rng.choice(np.delete(np.arange(P), i), size=n_edges, replace=False)
        ks = rng.choice(np.arange(1, K_true + 1), size=n_edges, replace=False)
        for j, k in zip(js, ks):
            A[k, i, int(j)] = coef * (1.0 if rng.random() < 0.5 else -1.0)
            edges.add((i, int(j), int(k)))
    F = np.zeros((K_true * P, K_true * P))
    for k in range(1, K_true + 1):
        F[:P, (k - 1) * P:k * P] = A[k]
    if K_true > 1:
        F[P:, :-P] = np.eye((K_true - 1) * P)
    rho = np.abs(np.linalg.eigvals(F)).max()
    if rho > rho_target:
        A *= rho_target / rho
    Tt = T + burn
    X = np.zeros((Tt, P))
    E = rng.standard_normal((Tt, P))
    for t in range(1, Tt):
        x = E[t].copy()
        for k in range(1, min(K_true, t) + 1):
            x += A[k] @ X[t - k]
        X[t] = x
    return X[burn:], edges


_CACHE = {}


def cached_var(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        _CACHE[key] = var_panel(**kw)
    X, e = _CACHE[key]
    return X.copy(), set(e)


def noise_panel(P=12, T=600, seed=0):
    return np.random.default_rng(seed).standard_normal((T, P))


def recall(res, edges):
    got = set(res.triples())
    return sum(1 for e in edges if e in got) / float(len(edges))


# --------------------------------------------------------------------------- #
# 1. no look-ahead -- asserted exactly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t0", [50, 137, 299])
def test_encoding_is_backward_only(t0):
    """Truncating the input at t0 must not change any encoded bit at rows <= t0."""
    X = noise_panel(P=7, T=320, seed=3)
    X[np.random.default_rng(4).random(X.shape) < 0.15] = np.nan
    full = encode_bits(X, max_lag=6)
    trunc = encode_bits(X[:t0 + 1], max_lag=6)
    for a, b, name in zip(full[:3], trunc[:3], ("b_s", "b_l", "v")):
        assert np.array_equal(a[:t0 + 1], b), f"{name} changed when the future was cut"


def test_lookahead_test_has_teeth():
    """The control: a FORWARD-looking encoder must FAIL the assertion above.

    Without this, ``test_encoding_is_backward_only`` could pass for a trivial reason.
    This is ARROW's own Definition 4.1 bit (relative to ``x^{t+w}``), which is exactly
    what this module refuses to use.
    """
    X = noise_panel(P=5, T=200, seed=5)
    w, t0 = 2, 120

    def leaky_bits(Z):                      # b[t] = Z[t] > Z[t + w]   (reads t + w)
        b = np.zeros_like(Z, dtype=bool)
        b[:len(Z) - w] = Z[:len(Z) - w] > Z[w:]
        return b

    assert not np.array_equal(leaky_bits(X)[:t0 + 1], leaky_bits(X[:t0 + 1])[:t0 + 1]), \
        "control failed: the forward encoder should not survive truncation"


@pytest.mark.parametrize("t0", [120, 240])
def test_online_result_ignores_the_future(t0):
    """The set reported at t0 is bit-identical when rows > t0 are scrambled."""
    X, _ = cached_var(P=20, T=300, seed=1)
    Y = X.copy()
    rng = np.random.default_rng(99)
    Y[t0 + 1:] = 50.0 * rng.standard_normal(Y[t0 + 1:].shape)   # nonsense future
    Y[t0 + 1:][rng.random(Y[t0 + 1:].shape) < 0.3] = np.nan

    kw = dict(max_lag=5, n_perm=40, seed=0)
    (_, ra), = screen_lags_online(X, checkpoints=[t0], **kw)
    (_, rb), = screen_lags_online(Y, checkpoints=[t0], **kw)
    rc = screen_lags(X[:t0 + 1], **kw)

    for r in (rb, rc):
        assert np.array_equal(ra.S, r.S)
        assert np.array_equal(ra.n_valid, r.n_valid)
        assert np.array_equal(ra.threshold, r.threshold)
        assert ra.candidates == r.candidates


def test_online_yields_every_checkpoint():
    X, _ = cached_var(P=15, T=250, seed=2)
    cps = [80, 150, 249]
    out = list(screen_lags_online(X, checkpoints=cps, max_lag=4, n_perm=0))
    assert [t for t, _ in out] == cps
    assert all(isinstance(r, ScreenResult) for _, r in out)
    with pytest.raises(ValueError):
        list(screen_lags_online(X, checkpoints=[X.shape[0]], max_lag=4, n_perm=0))


# --------------------------------------------------------------------------- #
# 2. recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stat", ["phi", "arrow"])
def test_recovery_recall_and_reduction(stat):
    X, edges = cached_var(P=100, T=400, seed=0)
    res = screen_lags(X, max_lag=8, stat=stat, n_perm=100, seed=0)
    rec = recall(res, edges)
    assert rec >= 0.95, f"recall {rec:.3f} < 0.95 ({stat})"
    assert res.reduction <= 0.10, f"reduction {res.reduction:.4f} > 0.10 ({stat})"
    # the forced self terms are all there and are not what earns the recall
    assert all((i, i, k) in set(res.triples()) for i in range(5) for k in (1, 8))


def test_recovery_is_stable_across_draws():
    for seed in range(4):
        X, edges = cached_var(P=100, T=400, seed=seed)
        res = screen_lags(X, max_lag=8, n_perm=100, seed=0)
        assert recall(res, edges) >= 0.95, f"seed {seed}"
        assert res.reduction <= 0.10


def test_all_significant_lags_are_kept_not_only_the_argmax():
    """A parent acting at two lags must appear at BOTH, not just at the argmax."""
    rng = np.random.default_rng(7)
    T, P = 900, 6
    X = rng.standard_normal((T, P))
    X[3:, 0] += 1.4 * X[2:-1, 3] + 1.4 * X[:-3, 3]          # j=3 drives i=0 at k=1 and k=3
    res = screen_lags(X, max_lag=5, n_perm=100, sis_c=0.0, seed=0)   # test only, no floor
    got = set(res.candidates[0])
    assert (3, 1) in got and (3, 3) in got, sorted(got)


# --------------------------------------------------------------------------- #
# 3. null behaviour
# --------------------------------------------------------------------------- #
def test_permutation_null_controls_fwer_on_iid_noise():
    """With the SIS floor off, at most ~alpha of target families may fire.

    ``sis_c=0`` is essential here: the SIS floor is a UNION and always contributes
    ``m_i`` candidates, so it would mask the test being measured.
    """
    alpha, P, reps = 0.05, 12, 10
    fired = 0
    for s in range(reps):
        res = screen_lags(noise_panel(P=P, T=600, seed=1000 + s), max_lag=4,
                          alpha=alpha, n_perm=200, sis_c=0.0, seed=s)
        fired += sum(any(j != i for j, _ in res.candidates[i]) for i in range(P))
    fwer = fired / float(P * reps)
    # Binomial(120, 0.05) has mean 6.0, sd 2.4 -- 15 is >3.7 sd out, so this bound is
    # tight enough to catch a broken null and loose enough not to flake.
    assert fired <= 15, f"observed FWER {fwer:.3f} (>{alpha}) -- {fired}/{P * reps}"


def test_null_threshold_rejects_degenerate_constant_streams():
    """Two constant (all-bits-zero) columns have phi == 1; the null must reject them.

    This is the case a hard-coded 1/3 or 1/4 threshold from ARROW's Theorem 4.2 would
    get catastrophically wrong.
    """
    rng = np.random.default_rng(11)
    X = rng.standard_normal((400, 6))
    X[:, 4] = 3.0                                     # constant
    X[:, 5] = 7.0                                     # constant
    res = screen_lags(X, max_lag=4, n_perm=200, sis_c=0.0, seed=0)
    assert res.S[:, 4, 5].max() == pytest.approx(1.0), "flat vs flat should score 1"
    assert res.threshold[4] >= 1.0, "threshold must rise to 1 for a degenerate target"
    assert (5, 1) not in res.candidates[4], "degenerate pair must not be selected"


def test_no_perm_means_no_test():
    X = noise_panel(P=8, T=300, seed=1)
    res = screen_lags(X, max_lag=3, n_perm=0, sis_c=0.0, force_self=False)
    assert np.isinf(res.threshold).all()
    assert res.n_candidates == 0 and res.reduction == 0.0


# --------------------------------------------------------------------------- #
# 4. force_self
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("maker", ["noise", "constant", "allnan_col"])
def test_force_self_is_unconditional(maker):
    rng = np.random.default_rng(21)
    X = rng.standard_normal((250, 9))
    if maker == "constant":
        X[:] = 1.0
    elif maker == "allnan_col":
        X[:, 3] = np.nan
    K = 6
    res = screen_lags(X, max_lag=K, n_perm=25, sis_c=0.0, seed=0)
    got = set(res.triples())
    for i in range(X.shape[1]):
        for k in range(1, K + 1):
            assert (i, i, k) in got, f"missing forced self term ({i},{i},{k}) [{maker}]"
    assert res.n_candidates >= X.shape[1] * K


def test_force_self_off_lets_self_be_tested():
    X = noise_panel(P=8, T=300, seed=2)
    res = screen_lags(X, max_lag=3, n_perm=50, sis_c=0.0, force_self=False, seed=0)
    assert res.n_candidates < 8 * 3, "self terms should not all survive a test on noise"


# --------------------------------------------------------------------------- #
# 5. backend equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stat", ["phi", "arrow"])
@pytest.mark.parametrize("miss", [0.0, 0.2])
def test_dense_and_bits_agree(stat, miss):
    rng = np.random.default_rng(33)
    X = rng.standard_normal((511, 11))                  # odd n -> odd packed width
    X[:, 2] = np.cumsum(rng.standard_normal(511))
    if miss:
        X[rng.random(X.shape) < miss] = np.nan
    kw = dict(max_lag=5, stat=stat, n_perm=30, seed=0)
    a = screen_lags(X, method="dense", **kw)
    b = screen_lags(X, method="bits", **kw)
    assert np.abs(a.S - b.S).max() <= 1e-6            # float32 tolerance as specified
    assert np.abs(a.S - b.S).max() == 0.0             # in fact bit-identical
    assert np.array_equal(a.n_valid, b.n_valid)
    assert np.array_equal(a.threshold, b.threshold)
    assert a.candidates == b.candidates


def test_auto_backend_picks_dense_at_normal_sizes():
    """dense measured 7-27x faster than bits at every tested P (see the module
    docstring): ``auto`` must therefore not pick bits unless memory forces it."""
    X = noise_panel(P=16, T=200, seed=0)
    assert screen_lags(X, max_lag=4, n_perm=0).meta["method"] == "dense"
    assert screen_lags(X, max_lag=4, n_perm=0,
                       mem_budget_bytes=1024).meta["method"] == "bits"


def test_popcount_lut_matches_unpackbits():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, size=(9, 64), dtype=np.uint8)
    assert np.array_equal(popcount_bytes(a), np.unpackbits(a, axis=1).sum(1))


# --------------------------------------------------------------------------- #
# 6. missingness
# --------------------------------------------------------------------------- #
def test_missingness_is_graceful_and_finite():
    X, edges = cached_var(P=100, T=800, seed=0)
    full = screen_lags(X, max_lag=8, n_perm=60, seed=0)
    rng = np.random.default_rng(123)
    Xm = X.copy()
    Xm[rng.random(X.shape) < 0.20] = np.nan
    miss = screen_lags(Xm, max_lag=8, n_perm=60, seed=0)

    assert np.isfinite(miss.S).all(), "S must never contain NaN or inf"
    assert np.isfinite(miss.threshold).all()
    assert (miss.S >= 0.0).all() and (miss.S <= 1.0).all()
    assert miss.n_valid.max() < full.n_valid.max()     # fewer usable pairs, as expected
    r_full, r_miss = recall(full, edges), recall(miss, edges)
    assert r_full >= 0.99
    assert r_miss >= 0.90, f"recall collapsed under 20% MCAR: {r_miss:.3f}"
    assert miss.reduction <= full.reduction            # the SIS floor shrinks with n


def test_explicit_obs_mask_matches_nan_encoding():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((300, 7))
    m = rng.random(X.shape) >= 0.25
    Xn = np.where(m, X, np.nan)
    a = screen_lags(X, obs=m, max_lag=4, n_perm=20, seed=0)
    b = screen_lags(Xn, max_lag=4, n_perm=20, seed=0)
    assert np.array_equal(a.S, b.S) and a.candidates == b.candidates


def test_all_missing_gives_zero_stats_but_still_self_terms():
    X = np.full((120, 5), np.nan)
    res = screen_lags(X, max_lag=3, n_perm=10, seed=0)
    assert np.all(res.S == 0.0) and np.all(res.n_valid == 0)
    assert len(res.triples()) == 5 * 3                 # only the forced self terms


# --------------------------------------------------------------------------- #
# panels, shapes, and the reduction contract
# --------------------------------------------------------------------------- #
def test_no_cross_entity_pairs_are_formed():
    """Pooled counts over a stacked panel must equal the sum of per-entity counts."""
    rng = np.random.default_rng(8)
    A = rng.standard_normal((200, 6))
    B = rng.standard_normal((200, 6)) * 3.0 + 5.0
    kw = dict(max_lag=5, n_perm=0, sis_c=0.0)
    na = screen_lags(A, **kw).n_valid
    nb = screen_lags(B, **kw).n_valid
    nab = screen_lags(np.stack([A, B]), **kw).n_valid
    assert np.array_equal(nab, na + nb), "a cross-entity lag pair leaked in"
    assert screen_lags(np.stack([A, B]), **kw).meta["n_entities"] == 2


def test_shapes_dtypes_and_reduction():
    X = noise_panel(P=9, T=300, seed=0)
    K = 4
    res = screen_lags(X, max_lag=K, n_perm=20, seed=0)
    P = X.shape[1]
    assert res.S.shape == (K, P, P) and res.S.dtype == np.float64
    assert res.n_valid.shape == (K, P, P) and res.n_valid.dtype == np.int64
    assert res.threshold.shape == (P,)
    assert set(res.candidates) == set(range(P))
    assert 0.0 < res.reduction <= 1.0
    assert res.reduction == pytest.approx(res.n_candidates / (P * P * K))
    assert res.argmax_lag.shape == (P, P)
    assert res.argmax_lag.min() >= 1 and res.argmax_lag.max() <= K
    assert all(1 <= k <= K and 0 <= j < P for i in res.candidates
               for (j, k) in res.candidates[i])
    for key in ("method", "stat", "w_short", "w_long", "n_perm", "t_stat_s"):
        assert key in res.meta


def test_statistics_stay_in_unit_interval():
    rng = np.random.default_rng(9)
    X = np.cumsum(rng.standard_normal((400, 8)), 0)
    for stat in ("phi", "arrow"):
        res = screen_lags(X, max_lag=4, stat=stat, n_perm=0)
        assert res.S.min() >= 0.0 and res.S.max() <= 1.0


def test_bad_arguments_raise():
    X = noise_panel(P=4, T=50, seed=0)
    for kw in (dict(stat="nope"), dict(method="nope"), dict(alpha=0.0),
               dict(max_lag=0), dict(max_lag=60)):
        with pytest.raises(ValueError):
            screen_lags(X, n_perm=0, **kw)
    with pytest.raises(ValueError):
        screen_lags(np.zeros((2, 3, 4, 5)), n_perm=0)
    with pytest.raises(ValueError):
        screen_lags(X, obs=np.ones((3, 3), bool), n_perm=0)


# --------------------------------------------------------------------------- #
# 7. numpy-only
# --------------------------------------------------------------------------- #
def test_module_source_has_no_forbidden_imports():
    src = open(_SCREEN_PY).read()
    for bad in ("scipy", "sklearn", "numba", "numexpr", "pandas", "polars", "torch"):
        assert f"import {bad}" not in src and f"from {bad}" not in src, bad


def test_imports_with_numpy_alone():
    """Import the module in a subprocess where every non-numpy dep is unimportable."""
    code = (
        "import sys, importlib.util, importlib.abc\n"
        "BAD = {'scipy','sklearn','numba','numexpr','pandas','polars','torch',"
        "'matplotlib','cafe'}\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BAD:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        f"spec = importlib.util.spec_from_file_location('_scr', {_SCREEN_PY!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['_scr'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "import numpy as np\n"
        "r = mod.screen_lags(np.random.default_rng(0).standard_normal((120, 5)),"
        " max_lag=3, n_perm=5)\n"
        "assert r.S.shape == (3, 5, 5)\n"
        "print('OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout
