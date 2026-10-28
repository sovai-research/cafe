"""CAFE-CD Stage 0 gate: the discovery surface must be FREE.

Stage 0 adds a residual/loadings read-out and a factor-adjusted dependency network on top
of the existing causal one-pass model. The whole point is that this is pure introspection:
the imputation itself, the published legacy dependency network, and the numpy-only core
must all be untouched. These tests are the contract that says so.

What is asserted
----------------
1.  PRODUCTION IS BIT-IDENTICAL.  ``cafe.impute`` on every one of the 16 harness datasets
    hashes to the value recorded at git f35d062 (the commit before Stage 0). Also, the
    traced path (``record=True``) returns byte-equal fills to the lean path, so switching
    tracing on can never move a number.
2.  NUMPY-ONLY.  ``import cafe`` + a full traced run + the new APIs must not pull in
    scipy / sklearn / pandas / polars / matplotlib / numba / torch.
3.  LEGACY NETWORK FROZEN.  ``dependency_network()`` with NO kwargs hashes to its
    pre-change value on a seeded fixture (paper/tables/byproduct.tex depends on it).
4.  RESIDUAL IDENTITY.  At observed cells the residual equals X - level - season -
    time_fe - factor to machine precision, and is NaN (never 0) at imputed cells.
5.  FREE CERTIFICATE.  The residuals inherit the imputer's causality: truncating the
    future revises nothing.
6.  ``verify_causal_series`` separates a genuinely causal reporter from a leaky one.
7.  ``partial=True`` returns a valid symmetric unit-diagonal correlation-like matrix.

Note on the hard-coded hashes: they are exact float64 checksums, so a mismatch means
either a real behaviour change (the thing this gate exists to catch) or a change of
numeric environment. They were verified stable across BLAS thread settings on the
reference machine.
"""
import hashlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                 # src/   -> cafe
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench"))  # bench/ -> harness

import numpy as np  # noqa: E402

import cafe  # noqa: E402
import cafe.audit  # noqa: E402


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. production output is bit-identical to the pre-Stage-0 baseline
# --------------------------------------------------------------------------- #
# sha256 of cafe.impute(masked, meta) for every harness dataset, masked with seed=0,
# recorded on git f35d062 (immediately before the Stage 0 commit).
BASELINE_SHA = {
    "1d_ar_mcar20": "74412536b8db594fb7cd73221a17b86c62c48e13296c0040da3ff764da3f0d24",
    "1d_seasonal_block": "ee175513e2b437a3c4ff868c1032cec9fd03ba9be8d2e37a0b984ab561dd5782",
    "2d_small_mcar20": "f4dc8ca6a9b785a03d41b13e145e8e5898d9b20d0c491114fb96aa8c65be7691",
    "2d_small_mar30": "c8c0ce51181988e01fb4a839d37de94ac44a313916468fa02e46e284df6e83b1",
    "2d_small_block20": "010910959087929109292c0200557eb0abddc06fdfcc3e81cb0910e812178843",
    "2d_small_mnar30": "66be5e1bea0d701cc2e3cb1246a36ffd3955b4761e33252a5eeee06a87420ad3",
    "2d_large_mcar30": "b51c4d3c18b81e5199b8298a86bd15d0ba02399eb357aa54311f8412fcfd6320",
    "2d_large_block30": "0849b8a6806168bda6e14544fcf7427c6c86ae42d757c2c05a992e20217333f3",
    "2d_highmiss_mcar60": "229cc680271cc47da410add29019ce830d6e56755f64e138829bc95b56af3c0b",
    "2d_highrank_mcar30": "2be699c3133aec5ff1ccbdbeb6494243441a8abd9e9ce9f3949ac5432af51ef3",
    "2d_drift_block": "0ba83d8df4afd8912814a8113718f85edd2ba6a630a959be9727eac33b4e4530",
    "2d_heavytail_mcar30": "9690082e64a9cdf7077d91b5b28a3339c59016f03dcec19fb14844475fa641c2",
    "panel_mcar20": "352de315b853eb2d82893ed68025ba8995292c57cd73488ab4287dc00f99a98d",
    "panel_mar40": "eab4ec0f3922e00647e68c16ec73866dee006578e4e1f3d1cf50da0343cb48e4",
    "panel_block30": "b2d85a766c9e7ecdc1a51850048eaf3a94b130347ff41e64c80a83a9b5e6e431",
    "panel_blackout25": "74070037a958c4f06a0669170b17b63fb1e421c7ee1dfe134cd006645855e6a9",
}


def _harness():
    return pytest.importorskip("harness", reason="bench/ harness not importable")


def _masked(name):
    h = _harness()
    clean, meta, mech, rate = h.DATASETS[name]
    M = h.MASKERS[mech](clean, rate, seed=0)
    Xobs = clean.copy()
    Xobs[M] = np.nan
    return Xobs, dict(meta)


@pytest.mark.parametrize("name", sorted(BASELINE_SHA))
def test_impute_bit_identical_to_pre_stage0(name):
    """Stage 0 must not move a single imputed value on any harness dataset."""
    h = _harness()
    assert name in h.DATASETS, f"{name} vanished from the harness registry"
    Xobs, meta = _masked(name)
    pred = np.asarray(cafe.impute(Xobs.copy(), meta), float)
    assert _sha(pred) == BASELINE_SHA[name], f"{name}: imputation changed"


def test_all_sixteen_datasets_covered():
    h = _harness()
    assert set(h.DATASETS) == set(BASELINE_SHA)
    assert len(BASELINE_SHA) == 16


@pytest.mark.parametrize("name", ["2d_small_mcar20", "2d_drift_block", "1d_ar_mcar20"])
def test_record_does_not_change_values(name):
    """The trace (record=True) is pure introspection: identical fills, to the bit.

    This is the machine-independent half of the guarantee -- it holds regardless of BLAS,
    and it is exactly the invariant the new resid_c/w_ver/W-snapshot recording could break.
    """
    Xobs, _meta = _masked(name)
    lean = np.asarray(cafe.impute(Xobs.copy()), float)
    traced = np.asarray(cafe.CAFE().run(Xobs.copy()).imputed, float)
    assert np.array_equal(lean, traced)


# --------------------------------------------------------------------------- #
# 2. the core stays numpy-only
# --------------------------------------------------------------------------- #
def test_core_is_numpy_only():
    """A traced run plus every Stage 0 API must not import a heavy optional dep."""
    import subprocess

    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {os.path.join(HERE, '..')!r})\n"
        "import cafe, cafe.audit\n"
        "rng = np.random.default_rng(0)\n"
        "Z = rng.standard_normal((90, 2)); W = rng.standard_normal((2, 5))\n"
        "X = Z @ W + 0.2 * rng.standard_normal((90, 5))\n"
        "X[rng.random((90, 5)) < 0.15] = np.nan\n"
        "r = cafe.CAFE().run(X)\n"
        "r.residuals(); r.loadings()\n"
        "r.dependency_network(); r.dependency_network(factor_adjusted=True)\n"
        "r.dependency_network(factor_adjusted=True, partial=True)\n"
        "cafe.audit.verify_causal_series(lambda Z_, t: np.zeros((2, 2)), X, n_prefixes=2)\n"
        "heavy = [m for m in ('scipy', 'sklearn', 'pandas', 'polars', 'matplotlib',\n"
        "                     'numba', 'torch') if m in sys.modules]\n"
        "print('HEAVY=' + ','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "HEAVY=\n" in out.stdout or out.stdout.strip() == "HEAVY=", out.stdout


# --------------------------------------------------------------------------- #
# 3. the legacy dependency network is frozen (paper/tables/byproduct.tex)
# --------------------------------------------------------------------------- #
LEGACY_DEPNET_SHA = "82e71221d78ec032bdcf3e5f6e7aed31ff409605d8b380bc65e6d1970d820ff8"


def _depnet_fixture():
    """Seeded 3-factor panel with 15% MCAR holes. Frozen -- do not retune."""
    rng = np.random.default_rng(20260904)
    T, N = 300, 8
    Z = np.zeros((T, 3))
    for t in range(1, T):
        Z[t] = 0.9 * Z[t - 1] + rng.standard_normal(3)
    X = Z @ rng.standard_normal((3, N)) + 0.4 * rng.standard_normal((T, N))
    X[rng.random((T, N)) < 0.15] = np.nan
    return X


def test_legacy_dependency_network_byte_identical():
    res = cafe.CAFE().run(_depnet_fixture())
    net = np.asarray(res.dependency_network(), float)
    assert _sha(net) == LEGACY_DEPNET_SHA, "the no-kwargs dependency network changed"


def test_new_network_paths_do_not_disturb_the_legacy_one():
    """Calling the new estimators must not mutate cached state behind the legacy path."""
    res = cafe.CAFE().run(_depnet_fixture())
    first = np.asarray(res.dependency_network(), float)
    res.dependency_network(factor_adjusted=True)
    res.dependency_network(factor_adjusted=True, partial=True, robust=True)
    res.residuals(robust=True)
    again = np.asarray(res.dependency_network(), float)
    assert np.array_equal(first, again)
    assert _sha(again) == LEGACY_DEPNET_SHA


# --------------------------------------------------------------------------- #
# 4. residual identity + NaN masking
# --------------------------------------------------------------------------- #
def _res_fixture(T=200, N=6, seed=7, miss=0.2):
    rng = np.random.default_rng(seed)
    Z = np.zeros((T, 2))
    for t in range(1, T):
        Z[t] = 0.85 * Z[t - 1] + rng.standard_normal(2)
    X = Z @ rng.standard_normal((2, N)) + 0.3 * rng.standard_normal((T, N))
    X[rng.random((T, N)) < miss] = np.nan
    return X


def test_residual_identity_at_observed_cells():
    """E == X - level - season - time_fe - factor, exactly, wherever X was observed."""
    X = _res_fixture()
    res = cafe.CAFE().run(X)
    E = np.asarray(res.residuals(factor_adjusted=True, robust=False), float)
    parts = {k: np.asarray(v, float) for k, v in res.decompose(full=True).items()}
    C = res._comp()
    obs = np.isfinite(X)
    expect = X - C["level"] - C["season"] - C["time_fe"] - C["factor"]
    assert np.allclose(E[obs], expect[obs], rtol=0, atol=1e-12)
    # and it coincides with decompose()'s own residual channel at observed cells
    assert np.allclose(E[obs], parts["residual"][obs], rtol=0, atol=1e-12)


def test_residual_without_factor_adjustment_keeps_the_factor():
    X = _res_fixture()
    res = cafe.CAFE().run(X)
    C = res._comp()
    obs = np.isfinite(X)
    E_raw = np.asarray(res.residuals(factor_adjusted=False), float)
    E_fa = np.asarray(res.residuals(factor_adjusted=True), float)
    assert np.allclose((E_raw - E_fa)[obs], C["factor"][obs], rtol=0, atol=1e-12)
    assert not np.allclose(E_raw[obs], E_fa[obs])          # the factor is actually there


def test_masked_imputed_cells_are_nan_not_zero():
    X = _res_fixture()
    res = cafe.CAFE().run(X)
    E = np.asarray(res.residuals(mask_imputed=True), float)
    imp = ~np.isfinite(X)
    assert imp.any()
    assert np.all(np.isnan(E[imp])), "imputed cells must be NaN"
    assert not np.any(E[imp] == 0.0), "a 0 would be indistinguishable from a datum"
    assert np.all(np.isfinite(E[np.isfinite(X)]))
    # unmasked they become finite numbers again -- which is exactly the hazard: they are
    # a deterministic function of the model's own fill, not data. With the robust
    # (winsorized) residual they are identically 0 by construction.
    E0 = np.asarray(res.residuals(mask_imputed=False), float)
    assert np.all(np.isfinite(E0[imp]))
    E0r = np.asarray(res.residuals(robust=True, factor_adjusted=False,
                                   mask_imputed=False), float)
    assert np.allclose(E0r[imp], 0.0, atol=0.0)


def test_robust_residual_is_winsorized_and_not_the_default():
    X = _res_fixture()
    X[10, 0] = 250.0                                       # a single extreme outlier
    res = cafe.CAFE().run(X)
    raw = np.asarray(res.residuals(robust=False), float)
    rob = np.asarray(res.residuals(robust=True), float)
    assert np.nanmax(np.abs(rob)) < np.nanmax(np.abs(raw)), "robust must clip the tail"
    # the DEFAULT is the raw (tail-preserving) residual -- direction identification needs it
    assert np.array_equal(np.nan_to_num(raw), np.nan_to_num(
        np.asarray(res.residuals(), float)))


def test_optional_trace_fields_are_guarded_by_presence():
    """A trace from an older core (no resid_c / w_ver) must degrade, not explode."""
    X = _res_fixture(T=120, N=4, seed=9)
    res = cafe.CAFE().run(X)
    for row in res._trace:                                 # simulate the old trace shape
        row.pop("resid_c", None)
        row.pop("w_ver", None)
    res._C = None
    res._Cx = None
    assert res._trace_field("resid_c") is None
    assert res._trace_field("w_ver") is None
    np.asarray(res.residuals())                            # raw path still works
    _Ws, _Sf, wv = res.loadings()
    assert wv.shape == (X.shape[0],) and np.all(wv == 0)   # falls back, does not raise
    with pytest.raises(NotImplementedError):
        res.residuals(robust=True)


def test_loadings_snapshots_are_consistent():
    X = _res_fixture(T=300)
    res = cafe.CAFE().run(X)
    Ws, Sf, wv = res.loadings()
    assert Ws.ndim == 3 and Ws.shape[1] == X.shape[1]
    assert Sf.shape == (Ws.shape[0], Ws.shape[2])
    assert wv.shape == (X.shape[0],)
    assert np.all(np.diff(wv) >= 0), "the loading version must be monotone in time"
    assert wv.max() <= Ws.shape[0]
    # the version recorded on a row must be the one that was live BEFORE that row's refit
    assert wv[0] == 0


# --------------------------------------------------------------------------- #
# 5. the causality certificate comes for free
# --------------------------------------------------------------------------- #
def test_residuals_are_certified_causal():
    X = _res_fixture(T=150, N=5, seed=3)
    rep = cafe.audit.verify_causal(
        lambda Z: np.nan_to_num(cafe.CAFE().run(Z).residuals()), X, n_prefixes=6)
    assert rep["n_comparisons"] > 0
    assert rep["max_revision"] == 0.0, rep
    assert rep["causal"]


def test_factor_adjusted_network_at_t_is_causal():
    X = _res_fixture(T=150, N=5, seed=4)
    rep = cafe.audit.verify_causal_series(
        lambda Z, t: cafe.CAFE().run(Z).dependency_network(factor_adjusted=True, at=t),
        X, n_prefixes=4)
    assert rep["max_revision"] <= 1e-6, rep
    assert rep["causal"]


# --------------------------------------------------------------------------- #
# 6. verify_causal_series itself
# --------------------------------------------------------------------------- #
def test_verify_causal_series_accepts_a_causal_reporter():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))

    def causal_fn(Z, t):                       # uses rows <= t only
        return np.array([[np.nansum(Z[: t + 1, j])] for j in range(Z.shape[1])])

    rep = cafe.audit.verify_causal_series(causal_fn, X, n_prefixes=5)
    assert rep["causal"] and rep["max_revision"] == 0.0
    assert rep["n_comparisons"] > 0


def test_verify_causal_series_rejects_a_leaky_reporter():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4))

    def leaky_fn(Z, t):                        # ignores t: uses the WHOLE array
        return np.array([[np.nansum(Z[:, j])] for j in range(Z.shape[1])])

    rep = cafe.audit.verify_causal_series(leaky_fn, X, n_prefixes=5)
    assert not rep["causal"]
    assert rep["max_revision"] > 1e-6


def test_verify_causal_series_on_set_valued_output():
    """Indicator (0/1) outputs make max_revision read as 'did the reported set change'."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 3))

    def stable_set(Z, t):
        return (np.abs(np.nanmean(Z[: t + 1], axis=0)) > 1e9).astype(float)

    def flipping_set(Z, t):
        return (np.arange(Z.shape[1]) < (Z.shape[0] % 3)).astype(float)

    assert cafe.audit.verify_causal_series(stable_set, X, n_prefixes=4)["causal"]
    bad = cafe.audit.verify_causal_series(flipping_set, X, n_prefixes=4)
    assert not bad["causal"] and bad["max_revision"] == 1.0


# --------------------------------------------------------------------------- #
# 7. partial correlations are a valid matrix
# --------------------------------------------------------------------------- #
def test_partial_network_is_symmetric_unit_diagonal_and_bounded():
    X = _res_fixture(T=250, N=7, seed=11)
    res = cafe.CAFE().run(X)
    P = np.asarray(res.dependency_network(factor_adjusted=True, partial=True), float)
    N = X.shape[1]
    assert P.shape == (N, N)
    assert np.all(np.isfinite(P))
    assert np.allclose(P, P.T, atol=1e-10)
    assert np.allclose(np.diag(P), 1.0, atol=1e-12)
    off = P[~np.eye(N, dtype=bool)]
    assert off.min() >= -1.0 and off.max() <= 1.0


def test_factor_adjusted_network_is_a_correlation_matrix():
    X = _res_fixture(T=250, N=7, seed=12)
    res = cafe.CAFE().run(X)
    C = np.asarray(res.dependency_network(factor_adjusted=True), float)
    assert np.allclose(C, C.T, atol=1e-10)
    assert np.allclose(np.diag(C), 1.0, atol=1e-12)
    assert C.min() >= -1.0 and C.max() <= 1.0


def test_at_argument_truncates_the_network():
    """`at` must give the network AS OF t: rows after t carry zero weight."""
    X = _res_fixture(T=200, N=5, seed=13)
    res_full = cafe.CAFE().run(X)
    res_pref = cafe.CAFE().run(X[:120])
    a = res_full.dependency_network(factor_adjusted=True, at=119)
    b = res_pref.dependency_network(factor_adjusted=True, at=119)
    assert np.allclose(a, b, atol=1e-10)


# --------------------------------------------------------------------------- #
# 8. panel surface
# --------------------------------------------------------------------------- #
def _panel_fixture(E=6, T=60, F=5, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((E * T, F))
    X[rng.random(X.shape) < 0.15] = np.nan
    meta = {"entity_ids": np.repeat(np.arange(E), T),
            "time_ids": np.tile(np.arange(T), E)}
    return X, meta


def test_panel_residuals_scatter_back_to_input_layout():
    X, meta = _panel_fixture()
    pr = cafe.CAFE().run(X, meta=meta)
    E, O = pr.residuals(stack=True)
    assert E.shape == X.shape and O.shape == X.shape
    assert np.array_equal(O, np.isfinite(X))
    assert np.all(np.isnan(E[~O]))
    per = pr.residuals(stack=False)
    assert set(per) == set(np.unique(meta["entity_ids"]).tolist())
    # every entity block reappears at exactly its own rows
    for e, (rows, _r) in pr._pe.items():
        assert np.allclose(np.nan_to_num(E[rows]), np.nan_to_num(per[e][0]))


def test_panel_legacy_dependency_network_unchanged_and_stats_pooling_works():
    X, meta = _panel_fixture()
    pr = cafe.CAFE().run(X, meta=meta)
    legacy = np.asarray(pr.dependency_network(), float)
    nets = [r.dependency_network() for _, (_, r) in pr._pe.items()]
    nets = [n for n in nets if np.all(np.isfinite(n))]
    assert np.array_equal(legacy, np.mean(nets, axis=0))    # untouched averaging path
    pooled = np.asarray(pr.dependency_network(factor_adjusted=True), float)
    assert pooled.shape == legacy.shape
    assert np.allclose(np.diag(pooled), 1.0, atol=1e-12)
    assert np.allclose(pooled, pooled.T, atol=1e-10)
