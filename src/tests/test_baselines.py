"""Tests for cafe.baselines: the classical / non-deep imputers shipped as first-class
library methods (the runnable backing for the paper's tab:special).

For every method we assert: finite output, same shape, observed cells preserved exactly,
container round-trip (numpy/pandas/polars, 1D/2D), and -- for the methods ported from a
bench reference -- bit-exact parity with that reference. We also exercise degenerate
inputs (all-missing column, 1x1, constant, all-missing matrix).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from cafe import baselines as B

# optional deps -- some methods are skipped if absent
_HAVE_SKLEARN = True
try:
    import sklearn  # noqa: F401
except Exception:
    _HAVE_SKLEARN = False

_HAVE_GCIMPUTE = True
try:
    import gcimpute  # noqa: F401
except Exception:
    _HAVE_GCIMPUTE = False

_OPTIONAL = {
    "knn": _HAVE_SKLEARN,
    "mice": _HAVE_SKLEARN,
    "gaussian_copula": _HAVE_GCIMPUTE,
}


def _available_methods():
    return [m for m in B.list_methods() if _OPTIONAL.get(m, True)]


def _data(T=40, N=6, seed=0, frac=0.2):
    rng = np.random.default_rng(seed)
    truth = np.cumsum(rng.standard_normal((T, N)), axis=0)
    X = truth.copy()
    X[rng.random((T, N)) < frac] = np.nan
    return X, truth


# --------------------------------------------------------------------------- #
# Registry sanity
# --------------------------------------------------------------------------- #
def test_registry_consistency():
    names = B.list_methods()
    assert set(names) == set(B.METHODS)
    causal = set(B.list_methods(causal=True))
    batch = set(B.list_methods(causal=False))
    assert causal.isdisjoint(batch)
    assert causal | batch == set(names)
    # known causal/batch labels match the paper's thesis
    assert "locf" in causal and "online_trmf" in causal and "ewcov" in causal
    assert "softimpute" in batch and "linear_interp" in batch and "knn" in batch
    for m in names:
        assert B.is_causal(m) is B.METHODS[m][1]


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        B.impute(_data()[0], method="does_not_exist")


# --------------------------------------------------------------------------- #
# Core contract: finite, same-shape, observed-cells-preserved, for every method
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _available_methods())
def test_contract(method):
    X, _ = _data()
    obs = np.isfinite(X)
    out = np.asarray(B.impute(X.copy(), method=method))
    assert out.shape == X.shape, f"{method}: shape changed"
    assert np.isfinite(out).all(), f"{method}: non-finite output"
    assert np.allclose(out[obs], X[obs]), f"{method}: observed cells altered"


@pytest.mark.parametrize("method", _available_methods())
def test_named_function_matches_dispatch(method):
    X, _ = _data(seed=1)
    fn = B.METHODS[method][0]
    a = np.asarray(fn(X.copy()))
    b = np.asarray(B.impute(X.copy(), method=method))
    assert np.array_equal(a, b, equal_nan=False)


# --------------------------------------------------------------------------- #
# Container round-trip: numpy/pandas/polars, 1D and 2D
# --------------------------------------------------------------------------- #
def test_numpy_1d_and_2d():
    X, _ = _data(N=4)
    out2d = np.asarray(B.softimpute(X.copy()))
    assert out2d.shape == X.shape and np.isfinite(out2d).all()
    x1d = X[:, 0]
    out1d = np.asarray(B.locf(x1d.copy()))
    assert out1d.ndim == 1 and out1d.shape == x1d.shape


def test_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    X, _ = _data(N=4)
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])
    out = B.ewcov(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(df.columns)
    obs = np.isfinite(X)
    assert np.allclose(out.to_numpy()[obs], X[obs])
    # 1D Series
    s = pd.Series(X[:, 0], name="v")
    os_ = B.kalman_local_level(s)
    assert isinstance(os_, pd.Series) and os_.name == "v"


def test_polars_roundtrip():
    pl = pytest.importorskip("polars")
    X, _ = _data(N=4)
    pf = pl.DataFrame({f"c{i}": X[:, i] for i in range(X.shape[1])})
    out = B.rolling_mean(pf, W=8)
    assert isinstance(out, pl.DataFrame) and out.columns == pf.columns
    ps = pl.Series("v", X[:, 1])
    ops = B.mean_impute(ps)
    assert isinstance(ops, pl.Series)


# --------------------------------------------------------------------------- #
# Parity with bench references (the methods we ported should match bit-exactly)
# --------------------------------------------------------------------------- #
_BENCH = os.path.join(os.path.dirname(__file__), "..", "..", "bench")


def _bench_available():
    return os.path.isdir(_BENCH)


@pytest.mark.skipif(not _bench_available(), reason="bench/ reference tree not present")
@pytest.mark.parametrize("seed", [3, 7, 11])
def test_parity_with_bench(seed):
    sys.path.insert(0, os.path.abspath(_BENCH))
    import importlib

    X, _ = _data(T=50, N=7, seed=seed, frac=0.2)

    def md(a, b):
        return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    checks = {}

    MS = importlib.import_module("m_softimpute")
    checks["softimpute"] = md(B.softimpute(X.copy()), MS.impute(X.copy(), {}))

    MT = importlib.import_module("m_trmf")
    checks["trmf"] = md(B.trmf(X.copy()), MT.impute(X.copy(), {}))

    MN = importlib.import_module("m_naive")
    checks["linear_interp"] = md(B.linear_interp(X.copy()),
                                 MN.linear_interp(X.copy(), None))
    # NB: B.locf is intentionally NOT compared to MN.locf here -- the library locf is
    # strictly causal (leading pre-first-obs cells -> 0), whereas MN.locf fills them with
    # the future-peeking column mean. They match on data without leading gaps but diverge
    # at the leading edge by design; locf causality is asserted separately below.

    CE = importlib.import_module("c_chal_ewcov")
    checks["ewcov"] = md(B.ewcov(X.copy()), CE._ewcov_2d(X.copy(), {}))

    CO = importlib.import_module("c_online_trmf")
    checks["online_trmf"] = md(B.online_trmf(X.copy()), CO.online_impute(X.copy(), {}))

    for name, dev in checks.items():
        assert dev < 1e-8, f"{name}: lib vs bench max|diff|={dev:.2e} (expected bit-exact)"


# --------------------------------------------------------------------------- #
# Degenerate / extreme inputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", _available_methods())
def test_all_missing_column(method):
    X, _ = _data(N=5)
    X[:, 2] = np.nan                                   # one entirely-missing column
    out = np.asarray(B.impute(X.copy(), method=method))
    assert out.shape == X.shape
    assert np.isfinite(out).all(), f"{method}: NaN with an all-missing column"


@pytest.mark.parametrize("method", _available_methods())
def test_1x1(method):
    out = np.asarray(B.impute(np.array([[3.5]]), method=method))
    assert out.shape == (1, 1) and np.isfinite(out).all()


@pytest.mark.parametrize("method", _available_methods())
def test_1x1_missing(method):
    out = np.asarray(B.impute(np.array([[np.nan]]), method=method))
    assert out.shape == (1, 1) and np.isfinite(out).all()


# Dynamic / state-space methods legitimately approach (rather than reproduce) a
# constant: a local-level Kalman filter ramps from its zero prior, and online-TRMF
# fits a learned low-rank state. We only require finiteness + a loose closeness for them.
_APPROX_CONST = {"kalman_local_level", "online_trmf"}


@pytest.mark.parametrize("method", _available_methods())
def test_constant_column(method):
    T, N = 30, 4
    X = np.full((T, N), 2.0)
    X[5:9, 1] = np.nan                                 # gap in a constant column
    out = np.asarray(B.impute(X.copy(), method=method))
    assert np.isfinite(out).all()
    if method not in _APPROX_CONST:
        # exact / interpolation / mean-type methods reproduce the constant precisely
        assert np.allclose(out[5:9, 1], 2.0, atol=1e-6), f"{method}: constant gap off"
    else:
        # dynamic state-space / learned low-rank methods only approach the constant;
        # require finiteness and the same order of magnitude (no blow-up)
        assert np.all(np.abs(out[5:9, 1] - 2.0) < 2.0), f"{method}: constant gap diverged"


@pytest.mark.parametrize("method", _available_methods())
def test_all_missing_matrix(method):
    X = np.full((10, 3), np.nan)
    out = np.asarray(B.impute(X.copy(), method=method))
    assert out.shape == X.shape and np.isfinite(out).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_locf_is_strictly_causal_at_leading_edge():
    """Regression: the library's causal-labelled LOCF must NOT peek at the future to
    fill cells before a column's first observation (it fills them with 0, not the
    full-column mean). Verified by truncation invariance via cafe.audit."""
    import numpy as np
    import cafe
    from cafe import audit, baselines
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    X[:5, 0] = np.nan                      # force a leading gap in column 0
    rep = audit.leakage_report(lambda Z: baselines.impute(Z, method="locf"), X)
    assert rep["causal"] and rep["max_revision"] < 1e-9, \
        f"library LOCF leaks: causal={rep['causal']} max_revision={rep['max_revision']:.3e}"
