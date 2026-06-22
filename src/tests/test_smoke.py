"""Smoke tests: container round-trip (numpy/pandas/polars), causality, and that every
advertised capability returns something sane."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import cafe


def _data(T=120, N=5, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, 2)); W = rng.standard_normal((2, N))
    X = Z @ W + 0.2 * rng.standard_normal((T, N))
    X[rng.random((T, N)) < 0.15] = np.nan
    return X


def test_numpy_roundtrip():
    X = _data()
    out = cafe.impute(X)
    assert isinstance(out, np.ndarray) and out.shape == X.shape
    assert np.isfinite(out).all()                      # no NaN/Inf out


def test_numpy_1d():
    x = _data(N=1)[:, 0]
    out = cafe.impute(x)
    assert out.ndim == 1 and out.shape == x.shape and np.isfinite(out).all()


def test_pandas_roundtrip():
    import pandas as pd
    X = _data()
    df = pd.DataFrame(X, columns=[f"s{i}" for i in range(X.shape[1])],
                      index=pd.date_range("2020-01-01", periods=len(X), freq="h"))
    out = cafe.impute(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(df.columns)
    assert out.index.equals(df.index)
    assert out.notna().all().all()


def test_polars_roundtrip():
    import polars as pl
    X = _data()
    df = pl.DataFrame({f"s{i}": X[:, i] for i in range(X.shape[1])})
    out = cafe.impute(df)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == df.columns
    assert out.null_count().sum_horizontal().item() == 0


def test_capabilities():
    res = cafe.CAFE().run(_data())
    assert np.isfinite(np.asarray(res.imputed)).all()
    assert res.factors().shape[0] == 120
    assert np.asarray(res.anomaly_scores()).shape[0] == 120
    net = res.dependency_network()
    assert net.shape == (5, 5)
    assert np.allclose(np.diag(net), 1.0, atol=1e-6)   # correlation matrix
    parts = res.decompose()
    assert set(parts) == {"level", "season", "factor"}
    assert isinstance(res.params["nu"], float)


def test_forecast():
    out = cafe.CAFE().forecast(_data(), horizon=12)
    assert np.asarray(out).shape[0] == 12 and np.isfinite(np.asarray(out)).all()


def test_causality_no_lookahead():
    """Imputations at early cells must not change when future rows are appended."""
    X = _data(T=100)
    a = np.asarray(cafe.impute(X[:60]))
    b = np.asarray(cafe.impute(X))[:60]
    obs = ~np.isnan(X[:60])
    # past imputations identical whether or not the future is present
    assert np.allclose(a[~obs], b[~obs], atol=1e-8)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} smoke tests passed.")
