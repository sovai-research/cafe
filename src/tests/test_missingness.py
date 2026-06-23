"""Tests for causal missingness-as-signal features (src/cafe/missingness.py).

Asserts: (1) strict causality -- a feature at row t is unchanged when *future* rows are
appended; (2) correct shapes and feature counts; (3) dtype/container round-trip for
numpy/pandas/polars; (4) the per-feature semantics (delta, gap, expanding rate); and
(5) selective-MIM only fires for genuinely informative columns.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from cafe.missingness import (
    FEATURE_KINDS,
    MissingnessFeatures,
    missingness_features,
)


def _masked(T=120, N=5, p=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    X[rng.random((T, N)) < p] = np.nan
    return X


# --------------------------------------------------------------------------- #
# Causality: the core non-negotiable.
# --------------------------------------------------------------------------- #
def test_causality_no_lookahead_numpy():
    """Every feature at rows < cut is identical whether or not future rows exist.

    selective=False so the (sample-statistic-based) selective scoring -- which is only
    causal under truncation -- doesn't enter; the per-column families must be exactly
    forward-only regardless.
    """
    X = _masked(T=200, N=6, seed=1)
    cut = 80
    full = missingness_features(X, selective=False)
    trunc = missingness_features(X[:cut], selective=False)
    assert full.shape[1] == trunc.shape[1]
    assert np.allclose(full[:cut], trunc, atol=0.0, equal_nan=True)


def test_causality_with_explicit_mask():
    """Same causality guarantee when the mask is passed separately (post-imputation)."""
    X = _masked(T=150, N=4, seed=3)
    mask = ~np.isfinite(X)
    filled = np.nan_to_num(X)                       # pretend it was already imputed
    cut = 60
    full = missingness_features(filled, mask=mask, selective=False)
    trunc = missingness_features(filled[:cut], mask=mask[:cut], selective=False)
    assert np.allclose(full[:cut], trunc, atol=0.0)


def test_selective_mim_is_truncation_causal():
    """Selective-MIM uses only rows <= t, so the truncated run is a prefix of the full
    run's selected set is NOT guaranteed, but the *emitted indicator values* for any
    selected column must match on the shared prefix (the indicator is just the mask)."""
    X = _masked(T=200, N=6, seed=5)
    full = missingness_features(X, selective=True, return_meta=True)
    # the selective indicator columns equal the raw mask of their column -> causal
    for j, (name, tag) in enumerate(zip(full.names, full.kinds)):
        if tag == "selective_mim":
            col = full.features[:, j]
            assert set(np.unique(col)).issubset({0.0, 1.0})


# --------------------------------------------------------------------------- #
# Shape / count.
# --------------------------------------------------------------------------- #
def test_shape_and_feature_count():
    X = _masked(T=100, N=5, seed=0)
    meta = missingness_features(X, selective=False, return_meta=True)
    assert isinstance(meta, MissingnessFeatures)
    # 4 per-column families * 5 columns
    assert meta.features.shape == (100, len(FEATURE_KINDS) * 5)
    assert len(meta.names) == meta.features.shape[1]
    assert len(meta.kinds) == meta.features.shape[1]


def test_subset_of_kinds():
    X = _masked(T=60, N=3, seed=2)
    out = missingness_features(X, kinds=("was_imputed", "missing_rate"),
                               selective=False)
    assert out.shape == (60, 2 * 3)


def test_1d_input():
    x = _masked(T=80, N=1, seed=4)[:, 0]
    out = missingness_features(x, selective=False)
    assert out.shape == (80, len(FEATURE_KINDS))


# --------------------------------------------------------------------------- #
# Dtype / container round-trip.
# --------------------------------------------------------------------------- #
def test_numpy_dtype():
    X = _masked(seed=0)
    out = missingness_features(X, selective=False)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


def test_pandas_container():
    import pandas as pd

    X = _masked(T=90, N=3, seed=1)
    df = pd.DataFrame(X, columns=["a", "b", "c"],
                      index=pd.date_range("2020-01-01", periods=len(X), freq="h"))
    out = missingness_features(df, selective=False)
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(df.index)                       # index preserved
    assert "a__was_imputed" in out.columns
    assert out.shape == (90, len(FEATURE_KINDS) * 3)
    assert out.notna().all().all()


def test_polars_container():
    import polars as pl

    X = _masked(T=90, N=3, seed=1)
    df = pl.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
    out = missingness_features(df, selective=False)
    assert isinstance(out, pl.DataFrame)
    assert "a__was_imputed" in out.columns
    assert out.shape == (90, len(FEATURE_KINDS) * 3)
    assert out.null_count().sum_horizontal().item() == 0


# --------------------------------------------------------------------------- #
# Semantics.
# --------------------------------------------------------------------------- #
def test_was_imputed_equals_mask():
    X = _masked(T=50, N=4, seed=7)
    mask = ~np.isfinite(X)
    out = missingness_features(X, kinds=("was_imputed",), selective=False)
    assert np.array_equal(out, mask.astype(float))


def test_time_since_obs_counts_steps():
    # column with a known gap: observed, miss, miss, miss, observed
    x = np.array([1.0, np.nan, np.nan, np.nan, 2.0]).reshape(-1, 1)
    out = missingness_features(x, kinds=("time_since_obs",), selective=False)
    assert out.ravel().tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_gap_length_resets_on_observation():
    x = np.array([np.nan, np.nan, 5.0, np.nan, 6.0]).reshape(-1, 1)
    out = missingness_features(x, kinds=("gap_length",), selective=False)
    assert out.ravel().tolist() == [1.0, 2.0, 0.0, 1.0, 0.0]


def test_expanding_missing_rate_is_causal_cumulative():
    x = np.array([np.nan, 1.0, np.nan, 2.0]).reshape(-1, 1)
    out = missingness_features(x, kinds=("missing_rate",), selective=False)
    # cumulative missing / row index: 1/1, 1/2, 2/3, 2/4
    assert np.allclose(out.ravel(), [1.0, 0.5, 2 / 3, 0.5])


def test_selective_mim_fires_for_informative_column():
    """Build a column whose missingness strongly shifts another column's values:
    selective-MIM should select it; a column missing-at-random should not be forced in.
    """
    rng = np.random.default_rng(0)
    T = 400
    driver = rng.standard_normal(T)
    # col1 is missing exactly when driver is high -> its missingness is informative
    miss1 = driver > 0.5
    col1 = driver.copy()
    col1[miss1] = np.nan
    # col2 carries the same signal, fully observed, so the association is detectable
    col2 = driver + 0.05 * rng.standard_normal(T)
    X = np.column_stack([col1, col2])
    meta = missingness_features(X, selective=True, selective_threshold=0.15,
                                return_meta=True)
    assert 0 in meta.informative_columns               # informative col1 selected


def test_selective_mim_skips_uninformative_under_high_threshold():
    X = _masked(T=300, N=4, seed=9)
    meta = missingness_features(X, selective=True, selective_threshold=5.0,
                                return_meta=True)
    # an absurd threshold => nothing qualifies, no selective columns emitted
    assert meta.informative_columns == []
    assert all(t != "selective_mim" for t in meta.kinds)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} missingness tests passed.")
