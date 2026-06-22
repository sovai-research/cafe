"""Property-based tests of CAFE's core guarantees (Hypothesis).

These encode the invariants that *define* the library and that we otherwise
re-verify by hand every session:

  1. No look-ahead (truncation invariance): imputing a prefix X[:t] gives the
     exact same result as imputing all of X and slicing to [:t]. This is the
     single property the whole point-in-time design exists to provide, fuzzed
     over many shapes, missing rates and truncation points.
  2. Observed values are preserved exactly.
  3. Output is always complete (no NaN/Inf) and shape-preserving.

Correctness-by-construction only counts if it is continuously tested.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import cafe  # noqa: E402


def _make(T, N, rate, seed):
    """A low-rank+noise (T, N) matrix with `rate` MCAR missingness, guaranteeing
    at least one observed entry per column (an all-missing column is degenerate
    and out of scope for the invariants under test)."""
    rng = np.random.default_rng(seed)
    R = max(1, min(N, 3))
    Z = rng.standard_normal((T, R))
    W = rng.standard_normal((R, N))
    X = Z @ W + 0.2 * rng.standard_normal((T, N))
    M = rng.random((T, N)) < rate
    for j in range(N):
        if M[:, j].all():
            M[rng.integers(T), j] = False
    X[M] = np.nan
    return X


@given(
    T=st.integers(20, 150),
    N=st.integers(1, 8),
    rate=st.floats(0.0, 0.6),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=60, deadline=None)
def test_output_finite_and_shape_preserving(T, N, rate, seed):
    X = _make(T, N, rate, seed)
    out = np.asarray(cafe.impute(X), dtype=float)
    assert out.shape == X.shape
    assert np.isfinite(out).all(), "imputation left NaN/Inf cells"


@given(
    T=st.integers(20, 150),
    N=st.integers(1, 8),
    rate=st.floats(0.0, 0.6),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=60, deadline=None)
def test_observed_values_preserved(T, N, rate, seed):
    X = _make(T, N, rate, seed)
    out = np.asarray(cafe.impute(X), dtype=float)
    obs = ~np.isnan(X)
    assert np.array_equal(out[obs], X[obs]), "an observed value was altered"


@given(
    T=st.integers(40, 150),
    N=st.integers(1, 6),
    rate=st.floats(0.0, 0.5),
    seed=st.integers(0, 10_000),
    frac=st.floats(0.3, 0.85),
)
@settings(max_examples=80, deadline=None)
def test_no_lookahead_truncation_invariance(T, N, rate, seed, frac):
    """The crown-jewel invariant: a past imputation cannot change when future
    rows are appended. Asserted on the FULL prefix (observed + imputed)."""
    X = _make(T, N, rate, seed)
    t = max(5, int(frac * T))
    prefix = np.asarray(cafe.impute(X[:t]), dtype=float)
    full = np.asarray(cafe.impute(X), dtype=float)[:t]
    assert np.allclose(prefix, full, atol=1e-8, rtol=0.0), (
        "look-ahead detected: prefix imputation differs from full-then-sliced"
    )


@given(
    T=st.integers(20, 120),
    rate=st.floats(0.0, 0.5),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=40, deadline=None)
def test_1d_series_same_invariants(T, rate, seed):
    x = _make(T, 1, rate, seed)[:, 0]
    out = np.asarray(cafe.impute(x), dtype=float)
    assert out.ndim == 1 and out.shape == x.shape
    assert np.isfinite(out).all()
    obs = ~np.isnan(x)
    assert np.array_equal(out[obs], x[obs])
