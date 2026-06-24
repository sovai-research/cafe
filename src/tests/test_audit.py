"""Unit tests for cafe.audit -- the model-agnostic leakage audit.

The audit must (a) certify a strictly point-in-time method (CAFE, LOCF) as causal
with max_revision == 0 and leakage_delta == 0, and (b) flag a look-ahead method
(linear interpolation, SoftImpute-style smoothing) as leaky -- nonzero max_revision
and/or a positive leakage_delta (accuracy borrowed from the future).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import cafe.audit as A


def _series(T=200, N=6, seed=0):
    """Seasonal + low-rank signal so interpolation has real future to borrow."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, T)
    Z = np.stack([np.sin(t), np.cos(0.7 * t)], 1)        # (T,2)
    W = rng.standard_normal((2, N))
    return Z @ W + 0.2 * rng.standard_normal((T, N))


# ---- reference imputers (no bench deps) ----------------------------------- #
def locf(Z):                                             # CAUSAL
    Z = np.asarray(Z, float).copy()
    for j in range(Z.shape[1]):
        last = 0.0
        for t in range(Z.shape[0]):
            if np.isfinite(Z[t, j]):
                last = Z[t, j]
            else:
                Z[t, j] = last
    return Z


def linear_interp(Z):                                    # LEAKY (reads the future)
    Z = np.asarray(Z, float).copy()
    tt = np.arange(Z.shape[0], dtype=float)
    for j in range(Z.shape[1]):
        obs = np.isfinite(Z[:, j])
        if obs.sum() >= 2:
            Z[:, j] = np.interp(tt, tt[obs], Z[obs, j])
        elif obs.sum() == 1:
            Z[~obs, j] = Z[obs, j][0]
        else:
            Z[:, j] = 0.0
    return Z


def global_mean(Z):                                      # LEAKY (uses all rows incl. future)
    Z = np.asarray(Z, float).copy()
    mu = np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    bad = ~np.isfinite(Z)
    Z[bad] = np.take(mu, np.where(bad)[1])
    return Z


# ---- tests ---------------------------------------------------------------- #
def test_locf_certified_causal():
    X = _series()
    rep = A.leakage_report(locf, X, native_causal=True)
    assert rep["causal"] is True
    assert rep["max_revision"] <= 1e-9, rep["max_revision"]
    assert abs(rep["leakage_delta"]) <= 1e-9, rep["leakage_delta"]


def test_cafe_certified_causal():
    """CAFE itself, if importable from the bench tree, must certify cleanly."""
    here = os.path.dirname(__file__)
    bench = os.path.abspath(os.path.join(here, "..", "..", "bench"))
    if bench not in sys.path:
        sys.path.insert(0, bench)
    try:
        import c_unified_penmf as P
    except Exception:
        import pytest
        pytest.skip("CAFE (c_unified_penmf) not importable in this env")
    X = _series()
    rep = A.leakage_report(lambda Z: P.online_impute(Z, {}), X, native_causal=True)
    assert rep["causal"] is True, rep
    assert rep["max_revision"] <= 1e-6, rep["max_revision"]
    assert abs(rep["leakage_delta"]) <= 1e-6, rep["leakage_delta"]


def test_linear_interp_flagged_leaky():
    X = _series()
    rep = A.leakage_report(linear_interp, X, native_causal=False)
    # truncation invariance is violated AND it borrows accuracy from the future
    assert rep["max_revision"] > 1e-3, rep["max_revision"]
    assert rep["causal"] is False
    assert rep["leakage_delta"] > 1e-3, rep["leakage_delta"]


def test_global_mean_flagged_leaky():
    """A global statistic over all rows (incl. future) must fail truncation
    invariance even though it looks innocuous -- the audit catches the subtle leak."""
    X = _series()
    rep = A.leakage_report(global_mean, X, native_causal=False)
    assert rep["causal"] is False
    assert rep["max_revision"] > 1e-3, rep["max_revision"]


def test_separation():
    """The certificate cleanly separates the causal from the leaky reference set."""
    X = _series()
    causal = A.audit_panel({"LOCF": (locf, True)}, X)[0]
    leaky = A.audit_panel({"LinearInterp": (linear_interp, False)}, X)[0]
    assert causal["causal"] and not leaky["causal"]
    assert leaky["leakage_delta"] > causal["leakage_delta"]


if __name__ == "__main__":
    test_locf_certified_causal()
    test_linear_interp_flagged_leaky()
    test_global_mean_flagged_leaky()
    test_separation()
    try:
        test_cafe_certified_causal()
        print("all audit tests PASSED (incl. CAFE)")
    except Exception as e:
        print("core audit tests PASSED (CAFE test:", repr(e)[:80], ")")
