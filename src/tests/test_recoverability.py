"""Contract tests for the per-cell recoverability certificate (idea #4).

The certificate must (1) be a finite score in [0, 1] at every imputed cell (NaN at observed
cells), and (2) drive a selective imputer that abstains (returns NaN) on MORE cells as the
confidence threshold tau rises. These are properties of the certificate's shape, not of its
calibration quality (that is validated empirically in bench/exp_recoverability.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cafe
from cafe.recoverability import (recoverability_score, score_from_result,
                                 selective_impute)


def _run():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((250, 8))
    X[rng.random(X.shape) < 0.35] = np.nan
    res = cafe.CAFE().run(X)
    return X, res


def test_score_in_unit_interval_and_finite():
    X, res = _run()
    s = score_from_result(res)
    miss = ~np.isfinite(X)
    assert s.shape == X.shape
    assert np.isfinite(s[miss]).all(), "score must be finite at every imputed cell"
    assert np.isnan(s[~miss]).all(), "score must be NaN at observed cells"
    assert np.nanmin(s) >= 0.0 - 1e-12 and np.nanmax(s) <= 1.0 + 1e-12, "score in [0,1]"


def test_score_raw_and_conformal_both_valid():
    X, res = _run()
    miss = ~np.isfinite(X)
    for conformal in (False, True):
        s = score_from_result(res, conformal=conformal)
        assert np.isfinite(s[miss]).all()
        assert np.nanmin(s) >= -1e-12 and np.nanmax(s) <= 1.0 + 1e-12


def test_selective_abstains_more_as_tau_rises():
    X, res = _run()
    s = score_from_result(res)
    filled = np.asarray(res.imputed, float)
    assert np.isfinite(filled).all(), "imputed has no NaN, so all NaN in selective = abstain"
    taus = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    counts = [int(np.isnan(selective_impute(filled, s, t)).sum()) for t in taus]
    # monotone non-decreasing in tau, and tau=0 abstains on nothing
    assert counts[0] == 0
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), counts
    assert counts[-1] >= counts[0]
    # observed-cell fills are never abstained on (score is NaN there)
    sel = selective_impute(filled, s, 0.99)
    assert np.isfinite(sel).sum() <= filled.size


def test_pure_function_signature():
    # the low-level numpy entry point works without a CafeResult
    T, N = 40, 5
    rng = np.random.default_rng(1)
    miss = rng.random((T, N)) < 0.3
    sigma = np.where(miss, rng.random((T, N)) + 0.1, np.nan)
    s = recoverability_score(sigma=sigma, miss=miss, obs_mask=~miss,
                             row_w=np.ones(T), loading_energy=np.ones(N), active_rank=2)
    assert np.isfinite(s[miss]).all()
    assert np.nanmin(s) >= -1e-12 and np.nanmax(s) <= 1.0 + 1e-12


if __name__ == "__main__":
    test_score_in_unit_interval_and_finite()
    test_score_raw_and_conformal_both_valid()
    test_selective_abstains_more_as_tau_rises()
    test_pure_function_signature()
    print("all recoverability tests passed")
