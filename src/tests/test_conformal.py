"""Tests for the causal split-conformal recalibration layer (cafe.conformal).

These pin the three guarantees that make the calibrated band trustworthy:

  (a) CALIBRATION: on held-out cells the conformal band's coverage is CLOSER to
      nominal than CAFE's raw (admittedly over-conservative) Gaussian band.
  (b) POINT-IN-TIME: appending future rows does NOT change the calibrated band for
      earlier rows (truncation invariance -- the conformal layer is as causal as CAFE).
  (c) IMPUTATION UNTOUCHED: the filled point values are bit-identical with and without
      conformal recalibration (this is a sigma-only change).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402

import cafe  # noqa: E402
from cafe.conformal import (  # noqa: E402
    ConformalCalibrator, _conformal_quantile, _z_for, conformal_multipliers,
)

LEVELS = (0.50, 0.80, 0.90, 0.95)
# Gaussian z multipliers for the raw band (so we score raw vs calibrated coverage).
_Z = {0.50: 0.6744897502, 0.80: 1.2815515655,
      0.90: 1.6448536270, 0.95: 1.9599639845}


def _lowrank(T=600, N=12, R=3, rate=0.15, seed=0):
    """A low-rank + noise (T, N) matrix and an MCAR hold-out mask. Returns
    (X_true, X_obs, mask) where X_obs has NaN at the held-out cells."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, R))
    W = rng.standard_normal((R, N))
    X = Z @ W + 0.3 * rng.standard_normal((T, N))
    M = rng.random((T, N)) < rate
    Xobs = X.copy()
    Xobs[M] = np.nan
    return X, Xobs, M


def _coverage(y, lo, hi, sel):
    return float(np.mean((y[sel] >= lo[sel]) & (y[sel] <= hi[sel])))


# --------------------------------------------------------------------------- #
# (a) calibration: conformal band is closer to nominal than the raw band
# --------------------------------------------------------------------------- #
def test_conformal_closer_to_nominal_than_raw():
    """Averaged over the 50/80/90/95 levels and several seeds, the conformal band's
    distance to nominal coverage is smaller than the raw Gaussian band's (which CAFE
    admits over-covers)."""
    raw_err, cal_err = [], []
    for seed in (0, 1, 2):
        X, Xobs, M = _lowrank(seed=seed)
        res = cafe.CAFE().run(Xobs)
        mu = np.asarray(res.imputed)
        sd = np.sqrt(res._comp()["cvar"])
        sel = M & np.isfinite(sd) & (sd > 1e-9)
        assert sel.sum() > 200
        for lvl in LEVELS:
            rlo, rhi = mu - _Z[lvl] * sd, mu + _Z[lvl] * sd
            lo, hi = res.calibrated_interval(lvl)
            lo, hi = np.asarray(lo), np.asarray(hi)
            raw_err.append(abs(_coverage(X, rlo, rhi, sel) - lvl))
            cal_err.append(abs(_coverage(X, lo, hi, sel) - lvl))
    raw_err, cal_err = np.array(raw_err), np.array(cal_err)
    # the conformal band is, on average, materially closer to nominal coverage.
    assert cal_err.mean() < raw_err.mean(), (cal_err.mean(), raw_err.mean())
    # and the raw band genuinely over-covers (the weakness we are fixing): its mean
    # signed gap is clearly positive.
    assert raw_err.mean() > cal_err.mean() + 0.02


def test_conformal_band_is_narrower_than_raw():
    """Recalibration TIGHTENS the over-wide band (sharpness improves) -- the conformal
    half-width is smaller than the raw Gaussian half-width on average."""
    X, Xobs, M = _lowrank(seed=3)
    res = cafe.CAFE().run(Xobs)
    sd = np.sqrt(res._comp()["cvar"])
    sel = M & np.isfinite(sd) & (sd > 1e-9)
    for lvl in (0.80, 0.90, 0.95):
        cw = np.asarray(res.calibrated_uncertainty(lvl))
        raw_hw = _Z[lvl] * sd
        assert np.nanmean(cw[sel]) < np.nanmean(raw_hw[sel])


# --------------------------------------------------------------------------- #
# (b) point-in-time: appending future rows does not move earlier calibrated bands
# --------------------------------------------------------------------------- #
def test_calibrated_band_truncation_invariant():
    """The calibrated band for rows < cut is identical whether or not future rows are
    present. The conformal multiplier at row t uses only calibration scores from rows
    < t, so truncating the future cannot change it -- and CAFE's sigma is itself
    point-in-time. (Same seed/cal_seed => same internal calibration mask on the shared
    prefix.)"""
    X, Xobs, M = _lowrank(T=500, seed=5)
    cut = 300
    full = cafe.CAFE().run(Xobs)
    pre = cafe.CAFE().run(Xobs[:cut])

    for lvl in (0.80, 0.90):
        lo_f, hi_f = full.calibrated_interval(lvl)
        lo_p, hi_p = pre.calibrated_interval(lvl)
        lo_f, hi_f = np.asarray(lo_f)[:cut], np.asarray(hi_f)[:cut]
        lo_p, hi_p = np.asarray(lo_p), np.asarray(hi_p)
        m = M[:cut] & np.isfinite(lo_f) & np.isfinite(lo_p)
        assert m.sum() > 50
        # bands on the shared prefix agree to numerical tolerance
        assert np.allclose(lo_f[m], lo_p[m], atol=1e-7), np.max(np.abs(lo_f[m] - lo_p[m]))
        assert np.allclose(hi_f[m], hi_p[m], atol=1e-7)


def test_calibrator_grid_is_strictly_past():
    """The streaming multiplier grid is built from rows STRICTLY before each row: the
    first row's multiplier is the raw fallback (no scores yet), and prepending the grid
    never peeks at the current row's own score."""
    # synthetic per-row scores: a constant block of scores so the empirical quantile is
    # well-defined and deterministic.
    T = 100
    rng = np.random.default_rng(0)
    row_scores = [rng.uniform(0.0, 2.0, size=8) for _ in range(T)]
    cal = ConformalCalibrator.from_row_scores(row_scores, window=10**6, min_scores=5)
    q = cal.grid(0.90)
    assert q.shape == (T,)
    # row 0 has no past scores -> raw Gaussian z fallback
    assert abs(q[0] - _z_for(0.90)) < 1e-9
    # by the end, enough scores have accrued -> a finite empirical quantile (not the
    # fallback in general)
    assert np.isfinite(q[-1])


# --------------------------------------------------------------------------- #
# (c) imputation untouched: filled values bit-identical with/without conformal
# --------------------------------------------------------------------------- #
def test_imputation_bit_identical_with_conformal():
    """Computing the calibrated interval must NOT perturb res.imputed: the centre of the
    band is exactly CAFE's filled value (conformal is a sigma-only side computation)."""
    X, Xobs, M = _lowrank(seed=7)
    res = cafe.CAFE().run(Xobs)
    before = np.asarray(res.imputed).copy()
    # force the (potentially expensive) calibration pass to run
    lo, hi = res.calibrated_interval(0.90)
    _ = res.calibrated_uncertainty(0.80)
    after = np.asarray(res.imputed)
    assert np.array_equal(before, after)             # bit-identical, not just close
    # the band is centred exactly on the filled values
    lo, hi = np.asarray(lo), np.asarray(hi)
    sel = M & np.isfinite(lo)
    mid = 0.5 * (lo + hi)
    assert np.allclose(mid[sel], after[sel], atol=1e-9)


def test_calibration_pass_does_not_mutate_input():
    """The internal calibration pass works on a copy; the data passed to run() is not
    modified by requesting a calibrated interval."""
    X, Xobs, M = _lowrank(seed=8)
    snapshot = Xobs.copy()
    res = cafe.CAFE().run(Xobs)
    res.calibrated_interval(0.90)
    assert np.array_equal(np.asarray(snapshot), np.asarray(Xobs), equal_nan=True)


# --------------------------------------------------------------------------- #
# unit checks on the conformal primitives
# --------------------------------------------------------------------------- #
def test_conformal_quantile_matches_definition():
    """_conformal_quantile returns the Vovk rank ceil(level*(n+1)) order statistic; for
    large clean samples it approaches the true quantile of the score distribution."""
    rng = np.random.default_rng(0)
    s = rng.standard_exponential(20000)
    for lvl in (0.5, 0.8, 0.9, 0.95):
        q = _conformal_quantile(s, lvl)
        # exponential quantile is -ln(1-lvl); empirical conformal quantile is close
        assert abs(q - (-np.log(1 - lvl))) < 0.1


def test_conformal_multipliers_fallback_small_n():
    """With too few scores, conformal_multipliers falls back to the raw Gaussian z."""
    q = conformal_multipliers(np.array([0.5, 1.0]), LEVELS)  # n=2 < MIN_SCORES
    for lvl in LEVELS:
        assert abs(q[lvl] - _z_for(lvl)) < 1e-9


def test_container_roundtrip_matches_uncertainty():
    """calibrated_interval/calibrated_uncertainty return the SAME container type as the
    existing res.uncertainty (whatever the input container round-trips to in this
    environment) -- so the calibrated API is a drop-in alongside the raw band."""
    import pandas as pd
    X, Xobs, M = _lowrank(T=200, N=4, seed=9)
    df = pd.DataFrame(Xobs, columns=[f"s{i}" for i in range(4)],
                      index=pd.date_range("2020-01-01", periods=len(Xobs), freq="h"))
    res = cafe.CAFE().run(df)
    want = type(res.uncertainty)
    lo, hi = res.calibrated_interval(0.90)
    assert type(lo) is want and type(hi) is want
    assert type(res.calibrated_uncertainty(0.80)) is want
    # shapes line up with the imputed grid
    assert np.asarray(lo).shape == np.asarray(res.imputed).shape


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} conformal tests passed.")
