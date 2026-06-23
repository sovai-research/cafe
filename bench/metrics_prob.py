"""Probabilistic + point metrics for imputation evaluation.

numpy/scipy-only, NaN-safe, mask-aware. Every metric scores ONLY the held-out
cells: pass a boolean ``mask`` (True where a value was held out and must be
scored), or rely on the automatic mask derived from finite truth values. Cells
that are NaN/Inf in the truth, prediction, or (for probabilistic metrics) the
spread are dropped pairwise so a single bad cell never poisons a whole run.

All functions are vectorized pure functions over arrays of arbitrary shape
(time x channels, panels, flat vectors -- shape is irrelevant, scoring is over
the flattened set of valid+selected cells).

Conventions
-----------
y        : ground-truth values at the held-out cells (true values).
mu       : point / mean prediction.
sigma    : predictive standard deviation (>0).
lo, hi   : lower / upper predictive-interval edges.
samples  : ensemble of predictive draws, shape (n_samples, *y.shape).
mask     : optional bool array broadcastable to ``y``; True = score this cell.

This module trains nothing and reads no files; it is a leaf utility safe to
import from any experiment script. It contains no look-ahead: it only compares
predictions to truth at already-chosen cells.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "crps_gaussian",
    "crps_ensemble",
    "picp",
    "coverage",
    "sharpness",
    "pit_values",
    "reliability_curve",
    "mae",
    "rmse",
    "mre",
]


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _as1d(*arrs):
    """Flatten inputs to 1-D float arrays, broadcasting to a common shape."""
    arrs = [np.asarray(a, dtype=float) for a in arrs]
    shape = np.broadcast_shapes(*[a.shape for a in arrs])
    return [np.broadcast_to(a, shape).reshape(-1) for a in arrs]


def _select(mask, *arrs):
    """Flatten ``arrs``, build the valid mask, and return the selected slices.

    The mask is broadcast against the (post-broadcast) flattened column shape so
    callers may pass a mask shaped like the original ``y`` array.
    """
    flat = _as1d(*arrs)
    n = flat[0].size
    keep = np.ones(n, dtype=bool)
    for c in flat:
        keep &= np.isfinite(c)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        # broadcast user mask to the original (un-flattened) broadcast shape,
        # then flatten -- matches how _as1d flattened the value columns.
        bshape = np.broadcast_shapes(*[np.asarray(a).shape for a in arrs])
        m = np.broadcast_to(m, bshape).reshape(-1)
        keep &= m
    return [c[keep] for c in flat]


# --------------------------------------------------------------------------- #
# point metrics
# --------------------------------------------------------------------------- #
def mae(y, mu, mask=None):
    """Mean absolute error over valid, selected cells. NaN if none."""
    yy, mm = _select(mask, y, mu)
    if yy.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yy - mm)))


def rmse(y, mu, mask=None):
    """Root mean squared error over valid, selected cells. NaN if none."""
    yy, mm = _select(mask, y, mu)
    if yy.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yy - mm) ** 2)))


def mre(y, mu, mask=None, eps=1e-12):
    """Mean relative error = sum|y-mu| / sum|y| over valid, selected cells.

    Uses the aggregate (sum/sum) form common in imputation papers, which is
    robust to individual near-zero truths. NaN if no valid cells or |y| sums
    to ~0.
    """
    yy, mm = _select(mask, y, mu)
    if yy.size == 0:
        return float("nan")
    denom = float(np.sum(np.abs(yy)))
    if denom < eps:
        return float("nan")
    return float(np.sum(np.abs(yy - mm)) / denom)


# --------------------------------------------------------------------------- #
# probabilistic metrics
# --------------------------------------------------------------------------- #
def crps_gaussian(y, mu, sigma, mask=None, eps=1e-12):
    """Mean CRPS for a Gaussian predictive distribution N(mu, sigma^2).

    Closed form (Gneiting & Raftery 2007):
        CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z=(y-mu)/sigma
    Lower is better; CRPS >= 0 always. Degenerate cells (sigma<=0) collapse to
    |y-mu| (the CRPS limit as sigma->0). Returns mean over valid cells; NaN if
    none.
    """
    yy, mm, ss = _select(mask, y, mu, sigma)
    if yy.size == 0:
        return float("nan")
    ss = np.maximum(ss, 0.0)
    out = np.empty_like(yy)
    deg = ss <= eps
    # degenerate (point mass): CRPS == absolute error
    out[deg] = np.abs(yy[deg] - mm[deg])
    nd = ~deg
    if np.any(nd):
        z = (yy[nd] - mm[nd]) / ss[nd]
        out[nd] = ss[nd] * (
            z * (2.0 * norm.cdf(z) - 1.0)
            + 2.0 * norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
    # numerical floor: CRPS is non-negative by construction
    return float(np.mean(np.maximum(out, 0.0)))


def crps_ensemble(y, samples, mask=None):
    """Mean CRPS from an empirical ensemble of predictive draws.

    Uses the energy / probability-weighted-moment form
        CRPS = E|X - y| - 0.5 * E|X - X'|,
    computed exactly from the sorted sample set in O(m log m) per cell:
        E|X-X'| = (2/m^2) * sum_i (2i - m + 1) * x_(i)   (1-indexed order stats).

    Parameters
    ----------
    y : array, shape S (the truth, any shape).
    samples : array, shape (m, *S) -- m predictive draws per cell.

    NaN draws are ignored per cell; cells with no finite draw or NaN truth are
    dropped. Returns mean over valid cells; CRPS >= 0. NaN if none valid.
    """
    y = np.asarray(y, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == y.ndim:  # single sample given -> degenerate to |y-x|
        samples = samples[None, ...]
    if samples.shape[1:] != y.shape:
        raise ValueError(
            f"samples.shape[1:] {samples.shape[1:]} must equal y.shape {y.shape}"
        )
    m = samples.shape[0]
    yf = y.reshape(-1)                       # (N,)
    sf = samples.reshape(m, -1)              # (m, N)

    cell_ok = np.isfinite(yf)
    if mask is not None:
        mk = np.broadcast_to(np.asarray(mask, dtype=bool), y.shape).reshape(-1)
        cell_ok &= mk
    if not np.any(cell_ok):
        return float("nan")

    out = np.full(yf.shape, np.nan)
    for j in np.flatnonzero(cell_ok):
        col = sf[:, j]
        col = col[np.isfinite(col)]
        k = col.size
        if k == 0:
            continue
        col.sort()
        term1 = np.mean(np.abs(col - yf[j]))            # E|X - y|
        if k == 1:
            term2 = 0.0
        else:
            i = np.arange(1, k + 1)                      # 1-indexed
            term2 = (2.0 / (k * k)) * np.sum((2 * i - k - 1) * col)
        out[j] = max(term1 - 0.5 * term2, 0.0)

    vals = out[np.isfinite(out)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def picp(y, lo, hi, mask=None):
    """Prediction Interval Coverage Probability: fraction of truths in [lo, hi].

    Returns a value in [0, 1] (e.g. should be ~0.9 for a well-calibrated 90%
    interval). NaN if no valid cells. Edges are inclusive.
    """
    yy, ll, hh = _select(mask, y, lo, hi)
    if yy.size == 0:
        return float("nan")
    inside = (yy >= np.minimum(ll, hh)) & (yy <= np.maximum(ll, hh))
    return float(np.mean(inside))


# alias: "coverage" is the common name for the same quantity
def coverage(y, lo, hi, mask=None):
    """Alias of :func:`picp` (empirical interval coverage), in [0, 1]."""
    return picp(y, lo, hi, mask=mask)


def sharpness(lo, hi, mask=None):
    """Mean predictive-interval width = mean(|hi - lo|) over valid cells.

    Smaller is sharper. Mask-aware; NaN if no valid cells. Independent of the
    truth (sharpness measures only the intervals).
    """
    ll, hh = _select(mask, lo, hi)
    if ll.size == 0:
        return float("nan")
    return float(np.mean(np.abs(hh - ll)))


def pit_values(y, mu, sigma, mask=None, eps=1e-12):
    """Probability Integral Transform values for a Gaussian forecast.

    PIT_i = Phi((y_i - mu_i)/sigma_i). For a well-specified predictive
    distribution the PIT values are Uniform(0,1). Returns the 1-D array of PIT
    values over valid cells (degenerate sigma<=0 cells are dropped, since their
    PIT is a step and uninformative for calibration).
    """
    yy, mm, ss = _select(mask, y, mu, sigma)
    keep = ss > eps
    yy, mm, ss = yy[keep], mm[keep], ss[keep]
    if yy.size == 0:
        return np.empty(0, dtype=float)
    return norm.cdf((yy - mm) / ss)


def reliability_curve(y, mu, sigma, n_bins=10, mask=None):
    """Reliability (calibration) curve from PIT values.

    Bins the PIT values into ``n_bins`` equal-width bins on [0,1] and returns
    (bin_centers, empirical_cdf) where empirical_cdf[k] is the fraction of PIT
    values <= the right edge of bin k. For a calibrated model this traces the
    diagonal y = x. Useful to plot against the ideal line.

    Returns
    -------
    centers : (n_bins,) bin centers (the nominal cumulative levels = right
              edges, i.e. (k+1)/n_bins) -- here we report the right edge as the
              nominal level so it can be plotted directly against observed.
    observed : (n_bins,) empirical fraction of PIT <= that nominal level.
    """
    pit = pit_values(y, mu, sigma, mask=mask)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    nominal = edges[1:]                       # right edges = nominal levels
    if pit.size == 0:
        return nominal, np.full(n_bins, np.nan)
    observed = np.array([np.mean(pit <= lvl) for lvl in nominal], dtype=float)
    return nominal, observed


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    n = 40000

    # ---- well-specified Gaussian: truth ~ N(mu, sigma^2) ------------------ #
    mu = rng.normal(size=n)
    sigma = rng.uniform(0.5, 2.0, size=n)
    y = rng.normal(loc=mu, scale=sigma)

    # sprinkle NaNs into truth/pred to exercise NaN-safety
    y_nan = y.copy()
    y_nan[::997] = np.nan
    mu_nan = mu.copy()
    mu_nan[::1501] = np.inf

    # ---- point metrics ---------------------------------------------------- #
    e_mae = mae(y_nan, mu_nan)
    e_rmse = rmse(y_nan, mu_nan)
    e_mre = mre(y_nan, mu_nan)
    assert np.isfinite(e_mae) and e_mae >= 0
    assert np.isfinite(e_rmse) and e_rmse >= e_mae - 1e-9  # rmse >= mae
    assert np.isfinite(e_mre) and e_mre >= 0
    # perfect prediction -> zero error
    assert mae(y, y) == 0.0 and rmse(y, y) == 0.0

    # ---- CRPS (Gaussian) -------------------------------------------------- #
    c_g = crps_gaussian(y, mu, sigma)
    assert np.isfinite(c_g) and c_g >= 0.0, "CRPS must be non-negative"
    # sharper (correct) sigma should beat a hugely inflated sigma on average
    c_wide = crps_gaussian(y, mu, sigma * 10.0)
    assert c_g < c_wide, "well-specified CRPS should beat over-dispersed"
    # closed-form sanity: y==mu, sigma=1 -> CRPS = 1/sqrt(pi) (2/sqrt(pi)-1/sqrt(pi))
    expect = 2.0 * norm.pdf(0.0) - 1.0 / np.sqrt(np.pi)
    got = crps_gaussian(np.zeros(1), np.zeros(1), np.ones(1))
    assert abs(got - expect) < 1e-9, (got, expect)

    # ---- CRPS (ensemble) converges to Gaussian CRPS ----------------------- #
    m = 600
    samples = rng.normal(loc=mu[None, :], scale=sigma[None, :], size=(m, n))
    samples[0, ::3000] = np.nan  # exercise NaN draws
    c_e = crps_ensemble(y, samples)
    assert np.isfinite(c_e) and c_e >= 0.0
    # ensemble CRPS should be close to the closed-form Gaussian CRPS
    rel = abs(c_e - c_g) / c_g
    assert rel < 0.02, f"ensemble vs gaussian CRPS rel err {rel:.4f} too large"

    # ---- coverage / sharpness -------------------------------------------- #
    z90 = norm.ppf(0.95)            # central 90% interval
    lo, hi = mu - z90 * sigma, mu + z90 * sigma
    cov = coverage(y, lo, hi)
    assert 0.0 <= cov <= 1.0, "coverage must be a probability"
    assert abs(cov - 0.90) < 0.02, f"90% interval coverage {cov:.3f} off"
    assert picp(y, lo, hi) == cov  # picp/coverage identical
    shp = sharpness(lo, hi)
    assert np.isfinite(shp) and shp > 0
    # wider intervals -> larger sharpness (worse), coverage higher
    assert sharpness(mu - 2 * z90 * sigma, mu + 2 * z90 * sigma) > shp

    # ---- PIT ~ Uniform(0,1) for well-specified Gaussian ------------------- #
    pit = pit_values(y, mu, sigma)
    assert pit.size > 0
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    assert abs(pit.mean() - 0.5) < 0.01, f"PIT mean {pit.mean():.3f} not ~0.5"
    assert abs(pit.std() - (1.0 / np.sqrt(12.0))) < 0.01, "PIT std not ~uniform"
    # KS-ish check: max deviation of empirical CDF from the diagonal is small
    nominal, observed = reliability_curve(y, mu, sigma, n_bins=20)
    assert np.max(np.abs(observed - nominal)) < 0.02, "reliability off diagonal"

    # ---- mask-awareness: scoring only held-out cells ---------------------- #
    held = rng.random(n) < 0.3
    # corrupt the NON-held cells; metrics must ignore them entirely
    mu_bad = mu.copy()
    mu_bad[~held] += 1000.0
    assert abs(mae(y, mu_bad, mask=held) - mae(y[held], mu[held])) < 1e-9
    assert abs(crps_gaussian(y, mu_bad, sigma, mask=held)
               - crps_gaussian(y[held], mu[held], sigma[held])) < 1e-9

    # ---- empty / all-invalid inputs return NaN, not crash ----------------- #
    assert np.isnan(mae(np.array([np.nan]), np.array([np.nan])))
    assert np.isnan(crps_gaussian(np.array([np.nan]), np.array([0.0]),
                                  np.array([1.0])))
    assert pit_values(np.array([np.nan]), np.array([0.0]),
                      np.array([1.0])).size == 0

    print("metrics_prob self-test: ALL PASS")
    print(f"  MAE={e_mae:.4f}  RMSE={e_rmse:.4f}  MRE={e_mre:.4f}")
    print(f"  CRPS gaussian={c_g:.4f}  ensemble={c_e:.4f}  (rel {rel:.4f})")
    print(f"  coverage(90%)={cov:.4f}  sharpness={shp:.4f}")
    print(f"  PIT mean={pit.mean():.4f} std={pit.std():.4f}")


if __name__ == "__main__":
    _selftest()
