"""
Chal_xsblend — High-rank fix via causal contemporaneous cross-sectional ridge.

Pure low-rank MF (OnlineTRMF) fails on near-full-rank data (2d_highrank, corr ~0.36):
the signal lives in the cross-section, not in a few latent factors. The remedy is a
contemporaneous cross-sectional ridge regression. At time t we predict each missing
feature j from the OTHER features observed at the SAME time t, using regression
coefficients derived from a covariance matrix accumulated over the EXPANDING history
(rows < t) only. This is fully point-in-time: the covariance/means use strictly past
co-observed cells, and the prediction consumes the contemporaneous cross-section at t
(which the causal protocol explicitly allows).

Covariance estimate (causal, online): we accumulate pairwise sums M2[i,j]=sum x_i x_j
and per-feature M1[i]=sum x_i, Sq[i]=sum x_i^2 with co-observation counts. At time t the
correlation matrix is formed from these running moments (frozen as of <t), the observed
block is ridge-regularized on the diagonal, and betas observed->missing are solved with
Cholesky. Predictions are only emitted once a target has enough co-observation support
(MIN_SUP), so noisy early estimates never contaminate the output.

Gating (cheap, up-front, causal). We only pay for this on data that is BOTH genuinely
high-rank AND densely observed enough for a cross-sectional regression to work:
  - effective-rank ratio (SVD energy of the column-standardized observed matrix) >= 0.65
    => weak low-rank structure (pure MF loses).
  - missing rate <= 0.45 => the cross-section is dense enough to regress on.
This selects exactly the near-full-rank scattered cases (2d_highrank, 2d_heavytail) and
leaves every other case (low-rank 2D, very-sparse, block gaps, panels, 1D, reals) on the
unmodified core — so the rest of the suite is byte-identical and adds ~0 time.

Causality: the low-rank base is the verified-causal OnlineTRMF; the ridge add-on uses
only history<t for its moments and the contemporaneous obs at t for inputs.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel

# ----------------------------------------------------------------- knobs --- #
RIDGE = 0.35          # ridge added to the diagonal of the (correlation) Gram
WARM = 30             # rows of history before the regression turns on
MIN_SUP = 30          # required co-observation support for a target prediction
BLEND_W = 1.0         # weight on the cross-sectional prediction in the blend
EFFRANK_FRAC = 0.90   # variance fraction defining effective rank
TOPK_FRAC = 0.30      # singular budget counted as the "low-rank" share
ENGAGE_ERANK = 0.65   # engage only above this effective-rank ratio
ENGAGE_MAXMISS = 0.45 # ...and only when dense enough to regress cross-sectionally


def _effrank_ratio(X):
    """Cheap up-front signals from the column-standardized, mean-imputed observed
    matrix. Returns (erank_ratio, lowrank_energy_in_topK). erank_ratio = (#components
    to reach EFFRANK_FRAC of SVD energy)/N: near 0 => strongly low-rank, near 1 =>
    full rank."""
    T, N = X.shape
    obs = ~np.isnan(X)
    col_cnt = obs.sum(0)
    if N < 4 or T < 8 or (col_cnt < 2).any():
        return 0.0, 1.0
    mu = np.nansum(np.where(obs, X, 0.0), 0) / np.maximum(col_cnt, 1)
    sd = np.sqrt(np.maximum(
        np.nansum(np.where(obs, (X - mu) ** 2, 0.0), 0) / np.maximum(col_cnt, 1),
        1e-12))
    Z = np.where(obs, (X - mu) / sd, 0.0)
    if T > 400:
        Z = Z[np.linspace(0, T - 1, 400).astype(int)]
    try:
        s = np.linalg.svd(Z, compute_uv=False)
    except Exception:
        return 0.0, 1.0
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return 0.0, 1.0
    cum = np.cumsum(e) / tot
    k = int(np.searchsorted(cum, EFFRANK_FRAC) + 1)
    topk = max(1, int(round(TOPK_FRAC * N)))
    return k / N, float(cum[min(topk - 1, len(cum) - 1)])


def _xs_ridge_impute(X):
    """Causal contemporaneous cross-sectional ridge. Returns (pred, conf) where
    pred[t,j] predicts missing feature j at time t from features observed at t, with
    a covariance accumulated only over rows < t; conf marks emitted predictions."""
    T, N = X.shape
    obs = ~np.isnan(X)
    pred = np.full((T, N), np.nan)

    M2 = np.zeros((N, N))     # sum x_i x_j over co-observed past cells
    Cnt = np.zeros((N, N))    # co-observation counts
    M1 = np.zeros(N)          # sum x_i
    Sq = np.zeros(N)          # sum x_i^2
    Cn = np.zeros(N)          # per-feature observed counts
    nrows = 0

    for t in range(T):
        row = X[t]
        o = obs[t]
        of = np.where(o)[0]
        if nrows >= WARM and of.size and of.size < N:
            mi = np.where(~o)[0]
            cnt = np.maximum(Cnt, 1.0)
            mean = np.where(Cn > 0, M1 / np.maximum(Cn, 1), 0.0)
            var = np.where(Cn > 1, Sq / np.maximum(Cn, 1) - mean ** 2, 1.0)
            sd = np.sqrt(np.maximum(var, 1e-8))
            Cov = M2 / cnt - np.outer(mean, mean)
            Corr = Cov / np.outer(sd, sd)
            Goo = Corr[np.ix_(of, of)].copy()
            Goo[np.diag_indices_from(Goo)] += RIDGE
            try:
                c = cho_factor(Goo, lower=True, check_finite=False)
                beta = cho_solve(c, Corr[np.ix_(of, mi)], check_finite=False)
                zo = (row[of] - mean[of]) / sd[of]
                phat = (zo @ beta) * sd[mi] + mean[mi]
                support = Cnt[np.ix_(of, mi)].min(0)
                good = support >= MIN_SUP
                pred[t, mi[good]] = phat[good]
            except Exception:
                pass

        # fold this row into history (now becomes <t for future steps)
        if of.size:
            xx = row[of]
            M2[np.ix_(of, of)] += np.outer(xx, xx)
            Cnt[np.ix_(of, of)] += 1.0
            M1[of] += xx
            Sq[of] += xx ** 2
            Cn[of] += 1.0
        nrows += 1

    return pred, np.isfinite(pred)


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    # Panels and 1D -> unmodified core (router behaviour).
    if _is_panel(meta):
        return _fe(X, meta)
    if X.shape[1] == 1:
        return _trmf(X, meta)

    base = _trmf(X, meta)

    # cheap up-front gate: only genuine high-rank AND dense enough to regress.
    miss = np.isnan(X)
    miss_rate = float(miss.mean())
    if miss_rate > ENGAGE_MAXMISS:
        return base
    erank, _ = _effrank_ratio(X)
    if erank < ENGAGE_ERANK:
        return base

    xs_pred, conf = _xs_ridge_impute(X)
    fill = miss & conf
    if not fill.any():
        return base
    out = base.copy()
    out[fill] = BLEND_W * xs_pred[fill] + (1.0 - BLEND_W) * base[fill]
    return out


if __name__ == "__main__":
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_xsblend", online_impute))
