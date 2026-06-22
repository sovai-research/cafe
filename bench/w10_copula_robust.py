"""
w10_copula_robust -- a UNIFIED robust narrow-2D core: one intrinsic model that
handles heavy tails, high effective-rank, and benign cross-sections simultaneously,
WITHOUT any value-based gate. It is meant to do the work of two separate gates in
c_chal_uni (winsor-for-heavytail + xsblend-ridge-for-highrank) in a single core.

IDEA -- a causal *Gaussian-copula* conditional mean with a *robust* EW correlation:

  (1) GAUSSIAN-RANK TRANSFORM (per column, expanding/causal). For each column j and
      time t with an observed value x, we map it to a latent z via the running
      empirical CDF over that column's OWN observed history at times <= t:
          z = Phi^-1( (rank_<=t(x)) / (n_obs_<=t + 1) ).
      A monotone copula transform neutralises heavy tails, skew and arbitrary
      marginals: a single huge outlier becomes at most ~Phi^-1(n/(n+1)) in z-space,
      so it CANNOT dominate the correlation. This replaces the heavy-tail winsor gate
      with an intrinsic, always-on transform that is harmless on benign data (a
      monotone reparametrisation of an already-Gaussian column is ~identity in rank).

  (2) ROBUST EW CORRELATION in latent space. We accumulate an exponentially-weighted
      correlation of the latent rows, with each row additionally downweighted by a
      Student-t / Mahalanobis factor  u = (nu + p) / (nu + d2)  (d2 = Mahalanobis
      distance of the row's observed latents under the current correlation). Rows that
      look like outliers even AFTER the rank transform get less weight. Unit diagonal
      is enforced. The FULL pxp correlation (not a rank-L factor) captures high
      effective-rank cross-sections, fixing the high-rank case without a separate gate.

  (3) CONDITIONAL MEAN + BACK-TRANSFORM. Missing latents are filled by the Gaussian
      conditional mean  z_m = Corr_mo Corr_oo^{-1} z_o, then mapped back to data space
      by inverting the column's running empirical CDF (interpolating the sorted
      observed history at times <= t). Everything below is strictly point-in-time.

CAUSALITY. We iterate strictly forward. Row t is imputed using ONLY: that column's
observed history at times < t (for the rank transform and inverse), the EW correlation
built from rows < t, and the contemporaneous observed cross-section at t (allowed). All
statistics are folded in AFTER imputing row t. Truncating future rows cannot change any
past imputation -> passes the look-ahead verifier with 0 leaks.

ROUTING (truncation-invariant structure only -> route identical on every prefix):
  panel                       -> CausalFE        (unchanged)
  1D (N==1)                   -> OnlineTRMF       (no cross-section to copula)
  wide 2D (N >= N_GATE)       -> EW-cov           (unchanged fast wide core)
  narrow 2D (N < N_GATE)      -> THIS copula-robust core
No value-based gate is used for the narrow-2D decision, so nothing can flip under
truncation.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtri  # inverse standard-normal CDF (Phi^-1), vectorised

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel
from c_chal_ewcov import _ewcov_2d, N_GATE
from c_chal_seasonal import _impute_1d

# ------------------------------------------------------------------- knobs --- #
HALFLIFE = 200.0   # EW forgetting half-life for the latent correlation (timesteps)
RIDGE = 5e-2       # ridge on the observed-block correlation (stabilises the solve)
WARM = 12          # min EW weight before trusting the correlation / robust weights
NU = 5.0           # Student-t dof for robust row downweighting (heavier-tail safe)
RANKWARM = 4       # min observed history in a column before rank-transforming it
MIN_COND = 3       # min observed conditioning columns to trust the copula prediction
BLEND_W = 0.6      # fixed weight on the copula cross-sectional prediction in the blend
EPS = 1e-9


def _rank_z(hist_sorted, x, n):
    """Map value x to latent z via the empirical CDF of the column's observed
    history (n past values, given pre-sorted). z = Phi^-1((r + 0.5)/(n + 1)) where
    r = #history strictly below x (mid-rank handling via +0.5). Causal: hist is the
    column's own past observed values."""
    # position of x among the sorted history
    r = float(np.searchsorted(hist_sorted, x, side="left"))
    req = float(np.searchsorted(hist_sorted, x, side="right"))
    r = 0.5 * (r + req)                      # average rank for ties
    p = (r + 0.5) / (n + 1.0)
    p = min(max(p, 1.0 / (n + 2.0)), 1.0 - 1.0 / (n + 2.0))
    return float(ndtri(p))


def _inv_rank(hist_sorted, z, n):
    """Inverse of _rank_z: map a latent z back to data space by interpolating the
    sorted observed history at the CDF level Phi(z)."""
    from scipy.special import ndtr
    p = float(ndtr(z))
    # target position in [0, n-1]
    pos = p * (n + 1.0) - 0.5 - 0.5   # invert (r+0.5)/(n+1)=p with r index space
    pos = min(max(pos, 0.0), n - 1.0)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(hist_sorted[lo] * (1.0 - frac) + hist_sorted[hi] * frac)


def _copula_robust_2d(X, meta):
    """Causal Gaussian-copula conditional-mean prediction with a robust EW latent
    correlation. Returns (pred, conf): pred[t,j] is the copula conditional-mean for a
    missing cell, conf[t,j] marks cells where a usable observed cross-section was
    available to condition on. Fully point-in-time."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    conf = np.zeros((T, N), dtype=bool)
    lam = 0.5 ** (1.0 / HALFLIFE)

    # per-column observed history (kept sorted for rank transform + inverse)
    hist = [[] for _ in range(N)]            # sorted lists of observed values
    # EW correlation accumulators in LATENT space
    w = 0.0
    Sz = np.zeros(N)                          # EW sum of latent z
    M2 = np.zeros((N, N))                     # EW sum of z z^T
    # expanding per-column mean (data space) for cold-start fallback
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)
    I = RIDGE * np.eye(N)
    have_cov = False

    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)
        miss = ~obs

        # --- transform observed entries to latent z using each column's past hist ---
        z = np.zeros(N)
        z_valid = np.zeros(N, dtype=bool)     # columns with a usable rank transform
        for j in np.where(obs)[0]:
            h = hist[j]
            n = len(h)
            if n >= RANKWARM:
                z[j] = _rank_z(h, row[j], n)
                z_valid[j] = True

        # --- build current latent correlation (from rows < t) ---
        corr = None
        mean = np.zeros(N)
        if have_cov and w > EPS:
            mean = Sz / w
            cov = M2 / w - np.outer(mean, mean)
            d = np.sqrt(np.maximum(np.diag(cov), EPS))
            corr = cov / np.outer(d, d)
            np.fill_diagonal(corr, 1.0)

        o = np.where(obs & z_valid)[0]        # condition only on rank-valid latents

        # --- impute missing entries ---
        if miss.any():
            gmean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
            filled = gmean.copy()
            m = np.where(miss)[0]
            if corr is not None and o.size:
                Coo = corr[np.ix_(o, o)] + I[np.ix_(o, o)]
                zo = z[o] - mean[o]
                try:
                    c = cho_factor(Coo, lower=True, check_finite=False)
                    sol = cho_solve(c, zo, check_finite=False)
                    Cmo = corr[np.ix_(m, o)]
                    zm = mean[m] + Cmo @ sol
                    cross_ok = o.size >= max(2, MIN_COND)
                    for idx, j in enumerate(m):
                        h = hist[j]
                        n = len(h)
                        if n >= RANKWARM:
                            filled[j] = _inv_rank(h, zm[idx], n)
                            if cross_ok:
                                conf[t, j] = True
                except Exception:
                    pass
            out[t, miss] = filled[miss]

        # --- fold row t into stats AFTER imputing (causal) ---
        # latent vector for the EW correlation: observed -> its rank z; missing/cold
        # columns -> 0 (their EW mean), so they neither help nor hurt the correlation.
        zf = np.where(z_valid, z, 0.0)
        # robust Student-t downweight from the row's Mahalanobis distance under the
        # current correlation (only the rank-valid coords).
        u = 1.0
        if have_cov and w > EPS:
            ov = np.where(z_valid)[0]
            p = ov.size
            if p:
                mean = Sz / w
                cov = M2 / w - np.outer(mean, mean)
                dd = np.sqrt(np.maximum(np.diag(cov), EPS))
                corr = cov / np.outer(dd, dd)
                np.fill_diagonal(corr, 1.0)
                Cb = corr[np.ix_(ov, ov)] + I[np.ix_(ov, ov)]
                zb = zf[ov] - mean[ov]
                try:
                    cc = cho_factor(Cb, lower=True, check_finite=False)
                    sb = cho_solve(cc, zb, check_finite=False)
                    d2 = float(zb @ sb)
                    u = (NU + p) / (NU + max(d2, 0.0))
                    u = min(u, 4.0)           # cap upweighting of "too clean" rows
                except Exception:
                    u = 1.0
        ww = lam * w + u
        Sz = lam * Sz + u * zf
        M2 = lam * M2 + u * np.outer(zf, zf)
        w = ww

        # update per-column histories (insertion-sorted) and cold-start means
        for j in np.where(obs)[0]:
            v = row[j]
            h = hist[j]
            pos = np.searchsorted(h, v)
            h.insert(pos, v)
            col_sum[j] += v
            col_cnt[j] += 1
        if w >= WARM:
            have_cov = True

    if np.isnan(out).any():
        gm = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(gm, idx[1])
    return out, conf


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if _is_panel(meta):
        return _fe(X, meta)
    T, N = X.shape
    if N == 1:                                   # univariate: causal seasonal / TRMF
        try:
            seas_out, seas_mask, used = _impute_1d(X)
            if used and seas_mask.any():
                base = np.asarray(_trmf(X, meta), float)
                base[seas_mask] = seas_out[seas_mask]
                return base
        except Exception:
            pass
        return _trmf(X, meta)
    if N >= N_GATE:
        return _ewcov_2d(X, meta)

    # Narrow 2D: blend the temporal low-rank base (TRMF, captures AR dynamics, robust
    # under block/whole-row gaps) with the copula-robust cross-sectional conditional
    # mean (captures heavy-tail + high effective-rank cross-sections). We trust the
    # copula prediction ONLY where it had a real observed cross-section to condition
    # on (conf); elsewhere the temporal base carries the cell. This makes ONE core
    # cover heavy-tail / high-rank / benign without any value-based gate.
    base = np.asarray(_trmf(X, meta), float)
    try:
        cop, conf = _copula_robust_2d(X, meta)
    except Exception:
        return base
    miss = np.isnan(X)
    use = miss & conf & np.isfinite(cop)
    if not use.any():
        return base
    out = base.copy()
    out[use] = BLEND_W * cop[use] + (1.0 - BLEND_W) * base[use]
    if np.isnan(out).any():
        out[np.isnan(out)] = base[np.isnan(out)]
    return out


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w10_copula_robust", online_impute))
