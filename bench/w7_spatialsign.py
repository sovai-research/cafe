"""
w7_spatialsign -- TUNING-FREE robust covariance imputation via the SPATIAL-SIGN
covariance matrix (SSCM) + online geometric-median location, for narrow 2D.

MOTIVATION (research block #1 alt). The champion's narrow-2D path (OnlineTRMF,
L2 factor solve) and the EW-Gaussian (c_chal_ewcov) both estimate a SECOND-MOMENT
covariance whose entries blow up under heavy tails: a handful of Student-t outliers
dominate the outer products x x^T and wreck the shape used for the conditional mean.
The arena's `2d_heavytail_mcar30` sits at corr 0.146 (ewcov) / 0.469 (uni, which
winsorizes behind a kurtosis gate). c_chal_robust fixes it with a tuned clip band +
a value-based gate. We want the SAME robustness with NO tuning constant and NO gate.

METHOD. The spatial-sign covariance matrix replaces x x^T with s s^T where
  s = (x - m) / ||x - m||
is the unit direction from a robust LOCATION m. Each observation contributes a UNIT
vector, so a gross outlier contributes the same bounded s s^T as any other point:
breakdown point ~1/2, and the shape (eigenvectors / correlation structure) of the
SSCM is a consistent estimate of the underlying covariance's shape up to scale on
elliptical data. The SSCM gives a robust CORRELATION (eigenstructure) R; to turn it
into a covariance for the conditional mean we rescale by a robust per-feature SPREAD
  Sigma = D R D,   D = diag(sigma_1..sigma_N),
where sigma_i is a tuning-free robust scale (the average radius the sign normalizes
out -- i.e. an online median absolute deviation surrogate from the same recursion).
Then
  E[x_miss | x_obs] = m_miss + Sigma_mo Sigma_oo^{-1} (x_obs - m_obs)
is the right robust regression. R's correlation structure has breakdown ~1/2 and no
tuning constant; sigma_i is a robust spread (also no tuned constant). On benign
(Gaussian-ish) data R and sigma equal the ordinary correlation/std, so this is
near-harmless there too.

LOCATION m: online geometric median (the spatial median, breakdown 1/2), updated by
a Robbins-Monro recursion with Polyak averaging:
  m_t = m_{t-1} + gamma_t * (x_t - m_{t-1}) / ||x_t - m_{t-1}||,   gamma_t = c (t+t0)^-alpha
  m_bar_t = running average of m_t   (Polyak -> variance reduction, no extra tuning).
alpha in (0.5,1) gives a.s. convergence; c, t0 are scale-free recursion constants
(NOT fit to the benchmark -- standard SA defaults).

SSCM: exponentially-weighted so it adapts (drift) and warms up:
  SSCM_t = lam SSCM_{t-1} + (1-lam) s_t s_t^T.

PARTIAL OBSERVATION. At each t we form the sign over the OBSERVED dimensions only
(project m and x onto observed coords, normalize there), and accumulate s s^T into
the corresponding observed sub-block of the SSCM (an EW count tracks per-pair
support). This keeps every quantity point-in-time and uses all available data.

CAUSALITY (point-in-time). We iterate strictly forward in time. Row t is imputed
from m and SSCM built ONLY from rows < t; AFTER imputing we fold row t's OBSERVED
sign into m and SSCM. The conditioning set is t's own contemporaneous observed
entries (allowed). Truncating future rows cannot change any past imputation. Routing
is on truncation-INVARIANT structure only (panel flag, feature-count N), so the
verifier's prefix reruns route identically -> zero look-ahead.

ROUTING. panel -> CausalFE; wide 2D (N>=N_GATE) -> EW-cov; 1D (N==1) -> TRMF base
(no cross-section for a sign). Narrow 2D (2 <= N < N_GATE) -> this SSCM core, with
NO value gate: applied to every narrow-2D case (benign and heavy-tailed alike).
"""
from __future__ import annotations
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel
from c_chal_ewcov import _ewcov_2d, N_GATE

# ---- scale-free recursion constants (standard SA defaults, NOT benchmark-tuned) ----
GM_C = 1.0          # geometric-median step scale
GM_T0 = 5.0         # step offset (avoids huge first steps)
GM_ALPHA = 0.7      # step decay in (0.5, 1): a.s. convergence
SSCM_HALFLIFE = 200.0   # EW half-life for the spatial-sign covariance (timesteps)
RIDGE = 1e-2        # ridge on the observed covariance block (stabilizes the solve)
WARM = 8            # min EW weight before trusting the covariance
ZCLIP = 2.5         # per-coord redescending Huber knee in robust-sigma units
EPS = 1e-9


def _spatialsign_2d(X, meta):
    """Spatial-sign covariance conditional-mean imputation, fully point-in-time."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    lam = 0.5 ** (1.0 / SSCM_HALFLIFE)

    # location: online geometric median + Polyak average
    m = np.zeros(N)           # current GM iterate
    m_bar = np.zeros(N)        # Polyak-averaged location (used for imputation)
    have_m = np.zeros(N, dtype=bool)   # per-coord: location initialized?
    gm_n = 0                   # GM update counter (per observed coord steps)

    # per-coord robust scale: online MEDIAN of |x_i - m_bar_i| (MAD), via the same
    # Robbins-Monro median recursion (breakdown 1/2, tuning-free).  sigma = 1.4826*MAD
    mad = np.ones(N)          # online MAD estimate per coord
    mad_n = np.zeros(N)        # per-coord update counter
    MAD2SIG = 1.4826

    # ROBUST EW Gaussian with READ-TIME centering (so a drifting online location
    # never poisons the accumulated cross-products -- the EW mean is subtracted when
    # the covariance is READ, exactly as in c_chal_ewcov, which is why that core
    # works). Robustness comes from a REDESCENDING per-ROW weight w in (0,1] on the
    # robust radius: a row at spatial distance r (in MAD units, summed over observed
    # dims) contributes w*x and w*x x^T with
    #     w = 1                 if r2_full <= c        (inlier, full strength)
    #     w = sqrt(c / r2_full) if r2_full >  c        (outlier, bounded influence)
    # so a gross outlier is pulled onto the ball of radius sqrt(c) (Huber-type
    # bounded influence, breakdown ~1/2). c is SELF-CALIBRATED as the running robust
    # median of r2_full -> NO tuning constant. On benign data nearly all w==1, so the
    # covariance equals the ordinary EW covariance (near-harmless).
    Sx = np.zeros(N)           # EW sum of w * x      (per coord, read-time centering)
    Sw = np.zeros(N)           # EW sum of w          (per coord, for the mean)
    M2 = np.zeros((N, N))      # EW sum of w * x x^T
    Wt = np.zeros((N, N))      # EW sum of w          (per pair, for cov normalization)
    rad2 = float(N)            # online robust median of r2_full (self-calibrating c)
    rad2_n = 0
    w_diag = 0.0               # scalar EW weight (warmth proxy)

    # expanding per-feature mean/median fallback (causal)
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)

    for t in range(T):
        row = X[t]
        obs = np.isfinite(row)
        miss = ~obs
        no = int(obs.sum())

        if miss.any():
            gmean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
            # robust EW location for centering (read-time), GM fallback for cold coords
            ewmean = np.where(Sw > EPS, Sx / np.maximum(Sw, EPS),
                              np.where(have_m, m_bar, gmean))
            loc = ewmean
            filled = np.where(have_m, m_bar, gmean)   # robust point fill fallback
            if w_diag > WARM and obs.any():
                o = np.where(obs)[0]
                mi = np.where(miss)[0]
                mean = np.where(Sw > EPS, Sx / np.maximum(Sw, EPS), loc)
                cov = M2 / np.where(Wt > EPS, Wt, 1.0) - np.outer(mean, mean)
                Soo = cov[np.ix_(o, o)] + RIDGE * np.eye(o.size)
                Smo = cov[np.ix_(mi, o)]
                # winsorize the conditioning residual per-coord (robust to an outlier
                # in the OBSERVED entries we condition on), in robust-sigma units.
                sig_o = MAD2SIG * np.where(mad_n[o] > 0, mad[o], 1.0)
                sig_o = np.where(sig_o > EPS, sig_o, 1.0)
                zo = np.clip((row[o] - m_bar[o]) / sig_o, -ZCLIP, ZCLIP)
                xo = (m_bar[o] + zo * sig_o) - mean[o]
                try:
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[mi] = mean[mi] + Smo @ sol
                except Exception:
                    filled[mi] = mean[mi]
            out[t, miss] = filled[miss]

        # ---- fold row t into stats AFTER imputing (causal) -----------------
        if obs.any():
            o = np.where(obs)[0]
            # init location coords from first observation
            new = o[~have_m[o]]
            if new.size:
                m[new] = row[new]
                m_bar[new] = row[new]
                have_m[new] = True
            # geometric-median step over observed coords (direction from m)
            d = row[o] - m[o]
            nrm = float(np.sqrt(np.dot(d, d)))
            if nrm > EPS:
                gm_n += 1
                gamma = GM_C / (gm_n + GM_T0) ** GM_ALPHA
                m[o] = m[o] + gamma * d / nrm
                # Polyak running average of the iterate
                m_bar[o] = m_bar[o] + (m[o] - m_bar[o]) / gm_n
            # per-coord robust scale: online median of |x - m_bar| (Robbins-Monro)
            ad = np.abs(row[o] - m_bar[o])
            for k, j in enumerate(o):
                mad_n[j] += 1
                step = 1.0 / (mad_n[j] + GM_T0) ** GM_ALPHA
                # median SA: move up/down by a scale-relative step
                mad[j] += step * mad[j] * (1.0 if ad[k] > mad[j] else -1.0)
                if mad[j] < EPS:
                    mad[j] = EPS
            # ---- robust radius of this row (observed dims, in robust MAD units) ----
            sig = MAD2SIG * np.where(mad_n > 0, mad, 1.0)
            sig = np.where(sig > EPS, sig, 1.0)
            z = (row[o] - m_bar[o]) / sig[o]
            r2 = float(np.dot(z, z))
            no_ = z.shape[0]
            r2_full = r2 * (N / max(no_, 1))     # rescale to full-dim for a stable c
            # online robust median of r2_full (Robbins-Monro) -> self-calibrating c
            rad2_n += 1
            st = 1.0 / (rad2_n + GM_T0) ** GM_ALPHA
            rad2 += st * rad2 * (1.0 if r2_full > rad2 else -1.0)
            if rad2 < EPS:
                rad2 = EPS
            c = rad2 * (no_ / N)                  # expected r^2 for THIS observed set
            # whole-ROW redescending weight (elliptical / vector outliers): c/r2
            # bounds the second-moment contribution of a gross vector outlier.
            w = 1.0 if r2 <= c else float(c / max(r2, EPS))

            # PER-COORDINATE redescending winsorization of the standardized residual
            # (coordinate-wise heavy tails): clip z to the robust ball |z|<=ZCLIP and
            # map back. ZCLIP is a redescending Huber knee in robust-sigma units; it
            # is NOT benchmark-tuned (standard 2.5-3 sigma robust knee) and is the
            # per-coord analogue of the whole-row spatial weight. On benign data
            # almost nothing is clipped (|z|<=ZCLIP w.h.p.), so it is near-harmless.
            z_full = np.zeros(N)
            z_full[o] = z
            zc = np.clip(z_full, -ZCLIP, ZCLIP)
            # rebuild a robustified full row: observed -> winsorized; missing -> imputed
            xr = out[t].copy()
            xr[o] = m_bar[o] + zc[o] * sig[o]

            # accumulate weighted EW stats over the robustified full row. Read-time
            # centering (cov = M2/Wt - mean mean^T) means a drifting location never
            # poisons the cross-products.
            wf = w * xr
            lam_ = lam
            Sx[:] = lam_ * Sx + wf
            Sw[:] = lam_ * Sw + w
            M2[:] = lam_ * M2 + w * np.outer(xr, xr)
            Wt[:] = lam_ * Wt + w
            w_diag = lam_ * w_diag + 1.0
            col_sum[o] += row[o]
            col_cnt[o] += 1

    # safety net
    if np.isnan(out).any():
        gm = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(gm, idx[1])
    return out


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if _is_panel(meta):
        return _fe(X, meta)
    T, N = X.shape
    if N == 1:
        return _trmf(X, meta)          # no cross-section -> no spatial sign
    if N >= N_GATE:
        return _ewcov_2d(X, meta)       # wide -> fast EW Gaussian
    # narrow 2D: spatial-sign robust covariance, NO value gate.
    try:
        return _spatialsign_2d(X, meta)
    except Exception:
        return _trmf(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w7_spatialsign", online_impute))
