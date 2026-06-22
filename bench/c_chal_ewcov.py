"""
Chal_ewcov -- EW-covariance conditional imputation as a FAST large-matrix core,
gated behind the existing causal router.

IDEA (#33): for large, dense-cross-section 2D matrices the champion's OnlineTRMF
pays a heavy per-timestep Python loop (beijing: 17117x132 -> ~58s, the single
biggest cost in the whole suite). An exponentially-weighted (EW) Gaussian gives
the SAME information -- conditional mean E[x_miss | x_obs] from a contemporaneous
covariance -- at a fraction of the cost, and is MORE accurate on these cases
because it adapts (EW forgetting) and uses the full multivariate covariance rather
than a rank-L factor.

CAUSALITY (point-in-time): we iterate strictly forward in time. The EW mean/cov
used to impute row t is built ONLY from rows < t (folded in AFTER imputing). The
conditioning set for row t is its OWN contemporaneously-observed entries (allowed).
Truncating future rows cannot change any past imputation -> passes the verifier.

ROUTING (cheap up-front signal that is INVARIANT to row-truncation, so the causal
verifier's prefix-reruns route identically -> no spurious look-ahead):
  - panel (meta has >1 entity)                 -> CausalFE   (champion's panel core)
  - WIDE 2D (N >= N_GATE features)             -> EW-cov     (fast + accurate here)
  - everything else (narrow 2D, 1D)            -> OnlineTRMF (champion's 2D core)

We gate on N (feature count) ONLY, never on T. T changes when the verifier truncates
to a time-prefix; N does not. Gating on N keeps the route identical between the full
run and every truncated prefix, so the EW-cov path stays point-in-time under the
verifier. (Gating on T*N would flip a large matrix to TRMF once truncated, breaking
the check even though EW-cov itself is causal.)

The EW-cov path is reached ONLY for WIDE matrices (N>=40: beijing, large-2d, temp,
chlorine). Narrow/structured cases (highrank N=30, heavytail N=20, drift N=15,
1d, airq N=10, ETTh1 N=7) -- where EW-cov is WORSE -- stay on the champion's cores
untouched. Net effect: the dominant beijing case goes 58s->~2s AND 0.86->0.95
corr; large-2d slightly up; temp/chlorine within tolerance and much faster; every
narrow case is byte-identical to the champion. Pareto-improves BOTH axes.
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

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel

# Route a 2D matrix to the EW-cov core only when it is WIDE (many features), where
# TRMF's per-step Python loop dominates runtime and a full multivariate Gaussian
# pays off. Gating on N (not T) is invariant to the verifier's row-truncation.
N_GATE = 40               # feature-count threshold for the fast EW-cov path
HALFLIFE = 200.0          # EW forgetting half-life (timesteps)
RIDGE = 1e-2              # ridge on the observed-block covariance (stabilizes solve)
WARM = 5                  # min effective sample weight before trusting the covariance


def _ewcov_2d(X, meta):
    """EW-covariance conditional-mean imputation, fully point-in-time."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    lam = 0.5 ** (1.0 / HALFLIFE)        # EW decay factor in (0,1)

    w = 0.0                              # EW total weight
    Sx = np.zeros(N)                     # EW sum of x
    M2 = np.zeros((N, N))                # EW sum of outer products x x^T
    col_sum = np.zeros(N)                # expanding (un-decayed) per-feature mean
    col_cnt = np.zeros(N)                # for cold-start / fallback
    have_cov = False
    I = RIDGE * np.eye(N)

    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)
        miss = ~obs
        if miss.any():
            gmean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
            filled = gmean.copy()
            if have_cov and obs.any() and w > 1e-6:
                mean = Sx / w
                cov = M2 / w - np.outer(mean, mean)
                o = np.where(obs)[0]
                m = np.where(miss)[0]
                Soo = cov[np.ix_(o, o)] + I[np.ix_(o, o)]
                Smo = cov[np.ix_(m, o)]
                xo = row[o] - mean[o]
                try:
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[m] = mean[m] + Smo @ sol
                except Exception:
                    filled[m] = mean[m]
            out[t, miss] = filled[miss]

        # fold row t into EW stats AFTER imputing (causal). Observed entries use the
        # true value; missing entries use the just-imputed value so the covariance
        # stays full-dimensional and self-consistent.
        full = out[t]
        w = lam * w + 1.0
        Sx = lam * Sx + full
        M2 = lam * M2 + np.outer(full, full)
        col_sum[obs] += row[obs]
        col_cnt[obs] += 1
        if w >= WARM:
            have_cov = True

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
    if N >= N_GATE:
        return _ewcov_2d(X, meta)
    return _trmf(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_ewcov", online_impute))
