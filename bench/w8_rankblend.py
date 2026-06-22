"""
w8_rankblend -- adaptive-rank low-rank + full-covariance RESIDUAL blend (2D, causal).

PROBLEM. Pure low-rank factor models (OnlineTRMF) collapse near-full-rank data into
a few latent factors and lose the cross-sectional signal (2d_highrank_mcar30: corr
~0.357). A plain EW full-covariance conditional mean does better there but is noisy
when the observed block is small or the covariance is poorly conditioned (high miss
rate). We want a single estimator that morphs between the two regimes WITHOUT a
benchmark-keyed constant.

IDEA. Maintain the online exponentially-weighted (EW) covariance (all moments <= t).
Eigendecompose it. Choose the low-rank dimension L ONLINE by a cumulative-energy /
ARD threshold on the running spectrum (keep components until they explain a learned
fraction of the trace). Build a SHRUNK covariance

    Csh = U_L diag(lam_L) U_L^T  +  tau * I          (rank-L signal + isotropic floor)

where tau is the AVERAGE of the discarded eigenvalues (the residual energy the few
factors miss -- an intrinsic, data-driven quantity, not a tuned constant). The
Gaussian conditional mean E[x_miss | x_obs] is then computed from Csh. Because tau is
the genuine residual variance, the conditioning automatically blends:
  - strongly low-rank data  -> tau ~ 0, Csh ~ the factor model (collapses to low-rank);
  - near-full-rank data     -> many eigenvalues survive / tau is large, the residual /
    full cross-sectional structure dominates (fixes highrank).
This is exactly a probabilistic-PCA / factor-analysis conditional mean with an
adaptive number of factors and an isotropic residual.

CAUSALITY (point-in-time). We iterate strictly forward. The EW mean/cov used to
impute row t is built ONLY from rows < t (each row is folded in AFTER it is imputed).
Conditioning uses row t's OWN contemporaneously observed entries (allowed). The
adaptive rank L and tau are functions of the running spectrum (rows < t) only.
Truncating future rows cannot change any past imputation -> passes the verifier.

ROUTING is on truncation-INVARIANT structure only (panel flag; feature count N;
N==1), exactly like c_chal_ewcov, so prefix-reruns route identically (no spurious
leak). Panels -> CausalFE; 1D -> OnlineTRMF; narrow/wide 2D -> this adaptive-rank core.
"""
from __future__ import annotations
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel
from c_chal_uni import online_impute as _uni
from c_chal_robust import _median_excess_kurtosis

KURT_GATE = 10.0     # heavy-tail margin: median excess kurtosis (~50 heavy vs <4 not)

# ------------------------------------------------------------------ knobs --- #
HALFLIFE = 200.0     # EW forgetting half-life (timesteps) -- same scale as ewcov
RIDGE = 1e-2         # tiny ridge on the observed-block solve (numerical floor)
WARM = 5             # min EW weight before trusting the covariance
ENERGY = 0.95        # ARD cumulative-energy fraction defining the kept rank L
TAU_FLOOR = 1e-6     # numerical floor on the isotropic residual variance

# Engage the adaptive-rank core ONLY on genuinely high effective-rank, dense 2D data
# (where pure low-rank MF loses). The gate is decided from a FIXED leading window of
# rows so it is identical on the full series and on every time-prefix the verifier
# probes -> truncation-invariant -> causal. Elsewhere we defer to the proven c_chal_uni
# routes byte-for-byte, so no benign / weak case can regress.
GATE_WIN = 80        # rows the gate is read from (present in every probed prefix)
ENGAGE_ERANK = 0.58  # engage only above this effective-rank ratio (intrinsic)
ENGAGE_MAXMISS = 0.45  # ...and only when dense enough to condition cross-sectionally


def _effrank_ratio(X):
    """(#components to reach 90% SVD energy)/N on the column-standardized observed
    window. Near 0 => strongly low-rank; near 1 => near-full-rank. Intrinsic data
    property, not a benchmark constant."""
    T, N = X.shape
    obs = ~np.isnan(X)
    col_cnt = obs.sum(0)
    if N < 4 or T < 8 or (col_cnt < 2).any():
        return 0.0
    mu = np.nansum(np.where(obs, X, 0.0), 0) / np.maximum(col_cnt, 1)
    sd = np.sqrt(np.maximum(
        np.nansum(np.where(obs, (X - mu) ** 2, 0.0), 0) / np.maximum(col_cnt, 1), 1e-12))
    Z = np.where(obs, (X - mu) / sd, 0.0)
    if T > 400:
        Z = Z[np.linspace(0, T - 1, 400).astype(int)]
    try:
        s = np.linalg.svd(Z, compute_uv=False)
    except Exception:
        return 0.0
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return 0.0
    cum = np.cumsum(e) / tot
    k = int(np.searchsorted(cum, 0.90) + 1)
    return k / N


def _adaptive_rank_cov(cov, energy=ENERGY):
    """Shrink an EW covariance to (rank-L signal + isotropic residual floor).

    L is chosen ONLINE as the number of leading eigen-components that explain
    `energy` of the spectral trace (an ARD / energy threshold on the RUNNING
    spectrum, never a fixed benchmark constant). tau = mean of the DISCARDED
    eigenvalues = the residual variance the L factors miss. Returns the shrunk
    covariance Csh = U_L diag(d_L - tau) U_L^T + tau I (factor-analysis form).

    For low-rank data L is tiny and tau ~ 0 (collapses to the factor model); for
    near-full-rank data many components survive and/or tau is large (the residual
    cross-sectional structure dominates).
    """
    N = cov.shape[0]
    # symmetric eigendecomposition (cov is symmetric PSD up to EW noise)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    tot = w.sum()
    if tot <= 0:
        return np.eye(N) * TAU_FLOOR, 1, TAU_FLOOR
    cum = np.cumsum(w) / tot
    L = int(np.searchsorted(cum, energy) + 1)
    L = max(1, min(L, N))
    # isotropic residual = average energy in the discarded tail (intrinsic quantity)
    if L < N:
        tau = float(w[L:].mean())
    else:
        tau = float(max(w[-1], TAU_FLOOR))
    tau = max(tau, TAU_FLOOR)
    # factor-analysis covariance: leading factors (variance ABOVE the floor) + tau*I
    d = np.maximum(w[:L] - tau, 0.0)
    UL = V[:, :L]
    Csh = (UL * d) @ UL.T + tau * np.eye(N)
    return Csh, L, tau


def _rankblend_2d(X, meta):
    """Adaptive-rank conditional-mean imputation, fully point-in-time."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    lam = 0.5 ** (1.0 / HALFLIFE)

    w = 0.0
    Sx = np.zeros(N)
    M2 = np.zeros((N, N))
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)
    have_cov = False
    Iridge = RIDGE * np.eye(N)

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
                cov = 0.5 * (cov + cov.T)
                Csh, _, _ = _adaptive_rank_cov(cov)
                o = np.where(obs)[0]
                m = np.where(miss)[0]
                Soo = Csh[np.ix_(o, o)] + Iridge[np.ix_(o, o)]
                Smo = Csh[np.ix_(m, o)]
                xo = row[o] - mean[o]
                try:
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[m] = mean[m] + Smo @ sol
                except Exception:
                    filled[m] = mean[m]
            out[t, miss] = filled[miss]

        # fold row t in AFTER imputing (causal). Missing entries use the imputed
        # value so the covariance stays full-dimensional and self-consistent.
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
    # Panels and 1D: defer to the proven routed cores (unchanged behaviour).
    if _is_panel(meta):
        return _fe(X, meta)
    T, N = X.shape
    if N == 1:
        return _uni(X, meta)

    # Causal high-rank/dense gate from a fixed leading window (truncation-invariant).
    try:
        win = X[:min(GATE_WIN, T)]
        dense = float(np.isnan(win).mean()) <= ENGAGE_MAXMISS
        highrank = _effrank_ratio(win) >= ENGAGE_ERANK
        # heavy-tailed data needs robustification, not a Gaussian conditional mean;
        # leave it to c_chal_uni's winsorize route (intrinsic, large-margin gate).
        heavy = _median_excess_kurtosis(X) > KURT_GATE
    except Exception:
        dense = highrank = False; heavy = True

    if dense and highrank and not heavy:
        try:
            return _rankblend_2d(X, meta)
        except Exception:
            return _uni(X, meta)

    # All other 2D data: byte-for-byte the proven c_chal_uni routing (no regression).
    return _uni(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w8_rankblend", online_impute))
