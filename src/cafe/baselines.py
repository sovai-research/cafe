"""Classical / non-deep imputation baselines, as first-class CAFE library methods.

This module promotes the benchmark baselines we already maintain into the library so
``cafe`` is a one-stop imputation toolkit AND the paper's "special cases of CAFE" table
(``tab:special``) is backed by *runnable* code rather than prose. Every method here is a
known classical estimator that CAFE recovers as a corner of its learned parameter space.

Uniform API
-----------
Two equivalent entry points, both container-native (numpy / pandas / polars, 1D or 2D)
via :func:`cafe.io.to_matrix` / :func:`cafe.io.from_matrix`, returning the SAME container
type as the input with observed cells preserved EXACTLY:

  * named functions -- ``softimpute(data, ...)``, ``locf(data)``, ``trmf(data)``, ...
  * a registry dispatcher -- ``impute(data, method="softimpute", **kw)``, plus the
    ``METHODS`` dict and :func:`list_methods`.

Causality labels (the whole point -- the paper's thesis is point-in-time imputation)
------------------------------------------------------------------------------------
Each method is tagged CAUSAL (point-in-time: the fill at (t, j) uses only data at
times <= t) or BATCH (non-causal: it may read future observations / the whole matrix).
The tag is first-class metadata in :data:`METHODS` and stated in every docstring.

  CAUSAL  : locf, mean_impute, rolling_mean, rolling_median, ewma, drift,
            kalman_local_level, xsec_mean, online_trmf, ewcov, gaussian_copula
  BATCH   : linear_interp, softimpute, trmf, mcnnm, knn, mice

The numerical cores are ported faithfully from the benchmark implementations
(``bench/m_softimpute.py``, ``bench/m_trmf.py``, ``bench/c_online_trmf.py``,
``bench/m_naive.py``, ``bench/causal_simple.py``, ``bench/c_baselines.py``,
``bench/c_chal_ewcov.py``, ``bench/online_competitors.py``); see ``src/tests/
test_baselines.py`` for the parity checks against those references.
"""
from __future__ import annotations

import warnings

import numpy as np

from .io import from_matrix, to_matrix

__all__ = [
    "impute", "list_methods", "METHODS",
    # named methods (causal)
    "locf", "mean_impute", "rolling_mean", "rolling_median", "ewma", "drift",
    "kalman_local_level", "xsec_mean", "online_trmf", "ewcov", "gaussian_copula",
    # named methods (batch / non-causal)
    "linear_interp", "softimpute", "trmf", "mcnnm", "knn", "mice",
]


# =========================================================================== #
# Shared NaN-safe helpers (matrix-level; X is (T, N) float64, NaN = missing)
# =========================================================================== #
def _col_mean_fill(X):
    """Per-column observed mean; all-NaN columns -> global mean -> 0.0."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col = np.nanmean(X, axis=0)
        g = np.nanmean(X)
    g = g if np.isfinite(g) else 0.0
    return np.where(np.isfinite(col), col, g)


def _finalize(out, X):
    """Preserve observed cells exactly and scrub any residual non-finite cell."""
    X = np.asarray(X, float)
    out = np.asarray(out, float)
    obs = np.isfinite(X)
    out = np.where(obs, X, out)
    bad = ~np.isfinite(out)
    if bad.any():
        colmean = _col_mean_fill(X)
        out[bad] = np.take(colmean, np.nonzero(bad)[1])
    return out


# =========================================================================== #
# Matrix-level cores (numpy in -> numpy out). Wrapped for containers below.
# =========================================================================== #
# --- CAUSAL: last observation carried forward ------------------------------ #
def _locf(X):
    # STRICTLY CAUSAL: x[t] = last value observed at or before t. Cells before a
    # column's first observation have no admissible past, so they are filled with 0
    # (the neutral prior) -- never the full-column mean, which would peek at future
    # observations and make the leading edge non-causal (the leakage audit flags that).
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    ar = np.arange(T)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        if not obs.any():
            out[:, j] = 0.0
            continue
        lidx = np.where(obs, ar, -1)
        prev = np.maximum.accumulate(lidx)
        valid = prev >= 0
        col[valid] = col[prev[valid]]
        out[:, j] = np.where(np.isfinite(col), col, 0.0)   # leading (pre-first-obs) -> 0
    return _finalize(out, X)


# --- CAUSAL: expanding per-column mean ------------------------------------- #
def _mean_impute(X):
    X = np.asarray(X, float)
    obs = np.isfinite(X)
    xv = np.where(obs, X, 0.0)
    csum = np.cumsum(xv, axis=0)
    ccnt = np.cumsum(obs, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        run = csum / np.maximum(ccnt, 1)
    run[ccnt == 0] = np.nan          # nothing seen yet -> filled by _finalize fallback
    return _finalize(run, X)


# --- CAUSAL: trailing-window mean / median --------------------------------- #
def _trailing_reduce(X, W, fn):
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    miss = ~np.isfinite(X)
    for t in range(T):
        lo = max(0, t - W + 1)
        window = X[lo:t + 1]                      # times <= t only (causal)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = fn(window, axis=0)
        m = miss[t]
        out[t, m] = val[m]
    return _finalize(out, X)


def _rolling_mean(X, W=24):
    return _trailing_reduce(X, W, np.nanmean)


def _rolling_median(X, W=24):
    return _trailing_reduce(X, W, np.nanmedian)


# --- CAUSAL: exponentially-weighted moving average ------------------------- #
def _ewma(X, halflife=5.0):
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    obs = np.isfinite(X)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    ew_num = np.zeros(N)
    ew_den = np.zeros(N)
    xs_sum = np.zeros(N)
    xs_cnt = np.zeros(N)
    for t in range(T):
        o = obs[t]
        miss = ~o
        if miss.any():
            xs_mean = np.where(xs_cnt > 0, xs_sum / np.maximum(xs_cnt, 1), np.nan)
            have = ew_den > 0
            ewv = np.where(have, ew_num / np.maximum(ew_den, 1e-300), xs_mean)
            out[t, miss] = ewv[miss]
        if o.any():
            ew_num[o] = (1 - alpha) * ew_num[o] + alpha * X[t, o]
            ew_den[o] = (1 - alpha) * ew_den[o] + alpha
            xs_sum[o] += X[t, o]
            xs_cnt[o] += 1
    return _finalize(out, X)


# --- CAUSAL: linear drift extrapolation ------------------------------------ #
def _drift(X):
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    obs = np.isfinite(X)
    for j in range(N):
        last_v = last_i = prev_v = prev_i = None
        for t in range(T):
            if obs[t, j]:
                prev_v, prev_i = last_v, last_i
                last_v, last_i = X[t, j], t
            else:
                if last_v is None:
                    continue                      # leading gap -> fallback
                elif prev_v is None or last_i == prev_i:
                    out[t, j] = last_v
                else:
                    slope = (last_v - prev_v) / (last_i - prev_i)
                    out[t, j] = last_v + slope * (t - last_i)
    return _finalize(out, X)


# --- CAUSAL: univariate local-level Kalman FILTER -------------------------- #
def _kalman_ll(X, q=0.05, r=1.0):
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    obs = np.isfinite(X)
    for j in range(N):
        m, P, started = 0.0, 1.0, False
        for t in range(T):
            mp, Pp = m, P + q                      # predict
            if obs[t, j]:
                K = Pp / (Pp + r)
                m = mp + K * (X[t, j] - mp)
                P = (1.0 - K) * Pp
                started = True
            else:
                if started:
                    out[t, j] = mp
                m, P = mp, Pp
    return _finalize(out, X)


# --- CAUSAL: contemporaneous cross-sectional mean -------------------------- #
def _xsec_mean(X):
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    obs = np.isfinite(X)
    xs_sum = np.zeros(N)
    xs_cnt = np.zeros(N)
    last_row = np.full(N, np.nan)
    for t in range(T):
        o = obs[t]
        miss = ~o
        if miss.any():
            row_obs = X[t, o]
            contemp = row_obs.mean() if row_obs.size else np.nan
            expanding = np.where(xs_cnt > 0, xs_sum / np.maximum(xs_cnt, 1), np.nan)
            fill = np.full(N, contemp)
            fill = np.where(np.isfinite(fill), fill, last_row)
            fill = np.where(np.isfinite(fill), fill, expanding)
            out[t, miss] = fill[miss]
        if o.any():
            last_row[o] = X[t, o]
            xs_sum[o] += X[t, o]
            xs_cnt[o] += 1
    return _finalize(out, X)


# --- BATCH: per-column linear interpolation (reads future) ----------------- #
def _linear_interp(X):
    X = np.asarray(X, float)
    T, N = X.shape
    out = X.copy()
    t = np.arange(T, dtype=float)
    colmean = _col_mean_fill(X)
    for j in range(N):
        col = out[:, j]
        obs = np.isfinite(col)
        k = int(obs.sum())
        if k == 0:
            out[:, j] = colmean[j]
        elif k == 1:
            out[:, j] = np.where(obs, col, col[obs][0])
        else:
            out[:, j] = np.interp(t, t[obs], col[obs])
    return _finalize(out, X)


# --- BATCH: SoftImpute-ALS (ported from bench/m_softimpute.py) -------------- #
def _standardize(X):
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(X, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return (X - mean) / std, mean, std


def _softimpute_als(Xs, mask_obs, rank, lam, max_iter=100, tol=1e-4, rng=None):
    T, N = Xs.shape
    rng = rng or np.random.default_rng(0)
    Xfill = np.where(mask_obs, Xs, 0.0)
    r = min(rank, T, N)
    # Use lam exactly when it is meaningfully positive (bit-exact parity with the
    # bench reference); only floor the ridge when lam is ~0 (e.g. a constant matrix
    # has top singular value 0 -> lam 0 -> singular solve) so the solve stays SPD.
    lam_solve = lam if lam > 1e-8 else 1e-8
    U = rng.standard_normal((T, r))
    U, _ = np.linalg.qr(U)
    D = np.ones(r)
    V = np.zeros((N, r))
    prev = Xfill.copy()
    for _ in range(max_iter):
        A = U * D
        AtA = A.T @ A
        rhs = A.T @ Xfill
        Bt = np.linalg.solve(AtA + lam_solve * np.eye(r), rhs)
        B = Bt.T
        Vv, dv, _ = np.linalg.svd(B, full_matrices=False)
        V = Vv
        model = A @ B.T
        Xfill = np.where(mask_obs, Xs, model)
        Bmat = V * dv
        BtB = Bmat.T @ Bmat
        rhs2 = Xfill @ Bmat
        Amat = np.linalg.solve(BtB + lam_solve * np.eye(r), rhs2.T).T
        Uu, du, _ = np.linalg.svd(Amat, full_matrices=False)
        U, D = Uu, du
        model = (U * D) @ V.T
        Xfill = np.where(mask_obs, Xs, model)
        num = np.linalg.norm(Xfill - prev)
        den = np.linalg.norm(prev) + 1e-9
        prev = Xfill.copy()
        if num / den < tol:
            break
    Uf, sf, Vtf = np.linalg.svd(Xfill, full_matrices=False)
    sf_shrunk = np.maximum(sf - lam, 0.0)
    keep = sf_shrunk > 0
    if not np.any(keep):
        keep[:1] = True
        sf_shrunk = sf_shrunk.copy()
        sf_shrunk[0] = max(sf_shrunk[0], 1e-6)
    return (Uf[:, keep] * sf_shrunk[keep]) @ Vtf[keep]


def _softimpute(X, rank_cap=30, max_iter=80, tol=1e-4):
    X = np.asarray(X, float)
    # Degenerate cases the low-rank ALS cannot fit (no observed cells, or a trivial
    # 1x1 / single-cell matrix) -> fall back to the mean fill. _finalize handles the rest.
    if not np.isfinite(X).any() or min(X.shape) < 2:
        return _finalize(X.copy(), X)
    Xs, mean, std = _standardize(X)
    mask_obs = np.isfinite(Xs)
    Xs = np.where(mask_obs, Xs, 0.0)
    T, N = Xs.shape
    rank = min(rank_cap, T, N)
    rng = np.random.default_rng(12345)
    try:
        s0 = np.linalg.svd(Xs, compute_uv=False)
        smax = float(s0[0])
    except Exception:
        smax = float(np.linalg.norm(Xs))
    base = smax / np.sqrt(max(T, N))
    obs_idx = np.argwhere(mask_obs)
    n_obs = len(obs_idx)
    best_lam = base * 0.05
    if n_obs >= 60:                                # only scan if enough held-out cells
        n_val = min(2000, max(50, n_obs // 20))
        sel = rng.choice(n_obs, size=n_val, replace=False)
        val_pos = obs_idx[sel]
        val_true = Xs[val_pos[:, 0], val_pos[:, 1]].copy()
        mask_scan = mask_obs.copy()
        mask_scan[val_pos[:, 0], val_pos[:, 1]] = False
        Xs_scan = np.where(mask_scan, Xs, 0.0)
        best_err = np.inf
        for lam in [base * m for m in (0.005, 0.02, 0.05, 0.15, 0.4, 1.0)]:
            recon = _softimpute_als(Xs_scan, mask_scan, rank, lam, max_iter=40,
                                    tol=1e-3, rng=np.random.default_rng(7))
            pred = recon[val_pos[:, 0], val_pos[:, 1]]
            err = np.mean((pred - val_true) ** 2)
            if err < best_err:
                best_err, best_lam = err, lam
    recon = _softimpute_als(Xs, mask_obs, rank, best_lam, max_iter=max_iter,
                            tol=tol, rng=rng)
    out = recon * std + mean
    return _finalize(out, X)


# --- BATCH: TRMF (ported from bench/m_trmf.py, 2D path) -------------------- #
def _trmf_solve_W(X, M, F, lam_w, k):
    T, N = X.shape
    W = np.zeros((N, k))
    I = lam_w * np.eye(k)
    for n in range(N):
        m = M[:, n]
        if not m.any():
            continue
        Fn = F[m]
        A = Fn.T @ Fn + I
        b = Fn.T @ X[m, n]
        W[n] = np.linalg.solve(A, b)
    return W


def _trmf_solve_theta(F, lags, lam_th):
    T, k = F.shape
    L = len(lags)
    maxlag = max(lags)
    theta = np.zeros((k, L))
    if T <= maxlag:
        return theta
    rows = np.arange(maxlag, T)
    for d in range(k):
        D = np.column_stack([F[rows - lag, d] for lag in lags])
        y = F[rows, d]
        A = D.T @ D + lam_th * np.eye(L)
        b = D.T @ y
        theta[d] = np.linalg.solve(A, b)
    return theta


def _trmf_solve_F(X, M, F, W, theta, lags, lam_f, k):
    T, N = X.shape
    maxlag = max(lags)
    for t in range(T):
        m = M[t]
        if m.any():
            Wt = W[m]
            A = Wt.T @ Wt
            b = Wt.T @ X[t, m]
        else:
            A = np.zeros((k, k))
            b = np.zeros(k)
        diag_add = np.zeros(k)
        if t >= maxlag:
            pred = np.zeros(k)
            for li, lag in enumerate(lags):
                pred += theta[:, li] * F[t - lag]
            diag_add += lam_f
            b += lam_f * pred
        for li, lag in enumerate(lags):
            s = t + lag
            if s < T and s >= maxlag:
                th = theta[:, li]
                res = F[s].copy()
                for lj, lagj in enumerate(lags):
                    sj = s - lagj
                    if sj == t:
                        continue
                    res -= theta[:, lj] * F[sj]
                diag_add += lam_f * th * th
                b += lam_f * th * res
        A = A + np.diag(diag_add) + 1e-8 * np.eye(k)
        F[t] = np.linalg.solve(A, b)
    return F


def _trmf(X, k=6, lags=(1, 2, 3), n_iter=20, lam_w=1.0, lam_f=2.0,
          lam_th=1.0, seed=0):
    X = np.asarray(X, float)
    T, N = X.shape
    k = min(k, max(1, T), max(1, N))
    M = np.isfinite(X)
    with warnings.catch_warnings():        # all-NaN columns -> handled by M.any(0) guard
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.where(M.any(0), np.nanmean(np.where(M, X, np.nan), 0), 0.0)
        sd = np.where(M.any(0), np.nanstd(np.where(M, X, np.nan), 0), 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Xs = np.where(M, (X - mu) / sd, 0.0)
    lags = tuple(int(l) for l in lags)
    rng = np.random.default_rng(20260622 + seed)
    F = 0.1 * rng.standard_normal((T, k))
    for _ in range(n_iter):
        W = _trmf_solve_W(Xs, M, F, lam_w, k)
        theta = _trmf_solve_theta(F, lags, lam_th)
        F = _trmf_solve_F(Xs, M, F, W, theta, lags, lam_f, k)
    R = F @ W.T
    out = X.copy()
    fill = R * sd + mu
    out[~M] = fill[~M]
    return _finalize(out, X)


# --- BATCH: MC-NNM (two-way fixed effects + low-rank completion) ----------- #
def _mcnnm(X, rank_cap=30, fe_iter=8, max_iter=80, tol=1e-4):
    """Matrix completion with two-way (row + column) fixed effects + low rank.

    Athey et al. 2021. We estimate additive row/column/grand effects from the
    observed cells (iterated two-way means), subtract them, run a nuclear-norm
    low-rank completion (SoftImpute-ALS) on the residual, then add the effects
    back. BATCH / non-causal (the SVD reconstruction reads the whole matrix).
    """
    X = np.asarray(X, float)
    T, N = X.shape
    M = np.isfinite(X)
    if not M.any() or min(T, N) < 2:
        # no observed cells, or a trivial single-row/col matrix the low-rank step
        # cannot fit -> additive two-way FE / mean fill via _finalize.
        return _finalize(X.copy(), X)
    grand = float(np.nanmean(X))
    row_fe = np.zeros(T)
    col_fe = np.zeros(N)
    for _ in range(fe_iter):
        R = X - grand - col_fe[None, :]
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rm = np.nanmean(np.where(M, R, np.nan), axis=1)
        row_fe = np.where(np.isfinite(rm), rm, 0.0)
        R = X - grand - row_fe[:, None]
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cm = np.nanmean(np.where(M, R, np.nan), axis=0)
        col_fe = np.where(np.isfinite(cm), cm, 0.0)
    base = grand + row_fe[:, None] + col_fe[None, :]
    resid = np.where(M, X - base, np.nan)
    # low-rank completion on the residual (scale-free since FE removed level)
    std = np.nanstd(resid)
    std = std if (np.isfinite(std) and std > 1e-8) else 1.0
    Rs = resid / std
    mask_obs = np.isfinite(Rs)
    Rs0 = np.where(mask_obs, Rs, 0.0)
    rank = min(rank_cap, T, N)
    try:
        smax = float(np.linalg.svd(Rs0, compute_uv=False)[0])
    except Exception:
        smax = float(np.linalg.norm(Rs0))
    lam = (smax / np.sqrt(max(T, N))) * 0.05
    recon = _softimpute_als(Rs0, mask_obs, rank, lam, max_iter=max_iter, tol=tol,
                            rng=np.random.default_rng(12345))
    out = base + recon * std
    return _finalize(out, X)


# --- CAUSAL: EW-covariance / Gaussian conditional mean --------------------- #
def _ewcov(X, halflife=200.0, ridge=1e-2, warm=5):
    """EW-covariance conditional-mean imputation, fully point-in-time.

    Ported from bench/c_chal_ewcov.py. Maintains an exponentially-weighted mean
    and covariance over rows < t; fills the missing entries of row t with the
    Gaussian conditional mean E[x_miss | x_obs] under that covariance.
    """
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    lam = 0.5 ** (1.0 / halflife)
    w = 0.0
    Sx = np.zeros(N)
    M2 = np.zeros((N, N))
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)
    have_cov = False
    I = ridge * np.eye(N)
    for t in range(T):
        row = X[t]
        obs = np.isfinite(row)
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
                    from scipy.linalg import cho_factor, cho_solve
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[m] = mean[m] + Smo @ sol
                except Exception:
                    try:
                        filled[m] = mean[m] + Smo @ np.linalg.solve(Soo, xo)
                    except Exception:
                        filled[m] = mean[m]
            out[t, miss] = filled[miss]
        full = out[t]
        # observed entries use the true value; missing entries use the just-imputed
        # value so the covariance stays full-dimensional and self-consistent.
        w = lam * w + 1.0
        Sx = lam * Sx + full
        M2 = lam * M2 + np.outer(full, full)
        col_sum[obs] += row[obs]
        col_cnt[obs] += 1
        if w >= warm:
            have_cov = True
    return _finalize(out, X)


# --- CAUSAL: online TRMF (ported from bench/c_online_trmf.py, 2D path) ------ #
def _online_trmf(X):
    # delegate to the faithful bench core via a vendored copy of its logic.
    return _online_trmf_core(np.asarray(X, float))


# --- BATCH: sklearn KNN ---------------------------------------------------- #
def _knn(X, n_neighbors=5):
    X = np.asarray(X, float)
    try:
        from sklearn.impute import KNNImputer
    except Exception as e:  # pragma: no cover
        raise ImportError("knn requires scikit-learn (sklearn.impute.KNNImputer)") from e
    T, N = X.shape
    nn = max(1, min(n_neighbors, T - 1)) if T > 1 else 1
    # KNNImputer silently drops all-NaN columns; pre-fill those with col mean.
    colmean = _col_mean_fill(X)
    Xpre = X.copy()
    allnan = np.all(np.isnan(X), axis=0)
    for j in np.where(allnan)[0]:
        Xpre[:, j] = colmean[j]
    out = KNNImputer(n_neighbors=nn).fit_transform(Xpre)
    if out.shape[1] != N:                          # defensive: shape guard
        out = Xpre
    return _finalize(out, X)


# --- BATCH: sklearn IterativeImputer (MICE-style) -------------------------- #
def _mice(X, max_iter=10, random_state=0):
    X = np.asarray(X, float)
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "mice requires scikit-learn (sklearn.impute.IterativeImputer)") from e
    colmean = _col_mean_fill(X)
    Xpre = X.copy()
    allnan = np.all(np.isnan(X), axis=0)
    for j in np.where(allnan)[0]:
        Xpre[:, j] = colmean[j]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = IterativeImputer(max_iter=max_iter,
                               random_state=random_state).fit_transform(Xpre)
    if out.shape[1] != X.shape[1]:
        out = Xpre
    return _finalize(out, X)


# --- CAUSAL: online Gaussian copula (gcimpute) ----------------------------- #
def _gaussian_copula(X):
    X = np.asarray(X, np.float64)
    try:
        from gcimpute.gaussian_copula import GaussianCopula
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "gaussian_copula requires the optional 'gcimpute' package "
            "(pip install gcimpute)") from e
    T, N = X.shape
    obs = np.isfinite(X)
    n_levels = np.array([np.unique(X[obs[:, j], j]).size for j in range(N)])
    if T < max(2, 20) or N < 1 or obs.sum() == 0 or (n_levels.size and n_levels.min() < 2):
        return _finalize(X.copy(), X)
    np.random.seed(12345)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianCopula(training_mode="minibatch-online", batch_size=20,
                                   window_size=40, const_stepsize=0.5,
                                   random_state=1, verbose=0)
            imp = model.fit_transform(X.copy(), continuous=list(range(N)))
    except Exception:
        return _finalize(X.copy(), X)
    return _finalize(np.asarray(imp, np.float64), X)


# =========================================================================== #
# Vendored OnlineTRMF core (faithful copy of bench/c_online_trmf.py, 2D path).
# Kept self-contained so the library has no dependency on the bench/ tree.
# =========================================================================== #
def _ot_solve_factor(WtW_full, W, x, obs, prior_mean, lam_f, lam_ar):
    from scipy.linalg import cho_factor, cho_solve
    L = W.shape[1]
    no = int(obs.sum())
    if no == 0:
        return prior_mean.copy()
    if no == obs.shape[0]:
        A = WtW_full
        rhs = W.T @ x
    else:
        Wo = W[obs]
        A = Wo.T @ Wo
        rhs = Wo.T @ x[obs]
    A = A + (lam_f + lam_ar) * np.eye(L)
    rhs = rhs + lam_ar * prior_mean
    try:
        c = cho_factor(A, lower=True, check_finite=False)
        return cho_solve(c, rhs, check_finite=False)
    except Exception:
        return np.linalg.solve(A, rhs)


def _ot_refresh_W(F_hist, Xf_hist, obs_hist, W, lam_w):
    from scipy.linalg import cho_factor, cho_solve
    M, L = F_hist.shape
    N = W.shape[0]
    Wnew = W.copy()
    if obs_hist.all():
        G = F_hist.T @ F_hist + lam_w * np.eye(L)
        try:
            c = cho_factor(G, lower=True, check_finite=False)
            return np.ascontiguousarray(cho_solve(c, F_hist.T @ Xf_hist,
                                                  check_finite=False).T)
        except Exception:
            return W
    for j in range(N):
        oj = obs_hist[:, j]
        if int(oj.sum()) < L + 1:
            continue
        Fj = F_hist[oj]
        G = Fj.T @ Fj + lam_w * np.eye(L)
        rhs = Fj.T @ Xf_hist[oj, j]
        try:
            c = cho_factor(G, lower=True, check_finite=False)
            Wnew[j] = cho_solve(c, rhs, check_finite=False)
        except Exception:
            pass
    return Wnew


def _ot_refresh_theta(F_hist):
    if F_hist.shape[0] < 3:
        return None
    a = F_hist[:-1]
    b = F_hist[1:]
    num = np.sum(a * b, axis=0)
    den = np.sum(a * a, axis=0) + 1e-8
    return np.clip(num / den, -0.999, 0.999)


def _ot_init_W(Xf, L, lam_w):
    M, N = Xf.shape
    try:
        U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
        Lk = min(L, len(s))
        W = np.zeros((N, L))
        W[:, :Lk] = (Vt[:Lk].T * s[:Lk]) / max(np.sqrt(M), 1.0)
    except Exception:
        W = np.random.default_rng(0).standard_normal((N, L)) * 0.1
    W = np.ascontiguousarray(W)
    return W, np.zeros(L), W.T @ W


def _ot_factors_for_buffer(Xf, Ob, W, WtW, lam_f):
    M, N = Xf.shape
    L = W.shape[1]
    F = np.zeros((M, L))
    zero = np.zeros(L)
    for i in range(M):
        o = Ob[i]
        F[i] = _ot_solve_factor(WtW, W, np.where(o, Xf[i], 0.0), o, zero, lam_f, 0.0)
    return F


def _online_trmf_core(X):
    """2D point-in-time TRMF. CAUSAL. Mirrors c_online_trmf.online_impute (2D)."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    L, LAM_W, LAM_F, LAM_AR = 6, 1e-1, 1e-2, 5.0
    REFRESH_K, MIN_WARM = 10, 8
    out = X.copy()
    obs_all = np.isfinite(X)
    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)
    rng_cap = T + 16
    F_buf = np.zeros((rng_cap, L))
    Xf_buf = np.zeros((rng_cap, N))
    Ob_buf = np.zeros((rng_cap, N), dtype=bool)
    buf_n = 0
    last_f = np.zeros(L)
    have_last = False
    W = None
    theta = np.zeros(L)
    WtW = None
    since = 0
    for t in range(T):
        cs_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        o = obs_all[t]
        miss = ~o
        if W is None:
            out[t, miss] = cs_mean[miss]
            resid = np.where(o, X[t], cs_mean)
            Xf_buf[buf_n] = resid
            Ob_buf[buf_n] = o
            F_buf[buf_n] = 0.0
            buf_n += 1
            col_sum[o] += X[t, o]
            col_cnt[o] += 1
            since += 1
            if buf_n >= MIN_WARM:
                W, theta, WtW = _ot_init_W(Xf_buf[:buf_n], L, LAM_W)
                F_buf[:buf_n] = _ot_factors_for_buffer(Xf_buf[:buf_n], Ob_buf[:buf_n],
                                                       W, WtW, LAM_F)
                th = _ot_refresh_theta(F_buf[:buf_n])
                if th is not None:
                    theta = th
                since = 0
            continue
        resid = np.where(o, X[t], 0.0)
        prior = theta * last_f if have_last else np.zeros(L)
        f = _ot_solve_factor(WtW, W, resid, o, prior, LAM_F, LAM_AR)
        if miss.any():
            rec_miss = W[miss] @ f
            out[t, miss] = rec_miss
            resid[miss] = rec_miss
        Xf_buf[buf_n] = resid
        Ob_buf[buf_n] = o
        F_buf[buf_n] = f
        buf_n += 1
        last_f = f
        have_last = True
        col_sum[o] += X[t, o]
        col_cnt[o] += 1
        since += 1
        if since >= REFRESH_K:
            W = _ot_refresh_W(F_buf[:buf_n], Xf_buf[:buf_n], Ob_buf[:buf_n], W, LAM_W)
            WtW = W.T @ W
            th = _ot_refresh_theta(F_buf[:buf_n])
            if th is not None:
                theta = th
            since = 0
    return _finalize(out, X)


# =========================================================================== #
# Container-aware wrappers: numpy/pandas/polars (1D/2D) in -> same type out.
# =========================================================================== #
def _wrap(core):
    """Lift a matrix-level core ``f(X, **kw) -> X`` to a container-native method."""
    def method(data, **kw):
        X, ctx = to_matrix(data)
        out = core(X, **kw)
        return from_matrix(out, ctx)
    method.__name__ = core.__name__.lstrip("_")
    return method


# Public named functions (container-native). Docstrings carry the causal/batch tag.
def locf(data):
    """CAUSAL (point-in-time). Last observation carried forward; cells before the first
    observation (no admissible past) -> 0, never the future-peeking column mean."""
    return _wrap(_locf)(data)


def mean_impute(data):
    """CAUSAL. Expanding per-column mean of observed values up to (and incl.) t."""
    return _wrap(_mean_impute)(data)


def rolling_mean(data, W=24):
    """CAUSAL. Trailing-window (last W steps) mean of observed values."""
    return _wrap(_rolling_mean)(data, W=W)


def rolling_median(data, W=24):
    """CAUSAL. Trailing-window (last W steps) median -- robust local baseline."""
    return _wrap(_rolling_median)(data, W=W)


def ewma(data, halflife=5.0):
    """CAUSAL. Exponentially-weighted moving average of each series' past observations."""
    return _wrap(_ewma)(data, halflife=halflife)


def drift(data):
    """CAUSAL. Linear extrapolation along the slope of the two most recent observations."""
    return _wrap(_drift)(data)


def kalman_local_level(data, q=0.05, r=1.0):
    """CAUSAL. Univariate local-level Kalman FILTER (random-walk state + obs noise)."""
    return _wrap(_kalman_ll)(data, q=q, r=r)


def xsec_mean(data):
    """CAUSAL. Contemporaneous cross-sectional mean at t (LOCF / expanding-mean fallback)."""
    return _wrap(_xsec_mean)(data)


def online_trmf(data):
    """CAUSAL. Online (point-in-time) Temporal Regularized Matrix Factorization."""
    return _wrap(_online_trmf)(data)


def ewcov(data, halflife=200.0, ridge=1e-2, warm=5):
    """CAUSAL. EW-covariance Gaussian conditional-mean imputation E[x_miss | x_obs]."""
    return _wrap(_ewcov)(data, halflife=halflife, ridge=ridge, warm=warm)


def gaussian_copula(data):
    """CAUSAL. Online Gaussian-copula imputation (gcimpute, AAAI 2022). Optional dep."""
    return _wrap(_gaussian_copula)(data)


def linear_interp(data):
    """BATCH / NON-CAUSAL. Per-column linear interpolation (reads the next observed value)."""
    return _wrap(_linear_interp)(data)


def softimpute(data, rank_cap=30, max_iter=80, tol=1e-4):
    """BATCH / NON-CAUSAL. SoftImpute-ALS nuclear-norm low-rank matrix completion."""
    return _wrap(_softimpute)(data, rank_cap=rank_cap, max_iter=max_iter, tol=tol)


def trmf(data, k=6, lags=(1, 2, 3), n_iter=20, lam_w=1.0, lam_f=2.0, lam_th=1.0):
    """BATCH / NON-CAUSAL. Temporal Regularized Matrix Factorization (batch ALS)."""
    return _wrap(_trmf)(data, k=k, lags=lags, n_iter=n_iter, lam_w=lam_w,
                        lam_f=lam_f, lam_th=lam_th)


def mcnnm(data, rank_cap=30, fe_iter=8, max_iter=80, tol=1e-4):
    """BATCH / NON-CAUSAL. MC-NNM: two-way fixed effects + low-rank completion."""
    return _wrap(_mcnnm)(data, rank_cap=rank_cap, fe_iter=fe_iter,
                         max_iter=max_iter, tol=tol)


def knn(data, n_neighbors=5):
    """BATCH / NON-CAUSAL. k-nearest-neighbour imputation (sklearn KNNImputer)."""
    return _wrap(_knn)(data, n_neighbors=n_neighbors)


def mice(data, max_iter=10, random_state=0):
    """BATCH / NON-CAUSAL. Iterative / MICE-style imputation (sklearn IterativeImputer)."""
    return _wrap(_mice)(data, max_iter=max_iter, random_state=random_state)


# =========================================================================== #
# Registry: name -> (function, causal_flag, description)
# =========================================================================== #
METHODS = {
    # causal / point-in-time
    "locf":               (locf,               True,  "Last observation carried forward"),
    "mean_impute":        (mean_impute,        True,  "Expanding per-column mean"),
    "rolling_mean":       (rolling_mean,       True,  "Trailing-window mean"),
    "rolling_median":     (rolling_median,     True,  "Trailing-window median"),
    "ewma":               (ewma,               True,  "Exponentially-weighted moving average"),
    "drift":              (drift,              True,  "Linear drift extrapolation"),
    "kalman_local_level": (kalman_local_level, True,  "Local-level Kalman filter (SSM)"),
    "xsec_mean":          (xsec_mean,          True,  "Contemporaneous cross-sectional mean"),
    "online_trmf":        (online_trmf,        True,  "Online point-in-time TRMF"),
    "ewcov":              (ewcov,              True,  "EW-covariance Gaussian conditional mean"),
    "gaussian_copula":    (gaussian_copula,    True,  "Online Gaussian copula (gcimpute)"),
    # batch / non-causal
    "linear_interp":      (linear_interp,      False, "Linear interpolation (reads future)"),
    "softimpute":         (softimpute,         False, "SoftImpute-ALS low-rank completion"),
    "trmf":               (trmf,               False, "Temporal regularized MF (batch)"),
    "mcnnm":              (mcnnm,              False, "MC-NNM: two-way FE + low rank"),
    "knn":                (knn,                False, "k-nearest-neighbour (sklearn)"),
    "mice":               (mice,               False, "Iterative / MICE (sklearn)"),
}


def list_methods(causal=None):
    """List available baseline methods.

    Parameters
    ----------
    causal : bool or None
        ``True`` -> only causal/point-in-time methods, ``False`` -> only batch
        methods, ``None`` (default) -> all. Returns a list of method-name strings.
    """
    if causal is None:
        return list(METHODS)
    return [n for n, (_f, c, _d) in METHODS.items() if c is causal]


def is_causal(method):
    """Return True if ``method`` is causal/point-in-time, False if batch/non-causal."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {list_methods()}")
    return METHODS[method][1]


def impute(data, method="softimpute", **kw):
    """Dispatch to a classical baseline by name, container-native.

    Accepts numpy / pandas / polars (1D or 2D), returns the SAME container type with
    missing cells filled and observed cells preserved exactly. ``method`` is one of
    :func:`list_methods`; extra keyword args are forwarded to the chosen method
    (e.g. ``impute(df, method="rolling_mean", W=12)``).

    Note the causal/batch distinction (see :data:`METHODS` / :func:`is_causal`): batch
    methods (softimpute, trmf, mcnnm, knn, mice, linear_interp) may use future
    information and are NOT point-in-time, unlike ``cafe.impute`` and the causal
    baselines here.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {list_methods()}")
    fn = METHODS[method][0]
    return fn(data, **kw)
