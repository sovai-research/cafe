import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_method, summarize

import numpy as np


# --------------------------------------------------------------------------- #
# TRMF: Temporal Regularized Matrix Factorization (Yu, Rao, Dhillon 2016)
#
#   X (T,N) ~= F @ W.T     F:(T,k) temporal factors,  W:(N,k) loadings
#
# Objective (observed entries only):
#   sum_{(t,n) in Obs} (X_tn - f_t . w_n)^2
#     + lam_w * ||W||_F^2                         (ridge on loadings)
#     + lam_f * sum_t || f_t - sum_l theta_l f_{t-Ll} ||^2   (AR temporal prior)
#     + lam_th * ||theta||^2
#
# Fit by alternating mask-aware ridge updates:
#   - W rows: per-feature ridge over observed times.
#   - F rows: per-time ridge with AR coupling -> we solve the temporal-graph
#     regularized least squares row-by-row using the residual of the AR chain.
#   - theta: ridge regression of f_t on its lagged values (per latent dim).
# --------------------------------------------------------------------------- #


def _solve_W(X, M, F, lam_w, k):
    """Per-feature ridge: w_n = argmin sum_t M_tn (X_tn - f_t.w_n)^2 + lam_w||w_n||^2"""
    T, N = X.shape
    W = np.zeros((N, k))
    I = lam_w * np.eye(k)
    for n in range(N):
        m = M[:, n]
        if not m.any():
            continue
        Fn = F[m]                       # (obs, k)
        xn = X[m, n]                    # (obs,)
        A = Fn.T @ Fn + I
        b = Fn.T @ xn
        W[n] = np.linalg.solve(A, b)
    return W


def _solve_theta(F, lags, lam_th):
    """Ridge regression: for each latent dim, fit f_t ~ sum_l theta_l f_{t-Ll}.
    Returns theta (k, L). Shared design over t >= max_lag."""
    T, k = F.shape
    L = len(lags)
    maxlag = max(lags)
    theta = np.zeros((k, L))
    if T <= maxlag:
        return theta
    rows = np.arange(maxlag, T)
    for d in range(k):
        # design (len, L): columns are f_{t-lag}
        D = np.column_stack([F[rows - lag, d] for lag in lags])
        y = F[rows, d]
        A = D.T @ D + lam_th * np.eye(L)
        b = D.T @ y
        theta[d] = np.linalg.solve(A, b)
    return theta


def _solve_F(X, M, F, W, theta, lags, lam_f, k):
    """Update F via Gauss-Seidel sweep over time rows.

    For each t we minimize wrt f_t:
        sum_n M_tn (X_tn - f_t.w_n)^2
        + lam_f * || f_t - sum_l theta_l f_{t-Ll} ||^2     (this row as target)
        + lam_f * sum_{s where t is a lag of s} || f_s - sum_l theta_l f_{s-Ll} ||^2

    The AR term couples f_t to neighbours. We build the per-dim normal
    equations with diagonal AR curvature (theta acts per latent dim, so the
    AR Hessian is diagonal in k). Data term mixes dims via W.
    """
    T, N = X.shape
    L = len(lags)
    maxlag = max(lags)
    # Precompute, for each lag index l, theta_l per dim (k,)
    # AR diagonal contribution per time row:
    #   from own row (t>=maxlag): +lam_f on diag
    #   from each future row s=t+lag that uses t: +lam_f*theta_l^2 on diag
    # We'll assemble A (k,k) and b (k,) per t.
    for t in range(T):
        m = M[t]
        # data term
        if m.any():
            Wt = W[m]                   # (obs,k)
            A = Wt.T @ Wt
            b = Wt.T @ X[t, m]
        else:
            A = np.zeros((k, k))
            b = np.zeros(k)
        diag_add = np.zeros(k)
        # AR term where t is the target (t >= maxlag)
        if t >= maxlag:
            pred = np.zeros(k)
            for li, lag in enumerate(lags):
                pred += theta[:, li] * F[t - lag]
            diag_add += lam_f
            b += lam_f * pred
        # AR terms where t is a predictor for a future row s = t + lag
        for li, lag in enumerate(lags):
            s = t + lag
            if s < T and s >= maxlag:
                th = theta[:, li]       # (k,)
                # residual of row s excluding contribution of f_t
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


def _trmf_2d(X, k=6, lags=(1, 2, 3), n_iter=20, lam_w=1.0, lam_f=2.0,
             lam_th=1.0, seed=0):
    T, N = X.shape
    M = np.isfinite(X)
    # standardize per column on observed
    mu = np.where(M.any(0), np.nanmean(np.where(M, X, np.nan), 0), 0.0)
    sd = np.where(M.any(0), np.nanstd(np.where(M, X, np.nan), 0), 1.0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = np.where(M, (X - mu) / sd, 0.0)

    rng = np.random.default_rng(20260622 + seed)
    F = 0.1 * rng.standard_normal((T, k))
    W = 0.1 * rng.standard_normal((N, k))
    theta = np.zeros((k, len(lags)))
    lags = tuple(int(l) for l in lags)

    for it in range(n_iter):
        W = _solve_W(Xs, M, F, lam_w, k)
        theta = _solve_theta(F, lags, lam_th)
        F = _solve_F(Xs, M, F, W, theta, lags, lam_f, k)

    R = F @ W.T
    out = X.copy()
    miss = ~M
    fill = R * sd + mu
    out[miss] = fill[miss]
    return out


def impute(X, meta):
    if meta and "entity_ids" in meta:
        # Panel: reshape entity-major stacked rows into per-entity (T,F) blocks.
        # TRMF temporal factors live on the time axis; share loadings W across
        # entities, but each entity has its own temporal factor sequence. We
        # build a stacked F of shape (E*T, k) where the AR prior is applied
        # WITHIN each entity block (lags don't cross entity boundaries because
        # rows are entity-major contiguous and we reset AR at block starts).
        eids = np.asarray(meta["entity_ids"])
        tids = np.asarray(meta["time_ids"])
        E, T, Fdim = int(meta["E"]), int(meta["T"]), int(meta["F"])
        # The simplest faithful approach: treat the (E*T, F) matrix directly
        # but make AR respect entity blocks. Since rows are entity-major and
        # time-sorted within entity (idx = e*T + t), a global AR with small
        # lags would leak across one boundary per entity. We instead impute
        # each entity block independently with its own TRMF, sharing nothing.
        out = X.copy()
        M = np.isfinite(X)
        # Iteratively estimate additive entity + time fixed effects per feature
        # from observed cells (two-way means), demean, run TRMF on the residual
        # arranged as a (T, E*F) time-by-(entity,feature) matrix so the AR prior
        # operates on the shared time axis (matching the AR temporal factor B).
        ent_fe = np.zeros((E, Fdim))
        time_fe = np.zeros((T, Fdim))
        gmean = np.zeros(Fdim)
        for f in range(Fdim):
            obs = M[:, f]
            if obs.any():
                gmean[f] = X[obs, f].mean()
        for _ in range(6):
            # entity effects
            for f in range(Fdim):
                r = X[:, f] - gmean[f] - time_fe[tids, f]
                for e in range(E):
                    sel = (eids == e) & M[:, f]
                    ent_fe[e, f] = r[sel].mean() if sel.any() else 0.0
            # time effects
            for f in range(Fdim):
                r = X[:, f] - gmean[f] - ent_fe[eids, f]
                for t in range(T):
                    sel = (tids == t) & M[:, f]
                    time_fe[t, f] = r[sel].mean() if sel.any() else 0.0
        base = gmean[None, :] + ent_fe[eids] + time_fe[tids]   # (E*T, F)
        resid = X - base                                        # NaN where missing

        # Reshape residual to (T, E*F): col index = e*Fdim + f
        RT = np.full((T, E * Fdim), np.nan)
        col = eids[:, None] * Fdim + np.arange(Fdim)[None, :]    # (E*T, F)
        RT[tids[:, None].repeat(Fdim, 1), col] = resid
        filledRT = _trmf_2d(RT, k=min(8, E * Fdim), lags=(1, 2, 3),
                            n_iter=15, lam_w=1.0, lam_f=2.0, lam_th=1.0, seed=0)
        # map back
        resid_full = filledRT[tids[:, None].repeat(Fdim, 1), col]  # (E*T, F)
        recon = base + resid_full
        miss = ~M
        out[miss] = recon[miss]
        return out
    # 2D path
    return _trmf_2d(X, k=6, lags=(1, 2, 3), n_iter=20,
                    lam_w=1.0, lam_f=2.0, lam_th=1.0)


if __name__ == "__main__":
    rows = run_method("TRMF", impute)
    summarize(rows)
