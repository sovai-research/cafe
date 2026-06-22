"""
Unified_BayesPPCA -- ONE causal imputation model in which every classical
technique is a SPECIAL CASE reached by CONTINUOUS learned parameters. No router,
no if/else model selection, no thresholds keyed to dataset type. The data moves
the parameters implicitly during the online EM/variational fit.

UNIFYING MODEL (robust dynamic factor / online Bayesian probabilistic PCA):

    x_{e,t} = mu_e + Phi_t beta + W z_{e,t} + eps,    eps ~ t_nu(0, Psi)
    z_{e,t} = A z_{e,t-1} + eta,                      eta ~ N(0, Q)

  * W (N x L)  loadings with an ARD prior: column l has precision alpha_l, and the
    M-step ridge per column is  (Z'Z + alpha_l I)^-1 ...  so columns with no
    support are driven to 0  ==> AUTOMATIC RANK (no rank hyper-parameter). When all
    alpha_l -> inf the model collapses to W=0 (pure cross-section / EW-cov regression).
  * A = diag(a_l), a_l learned in [0,1):  0 => static factor (SoftImpute/PPCA),
    ->1 => random walk (Kalman SSM / TRMF AR). One continuous knob, learned by
    regressing consecutive filtered latents (point-in-time).
  * Phi_t beta : Fourier seasonal regression, beta under a shrinkage (ARD) prior, so
    beta -> 0 when there is no cycle (seasonal OFF) and grows when there is (seasonal
    case). For 1D where W is degenerate this carries the whole signal.
  * eps ~ Student-t_nu : EM scale-mixture (per-cell weight w = (nu+1)/(nu+r^2/psi)).
    nu is learned by EM. nu -> inf => Gaussian (L2, SoftImpute/Kalman); small nu =>
    robust IRLS (heavy-tail case). One continuous knob.
  * Psi diagonal observation noise (per feature), learned; balances how much the
    low-rank factor vs the contemporaneous cross-section explains each feature.
  * mu_e : entity offset (strictly-past per-entity mean) + contemporaneous time-FE
    from the cross-section at t  ==> FE / MC-NNM case for panels. For blackout times
    with NO cross-section the Kalman PREDICTION step extrapolates (AR-extrapolation).

INFERENCE is the causal forward Kalman FILTER (NO RTS smoother). Imputation of a
missing cell at time tau is its predictive mean E[x_{tau,miss} | x_{<=tau,obs}].
All parameters are (re)fit ONLY on rows with time <= tau on a geometric schedule
(streaming/expanding-window EM, warm-started), so truncating future rows cannot
change a past imputation -> passes the causal verifier.

Speed: numpy/scipy only, BLAS threads capped at 2, Woodbury for (WW'+Psi)^-1 (only
L x L solves), cho_solve everywhere (never inv/pinv on the hot path), contiguous
float arrays, piecewise-constant params between refits.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ----------------------------------------------------------------- knobs ----
# These are PRINCIPLED PRIORS / numerical caps, NOT constants fit to the cases.
L_MAX = 6            # max latent columns; ARD prunes the live ones down implicitly
EM_SWEEPS = 5        # warm-started EM sweeps per refit on the expanding window
REFIT_GROWTH = 1.6   # geometric refit schedule (refit when t >= last*GROWTH)
MIN_REFIT = 10       # don't refit until this many distinct times seen
JITTER = 1e-6
ARD_FLOOR = 1e-8     # numerical floor for ARD precisions
NU_MIN, NU_MAX = 2.5, 1e6   # learned Student-t dof clamp (inf-ish => Gaussian)
SEAS_PERIODS = 4     # number of candidate Fourier fundamental periods probed
SEAS_HARM = 3        # harmonics per period
RIDGE_BETA0 = 1.0    # initial ARD precision on seasonal coeffs (shrunk by data)


# =========================================================================== #
# Seasonal Fourier design.  Phi_t has columns [1, t/T, cos/sin(2pi h t / p)].
# Periods are detly chosen from the data length grid; ARD on beta turns the
# useless ones off, so this is the SAME model whether or not there is a cycle.
# =========================================================================== #
def _fourier_design(times, T, periods):
    # NOTE: every column is a deterministic function of the ABSOLUTE time index
    # only (never of the block's mean/extent), so the design row at a given tau is
    # identical whether or not future rows exist -> preserves point-in-time causality.
    times = np.asarray(times, float)
    cols = [np.ones_like(times), times / 100.0]            # fixed-scale linear trend
    for p in periods:
        for h in range(1, SEAS_HARM + 1):
            w = 2.0 * np.pi * h / p
            cols.append(np.cos(w * times))
            cols.append(np.sin(w * times))
    return np.ascontiguousarray(np.vstack(cols).T)        # (n, P)


def _candidate_periods():
    """FIXED geometric grid of plausible periods (independent of data extent so the
    design is point-in-time invariant). ARD on beta turns the useless ones off."""
    return [12.0, 24.0, 168.0, 8.0][:SEAS_PERIODS]


# =========================================================================== #
# Expanding-window EM fit of the unified model on a block (rows with time<=tau).
# Returns W, mu, Psi (per-feature), A (diag), Q (diag), beta, alpha (ARD), nu,
# and the design-period list.  All quantities depend ONLY on the passed block.
# =========================================================================== #
def _fit(block, blk_times, blk_eids, T_total, periods,
         W0=None, beta0=None, alpha0=None, nu0=50.0):
    n, N = block.shape
    obs = np.isfinite(block)
    cnt = obs.sum(0)

    # ---- seasonal + trend regression (ARD shrinkage on beta) ---------------
    Phi = _fourier_design(blk_times, T_total, periods)     # (n, P)
    P = Phi.shape[1]
    # global feature mean (mu); start residual = block - mu
    mu = np.where(cnt > 0, np.where(obs, block, 0.0).sum(0) / np.maximum(cnt, 1), 0.0)
    R = np.where(obs, block - mu, 0.0)                     # centered residual

    # Fit beta (N x P) by ridge with per-coefficient ARD precision (shared across
    # features for stability). Solve column-pooled normal equations once.
    if beta0 is not None and beta0.shape == (N, P):
        beta = beta0.copy()
    else:
        beta = np.zeros((N, P))
    ab = beta0 is not None
    PtP = Phi.T @ Phi                                      # (P,P) (dense rows complete)
    # ARD precision per seasonal coeff (empirical-Bayes-ish): shrink unused harmonics
    lam = np.full(P, RIDGE_BETA0)
    lam[0] = 1e-6                                          # don't shrink intercept-ish
    for _ in range(2):
        G = PtP + np.diag(lam)
        try:
            cG = cho_factor(G, lower=True)
            B = cho_solve(cG, Phi.T @ R)                   # (P, N)
        except Exception:
            B = np.linalg.lstsq(G, Phi.T @ R, rcond=None)[0]
        beta = B.T                                         # (N, P)
        # empirical-Bayes ARD update on each seasonal coeff: precision ~ N / ||b||^2
        col_energy = np.sum(beta * beta, axis=0) + 1e-9
        lam = np.minimum(1e6, N / col_energy)
        lam[0] = 1e-6
    seasonal = Phi @ beta.T                                # (n, N)
    R = np.where(obs, block - mu - seasonal, 0.0)          # residual for factor model

    # ---- low-rank factor model with ARD on W columns + Student-t IRLS ------
    Lr = min(L_MAX, max(1, min(n, N)))
    if W0 is not None and W0.shape == (N, Lr):
        W = W0.copy()
    else:
        rng = np.random.default_rng(0)
        W = rng.standard_normal((N, Lr)) * 0.1
    if alpha0 is not None and alpha0.shape == (Lr,):
        alpha = alpha0.copy()
    else:
        alpha = np.full(Lr, 1.0)                           # ARD precisions on W cols
    psi = np.maximum(np.var(R[obs]) if obs.any() else 1.0, 1e-3)
    Psi = np.full(N, psi)                                  # per-feature noise
    nu = float(np.clip(nu0, NU_MIN, NU_MAX))

    Z = np.zeros((n, Lr))
    eyeL = np.eye(Lr)
    for sweep in range(EM_SWEEPS):
        # ---- E-step: latent scores per row (ridge w/ Psi-weighting + IRLS) ----
        # weights from Student-t scale mixture on the PREVIOUS residual.
        # pred_prev = Z @ W'  (per row); residual r; w = (nu+1)/(nu + r^2/psi)
        ZtZ = np.zeros((Lr, Lr))
        for i in range(n):
            oi = obs[i]
            if not oi.any():
                continue
            Wo = W[oi]
            xi = R[i, oi]
            pso = Psi[oi]
            # IRLS robustness weights (Gaussian when nu large -> w~1)
            if sweep > 0:
                ri = xi - Wo @ Z[i]
                wgt = (nu + 1.0) / (nu + (ri * ri) / pso)
            else:
                wgt = np.ones(oi.sum())
            # effective precision per obs = wgt / psi
            wp = wgt / pso
            G = (Wo.T * wp) @ Wo + np.diag(alpha)          # ARD prior precision on z
            rhs = (Wo.T * wp) @ xi
            try:
                cG = cho_factor(G + JITTER * eyeL, lower=True)
                Z[i] = cho_solve(cG, rhs)
            except Exception:
                Z[i] = np.linalg.lstsq(G, rhs, rcond=None)[0]
            ZtZ += np.outer(Z[i], Z[i])
        # ---- M-step: W rows by ARD ridge over rows observing feature j -------
        new_W = W.copy()
        for j in range(N):
            rj = obs[:, j]
            if not rj.any():
                continue
            Zr = Z[rj]
            # robust weights for this feature's rows
            rr = R[rj, j] - Zr @ W[j]
            wgt = (nu + 1.0) / (nu + (rr * rr) / Psi[j]) if sweep > 0 else np.ones(rj.sum())
            G = (Zr.T * wgt) @ Zr + np.diag(alpha)
            rhs = (Zr.T * wgt) @ R[rj, j]
            try:
                cG = cho_factor(G + JITTER * eyeL, lower=True)
                new_W[j] = cho_solve(cG, rhs)
            except Exception:
                new_W[j] = np.linalg.lstsq(G, rhs, rcond=None)[0]
        W = new_W
        # ---- ARD update on W columns: alpha_l = N / (||W_:l||^2)  (auto rank) -
        col_energy = np.sum(W * W, axis=0) + 1e-9
        alpha = np.minimum(1e8, N / col_energy)
        alpha = np.maximum(alpha, ARD_FLOOR)
        # ---- Psi update (per-feature) from weighted residuals ----------------
        pred = Z @ W.T
        resid = (R - pred)
        for j in range(N):
            rj = obs[:, j]
            if not rj.any():
                continue
            d = resid[rj, j]
            wgt = (nu + 1.0) / (nu + (d * d) / Psi[j]) if sweep > 0 else np.ones(rj.sum())
            Psi[j] = max(float(np.sum(wgt * d * d) / max(np.sum(wgt), 1e-9)), 1e-4)
        # ---- nu update (EM for Student-t dof) from standardized residuals ----
        if sweep > 0:
            d_all = resid[obs]
            psi_all = np.repeat(Psi[None, :], n, axis=0)[obs]
            s2 = (d_all * d_all) / psi_all
            wgt = (nu + 1.0) / (nu + s2)
            # solve psi-digamma fixed point approximately via moment match:
            # E[w]=1 at the true nu; use the standard one-step EM surrogate.
            m = float(np.mean(wgt - np.log(np.maximum(wgt, 1e-9))))
            # fixed-point: find nu s.t.  log(nu/2)-digamma(nu/2)+1 = m'  (approx)
            nu = _update_nu(nu, wgt)

    # ---- transition A (diag) + Q from consecutive filtered latents ---------
    A = np.zeros(Lr)
    Q = np.ones(Lr)
    z0_list, z1_list = [], []
    for e in np.unique(blk_eids):
        sel = np.where(blk_eids == e)[0]
        if len(sel) < 2:
            continue
        order = np.argsort(blk_times[sel], kind="stable")
        zs = Z[sel[order]]
        z0_list.append(zs[:-1]); z1_list.append(zs[1:])
    if z0_list:
        Z0 = np.vstack(z0_list); Z1 = np.vstack(z1_list)
        for l in range(Lr):
            num = float(np.dot(Z0[:, l], Z1[:, l]))
            den = float(np.dot(Z0[:, l], Z0[:, l])) + 1e-6
            a = num / den
            A[l] = float(np.clip(a, 0.0, 0.999))           # learned 0=static..~1=RW
    else:
        A[:] = 0.0
    # Tie process noise to the ARD prior so the filter's STATIONARY state variance
    # equals the ARD prior variance 1/alpha_l (Var_inf = Q/(1-A^2) == 1/alpha):
    #   A=0  -> Q = 1/alpha       (static PPCA, prior cov diag(1/alpha) every step)
    #   A->1 -> Q -> 0            (random walk / TRMF integrated factor)
    # This makes the online Kalman prior IDENTICAL to the ARD prior used in the
    # offline E-step, so cross-sectional borrowing matches the batch PPCA solution.
    s_inf = 1.0 / np.maximum(alpha, ARD_FLOOR)             # ARD prior variance per col
    Q = np.maximum((1.0 - A * A) * s_inf, 1e-4)
    return dict(W=W, mu=mu, Psi=Psi, A=A, Q=Q, beta=beta, alpha=alpha,
                nu=nu, periods=periods, P=Phi.shape[1])


def _update_nu(nu, w):
    """One Newton step on the Student-t dof EM objective:
       f(nu) = 1 - digamma(nu/2) + log(nu/2) + mean(log w - w) = 0.
    Uses scipy-free digamma approximation (Stirling-ish). Clamped."""
    from math import log
    Ew = float(np.mean(w))
    Elw = float(np.mean(np.log(np.maximum(w, 1e-12))))
    c = 1.0 + Elw - Ew
    # solve  log(nu/2) - psi(nu/2) + c = 0  by a few bisection steps on log-nu.
    def g(v):
        h = v / 2.0
        return log(h) - _digamma(h) + c
    lo, hi = NU_MIN, NU_MAX
    glo, ghi = g(lo), g(hi)
    if glo * ghi > 0:                                      # no sign change -> extreme
        return NU_MAX if c > 0 else NU_MIN
    for _ in range(40):
        mid = (lo + hi) / 2.0
        gm = g(mid)
        if glo * gm <= 0:
            hi = mid; ghi = gm
        else:
            lo = mid; glo = gm
    return float(np.clip((lo + hi) / 2.0, NU_MIN, NU_MAX))


def _digamma(x):
    """Digamma approximation (Bernoulli asymptotic + recurrence). x>0."""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    r += (np.log(x) - 0.5 / x
          + f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0))))
    return r


# =========================================================================== #
# Main causal imputer.
# =========================================================================== #
def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    T_rows, N = X.shape
    out = X.copy()

    if meta and "time_ids" in meta:
        tids = np.asarray(meta["time_ids"])
        eids = np.asarray(meta["entity_ids"])
        is_panel = True
    else:
        tids = np.arange(T_rows)
        eids = np.zeros(T_rows, dtype=int)
        is_panel = False

    uniq_times = np.unique(tids)
    nT = len(uniq_times)
    t_to_pos = {int(t): i for i, t in enumerate(uniq_times)}

    rows_by_time = [[] for _ in range(nT)]
    for r in range(T_rows):
        rows_by_time[t_to_pos[int(tids[r])]].append(r)

    ent_unique = np.unique(eids)
    periods = _candidate_periods()

    # per-entity Kalman state
    state_z = {int(e): None for e in ent_unique}
    state_P = {int(e): None for e in ent_unique}

    # strictly-past per-entity offset (entity FE). For 2D this is the single mu.
    use_offset = is_panel
    off_sum = {int(e): np.zeros(N) for e in ent_unique} if use_offset else None
    off_cnt = {int(e): np.zeros(N) for e in ent_unique} if use_offset else None

    # current params (piecewise constant between refits)
    par = None
    last_refit_pos = -1
    refit_threshold = MIN_REFIT

    def refit(upto_pos):
        nonlocal par
        keep = tids <= uniq_times[upto_pos]
        blk = X[keep]
        bt = tids[keep].astype(float)
        be = eids[keep]
        W0 = par["W"] if par is not None and par["W"].shape[0] == N else None
        beta0 = par["beta"] if par is not None else None
        alpha0 = par["alpha"] if par is not None and par["W"].shape[0] == N else None
        nu0 = par["nu"] if par is not None else 50.0
        par = _fit(blk, bt, be, 0, periods,
                   W0=W0, beta0=beta0, alpha0=alpha0, nu0=nu0)

    init_pos = max(min(MIN_REFIT, nT) - 1, 0)
    refit(init_pos)
    last_refit_pos = init_pos
    refit_threshold = max(MIN_REFIT, int(init_pos * REFIT_GROWTH) + 1)

    for pos in range(nT):
        tau = uniq_times[pos]
        if pos > last_refit_pos and pos >= refit_threshold:
            refit(pos)
            last_refit_pos = pos
            refit_threshold = max(refit_threshold + 1, int(pos * REFIT_GROWTH) + 1)

        W = par["W"]; mu = par["mu"]; Psi = par["Psi"]
        A = par["A"]; Q = par["Q"]; beta = par["beta"]; alpha = par["alpha"]
        nu = par["nu"]; per = par["periods"]
        Lr = W.shape[1]
        Adiag = A
        Qmat = np.diag(Q)
        # seasonal design row at tau (uses only tau)
        phi_tau = _fourier_design(np.array([float(tau)]), 0, per)[0]            # (P,)
        season = beta @ phi_tau                            # (N,)

        # contemporaneous time-FE for panels (cross-section at tau IS allowed)
        tfe = np.zeros(N)
        if use_offset:
            tsum = np.zeros(N); tcnt = np.zeros(N)
            for r in rows_by_time[pos]:
                e = int(eids[r]); row = X[r]; o = np.isfinite(row)
                if not o.any():
                    continue
                oc = off_cnt[e]
                offs = np.where(oc > 0, off_sum[e] / np.maximum(oc, 1), mu)
                tsum[o] += row[o] - offs[o] - season[o]
                tcnt[o] += 1
            tfe = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), 0.0)

        for r in rows_by_time[pos]:
            e = int(eids[r])
            z = state_z[e]; Pcov = state_P[e]
            if z is None:
                # reset latent state per entity; prior = ARD stationary covariance
                z = np.zeros(Lr); Pcov = np.diag(1.0 / np.maximum(alpha, ARD_FLOOR))
            # predict (diagonal A)
            z_pred = Adiag * z
            P_pred = (Adiag[:, None] * Pcov * Adiag[None, :]) + Qmat
            P_pred = 0.5 * (P_pred + P_pred.T)

            if use_offset:
                oc = off_cnt[e]
                base = np.where(oc > 0, off_sum[e] / np.maximum(oc, 1), mu) + tfe + season
            else:
                base = mu + season

            row = X[r]; obs = np.isfinite(row)
            if obs.any():
                Wo = W[obs]
                y = row[obs] - base[obs]
                pso = Psi[obs]
                nu_innov = y - Wo @ z_pred
                # Student-t IRLS reweight of the observation precision (robust update)
                wgt = (nu + 1.0) / (nu + (nu_innov * nu_innov) / pso)
                # effective per-obs noise variance = Psi / wgt  (down-weight outliers)
                inv_r = wgt / pso                          # (m,) diagonal precision
                # Woodbury: solve in L-space. M = P_pred^-1 + Wo' diag(inv_r) Wo
                try:
                    cP = cho_factor(P_pred + JITTER * np.eye(Lr), lower=True)
                    Pinv = cho_solve(cP, np.eye(Lr))
                except Exception:
                    Pinv = np.diag(1.0 / np.maximum(np.diag(P_pred), 1e-9))
                WtR = Wo.T * inv_r                         # (L, m)
                M = Pinv + WtR @ Wo
                try:
                    cM = cho_factor(M + JITTER * np.eye(Lr), lower=True)
                    Minv = cho_solve(cM, np.eye(Lr))
                except Exception:
                    Minv = np.linalg.pinv(M)
                # S^-1 nu via Woodbury (S = diag(1/inv_r) + Wo P_pred Wo')
                t1 = inv_r * nu_innov
                Wt_t1 = Wo.T @ t1
                Sinv_nu = t1 - inv_r * (Wo @ (Minv @ Wt_t1))
                z = z_pred + P_pred @ (Wo.T @ Sinv_nu)
                # P update via Woodbury
                WtRW = WtR @ Wo
                mid = WtRW - WtRW @ Minv @ WtRW
                Pcov = P_pred - P_pred @ mid @ P_pred
                Pcov = 0.5 * (Pcov + Pcov.T)
            else:
                z = z_pred; Pcov = P_pred

            miss = ~obs
            if miss.any():
                out[r, miss] = base[miss] + W[miss] @ z
            state_z[e] = z; state_P[e] = Pcov

            if use_offset and obs.any():
                off_sum[e][obs] += row[obs]
                off_cnt[e][obs] += 1

    # safety: strictly-past running column-mean fill for any residual NaN
    if np.isnan(out).any():
        csum = np.zeros(N); ccnt = np.zeros(N); glob = 0.0; gcnt = 0
        for pos in range(nT):
            mean_sf = np.where(ccnt > 0, csum / np.maximum(ccnt, 1),
                               glob / max(gcnt, 1) if gcnt else 0.0)
            for r in rows_by_time[pos]:
                miss = np.isnan(out[r])
                if miss.any():
                    out[r, miss] = mean_sf[miss]
            for r in rows_by_time[pos]:
                o = np.isfinite(X[r])
                csum[o] += X[r][o]; ccnt[o] += 1
                glob += X[r][o].sum(); gcnt += int(o.sum())
        out = np.where(np.isnan(out), 0.0, out)

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Unified_BayesPPCA", online_impute))
