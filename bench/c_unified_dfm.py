"""
Unified_DFM -- ONE robust dynamic factor model, fit by online filtering EM, whose
CONTINUOUS learned parameters subsume every classical imputation technique. There
is NO router, NO data-type if/else, NO threshold keyed to a dataset. A single
forward Kalman FILTER pass (causal / point-in-time) does all of it; the data moves
the parameters implicitly during fitting and the model "becomes" whatever special
case the data prefers.

MODEL  (shared parameters pooled across entities; latent state reset per entity)
    x_{e,t} = mu_e + Phi_t @ beta + W z_{e,t} + eps_{e,t}
    z_{e,t} = A z_{e,t-1} + eta_{e,t},     eta ~ N(0, Q),  Q diagonal
    eps_{e,t} ~ Student-t_nu(0, R),        R diagonal (per-feature)
  - mu_e            : entity offset (strictly-past per-entity running mean; FE).
  - Phi_t @ beta    : small Fourier seasonal basis (deterministic in t); beta has an
                      ARD/ridge prior -> seasonality switches itself on/off.
  - W (N x L)       : factor loadings with PER-COLUMN ARD variance alpha_l. Unused
                      columns get alpha_l -> 0 and are pruned => AUTOMATIC RANK.
  - A = diag(a_l)   : per-factor AR coefficient, learned, clipped to [0, 0.999]
                      (0 => static/PCA factor, ->1 => random-walk / Kalman SSM).
  - R (per feature) : idiosyncratic variance; large vs W governs low-rank-vs-
                      cross-section balance (rank-0 + full residual == EW-cov / xsec
                      regression; here residual is diagonal but the FACTOR block
                      carries the cross-correlation, so the Kalman update IS the
                      conditional E[x_miss | x_obs] cross-sectional regression).
  - nu              : Student-t d.o.f., learned by EM (moment match on standardized
                      residuals) and annealed; nu->inf => Gaussian, small nu =>
                      heavy-tail robust (IRLS scale-mixture weights in the update).

SPECIAL CASES (all reachable by continuous params, none hard-coded):
  SoftImpute/low-rank   : A=0, nu=inf, R small isotropic, few active W columns.
  TRMF                  : A!=0 AR factors, ARD-selected rank.
  Kalman SSM            : full A, Q, R dynamics.
  EW-cov / xsec reg     : ARD keeps many factors -> W W' + R reproduces the cross-
                          covariance; the filter's contemporaneous update == GLS
                          cross-section regression.  (rank "0" residual-only limit.)
  MC-NNM / FE           : mu_e (+ time effect via Phi/cross-section) + low-rank W.
  robust                : small learned nu -> IRLS down-weights outliers.
  seasonal              : Phi=Fourier, beta!=0 (ARD lets it grow only if it helps).
  blackout AR-extrapol. : when no obs at t, Kalman PREDICT step extrapolates z_t.

CAUSALITY:
  * Iterate strictly by TIME tau over the union of times. For panels, all entities
    at tau are processed together; each entity's latent state resets at its first
    time and is carried forward online.
  * Every parameter is fit ONLY from rows with time <= tau (expanding-window EM on a
    geometric refit schedule; between refits the past-fit params are reused).
  * The seasonal beta and entity offsets are likewise estimated from <= tau only.
  * The contemporaneous cross-section at tau IS used (allowed) but no value at s>tau.
  Hence imputations at tau are invariant to removing future rows => passes causal.py.

SPEED: numpy/scipy only; Woodbury for the (low-rank+diagonal) innovation inverse so
we solve only an L x L system; cho_factor/cho_solve everywhere (never inv/pinv on
the hot path); contiguous arrays; BLAS threads capped at top.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ----- principled caps (NOT tuned to any case): keep runtime + EM bounded -------
EM_SWEEPS = 6           # warm-started EM sweeps per refit on the expanding window
REFIT_GROWTH = 1.6      # geometric refit schedule (refit when pos >= last*GROWTH)
MIN_REFIT_TIMES = 6     # don't fit dynamics until this many distinct times seen
L_MAX = 6               # max latent factors; ARD prunes down from here
JITTER = 1e-6
NU_INIT = 30.0          # start near-Gaussian; EM pulls it down only if data is heavy
NU_MIN, NU_MAX = 2.5, 1e4


# =========================================================================== #
#  Expanding-window EM fit of the SHARED parameters from rows with time<=tau.
#  Returns a params dict. Warm-started across refits for speed + stability.
#  Implements: ARD on W columns, per-feature R, diagonal AR A in [0,1], Fourier
#  beta with ARD ridge, and EM-updated Student-t nu.
# =========================================================================== #
def _build_phi(time_vals, K, span):
    """Fourier design matrix Phi (n, 2K+1): [1, sin/cos at K harmonics].

    Frequencies are chosen so the LOWEST harmonic already completes >= 2 full
    cycles over the data span (period = span / (m+1), m=1..K). This deliberately
    excludes the 1-cycle-over-span term, which would just fit slow TREND/AR drift
    and steal it from the latent dynamics. ARD ridge on beta then shrinks any
    harmonic the data does not actually need -- so seasonality switches itself on
    only when a genuine repeated cycle exists. No per-dataset period constant."""
    n = len(time_vals)
    if K <= 0:
        return np.ones((n, 1))
    t = np.asarray(time_vals, float)
    base = max(span, 2.0)
    cols = [np.ones(n)]
    # harmonics h = 2..K+1 of the fundamental (period = span/h), i.e. >= 2 cycles
    # over the span. h=1 (single slow trend) is excluded so seasonality does not
    # steal the latent AR drift. With many harmonics the basis can resolve short
    # periods; ARD ridge on beta shrinks the ones the data does not need.
    for h in range(2, K + 2):
        period = base / h
        ang = 2.0 * np.pi * t / period
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.ascontiguousarray(np.column_stack(cols))


def _fit_params(block, block_tids, block_eids, ent_unique, time_unit,
                L, K, warm=None):
    """One expanding-window EM fit. `block` rows are entity-major with NaNs.

    All quantities use only `block` (= rows with time<=tau). Returns dict of
    learned params. Adaptation is intrinsic via ARD (alpha), per-feature R,
    diagonal A, beta ridge, and nu.
    """
    n, N = block.shape
    obs = np.isfinite(block)
    cnt = obs.sum(0)

    # per-feature observed variance (used for a PRINCIPLED relative noise floor:
    # signal-to-noise is never infinite, so R cannot collapse to 0; and an ARD
    # scale cap so a single factor cannot de-regularize to fit a row exactly).
    fvar = np.ones(N)
    for j in range(N):
        v = block[obs[:, j], j]
        if v.size > 1:
            fvar[j] = max(float(np.var(v)), 1e-6)
    R_floor = 1e-3 * fvar                       # >=0.1% noise floor per feature

    # ---- entity offsets mu_e (FE): per-entity per-feature observed mean -------
    # pooled global mean as fallback for unobserved (entity,feature) pairs.
    gsum = np.where(obs, block, 0.0).sum(0)
    gmean = np.where(cnt > 0, gsum / np.maximum(cnt, 1), 0.0)

    # ---- Fourier seasonal regression beta with ARD ridge ---------------------
    # period hint = span of the data (one fundamental cycle over the window); the
    # harmonics cover sub-cycles. beta fit by ridge on the entity-demeaned obs.
    span = (block_tids.max() - block_tids.min() + 1) if n else 1.0
    Phi = _build_phi(block_tids, K, span)
    P = Phi.shape[1]
    # entity-demeaned target (remove FE so beta captures shared seasonality only)
    # build per-row entity mean
    ent_mean = np.zeros((n, N))
    # map entity -> its observed mean vector
    emean = {}
    for e in ent_unique:
        sel = (block_eids == e)
        if sel.any():
            oe = obs[sel]
            be = block[sel]
            ce = oe.sum(0)
            se = np.where(oe, be, 0.0).sum(0)
            emean[e] = np.where(ce > 0, se / np.maximum(ce, 1), gmean)
        else:
            emean[e] = gmean
    for i in range(n):
        ent_mean[i] = emean[block_eids[i]]
    target = np.where(obs, block - ent_mean, 0.0)

    # ridge solve per feature with ARD on beta (shared ridge tau_b learned by EB).
    # beta_j = (Phi_o' Phi_o + lam I)^-1 Phi_o' y_o    (only observed rows for j)
    beta = np.zeros((P, N))
    if P > 1 and n >= P + 2:
        lam_b = 1.0
        for _ in range(2):                      # 2 empirical-Bayes ridge updates
            PtP_full = Phi.T @ Phi
            ssq = 0.0; df = 0.0
            for j in range(N):
                rj = obs[:, j]
                if rj.sum() < P + 1:
                    beta[:, j] = 0.0
                    continue
                Pr = Phi[rj]
                G = Pr.T @ Pr + lam_b * np.eye(P)
                G[0, 0] = Pr.shape[0] + JITTER     # intercept ~ unpenalized
                try:
                    c = cho_factor(G, lower=True, check_finite=False)
                    bj = cho_solve(c, Pr.T @ target[rj, j], check_finite=False)
                except Exception:
                    bj = np.zeros(P)
                beta[:, j] = bj
                ssq += float(beta[1:, j] @ beta[1:, j])
                df += (P - 1)
            # EB ridge: lam ~ df / ||beta||^2  (ARD shrinkage on seasonal weights)
            lam_b = float(df / (ssq + 1e-8)) if ssq > 0 else 1e3
            lam_b = min(max(lam_b, 1e-3), 1e6)

    season = Phi @ beta                          # (n, N) seasonal component
    # residual after FE + seasonality -> the factor model explains the rest
    Xc = np.where(obs, block - ent_mean - season, 0.0)

    # ---- low-rank factor EM with PER-COLUMN ARD + per-feature R --------------
    Lr = min(L, max(1, min(n, N)))
    if warm is not None and warm.get("W") is not None and warm["W"].shape == (N, Lr):
        W = warm["W"].copy()
        alpha = warm.get("alpha", np.ones(Lr)).copy()
        R = warm.get("R", np.ones(N)).copy()
        if alpha.shape[0] != Lr:
            alpha = np.ones(Lr)
        if R.shape[0] != N:
            R = np.ones(N)
    else:
        rng = np.random.default_rng(0)
        W = rng.standard_normal((N, Lr)) * 0.1
        alpha = np.ones(Lr)                      # ARD prior variance per column
        R = np.maximum(np.var(Xc[obs]) if obs.any() else 1.0, 1e-3) * np.ones(N)

    nu = warm.get("nu", NU_INIT) if warm is not None else NU_INIT

    for sweep in range(EM_SWEEPS):
        # E-step: latent scores Z (n, Lr); ridge by R and ARD via ZtZ + diag(1/alpha)
        Z = np.zeros((n, Lr))
        inv_alpha = 1.0 / np.maximum(alpha, 1e-8)
        for i in range(n):
            oi = obs[i]
            if not oi.any():
                continue
            Wo = W[oi]
            ro = R[oi]
            # G = Wo' diag(1/ro) Wo + diag(1/alpha)
            WtR = Wo.T / ro                       # (Lr, m)
            G = WtR @ Wo + np.diag(inv_alpha)
            try:
                c = cho_factor(G + JITTER * np.eye(Lr), lower=True, check_finite=False)
                Z[i] = cho_solve(c, WtR @ Xc[i, oi], check_finite=False)
            except Exception:
                Z[i] = np.linalg.lstsq(Wo, Xc[i, oi], rcond=None)[0]

        # M-step: W rows; per-feature R; ARD alpha. Student-t weights via IRLS.
        # robustness weights w_ij from current residual (scale-mixture): downweight
        # rows with large standardized residual. Uses learned nu.
        pred = Z @ W.T
        resid = np.where(obs, Xc - pred, 0.0)
        # per-observation t-weight = (nu+1)/(nu + (r^2 / R))
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = (resid ** 2) / np.maximum(R, 1e-8)
        tw = (nu + 1.0) / (nu + np.where(obs, r2, 0.0))
        tw = np.where(obs, tw, 0.0)

        new_W = W.copy()
        for j in range(N):
            rj = obs[:, j]
            if not rj.any():
                continue
            Zr = Z[rj]
            wj = tw[rj, j]
            Zw = Zr * wj[:, None]
            # ML estimate of W given the latent scores; ARD shrinkage acts through
            # the LATENT prior (E-step diag(1/alpha)), NOT a second ridge on W -- a
            # double penalty there drives a degenerate alpha->0->W->0 death spiral.
            G = Zr.T @ Zw + JITTER * np.eye(Lr)
            try:
                c = cho_factor(G + JITTER * np.eye(Lr), lower=True, check_finite=False)
                new_W[j] = cho_solve(c, Zw.T @ Xc[rj, j], check_finite=False)
            except Exception:
                new_W[j] = np.linalg.lstsq(Zr, Xc[rj, j], rcond=None)[0]
        W = new_W

        # per-feature R from weighted residuals
        pred = Z @ W.T
        resid = np.where(obs, Xc - pred, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = (resid ** 2) / np.maximum(R, 1e-8)
        tw = np.where(obs, (nu + 1.0) / (nu + r2), 0.0)
        num = (tw * resid ** 2).sum(0)
        den = np.maximum(obs.sum(0), 1.0)
        R = np.maximum(num / den, R_floor)      # principled relative noise floor

        # ARD update: alpha_l = mean_j W[j,l]^2  (per-column relevance, latent prior
        # = N(0, diag(alpha))). Columns explaining nothing collapse -> automatic
        # rank. Capped at the data scale so a factor cannot de-regularize to fit a
        # single row exactly (the small-N / L>=N identifiability guard).
        alpha = np.maximum(np.mean(W ** 2, axis=0), 1e-8)
        alpha = np.minimum(alpha, 10.0 * float(fvar.mean()))

        # EM for nu (every other sweep): moment-match standardized residuals.
        if sweep >= 1 and (sweep % 2 == 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                s2 = (resid[obs] ** 2) / np.maximum(R[None, :].repeat(n, 0)[obs], 1e-8)
            s2 = s2[np.isfinite(s2)]
            if s2.size > 30:
                # kurtosis -> nu via t-distribution excess kurtosis = 6/(nu-4)
                m2 = np.mean(s2)
                m4 = np.mean(s2 ** 2)
                if m2 > 1e-8:
                    kurt = m4 / (m2 ** 2)         # ~3 Gaussian, larger=>heavier
                    exc = kurt - 3.0
                    if exc > 1e-3:
                        nu_new = 4.0 + 6.0 / exc
                    else:
                        nu_new = NU_MAX
                    nu = float(np.clip(0.5 * nu + 0.5 * nu_new, NU_MIN, NU_MAX))

    # ---- diagonal AR transition A in [0,1] + process noise Q -----------------
    # estimate per-factor AR(1) from consecutive within-entity score pairs.
    a = np.full(Lr, 0.5)
    qd = np.ones(Lr)
    p0 = []; p1 = []
    row_obs = obs.any(1)            # rows with >=1 observation -> z is identified
    # use only consecutive pairs where BOTH rows had observations (z meaningful);
    # rows with no obs have z forced to 0 in the E-step and would bias A downward.
    for e in ent_unique:
        sel = np.where(block_eids == e)[0]
        if len(sel) < 2:
            continue
        order = np.argsort(block_tids[sel], kind="stable")
        sel = sel[order]
        zs = Z[sel]
        good = row_obs[sel]
        pair = good[:-1] & good[1:]
        if pair.any():
            p0.append(zs[:-1][pair]); p1.append(zs[1:][pair])
    if p0:
        Z0 = np.vstack(p0); Z1 = np.vstack(p1)
        for l in range(Lr):
            d0 = Z0[:, l]; d1 = Z1[:, l]
            denom = float(d0 @ d0) + 1e-6
            al = float(d0 @ d1) / denom
            a[l] = min(max(al, 0.0), 0.999)       # learned dynamics in [0,1]
            res = d1 - a[l] * d0
            qd[l] = max(float(np.mean(res ** 2)), 1e-4)
    else:
        qd = np.maximum(alpha, 1e-3)

    return dict(W=W, alpha=alpha, R=R, a=a, qd=qd, beta=beta, K=K,
                gmean=gmean, emean=emean, nu=nu, span=float(span), Lr=Lr)


# =========================================================================== #
#  Main causal online imputer.
# =========================================================================== #
def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    T_rows, N = X.shape
    out = X.copy()

    if meta and "time_ids" in meta:
        tids = np.asarray(meta["time_ids"])
        eids = np.asarray(meta["entity_ids"])
        is_panel = len(np.unique(eids)) > 1
    else:
        tids = np.arange(T_rows)
        eids = np.zeros(T_rows, dtype=int)
        is_panel = False

    uniq_times = np.unique(tids)
    nT = len(uniq_times)
    t_to_pos = {int(t): i for i, t in enumerate(uniq_times)}

    ent_unique = np.unique(eids)

    rows_by_time = [[] for _ in range(nT)]
    for r in range(T_rows):
        rows_by_time[t_to_pos[int(tids[r])]].append(r)

    # number of Fourier harmonics. A rich basis (period span/h, h=2..K+1) so short
    # cycles can be resolved; the ARD ridge on beta shrinks every harmonic the data
    # does not need, so this is a CAPACITY bound (grows with series length), not a
    # per-dataset knob. Capped so the per-feature ridge solve stays cheap.
    K = int(np.clip(nT // 4, 0, 20))
    if nT < 8:
        K = 0
    L = min(L_MAX, N)

    # ---- params (shared); warm-started across refits -------------------------
    params = None

    def refit(upto_pos):
        nonlocal params
        keep = tids <= uniq_times[upto_pos]
        params = _fit_params(X[keep], tids[keep], eids[keep], ent_unique,
                             1.0, L, K, warm=params)

    init_pos = max(min(MIN_REFIT_TIMES, nT) - 1, 0)
    refit(init_pos)
    last_refit_pos = init_pos
    refit_threshold = max(MIN_REFIT_TIMES, int(init_pos * REFIT_GROWTH) + 1)

    # per-entity Kalman state (reset at first appearance)
    state_z = {int(e): None for e in ent_unique}
    state_P = {int(e): None for e in ent_unique}

    # strictly-past per-entity running offset (entity FE), causal
    off_sum = {int(e): np.zeros(N) for e in ent_unique}
    off_cnt = {int(e): np.zeros(N) for e in ent_unique}

    for pos in range(nT):
        tau = uniq_times[pos]
        if pos > last_refit_pos and pos >= refit_threshold:
            refit(pos)
            last_refit_pos = pos
            refit_threshold = max(refit_threshold + 1, int(pos * REFIT_GROWTH) + 1)

        W = params["W"]; alpha = params["alpha"]; R = params["R"]
        a = params["a"]; qd = params["qd"]; beta = params["beta"]
        gmean = params["gmean"]; emean = params["emean"]; nu = params["nu"]
        span = params["span"]; Lr = params["Lr"]
        A = np.diag(a)
        Q = np.diag(qd)
        invR = 1.0 / np.maximum(R, 1e-8)

        # seasonal component at this time (deterministic in tau)
        phi_t = _build_phi(np.array([tau]), params["K"], span)[0]   # (P,)
        season = phi_t @ beta                                       # (N,)

        # CONTEMPORANEOUS time effect (cross-section @ tau is allowed): pooled mean
        # over entities observed at tau of (x - entity_offset - season). Captures a
        # shared period level so blacked-out entities still get the right level.
        tfe = np.zeros(N)
        if is_panel:
            tsum = np.zeros(N); tcnt = np.zeros(N)
            for r in rows_by_time[pos]:
                e = int(eids[r]); row = X[r]; o = np.isfinite(row)
                if not o.any():
                    continue
                oc = off_cnt[e]
                offs = np.where(oc > 0, off_sum[e] / np.maximum(oc, 1), emean.get(e, gmean))
                tsum[o] += row[o] - offs[o] - season[o]
                tcnt[o] += 1
            tfe = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), 0.0)

        for r in rows_by_time[pos]:
            e = int(eids[r])
            z = state_z[e]; P = state_P[e]
            if z is None:
                z = np.zeros(Lr); P = np.eye(Lr)

            # predict (Kalman; for blackout this alone extrapolates the state)
            z_pred = A @ z
            P_pred = A @ P @ A.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)

            row = X[r]; obs = np.isfinite(row)

            # baseline = entity offset (+ contemporaneous time effect for panels)
            if is_panel:
                oc = off_cnt[e]
                base = np.where(oc > 0, off_sum[e] / np.maximum(oc, 1),
                                emean.get(e, gmean)) + tfe + season
            else:
                base = gmean + season

            if obs.any():
                o = obs
                Wo = W[o]; ro = R[o]; invRo = invR[o]
                y = row[o] - base[o]
                nu_pred = y - Wo @ z_pred                  # innovation

                # Student-t IRLS weights from the innovation (causal: uses obs@tau
                # + past params only). Downweights outliers -> robust update.
                # diag innovation scale ~ sum_l Wo_l^2 P_pred_ll + ro
                s_diag = ro + np.einsum("ml,l->m", Wo ** 2, np.diag(P_pred))
                stand = (nu_pred ** 2) / np.maximum(s_diag, 1e-8)
                wt = (nu + 1.0) / (nu + stand)             # (m,)
                invRo_w = invRo * wt                       # weighted obs precision

                # Woodbury in L-space with WEIGHTED diagonal observation precision.
                # M = P_pred^-1 + Wo' diag(invRo_w) Wo
                try:
                    cP = cho_factor(P_pred + JITTER * np.eye(Lr), lower=True,
                                    check_finite=False)
                    Pinv = cho_solve(cP, np.eye(Lr), check_finite=False)
                except Exception:
                    Pinv = np.linalg.pinv(P_pred)
                WtRw = Wo.T * invRo_w                       # (Lr, m)
                M = Pinv + WtRw @ Wo
                try:
                    cM = cho_factor(M + JITTER * np.eye(Lr), lower=True,
                                    check_finite=False)
                    # posterior mean: z = z_pred + M^-1 Wo' diag(invRo_w) nu_pred
                    rhs = WtRw @ nu_pred
                    z = z_pred + cho_solve(cM, rhs, check_finite=False)
                    P = cho_solve(cM, np.eye(Lr), check_finite=False)
                except Exception:
                    Minv = np.linalg.pinv(M)
                    z = z_pred + Minv @ (WtRw @ nu_pred)
                    P = Minv
                P = 0.5 * (P + P.T)
            else:
                z = z_pred
                P = P_pred

            miss = ~obs
            if miss.any():
                out[r, miss] = base[miss] + W[miss] @ z

            state_z[e] = z; state_P[e] = P
            if obs.any():
                off_sum[e][obs] += row[obs]
                off_cnt[e][obs] += 1

    # safety: strictly-past running column-mean fill for any residual NaN
    if np.isnan(out).any():
        csum = np.zeros(N); ccnt = np.zeros(N)
        for pos in range(nT):
            mean_so_far = np.where(ccnt > 0, csum / np.maximum(ccnt, 1), 0.0)
            for r in rows_by_time[pos]:
                m = np.isnan(out[r])
                if m.any():
                    out[r, m] = mean_so_far[m]
            for r in rows_by_time[pos]:
                orow = X[r]; o = np.isfinite(orow)
                csum[o] += orow[o]; ccnt[o] += 1
        out = np.where(np.isnan(out), 0.0, out)

    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Unified_DFM", online_impute))
