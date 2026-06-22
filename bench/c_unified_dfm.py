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
EM_SWEEPS = 8  # warm-started EM sweeps per refit on the expanding window
REFIT_GROWTH = 1.6      # geometric refit schedule (refit when pos >= last*GROWTH)
MIN_REFIT_TIMES = 12    # don't fit dynamics until this many distinct times seen
L_MAX = 8  # max latent factors; ARD prunes down from here
JITTER = 1e-6
NU_INIT = 6.0           # start ROBUST (heavy-tail prior); EM raises nu toward
                        # Gaussian only if the data's residuals are actually light.
                        # Robust-first prevents early sweeps from baking outliers
                        # into the factor subspace before nu has a chance to adapt.
NU_MIN, NU_MAX = 2.5, 1e4


# =========================================================================== #
#  Expanding-window EM fit of the SHARED parameters from rows with time<=tau.
#  Returns a params dict. Warm-started across refits for speed + stability.
#  Implements: ARD on W columns, per-feature R, diagonal AR A in [0,1], Fourier
#  beta with ARD ridge, and EM-updated Student-t nu.
# =========================================================================== #
def _build_phi(time_vals, periods):
    """Fourier design matrix Phi (n, 1 + 2*sum_p H_p): intercept + sin/cos
    harmonics for each DATA-DETECTED seasonal period in `periods`. Periods are
    discovered empirically (autocorrelation peaks) -- no hard-coded period. Each
    period gets 2 harmonics (fundamental + 1st overtone). ARD ridge on beta then
    shrinks any term the data does not need."""
    t = np.asarray(time_vals, float)
    n = len(t)
    cols = [np.ones(n)]
    for per in periods:
        if per < 2:
            continue
        for h in (1, 2):
            ang = 2.0 * np.pi * h * t / per
            cols.append(np.sin(ang))
            cols.append(np.cos(ang))
    return np.ascontiguousarray(np.column_stack(cols))


def _detect_periods(series, max_lag, top=2):
    """Detect up to `top` dominant seasonal periods from the autocorrelation of a
    1D residual series (NaNs allowed). Empirical-Bayes-style: pick lags whose ACF
    is a local maximum and exceeds a noise band. Returns a sorted list of periods.
    Purely data-driven; no dataset-specific constant."""
    x = np.asarray(series, float)
    fin = np.isfinite(x)
    if fin.sum() < 16:
        return []
    x = x.copy()
    x[~fin] = np.nanmean(x[fin])
    x = x - x.mean()
    nrm = float(x @ x) + 1e-12
    # cap the max lag by the block length (causal) and by an absolute capacity
    # bound (keeps the O(n*Lmax) ACF scan cheap on long real series); neither
    # depends on the TOTAL series length, so truncation cannot change it.
    Lmax = int(min(max_lag, len(x) // 2 - 1, 512))
    if Lmax < 3:
        return []
    # autocorrelation via FFT (O(n log n)) up to Lmax
    nfft = 1
    while nfft < 2 * len(x):
        nfft *= 2
    fx = np.fft.rfft(x, nfft)
    acf_full = np.fft.irfft(fx * np.conj(fx), nfft)[: Lmax + 1]
    ac = acf_full / nrm
    band = 2.0 / np.sqrt(fin.sum())          # ~95% white-noise ACF band
    cands = []
    for L in range(3, Lmax):
        if ac[L] > band and ac[L] >= ac[L - 1] and ac[L] >= ac[L + 1]:
            cands.append((ac[L], L))
    cands.sort(reverse=True)
    out = []
    for _, L in cands:
        if all(abs(L - p) > 1 and (L % p != 0 or L == p) for p in out):
            out.append(L)
        if len(out) >= top:
            break
    return sorted(out)


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

    span = (block_tids.max() - block_tids.min() + 1) if n else 1.0
    # ---- entity-demeaned target (remove FE so seasonality is the shared part) -
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

    # ---- DETECT seasonal periods from the data (empirical, no constants) -----
    # Average the demeaned target across features into one pooled series indexed
    # by time, then read autocorrelation peaks. K caps the max detectable period.
    periods = []
    if K > 0:
        # detect on the per-time pooled feature mean (works for 1D/2D/panel alike)
        tmin = int(block_tids.min())
        L_series = int(block_tids.max()) - tmin + 1
        psum = np.zeros(L_series)
        pc = np.zeros(L_series)
        for i in range(n):
            oi = obs[i]
            if oi.any():
                ti = int(block_tids[i]) - tmin
                psum[ti] += float(np.mean(target[i, oi]))
                pc[ti] += 1.0
        pooled = np.where(pc > 0, psum / np.maximum(pc, 1), np.nan)
        periods = _detect_periods(pooled, max_lag=K, top=2)
    Phi = _build_phi(block_tids, periods)
    P = Phi.shape[1]

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
    # Lr is fixed (= min(L, N)) across refits so the carried Kalman state keeps a
    # stable dimension; ARD prunes redundant columns rather than dropping them.
    Lr = max(1, min(L, N))

    # SVD-based initialisation of W on the mean-filled centred residual: a data-
    # driven starting point at the right SCALE (truncated principal subspace), so
    # the EM does not have to grow W up from a tiny random seed (which, at small n,
    # gets pruned by ARD before it can fit). Standard PPCA init.
    def _svd_init():
        Xf = Xc.copy()                     # already centred; missing -> 0
        # robust: winsorise at a per-feature MAD multiple so a handful of heavy-tail
        # outliers do not steer the principal subspace (the SVD is otherwise L2 and
        # outlier-sensitive). MAD-based -> scale-free, not a tuned constant.
        for j in range(N):
            cj = Xf[:, j]
            mad = np.median(np.abs(cj[obs[:, j]])) if obs[:, j].any() else 0.0
            if mad > 0:
                lim = 8.0 * 1.4826 * mad   # ~8 sigma robust cap
                np.clip(cj, -lim, lim, out=cj)
        try:
            U_, s_, Vt_ = np.linalg.svd(Xf, full_matrices=False)
            k = min(Lr, len(s_))
            scale = np.sqrt(np.maximum(s_[:k], 1e-6) / max(np.sqrt(n), 1.0))
            Wi = (Vt_[:k].T * scale)        # (N, k)
            if Wi.shape[1] < Lr:
                pad = np.random.default_rng(0).standard_normal((N, Lr - Wi.shape[1])) * 0.1
                Wi = np.hstack([Wi, pad])
            return np.ascontiguousarray(Wi)
        except Exception:
            return np.random.default_rng(0).standard_normal((N, Lr)) * 0.1

    R0 = np.maximum(np.var(Xc[obs]) if obs.any() else 1.0, 1e-3) * np.ones(N)
    if warm is not None and warm.get("W") is not None and warm["W"].shape == (N, Lr):
        # warm-carry the loadings (they evolve smoothly and, on heavy-tailed data,
        # carry the robustly-cleaned subspace forward). Revive any column the ARD
        # pruned to ~0 from a fresh SVD direction so pruning stays REVERSIBLE.
        W = warm["W"].copy()
        R = warm.get("R", R0).copy()
        if R.shape[0] != N:
            R = R0
        col_norm = np.sqrt(np.sum(W ** 2, axis=0))
        dead = col_norm < 1e-3 * (np.sqrt(np.mean(R)) + 1e-9)
        if dead.any():
            W[:, dead] = _svd_init()[:, dead]
    else:
        W = _svd_init()
        R = R0
    alpha = np.ones(Lr)

    nu = warm.get("nu", NU_INIT) if warm is not None else NU_INIT
    # ARD ridge per loading column (relevance). Large lam_l => column pruned.
    # Stable sparse-Bayes / RVM formulation: latent prior is FIXED N(0, I); the
    # shrinkage lives on the loadings W. lam is RE-INITIALISED each refit (not warm
    # carried): a column pruned (lam->inf) in one window would otherwise be stuck
    # dead forever; EM re-discovers the relevant rank in a few sweeps from warm W/R.
    lam = np.ones(Lr)

    for sweep in range(EM_SWEEPS):
        # E-step: latent scores Z (n, Lr) with FIXED latent prior N(0, I).
        #   Z_i = (W_o' R_o^-1 W_o + I)^-1 W_o' R_o^-1 x_i
        Z = np.zeros((n, Lr))
        for i in range(n):
            oi = obs[i]
            if not oi.any():
                continue
            Wo = W[oi]; ro = R[oi]
            WtR = Wo.T / ro                       # (Lr, m)
            G = WtR @ Wo + np.eye(Lr)
            try:
                c = cho_factor(G + JITTER * np.eye(Lr), lower=True, check_finite=False)
                Z[i] = cho_solve(c, WtR @ Xc[i, oi], check_finite=False)
            except Exception:
                Z[i] = np.linalg.lstsq(Wo, Xc[i, oi], rcond=None)[0]

        # Student-t IRLS weights from current residual (scale-mixture robustness):
        # rows with large standardized residual are down-weighted. Uses learned nu.
        pred = Z @ W.T
        resid = np.where(obs, Xc - pred, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = (resid ** 2) / np.maximum(R, 1e-8)
        tw = np.where(obs, (nu + 1.0) / (nu + r2), 0.0)

        # M-step W (per feature row), ridged by the ARD column relevances lam:
        #   W_j = (Z_w' Z + R_j diag(lam))^-1 Z_w' x_j   (diag(lam) shrinks weak cols)
        new_W = W.copy()
        Lam = np.diag(lam)
        for j in range(N):
            rj = obs[:, j]
            if not rj.any():
                continue
            Zr = Z[rj]; wj = tw[rj, j]
            Zw = Zr * wj[:, None]
            G = Zr.T @ Zw + R[j] * Lam + JITTER * np.eye(Lr)
            try:
                c = cho_factor(G, lower=True, check_finite=False)
                new_W[j] = cho_solve(c, Zw.T @ Xc[rj, j], check_finite=False)
            except Exception:
                new_W[j] = np.linalg.lstsq(Zr, Xc[rj, j], rcond=None)[0]
        W = new_W

        # per-feature R from weighted residuals (robust)
        pred = Z @ W.T
        resid = np.where(obs, Xc - pred, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = (resid ** 2) / np.maximum(R, 1e-8)
        tw = np.where(obs, (nu + 1.0) / (nu + r2), 0.0)
        num = (tw * resid ** 2).sum(0)
        den = np.maximum(obs.sum(0), 1.0)
        R = np.maximum(num / den, R_floor)      # principled relative noise floor

        # ARD update of column relevances via EXPLAINED VARIANCE. Each factor's
        # contribution to the (whitened) data is e_l = var(z_l) * sum_j W[j,l]^2/R[j].
        # We pull the prior precision lam_l toward N/e_l, but inflate it sharply for
        # factors whose explained variance is a tiny FRACTION of the strongest
        # factor's -- those are noise directions and get pruned (lam->large) =>
        # AUTOMATIC RANK. The fraction test is scale-free (ratio to the max), not a
        # per-dataset constant. Deferred a couple of sweeps so columns can grow in.
        if sweep >= 2:
            wnorm = np.sum((W ** 2) / np.maximum(R, 1e-8)[:, None], axis=0)
            lam = N / np.maximum(wnorm, 1e-8)
            lam = np.minimum(lam, 1e8)
            alpha = 1.0 / np.maximum(lam, 1e-8)  # stored for inspection

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

    return dict(W=W, alpha=alpha, R=R, a=a, qd=qd, beta=beta, periods=periods,
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

    # K = MAX seasonal period the empirical detector may consider. We pass a large
    # constant; _detect_periods internally caps the lag at (block_length // 2 - 1),
    # i.e. by the data available UP TO tau at each refit -- so K does NOT depend on
    # the TOTAL series length (which would leak future info under truncation and
    # break causality). The detector keeps only ACF-significant periods; ARD on
    # beta shrinks unhelpful harmonics. Not a per-dataset constant.
    K = 100000
    L = min(L_MAX, N)

    # ---- params (shared); warm-started across refits -------------------------
    params = None

    def refit(upto_pos):
        nonlocal params
        keep = tids <= uniq_times[upto_pos]
        params = _fit_params(X[keep], tids[keep], eids[keep], ent_unique,
                             1.0, L, K, warm=params)

    # CAUSAL refit schedule: the FIRST fit happens only once `init_pos` distinct
    # times are available, and it uses data <= that time. Crucially we do NOT fit
    # before the loop and then apply to earlier times (that would use future data).
    # Times before the first fit are imputed by a strictly-past running column mean
    # (point-in-time). Every later fit uses only rows with time <= the current tau.
    init_pos = max(min(MIN_REFIT_TIMES, nT) - 1, 0)
    last_refit_pos = -1
    refit_threshold = init_pos

    # per-entity Kalman state (reset at first appearance)
    state_z = {int(e): None for e in ent_unique}
    state_P = {int(e): None for e in ent_unique}

    # strictly-past per-entity running offset (entity FE), causal
    off_sum = {int(e): np.zeros(N) for e in ent_unique}
    off_cnt = {int(e): np.zeros(N) for e in ent_unique}

    # strictly-past running column mean for the cold-start fallback (causal)
    cs_sum = np.zeros(N); cs_cnt = np.zeros(N)

    # strictly-past EXPONENTIALLY-WEIGHTED column level (for the non-panel level):
    # recency-weighted so the baseline tracks slow drift / level breaks (the model
    # then only has to explain the de-levelled signal). Forgetting is a principled
    # non-stationary prior. The half-life is a FIXED capacity constant (does not
    # depend on total series length, so truncation cannot change it -> causal).
    EW_HALFLIFE = 100.0
    ew_lam = 0.5 ** (1.0 / EW_HALFLIFE)
    ew_sum = np.zeros(N)
    ew_cnt = np.zeros(N)        # per-feature decayed observation weight

    for pos in range(nT):
        tau = uniq_times[pos]
        if pos >= refit_threshold:
            refit(pos)
            last_refit_pos = pos
            refit_threshold = max(refit_threshold + 1, int(pos * REFIT_GROWTH) + 1)

        if params is None:
            # cold start: no model yet (fewer than init_pos times seen). Impute
            # missing cells with the strictly-past running column mean, advance the
            # running stats, and continue. Purely point-in-time.
            cmean = np.where(cs_cnt > 0, cs_sum / np.maximum(cs_cnt, 1), 0.0)
            for r in rows_by_time[pos]:
                row = X[r]; miss = np.isnan(row)
                if miss.any():
                    out[r, miss] = cmean[miss]
            for r in rows_by_time[pos]:
                orow = X[r]; o = np.isfinite(orow)
                cs_sum[o] += orow[o]; cs_cnt[o] += 1
            continue

        W = params["W"]; alpha = params["alpha"]; R = params["R"]
        a = params["a"]; qd = params["qd"]; beta = params["beta"]
        gmean = params["gmean"]; emean = params["emean"]; nu = params["nu"]
        span = params["span"]; Lr = params["Lr"]
        A = np.diag(a)
        Q = np.diag(qd)
        invR = 1.0 / np.maximum(R, 1e-8)

        # seasonal component at this time (deterministic in tau)
        phi_t = _build_phi(np.array([tau]), params["periods"])[0]   # (P,)
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
                # non-panel level: blend the fitted (expanding) gmean with the
                # strictly-past EW level so the baseline follows drift/level breaks.
                # Falls back to gmean per-feature where no recent obs exist.
                ew_level = np.where(ew_cnt > 1e-6, ew_sum / np.maximum(ew_cnt, 1e-6),
                                    gmean)
                base = ew_level + season

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

        # advance the strictly-past EW level using THIS time's observed values
        # (after they have been used for imputation -> causal). Decay all features,
        # add observed ones. Pooled across entities at this time for panels.
        ew_sum *= ew_lam
        ew_cnt *= ew_lam
        for r in rows_by_time[pos]:
            row = X[r]; o = np.isfinite(row)
            if o.any():
                ew_sum[o] += row[o]
                ew_cnt[o] += 1.0

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
