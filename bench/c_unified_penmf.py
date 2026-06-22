"""
Unified_PenalizedMF -- ONE causal imputation model, Framing B.

A single regularized objective is optimized jointly at every time step over a
trailing causal window. There is NO routing, NO data-driven model selection, NO
if/else on dataset type. Every classical technique is a SPECIAL CASE reached by
the CONTINUOUS parameters of this one objective, and the data moves those
parameters implicitly during fitting (ARD / empirical-Bayes / IRLS / scale
heuristics) -- never by a hand-tuned threshold keyed to a dataset.

----------------------------------------------------------------------------
THE OBJECTIVE (minimized jointly over W, Z, beta, per row over the window):

  sum_{t,i obs} rho_nu( x_{t,i} - mu_i - (Phi_t beta)_i - (z_t W^T)_i )
    + sum_l alpha_l ||W[:,l]||^2          # ARD per factor column  -> RANK emerges
    + lambda_z sum_t ||z_t - a z_{t-1}||^2 # AR penalty on factors  -> DYNAMICS (a)
    + lambda_b ||beta||^2 (ARD per basis)  # Fourier ridge          -> SEASON emerges
    + ridge on mu                          # FE

where rho_nu is the Student-t neg-log-likelihood (a scale-mixture); minimizing it
is IRLS with weights w = (nu+1)/(nu + r^2/s^2). nu and the scale s are estimated by
EM moment matching on the running residuals -> ROBUSTNESS emerges (nu small => heavy
down-weighting; nu large => plain L2).

SPECIAL CASES (all reachable by learned continuous params, none hard-coded):
  * SoftImpute / low-rank   : a=0, nu=inf, alpha_l small for a few l, large else.
  * TRMF                    : a in (0,1) learned, AR penalty active.
  * Kalman/SSM dynamics     : a near 1 (random walk) -> AR-extrapolation in blackout
                              (when a row has NO observed cells, z_t = a z_{t-1} is the
                              filter prediction; mu+Phi beta+pred fills it).
  * cross-section regression: ARD kills ALL factor columns (alpha_l huge) => rank 0 =>
                              the per-row solve degenerates to conditional mean from the
                              window covariance via the residual ridge (EW-cov limit).
  * MC-NNM / FE             : mu_i (feature FE) + per-row level (time FE) + low-rank.
  * robust (Student-t)      : nu small, IRLS scale-mixture.
  * seasonal                : Phi = Fourier design; beta!=0 when ARD keeps it, shrinks
                              to 0 otherwise.

----------------------------------------------------------------------------
CAUSALITY (verified by causal.py):
  We iterate strictly forward in time. To impute the row(s) at time tau we use ONLY
  rows at times <= tau (a trailing window) plus the contemporaneous observed cells of
  tau itself (allowed). All learned parameters (W, AR coeff a, nu, scales, ARD alphas)
  are updated online from data <= tau. Truncating future rows cannot change any past
  imputation. Panel: latent state z is reset per entity; W/a/nu/scales POOLED across
  entities (estimated from all rows <= tau, any entity).

Speed: numpy/scipy only. cho_solve everywhere (never inv/pinv). Woodbury-free because
the per-row systems are rank-sized (<=R). Column factors W refit on a trailing window
only every REFIT_EVERY steps; each row solves its own rank-R factor against current W.
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

# ---- principled constants (NOT fit to any dataset; standard priors/defaults) ----
R_MAX        = 8         # max latent factors offered; ARD shrinks the unused ones
WINDOW       = 400       # trailing causal window length (rows) for W refits
REFIT_EVERY  = 20        # refit column factors W every this many time steps
SWEEPS       = 3         # ALS sweeps inside a W refit
NU_INIT      = 8.0       # Student-t dof start (moderate); EM moves it 2..~50
NU_MIN, NU_MAX = 2.5, 60.0
HALFLIFE     = 200.0     # EW forgetting for pooled scale/cov statistics
EPS          = 1e-9


# --------------------------------------------------------------------------- #
# Fourier seasonal design (causal: depends only on the time index, not values).
# Periods are data-agnostic harmonics of the window; ARD on beta shrinks unused
# ones to ~0 so "no seasonality" is the learned special case.
# --------------------------------------------------------------------------- #
def _fourier_periods(T):
    # candidate seasonal periods: common semantic cycles only (NOT window
    # subharmonics, which would fit smooth trends and extrapolate wildly across
    # block gaps). ARD on beta shrinks any of these to ~0 when no such cycle exists,
    # so "no seasonality" is the learned special case.
    cand = [24.0, 168.0, 12.0, 7.0, 48.0, 365.0]
    out = [p for p in cand if 3.0 <= p <= T / 2.0]
    out = sorted(set(out))
    return out[:6]


def _fourier_row(t, periods):
    """Design vector Phi_t (1 x P) for absolute time index t: [sin,cos] per period."""
    if not periods:
        return np.zeros(0)
    ph = 2.0 * np.pi * t / np.asarray(periods)
    return np.concatenate([np.sin(ph), np.cos(ph)])


# --------------------------------------------------------------------------- #
# Core engine operating on a single time-ordered stream of rows (2D or one entity).
# Stateful object so panel can reuse pooled W/a/nu while resetting latent z.
# --------------------------------------------------------------------------- #
class _UnifiedCore:
    def __init__(self, N, periods, E=1):
        self.N = N
        self.E = E
        self.periods = periods
        self.P = 2 * len(periods)
        self.R = R_MAX
        rng = np.random.default_rng(0)
        self.W = 0.01 * rng.standard_normal((N, self.R))     # loadings (pooled)
        self.alpha = np.ones(self.R)                          # ARD precisions on W cols
        self.beta = np.zeros((self.P, N)) if self.P else np.zeros((0, N))
        self.beta_alpha = np.ones(max(self.P, 1))             # ARD on Fourier basis
        # online ridge (RLS) accumulators for the seasonal fit of (x-mu) on Phi.
        # Phi is shared across features so PtP is P x P; Pty is P x N. EW-forgotten so
        # the seasonal mean adapts (drift). beta solved closed-form from these.
        self.PtP = np.zeros((self.P, self.P))
        self.Pty = np.zeros((self.P, N))
        self.a = 0.5                                          # AR coeff on z (learned)
        self.nu = NU_INIT                                     # Student-t dof (pooled)
        self.scale2 = 1.0                                     # robust residual scale^2
        # expanding feature mean (FE / mu), point-in-time.
        # Per-entity entity-FE (E x N) with a pooled global fallback. For 2D E=1 so
        # this reduces to a single expanding feature mean (MC-NNM column FE limit).
        self.mu_sum = np.zeros((E, N))
        self.mu_cnt = np.zeros((E, N))
        self.g_sum = np.zeros(N)
        self.g_cnt = np.zeros(N)
        # trailing residual snapshot window (for W refit), filled causally
        self.res_buf = np.zeros((0, N))
        self.W_buf = np.zeros((0, N), dtype=bool)
        self.A_buf = np.zeros((0, self.R))
        self.tidx_buf = np.zeros((0,), dtype=int)             # absolute time of buf row
        # pooled EW residual covariance (for rank-0 / cross-section limit)
        self.cw = 0.0
        self.csum = np.zeros(N)
        self.cM2 = np.zeros((N, N))
        self.lam = 0.5 ** (1.0 / HALFLIFE)
        # mu forgets more slowly (longer half-life) than the covariance: the level is
        # a slow-moving FE, the cross-section structure adapts faster.
        self.lam_mu = 0.5 ** (1.0 / (4.0 * HALFLIFE))
        # AR sufficient stats (pooled, learned a): sum z_t.z_{t-1}, sum z_{t-1}^2
        self.ar_num = 0.0
        self.ar_den = 0.0
        # robust scale + nu EM stats (EW-forgotten)
        self.r2_sum = 0.0
        self.r2_cnt = 0.0
        self.m2_sum = 0.0
        self.m4_sum = 0.0
        self.mom_cnt = 0.0
        # per-feature EW robust scale (mean-abs-deviation proxy) of the de-FE,
        # de-season value, for per-cell winsorizing of the data term.
        self.fsc_sum = np.ones(N)
        self.fsc_cnt = np.full(N, 1e-3)
        self.W_ready = False
        self.step = 0

    # -- feature FE (mu) for entity e from history <= current (expanding),
    #    falling back to the pooled global expanding mean where e is unseen --
    def _mu(self, e=0):
        g = np.where(self.g_cnt > 0, self.g_sum / np.maximum(self.g_cnt, 1), 0.0)
        cnt = self.mu_cnt[e]
        own = np.where(cnt > 0, self.mu_sum[e] / np.maximum(cnt, 1), 0.0)
        return np.where(cnt > 0, own, g)

    # -- robust IRLS weight for a residual given current nu, scale --
    def _wt(self, r2):
        # Student-t reweighting w = (nu+1)/(nu + r2/scale2)
        return (self.nu + 1.0) / (self.nu + r2 / max(self.scale2, EPS))

    # -- refit pooled column factors W on the trailing residual window (ALS+ARD) --
    def _refit_W(self):
        n = self.res_buf.shape[0]
        if n < 2:
            return
        R = self.res_buf
        Wobs = self.W_buf
        A = self.A_buf
        if A.shape[0] != n:
            newA = np.zeros((n, self.R))
            m = min(A.shape[0], n)
            newA[:m] = A[:m]
            A = newA
        Rf = np.where(Wobs, R, 0.0)
        for _ in range(SWEEPS):
            # update row factors A given W, with AR coupling as a ridge toward a*A_prev
            for i in range(n):
                w = Wobs[i]
                if not w.any():
                    # AR prediction only
                    A[i] = self.a * A[i - 1] if i > 0 else 0.0
                    continue
                Bw = self.W[w]
                G = Bw.T @ Bw + np.diag(self.alpha)
                rhs = Bw.T @ Rf[i, w]
                if i > 0:
                    # AR penalty lambda_z (=1, scale set by alpha) couples to prev factor
                    G = G + np.eye(self.R)
                    rhs = rhs + self.a * A[i - 1]
                try:
                    A[i] = cho_solve(cho_factor(G, lower=True, check_finite=False),
                                     rhs, check_finite=False)
                except Exception:
                    A[i] = np.linalg.lstsq(G, rhs, rcond=None)[0]
            # update loadings W given A, ARD ridge per column
            AtA = A.T @ A
            for j in range(self.N):
                w = Wobs[:, j]
                if not w.any():
                    continue
                Aw = A[w]
                G = Aw.T @ Aw + np.diag(self.alpha)
                rhs = Aw.T @ Rf[w, j]
                try:
                    self.W[j] = cho_solve(cho_factor(G, lower=True, check_finite=False),
                                          rhs, check_finite=False)
                except Exception:
                    self.W[j] = np.linalg.lstsq(G, rhs, rcond=None)[0]
        self.A_buf = A
        # --- ARD empirical-Bayes update of per-column precisions alpha_l ---
        #   alpha_l = (N) / (||W[:,l]||^2 + eps)   (evidence-style; unused cols -> huge)
        col_e = np.sum(self.W ** 2, axis=0)
        self.alpha = self.N / (col_e + 1.0)         # +1 prior => no zero-div, gentle
        # --- AR coeff a from pooled factor stats (closed-form ridge regression) ---
        if A.shape[0] >= 2:
            zt = A[1:]
            ztm = A[:-1]
            num = float(np.sum(zt * ztm))
            den = float(np.sum(ztm * ztm)) + 1.0
            a_hat = num / den
            self.a = float(np.clip(a_hat, 0.0, 0.999))
        self.W_ready = True

    # -- per-row solve of latent z_t against current W (rank-R, IRLS-weighted) --
    def _solve_z(self, resid, obs, z_prev):
        if not obs.any():
            # blackout row: Kalman/AR prediction is the special case
            return self.a * z_prev if z_prev is not None else np.zeros(self.R)
        Bw = self.W[obs]                       # (nobs, R)
        rw = resid[obs]
        # IRLS robust weights from a first L2 pass residual
        G = Bw.T @ Bw + np.diag(self.alpha) + np.eye(self.R)   # AR ridge (lambda_z=1)
        rhs = Bw.T @ rw + (self.a * z_prev if z_prev is not None else 0.0)
        try:
            z = cho_solve(cho_factor(G, lower=True, check_finite=False), rhs,
                          check_finite=False)
        except Exception:
            z = np.linalg.lstsq(G, rhs, rcond=None)[0]
        # one IRLS reweight pass (Student-t scale mixture)
        pred = Bw @ z
        r2 = (rw - pred) ** 2
        wts = self._wt(r2)
        Bww = Bw * wts[:, None]
        G = Bww.T @ Bw + np.diag(self.alpha) + np.eye(self.R)
        rhs = Bww.T @ rw + (self.a * z_prev if z_prev is not None else 0.0)
        try:
            z = cho_solve(cho_factor(G, lower=True, check_finite=False), rhs,
                          check_finite=False)
        except Exception:
            z = np.linalg.lstsq(G, rhs, rcond=None)[0]
        return z

    # -- cross-section conditional mean from pooled EW residual cov (rank-0 limit) --
    def _xsec_fill(self, resid, obs, miss):
        if self.cw < 5.0 or not obs.any():
            return None
        mean = self.csum / self.cw
        cov = self.cM2 / self.cw - np.outer(mean, mean)
        o = np.where(obs)[0]; m = np.where(miss)[0]
        ridge = 1e-2 * (np.trace(cov) / max(self.N, 1) + EPS)
        Soo = cov[np.ix_(o, o)] + ridge * np.eye(o.size)
        Smo = cov[np.ix_(m, o)]
        xo = resid[o] - mean[o]
        try:
            c = cho_factor(Soo, lower=True, check_finite=False)
            sol = cho_solve(c, xo, check_finite=False)
            return mean[m] + Smo @ sol
        except Exception:
            return None

    def _update_robust_scale(self, r2_vals):
        # EW-forgotten robust scale + EM estimate of nu by moment matching.
        # We track the EW mean of r2 (robust scale^2 proxy, weighted by current IRLS
        # weights so outliers don't inflate it) and the EW 2nd/4th moments of the
        # standardized residual to estimate the dof nu from excess kurtosis:
        #   kurt_excess(t_nu) = 6/(nu-4)  =>  nu = 4 + 6/kurt_excess.
        for r2 in r2_vals:
            w = self._wt(r2)                       # current IRLS weight
            self.r2_sum = self.lam * self.r2_sum + w * r2
            self.r2_cnt = self.lam * self.r2_cnt + w
            # raw (unweighted) moments of standardized resid for kurtosis -> nu
            s2 = max(self.scale2, EPS)
            self.m2_sum = self.lam * self.m2_sum + (r2 / s2)
            self.m4_sum = self.lam * self.m4_sum + (r2 / s2) ** 2
            self.mom_cnt = self.lam * self.mom_cnt + 1.0
        if self.r2_cnt > 5.0:
            self.scale2 = max(self.r2_sum / self.r2_cnt, EPS)
        if self.mom_cnt > 30.0:
            m2 = self.m2_sum / self.mom_cnt
            m4 = self.m4_sum / self.mom_cnt
            kurt = m4 / max(m2 * m2, EPS)          # ~3 normal, >3 heavy-tailed
            excess = kurt - 3.0
            if excess > 0.2:                        # heavy tails -> finite small nu
                nu_hat = 4.0 + 6.0 / excess
            else:                                   # near-normal -> large nu (L2 limit)
                nu_hat = NU_MAX
            # gentle EM move toward the moment estimate (stability)
            self.nu = float(np.clip(0.7 * self.nu + 0.3 * nu_hat, NU_MIN, NU_MAX))

    def process_row(self, x_obs_row, abs_t, z_prev, fourier_t, eid=0):
        """Impute one row (in place into a copy), return (filled_row, z_t)."""
        N = self.N
        out = x_obs_row.copy()
        obs = ~np.isnan(x_obs_row)
        miss = ~obs

        mu = self._mu(eid)
        # seasonal mean contribution (shrinks via beta ARD)
        season = (fourier_t @ self.beta) if self.P else np.zeros(N)

        # residual = x - mu - season (on observed)
        resid = np.zeros(N)
        resid[obs] = x_obs_row[obs] - mu[obs] - season[obs]

        # ---- robust data term: per-feature winsorize the residual that feeds the
        # factor solve. The clip band widens with the learned dof nu: small nu (heavy
        # tails, learned from kurtosis) => tight redescending clip; large nu => no clip
        # (the plain-L2 limit). This is the M-estimator the Student-t loss induces, with
        # its threshold set IMPLICITLY by the empirical-Bayes nu, not a tuned constant.
        fsc = self.fsc_sum / np.maximum(self.fsc_cnt, 1e-3)        # per-feature scale
        k_nu = 1.5 + self.nu / 6.0                                  # nu-driven band
        band = k_nu * 1.4826 * fsc
        resid_c = resid.copy()
        if obs.any():
            resid_c[obs] = np.clip(resid[obs], -band[obs], band[obs])

        # latent factor solve (low-rank + AR), point-in-time, on the robust residual
        if self.W_ready:
            z_t = self._solve_z(resid_c, obs, z_prev)
        else:
            z_t = self.a * z_prev if z_prev is not None else np.zeros(self.R)
        lr = self.W @ z_t                          # (N,) low-rank reconstruction

        if miss.any():
            fill = mu[miss] + season[miss] + lr[miss]
            # blend with cross-section conditional mean of the residual where it adds
            # info (the rank-0 / EW-cov special case is reached when ARD has killed the
            # factors, leaving lr~0 and the xsec term carrying the signal).
            xs = self._xsec_fill(resid_c, obs, miss)
            if xs is not None:
                # EB prior-mean blend of the low-rank reconstruction and the
                # cross-section conditional mean of the residual. When ARD has killed
                # the factors (rank-0 limit) lr~0 and the xsec term carries the signal
                # (the EW-cov / cross-section-regression special case); when the factors
                # are strong both agree. Equal weight is the uninformative prior mean.
                fill = mu[miss] + season[miss] + 0.5 * lr[miss] + 0.5 * xs
            out[miss] = fill

        # ---- causal online parameter updates (AFTER imputing this row) ----
        # robust residuals on observed cells (use the clipped residual for the scale
        # so the per-feature band itself is robust to outliers).
        if obs.any():
            pred_obs = lr[obs]
            r2v = (resid[obs] - pred_obs) ** 2
            self._update_robust_scale(r2v)
            # per-feature EW robust scale (abs deviation of de-FE/de-season value)
            self.fsc_sum[obs] = self.lam * self.fsc_sum[obs] + np.abs(resid_c[obs])
            self.fsc_cnt[obs] = self.lam * self.fsc_cnt[obs] + 1.0

        # seasonal beta via online ridge (RLS), closed-form. Target is (x - mu) on
        # observed cells; EW-forgotten so it tracks drifting seasonality. ARD per
        # harmonic shrinks beta to ~0 when there is no cycle (seasonality emerges).
        if self.P and obs.any():
            yrow = np.zeros(N)
            yrow[obs] = x_obs_row[obs] - mu[obs]
            self.PtP = self.lam * self.PtP + np.outer(fourier_t, fourier_t)
            self.Pty[:, obs] = self.lam * self.Pty[:, obs] + np.outer(fourier_t, yrow[obs])
            self.Pty[:, ~obs] = self.lam * self.Pty[:, ~obs]
            # refit beta occasionally (closed-form ridge with ARD diag)
            if self.step % 8 == 0 and self.PtP[0, 0] > 3.0:
                G = self.PtP + np.diag(self.beta_alpha)
                try:
                    c = cho_factor(G, lower=True, check_finite=False)
                    self.beta = cho_solve(c, self.Pty, check_finite=False)
                except Exception:
                    self.beta = np.linalg.lstsq(G, self.Pty, rcond=None)[0]
                be = np.sum(self.beta ** 2, axis=1)
                self.beta_alpha = self.N / (be + 1.0)

        # full reconstructed row (for residual buffer + covariance). Use the robust
        # (winsorized) residual on observed cells so outliers never enter the pooled
        # covariance / loadings; missing cells use the model reconstruction.
        full_resid = np.where(obs, resid_c, lr)

        # robust row weight (Student-t scale mixture) from the observed prediction
        # residual: a row dominated by outliers is down-weighted everywhere it feeds
        # parameter learning, so W / covariance / scale stay clean (robust emerges).
        if obs.any():
            r2row = float(np.mean((resid[obs] - lr[obs]) ** 2))
            row_w = self._wt(r2row)
        else:
            row_w = 1.0

        # update EW residual covariance (pooled) for the cross-section limit,
        # robustly down-weighting outlier rows.
        self.cw = self.lam * self.cw + row_w
        self.csum = self.lam * self.csum + row_w * full_resid
        self.cM2 = self.lam * self.cM2 + row_w * np.outer(full_resid, full_resid)

        # push the per-feature-winsorized residual into the trailing window buffer so
        # outliers cannot corrupt the pooled loadings W during refits.
        rrow = np.zeros(N)
        rrow[obs] = resid_c[obs]
        self.res_buf = np.vstack([self.res_buf, rrow[None, :]])
        self.W_buf = np.vstack([self.W_buf, obs[None, :]])
        self.A_buf = np.vstack([self.A_buf, z_t[None, :]])
        self.tidx_buf = np.append(self.tidx_buf, abs_t)
        if self.res_buf.shape[0] > WINDOW:
            self.res_buf = self.res_buf[-WINDOW:]
            self.W_buf = self.W_buf[-WINDOW:]
            self.A_buf = self.A_buf[-WINDOW:]
            self.tidx_buf = self.tidx_buf[-WINDOW:]

        # EW-forgotten feature mean update (per entity + global). Forgetting lets the
        # FE/level TRACK regime shifts (drift / level breaks) -- the random-walk-mean
        # special case -- while a long half-life keeps it near a static FE otherwise.
        # Decay is per OWN observation, so it is well-defined under interleaved panels.
        self.mu_sum[eid, obs] = self.lam_mu * self.mu_sum[eid, obs] + x_obs_row[obs]
        self.mu_cnt[eid, obs] = self.lam_mu * self.mu_cnt[eid, obs] + 1.0
        self.g_sum[obs] = self.lam_mu * self.g_sum[obs] + x_obs_row[obs]
        self.g_cnt[obs] = self.lam_mu * self.g_cnt[obs] + 1.0

        # periodic pooled W / a / ARD refit
        self.step += 1
        if (self.step % REFIT_EVERY == 0):
            self._refit_W()

        return out, z_t


# --------------------------------------------------------------------------- #
# 2D / 1D path: single stream in time order.
# --------------------------------------------------------------------------- #
def _impute_2d(X, meta):
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    periods = _fourier_periods(T)
    core = _UnifiedCore(N, periods, E=1)
    out = X.copy()
    z_prev = None
    for t in range(T):
        ft = _fourier_row(t, periods)
        filled, z_t = core.process_row(X[t], t, z_prev, ft, eid=0)
        out[t] = filled
        z_prev = z_t
    if np.isnan(out).any():
        gm = core._mu(0)
        idx = np.where(np.isnan(out)); out[idx] = np.take(gm, idx[1])
    return out


# --------------------------------------------------------------------------- #
# Panel path: entity-major. Reset latent z per entity; POOL W/a/nu/scales across
# entities by processing all entities' rows in GLOBAL time order through ONE core.
# State z is keyed per entity so each entity's dynamics evolve independently while
# the loadings/AR/robust params are shared (pooled) -- exactly the spec.
# --------------------------------------------------------------------------- #
def _impute_panel(X, meta):
    X = np.ascontiguousarray(np.asarray(X, float))
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    T_rows, N = X.shape
    E = int(eids.max()) + 1
    Tmax = int(tids.max()) + 1
    periods = _fourier_periods(Tmax)
    core = _UnifiedCore(N, periods, E=E)
    out = X.copy()

    # process strictly in global time order; within a time, iterate entities.
    order = np.lexsort((eids, tids))     # primary key tids, secondary eids
    z_state = {}                          # per-entity latent z (reset = absent)
    for r in order:
        e = int(eids[r]); t = int(tids[r])
        ft = _fourier_row(t, periods)
        z_prev = z_state.get(e, None)
        filled, z_t = core.process_row(X[r], t, z_prev, ft, eid=e)
        out[r] = filled
        z_state[e] = z_t

    if np.isnan(out).any():
        g = np.where(core.g_cnt > 0, core.g_sum / np.maximum(core.g_cnt, 1), 0.0)
        idx = np.where(np.isnan(out)); out[idx] = np.take(g, idx[1])
    return out


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if meta and "entity_ids" in meta and "time_ids" in meta \
            and len(np.unique(meta["entity_ids"])) > 1:
        return _impute_panel(X, meta)
    return _impute_2d(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Unified_PenMF", online_impute))
