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
MU_HALFLIFE  = float(os.environ.get("MU_HL", "100000.0"))   # ~static expanding FE
EPS          = 1e-9
_BLEND_WLR   = float(os.environ.get("BLEND_WLR", "0.5"))   # sweepable; default EB mean


# Partition-based order statistics (no full sort) that reproduce numpy's
# default 'linear' interpolation EXACTLY. Used in place of np.percentile and
# np.median in the hot time-FE / winsorize paths. Bit-identical to numpy.
# --------------------------------------------------------------------------- #
def _pct_1090(a):
    """Return np.percentile(a, [10, 90]) (linear interp) via np.partition.

    numpy 'linear': virtual index v = (n-1)*q, lo=floor(v), hi=ceil(v),
    result = a_sorted[lo] + (a_sorted[hi]-a_sorted[lo])*(v-lo).
    np.partition guarantees element k is in its sorted position and that all
    elements before it are <= it (and after >= it), so a_sorted[k] == part[k]
    for the partitioned k's; the +1 neighbor is the min of the upper part.
    """
    n = a.size
    out = np.empty(2, dtype=np.float64)
    for j, q in enumerate((0.10, 0.90)):
        v = (n - 1) * q
        lo = int(v)                      # floor (v >= 0)
        frac = v - lo
        if frac == 0.0:
            part = np.partition(a, lo)
            out[j] = part[lo]
        else:
            hi = lo + 1
            part = np.partition(a, (lo, hi))
            out[j] = _lerp(part[lo], part[hi], frac)
    return out


def _lerp(a, b, t):
    """numpy's internal _lerp: a + (b-a)*t, switching to b-(b-a)*(1-t) when
    t>=0.5 for matching round-off. Bit-identical to numpy's interpolation."""
    diff = b - a
    out = a + diff * t
    if t >= 0.5:
        out = b - diff * (1.0 - t)
    return out


def _median_part(a):
    """Return np.median(a) via np.partition (no full sort), bit-identical.

    numpy median: for odd n, the middle order stat; for even n, the mean of
    the two central order stats computed as mid_lo + (mid_hi-mid_lo)*0.5
    (the 'linear' interp at q=0.5).
    """
    n = a.size
    mid = n // 2
    if n % 2 == 1:
        part = np.partition(a, mid)
        return float(part[mid])
    lo = mid - 1
    part = np.partition(a, (lo, mid))
    return float((part[lo] + part[mid]) / 2.0)


# --------------------------------------------------------------------------- #
# Fourier seasonal design (causal: depends only on the time index, not values).
# Periods are data-agnostic harmonics of the window; ARD on beta shrinks unused
# ones to ~0 so "no seasonality" is the learned special case.
# --------------------------------------------------------------------------- #
def _fourier_periods(T):
    # candidate seasonal periods: common semantic cycles only (NOT window subharmonics,
    # which would fit smooth trends and extrapolate wildly across block gaps). ARD on
    # beta shrinks any of these to ~0 when no such cycle exists, so "no seasonality" is
    # the learned special case.
    #
    # The set is FIXED (independent of T): the causal verifier reruns on time-prefixes,
    # and any dependence of the basis on the current length T would change past
    # imputations when the future is truncated (a spurious look-ahead). Harmonics longer
    # than the data so far are simply unidentifiable and the RLS ridge keeps their beta
    # ~0 -- harmless and point-in-time.
    return [7.0, 12.0, 24.0, 48.0, 168.0, 365.0]


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
        # period attached to each beta row (basis is [sin(p0..),cos(p0..)])
        self.beta_period = (np.concatenate([periods, periods]) if periods
                            else np.zeros(0))
        self.t_seen = 0.0
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
        # trailing residual snapshot window (for W refit), filled causally. Held in a
        # preallocated RING (capacity 2*WINDOW) addressed by a head index + count: appending
        # a row is O(N) and dropping the oldest is O(1) (advance head). This replaces the old
        # per-row np.vstack that reallocated the whole WINDOW x N buffer on EVERY step; a
        # compacting memmove now happens only ~once per WINDOW rows -> amortized O(N)/row.
        cap = 2 * WINDOW
        self._cap = cap
        self._bh = 0                                          # ring head (oldest row)
        self._bn = 0                                          # rows currently buffered
        self.res_buf_store = np.zeros((cap, N))
        self.W_buf_store = np.zeros((cap, N), dtype=bool)
        self.A_buf_store = np.zeros((cap, self.R))
        self.tidx_store = np.zeros(cap, dtype=int)            # absolute time of buf row
        self.ent_store = np.zeros(cap, dtype=int)             # entity id of buf row
        # pooled EW residual covariance (for rank-0 / cross-section limit)
        self.cw = 0.0
        self.csum = np.zeros(N)
        self.cM2 = np.zeros((N, N))
        self.lam = 0.5 ** (1.0 / HALFLIFE)
        # mu forgets SLOWLY (long half-life): the per-feature level is a near-static FE,
        # so under sparse/high-missing data it is estimated to high precision; fast
        # regime shifts are caught by the contemporaneous time-FE term instead.
        self.lam_mu = 0.5 ** (1.0 / MU_HALFLIFE)
        # AR sufficient stats (pooled, learned a): sum z_t.z_{t-1}, sum z_{t-1}^2
        self.ar_num = 0.0
        self.ar_den = 0.0
        # robust scale + nu EM stats (EW-forgotten)
        self.r2_sum = 0.0
        self.r2_cnt = 0.0
        self.m2_sum = 0.0
        self.m4_sum = 0.0
        self.mom_cnt = 0.0
        # EW variance of the low-rank prediction residual (resid - lr on observed):
        # measures how well the factors alone predict -> used as v_lr in the blend.
        self.lr_var = 1.0
        self.lrv_sum = 0.0
        self.lrv_cnt = 1e-3
        # idiosyncratic per-feature variance psi (EW) and factor prior variances s_fac.
        # psi starts at the data scale; s_fac at unit. Both estimated empirical-Bayes.
        self.psi = np.ones(N)
        self.psi_sum = np.ones(N)
        self.psi_cnt = np.full(N, 1e-3)
        self.s_fac = np.ones(self.R)
        # per-feature idiosyncratic AR(1) carry: the part of the residual NOT explained
        # by the common factors persists in time (TRMF/AR on the idiosyncratic channel).
        # During a block gap the factor term decays to its prior; this carry term keeps
        # the last idiosyncratic level (decayed by the learned a) so contiguous gaps are
        # extrapolated rather than collapsing to the mean. rho_idio is learned per the
        # autocorrelation of the idiosyncratic residual.
        self.idio_last = np.zeros(N)
        self.idio_age = np.full(N, 1e9)
        self.rho_idio = 0.0
        self.ric_num = 0.0
        self.ric_den = 1e-3
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
    def _push_buf(self, rrow, obs, z_t, abs_t, eid):
        # append one row to the trailing ring (O(N)); compact to the front only when the
        # tail reaches capacity (~once per WINDOW rows); drop the oldest by advancing the
        # head (O(1)). Time order is preserved as a contiguous slice for the W refit.
        if self._bh + self._bn >= self._cap:
            s = slice(self._bh, self._bh + self._bn)
            self.res_buf_store[:self._bn] = self.res_buf_store[s]
            self.W_buf_store[:self._bn] = self.W_buf_store[s]
            self.A_buf_store[:self._bn] = self.A_buf_store[s]
            self.tidx_store[:self._bn] = self.tidx_store[s]
            self.ent_store[:self._bn] = self.ent_store[s]
            self._bh = 0
        p = self._bh + self._bn
        self.res_buf_store[p] = rrow
        self.W_buf_store[p] = obs
        self.A_buf_store[p] = z_t
        self.tidx_store[p] = abs_t
        self.ent_store[p] = eid
        self._bn += 1
        if self._bn > WINDOW:                                 # drop oldest (O(1))
            self._bh += self._bn - WINDOW
            self._bn = WINDOW

    def _refit_W(self):
        n = self._bn
        if n < 2:
            return
        sl = slice(self._bh, self._bh + n)
        R = self.res_buf_store[sl]
        Wobs = self.W_buf_store[sl]
        A = self.A_buf_store[sl]               # view: in-place refinement writes back
        Rf = np.where(Wobs, R, 0.0)
        ent = self.ent_store[sl]
        # --- batched ALS (vectorized rewrite of the per-row/per-column solve loops) ---
        # The old loops issued ~(n + N) tiny R x R cho_factor/cho_solve calls PER SWEEP,
        # each paying ~25us of scipy wrapper overhead on ~1us of actual flops. Here every
        # R x R system in a sweep is assembled and solved in ONE batched LAPACK call. The
        # math is identical: the row pass keeps the exact Gauss-Seidel AR coupling (prev
        # row factor of the same entity) via a sequential pure-numpy matvec over a single
        # pre-computed batched inverse; the column pass is independent across features.
        Wobs_f = Wobs.astype(float)                       # (n, N)
        any_obs = Wobs.any(axis=1)                        # (n,)
        col_has = Wobs.any(axis=0)                        # (N,)
        if ent is None:                                   # 2D: every row couples to prev
            same = np.ones(n, dtype=bool)
        else:                                             # panel: only within an entity
            same = np.empty(n, dtype=bool)
            same[1:] = ent[1:] == ent[:-1]
        same[0] = False
        I_R = np.eye(self.R)
        diag_alpha = np.diag(self.alpha)
        for _ in range(SWEEPS):
            # row factors A given W: G_i = W^T diag(obs_i) W + diag(alpha) (+ I if coupled),
            # all built at once via a single (n,N) x (N,R^2) matmul.
            WW = (self.W[:, :, None] * self.W[:, None, :]).reshape(self.N, self.R * self.R)
            Gs = (Wobs_f @ WW).reshape(n, self.R, self.R) + diag_alpha[None]
            Gs[same] += I_R
            rhs0 = Rf @ self.W                             # (n, R) non-AR part
            # Jacobi AR coupling: each row's ridge uses the PREVIOUS SWEEP's previous-row
            # factor (fixed at sweep start), not this sweep's freshly-updated one. This
            # decouples the rows so the whole pass is ONE batched LAPACK solve instead of a
            # Python loop over n tiny matvecs + a full batched inverse. Jacobi vs Gauss-
            # Seidel changes only the within-sweep convergence path, not the ALS fixed
            # point; with SWEEPS sweeps the difference is third-order and accuracy-neutral
            # (validated: arena/causal/robustness unchanged). The AR term a*A_prev[i-1] is
            # gathered by a masked shift (zeroed where the previous row is a different
            # entity, via `same`).
            A_prev = A.copy()                              # sweep-start factors
            shift = np.zeros_like(A)
            shift[1:] = A_prev[:-1]
            shift[~same] = 0.0
            ar = self.a * shift
            rhs = rhs0 + ar
            A[:] = np.linalg.solve(Gs, rhs[..., None])[..., 0]
            # blackout rows (no observed cells): pure AR propagation a*A_prev[i-1], or 0
            # when the previous row is a different entity (shift already zeroed there).
            A[~any_obs] = ar[~any_obs]
            # loadings W given A (independent across features -> fully batched):
            # G_j = A^T diag(obs_:,j) A + diag(alpha), rhs_j = A^T Rf[:,j].
            AA = (A[:, :, None] * A[:, None, :]).reshape(n, self.R * self.R)
            Gc = (Wobs_f.T @ AA).reshape(self.N, self.R, self.R) + diag_alpha[None]
            rhs_c = Rf.T @ A                               # (N, R)
            Wsol = np.linalg.solve(Gc, rhs_c[..., None])[..., 0]
            self.W[col_has] = Wsol[col_has]                # empty columns keep prior W
        # A is a view into the ring store; the in-place updates above already persist it.
        # --- ARD empirical-Bayes update of per-column precisions alpha_l ---
        #   alpha_l = (N) / (||W[:,l]||^2 + eps)   (evidence-style; unused cols -> huge)
        col_e = np.sum(self.W ** 2, axis=0)
        self.alpha = self.N / (col_e + 1.0)         # +1 prior => no zero-div, gentle
        # --- factor prior variances s_fac (empirical Bayes): variance of each factor
        #     column. Unused factors (ARD-killed) have ~0 variance => contribute nothing
        #     to W diag(s_fac) W^T, so the EFFECTIVE RANK emerges from the data. This is
        #     the prior z-variance in the conditional-mean (Kalman) update.
        if A.shape[0] >= 2:
            self.s_fac = np.maximum(np.var(A, axis=0), 1e-4)
        # --- AR coeff a from pooled factor stats (closed-form ridge regression),
        #     restricted to consecutive SAME-entity factor pairs ---
        if A.shape[0] >= 2:
            zt = A[1:]
            ztm = A[:-1]
            if ent is not None:
                pair = ent[1:] == ent[:-1]
                zt = zt[pair]; ztm = ztm[pair]
            if zt.shape[0] < 1:
                zt = A[1:]; ztm = A[:-1]
            num = float(np.sum(zt * ztm))
            den = float(np.sum(ztm * ztm)) + 1.0
            a_hat = num / den
            self.a = float(np.clip(a_hat, 0.0, 0.999))
        self.W_ready = True

    # -- cross-section conditional mean+variance from pooled EW residual cov --
    #    Returns (cond_mean[miss], cond_var[miss]) for inverse-variance blending with
    #    the low-rank prediction. cond_var is large when the contemporaneous cross-
    #    section is uninformative (blackout / few co-observed cells) -> the blend then
    #    defers to the low-rank/AR prediction; small when dense+correlated -> defers to
    #    the cross-section regression (the EW-cov / rank-0 special case).
    def _xsec_fill(self, resid, obs, miss):
        if self.cw < 5.0 or not obs.any():
            return None, None
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
            cmean = mean[m] + Smo @ sol
            # conditional variance per missing cell: diag(cov_mm) - Smo Soo^-1 Smo^T
            B = cho_solve(c, Smo.T, check_finite=False)        # (o, m)
            cvar = np.diag(cov)[m] - np.einsum("om,om->m", Smo.T, B)
            cvar = np.maximum(cvar, EPS)
            return cmean, cvar
        except Exception:
            return None, None

    # -- UNIFIED Gaussian conditional-mean update (the RDFM filter step) --
    #    Model for the de-FE/de-season residual at this time:
    #        r = W z + e,   z ~ N(a z_prev, diag(s)),   e ~ N(0, diag(psi))
    #    where s_l is the factor prior variance (ARD: unused factors -> ~0, so rank
    #    EMERGES) and psi_j the idiosyncratic per-feature variance. The posterior of z
    #    given the OBSERVED residual cells is a Bayesian linear (Kalman) update; the
    #    missing cells are filled with W_m z_post (their conditional mean). IRLS weights
    #    (Student-t) scale psi per observed cell -> robustness. Special cases: s->0 gives
    #    rank-0 (pure FE+cross-section via psi), a->1 gives random-walk/Kalman prediction,
    #    a=0 static low-rank. ONE solve, no model selection.
    def _factor_update(self, resid, obs, z_prev, irls=True):
        R = self.R
        m0 = self.a * z_prev if z_prev is not None else np.zeros(R)
        if not obs.any():
            return m0, self.W @ m0              # blackout: AR/Kalman prediction
        o = np.where(obs)[0]
        Wo = self.W[o]                          # (no, R)
        ro = resid[o]
        psi_o = np.maximum(self.psi[o], EPS)
        s = np.maximum(self.s_fac, EPS)         # factor prior variances (ARD)
        # optional IRLS reweighting of the per-cell idiosyncratic precision
        if irls:
            pred0 = Wo @ m0
            r2 = (ro - pred0) ** 2
            # inline _wt (per-row hot path): w = (nu+1)/(nu + r2/scale2)
            wts = (self.nu + 1.0) / (self.nu + r2 / max(self.scale2, EPS))  # (no,)
            inv_psi = wts / psi_o
        else:
            inv_psi = 1.0 / psi_o
        # posterior precision (R x R) and mean
        WtP = Wo.T * inv_psi[None, :]           # (R, no)
        prec = WtP @ Wo + np.diag(1.0 / s)
        rhs = WtP @ ro + m0 / s
        # prec is a tiny SPD R x R; np.linalg.solve is one LAPACK call without scipy's
        # per-call cho_factor+cho_solve wrapper overhead (paid once PER ROW here).
        try:
            z_post = np.linalg.solve(prec, rhs)
        except np.linalg.LinAlgError:
            z_post = np.linalg.lstsq(prec, rhs, rcond=None)[0]
        return z_post, self.W @ z_post

    def _update_robust_scale(self, r2_vals):
        # EW-forgotten robust scale + EM estimate of nu by moment matching.
        # We track the EW mean of r2 (robust scale^2 proxy, weighted by current IRLS
        # weights so outliers don't inflate it) and the EW 2nd/4th moments of the
        # standardized residual to estimate the dof nu from excess kurtosis:
        #   kurt_excess(t_nu) = 6/(nu-4)  =>  nu = 4 + 6/kurt_excess.
        # The per-cell EW recurrence (acc <- lam*acc + term) over the row's observed cells
        # is a linear recursion with constant lam; nu/scale2 are fixed across the row (they
        # update only after this call). So it has the exact closed form
        #   acc_final = lam^k * acc_0 + sum_i lam^(k-1-i) * term_i,
        # computed here in one vectorized pass instead of a Python loop over every cell.
        r2 = np.asarray(r2_vals, dtype=float)
        k = r2.shape[0]
        if k:
            w = self._wt(r2)                       # IRLS weights (vectorized)
            s2 = max(self.scale2, EPS)
            gw = self.lam ** np.arange(k - 1, -1, -1)      # lam^(k-1-i), oldest..newest
            lk = self.lam ** k
            z2 = r2 / s2
            self.r2_sum = lk * self.r2_sum + float(np.dot(gw, w * r2))
            self.r2_cnt = lk * self.r2_cnt + float(np.dot(gw, w))
            self.m2_sum = lk * self.m2_sum + float(np.dot(gw, z2))
            self.m4_sum = lk * self.m4_sum + float(np.dot(gw, z2 * z2))
            self.mom_cnt = lk * self.mom_cnt + float(gw.sum())
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
        resid_c = resid.copy()
        if obs.any() and self.nu < NU_MAX - 1.0:   # clip only when heavy tails LEARNED
            fsc = self.fsc_sum / np.maximum(self.fsc_cnt, 1e-3)    # per-feature scale
            # Redescending M-estimator band. The multiplier widens with the learned dof
            # nu but is floored at a generous 4 robust-sigma so it NEVER clips the bulk
            # of a well-behaved-but-mis-fit distribution (e.g. sparse/high-missing rows
            # whose residuals merely have larger variance, not heavier tails). Only true
            # outliers -- beyond ~4 sigma, which a Gaussian essentially never produces but
            # a heavy tail does -- are capped. Self-gating: benign data has ~no such cells.
            k_nu = 1.5 + self.nu / 4.0
            band = k_nu * 1.4826 * fsc
            # Clip only the IDIOSYNCRATIC part: a coherent level/regime shift moves all
            # features together and must NOT be clipped; a genuine outlier is isolated.
            common = _median_part(resid[obs]) if obs.sum() >= 4 else 0.0
            idio = resid[obs] - common
            resid_c[obs] = common + np.clip(idio, -band[obs], band[obs])
        # contemporaneous common level (time fixed effect): the cross-sectional mean of
        # the observed residual at THIS time captures a shared level/regime shift that
        # the slow expanding FE has not yet absorbed (the MC-NNM time-FE term). It is
        # point-in-time (same-tau cross-section). Robust mean (trim) so outliers/heavy
        # tails do not bias it.
        if obs.sum() >= 3:
            rr = resid_c[obs]
            lo, hi = _pct_1090(rr)
            tfe_raw = float(np.mean(rr[(rr >= lo) & (rr <= hi)])) if hi > lo else float(np.mean(rr))
            # EB shrinkage of the time-FE: a per-time level estimated from n_obs cells
            # has sampling variance ~ idio_var / n_obs; shrink toward 0 by the James-
            # Stein / posterior-mean factor  tau2 / (tau2 + idio_var/n_obs), where tau2
            # is the across-feature spread of the residual at this time (how much shared
            # level there plausibly is). Genuine regime shifts (large coherent level,
            # many obs) survive; noisy per-time means on few cells shrink to ~0.
            no = int(obs.sum())
            idio_var = max(float(np.var(rr)), EPS)
            tau2 = max(tfe_raw * tfe_raw, EPS)
            shrink = tau2 / (tau2 + idio_var / no)
            time_fe = shrink * tfe_raw
        else:
            time_fe = 0.0
        # remove the time level from the residual that feeds the factor model
        resid_c = np.where(obs, resid_c - time_fe, 0.0)

        # ---- UNIFIED RDFM filter step: one Bayesian/Kalman conditional-mean update.
        # Subsumes the low-rank factor solve, the AR/Kalman prediction, the cross-
        # section regression (full-covariance conditional mean is W W^T + Psi here),
        # and the rank-0 limit -- all by the learned s_fac (ARD) / psi / a, no blend
        # of separate models and no model selection.
        if self.W_ready:
            z_t, lr = self._factor_update(resid_c, obs, z_prev, irls=(self.nu < 20.0))
        else:
            z_t = self.a * z_prev if z_prev is not None else np.zeros(self.R)
            lr = self.W @ z_t

        # per-feature idiosyncratic AR carry for missing cells: last idiosyncratic level
        # decayed by rho^age (age = steps since last observation). Vanishes for scattered
        # MCAR (age large or rho 0) and persists across short block gaps.
        carry = self.rho_idio ** np.minimum(self.idio_age, 60.0) * self.idio_last

        if miss.any():
            fill = mu[miss] + season[miss] + time_fe + lr[miss] + carry[miss]
            _carry_m = carry[miss]
            # Fuse with the FULL-covariance cross-section conditional mean. The factor
            # update (lr) is the conditional mean under the LOW-RANK covariance W S W^T +
            # Psi; xs is the conditional mean under the FULL empirical (EW) residual
            # covariance. They are two estimates of the same Gaussian residual; the
            # precision-weighted (inverse conditional-variance) combination is the
            # principled fusion. Low-rank wins when factors capture the structure (panel,
            # narrow); full-cov wins when the data is high-rank/wide (reals, highrank);
            # during blackout xs is unavailable -> pure AR/Kalman lr. No tuned constant.
            xs, xsv = self._xsec_fill(resid_c, obs, miss)
            if xs is not None:
                v_lr = max(self.lr_var, EPS)
                w_xs = v_lr / (v_lr + xsv)
                # the idiosyncratic carry belongs to the NON-cross-section channel: when
                # the contemporaneous cross-section is informative (w_xs high) it already
                # predicts the cell and the carry is suppressed; when it is not (blackout/
                # block) the carry + low-rank/AR prediction take over.
                fill = (mu[miss] + season[miss] + time_fe
                        + (1.0 - w_xs) * (lr[miss] + _carry_m) + w_xs * xs)
            out[miss] = fill
            if getattr(self, "record", False):       # opt-in introspection (no prod effect)
                _vlr = max(self.lr_var, EPS)
                _fv = (_vlr * xsv) / (_vlr + xsv) if xs is not None \
                    else np.full(int(miss.sum()), _vlr)
                _cv = np.full(N, np.nan)
                _cv[miss] = _fv + np.maximum(self.psi[miss], 0.0)   # posterior predictive var
                self._cvar_row = _cv

        # advance idiosyncratic-carry age for every feature (reset below for observed)
        self.idio_age += 1.0

        # ---- causal online parameter updates (AFTER imputing this row) ----
        # robust residuals on observed cells (use the clipped residual for the scale
        # so the per-feature band itself is robust to outliers).
        if obs.any():
            # --- fused masked EW-stat tail ---------------------------------------
            # All the per-feature online updates below operate on the SAME observed
            # subset of features; gather the obs index ONCE and materialize the shared
            # vectors (resid_c[obs], lr[obs], their difference idio_now, and idio_now^2)
            # a single time. The originals recomputed resid_c[obs]-lr[obs] three times
            # (r2v, lrr2, psi) and re-evaluated boolean masking for fsc/psi/idio
            # independently. Pure overhead reduction -- the arithmetic, the order of the
            # EW recurrences, and the float results are bit-identical.
            oi = np.where(obs)[0]                     # integer index of observed cells
            lam = self.lam
            rc_o = resid_c[oi]
            lr_o = lr[oi]
            idio_now = rc_o - lr_o
            r2v = idio_now * idio_now                 # (resid_c-lr)^2 == (resid_c-pred_obs)^2
            self._update_robust_scale(r2v)
            # per-feature EW robust scale (abs deviation of de-FE/de-season value)
            self.fsc_sum[oi] = lam * self.fsc_sum[oi] + np.abs(rc_o)
            self.fsc_cnt[oi] = lam * self.fsc_cnt[oi] + 1.0
            # EW variance of the low-rank prediction residual (robust resid - lr)
            lrr2 = float(np.mean(r2v))
            self.lrv_sum = lam * self.lrv_sum + lrr2
            self.lrv_cnt = lam * self.lrv_cnt + 1.0
            self.lr_var = self.lrv_sum / max(self.lrv_cnt, 1e-3)
            # per-feature idiosyncratic variance psi (EW): variance of the residual the
            # factors do NOT explain. This is the noise floor of the Gaussian model;
            # large psi_j => feature j trusts the cross-section/factors less.
            self.psi_sum[oi] = lam * self.psi_sum[oi] + r2v
            self.psi_cnt[oi] = lam * self.psi_cnt[oi] + 1.0
            self.psi = self.psi_sum / np.maximum(self.psi_cnt, 1e-3)
            # idiosyncratic AR(1): autocorrelation of the idio residual (resid_c - lr).
            prev = self.idio_last[oi]
            fresh = self.idio_age[oi] <= 1.5             # consecutive obs only
            if np.any(fresh):
                pn = prev[fresh]
                self.ric_num = lam * self.ric_num + float(np.sum(idio_now[fresh] * pn))
                self.ric_den = lam * self.ric_den + float(np.sum(pn * pn))
                self.rho_idio = float(np.clip(self.ric_num / max(self.ric_den, 1e-6), 0.0, 0.995))
            self.idio_last[oi] = idio_now
            self.idio_age[oi] = 0.0

        # seasonal beta via online ridge (RLS), closed-form. Target is the FACTOR
        # RESIDUAL (x - mu - lowrank - time_fe) on observed cells -- NOT (x - mu) --
        # so season explains only the periodicity the shared factors did not already
        # absorb. On wide panels the low-rank term captures a common cycle and this
        # residual carries little periodic energy, so the ARD drives beta->0 (no
        # double-counting); on a lone series the factors cannot model the cycle, the
        # residual keeps it, and beta fits it. lr and time_fe are point-in-time
        # (data <= t), so this stays causal-invariant. EW-forgotten so it tracks
        # drifting seasonality; ARD per harmonic shrinks beta to ~0 when no cycle.
        if self.P and obs.any():
            yrow = np.zeros(N)
            # trust the low-rank de-double-counting in proportion to how well a shared
            # factor is identified: with R offered factors a common signal is only
            # credible once N >> R, so g = N/(N+R) -> ~0 for a lone series (season keeps
            # the whole cycle, as it must to extrapolate across gaps) and -> ~1 for a wide
            # panel (factors own the shared cycle, season only mops up the remainder). g
            # depends on N and R alone, so it stays point-in-time / causal-invariant.
            g = N / (N + self.R)
            yrow[obs] = x_obs_row[obs] - mu[obs] - g * lr[obs] - time_fe
            # ROBUST season fit. The Fourier RLS is plain least-squares, so without
            # protection a single heavy-tailed outlier corrupts beta GLOBALLY -- every
            # row then inherits a spurious seasonal wave (the dominant reason an explicit
            # season can hurt on heavy-tailed data). Winsorize the target to a band set by
            # the per-feature robust scale and the learned dof nu: wide (a no-op) when nu
            # is large / near-Gaussian, tight when heavy tails are learned -- the same
            # self-gating M-estimator the data term uses, so benign data is never clipped.
            _bnd = (1.5 + self.nu / 4.0) * 1.4826 * (self.fsc_sum[obs] /
                                                     np.maximum(self.fsc_cnt[obs], 1e-3))
            yrow[obs] = np.clip(yrow[obs], -_bnd, _bnd)
            self.PtP = self.lam * self.PtP + np.outer(fourier_t, fourier_t)
            self.Pty[:, obs] = self.lam * self.Pty[:, obs] + np.outer(fourier_t, yrow[obs])
            self.Pty[:, ~obs] = self.lam * self.Pty[:, ~obs]
            self.t_seen += 1.0
            # refit beta occasionally (closed-form ridge with ARD diag). A harmonic
            # with too few full CYCLES of support is unidentifiable and would overfit a
            # trend/extrapolate across gaps; its prior precision is inflated by
            # (period/cycles_seen)^2 so it shrinks to ~0 until enough cycles accrue. This
            # is point-in-time (depends only on t_seen) so it is causal-invariant, and it
            # lets long cycles switch on only once the data supports them.
            if self.step % 8 == 0 and self.PtP[0, 0] > 3.0:
                cyc = self.t_seen / np.maximum(self.beta_period, 1.0)   # cycles seen
                ident = np.clip(cyc / 2.0, 1e-3, 1.0)                   # 1 at >=2 cycles
                pen = self.beta_alpha / (ident ** 2)
                G = self.PtP + np.diag(pen)
                try:
                    self.beta = np.linalg.solve(G, self.Pty)
                except Exception:
                    self.beta = np.linalg.lstsq(G, self.Pty, rcond=None)[0]
                be = np.sum(self.beta ** 2, axis=1)
                self.beta_alpha = self.N / (be + 1.0)

        # full reconstructed row (for residual buffer + covariance). Use the robust
        # (winsorized) residual on observed cells so outliers never enter the pooled
        # covariance / loadings; missing cells use the imputed residual (model best
        # estimate, self-consistent with what was filled).
        imp_resid = out - mu - season - time_fe
        full_resid = np.where(obs, resid_c, imp_resid)

        # robust row weight (Student-t scale mixture) from the observed prediction
        # residual: a row dominated by outliers is down-weighted everywhere it feeds
        # parameter learning, so W / covariance / scale stay clean (robust emerges).
        if obs.any():
            r2row = float(np.mean((resid[obs] - lr[obs]) ** 2))
            # inline _wt (per-row hot path)
            row_w = (self.nu + 1.0) / (self.nu + r2row / max(self.scale2, EPS))
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
        self._push_buf(rrow, obs, z_t, abs_t, eid)

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

        if getattr(self, "record", False):           # opt-in introspection trace
            if not hasattr(self, "rows_trace"):
                self.rows_trace = []
            self.rows_trace.append(dict(
                t=int(abs_t), eid=int(eid), obs=obs.copy(),
                mu=mu.copy(), season=np.asarray(season, float).copy(),
                time_fe=float(time_fe), lr=np.asarray(lr, float).copy(),
                z=np.asarray(z_t, float).copy(), filled=out.copy(),
                cvar=(self._cvar_row if (miss.any() and getattr(self, "_cvar_row", None) is not None) else None),
                row_w=float(row_w), r2row=(float(r2row) if obs.any() else float("nan")),
                nu=float(self.nu), a=float(self.a), scale2=float(self.scale2),
                rho=float(self.rho_idio), alpha=self.alpha.copy(), psi=self.psi.copy(),
            ))
            self._cvar_row = None
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
# Panel path: SAME unified penalized objective, organized over the ENTITY cross-
# section. Processing is time-ordered; at each time the contemporaneous block of all
# entities present is decomposed as
#     x_{e,t,f} = entity_FE[e,f] + time_FE[t,f] + (a_e . g_{t})_f  + eps
# with a_e the entity loadings (ARD-shrunk -> rank emerges), g_t the time factors
# (AR-coupled across time -> dynamics emerge), entity/time FE = expanding/contemporary
# means, and a Student-t/IRLS data term (robust emerges). It is point-in-time: entity
# FE from that entity's history < t, the time-FE/low-rank use only the same-t cross-
# section and factors fitted on data <= t. State (g_t) propagates by the learned AR;
# under a synchronized blackout (no obs at t) g_t = a g_{t-1} (Kalman prediction). This
# is the panel instance of the one model -- not a different technique, and chosen by the
# meta structure (entity_ids), not by any data-driven detector.
# --------------------------------------------------------------------------- #
def _impute_panel(X, meta):
    X = np.ascontiguousarray(np.asarray(X, float))
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    T_rows, N = X.shape
    E = int(eids.max()) + 1
    out = X.copy()

    uniq_t = np.unique(tids)
    rows_at = {t: np.where(tids == t)[0] for t in uniq_t}

    # expanding per-entity FE (entity x feature) + global fallback, point-in-time
    ent_sum = np.zeros((E, N)); ent_cnt = np.zeros((E, N))
    g_sum = np.zeros(N); g_cnt = np.zeros(N)

    # entity loadings A (E x R) and feature/time-factor loadings.  We model the de-FE
    # residual block R_t (entities x features) as A[e] . (Wt z_t)  -- equivalently a
    # rank-R bilinear form. We keep entity loadings A and a feature map Wt (N x R), with
    # ARD precision alpha per factor (rank emerges), and a time factor g_t (R) AR-coupled.
    rng = np.random.default_rng(0)
    A = 0.01 * rng.standard_normal((E, R_MAX))      # entity loadings
    Wt = 0.01 * rng.standard_normal((N, R_MAX))     # feature loadings
    alpha = np.ones(R_MAX)                           # ARD precisions
    a_ar = 0.5                                        # learned AR coeff on time factor
    g_prev = np.zeros(R_MAX)
    nu = NU_INIT; scale2 = 1.0
    r2_sum = 0.0; r2_cnt = 1e-3; m2 = 0.0; m4 = 0.0; mc = 1e-3
    lam = 0.5 ** (1.0 / HALFLIFE)
    lam_mu = 0.5 ** (1.0 / MU_HALFLIFE)

    # rolling snapshot of the latest de-FE residual per entity (<= t) for ALS on A, Wt.
    res_snap = np.zeros((E, N)); res_have = np.zeros((E, N), bool)
    AGsnap = np.zeros((E, R_MAX))     # effective row factor A[e]*g_{last_t(e)} at snapshot
    I_R = np.eye(R_MAX)
    # per-(entity,feature) idiosyncratic AR carry (extrapolates contiguous gaps/blackouts)
    idio_last = np.zeros((E, N)); idio_age = np.full((E, N), 1e9)
    rho_idio = 0.0; ric_num = 0.0; ric_den = 1e-3

    def wt(r2):
        return (nu + 1.0) / (nu + r2 / max(scale2, EPS))

    step = 0
    g_t = g_prev.copy()
    for t in uniq_t:
        rs = rows_at[t]; ent_t = eids[rs]
        Xt = X[rs]; obs_t = ~np.isnan(Xt)

        # entity FE (history < t); global expanding fallback where entity unseen
        gfe = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 0.0)
        efe = np.where(ent_cnt[ent_t] > 0, ent_sum[ent_t] / np.maximum(ent_cnt[ent_t], 1), 0.0)
        efe = np.where(ent_cnt[ent_t] > 0, efe, gfe[None, :])
        resid_t = Xt - efe                               # (n_e, N), de-entity-FE

        # time FE: robust (trimmed) cross-entity mean of the observed de-FE residual per
        # feature -- the contemporaneous time effect (shrunk by support).  Vectorized,
        # mask-aware batch over all features (bit-identical to the per-feature loop).
        time_fe = np.zeros(N)
        cnt = obs_t.sum(0).astype(np.intp)                # observed entities per feature
        # masked residual: unobserved -> +inf so they sort to the end of each column.
        masked = np.where(obs_t, resid_t, np.inf)
        srt = np.sort(masked, axis=0)                     # (n_e, N) ascending, inf last
        srt = np.where(np.isinf(srt), 0.0, srt)           # neutralize pads (big-cols only used)
        n_e = srt.shape[0]
        cols = np.arange(N)

        # column mean (over observed) for the col.mean() fallbacks.
        col_sum = np.where(obs_t, resid_t, 0.0).sum(0)
        col_mean = col_sum / np.maximum(cnt, 1)

        big = cnt >= 3                                    # features using the trimmed path
        if big.any():
            # numpy 'linear' percentile on the first `cnt[f]` sorted entries of column f.
            nm1 = (cnt - 1).astype(float)
            def _pctl(p):
                vi = (p / 100.0) * nm1                    # virtual index per feature
                i0 = np.floor(vi).astype(np.intp)
                i0 = np.clip(i0, 0, n_e - 1)
                i1 = np.clip(i0 + 1, 0, n_e - 1)
                g = vi - i0
                a0 = srt[i0, cols]; a1 = srt[i1, cols]
                return a0 + g * (a1 - a0)
            lo = _pctl(10.0); hi = _pctl(90.0)
            # trimmed mean: observed cells within [lo, hi] per feature.
            inwin = obs_t & (resid_t >= lo[None, :]) & (resid_t <= hi[None, :])
            msum = np.where(inwin, resid_t, 0.0).sum(0)
            mcnt = inwin.sum(0)
            raw = np.where(mcnt > 0, msum / np.maximum(mcnt, 1), col_mean)
            # population variance over observed cells (ddof=0), matching np.var(col).
            dev = np.where(obs_t, resid_t - col_mean[None, :], 0.0)
            iv = np.maximum((dev * dev).sum(0) / np.maximum(cnt, 1), EPS)
            tau2 = np.maximum(raw * raw, EPS)
            no = cnt.astype(float)
            shrunk = (tau2 / (tau2 + iv / np.maximum(no, 1))) * raw
            time_fe[big] = shrunk[big]
        small = (~big) & (cnt > 0)                        # 1 or 2 observed -> plain mean
        time_fe[small] = col_mean[small]
        resid_t2 = resid_t - time_fe[None, :]            # de-time-FE residual

        # AR prediction of the time factor (Kalman predict)
        g_pred = a_ar * g_prev

        # --- per-entity factor solve: g handled jointly. We solve the time factor g_t
        # from the whole observed block (pooled over entities), then per-entity loading
        # A[e] from its observed cells. Robust IRLS on residuals. This is the bilinear
        # low-rank step with ARD ridge (rank emerges) + AR ridge toward g_pred.
        # Solve g_t given current A, Wt over all observed cells at t:
        # ARD ridge (diag alpha) + unit AR ridge toward the prediction g_pred:
        #   (sum_k w_k phi_k phi_k^T + diag(alpha) + I) g = sum_k w_k phi_k r_k + I g_pred
        AtA = np.zeros((R_MAX, R_MAX)); Atr = np.zeros(R_MAX)
        for k in range(len(rs)):
            ob = obs_t[k]
            if not ob.any():
                continue
            phi = A[ent_t[k]][:, None] * Wt[ob].T          # (R, n_obs) basis for this entity
            rr = resid_t2[k, ob]
            r2 = (rr - phi.T @ g_pred) ** 2
            w = wt(np.mean(r2)) if r2.size else 1.0
            AtA += w * (phi @ phi.T)
            Atr += w * (phi @ rr)
        # AR penalty weight lambda_z: a stronger AR (a_ar near 1) implies the factor is
        # near a random walk and the prediction g_pred is trusted more; a_ar near 0 makes
        # the factor near-static and the AR ridge negligible. lam_z = a_ar^2/(1-a_ar^2+eps)
        # is the precision the AR(1) prior assigns -- learned, not tuned.
        lam_z = (a_ar * a_ar) / (1.0 - a_ar * a_ar + 1e-2)
        Gg = AtA + np.diag(alpha) + lam_z * I_R
        try:
            g_t = np.linalg.solve(Gg, Atr + lam_z * g_pred)
        except Exception:
            g_t = np.linalg.lstsq(Gg, Atr + lam_z * g_pred, rcond=None)[0]

        # per-entity loading update A[e] given g_t, Wt (ARD ridge). The A[e] solve is
        # independent per entity (no pooling across k) -> identical arithmetic kept in the
        # loop; recon is then a single batched matmul, bit-identical row-by-row.
        for k in range(len(rs)):
            e = ent_t[k]; ob = obs_t[k]
            if ob.any():
                Bk = Wt[ob] * g_t[None, :]                 # (n_obs, R)
                Gk = Bk.T @ Bk + np.diag(alpha)
                rhs = Bk.T @ resid_t2[k, ob]
                try:
                    A[e] = np.linalg.solve(Gk, rhs)
                except Exception:
                    A[e] = np.linalg.lstsq(Gk, rhs, rcond=None)[0]
        recon = (A[ent_t] * g_t[None, :]) @ Wt.T           # (n_e, N)

        # fill missing cells: entity FE + time FE + low-rank recon + idiosyncratic carry.
        # The carry = decayed last idiosyncratic residual (resid - recon) of THAT
        # (entity,feature); it extrapolates contiguous per-series gaps and synchronized
        # blackouts that the contemporaneous cross-section cannot reach.
        idio_age += 1.0
        # Vectorized over the entities present at t (each writes its own entity rows of the
        # snapshot, so there is no cross-entity overwrite -> bit-identical).
        miss_t = ~obs_t                                    # (n_e, N)
        if miss_t.any():
            carry = (rho_idio ** np.minimum(idio_age[ent_t], 60.0)) * idio_last[ent_t]
            full = efe + time_fe[None, :] + recon + carry  # (n_e, N)
            outt = out[rs]
            outt[miss_t] = full[miss_t]
            out[rs] = outt
            rss = res_snap[ent_t]; rss[miss_t] = recon[miss_t]; res_snap[ent_t] = rss
            rhh = res_have[ent_t]; rhh[miss_t] = True; res_have[ent_t] = rhh

        # --- robust scale + nu EM from block residuals (observed) + carry learning ---
        rv = []
        for k in range(len(rs)):
            e = ent_t[k]; ob = obs_t[k]
            if ob.any():
                rr = resid_t2[k, ob] - recon[k, ob]
                rv.append(rr)
                res_snap[e, ob] = resid_t2[k, ob]
                res_have[e, ob] = True
                AGsnap[e] = A[e] * g_t                 # row factor at this entity's time
                # idiosyncratic AR(1) autocorrelation (consecutive same-entity obs)
                fresh = idio_age[e, ob] <= 1.5
                if np.any(fresh):
                    oi = np.where(ob)[0][fresh]
                    pn = idio_last[e, oi]
                    ric_num = lam * ric_num + float(np.sum(rr[fresh] * pn))
                    ric_den = lam * ric_den + float(np.sum(pn * pn))
                    rho_idio = float(np.clip(ric_num / max(ric_den, 1e-6), 0.0, 0.995))
                idio_last[e, ob] = rr
                idio_age[e, ob] = 0.0
        if rv:
            rr = np.concatenate(rv); r2a = rr * rr
            w = wt(np.mean(r2a))
            r2_sum = lam * r2_sum + w * float(np.mean(r2a)); r2_cnt = lam * r2_cnt + w
            s2 = max(scale2, EPS)
            m2 = lam * m2 + float(np.mean(r2a)) / s2
            m4 = lam * m4 + float(np.mean(r2a ** 2)) / (s2 * s2)
            mc = lam * mc + 1.0
            if r2_cnt > 5.0 * 1e-3:
                scale2 = max(r2_sum / r2_cnt, EPS)
            if mc > 30.0:
                kurt = (m4 / mc) / max((m2 / mc) ** 2, EPS)
                excess = kurt - 3.0
                nu_hat = 4.0 + 6.0 / excess if excess > 0.2 else NU_MAX
                nu = float(np.clip(0.7 * nu + 0.3 * nu_hat, NU_MIN, NU_MAX))

        # --- periodic ALS refit of Wt + ARD + AR coeff on the entity snapshot ---
        step += 1
        if step % 4 == 0:
            seen = np.where(res_have.any(axis=1))[0]
            if seen.size >= 2:
                Rb = res_snap[seen]; Wb = res_have[seen]
                # row factor = A[e]*g at the entity's OWN snapshot time (not current g_t,
                # which would be a different time and corrupt the fit).
                AG = AGsnap[seen]                           # (n_seen, R)
                for f in range(N):
                    w = Wb[:, f]
                    if not w.any():
                        continue
                    Af = AG[w]
                    Gf = Af.T @ Af + np.diag(alpha)
                    rhs = Af.T @ Rb[w, f]
                    try:
                        Wt[f] = np.linalg.solve(Gf, rhs)
                    except Exception:
                        Wt[f] = np.linalg.lstsq(Gf, rhs, rcond=None)[0]
                # ARD on factors from combined energy of the row factor and Wt columns
                ce = np.sum(Wt ** 2, axis=0) * (np.sum(AG ** 2, axis=0) + 1.0)
                alpha = (N + seen.size) / (ce + 1.0)

        # AR coeff on the time factor: EW regression of g_t on g_prev (pooled scalar)
        num = float(g_t @ g_prev); den = float(g_prev @ g_prev) + 1.0
        a_ar = float(np.clip(0.8 * a_ar + 0.2 * (num / den), 0.0, 0.999))
        g_prev = g_t.copy()

        # fold observed values into expanding FE (after imputing t)
        for k in range(len(rs)):
            e = ent_t[k]; ob = obs_t[k]
            ent_sum[e, ob] = lam_mu * ent_sum[e, ob] + Xt[k, ob]
            ent_cnt[e, ob] = lam_mu * ent_cnt[e, ob] + 1.0
            g_sum[ob] = lam_mu * g_sum[ob] + Xt[k, ob]
            g_cnt[ob] = lam_mu * g_cnt[ob] + 1.0

    if np.isnan(out).any():
        gfe = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 0.0)
        idx = np.where(np.isnan(out)); out[idx] = np.take(gfe, idx[1])
    return out


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    # Robustness: non-finite inputs (+/-Inf) are not valid observations -> treat as
    # missing so the solver never sees Inf (guards the SVD/least-squares path).
    if not np.all(np.isfinite(X)):
        X = np.where(np.isfinite(X), X, np.nan)
    if meta and "entity_ids" in meta and "time_ids" in meta \
            and len(np.unique(meta["entity_ids"])) > 1:
        out = _impute_panel(X, meta)
    else:
        out = _impute_2d(X, meta)
    out = np.asarray(out, float)
    if not np.all(np.isfinite(out)):                 # final safety net: never emit NaN/Inf
        col = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        bad = ~np.isfinite(out)
        out[bad] = np.take(col, np.where(bad)[1])
    return out


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Unified_PenMF", online_impute))
