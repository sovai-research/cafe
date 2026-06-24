"""
Multivariate-volatility models for Experiment 3.

Three estimators of the one-step-ahead conditional covariance Sigma_t:

  * CCC-GARCH (Bollerslev)   : per-series GARCH variances + CONSTANT sample
                               correlation.  Sigma_t = D_t R D_t.
  * DCC-GARCH (Engle 2002)   : per-series GARCH + a DYNAMIC correlation recursion
                               Q_t = (1-a-b) Qbar + a z z' + b Q_{t-1}.  The standard
                               multivariate-volatility frontier model.
  * CAFE-FV (proposed)       : low-rank-plus-diagonal covariance in CAFE's own form
                               Sigma_t = B diag(hf_t) B' + diag(he_t), where each
                               factor vol hf and idiosyncratic vol he follows the
                               Beta-t-GARCH recursion (Exp 2).  Scales as O(N*K) and
                               ingests ragged gaps natively (masked factor projection).

All three follow the same point-in-time backtest protocol: parameters are estimated
on a training prefix, then the conditional covariance is FILTERED forward over a
disjoint evaluation window using only past returns.  CCC/DCC require complete data
(NaN breaks the univariate filters); CAFE-FV does not.
"""
from __future__ import annotations
import time
import numpy as np

import garch_lib as gl


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _uni_garch_filter(r, omega, alpha, beta):
    """Plain Gaussian GARCH(1,1) one-step variance filter."""
    T = len(r); h = np.empty(T)
    h[0] = omega / max(1 - alpha - beta, 1e-3)
    for t in range(T):
        if t + 1 < T:
            h[t + 1] = omega + alpha * r[t] ** 2 + beta * h[t]
    return h


def _fit_uni_garch(r):
    """MLE a univariate GARCH(1,1) via arch; return (omega,alpha,beta) at data scale."""
    from arch import arch_model
    import warnings
    from arch.utility.exceptions import DataScaleWarning
    s = 100.0 / max(np.std(r), 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataScaleWarning)
        res = arch_model(r * s, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    p = res.params
    # undo the x s rescale: variance scales by s^2, so omega divides by s^2
    return p["omega"] / s ** 2, p["alpha[1]"], p["beta[1]"]


# --------------------------------------------------------------------------- #
# CCC / DCC
# --------------------------------------------------------------------------- #
class DCCModel:
    def __init__(self, dynamic=True):
        self.dynamic = dynamic

    def fit(self, R_train):
        """R_train: (T,N) complete returns. Fit per-series GARCH + (D)CC params."""
        T, N = R_train.shape
        self.N = N
        self.uni = [(_fit_uni_garch(R_train[:, i])) for i in range(N)]
        H = np.empty((T, N))
        for i, (o, a, b) in enumerate(self.uni):
            H[:, i] = _uni_garch_filter(R_train[:, i], o, a, b)
        Z = R_train / np.sqrt(np.maximum(H, 1e-12))         # standardized resids
        self.Qbar = np.corrcoef(Z, rowvar=False)
        self.Qbar = np.atleast_2d(self.Qbar)
        # last in-sample variances/Q to seed the eval-window filter
        self._H_last = H[-1]
        self._r_last = R_train[-1]
        if self.dynamic:
            self.a, self.b = self._fit_dcc_params(Z)
            self._Q_last = self.Qbar.copy()
            self._z_last = Z[-1]
        return self

    def _fit_dcc_params(self, Z):
        """Two-step DCC: maximize the correlation-part quasi-likelihood over (a,b)."""
        from scipy.optimize import minimize
        T, N = Z.shape
        Qbar = self.Qbar

        def nll(theta):
            a = 1.0 / (1 + np.exp(-theta[0])) * 0.5
            b = 1.0 / (1 + np.exp(-theta[1])) * (0.999 - a)
            Q = Qbar.copy(); ll = 0.0
            for t in range(T):
                d = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
                Rc = Q / np.outer(d, d)
                try:
                    L = np.linalg.cholesky(Rc)
                except np.linalg.LinAlgError:
                    return 1e10
                sol = np.linalg.solve(L, Z[t])
                ll += 2 * np.log(np.diag(L)).sum() + sol @ sol
                z = Z[t]
                Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
            return 0.5 * ll
        res = minimize(nll, np.array([-2.0, 2.0]), method="Nelder-Mead",
                       options=dict(maxiter=300, xatol=1e-4, fatol=1e-4))
        a = 1.0 / (1 + np.exp(-res.x[0])) * 0.5
        b = 1.0 / (1 + np.exp(-res.x[1])) * (0.999 - a)
        return a, b

    def filter_eval(self, R_eval):
        """One-step Sigma_t over the eval window (uses only past returns)."""
        Te, N = R_eval.shape
        Sig = np.empty((Te, N, N))
        H_prev = self._H_last.copy(); r_prev = self._r_last.copy()
        if self.dynamic:
            Q = self._Q_last.copy(); z_prev = self._z_last.copy()
        for t in range(Te):
            H_t = np.array([o + a * r_prev[i] ** 2 + b * H_prev[i]
                            for i, (o, a, b) in enumerate(self.uni)])
            d = np.sqrt(np.maximum(H_t, 1e-12))
            if self.dynamic:
                Q = ((1 - self.a - self.b) * self.Qbar
                     + self.a * np.outer(z_prev, z_prev) + self.b * Q)
                dd = np.sqrt(np.clip(np.diag(Q), 1e-12, None))
                Rc = Q / np.outer(dd, dd)
            else:
                Rc = self.Qbar
            Sig[t] = (d[:, None] * Rc) * d[None, :]
            # advance using the realized return at t (now becomes "past" for t+1)
            r_prev = R_eval[t]; H_prev = H_t
            if self.dynamic:
                z_prev = R_eval[t] / d
        return Sig


# --------------------------------------------------------------------------- #
# CAFE-FV : low-rank-plus-diagonal Beta-t factor volatility (the proposed upgrade)
# --------------------------------------------------------------------------- #
class CafeFV:
    def __init__(self, K=3, fast_idio=True):
        self.K = K
        self.fast_idio = fast_idio       # cheap method-of-moments idio vols -> O(K) MLEs

    def fit(self, R_train, obs_train=None):
        """Estimate loadings B (trailing PCA on de-meaned, gap-aware) and the Beta-t
        params for each factor and each idiosyncratic series, on the training prefix."""
        T, N = R_train.shape
        self.N = N
        if obs_train is None:
            obs_train = ~np.isnan(R_train)
        Rc = np.where(obs_train, R_train, np.nan)
        self.mean = np.nanmean(Rc, axis=0)
        X = np.where(obs_train, Rc - self.mean, 0.0)
        # gap-aware covariance: pairwise available-case, then nearest-SPD projection
        C = self._masked_cov(X, obs_train)
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        self.B = V[:, order[:self.K]] * np.sqrt(np.maximum(w[order[:self.K]], 1e-8))
        # factor series by masked least squares per row, idiosyncratic = resid
        F = self._project(X, obs_train)                      # (T,K)
        recon = F @ self.B.T
        Eps = np.where(obs_train, X - recon, np.nan)
        # Beta-t params per factor and per idiosyncratic series
        self.fac_par = [gl.fit_betat_garch(F[:, k]) for k in range(self.K)]
        self.idio_par = []
        for i in range(N):
            ei = Eps[:, i]; ei = ei[~np.isnan(ei)]
            v = float(np.var(ei)) if len(ei) > 5 else 1.0
            if (not self.fast_idio) and len(ei) > 50:
                self.idio_par.append(gl.fit_betat_garch(ei))
            else:
                # method-of-moments Beta-t-GARCH: fixed RiskMetrics-like reaction/
                # persistence, long-run level matched to the sample variance. O(1) per
                # series -> CAFE-FV needs only K (factor) MLE fits regardless of N.
                a, b = 0.06, 0.90
                self.idio_par.append(dict(omega=v * (1 - a - b), alpha=a, beta=b, nu=8.0))
        # seed states (last training variances) by filtering training factors/idio
        self._seed(F, Eps)
        return self

    def _masked_cov(self, X, obs):
        # pairwise available-case covariance (X already zero-filled where missing)
        S = X.T @ X
        cnt = obs.T.astype(float) @ obs.astype(float)
        C = S / np.maximum(cnt, 1.0)
        # symmetric nearest-PSD: clip negative eigenvalues
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-6 * w.max() if w.max() > 0 else 1e-6, None)
        return (V * w) @ V.T

    def _project(self, X, obs):
        """Masked least-squares factor scores per row: solve (B_o'B_o) f = B_o' x_o."""
        T, N = X.shape
        F = np.zeros((T, self.K))
        BtB_full = self.B.T @ self.B
        for t in range(T):
            o = obs[t]
            if o.sum() < self.K:
                F[t] = 0.0
                continue
            Bo = self.B[o]
            G = Bo.T @ Bo + 1e-6 * np.eye(self.K)
            F[t] = np.linalg.solve(G, Bo.T @ X[t, o])
        return F

    def _seed(self, F, Eps):
        self._hf_last = np.empty(self.K); self._f_last = F[-1].copy()
        for k, p in enumerate(self.fac_par):
            hf = gl.betat_garch_filter(F[:, k], p["omega"], p["alpha"], p["beta"], p["nu"])
            self._hf_last[k] = hf[-1]
        self._he_last = np.empty(self.N); self._e_last = np.zeros(self.N)
        for i, p in enumerate(self.idio_par):
            ei = Eps[:, i].copy()
            last_obs = ei[~np.isnan(ei)]
            ei = np.where(np.isnan(ei), 0.0, ei)
            he = gl.betat_garch_filter(ei, p["omega"], p["alpha"], p["beta"], p["nu"])
            self._he_last[i] = he[-1] if len(last_obs) else p["omega"]
            self._e_last[i] = last_obs[-1] if len(last_obs) else 0.0

    def filter_eval(self, R_eval, obs_eval=None):
        """One-step Sigma_t = B diag(hf) B' + diag(he), gap-aware. Past returns only."""
        Te, N = R_eval.shape
        if obs_eval is None:
            obs_eval = ~np.isnan(R_eval)
        Sig = np.empty((Te, N, N))
        hf = self._hf_last.copy(); f_prev = self._f_last.copy()
        he = self._he_last.copy(); e_prev = self._e_last.copy()
        Xc = np.where(obs_eval, R_eval - self.mean, 0.0)
        for t in range(Te):
            # advance factor & idio Beta-t vols using the previous step's innovations
            for k, p in enumerate(self.fac_par):
                w = (p["nu"] + 1) / (p["nu"] + f_prev[k] ** 2 / max(hf[k], 1e-9))
                hf[k] = p["omega"] + p["alpha"] * w * f_prev[k] ** 2 + p["beta"] * hf[k]
            for i, p in enumerate(self.idio_par):
                w = (p["nu"] + 1) / (p["nu"] + e_prev[i] ** 2 / max(he[i], 1e-9))
                he[i] = p["omega"] + p["alpha"] * w * e_prev[i] ** 2 + p["beta"] * he[i]
            Sig[t] = (self.B * hf) @ self.B.T + np.diag(np.maximum(he, 1e-9))
            # realized innovations at t become "past" for t+1 (gap-aware projection)
            o = obs_eval[t]
            if o.sum() >= self.K:
                Bo = self.B[o]
                G = Bo.T @ Bo + 1e-6 * np.eye(self.K)
                f_prev = np.linalg.solve(G, Bo.T @ Xc[t, o])
            else:
                f_prev = self.fac_decay(f_prev)
            recon = self.B @ f_prev
            e_new = np.where(o, Xc[t] - recon, 0.0)
            # keep last observed idio where missing (carry), else update
            e_prev = np.where(o, e_new, e_prev)
        return Sig

    @staticmethod
    def fac_decay(f):
        return 0.9 * f


# --------------------------------------------------------------------------- #
# simple covariance baselines
# --------------------------------------------------------------------------- #
class EWMACov:
    """RiskMetrics multivariate EWMA: Sigma_t = lam Sigma_{t-1} + (1-lam) r r'."""
    def __init__(self, lam=0.94):
        self.lam = lam

    def fit(self, R_train, obs_train=None):
        Rc = np.nan_to_num(R_train)
        self.N = R_train.shape[1]
        self._S = np.cov(Rc, rowvar=False) + 1e-6 * np.eye(self.N)
        self._r_last = Rc[-1]
        return self

    def filter_eval(self, R_eval, obs_eval=None):
        Te, N = R_eval.shape
        Sig = np.empty((Te, N, N)); S = self._S.copy(); r_prev = self._r_last.copy()
        for t in range(Te):
            S = self.lam * S + (1 - self.lam) * np.outer(r_prev, r_prev)
            Sig[t] = S
            r_prev = np.nan_to_num(R_eval[t])
        return Sig


class SampleCov:
    """Static sample covariance from the training window (the CCC-without-GARCH null)."""
    def fit(self, R_train, obs_train=None):
        Rc = np.nan_to_num(R_train); self.N = R_train.shape[1]
        self._S = np.cov(Rc, rowvar=False) + 1e-6 * np.eye(self.N)
        return self

    def filter_eval(self, R_eval, obs_eval=None):
        Te, N = R_eval.shape
        return np.repeat(self._S[None], Te, axis=0)


# --------------------------------------------------------------------------- #
# high-dimensional STATIC covariance baselines (the real incumbents at large N)
# --------------------------------------------------------------------------- #
def _analytical_nls(X):
    """Ledoit-Wolf (2020) ANALYTICAL nonlinear shrinkage of the sample covariance.
    The state-of-the-art well-conditioned static estimator for N near/above T --
    exactly the regime where DCC/CCC's sample correlation breaks. X: (n,p) de-meaned."""
    n, p = X.shape
    S = (X.T @ X) / n
    lam, u = np.linalg.eigh(S)
    lam = np.maximum(lam, 0.0)
    lam = lam[max(0, p - n):]                          # effective eigenvalues
    L = lam.reshape(-1, 1) * np.ones((1, len(lam)))
    h = n ** (-1 / 3)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3 / 4 / np.sqrt(5)) * np.mean(np.maximum(1 - x ** 2 / 5, 0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hf = (-3 / 10 / np.pi) * x + (3 / 4 / np.sqrt(5) / np.pi) * (1 - x ** 2 / 5) * \
            np.log(np.abs((np.sqrt(5) - x) / (np.sqrt(5) + x)))
    Hf[np.abs(x) == np.sqrt(5)] = (-3 / 10 / np.pi) * x[np.abs(x) == np.sqrt(5)]
    Hftilde = np.mean(Hf / H, axis=1)
    if p <= n:
        dtilde = lam / ((np.pi * (p / n) * lam * ftilde) ** 2 +
                        (1 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2)
    else:
        Hftilde0 = (1 / np.pi) * (3 / 10 / h ** 2 + 3 / 4 / np.sqrt(5) / h * (1 - 1 / 5 / h ** 2)
                                  * np.log((1 + np.sqrt(5) * h) / (1 - np.sqrt(5) * h))) \
            * np.mean(1 / lam)
        dtilde0 = 1 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lam / (np.pi ** 2 * lam ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate([dtilde0 * np.ones(p - n), dtilde1])
    return (u * dtilde) @ u.T


class NLSCov:
    """Analytical nonlinear-shrinkage covariance (Ledoit-Wolf 2020), static. The
    genuine high-dimensional incumbent -- well-conditioned where the sample/DCC
    correlation is not -- but with NO volatility timing (one matrix for the window)."""
    def fit(self, R_train, obs_train=None):
        Rc = np.nan_to_num(R_train); self.N = R_train.shape[1]
        X = Rc - Rc.mean(0)
        self._S = _analytical_nls(X) + 1e-8 * np.eye(self.N)
        return self

    def filter_eval(self, R_eval, obs_eval=None):
        return np.repeat(self._S[None], len(R_eval), axis=0)


class LWLinearCov:
    """Ledoit-Wolf LINEAR shrinkage toward a scaled identity (sklearn), static."""
    def fit(self, R_train, obs_train=None):
        from sklearn.covariance import LedoitWolf
        Rc = np.nan_to_num(R_train); self.N = R_train.shape[1]
        self._S = LedoitWolf().fit(Rc - Rc.mean(0)).covariance_ + 1e-8 * np.eye(self.N)
        return self

    def filter_eval(self, R_eval, obs_eval=None):
        return np.repeat(self._S[None], len(R_eval), axis=0)


class StaticFactorCov(CafeFV):
    """ABLATION: CAFE-FV's low-rank-plus-diagonal STRUCTURE with STATIC factor and
    idiosyncratic variances -- the Beta-t score-driven dynamics switched OFF. Same
    loadings, same gap-aware projection, but Sigma is constant over the window. The
    gap between this and CAFE-FV is exactly the contribution of the score recursion;
    the gap between this and DCC/NLS is the contribution of the low-rank structure."""
    def fit(self, R_train, obs_train=None):
        super().fit(R_train, obs_train)
        if obs_train is None:
            obs_train = ~np.isnan(R_train)
        X = np.where(obs_train, R_train - self.mean, 0.0)
        F = self._project(X, obs_train)
        hf = np.maximum(F.var(axis=0), 1e-9)
        recon = F @ self.B.T
        Eps = np.where(obs_train, X - recon, np.nan)
        he = np.array([np.nanvar(Eps[:, i]) for i in range(self.N)])
        he = np.where(np.isfinite(he), he, 1.0)
        self._Sig_static = (self.B * hf) @ self.B.T + np.diag(np.maximum(he, 1e-9))
        return self

    def filter_eval(self, R_eval, obs_eval=None):
        return np.repeat(self._Sig_static[None], len(R_eval), axis=0)


# --------------------------------------------------------------------------- #
# evaluation metrics
# --------------------------------------------------------------------------- #
def mv_portfolio_realized_var(Sig, R_eval):
    """Global minimum-variance portfolio: w_t = Sig_t^-1 1 / (1' Sig_t^-1 1).
    Out-of-sample realized variance = mean (w_t' r_t)^2 -- the standard MGARCH
    economic loss (lower = a better covariance forecast). Returns (rvar, ok_frac)."""
    Te, N = R_eval.shape
    ones = np.ones(N)
    pr = []
    for t in range(Te):
        if np.any(~np.isfinite(R_eval[t])):
            continue
        try:
            inv1 = np.linalg.solve(Sig[t] + 1e-8 * np.eye(N), ones)
        except np.linalg.LinAlgError:
            continue
        w = inv1 / (ones @ inv1)
        pr.append((w @ R_eval[t]) ** 2)
    if not pr:
        return np.nan, 0.0
    return float(np.mean(pr)), len(pr) / Te


def cov_qlike(Sig, Sig_true):
    """Multivariate QLIKE: mean[ tr(Sig^-1 S_true) - logdet(Sig^-1 S_true) - N ]."""
    Te, N, _ = Sig.shape
    vals = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for t in range(Te):
            try:
                iS = np.linalg.inv(Sig[t] + 1e-8 * np.eye(N))
            except np.linalg.LinAlgError:
                continue
            M = iS @ Sig_true[t]
            sign, ld = np.linalg.slogdet(M)
            if sign <= 0 or not np.isfinite(ld):
                continue
            vals.append(np.trace(M) - ld - N)
    return float(np.mean(vals)) if vals else np.nan
