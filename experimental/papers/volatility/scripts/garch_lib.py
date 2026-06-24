"""
Shared utilities for the CAFE-as-volatility-model experiments.

Everything here is strictly point-in-time (one-step-ahead): a forecast h_t for the
variance of return r_t is formed using ONLY information up to t-1. That mirrors how
both GARCH and CAFE actually operate, so the comparison is apples-to-apples.

Contents
--------
  * GARCH(1,1) data-generating processes (Gaussian and Student-t innovations),
    plus a factor-GARCH panel DGP with controllable cross-sectional structure.
  * EWMA / RiskMetrics variance.
  * A FAITHFUL re-implementation of CAFE's robust EW residual-scale recursion
    (the exact math in cafe._core._UnifiedCore._update_robust_scale, HALFLIFE=200,
    Student-t IRLS weights, kurtosis-EM for nu). cross_check_cafe_scale() verifies
    it tracks the genuine library object to <1e-9.
  * Beta-t-GARCH: a score-driven variance recursion whose innovation is exactly the
    CAFE IRLS weight w = (nu+1)/(nu + r^2/h) times r^2 -- i.e. the upgrade the writeup
    proposes, built from machinery CAFE already computes.
  * Forecast-evaluation metrics: QLIKE, variance-MSE, VaR back-test (Kupiec).
"""
from __future__ import annotations

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# CAFE's own constant -- imported, not copied, so the experiment stays in lock-step
# with the library if it ever changes.
from cafe._core import HALFLIFE, NU_MIN, NU_MAX  # noqa: E402


# --------------------------------------------------------------------------- #
# Data-generating processes
# --------------------------------------------------------------------------- #
def sim_garch(T, omega=0.05, alpha=0.08, beta=0.90, dist="normal", nu=6.0,
              seed=0, burn=500):
    """Simulate a GARCH(1,1): h_t = omega + alpha r_{t-1}^2 + beta h_{t-1}.

    Persistence alpha+beta < 1 gives a genuine long-run variance omega/(1-alpha-beta)
    that the series mean-reverts to -- the defining feature CAFE's EWMA scale lacks.
    Returns (r, h) where h is the TRUE latent conditional variance (known in sim).
    """
    rng = np.random.default_rng(seed)
    n = T + burn
    h = np.empty(n)
    r = np.empty(n)
    h[0] = omega / max(1.0 - alpha - beta, 1e-3)
    if dist == "normal":
        z = rng.standard_normal(n)
    elif dist == "t":
        # unit-variance Student-t innovations
        z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    else:
        raise ValueError(dist)
    r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    return r[burn:], h[burn:]


def add_additive_outliers(r, rate=0.01, scale=8.0, seed=1):
    """Inject additive (measurement) outliers that are NOT part of the vol process.

    These are the jumps/fat-finger prints a robust vol model should ignore but a
    Gaussian GARCH over-reacts to (it treats the squared jump as a vol signal).
    Returns (r_contaminated, outlier_mask).
    """
    rng = np.random.default_rng(seed)
    r = r.copy()
    mask = rng.random(r.shape[0]) < rate
    sd = np.std(r)
    signs = rng.choice([-1.0, 1.0], size=mask.sum())
    r[mask] += signs * scale * sd
    return r, mask


def sim_factor_garch_panel(T, N, K=3, dist="t", nu=7.0, seed=0, burn=500,
                           rho_load=0.0):
    """Factor-GARCH panel: r_{i,t} = sum_k B_{i,k} f_{k,t} + e_{i,t}.

    Each common factor f_k and each idiosyncratic e_i follows its own GARCH(1,1),
    so the TRUE conditional covariance is  Sigma_t = B diag(hf_t) B' + diag(he_t)
    -- low-rank-plus-diagonal, exactly the structure CAFE represents. Returns
    (R [T,N], Sigma [T,N,N], B [N,K]).
    """
    rng = np.random.default_rng(seed)
    # factor GARCH params (persistent, mean-reverting)
    fac = [dict(omega=0.05, alpha=0.07 + 0.02 * k / max(K - 1, 1),
                beta=0.90 - 0.02 * k / max(K - 1, 1)) for k in range(K)]
    F = np.empty((T, K))
    HF = np.empty((T, K))
    for k, p in enumerate(fac):
        f, hf = sim_garch(T, p["omega"], p["alpha"], p["beta"], dist=dist, nu=nu,
                          seed=seed * 100 + k, burn=burn)
        F[:, k] = f
        HF[:, k] = hf
    # loadings: each asset loads on a primary factor + spillover; positive-ish so the
    # cross-section is genuinely correlated (where MGARCH's curse of dimensionality bites)
    B = 0.3 * rng.standard_normal((N, K))
    B[:, 0] += 0.8                                   # a broad market factor
    # idiosyncratic GARCH (milder, asset-specific)
    E = np.empty((T, N))
    HE = np.empty((T, N))
    for i in range(N):
        e, he = sim_garch(T, 0.08, 0.05, 0.90, dist=dist, nu=nu,
                          seed=seed * 1000 + 7 + i, burn=burn)
        E[:, i] = e * 0.6
        HE[:, i] = he * 0.36
    R = F @ B.T + E
    Sigma = np.empty((T, N, N))
    for t in range(T):
        Sigma[t] = (B * HF[t]) @ B.T + np.diag(HE[t])
    return R, Sigma, B


def induce_async_gaps(R, gap_rate=0.15, mean_gap=20, seed=0):
    """Asynchronous block gaps per series (non-synchronous trading / halts).

    Returns (R_with_nan, mask_observed). Each series independently gets blackout
    blocks -- the ragged, non-aligned missingness classical MGARCH cannot ingest.
    """
    rng = np.random.default_rng(seed)
    T, N = R.shape
    Rg = R.copy()
    obs = np.ones((T, N), bool)
    for i in range(N):
        t = 0
        target = int(gap_rate * T)
        dropped = 0
        while dropped < target and t < T:
            if rng.random() < gap_rate / max(mean_gap, 1):
                L = max(1, int(rng.exponential(mean_gap)))
                Rg[t:t + L, i] = np.nan
                obs[t:t + L, i] = False
                dropped += L
                t += L
            else:
                t += 1
    return Rg, obs


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def ewma_var(r, lam=0.94, h0=None):
    """RiskMetrics EWMA: h_t = lam h_{t-1} + (1-lam) r_{t-1}^2. One-step, causal.

    lam=0.94 is the RiskMetrics daily default (half-life ~= 11 days)."""
    r = np.asarray(r, float)
    h = np.empty_like(r)
    h[0] = r[0] ** 2 if h0 is None else h0
    for t in range(1, len(r)):
        h[t] = lam * h[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return h


def lam_from_halflife(hl):
    return 0.5 ** (1.0 / hl)


def halflife_from_lam(lam):
    return np.log(0.5) / np.log(lam)


# --------------------------------------------------------------------------- #
# CAFE's actual variance machinery, isolated (faithful to _core._update_robust_scale)
# --------------------------------------------------------------------------- #
def cafe_scale_var(r, halflife=HALFLIFE, nu_init=8.0, robust=True, demean=True):
    """One-step variance forecast h_t = CAFE's robust EW residual scale^2 at t-1.

    This reproduces, for a single de-meaned series, exactly the recursion CAFE runs
    in _UnifiedCore: an IRLS/Student-t down-weighted, EW-forgotten mean of r^2 with
    the library half-life, plus the kurtosis-EM update of nu. It is the conditional
    variance CAFE would report for this series AS WRITTEN. h_t uses only data < t.
    """
    r = np.asarray(r, float)
    T = len(r)
    lam = lam_from_halflife(halflife)
    EPS = 1e-9
    # state (mirrors _UnifiedCore fields)
    nu = float(nu_init)
    scale2 = 1.0
    r2_sum = 0.0; r2_cnt = 0.0
    m2_sum = 0.0; m4_sum = 0.0; mom_cnt = 0.0
    # expanding (near-static) mean, like CAFE's mu with MU_HALFLIFE ~ 1e5
    mu_sum = 0.0; mu_cnt = 0.0
    h = np.empty(T)
    for t in range(T):
        h[t] = max(scale2, EPS)                       # forecast made from past state
        mu = mu_sum / mu_cnt if mu_cnt > 0 else 0.0
        resid = (r[t] - mu) if demean else r[t]
        r2 = resid * resid
        s2_old = max(scale2, EPS)                      # library uses OLD scale2 below
        if robust:
            w = (nu + 1.0) / (nu + r2 / s2_old)
        else:
            w = 1.0
        z2 = r2 / s2_old
        # EW robust scale^2 + standardized moments (k=1 case of the recurrence)
        r2_sum = lam * r2_sum + w * r2
        r2_cnt = lam * r2_cnt + w
        m2_sum = lam * m2_sum + z2
        m4_sum = lam * m4_sum + z2 * z2
        mom_cnt = lam * mom_cnt + 1.0
        if r2_cnt > 5.0:                               # scale2 updated AFTER moments
            scale2 = max(r2_sum / r2_cnt, EPS)
        if mom_cnt > 30.0 and robust:
            m2 = m2_sum / mom_cnt
            m4 = m4_sum / mom_cnt
            kurt = m4 / max(m2 * m2, EPS)
            excess = kurt - 3.0
            nu_hat = 4.0 + 6.0 / excess if excess > 0.2 else NU_MAX
            nu = float(np.clip(0.7 * nu + 0.3 * nu_hat, NU_MIN, NU_MAX))
        # update near-static mean
        if demean:
            mu_sum += r[t]; mu_cnt += 1.0
    return h


def cross_check_cafe_scale(r):
    """Drive the GENUINE library recursion cafe._core._UnifiedCore._update_robust_scale
    with the de-meaned squared returns and compare its scale2 trajectory to
    cafe_scale_var(). This isolates exactly the recursion the writeup identifies as
    CAFE's volatility machinery (the EW robust residual scale^2 + kurtosis-EM nu),
    without the rank-8 factor degeneracy that swamps a single fully-observed series.

    A near-zero deviation confirms cafe_scale_var is a faithful isolation of the
    library object, not an independent re-derivation. Returns (max|dev|, lib, proxy).
    """
    from cafe._core import _UnifiedCore
    core = _UnifiedCore(1, periods=[], E=1)
    # near-static de-meaning, matching cafe_scale_var / CAFE's long-half-life mu
    mu_sum = 0.0; mu_cnt = 0.0
    lib = np.empty(len(r))
    for t in range(len(r)):
        lib[t] = max(core.scale2, 1e-9)                # forecast = pre-update scale2
        mu = mu_sum / mu_cnt if mu_cnt > 0 else 0.0
        resid = r[t] - mu
        core._update_robust_scale(np.array([resid * resid]))   # the genuine recursion
        mu_sum += r[t]; mu_cnt += 1.0
    proxy = cafe_scale_var(r)
    return float(np.max(np.abs(lib - proxy))), lib, proxy


# --------------------------------------------------------------------------- #
# Beta-t-GARCH  (the proposed upgrade -- innovation IS the CAFE IRLS weight)
# --------------------------------------------------------------------------- #
def betat_garch_filter(r, omega, alpha, beta, nu):
    """Score-driven variance recursion (Harvey/Creal-Koopman-Lucas Beta-t-GARCH).

        w_t = (nu+1) / (nu + r_t^2 / h_t)          <-- CAFE's exact IRLS weight
        u_t = w_t * r_t^2                            <-- the Student-t score for scale
        h_{t+1} = omega + alpha * u_t + beta * h_t

    The single line `w_t = (nu+1)/(nu + r^2/h)` is byte-for-byte the weight CAFE
    already computes in _factor_update / _update_robust_scale. The only additions
    over CAFE-as-is are the learned reaction alpha, persistence beta, and constant
    omega (the long-run level CAFE lacks). Returns h (one-step variance forecasts).
    """
    r = np.asarray(r, float)
    T = len(r)
    h = np.empty(T)
    h[0] = omega / max(1.0 - beta, 1e-3)
    EPS = 1e-9
    for t in range(T):
        ht = max(h[t], EPS)
        w = (nu + 1.0) / (nu + r[t] ** 2 / ht)
        u = w * r[t] ** 2
        if t + 1 < T:
            h[t + 1] = omega + alpha * u + beta * ht
    return h


def _student_t_nll(r, h, nu):
    """Negative log-likelihood of r under Student-t with conditional variance h."""
    h = np.maximum(h, 1e-12)
    s2 = h * (nu - 2.0) / nu                                  # scale^2 of the t
    from scipy.special import gammaln
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu * s2)
    ll = c - (nu + 1) / 2 * np.log1p(r ** 2 / (nu * s2))
    return -np.sum(ll)


def fit_betat_garch(r, nu_fixed=None):
    """ML-fit the Beta-t-GARCH (omega, alpha, beta[, nu]) by numerical optimization."""
    from scipy.optimize import minimize
    r = np.asarray(r, float)
    var = np.var(r)

    def unpack(theta):
        omega = np.exp(theta[0])
        # alpha, beta in (0,1) via logistic, persistence kept < 1
        alpha = 1.0 / (1.0 + np.exp(-theta[1])) * 0.3
        beta = 1.0 / (1.0 + np.exp(-theta[2])) * 0.999
        nu = nu_fixed if nu_fixed is not None else (2.1 + np.exp(theta[3]))
        return omega, alpha, beta, nu

    def obj(theta):
        omega, alpha, beta, nu = unpack(theta)
        h = betat_garch_filter(r, omega, alpha, beta, nu)
        if not np.all(np.isfinite(h)):
            return 1e10
        return _student_t_nll(r, h, nu)

    x0 = [np.log(0.05 * var + 1e-6), -1.0, 2.0]
    if nu_fixed is None:
        x0.append(np.log(6.0))
    best = None
    for seed_theta in (x0, [np.log(0.1 * var + 1e-6), 0.0, 1.5] + ([np.log(10.0)] if nu_fixed is None else [])):
        res = minimize(obj, np.array(seed_theta), method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta, nu = unpack(best.x)
    h = betat_garch_filter(r, omega, alpha, beta, nu)
    return dict(omega=omega, alpha=alpha, beta=beta, nu=nu, nll=best.fun, h=h)


# --------------------------------------------------------------------------- #
# arch wrappers (gold-standard GARCH MLE), one-step in-sample conditional variance
# --------------------------------------------------------------------------- #
def arch_garch_var(r, dist="normal"):
    """Fit GARCH(1,1) via the `arch` package; return one-step conditional variance.

    `arch` works best on percent-scale returns; we rescale internally and undo it.
    """
    import warnings
    from arch import arch_model
    from arch.utility.exceptions import DataScaleWarning
    r = np.asarray(r, float)
    s = 100.0 / max(np.std(r), 1e-8)                          # rescale to ~unit-ish %
    am = arch_model(r * s, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataScaleWarning)
        res = am.fit(disp="off")
    h = res.conditional_volatility ** 2 / s ** 2              # undo rescale
    return np.asarray(h), res


# --------------------------------------------------------------------------- #
# Forecast-evaluation metrics
# --------------------------------------------------------------------------- #
def qlike(h, proxy):
    """QLIKE loss = mean(log h + proxy/h). Robust to a noisy variance proxy
    (Patton 2011); lower is better. proxy is usually r^2 or the true h."""
    h = np.maximum(np.asarray(h, float), 1e-12)
    proxy = np.asarray(proxy, float)
    return float(np.mean(np.log(h) + proxy / h))


def var_mse(h, h_true):
    return float(np.mean((np.asarray(h) - np.asarray(h_true)) ** 2))


def var_backtest(r, h, p=0.01, dist="normal", nu=6.0):
    """One-sided Value-at-Risk back-test at coverage p.

    Returns (violation_rate, kupiec_LR, kupiec_pvalue). A well-calibrated forecast
    has violation_rate ~= p and a non-significant Kupiec unconditional-coverage LR.
    """
    from scipy import stats
    r = np.asarray(r, float)
    h = np.maximum(np.asarray(h, float), 1e-12)
    if dist == "normal":
        q = stats.norm.ppf(p)
    else:
        q = stats.t.ppf(p, nu) * np.sqrt((nu - 2.0) / nu)
    VaR = q * np.sqrt(h)                                       # lower-tail threshold
    viol = r < VaR
    n = len(r); x = int(viol.sum()); pi = x / n
    # Kupiec POF likelihood-ratio test
    if 0 < x < n:
        lr = -2.0 * (np.log((1 - p) ** (n - x) * p ** x)
                     - np.log((1 - pi) ** (n - x) * pi ** x))
    else:
        lr = np.nan
    pval = float(1 - stats.chi2.cdf(lr, 1)) if np.isfinite(lr) else np.nan
    return float(pi), float(lr), pval


if __name__ == "__main__":
    # smoke test + the faithful-proxy cross-check
    r, h = sim_garch(3000, seed=0)
    d, lib, proxy = cross_check_cafe_scale(r)
    print(f"CAFE scale faithful-proxy max|dev| over 3000 steps: {d:.2e}")
    print(f"  library scale2[-1]={lib[-1]:.4f}  proxy[-1]={proxy[-1]:.4f}  true h mean={h.mean():.4f}")
