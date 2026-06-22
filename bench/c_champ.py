"""
c_champ — composite causal imputer. Cheap structural detectors pick ONE technique
per dataset (so only one runs => cost stays near the base router). Each underlying
technique is itself causal; routing is a model-selection step on global structure.

Per-case winners (from the arena sweep) this reproduces:
  heavy-tail->robust  high-rank->xsblend  drift->driftgate  seasonal-1D->seasonal
  panel-block/blackout->blackout  dense-MCAR->ewcov  else->router(ewcov base)
"""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from c_router import _is_panel
from c_chal_ewcov import online_impute as f_ewcov
from c_chal_seasonal import online_impute as f_seasonal
from c_chal_xsblend import online_impute as f_xsblend
from c_chal_driftgate import online_impute as f_drift
from c_chal_robust import online_impute as f_robust
from c_chal_blackout import online_impute as f_blackout
import c_online_trmf, c_fe_lowrank

DEBUG = os.environ.get("TIMARA_CHAMP_DEBUG", "")


def _router(X, meta):
    return c_fe_lowrank.online_impute(X, meta) if _is_panel(meta) else c_online_trmf.online_impute(X, meta)


def _excess_kurtosis(X):
    v = X[~np.isnan(X)]
    if v.size < 20:
        return 0.0
    v = v - v.mean(); s = v.std() + 1e-9
    return float(np.mean((v / s) ** 4) - 3.0)


def _contiguity(X):
    """lag-1 autocorr of the missing mask (per col, averaged). ~0 MCAR, high block."""
    nan = np.isnan(X).astype(float); T, N = nan.shape
    if T < 3:
        return 0.0
    acs = []
    for j in range(N):
        a, b = nan[:-1, j], nan[1:, j]
        if a.std() > 1e-9 and b.std() > 1e-9:
            acs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(acs)) if acs else 0.0


def _eff_rank_ratio(X):
    """participation-ratio effective rank / N, on mean-filled observed (cheap)."""
    Xf = X.copy(); col = np.nanmean(Xf, 0)
    col = np.where(np.isfinite(col), col, 0.0)
    idx = np.where(np.isnan(Xf)); Xf[idx] = np.take(col, idx[1])
    Xf = Xf - Xf.mean(0)
    if min(Xf.shape) < 3:
        return 0.0
    # subsample rows for speed on long series
    if Xf.shape[0] > 600:
        Xf = Xf[np.linspace(0, Xf.shape[0] - 1, 600).astype(int)]
    s = np.linalg.svd(Xf, compute_uv=False)
    pr = (s.sum() ** 2) / (np.sum(s ** 2) + 1e-12)
    return float(pr / Xf.shape[1])


def _drift_score(X):
    """level shift between first and last third of observed (per col, normalized)."""
    T, N = X.shape; a, b = X[:T // 3], X[-T // 3:]
    ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
    sd = np.nanstd(X, 0) + 1e-9
    d = np.abs(ma - mb) / sd
    return float(np.nanmean(d))


def _seasonal_strength(x):
    """max autocorr at lag in [3..T/3] for a single observed series."""
    x = x[np.isfinite(x)]
    if x.size < 30:
        return 0.0, 0
    x = x - x.mean()
    n = x.size; best, lag = 0.0, 0
    denom = np.dot(x, x) + 1e-12
    for L in range(3, min(n // 3, 200)):
        ac = np.dot(x[:-L], x[L:]) / denom
        if ac > best:
            best, lag = ac, L
    return float(best), lag


def online_impute(X, meta):
    X = np.asarray(X, float)
    T, N = X.shape
    panel = _is_panel(meta)
    contig = _contiguity(X)
    pick, fn = "router", _router

    if panel:
        if contig > 0.25:                       # contiguous panel gaps -> blackout specialist
            pick, fn = "blackout", f_blackout
        else:
            pick, fn = "router", _router
    elif N == 1:                                # 1D
        ss, _ = _seasonal_strength(X[:, 0])
        if ss > 0.30:
            pick, fn = "seasonal", f_seasonal
        else:
            pick, fn = "router", _router
    else:                                       # 2D
        kurt = _excess_kurtosis(X)
        errk = _eff_rank_ratio(X)
        drift = _drift_score(X)
        if N >= 40 and contig < 0.15:           # large dense scattered -> ewcov
            pick, fn = "ewcov", f_ewcov
        elif drift > 1.0 and contig > 0.25:     # non-stationary contiguous -> drift gate
            pick, fn = "driftgate", f_drift
        elif kurt > 8.0 and contig < 0.3:       # heavy tails (scattered) -> robust
            pick, fn = "robust", f_robust
        elif errk > 0.8 and contig < 0.3:       # high effective rank (scattered) -> xsblend
            pick, fn = "xsblend", f_xsblend
        else:
            pick, fn = "router", _router
        if DEBUG:
            print(f"  [champ] {X.shape} contig={contig:.2f} kurt={kurt:.1f} "
                  f"errk={errk:.2f} drift={drift:.2f} -> {pick}", file=sys.stderr)
    if DEBUG and (panel or N == 1):
        print(f"  [champ] {X.shape} panel={panel} contig={contig:.2f} -> {pick}", file=sys.stderr)
    try:
        return fn(X.copy(), dict(meta))
    except Exception:                           # robust fallback
        return _router(X.copy(), dict(meta))


if __name__ == "__main__":
    os.environ["TIMARA_CHAMP_DEBUG"] = "1"
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("c_champ", online_impute))
