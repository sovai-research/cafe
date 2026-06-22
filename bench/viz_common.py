"""Shared introspection helper for the CAFE paper figures.

Runs the REAL c_unified_penmf core with its opt-in trace enabled and returns the
genuine per-step internals (level / season / factor decomposition, latent factors,
per-cell predictive variance, Student-t anomaly weights, learned nu / AR / ARD,
and the pooled residual covariance). Nothing here re-implements the model: every
quantity is read straight out of the running estimator.
"""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import c_unified_penmf as P

PALETTE = dict(blue="#2b6cb0", teal="#2c7a7b", amber="#dd6b20", red="#c53030",
               slate="#475569", green="#2f855a", purple="#6b46c1",
               grey="#94a3b8", ink="#1a202c")


def demo_series(T=400, N=8, period=48, rank=2, outlier_frac=0.01, miss=0.15, seed=0):
    """A synthetic 2D panel with KNOWN structure: level + seasonal cycle + low-rank
    common factors + heavy-tailed noise, plus block+scattered missingness. Returns
    (X_obs, X_true, mask, parts) where parts holds the ground-truth components."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    level = (np.linspace(-1, 1, T)[:, None] * rng.standard_normal(N) * 0.6
             + rng.standard_normal(N))                       # slow per-series level
    season_amp = rng.uniform(0.5, 1.5, N)
    season = np.sin(2 * np.pi * t[:, None] / period) * season_amp \
        + 0.4 * np.cos(2 * np.pi * t[:, None] / (period / 2)) * season_amp
    G = np.cumsum(rng.standard_normal((T, rank)) * 0.15, axis=0)   # smooth factors
    Wl = rng.standard_normal((rank, N))
    factor = G @ Wl
    noise = 0.25 * rng.standard_normal((T, N))
    Xtrue = level + season + factor + noise
    # heavy-tailed outliers
    om = rng.random((T, N)) < outlier_frac
    Xtrue_o = Xtrue + om * rng.standard_normal((T, N)) * 8.0
    # missingness: scattered + one contiguous block per a few series
    M = rng.random((T, N)) < miss
    for j in range(0, N, 3):
        s = int(rng.integers(T // 4, T // 2)); M[s:s + T // 8, j] = True
    Xobs = Xtrue_o.copy(); Xobs[M] = np.nan
    parts = dict(level=level, season=season, factor=factor, noise=noise,
                 outliers=om, true=Xtrue, true_outlier=Xtrue_o)
    return Xobs, Xtrue, M, parts


def run_traced(Xobs, meta=None):
    """Run the real model with tracing; return (filled, trace_list, core)."""
    X = np.ascontiguousarray(np.asarray(Xobs, float))
    T, N = X.shape
    periods = P._fourier_periods(T)
    core = P._UnifiedCore(N, periods, E=1)
    core.record = True
    out = X.copy(); z_prev = None
    for tt in range(T):
        ft = P._fourier_row(tt, periods)
        filled, z_t = core.process_row(X[tt], tt, z_prev, ft, eid=0)
        out[tt] = filled; z_prev = z_t
    return out, core.rows_trace, core


def stack(trace, key):
    """Stack a per-row trace key into an array (T, ...)."""
    return np.array([r[key] for r in trace])


def components(trace, N):
    """Return dict of (T,N)/(T,) component matrices read from the trace."""
    T = len(trace)
    level = stack(trace, "mu"); season = stack(trace, "season")
    lr = stack(trace, "lr"); filled = stack(trace, "filled")
    time_fe = stack(trace, "time_fe")
    Z = stack(trace, "z")
    cvar = np.full((T, N), np.nan)
    for i, r in enumerate(trace):
        if r["cvar"] is not None:
            cvar[i] = r["cvar"]
    return dict(level=level, season=season, factor=lr, time_fe=time_fe,
                filled=filled, Z=Z, cvar=cvar,
                nu=stack(trace, "nu"), a=stack(trace, "a"),
                rho=stack(trace, "rho"), row_w=stack(trace, "row_w"),
                alpha=stack(trace, "alpha"))


def residual_cov(core):
    """Pooled EW residual covariance from the running core (dependency network)."""
    if core.cw < 1e-6:
        return None
    mean = core.csum / core.cw
    return core.cM2 / core.cw - np.outer(mean, mean)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.18, lw=0.6)


if __name__ == "__main__":
    Xo, Xt, M, parts = demo_series()
    filled, tr, core = run_traced(Xo)
    C = components(tr, Xo.shape[1])
    obs = ~np.isnan(Xo)
    err = np.abs(filled[M] - Xt[M]).mean()
    print(f"traced run OK: T={len(tr)} rows, MAE on missing={err:.3f}, "
          f"final nu={C['nu'][-1]:.1f}, a={C['a'][-1]:.2f}, "
          f"eff-rank={(C['alpha'][-1] < np.median(C['alpha'][-1])).sum()}")
