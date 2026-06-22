"""
Shared benchmark harness for TIMARA-Core experiments.

EVERY candidate imputer is evaluated through THIS module so the data, masking,
and metrics are byte-identical across methods. Do not fork data generation in a
method file -- import from here.

Interface a method must provide:
    def impute(X: np.ndarray, meta: dict) -> np.ndarray
        X    : (T, N) float array with np.nan in the missing entries
        meta : dict with optional keys for panel methods:
               'entity_ids' (len T int array), 'time_ids' (len T int array)
        returns a (T, N) array with all NaNs filled (observed entries may be
        returned unchanged or reconstructed; metrics only score masked cells).

Run with:
    from harness import DATASETS, evaluate, summarize
    rows = [evaluate('MyMethod', impute, name, d) for name, d in DATASETS.items()]
"""
from __future__ import annotations
import time
import numpy as np

RNG_SEED = 20260622


# --------------------------------------------------------------------------- #
# Ground-truth data generators
# --------------------------------------------------------------------------- #
def _ar_factor(T, L, rho, rng):
    """L latent temporal factors, each an AR(1) with coefficient rho."""
    Z = np.zeros((T, L))
    Z[0] = rng.standard_normal(L)
    for t in range(1, T):
        Z[t] = rho * Z[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(L)
    return Z


def gen_2d(T=400, N=20, L=4, rho=0.95, seasonal=True, noise=0.1, seed=0):
    """Low-rank temporal + seasonality + noise. Returns clean (T,N) matrix."""
    rng = np.random.default_rng(RNG_SEED + seed)
    Z = _ar_factor(T, L, rho, rng)              # (T, L) temporal factors
    U = rng.standard_normal((N, L))             # (N, L) loadings
    X = Z @ U.T                                 # (T, N) low-rank signal
    if seasonal:
        t = np.arange(T)
        for k in (1, 2):
            period = T / (3 * k)
            phase = rng.standard_normal(N)
            amp = 0.5 * rng.random(N)
            X += amp * np.sin(2 * np.pi * k * t[:, None] / period + phase)
    X += noise * X.std() * rng.standard_normal((T, N))
    return X


def gen_panel(E=15, T=60, F=8, L=3, rho=0.9, noise=0.1, seed=0):
    """
    Panel: E entities x T time x F features, stacked to (E*T, F) row-major
    (entity-major). Includes entity fixed effects + time fixed effects + a
    low-rank entity-time interaction. Returns (X2d, meta) with entity/time ids.
    """
    rng = np.random.default_rng(RNG_SEED + 1000 + seed)
    ent_fe = rng.standard_normal((E, F)) * 2.0          # entity fixed effects
    time_fe = rng.standard_normal((T, F)) * 1.0         # time fixed effects
    # low-rank entity-time interaction per feature
    A = rng.standard_normal((E, L))
    B = _ar_factor(T, L, rho, rng)                      # AR temporal factor
    C = rng.standard_normal((L, F))
    X = np.empty((E * T, F))
    eids = np.empty(E * T, dtype=int)
    tids = np.empty(E * T, dtype=int)
    idx = 0
    for e in range(E):
        for t in range(T):
            inter = (A[e][None, :] * B[t][None, :]) @ C  # (1,L)*(1,L)->(1,L)@ (L,F)
            val = ent_fe[e] + time_fe[t] + inter.ravel()
            X[idx] = val
            eids[idx] = e
            tids[idx] = t
            idx += 1
    X += noise * X.std() * rng.standard_normal(X.shape)
    meta = {"entity_ids": eids, "time_ids": tids, "E": E, "T": T, "F": F}
    return X, meta


# --------------------------------------------------------------------------- #
# Missingness mechanisms
# --------------------------------------------------------------------------- #
def mask_mcar(X, rate, seed=0):
    rng = np.random.default_rng(RNG_SEED + 7 + seed)
    return rng.random(X.shape) < rate


def mask_mar(X, rate, seed=0):
    """Missingness in each column driven by the value of the previous column."""
    rng = np.random.default_rng(RNG_SEED + 11 + seed)
    T, N = X.shape
    M = np.zeros((T, N), dtype=bool)
    for j in range(N):
        driver = X[:, j - 1] if j > 0 else X[:, -1]
        p = 1.0 / (1.0 + np.exp(-(driver - driver.mean()) / (driver.std() + 1e-9)))
        p = p * (2 * rate)            # scale so mean ~ rate
        M[:, j] = rng.random(T) < np.clip(p, 0, 0.95)
    return M


def mask_block(X, rate, seed=0):
    """Contiguous time-blocks missing per column (sensor blackout style)."""
    rng = np.random.default_rng(RNG_SEED + 13 + seed)
    T, N = X.shape
    M = np.zeros((T, N), dtype=bool)
    target = int(rate * T * N)
    placed = 0
    while placed < target:
        j = rng.integers(N)
        blen = int(rng.integers(max(2, T // 50), max(3, T // 8)))
        start = int(rng.integers(0, max(1, T - blen)))
        seg = slice(start, start + blen)
        newly = np.sum(~M[seg, j])
        M[seg, j] = True
        placed += newly
    return M


def mask_mnar(X, rate, seed=0):
    """Self-masking: large values are more likely to be missing (value-dependent)."""
    rng = np.random.default_rng(RNG_SEED + 17 + seed)
    z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = 1.0 / (1.0 + np.exp(-(z - 1.0)))          # high z -> high miss prob
    p *= (rate / (p.mean() + 1e-9))               # rescale to target rate
    return rng.random(X.shape) < np.clip(p, 0, 0.97)


def mask_blackout(X, rate, seed=0):
    """SYNCHRONIZED blackout: contiguous time windows where ALL features vanish
    at once (the bench-vldb20 hard scenario — no contemporaneous cross-section)."""
    rng = np.random.default_rng(RNG_SEED + 19 + seed)
    T, N = X.shape
    M = np.zeros((T, N), dtype=bool)
    placed, target = 0, int(rate * T)
    while placed < target:
        blen = int(rng.integers(max(2, T // 40), max(3, T // 10)))
        start = int(rng.integers(0, max(1, T - blen)))
        newly = np.sum(~M[start:start + blen, 0])
        M[start:start + blen, :] = True
        placed += newly
    return M


MASKERS = {"mcar": mask_mcar, "mar": mask_mar, "block": mask_block,
           "mnar": mask_mnar, "blackout": mask_blackout}


# --------------------------------------------------------------------------- #
# Extra generators for benchmark DIVERSITY (no single method should win by luck)
# --------------------------------------------------------------------------- #
def gen_1d(T=600, kind="ar", seed=0):
    """Single series (T,1). kind: 'ar' (smooth AR) or 'seasonal' (strong cycle)."""
    rng = np.random.default_rng(RNG_SEED + 200 + seed)
    if kind == "seasonal":
        t = np.arange(T)
        x = (2.0 * np.sin(2 * np.pi * t / 24) + 1.0 * np.sin(2 * np.pi * t / 168)
             + 0.3 * rng.standard_normal(T) + 0.002 * t)
    else:
        x = _ar_factor(T, 1, 0.97, rng).ravel() * 3.0 + 0.1 * rng.standard_normal(T)
    return x.reshape(T, 1)


def gen_highrank(T=500, N=30, seed=0):
    """High effective rank (weak low-rank structure) — punishes pure low-rank MF."""
    rng = np.random.default_rng(RNG_SEED + 300 + seed)
    L = N - 2                                     # nearly full rank
    Z = _ar_factor(T, L, 0.6, rng)
    X = Z @ rng.standard_normal((N, L)).T
    return X + 0.15 * X.std() * rng.standard_normal((T, N))


def gen_drift(T=800, N=15, seed=0):
    """Non-stationary: regime shifts in level/loadings (rewards adaptive forgetting)."""
    rng = np.random.default_rng(RNG_SEED + 400 + seed)
    L = 4
    Z = _ar_factor(T, L, 0.9, rng)
    U1 = rng.standard_normal((N, L)); U2 = rng.standard_normal((N, L))
    X = np.empty((T, N))
    for t in range(T):
        w = t / T                                 # loadings drift over time
        U = (1 - w) * U1 + w * U2
        X[t] = Z[t] @ U.T + (3.0 if t > T // 2 else 0.0)   # level break at midpoint
    return X + 0.1 * X.std() * rng.standard_normal((T, N))


def gen_heavytail(T=500, N=20, seed=0):
    """Low-rank signal + Student-t outliers (rewards robust loss)."""
    rng = np.random.default_rng(RNG_SEED + 500 + seed)
    base = gen_2d(T=T, N=N, seed=seed, noise=0.05)
    out = rng.standard_t(2.5, size=(T, N)) * (rng.random((T, N)) < 0.05) * base.std() * 4
    return base + out


# --------------------------------------------------------------------------- #
# Dataset registry  (clean matrix, meta, list of (mech, rate) configs)
# --------------------------------------------------------------------------- #
def _build_datasets():
    ds, desc = {}, {}
    # --- 1D univariate (the library must nail single series too) ---
    s_ar = gen_1d(kind="ar"); s_se = gen_1d(kind="seasonal")
    ds["1d_ar_mcar20"] = (s_ar, {}, "mcar", 0.20)
    desc["1d_ar_mcar20"] = "1D smooth AR(.97) series, scattered point gaps. No cross-section; pure temporal carry/AR."
    ds["1d_seasonal_block"] = (s_se, {}, "block", 0.25)
    desc["1d_seasonal_block"] = "1D strong daily+weekly seasonal series, contiguous gaps. Rewards seasonal-naive/Fourier; LOCF fails across cycles."

    # --- 2D time-series, varied structure × mechanism × rate ---
    base2d = gen_2d(T=400, N=20, seed=0)
    ds["2d_small_mcar20"] = (base2d, {}, "mcar", 0.20)
    desc["2d_small_mcar20"] = "2D low-rank+AR+seasonal, light scattered missing. Easy; cross-section + AR both help."
    ds["2d_small_mar30"] = (base2d, {}, "mar", 0.30)
    desc["2d_small_mar30"] = "2D, missing in a column driven by a neighbor column (MAR). Cross-section still mostly observed."
    ds["2d_small_block20"] = (base2d, {}, "block", 0.20)
    desc["2d_small_block20"] = "2D, per-column contiguous blackouts. Needs temporal carry-over; cross-section partial."
    ds["2d_small_mnar30"] = (base2d, {}, "mnar", 0.30)
    desc["2d_small_mnar30"] = "2D, large values self-mask (MNAR). Tests bias: mean/low-rank under-predict the missing high tail."
    big2d = gen_2d(T=2000, N=60, seed=1)
    ds["2d_large_mcar30"] = (big2d, {}, "mcar", 0.30)
    desc["2d_large_mcar30"] = "Large 2D, 30% scattered. Scale + speed test; low-rank shines."
    ds["2d_large_block30"] = (big2d, {}, "block", 0.30)
    desc["2d_large_block30"] = "Large 2D, 30% per-column blackouts. Scale + hard temporal gaps."
    ds["2d_highmiss_mcar60"] = (base2d, {}, "mcar", 0.60)
    desc["2d_highmiss_mcar60"] = "2D, 60% missing. Stress test: sparse observations, easy to overfit/diverge."

    # --- 2D adversarial structures ---
    ds["2d_highrank_mcar30"] = (gen_highrank(), {}, "mcar", 0.30)
    desc["2d_highrank_mcar30"] = "Near-full-rank data (weak low-rank). Punishes pure MF; rewards cross-sectional regression."
    ds["2d_drift_block"] = (gen_drift(), {}, "block", 0.25)
    desc["2d_drift_block"] = "Non-stationary: drifting loadings + mid-series level break, with blackouts. Rewards adaptive forgetting (RLS)."
    ds["2d_heavytail_mcar30"] = (gen_heavytail(), {}, "mcar", 0.30)
    desc["2d_heavytail_mcar30"] = "Low-rank + 5% Student-t(2.5) outliers. Rewards robust loss; punishes L2/mean."

    # --- 3D panel (entity × time × feature) ---
    pX, pmeta = gen_panel(E=20, T=80, F=10, seed=0)
    ds["panel_mcar20"] = (pX, pmeta, "mcar", 0.20)
    desc["panel_mcar20"] = "Panel, scattered. Entity+time FE + shared AR factor; FE absorption + cross-section win."
    ds["panel_mar40"] = (pX, pmeta, "mar", 0.40)
    desc["panel_mar40"] = "Panel, heavy MAR. High rate stresses online FE/loadings estimation."
    ds["panel_block30"] = (pX, pmeta, "block", 0.30)
    desc["panel_block30"] = "Panel, per-series blackouts (THE weak spot). Some entities dark for spans; FE + AR-on-residual needed."
    ds["panel_blackout25"] = (pX, pmeta, "blackout", 0.25)
    desc["panel_blackout25"] = "Panel, SYNCHRONIZED blackout (all entities+features dark together). No contemporaneous cross-section at all — hardest causal case."
    return ds, desc


DATASETS, DATASET_DESC = _build_datasets()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def metrics(true, pred, mask):
    t = true[mask]
    p = pred[mask]
    finite = np.isfinite(p)
    if not np.all(finite):              # penalize methods that leave NaNs
        p = np.where(finite, p, np.nanmean(true))
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.mean(np.abs(t)) + 1e-9
    mre = float(np.mean(np.abs(err)) / denom)        # mean relative error
    if np.std(p) > 1e-12 and np.std(t) > 1e-12:
        corr = float(np.corrcoef(t, p)[0, 1])
    else:
        corr = float("nan")
    return dict(mae=mae, rmse=rmse, mre=mre, corr=corr, n=int(mask.sum()))


def evaluate(method_name, impute_fn, dataset_name, dataset, repeats=1):
    """Run one method on one dataset; return a result row dict."""
    clean, meta, mech, rate = dataset
    rows = []
    for r in range(repeats):
        M = MASKERS[mech](clean, rate, seed=r)
        Xobs = clean.copy()
        Xobs[M] = np.nan
        t0 = time.perf_counter()
        try:
            pred = impute_fn(Xobs.copy(), dict(meta))
            dt = time.perf_counter() - t0
            m = metrics(clean, np.asarray(pred, dtype=float), M)
            m.update(success=True, error="")
        except Exception as e:                       # noqa: BLE001
            dt = time.perf_counter() - t0
            m = dict(mae=float("nan"), rmse=float("nan"), mre=float("nan"),
                     corr=float("nan"), n=int(M.sum()), success=False,
                     error=f"{type(e).__name__}: {e}")
        m.update(time_s=dt)
        rows.append(m)
    # average over repeats
    agg = dict(method=method_name, dataset=dataset_name, mechanism=mech,
               rate=rate, shape=f"{clean.shape[0]}x{clean.shape[1]}")
    for k in ("mae", "rmse", "mre", "corr", "time_s"):
        agg[k] = float(np.nanmean([row[k] for row in rows]))
    agg["success"] = all(row["success"] for row in rows)
    agg["error"] = next((row["error"] for row in rows if not row["success"]), "")
    return agg


def run_method(method_name, impute_fn, datasets=None, repeats=1):
    datasets = datasets or DATASETS
    return [evaluate(method_name, impute_fn, name, d, repeats)
            for name, d in datasets.items()]


def summarize(rows):
    """Pretty one-line-per-row print."""
    hdr = f"{'method':22s} {'dataset':18s} {'shape':10s} {'MAE':>8s} {'RMSE':>8s} {'corr':>6s} {'time_s':>9s}  ok"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method'][:22]:22s} {r['dataset'][:18]:18s} {r['shape']:10s} "
              f"{r['mae']:8.4f} {r['rmse']:8.4f} {r['corr']:6.3f} {r['time_s']:9.4f}  "
              f"{'Y' if r['success'] else 'N ' + r['error'][:40]}")


if __name__ == "__main__":
    # smoke test with a trivial column-mean imputer
    def mean_impute(X, meta):
        col = np.nanmean(X, axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        out = X.copy()
        idx = np.where(np.isnan(out))
        out[idx] = np.take(col, idx[1])
        return out

    rows = run_method("ColumnMean", mean_impute)
    summarize(rows)
    print(f"\n{len(DATASETS)} datasets, {len(rows)} result rows.")
