"""E1 — Does cross-entity tensor structure actually help?

Generate a genuine low-rank TENSOR x[e,t,f] = mu[e,f] + sum_r A[e,r] z[t,r] W[f,r] + eps
with z_t AR(1) factors SHARED across entities (common 'market' factors). This is the
regime where contemporaneous cross-entity info is informative: an entity missing a cell
at time t can borrow from other entities observed at t.

Compare on MCAR-masked cells:
  - PANEL  : cafe.impute(stacked, meta=...)         -- sees all entities jointly (trilinear)
  - PERENT : cafe.impute(X_e) per entity, 2D        -- each ticker imputed in isolation
  - COLMEAN: per (entity,feature) expanding-ish mean floor (point-in-time-ish)
Lower MAE = better. If PANEL << PERENT, the tensor mode earns its keep.
"""
import sys, time
import numpy as np
sys.path.insert(0, "src")
import cafe


def make_tensor(E, T, F, R, seed, shared_strength=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((E, R))
    W = rng.standard_normal((F, R))
    # AR(1) shared factors
    z = np.zeros((T, R))
    z[0] = rng.standard_normal(R)
    phi = 0.85
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(R)
    mu = 0.5 * rng.standard_normal((E, F))
    X = np.empty((E, T, F))
    for e in range(E):
        common = (A[e] * shared_strength) @ z.T  # (R,)·(R,T) -> need einsum
    # vectorized trilinear
    core = np.einsum("er,tr,fr->etf", A * shared_strength, z, W)
    X = mu[:, None, :] + core + 0.3 * rng.standard_normal((E, T, F))
    return X


def mask_mcar(X, rate, seed):
    rng = np.random.default_rng(seed + 9999)
    m = rng.random(X.shape) < rate
    Xm = X.copy()
    Xm[m] = np.nan
    return Xm, m


def mae(true, pred, m):
    return float(np.mean(np.abs(true[m] - pred[m])))


def stack(Xm):
    E, T, F = Xm.shape
    ent = np.repeat(np.arange(E), T)
    tim = np.tile(np.arange(T), E)
    rows = Xm.reshape(E * T, F)
    return rows, {"entity_ids": ent, "time_ids": tim}, (E, T, F)


def run_panel(Xm):
    rows, meta, (E, T, F) = stack(Xm)
    filled = np.asarray(cafe.impute(rows, meta=meta), float)
    return filled.reshape(E, T, F)


def run_perentity(Xm):
    E, T, F = Xm.shape
    out = np.empty_like(Xm)
    for e in range(E):
        out[e] = np.asarray(cafe.impute(Xm[e]), float)  # 2D (T,F)
    return out


def run_colmean(Xm):
    # cumulative per-(entity,feature) mean as a naive causal-ish floor
    E, T, F = Xm.shape
    out = Xm.copy()
    for e in range(E):
        for f in range(F):
            col = Xm[e, :, f]
            running = np.nan
            csum = 0.0; ccnt = 0
            for t in range(T):
                v = col[t]
                if np.isnan(v):
                    out[e, t, f] = (csum / ccnt) if ccnt else 0.0
                else:
                    csum += v; ccnt += 1
    return out


def main():
    seeds = list(range(8))
    configs = [
        dict(E=12, T=60, F=8, R=3, rate=0.2, shared=1.0, tag="base"),
        dict(E=30, T=60, F=8, R=3, rate=0.2, shared=1.0, tag="more_entities"),
        dict(E=12, T=60, F=8, R=3, rate=0.4, shared=1.0, tag="heavy_missing"),
        dict(E=12, T=60, F=8, R=3, rate=0.2, shared=0.3, tag="weak_shared"),
        dict(E=12, T=120, F=20, R=5, rate=0.2, shared=1.0, tag="wide_long"),
    ]
    print(f"{'config':>14} | {'panel':>16} | {'perentity':>16} | {'colmean':>10} | panel<peradv")
    for cfg in configs:
        pm, qm, cm = [], [], []
        tp = tq = 0.0
        for s in seeds:
            X = make_tensor(cfg["E"], cfg["T"], cfg["F"], cfg["R"], s, cfg["shared"])
            Xm, m = mask_mcar(X, cfg["rate"], s)
            t0 = time.time(); fp = run_panel(Xm); tp += time.time() - t0
            t0 = time.time(); fq = run_perentity(Xm); tq += time.time() - t0
            fc = run_colmean(Xm)
            pm.append(mae(X, fp, m)); qm.append(mae(X, fq, m)); cm.append(mae(X, fc, m))
        pm, qm, cm = map(np.array, (pm, qm, cm))
        adv = 100 * (qm - pm) / qm  # % panel improves over per-entity, per seed
        sig = f"{adv.mean():+5.1f}% (wins {int((pm<qm).sum())}/{len(seeds)})"
        print(f"{cfg['tag']:>14} | {pm.mean():.4f} ± {pm.std():.3f} | "
              f"{qm.mean():.4f} ± {qm.std():.3f} | {cm.mean():>9.4f} | {sig}")
    print(f"\ntiming: panel total {tp:.1f}s, perentity total {tq:.1f}s (last config row counts)")


if __name__ == "__main__":
    main()
