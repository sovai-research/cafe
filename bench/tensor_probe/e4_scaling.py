"""E4 — Scaling probe: what panel sizes are feasible, and where does cost grow?

Relevant to the '#13 N x N covariance' question. The panel per-time block is
(entities present) x (features). We sweep E (entities), T (time), F (features)
independently and report wall-clock + peak nan-fill correctness, to find the practical
ceiling and the dominant cost axis.
"""
import sys, time, gc
import numpy as np
sys.path.insert(0, "src")
import cafe


def make_and_time(E, T, F, rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    R = 3
    A = rng.standard_normal((E, R)); W = rng.standard_normal((F, R))
    z = np.cumsum(rng.standard_normal((T, R)) * 0.3, axis=0)
    X = np.einsum("er,tr,fr->etf", A, z, W) + 0.3 * rng.standard_normal((E, T, F))
    Xm = X.copy(); Xm[rng.random(X.shape) < rate] = np.nan
    ent = np.repeat(np.arange(E), T); tim = np.tile(np.arange(T), E)
    rows = Xm.reshape(E * T, F)
    meta = {"entity_ids": ent, "time_ids": tim}
    gc.collect()
    t0 = time.time()
    filled = np.asarray(cafe.impute(rows, meta=meta), float)
    dt = time.time() - t0
    nan_left = int(np.isnan(filled).sum())
    return dt, nan_left, E * T * F


def main():
    print("sweep E (entities), T=60, F=8:")
    for E in [10, 25, 50, 100, 200]:
        dt, nl, cells = make_and_time(E, 60, 8)
        print(f"  E={E:>4}  rows={E*60:>6}  cells={cells:>7}  {dt:7.3f}s  nan_left={nl}")

    print("sweep F (features), E=25, T=60   [F drives the cross-section solve width]:")
    for F in [4, 8, 16, 32, 64, 128]:
        dt, nl, cells = make_and_time(25, 60, F)
        print(f"  F={F:>4}  cells={cells:>7}  {dt:7.3f}s  nan_left={nl}")

    print("sweep T (time), E=25, F=8:")
    for T in [50, 100, 200, 400, 800]:
        dt, nl, cells = make_and_time(25, T, 8)
        print(f"  T={T:>4}  rows={25*T:>6}  cells={cells:>7}  {dt:7.3f}s  nan_left={nl}")


if __name__ == "__main__":
    main()
