"""E3 — Genuine 3D numpy array on-ramp (entity x time x feature).

Today io.py:120 rejects 3D arrays outright. This prototypes the alternative: accept an
(E, T, F) array, flatten to stacked (E*T, F) + auto meta, route through the panel path,
and reshape back to (E, T, F). Proves the shape round-trips and observed cells survive.
Also checks an irregular/ragged case is handled by the same machinery (entities with
different numbers of observed times via a mask), which a strict tensor cannot express but
the stacked-meta form can.
"""
import sys
import numpy as np
sys.path.insert(0, "src")
import cafe


def impute_3d(arr3d):
    """Candidate cafe.impute path for ndim==3."""
    E, T, F = arr3d.shape
    ent = np.repeat(np.arange(E), T)
    tim = np.tile(np.arange(T), E)
    rows = arr3d.reshape(E * T, F)
    filled = np.asarray(cafe.impute(rows, meta={"entity_ids": ent, "time_ids": tim}), float)
    return filled.reshape(E, T, F)


def main():
    rng = np.random.default_rng(3)
    E, T, F = 10, 50, 6
    A = rng.standard_normal((E, 3)); W = rng.standard_normal((F, 3))
    z = np.cumsum(rng.standard_normal((T, 3)) * 0.3, axis=0)
    X = np.einsum("er,tr,fr->etf", A, z, W) + 0.3 * rng.standard_normal((E, T, F))
    m = rng.random(X.shape) < 0.25
    Xm = X.copy(); Xm[m] = np.nan

    out = impute_3d(Xm)
    assert out.shape == (E, T, F), out.shape
    nan_left = int(np.isnan(out).sum())
    obs = ~m
    drift = float(np.max(np.abs(out[obs] - Xm[obs])))
    err = float(np.mean(np.abs(out[m] - X[m])))
    print(f"[3D] shape round-trip {out.shape} OK | nan_left={nan_left} | "
          f"obs drift={drift:.2e} | recovery MAE on missing={err:.4f}")

    # ragged: entity 7 only observed for first 20 times (rest fully NaN rows)
    Xr = Xm.copy()
    Xr[7, 20:, :] = np.nan
    out_r = impute_3d(Xr)
    nan_left_r = int(np.isnan(out_r).sum())
    print(f"[ragged] entity-7 blackout after t=20 -> nan_left={nan_left_r} "
          f"(0 = panel filled the blackout via cross-section/AR)")


if __name__ == "__main__":
    main()
