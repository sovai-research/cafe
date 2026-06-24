"""HIGH-DIMENSIONAL (wide) panels for the CAFE scale benchmark.

Gap #4 answer: the existing race panels are all mid-size (<=100 channels). This
module builds genuinely HIGH-N panels -- a *decade* more width than the race --
so the "strongly sub-linear in N" CPU-scaling claim can be measured at a width
that actually stresses it, not just asserted on narrow data.

Conventions mirror bench/datasets_real.py and bench/datasets_extra.py:
  * each loader returns a numpy matrix X of shape (time, features), dtype
    float64, fully finite, z-scored per column (mean 0, std 1).
  * the WIDE clean matrix is cached to data/<name>_wide_clean.npy.

The wide panels
---------------
traffic_wide   traffic   2500 x 862   Caltrans PEMS freeway occupancy, FULL width
electric_wide  energy    2500 x 321   UCI ElectricityLoadDiagrams, FULL width

Provenance & slice (point-in-time honest):
  traffic_wide
    Source: data/traffic.txt.gz -- San Francisco Bay Area freeway road-occupancy
      rates (Caltrans PEMS), hourly, the LSTNet/Lai-et-al. benchmark. Raw on disk
      is 17544 x 862 (862 sensors). COMPLETE, no native gaps.
    Slice: keep ALL 862 sensors (drop only any constant/degenerate column), cap
      to the first 2500 hours for speed. -> ~2500 x 862. This is ~8.6x wider than
      the widest race panel (100 channels) and is a real, spatially-correlated
      sensor network (neighbouring sensors share a strong low-rank factor).
  electric_wide
    Source: data/electricity.txt.gz -- UCI ElectricityLoadDiagrams2011-2014
      hourly consumption (kWh) of 321 clients, LSTNet pre-processed version. Raw
      on disk is 26304 x 321. COMPLETE.
    Slice: keep ALL 321 clients (drop constant columns), subsample every 3rd hour
      to vary the cadence, cap to the first 2500 rows -> ~2500 x 321. ~3.2x wider
      than the race panels, different domain (demand vs occupancy).

Both are z-scored per column on disk and finite. Nothing here is re-fetched: it
reads the .gz files already present in data/.
"""
import os
import gzip
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

MAX_ROWS = 2500            # cap rows for speed; width is the point here, not length


# --------------------------------------------------------------------------- #
# helpers (same primitives as the sibling loaders)
# --------------------------------------------------------------------------- #
def _zscore(X):
    X = np.asarray(X, dtype=float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _drop_degenerate(X):
    """Drop constant / non-finite columns so the z-score is well defined."""
    X = np.asarray(X, float)
    good = (np.nanstd(X, 0) > 1e-12) & np.isfinite(X).all(0)
    return X[:, good]


def _finite_check(X, name):
    assert X.ndim == 2, f"{name}: expected 2D, got {X.ndim}D"
    assert np.isfinite(X).all(), f"{name}: clean matrix has non-finite values"
    return X


# --------------------------------------------------------------------------- #
# 1. traffic_wide -- FULL-width Caltrans PEMS panel (~862 sensors)
# --------------------------------------------------------------------------- #
def load_traffic_wide():
    """FULL-width Caltrans PEMS occupancy panel: ~2500 x 862 (all sensors)."""
    path = os.path.join(DATA, "traffic.txt.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, header=None)
    X = df.to_numpy(float)
    X = X[:MAX_ROWS]                       # cap rows for speed; KEEP all 862 cols
    X = _drop_degenerate(X)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "traffic_wide")
    np.save(os.path.join(DATA, "traffic_wide_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 2. electric_wide -- FULL-width UCI electricity panel (~321 clients)
# --------------------------------------------------------------------------- #
def load_electric_wide():
    """FULL-width UCI electricity panel: ~2500 x 321 (all clients, 3x subsampled)."""
    path = os.path.join(DATA, "electricity.txt.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, header=None)
    X = df.to_numpy(float)
    X = X[::3]                             # vary cadence; KEEP all 321 cols
    X = X[:MAX_ROWS]
    X = _drop_degenerate(X)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "electric_wide")
    np.save(os.path.join(DATA, "electric_wide_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
DATASETS_SCALE = {
    "traffic_wide": load_traffic_wide,
    "electric_wide": load_electric_wide,
}

DATASET_DESC_SCALE = {
    "traffic_wide": "Caltrans PEMS freeway occupancy, FULL 862-sensor width (LSTNet)",
    "electric_wide": "UCI ElectricityLoadDiagrams, FULL 321-client width (LSTNet)",
}


def _pc_var(X, k):
    """% cross-sectional variance in the top-k PCs (low-rank structure check)."""
    Z = np.nan_to_num(_zscore(X), nan=0.0)
    s = np.linalg.svd(Z, full_matrices=False, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    return 100.0 * cum[min(k, len(cum)) - 1]


def main():
    print(f"{'dataset':14s} {'shape':>14s} {'PC10%':>7s}  description")
    print("-" * 90)
    for name, loader in DATASETS_SCALE.items():
        X = loader()
        pc10 = _pc_var(X, 10)
        print(f"{name:14s} {str(X.shape):>14s} {pc10:>6.1f}%  "
              f"{DATASET_DESC_SCALE[name]}")
    print("-" * 90)
    print("wide clean arrays z-scored, finite, saved to "
          "data/<name>_wide_clean.npy")


if __name__ == "__main__":
    main()
