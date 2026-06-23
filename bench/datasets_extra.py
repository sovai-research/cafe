"""EXTRA structured multivariate-time-series datasets for the CAFE horse race.

This module extends bench/datasets_real.py with FOUR additional, genuinely
*structured* real-world panels, so the structured benchmark rests on 8 datasets
instead of 4.  "Structured" here means a low-rank common-factor structure: the
top few principal components explain a large share of the cross-sectional
variance (top-5 PCs > 50-60%).  (FX/exchange deliberately fails this test and is
the no-structure control -- nothing of that kind is added here.)

Conventions mirror bench/datasets_real.py exactly:
  * each loader returns a numpy matrix X of shape (time, features), dtype
    float64, fully finite, z-scored per column (mean 0, std 1).
  * the clean matrix is cached to data/<name>_clean.npy.
  * every panel is capped to <= 2000 rows and <= 100 columns to stay
    small/fast and comparable to the existing race datasets.

The four EXTRA structured datasets:

  traffic2  traffic   2000 x 100   Caltrans PEMS occupancy, ALREADY on disk
  etth      energy     ~2000 x 7   ETT-h1 transformer-temperature, on disk
  solar     energy     2000 x 100  NREL/LSTNet solar-plant power (very strong PC1)
  electric  energy     ~1880 x 100 UCI ElectricityLoadDiagrams (LSTNet version)

NOTE on traffic/etth: traffic_clean.npy and ETTh1_clean.npy were already on disk
and are NOT in the current race.  Both pass the factor-structure test (traffic
top-5 PCs = 79.7%, ETTh top-5 = 99.7%), so they are free, honest additions.
We re-derive a capped (<=2000 x <=100) clean view from those on-disk arrays so
this module is self-contained and never re-fetches them.

Sources (direct, no-auth downloads):
  solar     https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz
  electric  https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz
  (the gz files are cached to data/solar_AL.txt.gz and data/electricity.txt.gz)
"""
import os
import gzip
import urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

MAX_ROWS = 2000
MAX_COLS = 100

_URLS = {
    "solar_AL.txt.gz": (
        "https://raw.githubusercontent.com/laiguokun/"
        "multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"
    ),
    "electricity.txt.gz": (
        "https://raw.githubusercontent.com/laiguokun/"
        "multivariate-time-series-data/master/electricity/electricity.txt.gz"
    ),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _zscore(X):
    X = np.asarray(X, dtype=float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _finite_check(X, name):
    assert X.ndim == 2, f"{name}: expected 2D, got {X.ndim}D"
    assert np.isfinite(X).all(), f"{name}: clean matrix has non-finite values"
    return X


def _ensure(fname):
    """Download <fname> into data/ if not already present; return its path."""
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        url = _URLS[fname]
        os.makedirs(DATA, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _cap_cols(X):
    """Keep at most MAX_COLS columns, dropping constant/all-NaN ones first."""
    X = np.asarray(X, float)
    good = (np.nanstd(X, 0) > 1e-12) & np.isfinite(X).any(0)
    X = X[:, good]
    if X.shape[1] > MAX_COLS:
        X = X[:, :MAX_COLS]
    return X


def _factor_score(X):
    """Return (pc1, pc3, pc5, pc10) cumulative %% variance explained + lag-1 ac.

    Uses a plain numpy SVD on the per-column z-scored (NaN->0) matrix, so no
    sklearn dependency is required.
    """
    X = np.asarray(X, float)
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0) + 1e-9
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    s = np.linalg.svd(Z, full_matrices=False, compute_uv=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()

    def pc(k):
        return 100.0 * cum[min(k, len(cum)) - 1]

    Zf = (X - mu) / sd
    acs = []
    for j in range(X.shape[1]):
        c = Zf[:, j]
        c = c[np.isfinite(c)]
        if len(c) > 3:
            a = np.corrcoef(c[:-1], c[1:])[0, 1]
            if np.isfinite(a):
                acs.append(a)
    ac1 = float(np.mean(acs)) if acs else float("nan")
    return pc(1), pc(3), pc(5), pc(10), ac1


# --------------------------------------------------------------------------- #
# 1. Traffic2  (traffic) -- Caltrans PEMS occupancy, re-capped from disk
# --------------------------------------------------------------------------- #
def load_traffic2():
    """Caltrans PEMS freeway-occupancy panel (LSTNet/Lai et al.).

    Re-derived from the already-on-disk traffic_clean.npy (3509 x 100) so this
    module never re-fetches it.  We take the first MAX_ROWS hours to cap to a
    <=2000 x 100 panel.  Highly correlated neighbouring sensors -> strong
    spatial common-factor structure (top-5 PCs ~80%).
    """
    src = np.load(os.path.join(DATA, "traffic_clean.npy"))
    X = src[:MAX_ROWS]
    X = _cap_cols(X)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "traffic2")
    np.save(os.path.join(DATA, "traffic2_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 2. ETT-h  (energy) -- electricity-transformer temperature, re-capped from disk
# --------------------------------------------------------------------------- #
def load_etth():
    """ETT-h1 electricity-transformer panel (Zhou et al., Informer): 6 load
    series + oil temperature, hourly.

    Re-derived from the already-on-disk ETTh1_clean.npy (17420 x 7) so this
    module never re-fetches it.  We take the first MAX_ROWS hours -> 2000 x 7.
    Very strong shared load/temperature factor (top-5 PCs ~99%).
    """
    src = np.load(os.path.join(DATA, "ETTh1_clean.npy"))
    X = src[:MAX_ROWS]
    X = _cap_cols(X)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "etth")
    np.save(os.path.join(DATA, "etth_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 3. Solar-Energy  (energy) -- NREL solar-plant power, LSTNet benchmark
# --------------------------------------------------------------------------- #
def load_solar():
    """Solar power production of PV plants in Alabama, 2006 (NREL via LSTNet).

    Raw solar_AL.txt.gz: 52560 x 137 at 10-minute cadence, complete.
    Documented cap: subsample every 6th row (-> hourly cadence), keep the first
    MAX_ROWS hours and the first MAX_COLS plants -> 2000 x 100.  A single solar
    irradiance factor dominates (PC1 ~90%), so this is the strongest-structured
    panel in the race.
    """
    path = _ensure("solar_AL.txt.gz")
    with gzip.open(path, "rt") as f:
        X = np.loadtxt(f, delimiter=",")
    X = X[::6]                       # 10-min -> hourly
    X = X[:MAX_ROWS]
    X = _cap_cols(X)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "solar")
    np.save(os.path.join(DATA, "solar_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 4. Electricity  (energy) -- UCI ElectricityLoadDiagrams, LSTNet version
# --------------------------------------------------------------------------- #
def load_electric():
    """Hourly electricity consumption of 321 clients (UCI
    ElectricityLoadDiagrams2011-2014, LSTNet pre-processed version).

    Raw electricity.txt.gz: 26304 x 321 hourly kWh, complete.
    Documented cap: subsample every 14th hour to stay small, keep the first
    MAX_ROWS rows and the first MAX_COLS clients -> ~1879 x 100.  Strong shared
    daily/weekly demand factor (top-5 PCs ~75%).
    """
    path = _ensure("electricity.txt.gz")
    with gzip.open(path, "rt") as f:
        X = np.loadtxt(f, delimiter=",")
    X = X[::14]                      # subsample to keep small/fast
    X = X[:MAX_ROWS]
    X = _cap_cols(X)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "electric")
    np.save(os.path.join(DATA, "electric_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
DATASETS_EXTRA = {
    "traffic2": load_traffic2,
    "etth": load_etth,
    "solar": load_solar,
    "electric": load_electric,
}

DATASET_DESC_EXTRA = {
    "traffic2": "Caltrans PEMS freeway occupancy (LSTNet); capped spatial panel",
    "etth": "ETT-h1 electricity-transformer load + oil temperature (Informer); hourly",
    "solar": "NREL solar-plant power, Alabama (LSTNet); hourly, dominant irradiance factor",
    "electric": "UCI ElectricityLoadDiagrams hourly demand (LSTNet); 100 clients subsampled",
}


def main():
    print(f"{'dataset':10s} {'shape':>14s} {'PC5%':>7s} {'lag1ac':>7s}  description")
    print("-" * 100)
    for name, loader in DATASETS_EXTRA.items():
        X = loader()
        _, _, pc5, _, ac1 = _factor_score(X)
        # simple fill then assert finiteness (clean is already finite, but the
        # harness applies masks -> verify a mean-fill of a masked copy is finite)
        Xm = X.copy()
        Xm[::7] = np.nan
        filled = np.where(np.isfinite(Xm), Xm, np.nanmean(X))
        assert np.isfinite(filled).all(), f"{name}: fill produced non-finite values"
        assert pc5 > 50.0, f"{name}: NOT structured (top-5 PCs = {pc5:.1f}% <= 50%)"
        print(f"{name:10s} {str(X.shape):>14s} {pc5:>6.1f}% {ac1:>7.3f}  "
              f"{DATASET_DESC_EXTRA[name]}")
    print("-" * 100)
    print("all EXTRA datasets structured (top-5 PCs > 50%), finite after fill; "
          "clean arrays saved to data/<name>_clean.npy")


if __name__ == "__main__":
    main()
