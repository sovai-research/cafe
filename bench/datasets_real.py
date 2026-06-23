"""Diverse, REAL, SMALL real-world datasets for imputation benchmarking.

Mirrors the conventions in bench/prep_real.py:
  * each loader returns a numpy matrix X of shape (time, features), dtype float64,
    fully finite, z-scored per column (mean 0, std 1) -- the "clean" complete
    ground-truth matrix used for masked-imputation eval (synthetic masks applied
    by the harness/arena).
  * the clean matrix is cached to data/<name>_clean.npy.
  * for datasets with GENUINE NATIVE missingness we ALSO cache the (z-scored)
    native matrix with NaNs to data/<name>_native.npy plus a boolean observed
    mask data/<name>_obsmask.npy, so we can later evaluate real-missingness
    imputation (not just synthetic masks).

Five NEW small diverse real datasets (see paper/datasets_real.md for the full
rank-ordered candidate table, provenance and licenses):

  fredmd       macro        ~728 x 107   native ragged-edge missingness
  exchange     finance      7588 x 8     complete (no native gaps)
  airquality   environment  ~6941 x 11   heavy native missingness (-200 sentinel)
  appliances   energy       ~4934 x 26   complete (subsampled rows)
  traffic      traffic      ~3509 x 100  complete (sliced sensors + subsampled)

We already have (do NOT re-fetch): beijing_clean.npy, ETTh1_clean.npy, and the
VLDB txts (airq/chlorine/temp/drift).
"""
import os
import io
import gzip
import zipfile
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


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


# --------------------------------------------------------------------------- #
# 1. FRED-MD  (macroeconomics)  -- THE canonical dynamic-factor-model panel
# --------------------------------------------------------------------------- #
def load_fredmd():
    """McCracken & Ng FRED-MD monthly macro panel.

    Raw current.csv: row 1 = header, row 2 = 'Transform:' codes (dropped),
    first column = 'sasdate'. ~728 months x 128 series, 1959-2019 vintage.
    Has real ragged-edge / patchy native missingness.

    Clean matrix: drop series and rows so that the densest fully-observed block
    is kept (drop the few near-empty columns, then drop rows with any NaN).
    Native matrix: keep all series, z-scored, NaNs preserved.
    """
    path = os.path.join(DATA, "fredmd_current.csv")
    df = pd.read_csv(path)
    # drop the 'Transform:' code row (its sasdate cell == 'Transform:')
    df = df[df["sasdate"].astype(str).str.contains("Transform", na=False) == False]
    # drop fully-empty trailing rows (blank sasdate)
    df = df[df["sasdate"].notna() & (df["sasdate"].astype(str).str.strip() != "")]
    df = df.drop(columns=["sasdate"])
    df = df.apply(pd.to_numeric, errors="coerce")
    raw = df.to_numpy(float)  # (T, N) with native NaNs

    # ---- native (z-scored, NaNs preserved): keep ALL series so the native view
    #      reflects the TRUE ragged-edge missingness (this vintage is ~1% missing).
    col_cov = np.isfinite(raw).mean(0)
    nat = raw
    mu = np.nanmean(nat, 0)
    sd = np.nanstd(nat, 0) + 1e-9
    nat_z = (nat - mu) / sd
    obs = np.isfinite(nat_z)
    np.save(os.path.join(DATA, "fredmd_native.npy"), nat_z)
    np.save(os.path.join(DATA, "fredmd_obsmask.npy"), obs)

    # ---- clean: keep columns with >=98% coverage, then drop rows with any NaN
    keep_clean = col_cov >= 0.98
    sub = raw[:, keep_clean]
    full = sub[np.isfinite(sub).all(1)]
    Xz = _zscore(full)
    Xz = _finite_check(Xz, "fredmd")
    np.save(os.path.join(DATA, "fredmd_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 2. Exchange-Rate  (finance) -- 8 currencies daily, LSTNet benchmark
# --------------------------------------------------------------------------- #
def load_exchange():
    """Daily exchange rates of 8 countries (1990-2016), LSTNet/Lai et al.

    Complete (no native missingness). 7588 x 8. Fits the <=12k x <=150 budget
    directly, so we keep all of it.
    """
    path = os.path.join(DATA, "exchange_rate.txt.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, header=None)
    X = df.to_numpy(float)
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "exchange")
    np.save(os.path.join(DATA, "exchange_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 3. AirQualityUCI  (environment) -- de Vito et al., HEAVY native missingness
# --------------------------------------------------------------------------- #
def load_airquality():
    """Italian-city air-quality multisensor array (de Vito 2008).

    European CSV: ';' separated, ',' decimals, -200 == missing, two trailing
    blank columns, blank trailing rows. 11 numeric pollutant/met series.
    Genuine, heavy native missingness (one column, NMHC(GT), is ~90% missing
    and is dropped from BOTH views).

    Clean: drop the near-empty NMHC column, then drop rows with any remaining
    -200/NaN -> densest fully-observed block.
    Native: same 11 columns, z-scored, NaNs preserved.
    """
    with zipfile.ZipFile(os.path.join(DATA, "airquality_uci.zip")) as z:
        raw = z.read("AirQualityUCI.csv").decode("latin-1")
    df = pd.read_csv(io.StringIO(raw), sep=";", decimal=",")
    # drop trailing unnamed blank columns and blank rows
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(axis=0, how="all")
    # numeric sensor/met columns (drop Date, Time)
    num = df.drop(columns=[c for c in ("Date", "Time") if c in df.columns])
    num = num.apply(pd.to_numeric, errors="coerce")
    # -200 is the missing sentinel
    num = num.mask(num <= -200.0)
    # drop the ~90%-empty NMHC(GT) column from both views
    cov = num.notna().mean(0)
    num = num.loc[:, cov >= 0.20]
    # drop rows that are entirely missing
    num = num.dropna(axis=0, how="all")
    arr = num.to_numpy(float)

    # ---- native (z-scored, NaNs preserved)
    mu = np.nanmean(arr, 0)
    sd = np.nanstd(arr, 0) + 1e-9
    nat_z = (arr - mu) / sd
    obs = np.isfinite(nat_z)
    np.save(os.path.join(DATA, "airquality_native.npy"), nat_z)
    np.save(os.path.join(DATA, "airquality_obsmask.npy"), obs)

    # ---- clean (densest fully-observed block)
    full = arr[np.isfinite(arr).all(1)]
    Xz = _zscore(full)
    Xz = _finite_check(Xz, "airquality")
    np.save(os.path.join(DATA, "airquality_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 4. Appliances Energy  (energy) -- UCI low-energy house, 10-min sampling
# --------------------------------------------------------------------------- #
def load_appliances():
    """Appliances energy prediction (Candanedo 2017): house energy + indoor
    T/RH sensors + weather, 10-min steps over ~4.5 months.

    Complete (no native gaps). Raw 19735 x 29; we drop the two non-physical
    'rv1','rv2' random columns and 'lights' constant-ish, keep 26 sensor series
    and SUBSAMPLE every 4th row (40-min cadence) to stay small/fast -> ~4934 rows.
    Documented choice: subsample factor 4, columns = all numeric except date,
    rv1, rv2.
    """
    with zipfile.ZipFile(os.path.join(DATA, "appliances_energy.zip")) as z:
        raw = z.read("energydata_complete.csv").decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    drop = [c for c in ("date", "rv1", "rv2") if c in df.columns]
    df = df.drop(columns=drop)
    df = df.apply(pd.to_numeric, errors="coerce")
    X = df.to_numpy(float)
    X = X[np.isfinite(X).all(1)]          # already complete
    X = X[::4]                            # subsample to ~40-min cadence
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "appliances")
    np.save(os.path.join(DATA, "appliances_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
# 5. Traffic (PEMS) slice  (traffic) -- spatial road-occupancy panel
# --------------------------------------------------------------------------- #
def load_traffic():
    """San Francisco Bay Area freeway occupancy (Caltrans PEMS), hourly,
    Lai et al. LSTNet benchmark. Raw 17544 x 862 (too wide).

    Documented slice: keep the first 100 sensors and subsample every 5th hour
    -> ~3509 x 100, complete. This is a genuine spatial multivariate panel
    (highly correlated neighbouring sensors) -- good for low-rank + block-missing
    stress later.
    """
    path = os.path.join(DATA, "traffic.txt.gz")
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, header=None)
    X = df.to_numpy(float)
    X = X[:, :100]      # first 100 sensors
    X = X[::5]          # subsample to keep small/fast
    X = X[np.isfinite(X).all(1)]
    Xz = _zscore(X)
    Xz = _finite_check(Xz, "traffic")
    np.save(os.path.join(DATA, "traffic_clean.npy"), Xz)
    return Xz


# --------------------------------------------------------------------------- #
DATASETS_REAL = {
    "fredmd": load_fredmd,
    "exchange": load_exchange,
    "airquality": load_airquality,
    "appliances": load_appliances,
    "traffic": load_traffic,
}

_DESC = {
    "fredmd": "FRED-MD monthly US macro panel (McCracken & Ng); native ragged-edge gaps",
    "exchange": "Daily exchange rates, 8 currencies (LSTNet); complete",
    "airquality": "Italian air-quality multisensor (de Vito); heavy native missingness",
    "appliances": "UCI appliances energy + indoor/weather sensors; complete (4x subsampled)",
    "traffic": "Caltrans PEMS freeway occupancy slice (LSTNet); complete spatial panel",
}

_NATIVE = {  # datasets that also expose a native-missing view
    "fredmd": "fredmd_native.npy",
    "airquality": "airquality_native.npy",
}


def main():
    print(f"{'dataset':12s} {'shape':>14s} {'clean_miss%':>11s} "
          f"{'native_miss%':>13s}  description")
    print("-" * 110)
    for name, loader in DATASETS_REAL.items():
        X = loader()
        clean_miss = 100.0 * (1.0 - np.isfinite(X).mean())
        nat_str = "-"
        if name in _NATIVE:
            nat = np.load(os.path.join(DATA, _NATIVE[name]))
            nat_str = f"{100.0 * (1.0 - np.isfinite(nat).mean()):.1f}"
        print(f"{name:12s} {str(X.shape):>14s} {clean_miss:>10.1f}% "
              f"{nat_str:>12s}%  {_DESC[name]}")
    print("-" * 110)
    print("clean arrays z-scored, finite, saved to data/<name>_clean.npy; "
          "native views to data/<name>_native.npy (+ _obsmask.npy)")


if __name__ == "__main__":
    main()
