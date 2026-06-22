"""Prepare real point-protocol datasets into clean (T,N) arrays for arena.py."""
import os, io, zipfile
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def prep_etth1():
    df = pd.read_csv(os.path.join(DATA, "ETTh1.csv"), parse_dates=["date"]).drop(columns=["date"])
    X = df.to_numpy(float)
    X = X[np.isfinite(X).all(1)]                       # already clean
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)           # z-score (TSI-Bench scale)
    np.save(os.path.join(DATA, "ETTh1_clean.npy"), Xz)
    print(f"ETTh1_clean.npy {Xz.shape}")


def prep_beijing():
    cont = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    outer = zipfile.ZipFile(os.path.join(DATA, "beijing.zip"))
    inner_name = [n for n in outer.namelist() if n.endswith(".zip")][0]
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    csvs = sorted(n for n in inner.namelist() if n.endswith(".csv") and "PRSA" in n)
    cols = {}
    for c in csvs:
        df = pd.read_csv(io.BytesIO(inner.read(c)))
        stn = df["station"].iloc[0]
        idx = pd.to_datetime(df[["year", "month", "day", "hour"]])
        for v in cont:
            cols[f"{stn}_{v}"] = pd.Series(df[v].to_numpy(float), index=idx)
    wide = pd.DataFrame(cols).sort_index()             # (T, 12*11=132)
    # take the longest fully-observed contiguous slice for clean ground truth
    full = wide.dropna(axis=0, how="any")
    X = full.to_numpy(float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    np.save(os.path.join(DATA, "beijing_clean.npy"), Xz)
    print(f"beijing_clean.npy {Xz.shape}  (kept {len(full)}/{len(wide)} fully-observed rows)")


if __name__ == "__main__":
    prep_etth1()
    try:
        prep_beijing()
    except Exception as e:
        print(f"beijing prep failed: {type(e).__name__}: {e}")
