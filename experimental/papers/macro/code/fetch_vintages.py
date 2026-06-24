"""Fetch GENUINE real-time vintage data from ALFRED (archival FRED).

This is the gold-standard Croushore-Stark real-time protocol: for each historical
*vintage date* we download every series exactly as it was known on that date, including
the ragged edge (publication lag) that the data actually had. Nothing is simulated.

Source endpoint (returns one column per vintage, NaN before first availability):
    https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=<SERIES>&vintage_date=YYYY-MM-DD

We cache every CSV under data/alfred_cache/ so the experiment is reproducible offline
once fetched. We also fetch the FINAL (latest) vintage of the target, which is the
"truth" each real-time nowcast is scored against.

Target  : INDPRO  -- Industrial Production index, a canonical monthly coincident-activity
                     series central banks nowcast (also the headline of FRED-MD).
Predictors: a basket of monthly FRED-MD activity / labour / sales indicators whose ALFRED
            vintages are available (verified by probe). All strictly real-time.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import urllib.parse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(DATA, "alfred_cache")
os.makedirs(CACHE, exist_ok=True)

TARGET = "INDPRO"

# Monthly FRED-MD predictors with confirmed ALFRED vintage coverage. Chosen to span
# output, labour, income, consumption, and orders -- the standard nowcasting cross-section.
PREDICTORS = [
    "INDPRO",            # the target's own past releases (autoregressive content)
    "PAYEMS",            # nonfarm payroll employment
    "UNRATE",            # unemployment rate
    "RPI",               # real personal income
    "DPCERA3M086SBEA",   # real personal consumption expenditures
    "CUMFNS",            # capacity utilisation, manufacturing
    "DGORDER",           # manufacturers' new orders, durable goods
    "AWHMAN",            # average weekly hours, manufacturing
    "HOUST",             # housing starts
    "MANEMP",            # manufacturing employment
    "CE16OV",            # civilian employment
]

BASE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
HDRS = {"User-Agent": "Mozilla/5.0 (research; real-time-nowcasting)"}


def _vintage_dates(start="2015-07-01", end="2025-05-01", step=1):
    """Vintage (nowcast-origin) dates: the 1st of each month in the window, every
    ``step`` months. ``step=1`` is monthly (full real-time cadence)."""
    all_m = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="MS")]
    return all_m[::step]


def _fetch(series: str, vintage: str) -> pd.Series | None:
    """Download (and cache) one series at one vintage. Returns a date-indexed Series, or
    None if the series does not exist at that vintage (404)."""
    safe = series.replace("/", "_")
    fn = os.path.join(CACHE, f"{safe}__{vintage}.csv")
    if os.path.exists(fn):
        txt = open(fn).read()
        if txt.startswith("__404__"):
            return None
    else:
        url = f"{BASE}?id={urllib.parse.quote(series)}&vintage_date={vintage}"
        # curl is markedly more reliable against ALFRED than urllib (which the server
        # intermittently throttles to a hang); retry a few times on transient failure.
        txt = None
        for attempt in range(6):
            try:
                # NB: plain curl (default HTTP/2, default UA) is what ALFRED serves
                # reliably; --http1.1 and a custom UA get throttled to a hang here.
                out = subprocess.run(
                    ["curl", "-sL", "-m", "40", url],
                    capture_output=True, text=True, timeout=55)
                if out.returncode == 0 and out.stdout:
                    txt = out.stdout
                    break
            except Exception:
                pass
            # ALFRED rate-limits bursts (empty body / rc 92 / rc 000): back off, but
            # cap the escalation so one transient failure does not stall the whole run.
            time.sleep(min(5.0 + 3.0 * attempt, 20.0))
        if txt is None:
            return None  # transient failure: skip this (series, vintage); not cached
        low = txt.lstrip().lower()
        if low.startswith("404") or "<html" in low[:200]:
            open(fn, "w").write("__404__")
            return None
        if not low.startswith("observation_date"):
            open(fn, "w").write("__404__")
            return None
        open(fn, "w").write(txt)
        time.sleep(3.0)  # be polite: ALFRED blocks bursts, so fetch slowly + serially
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.Series(df["value"].values, index=df["date"].values, dtype=float)
    return s


def fetch_all(start="2015-07-01", end="2025-05-01", step=2):
    vintages = _vintage_dates(start, end, step=step)
    print(f"fetching {len(PREDICTORS)} series x {len(vintages)} vintages from ALFRED ...")
    # Per-vintage panel of predictors (each is the as-known-at-vintage path).
    panel = {}     # vintage -> DataFrame(date x series), real-time
    target_rt = {}  # vintage -> Series of INDPRO as known at that vintage
    for vi, v in enumerate(vintages):
        cols = {}
        ok = 0
        for s in PREDICTORS:
            ser = _fetch(s, v)
            if ser is not None and ser.notna().sum() > 24:
                cols[s] = ser
                ok += 1
        print(f"  vintage {v}: {ok}/{len(PREDICTORS)} series", flush=True)
        if TARGET in cols:
            target_rt[v] = cols[TARGET].dropna()
        if cols:
            panel[v] = pd.DataFrame(cols)
        if (vi + 1) % 12 == 0:
            print(f"  ... {vi+1}/{len(vintages)} vintages")
    # FINAL revised target = latest vintage's full path
    final = _fetch(TARGET, vintages[-1]).dropna()
    print(f"done. {len(panel)} usable vintages; final target spans "
          f"{final.index.min().date()}..{final.index.max().date()}")
    return panel, target_rt, final


if __name__ == "__main__":
    panel, target_rt, final = fetch_all()
    # quick integrity report on the ragged edge
    print("\nRAGGED-EDGE CHECK (last observed month per vintage, target vs predictors):")
    for v in list(panel)[::18]:
        df = panel[v]
        tgt_last = target_rt[v].index.max().date() if v in target_rt else None
        pred_last = df.index.max().date()
        print(f"  vintage {v}: predictors end {pred_last}, INDPRO ends {tgt_last}")
    # save a compact npz snapshot of what we need downstream
    import pickle
    with open(os.path.join(DATA, "vintages.pkl"), "wb") as f:
        pickle.dump({"panel": panel, "target_rt": target_rt, "final": final,
                     "predictors": PREDICTORS, "target": TARGET}, f)
    print(f"\nsaved {os.path.join(DATA, 'vintages.pkl')}")
