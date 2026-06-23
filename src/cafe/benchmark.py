"""One-line, honest imputation benchmark.

    import cafe
    cafe.benchmark()                 # synthetic data, CAFÉ vs baselines, printed table
    cafe.benchmark(df)               # your DataFrame (numpy/pandas/polars)
    cafe.benchmark("beijing")        # a named local dataset + published SOTA references
    r = cafe.benchmark(df); r.plot() # grouped bar chart (causal vs bidirectional)

Design principles (grounded in the imputation-benchmark literature):

* **Real, not asserted.** Every baseline here is *run live* on the same masked data with
  the same seed and scored on the same held-out cells — apples-to-apples. Published
  deep-learning numbers appear only as clearly-labelled **reference rows with a citation**,
  are never silently merged with the live results, and CAFÉ is **never ranked among
  them** (different data variant / mask / windowed protocol — context, not a board).
  All published numbers mirror the one registry ``bench/refs_published.py``; where two
  sources disagree on a cell (e.g. Beijing SAITS .137 vs .155) **both** are shown, each
  tagged with its source — we never pick one. Note CSDI (.102, TSI-Bench) is the lowest
  published Beijing MAE, so no protocol-independent "lowest MAE" claim is made.
* **Causal vs bidirectional is a hard separation.** Causal methods fill X[t] from data
  ≤ t (filtering, backtest-safe, deployable online); bidirectional methods use the whole
  series incl. the future (smoothing — forbidden look-ahead bias in a backtest). The table
  is grouped into the two tiers and sorted within each; they are never ranked as equals.
* **The predict-the-mean trap.** A `global mean` baseline is always included; a method that
  barely beats it has no real skill.
* Standardised (z-score, fit on the train portion only) so MAE/RMSE are comparable to the
  published tables, which all standardise.
* numpy-only (the baselines too) — matching the library's zero-heavy-deps promise.
"""
from __future__ import annotations

import time

import numpy as np

from .io import to_matrix
from .model import impute as _cafe_impute

__all__ = ["benchmark", "BenchmarkResult"]


# --------------------------------------------------------------------------- #
# Missingness injection (held-out cells, fixed seed, same mask for every method)
# --------------------------------------------------------------------------- #
def _inject(X, rate, pattern, seed):
    rng = np.random.default_rng(seed)
    obs = ~np.isnan(X)
    if pattern in ("mcar", "point"):
        return (rng.random(X.shape) < rate) & obs
    if pattern == "block":                       # contiguous per-column gaps (harder)
        T, N = X.shape
        M = np.zeros(X.shape, bool)
        target, placed, guard = int(rate * obs.sum()), 0, 0
        while placed < target and guard < 100000:
            guard += 1
            j = int(rng.integers(N))
            blen = int(rng.integers(max(2, T // 50), max(3, T // 8)))
            s = int(rng.integers(0, max(1, T - blen)))
            seg = slice(s, s + blen)
            placed += int((obs[seg, j] & ~M[seg, j]).sum())
            M[seg, j] = True
        return M & obs
    raise ValueError(f"pattern must be 'mcar'/'point' or 'block'; got {pattern!r}")


def _scaler(X, train_frac):
    """z-score stats from the first ``train_frac`` of rows only (no leakage)."""
    tr = X[: max(1, int(train_frac * X.shape[0]))]
    mu = np.nanmean(tr, 0)
    sd = np.nanstd(tr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return mu, sd


# --------------------------------------------------------------------------- #
# Baselines — pure numpy. Each takes a (T,N) array with NaNs, returns it filled.
# `causal=True` means the fill at t uses only data <= t.
# --------------------------------------------------------------------------- #
def _b_global_mean(X):                            # bidirectional (whole column)
    col = np.where(np.isfinite(np.nanmean(X, 0)), np.nanmean(X, 0), 0.0)
    out = X.copy()
    idx = np.where(np.isnan(out))
    out[idx] = np.take(col, idx[1])
    return out


def _b_mean_past(X):                              # causal expanding mean
    obs = ~np.isnan(X)
    run = np.cumsum(np.where(obs, X, 0.0), 0) / np.maximum(np.cumsum(obs, 0), 1)
    out = np.where(obs, X, run)
    out[np.cumsum(obs, 0) == 0] = 0.0
    return out


def _b_locf(X):                                   # causal last-obs-carried-forward
    T = X.shape[0]
    obs = ~np.isnan(X)
    last = np.where(obs, np.arange(T)[:, None], -1)
    np.maximum.accumulate(last, axis=0, out=last)
    out = np.take_along_axis(np.where(obs, X, 0.0), np.where(last < 0, 0, last), axis=0)
    out[last < 0] = 0.0
    return out


def _b_nocb(X):                                   # bidirectional next-obs-carried-backward
    bwd = _b_locf(X[::-1])[::-1]                   # next future observation
    out = X.copy()
    nan = np.isnan(out)
    out[nan] = bwd[nan]
    return out


def _b_linear(X):                                 # bidirectional linear interpolation
    T, N = X.shape
    xs = np.arange(T)
    out = X.copy()
    for j in range(N):
        c = X[:, j]; o = ~np.isnan(c)
        if o.sum() == 0:
            out[:, j] = 0.0
        elif o.sum() == 1:
            out[:, j] = c[o][0]
        else:
            out[:, j] = np.interp(xs, xs[o], c[o])
    return out


def _b_holt(X, alpha=0.4, beta=0.1, cap=10.0):    # causal double-exponential (level+trend)
    T, N = X.shape
    out = X.copy()
    lev = np.zeros(N); tr = np.zeros(N); age = np.zeros(N); started = np.zeros(N, bool)
    rsum = np.zeros(N); rcnt = np.zeros(N)
    for t in range(T):
        row = X[t]; obs = ~np.isnan(row); miss = ~obs
        if miss.any():
            fb = np.where(rcnt > 0, rsum / np.maximum(rcnt, 1), 0.0)
            fc = lev + np.minimum(age + 1.0, cap) * tr
            fill = np.where(started, fc, fb)
            out[t, miss] = fill[miss]
        if obs.any():
            o = obs
            prev = lev[o]
            nl = np.where(started[o], alpha * row[o] + (1 - alpha) * (prev + tr[o]), row[o])
            nb = np.where(started[o], beta * (nl - prev) + (1 - beta) * tr[o], 0.0)
            lev[o] = nl; tr[o] = nb; age[o] = 0.0; started[o] = True
            rsum[o] += row[o]; rcnt[o] += 1
        age += 1.0
    return np.where(np.isnan(out), 0.0, out)


def _b_kalman(X, q=1e-2, r=1.0):                  # causal local-level Kalman FILTER (no smoother)
    T, N = X.shape
    out = X.copy()
    mu = np.zeros(N); P = np.full(N, 1e3); started = np.zeros(N, bool)
    rsum = np.zeros(N); rcnt = np.zeros(N)
    for t in range(T):
        row = X[t]; obs = ~np.isnan(row); miss = ~obs
        if miss.any():
            fb = np.where(rcnt > 0, rsum / np.maximum(rcnt, 1), 0.0)
            out[t, miss] = np.where(started, mu, fb)[miss]
        if obs.any():
            o = obs
            Pm = P[o] + q
            fresh = ~started[o]
            K = np.where(fresh, 1.0, Pm / (Pm + r))
            mu[o] = np.where(fresh, row[o], mu[o] + K * (row[o] - mu[o]))
            P[o] = np.where(fresh, r, (1.0 - K) * Pm)
            started[o] = True
            rsum[o] += row[o]; rcnt[o] += 1
    return np.where(np.isnan(out), 0.0, out)


def _b_svdimpute(X, rank=10, n_iter=25, tol=1e-4):  # bidirectional fixed-rank iterative SVD
    obs = ~np.isnan(X)
    col = np.where(np.isfinite(np.nanmean(X, 0)), np.nanmean(X, 0), 0.0)
    F = np.where(obs, X, col)
    r = max(1, min(rank, min(X.shape) - 1))
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        rec = (U[:, :r] * s[:r]) @ Vt[:r]
        new = np.where(obs, X, rec)
        if np.linalg.norm(new - F) / (np.linalg.norm(F) + 1e-9) < tol:
            return new
        F = new
    return F


def _b_softimpute(X, n_iter=50, tol=1e-3):        # bidirectional nuclear-norm matrix completion
    obs = ~np.isnan(X)
    col = np.where(np.isfinite(np.nanmean(X, 0)), np.nanmean(X, 0), 0.0)
    F = np.where(obs, X, col)
    lam = None
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        if lam is None:
            lam = 0.05 * s[0]
        new = np.where(obs, X, (U * np.maximum(s - lam, 0.0)) @ Vt)
        if np.linalg.norm(new - F) / (np.linalg.norm(F) + 1e-9) < tol:
            return new
        F = new
    return F


# (name, fn, causal)
_BASELINES = [
    ("CAFÉ",          lambda Xo: np.asarray(_cafe_impute(Xo), float), True),
    ("mean-of-past",  _b_mean_past,  True),
    ("LOCF",          _b_locf,       True),
    ("Holt (lvl+tr)", _b_holt,       True),
    ("Kalman filter", _b_kalman,     True),
    ("global mean",   _b_global_mean, False),
    ("NOCB",          _b_nocb,       False),
    ("linear interp", _b_linear,     False),
    ("SVDImpute",     _b_svdimpute,  False),
    ("SoftImpute",    _b_softimpute, False),
]


# --------------------------------------------------------------------------- #
# Published reference numbers.
#
# SINGLE SOURCE OF TRUTH: bench/refs_published.py (the structured `REFS`
# registry). That file lives in the research harness (not in the installed
# package), so it cannot be imported here at runtime; the rows below are a
# verbatim MIRROR of it and MUST be kept in sync with it — do not edit values
# here without editing bench/refs_published.py. This replaces the repo's old
# second, contradictory registry.
#
# All entries are BIDIRECTIONAL + GPU (each fill sees the whole series, incl.
# the future) under a WINDOWED train/val/test protocol — a DIFFERENT data
# variant/mask/setting from CAFÉ's full-series causal online imputation. They
# are shown as clearly-labelled CONTEXT, never as a like-for-like leaderboard,
# and CAFÉ is deliberately not ranked among them.
#
# Where two sources disagree on the same (method, dataset) cell we keep BOTH,
# each tagged with its source (e.g. Beijing SAITS .137 [du2023] vs .155
# [tsibench]) — we never silently pick one. Note that under the TSI-Bench
# source CSDI (.102) is the strongest Beijing MAE, so there is no protocol-
# independent "lowest MAE" claim.
#
# Each row: (method, MAE, source-tag). Source tags resolve via _SOURCES below.
# --------------------------------------------------------------------------- #
_SOURCES = {
    "du2023":   "Du et al. 2023, SAITS (ESWA), Table 2",
    "tsibench": "TSI-Bench, arXiv:2406.12747 (NeurIPS'24 D&B)",
    "fgti2024": "FGTI, NeurIPS'24 (frequency-domain diffusion)",
}

_PUBLISHED = {
    # Beijing Multi-Site Air-Quality — MAE @ 10% MCAR-point.
    # du2023 and TSI-Bench disagree on every shared method: BOTH are kept.
    "beijing": [
        ("CSDI",         0.102, "tsibench"),   # strongest under tsibench (< CAFÉ .108)
        ("iTransformer", 0.123, "tsibench"),
        ("BRITS",        0.127, "tsibench"),
        ("BRITS",        0.153, "du2023"),
        ("Transformer",  0.142, "tsibench"),
        ("Transformer",  0.158, "du2023"),
        ("SAITS",        0.137, "du2023"),
        ("SAITS",        0.155, "tsibench"),
        ("FGTI",         0.149, "fgti2024"),
        ("GP-VAE",       0.268, "du2023"),
        ("M-RNN",        0.294, "du2023"),
    ],
    # PhysioNet-2012 — MAE @ 10% (SAITS Table 2).
    "physionet": [
        ("SAITS",        0.186, "du2023"),
        ("Transformer",  0.190, "du2023"),
        ("BRITS",        0.256, "du2023"),
        ("M-RNN",        0.533, "du2023"),
    ],
    # Electricity — MAE @ 10% (SAITS Table 2).
    "electricity": [
        ("SAITS",        0.735, "du2023"),
        ("Transformer",  0.823, "du2023"),
        ("BRITS",        0.847, "du2023"),
    ],
}

_LOCAL = {
    "beijing":  ("beijing_clean.npy", "npy"),
    "etth1":    ("ETTh1_clean.npy",   "npy"),
    "airq":     ("airq_normal.txt",   "txt"),
    "chlorine": ("chlorine_normal.txt", "txt"),
    "temp":     ("temp_normal.txt",   "txt"),
}


def _synthetic(T=600, N=20, seed=0):
    rng = np.random.default_rng(seed)
    L = 4
    Z = np.zeros((T, L)); Z[0] = rng.standard_normal(L)
    for t in range(1, T):
        Z[t] = 0.95 * Z[t - 1] + np.sqrt(1 - 0.95 ** 2) * rng.standard_normal(L)
    X = Z @ rng.standard_normal((N, L)).T
    t = np.arange(T)
    X += 0.6 * np.sin(2 * np.pi * t[:, None] / 24 + rng.standard_normal(N))
    return X + 0.1 * X.std() * rng.standard_normal((T, N))


def _resolve(data):
    if data is None:
        return _synthetic(), "synthetic", None
    if isinstance(data, str):
        key = data.lower()
        if key not in _LOCAL:
            raise ValueError(f"unknown dataset {data!r}; known: {list(_LOCAL)} "
                             "(or pass a numpy/pandas/polars object directly)")
        import os
        fn, kind = _LOCAL[key]
        for base in ("data", os.path.join("..", "data"),
                     os.path.join(os.path.dirname(__file__), "..", "..", "data")):
            p = os.path.join(base, fn)
            if os.path.exists(p):
                X = np.load(p) if kind == "npy" else np.loadtxt(p)
                return np.asarray(X, float), key, (key if key in _PUBLISHED else None)
        raise FileNotFoundError(
            f"dataset {data!r} ({fn}) not found under ./data or ../data. Named datasets "
            "are a dev convenience; pass your own DataFrame instead."
        )
    X, _ = to_matrix(data)
    return np.asarray(X, float), "data", None


def _score(truth, pred, M):
    e = pred[M] - truth[M]
    finite = np.isfinite(e)
    e = e[finite]
    t = truth[M][finite]
    return dict(
        mae=float(np.mean(np.abs(e))),
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mre=float(np.sum(np.abs(e)) / (np.sum(np.abs(t)) + 1e-12)),
    )


class BenchmarkResult:
    """Outcome of :func:`benchmark`. Prints grouped by kind (causal first, then
    bidirectional), each sorted by MAE. ``.plot()`` charts it; ``.to_pandas()`` returns
    the rows; ``.published`` holds a list of ``(method, MAE, source-tag)`` cited
    reference rows (bidirectional, different-protocol context — never head-to-head)."""

    def __init__(self, name, rows, published, meta):
        # causal block first, bidir block second; within each, best MAE first; failures last
        self.rows = sorted(rows, key=lambda r: (not r["ok"], not r["causal"], r["mae"]))
        self.name = name
        self.published = published
        self.meta = meta

    @property
    def best_causal(self):
        ok = [r for r in self.rows if r["ok"] and r["causal"]]
        return min(ok, key=lambda r: r["mae"]) if ok else None

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.rows)

    def __repr__(self):
        m = self.meta
        best = self.best_causal
        L = [f"CAFÉ benchmark — {self.name}  "
             f"({m['shape'][0]}×{m['shape'][1]}, {int(m['missing']*100)}% {m['pattern']} "
             f"missing, seed {m['seed']}, standardised)"]
        hdr = f"  {'method':16s} {'MAE':>8s} {'RMSE':>8s} {'MRE':>7s} {'time':>8s}"
        last_kind = None
        for r in self.rows:
            if r["causal"] != last_kind:
                last_kind = r["causal"]
                L += ["", "CAUSAL  (online / point-in-time, X[t] from data ≤ t — backtest-safe)"
                      if r["causal"] else
                      "BIDIRECTIONAL  (uses the whole series incl. the future — smoothing)", hdr]
            star = " ★" if (best and r["method"] == best["method"]) else ""
            if r["ok"]:
                L.append(f"  {r['method']:16s} {r['mae']:8.3f} {r['rmse']:8.3f} "
                         f"{r['mre']*100:6.1f}% {r['time_s']:7.2f}s{star}")
            else:
                L.append(f"  {r['method']:16s} {'FAILED':>8s}  ({r.get('err','')[:28]})")
        if self.published:
            prows = sorted(self.published, key=lambda r: r[1])
            used = sorted({tag for _, _, tag in prows})
            L += ["",
                  "PUBLISHED reference — CONTEXT ONLY, *not* a head-to-head leaderboard.",
                  "  All rows below are BIDIRECTIONAL + GPU (each fill sees the whole",
                  "  series incl. the future), under a windowed train/val/test protocol",
                  "  on a DIFFERENT data variant/mask. CAFÉ above runs the FULL series",
                  "  causally/online — so CAFÉ is deliberately NOT ranked among these.",
                  "  Where sources disagree both values are kept, each tagged.",
                  f"  {'method':16s} {'MAE':>8s}  source"]
            for nm, mae, tag in prows:
                L.append(f"  {nm:16s} {mae:8.3f}  [{tag}]")
            L += ["", "  sources:  " + ";  ".join(
                f"{k} = {_SOURCES[k]}" for k in used)]
        L += ["", "★ = best causal method (CAFÉ vs causal baselines, run LIVE here)."]
        return "\n".join(L)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        ok = [r for r in self.rows if r["ok"]]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.46 * len(ok) + 1.2))
        y = np.arange(len(ok))[::-1]
        maes = [r["mae"] for r in ok]
        colors = ["#2c7fb8" if r["causal"] else "#bdbdbd" for r in ok]
        ax.barh(y, maes, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels([("★ " if (self.best_causal and r["method"] == self.best_causal["method"])
                             else "") + r["method"] for r in ok])
        for yi, v in zip(y, maes):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=9)
        # divider between the causal block (top) and bidirectional block (bottom)
        n_causal = sum(r["causal"] for r in ok)
        if 0 < n_causal < len(ok):
            ax.axhline(len(ok) - n_causal - 0.5, color="0.3", lw=0.8, ls=":")
        if self.published:
            # Best published BIDIRECTIONAL number, drawn as a context line only
            # (different protocol — not a like-for-like target).
            ceil = min(v for _, v, _ in self.published)
            ax.axvline(ceil, color="C3", ls="--", lw=1)
            ax.text(ceil, 0.1, "  best published\n  (bidir, diff. protocol)", color="C3",
                    fontsize=8, va="bottom", ha="left")
        ax.set_xlabel("MAE (standardised, lower = better)")
        ax.set_title(f"{self.name}: causal (blue, top) vs bidirectional (grey, bottom)")
        return ax


def benchmark(data=None, *, missing=0.1, pattern="mcar", seed=0, train_frac=0.6,
              methods=None, standardize=True, verbose=True):
    """Run CAFÉ against baseline imputers on ``data`` and return a :class:`BenchmarkResult`.

    Parameters
    ----------
    data : None | array | DataFrame | str
        ``None`` -> a self-contained synthetic dataset; a numpy/pandas/polars object ->
        your data (numeric columns); a name like ``"beijing"`` -> a local dataset (dev)
        plus published SOTA reference rows.
    missing : float
        Fraction of observed cells to hold out and score on.
    pattern : 'mcar' | 'block'
        Scattered (easy) vs contiguous per-column gaps (hard).
    seed : int
        The *same* mask is used for every method.
    """
    X, name, pub_key = _resolve(data)
    if standardize:
        mu, sd = _scaler(X, train_frac)
        Z = (X - mu) / sd
    else:
        Z = X.astype(float).copy()
    M = _inject(Z, missing, pattern, seed)
    if not M.any():
        raise ValueError("no cells held out — increase `missing` or check the data has observed values")
    Xo = Z.copy(); Xo[M] = np.nan

    rows = []
    for nm, fn, causal in (methods or _BASELINES):
        t0 = time.perf_counter()
        try:
            sc = _score(Z, np.asarray(fn(Xo.copy()), float), M)
            sc.update(method=nm, causal=causal, time_s=time.perf_counter() - t0, ok=True)
        except Exception as e:                                            # noqa: BLE001
            sc = dict(mae=float("inf"), rmse=float("inf"), mre=float("inf"),
                      method=nm, causal=causal, time_s=float("nan"), ok=False, err=str(e))
        rows.append(sc)

    res = BenchmarkResult(name, rows, _PUBLISHED.get(pub_key),
                          dict(missing=missing, pattern=pattern, seed=seed, shape=X.shape))
    if verbose:
        print(res)
    return res
