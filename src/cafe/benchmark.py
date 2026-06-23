"""One-line, honest imputation benchmark.

    import cafe
    cafe.benchmark()                 # synthetic data, CAFÉ vs baselines, printed table
    cafe.benchmark(df)               # your DataFrame (numpy/pandas/polars)
    cafe.benchmark("beijing")        # a named local dataset + published SOTA references
    r = cafe.benchmark(df); r.plot() # bar chart of MAE by method

Design principles (grounded in the imputation-benchmark literature):

* **Real, not asserted.** Every baseline here is *run live* on the same masked data
  with the same seed, and scored on the same held-out cells -- so the comparison is
  apples-to-apples. Published deep-learning numbers (SAITS, BRITS, CSDI, ...) are shown
  only as clearly-labelled **reference rows with a citation**; they are never silently
  mixed with the live results.
* **Causal vs bidirectional is a hard separation.** Almost all published SOTA imputers
  (SAITS, BRITS, CSDI, TimesNet, ImputeFormer, foundation models) impute X[t] using the
  *whole window incl. the future* -- a smoothing task. CAFÉ and the causal baselines use
  only data <= t (filtering). A bidirectional method has a structural information
  advantage, so the table marks each method ``causal`` or ``bidir`` and never ranks them
  as if equal.
* **The predict-the-mean trap.** A `global mean` baseline is always included; if a method
  barely beats it, its low error is the trap, not skill.
* Standardised (z-score, **fit on the train portion only**) so MAE/RMSE are comparable to
  the published tables, which all standardise.
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
        target, placed = int(rate * obs.sum()), 0
        guard = 0
        while placed < target and guard < 100000:
            guard += 1
            j = int(rng.integers(N))
            blen = int(rng.integers(max(2, T // 50), max(3, T // 8)))
            s = int(rng.integers(0, max(1, T - blen)))
            seg = slice(s, s + blen)
            newly = int((obs[seg, j] & ~M[seg, j]).sum())
            M[seg, j] = True
            placed += newly
        return M & obs
    raise ValueError(f"pattern must be 'mcar'/'point' or 'block'; got {pattern!r}")


def _scaler(X, train_frac):
    """z-score stats from the first ``train_frac`` of rows only (no leakage)."""
    T = X.shape[0]
    tr = X[: max(1, int(train_frac * T))]
    mu = np.nanmean(tr, 0)
    sd = np.nanstd(tr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return mu, sd


# --------------------------------------------------------------------------- #
# Baselines -- pure numpy, each takes a (T,N) array with NaNs and returns it filled.
# `causal=True` means the fill at t uses only data <= t.
# --------------------------------------------------------------------------- #
def _b_global_mean(X):                            # bidirectional (uses whole column)
    col = np.nanmean(X, 0)
    col = np.where(np.isfinite(col), col, 0.0)
    out = X.copy()
    idx = np.where(np.isnan(out))
    out[idx] = np.take(col, idx[1])
    return out


def _b_mean_past(X):                              # causal expanding mean
    obs = ~np.isnan(X)
    vals = np.where(obs, X, 0.0)
    run = np.cumsum(vals, 0) / np.maximum(np.cumsum(obs, 0), 1)
    out = np.where(obs, X, run)
    out[np.cumsum(obs, 0) == 0] = 0.0             # leading gap -> mean (0 in z-space)
    return out


def _b_locf(X):                                   # causal last-observation-carried-forward
    T = X.shape[0]
    obs = ~np.isnan(X)
    last = np.where(obs, np.arange(T)[:, None], -1)
    np.maximum.accumulate(last, axis=0, out=last)
    safe = np.where(last < 0, 0, last)
    out = np.take_along_axis(np.where(obs, X, 0.0), safe, axis=0)
    out[last < 0] = 0.0
    return out


def _b_linear(X):                                 # bidirectional (interpolates to future point)
    T, N = X.shape
    xs = np.arange(T)
    out = X.copy()
    for j in range(N):
        c = X[:, j]
        o = ~np.isnan(c)
        if o.sum() == 0:
            out[:, j] = 0.0
        elif o.sum() == 1:
            out[:, j] = c[o][0]
        else:
            out[:, j] = np.interp(xs, xs[o], c[o])
    return out


def _b_softimpute(X, n_iter=50, tol=1e-3):        # bidirectional low-rank matrix completion
    obs = ~np.isnan(X)
    col = np.nanmean(X, 0)
    col = np.where(np.isfinite(col), col, 0.0)
    filled = np.where(obs, X, col)
    lam = None
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(filled, full_matrices=False)
        if lam is None:
            lam = 0.05 * s[0]                     # shrinkage from the spectrum scale
        low = (U * np.maximum(s - lam, 0.0)) @ Vt
        new = np.where(obs, X, low)
        if np.linalg.norm(new - filled) / (np.linalg.norm(filled) + 1e-9) < tol:
            return new
        filled = new
    return filled


# (name, fn, causal)
_BASELINES = [
    ("CAFÉ",          lambda Xo: np.asarray(_cafe_impute(Xo), float), True),
    ("mean-of-past",  _b_mean_past,  True),
    ("LOCF",          _b_locf,       True),
    ("global mean",   _b_global_mean, False),
    ("linear interp", _b_linear,     False),
    ("SoftImpute",    _b_softimpute, False),
]


# --------------------------------------------------------------------------- #
# Published reference numbers (standardised MAE @ 10% point/MCAR). ALL bidirectional.
# Cited, never re-run, shown only as context. Sources verified from the papers.
# --------------------------------------------------------------------------- #
_PUBLISHED = {
    "beijing": [
        ("CSDI",        0.102, "TSI-Bench 2024 (arXiv:2406.12747)"),
        ("SAITS",       0.137, "Du+2023 ESWA, Table 2"),
        ("BRITS",       0.153, "Du+2023 ESWA, Table 2"),
        ("Transformer", 0.158, "Du+2023 ESWA, Table 2"),
        ("GP-VAE",      0.268, "Du+2023 ESWA, Table 2"),
    ],
    "physionet": [
        ("SAITS",       0.186, "Du+2023 ESWA, Table 2"),
        ("Transformer", 0.190, "Du+2023 ESWA, Table 2"),
        ("CSDI",        0.217, "Tashiro+2021 NeurIPS, Table 3"),
        ("BRITS",       0.256, "Du+2023 ESWA, Table 2"),
    ],
    "electricity": [
        ("SSSD",        0.345, "Alcaraz+2022 TMLR, Table 3"),
        ("SAITS",       0.735, "Du+2023 ESWA, Table 2"),
        ("Transformer", 0.823, "Du+2023 ESWA, Table 2"),
        ("BRITS",       0.847, "Du+2023 ESWA, Table 2"),
    ],
}

# local files (dev convenience) — name -> (filename, loader)
_LOCAL = {
    "beijing":  ("beijing_clean.npy", "npy"),
    "etth1":    ("ETTh1_clean.npy",   "npy"),
    "airq":     ("airq_normal.txt",   "txt"),
    "chlorine": ("chlorine_normal.txt", "txt"),
    "temp":     ("temp_normal.txt",   "txt"),
}


def _synthetic(T=600, N=20, seed=0):
    """A self-contained low-rank + AR + seasonal + noise matrix (so the zero-arg call
    always works, anywhere, with no data files)."""
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
    """Return (clean_matrix, name, published_key)."""
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
            f"dataset {data!r} ({fn}) not found under ./data or ../data. "
            "Named datasets are a dev convenience; pass your own DataFrame instead."
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
    """The outcome of :func:`benchmark`. Prints as a ranked table; ``.plot()`` charts it;
    ``.to_pandas()`` returns the rows; ``.published`` holds the cited reference rows."""

    def __init__(self, name, rows, published, meta):
        self.name = name
        self.rows = sorted(rows, key=lambda r: (not r["ok"], r["mae"]))
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
        L = [f"CAFÉ benchmark — {self.name}  "
             f"({m['shape'][0]}×{m['shape'][1]}, {int(m['missing']*100)}% {m['pattern']} "
             f"missing, seed {m['seed']}, standardised)",
             f"{'method':16s} {'kind':6s} {'MAE':>8s} {'RMSE':>8s} {'MRE':>7s} {'time':>8s}",
             "-" * 56]
        best = self.best_causal
        for r in self.rows:
            star = "  ★" if (best and r["method"] == best["method"]) else ""
            kind = "causal" if r["causal"] else "bidir"
            if r["ok"]:
                L.append(f"{r['method']:16s} {kind:6s} {r['mae']:8.3f} {r['rmse']:8.3f} "
                         f"{r['mre']*100:6.1f}% {r['time_s']:7.2f}s{star}")
            else:
                L.append(f"{r['method']:16s} {kind:6s} {'FAILED':>8s} ({r.get('err','')[:24]})")
        if self.published:
            L += ["", "published reference (all BIDIRECTIONAL — use future context; "
                  "not run here, cited for context):",
                  f"{'method':16s} {'kind':6s} {'MAE':>8s}   source"]
            for nm, mae, src in self.published:
                L.append(f"{nm:16s} {'bidir':6s} {mae:8.3f}   {src}")
        L += ["", "causal = fills X[t] from data ≤ t only (backtest-safe); "
              "bidir = uses the whole window incl. the future.",
              "★ = best causal method."]
        return "\n".join(L)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        ok = [r for r in self.rows if r["ok"]]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.5 * len(ok) + 1))
        names = [r["method"] for r in ok]
        maes = [r["mae"] for r in ok]
        colors = ["#2c7fb8" if r["causal"] else "#bdbdbd" for r in ok]
        y = np.arange(len(ok))[::-1]
        ax.barh(y, maes, color=colors)
        ax.set_yticks(y); ax.set_yticklabels(names)
        for yi, v in zip(y, maes):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=9)
        if self.published:
            best_pub = min(p[1] for p in self.published)
            ax.axvline(best_pub, color="C3", ls="--", lw=1)
            ax.text(best_pub, 0.15, "  best published\n  (bidirectional)",
                    color="C3", fontsize=8, va="bottom", ha="left")
        ax.set_xlabel("MAE (standardised, lower = better)")
        ax.set_title(f"{self.name}: imputation MAE  (blue = causal, grey = bidirectional)")
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
        raise ValueError("no cells were held out — increase `missing` or check the data has observed values")
    Xo = Z.copy(); Xo[M] = np.nan

    rows = []
    for nm, fn, causal in (methods or _BASELINES):
        t0 = time.perf_counter()
        try:
            pred = np.asarray(fn(Xo.copy()), float)
            sc = _score(Z, pred, M)
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
