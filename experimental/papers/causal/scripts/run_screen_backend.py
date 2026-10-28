#!/usr/bin/env python
"""Bit-packed vs dense lag screening -- a LIVE re-run of our own null result.

``cafe._screen`` ships two backends for the backward-only lag-screening statistic:

``dense``  bits -> {-1, 0, +1} float32, then one BLAS-3 ``D[k:].T @ D[:-k]`` per lag.
``bits``   ``np.packbits`` + a uint16 popcount lookup table.

The bit-packed path was implemented on the expectation that it would win (it is the
mechanism behind ARROW's reported speed-ups).  It does not.  This script re-measures
the claim rather than transcribing it: for each ``(P, stat)`` cell it times both
backends on identical input with the permutation null disabled (``n_perm=0``), so the
comparison is the statistic only, and it checks that the two produce bit-identical
statistic surfaces (``max|dS|``) -- a speed claim about a different answer is worthless.

Writes ``data/screen_backend.json``.

    .venv-bench/bin/python experimental/papers/causal/scripts/run_screen_backend.py
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PAPER)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cafe._screen import screen_lags  # noqa: E402

N = 4000
MAX_LAG = 8
PS = [20, 50, 100, 200]
STATS = ["phi", "arrow"]
REPEATS = 3


def make_panel(P, n=N, seed=0, p_miss=0.0):
    """A sparse-VAR residual panel: real serial structure, not white noise."""
    rng = np.random.default_rng(seed)
    A = np.zeros((P, P))
    A[np.arange(P), np.arange(P)] = 0.5
    for _ in range(2 * P):
        i, j = rng.integers(0, P, 2)
        if i != j:
            A[i, j] = rng.normal(scale=0.3)
    A *= min(1.0, 0.9 / max(np.abs(np.linalg.eigvals(A)).max(), 1e-9))
    X = np.zeros((n, P))
    for t in range(1, n):
        X[t] = A @ X[t - 1] + rng.normal(size=P)
    if p_miss:
        X[rng.random(X.shape) < p_miss] = np.nan
    return X


def time_backend(X, method, stat):
    best = float("inf")
    out = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        r = screen_lags(X, max_lag=MAX_LAG, stat=stat, method=method, n_perm=0)
        dt = time.perf_counter() - t0
        best = min(best, dt)
        out = r
    return best, out


def main():
    rows = []
    for P in PS:
        X = make_panel(P)
        for stat in STATS:
            t_dense, r_dense = time_backend(X, "dense", stat)
            t_bits, r_bits = time_backend(X, "bits", stat)
            d = float(np.nanmax(np.abs(np.asarray(r_dense.S, float)
                                       - np.asarray(r_bits.S, float))))
            rows.append(dict(P=P, stat=stat, n=N, max_lag=MAX_LAG,
                             dense_s=t_dense, bits_s=t_bits,
                             ratio=t_bits / t_dense, max_abs_dS=d,
                             auto=r_dense.meta.get("method")))
            print(f"P={P:<4} {stat:<6} dense={t_dense:.4f}s bits={t_bits:.4f}s "
                  f"ratio={t_bits / t_dense:5.1f}x  max|dS|={d:.1e}")

    # one missing-data cell, to show the ordering is not an artefact of a full panel
    Xm = make_panel(100, p_miss=0.20, seed=1)
    t_dense, r_dense = time_backend(Xm, "dense", "phi")
    t_bits, r_bits = time_backend(Xm, "bits", "phi")
    mcar = dict(P=100, stat="phi", p_miss=0.20, dense_s=t_dense, bits_s=t_bits,
                ratio=t_bits / t_dense,
                max_abs_dS=float(np.nanmax(np.abs(np.asarray(r_dense.S, float)
                                                  - np.asarray(r_bits.S, float)))))
    print(f"P=100 phi 20% MCAR  dense={t_dense:.4f}s bits={t_bits:.4f}s "
          f"ratio={mcar['ratio']:5.1f}x")

    # what the permutation null costs, at the default n_perm
    X = make_panel(50)
    t0 = time.perf_counter()
    r = screen_lags(X, max_lag=MAX_LAG, stat="phi", method="dense", n_perm=200)
    t_full = time.perf_counter() - t0
    perm_share = r.meta["t_perm_s"] / max(r.meta["t_total_s"], 1e-12)

    out = dict(
        rows=rows, mcar=mcar,
        perm=dict(P=50, n_perm=200, total_s=t_full,
                  t_stat_s=r.meta["t_stat_s"], t_perm_s=r.meta["t_perm_s"],
                  perm_share=perm_share),
        crossover=None,
        n_cells=len(rows),
        bits_never_wins=bool(all(x["ratio"] > 1.0 for x in rows)),
        ratio_range=[min(x["ratio"] for x in rows), max(x["ratio"] for x in rows)],
        identical=bool(max(x["max_abs_dS"] for x in rows) == 0.0),
        env=dict(platform=platform.platform(), numpy=np.__version__,
                 python=platform.python_version()),
    )
    os.makedirs(os.path.join(PAPER, "data"), exist_ok=True)
    with open(os.path.join(PAPER, "data", "screen_backend.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nbits loses in {sum(x['ratio'] > 1 for x in rows)}/{len(rows)} cells; "
          f"ratio range {out['ratio_range'][0]:.1f}x-{out['ratio_range'][1]:.1f}x; "
          f"surfaces identical: {out['identical']}")
    print(f"permutation null is {100 * perm_share:.0f}% of the default-config wall clock")
    print("wrote data/screen_backend.json")


if __name__ == "__main__":
    main()
