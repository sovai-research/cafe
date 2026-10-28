#!/usr/bin/env python
"""Engine-router probe for the CAFE-CD results paper -- a LIVE run, not a transcription.

Question
--------
CAFE-CD routes a panel to a pooled or a per-entity loading engine using
``cafe.loading_homogeneity_ratio``: the between-entity alignment of half-sample
loading subspaces divided by the within-entity alignment of the same objects.  The
denominator is the sampling-noise ceiling that *identical* loadings would produce, so
the statistic has a null of 1 by construction and needs no fitted cutoff.

Two things are measured here, both end-to-end:

1. **Does a fixed cutoff on the RAW between-entity alignment work?**  We build four
   homogeneous-loading panels (short + noisy) and four heterogeneous-loading panels
   (long + clean) and check whether the raw alignments of the two families interleave.
   If they do, no absolute threshold can order them and the normalisation is doing
   real work.

2. **Does the router pick the engine that is actually better?**  For every panel we
   hold out 10% of cells MCAR, run ``cafe.CAFE().run(X, engine=...)`` under BOTH
   engines, and score held-out MAE.  The router's choice is scored against the
   engine that actually won.  This is an outcome the router never sees: the ratio is
   a property of the design matrix only, which is what keeps it causal-clean.

Writes ``data/engine_router.json`` and ``tables/router.tex``.

    .venv-bench/bin/python experimental/papers/causal/scripts/run_engine_router.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PAPER)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cafe  # noqa: E402
from cafe.discovery import choose_engine, loading_homogeneity_ratio  # noqa: E402

# The two fixture families, verbatim from src/tests/test_discovery.py.  They are
# deliberately built so the RAW alignments interleave: the homogeneous panel is short
# and noisy, the heterogeneous one long and clean.
HOMO = dict(n_ent=6, T=100, P=12, R=3, noise=1.5, rot=0.0)
HETERO = dict(n_ent=5, T=800, P=12, R=3, noise=0.1, rot=0.25)
SEEDS = [0, 1, 2, 3]
P_MISS = 0.10


def loading_panel(n_ent, T, P, R, noise, rot, seed):
    """Factor panel whose entities share (``rot=0``) or do not share their loadings."""
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(P, R))
    out = []
    for _ in range(n_ent):
        W = W0 + rot * rng.normal(size=(P, R)) if rot > 0 else W0
        F = rng.normal(size=(T, R))
        out.append(F @ W.T + noise * rng.normal(size=(T, P)))
    return np.stack(out)


def holdout_mae(X, engine, seed):
    """Held-out MAE of a CAFE pass under a pinned loading engine."""
    rng = np.random.default_rng(1000 + seed)
    mask = rng.random(X.shape) < P_MISS
    Xm = X.copy()
    Xm[mask] = np.nan
    res = cafe.CAFE().run(Xm, engine=engine)
    fill = np.asarray(res.imputed)
    err = np.abs(fill[mask] - X[mask])
    err = err[np.isfinite(err)]
    return float(np.mean(err)) if err.size else float("nan")


def main():
    rows = []
    for family, kw in (("homogeneous", HOMO), ("heterogeneous", HETERO)):
        for seed in SEEDS:
            X = loading_panel(seed=seed, **kw)
            t0 = time.perf_counter()
            parts = loading_homogeneity_ratio(X, return_parts=True)
            t_route = time.perf_counter() - t0
            engine, _ = choose_engine(X)
            mae = {e: holdout_mae(X, e, seed) for e in ("pooled", "per_entity")}
            better = min(mae, key=lambda e: mae[e])
            rows.append(dict(
                family=family, seed=seed, within=parts["within"],
                between=parts["between"], ratio=parts["ratio"], r=parts["r"],
                engine=engine, t_route_s=t_route,
                mae_pooled=mae["pooled"], mae_per_entity=mae["per_entity"],
                better=better, correct=bool(engine == better),
                mae_gap=float(mae["per_entity"] - mae["pooled"]),
            ))
            print(f"{family:<14} seed={seed}  within={parts['within']:.3f} "
                  f"between={parts['between']:.3f} ratio={parts['ratio']:.3f} "
                  f"-> {engine:<10} | MAE pooled={mae['pooled']:.4f} "
                  f"per_ent={mae['per_entity']:.4f} best={better} "
                  f"{'OK' if engine == better else 'MISS'}")

    raw_h = [r["between"] for r in rows if r["family"] == "homogeneous"]
    raw_e = [r["between"] for r in rows if r["family"] == "heterogeneous"]
    interleaved = bool(max(raw_e) > min(raw_h) and max(raw_h) > min(raw_e))
    n_correct = sum(r["correct"] for r in rows)
    out = dict(
        rows=rows,
        n_panels=len(rows), n_correct=n_correct,
        raw_between_homogeneous=[min(raw_h), max(raw_h)],
        raw_between_heterogeneous=[min(raw_e), max(raw_e)],
        raw_alignment_interleaves=interleaved,
        ratio_homogeneous=[min(r["ratio"] for r in rows if r["family"] == "homogeneous"),
                           max(r["ratio"] for r in rows if r["family"] == "homogeneous")],
        ratio_heterogeneous=[min(r["ratio"] for r in rows if r["family"] == "heterogeneous"),
                             max(r["ratio"] for r in rows if r["family"] == "heterogeneous")],
        p_miss=P_MISS, seeds=SEEDS, homo_kw=HOMO, hetero_kw=HETERO,
        cutoff=0.95,
    )
    os.makedirs(os.path.join(PAPER, "data"), exist_ok=True)
    with open(os.path.join(PAPER, "data", "engine_router.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nrouter picked the lower-MAE engine on {n_correct}/{len(rows)} panels; "
          f"raw between-alignment interleaves: {interleaved}")
    print("wrote data/engine_router.json")


if __name__ == "__main__":
    main()
