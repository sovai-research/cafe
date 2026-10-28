#!/usr/bin/env python
"""Measure var-sortability directly on CausalRivers, instead of inferring it.

The rescaling ladder in ``bench/bench_rivers_cache.json`` shows that a VAR scoring
``|beta|`` on unrescaled discharge gets AUROC 0.849, and that rescaling the series in
either of the two places a harness normally rescales drops it below chance.  That is
strong circumstantial evidence for a scale prior.  This script measures the prior
itself.

``varsortability`` (Reisach, Seiler & Weichwald, NeurIPS 2021) is the fraction of
directed paths in the true graph along which marginal variance increases.  0.5 means
the node ordering carries no information about causal direction; 1.0 means "sort the
nodes by variance" recovers the causal order for free, with no learning at all.
``R^2-sortability`` (Reisach et al., NeurIPS 2023) is its standardisation-proof
cousin: z-scoring the data destroys varsortability but not the share of the order that
leaks through predictability.

Both are computed here on the exact published cell -- flood graph, Random-5, 6 h,
``normalize=False`` -- over every subgraph, with no model fitted at any point.

A z-scored control is deliberately NOT reported: standardising makes every marginal
variance exactly 1, so varsortability degenerates into a comparison of floating-point
noise and any value it returns is an artefact, not a measurement.  ``R^2``-sortability
is the standardisation-proof diagnostic and is reported instead.

Writes ``data/varsortability.json``.

    .venv-bench/bin/python experimental/papers/causal/scripts/run_varsortability.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PAPER)))
sys.path.insert(0, os.path.join(ROOT, "bench"))

from datasets_causal import iter_causal_rivers  # noqa: E402
from sortability import r2_sortability, var_sortability  # noqa: E402

# the published VAR cell, verbatim from docs/causal_baselines.md section 6a
CELL = dict(region="flood_full", n_vars=5, strategy="random", resolution="6h",
            normalize=False, interpolate=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    vs, rs, n_edges = [], [], []
    n_skip = 0
    for X, labels, meta in iter_causal_rivers(limit=args.limit, **CELL):
        # CausalRivers labels are [effect, cause]; sortability wants [cause, effect]
        skel = np.asarray(labels, float).T > 0
        if skel.sum() == 0:
            n_skip += 1
            continue
        A = np.asarray(X, float)
        ok = np.isfinite(A).all(axis=1)
        A = A[ok]
        if A.shape[0] < 32 or not np.all(A.std(axis=0) > 0):
            n_skip += 1
            continue
        vs.append(var_sortability(A, skel))
        rs.append(r2_sortability(A, skel))
        n_edges.append(int(skel.sum()))

    def summ(x):
        x = np.asarray([v for v in x if np.isfinite(v)], float)
        return dict(mean=float(x.mean()), sd=float(x.std()), n=int(x.size),
                    median=float(np.median(x)))

    out = dict(cell=CELL, n_samples=len(vs), n_skipped=n_skip,
               mean_true_edges=float(np.mean(n_edges)),
               var_sortability=summ(vs), r2_sortability=summ(rs),
               wall_s=time.perf_counter() - t0, limit=args.limit)
    os.makedirs(os.path.join(PAPER, "data"), exist_ok=True)
    name = ("varsortability.json" if args.limit is None
            else f"varsortability_limit{args.limit}.json")
    with open(os.path.join(PAPER, "data", name), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"n = {len(vs)} subgraphs ({n_skip} skipped), {out['wall_s']:.1f}s")
    print(f"  var-sortability     {out['var_sortability']['mean']:.4f} "
          f"(sd {out['var_sortability']['sd']:.4f})")
    print(f"  R^2-sortability     {out['r2_sortability']['mean']:.4f} "
          f"(sd {out['r2_sortability']['sd']:.4f})")
    print(f"wrote data/{name}")


if __name__ == "__main__":
    main()
