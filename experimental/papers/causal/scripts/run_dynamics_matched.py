#!/usr/bin/env python
"""Re-run CAFE-CD on CausalDynamics `simple` with today's library, matched to the
baseline board.

Why this script exists.  ``bench/bench_dynamics_cache.json`` holds CAFE-CD twice:
a stale set of rows written under the cache tag ``batch`` when CAFE-CD was still
a member of ``METHODS``, and the current dedicated ``cafecd`` tag.  The two
disagree on the *support* itself, so they are two different pipelines and only
one can be current.  Re-running the identical job
(``exp_bench_dynamics._h2h_one_system``) settles it and additionally picks up the
``CausalGraph.scores()`` continuous-evidence fix, which landed after the cached
``cafecd`` run and which that cache's own ``_status.provisional`` flag warns
about.  Support metrics (F1, SHD, density) are unaffected by that fix and are
used here as the identity check.

Nothing under ``bench/`` is written; the result goes to
``experimental/papers/causal/data/dynamics_matched.json``.

    .venv-bench/bin/python experimental/papers/causal/scripts/run_dynamics_matched.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PAPER)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import exp_bench_dynamics as E                                   # noqa: E402

SETTING = "noise=0.00_confounder=False"
LABEL = f"simple/{SETTING}"
WORKERS = int(os.environ.get("DYN_WORKERS", "4"))
OUT = os.path.join(PAPER, "data", "dynamics_matched.json")


def main() -> int:
    cache = json.load(open(os.path.join(ROOT, "bench",
                                        "bench_dynamics_cache.json")))["h2h"]
    paths = E.system_paths("simple", SETTING)
    systems = [os.path.basename(p)[:-3] for p in paths]
    # only systems the baseline board actually scored, so the comparison is matched
    keep = [(s, p) for s, p in zip(systems, paths)
            if f"batch|{LABEL}|{s}" in cache
            and all(m in cache[f"batch|{LABEL}|{s}"] for m in E.METHODS)]
    print(f"{len(paths)} systems in {LABEL}; {len(keep)} scored by all "
          f"{len(E.METHODS)} baselines")

    jobs = [(f"live|{LABEL}|{s}", p, "simple", [E.CAFE_CD], E.MAX_LAG,
             E.TIMEOUT_S, None, 1.0) for s, p in keep]
    rows, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(E._h2h_one_system, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            k, v = fut.result()
            rows[k.split("|")[2]] = v
            print(f"  {i}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    # identity check against the cached `cafecd` tag: same support => same run,
    # and any AUROC/AUPRC gap is the scores() fix alone.
    ident = {"n": 0, "same_support": 0, "d_auprc": [], "d_auroc": []}
    for s, v in rows.items():
        c = cache.get(f"cafecd|{LABEL}|{s}", {}).get("cafe_cd")
        if not c:
            continue
        ident["n"] += 1
        ident["same_support"] += int(
            abs(c["shd"] - v["cafe_cd"]["shd"]) < 1e-9
            and abs(c["density_hat"] - v["cafe_cd"]["density_hat"]) < 1e-9
            and abs(c["f1"] - v["cafe_cd"]["f1"]) < 1e-9)
        ident["d_auprc"].append(v["cafe_cd"]["auprc"] - c["auprc"])
        ident["d_auroc"].append(v["cafe_cd"]["auroc"] - c["auroc"])
    for k in ("d_auprc", "d_auroc"):
        ident[k] = float(np.mean(ident[k])) if ident[k] else float("nan")

    board = {}
    for m in list(E.METHODS):
        board[m] = {
            "auprc": float(np.mean([cache[f"batch|{LABEL}|{s}"][m]["auprc"]
                                    for s, _ in keep])),
            "auroc": float(np.mean([cache[f"batch|{LABEL}|{s}"][m]["auroc"]
                                    for s, _ in keep])),
            "f1": float(np.mean([cache[f"batch|{LABEL}|{s}"][m]["f1"]
                                 for s, _ in keep])),
            "source": "bench/bench_dynamics_cache.json (tag batch)"}
    board["cafe_cd"] = {
        "auprc": float(np.mean([rows[s]["cafe_cd"]["auprc"] for s, _ in keep])),
        "auroc": float(np.mean([rows[s]["cafe_cd"]["auroc"] for s, _ in keep])),
        "f1": float(np.mean([rows[s]["cafe_cd"]["f1"] for s, _ in keep])),
        "source": "live re-run, this script"}
    # the stale in-batch CAFE-CD rows, kept for the record
    stale = [cache[f"batch|{LABEL}|{s}"]["cafe_cd"] for s, _ in keep
             if "cafe_cd" in cache[f"batch|{LABEL}|{s}"]]
    cached = [cache[f"cafecd|{LABEL}|{s}"]["cafe_cd"] for s, _ in keep
              if f"cafecd|{LABEL}|{s}" in cache]

    out = {
        "setting": LABEL, "n_systems": len(keep),
        "systems": [s for s, _ in keep],
        "max_lag": E.MAX_LAG, "board": board,
        "cafe_cd_variants": {
            "live (this script, post scores() fix)": board["cafe_cd"],
            "cached tag cafecd (pre scores() fix)": {
                "auprc": float(np.mean([c["auprc"] for c in cached])),
                "auroc": float(np.mean([c["auroc"] for c in cached])),
                "f1": float(np.mean([c["f1"] for c in cached])),
                "n": len(cached)} if cached else None,
            "cached tag batch (stale pipeline)": {
                "auprc": float(np.mean([c["auprc"] for c in stale])),
                "auroc": float(np.mean([c["auroc"] for c in stale])),
                "f1": float(np.mean([c["f1"] for c in stale])),
                "n": len(stale)} if stale else None},
        "identity_check_vs_cached_cafecd": ident,
        "rank_auprc": sorted(board, key=lambda m: -board[m]["auprc"]),
        "wall_s": time.time() - t0,
        "note": ("CAFE-CD re-run live with the current src/cafe; the 12 baselines "
                 "are read unchanged from the batch board. Matched systems only."),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}")
    for i, m in enumerate(out["rank_auprc"], 1):
        print("  %2d %-18s auprc %.4f auroc %.4f f1 %.4f"
              % (i, m, board[m]["auprc"], board[m]["auroc"], board[m]["f1"]))
    print("identity check:", json.dumps(ident))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
