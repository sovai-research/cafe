"""Regenerate src/tests/baseline_scores.json from the current model.

Run after an intentional, validated model change that moves the benchmark numbers:
    python src/tests/gen_baseline.py
The benchmark regression test gates against this file (current minus tolerance),
so it is the single place where "the numbers got better, lock them in" is recorded.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench"))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402
from harness import run_method  # noqa: E402

import cafe  # noqa: E402


def main():
    rows = run_method("CAFE", cafe.impute)
    assert all(r["success"] for r in rows), "model failed on a dataset; fix before baselining"
    base = {r["dataset"]: round(float(r["corr"]), 4) for r in rows}
    out = {
        "_note": "Per-dataset corr for cafe.impute on bench/harness DATASETS. "
                 "Regression gate floors = these minus tolerance. "
                 "Regenerate: python src/tests/gen_baseline.py",
        "mean_corr": round(float(np.nanmean(list(base.values()))), 4),
        "min_corr": round(float(np.nanmin(list(base.values()))), 4),
        "per_dataset": base,
    }
    path = os.path.join(HERE, "baseline_scores.json")
    json.dump(out, open(path, "w"), indent=2, sort_keys=True)
    print(f"wrote {path}: mean {out['mean_corr']:.4f} min {out['min_corr']:.4f} "
          f"over {len(base)} datasets")


if __name__ == "__main__":
    main()
