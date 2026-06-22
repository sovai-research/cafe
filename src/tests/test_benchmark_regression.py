"""Benchmark regression gate.

"Beats published benchmarks" and "fast" are load-bearing claims, so every change
is measured against a recorded baseline (src/tests/baseline_scores.json) and fails
if accuracy regresses. Speed is asserted against a generous ceiling so genuine
algorithmic slowdowns surface without flaking on a busy CI box.

Marked `benchmark` (and `slow`): runs the full synthetic suite (~2-3s). Deselect
with `-m "not benchmark"`. Regenerate the baseline after a validated improvement
with `python src/tests/gen_baseline.py`.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                 # src/  -> cafe
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench"))  # bench/ -> harness

harness = pytest.importorskip(
    "harness",
    reason="bench/ harness not importable",
)  # noqa: E402  -- importorskip must run before importing the model harness

import numpy as np  # noqa: E402

import cafe  # noqa: E402

# tolerances: how far a single dataset, or the aggregates, may slip below baseline.
PER_DATASET_DROP = 0.03   # mirrors the arena --maxdrop "no case regresses" rule
AGGREGATE_DROP = 0.02
TIME_CEILING_S = 15.0     # generous; only catches order-of-magnitude slowdowns


@pytest.fixture(scope="module")
def baseline():
    with open(os.path.join(HERE, "baseline_scores.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def results():
    return harness.run_method("CAFE", cafe.impute)


@pytest.mark.benchmark
@pytest.mark.slow
def test_all_datasets_succeed(results):
    failed = [(r["dataset"], r["error"]) for r in results if not r["success"]]
    assert not failed, f"model errored on datasets: {failed}"


@pytest.mark.benchmark
@pytest.mark.slow
def test_no_per_dataset_regression(results, baseline):
    base = baseline["per_dataset"]
    got = {r["dataset"]: float(r["corr"]) for r in results}
    regressions = {
        d: (base[d], got[d])
        for d in base
        if d in got and got[d] < base[d] - PER_DATASET_DROP
    }
    assert not regressions, (
        "accuracy regressed (baseline, now): "
        + ", ".join(f"{d} {b:.4f}->{n:.4f}" for d, (b, n) in regressions.items())
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_aggregate_accuracy(results, baseline):
    corrs = [float(r["corr"]) for r in results]
    mean_c, min_c = float(np.nanmean(corrs)), float(np.nanmin(corrs))
    assert mean_c >= baseline["mean_corr"] - AGGREGATE_DROP, (
        f"mean corr {mean_c:.4f} below baseline {baseline['mean_corr']:.4f}"
    )
    assert min_c >= baseline["min_corr"] - AGGREGATE_DROP, (
        f"min corr {min_c:.4f} below baseline {baseline['min_corr']:.4f}"
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_speed_ceiling(results):
    total = sum(r["time_s"] for r in results)
    assert total < TIME_CEILING_S, f"suite took {total:.2f}s (ceiling {TIME_CEILING_S}s)"
