"""
Reference benchmark: EXISTING TIMARA / MISSNET (pure-Python fallback, no numba).

Method name: MISSNET(existing)

This is the high-fidelity reference point: accuracy ceiling + the (slow) timing
that the redesign is trying to beat.

Strategy:
  - 2d_large_* datasets are SKIPPED (too slow without numba) -> success=false.
  - Every other dataset runs the real missnet_impute() with a 120s per-call
    wall-clock guard (signal.alarm). If a single fit exceeds the guard, that
    dataset is marked skipped (success=false).
"""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import sys
import time
import signal
import warnings

import numpy as np

sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA")

from harness import DATASETS, evaluate, summarize  # noqa: E402

TIME_GUARD_S = 120
SKIP_DATASETS = {"2d_large_mcar30", "2d_large_block30"}


class _Timeout(Exception):
    pass


def make_imputer():
    """Return an impute(X, meta) closure with a hard per-call wall-clock guard.

    Uses signal.alarm; the pure-Python EM fallback returns control to the
    interpreter frequently enough for the SIGALRM handler to fire.
    """
    from missnet_imputer import missnet_impute

    def _handler(signum, frame):
        raise _Timeout(f"skipped: exceeded {TIME_GUARD_S}s wall-clock guard")

    def impute(X, meta):
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(TIME_GUARD_S)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = missnet_impute(X, verbose=False, auto_tune=True)
            return np.asarray(out, dtype=float)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    return impute


def main():
    impute = make_imputer()
    rows = []
    for name, dataset in DATASETS.items():
        clean, meta, mech, rate = dataset
        if name in SKIP_DATASETS:
            rows.append(dict(
                method="MISSNET(existing)", dataset=name, mechanism=mech,
                rate=rate, shape=f"{clean.shape[0]}x{clean.shape[1]}",
                mae=float("nan"), rmse=float("nan"), mre=float("nan"),
                corr=float("nan"), time_s=float("nan"), success=False,
                error="skipped: too slow without numba"))
            print(f"[skip] {name}: too slow without numba")
            continue

        print(f"[run ] {name}  shape={clean.shape}  {mech}@{rate} ...", flush=True)
        t0 = time.perf_counter()
        row = evaluate("MISSNET(existing)", impute, name, dataset)
        dt = time.perf_counter() - t0
        ok = row["success"]
        print(f"       -> ok={ok} time_s={row['time_s']:.2f} "
              f"(wall {dt:.1f}s) mae={row['mae']:.4f} err={row['error'][:60]}",
              flush=True)
        rows.append(row)

    print()
    summarize(rows)
    return rows


if __name__ == "__main__":
    main()
