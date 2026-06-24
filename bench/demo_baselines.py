"""Tiny demo: race cafe.baselines classical methods vs cafe.impute on one panel.

Prints REAL masked-cell MAE so the "special cases of CAFE" table (tab:special) is
visibly backed by runnable code. Run:

    python3 bench/demo_baselines.py
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

import cafe
from cafe import baselines as B


def _make_panel(T=200, N=12, seed=0, frac=0.2):
    """Low-rank + AR signal with MCAR holes (the regime tab:special methods target)."""
    rng = np.random.default_rng(seed)
    R = 3
    Z = np.zeros((T, R))
    for t in range(1, T):
        Z[t] = 0.9 * Z[t - 1] + rng.standard_normal(R)
    W = rng.standard_normal((R, N))
    truth = Z @ W + 0.3 * rng.standard_normal((T, N))
    X = truth.copy()
    mask = rng.random((T, N)) < frac
    X[mask] = np.nan
    return X, truth, mask


def main():
    X, truth, mask = _make_panel()
    print(f"demo panel: shape={X.shape}  missing={mask.mean():.1%}  "
          f"(low-rank+AR signal, MCAR holes)\n")

    def mae(pred):
        p = np.asarray(pred, float)
        return float(np.mean(np.abs(p[mask] - truth[mask])))

    rows = []
    # CAFE itself
    rows.append(("cafe.impute", "CAUSAL", mae(cafe.impute(X.copy()))))
    # every baseline (skip optional deps that are unavailable)
    for name in B.list_methods():
        tag = "CAUSAL" if B.is_causal(name) else "BATCH"
        try:
            out = B.impute(X.copy(), method=name)
        except ImportError as e:
            rows.append((name, tag, f"skipped ({e.args[0].split(' (')[0]})"))
            continue
        rows.append((name, tag, mae(out)))

    print(f"{'method':<20}{'kind':<9}{'MAE(masked)':>12}")
    print("-" * 41)
    for name, tag, m in rows:
        s = f"{m:.4f}" if isinstance(m, float) else m
        print(f"{name:<20}{tag:<9}{s:>12}")
    print(f"\n(data std = {np.nanstd(truth):.3f})")


if __name__ == "__main__":
    main()
