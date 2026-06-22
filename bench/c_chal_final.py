"""
c_chal_final -- the consolidated causal imputer: c_chal_uni plus the three
research-driven specialist routes that each beat uni on a DISJOINT structural slice
with no regression elsewhere. Every route is truncation-invariant or intrinsic-
online, so the whole thing stays point-in-time (assert_causal clean).

  panel                          -> w6 panel contiguous-gap specialist
                                    (scattered panels fall through to CausalFE)
  1D (N==1)  /  wide (N>=N_GATE)  -> c_chal_uni  (seasonal / EW-cov, unchanged)
  narrow 2D, high-eff-rank+dense  -> w8 adaptive-rank low-rank+full-cov blend
  narrow 2D, otherwise            -> w9 CUSUM-reset EW-cov blend
                                    (heavy-tail defers to uni winsor; if no regime
                                     break is detected it is byte-close to uni)

Each specialist only runs on its own slice, so cost is added on-target rather than
across the whole suite.
"""
from __future__ import annotations
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from c_router import _is_panel
from c_chal_ewcov import N_GATE
import c_chal_uni
import w6_panel_blackout as _w6
import w8_rankblend as _w8
import w9_cusum_reset as _w9


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if _is_panel(meta):
        return _w6.online_impute(X, meta)          # panel specialist (else -> FE)
    T, N = X.shape
    if N == 1 or N >= N_GATE:
        return c_chal_uni.online_impute(X, meta)    # 1D seasonal / wide EW-cov

    # narrow 2D: high-eff-rank dense non-heavy -> rankblend; everything else -> CUSUM blend
    try:
        win = X[:min(_w8.GATE_WIN, T)]
        dense = float(np.isnan(win).mean()) <= _w8.ENGAGE_MAXMISS
        highrank = _w8._effrank_ratio(win) >= _w8.ENGAGE_ERANK
        heavy = _w8._median_excess_kurtosis(X) > _w8.KURT_GATE
        if dense and highrank and not heavy:
            return _w8.online_impute(X, meta)
    except Exception:
        pass
    return _w9.online_impute(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("c_chal_final", online_impute))
