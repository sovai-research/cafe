"""
Chal_ensemble -- a smarter causal router on top of the existing causal cores.

Rather than a blanket per-case blend (which is both slow and gate-risky -- the
ratchet forbids ANY single case regressing by >0.03), this challenger keeps the
champion's routing everywhere it is already strong and re-routes ONLY where a
*shape-based*, strictly point-in-time signal proves the other core is better on
BOTH accuracy and time:

  WIDE-LONG gate (large, wide 2D):
     when a 2D matrix is both very long and very wide (N>=WIDE_N and T>=LONG_T)
     the TRMF per-time factor solve scales O(T*N*L) and is dominated by the FE
     core, which is BOTH faster and slightly more accurate here
     (real_beijing 17117x132: FE 0.867 in ~12s vs TRMF 0.861 in ~53s).
     -> route FE. This single re-route drops total_time ~40s while corr ticks UP.
     Narrow long series (ETTh1, N=7) are NOT caught (FE collapses to 0.68 there),
     and shorter/narrower 2D (temp 5000x50, large 2000x60, drift-vldb 1000x100)
     stay on TRMF, where it is stronger.

WHY NO DRIFT GATE: an obvious second win is routing the non-stationary
2d_drift_block to FE (0.77 vs TRMF 0.42, +0.34). But any data-driven drift
detector (first-half vs second-half column means) is by construction a
future-vs-past statistic: on a time-truncated prefix the "halves" differ, so the
routing decision -- and hence the imputations at tau -- changes when future rows
are removed. That is LOOK-AHEAD and FAILS the causal verifier
(assert_causal: LOOK-AHEAD@tau, max_drift~0.27). An early-fixed-window detector
is point-in-time but cannot separate drift_block from real_temp/chlorine, so it
would falsely re-route temp (0.989->0.942, a fatal -0.05 regression). The drift
win is therefore not reachable from a causal *router* signal and is dropped; it
would require a core that detects drift online from history <= tau.

The WIDE-LONG gate is shape-only (N, T), independent of cell values and of the
suffix, so on any prefix with T<LONG_T it simply stays on TRMF and the imputation
of any earlier tau is identical -- it passes the causal verifier. We only ever
dispatch to one of the two already-verified causal cores.

Libs: numpy/scipy only (via the cores). Thread caps set at import.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from c_online_trmf import online_impute as trmf_core
from c_fe_lowrank import online_impute as fe_core
from c_router import _is_panel

WIDE_N = 100              # features to count as "wide"
LONG_T = 5000             # rows to count as "long"


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    # Panels keep the champion route (FE).
    if _is_panel(meta):
        return fe_core(X, meta)

    T, N = X.shape
    # WIDE-LONG (shape-only, point-in-time): TRMF's per-time solve is the time
    # sink on very long+wide 2D; FE wins both axes there (e.g. beijing).
    if N >= WIDE_N and T >= LONG_T:
        return fe_core(X, meta)

    # 1D and all other 2D default to TRMF (champion route).
    return trmf_core(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_ensemble", online_impute))
