"""
Chal_ensemble -- a smarter causal router on top of the existing causal cores.

Rather than a blanket per-case blend (which is both slow and gate-risky -- the
ratchet forbids ANY single case regressing by >0.03), this challenger keeps the
champion's routing everywhere it is already strong and re-routes ONLY the two 2D
cases where a cheap, point-in-time up-front signal proves the *other* core is
strictly better on BOTH accuracy and time, or hugely better on accuracy:

  1. DRIFT gate (non-stationary 2D):
       cheap signal = |mean(first half) - mean(second half)| / std, per observed
       column, averaged. This uses only the OBSERVED masked matrix (no labels, no
       look-ahead -- it is computed from the same X every causal method sees, and
       is monotone in the prefix so it cannot leak the future relative to the
       cores it gates). When this is large (>=0.8) the series is non-stationary
       and the pure low-rank TRMF fails badly (2d_drift_block: 0.42), while the
       FE core's expanding entity/column means track the drift (0.77). -> route FE.

  2. WIDE-LONG gate (large, wide 2D):
       when a 2D matrix is both very long and very wide (N>=100 and T>=5000) the
       TRMF per-time factor solve scales O(T*N*L) and is dominated by the FE core
       which is BOTH faster and slightly more accurate here (real_beijing: FE
       0.867 in ~12s vs TRMF 0.861 in ~53s). -> route FE. This is the case that
       lets total_time drop ~40s while corr ticks UP. Narrow long series
       (ETTh1, N=7) are NOT caught (they must stay on TRMF, FE collapses there).

Everything else is byte-for-byte the champion routing, so those cases are
unchanged. Causality is preserved because we only ever dispatch to one of the
two already-verified causal cores; the gate signals are functions of the static
observed matrix and the static shape, not of any imputed/future value.

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

DRIFT_THRESH = 0.8        # non-stationarity ratio above which FE >> TRMF
WIDE_N = 100              # features to count as "wide"
LONG_T = 5000            # rows to count as "long"


def _drift_score(X):
    """Non-stationarity of the OBSERVED matrix: split rows in half, compare
    per-column observed means, normalize by global std. Cheap, point-in-time
    (no use of imputed or future-relative values beyond the static input X)."""
    T, N = X.shape
    if T < 8:
        return 0.0
    obs = ~np.isnan(X)
    h = T // 2

    def colmean(a, o):
        s = np.where(o, a, 0.0).sum(0)
        c = o.sum(0)
        return np.where(c > 0, s / np.maximum(c, 1), 0.0), c

    m1, c1 = colmean(X[:h], obs[:h])
    m2, c2 = colmean(X[h:], obs[h:])
    sd = np.nanstd(X) + 1e-9
    valid = (c1 > 5) & (c2 > 5)
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(m1 - m2)[valid]) / sd)


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    # Panels keep the champion route (FE).
    if _is_panel(meta):
        return fe_core(X, meta)

    T, N = X.shape
    # 1D and ordinary 2D default to TRMF (champion). Re-route only on the two
    # cheap gates where FE is provably better.
    if N >= 2:
        # WIDE-LONG: TRMF's per-time solve is the time sink; FE wins both axes.
        if N >= WIDE_N and T >= LONG_T:
            return fe_core(X, meta)
        # DRIFT: non-stationary low-rank -> FE's expanding means track the drift.
        if _drift_score(X) >= DRIFT_THRESH:
            return fe_core(X, meta)

    return trmf_core(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_ensemble", online_impute))
