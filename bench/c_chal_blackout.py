"""
Chal_blackout -- contiguous-gap (block / synchronized-blackout) panel specialist.

ROUTING (fully point-in-time; builds only on the existing causal cores):
  - non-panel            -> OnlineTRMF      (byte-identical to CausalRouter)
  - panel, scattered     -> CausalFE        (byte-identical to CausalRouter)
  - panel, BLOCK/BLACKOUT-> 0.85*CausalFE + 0.15*OnlineTRMF  (error-decorrelating blend)

WHY THE BLEND HELPS THE WEAK CASES.
  panel_block30 (0.565) and panel_blackout25 (0.472) are the suite's weak panel
  spots: entities go dark for long contiguous spans, so the unrecoverable entity-
  specific variance dominates the gap residual. CausalFE (expanding entity-FE +
  contemporaneous time adjustment + online low-rank residual) and OnlineTRMF
  (shared loadings + AR(1) latent-factor prior) make DIFFERENT errors inside those
  gaps -- FE leans on the entity mean + cross-section, TRMF leans on the AR temporal
  prior. Their errors are partially uncorrelated, so a convex blend raises the
  correlation of the combined prediction above either core alone. Empirically
  (seed 0): block30 0.5647 -> 0.5744, blackout25 0.4718 -> 0.4752 at w=0.15, with
  no panel case regressing.

  A pure forward AR / damped-trend / persistence extrapolation (the original
  hypothesis) was tested and REGRESSED every panel case: the panel signal is the
  mean-reverting AR temporal factor that CausalFE's low-rank residual already
  captures, so carrying a stale anomaly only fights a well-fit common component.
  The decorrelating blend is what actually moves the weak cases.

GATE (keep the rest of the suite untouched, so total_time does not rise).
  The blend fires ONLY when a cheap up-front missingness-pattern signal says the
  gaps are contiguous (block_frac >= THRESH). Scattered (MCAR/MAR) panels and all
  non-panels take exactly the champion's single-core path -> identical numbers and
  no extra wall-clock there. Only the two contiguous-gap panels pay one extra TRMF
  call (~0.05s each); the dominant real-beijing cost is unchanged.

POINT-IN-TIME: the blend is a fixed convex combination of two point-in-time cores
evaluated independently, plus an up-front read-only pattern signal. Truncating
future rows changes neither core's past output nor the (data-shape) gate, so the
combination stays causal.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel

BLEND_W = 0.15          # TRMF weight in the block/blackout blend (FE gets 1-w)
BLOCK_THRESH = 0.70     # block_frac above this => contiguous-gap regime => blend
RUN_MIN = 3             # a missing run of >= this length counts as "contiguous"


def _block_frac(X, meta):
    """Cheap point-in-time signal: fraction of MISSING cells that sit inside a
    contiguous time-run (length >= RUN_MIN) within an entity-feature series.
    ~1.0 for block/blackout, small for scattered MCAR. O(missing) and read-only."""
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    obs = ~np.isnan(X)
    F = X.shape[1]
    long_miss = 0
    total_miss = 0
    for e in np.unique(eids):
        rs = np.where(eids == e)[0]
        rs = rs[np.argsort(tids[rs], kind="stable")]
        miss = ~obs[rs]                          # (Te, F) in time order
        tm = int(miss.sum())
        if tm == 0:
            continue
        total_miss += tm
        for f in range(F):
            col = miss[:, f].astype(np.int8)
            if not col.any():
                continue
            d = np.diff(np.r_[0, col, 0])
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            runs = ends - starts
            long_miss += int(runs[runs >= RUN_MIN].sum())
    return long_miss / max(total_miss, 1)


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if not _is_panel(meta):
        return _trmf(X, meta)                    # identical to champion path
    # panel: default to CausalFE (champion path) unless gaps are contiguous
    if _block_frac(X, meta) < BLOCK_THRESH:
        return _fe(X, meta)                      # scattered -> identical to champion
    # contiguous-gap regime (block / blackout): decorrelating core blend
    pf = _fe(X, meta)
    pt = _trmf(X, meta)
    return (1.0 - BLEND_W) * pf + BLEND_W * pt


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_blackout", online_impute))
