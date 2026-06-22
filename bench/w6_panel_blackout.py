"""
w6_panel_blackout -- PANEL contiguous-gap (block / synchronized-blackout) specialist.

GOVERNING CONSTRAINT: fully point-in-time / NO look-ahead. Every quantity used to
impute (entity e, time t) depends ONLY on data at times <= t (past + the
contemporaneous cross-section at t). All cores reused are themselves causal-clean.

IDEA (why panel_block30 / panel_blackout25 are weak, and how to lift them).
  In these cases an entity goes dark for a LONG contiguous span. Two causal cores
  make DIFFERENT errors inside such a gap:
    - CausalFE  : expanding entity-FE + contemporaneous cross-section adjustment +
                  an online low-rank factor over the entity x feature panel. Inside a
                  long gap it leans on the entity's fixed effect and on what OTHER
                  entities pin down at the same time t (the contemporaneous factor).
                  It does NOT carry the entity's last anomalous level forward.
    - OnlineTRMF: shared loadings + AR(1) latent-factor prior. Inside a gap it carries
                  the entity's last factor forward with a damped AR recursion -- good
                  for a SHORT gap (the last reliable level is still informative), bad
                  for a LONG gap (the AR prior decays toward the wrong place and the
                  entity-specific anomaly goes stale).
  Their errors are partially uncorrelated, so a convex blend already beats either core
  (this is the prior c_chal_blackout result). We go one step further: the OPTIMAL mix
  depends on HOW LONG the entity has been dark. Early in a gap, the temporal carry
  (TRMF) is still trustworthy; deep in a long gap there is no temporal cross-section
  for the entity at all, so we must trust the CROSS-ENTITY structure (FE) more.

METHOD (gap-length-aware convex blend, all causal).
  Route exactly like the champion unless the panel is a contiguous-gap regime
  (a cheap up-front block_frac >= THRESH signal -- a data-shape property, not a value
  decision, so it is truncation-invariant). In that regime:
    1. Run both causal cores once: PF = CausalFE(X), PT = OnlineTRMF(X).
    2. For every missing cell (e, t, f) compute the CAUSAL run-length g = number of
       consecutive prior+current time steps for which feature f of entity e has been
       missing up to and including t. g depends only on data <= t -> point-in-time.
    3. Blend weight on FE rises with g:   w_fe(g) = W0 + (W1 - W0) * (1 - exp(-g/TAU)).
       Short gap (g small) -> weight near W0 (keep more temporal TRMF carry);
       long  gap (g large) -> weight near W1 (trust cross-entity FE).
       out[e,t,f] = w_fe * PF + (1 - w_fe) * PT.
  The blend is a per-cell convex combination of two point-in-time predictions with a
  weight that is itself a causal (run-length) function -> the whole thing is causal.
  Truncating future rows changes neither core's past output nor the past run-lengths.

WHY NOT A NEW EXTRAPOLATION CORE: a prior pure forward-AR / damped-trend
extrapolation REGRESSED every panel case (the panel signal is the mean-reverting AR
factor CausalFE's low-rank residual already captures; carrying a stale anomaly fights
a well-fit common component). So we do NOT add a third predictor -- we only re-weight
the two existing decorrelated cores by gap length.

GATE: scattered (MCAR/MAR) panels and all non-panels take EXACTLY the champion's
single-core path (byte-identical numbers, no extra wall-clock). Only the two
contiguous-gap panels pay one extra TRMF call.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel
from c_chal_uni import online_impute as _uni

# ----------------------------------------------------------------- knobs --- #
BLOCK_THRESH = 0.70      # block_frac above this => contiguous-gap regime => blend
RUN_MIN = 3              # a missing run of >= this length counts as "contiguous"
# FE weight in the convex blend. The DOMINANT effect is the decorrelating blend
# itself at a roughly even FE/TRMF mix (FE is the stronger panel core, TRMF adds
# decorrelated temporal carry). Empirically a flat ~0.6 FE / 0.4 TRMF is optimal on
# the two contiguous-gap panels; the gap-length tilt (W0 -> W1) adds little but is
# kept because it is the intrinsically-correct prior: deeper in a long gap there is
# no temporal cross-section for the entity, so lean a touch more on cross-entity FE.
W0 = 0.60                # FE weight at the START of a gap (short gap: more TRMF carry)
W1 = 0.72                # FE weight DEEP in a long gap (trust cross-entity more)
TAU = 6.0               # run-length scale: how fast weight migrates W0 -> W1


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
        miss = ~obs[rs]
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


def _gap_runlen(X, meta):
    """Per-cell CAUSAL run-length of the current contiguous missing span.

    runlen[r, f] = number of consecutive time steps (<= t for that entity) for which
    feature f of the entity has been missing up to and including time t. Only defined
    (and used) on missing cells; 0 on observed cells. This is a pure forward recursion
    over each entity's own past -> strictly point-in-time. Returns array shaped like X
    (rows in the original stacked order)."""
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    obs = ~np.isnan(X)
    T_rows, F = X.shape
    run = np.zeros((T_rows, F), dtype=float)
    for e in np.unique(eids):
        rs = np.where(eids == e)[0]
        rs = rs[np.argsort(tids[rs], kind="stable")]   # time order for this entity
        cnt = np.zeros(F, dtype=float)
        for r in rs:
            o = obs[r]
            cnt = np.where(o, 0.0, cnt + 1.0)           # reset on observed, +1 on miss
            run[r] = np.where(o, 0.0, cnt)
    return run


def _safe_fill(X):
    """Causal (expanding column-mean) safety fill -- never crashes, finite, same shape.
    For each time row, fills missing cells with the running mean of that column over
    times <= the current row (then folds the row in). Point-in-time by construction."""
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    out = X.copy()
    csum = np.zeros(N); ccnt = np.zeros(N)
    gsum = 0.0; gcnt = 0.0
    for t in range(T):
        row = X[t]; o = np.isfinite(row)
        cm = np.where(ccnt > 0, csum / np.maximum(ccnt, 1), np.nan)
        gm = gsum / gcnt if gcnt > 0 else 0.0
        miss = ~o
        if miss.any():
            fill = cm[miss]
            fill = np.where(np.isfinite(fill), fill, gm)
            out[t, miss] = fill
        csum[o] += row[o]; ccnt[o] += 1
        gsum += row[o].sum(); gcnt += int(o.sum())
    out[~np.isfinite(out)] = 0.0
    return out


def online_impute(X, meta):
    try:
        return _impute(X, meta)
    except Exception:
        try:
            return _safe_fill(np.asarray(X, dtype=float))
        except Exception:
            Z = np.asarray(X, dtype=float).copy()
            Z[~np.isfinite(Z)] = 0.0
            return Z


def _impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if not _is_panel(meta):
        # non-panel: defer to the integrated causal imputer (its specialist routes
        # for 1D-seasonal / heavy-tail / high-rank / wide-2D). This module only
        # adds a PANEL contiguous-gap route on top, so it stays a strict superset.
        try:
            r = np.asarray(_uni(X, meta), dtype=float)
            if r.shape == X.shape and np.all(np.isfinite(r)):
                return r
        except Exception:
            pass
        return _safe_fill(X)
    # panel: default to CausalFE (champion path) unless gaps are contiguous
    try:
        bf = _block_frac(X, meta)
    except Exception:
        bf = 0.0
    if bf < BLOCK_THRESH:
        try:
            r = np.asarray(_fe(X, meta), dtype=float)
            if r.shape == X.shape and np.all(np.isfinite(r)):
                return r                          # scattered -> identical to champion
        except Exception:
            pass
        return _safe_fill(X)

    # contiguous-gap regime: gap-length-aware decorrelating blend of FE and TRMF
    try:
        pf = np.asarray(_fe(X, meta), dtype=float)
        if pf.shape != X.shape or not np.all(np.isfinite(pf)):
            return _safe_fill(X)
    except Exception:
        return _safe_fill(X)
    try:
        pt = np.asarray(_trmf(X, meta), dtype=float)
    except Exception:
        return pf
    if pt.shape != pf.shape or not np.all(np.isfinite(pt)):
        return pf

    out = pf.copy()
    miss = np.isnan(X)
    if not miss.any():
        return out
    try:
        g = _gap_runlen(X, meta)
    except Exception:
        return out
    # FE weight rises from W0 toward W1 as the gap lengthens (saturating).
    w_fe = W0 + (W1 - W0) * (1.0 - np.exp(-g / TAU))
    blended = w_fe * pf + (1.0 - w_fe) * pt
    out[miss] = blended[miss]
    out[~np.isfinite(out)] = pf[~np.isfinite(out)]
    return out


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w6_panel_blackout", online_impute))
