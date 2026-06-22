"""
CAUSAL FLOOR baselines for the TIMARA redesign.

Three fully point-in-time imputers, each run through `run_causal` (so each is its
own verified method) and combined with the baseline name embedded in the 'dataset'
field as "<base>:<dataset>". Top-level method label = "CausalBaselines".

GOVERNING CONSTRAINT: an imputed value for (entity e, time tau, feature i) uses ONLY
data at time <= tau -- past values of that series AND the contemporaneous cross-section
at tau. NEVER any value at time s > tau. We iterate by TIME (grouping all entities that
share a time_id), so panels (entity-major stacked rows) never leak a later entity's
future into an earlier entity's past.

Baselines
---------
(a) LOCF           : per-(entity,feature) forward-fill of last observed value, by time.
                     Leading gaps (no past observation yet for that series) fall back to
                     the EXPANDING cross-sectional mean of that feature at time tau
                     (mean over all entities/features-instances observed at times <= tau).
(b) CausalEWMA     : exponentially-weighted mean of each series' past observed values
                     (recursive, by time). Leading gaps -> expanding cross-sectional mean.
(c) CausalXSecMean : for a missing (e,tau,i) use the mean over entities of feature i AT
                     TIME tau (the contemporaneous cross-section, fully causal). Fall back
                     to the series' last observed value, then to the expanding x-sec mean.

All three pass the causality verifier on all 7 datasets, panels included.

Time axis: for 2D datasets time_id of row r is r (each row a distinct time, one
"entity"). For panels time_id/entity_id come from meta.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
from causal import run_causal, summarize_causal  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _time_groups(meta, T):
    """Return (entity_ids, time_ids, ordered_times, rows_per_time).

    rows_per_time[k] is the array of row indices whose time_id == ordered_times[k].
    For 2D data (no meta) every row is its own time and entity 0.
    """
    if meta and "time_ids" in meta:
        tid = np.asarray(meta["time_ids"])
        eid = np.asarray(meta["entity_ids"]) if "entity_ids" in meta \
            else np.zeros(T, dtype=int)
    else:
        tid = np.arange(T)
        eid = np.zeros(T, dtype=int)
    order = np.argsort(tid, kind="stable")
    uniq = np.unique(tid)
    rows_per_time = []
    # build per-time row index lists in ascending time order
    sorted_tid = tid[order]
    # boundaries of each unique time within the sorted order
    pos = 0
    for ut in uniq:
        cnt = int(np.count_nonzero(sorted_tid == ut))
        rows_per_time.append(order[pos:pos + cnt])
        pos += cnt
    return eid, tid, uniq, rows_per_time


# --------------------------------------------------------------------------- #
# (a) LOCF -- last observation carried forward, by time, per (entity, feature)
# --------------------------------------------------------------------------- #
def locf_impute(X, meta):
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    eid, tid, times, rows_per_time = _time_groups(meta, T)
    n_ent = int(eid.max()) + 1 if eid.size else 1
    out = X.copy()

    # last observed value per (entity, feature); NaN until first observation
    last_val = np.full((n_ent, N), np.nan)
    # expanding cross-sectional accumulator per feature (over all observed cells
    # at times <= current), used only for leading gaps with no past series value
    xs_sum = np.zeros(N)
    xs_cnt = np.zeros(N)

    for rows in rows_per_time:
        # current expanding cross-sectional mean (from strictly past times) for fallback
        xs_mean = np.where(xs_cnt > 0, xs_sum / np.maximum(xs_cnt, 1), 0.0)
        for r in rows:
            e = eid[r]
            row = X[r]
            miss = np.isnan(row)
            if miss.any():
                lv = last_val[e]
                fill = np.where(np.isfinite(lv), lv, xs_mean)
                out[r, miss] = fill[miss]
        # AFTER filling this whole time slice, fold its observations into state
        for r in rows:
            e = eid[r]
            row = X[r]
            obs = ~np.isnan(row)
            if obs.any():
                last_val[e, obs] = row[obs]
                xs_sum[obs] += row[obs]
                xs_cnt[obs] += 1
    # any residual NaN (feature never observed anywhere up to its time) -> 0
    if np.isnan(out).any():
        out = np.where(np.isnan(out), 0.0, out)
    return out


# --------------------------------------------------------------------------- #
# (b) CausalEWMA -- EW mean of each series' past observed values, by time
# --------------------------------------------------------------------------- #
def make_ewma(halflife=5.0):
    alpha = 1.0 - 0.5 ** (1.0 / halflife)  # smoothing factor

    def ewma_impute(X, meta):
        X = np.asarray(X, dtype=float)
        T, N = X.shape
        eid, tid, times, rows_per_time = _time_groups(meta, T)
        n_ent = int(eid.max()) + 1 if eid.size else 1
        out = X.copy()

        # recursive EWMA state per (entity, feature): numerator / denominator form
        # ewm_t = (1-a)*ewm_{t-1} + a*x_t implemented via weighted sums for correctness
        ew_num = np.zeros((n_ent, N))
        ew_den = np.zeros((n_ent, N))       # 0 => no past obs yet
        xs_sum = np.zeros(N)
        xs_cnt = np.zeros(N)

        for rows in rows_per_time:
            xs_mean = np.where(xs_cnt > 0, xs_sum / np.maximum(xs_cnt, 1), 0.0)
            for r in rows:
                e = eid[r]
                row = X[r]
                miss = np.isnan(row)
                if miss.any():
                    have = ew_den[e] > 0
                    ewv = np.where(have, ew_num[e] / np.maximum(ew_den[e], 1e-300),
                                   xs_mean)
                    out[r, miss] = ewv[miss]
            # fold observations of this time slice into the recursive state
            for r in rows:
                e = eid[r]
                row = X[r]
                obs = ~np.isnan(row)
                if obs.any():
                    # decay then add: num = (1-a)*num + a*x ; den = (1-a)*den + a
                    ew_num[e, obs] = (1 - alpha) * ew_num[e, obs] + alpha * row[obs]
                    ew_den[e, obs] = (1 - alpha) * ew_den[e, obs] + alpha
                    xs_sum[obs] += row[obs]
                    xs_cnt[obs] += 1
        if np.isnan(out).any():
            out = np.where(np.isnan(out), 0.0, out)
        return out

    return ewma_impute


# --------------------------------------------------------------------------- #
# (c) CausalXSecMean -- contemporaneous cross-sectional mean at tau, with fallback
# --------------------------------------------------------------------------- #
def xsecmean_impute(X, meta):
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    eid, tid, times, rows_per_time = _time_groups(meta, T)
    n_ent = int(eid.max()) + 1 if eid.size else 1
    out = X.copy()

    last_val = np.full((n_ent, N), np.nan)   # series' last observed value
    xs_sum = np.zeros(N)                     # expanding x-sec accumulator (past times)
    xs_cnt = np.zeros(N)

    for rows in rows_per_time:
        expanding_mean = np.where(xs_cnt > 0, xs_sum / np.maximum(xs_cnt, 1), 0.0)
        # CONTEMPORANEOUS cross-section: mean over entities of each feature AT this tau,
        # computed from the OBSERVED cells in this time slice only (no future, no
        # imputed values). Fully causal: only uses data at time == tau.
        block = X[rows]                       # (n_rows_at_tau, N)
        obs_blk = ~np.isnan(block)
        col_sum = np.where(obs_blk, block, 0.0).sum(axis=0)
        col_cnt = obs_blk.sum(axis=0)
        contemp_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), np.nan)

        for r in rows:
            e = eid[r]
            row = X[r]
            miss = np.isnan(row)
            if miss.any():
                lv = last_val[e]
                # primary: contemporaneous x-sec mean; fallback: series last value;
                # final fallback: expanding x-sec mean.
                fill = np.where(np.isfinite(contemp_mean), contemp_mean, np.nan)
                fill = np.where(np.isfinite(fill), fill, lv)
                fill = np.where(np.isfinite(fill), fill, expanding_mean)
                out[r, miss] = fill[miss]
        for r in rows:
            e = eid[r]
            row = X[r]
            obs = ~np.isnan(row)
            if obs.any():
                last_val[e, obs] = row[obs]
                xs_sum[obs] += row[obs]
                xs_cnt[obs] += 1
    if np.isnan(out).any():
        out = np.where(np.isnan(out), 0.0, out)
    return out


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
BASELINES = {
    "LOCF": locf_impute,
    "CausalEWMA": make_ewma(halflife=5.0),
    "CausalXSecMean": xsecmean_impute,
}


def run_all():
    all_rows = []
    for base, fn in BASELINES.items():
        rows = run_causal("CausalBaselines", fn, verify=True)
        for r in rows:
            r = dict(r)
            r["dataset"] = f"{base}:{r['dataset']}"
            all_rows.append(r)
    return all_rows


if __name__ == "__main__":
    rows = run_all()
    summarize_causal(rows)
    n_causal = sum(1 for r in rows if r["causal"])
    print(f"\n{len(rows)} rows, causal on {n_causal}/{len(rows)}")
