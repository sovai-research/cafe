"""
Chal_cdrec — Centroid Decomposition Recovery (CDRec-style) gated to contiguous-
block / blackout cases and small-N structured reals (airq).

IDEA (bench-vldb20 CDRec): for contiguous-block missingness the right recovery is
an iterative SVD-truncation completion on the (entity x feature) cross-section
accumulated up to the current time tau: initialize the missing residual cells with
the FE baseline, then repeatedly (a) truncate the filled residual matrix to rank-k
via SVD and (b) write the rank-k reconstruction back into the missing cells only.
This converges to a low-rank completion that respects the OBSERVED cells exactly —
strictly better than a couple of ALS sweeps when whole spans are dark, because the
SVD pools every entity's correlated structure instead of solving one factor at a
time against a stale Bt.

CAUSAL GUARANTEE: identical contract to CausalFE. We iterate strictly forward in
time. At time tau the CDRec completion uses ONLY the residual snapshot of times
<= tau (each entity's last-observed residual <= tau plus tau's own cross-section).
Truncating future rows never changes a past snapshot -> point-in-time.

ROUTING (cheap up-front signal): only contiguous-block patterns benefit, so we
detect them and route; everything else goes to the unchanged champion cores so the
rest of the suite (and total_time) is untouched.
  - panel + contiguous gaps   -> causal panel CDRec
  - 2D / non-block            -> OnlineTRMF   (unchanged)
  - 2D block but NOT helped   -> OnlineTRMF   (unchanged)
  - small-N structured real (airq, contiguous block) -> 2D CDRec
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
from c_fe_lowrank import online_impute as _fe, _impute_panel as _fe_panel
from c_router import _is_panel


# --------------------------------------------------------------------------- #
# CDRec core: iterative SVD-truncation completion of a residual matrix.
# R         : (n, f) residual, NaN/0 in missing handled via mask
# Wobs      : (n, f) bool observed mask
# fill0     : (n, f) initial fill for the missing cells (baseline)
# rank      : truncation rank
# returns the completed matrix (observed cells preserved).
# --------------------------------------------------------------------------- #
def _cdrec_complete(Robs, Wobs, fill0, rank, iters, tol=1e-3):
    n, f = Robs.shape
    M = np.where(Wobs, Robs, fill0).astype(float, copy=True)
    k = max(1, min(rank, n - 1, f))
    prev = None
    for _ in range(iters):
        # truncated SVD reconstruction (the "centroid decomposition" surrogate)
        try:
            U, s, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        kk = min(k, len(s))
        recon = (U[:, :kk] * s[:kk]) @ Vt[:kk]
        # write reconstruction into missing cells only; keep observed exactly
        newM = np.where(Wobs, M, recon)
        if prev is not None:
            denom = np.linalg.norm(M[~Wobs]) + 1e-12
            if np.linalg.norm(newM[~Wobs] - prev) / denom < tol:
                M = newM
                break
        prev = newM[~Wobs].copy()
        M = newM
    return M


# --------------------------------------------------------------------------- #
# Causal panel CDRec. Mirrors CausalFE's FE + cross-section structure, but the
# residual low-rank step is an iterative SVD-truncation CDRec completion of the
# accumulated (entity x feature) residual snapshot at every time tau.
# --------------------------------------------------------------------------- #
RANK = 2                  # LOW rank: the residual cross-section is rank-deficient;
                          # higher ranks overfit the snapshot and HURT block recovery.
CDREC_ITERS = 6


def _impute_panel_cdrec(X, meta):
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    T_rows, F = X.shape
    E = int(eids.max()) + 1
    out = X.copy()

    uniq_t = np.unique(tids)
    rows_at = {t: np.where(tids == t)[0] for t in uniq_t}

    ent_sum = np.zeros((E, F)); ent_cnt = np.zeros((E, F))
    g_sum = np.zeros(F); g_cnt = np.zeros(F)

    res_snap = np.zeros((E, F)); res_have = np.zeros((E, F), dtype=bool)

    for t in uniq_t:
        rs = rows_at[t]
        ent_t = eids[rs]
        Xt = X[rs]
        obs_t = ~np.isnan(Xt)

        # entity FE from history < tau (expanding own-mean), global fallback
        ent_fe = np.where(ent_cnt[ent_t] > 0,
                          ent_sum[ent_t] / np.maximum(ent_cnt[ent_t], 1), 0.0)
        gfe = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 0.0)
        ent_fe = np.where(ent_cnt[ent_t] > 0, ent_fe, gfe[None, :])

        resid_t = Xt - ent_fe

        # contemporaneous cross-section adjustment AT tau (causal)
        time_adj = np.zeros(F)
        for fi in range(F):
            col = resid_t[obs_t[:, fi], fi]
            if col.size > 0:
                time_adj[fi] = col.mean()
        resid_t2 = resid_t - time_adj[None, :]

        # update residual snapshot (point-in-time) with tau's observed cells
        for k, e in enumerate(ent_t):
            ob = obs_t[k]
            res_snap[e, ob] = resid_t2[k, ob]
            res_have[e, ob] = True

        # CDRec completion of the accumulated residual snapshot (rows = seen ents)
        seen_ent = np.where(res_have.any(axis=1))[0]
        lowrank_full = np.zeros((E, F))
        if seen_ent.size >= 2:
            Rblk = res_snap[seen_ent]
            Wblk = res_have[seen_ent]
            # baseline fill for missing snapshot cells: column mean of observed
            colm = np.zeros(F)
            for fi in range(F):
                c = Rblk[Wblk[:, fi], fi]
                colm[fi] = c.mean() if c.size else 0.0
            fill0 = np.broadcast_to(colm, Rblk.shape)
            comp = _cdrec_complete(Rblk, Wblk, fill0, RANK, CDREC_ITERS)
            lowrank_full[seen_ent] = comp

        # fill missing cells at tau: FE + time_adj + CDRec residual
        for k, e in enumerate(ent_t):
            miss = ~obs_t[k]
            if miss.any():
                out[rs[k], miss] = (ent_fe[k, miss] + time_adj[miss]
                                    + lowrank_full[e, miss])

        # fold tau's observed values into expanding FE
        for k, e in enumerate(ent_t):
            ob = obs_t[k]
            ent_sum[e, ob] += Xt[k, ob]; ent_cnt[e, ob] += 1
            g_sum[ob] += Xt[k, ob]; g_cnt[ob] += 1

    if np.isnan(out).any():
        gfe = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 0.0)
        nanidx = np.where(np.isnan(out))
        out[nanidx] = np.take(gfe, nanidx[1])
    return out


# --------------------------------------------------------------------------- #
# Causal 2D CDRec for small-N structured reals (airq). Column FE expanding;
# CDRec completion of the accumulated residual snapshot at each time (windowed).
# --------------------------------------------------------------------------- #
TWO_D_RANK = 4
TWO_D_ITERS = 10
TWO_D_WINDOW = 400


def _impute_2d_cdrec(X, meta):
    T, N = X.shape
    out = X.copy()
    col_sum = np.zeros(N); col_cnt = np.zeros(N)
    res_snap = np.zeros((T, N)); res_have = np.zeros((T, N), dtype=bool)

    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)
        col_fe = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        row_fe = float((row[obs] - col_fe[obs]).mean()) if obs.any() else 0.0

        resid = np.zeros(N)
        resid[obs] = row[obs] - col_fe[obs] - row_fe
        res_snap[t, obs] = resid[obs]; res_have[t, obs] = True

        lr_row = np.zeros(N)
        if t >= 2:
            lo = max(0, t - TWO_D_WINDOW + 1)
            seen = np.arange(lo, t + 1)
            seen = seen[res_have[seen].any(axis=1)]
            if seen.size >= 3:
                Rblk = res_snap[seen]; Wblk = res_have[seen]
                colm = np.zeros(N)
                for fi in range(N):
                    c = Rblk[Wblk[:, fi], fi]
                    colm[fi] = c.mean() if c.size else 0.0
                fill0 = np.broadcast_to(colm, Rblk.shape)
                comp = _cdrec_complete(Rblk, Wblk, fill0, TWO_D_RANK, TWO_D_ITERS)
                lr_row = comp[np.where(seen == t)[0][0]]

        miss = ~obs
        if miss.any():
            out[t, miss] = col_fe[miss] + row_fe + lr_row[miss]
        col_sum[obs] += row[obs]; col_cnt[obs] += 1

    if np.isnan(out).any():
        col_fe = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        nanidx = np.where(np.isnan(out))
        out[nanidx] = np.take(col_fe, nanidx[1])
    return out


# --------------------------------------------------------------------------- #
# Cheap up-front routing signal: is the missingness contiguous (block-like)?
# Measure mean run-length of missing cells per column vs. an MCAR expectation.
# --------------------------------------------------------------------------- #
def _is_blocky(X):
    miss = np.isnan(X)
    rate = miss.mean()
    if rate <= 0:
        return False
    T, N = X.shape
    # count vertical runs of missing per column: runs = transitions/2-ish
    # number of missing cells / number of run-starts = mean run length
    starts = miss & ~np.vstack([np.zeros((1, N), bool), miss[:-1]])
    n_runs = starts.sum()
    if n_runs == 0:
        return False
    mean_run = miss.sum() / n_runs
    # MCAR mean run length ~ 1/(1-rate). Block-like if observed runs much longer.
    mcar_run = 1.0 / max(1e-6, 1.0 - rate)
    return mean_run >= 2.0 * mcar_run and mean_run >= 3.0


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if _is_panel(meta):
        # Only PER-SERIES contiguous blocks benefit from CDRec. Synchronized
        # blackout (no cross-section anywhere) and scattered MCAR/MAR are handled
        # better by the warm-started ALS in CausalFE -> leave them untouched.
        if _is_blocky(X) and not _is_blackout_panel(X, meta):
            return _impute_panel_cdrec(X, meta)
        return _fe(X, meta)
    # 2D: champion cores are already best (CDRec regressed airq/1d) -> untouched.
    return _trmf(X, meta)


def _is_blackout_panel(X, meta):
    """Synchronized blackout: at some times EVERY entity is fully missing. Detect by
    checking whether any time has zero observed cells across all its rows."""
    tids = np.asarray(meta["time_ids"])
    miss = np.isnan(X)
    allmiss_rows = miss.all(axis=1)              # rows fully missing
    if not allmiss_rows.any():
        return False
    # a time is blacked out if ALL its rows are fully missing
    for t in np.unique(tids[allmiss_rows]):
        rows = tids == t
        if miss[rows].all():
            return True
    return False


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_cdrec", online_impute))
