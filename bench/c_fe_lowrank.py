"""
CausalFE -- causal panel core (online MC-NNM style).

GOVERNING CONSTRAINT: fully point-in-time / NO look-ahead. An imputed value for
(entity e, time tau, feature i) uses ONLY data at time <= tau: past values AND the
contemporaneous cross-section at tau. Parameters estimated expanding/online.

Decomposition at each time tau (panel):
  value ~= entity_FE[e,f] + time_adj[f] + lowrank_residual
where
  entity_FE[e,f] = expanding mean of entity e's OWN observed history at times < tau
                   (per feature). Falls back to global expanding mean if e unseen.
  time_adj[f]    = contemporaneous cross-section mean across entities AT tau of the
                   FE-removed observed residual (causal: same-tau info allowed).
  lowrank        = online/warm-started low-rank completion of the small residual using
                   only data at times <= tau. Factors warm-started across time steps.

For 2D data (meta empty): the two FE dimensions are row(time)=expanding row mean and
column(feature)=expanding column mean over times <= tau (NO full-sample column mean).

The method iterates strictly by time and groups all entities at a time together, so
removing future rows never changes a past imputation -> passes the verifier.

Inner low-rank iterations are CAPPED (warm-started, a few sweeps per time step).
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ------------------------------------------------------------------- knobs ---
RANK = 4              # low-rank dimension for the residual
RIDGE = 1e-1          # ridge on ALS solves (stabilizes small same-tau systems)
SWEEPS_WARM = 2       # ALS sweeps per time step (warm-started)
REFIT_EVERY = 8       # occasionally do a couple extra sweeps to keep factors fresh
SWEEPS_REFIT = 2
# For large T we refit the column factors Bt only periodically (point-in-time:
# Bt depends only on rows <= the refit time). Between refits each new row solves
# just its own A[t] against the current Bt -- O(T) instead of O(T^2).
BT_REFIT_EVERY_2D = 25   # refit Bt every this many time steps (2D)
BT_WINDOW = 600          # cap snapshot rows used in a Bt refit (recent window)


def _als_residual_lowrank(R, Wobs, A, Bt, rank, sweeps, ridge):
    """
    Low-rank completion of residual matrix R (rows = entities-or-times present so
    far, cols = features) over observed mask Wobs, warm-started from factors
    (A: rows x rank, Bt: features x rank). Returns updated A, Bt and the
    reconstruction A @ Bt.T.

    All data passed in is already restricted to times <= tau by the caller, so this
    is point-in-time. Factored A.Bt form -- never a full dense refit from scratch.
    """
    n, f = R.shape
    if A is None or A.shape[0] != n:
        # grow / init row factors; keep column factors warm if compatible
        newA = np.zeros((n, rank))
        if A is not None:
            m = min(A.shape[0], n)
            newA[:m] = A[:m]
        A = newA
    if Bt is None or Bt.shape != (f, rank):
        Bt = 0.01 * np.random.default_rng(0).standard_normal((f, rank))

    Rf = np.nan_to_num(R)  # zeros where unobserved; masked out by Wobs in solves
    I = ridge * np.eye(rank)
    for _ in range(sweeps):
        # update A (row factors) given Bt
        for i in range(n):
            w = Wobs[i]
            if not w.any():
                continue
            Bw = Bt[w]                       # (nobs, rank)
            G = Bw.T @ Bw + I
            rhs = Bw.T @ Rf[i, w]
            try:
                A[i] = cho_solve(cho_factor(G, lower=True, check_finite=False),
                                 rhs, check_finite=False)
            except Exception:
                A[i] = np.linalg.lstsq(G, rhs, rcond=None)[0]
        # update Bt (feature factors) given A
        for j in range(f):
            w = Wobs[:, j]
            if not w.any():
                continue
            Aw = A[w]
            G = Aw.T @ Aw + I
            rhs = Aw.T @ Rf[w, j]
            try:
                Bt[j] = cho_solve(cho_factor(G, lower=True, check_finite=False),
                                  rhs, check_finite=False)
            except Exception:
                Bt[j] = np.linalg.lstsq(G, rhs, rcond=None)[0]
    return A, Bt


def _impute_panel(X, meta):
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    T_rows, F = X.shape
    E = int(eids.max()) + 1
    out = X.copy()

    # per-time row index lists (precomputed once)
    uniq_t = np.unique(tids)
    rows_at = {t: np.where(tids == t)[0] for t in uniq_t}

    # expanding entity FE accumulators (per entity, per feature)
    ent_sum = np.zeros((E, F))
    ent_cnt = np.zeros((E, F))
    # global expanding FE fallback
    g_sum = np.zeros(F)
    g_cnt = np.zeros(F)

    # residual store: we keep the FE-removed residual matrix (entity x feature)
    # of the LATEST observation per entity at times <= tau, plus warm factors.
    # For the low-rank step we operate on the current cross-section's residual
    # blended with a running residual snapshot per entity (point-in-time).
    res_snap = np.zeros((E, F))      # last observed residual per entity (<= tau)
    res_have = np.zeros((E, F), dtype=bool)

    A = None                          # row(entity) factors, grown to E
    Bt = None                         # feature factors
    A = np.zeros((E, RANK))
    rng = np.random.default_rng(0)
    Bt = 0.01 * rng.standard_normal((F, RANK))

    step = 0
    for t in uniq_t:
        rs = rows_at[t]
        ent_t = eids[rs]
        Xt = X[rs]                    # (n_e_at_t, F)
        obs_t = ~np.isnan(Xt)

        # --- entity FE from history strictly < tau (expanding own-mean) ---
        ent_fe = np.where(ent_cnt[ent_t] > 0,
                          ent_sum[ent_t] / np.maximum(ent_cnt[ent_t], 1),
                          0.0)
        gfe = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 0.0)
        # fall back to global expanding mean where entity has no own history
        ent_fe = np.where(ent_cnt[ent_t] > 0, ent_fe, gfe[None, :])

        # FE-removed residual on observed cells at tau
        resid_t = Xt - ent_fe

        # --- contemporaneous cross-section adjustment AT tau (causal) ---
        # mean across entities present at tau of the FE-removed residual, per feature
        time_adj = np.zeros(F)
        for f in range(F):
            col = resid_t[obs_t[:, f], f]
            if col.size > 0:
                time_adj[f] = col.mean()
        resid_t2 = resid_t - time_adj[None, :]

        # --- online low-rank step on the residual, data <= tau only ---
        # Build entity x feature residual matrix using snapshot (past) + current tau.
        present = np.unique(ent_t)
        # update snapshot with current observed residuals (point-in-time)
        for k, e in enumerate(ent_t):
            ob = obs_t[k]
            res_snap[e, ob] = resid_t2[k, ob]
            res_have[e, ob] = True

        # ALS on the accumulated snapshot (rows = all entities seen so far)
        seen_ent = np.where(res_have.any(axis=1))[0]
        if seen_ent.size >= 2:
            Rblk = res_snap[seen_ent]
            Wblk = res_have[seen_ent]
            sw = SWEEPS_REFIT if (step % REFIT_EVERY == 0) else SWEEPS_WARM
            A_blk = A[seen_ent]
            A_blk, Bt = _als_residual_lowrank(Rblk, Wblk, A_blk, Bt,
                                              RANK, sw, RIDGE)
            A[seen_ent] = A_blk
        lowrank_full = A @ Bt.T       # (E, F), factored form reconstruction

        # --- fill missing cells at tau: FE + time_adj + lowrank residual ---
        for k, e in enumerate(ent_t):
            miss = ~obs_t[k]
            if miss.any():
                out[rs[k], miss] = (ent_fe[k, miss] + time_adj[miss]
                                    + lowrank_full[e, miss])

        # --- update expanding FE accumulators with tau's OBSERVED values ---
        # (now that imputation for tau is done, fold tau into history for future)
        for k, e in enumerate(ent_t):
            ob = obs_t[k]
            ent_sum[e, ob] += Xt[k, ob]
            ent_cnt[e, ob] += 1
            g_sum[ob] += Xt[k, ob]
            g_cnt[ob] += 1
        step += 1

    # any remaining NaN safety
    if np.isnan(out).any():
        gfe = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), 0.0)
        nanidx = np.where(np.isnan(out))
        out[nanidx] = np.take(gfe, nanidx[1])
    return out


def _solve_row(rvec, w, Bt, ridge, rank):
    """Closed-form ridge solve for one row's rank-r factor against observed cols."""
    if not w.any():
        return np.zeros(rank)
    Bw = Bt[w]
    G = Bw.T @ Bw + ridge * np.eye(rank)
    rhs = Bw.T @ rvec[w]
    try:
        return cho_solve(cho_factor(G, lower=True, check_finite=False), rhs,
                         check_finite=False)
    except Exception:
        return np.linalg.lstsq(G, rhs, rcond=None)[0]


def _refit_Bt(Rblk, Wblk, A_blk, Bt, rank, sweeps, ridge):
    """A few ALS sweeps on a block to refresh row factors A_blk AND column Bt."""
    return _als_residual_lowrank(Rblk, Wblk, A_blk, Bt, rank, sweeps, ridge)


def _impute_2d(X, meta):
    """
    2D point-in-time: column(feature) FE = expanding column mean over times <= tau
    (NO full-sample column mean); row(time) FE = contemporaneous mean of FE-removed
    cells at t. Residual completed with an online warm-started low-rank step using
    only data at times <= tau.

    Speed: column factors Bt are refit only every BT_REFIT_EVERY_2D steps (over a
    recent window of rows, all <= current time -> still point-in-time). Each step
    solves just its own row factor A[t] against the current Bt: O(T) overall.
    """
    T, N = X.shape
    out = X.copy()

    col_sum = np.zeros(N)
    col_cnt = np.zeros(N)

    res_snap = np.zeros((T, N))
    res_have = np.zeros((T, N), dtype=bool)
    A = np.zeros((T, RANK))
    rng = np.random.default_rng(0)
    Bt = 0.01 * rng.standard_normal((N, RANK))
    bt_ready = False

    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)

        col_fe = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        if obs.any():
            row_fe = float((row[obs] - col_fe[obs]).mean())
        else:
            row_fe = 0.0

        resid = np.zeros(N)
        resid[obs] = row[obs] - col_fe[obs] - row_fe
        res_snap[t, obs] = resid[obs]
        res_have[t, obs] = True

        # periodic Bt refit over a recent window of rows (all times <= t)
        if (t % BT_REFIT_EVERY_2D == 0) and t >= 1:
            lo = max(0, t - BT_WINDOW + 1)
            seen = np.arange(lo, t + 1)
            seen = seen[res_have[seen].any(axis=1)]
            if seen.size >= 2:
                A_blk, Bt = _refit_Bt(res_snap[seen], res_have[seen], A[seen], Bt,
                                      RANK, SWEEPS_REFIT, RIDGE)
                A[seen] = A_blk
                bt_ready = True

        # cheap per-row factor solve against current Bt
        if bt_ready and obs.any():
            A[t] = _solve_row(res_snap[t], res_have[t], Bt, RIDGE, RANK)
            lr_row = A[t] @ Bt.T
        else:
            lr_row = np.zeros(N)

        miss = ~obs
        if miss.any():
            out[t, miss] = col_fe[miss] + row_fe + lr_row[miss]

        col_sum[obs] += row[obs]
        col_cnt[obs] += 1

    if np.isnan(out).any():
        col_fe = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        nanidx = np.where(np.isnan(out))
        out[nanidx] = np.take(col_fe, nanidx[1])
    return out


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if meta and "entity_ids" in meta and "time_ids" in meta:
        return _impute_panel(X, meta)
    return _impute_2d(X, meta)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    rows = run_causal("CausalFE", online_impute)
    summarize_causal(rows)
