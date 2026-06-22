"""
MC-NNM-FE : Athey et al. (2021) matrix completion with two-way fixed effects.

Pipeline:
  1. Remove UNREGULARIZED two-way fixed effects by iterative demeaning over the
     OBSERVED cells only (Method of Alternating Projections / FWL).
        - PANEL datasets: the two FE dimensions are ENTITY and TIME, taken from
          meta['entity_ids'] / meta['time_ids']. Each is removed PER FEATURE
          (column), since each feature has its own entity & time effects.
        - 2D datasets (meta empty): the two FE dimensions are ROW (time) and
          COLUMN (feature). Remove row means + column means iteratively.
  2. Complete the low-rank residual with SoftImpute-ALS (nuclear-norm
     regularized matrix completion via alternating ridge on observed cells).
  3. ADD the fixed effects back; return the dense reconstruction.

This unifies the 2D and panel cases: panel FE-partialling is just two-way
demeaning on richer index sets.
"""
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from harness import run_method, summarize


# --------------------------------------------------------------------------- #
# Two-way fixed-effect demeaning over OBSERVED cells (MAP / FWL)
# --------------------------------------------------------------------------- #
def _demean_two_way_2d(X, obs, max_iter=100, tol=1e-7):
    """
    Remove row-effect + col-effect from matrix X using only observed cells (obs
    boolean mask). Returns (resid, row_eff, col_eff, grand). Reconstruction is
    grand + row_eff[:,None] + col_eff[None,:] + resid (on observed cells the
    sum of effects approximates X; resid carries the rest).
    """
    T, N = X.shape
    Xo = np.where(obs, X, 0.0)
    row_cnt = obs.sum(axis=1)            # (T,)
    col_cnt = obs.sum(axis=0)            # (N,)
    row_cnt_s = np.where(row_cnt > 0, row_cnt, 1)
    col_cnt_s = np.where(col_cnt > 0, col_cnt, 1)

    grand = Xo.sum() / max(obs.sum(), 1)
    row_eff = np.zeros(T)
    col_eff = np.zeros(N)

    for _ in range(max_iter):
        # current residual on observed cells (without row/col effects)
        # update row effects: mean over observed cols of (X - grand - col_eff)
        partial = Xo - obs * (grand + col_eff[None, :] + row_eff[:, None])
        d_row = partial.sum(axis=1) / row_cnt_s
        row_eff = row_eff + d_row

        partial = Xo - obs * (grand + col_eff[None, :] + row_eff[:, None])
        d_col = partial.sum(axis=0) / col_cnt_s
        col_eff = col_eff + d_col

        if max(np.max(np.abs(d_row)), np.max(np.abs(d_col))) < tol:
            break

    fitted = grand + row_eff[:, None] + col_eff[None, :]
    resid = np.where(obs, X - fitted, 0.0)
    return resid, fitted


def _demean_panel(X, obs, eids, tids, E, T, max_iter=200, tol=1e-7):
    """
    Panel two-way FE per feature (column). For each feature f independently,
    remove entity FE (over E entities) and time FE (over T periods) using only
    observed cells. Vectorized across features.

    X    : (ET, F) stacked entity-major
    obs  : (ET, F) bool
    eids : (ET,) entity index, tids : (ET,) time index
    Returns (resid, fitted) both (ET, F).
    """
    ET, F = X.shape
    Xo = np.where(obs, X, 0.0)

    grand = Xo.sum(axis=0) / np.maximum(obs.sum(axis=0), 1)   # (F,)
    ent_eff = np.zeros((E, F))
    time_eff = np.zeros((T, F))

    # counts per (entity,feature) and (time,feature)
    ent_cnt = np.zeros((E, F))
    np.add.at(ent_cnt, eids, obs.astype(float))
    time_cnt = np.zeros((T, F))
    np.add.at(time_cnt, tids, obs.astype(float))
    ent_cnt_s = np.where(ent_cnt > 0, ent_cnt, 1.0)
    time_cnt_s = np.where(time_cnt > 0, time_cnt, 1.0)

    for _ in range(max_iter):
        fitted = grand[None, :] + ent_eff[eids] + time_eff[tids]
        partial = Xo - obs * fitted                          # (ET,F)
        # entity update
        d_ent = np.zeros((E, F))
        np.add.at(d_ent, eids, partial)
        d_ent /= ent_cnt_s
        ent_eff = ent_eff + d_ent

        fitted = grand[None, :] + ent_eff[eids] + time_eff[tids]
        partial = Xo - obs * fitted
        d_time = np.zeros((T, F))
        np.add.at(d_time, tids, partial)
        d_time /= time_cnt_s
        time_eff = time_eff + d_time

        if max(np.max(np.abs(d_ent)), np.max(np.abs(d_time))) < tol:
            break

    fitted = grand[None, :] + ent_eff[eids] + time_eff[tids]
    resid = np.where(obs, X - fitted, 0.0)
    return resid, fitted


# --------------------------------------------------------------------------- #
# SoftImpute-ALS: nuclear-norm regularized low-rank matrix completion
# --------------------------------------------------------------------------- #
def _soft_impute_als(R, obs, rank, lam, max_iter=100, tol=1e-5):
    """
    Solve min_{Z} 1/2 ||P_obs(R - Z)||^2 + lam ||Z||_*  via ALS with factors
    Z = U V^T (U: m x r, V: n x r). Alternating ridge solves, with the
    standard SoftImpute filling of unobserved cells by current estimate.

    R   : (m,n) residual with zeros at unobserved cells (already centered)
    obs : (m,n) bool mask of observed cells
    """
    m, n = R.shape
    r = min(rank, m, n)
    if r < 1:
        return np.zeros_like(R)
    rng = np.random.default_rng(0)
    U = rng.standard_normal((m, r)) * 0.01
    V = rng.standard_normal((n, r)) * 0.01
    Z = U @ V.T
    Ir = np.eye(r)

    prev = None
    for _ in range(max_iter):
        # Fill: on observed cells use R, on missing use current Z
        Xf = np.where(obs, R, Z)
        # Update U: for each row solve (V^T V + lam I) u = V^T x_row
        VtV = V.T @ V + lam * Ir
        U = np.linalg.solve(VtV, (Xf @ V).T).T
        # Update V
        Xf = np.where(obs, R, U @ V.T)
        UtU = U.T @ U + lam * Ir
        V = np.linalg.solve(UtU, (Xf.T @ U).T).T
        Z = U @ V.T
        # convergence on observed-cell fit
        cur = np.sum((np.where(obs, R - Z, 0.0)) ** 2)
        if prev is not None and abs(prev - cur) <= tol * (prev + 1e-12):
            break
        prev = cur
    return Z


# --------------------------------------------------------------------------- #
# The imputer
# --------------------------------------------------------------------------- #
def impute(X, meta):
    obs = np.isfinite(X)
    Xc = np.where(obs, X, 0.0)

    if meta and "entity_ids" in meta:
        eids = np.asarray(meta["entity_ids"], dtype=int)
        tids = np.asarray(meta["time_ids"], dtype=int)
        E = int(meta["E"]); T = int(meta["T"]); F = int(meta["F"])
        resid, fitted = _demean_panel(Xc, obs, eids, tids, E, T)
        # residual is an entity-time interaction; low rank in F is small,
        # the meaningful structure is along entity/time. Complete on (ET,F).
        rank = min(F, 6)
        # scale lambda to residual energy
        s = np.std(resid[obs]) + 1e-9
        lam = 0.05 * s * np.sqrt(max(resid.shape))
        Zres = _soft_impute_als(resid, obs, rank=rank, lam=lam, max_iter=150)
        out = fitted + Zres
    else:
        T, N = X.shape
        resid, fitted = _demean_two_way_2d(Xc, obs)
        rank = min(N - 1, max(2, N // 2), 12)
        s = np.std(resid[obs]) + 1e-9
        lam = 0.03 * s * np.sqrt(max(T, N))
        Zres = _soft_impute_als(resid, obs, rank=rank, lam=lam, max_iter=200)
        out = fitted + Zres

    out = np.where(obs, X, out)
    return out


# --------------------------------------------------------------------------- #
# Ablation imputer: NO fixed-effect partialling (just SoftImpute on raw X).
# Used only to report whether FE-partialling helps.
# --------------------------------------------------------------------------- #
def impute_nofe(X, meta):
    obs = np.isfinite(X)
    col = np.where(np.isfinite(np.nanmean(np.where(obs, X, np.nan), axis=0)),
                   np.nanmean(np.where(obs, X, np.nan), axis=0), 0.0)
    R = np.where(obs, X - col[None, :], 0.0)
    m, n = X.shape
    rank = min(m, n, 12)
    s = np.std(R[obs]) + 1e-9
    lam = 0.03 * s * np.sqrt(max(m, n))
    Z = _soft_impute_als(R, obs, rank=rank, lam=lam, max_iter=200)
    out = col[None, :] + Z
    out = np.where(obs, X, out)
    return out


if __name__ == "__main__":
    rows = run_method("MC-NNM-FE", impute)
    summarize(rows)
    print()
    rows_nofe = run_method("SoftImpute(no FE)", impute_nofe)
    summarize(rows_nofe)
