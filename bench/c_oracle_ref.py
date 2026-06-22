"""
c_oracle_ref.py -- NON-CAUSAL ORACLE CEILING for the TIMARA causal benchmark.

These methods INTENTIONALLY look ahead (full-series fits) and define the
cost-of-causality denominator. We call run_causal(..., verify=False) so the
causality verifier is NOT run, and we force every row's causal=False with
causal_detail="oracle: bidirectional, non-causal ceiling".

Two bidirectional baselines on the SAME 7 datasets:
  1. TRMF-oracle : low-rank temporal regularized matrix factorization. Factors
     X ~ W @ H^T where the temporal factor W carries an AR(p) smoothness
     penalty over the FULL series (bidirectional). For panels we demean by
     two-way (entity + time) fixed effects first, factor the residual on the
     (T x F)-per-entity dense tensor reshaped as a (T, E*F) matrix, then add FE
     back. ALS with closed-form ridge updates; Gram caching.
  2. MCNNM-oracle : Matrix Completion with Nuclear-Norm Minimization
     (Athey et al). Two-way fixed effects + SoftImpute (SVD soft-threshold)
     on the residual matrix over the full sample.

Speed tricks: thread caps at top, closed-form ridge ALS, cached Gram B^T B,
cho_solve, factored low-rank reconstruction, capped inner iterations.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")

import numpy as np
from scipy.linalg import cho_factor, cho_solve, svd
from causal import run_causal, summarize_causal


# --------------------------------------------------------------------------- #
# Panel reshaping helpers (entity-major stacked  <->  (T, E*F) dense)
# --------------------------------------------------------------------------- #
def _panel_to_dense(X, meta):
    """(E*T, F) entity-major stacked -> dense (T, E*F) matrix + index maps.

    Returns Xd (T, E*F), and a map so we can scatter back. Column block for
    entity e occupies columns [e*F : (e+1)*F]. Missing entries stay NaN.
    """
    eids = np.asarray(meta["time_ids"]) * 0 + np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    E, T, F = int(meta["E"]), int(meta["T"]), int(meta["F"])
    Xd = np.full((T, E * F), np.nan)
    for r in range(X.shape[0]):
        e = int(eids[r]); t = int(tids[r])
        Xd[t, e * F:(e + 1) * F] = X[r]
    return Xd, (E, T, F, eids, tids)


def _dense_to_panel(Xd, info, like):
    E, T, F, eids, tids = info
    out = like.copy()
    for r in range(like.shape[0]):
        e = int(eids[r]); t = int(tids[r])
        out[r] = Xd[t, e * F:(e + 1) * F]
    return out


# --------------------------------------------------------------------------- #
# Two-way fixed effects (non-causal: uses full sample means)
# --------------------------------------------------------------------------- #
def _twoway_fe(Xd):
    """Iterative two-way demeaning on a (T, C) matrix with NaNs.

    Returns residual R (NaN where input NaN) and the additive components
    (grand, row_eff (T,), col_eff (C,)) so we can reconstruct.
    """
    T, C = Xd.shape
    obs = ~np.isnan(Xd)
    grand = np.nanmean(Xd)
    row_eff = np.zeros(T)
    col_eff = np.zeros(C)
    Xc = np.where(obs, Xd, np.nan)
    row_has = obs.any(axis=1)
    col_has = obs.any(axis=0)
    for _ in range(20):
        cur = Xc - grand - row_eff[:, None] - col_eff[None, :]
        # row update
        with np.errstate(invalid="ignore", divide="ignore"):
            r_adj = np.where(row_has, np.nansum(np.where(obs, cur, 0.0), axis=1)
                             / np.maximum(obs.sum(axis=1), 1), 0.0)
        row_eff = row_eff + r_adj
        cur = Xc - grand - row_eff[:, None] - col_eff[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            c_adj = np.where(col_has, np.nansum(np.where(obs, cur, 0.0), axis=0)
                             / np.maximum(obs.sum(axis=0), 1), 0.0)
        col_eff = col_eff + c_adj
    R = Xd - grand - row_eff[:, None] - col_eff[None, :]
    return R, (grand, row_eff, col_eff)


def _fe_reconstruct(grand, row_eff, col_eff):
    return grand + row_eff[:, None] + col_eff[None, :]


# --------------------------------------------------------------------------- #
# Bidirectional TRMF (full-series, NON-CAUSAL)
# --------------------------------------------------------------------------- #
def _trmf(Xd, rank=6, lam=1.0, eta=10.0, ar_p=1, n_iter=30, seed=0):
    """Temporal Regularized Matrix Factorization on dense (T, C) with NaNs.

    X ~ W (T x k) @ H^T (C x k). W carries an AR smoothness penalty that ties
    W[t] to a linear combo of W[t-1..t-p] over the FULL series (bidirectional,
    look-ahead). ALS with closed-form ridge; missing entries skipped per row/col.
    """
    rng = np.random.default_rng(seed)
    T, C = Xd.shape
    obs = ~np.isnan(Xd)
    Xz = np.where(obs, Xd, 0.0)
    # init
    col_std = np.nanstd(Xd) + 1e-6
    W = rng.standard_normal((T, rank)) * 0.1
    H = rng.standard_normal((C, rank)) * (col_std / np.sqrt(rank))

    Ik = np.eye(rank)

    # global AR coefficient(s) estimated from W each sweep (full series)
    for it in range(n_iter):
        # ---- update H (per column, closed-form ridge over observed rows) ----
        for j in range(C):
            ridx = np.nonzero(obs[:, j])[0]
            if ridx.size == 0:
                continue
            Wj = W[ridx]                       # (n_j, k)
            G = Wj.T @ Wj + lam * Ik
            rhs = Wj.T @ Xd[ridx, j]
            try:
                cf = cho_factor(G, lower=True, check_finite=False)
                H[j] = cho_solve(cf, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                H[j] = np.linalg.solve(G, rhs)

        # ---- estimate AR(p) coefficients on W (full series, bidirectional) --
        if ar_p >= 1 and T > ar_p + 2:
            # build lagged design for each latent dim jointly (shared coefs)
            Yt = W[ar_p:]                      # (T-p, k)
            Lag = np.concatenate([W[ar_p - l - 1:T - l - 1] for l in range(ar_p)],
                                 axis=1)       # (T-p, p*k) but we want per-lag scalar
            # shared scalar coef per lag: regress vec(Yt) on vec of each lag
            # Build (n, p) design by stacking dims
            n = Yt.shape[0] * rank
            yv = Yt.reshape(-1)
            Xdes = np.empty((n, ar_p))
            for l in range(ar_p):
                Xdes[:, l] = W[ar_p - l - 1:T - l - 1].reshape(-1)
            GG = Xdes.T @ Xdes + 1e-6 * np.eye(ar_p)
            theta = np.linalg.solve(GG, Xdes.T @ yv)
        else:
            theta = np.zeros(max(ar_p, 1))

        # ---- update W (per time, closed-form ridge with AR temporal penalty) -
        # penalty: eta * || W[t] - sum_l theta_l W[t-l] ||^2  (and forward terms)
        HtH_cache = None
        for t in range(T):
            cidx = np.nonzero(obs[t])[0]
            A = lam * Ik + eta * Ik            # base ridge + temporal anchor self term
            b = np.zeros(rank)
            if cidx.size > 0:
                Ht = H[cidx]
                A = A + Ht.T @ Ht
                b = b + Ht.T @ Xd[t, cidx]
            # temporal anchor: pull W[t] toward AR prediction from past lags
            pred = np.zeros(rank)
            if ar_p >= 1:
                for l in range(ar_p):
                    if t - l - 1 >= 0:
                        pred = pred + theta[l] * W[t - l - 1]
            b = b + eta * pred
            try:
                cf = cho_factor(A, lower=True, check_finite=False)
                W[t] = cho_solve(cf, b, check_finite=False)
            except np.linalg.LinAlgError:
                W[t] = np.linalg.solve(A, b)

    recon = W @ H.T
    return recon


# --------------------------------------------------------------------------- #
# Bidirectional MC-NNM: two-way FE + SoftImpute (NON-CAUSAL)
# --------------------------------------------------------------------------- #
def _softimpute(Xd, R, fe_parts, max_rank=None, n_iter=120, tol=1e-5,
                lam_floor_frac=0.02):
    """SoftImpute on residual R (NaN where missing).

    Warm-start lambda from the top singular value and anneal geometrically to a
    small floor (a fraction of s_max), so the ceiling reconstructs as much
    low-rank structure as possible. Inner iters capped at n_iter.
    """
    T, C = R.shape
    obs = ~np.isnan(R)
    if max_rank is None:
        max_rank = min(T, C, 40)
    M = np.zeros((T, C))
    # initial spectrum on mean-filled residual
    s_max = svd(np.where(obs, R, 0.0), full_matrices=False, compute_uv=False)[0]
    lam = s_max
    lam_floor = lam_floor_frac * s_max + 1e-9
    prev = np.inf
    for it in range(n_iter):
        filled = np.where(obs, R, M)
        U, s, Vt = svd(filled, full_matrices=False)
        s_th = np.maximum(s - lam, 0.0)
        k = int(np.sum(s_th > 0))
        if k == 0:
            k = 1
            s_th = s.copy(); s_th[1:] = 0.0
        if max_rank is not None and k > max_rank:
            k = max_rank
        Mnew = (U[:, :k] * s_th[:k]) @ Vt[:k]
        change = np.linalg.norm(Mnew - M) / (np.linalg.norm(M) + 1e-9)
        M = Mnew
        if lam > lam_floor:
            lam = max(lam * 0.85, lam_floor)
        elif change < tol and it > 10:
            break
        prev = change
    grand, row_eff, col_eff = fe_parts
    return _fe_reconstruct(grand, row_eff, col_eff) + M


# --------------------------------------------------------------------------- #
# Public imputers (online_impute signature) -- ORACLE, look-ahead allowed
# --------------------------------------------------------------------------- #
def _is_panel(meta):
    return bool(meta) and "time_ids" in meta


def trmf_oracle(X, meta):
    if _is_panel(meta):
        Xd, info = _panel_to_dense(X, meta)
        R, fe = _twoway_fe(Xd)
        recon_R = _trmf(R, rank=12, lam=0.5, eta=5.0, ar_p=2, n_iter=35)
        Xd_hat = _fe_reconstruct(*fe) + recon_R
        # fill only missing in dense, keep observed
        miss = np.isnan(Xd)
        Xd_filled = np.where(miss, Xd_hat, Xd)
        return _dense_to_panel(Xd_filled, info, X)
    else:
        T, N = X.shape
        recon = _trmf(X, rank=min(8, N), lam=1.0, eta=10.0, ar_p=2, n_iter=30)
        out = np.where(np.isnan(X), recon, X)
        return out


def mcnnm_oracle(X, meta):
    if _is_panel(meta):
        Xd, info = _panel_to_dense(X, meta)
        R, fe = _twoway_fe(Xd)
        Xd_hat = _softimpute(Xd, R, fe, max_rank=20, n_iter=60)
        miss = np.isnan(Xd)
        Xd_filled = np.where(miss, Xd_hat, Xd)
        return _dense_to_panel(Xd_filled, info, X)
    else:
        T, N = X.shape
        R, fe = _twoway_fe(X)
        Xhat = _softimpute(X, R, fe, max_rank=min(T, N, 25), n_iter=60)
        out = np.where(np.isnan(X), Xhat, X)
        return out


# --------------------------------------------------------------------------- #
def _force_oracle(rows, tag):
    out = []
    for r in rows:
        r = dict(r)
        r["dataset"] = f"{tag}:{r['dataset']}"
        r["causal"] = False
        r["causal_detail"] = "oracle: bidirectional, non-causal ceiling"
        r["method"] = "OracleCeiling"
        out.append(r)
    return out


def main():
    trmf_rows = run_causal("OracleCeiling", trmf_oracle, verify=False)
    mcnnm_rows = run_causal("OracleCeiling", mcnnm_oracle, verify=False)
    trmf_rows = _force_oracle(trmf_rows, "TRMF-oracle")
    mcnnm_rows = _force_oracle(mcnnm_rows, "MCNNM-oracle")
    all_rows = trmf_rows + mcnnm_rows
    summarize_causal(all_rows)
    return all_rows


if __name__ == "__main__":
    main()
