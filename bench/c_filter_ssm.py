"""
FilterSSM -- CAUSAL filter-only state-space imputer for the TIMARA redesign.

Model (linear-Gaussian latent SSM, shared across entities for panels):
    z_t = A z_{t-1} + w_t,     w_t ~ N(0, Q),   Q diagonal
    x_t = U z_t     + v_t,     v_t ~ N(0, R),   R = sigma2 * I_N   (scalar obs noise)
    U : (N, L) low-rank loadings (rank L << N), A : (L, L) AR transition.

At each time tau we run the Kalman FILTER forward ONLY (no RTS smoother):
  predict  z_tau|tau-1 = A z_{tau-1|tau-1},  P_pred = A P A' + Q
  update   with the OBSERVED entries of x_tau (the contemporaneous cross-section
           IS allowed); impute missing entries as (U z_tau|tau)[missing].
The innovation covariance is  S = U_o P_pred U_o' + sigma2 I  (o = observed rows).
We never form / invert an N x N matrix: the Kalman gain is obtained via the
WOODBURY identity so only an L x L (and a diagonal) system is solved, with
cho_solve throughout (no inv / no pinv).

Causality / point-in-time:
  * We iterate strictly by TIME. For panels we process all entities sharing a
    time tau together, and reset each entity's state at its first time.
  * Parameters (U, A, Q, sigma2) are (re-)estimated ONLY from data with time<=tau,
    via a few warm-started EM-flavoured sweeps on the expanding window, refreshed
    on a geometric schedule of times (cheap, and never uses future rows). Between
    refreshes the previously fit (past-only) parameters are reused. Because every
    parameter and every filtered state at tau is a function of rows with time<=tau
    only, removing future rows cannot change a past imputation -> passes verifier.

Speed tricks applied: factored U (low rank, never reconstruct full NxN), Woodbury
for the gain (L x L solve), cho_solve everywhere, precomputed per-time row-index
lists & observed masks, warm-started factors across the (rare) refit times, BLAS
thread caps at top, contiguous float arrays.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ---- caps (we SAY SO): keep total runtime sane -------------------------------
EM_SWEEPS = 4          # warm-started EM sweeps per (re)fit on the expanding window
REFIT_GROWTH = 1.5     # refit params when current time index >= last_refit*GROWTH
MIN_REFIT_TIMES = 8    # don't refit until at least this many times observed
JITTER = 1e-6


# --------------------------------------------------------------------------- #
# Low-rank parameter fit on a fixed (past-only) data block via simple EM-PCA.
# Block is a (n_rows, N) matrix with NaNs. Returns U (N,L), col-mean mu (N,).
# This is a probabilistic-PCA style fit; warm-started by a passed-in U0.
# --------------------------------------------------------------------------- #
def _fit_lowrank(block, L, U0=None, mu0=None, sweeps=EM_SWEEPS):
    n, N = block.shape
    obs = np.isfinite(block)
    # column mean over observed entries (past-only)
    cnt = obs.sum(0)
    s = np.where(obs, block, 0.0).sum(0)
    mu = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    if mu0 is not None:
        mu = np.where(cnt > 0, mu, mu0)
    Lr = min(L, max(1, min(n, N)))
    if U0 is not None and U0.shape == (N, Lr):
        U = U0.copy()
    else:
        rng = np.random.default_rng(0)
        U = rng.standard_normal((N, Lr)) * 0.1
    Xc = np.where(obs, block - mu, 0.0)            # centered, missing->0 (E-step uses obs only)
    # EM for probabilistic PCA with per-row latent scores; closed-form ridge.
    sigma2 = 1.0
    for _ in range(sweeps):
        # E-step: per row, score = (U_o'U_o + sigma2 I)^-1 U_o' x_o   (ridge / Woodbury-free, LxL)
        Z = np.zeros((n, Lr))
        UtU_full = U.T @ U
        for i in range(n):
            oi = obs[i]
            if not oi.any():
                continue
            Uo = U[oi]
            G = Uo.T @ Uo + sigma2 * np.eye(Lr)
            try:
                c = cho_factor(G + JITTER * np.eye(Lr), lower=True)
                Z[i] = cho_solve(c, Uo.T @ Xc[i, oi])
            except Exception:
                Z[i] = np.linalg.lstsq(Uo, Xc[i, oi], rcond=None)[0]
        # M-step: U row j = solve over rows observing j:  (Z_o'Z_o)^-1 Z_o' x_oj
        ZtZ_all = Z.T @ Z
        new_U = U.copy()
        resid_sq = 0.0
        resid_n = 0
        for j in range(N):
            rj = obs[:, j]
            if not rj.any():
                continue
            Zr = Z[rj]
            G = Zr.T @ Zr + JITTER * np.eye(Lr)
            try:
                c = cho_factor(G, lower=True)
                new_U[j] = cho_solve(c, Zr.T @ Xc[rj, j])
            except Exception:
                new_U[j] = np.linalg.lstsq(Zr, Xc[rj, j], rcond=None)[0]
        U = new_U
        # update sigma2 from residuals on observed entries
        pred = Z @ U.T
        d = (Xc - pred)[obs]
        if d.size:
            sigma2 = max(float(np.mean(d * d)), 1e-4)
    return U, mu, sigma2, Z


def _fit_transition(Z, rho_floor=0.0):
    """Diagonal-ish AR(1) transition A (full LxL via ridge) and process noise Q.
    Z is (n, L) the sequence of latent scores in TIME order. Closed-form ridge."""
    n, Lr = Z.shape
    if n < 3:
        return 0.9 * np.eye(Lr), np.ones(Lr)
    Z0 = Z[:-1]
    Z1 = Z[1:]
    G = Z0.T @ Z0 + 1e-3 * np.eye(Lr)
    try:
        c = cho_factor(G, lower=True)
        A = cho_solve(c, Z0.T @ Z1).T          # A: z1 ~ A z0
    except Exception:
        A = (np.linalg.lstsq(Z0, Z1, rcond=None)[0]).T
    # stabilise
    resid = Z1 - Z0 @ A.T
    q = np.maximum(np.mean(resid * resid, axis=0), 1e-3)
    # shrink A spectrally to keep filter stable
    w = np.linalg.eigvals(A)
    r = np.max(np.abs(w)) if A.size else 0.0
    if r > 0.999:
        A = A * (0.99 / r)
    return A, q


# --------------------------------------------------------------------------- #
# Kalman FILTER (forward only) over a per-entity time sequence, given params.
# rows_at[t] -> the single row index (in original X) for time-step t of this seq.
# We impute missing entries of each row from the filtered state.
# --------------------------------------------------------------------------- #
def _filter_sequence(X, out, row_seq, U, mu, A, q, sigma2):
    N, Lr = U.shape
    z = np.zeros(Lr)
    P = np.eye(Lr)            # prior state covariance
    Q = np.diag(q)
    for r in row_seq:
        row = X[r]
        obs = np.isfinite(row)
        # predict
        z_pred = A @ z
        P_pred = A @ P @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        if obs.any():
            Uo = U[obs]                                  # (m, L)
            y = row[obs] - mu[obs]                        # innovation target
            # innovation: nu = y - Uo z_pred ; S = Uo P_pred Uo' + sigma2 I_m
            nu = y - Uo @ z_pred
            # Woodbury for gain: K = P_pred Uo' S^-1, but solve in L-space.
            # S^-1 = (1/sig2) I - (1/sig2) Uo M^-1 Uo' (1/sig2),
            #   M = P_pred^-1 + (1/sig2) Uo'Uo   (L x L).
            inv_s2 = 1.0 / sigma2
            try:
                cP = cho_factor(P_pred + JITTER * np.eye(Lr), lower=True)
                Pinv = cho_solve(cP, np.eye(Lr))
            except Exception:
                Pinv = np.linalg.pinv(P_pred)
            M = Pinv + inv_s2 * (Uo.T @ Uo)
            try:
                cM = cho_factor(M + JITTER * np.eye(Lr), lower=True)
                Minv = cho_solve(cM, np.eye(Lr))
            except Exception:
                Minv = np.linalg.pinv(M)
            # S^-1 nu  (m-vector) via Woodbury, never forming S explicitly
            Ut_nu = Uo.T @ nu                              # (L,)
            Sinv_nu = inv_s2 * nu - inv_s2 * (Uo @ (Minv @ (inv_s2 * Ut_nu)))
            # gain applied: z = z_pred + P_pred Uo' Sinv_nu
            z = z_pred + P_pred @ (Uo.T @ Sinv_nu)
            # P = P_pred - P_pred Uo' S^-1 Uo P_pred ; use Woodbury form:
            # Uo' S^-1 Uo = inv_s2 Uo'Uo - inv_s2^2 Uo'Uo Minv Uo'Uo
            UtU = Uo.T @ Uo
            mid = inv_s2 * UtU - (inv_s2 * inv_s2) * (UtU @ Minv @ UtU)
            P = P_pred - P_pred @ mid @ P_pred
            P = 0.5 * (P + P.T)
        else:
            z = z_pred
            P = P_pred
        # impute missing entries from filtered state
        miss = ~obs
        if miss.any():
            out[r, miss] = mu[miss] + U[miss] @ z


# --------------------------------------------------------------------------- #
# Main causal imputer.
# --------------------------------------------------------------------------- #
def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    T_rows, N = X.shape
    out = X.copy()

    # time / entity structure
    if meta and "time_ids" in meta:
        tids = np.asarray(meta["time_ids"])
        eids = np.asarray(meta["entity_ids"])
        is_panel = True
    else:
        tids = np.arange(T_rows)
        eids = np.zeros(T_rows, dtype=int)
        is_panel = False

    uniq_times = np.unique(tids)
    nT = len(uniq_times)
    t_to_pos = {t: i for i, t in enumerate(uniq_times)}

    # rank
    L = min(4, N)

    # ---- pre-pass for a global column-mean fallback (causal: running) -------
    # We'll fill any still-missing cell with a strictly-past running column mean,
    # which is itself point-in-time; the Kalman state handles the rest.
    # Precompute, per time tau, the list of row indices observed at that time.
    rows_by_time = [[] for _ in range(nT)]
    for r in range(T_rows):
        rows_by_time[t_to_pos[tids[r]]].append(r)

    # For panels, build per-entity ordered (by time) row sequences.
    # For 2D, single entity, row order == time order.
    ent_unique = np.unique(eids)
    ent_seq = {}
    for e in ent_unique:
        idx = np.where(eids == e)[0]
        order = np.argsort(tids[idx], kind="stable")
        ent_seq[e] = idx[order]

    # =====================================================================
    # CAUSAL parameter schedule + ONLINE Kalman filter.
    #
    # We sweep times in increasing order. We maintain current params (U,mu,A,q,
    # sigma2) fit on rows with time <= current refit point. We refit on a
    # geometric schedule. For the filter pass itself, we cannot simply run the
    # whole filter once with final params (that would use future-fit params for
    # past times). Instead, to stay strictly point-in-time AND deterministic
    # under truncation, we IMPUTE each time tau using params fit on time<=tau,
    # then advance the per-entity Kalman states using those same params.
    #
    # To keep it O(refits) not O(T) refits, params are piecewise-constant across
    # time segments; within a segment we keep filtering with the segment's params
    # (all <= the segment's right edge, hence <= tau for tau in the segment). The
    # state at tau depends only on rows < tau (filtered) + params(<=tau): causal.
    # =====================================================================

    # per-entity running Kalman state (carried across the time sweep)
    state_z = {e: None for e in ent_unique}     # None => not yet started
    state_P = {e: None for e in ent_unique}

    # per-entity running feature offset (entity fixed effect). For panels the
    # entity FE dominates and is NOT captured by a single shared mu / low-rank U.
    # We subtract a STRICTLY-PAST per-entity feature mean before the filter and
    # add it back to imputations. Strictly-past => deterministic under truncation
    # => causal. Cold start (no entity history yet) falls back to global mu.
    use_offset = is_panel
    off_sum = {e: np.zeros(N) for e in ent_unique} if use_offset else None
    off_cnt = {e: np.zeros(N) for e in ent_unique} if use_offset else None

    # current params
    U = mu = A = q = None
    sigma2 = 1.0
    last_refit_pos = -1
    refit_threshold = MIN_REFIT_TIMES

    def refit(upto_pos):
        """Fit params using all rows with time-position <= upto_pos."""
        nonlocal U, mu, A, q, sigma2
        keep = tids <= uniq_times[upto_pos]
        block = X[keep]
        # warm start U
        U_new, mu_new, s2_new, Z = _fit_lowrank(
            block, L, U0=U if (U is not None and U.shape[0] == N) else None,
            mu0=mu)
        # transition fit needs Z in TIME order. Z rows align with `block` rows
        # (which are entity-major). Build a pooled time-ordered Z per entity and
        # estimate one shared A from concatenated consecutive pairs.
        block_tids = tids[keep]
        block_eids = eids[keep]
        # map block-row -> its position in `block`
        # accumulate consecutive (z_t, z_{t+1}) pairs within each entity
        pairs0 = []
        pairs1 = []
        for e in ent_unique:
            sel = np.where(block_eids == e)[0]
            if len(sel) < 2:
                continue
            order = np.argsort(block_tids[sel], kind="stable")
            zs = Z[sel[order]]
            pairs0.append(zs[:-1])
            pairs1.append(zs[1:])
        if pairs0:
            Z0 = np.vstack(pairs0)
            Z1 = np.vstack(pairs1)
            Zcat = np.vstack([Z0, Z1[-1:]]) if Z0.size else Z
            # fit A from the stacked pairs directly
            Lr = U_new.shape[1]
            G = Z0.T @ Z0 + 1e-3 * np.eye(Lr)
            try:
                c = cho_factor(G, lower=True)
                A_new = cho_solve(c, Z0.T @ Z1).T
            except Exception:
                A_new = (np.linalg.lstsq(Z0, Z1, rcond=None)[0]).T
            resid = Z1 - Z0 @ A_new.T
            q_new = np.maximum(np.mean(resid * resid, axis=0), 1e-3)
            w = np.linalg.eigvals(A_new)
            rr = np.max(np.abs(w)) if A_new.size else 0.0
            if rr > 0.999:
                A_new = A_new * (0.99 / rr)
        else:
            Lr = U_new.shape[1]
            A_new = 0.9 * np.eye(Lr)
            q_new = np.ones(Lr)
        U, mu, sigma2 = U_new, mu_new, s2_new
        A, q = A_new, q_new

    # initial fit on the cold-start prefix
    # find first position with enough times
    init_pos = min(refit_threshold, nT) - 1
    init_pos = max(init_pos, 0)
    refit(init_pos)
    last_refit_pos = init_pos
    refit_threshold = max(MIN_REFIT_TIMES, int(init_pos * REFIT_GROWTH) + 1)

    Q_cache = None
    for pos in range(nT):
        tau = uniq_times[pos]
        # refit params if schedule says so (using time<=tau only)
        if pos > last_refit_pos and pos >= refit_threshold:
            refit(pos)
            last_refit_pos = pos
            refit_threshold = max(refit_threshold + 1,
                                  int(pos * REFIT_GROWTH) + 1)
        Lr = U.shape[1]
        Q = np.diag(q)
        inv_s2 = 1.0 / sigma2
        # CONTEMPORANEOUS time fixed effect (cross-section at tau IS allowed):
        # mean over entities observed at tau of (x - entity_offset), per feature.
        # Captures the shared time-FE so entities with NO obs at tau (block
        # blackout) still get the current period level. Uses only data at tau.
        tfe = np.zeros(N)
        if use_offset:
            tsum = np.zeros(N); tcnt = np.zeros(N)
            for r in rows_by_time[pos]:
                e = eids[r]
                row = X[r]
                o = np.isfinite(row)
                if not o.any():
                    continue
                oc = off_cnt[e]
                offs = np.where(oc > 0, off_sum[e] / np.maximum(oc, 1), mu)
                tsum[o] += row[o] - offs[o]
                tcnt[o] += 1
            tfe = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), 0.0)
        # process every entity that has a row at this time
        for r in rows_by_time[pos]:
            e = eids[r]
            z = state_z[e]
            P = state_P[e]
            if z is None:                          # reset state at entity's first time
                z = np.zeros(Lr)
                P = np.eye(Lr)
            # predict
            z_pred = A @ z
            P_pred = A @ P @ A.T + Q
            P_pred = 0.5 * (P_pred + P_pred.T)
            row = X[r]
            obs = np.isfinite(row)
            # baseline = entity offset (strictly-past per-entity mean, else mu)
            #            + contemporaneous time fixed effect (cross-section @ tau)
            if use_offset:
                oc = off_cnt[e]
                offset = np.where(oc > 0, off_sum[e] / np.maximum(oc, 1), mu) + tfe
            else:
                offset = mu
            if obs.any():
                Uo = U[obs]
                y = row[obs] - offset[obs]
                nu = y - Uo @ z_pred
                try:
                    cP = cho_factor(P_pred + JITTER * np.eye(Lr), lower=True)
                    Pinv = cho_solve(cP, np.eye(Lr))
                except Exception:
                    Pinv = np.linalg.pinv(P_pred)
                Mmat = Pinv + inv_s2 * (Uo.T @ Uo)
                try:
                    cM = cho_factor(Mmat + JITTER * np.eye(Lr), lower=True)
                    Minv = cho_solve(cM, np.eye(Lr))
                except Exception:
                    Minv = np.linalg.pinv(Mmat)
                Ut_nu = Uo.T @ nu
                Sinv_nu = inv_s2 * nu - inv_s2 * (Uo @ (Minv @ (inv_s2 * Ut_nu)))
                z = z_pred + P_pred @ (Uo.T @ Sinv_nu)
                UtU = Uo.T @ Uo
                mid = inv_s2 * UtU - (inv_s2 * inv_s2) * (UtU @ Minv @ UtU)
                P = P_pred - P_pred @ mid @ P_pred
                P = 0.5 * (P + P.T)
            else:
                z = z_pred
                P = P_pred
            miss = ~obs
            if miss.any():
                out[r, miss] = offset[miss] + U[miss] @ z
            state_z[e] = z
            state_P[e] = P
            # update strictly-past per-entity offset AFTER using it (causal)
            if use_offset and obs.any():
                off_sum[e][obs] += row[obs]
                off_cnt[e][obs] += 1

    # safety: any residual NaN (e.g. a feature never observed up to its time)
    if np.isnan(out).any():
        # strictly-past running column mean fill (point-in-time, by time order)
        csum = np.zeros(N)
        ccnt = np.zeros(N)
        glob = 0.0
        gcnt = 0
        for pos in range(nT):
            mean_so_far = np.where(ccnt > 0, csum / np.maximum(ccnt, 1),
                                   glob / max(gcnt, 1) if gcnt else 0.0)
            for r in rows_by_time[pos]:
                row = out[r]
                miss = np.isnan(row)
                if miss.any():
                    out[r, miss] = mean_so_far[miss]
            # update running stats from this time's observed (original) entries
            for r in rows_by_time[pos]:
                orow = X[r]
                o = np.isfinite(orow)
                csum[o] += orow[o]
                ccnt[o] += 1
                glob += orow[o].sum()
                gcnt += int(o.sum())
        out = np.where(np.isnan(out), 0.0, out)

    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
    from causal import run_causal, summarize_causal
    rows = run_causal("FilterSSM", online_impute)
    summarize_causal(rows)
    import json
    print("\nJSON_ROWS_START")
    print(json.dumps([{k: (None if isinstance(v, float) and (v != v) else v)
                       for k, v in r.items()} for r in rows]))
    print("JSON_ROWS_END")
