"""
ONLINE / CAUSAL competitor stand-ins for TIMARA.

WHY THIS FILE EXISTS
--------------------
The causal column of the benchmark must not be just mean / LOCF / EWMA
(see c_baselines.py). Reviewers comparing a *causal, CPU-only* imputer will
immediately ask how it stacks up against the published *online / streaming*
literature: GROUSE (subspace tracking from partial observations), online
Gaussian-copula / online Gaussian models (Zhao & Udell, AAAI'22), and
streaming Bayesian models such as BayOTIDE (ICML'24). Those published systems
are heavier; here we provide faithful, lightweight *reimplementations of the
public ALGORITHMS* (allowed) as honest causal stand-ins so the causal column
has non-trivial, recognizably-named competitors.

THREE METHODS (all pure numpy, all strictly CAUSAL / forward-only):

  (1) online_ewcov     -- Online Gaussian conditional-mean imputation. Tracks an
                          exponentially-weighted mean and covariance of the
                          feature cross-section from PAST fully/partially observed
                          rows; for a missing cell it uses the Gaussian conditional
                          expectation E[x_miss | x_obs] under that running model.
                          (A streaming, EW-decayed cousin of the online-Gaussian /
                          online-copula family, minus the marginal copula transform.)

  (2) grouse_lite      -- GROUSE-style online low-rank subspace tracking from
                          PARTIAL observations (Balzano, Nowak & Recht, 2010).
                          Maintains an orthonormal basis U (T x r is the data; U is
                          N x r over features) updated by one incremental-gradient
                          step per time row using only that row's observed entries,
                          then imputes the missing entries from the rank-r fit of
                          the observed ones. Subspace at time tau depends only on
                          rows <= tau.

  (3) online_mean_var  -- Trivial but honest causal floor: expanding (EW) per-
                          feature mean; imputes a missing cell with the running mean
                          of that feature over rows seen so far. Serves as the
                          "did the fancy methods actually buy anything" anchor in
                          the online column.

GOVERNING CAUSAL CONSTRAINT (verified, not asserted)
----------------------------------------------------
An imputed value for (row r at time tau, feature i) uses ONLY information at time
<= tau: the contemporaneous cross-section at tau (other observed features in the
same row, and other entities sharing time tau) and the running state estimated
from rows at times STRICTLY < tau. We iterate by TIME group (all rows sharing a
time_id together), impute the whole tau-slice FIRST using state built from times
< tau, and only THEN fold the tau observations into the running state. This makes
the imputation at tau invariant to the removal of future rows -> passes
bench/causal.py assert_causal. Panels (entity-major stacked rows) are handled via
meta['time_ids'] so a later entity's future never leaks into an earlier read.

Robustness: every method sanitizes non-finite inputs (NaN = missing; +-Inf are
treated as missing, never propagated), guards empty / all-missing / constant /
single-row / huge / tiny inputs, and always returns a finite (T, N) array.

These pass bench/causal.py assert_causal and bench/robustness.py (see __main__).
"""
from __future__ import annotations
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _time_groups(meta, T):
    """Return ordered list of row-index arrays, one per unique time, ascending.

    For 2D data (no meta) every row is its own time. For panels we group by
    meta['time_ids'] so all rows sharing a time are imputed together as one
    contemporaneous cross-section.
    """
    if meta and "time_ids" in meta:
        tid = np.asarray(meta["time_ids"])
    else:
        tid = np.arange(T)
    order = np.argsort(tid, kind="stable")
    sorted_tid = tid[order]
    rows_per_time = []
    pos = 0
    for ut in np.unique(tid):
        cnt = int(np.count_nonzero(sorted_tid == ut))
        rows_per_time.append(order[pos:pos + cnt])
        pos += cnt
    return rows_per_time


def _sanitize(X):
    """Coerce to float64 (T,N); map +-Inf to NaN (treated as missing).

    Returns (Xclean, miss_mask) where miss_mask marks cells we must fill.
    """
    X = np.array(X, dtype=float, copy=True)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    bad = ~np.isfinite(X)          # NaN OR +-Inf
    X[bad] = np.nan
    return X, bad


def _col_nanmean(block):
    """Per-column mean over non-NaN cells; NaN for all-NaN columns. No warnings."""
    obs = ~np.isnan(block)
    cnt = obs.sum(axis=0)
    s = np.where(obs, block, 0.0).sum(axis=0)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _finalize(out, fallback=0.0):
    """Replace any residual non-finite cell with a finite fallback value."""
    if not np.all(np.isfinite(out)):
        col_mean = np.nanmean(np.where(np.isfinite(out), out, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, fallback)
        bad = ~np.isfinite(out)
        # broadcast per-column fallback
        out = np.where(bad, col_mean[None, :], out)
        out = np.where(np.isfinite(out), out, fallback)
    return out


# --------------------------------------------------------------------------- #
# (3) online_mean_var -- expanding / EW per-feature mean (causal floor)
# --------------------------------------------------------------------------- #
def make_online_mean_var(halflife=20.0):
    """Factory for an EW running-mean causal imputer (halflife in time steps).

    Missing cell (tau, i) <- EW mean of feature i over OBSERVED rows at times < tau.
    Leading gap (feature never seen yet) -> contemporaneous cross-row mean at tau,
    else 0.
    """
    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1e-6))

    def online_mean_var(X, meta=None):
        X, _ = _sanitize(X)
        T, N = X.shape
        out = X.copy()
        if T == 0 or N == 0:
            return _finalize(out)
        ew_num = np.zeros(N)
        ew_den = np.zeros(N)        # 0 => feature unseen so far
        for rows in _time_groups(meta, T):
            have = ew_den > 0
            running = np.where(have, ew_num / np.maximum(ew_den, 1e-300), np.nan)
            block = X[rows]
            # contemporaneous cross-row mean at tau, for leading gaps only
            contemp = _col_nanmean(block)
            for r in rows:
                miss = np.isnan(X[r])
                if miss.any():
                    fill = np.where(np.isfinite(running), running, contemp)
                    fill = np.where(np.isfinite(fill), fill, 0.0)
                    out[r, miss] = fill[miss]
            # fold tau observations into EW state (AFTER imputing tau -> causal)
            obs_blk = ~np.isnan(block)
            if obs_blk.any():
                col_sum = np.where(obs_blk, block, 0.0).sum(axis=0)
                col_cnt = obs_blk.sum(axis=0)
                seen = col_cnt > 0
                slice_mean = np.where(seen, col_sum / np.maximum(col_cnt, 1), 0.0)
                # one decayed update per time slice using the slice mean
                ew_num[seen] = (1 - alpha) * ew_num[seen] + alpha * slice_mean[seen]
                ew_den[seen] = (1 - alpha) * ew_den[seen] + alpha
        return _finalize(out)

    return online_mean_var


online_mean_var = make_online_mean_var(halflife=20.0)


# --------------------------------------------------------------------------- #
# (1) online_ewcov -- online Gaussian conditional-mean imputation
# --------------------------------------------------------------------------- #
def make_online_ewcov(halflife=40.0, ridge=1e-2):
    """Factory for streaming EW Gaussian conditional-mean imputation.

    State (all from rows at times < tau):
        mu  : EW running mean over features         (N,)
        S   : EW running second-moment (uncentered) (N, N)
        w   : EW running weight (sum of decays)     scalar
    Covariance C = S/w - mu mu^T (regularized). For a row with observed set O and
    missing set Mi at time tau, impute
        x_Mi = mu_Mi + C[Mi,O] (C[O,O] + lam I)^-1 (x_O - mu_O).
    The contemporaneous observed entries x_O are at time tau (allowed). All
    PARAMETERS (mu, C) are from times < tau, so removing future rows cannot change
    the tau imputation -> causal.
    """
    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1e-6))

    def online_ewcov(X, meta=None):
        X, _ = _sanitize(X)
        T, N = X.shape
        out = X.copy()
        if T == 0 or N == 0:
            return _finalize(out)

        mu = np.zeros(N)
        S = np.zeros((N, N))
        w = 0.0
        # per-feature EW mean fallback for leading gaps (feature never seen)
        f_num = np.zeros(N)
        f_den = np.zeros(N)

        for rows in _time_groups(meta, T):
            block = X[rows]
            obs_blk = ~np.isnan(block)
            seen_now = obs_blk.any(axis=0)
            contemp = _col_nanmean(block)

            # current model from STRICTLY PAST rows
            if w > 1e-12:
                cov = S / w - np.outer(mu, mu)
            else:
                cov = np.zeros((N, N))
            diag = np.diag(cov).copy()
            diag = np.where(diag > 0, diag, 0.0)
            # adaptive ridge scaled to feature variance for conditioning
            lam = ridge * (diag.mean() + 1e-9) + 1e-9

            have_mu = f_den > 0
            mu_run = np.where(have_mu, f_num / np.maximum(f_den, 1e-300), np.nan)

            for r in rows:
                row = X[r]
                miss = np.isnan(row)
                if not miss.any():
                    continue
                obs = ~miss
                # baseline guess = running mean (or contemp / 0 fallback)
                guess = np.where(have_mu, mu_run, contemp)
                guess = np.where(np.isfinite(guess), guess, 0.0)
                out_row = guess.copy()
                # Gaussian conditional correction only if we have a model AND
                # at least one observed predictor in this row.
                if w > 1e-9 and obs.any():
                    Coo = cov[np.ix_(obs, obs)] + lam * np.eye(int(obs.sum()))
                    Cmo = cov[np.ix_(miss, obs)]
                    resid = row[obs] - mu[obs]
                    try:
                        sol = np.linalg.solve(Coo, resid)
                        corr = Cmo @ sol
                        out_row[miss] = mu[miss] + corr
                    except np.linalg.LinAlgError:
                        pass   # keep baseline guess
                # numeric guard
                out_row = np.where(np.isfinite(out_row), out_row, guess)
                out[r, miss] = out_row[miss]

            # ---- fold tau observations into EW state (AFTER imputing tau) ----
            # Use a complete-case proxy for the slice: per-feature observed mean,
            # which lets us update mu / second-moment without leaking imputed values.
            col_cnt = obs_blk.sum(axis=0)
            col_sum = np.where(obs_blk, block, 0.0).sum(axis=0)
            slice_mean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1),
                                  np.where(have_mu, mu_run, mu))
            slice_mean = np.where(np.isfinite(slice_mean), slice_mean, 0.0)
            # update second moment from the per-row observed pairs (pairwise EW).
            # For causal correctness we only need PAST<tau in the model used AT tau,
            # which we already enforced; here we just accumulate this slice.
            for r in rows:
                row = X[r]
                obs = ~np.isnan(row)
                if not obs.any():
                    continue
                xr = np.where(obs, row, slice_mean)          # fill gaps w/ slice mean
                mu[:] = (1 - alpha) * mu + alpha * xr
                S[:, :] = (1 - alpha) * S + alpha * np.outer(xr, xr)
                w = (1 - alpha) * w + alpha
            upd = seen_now
            if upd.any():
                f_num[upd] = (1 - alpha) * f_num[upd] + alpha * slice_mean[upd]
                f_den[upd] = (1 - alpha) * f_den[upd] + alpha

        return _finalize(out)

    return online_ewcov


online_ewcov = make_online_ewcov(halflife=40.0, ridge=1e-2)


# --------------------------------------------------------------------------- #
# (2) grouse_lite -- GROUSE-style online subspace tracking, partial obs
# --------------------------------------------------------------------------- #
def make_grouse_lite(rank=5, step=0.5, max_step=2.0):
    """Factory for GROUSE-style online low-rank subspace tracking.

    Balzano, Nowak & Recht (2010): maintain an orthonormal N x r basis U of the
    feature subspace. At each time row x (with observed index set O):
        1. impute missing entries from the rank-r fit of the OBSERVED entries:
              w = argmin_w || U_O w - x_O ||  ;  x_hat = U w
        2. take ONE incremental gradient step rotating U toward the new direction.
    Update happens AFTER imputation; the basis used to impute row tau is built only
    from rows < tau -> causal. We process panels time-slice by time-slice (each row
    in the slice imputed against the SAME past basis, then all folded in).
    """
    def grouse_lite(X, meta=None):
        X, _ = _sanitize(X)
        T, N = X.shape
        out = X.copy()
        if T == 0 or N == 0:
            return _finalize(out)
        r = int(min(max(1, rank), N))

        # init orthonormal basis (deterministic seed -> verifier reproducibility)
        rng = np.random.default_rng(0)
        U, _ = np.linalg.qr(rng.standard_normal((N, r)))

        # running per-feature mean (centering) and fallback, from past rows only
        m_num = np.zeros(N)
        m_den = np.zeros(N)
        initialized = False

        for rows in _time_groups(meta, T):
            block = X[rows]
            obs_blk = ~np.isnan(block)
            have_mu = m_den > 0
            mu = np.where(have_mu, m_num / np.maximum(m_den, 1e-300), 0.0)
            contemp = _col_nanmean(block)

            # --- impute every row in this slice against the PAST basis U, mu ---
            for r_idx in rows:
                row = X[r_idx]
                miss = np.isnan(row)
                if not miss.any():
                    continue
                obs = ~miss
                guess = np.where(have_mu, mu, contemp)
                guess = np.where(np.isfinite(guess), guess, 0.0)
                fill = guess.copy()
                if initialized and obs.any():
                    Uo = U[obs]                 # (|O|, r)
                    xo = row[obs] - mu[obs]      # centered observed
                    # least-squares weights from observed sub-rows
                    A = Uo.T @ Uo + 1e-6 * np.eye(r)
                    try:
                        wv = np.linalg.solve(A, Uo.T @ xo)
                        recon = U @ wv + mu      # full reconstruction
                        fill = np.where(np.isfinite(recon), recon, guess)
                    except np.linalg.LinAlgError:
                        pass
                out[r_idx, miss] = fill[miss]

            # --- update basis & mean from this slice's OBSERVED entries (post-tau)
            for r_idx in rows:
                row = X[r_idx]
                obs = ~np.isnan(row)
                if not obs.any():
                    continue
                mu_now = np.where(m_den > 0, m_num / np.maximum(m_den, 1e-300), 0.0)
                xc = np.zeros(N)
                xc[obs] = row[obs] - mu_now[obs]      # centered, missing as 0
                Uo = U[obs]
                A = Uo.T @ Uo + 1e-6 * np.eye(r)
                try:
                    wv = np.linalg.solve(A, Uo.T @ xc[obs])
                except np.linalg.LinAlgError:
                    wv = np.zeros(r)
                p = U @ wv                            # predicted full vector
                # residual on OBSERVED coords only (partial-observation GROUSE)
                res = np.zeros(N)
                res[obs] = xc[obs] - p[obs]
                rn = np.linalg.norm(res)
                pn = np.linalg.norm(wv)
                if rn > 1e-12 and pn > 1e-12:
                    # GROUSE rank-1 rotation: U <- U + (gradient) step
                    sigma = rn * pn
                    eta = min(step * sigma / (sigma + 1e-9), max_step)
                    U = U + (eta * (res / rn)[:, None]) @ (wv / pn)[None, :]
                    # re-orthonormalize to keep U a valid basis (cheap, r small)
                    U, _ = np.linalg.qr(U)
                # update running mean
                m_num[obs] = m_num[obs] + (row[obs])
                m_den[obs] = m_den[obs] + 1.0
                initialized = True

        return _finalize(out)

    return grouse_lite


grouse_lite = make_grouse_lite(rank=5, step=0.5)


# --------------------------------------------------------------------------- #
# Registry + drivers (mirror c_baselines.py)
# --------------------------------------------------------------------------- #
ONLINE_BASELINES = {
    "OnlineEWCov": online_ewcov,
    "GrouseLite": grouse_lite,
    "OnlineMeanVar": online_mean_var,
}


def run_all():
    """Run every online baseline through the causal evaluator (verify=True)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from causal import run_causal  # noqa: E402

    all_rows = []
    for base, fn in ONLINE_BASELINES.items():
        rows = run_causal("OnlineBaselines", fn, verify=True)
        for r in rows:
            r = dict(r)
            r["dataset"] = f"{base}:{r['dataset']}"
            all_rows.append(r)
    return all_rows


# --------------------------------------------------------------------------- #
# __main__ smoke: tiny synthetic + causal verify + robustness battery
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)

    # ---- 1. quick functional smoke on a tiny synthetic low-rank matrix ----
    print("=" * 70)
    print("SMOKE: tiny low-rank 2D matrix")
    print("=" * 70)
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((60, 3))
    L = rng.standard_normal((6, 3))
    Xtrue = Z @ L.T
    M = rng.random(Xtrue.shape) < 0.25
    Xobs = Xtrue.copy()
    Xobs[M] = np.nan
    for name, fn in ONLINE_BASELINES.items():
        pred = fn(Xobs.copy(), {})
        ok_shape = pred.shape == Xtrue.shape
        ok_fin = bool(np.all(np.isfinite(pred)))
        mae = float(np.mean(np.abs(pred[M] - Xtrue[M])))
        print(f"  {name:14s} shape_ok={ok_shape} finite={ok_fin} MAE={mae:.4f}")

    # ---- 2. causal verification on all datasets via causal.assert_causal ----
    print("\n" + "=" * 70)
    print("CAUSAL VERIFY (assert_causal on every dataset)")
    print("=" * 70)
    from causal import run_causal, summarize_causal  # noqa: E402
    all_rows = []
    n_causal = n_total = 0
    for base, fn in ONLINE_BASELINES.items():
        rows = run_causal("OnlineBaselines", fn, verify=True)
        for r in rows:
            n_total += 1
            n_causal += int(bool(r["causal"]))
            rr = dict(r)
            rr["dataset"] = f"{base}:{rr['dataset']}"
            all_rows.append(rr)
    summarize_causal(all_rows)
    print(f"\nCAUSAL: {n_causal}/{n_total} rows verified causal")

    # ---- 3. robustness battery via robustness.run ----
    print("\n" + "=" * 70)
    print("ROBUSTNESS BATTERY (edge-case invariants)")
    print("=" * 70)
    import robustness  # noqa: E402
    rob_summary = {}
    for name, fn in ONLINE_BASELINES.items():
        print(f"\n----- {name} -----")
        npass, ntot, fails = robustness.run(fn, name)
        rob_summary[name] = (npass, ntot, fails)

    # ---- final verdict ----
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"causal: {n_causal}/{n_total} rows")
    for name, (npass, ntot, fails) in rob_summary.items():
        print(f"robust {name:14s}: {npass}/{ntot}"
              + (f"  FAILS={[f[0] for f in fails]}" if fails else ""))
    all_robust = all(p == t for p, t, _ in rob_summary.values())
    print(f"\nALL CAUSAL: {n_causal == n_total}   ALL ROBUST: {all_robust}")
