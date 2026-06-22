"""
SoftImpute-ALS (Hastie, Mazumder, Lee, Zadeh 2015) implemented from scratch in numpy.

Nuclear-norm regularized matrix completion solved by alternating least squares on a
low-rank factorization X ~ A B^T with a ridge (lambda) penalty that is exactly
equivalent to soft-thresholding the SVD by lambda. We:

  1. Standardize columns (store mean/std), restore at the end.
  2. Initialize missing entries with column means (zero after standardization).
  3. ALS sweeps: given the current low-rank model, "fill" the matrix as
        Xfilled = Pobs(Xobs) + Pmiss(A B^T)
     then update A (ridge regression of Xfilled on B), then B similarly, while
     maintaining the SVD form so we can soft-threshold singular values by lambda.
  4. Iterate until relative change of the filled matrix < tol or max_iter.

lambda is chosen by a tiny internal scan over a few multiples of an automatic
scale (a fraction of the top singular value of the mean-filled matrix), picking
the lambda with best reconstruction on a small held-out subset of OBSERVED cells.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from harness import run_method, summarize


def _standardize(X):
    """Column standardize ignoring NaNs. Returns Xs, mean, std."""
    mean = np.nanmean(X, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(X, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    Xs = (X - mean) / std
    return Xs, mean, std


def _softimpute_als(Xs, mask_obs, rank, lam, max_iter=100, tol=1e-4, rng=None):
    """
    Core SoftImpute-ALS on standardized matrix Xs (NaNs already present).
    mask_obs: boolean (T,N) True where observed.
    rank: max rank cap.  lam: nuclear-norm penalty.
    Returns reconstructed (T,N) low-rank matrix (dense, no NaN).
    """
    T, N = Xs.shape
    rng = rng or np.random.default_rng(0)

    # working filled matrix: observed at true values, missing at 0 (col mean already 0)
    Xfill = np.where(mask_obs, Xs, 0.0)

    r = min(rank, T, N)
    # init U (T,r) orthonormal, D (r,), V (N,r) orthonormal via random + SVD of small
    U = rng.standard_normal((T, r))
    U, _ = np.linalg.qr(U)
    D = np.ones(r)
    V = np.zeros((N, r))

    prev = Xfill.copy()
    lam2 = lam ** 2

    for it in range(max_iter):
        # --- B-step: solve for V given A = U*D ---
        A = U * D                                   # (T,r)
        # ridge: V = (A^T A + lam I)^-1 A^T Xfill  but use sufficient stats
        AtA = A.T @ A                               # (r,r)
        rhs = A.T @ Xfill                           # (r,N)
        Bt = np.linalg.solve(AtA + lam * np.eye(r), rhs)   # (r,N)
        B = Bt.T                                     # (N,r)
        # SVD-renormalize B: B = V * Dv ; keep V orthonormal
        Vv, dv, _ = np.linalg.svd(B, full_matrices=False)
        V = Vv
        # update filled missing entries with current model A B^T
        model = A @ B.T
        Xfill = np.where(mask_obs, Xs, model)

        # --- A-step: solve for U given B-part ---
        Bmat = V * dv                               # (N,r)
        BtB = Bmat.T @ Bmat
        rhs2 = Xfill @ Bmat                          # (T,r)
        Amat = np.linalg.solve(BtB + lam * np.eye(r), rhs2.T).T   # (T,r)
        Uu, du, _ = np.linalg.svd(Amat, full_matrices=False)
        U = Uu
        D = du
        model = (U * D) @ V.T
        Xfill = np.where(mask_obs, Xs, model)

        # convergence on filled matrix
        num = np.linalg.norm(Xfill - prev)
        den = np.linalg.norm(prev) + 1e-9
        prev = Xfill.copy()
        if num / den < tol:
            break

    # final reconstruction: soft-threshold singular values by lam and rebuild
    # one clean SVD of the filled matrix, shrink, reconstruct
    Uf, sf, Vtf = np.linalg.svd(Xfill, full_matrices=False)
    sf_shrunk = np.maximum(sf - lam, 0.0)
    keep = sf_shrunk > 0
    if not np.any(keep):
        keep[:1] = True
        sf_shrunk = sf_shrunk.copy()
        sf_shrunk[0] = max(sf_shrunk[0], 1e-6)
    recon = (Uf[:, keep] * sf_shrunk[keep]) @ Vtf[keep]
    return recon


def _impute_core(X, rank_cap=30, max_iter=80, tol=1e-4):
    Xs, mean, std = _standardize(X)
    mask_obs = np.isfinite(Xs)
    Xs = np.where(mask_obs, Xs, 0.0)
    T, N = Xs.shape
    rank = min(rank_cap, T, N)
    rng = np.random.default_rng(12345)

    # automatic lambda scale: fraction of top singular value of mean-filled matrix
    try:
        s0 = np.linalg.svd(Xs, compute_uv=False)
        smax = float(s0[0])
    except Exception:
        smax = float(np.linalg.norm(Xs))
    base = smax / np.sqrt(max(T, N))

    # tiny internal scan over lambda multiples using a held-out subset of observed cells
    obs_idx = np.argwhere(mask_obs)
    n_obs = len(obs_idx)
    n_val = min(2000, max(50, n_obs // 20))
    sel = rng.choice(n_obs, size=n_val, replace=False)
    val_pos = obs_idx[sel]
    val_true = Xs[val_pos[:, 0], val_pos[:, 1]].copy()

    mask_scan = mask_obs.copy()
    mask_scan[val_pos[:, 0], val_pos[:, 1]] = False
    Xs_scan = np.where(mask_scan, Xs, 0.0)

    lam_candidates = [base * m for m in (0.005, 0.02, 0.05, 0.15, 0.4, 1.0)]
    best_lam, best_err = lam_candidates[0], np.inf
    for lam in lam_candidates:
        recon = _softimpute_als(Xs_scan, mask_scan, rank, lam,
                                max_iter=40, tol=1e-3, rng=np.random.default_rng(7))
        pred = recon[val_pos[:, 0], val_pos[:, 1]]
        err = np.mean((pred - val_true) ** 2)
        if err < best_err:
            best_err, best_lam = err, lam

    # final fit on full observed set with chosen lambda
    recon = _softimpute_als(Xs, mask_obs, rank, best_lam,
                            max_iter=max_iter, tol=tol, rng=rng)
    out = recon * std + mean
    # keep observed entries exact
    out = np.where(mask_obs, X, out)
    return out


def impute(X, meta):
    """Generic 2D imputer. For panel data we exploit entity/time structure lightly
    by just running the same low-rank completion on the stacked (E*T, F) matrix,
    which already captures entity/time effects through the low-rank factors."""
    return _impute_core(X, rank_cap=30, max_iter=80, tol=1e-4)


if __name__ == "__main__":
    rows = run_method("SoftImpute-ALS", impute)
    summarize(rows)
    print()
    import json
    print(json.dumps(rows, default=float))
