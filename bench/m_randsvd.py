import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np
from sklearn.utils.extmath import randomized_svd
from harness import run_method, summarize, DATASETS


# --------------------------------------------------------------------------- #
# Common preprocessing: column-standardize, fill missing with 0 (= col mean)
# --------------------------------------------------------------------------- #
def _prep(X):
    M = np.isnan(X)
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Xs = (X - mu) / sd
    Xs[M] = 0.0
    return Xs, M, mu, sd


def _restore(Z, mu, sd):
    return Z * sd + mu


# --------------------------------------------------------------------------- #
# SoftImpute-ALS, full SVD on the dense filled matrix each iteration
# --------------------------------------------------------------------------- #
def softimpute_full(X, meta, rank=None, lam=None, max_iter=100, tol=1e-4):
    Xs, M, mu, sd = _prep(X)
    T, N = Xs.shape
    obs = ~M
    if rank is None:
        rank = min(T, N, 20)
    Z = Xs.copy()  # current estimate (standardized)
    prev = None
    if lam is None:
        # set lambda as a fraction of the top singular value of the filled matrix
        s0 = np.linalg.svd(Z, compute_uv=False)[0]
        lam = 0.02 * s0
    for it in range(max_iter):
        # full SVD of the completed matrix
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        s_th = np.maximum(s[:rank] - lam, 0.0)
        Zhat = (U[:, :rank] * s_th) @ Vt[:rank]
        # plug observed values back, keep low-rank estimate on missing
        Znew = np.where(obs, Xs, Zhat)
        if prev is not None:
            num = np.linalg.norm(Znew - prev)
            den = np.linalg.norm(prev) + 1e-12
            if num / den < tol:
                Z = Znew
                break
        prev = Znew
        Z = Znew
    out = X.copy()
    filled = _restore(Z, mu, sd)
    out[M] = filled[M]
    return out


# --------------------------------------------------------------------------- #
# SoftImpute with randomized_svd (fixed rank) + warm-started factors.
# We warm-start the randomized_svd via the v0/transpose flow by passing the
# previous right factors as the initial subspace is not directly supported by
# sklearn, so we warm-start the COMPLETION (Z) which is the dominant cost driver
# and additionally reuse a small power-iteration count. The key speed lever is
# truncated rank-r randomized SVD instead of full SVD.
# --------------------------------------------------------------------------- #
def softimpute_rand(X, meta, rank=None, lam=None, max_iter=100, tol=1e-4,
                    n_oversamples=10, n_power=4):
    Xs, M, mu, sd = _prep(X)
    T, N = Xs.shape
    obs = ~M
    if rank is None:
        rank = min(T, N, 20)
    rank = min(rank, min(T, N))
    Z = Xs.copy()
    prev = None
    rng = np.random.RandomState(0)
    if lam is None:
        # cheap estimate of top singular value via a few power iterations
        v = rng.standard_normal(N)
        for _ in range(5):
            v = Z.T @ (Z @ v)
            v /= (np.linalg.norm(v) + 1e-12)
        s0 = np.linalg.norm(Z @ v)
        lam = 0.02 * s0
    for it in range(max_iter):
        U, s, Vt = randomized_svd(
            Z, n_components=rank, n_oversamples=n_oversamples,
            n_iter=n_power, power_iteration_normalizer="QR",
            random_state=0,
        )
        s_th = np.maximum(s - lam, 0.0)
        Zhat = (U * s_th) @ Vt
        Znew = np.where(obs, Xs, Zhat)
        if prev is not None:
            num = np.linalg.norm(Znew - prev)
            den = np.linalg.norm(prev) + 1e-12
            if num / den < tol:
                Z = Znew
                break
        prev = Znew
        Z = Znew
    out = X.copy()
    filled = _restore(Z, mu, sd)
    out[M] = filled[M]
    return out


if __name__ == "__main__":
    # Matched hyperparameters between the two variants for a fair comparison.
    RANK = 20
    MAXIT = 100
    TOL = 1e-4

    def f_full(X, meta):
        return softimpute_full(X, meta, rank=RANK, max_iter=MAXIT, tol=TOL)

    def f_rand(X, meta):
        return softimpute_rand(X, meta, rank=RANK, max_iter=MAXIT, tol=TOL)

    rows_rand = run_method("SoftImpute-RandSVD", f_rand)

    # Full-SVD rows: relabel dataset field with FullSVD: prefix for disambiguation
    full_rows_raw = run_method("SoftImpute-FullSVD", f_full)
    rows_full = []
    for r in full_rows_raw:
        r2 = dict(r)
        r2["dataset"] = "FullSVD:" + r["dataset"]
        rows_full.append(r2)

    all_rows = rows_rand + rows_full
    summarize(all_rows)

    # Print a speed/accuracy comparison summary
    print("\n=== Speedup (FullSVD time / RandSVD time) and accuracy delta ===")
    rmap = {r["dataset"]: r for r in rows_rand}
    for fr in full_rows_raw:
        ds = fr["dataset"]
        rr = rmap[ds]
        if fr["time_s"] > 0 and rr["time_s"] > 0:
            speedup = fr["time_s"] / rr["time_s"]
        else:
            speedup = float("nan")
        d_rmse = rr["rmse"] - fr["rmse"]
        print(f"{ds:20s} shape={fr['shape']:9s} "
              f"full={fr['time_s']:.4f}s rand={rr['time_s']:.4f}s "
              f"speedup={speedup:5.2f}x  rmse full={fr['rmse']:.4f} rand={rr['rmse']:.4f} "
              f"(drand-dfull={d_rmse:+.4f})")
