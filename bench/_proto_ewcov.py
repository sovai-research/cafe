import os
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"2")
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def ewcov_impute(X, meta, halflife=200.0, ridge=1e-2, refit_every=1):
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    lam = 0.5 ** (1.0 / halflife)   # EW decay
    # EW accumulators (causal: only rows < t fold in before imputing t... but we
    # use rows <= t-1; contemporaneous obs of row t are the conditioning set)
    w = 0.0
    mu = np.zeros(N)
    M2 = np.zeros((N, N))   # EW sum of outer products (uncentered)
    Sx = np.zeros(N)        # EW sum of x
    col_sum = np.zeros(N); col_cnt = np.zeros(N)
    have_cov = False
    for t in range(T):
        row = X[t]
        obs = ~np.isnan(row)
        miss = ~obs
        if miss.any():
            # global causal mean fallback
            gmean = np.where(col_cnt > 0, col_sum/np.maximum(col_cnt,1), 0.0)
            filled = gmean.copy()
            if have_cov and obs.any() and w > 1e-6:
                mean = Sx / w
                cov = M2 / w - np.outer(mean, mean)
                o = np.where(obs)[0]; m = np.where(miss)[0]
                Soo = cov[np.ix_(o,o)] + ridge*np.eye(o.size)
                Smo = cov[np.ix_(m,o)]
                xo = row[o] - mean[o]
                try:
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[m] = mean[m] + Smo @ sol
                except Exception:
                    filled[m] = mean[m]
            out[t, miss] = filled[miss]
        # update EW stats with row t's OBSERVED entries.
        # For missing entries, use the just-imputed value (self-consistent) so cov stays full-dim.
        full = out[t].copy()
        w = lam*w + 1.0
        Sx = lam*Sx + full
        M2 = lam*M2 + np.outer(full, full)
        col_sum[obs] += row[obs]; col_cnt[obs] += 1
        if w > 5: have_cov = True
    if np.isnan(out).any():
        gm = np.where(col_cnt>0, col_sum/np.maximum(col_cnt,1), 0.0)
        idx = np.where(np.isnan(out)); out[idx] = np.take(gm, idx[1])
    return out

if __name__ == "__main__":
    import sys, time
    sys.path.insert(0,"/Users/dereksnow/Sovai/Github/TIMARA/bench")
    import arena, numpy as np
    from harness import MASKERS, metrics
    suite = arena.SUITE
    for name in ("real_beijing_mcar10","real_ETTh1_mcar10","2d_highrank_mcar30","2d_heavytail_mcar30","2d_large_mcar30","2d_small_mcar20","real_temp_block10","real_chlorine_block10"):
        clean, meta, mech, rate, grp = suite[name]
        masker = arena._real_block_mask if mech=="block" and grp=="real-vldb" else MASKERS[mech]
        M = masker(clean, rate, 0) if mech!="block" or grp=="real-vldb" else MASKERS["block"](clean,rate,0)
        Xo = clean.copy(); Xo[M]=np.nan
        t0=time.perf_counter()
        pred = ewcov_impute(Xo.copy(), dict(meta))
        dt=time.perf_counter()-t0
        mt = metrics(clean, pred, M)
        print(f"{name:22s} corr={mt['corr']:.4f} t={dt:.3f}")
