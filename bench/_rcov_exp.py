"""Experimental robust online EW-covariance core — measure standalone on all cases.

Robustification (all intrinsic-online, hence causal):
  - GAUSSIAN-RANK transform per column using running observed mean/MAD-scale, then a
    soft tanh winsor: z = tanh((x-med)/(c*scale)) maps heavy tails into a bounded
    range so a single outlier cannot dominate the covariance. We invert the transform
    when emitting imputations.
  - HUBER-downweighted EW moment update: each row's contribution to the EW cov is
    scaled by a weight w_t = min(1, k / d_t) where d_t is the (robust) Mahalanobis
    distance under the current estimate. Caps outlier leverage.
All stats for row t use only rows < t -> point-in-time.
"""
import os, sys
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v,"2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_factor, cho_solve

HALFLIFE = 200.0
RIDGE = 1e-2
WARM = 5
HUBER_K = 9.0       # chi2-ish leverage cap (downweight rows with d_t > k)


def rcov_2d(X, meta=None, robust=True, winsor=True):
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    out = X.copy()
    lam = 0.5 ** (1.0 / HALFLIFE)
    w = 0.0; Sx = np.zeros(N); M2 = np.zeros((N, N))
    col_sum = np.zeros(N); col_cnt = np.zeros(N)
    # running robust scale per column (EW mean + EW mean-abs-dev), for winsor transform
    rc_mu = np.zeros(N); rc_sc = np.ones(N); rc_w = 0.0
    have_cov = False
    I = RIDGE * np.eye(N)

    def _wins(vec, obs):
        if not winsor:
            return vec.copy()
        z = vec.copy()
        sc = np.maximum(rc_sc, 1e-6)
        z[obs] = rc_mu[obs] + sc[obs] * np.tanh((vec[obs] - rc_mu[obs]) / (3.0 * sc[obs]))
        return z

    for t in range(T):
        row = X[t]; obs = ~np.isnan(row); miss = ~obs
        roww = _wins(row, obs)                          # transformed row for cov use
        if miss.any():
            gmean = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
            filled = gmean.copy()
            if have_cov and obs.any() and w > 1e-6:
                mean = Sx / w
                cov = M2 / w - np.outer(mean, mean)
                o = np.where(obs)[0]; m = np.where(miss)[0]
                Soo = cov[np.ix_(o, o)] + I[np.ix_(o, o)]
                Smo = cov[np.ix_(m, o)]
                xo = roww[o] - mean[o]
                try:
                    c = cho_factor(Soo, lower=True, check_finite=False)
                    sol = cho_solve(c, xo, check_finite=False)
                    filled[m] = mean[m] + Smo @ sol
                except Exception:
                    filled[m] = mean[m]
            out[t, miss] = filled[miss]

        full = out[t].copy()
        fullw = _wins(full, np.ones(N, bool))
        # robust leverage weight under current estimate
        wt = 1.0
        if robust and have_cov and w > 1e-6:
            mean = Sx / w
            cov = M2 / w - np.outer(mean, mean) + I
            d = fullw - mean
            try:
                c = cho_factor(cov, lower=True, check_finite=False)
                md = float(d @ cho_solve(c, d, check_finite=False))
                if md > HUBER_K:
                    wt = HUBER_K / md
            except Exception:
                wt = 1.0
        w = lam * w + wt
        Sx = lam * Sx + wt * fullw
        M2 = lam * M2 + wt * np.outer(fullw, fullw)
        col_sum[obs] += row[obs]; col_cnt[obs] += 1
        # update running robust scale (EW) from observed
        if obs.any():
            rc_w = lam * rc_w + 1.0
            a = 1.0 / rc_w
            rc_mu[obs] = (1 - a) * rc_mu[obs] + a * row[obs]
            rc_sc[obs] = (1 - a) * rc_sc[obs] + a * np.abs(row[obs] - rc_mu[obs]) * 1.4826
        if w >= WARM:
            have_cov = True

    if np.isnan(out).any():
        gm = np.where(col_cnt > 0, col_sum / np.maximum(col_cnt, 1), 0.0)
        idx = np.where(np.isnan(out)); out[idx] = np.take(gm, idx[1])
    return out


if __name__ == "__main__":
    from arena import SUITE, MASKERS, metrics
    from c_online_trmf import online_impute as trmf
    print(f"{'case':24s} {'TRMF':>7} {'rcov':>7} {'rcov_nr':>7} {'plain':>7}")
    for name,(clean,meta,mech,rate,grp) in SUITE.items():
        if meta: continue   # 2D/1D only
        M=MASKERS[mech](clean,rate,0); Xo=clean.copy(); Xo[M]=np.nan
        r_trmf=metrics(clean,trmf(Xo.copy(),dict(meta)),M)['corr']
        r_rc=metrics(clean,rcov_2d(Xo.copy(),meta,True,True),M)['corr']
        r_rcnr=metrics(clean,rcov_2d(Xo.copy(),meta,False,True),M)['corr']
        r_pl=metrics(clean,rcov_2d(Xo.copy(),meta,False,False),M)['corr']
        flag='  <--' if r_rc>r_trmf+0.02 else ''
        print(f"{name:24s} {r_trmf:7.3f} {r_rc:7.3f} {r_rcnr:7.3f} {r_pl:7.3f}{flag}")
