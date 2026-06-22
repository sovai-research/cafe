"""
Chal_robust -- robust-loss wrapper for heavy-tailed 2D data.

PROBLEM: the champion (CausalRouter -> OnlineTRMF on 2D) uses an L2 factor solve.
On data with sparse heavy-tailed outliers (Student-t contamination), a handful of
extreme observed values dominate the W / factor estimation, wrecking the low-rank
fit. The arena's `2d_heavytail_mcar30` case sits at corr=0.146 (the suite minimum).

FIX (gated, point-in-time): detect heavy tails up front with a CHEAP signal -- the
MEDIAN per-column excess kurtosis of the observed values. Synthetic/real benign
data has median kurtosis ~< 4 (beijing 3.5, drift 2.1, ETTh1 2.0, airq/chlorine/
temp < 0, all the synthetic 2D ~ -0.1); the heavy-tail case has median 62.9. A
threshold of 10 cleanly isolates it and leaves every expensive real dataset
(beijing 53s, ETTh1 3s, temp/drift) routed through the UNCHANGED champion, so
total time and all other case accuracies are untouched.

For a gated heavy-tail matrix we Winsorize the OBSERVED values causally before
handing the matrix to the unchanged OnlineTRMF core: at each time t, column j's
observed value is clipped to a robust band  median +/- k*1.4826*MAD  computed from
that column's OWN observed history at times <= t (expanding, warm-started after a
few points). This is strictly point-in-time -- only past+current observations feed
the band, and we only ever clip INPUTS that feed the factor solve (missing cells
are imputed by the core as usual, never affected by look-ahead). Clipping caps the
leverage of outliers on the L2 fit (a hard-redescending M-estimator surrogate),
recovering the underlying low-rank signal.

Everything else routes verbatim to CausalRouter (panel -> CausalFE, benign 2D ->
OnlineTRMF), so the rest of the suite is byte-for-byte the champion.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
sys.path.insert(0, "/Users/dereksnow/Sovai/Github/TIMARA/bench")
import numpy as np

from c_router import online_impute as _router, _is_panel
from c_online_trmf import online_impute as _trmf

# ----------------------------------------------------------------- knobs ---
KURT_GATE = 10.0   # median per-column excess kurtosis above this -> heavy-tail
WINZ_K = 2.0       # clip band half-width in robust-sigma units (k * 1.4826 * MAD)
WINZ_WARM = 8      # min observed points in a column before clipping kicks in


def _median_excess_kurtosis(X):
    """Median across columns of per-column excess (Fisher) kurtosis, observed only.

    Cheap O(T*N) up-front signal. Columns with < ~12 observations are skipped
    (kurtosis is unstable on tiny samples). Returns 0.0 if nothing qualifies.
    """
    ks = []
    for j in range(X.shape[1]):
        c = X[:, j]
        c = c[np.isfinite(c)]
        if c.size < 12:
            continue
        m = c.mean()
        d = c - m
        v = np.mean(d * d)
        if v <= 1e-12:
            continue
        k = np.mean(d ** 4) / (v * v) - 3.0
        ks.append(k)
    if not ks:
        return 0.0
    return float(np.median(ks))


def _causal_winsorize(X, k=WINZ_K, warm=WINZ_WARM):
    """Clip each observed value to median +/- k*1.4826*MAD of that column's OWN
    observed history at times <= t (expanding). Strictly point-in-time. The
    running median/MAD use the clipped values (more stable, still causal).
    Missing cells are left as NaN for the downstream core to impute.
    """
    X = X.copy()
    T, N = X.shape
    for j in range(N):
        c = X[:, j]
        obs_idx = np.where(np.isfinite(c))[0]
        if obs_idx.size < warm + 1:
            continue
        buf = []
        for t in obs_idx:
            v = c[t]
            if len(buf) >= warm:
                a = np.asarray(buf)
                med = np.median(a)
                mad = np.median(np.abs(a - med)) + 1e-9
                band = k * 1.4826 * mad
                cv = med - band if v < med - band else (med + band if v > med + band else v)
                c[t] = cv
                buf.append(cv)
            else:
                buf.append(v)
        X[:, j] = c
    return X


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    # Panels and 1D univariate: nothing to gain from column winsorizing here;
    # route verbatim to the champion.
    if _is_panel(meta) or X.shape[1] < 2:
        return _router(X, meta)
    # Cheap up-front heavy-tail gate on 2D data.
    if _median_excess_kurtosis(X) > KURT_GATE:
        Xw = _causal_winsorize(X)
        return _trmf(Xw, dict(meta) if meta else {})
    # Benign 2D -> unchanged champion route.
    return _router(X, meta)


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("Chal_robust", online_impute))
