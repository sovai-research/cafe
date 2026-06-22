"""
w5_boa -- Causal ONLINE MODEL AGGREGATION (BOA, Bernstein Online Aggregation,
Wintenberger 2017) over K=3 causal experts for 2D matrices. Headline idea:
replace model SELECTION (which leaks, because a "which expert is best" decision
read off the whole series flips once the verifier truncates time) with an online
BLEND whose mixing weights are a forward recursion of PAST realized loss only.

EXPERTS (all import-only, all individually causal-clean):
    k=0  trmf       (low-rank + AR(1))           -> low-rank / temporal structure
    k=1  ewcov      (EW multivariate Gaussian)   -> dense wide cross-section
    k=2  xsblend    (cross-sectional ridge)      -> high-rank cross-section

THE CAUSAL LOSS SIGNAL (the crux). BOA needs a per-expert loss at every row, but
the imputation truth at the missing cells is hidden. We manufacture an honest,
point-in-time, self-supervised loss WITHOUT peeking at the missing truth:

  * Pick a deterministic HELD-OUT probe set among the OBSERVED cells. The choice is
    a pure function of (row index, column index) -- it does NOT depend on time-
    truncation, so it is identical on the full run and on every prefix the verifier
    probes (truncation-invariant => causal-clean).
  * Additionally NaN-out the probe cells and run each expert ONCE on this probed
    matrix, yielding predictions at the probe positions. Because each expert is
    causal, its prediction at probe cell (t,j) uses only data at times <= t.
  * Per row t we have, for each expert k, a squared error r_k(t) on the probe cells
    of row t. Losses are min-max normalized across experts to [0,1] per row (a
    contemporaneous, leak-free transform). The instantaneous regret of expert k is
    r_k = loss_k - blend_loss.

  Crucially the BOA weight used to impute row t is built ONLY from probe losses at
  rows STRICTLY < t. So nothing at time >= t (and certainly no missing truth) enters
  the weight for t. Forward recursion => point-in-time.

SELF-TUNING BOA WEIGHT RECURSION (per expert k, exactly as specified):
    eta_inv2_k += C * r_k^2                  (C self-tuning learning-rate growth;
                                              tuned to 40 for K=3 short-2D regret,
                                              NOT a per-case constant -- it is a single
                                              global rate, see BOA_C note below)
    eta_k       = 1/sqrt(eta_inv2_k)
    r_reg       = r_k - r_k^2 / sqrt(eta_inv2_k)   (Bernstein/2nd-order correction)
    R_k        += r_reg                       (cumulative regularized regret)
    log_w_k     = eta_k * R_k + 0.5*log(eta_k^2)
    p           = softmax(log_w)
    p           = (1-FLOOR)*p + FLOOR/K       (fixed-share floor -> tracks switches)

IMPUTATION. Run each expert ONCE more on the ORIGINAL matrix (probe cells present)
to get full prediction matrices P_k. Missing cell (t,j) is filled with
sum_k p_k(t) * P_k[t,j], where p_k(t) is the BOA weight from probe-losses < t.

PANELS / 1D: BOA's three 2D experts don't all apply; we defer to the existing
router behaviour (FE for panels, trmf for 1D) -- byte-identical to the cores there.
Probing requires enough observed cells; degenerate inputs fall back to trmf.
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from c_online_trmf import online_impute as _trmf
from c_fe_lowrank import online_impute as _fe
from c_router import _is_panel
from c_chal_ewcov import _ewcov_2d
from c_chal_xsblend import online_impute as _xsblend

# ----------------------------------------------------------------- knobs --- #
BOA_C = 40.0         # self-tuning learning-rate growth (eta_inv2 += C*r^2). Larger
                     # than the textbook 2.2 because with only K=3 experts and short
                     # 2D series the regret signal must concentrate quickly to shed
                     # the consistently-weak ewcov expert before the series ends.
FLOOR = 0.005        # fixed-share floor on the simplex (switch tracking)
PROBE_STRIDE = 7     # ~1/7 of observed cells held out (deterministic, position-based)
WARM = 8             # rows before the blend turns on (uniform weights until then)
MIN_OBS_FRAC = 0.05  # if too few observed cells, skip BOA -> trmf fallback


def _experts():
    """Ordered list of (name, fn). All causal-clean 2D cores."""
    return [("trmf", _trmf), ("ewcov", _ewcov_2d), ("xsblend", _xsblend)]


def _probe_mask(X):
    """Deterministic held-out probe set among OBSERVED cells. Pure function of cell
    position (row*N + col), so it is invariant to time-truncation (causal-clean).
    Returns a boolean (T,N) mask of cells to hold out."""
    T, N = X.shape
    obs = ~np.isnan(X)
    lin = (np.arange(T)[:, None] * N + np.arange(N)[None, :])
    pm = obs & ((lin % PROBE_STRIDE) == 0)
    return pm


def _run_expert(fn, X, meta):
    """Run an expert, returning a finite (T,N) prediction or None on failure."""
    try:
        P = np.asarray(fn(X.copy(), dict(meta)), dtype=float)
        if P.shape != X.shape or not np.all(np.isfinite(P)):
            return None
        return P
    except Exception:
        return None


def online_impute(X, meta):
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    # Panels and 1D -> unmodified router cores (BOA's 2D experts don't all apply).
    if _is_panel(meta):
        return _fe(X, meta)
    T, N = X.shape
    if N < 2 or T < WARM + 2:
        return _trmf(X, meta)

    miss = np.isnan(X)
    obs = ~miss
    n_obs = int(obs.sum())
    if n_obs < max(8, int(MIN_OBS_FRAC * T * N)):
        return _trmf(X, meta)

    experts = _experts()
    K = len(experts)

    # ---- full-matrix predictions on the ORIGINAL data (probe cells present) --- #
    P_full = []
    for _, fn in experts:
        P = _run_expert(fn, X, meta)
        P_full.append(P)
    # Need at least the trmf base to exist for a safe fallback.
    if P_full[0] is None:
        # try any working expert as the base
        base = next((p for p in P_full if p is not None), None)
        if base is None:
            return _trmf(X, meta)
    else:
        base = P_full[0]

    # ---- probed predictions (held-out observed cells masked) for the loss --- #
    pm = _probe_mask(X)
    if pm.sum() < max(4, K):
        # not enough probes to learn weights -> uniform blend of available experts
        avail = [p for p in P_full if p is not None]
        out = X.copy()
        if avail:
            blend = np.mean(np.stack(avail, 0), 0)
            out[miss] = blend[miss]
        else:
            out[miss] = base[miss]
        if np.isnan(out).any():
            out[np.isnan(out)] = 0.0
        return out

    Xp = X.copy()
    Xp[pm] = np.nan
    P_probe = []
    for _, fn in experts:
        P_probe.append(_run_expert(fn, Xp, meta))

    # ---- per-row, per-expert squared error on probe cells -------------------- #
    # err[t,k] = sum of squared error of expert k over row t's probe cells (NaN if
    # the row has no probe cells or expert k failed).
    truth = X  # probe cells hold the real observed value
    sqerr = np.full((T, K), np.nan)
    for k in range(K):
        Pk = P_probe[k]
        if Pk is None:
            continue
        e = np.where(pm, (Pk - truth) ** 2, 0.0)
        cnt = pm.sum(1)
        se = e.sum(1)
        sqerr[:, k] = np.where(cnt > 0, se / np.maximum(cnt, 1), np.nan)

    # Experts that failed entirely -> treated as worst (loss=1) so they get no weight.
    failed = np.array([P_full[k] is None for k in range(K)])

    # ---- forward BOA recursion (weights from probe-losses at rows < t) ------- #
    eta_inv2 = np.full(K, 1e-6)   # accumulator, small positive start
    R = np.zeros(K)               # cumulative regularized regret
    p = np.full(K, 1.0 / K)       # current mixing weights (used for the NEXT row)

    out = X.copy()
    seen = 0                      # number of rows with a usable probe loss so far

    for t in range(T):
        # ---- impute row t with the CURRENT weights p (built from rows < t) ---- #
        m_t = miss[t]
        if m_t.any():
            num = np.zeros(N)
            wsum = 0.0
            for k in range(K):
                Pk = P_full[k]
                if Pk is None:
                    continue
                num += p[k] * Pk[t]
                wsum += p[k]
            if wsum > 1e-12:
                out[t, m_t] = (num / wsum)[m_t]
            else:
                out[t, m_t] = base[t, m_t]

        # ---- update weights from THIS row's probe loss (for use at t+1) ------- #
        row_se = sqerr[t]
        valid = np.isfinite(row_se)
        if valid.sum() >= 1 and pm[t].any():
            # normalize losses across experts to [0,1] this row (leak-free transform)
            loss = np.ones(K)                 # failed/invalid experts -> worst loss 1
            v = row_se[valid]
            lo = v.min(); hi = v.max()
            if hi - lo > 1e-12:
                loss[valid] = (v - lo) / (hi - lo)
            else:
                loss[valid] = 0.0             # all equal -> no regret signal
            loss[failed] = 1.0

            blend_loss = float(np.dot(p, loss))
            r = blend_loss - loss             # instantaneous regret (gain if r>0)

            eta_inv2 = eta_inv2 + BOA_C * r * r
            eta = 1.0 / np.sqrt(eta_inv2)
            r_reg = r - (r * r) / np.sqrt(eta_inv2)
            R = R + r_reg
            log_w = eta * R + 0.5 * np.log(eta * eta)
            log_w = log_w - log_w.max()
            w = np.exp(log_w)
            w[failed] = 0.0
            s = w.sum()
            if s > 1e-12 and seen + 1 >= WARM:
                p = w / s
                p = (1.0 - FLOOR) * p + FLOOR / K
                p[failed] = 0.0
                ps = p.sum()
                if ps > 1e-12:
                    p = p / ps
                else:
                    p = np.where(failed, 0.0, 1.0)
                    p = p / p.sum()
            seen += 1

    if np.isnan(out).any():
        out[np.isnan(out)] = base[np.isnan(out)] if base is not None else 0.0
        out[np.isnan(out)] = 0.0
    return out


if __name__ == "__main__":
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("w5_boa", online_impute))
