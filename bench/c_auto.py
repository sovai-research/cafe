"""
c_auto — GENERALIZABLE causal imputer with NO benchmark-tuned thresholds.

Anti-overfit principle: the choice of technique is made per-dataset by SELF-
SUPERVISED validation, not by hand-picked rules calibrated to our benchmark.

For each dataset we:
  1. detect only LEGITIMATE, intrinsic facts: is it panel? is it 1D? and what is
     the dataset's own missingness PATTERN (scattered vs contiguous) — these are
     properties of the input, not labels fit to a test set.
  2. build a validation mask by hiding some of the dataset's OWN observed cells,
     MATCHING its missingness pattern (scattered->scattered, block->block).
  3. run each candidate technique, score recovery on the validation cells.
  4. keep the winner ONLY if it beats the robust default by a margin (regularized
     selection -> bias to the simple model, avoids selection-variance overfit).
Then impute the real missing cells with the winner.

This is "internal cross-validation for imputation": it adapts to ANY data because
the signal comes from the data itself, never from tuned constants.
"""
import os, sys
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from c_router import _is_panel
import c_online_trmf, c_fe_lowrank
from c_chal_ewcov import online_impute as f_ewcov
from c_chal_seasonal import online_impute as f_seasonal
from c_chal_xsblend import online_impute as f_xsblend
from c_chal_robust import online_impute as f_robust
from c_chal_blackout import online_impute as f_blackout
from c_chal_driftgate import online_impute as f_drift

DEBUG = os.environ.get("TIMARA_AUTO_DEBUG", "")
SEL_CAP = 400            # rows used for the (cheap) selection pass
MARGIN = 0.02            # relative MAE improvement required to leave the default
N_VAL_SEEDS = 2          # average selection over a couple of validation folds


def _router(X, meta):
    return c_fe_lowrank.online_impute(X, meta) if _is_panel(meta) else c_online_trmf.online_impute(X, meta)


def _contiguity(mask):
    """lag-1 autocorr of the missing mask -> ~0 scattered, high contiguous."""
    m = mask.astype(float); T, N = m.shape
    if T < 3:
        return 0.0
    acs = []
    for j in range(N):
        a, b = m[:-1, j], m[1:, j]
        if a.std() > 1e-9 and b.std() > 1e-9:
            acs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(acs)) if acs else 0.0


def _make_val_mask(obs, contig, rate, seed):
    """Hide a `rate` fraction of OBSERVED cells, matching the missingness pattern."""
    rng = np.random.default_rng(seed)
    T, N = obs.shape
    V = np.zeros((T, N), bool)
    target = int(rate * obs.sum())
    if contig > 0.25:                                  # contiguous: hide blocks in observed runs
        placed = 0; guard = 0
        blen = max(2, T // 20)
        while placed < target and guard < 10000:
            guard += 1
            j = int(rng.integers(N)); s = int(rng.integers(0, max(1, T - blen)))
            seg = obs[s:s + blen, j] & ~V[s:s + blen, j]
            V[s:s + blen, j] |= obs[s:s + blen, j]
            placed += int(seg.sum())
    else:                                              # scattered
        oi, oj = np.where(obs)
        if len(oi):
            pick = rng.choice(len(oi), size=min(target, len(oi)), replace=False)
            V[oi[pick], oj[pick]] = True
    return V


def _candidates(X, meta):
    """Portfolio chosen ONLY by intrinsic structure (panel/1D/2D) — not by tuned thresholds."""
    if _is_panel(meta):
        return {"router": _router, "blackout": f_blackout, "ewcov": f_ewcov}
    if X.shape[1] == 1:
        return {"router": _router, "seasonal": f_seasonal}
    return {"router": _router, "ewcov": f_ewcov, "robust": f_robust, "xsblend": f_xsblend}


def _score(true_vals, pred, vmask):
    p = np.asarray(pred, float)[vmask]
    t = true_vals[vmask]
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return np.inf
    return float(np.mean(np.abs(p[ok] - t[ok])))


def online_impute(X, meta):
    X = np.asarray(X, float)
    T, N = X.shape
    obs = ~np.isnan(X)
    cand = _candidates(X, meta)
    if len(cand) == 1:
        return _router(X.copy(), dict(meta))

    contig = _contiguity(~obs)
    miss_rate = 1.0 - obs.mean()
    val_rate = float(np.clip(miss_rate, 0.05, 0.25))

    # selection pass on a (cheap) subsample, pattern-matched validation
    cap = min(T, SEL_CAP)
    Xs = X[:cap]; obss = obs[:cap]
    meta_s = meta
    if _is_panel(meta) and "entity_ids" in meta:        # keep selection subsample valid for panel
        Xs, obss, meta_s = X, obs, meta                 # panel: don't truncate rows (entity-major)
        cap = T
    scores = {k: [] for k in cand}
    for sd in range(N_VAL_SEEDS):
        V = _make_val_mask(obss, contig, val_rate, seed=1000 + sd)
        if V.sum() < 5:
            continue
        Xv = Xs.copy(); Xv[V] = np.nan
        truth = Xs
        for k, fn in cand.items():
            try:
                pred = fn(Xv.copy(), dict(meta_s))
                scores[k].append(_score(truth, pred, V))
            except Exception:
                scores[k].append(np.inf)
    mean_score = {k: (np.mean(v) if v else np.inf) for k, v in scores.items()}

    # regularized choice: default 'router' unless a candidate beats it by MARGIN
    base = mean_score.get("router", np.inf)
    best_k, best_s = "router", base
    for k, s in mean_score.items():
        if k == "router":
            continue
        if np.isfinite(s) and np.isfinite(base) and s < base * (1 - MARGIN) and s < best_s:
            best_k, best_s = k, s
    if not np.isfinite(base):                            # router failed selection; take best finite
        fin = {k: s for k, s in mean_score.items() if np.isfinite(s)}
        if fin:
            best_k = min(fin, key=fin.get)

    if DEBUG:
        srep = " ".join(f"{k}={mean_score[k]:.3f}" for k in cand)
        print(f"  [auto] {X.shape} panel={_is_panel(meta)} contig={contig:.2f} "
              f"[{srep}] -> {best_k}", file=sys.stderr)
    try:
        return cand[best_k](X.copy(), dict(meta))
    except Exception:
        return _router(X.copy(), dict(meta))


if __name__ == "__main__":
    os.environ["TIMARA_AUTO_DEBUG"] = "1"
    from causal import run_causal, summarize_causal
    summarize_causal(run_causal("c_auto", online_impute))
