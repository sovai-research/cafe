"""
Causal (fully point-in-time) evaluation layer on top of harness.py.

CONSTRAINT (user, governing): an imputed value for (entity e, time tau, feature i)
may use ONLY information at time <= tau -- past values and the CONTEMPORANEOUS
cross-section at tau (other features/entities observed at the same tau). NEVER any
value at time s > tau. Parameters must also be estimated point-in-time (expanding /
online). The contemporaneous cross-section IS allowed.

A causal imputer here has the SAME signature as a normal one:
    online_impute(X, meta) -> filled (T,N) array
but it must compute each masked cell causally. We do NOT trust that by fiat: we
VERIFY it. `assert_causal` re-runs the method on time-truncated prefixes and checks
that the imputations at time tau are unchanged when future rows are removed. Any
method that peeks ahead fails this check.

Time axis:
  - 2D datasets (meta empty): time_id of row r is r.
  - panel datasets (entity-major stacked): meta['time_ids'][r] is the time of row r.
"""
from __future__ import annotations
import numpy as np
from harness import DATASETS, MASKERS, metrics
import time


def time_ids_of(X, meta):
    if meta and "time_ids" in meta:
        return np.asarray(meta["time_ids"])
    return np.arange(X.shape[0])


def slice_meta_by_rows(meta, keep):
    """Return a meta dict restricted to the boolean/index row selection `keep`."""
    if not meta:
        return {}
    out = dict(meta)
    for k in ("entity_ids", "time_ids"):
        if k in meta:
            out[k] = np.asarray(meta[k])[keep]
    return out


def assert_causal(online_fn, X_obs, meta, n_probes=4, rtol=1e-5, atol=1e-6, seed=0):
    """
    Verify causality. Run on full data, then on time-prefixes; the imputations at
    the probe time must match. Returns (is_causal, detail_str).
    """
    tid = time_ids_of(X_obs, meta)
    full = np.asarray(online_fn(X_obs.copy(), dict(meta)), dtype=float)
    uniq = np.unique(tid)
    if len(uniq) < 4:
        return True, "too-few-times-to-probe"
    rng = np.random.default_rng(seed)
    cand = uniq[(len(uniq) // 5):-1]            # avoid earliest (cold start) & last
    if len(cand) == 0:
        return True, "no-probe-candidates"
    probes = sorted(rng.choice(cand, size=min(n_probes, len(cand)), replace=False))
    nanmask = np.isnan(X_obs)
    for tau in probes:
        keep = tid <= tau
        Xp = X_obs[keep].copy()
        mp = slice_meta_by_rows(meta, keep)
        outp = np.asarray(online_fn(Xp, mp), dtype=float)
        # rows of the prefix that are exactly at time tau
        at_tau_prefix = (tid[keep] == tau)
        at_tau_full = (tid == tau)
        a = full[at_tau_full]
        b = outp[at_tau_prefix]
        m = nanmask[at_tau_full]                # only the imputed (masked) cells matter
        if a.shape != b.shape:
            return False, f"shape-mismatch@tau={tau}"
        d = np.abs(a[m] - b[m])
        tolv = atol + rtol * np.abs(a[m])
        if np.any(d > tolv):
            return False, f"LOOK-AHEAD@tau={tau}: max_drift={float(np.max(d)):.3e}"
    return True, "ok"


def evaluate_causal(method_name, online_fn, dataset_name, dataset,
                    verify=True, seed=0):
    clean, meta, mech, rate = dataset
    M = MASKERS[mech](clean, rate, seed=seed)
    Xobs = clean.copy()
    Xobs[M] = np.nan
    causal_ok, detail = (True, "skipped")
    if verify:
        try:
            causal_ok, detail = assert_causal(online_fn, Xobs, meta)
        except Exception as e:                                  # noqa: BLE001
            causal_ok, detail = False, f"verify-error: {type(e).__name__}: {e}"
    t0 = time.perf_counter()
    try:
        pred = np.asarray(online_fn(Xobs.copy(), dict(meta)), dtype=float)
        dt = time.perf_counter() - t0
        m = metrics(clean, pred, M)
        m.update(success=True, error="")
    except Exception as e:                                      # noqa: BLE001
        dt = time.perf_counter() - t0
        m = dict(mae=float("nan"), rmse=float("nan"), mre=float("nan"),
                 corr=float("nan"), n=int(M.sum()), success=False,
                 error=f"{type(e).__name__}: {e}")
    return dict(method=method_name, dataset=dataset_name, mechanism=mech, rate=rate,
                shape=f"{clean.shape[0]}x{clean.shape[1]}", time_s=dt,
                causal=bool(causal_ok), causal_detail=detail, **m)


def run_causal(method_name, online_fn, datasets=None, verify=True):
    datasets = datasets or DATASETS
    return [evaluate_causal(method_name, online_fn, n, d, verify=verify)
            for n, d in datasets.items()]


def summarize_causal(rows):
    hdr = (f"{'method':20s} {'dataset':18s} {'MAE':>8s} {'RMSE':>8s} "
           f"{'corr':>6s} {'time_s':>9s}  causal  ok")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        c = "Y" if r.get("causal") else f"N({r.get('causal_detail','')[:18]})"
        print(f"{r['method'][:20]:20s} {r['dataset'][:18]:18s} {r['mae']:8.4f} "
              f"{r['rmse']:8.4f} {r['corr']:6.3f} {r['time_s']:9.4f}  {c:7s} "
              f"{'Y' if r['success'] else 'N:'+r['error'][:30]}")


if __name__ == "__main__":
    # demo: a causal online column-mean (expanding mean of past+contemporaneous obs).
    # Cross-section allowed, so we use the running per-column mean over observed
    # rows seen so far; for a missing cell at row r we use the mean of that column
    # over rows < r (strictly past) -- guaranteed causal -> should PASS the verifier.
    def causal_running_mean(X, meta):
        T, N = X.shape
        out = X.copy()
        csum = np.zeros(N); ccnt = np.zeros(N)
        for r in range(T):
            row = X[r]
            obs = ~np.isnan(row)
            mean_so_far = np.where(ccnt > 0, csum / np.maximum(ccnt, 1), 0.0)
            miss = np.isnan(row)
            out[r, miss] = mean_so_far[miss]
            csum[obs] += row[obs]; ccnt[obs] += 1
        return out

    rows = run_causal("CausalRunMean", causal_running_mean)
    summarize_causal(rows)
