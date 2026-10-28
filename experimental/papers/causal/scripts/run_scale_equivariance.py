#!/usr/bin/env python
"""Is CAFE's factor adjustment scale-equivariant, and does it engage on CausalRivers?

Three measurements, all live:

1.  **The equivariance sweep.**  One synthetic panel, one global multiplier
    ``x -> c*x``, everything else identical.  A factor model should be equivariant
    under that, because a global rescale of the data rescales the loadings and
    nothing else.  ``src/cafe/_core.py`` sets the ARD precision
    ``alpha_l = N / (||W[:,l]||^2 + 1.0)``: the ``+1.0`` does not rescale with the
    data, so below unit scale it dominates and drives every column's precision to
    its maximum.  This sweep measures the consequence rather than asserting it.

2.  **Which arm the mechanism is actually running in.**  The paper's two
    CausalRivers arms differ only in scale convention -- unrescaled (the published
    cell) vs z-scored (scale-free).  For each we measure how much variance the
    factor adjustment removes, so a benchmark number can be attributed to the
    method or to a mechanism that never engaged.

3.  **Falsifying the obvious objection**: "residualisation removes the scale
    prior, which is why CAFE-CD loses the scale-exploiting arm".  Varsortability
    is measured on the raw series and on the CAFE residual of the same sample.

The factor-activity metric is deliberately the PER-COLUMN variance share the
factor adjustment removes, read from ONE fit as
``residuals(factor_adjusted=True)`` against ``residuals(factor_adjusted=False)``.
Both readouts share the same level, season, fixed-effect and AR terms, so those
cancel exactly and what is left is the factor projection alone.  Per column, not
pooled: unrescaled discharge columns differ by three orders of magnitude and a
pooled sum of squares would report only the largest one.

``effective_rank`` is reported alongside but is NOT used as the indicator -- part
of the point is that the two disagree.

Writes ``experimental/papers/causal/data/scale_equivariance.json``.  Nothing under
``bench/`` or ``src/`` is written.

    .venv-bench/bin/python experimental/papers/causal/scripts/run_scale_equivariance.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PAPER)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import datasets_causal as DC                                     # noqa: E402
from gen_causal_panel import gen_causal_panel                    # noqa: E402
from sortability import r2_sortability, var_sortability          # noqa: E402
import cafe                                                      # noqa: E402
from sklearn.metrics import roc_auc_score                        # noqa: E402

N_RIVERS = int(os.environ.get("SCALE_N", "25"))
MULTS = [0.01, 0.1, 1.0, 3.0, 10.0, 100.0, 1000.0]
CELL = dict(region="flood_full", n_vars=5, strategy="random", resolution="6h",
            normalize=False)
OUT = os.path.join(PAPER, "data", "scale_equivariance.json")


def _resid(res, adjusted):
    """(residual, observed mask) from any CAFE result object, panel or single-series.

    Panel results always return the (E, obs) pair and reject ``return_mask``; the
    single-series result requires it.  Same dispatch as
    ``cafe.discovery._residual_cube``.
    """
    kw = dict(factor_adjusted=bool(adjusted), mask_imputed=True, robust=False)
    if hasattr(res, "_pe") or hasattr(res, "stream_result"):
        E_, o_ = res.residuals(**kw)
    else:
        E_, o_ = res.residuals(return_mask=True, **kw)
    E_ = np.asarray(E_, float).reshape(-1, np.shape(E_)[-1])
    o_ = np.asarray(o_, bool).reshape(E_.shape)
    return E_, o_


def factor_activity(X):
    """(per-column variance share the factor adjustment removes, effective_rank).

    One fit, two read-outs, so every non-factor term cancels.
    """
    res = cafe.CAFE().run(np.asarray(X, float), engine="per_entity")
    Ea, oa = _resid(res, True)
    Ek, ok = _resid(res, False)
    m = np.asarray(oa, bool) & np.asarray(ok, bool) & np.isfinite(Ea) & np.isfinite(Ek)
    sh = []
    for j in range(Ea.shape[1]):
        mj = m[:, j]
        if mj.sum() < 32:
            continue
        vk = float(np.var(Ek[mj, j]))
        if vk <= 0:
            continue
        sh.append(1.0 - float(np.var(Ea[mj, j])) / vk)
    return (float(np.mean(sh)) if sh else float("nan"),
            int(res.effective_rank()), [float(x) for x in sh])


def auc_offdiag(S, A):
    S = np.abs(np.asarray(S, float))
    if S.ndim == 3:                      # (K, P, P) lag stack -> lag-summary graph
        S = np.where(np.isfinite(S), S, 0.0).max(axis=0)
    P = A.shape[0]
    off = ~np.eye(P, dtype=bool)
    y = (np.asarray(A) != 0)[off].astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.nan_to_num(np.asarray(S, float))[off]))


def main() -> int:
    t0 = time.time()
    out = {"mults": MULTS, "cell": CELL, "n_rivers": N_RIVERS}

    # ---- 1. equivariance sweep on our own generator ------------------------ #
    X3, _X2, _meta, truth = gen_causal_panel(E=10, T=120, P=24, R=3, n_lags=2, deg=2,
                                             chi=0.90, noise="laplace",
                                             beta_hetero=0.6, seed=0)
    X3 = np.asarray(X3, float)
    A_true = (np.abs(np.asarray(truth.A_lag, float)).max(axis=0) != 0)
    base_sd = float(np.mean(np.std(X3.reshape(-1, X3.shape[-1]), axis=0)))
    sweep = []
    for c in MULTS:
        sh, rk, _ = factor_activity(X3 * c)
        g = cafe.discover(X3 * c, max_lag=2)
        sweep.append(dict(mult=c, mean_sd=base_sd * c, factor_share=sh,
                          effective_rank=rk,
                          auc=auc_offdiag(g.scores(lag="lagged"), A_true)))
        print("  x%-8g mean sd %9.3f  factor removes %7.2f%%  rank %d  auc %.4f"
              % (c, base_sd * c, 100 * sh, rk, sweep[-1]["auc"]), flush=True)
    out["sweep"] = sweep
    out["panel"] = dict(E=10, T=120, P=24, R=3, K=2, chi=0.90, base_mean_sd=base_sd)

    # ---- 2 + 3. the two CausalRivers arms, and the sortability control ----- #
    rows = []
    for i in range(N_RIVERS):
        try:
            X, A, meta = DC.load_causal_rivers(index=i, **CELL)
        except Exception as exc:                                 # pragma: no cover
            print("  sample %d: %s" % (i, exc))
            continue
        X = np.asarray(X, float)
        ok = np.isfinite(X).all(axis=1)
        Xo = X[ok]
        if Xo.shape[0] < 64 or not np.all(Xo.std(axis=0) > 0):
            continue
        Z = (Xo - Xo.mean(0)) / Xo.std(0)
        su, ru, _ = factor_activity(Xo)
        sz, rz, _ = factor_activity(Z)
        # sortability: labels are [effect, cause]; sortability wants [cause, effect]
        skel = np.asarray(A, float).T > 0
        res = cafe.CAFE().run(Xo, engine="per_entity")
        Er, om = res.residuals(factor_adjusted=True, return_mask=True,
                               mask_imputed=True, robust=False)
        Er = np.asarray(Er, float)
        keep = np.asarray(om, bool).all(axis=1) & np.isfinite(Er).all(axis=1)
        r = dict(index=i, T=int(Xo.shape[0]),
                 col_sd_min=float(Xo.std(0).min()),
                 col_sd_med=float(np.median(Xo.std(0))),
                 col_sd_max=float(Xo.std(0).max()),
                 share_unrescaled=su, rank_unrescaled=ru,
                 share_zscored=sz, rank_zscored=rz)
        if skel.sum() and keep.sum() > 64 and np.all(Er[keep].std(0) > 0):
            r.update(varsort_raw=float(var_sortability(Xo, skel)),
                     varsort_resid=float(var_sortability(Er[keep], skel)),
                     r2sort_raw=float(r2_sortability(Xo, skel)),
                     r2sort_resid=float(r2_sortability(Er[keep], skel)))
        rows.append(r)
        print("  rivers %2d  unrescaled %8.2f%% (rank %d)  z-scored %6.2f%% (rank %d)"
              % (i, 100 * su, ru, 100 * sz, rz), flush=True)
    out["rivers"] = rows

    def mean(k):
        xs = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.mean(xs)) if xs else float("nan")

    out["rivers_summary"] = dict(
        n=len(rows),
        share_unrescaled=mean("share_unrescaled"),
        share_zscored=mean("share_zscored"),
        n_unrescaled_negative=int(sum(r["share_unrescaled"] < 0 for r in rows)),
        n_zscored_negative=int(sum(r["share_zscored"] < 0 for r in rows)),
        col_sd_med=mean("col_sd_med"),
        col_sd_min=float(min(r["col_sd_min"] for r in rows)) if rows else float("nan"),
        col_sd_max=float(max(r["col_sd_max"] for r in rows)) if rows else float("nan"),
        frac_samples_with_a_subunit_column=float(
            np.mean([r["col_sd_min"] < 1.0 for r in rows])) if rows else float("nan"),
        varsort_raw=mean("varsort_raw"), varsort_resid=mean("varsort_resid"),
        r2sort_raw=mean("r2sort_raw"), r2sort_resid=mean("r2sort_resid"),
        n_sortability=int(sum("varsort_raw" in r for r in rows)))
    out["wall_s"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote %s (%.1f min)" % (OUT, out["wall_s"] / 60))
    print(json.dumps(out["rivers_summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
