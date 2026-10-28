#!/usr/bin/env python
"""Why CAFE-CD loses on CausalRivers: a 2x2 decomposition of the two design choices.

CAFE-CD ranks 11/13 on the published CausalRivers cell (see the paper's Table 6).
This script isolates *which* stage costs the ranking, by holding the estimator
fixed and varying only the two things CAFE-CD does differently from the
baselines it loses to:

  * the REPRESENTATION -- the raw panel vs the CAFE factor-adjusted residual;
  * the CONDITIONING SET -- for target ``i`` and source ``j``, regress on
    ``i``'s own past plus ALL other sources (fully conditional, what the
    skeleton does) or plus exactly ONE other source (bivariate Granger).

Both cells of the second axis come from the SAME estimator on the SAME moment
tensor -- ``cafe.discovery._refit_t`` and ``cafe._skeleton.bivariate_granger``
-- so the contrast is the conditioning set and nothing else.  ``var_ols`` and
``granger_pairwise`` are run alongside as external identity checks: the
fully-conditional cell should reproduce ``var_ols`` and the bivariate cell
should reproduce ``granger_pairwise``.

Also measured, so the alternative explanations are ruled out rather than
asserted: the panel's collinearity (mean/max off-diagonal |corr|, condition
number, leading-eigenvalue share), CAFE's own selected rank and residual
variance share, a naive marginal lagged |corr| ranking, and the structural
conditioning diagnostic that the scoring router reads.

Everything is scored off-diagonally with CausalRivers' own protocol (AUROC over
the ordered pairs of the lag-summarised graph, diagonal removed).

    .venv-bench/bin/python experimental/papers/causal/scripts/run_conditioning_diagnosis.py

Writes ``experimental/papers/causal/data/conditioning.json``.  Nothing under
``bench/`` or ``src/`` is written.
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

import causal_baselines as cb                                    # noqa: E402
import datasets_causal as DC                                     # noqa: E402
import cafe                                                      # noqa: E402
from cafe import discovery as DISC                                # noqa: E402
from cafe._skeleton import (build_moment_tensor,                  # noqa: E402
                            conditioning_diagnostic, bivariate_granger)
from sklearn.metrics import (average_precision_score,             # noqa: E402
                             precision_recall_curve, roc_auc_score)

K = 3                       # the published CausalRivers cell's lag budget
N_SAMPLES = int(os.environ.get("DIAG_N", "30"))
OUT = os.path.join(PAPER, "data", "conditioning.json")


def _lagsum(T3):
    """(K, P, P) -> (P, P): the lag-summary graph, max |value| over lags."""
    T3 = np.abs(np.asarray(T3, float))
    return np.where(np.isfinite(T3), T3, 0.0).max(axis=0)


def _best_f1(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmax(2 * rc * pr / (rc + pr)))


def _scorer(A):
    off = ~np.eye(A.shape[0], dtype=bool)
    y = (np.asarray(A) != 0)[off].astype(int)

    def sc(M):
        p = np.nan_to_num(np.asarray(M, float))[off]
        if len(np.unique(y)) < 2:
            return None
        return dict(auroc=float(roc_auc_score(y, p)),
                    auprc=float(average_precision_score(y, p)),
                    f1max=_best_f1(y, p),
                    top_k_hits=int(y[np.argsort(-p)[:int(y.sum())]].sum()))
    return sc, y


def collinearity(X):
    C = np.corrcoef(np.asarray(X, float).T)
    P = C.shape[0]
    o = C[~np.eye(P, dtype=bool)]
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return dict(mean_abs_offdiag_corr=float(np.abs(o).mean()),
                max_abs_offdiag_corr=float(np.abs(o).max()),
                condition_number=float(ev[0] / max(ev[-1], 1e-300)),
                leading_eig_share=float(ev[0] / ev.sum()))


def marginal_lagged_corr(X, K):
    Xs = np.asarray(X, float)
    Xs = (Xs - Xs.mean(0)) / np.maximum(Xs.std(0), 1e-12)
    P = Xs.shape[1]
    M = np.zeros((P, P))
    for k in range(1, K + 1):
        C = np.abs(np.corrcoef(Xs[k:].T, Xs[:-k].T)[:P, P:])   # [target, source]
        M = np.maximum(M, C)
    np.fill_diagonal(M, 0.0)
    return M


def arms(Ecube, obs, P):
    """(conditional |t|, bivariate |t|, conditioning diagnostic) for one design."""
    mt = build_moment_tensor(Ecube, obs, max_lag=K)
    t_cond, _b, tier = DISC._refit_t(mt, K, P)
    t_biv = bivariate_granger(mt, max_lag=K)
    d = conditioning_diagnostic(mt, max_lag=K)
    return _lagsum(t_cond), _lagsum(t_biv), d, tier


def one_sample(i):
    X, A, meta = DC.load_causal_rivers(region="flood_full", n_vars=5,
                                       strategy="random", index=i,
                                       resolution="6h", normalize=False)
    P = X.shape[1]
    sc, y = _scorer(A)
    if len(np.unique(y)) < 2:
        return None
    row = dict(index=i, T=int(X.shape[0]), P=int(P), n_true_offdiag=int(y.sum()),
               nodes=[int(n) for n in meta["nodes"]], **collinearity(X))

    # --- the two designs ------------------------------------------------- #
    Xc = np.asarray(X, float)[None]
    obs_raw = np.isfinite(Xc)
    t0 = time.perf_counter()
    res = cafe.CAFE().run(np.asarray(X, float), engine="per_entity")
    Eres, obs = DISC._residual_cube(res, (1, X.shape[0], P))
    obs = obs & np.isfinite(Eres)
    row["cafe_model_s"] = time.perf_counter() - t0
    row["effective_rank"] = int(res.effective_rank())
    v_raw = np.nanvar(np.asarray(X, float), axis=0)
    v_res = np.nanvar(np.where(obs[0], Eres[0], np.nan), axis=0)
    row["residual_var_share"] = float(np.nansum(v_res) / max(np.nansum(v_raw), 1e-300))

    for name, cube, ob in (("raw", Xc, obs_raw), ("residual", Eres, obs)):
        t_cond, t_biv, d, tier = arms(cube, ob, P)
        row[f"{name}/conditional"] = sc(t_cond)
        row[f"{name}/bivariate"] = sc(t_biv)
        row[f"{name}/tolerance"] = float(d["tolerance"])
        row[f"{name}/min_tolerance"] = float(d["min_tolerance"])
        row[f"{name}/n_below_half"] = int(d["n_below_half"])
        row[f"{name}/t_tier"] = tier

    # --- the shipped pipeline, and two external identity checks ---------- #
    t0 = time.perf_counter()
    g = cafe.discover(np.asarray(X, float), max_lag=K)
    row["cafe_cd_s"] = time.perf_counter() - t0
    row["cafe_cd"] = sc(g.scores(lag=None))
    row["cafe_cd_conditioning"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                                   for k, v in g.meta.get("conditioning", {}).items()}
    for m in ("var_ols", "granger_pairwise", "dynotears"):
        try:
            r = cb.run_baseline(m, np.asarray(X, float), max_lag=K)
            row[m] = sc(cb.summary_graph(r, how="max", include_contemp=True))
        except Exception as exc:                                 # pragma: no cover
            row[m] = None
            row[m + "_err"] = f"{type(exc).__name__}: {exc}"
    row["marginal_lagged_corr"] = sc(marginal_lagged_corr(X, K))
    return row


def main() -> int:
    rows, t0 = [], time.time()
    for i in range(N_SAMPLES):
        try:
            r = one_sample(i)
        except Exception as exc:
            print(f"  sample {i}: {type(exc).__name__}: {exc}")
            continue
        if r is None:
            continue
        rows.append(r)
        print("  %2d/%d  cafe_cd %.3f | raw cond %.3f biv %.3f | resid cond %.3f "
              "biv %.3f | var_ols %.3f granger %.3f  (%.1f min)"
              % (i + 1, N_SAMPLES, r["cafe_cd"]["auroc"],
                 r["raw/conditional"]["auroc"], r["raw/bivariate"]["auroc"],
                 r["residual/conditional"]["auroc"], r["residual/bivariate"]["auroc"],
                 r["var_ols"]["auroc"], r["granger_pairwise"]["auroc"],
                 (time.time() - t0) / 60), flush=True)

    def mean(key, sub="auroc"):
        xs = [r[key][sub] for r in rows if r.get(key)]
        return float(np.mean(xs)) if xs else float("nan")

    arms_mean = {k: {s: mean(k, s) for s in ("auroc", "auprc", "f1max")}
                 for k in ("raw/conditional", "raw/bivariate",
                           "residual/conditional", "residual/bivariate",
                           "cafe_cd", "var_ols", "granger_pairwise", "dynotears",
                           "marginal_lagged_corr")}
    ident = {
        "raw/conditional vs var_ols": float(np.mean(
            [abs(r["raw/conditional"]["auroc"] - r["var_ols"]["auroc"]) for r in rows])),
        "raw/bivariate vs granger_pairwise": float(np.mean(
            [abs(r["raw/bivariate"]["auroc"] - r["granger_pairwise"]["auroc"])
             for r in rows])),
    }
    out = dict(
        config="CausalRivers flood_full/random_5, 6h, normalize=False "
               "(the published VAR cell); diagonal removed",
        max_lag=K, n_samples=len(rows),
        mean_T=float(np.mean([r["T"] for r in rows])),
        mean_true_edges=float(np.mean([r["n_true_offdiag"] for r in rows])),
        collinearity={k: float(np.mean([r[k] for r in rows])) for k in
                      ("mean_abs_offdiag_corr", "max_abs_offdiag_corr",
                       "condition_number", "leading_eig_share")},
        effective_rank_mean=float(np.mean([r["effective_rank"] for r in rows])),
        effective_rank_zero_frac=float(np.mean(
            [r["effective_rank"] == 0 for r in rows])),
        residual_var_share=float(np.mean([r["residual_var_share"] for r in rows])),
        tolerance={"raw": float(np.mean([r["raw/tolerance"] for r in rows])),
                   "residual": float(np.mean([r["residual/tolerance"] for r in rows]))},
        w_bivariate=float(np.mean([r["cafe_cd_conditioning"].get("w_bivariate", 0.0)
                                   for r in rows])),
        arms=arms_mean, identity_checks=ident,
        traced_sample=rows[0] if rows else None,
        rows=rows, wall_s=time.time() - t0)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}  ({len(rows)} samples, {out['wall_s'] / 60:.1f} min)")
    for k, v in arms_mean.items():
        print("  %-24s AUROC %.4f  AUPRC %.4f  maxF1 %.4f"
              % (k, v["auroc"], v["auprc"], v["f1max"]))
    print("  identity checks (mean |diff| in AUROC):", json.dumps(ident))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
