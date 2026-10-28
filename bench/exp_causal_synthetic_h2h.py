"""CAFE-CD vs the published-method field, on ground truth we control.

QUESTION
    The public benchmarks each have a defect that limits what a win there means:
    CausalRivers' headline is substantially a scale prior (the same VAR moves 0.849 ->
    0.593 by switching to a scale-free statistic), and CausalDynamics' `simple` split
    sits at chance for every method (0.426-0.518 against a 0.500 null).  So we also
    need a venue where the truth is known exactly, the generator is audited, and no
    artefact is doing the work.

DESIGN
    Panels from bench/gen_causal_panel.py: latent AR factors with entity-specific
    exposures, a KNOWN sparse lagged SVAR on the idiosyncratic component, variance-
    equalised so the generator does not leak the causal order through scale
    (var-/R2-sortability audited in [0.40, 0.60]).

    Every method sees the identical data.  2-D baselines are applied per entity and
    their scores averaged -- the standard way to run a series method on a panel.

    Swept over the common-variance share chi, because that is the axis on which
    factor adjustment is known to pay off or not.

    PRIMARY METRIC is auc_cross: OFF-DIAGONAL lagged edges.  The lag-1 diagonal is
    dense by construction (every node has an AR self-loop) and trivially detectable,
    so the all-cells auc flatters any detector that only fires there.

WHAT IT WRITES
    bench/causal_synthetic_h2h_cache.json   the cache (this file writes ONLY this)
  The paper table is rendered from that cache by
    experimental/papers/causal/scripts/make_tables.py  -> tables/h2h.tex \\label{tab:h2h}

Run:  python3 bench/exp_causal_synthetic_h2h.py [--tables-only]
Env:  SH_CHIS=0.5,0.7,0.9,0.95  SH_SEEDS=4  SH_P=24  SH_E=10  SH_T=120
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_ROOT, "src"))

import cafe                                            # noqa: E402
import causal_metrics as cm                            # noqa: E402
from gen_causal_panel import gen_causal_panel          # noqa: E402
from causal_baselines import run_baseline              # noqa: E402

CACHE = os.path.join(_HERE, "causal_synthetic_h2h_cache.json")
TABLE = os.path.join(_ROOT, "paper", "tables", "causal_synthetic_h2h.tex")

CHIS  = [float(x) for x in os.environ.get("SH_CHIS", "0.5,0.7,0.9,0.95").split(",")]
SEEDS = int(os.environ.get("SH_SEEDS", "4"))
P     = int(os.environ.get("SH_P", "24"))
E     = int(os.environ.get("SH_E", "10"))
T     = int(os.environ.get("SH_T", "120"))
K     = 2
METHODS = ["var_ols", "granger_pairwise", "varlingam", "dynotears",
           "dagma_linear", "pc_ts", "pcmci"]


def baseline_scores(name, X3, max_lag):
    """Apply a 2-D series method per entity and average |score|.  (K,P,P) or None.

    Two arms are run for every baseline, because otherwise the comparison is open to
    an obvious objection:

      RAW    the baseline sees the observed panel, as it would be used in practice.
      +RESID the baseline sees the SAME CAFE idiosyncratic residuals that CAFE-CD
             consumes.  This isolates the DISCOVERY algorithm from the preprocessing:
             if CAFE-CD only wins in the RAW arm, then factor residualisation is doing
             the work and any method could have had it; if it also wins in +RESID, the
             screening/skeleton/orientation machinery is contributing.
    """
    acc, ok, t0 = None, 0, time.time()
    for e in range(X3.shape[0]):
        r = run_baseline(name, X3[e], max_lag=max_lag, timeout_s=180.0)
        if not r.ok or r.A_lag is None:
            continue
        A = np.abs(np.asarray(r.A_lag, float))
        if A.shape != (max_lag, X3.shape[2], X3.shape[2]):
            continue
        acc = A if acc is None else acc + A
        ok += 1
    if ok == 0:
        return None, time.time() - t0, 0
    return acc / ok, time.time() - t0, ok


def main():
    rows = []
    for chi in CHIS:
        for seed in range(SEEDS):
            X3, _X2, _meta, truth = gen_causal_panel(
                E=E, T=T, P=P, R=3, n_lags=K, deg=2, chi=chi,
                beta_hetero=0.6, noise="laplace", equal_variance=True, seed=seed)
            A = np.asarray(truth.A_lag, float)

            t0 = time.time()
            g = cafe.discover(X3, max_lag=K)
            t_cafe = time.time() - t0
            S = np.abs(np.asarray(g.scores(lag="lagged"), float))
            adj = np.asarray(g.adjacency(lag="lagged"), bool)
            m = cm.summarize_graph(A, S, adj)
            rows.append(dict(chi=chi, seed=seed, method="cafe_cd",
                             auc_cross=float(m["auc_cross"]), auc=float(m["auc"]),
                             f1=float(m.get("f1", np.nan)), shd=int(m.get("shd", -1)),
                             runtime_s=t_cafe, n_entities_ok=E))
            print(f"chi={chi} s={seed} cafe_cd auc_cross={m['auc_cross']:.3f} "
                  f"f1={m.get('f1', float('nan')):.3f} {t_cafe:.1f}s", flush=True)

            # the residual view CAFE-CD itself consumes -- handed to every baseline
            res = cafe.CAFE().run(X3)
            Ec, Oc = res.residuals(stack=True, factor_adjusted=True, mask_imputed=True)
            Ec = np.asarray(Ec, float).reshape(X3.shape)
            Ec = np.where(np.isfinite(Ec), Ec, 0.0)

            for name in METHODS:
              for arm, data in (("", X3), ("+resid", Ec)):
                Sb, tb, nok = baseline_scores(name, data, K)
                name_a = name + arm
                if Sb is None:
                    rows.append(dict(chi=chi, seed=seed, method=name_a, auc_cross=None,
                                     auc=None, f1=None, shd=None, runtime_s=tb,
                                     n_entities_ok=0))
                    print(f"chi={chi} s={seed} {name_a:24} FAILED", flush=True)
                    continue
                mb = cm.summarize_graph(A, Sb, None)
                rows.append(dict(chi=chi, seed=seed, method=name_a,
                                 auc_cross=float(mb["auc_cross"]), auc=float(mb["auc"]),
                                 f1=None, shd=None, runtime_s=tb, n_entities_ok=nok))
                print(f"chi={chi} s={seed} {name_a:24} auc_cross={mb['auc_cross']:.3f} "
                      f"{tb:.1f}s ({nok}/{E} entities)", flush=True)

            with open(CACHE, "w") as f:
                json.dump({"rows": rows, "meta": dict(
                    CHIS=CHIS, SEEDS=SEEDS, P=P, E=E, T=T, K=K, methods=METHODS,
                    primary="auc_cross", beta_hetero=0.6, noise="laplace")}, f, indent=1)
    print(f"\nwrote {CACHE} ({len(rows)} rows)")


if __name__ == "__main__":
    if "--tables-only" not in sys.argv:
        main()
