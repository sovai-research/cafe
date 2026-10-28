r"""
CAFE-CD diagnosis: WHEN does blending several evidence statistics beat the best one?

THE QUESTION
------------
``cafe.discovery`` scores a lagged edge by a rank-blend of three statistics
(``SCORE_WEIGHTS = (0.60, 0.20, 0.20)``):

    t_direct   the multivariate refit |t| -- fully conditional
    path       the panel group lasso's lambda-path occupancy
    screen     the backward-only screening statistic -- marginal

Those weights were calibrated on a PANEL.  On the single-series public benchmarks
(CausalRivers, CausalDynamics, CauseMe -- all ``E = 1``) CAFE-CD loses to a plain VAR,
and a component-wise read of one rivers configuration showed the refit ALONE scoring
above the blend.  That is either a property of hydrological data or a property of
``E = 1``, and the two have completely different consequences: the first is a data
mismatch we can only report, the second is a design fact we could act on.

THE CONTROLLED EXPERIMENT
-------------------------
Vary ONLY ``E``, on our own audited generator, and read the blend premium

    premium = AUC(blend) - AUC(t_direct)

The confound to kill is total sample size: an ``E = 1`` panel has ``E`` times fewer
rows, so a premium that vanished there might be about rows rather than entities.  So
the grid carries TWO arms:

    fixed-N     E * T held constant (E=1,T=1200  ...  E=20,T=60) -- the clean contrast
    fixed-T     T held constant (T=120 at every E) -- the practitioner's contrast

Everything else is pinned: ``P=24, R=3, K=2, deg=2, beta_hetero=0.6, noise='laplace',
equal_variance=True``, the same configuration the head-to-head grid uses.

WHY THE ANSWER IS NOT OBVIOUS EITHER WAY
----------------------------------------
``path`` and ``screen`` are both pooled across entities before they are ranked, so
their sampling variance falls like ``1/E`` while their bias does not.  A rank-average
of a low-variance estimator with two higher-variance ones is worth taking only while
the noise it adds is smaller than the independent signal it brings.  At ``E = 1`` the
two auxiliaries carry the same bias with ``E`` times the variance, and the blend can be
strictly worse than its own strongest component.  That is a prediction, not an
observation, until this file runs.

WHAT IS INSTRUMENTED, AND WHY IT IS A PRIVATE IMPORT
----------------------------------------------------
The component arrays are internal to the score assembler, so this file wraps the
private ``cafe.discovery._blend`` to record the arrays it is handed and then calls
through unchanged.  It measures the SHIPPED pipeline -- no reimplementation, no second
code path -- which is the whole point; ``exp_causal_sparse_gate.py`` imports
``cafe._skeleton.panel_group_lasso`` on the same principle.  The wrapper is installed
and removed around each call and never alters a return value.

METRIC
------
``auc_cross`` from ``causal_metrics.summarize_graph`` -- off-diagonal lagged edges
only.  The lag-1 diagonal is dense by construction and would flatter any detector that
only fires there.

WHAT IT WRITES
--------------
    bench/causal_replication_cache.json          (checkpointed after every cell)
    experimental/papers/causal/tables/replication.tex   \label{tab:causalreplication}

Run:
    python3 bench/exp_causal_replication.py
    python3 bench/exp_causal_replication.py --tables-only
    python3 bench/exp_causal_replication.py --force
Env knobs:
    RP_SEEDS   number of seeds per cell (default 4)
    RP_CHIS    comma-separated chi values (default 0.5,0.9)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import causal_metrics as cm  # noqa: E402
from gen_causal_panel import gen_causal_panel  # noqa: E402

import cafe.discovery as D  # noqa: E402

CACHE = os.path.join(_HERE, "causal_replication_cache.json")
TABLE = os.path.join(_ROOT, "experimental", "papers", "causal", "tables",
                     "replication.tex")

#: (E, T) pairs.  The first block holds E*T = 1200 fixed; the second holds T = 120.
GRID_FIXED_N = [(1, 1200), (2, 600), (4, 300), (10, 120), (20, 60)]
GRID_FIXED_T = [(1, 120), (2, 120), (4, 120), (10, 120), (20, 120)]
CHIS = [float(x) for x in os.environ.get("RP_CHIS", "0.5,0.9").split(",")]
SEEDS = int(os.environ.get("RP_SEEDS", "4"))
P, R, K, DEG = 24, 3, 2, 2
COMPONENTS = ["t_direct", "t_biv", "path", "screen"]


# --------------------------------------------------------------------------- #
def _auc_cross(A, truth_lagged):
    """Off-diagonal lagged AUC of one component array against the true ``A_lag``."""
    S = np.abs(np.asarray(A, float))
    if S.ndim == 2:
        S = S[None]
    return float(cm.summarize_graph(np.asarray(truth_lagged, float), S, None)["auc_cross"])


class _BlendSpy:
    """Record the components handed to the shipped score assembler; call through."""

    def __init__(self):
        self.calls = []
        self._orig = None

    def __enter__(self):
        self._orig = D._blend

        def spy(components):
            self.calls.append(components)
            return self._orig(components)

        D._blend = spy
        return self

    def __exit__(self, *exc):
        D._blend = self._orig
        return False

    def lagged(self):
        """The 4-component call is the lagged blend; the 2-component one is contemp."""
        for c in self.calls:
            if len(c) == len(COMPONENTS):
                return c
        return self.calls[0] if self.calls else []


def run_cell(E, T, chi, seed):
    X3, _x2, _meta, truth = gen_causal_panel(
        E=E, T=T, P=P, R=R, n_lags=K, deg=DEG, chi=chi, beta_hetero=0.6,
        noise="laplace", equal_variance=True, seed=seed)
    A = np.asarray(truth.A_lag, float)
    t0 = time.perf_counter()
    with _BlendSpy() as spy:
        g = D.discover(X3, max_lag=K)
    dt = time.perf_counter() - t0
    comps = {}
    for name, (w, arr) in zip(COMPONENTS, spy.lagged()):
        comps[name] = dict(weight=float(w),
                           auc=None if arr is None else _auc_cross(arr, A))
    blend = _auc_cross(np.asarray(g.scores(lag="lagged"), float), A)
    return dict(E=E, T=T, chi=chi, seed=seed, blend=blend, components=comps,
                runtime_s=dt)


# --------------------------------------------------------------------------- #
def _key(r):
    return (int(r["E"]), int(r["T"]), float(r["chi"]), int(r["seed"]))


def main(force=False):
    rows = []
    if os.path.exists(CACHE) and not force:
        rows = json.load(open(CACHE)).get("rows", [])
    have = {_key(r) for r in rows}
    cells = [(E, T, "fixed-N") for E, T in GRID_FIXED_N]
    cells += [(E, T, "fixed-T") for E, T in GRID_FIXED_T if (E, T) not in GRID_FIXED_N]
    for chi in CHIS:
        for seed in range(SEEDS):
            for E, T, arm in cells:
                if (E, T, chi, seed) in have:
                    continue
                r = run_cell(E, T, chi, seed)
                r["arm"] = arm
                rows.append(r)
                have.add(_key(r))
                td = r["components"]["t_direct"]["auc"]
                print(f"E={E:3d} T={T:5d} chi={chi} s={seed}  blend={r['blend']:.4f} "
                      f"t_direct={td:.4f}  premium={r['blend'] - td:+.4f} "
                      f"({r['runtime_s']:.1f}s)", flush=True)
                with open(CACHE, "w") as f:
                    json.dump({"rows": rows, "meta": _meta()}, f, indent=1)
    return rows


def _meta():
    return dict(P=P, R=R, K=K, deg=DEG, chis=CHIS, seeds=SEEDS,
                grid_fixed_N=GRID_FIXED_N, grid_fixed_T=GRID_FIXED_T,
                beta_hetero=0.6, noise="laplace", equal_variance=True,
                metric="auc_cross", weights=list(D.SCORE_WEIGHTS))


def _fmt(x, nd=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def tables(rows):
    by = {}
    for r in rows:
        by.setdefault((r.get("arm", "fixed-N"), int(r["E"]), int(r["T"])), []).append(r)
    lines = [
        r"% generated by bench/exp_causal_replication.py -- do not edit",
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{\textbf{The evidence blend needs cross-entity replication.} Same "
        r"generator, same $\chi$, same $P{=}24$; only the number of entities $E$ "
        r"varies. \emph{Premium} is $\mathrm{AUC}(\text{blend}) - "
        r"\mathrm{AUC}(t_{\text{direct}})$: positive means blending the three evidence "
        r"statistics beats the strongest one alone. The \textsc{fixed-}$N$ block holds "
        r"$ET$ constant so the contrast is entities, not rows. Off-diagonal lagged "
        r"\textsc{auc}, mean over $\chi \in \{0.5, 0.9\}$ and seeds.}",
        r"\label{tab:causalreplication}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"arm & $E \times T$ & $t_{\text{direct}}$ & path & screen & blend & premium \\",
        r"\midrule",
    ]
    for arm in ("fixed-N", "fixed-T"):
        keys = sorted([k for k in by if k[0] == arm], key=lambda k: k[1])
        if not keys:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{arm}}}}} \\")
        for k in keys:
            rs = by[k]
            def m(f):
                v = [f(r) for r in rs]
                v = [x for x in v if x is not None and np.isfinite(x)]
                return float(np.mean(v)) if v else None
            td = m(lambda r: r["components"]["t_direct"]["auc"])
            pa = m(lambda r: r["components"]["path"]["auc"])
            sc = m(lambda r: r["components"]["screen"]["auc"])
            bl = m(lambda r: r["blend"])
            prem = None if (bl is None or td is None) else bl - td
            wins = sum(1 for r in rs
                       if r["blend"] - (r["components"]["t_direct"]["auc"] or 0) > 0)
            mark = r"\,$^{\ast}$" if prem is not None and prem > 0 else ""
            lines.append(
                f"{arm} & ${k[1]} \\times {k[2]}$ & {_fmt(td)} & {_fmt(pa)} & "
                f"{_fmt(sc)} & {_fmt(bl)} & {_fmt(prem, 4)}{mark} "
                f"~{{\\scriptsize({wins}/{len(rs)})}} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[wrote] {TABLE}")
    # console summary
    print(f"\n{'arm':<9} {'E x T':>10} {'t_direct':>9} {'path':>7} {'screen':>7} "
          f"{'blend':>7} {'premium':>8} wins")
    for arm in ("fixed-N", "fixed-T"):
        for k in sorted([k for k in by if k[0] == arm], key=lambda k: k[1]):
            rs = by[k]
            td = float(np.mean([r["components"]["t_direct"]["auc"] for r in rs]))
            pa = float(np.mean([r["components"]["path"]["auc"] for r in rs]))
            sc = float(np.mean([r["components"]["screen"]["auc"] for r in rs]))
            bl = float(np.mean([r["blend"] for r in rs]))
            w = sum(1 for r in rs if r["blend"] > r["components"]["t_direct"]["auc"])
            print(f"{arm:<9} {k[1]:>4} x {k[2]:<4} {td:9.4f} {pa:7.4f} {sc:7.4f} "
                  f"{bl:7.4f} {bl - td:+8.4f} {w}/{len(rs)}")


def main_entry():
    args = sys.argv[1:]
    if "--tables-only" in args:
        rows = json.load(open(CACHE)).get("rows", [])
    else:
        rows = main(force="--force" in args)
    tables(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main_entry())
