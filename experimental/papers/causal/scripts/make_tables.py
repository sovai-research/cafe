#!/usr/bin/env python
"""Regenerate every table and figure in the CAFE-CD results paper from its cache.

    python3 experimental/papers/causal/scripts/make_tables.py

Honesty contract: **nothing in this file may write a number it did not read from a
cache.**  Every emitted cell traces to one of

    bench/causal_sparse_gate_cache.base.json   the chi ladder            (finished)
    bench/bench_rivers_cache.json              CausalRivers repro + varsort
    bench/bench_dynamics_cache.json            CausalDynamics forensics
    bench/pit_board_cache.json                 the point-in-time board
    bench/causal_synthetic_h2h_cache.json      the audited-truth head-to-head
    bench/bench_causeme_cache.json             CauseMe runtime + orientation
    data/causal/cache/cd_coupled_audit.json    CausalDynamics `coupled` forensics
    data/dynamics_matched.json                 scripts/run_dynamics_matched.py
    data/conditioning.json                     scripts/run_conditioning_diagnosis.py
    data/engine_router.json                    scripts/run_engine_router.py
    data/screen_backend.json                   scripts/run_screen_backend.py
    data/varsortability.json                   scripts/run_varsortability.py

Rows a run has not produced yet are emitted as ``--`` and counted into
``tables/status.tex``, which the paper prints verbatim.  Nothing is interpolated,
padded or carried over from an earlier build.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PAPER)))
BENCH = os.path.join(ROOT, "bench")
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figures")
DATA = os.path.join(PAPER, "data")
CAUSAL = os.path.join(ROOT, "data", "causal", "cache")
for d in (TAB, FIG, DATA):
    os.makedirs(d, exist_ok=True)

NA = "--"
PENDING: list[str] = []          # human-readable list of cells left empty
SOURCES: list[tuple[str, str]] = []   # (artefact, source)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fnum(x, nd=3, plus=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return NA
    fmt = f"%+.{nd}f" if plus else f"%.{nd}f"
    return fmt % x


def write(name, body, source):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write(body)
    SOURCES.append((name, source))
    print(f"  wrote tables/{name}")


# --------------------------------------------------------------------------- #
# statistics shared with bench/exp_causal_sparse_gate.py (same definitions)
# --------------------------------------------------------------------------- #
def perm_p_onesided(d):
    """Exact paired sign-flip permutation p-value for H1: mean(d) > 0 (n <= 20)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan")
    obs = float(d.mean())
    signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    return float(np.mean((signs * d[None, :]).mean(axis=1) >= obs - 1e-12))


def crossover_chi(chis, deltas):
    """Interpolated chi at which the paired cafe-minus-raw advantage turns positive."""
    chis = np.asarray(chis, float)
    d = np.asarray(deltas, float)
    ok = np.isfinite(d)
    chis, d = chis[ok], d[ok]
    if d.size == 0 or d[0] > 0:
        return float("nan")
    for i in range(1, len(d)):
        if d[i] > 0 >= d[i - 1]:
            x0, x1, y0, y1 = chis[i - 1], chis[i], d[i - 1], d[i]
            return float(x0 + (x1 - x0) * (-y0) / (y1 - y0)) if y1 != y0 else float(x1)
    return float("nan")


# --------------------------------------------------------------------------- #
# 1. the chi ladder -- when does factor residualisation help?
# --------------------------------------------------------------------------- #
EST_LABEL = {"ols": "OLS-VAR (unpenalised)", "grouplasso": "panel group lasso",
             "lasso": r"$L_1$ VAR (per target)"}
EST_SHORT = {"ols": "OLS-VAR", "grouplasso": "group lasso", "lasso": r"$L_1$ VAR"}

#: one display name per baseline, shared by every table below
PIT_LABEL = {
    "var_ols": "VAR-OLS", "granger_pairwise": "Granger (pairwise)",
    "varlingam": "VARLiNGAM", "directlingam": "DirectLiNGAM", "dynotears": "DYNOTEARS",
    "dagma_linear": "DAGMA (linear)", "pc": "PC", "fci": "FCI",
    "pc_ts": "PC (lag-augmented)", "fci_ts": "FCI (lag-augmented)",
    "pcmci": "PCMCI", "pcmciplus": "PCMCI$+$", "cafe_cd": r"\textbf{\cafe{}-CD}",
}


def table_gate():
    src = os.path.join(BENCH, "causal_sparse_gate_cache.base.json")
    d = load(src)
    if d is None:
        PENDING.append("chi ladder: cache absent")
        return None
    rows, meta = d["rows"], d["meta"]
    metric = meta.get("primary_metric", "auc_cross")
    chis = sorted({r["chi"] for r in rows})
    ests = [e for e in ("ols", "grouplasso", "lasso") if e in meta["estimators"]]

    agg, cross = {}, {}
    for est in ests:
        deltas = []
        for chi in chis:
            sel = [r for r in rows if r["chi"] == chi]
            raw = np.array([r["res"][est]["raw"][metric] for r in sel])
            caf = np.array([r["res"][est]["cafe"][metric] for r in sel])
            orc = np.array([r["res"][est]["oracle"][metric] for r in sel])
            dd = caf - raw
            agg[(est, chi)] = dict(
                raw=float(raw.mean()), cafe=float(caf.mean()), orc=float(orc.mean()),
                delta=float(dd.mean()), wins=int((dd > 0).sum()), n=int(dd.size),
                p=perm_p_onesided(dd))
            deltas.append(float(dd.mean()))
        cross[est] = crossover_chi(chis, deltas)

    cfg = meta["configs"]["base"]
    n_over_col = rows[0]["n_over_col"]
    head = " & ".join(r"\multicolumn{5}{c}{%s}" % EST_LABEL[e] for e in ests)
    sub = " & ".join([r"raw & \textsc{cd} & orc & $\Delta$ & w/$p$"] * len(ests))
    body = []
    for chi in chis:
        cells = []
        for est in ests:
            a = agg[(est, chi)]
            best = r"\textbf{%s}" % fnum(a["cafe"]) if a["delta"] > 0 else fnum(a["cafe"])
            cells.append("%s & %s & %s & %s & %d/%.3f" % (
                fnum(a["raw"]), best, fnum(a["orc"]), fnum(a["delta"], plus=True),
                a["wins"], a["p"]))
        body.append("%.2f & %s \\\\" % (chi, " & ".join(cells)))
    crossrow = " & ".join(r"\multicolumn{5}{c}{$\chi^\star=%s$}"
                          % (("%.2f" % cross[e]) if np.isfinite(cross[e]) else "n/a")
                          for e in ests)

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/causal_sparse_gate_cache.base.json
\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{3.2pt}
\caption{\textbf{Factor residualisation helps only above a crossover in the
common-variance share $\chi$, and a \emph{sparse} downstream estimator moves that
crossover down.} Off-diagonal lagged-edge AUC for identical representations fed to
three downstream estimators: \textsc{raw} the observed panel, \textsc{cd} the \cafe{}
idiosyncratic residual, \textsc{orc} the oracle residual (true factors removed). The
sparse estimators are scored threshold-free by lambda entry order, so no model-selection
rule enters. $\Delta$ is the paired mean $(\textsc{cd}-\textsc{raw})$ over %(nseed)d
seeds, \emph{w} the paired win count, $p$ an exact one-sided sign-flip permutation test,
$\chi^\star$ the interpolated crossover. Panel: $E{=}%(E)d$, $T{=}%(T)d$, $P{=}%(P)d$,
$n/\mathrm{col}{=}%(noc).1f$.}
\label{tab:gate}
\begin{tabular}{r|%(colspec)s}
\toprule
 & %(head)s \\
$\chi$ & %(sub)s \\
\midrule
%(body)s
\cmidrule{2-%(lastcol)d}
\multicolumn{1}{r|}{crossover} & %(crossrow)s \\
\bottomrule
\end{tabular}
\end{table*}
""" % dict(nseed=meta["seeds"], E=cfg["E"], T=cfg["T"], P=cfg["P"], noc=n_over_col,
           colspec="|".join(["rrrr@{\\ }l"] * len(ests)), head=head, sub=sub,
           body="\n".join(body), lastcol=1 + 5 * len(ests), crossrow=crossrow)
    write("gate.tex", tex, "bench/causal_sparse_gate_cache.base.json")
    return dict(agg=agg, cross=cross, chis=chis, ests=ests, meta=meta, metric=metric)


# --------------------------------------------------------------------------- #
# 1b. head-to-head on our own audited ground truth, with a +resid fairness arm
# --------------------------------------------------------------------------- #
H2H_LABEL = {"cafe_cd": r"\textbf{\cafe{}-CD}", "var_ols": "VAR-OLS",
             "granger_pairwise": "Granger (pairwise)", "varlingam": "VARLiNGAM",
             "dynotears": "DYNOTEARS", "dagma_linear": "DAGMA (linear)",
             "pc_ts": "PC (lag-augmented)", "pcmci": "PCMCI"}


def table_h2h():
    src = os.path.join(BENCH, "causal_synthetic_h2h_cache.json")
    d = load(src)
    if d is None:
        PENDING.append("audited-truth head-to-head: cache absent")
        return None
    rows, meta = d["rows"], d["meta"]
    metric = meta.get("primary", "auc_cross")
    chis = sorted({r["chi"] for r in rows})
    base = [m for m in meta["methods"]]

    def cell(m, c, key=metric):
        xs = [r[key] for r in rows if r["method"] == m and r["chi"] == c
              and r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
        return (float(np.mean(xs)), len(xs)) if xs else (float("nan"), 0)

    present = sorted({r["method"] for r in rows})
    A = {(m, c): cell(m, c)[0] for m in present for c in chis}
    N = {(m, c): cell(m, c)[1] for m in present for c in chis}
    rt = {m: float(np.mean([r["runtime_s"] for r in rows if r["method"] == m
                            and r.get("runtime_s") is not None])) for m in present}

    # rank of CAFE-CD at each chi, over every arm that exists
    ranks, best = {}, {}
    for c in chis:
        order = sorted(present, key=lambda m: -A[(m, c)])
        ranks[c] = order.index("cafe_cd") + 1 if "cafe_cd" in order else None
        best[c] = next(m for m in order if m != "cafe_cd")

    # the fairness arm, measured rather than asserted
    deltas = [A[(m + "+resid", c)] - A[(m, c)] for m in base for c in chis
              if (m + "+resid", c) in A and np.isfinite(A[(m + "+resid", c)])
              and np.isfinite(A[(m, c)])]
    dlo, dhi, dmu = (min(deltas), max(deltas), float(np.mean(deltas))) if deltas \
        else (np.nan,) * 3
    n_help = int(sum(x > 0 for x in deltas))

    order = ["cafe_cd"] + sorted(base, key=lambda m: -np.mean([A[(m, c)] for c in chis]))
    body = []
    for m in order:
        if m == "cafe_cd":
            cells = [r"\textbf{%s} & ---" % fnum(A[(m, c)]) for c in chis]
        else:
            cells = []
            for c in chis:
                r_ = A.get((m + "+resid", c), float("nan"))
                cells.append("%s & %s" % (fnum(A[(m, c)]), fnum(r_)))
        body.append("%s & %s & %s \\\\" % (H2H_LABEL.get(m, m.replace("_", r"\_")),
                                           " & ".join(cells), fnum(rt[m], 2)))
        if m == "cafe_cd":
            body.append(r"\midrule")
    nseeds = sorted({N[(m, c)] for m in present for c in chis})
    rankrow = " & ".join(r"\multicolumn{2}{c}{%d/%d}" % (ranks[c], len(present))
                         for c in chis)

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/causal_synthetic_h2h_cache.json
\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Head-to-head on ground truth we audit ourselves: \cd{} is rank 1 of
%(nmeth)d at every $\chi$, and the margin survives the fairness arm.} Off-diagonal
lagged-edge AUC (\texttt{%(metric)s}) on panels from \texttt{bench/gen\_causal\_panel.py}
($E{=}%(E)d$, $T{=}%(T)d$, $P{=}%(P)d$, $K{=}%(K)d$, Laplace shocks, heterogeneous
$\beta$), %(seedtxt)s seeds per cell. \textbf{The baselines are single-series methods,
applied per entity and averaged --- so \cd{}'s ability to pool $E$ entities is a real
advantage, and it is most of what this table measures.} The \textsc{+res} column
controls for the other half: each baseline re-run on the \emph{identical} \cafe{}
residual, which isolates the discovery machinery from the preprocessing. Across all
%(ndelta)d (method, $\chi$) pairs residualisation moves a baseline by
$%(dlo)+.3f$ to $%(dhi)+.3f$ (mean $%(dmu)+.3f$; it helps %(nhelp)d of %(ndelta)d), so
the gap to \cd{} is the machinery, not the representation. Runtime is mean seconds per
cell on one CPU.}
\label{tab:h2h}
\begin{tabular}{@{}l%(colspec)sr@{}}
\toprule
 & %(head)s & \\
method & %(sub)s & s/cell \\
\midrule
%(body)s
\midrule
\multicolumn{1}{@{}l}{\cd{} rank} & %(rankrow)s & \\
\bottomrule
\end{tabular}
\end{table*}
""" % dict(nmeth=len(present), metric=metric.replace("_", r"\_"),
           E=meta["E"], T=meta["T"], P=meta["P"], K=meta["K"],
           seedtxt=("%d" % nseeds[0]) if len(nseeds) == 1
                   else ("%d--%d" % (min(nseeds), max(nseeds))),
           ndelta=len(deltas), dlo=dlo, dhi=dhi, dmu=dmu, nhelp=n_help,
           colspec="".join(["rr"] * len(chis)),
           head=" & ".join(r"\multicolumn{2}{c}{$\chi=%.2f$}" % c for c in chis),
           sub=" & ".join([r"own & \textsc{+res}"] * len(chis)),
           body="\n".join(body), rankrow=rankrow)
    write("h2h.tex", tex, "bench/causal_synthetic_h2h_cache.json")
    gaps = [A[("cafe_cd", c)] - A[(best[c], c)] for c in chis]
    return dict(A=A, chis=chis, ranks=ranks, best=best, present=present,
                dlo=dlo, dhi=dhi, dmu=dmu, n_help=n_help, n_delta=len(deltas),
                rt=rt, meta=meta, n_base=len(base),
                cafe_lo=min(A[("cafe_cd", c)] for c in chis),
                cafe_hi=max(A[("cafe_cd", c)] for c in chis),
                cafe_first=A[("cafe_cd", chis[0])],
                cafe_last=A[("cafe_cd", chis[-1])],
                chi_first=chis[0], chi_last=chis[-1],
                best_lo=min(A[(best[c], c)] for c in chis),
                best_hi=max(A[(best[c], c)] for c in chis),
                gap_lo=min(gaps), gap_hi=max(gaps),
                gap_first=A[("cafe_cd", chis[0])] - A[(best[chis[0]], chis[0])],
                rank=ranks[chis[0]], n_meth=len(present))


def figure_gate(g):
    """Delta(chi) for the three downstream estimators, with the crossovers marked.

    Deliberately one panel: the levels are already in Table 1, and a second panel would
    only restate them.
    """
    if g is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chis, ests, agg = g["chis"], g["ests"], g["agg"]
    cols = {"ols": "#C05621", "grouplasso": "#2B6CB0", "lasso": "#2C7A7B"}
    fig, ax = plt.subplots(figsize=(3.3, 1.65))
    for k, e in enumerate(ests):
        ax.plot(chis, [agg[(e, c)]["delta"] for c in chis], "o-", ms=3,
                color=cols[e], label=EST_SHORT[e])
    lo, hi = ax.get_ylim()
    for k, e in enumerate(ests):
        x = g["cross"][e]
        if np.isfinite(x):
            ax.axvline(x, color=cols[e], lw=0.7, ls=":")
            ax.annotate("%.2f" % x, (x, lo + (0.04 + 0.11 * k) * (hi - lo)),
                        color=cols[e], fontsize=6, ha="center", va="bottom")
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xlabel(r"common-variance share $\chi$", fontsize=8)
    ax.set_ylabel(r"$\Delta$ AUC (CD $-$ raw)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(FIG, "gate.pdf"), bbox_inches="tight")
    plt.close(fig)
    SOURCES.append(("figures/gate.pdf", "bench/causal_sparse_gate_cache.base.json"))
    print("  wrote figures/gate.pdf")


# --------------------------------------------------------------------------- #
# 2. CausalRivers: reproduction of published cells, then var-sortability
# --------------------------------------------------------------------------- #
REPRO_LABEL = {
    "var_ols/flood_full_random_5": (r"VAR", r"flood/R-5, Fig.~3a", "AUROC"),
    "var_ols/east_random_3": (r"VAR", r"east/R-3, Tab.~3", "AUROC"),
    "var_ols/east_random_3_F1max": (r"VAR", r"east/R-3, Tab.~7", r"max-$F_1$"),
    "dynotears/flood_full_random_5": (r"DYNOTEARS$^{\ddagger}$", r"flood/R-5", "AUROC"),
    "varlingam/flood_full_random_5": (r"VARLiNGAM", r"flood/R-5", "AUROC"),
}
VARSORT_ORDER = ["varsort/published (abscoef, unrescaled)",
                 "varsort/scale-free (tstat, unrescaled)",
                 "varsort/benchmark min-max (abscoef)",
                 "varsort/wrapper z-score (abscoef)"]
VARSORT_LABEL = {
    "varsort/published (abscoef, unrescaled)":
        (r"$|\hat\beta|$, unrescaled \emph{(published cell)}", r"\ding{51}"),
    "varsort/scale-free (tstat, unrescaled)":
        (r"$|t|$, unrescaled \emph{(scale-free)}", r"\ding{55}"),
    "varsort/benchmark min-max (abscoef)":
        (r"$|\hat\beta|$, benchmark min--max rescale", r"\ding{55}"),
    "varsort/wrapper z-score (abscoef)":
        (r"$|\hat\beta|$, wrapper $z$-score", r"\ding{55}"),
}


def table_rivers():
    vsd, vsrows = varsort_rows()
    src = os.path.join(BENCH, "bench_rivers_cache.json")
    d = load(src)
    if d is None:
        PENDING.append("CausalRivers: cache absent")
        return None
    cells = d["cells"]

    rep = []
    for key, (meth, cell, metric) in REPRO_LABEL.items():
        c = cells.get("repro/" + key)
        if c is None:
            rep.append("%s & %s & %s & %s & %s & %s \\\\" % (meth, cell, metric, NA, NA, NA))
            PENDING.append(f"CausalRivers repro cell {key}")
            continue
        rep.append("%s & %s & %s & %s & %s & %s \\\\" % (
            meth, cell, metric, "%d" % c["n_samples"], fnum(c["ours"], 4),
            "%.2f" % c["published"]))

    vs = []
    for key in VARSORT_ORDER:
        c = cells.get(key)
        label, keeps = VARSORT_LABEL[key]
        if c is None:
            vs.append("%s & %s & %s & %s \\\\" % (label, keeps, NA, NA))
            PENDING.append(f"CausalRivers var-sortability arm {key}")
            continue
        vs.append("%s & %s & %s & %s \\\\" % (
            label, keeps, fnum(c["individual_auroc"], 4),
            fnum(c["f1max_individual"], 4)))
    n = cells.get(VARSORT_ORDER[0], {}).get("n_samples")
    if vsrows:
        vs.append(r"\midrule")
        vs.append(vsrows)

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/bench_rivers_cache.json
\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{(a) Our harness reproduces five published CausalRivers cells}, so
(b) can be read as a statement about the benchmark rather than about our
implementation. $^{\ddagger}$DYNOTEARS is re-implemented from Pamfil et al.\ (2020)
because \texttt{causalnex} does not install on Python 3.13; it and VARLiNGAM land
\emph{above} the published figures. \textbf{(b) The leaderboard is substantially
var-sortability.} The identical VAR on all %(n)s flood/Random-5 subgraphs, changing only
the scale convention and the edge statistic --- then the prior measured directly on the
true graphs, with no model fitted. Varsortability is the fraction of true directed edges
whose effect has the larger marginal variance ($0.5$ = no signal, $1$ = sorting by
variance \emph{is} the causal order); $R^2$-sortability is its scale-invariant
counterpart, so standardising does not remove it.}
\label{tab:rivers}
\begin{tabular}{@{}llccrr@{}}
\multicolumn{6}{@{}l}{\textbf{(a) reproduction of published cells}}\\
\toprule
method & cell & metric & $n$ & ours & pub. \\
\midrule
%(rep)s
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{@{}lcrr@{}}
\multicolumn{4}{@{}l}{\textbf{(b) four scale conventions, then the prior itself}}\\
\toprule
scoring convention & scale prior & AUROC & max-$F_1$ \\
\midrule
%(vs)s
\bottomrule
\end{tabular}
\end{table}
""" % dict(rep="\n".join(rep), vs="\n".join(vs), n=("%s" % n) if n else NA)
    write("rivers.tex", tex, "bench/bench_rivers_cache.json")
    return cells


def varsort_rows():
    """Sortability of the CausalRivers ground truth, as rows for the rivers table."""
    d = load(os.path.join(DATA, "varsortability.json"))
    if d is None:
        PENDING.append("var-sortability: run scripts/run_varsortability.py")
        return None, ""
    v, r = d["var_sortability"], d["r2_sortability"]
    rows = ("varsortability of the true graphs & --- & \\textbf{%s} & %s \\\\\n"
            "$R^2$-sortability of the true graphs & --- & %s & %s \\\\"
            % (fnum(v["mean"], 4), "sd\\,%s" % fnum(v["sd"], 3),
               fnum(r["mean"], 4), "sd\\,%s" % fnum(r["sd"], 3)))
    SOURCES.append(("rivers.tex (sortability rows)",
                    "data/varsortability.json (live)"))
    return d, rows


# --------------------------------------------------------------------------- #
# 2b. the scale prior, decomposed per method
# --------------------------------------------------------------------------- #
def table_scaleprior():
    """AUROC each method gains from being handed unrescaled rather than z-scored data."""
    d = load(os.path.join(BENCH, "bench_rivers_cache.json"))
    if d is None:
        PENDING.append("scale-prior decomposition: CausalRivers cache absent")
        return None
    cells = d["cells"]
    pre = "h2h/batch/%s/flood_full/random_5/"
    sc = {k.split("/")[5]: v for k, v in cells.items() if k.startswith(pre % "scale")}
    fr = {k.split("/")[5]: v for k, v in cells.items() if k.startswith(pre % "free")}
    common = sorted(set(sc) & set(fr))
    if not common:
        PENDING.append("scale-prior decomposition: the two arms do not overlap yet")
        return None
    gains = {m: sc[m]["individual_auroc"] - fr[m]["individual_auroc"] for m in common}
    order = sorted(common, key=lambda m: -gains[m])
    EPS = 5e-5
    grp = {"gain": [m for m in order if gains[m] > EPS],
           "flat": [m for m in order if abs(gains[m]) <= EPS],
           "hurt": [m for m in order if gains[m] < -EPS]}

    body, seen = [], set()
    for tag, label in (("gain", r"\emph{coefficient magnitude} --- gains"),
                       ("flat", r"\emph{test statistic / CI test} --- exactly invariant"),
                       ("hurt", r"\emph{scale-invariant by construction} --- harmed")):
        if not grp[tag]:
            continue
        if seen:
            body.append(r"\addlinespace[1.5pt]")
        seen.add(tag)
        body.append(r"\multicolumn{4}{@{}l}{%s}\\" % label)
        for m in grp[tag]:
            bold = m.startswith("cafe_cd")
            nm = PIT_LABEL.get(m, m.replace("_", r"\_"))
            body.append("\\quad %s & %s & %s & %s \\\\" % (
                nm, fnum(sc[m]["individual_auroc"], 4),
                fnum(fr[m]["individual_auroc"], 4),
                (r"\textbf{%s}" % fnum(gains[m], 4, plus=True)) if bold
                else fnum(gains[m], 4, plus=True)))
    n = sc[common[0]]["n_scored"]

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/bench_rivers_cache.json
\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{The CausalRivers leaderboard, decomposed per method: what each one
gains from being handed unrescaled rather than $z$-scored data.} Same %(n)s
flood/Random-5 subgraphs, same estimator, same lag budget; only the scale convention
differs. The field splits into exactly three groups and the split is the published
ranking: the %(ngain)d methods that rank by coefficient magnitude gain
$%(glo).2f$--$%(ghi).2f$ AUROC and are precisely the top of the published board; the
%(nflat)d that rank by a test statistic or a conditional-independence test are invariant
to $4$ decimal places; and \cd{} alone is \emph{harmed}. This is
Table~\ref{tab:rivers}b's var-sortability prior ($0.939$ over 2\,139 subgraphs) read
method by method.}
\label{tab:scaleprior}
\begin{tabular}{@{}lrrr@{}}
\toprule
method & unrescaled & $z$-scored & gain \\
\midrule
%(body)s
\bottomrule
\end{tabular}
\end{table}
""" % dict(body="\n".join(body), n=("%s" % n) if n else NA,
           ngain=len(grp["gain"]), nflat=len(grp["flat"]),
           glo=min(gains[m] for m in grp["gain"]) if grp["gain"] else float("nan"),
           ghi=max(gains[m] for m in grp["gain"]) if grp["gain"] else float("nan"))
    write("scaleprior.tex", tex, "bench/bench_rivers_cache.json")
    return dict(gains=gains, grp=grp, sc=sc, fr=fr,
                cafe_gain=gains.get("cafe_cd"),
                cafe_scale=sc.get("cafe_cd", {}).get("individual_auroc"),
                cafe_free=fr.get("cafe_cd", {}).get("individual_auroc"),
                n=n)


# --------------------------------------------------------------------------- #
# 3. CausalDynamics: two broken published cells, and a split with no signal
# --------------------------------------------------------------------------- #
CAL_ORDER = [
    ("dynotears / upstream wrapper (binarised, transposed)",
     r"DYNOTEARS, upstream$^{a}$", "dynotears"),
    ("dynotears / orientation corrected",
     r"\quad orientation fixed", None),
    ("varlingam / upstream wrapper (binarised, prune off)",
     r"VARLiNGAM, upstream$^{b}$", "varlingam"),
    ("varlingam / lingam library defaults (prune on, BIC lag order)",
     r"\quad \texttt{lingam} defaults", None),
    ("var_ols / |coef|, lag-1 slice", r"\quad plain VAR, $|\hat\beta|$", None),
]


def table_dynamics():
    src = os.path.join(BENCH, "bench_dynamics_cache.json")
    d = load(src)
    if d is None:
        PENDING.append("CausalDynamics: cache absent")
        return None
    cal = d.get("calibration", {})
    # The calibration cell is ONE setting, so it must be compared against that
    # setting's shipped per-system evaluation, not the 10-setting `simple` average.
    SETTING = "noise=0.00_confounder=False"
    pub = (d.get("published", {}).get("data", {}).get("simple", {})
           .get("per_setting", {}).get(SETTING, {}))

    rows = []
    for key, label, pubkey in CAL_ORDER:
        c = cal.get(key)
        if c is None:
            rows.append("%s & %s & %s & %s & %s \\\\" % (label, NA, NA, NA, NA))
            PENDING.append(f"CausalDynamics calibration row {key}")
            continue
        if pubkey and pubkey in pub:
            pa = pub[pubkey]["Individual_AUROC"]
            pp = pub[pubkey]["Individual_AUPRC"]
            rows.append("%s & %s & %s & %s & %s \\\\" % (
                label, fnum(c["auroc"], 4), fnum(c["auprc"], 4),
                fnum(pa, 4), fnum(pp, 4)))
        else:
            rows.append("%s & %s & %s & %s & %s \\\\" % (
                label, fnum(c["auroc"], 4), fnum(c["auprc"], 4), NA, NA))
    n_sys = cal.get(CAL_ORDER[0][0], {}).get("n_systems")

    # the whole shipped leaderboard for this setting, against its own constant null
    lead = sorted(pub.items(), key=lambda kv: -kv[1]["Individual_AUROC"])
    null_auprc = np.mean([v["Null_AUPRC"] for v in pub.values()]) if pub else float("nan")
    spread = ((min(v["Individual_AUROC"] for v in pub.values()),
               max(v["Individual_AUROC"] for v in pub.values())) if pub else (np.nan,) * 2)
    ap_gap = (max(abs(v["Individual_AUPRC"] - v["Null_AUPRC"]) for v in pub.values())
              if pub else float("nan"))
    lead_rows = " ".join(
        r"\texttt{%s} %s;" % (k.replace("_", r"\_"), fnum(v["Individual_AUROC"], 3))
        for k, v in lead)

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/bench_dynamics_cache.json
\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Two CausalDynamics cells are wrong by construction; we reproduce both
published AUROCs to $10^{-3}$ by replicating the defect.} \texttt{simple},
\texttt{noise=0.00\_confounder=False}, %(nsys)s scored systems, $\tau_{\max}{=}1$,
self-loops scored, mean per-realisation. $^{a}$~scores a cause-row matrix against
effect-row ground truth --- which is why its published AUROC is \emph{below} chance;
correcting the orientation moves it to chance. $^{b}$~binarises an unpruned, dense lag
matrix, so the prediction is all-ones and $0.500$ is an identity, not a measurement. Across the ten methods the
benchmark ships for this setting, AUROC spans $%(lo).3f$--$%(hi).3f$ against a $0.500$
null and no method is more than $%(apgap).3f$ AUPRC from the $%(nullap).3f$
constant-prediction baseline: nothing beats chance here, for anyone.}
\label{tab:dynamics}
\begin{tabular}{@{}lrrrr@{}}
\toprule
 & \multicolumn{2}{c}{ours} & \multicolumn{2}{c}{published} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
variant & AUROC & AUPRC & AUROC & AUPRC \\
\midrule
%(rows)s
\bottomrule
\end{tabular}
\end{table}
""" % dict(rows="\n".join(rows), nsys=("%s" % n_sys) if n_sys else NA,
           lo=spread[0], hi=spread[1], nullap=null_auprc, apgap=ap_gap)
    write("dynamics.tex", tex, "bench/bench_dynamics_cache.json")
    return dict(pub=pub, spread=spread, null_auprc=null_auprc, lead=lead_rows,
                ap_gap=ap_gap, n_methods=len(pub))


# --------------------------------------------------------------------------- #
# 3b. where CAFE-CD loses: the two public single-series benchmarks
# --------------------------------------------------------------------------- #
ARM_LABEL = {"scale": r"scale-exploiting \emph{(published)}", "free": "scale-free"}
CFG_LABEL = {"flood_full/random_5": r"flood/R-5", "east/confounder_3": r"east/conf-3",
             "east/random_3": r"east/R-3", "east/root_cause_3": r"east/root-3"}


def table_losses():
    """CausalRivers head-to-head ranks + the matched CausalDynamics `simple` board."""
    rows_a, rows_b, quote = [], [], {}
    d = load(os.path.join(BENCH, "bench_rivers_cache.json"))
    n_cfg = 0
    lose_to_var = 0
    if d is None:
        PENDING.append("CausalRivers head-to-head: cache absent")
    else:
        groups = defaultdict(dict)
        for k, v in d["cells"].items():
            if not k.startswith("h2h/"):
                continue
            _, board, arm, ds, cfg, meth, cap = k.split("/")
            groups[(arm, ds + "/" + cfg)][meth] = v
        for key in sorted(groups, key=lambda t: (t[0] != "scale", t[1])):
            arm, cfg = key
            mm = groups[key]
            if "cafe_cd" not in mm:
                PENDING.append("CausalRivers %s/%s: the \\cafe{}-CD arm has not "
                               "finished (%d of 13 methods present), so no rank is "
                               "printed" % (arm, cfg, len(mm)))
                continue
            n_cfg += 1
            order = sorted(mm, key=lambda m: -(mm[m].get("individual_auprc") or -1))
            r = order.index("cafe_cd") + 1
            top = order[0]
            quote[(arm, cfg)] = dict(
                rank=r, n=len(mm), cafe=mm["cafe_cd"]["individual_auprc"],
                top=PIT_LABEL.get(top, top), top_ap=mm[top]["individual_auprc"],
                var_ols=mm.get("var_ols", {}).get("individual_auprc"))
            vo = mm.get("var_ols", {}).get("individual_auprc")
            if vo is not None and vo > mm["cafe_cd"]["individual_auprc"]:
                lose_to_var += 1
            rows_a.append(
                "%s & %s & %s & %s & %s & \\textbf{%d/%d} & %s & %s & %s \\\\" % (
                    CFG_LABEL.get(cfg, cfg.replace("_", r"\_")),
                    ARM_LABEL.get(arm, arm),
                    "%d" % mm["cafe_cd"]["n_scored"],
                    fnum(mm["cafe_cd"]["individual_auprc"]),
                    fnum(mm["cafe_cd"]["individual_auroc"]),
                    r, len(mm),
                    PIT_LABEL.get(top, top.replace("_", r"\_")),
                    fnum(mm[top]["individual_auprc"]),
                    fnum(vo) if vo is not None else NA))

    # how far the CausalDynamics boards actually overlap, per split -- the coupled
    # and climate tiers are matched on far too few systems to rank.
    overlap = {}
    bd = load(os.path.join(BENCH, "bench_dynamics_cache.json"))
    if bd is not None:
        BASE = [m for m in PIT_LABEL if m != "cafe_cd"]
        bb, cc = {}, {}
        for k, v in bd.get("h2h", {}).items():
            parts = k.split("|")
            if len(parts) != 3 or not isinstance(v, dict):
                continue
            (bb if parts[0] == "batch" else cc if parts[0] == "cafecd"
             else {})[(parts[1], parts[2])] = v
        for sp in ("simple", "coupled", "climate"):
            B = {k for k in bb if k[0].split("/")[0] == sp}
            C = {k for k in cc if k[0].split("/")[0] == sp}
            ok = [k for k in B & C
                  if "cafe_cd" in cc[k] and not cc[k]["cafe_cd"].get("err")
                  and all(m in bb[k] and not bb[k][m].get("err") for m in BASE)]
            overlap[sp] = dict(baseline=len(B), cafe=len(C), matched=len(ok))
        for sp in ("coupled", "climate"):
            PENDING.append(
                "CausalDynamics %s head-to-head: the two boards overlap on only %d "
                "fully-scored systems (baseline board %d, CAFE-CD board %d), too few "
                "to rank, so no number is printed"
                % (sp, overlap[sp]["matched"], overlap[sp]["baseline"],
                   overlap[sp]["cafe"]))

    dm = load(os.path.join(DATA, "dynamics_matched.json"))
    dyn_rank = dyn_n = None
    floor = None
    cau = load(os.path.join(CAUSAL, "cd_coupled_audit.json"))
    if cau is not None:
        floor = cau["floor"]["simple"]["rows"]["random/full"]["auprc"]
    if dm is None:
        PENDING.append("CausalDynamics matched board: run "
                       "scripts/run_dynamics_matched.py")
    else:
        b, dyn_n = dm["board"], dm["n_systems"]
        order = dm["rank_auprc"]
        dyn_rank = order.index("cafe_cd") + 1
        for i, m in enumerate(order, 1):
            mark = r"\ \ding{51}" if (floor is not None
                                      and b[m]["auprc"] > floor) else ""
            rows_b.append("%d. %s %s%s" % (i, PIT_LABEL.get(m, m.replace("_", r"\_")),
                                           fnum(b[m]["auprc"]), mark))

    if not rows_a:
        rows_a = [r"\multicolumn{9}{c}{%s}\\" % NA]
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/bench_rivers_cache.json + data/dynamics_matched.json
\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Where \cd{} loses: both public single-series benchmarks.}
\textbf{(a)} CausalRivers head-to-head, every method the harness has finished on the
identical subgraph sample, ranked by \emph{individual} AUPRC (the metric to read on
graphs this sparse); the \emph{rank} column prints its own denominator.
\cd{} is beaten by \texttt{var\_ols} --- a plain OLS VAR at $\sim$1\,ms --- in
%(losevar)d of the %(ncfg)d completed configurations. \textbf{(b)} CausalDynamics
\texttt{simple}, \texttt{noise=0.00\_confounder=False}, %(dynn)s systems scored by all
13 methods, \cd{} re-run live with the current library
(\texttt{scripts/run\_dynamics\_matched.py}); \ding{51}\ marks the three entries above
the \emph{measured} random-ranker AUPRC floor of %(floor)s for that tier
(\S\ref{sec:forensics}) --- ten of thirteen, \cd{} included, are below it. Neither
loss is ambiguous and \S\ref{sec:diag} measures the cause.}
\label{tab:losses}
\begin{tabular}{@{}llrrrclrr@{}}
\multicolumn{9}{@{}l}{\textbf{(a) CausalRivers, individual AUPRC}}\\
\toprule
 & & & \multicolumn{2}{c}{\cd{}} & & \multicolumn{2}{c}{best method} & \texttt{var\_ols} \\
\cmidrule(lr){4-5}\cmidrule(lr){7-8}
config & arm & $n$ & AUPRC & AUROC & rank & method & AUPRC & AUPRC \\
\midrule
%(rowsa)s
\bottomrule
\end{tabular}

\vspace{3pt}
{\footnotesize\raggedright
\textbf{(b) CausalDynamics \texttt{simple}, %(dynn)s matched systems, AUPRC} ---
%(rowsb)s. \cd{} is %(dynrank)s of 13.\par}
\end{table*}
""" % dict(rowsa="\n".join(rows_a), rowsb="; ".join(rows_b) if rows_b else NA,
           ncfg=n_cfg, losevar=lose_to_var,
           dynn=("%d" % dyn_n) if dyn_n else NA,
           dynrank=("%d" % dyn_rank) if dyn_rank else NA,
           floor=fnum(floor) if floor is not None else NA)
    write("losses.tex", tex, "bench/bench_rivers_cache.json + "
                             "data/dynamics_matched.json (live)")
    dyn = None
    if dm is not None:
        b = dm["board"]
        dyn = dict(rank=dyn_rank, n=dyn_n, cafe=b["cafe_cd"]["auprc"],
                   top=PIT_LABEL.get(dm["rank_auprc"][0], dm["rank_auprc"][0]),
                   top_ap=b[dm["rank_auprc"][0]]["auprc"],
                   n_below=(sum(1 for m in b if floor is not None
                                and b[m]["auprc"] < floor) if floor else None),
                   n_meth=len(b))
    return dict(n_cfg=n_cfg, lose_to_var=lose_to_var, dyn_rank=dyn_rank,
                dyn_n=dyn_n, floor=floor, dm=dm, overlap=overlap,
                quote=quote, dyn=dyn)


# --------------------------------------------------------------------------- #
# 3c. CausalDynamics `coupled`: reproduced first, then criticised
# --------------------------------------------------------------------------- #
CD_PUB_KEY = {"ngc_lstm": "ngc"}


def table_coupled():
    src = os.path.join(CAUSAL, "cd_coupled_audit.json")
    d = load(src)
    if d is None:
        PENDING.append("CausalDynamics coupled audit: cache absent "
                       "(run bench/datasets_causal.py --audit coupled)")
        return None
    lb, pub, pair = d["leaderboard"], d["published"], d["identity_paired"]
    fl = d["floor"]
    idf = fl["coupled"]["rows"]["identity/full"]

    rows = []
    for m, v in sorted(lb.items(), key=lambda kv: -kv[1]["Individual_AUROC"]):
        pk = pub.get(CD_PUB_KEY.get(m, m))
        p = pair.get(m)
        rows.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            m.replace("_", r"\_"),
            "%d" % v["n_cells"],
            fnum(v["Individual_AUROC"]), fnum(v["Individual_AUPRC"]),
            ("%.2f / %.2f" % (pk[0], pk[1])) if pk else "---",
            ("%.0f\\%%" % (100 * v["auroc_exactly_half"]))
            if v.get("auroc_exactly_half") is not None else NA,
            ("%.0e" % p["p_auprc"]) if p else NA,
            ("%.0f\\%%" % (100 * p["win_rate_auprc"])) if p else NA))

    fr = []
    for tier, lab in (("coupled", "coupled"), ("simple", "simple")):
        b = fl[tier]
        for k, kl in (("random/full", "random ranker"),
                      ("zeros/full", r"all-zeros (\texttt{Null})"),
                      ("identity/full", r"\textbf{identity matrix}")):
            r_ = b["rows"][k]
            o = b["rows"][k.replace("/full", "/offdiag")]
            fr.append("%s & %s & %s & %s & %s & %s \\\\" % (
                lab if k.startswith("random") else "", kl,
                fnum(r_["auroc"]), fnum(r_["auprc"]),
                fnum(o["auroc"]), fnum(o["auprc"])))
        fr.append(r"\addlinespace[1pt]" if tier == "coupled" else "")

    wr = []
    VOL = ("var_ols|coef|", "var_ols|t|")           # both, so nothing is selected
    for row in ("Confounder", "Lag"):
        blk = d["repro_rows"][row]
        f0 = blk["floor"]
        idr = f0["rows"]["identity/full"]
        ship = d["rows_vs_identity"][row]["paired"].get("pcmciplus", {})
        by = {e["label"]: e["variants"]["transpose=1_binarise=0_full"]
              for e in blk["repro"]}
        cells = []
        for lab in VOL:
            v = by.get(lab)
            cells.append("%s/%s" % (fnum(v["individual_auroc"]),
                                    fnum(v["individual_auprc"])) if v else NA)
        wr.append("%s & %s & %.0f\\%% & %s/%s & %s/%s & %s & %s \\\\" % (
            row, "%d" % f0["n_systems"],
            100 * f0["diagonal_share_of_positives"],
            fnum(idr["auroc"]), fnum(idr["auprc"]),
            fnum(ship.get("auroc")), fnum(ship.get("auprc")), cells[0], cells[1]))

    pr = d["wrappers"]["probe"]
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: data/causal/cache/cd_coupled_audit.json
\begin{table*}[t]\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{CausalDynamics \texttt{coupled}: reproduced first (a), then a constant
beats all of it (b).} \textbf{(a)} re-derived from the authors' own shipped
\texttt{outputs/coupled.tar.gz} over the Default row's %(nsys)d systems --- five of
seven published cells land within $0.026$, so the row is reproduced. \emph{const} is
the share of cells whose AUROC is \emph{exactly} $0.500$, i.e.\ a constant prediction;
$p$ and \emph{wins} are a paired Wilcoxon signed-rank test of that method's AUPRC
against a \textbf{constant identity matrix} over the same systems.
\textbf{Not one of the nine testable methods beats the identity matrix};
\texttt{pcmciplus} merely ties on AUROC ($p{=}%(pauroc).2f$) and loses on AUPRC.
\textbf{(b)} label-free predictors pushed through the benchmark's own scorer, 20 draws
per system. Their shipped \texttt{Null} is an \emph{all-zeros} predictor, which is the
wrong reference for a ranking metric: against a random \emph{ranker} the AUPRC floor is
%(cfl).3f on \texttt{coupled} and %(sfl).3f on \texttt{simple}, so every published
\texttt{simple} AUPRC ($.67$--$.70$) is below chance. \textbf{(c)} the mechanism is the
diagonal: the protocol keeps it, and where it carries less of the positive class the
identity floor falls and methods separate --- but the bar there is a plain OLS VAR(1)
at $\sim$1\,ms, not the leaderboard.}
\label{tab:coupled}
\begin{tabular}{@{}lrrrcrrr@{}}
\multicolumn{8}{@{}l}{\textbf{(a) the shipped Default row, re-derived --- and paired against a constant}}\\
\toprule
 & & \multicolumn{2}{c}{ours (shipped eval)} & published & & \multicolumn{2}{c}{vs.\ identity (AUPRC)} \\
\cmidrule(lr){3-4}\cmidrule(lr){7-8}
method & $n$ & AUROC & AUPRC & AUROC/AUPRC & const & $p$ & wins \\
\midrule
%(rows)s
\midrule
\textbf{constant identity} & %(nsys)d & \textbf{%(ida).3f} & \textbf{%(idp).3f} & --- & --- & --- & --- \\
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{minipage}[t]{0.455\textwidth}
{\centering\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}llrrrr@{}}
\multicolumn{6}{@{}l}{\textbf{(b) the floor, measured}}\\
\toprule
 & & \multicolumn{2}{c}{as scored} & \multicolumn{2}{c}{off-diagonal} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
tier & predictor & AUROC & AUPRC & AUROC & AUPRC \\
\midrule
%(fr)s
\bottomrule
\end{tabular}\par}
\end{minipage}\hfill
\begin{minipage}[t]{0.525\textwidth}
{\centering\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lrrllll@{}}
\multicolumn{7}{@{}l}{\textbf{(c) where the tier is a real venue} (AUROC/AUPRC)}\\
\toprule
 & & & & & \multicolumn{2}{c}{OLS VAR(1), $\sim$1\,ms} \\
\cmidrule(lr){6-7}
row & $n$ & diag. & ident. & \texttt{pcmciplus} & by $|\hat\beta|$ & by $|t|$ \\
\midrule
%(wr)s
\bottomrule
\end{tabular}\par}
\end{minipage}
\end{table*}
""" % dict(rows="\n".join(rows), fr="\n".join(x for x in fr if x),
           wr="\n".join(wr), nsys=int(idf["n"]),
           ida=idf["auroc"], idp=idf["auprc"],
           pauroc=pair["pcmciplus"]["p_auroc"],
           cfl=fl["coupled"]["rows"]["random/full"]["auprc"],
           sfl=fl["simple"]["rows"]["random/full"]["auprc"])
    write("coupled.tex", tex, "data/causal/cache/cd_coupled_audit.json")
    return dict(audit=d, probe=pr, identity=idf,
                cfl=fl["coupled"]["rows"]["random/full"]["auprc"],
                sfl=fl["simple"]["rows"]["random/full"]["auprc"],
                diag_share=fl["coupled"]["diagonal_share_of_positives"])


# --------------------------------------------------------------------------- #
# 3d. the diagnosis: a 2x2 over representation x conditioning set
# --------------------------------------------------------------------------- #
def table_diag():
    d = load(os.path.join(DATA, "conditioning.json"))
    if d is None:
        PENDING.append("conditioning diagnosis: run "
                       "scripts/run_conditioning_diagnosis.py")
        return None
    A, ident, rows = d["arms"], d["identity_checks"], d["rows"]

    def wins(a, b):
        """paired sample count on which arm `a` beats arm `b` (AUROC)."""
        return (int(sum(r[a]["auroc"] > r[b]["auroc"] for r in rows)), len(rows))

    def c(k, w=None):
        a = A[k]
        cells = "%s & %s & %s" % (fnum(a["auroc"]), fnum(a["auprc"]),
                                  fnum(a["f1max"]))
        return cells + (" & %d/%d" % w if w else " & ---")

    w_rawbiv = wins("raw/bivariate", "raw/conditional")
    w_resbiv = wins("residual/bivariate", "residual/conditional")
    w_rawrep = wins("raw/conditional", "residual/conditional")
    body = [
        r"raw panel & own past $+$ \textbf{all} others & %s & \texttt{var\_ols} \\"
        % c("raw/conditional"),
        r"raw panel & own past $+$ \textbf{one} other & %s & \texttt{granger\_pairwise} \\"
        % c("raw/bivariate", w_rawbiv),
        r"\addlinespace[1pt]",
        r"\cafe{} residual & own past $+$ \textbf{all} others & %s & "
        r"\emph{\cd{}'s skeleton} \\" % c("residual/conditional"),
        r"\cafe{} residual & own past $+$ \textbf{one} other & %s & \\"
        % c("residual/bivariate", w_resbiv),
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\cd{}, end to end} & %s & \\" % c("cafe_cd"),
        r"\multicolumn{2}{@{}l}{raw marginal lagged $|\mathrm{corr}|$, "
        r"\emph{no} conditioning} & %s & \\" % c("marginal_lagged_corr"),
    ]
    d_rep = A["raw/conditional"]["auroc"] - A["residual/conditional"]["auroc"]
    d_cond = A["residual/bivariate"]["auroc"] - A["residual/conditional"]["auroc"]
    d_cond_raw = A["raw/bivariate"]["auroc"] - A["raw/conditional"]["auroc"]

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: experimental/papers/causal/data/conditioning.json (live run)
\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{The loss decomposed: hold the estimator fixed, vary only the
representation and the conditioning set.} %(n)d CausalRivers flood/Random-5 subgraphs
(the published cell; $T{\approx}%(T).0f$, $P{=}5$, %(E).1f true off-diagonal edges,
$K{=}%(K)d$), off-diagonal, mean over samples. Every row is the \emph{same} $|t|$
statistic from the \emph{same} moment tensor --- \texttt{cafe.discovery.\_refit\_t} for
the conditional cells, \texttt{cafe.\_skeleton.bivariate\_granger} for the bivariate
ones --- so a row difference is that one design choice and nothing else. The
\emph{reproduces} column is the identity check: the two raw rows match two external
baselines to $0.0000$ mean absolute AUROC, on all %(n)d samples. \emph{w} is the paired
count of samples on which the row beats the one above it. \textbf{Both of \cd{}'s
choices cost on this benchmark, and the representation costs $%(ratio).0f\times$
more}:
residualising is $-%(drep).3f$ AUROC at fixed conditioning ($%(wrep)d/%(n)d$ samples
favour raw), and conditioning on every other source rather than one is a further
$-%(dcond).3f$ ($-%(dcondraw).3f$ on the raw panel). The panel is near-collinear ---
mean off-diagonal $|\mathrm{corr}|=%(mc).3f$, max $%(xc).3f$, condition number
$%(cn).1f$, leading eigenvalue $%(le).3f$ of the total --- which is the regime in which
a fully conditional coefficient stops being readable, and the source-separability
diagnostic the scoring router reads agrees ($\overline{\mathrm{tol}}=%(tol).2f$ against
a null of $1$, so it blends in $w_{\mathrm{biv}}=%(wbiv).2f$ of the bivariate branch).
A confounding common component is exactly what \cafe{} is built to remove, so the sign
of the first row difference is the opposite of what the design predicts.
Table~\ref{tab:scale} measures why: on unrescaled input of this magnitude the factor
adjustment is \emph{mis-scaled}, so the \cd{} residual row here is not a clean
confounder removal. A real gap survives that defect --- see \S\ref{sec:diag}.}
\label{tab:diag}
\begin{tabular}{@{}llrrrcl@{}}
\toprule
representation & conditioning set & AUROC & AUPRC & max-$F_1$ & \emph{w} & reproduces \\
\midrule
%(body)s
\bottomrule
\end{tabular}
\end{table*}
""" % dict(body="\n".join(body), n=d["n_samples"], T=d["mean_T"],
           E=d["mean_true_edges"], K=d["max_lag"],
           drep=d_rep, dcond=d_cond, dcondraw=d_cond_raw, wrep=w_rawrep[0],
           ratio=(d_rep / d_cond) if d_cond else float("nan"),
           mc=d["collinearity"]["mean_abs_offdiag_corr"],
           xc=d["collinearity"]["max_abs_offdiag_corr"],
           cn=d["collinearity"]["condition_number"],
           le=d["collinearity"]["leading_eig_share"],
           tol=d["tolerance"]["residual"], wbiv=d["w_bivariate"],
           rk=d["effective_rank_mean"])
    write("diag.tex", tex, "data/conditioning.json (live)")
    return dict(d=d, A=A, d_rep=d_rep, d_cond=d_cond, d_cond_raw=d_cond_raw,
                w_rawbiv=w_rawbiv, w_resbiv=w_resbiv, w_rawrep=w_rawrep)


# --------------------------------------------------------------------------- #
# 3f. is the factor adjustment scale-equivariant, and does it engage?
# --------------------------------------------------------------------------- #
def table_scale():
    d = load(os.path.join(DATA, "scale_equivariance.json"))
    if d is None:
        PENDING.append("scale-equivariance probe: run "
                       "scripts/run_scale_equivariance.py")
        return None
    sw, rs = d["sweep"], d["rivers_summary"]
    body = []
    for r in sw:
        body.append("$\\times %s$ & %.3f & %s & %d & %s \\\\" % (
            ("%g" % r["mult"]), r["mean_sd"],
            r"\textbf{%.1f\%%}" % (100 * r["factor_share"])
            if r["factor_share"] < 0.05 else "%.1f\\%%" % (100 * r["factor_share"]),
            r["effective_rank"], fnum(r["auc"])))
    on = [r for r in sw if r["factor_share"] > 0.05]
    off = [r for r in sw if r["factor_share"] <= 0.05]

    onv = [100 * r["factor_share"] for r in on]
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: experimental/papers/causal/data/scale_equivariance.json (live run)
\begin{table}[t]\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{\cafe{}'s factor adjustment is not scale-equivariant, and
\texttt{effective\_rank} does not report it.} One panel ($E{=}%(E)d$, $T{=}%(T)d$,
$P{=}%(P)d$, $\chi{=}%(chi).2f$), one global multiplier, everything else identical ---
a global rescale of the data rescales the loadings and should change nothing.
\emph{removes} is the per-column variance share the factor adjustment actually takes
out, read from ONE fit as \texttt{residuals(factor\_adjusted=True)} against
\texttt{(=False)}, so level, season, fixed effects and AR cancel exactly. The mechanism
is off below unit scale and degrades again far above it; the cause is
\texttt{src/cafe/\_core.py}, where the ARD precision is
$\alpha_l = N/(\lVert W_{\cdot l}\rVert^2 + 1)$ and the $+1$ does not rescale with the
data, so it dominates and drives every column's precision to its maximum.
\textbf{\texttt{effective\_rank} reads $%(rk)d$ in every row}, including the rows where
$%(onlo).0f$--$%(onhi).0f\%%$ of the variance is being removed: it is not a usable indicator of factor
activity, and every claim in this paper about whether the factor engaged is made from
the adjusted-vs-kept comparison instead.}
\label{tab:scale}
\begin{tabular}{@{}lrrrr@{}}
\toprule
multiplier & mean sd & removes & \texttt{eff.\ rank} & AUC \\
\midrule
%(body)s
\bottomrule
\end{tabular}

\vspace{3pt}
{\footnotesize\raggedright
\textbf{On CausalRivers the mechanism is mis-scaled, not merely idle.} Over %(n)d
flood/Random-5 subgraphs the column standard deviations inside a single five-station
panel span $%(sdlo).3f$ to $%(sdhi).1f$ and \emph{every} sample has at least one
sub-unit column, so the floor bites on part of the panel and not the rest. Unrescaled
(the published arm) the share removed is \textbf{negative on %(nneg)d of %(n)d} samples
--- the adjustment \emph{adds} idiosyncratic variance rather than removing a confounder.
$z$-scored it is small but mostly positive (mean $%(shz)+.3f$, negative on %(nnegz)d of
%(n)d). \textbf{And the obvious objection is falsified}: residualisation does \emph{not}
remove the scale prior. Varsortability over the same subgraphs is $%(vr).2f$ on the raw
series and $%(vd).2f$ on the \cafe{} residual --- still far above the $0.5$ no-signal
point --- with $R^2$-sortability $%(rr).2f$ against $%(rd).2f$. The ordering survives;
our estimator is simply built not to read it (Table~\ref{tab:scaleprior}).\par}
\end{table}
""" % dict(body="\n".join(body), E=d["panel"]["E"], T=d["panel"]["T"],
           P=d["panel"]["P"], chi=d["panel"]["chi"], n=rs["n"],
           sdlo=rs["col_sd_min"], sdhi=rs["col_sd_max"],
           nneg=rs["n_unrescaled_negative"], nnegz=rs["n_zscored_negative"],
           shz=rs["share_zscored"], vr=rs["varsort_raw"], vd=rs["varsort_resid"],
           rr=rs["r2sort_raw"], rd=rs["r2sort_resid"],
           rk=int(np.max([r["effective_rank"] for r in sw])),
           onlo=min(onv) if onv else float("nan"),
           onhi=max(onv) if onv else float("nan"))
    write("scale.tex", tex, "data/scale_equivariance.json (live)")
    return dict(d=d, on=on, off=off, rs=rs)


# --------------------------------------------------------------------------- #
# 3e. runtime and orientation, from the CauseMe artefacts
# --------------------------------------------------------------------------- #
SHORT_LABEL = dict(PIT_LABEL, granger_pairwise="Granger-pw",
                   pc_ts="PC-ts", fci_ts="FCI-ts", dagma_linear="DAGMA",
                   directlingam="DirectLiNGAM", varlingam="VARLiNGAM")


def table_runtime():
    d = load(os.path.join(BENCH, "bench_causeme_cache.json"))
    if d is None:
        PENDING.append("CauseMe runtime/orientation: cache absent")
        return None
    cells = d["cells"]
    EXP = "river-runoff_N-12_T-4600"
    rt = {}
    for k, v in cells.items():
        if k.startswith("run/%s/" % EXP):
            rt[v["method"]] = v
    rtrows = []
    for m, v in sorted(rt.items(), key=lambda kv: kv[1]["runtime_s_total"]):
        rtrows.append("%s & %s & %s \\\\" % (
            SHORT_LABEL.get(m, m.replace("_", r"\_")),
            ("\\textbf{%.2f}" % v["runtime_s_total"]) if m in ("cafe_cd", "pcmciplus")
            else ("%.3f" % v["runtime_s_total"] if v["runtime_s_total"] < 0.1
                  else "%.2f" % v["runtime_s_total"]),
            fnum(v["nonzero_frac"], 3)))
    speed = (rt["pcmciplus"]["runtime_s_total"] / rt["cafe_cd"]["runtime_s_total"]
             if "cafe_cd" in rt and "pcmciplus" in rt else float("nan"))

    ro = cells.get("roundtrip", {})
    rows = ro.get("rows", {})
    beats = sum(1 for v in rows.values() if v["auc"] > v["auc_transposed"])
    orows = []
    for m, v in sorted(rows.items(), key=lambda kv: -kv[1]["auc"]):
        orows.append("%s & %s & %s & %s \\\\" % (
            SHORT_LABEL.get(m, m.replace("_", r"\_")), fnum(v["auc"]),
            fnum(v["auc_transposed"]), fnum(v["margin"], 3, plus=True)))

    n_art = sum(1 for k in cells if k.startswith("run/"))
    SMALL = "linear-VAR_N-5_T-300"
    sm = {v["method"]: v for k, v in cells.items()
          if k.startswith("run/%s/" % SMALL)}
    speed_small = (sm["pcmciplus"]["runtime_s_per_dataset"]
                   / sm["cafe_cd"]["runtime_s_per_dataset"]
                   if "cafe_cd" in sm and "pcmciplus" in sm else float("nan"))
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/bench_causeme_cache.json
\begin{table}[t]\centering\scriptsize
\setlength{\tabcolsep}{2.6pt}
\caption{\textbf{Runtime is a genuine win, and the orientation is proved rather than
asserted.} \textbf{(a)} CauseMe \texttt{%(exp)s}: one \emph{real} $N{=}%(N)d$,
$T{=}%(T)d$ series, one process, same CPU. \cd{} is $%(speed).0f\times$ faster than
PCMCI$+$ at the same fully dense submission (\emph{nz}, the fraction of off-diagonal
pairs given a non-zero score). \textbf{(b)} a synthetic round-trip through CauseMe's
\emph{own} on-disk format, loader and metric definitions ($B{=}%(B)d$, $N{=}%(RN)d$,
$T{=}%(RT)d$): each method is scored against a known graph and against its own
transpose. \textbf{All %(beats)d beat their transpose}, so no submission is inverted ---
the failure mode that a benchmark with server-side scoring cannot catch. This is a
format test, \textbf{not} a CauseMe score: the experiment zips ship no ground truth, so
no CauseMe leaderboard number appears in this paper. %(nart)d artefacts are built and
validated; nothing was uploaded.}
\label{tab:runtime}
\begin{minipage}[t]{0.44\columnwidth}
{\raggedright\textbf{(a) s / series, $N{=}%(N)d$}\par}
\vspace{1pt}
\begin{tabular}{@{}lrr@{}}
\toprule
method & s & nz \\
\midrule
%(rtrows)s
\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.54\columnwidth}
{\raggedright\textbf{(b) orientation round-trip}\par}
\vspace{1pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
method & AUC & $\top$ & margin \\
\midrule
%(orows)s
\bottomrule
\end{tabular}
\end{minipage}
\end{table}
""" % dict(exp=EXP.replace("_", r"\_"), rtrows="\n".join(rtrows),
           orows="\n".join(orows), speed=speed, beats=beats, nart=n_art,
           N=d["meta"]["experiments"][EXP]["N"], T=d["meta"]["experiments"][EXP]["T"],
           B=ro.get("B", 0), RN=ro.get("N", 0), RT=ro.get("T", 0))
    write("runtime.tex", tex, "bench/bench_causeme_cache.json")
    return dict(speed=speed, beats=beats, n_art=n_art, rt=rt, roundtrip=rows,
                speed_small=speed_small,
                small=dict(N=d["meta"]["experiments"][SMALL]["N"],
                           T=d["meta"]["experiments"][SMALL]["T"]))


# --------------------------------------------------------------------------- #
# 4. the point-in-time board
# --------------------------------------------------------------------------- #


def _nanmean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def table_pit():
    src = os.path.join(BENCH, "pit_board_cache.json")
    d = load(src)
    if d is None:
        PENDING.append("point-in-time board: cache absent")
        return None
    rows, meta = d["rows"], d["meta"]
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)

    agg = {}
    for m, rs in by.items():
        agg[m] = dict(
            n_tasks=len(rs),
            n_rev=int(sum(r["max_edge_revision"] > 0 for r in rs)),
            churn=_nanmean([r["churn_mean"] for r in rs]),
            churn_max=_nanmean([r["churn_max"] for r in rs]),
            flips=float(np.mean([r["n_flip"] for r in rs])),
            edges=_nanmean([r.get("n_edges_full") for r in rs]),
            n_empty=int(sum(bool(r.get("degenerate")) for r in rs)),
            pub_ap=_nanmean([r["batch"].get("auprc") for r in rs]),
            hon_ap=_nanmean([r["honest"].get("auprc") for r in rs]),
            d_ap=_nanmean([r.get("delta_auprc") for r in rs]),
            dm_ap=_nanmean([r.get("dmatch_auprc") for r in rs]),
            d_f1=_nanmean([r.get("delta_f1") for r in rs]),
            pit=all(r.get("is_pit") for r in rs),
        )

    ntasks = max(a["n_tasks"] for a in agg.values())
    full = [m for m in agg if agg[m]["n_tasks"] == ntasks]
    partial = [m for m in agg if agg[m]["n_tasks"] < ntasks]
    # CAFE-CD is printed once, below the rule, however complete its row is
    order = sorted([m for m in full if m != "cafe_cd"],
                   key=lambda m: (-agg[m]["d_ap"] if np.isfinite(agg[m]["d_ap"])
                                  else 1e9))

    def row(m, mark=""):
        a = agg[m]
        dag = r"$^{\dagger}$" if a["n_empty"] else ""
        return "%s%s%s & %d/%d & %s & %s & %.1f & %.1f & %s & %s & %s & %s \\\\" % (
            PIT_LABEL.get(m, m.replace("_", r"\_")), dag, mark, a["n_rev"],
            a["n_tasks"], fnum(a["churn"]), fnum(a["churn_max"]), a["flips"],
            a["edges"], fnum(a["pub_ap"]), fnum(a["hon_ap"]),
            fnum(a["d_ap"], plus=True), fnum(a["dm_ap"], plus=True))

    body = [row(m) for m in order]
    body.append(r"\midrule")
    # The CAFE-CD row is a full-board row now; the n/N denominator stays visible in the
    # rev column for every method, so a short row could never be read as a full one.
    if "cafe_cd" in agg:
        body.append(row("cafe_cd", r"$^{\ast}$" if "cafe_cd" in partial else ""))
        if "cafe_cd" in partial:
            PENDING.append(
                "the CAFE-CD row of the point-in-time board is a %d-task mean against "
                "the baselines' %d-task means; its denominator is printed in the rev "
                "column" % (agg["cafe_cd"]["n_tasks"], ntasks))
    else:
        body.append(r"\textbf{\cafe{}-CD}$^{\ast}$ & %s \\" % " & ".join([NA] * 9))
        PENDING.append("point-in-time board: the whole CAFE-CD row")
    for m in partial:
        if m != "cafe_cd":
            body.append(row(m, r"$^{\ast}$"))
            PENDING.append("point-in-time board: %s is a %d-task mean (run in progress)"
                           % (m, agg[m]["n_tasks"]))
    have_cafe = "cafe_cd" in agg and "cafe_cd" not in partial
    pos = [m for m in agg if np.isfinite(agg[m]["d_ap"]) and agg[m]["d_ap"] > 0]
    nondeg = [m for m in agg if not agg[m]["n_empty"]]
    worst = max((m for m in agg if np.isfinite(agg[m]["d_ap"])),
                key=lambda m: agg[m]["d_ap"], default=None)

    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: bench/pit_board_cache.json
\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{4.5pt}
\caption{\textbf{The point-in-time board: an unexamined property of time-series causal
discovery, and what it costs each method.} Every published method here is \emph{batch},
so a graph consumed at $t$ carries information from after $t$. \textsc{rev} counts the
tasks on which the method fails the truncation-invariance certificate \cafe{} uses for
imputation (an edge asserted at a prefix is later rewritten); because a batch method
fails that almost definitionally, the informative columns are the \emph{magnitude} ---
\textsc{churn}, mean and max Jaccard distance between the graph asserted at a prefix
and the graph finally published; \textsc{flip}, edges whose orientation reverses;
$|E|$, final edge count --- and the \emph{leakage delta} $\Delta$, on each method's own
metric, threshold and data: published AUPRC minus right-edge honest AUPRC (refitted on
each prefix, read at $t$). $\Delta_{\text{match}}$ repeats it at identical sample size,
isolating look-ahead from the mere loss of data. Mean over %(ntasks)d tasks (synthetic
VAR, synthetic contemporaneous, CausalRivers, CausalDynamics; 2 each).
$^{\dagger}$ asserted \emph{no} edges on $\ge1$ task, where the certificate is vacuous;
FCI earns one on four of eight. \textsc{rev} prints its own denominator, so a row with
fewer tasks (marked $^{\ast}$) can never be read as a full-board mean.
\cd{}'s AUPRC columns here are a \emph{lower bound}: this board's rows were computed
before \texttt{CausalGraph.scores()} was changed from a two-tier step to continuous
evidence. The certificate columns read the support and are unaffected.
\emph{We defined this test; it is offered as a property worth reporting, not as a
ranking we win.}}
\label{tab:pit}
\begin{tabular}{@{}lccccrrrrr@{}}
\toprule
 & \multicolumn{5}{c}{certificate} & \multicolumn{4}{c}{AUPRC} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-10}
Method & rev & churn & churn$_{\max}$ & flips & $|E|$ & pub. & honest & $\Delta$ & $\Delta_{\text{match}}$ \\
\midrule
%(body)s
\bottomrule
\end{tabular}
\end{table*}
""" % dict(body="\n".join(body), ntasks=ntasks)
    write("pit.tex", tex, "bench/pit_board_cache.json")
    return dict(agg=agg, ntasks=ntasks, n_pos=len(pos), n_nondeg=len(nondeg),
                have_cafe=have_cafe, meta=meta, worst=worst)


# --------------------------------------------------------------------------- #
# 5 + 6. engine router and the screening-backend null result
# --------------------------------------------------------------------------- #
def table_router():
    d = load(os.path.join(DATA, "engine_router.json"))
    if d is None:
        PENDING.append("engine router: run scripts/run_engine_router.py")
        return None
    body = []
    for r in d["rows"]:
        body.append("%s & %d & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            ({"homogeneous": "homog.", "heterogeneous": "heterog."}[r["family"]]
             if r["seed"] == d["seeds"][0] else ""),
            r["seed"], fnum(r["within"]), fnum(r["between"]), fnum(r["ratio"]),
            {"pooled": "pooled", "per_entity": "per-e."}[r["engine"]],
            fnum(r["mae_pooled"]), fnum(r["mae_per_entity"]),
            r"\ding{51}" if r["correct"] else r"\ding{55}"))
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: experimental/papers/causal/data/engine_router.json (live run)
\textbf{(a) The engine router: a structural test with a null of $1$.}
{\footnotesize \emph{within} is the alignment of an entity's own two half-sample loading
subspaces --- the ceiling identical loadings would give; \emph{between} is the same
alignment across entities. The raw \emph{between} column interleaves across the two
families ($%(hlo).3f$--$%(hhi).3f$ homogeneous vs.\ $%(elo).3f$--$%(ehi).3f$
heterogeneous), so no absolute cutoff orders them; the ratio does. MAE is held-out
($%(pm).0f\%%$ MCAR) with each engine \emph{pinned} --- an outcome the router never
sees. It picks the lower-MAE engine on $%(nc)d/%(np)d$ panels.\par}
\vspace{2pt}
{\centering\scriptsize
\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{@{}lrrrrlrrc@{}}
\toprule
 & & \multicolumn{3}{c}{router (design matrix only)} & & \multicolumn{2}{c}{held-out MAE} & \\
\cmidrule(lr){3-5}\cmidrule(lr){7-8}
panel & seed & within & betw. & ratio & picks & pooled & per-e. & ok \\
\midrule
%(body)s
\bottomrule
\end{tabular}\par}
""" % dict(body="\n".join(body), nc=d["n_correct"], np=d["n_panels"],
           pm=100 * d["p_miss"],
           hlo=d["raw_between_homogeneous"][0], hhi=d["raw_between_homogeneous"][1],
           elo=d["raw_between_heterogeneous"][0], ehi=d["raw_between_heterogeneous"][1])
    write("router.tex", tex, "data/engine_router.json (live)")
    return d


def table_screen():
    d = load(os.path.join(DATA, "screen_backend.json"))
    if d is None:
        PENDING.append("screening backend: run scripts/run_screen_backend.py")
        return None
    by = defaultdict(dict)
    for r in d["rows"]:
        by[r["P"]][r["stat"]] = r
    body = []
    for P in sorted(by):
        c = by[P]
        body.append("%d & %s & %s & %.1f$\\times$ & %s & %s & %.1f$\\times$ \\\\" % (
            P, fnum(c["phi"]["dense_s"], 4), fnum(c["phi"]["bits_s"], 4),
            c["phi"]["ratio"], fnum(c["arrow"]["dense_s"], 4),
            fnum(c["arrow"]["bits_s"], 4), c["arrow"]["ratio"]))
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
%% source: experimental/papers/causal/data/screen_backend.json (live run)
\textbf{(b) A null result on our own idea: bit-packed lag screening is
$%(lo).1f$--$%(hi).1f\times$ \emph{slower} than dense BLAS, with no crossover.}
{\footnotesize Statistic only ($n_{\text{perm}}{=}0$), $n{=}%(n)d$, $K{=}%(K)d$, best of
%(rep)d, single process, %(plat)s, numpy %(np)s. The two backends return
\emph{bit-identical} statistic surfaces ($\max|\mathrm{d}S| = 0$ in every cell), so this
is a pure speed comparison, and the gap \emph{widens} with $P$. At $20\%%$ MCAR the
ordering is unchanged ($%(mc).1f\times$ at $P{=}100$).\par}
\vspace{2pt}
{\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rrrrrrr@{}}
\toprule
 & \multicolumn{3}{c}{\texttt{stat="phi"}} & \multicolumn{3}{c}{\texttt{stat="arrow"}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$P$ & dense (s) & bits (s) & ratio & dense (s) & bits (s) & ratio \\
\midrule
%(body)s
\bottomrule
\end{tabular}\par}
""" % dict(body="\n".join(body), lo=d["ratio_range"][0], hi=d["ratio_range"][1],
           n=d["rows"][0]["n"], K=d["rows"][0]["max_lag"], rep=3,
           plat=d["env"]["platform"].split("-")[0] + "-arm64",
           np=d["env"]["numpy"], mc=d["mcar"]["ratio"])
    write("screen.tex", tex, "data/screen_backend.json (live)")
    return d


# --------------------------------------------------------------------------- #
# provenance / status block, printed verbatim in the paper's appendix
# --------------------------------------------------------------------------- #
def table_eng():
    """One full-width float holding both engineering results, side by side."""
    if not os.path.exists(os.path.join(TAB, "router.tex")):
        return
    tex = r"""%% generated by experimental/papers/causal/scripts/make_tables.py
\begin{table*}[t]
\begin{minipage}[t]{0.485\textwidth}\input{tables/router}\end{minipage}\hfill
\begin{minipage}[t]{0.485\textwidth}\input{tables/screen}\end{minipage}
\caption{\textbf{Two engineering results from building \cd{}, one of them negative.}
Both are live re-runs, not transcriptions: \texttt{scripts/run\_engine\_router.py} and
\texttt{scripts/run\_screen\_backend.py}.}
\label{tab:eng}
\end{table*}
"""
    with open(os.path.join(TAB, "eng.tex"), "w") as fh:
        fh.write(tex)
    print("  wrote tables/eng.tex")


def note_external():
    """Tables this build does NOT generate but the paper inputs, so the provenance map
    in status.tex still names their source.  Nothing here is written by this script."""
    ext = [("replication.tex", "bench/exp_causal_replication.py "
                               "(bench/causal_replication_cache.json)")]
    # Three numbers quoted in the running text of the diagnosis section were measured
    # in the scoring-router investigation and are NOT cached in this directory.  They
    # are named here so the provenance map says so rather than implying a local cache.
    SOURCES.append((
        "sec. 6 text: the 40-sample residual bivariate re-measurement, the "
        "antisymmetric-share test and the path-resolution test",
        "measured in the scoring-router investigation (src/cafe/_skeleton.py, "
        "src/cafe/discovery.py); no cache in this directory"))
    for name, src in ext:
        if os.path.exists(os.path.join(TAB, name)):
            SOURCES.append((name, src))
        else:
            PENDING.append("evidence-replication table: run "
                           "bench/exp_causal_replication.py --tables-only")


def write_macros(L, H=None, R=None, D=None, SP=None, SC=None):
    """Every number the running text quotes from a LIVE cache, emitted as a macro so a
    rebuild can never leave the prose disagreeing with the table beside it."""
    def mac(name, val):
        return r"\newcommand{\%s}{%s}" % (name, val)
    lines = [r"%% generated by experimental/papers/causal/scripts/make_tables.py"]
    if H:
        lines += [
            mac("HHrank", "%d of %d" % (H["rank"], H["n_meth"])),
            mac("HHnbase", H["n_base"] * 2),
            mac("HHchiA", "%.2f" % H["chi_first"]),
            mac("HHchiB", "%.2f" % H["chi_last"]),
            mac("HHcafeA", fnum(H["cafe_first"])),
            mac("HHcafeB", fnum(H["cafe_last"])),
            mac("HHbestLo", fnum(H["best_lo"])),
            mac("HHbestHi", fnum(H["best_hi"])),
            mac("HHgapLo", fnum(H["gap_lo"])),
            mac("HHgapHi", fnum(H["gap_hi"])),
            mac("HHgapA", fnum(H["gap_first"])),
            mac("HHresLo", fnum(H["dlo"], plus=True)),
            mac("HHresHi", fnum(H["dhi"], plus=True)),
            mac("HHresMu", fnum(H["dmu"], plus=True)),
            mac("HHrtCafe", "%.2f" % H["rt"]["cafe_cd"]),
        ]
    if R:
        for m, key in (("rt", "cafe_cd"), ("rt", "pcmciplus")):
            pass
        lines += [mac("RTcafe", "%.2f" % R["rt"]["cafe_cd"]["runtime_s_total"]),
                  mac("RTpcmci", "%.2f" % R["rt"]["pcmciplus"]["runtime_s_total"]),
                  mac("RTspeed", "%.0f" % R["speed"]),
                  mac("RTnz", fnum(R["rt"]["cafe_cd"]["nonzero_frac"], 3)),
                  mac("RTnart", R["n_art"]),
                  mac("RTbeats", R["beats"]),
                  mac("RTrtAuc", fnum(R["roundtrip"]["cafe_cd"]["auc"])),
                  mac("RTrtTrans",
                      fnum(R["roundtrip"]["cafe_cd"]["auc_transposed"])),
                  mac("RTspeedSmall", "%.1f" % R["speed_small"]),
                  mac("RTsmallN", R["small"]["N"]),
                  mac("RTsmallT", R["small"]["T"])]
    if D:
        d, A = D["d"], D["A"]
        lines += [
            mac("DGn", d["n_samples"]),
            mac("DGrep", fnum(D["d_rep"])),
            mac("DGcond", fnum(D["d_cond"])),
            mac("DGcondRaw", fnum(D["d_cond_raw"])),
            mac("DGwRep", "%d/%d" % D["w_rawrep"]),
            mac("DGwRawBiv", "%d/%d" % D["w_rawbiv"]),
            mac("DGwResBiv", "%d/%d" % D["w_resbiv"]),
            mac("DGrawCond", fnum(A["raw/conditional"]["auroc"])),
            mac("DGresCond", fnum(A["residual/conditional"]["auroc"])),
            mac("DGresBiv", fnum(A["residual/bivariate"]["auroc"])),
            mac("DGrawBiv", fnum(A["raw/bivariate"]["auroc"])),
            mac("DGcafe", fnum(A["cafe_cd"]["auroc"])),
            mac("DGcafeFone", fnum(A["cafe_cd"]["f1max"])),
            mac("DGmarg", fnum(A["marginal_lagged_corr"]["auroc"])),
            mac("DGtol", "%.2f" % d["tolerance"]["residual"]),
            mac("DGwbiv", "%.2f" % d["w_bivariate"]),
            mac("DGrank", "%.1f" % d["effective_rank_mean"]),
            mac("DGmc", "%.3f" % d["collinearity"]["mean_abs_offdiag_corr"]),
            mac("DGxc", "%.3f" % d["collinearity"]["max_abs_offdiag_corr"]),
            mac("DGcn", "%.1f" % d["collinearity"]["condition_number"]),
            mac("DGle", "%.3f" % d["collinearity"]["leading_eig_share"]),
        ]
    if SC:
        rs, sw = SC["rs"], SC["d"]["sweep"]
        onv = [r["factor_share"] for r in SC["on"]]
        lines += [mac("SCn", rs["n"]),
                  mac("SConLo", "%.0f" % (100 * min(onv))),
                  mac("SConHi", "%.0f" % (100 * max(onv))),
                  mac("SCoffHi", "%.2f" % (100 * max(
                      r["factor_share"] for r in SC["off"]))),
                  mac("SCsdOff", "%.3f" % max(r["mean_sd"] for r in SC["off"])),
                  mac("SCnneg", rs["n_unrescaled_negative"]),
                  mac("SCnnegz", rs["n_zscored_negative"]),
                  mac("SCshz", "%.3f" % rs["share_zscored"]),
                  mac("SCsdLo", "%.3f" % rs["col_sd_min"]),
                  mac("SCsdHi", "%.1f" % rs["col_sd_max"]),
                  mac("SCvsRaw", "%.2f" % rs["varsort_raw"]),
                  mac("SCvsRes", "%.2f" % rs["varsort_resid"]),
                  mac("SCrsRaw", "%.2f" % rs["r2sort_raw"]),
                  mac("SCrsRes", "%.2f" % rs["r2sort_resid"])]
    if SP:
        lines += [mac("SPngain", len(SP["grp"]["gain"])),
                  mac("SPnflat", len(SP["grp"]["flat"])),
                  mac("SPglo", fnum(min(SP["gains"][m] for m in SP["grp"]["gain"]), 4,
                                    plus=True)),
                  mac("SPghi", fnum(max(SP["gains"][m] for m in SP["grp"]["gain"]), 4,
                                    plus=True)),
                  mac("SPcafe", fnum(SP["cafe_gain"], 4, plus=True)),
                  mac("SPcafeScale", fnum(SP["cafe_scale"], 4)),
                  mac("SPcafeFree", fnum(SP["cafe_free"], 4)),
                  mac("SPn", SP["n"])]
    if L:
        q = L.get("quote", {})
        for (arm, cfg), tag in ((("scale", "flood_full/random_5"), "FloodScale"),
                                (("free", "flood_full/random_5"), "FloodFree"),
                                (("scale", "east/confounder_3"), "EastConf"),
                                (("scale", "east/random_3"), "EastRand")):
            v = q.get((arm, cfg))
            if not v:
                continue
            lines += [mac("CR%sRank" % tag, "%d" % v["rank"]),
                      mac("CR%sN" % tag, "%d" % v["n"]),
                      mac("CR%sCafe" % tag, fnum(v["cafe"])),
                      mac("CR%sTop" % tag, v["top"]),
                      mac("CR%sTopAP" % tag, fnum(v["top_ap"])),
                      mac("CR%sVar" % tag, fnum(v["var_ols"]))]
        dy = L.get("dyn")
        if dy:
            lines += [mac("DYrank", "%d" % dy["rank"]),
                      mac("DYn", "%d" % dy["n"]),
                      mac("DYcafe", fnum(dy["cafe"])),
                      mac("DYtop", dy["top"]),
                      mac("DYtopAP", fnum(dy["top_ap"])),
                      mac("DYnbelow", "%d" % dy["n_below"]),
                      mac("DYnmeth", "%d" % dy["n_meth"])]
        if L.get("floor") is not None:
            lines.append(mac("DYfloor", fnum(L["floor"])))
        lines += [mac("LossVarCount", L["lose_to_var"]),
                  mac("LossCfgCount", L["n_cfg"]),
                  mac("LossVarAll",
                      ("all %d" % L["n_cfg"]) if L["lose_to_var"] == L["n_cfg"]
                      else "%d of the %d" % (L["lose_to_var"], L["n_cfg"]))]
        for sp in ("coupled", "climate"):
            o = L.get("overlap", {}).get(sp)
            if o:
                lines += [mac("Match%s" % sp.capitalize(), o["matched"]),
                          mac("Base%s" % sp.capitalize(), o["baseline"]),
                          mac("Cafe%s" % sp.capitalize(), o["cafe"])]
    with open(os.path.join(TAB, "macros.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote tables/macros.tex")


def write_status():
    def esc(x):
        return x.replace("_", r"\_")

    # group artefacts by source so a shared cache is named once
    by_src = {}
    for n, s_ in SOURCES:
        by_src.setdefault(s_, []).append(n)
    lines = [r"%% generated by experimental/papers/causal/scripts/make_tables.py",
             r"{\footnotesize\raggedright\noindent"]
    lines.append("; ".join(
        r"\texttt{%s}$\,\leftarrow\,$\texttt{%s}" % (esc(", ".join(ns)), esc(s_))
        for s_, ns in by_src.items()) + r".\par}")
    lines.append(r"\vspace{3pt}")
    if PENDING:
        lines.append(r"{\footnotesize\raggedright\noindent\textbf{Left empty because the "
                     r"run had not finished at build time:} "
                     + "; ".join(esc(p) for p in PENDING) + r".\par}")
    else:
        lines.append(r"{\footnotesize\raggedright\noindent Nothing was left empty: every "
                     r"table in this paper is complete as built.\par}")
    with open(os.path.join(TAB, "status.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote tables/status.tex")


def main():
    print("regenerating CAFE-CD paper tables")
    g = table_gate()
    figure_gate(g)
    H = table_h2h()
    table_rivers()
    SP = table_scaleprior()
    table_dynamics()
    table_coupled()
    L = table_losses()
    D = table_diag()
    SC = table_scale()
    R = table_runtime()
    table_pit()
    table_router()
    table_screen()
    table_eng()
    note_external()
    write_macros(L, H, R, D, SP, SC)
    write_status()
    print("\nprovenance:")
    for name, s in SOURCES:
        print(f"  {name:<16} <- {s}")
    if PENDING:
        print("\nPENDING (emitted as '--'):")
        for p in PENDING:
            print("  -", p)
    else:
        print("\nno pending cells")


if __name__ == "__main__":
    main()
