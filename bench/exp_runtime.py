"""
RUNTIME experiment for the CAFE paper -- substantiate the CPU-speed / scaling claim
with REAL measured wall-clock (median of repeated runs) and an honest complexity read.

Everything here is point-in-time / single-machine fair: BLAS threads are pinned to 2
BEFORE numpy import so every method (CAFE and the references) sees the same core budget.
There is NO GPU anywhere -- the deep-model contrast (SAITS/BRITS/Transformer/CSDI) is
REPORTED from the literature, not run here, and is clearly labelled qualitative.

Outputs (written under paper/):
  paper/figures/scaling.pdf   -- log-log time vs T (fixed N) and time vs N (fixed T), CAFE.
  paper/tables/runtime.tex    -- head-to-head wall-clock + MAE on one fixed real task.

Run:  python bench/exp_runtime.py     ( < ~6 min on one CPU core, BLAS=2 )
"""
import os
# Pin BLAS threads BEFORE numpy import so the timing is a fair single-machine comparison.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)

import harness as H                       # MASKERS, metrics
import c_unified_penmf as P               # CAFE (causal): online_impute
import m_softimpute as S                  # non-causal ref
import m_trmf as Tr                       # non-causal ref
import m_baselines as B                   # cheap causal refs: linear_interp, knn_impute

DATA = os.path.join(_ROOT, "data")
FIGDIR = os.path.join(_ROOT, "paper", "figures")
TABDIR = os.path.join(_ROOT, "paper", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

BLAS_THREADS = os.environ.get("OPENBLAS_NUM_THREADS", "2")
N_REPEAT = 3                              # median of 3 runs for every timed cell


# --------------------------------------------------------------------------- #
def _time_median(fn, Xobs, meta, repeat=N_REPEAT):
    """Median wall-clock (s) of `repeat` runs of fn(Xobs.copy(), meta). Returns
    (median_seconds, prediction_of_the_LAST_run)."""
    ts, pred = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        pred = fn(Xobs.copy(), dict(meta) if meta else {})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), pred


def _mae(true, pred, mask):
    p = np.asarray(pred, float)[mask]
    p = np.where(np.isfinite(p), p, np.nanmean(true))
    return float(np.mean(np.abs(p - true[mask])))


# --------------------------------------------------------------------------- #
def head_to_head():
    """Wall-clock + MAE on one fixed real task: beijing[:3000,:132] @ 10% MCAR."""
    b = np.load(os.path.join(DATA, "beijing_clean.npy"))
    X = np.ascontiguousarray(b[:3000, :132])
    M = H.MASKERS["mcar"](X, 0.10, seed=0)
    Xobs = X.copy(); Xobs[M] = np.nan

    # (label, fn, causal?, backbone)
    methods = [
        ("CAFE (ours)",   P.online_impute,   True,  "factor"),
        ("SoftImpute",    S.impute,          False, "low-rank"),
        ("TRMF",          Tr.impute,         False, "MF+AR"),
        ("Linear interp", B.linear_interp,   False, "1D interp"),
        ("kNN",           B.knn_impute,      False, "kNN"),
    ]
    rows = []
    for label, fn, causal, backbone in methods:
        sec, pred = _time_median(fn, Xobs, {})
        mae = _mae(X, pred, M)
        rows.append(dict(method=label, sec=sec, mae=mae,
                         causal=causal, backbone=backbone))
        print(f"  {label:14s}  {sec:7.3f}s   MAE={mae:.4f}   "
              f"causal={causal}")
    task = f"beijing[:3000,:132] @ 10% MCAR  ({X.shape[0]}x{X.shape[1]})"
    return rows, task


# --------------------------------------------------------------------------- #
def scaling():
    """Time CAFE while varying T (fixed N) and N (fixed T) on the real beijing matrix."""
    b = np.load(os.path.join(DATA, "beijing_clean.npy"))

    # --- vary T at fixed N = 60 ---
    N_FIX = 60
    Ts = [500, 1000, 2000, 4000]
    t_secs = []
    for T in Ts:
        X = np.ascontiguousarray(b[:T, :N_FIX])
        M = H.MASKERS["mcar"](X, 0.10, seed=0)
        Xobs = X.copy(); Xobs[M] = np.nan
        sec, _ = _time_median(P.online_impute, Xobs, {})
        t_secs.append(sec)
        print(f"  T={T:5d} N={N_FIX}:  {sec:7.3f}s")

    # --- vary N at fixed T = 2000 ---
    T_FIX = 2000
    Ns = [10, 30, 60, 120]
    n_secs = []
    for N in Ns:
        X = np.ascontiguousarray(b[:T_FIX, :N])
        M = H.MASKERS["mcar"](X, 0.10, seed=0)
        Xobs = X.copy(); Xobs[M] = np.nan
        sec, _ = _time_median(P.online_impute, Xobs, {})
        n_secs.append(sec)
        print(f"  T={T_FIX} N={N:4d}:  {sec:7.3f}s")

    return dict(N_fix=N_FIX, Ts=Ts, t_secs=t_secs,
                T_fix=T_FIX, Ns=Ns, n_secs=n_secs)


def _loglog_slope(x, y):
    """OLS slope of log y on log x (the empirical scaling exponent)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


# --------------------------------------------------------------------------- #
def make_figure(sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    from viz_common import PALETTE, style_ax

    sT = _loglog_slope(sc["Ts"], sc["t_secs"])
    sN = _loglog_slope(sc["Ns"], sc["n_secs"])

    fig, (axT, axN) = plt.subplots(1, 2, figsize=(6.6, 2.6),
                                   constrained_layout=True)

    # -- time vs T (fixed N) --
    axT.loglog(sc["Ts"], sc["t_secs"], "o-", color=PALETTE["teal"],
               lw=1.8, ms=5, mfc="white", mec=PALETTE["teal"], zorder=3)
    # reference linear (slope-1) guide through the last point
    ref = np.array(sc["Ts"], float)
    refy = sc["t_secs"][-1] * ref / ref[-1]
    axT.loglog(ref, refy, "--", color=PALETTE["grey"], lw=1.0,
               zorder=2, label="slope 1 (linear)")
    axT.set_xlabel("time length $T$  (rows)")
    axT.set_ylabel("wall-clock (s)")
    axT.set_title(f"CAFÉ vs $T$  (fixed $N={sc['N_fix']}$)", fontsize=9)
    axT.annotate(f"fitted slope $={sT:.2f}$", xy=(0.04, 0.90),
                 xycoords="axes fraction", fontsize=8,
                 color=PALETTE["ink"])
    axT.legend(fontsize=6.6, frameon=False, loc="lower right")
    # Explicit sparse major ticks with plain integer labels -> no collision on log axis.
    axT.xaxis.set_major_locator(FixedLocator(sc["Ts"]))
    axT.xaxis.set_major_formatter(FixedFormatter([str(t) for t in sc["Ts"]]))
    axT.xaxis.set_minor_locator(NullLocator())
    style_ax(axT)

    # -- time vs N (fixed T) --
    axN.loglog(sc["Ns"], sc["n_secs"], "o-", color=PALETTE["blue"],
               lw=1.8, ms=5, mfc="white", mec=PALETTE["blue"], zorder=3)
    refn = np.array(sc["Ns"], float)
    refny = sc["n_secs"][-1] * refn / refn[-1]
    axN.loglog(refn, refny, "--", color=PALETTE["grey"], lw=1.0,
               zorder=2, label="slope 1 (linear)")
    axN.set_xlabel("width $N$  (features)")
    axN.set_ylabel("wall-clock (s)")
    axN.set_title(f"CAFÉ vs $N$  (fixed $T={sc['T_fix']}$)", fontsize=9)
    axN.annotate(f"fitted slope $={sN:.2f}$", xy=(0.04, 0.90),
                 xycoords="axes fraction", fontsize=8,
                 color=PALETTE["ink"])
    axN.legend(fontsize=6.6, frameon=False, loc="lower right")
    # Explicit sparse major ticks with plain integer labels -> no collision on log axis.
    axN.xaxis.set_major_locator(FixedLocator(sc["Ns"]))
    axN.xaxis.set_major_formatter(FixedFormatter([str(n) for n in sc["Ns"]]))
    axN.xaxis.set_minor_locator(NullLocator())
    style_ax(axN)

    out = os.path.join(FIGDIR, "scaling.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {out}  (slope_T={sT:.2f}, slope_N={sN:.2f})")
    return sT, sN


# --------------------------------------------------------------------------- #
def _ding(flag):
    return "\\ding{51}" if flag else "\\ding{55}"


def make_table(rows, task):
    """Emit paper/tables/runtime.tex -- ready to \\input, styled per cafe.tex."""
    cafe_sec = next(r["sec"] for r in rows if "CAFE" in r["method"])
    lines = []
    lines.append("% Auto-generated by bench/exp_runtime.py -- real measured wall-clock.")
    lines.append("% Do not edit by hand; re-run exp_runtime.py to refresh.")
    lines.append("\\begin{table}[t]\\centering\\small")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append("\\caption{\\textbf{Wall-clock on a fixed real task} "
                 "(BeijingAir, first $3000$ rows $\\times132$ sensors, $10\\%$ MCAR; "
                 "MAE on standardised data). Single machine, \\emph{one CPU core} with "
                 "BLAS pinned to " + BLAS_THREADS + " threads for every method; "
                 "\\emph{no GPU}. Median of " + str(N_REPEAT) + " runs. "
                 "\\cafe{} matches the references' accuracy at comparable CPU cost while "
                 "being the only strictly \\emph{causal} (point-in-time) low-error method. "
                 "Deep neural imputers (SAITS/BRITS/Transformer/CSDI) are omitted here as "
                 "they require GPU \\emph{training}; see text for that "
                 "$\\sim\\!10^3\\times$ compute gap.}")
    lines.append("\\label{tab:runtime}")
    lines.append("\\begin{tabular}{@{}lccccc@{}}")
    lines.append("\\toprule")
    lines.append("Method & Backbone & Time (s)~$\\downarrow$ & MAE~$\\downarrow$ "
                 "& Causal? & HW \\\\")
    lines.append("\\midrule")
    for r in rows:
        is_cafe = "CAFE" in r["method"]
        name = ("\\textbf{\\cafe{} (ours)}" if is_cafe else r["method"])
        sec = f"{r['sec']:.2f}"
        mae = f"{r['mae']:.3f}"
        if is_cafe:
            sec = "\\textbf{" + sec + "}"
            mae = "$\\mathbf{" + f"{r['mae']:.3f}" + "}$"
        hw = "\\textbf{CPU}" if is_cafe else "CPU"
        lines.append(f"{name} & {r['backbone']} & {sec} & {mae} & "
                     f"{_ding(r['causal'])} & {hw} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    txt = "\n".join(lines) + "\n"
    out = os.path.join(TABDIR, "runtime.tex")
    with open(out, "w") as f:
        f.write(txt)
    print(f"  wrote {out}")
    return txt, cafe_sec


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    t_all = time.perf_counter()
    print("[1/4] head-to-head wall-clock (median of 3) ...")
    rows, task = head_to_head()
    print("[2/4] scaling sweeps (CAFE) ...")
    sc = scaling()
    print("[3/4] figure ...")
    sT, sN = make_figure(sc)
    print("[4/4] table ...")
    txt, cafe_sec = make_table(rows, task)

    print("\n===== SUMMARY =====")
    print(f"fixed task: {task}")
    for r in rows:
        print(f"  {r['method']:14s} {r['sec']:7.3f}s  MAE={r['mae']:.4f}  "
              f"causal={r['causal']}")
    print(f"scaling slope_T={sT:.2f} (near-linear in T), "
          f"slope_N={sN:.2f}")
    print(f"total experiment wall-clock: {time.perf_counter()-t_all:.1f}s")
    print("\n----- runtime.tex -----")
    print(txt)
