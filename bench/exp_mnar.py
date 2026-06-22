"""Experiment MNAR: robustness of CAFE across missingness MECHANISMS.

The field is moving from "MCAR at 20%" toward stress-testing imputers under
harder, structured missingness mechanisms. This script measures CAFE (the causal
c_unified_penmf core) against simple local causal baselines (LOCF, linear
interpolation) and a strong NON-causal batch reference (SoftImpute-ALS) across
four mechanisms defined in harness.py:

  mcar  : entries dropped independently and uniformly (rate 20%).            [easy]
  mar   : missingness in column j is a logistic function of the value of the
          neighbouring column j-1 -- missing-at-random given observed data.
  mnar  : self-masking. Each cell's missing probability rises with its own
          (standardized) value (p ~ sigmoid(z-1)); large values self-censor.
          The classic hard case: a low-rank/mean model under-predicts the
          missing high tail, so MAE is biased upward for everyone.
  block : per-column contiguous time-blocks vanish (sensor-blackout style),
          so there is no nearby observation to interpolate from.

All four run at ~20% missing, 3 seeds each, on beijing[:3000], ETTh1[:3000]
and a synthetic gen_2d panel. We report MAE (lower better) and corr.

Thesis: CAFE stays robust where LOCF / linear-interp crater under block and
MNAR, and stays competitive with the non-causal batch reference -- arguing from
RELATIVE robustness, since MNAR is genuinely hard and biases everyone.

Outputs:
  paper/tables/mnar.tex      (booktabs, \\input-ready, rows=mechanism)
  paper/figures/mnar.pdf     (grouped bars of MAE by mechanism)
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness import MASKERS, metrics, gen_2d
from c_unified_penmf import online_impute as cafe_impute
from c_baselines import locf_impute
from m_baselines import linear_interp
import m_softimpute

import viz_common as V

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG_OUT = os.path.join(ROOT, "paper", "figures", "mnar.pdf")
TAB_OUT = os.path.join(ROOT, "paper", "tables", "mnar.tex")

ROWS_CAP = 3000
RATE = 0.20
SEEDS = (0, 1, 2)
MECHS = ("mcar", "mar", "mnar", "block")
MECH_LABEL = {"mcar": "MCAR", "mar": "MAR", "mnar": "MNAR", "block": "Block"}

# method label -> (callable, is_causal)
METHODS = [
    ("CAFE", cafe_impute, True),
    ("LOCF", locf_impute, True),
    ("LinInterp", linear_interp, True),
    ("SoftImpute", m_softimpute.impute, False),
]


def eval_cell(X, mech, fn):
    """Mean MAE / corr over SEEDS for one (dataset, mechanism, method)."""
    maes, corrs = [], []
    for s in SEEDS:
        M = MASKERS[mech](X, RATE, seed=s)
        Xobs = X.copy()
        Xobs[M] = np.nan
        try:
            pred = np.asarray(fn(Xobs.copy(), {}), float)
            m = metrics(X, pred, M)
            maes.append(m["mae"]); corrs.append(m["corr"])
        except Exception as e:                       # noqa: BLE001
            print(f"    FAIL {mech}: {type(e).__name__}: {e}", flush=True)
            maes.append(np.nan); corrs.append(np.nan)
    return float(np.nanmean(maes)), float(np.nanmean(corrs))


def main():
    t_start = time.perf_counter()
    beijing = np.load(os.path.join(ROOT, "data", "beijing_clean.npy"))[:ROWS_CAP]
    etth1 = np.load(os.path.join(ROOT, "data", "ETTh1_clean.npy"))[:ROWS_CAP]
    syn = gen_2d(T=1500, N=20, seed=7)
    datasets = [
        ("Beijing", np.ascontiguousarray(beijing, float)),
        ("ETTh1", np.ascontiguousarray(etth1, float)),
        ("Synthetic", np.ascontiguousarray(syn, float)),
    ]

    # res[mech][method] = list of (mae, corr) per dataset
    res = {m: {meth[0]: [] for meth in METHODS} for m in MECHS}
    for dname, X in datasets:
        print(f"[{dname}] X={X.shape}", flush=True)
        for mech in MECHS:
            line = f"  {mech:6s}"
            for mlabel, fn, _ in METHODS:
                mae, corr = eval_cell(X, mech, fn)
                res[mech][mlabel].append((mae, corr))
                line += f"  {mlabel}={mae:.3f}"
            print(line, flush=True)

    # average over datasets
    avg = {m: {} for m in MECHS}
    for mech in MECHS:
        for mlabel, _, _ in METHODS:
            pairs = res[mech][mlabel]
            maes = [p[0] for p in pairs]
            corrs = [p[1] for p in pairs]
            avg[mech][mlabel] = (float(np.nanmean(maes)), float(np.nanmean(corrs)))

    tex = make_table(avg)
    with open(TAB_OUT, "w") as f:
        f.write(tex)
    sz = make_figure(avg)

    print(f"\nsaved table  {TAB_OUT}", flush=True)
    print(f"saved figure {FIG_OUT} ({sz} bytes)", flush=True)
    print(f"\ntotal {time.perf_counter() - t_start:.1f}s", flush=True)

    print("\n=== MAE (avg over 3 datasets, 3 seeds) ===", flush=True)
    hdr = "mech    " + "".join(f"{m[0]:>12s}" for m in METHODS)
    print(hdr)
    for mech in MECHS:
        row = f"{mech:6s}  "
        for mlabel, _, _ in METHODS:
            row += f"{avg[mech][mlabel][0]:12.3f}"
        print(row)
    print("\n=== TABLE LATEX ===")
    print(tex)
    return avg


# --------------------------------------------------------------------------- #
def make_table(avg):
    causal_labels = [m[0] for m in METHODS if m[2]]
    lines = []
    lines.append(r"\begin{table}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(
        r"\caption{Robustness across missingness \emph{mechanisms} "
        r"(MAE $\downarrow$, averaged over Beijing, ETTh1 and a synthetic "
        r"panel; $20\%$ missing, 3 seeds). \textbf{MCAR}: entries dropped "
        r"independently and uniformly. \textbf{MAR}: missingness in a column "
        r"is a logistic function of the neighbouring column's value "
        r"(missing-at-random given observed data). \textbf{MNAR}: self-masking "
        r"-- each cell's drop probability rises with its own value, so large "
        r"values censor themselves (the hard case that biases every estimator). "
        r"\textbf{Block}: per-column contiguous blackouts, leaving no nearby "
        r"observation to interpolate from. \cafe{} and LOCF/LinInterp are "
        r"strictly causal (point-in-time); SoftImpute is a non-causal batch "
        r"reference (\emph{italic}). Bold $=$ best \emph{causal} method per row.}")
    lines.append(r"\label{tab:mnar}")
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(METHODS) + r"@{}}")
    lines.append(r"\toprule")
    hdr = "Mechanism"
    for mlabel, _, causal in METHODS:
        hdr += " & " + (mlabel if causal else r"\textit{" + mlabel + r"}")
    lines.append(hdr + r" \\")
    lines.append(r"\midrule")
    for mech in MECHS:
        # best causal MAE for bolding
        best = min(avg[mech][c][0] for c in causal_labels)
        cells = []
        for mlabel, _, causal in METHODS:
            mae = avg[mech][mlabel][0]
            s = f"{mae:.3f}"
            if causal and abs(mae - best) < 1e-9:
                s = r"\textbf{" + s + r"}"
            elif not causal:
                s = r"\textit{" + s + r"}"
            cells.append(s)
        lines.append(f"{MECH_LABEL[mech]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_figure(avg):
    PAL = V.PALETTE
    colors = {"CAFE": PAL["blue"], "LOCF": PAL["amber"],
              "LinInterp": PAL["green"], "SoftImpute": PAL["grey"]}
    labels = [m[0] for m in METHODS]
    fig, ax = plt.subplots(figsize=(6.2, 2.9), constrained_layout=True)
    nmech = len(MECHS)
    nmeth = len(labels)
    width = 0.8 / nmeth
    x = np.arange(nmech)
    for i, lab in enumerate(labels):
        vals = [avg[m][lab][0] for m in MECHS]
        off = (i - (nmeth - 1) / 2) * width
        hatch = "//" if not METHODS[i][2] else None
        ax.bar(x + off, vals, width=width * 0.95, color=colors[lab],
               label=(lab + " (ref)" if not METHODS[i][2] else lab),
               hatch=hatch, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([MECH_LABEL[m] for m in MECHS])
    ax.set_ylabel("MAE (lower better)", fontsize=9)
    ax.set_title("Imputation error by missingness mechanism", fontsize=9.5, pad=6)
    ax.legend(fontsize=7.0, frameon=False, ncol=2, handlelength=1.3,
              columnspacing=0.9, loc="upper left")
    V.style_ax(ax)
    fig.savefig(FIG_OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return os.path.getsize(FIG_OUT)


if __name__ == "__main__":
    main()
