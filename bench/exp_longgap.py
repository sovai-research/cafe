"""
LONGGAP experiment for the CAFE paper.

Question: how does imputation error grow as the length of a SINGLE contiguous
block gap (per column) increases? Local / point methods (LOCF carries a flat
value across the whole gap, linear interpolation extrapolates a straight line)
must blow up as the gap spans more of the dynamics, whereas a structural model
(CAFE: AR factor dynamics + Fourier seasonality + cross-sectional factors)
should degrade gracefully. SoftImpute is a strong NON-causal batch reference.

Masker: for each column we mask ONE contiguous run of length L = round(frac*T),
starting at a seeded position. The same start logic is reused across methods and
fractions so the only thing that changes is gap LENGTH. We sweep
frac in {0.02, 0.05, 0.10, 0.20, 0.30, 0.40} over several seeds and report MAE /
RMSE on the held-out block cells only.

Self-contained, < ~6 min. Writes:
    paper/figures/longgap.pdf
    paper/tables/longgap.tex
and prints the raw numbers.
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import harness
from viz_common import PALETTE, style_ax
from c_unified_penmf import online_impute as cafe_impute
from c_baselines import locf_impute
from m_baselines import linear_interp
import m_softimpute

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FRACS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2]
CAP_ROWS = 4000

METHODS = [
    ("CAFE",         cafe_impute,            True),   # causal (ours)
    ("LOCF",         locf_impute,            True),   # causal local
    ("LinearInterp", linear_interp,          False),  # np.interp reads the gap's far (future) endpoint -> non-causal
    ("SoftImpute",   m_softimpute.impute,    False),  # non-causal batch ref
]


# --------------------------------------------------------------------------- #
# Contiguous-block masker keyed by gap LENGTH
# --------------------------------------------------------------------------- #
def mask_longgap(T, N, frac, seed=0):
    """ONE contiguous run of length L = round(frac*T) per column, seeded start.

    Starts are spread across the series (and jittered by seed) so the gap does
    not always land on the same dynamics. Every masked column gets exactly the
    same gap length, so error-vs-length is clean. Returns boolean (T, N)."""
    rng = np.random.default_rng(harness.RNG_SEED + 4242 + seed)
    L = max(2, int(round(frac * T)))
    L = min(L, T - 2)                      # always keep some context on each side
    M = np.zeros((T, N), dtype=bool)
    hi = T - L                             # last valid start
    for j in range(N):
        # spread starts deterministically across the column index, jitter per seed
        base = int((j + 0.5) / N * hi)
        jit = int(rng.integers(-hi // 6, hi // 6 + 1)) if hi > 6 else 0
        start = int(np.clip(base + jit, 0, hi))
        M[start:start + L, j] = True
    return M


# --------------------------------------------------------------------------- #
# Data sources (clean matrices), capped for speed
# --------------------------------------------------------------------------- #
def _load():
    out = {}
    # synthetic strong-seasonal 2D harness (known structure)
    syn = harness.gen_2d(T=800, N=20, L=4, rho=0.95, seasonal=True, seed=0)
    out["Synthetic"] = syn
    # ETTh1 real (capped)
    ett = np.load(os.path.join(ROOT, "data", "ETTh1_clean.npy"))[:CAP_ROWS]
    out["ETTh1"] = np.ascontiguousarray(ett, float)
    # Beijing real (capped rows; subset of columns for speed)
    bj = np.load(os.path.join(ROOT, "data", "beijing_clean.npy"))[:CAP_ROWS, :40]
    out["Beijing"] = np.ascontiguousarray(bj, float)
    return out


def _run_one(fn, Xclean, M):
    Xobs = Xclean.copy()
    Xobs[M] = np.nan
    t0 = time.perf_counter()
    pred = np.asarray(fn(Xobs.copy(), {}), dtype=float)
    dt = time.perf_counter() - t0
    m = harness.metrics(Xclean, pred, M)
    return m["mae"], m["rmse"], dt


def run():
    data = _load()
    # results[method][frac] = list of MAE over (dataset, seed); same for rmse
    res_mae = {name: {f: [] for f in FRACS} for name, _, _ in METHODS}
    res_rmse = {name: {f: [] for f in FRACS} for name, _, _ in METHODS}
    timings = {name: [] for name, _, _ in METHODS}

    t_start = time.perf_counter()
    for dname, Xclean in data.items():
        T, N = Xclean.shape
        for f in FRACS:
            for s in SEEDS:
                M = mask_longgap(T, N, f, seed=s)
                for name, fn, _causal in METHODS:
                    mae, rmse, dt = _run_one(fn, Xclean, M)
                    res_mae[name][f].append(mae)
                    res_rmse[name][f].append(rmse)
                    timings[name].append(dt)
        print(f"  [{dname}] {T}x{N} done  "
              f"(elapsed {time.perf_counter()-t_start:.1f}s)")

    # aggregate (mean over datasets+seeds)
    agg_mae = {name: [float(np.mean(res_mae[name][f])) for f in FRACS]
               for name, _, _ in METHODS}
    agg_rmse = {name: [float(np.mean(res_rmse[name][f])) for f in FRACS]
                for name, _, _ in METHODS}
    print(f"\nTotal compute: {time.perf_counter()-t_start:.1f}s")
    return agg_mae, agg_rmse, timings


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
STYLE = {
    "CAFE":         dict(color=PALETTE["teal"],  marker="o", lw=2.2, ms=6, zorder=5),
    "LOCF":         dict(color=PALETTE["amber"], marker="s", lw=1.6, ms=5, zorder=3),
    "LinearInterp": dict(color=PALETTE["red"],   marker="^", lw=1.6, ms=5, zorder=3),
    "SoftImpute":   dict(color=PALETTE["slate"], marker="D", lw=1.6, ms=4.5,
                         zorder=2, ls="--"),
}
LABELS = {"CAFE": "CAFÉ (causal, ours)", "LOCF": "LOCF (causal)",
          "LinearInterp": "Linear interp (non-causal)",
          "SoftImpute": "SoftImpute (non-causal ref.)"}


def make_figure(agg_mae):
    x = [int(round(f * 100)) for f in FRACS]
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    for name, _, _ in METHODS:
        st = STYLE[name]
        ax.plot(x, agg_mae[name], label=LABELS[name],
                markeredgecolor="white", markeredgewidth=0.5, **st)
    style_ax(ax)
    ax.set_xlabel("Contiguous gap length (% of series)", fontsize=8.5)
    ax.set_ylabel("MAE on gap cells (lower better)", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", handlelength=1.8)
    ax.set_title("Error vs. gap length", fontsize=9)
    out = os.path.join(ROOT, "paper", "figures", "longgap.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")
    return out


def make_table(agg_mae):
    # representative short / medium / long gap lengths
    reps = [(0.02, "Short"), (0.10, "Medium"), (0.40, "Long")]
    idx = [FRACS.index(f) for f, _ in reps]
    causal_names = [n for n, _, c in METHODS if c]

    def fmt(v, bold=False):
        s = f"{v:.3f}"
        return ("$\\mathbf{%s}$" % s) if bold else s

    lines = []
    lines.append(r"\begin{table}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\caption{\textbf{Graceful degradation under long contiguous "
                 r"gaps.} MAE $\downarrow$ on the held-out block cells as a single "
                 r"per-column gap grows from short ($2\%$ of the series) to long "
                 r"($40\%$), averaged over three datasets (Synthetic, ETTh1, "
                 r"Beijing) and three seeds. The cheap local baselines saturate: "
                 r"causal LOCF carries the last value, while linear interpolation "
                 r"draws a straight line across the gap from its \emph{far} (future) "
                 r"endpoint---non-causal, yet still blows up. \cafe{} stays bounded "
                 r"via its AR\,+\,season\,+\,factor structure. SoftImpute is a "
                 r"non-causal batch reference. Non-causal methods are "
                 r"\emph{italic}; \textbf{bold} $=$ best \emph{causal} method.}")
    lines.append(r"\label{tab:longgap}")
    lines.append(r"\begin{tabular}{@{}lccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Short ($2\%$) & Medium ($10\%$) & Long ($40\%$) \\")
    lines.append(r"\midrule")

    # best causal per representative length
    best_causal = []
    for k in idx:
        vals = [(n, agg_mae[n][k]) for n in causal_names]
        best_causal.append(min(vals, key=lambda kv: kv[1])[0])

    disp = {"CAFE": r"\textbf{\cafe{} (ours)}", "LOCF": "LOCF",
            "LinearInterp": r"\emph{Linear interp}", "SoftImpute": r"\emph{SoftImpute}"}
    _seen_noncausal = False
    for name, _, causal in METHODS:
        if not causal and not _seen_noncausal:
            lines.append(r"\midrule")
            _seen_noncausal = True
        cells = []
        for col, k in enumerate(idx):
            bold = causal and (name == best_causal[col])
            cells.append(fmt(agg_mae[name][k], bold=bold))
        lines.append(f"{disp[name]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"
    out = os.path.join(ROOT, "paper", "tables", "longgap.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(tex)
    print(f"table  -> {out}")
    return out, tex


if __name__ == "__main__":
    agg_mae, agg_rmse, timings = run()

    print("\n=== MAE vs gap length (mean over datasets+seeds) ===")
    hdr = "gap%  " + "".join(f"{n:>14s}" for n, _, _ in METHODS)
    print(hdr)
    for i, f in enumerate(FRACS):
        row = f"{int(f*100):>4d}  " + "".join(
            f"{agg_mae[n][i]:14.4f}" for n, _, _ in METHODS)
        print(row)

    print("\n=== RMSE vs gap length ===")
    print(hdr)
    for i, f in enumerate(FRACS):
        row = f"{int(f*100):>4d}  " + "".join(
            f"{agg_rmse[n][i]:14.4f}" for n, _, _ in METHODS)
        print(row)

    print("\n=== mean per-call time (s) ===")
    for n, _, _ in METHODS:
        print(f"  {n:14s} {np.mean(timings[n]):.3f}")

    fig_path = make_figure(agg_mae)
    tab_path, tex = make_table(agg_mae)
    print("\n----- longgap.tex -----")
    print(tex)
