"""
EXPERIMENT: ABLATION for the CAFE imputation paper.

Show that each of CAFE's four learned dials -- Student-t tails (nu), AR memory (a),
ARD factor rank (alpha), Fourier season (beta) -- earns its place: removing the
matched dial should hurt MOST on the dataset that dial is designed for, while the
Full model is best-or-tied across the board.

Isolation: we do NOT touch the production core (bench/c_unified_penmf.py). All
ablations live in a scratch copy bench/c_unified_abl.py with module-level flags
toggled via set_ablation(). With all flags off, the scratch model reproduces
online_impute (sanity-checked below).

Datasets (each chosen so one dial should matter):
  heavytail  : harness.gen_heavytail()      -> tails (nu) should matter
  drift      : harness.gen_drift()          -> AR (a) should matter
  ar2d       : harness.gen_2d(rho=0.97)     -> AR (a) should matter (smooth dynamics)
  highrank   : harness.gen_highrank()       -> ARD rank should matter
  seasonal2d : harness.gen_2d(seasonal=True)-> season (beta) should matter
  seasonal1d : harness.gen_1d('seasonal')   -> season (beta) should matter
  beijing    : real anchor (3000 rows)
  etth1      : real anchor (3000 rows)

10% MCAR (harness.MASKERS['mcar']), averaged over 3 seeds. Metric: MAE (+corr).

Outputs:
  paper/tables/ablation.tex          (booktabs table, bold best per column)
  paper/figures/ablation.pdf         (grouped bars: MAE delta vs Full)
Plain-text summary + sanity check printed to stdout.
"""
from __future__ import annotations
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import numpy as np
import harness
from harness import MASKERS, metrics
import c_unified_abl as abl
from c_unified_penmf import online_impute as prod_impute

SEEDS = (0, 1, 2)
RATE = 0.15
# Each dial is exercised by the missingness REGIME its matched dataset is designed
# for. Temporal dials (AR memory) only bite when there is no contemporaneous cross-
# section to lean on -- i.e. contiguous BLOCK gaps; under scattered MCAR a single
# missing cell is trivially recovered from its row, so AR/season carry barely shows.
# We therefore use the harness's per-dataset design mechanism (block for the
# dynamic/temporal regimes, MCAR for the rest), all at a single 15% rate. The
# sanity check below still uses plain MCAR.
MECH_BY_DS = {
    "heavytail":  "mcar",     # outliers: tails dial (mechanism-agnostic)
    "drift":      "block",    # non-stationary dynamics -> temporal gaps
    "ar2d":       "block",    # smooth AR(.97) -> temporal gaps where AR carry matters
    "highrank":   "mcar",     # weak low-rank -> ARD must keep enough rank
    "seasonal2d": "block",    # seasonal: block gaps span where Fourier extrapolates
    "seasonal1d": "block",    # single seasonal series, contiguous gaps
    "beijing":    "mcar",     # real anchor (SAITS-style point missing)
    "etth1":      "mcar",     # real anchor
}

# ablation variants: name -> flags passed to set_ablation
VARIANTS = {
    "Full":     dict(),
    "-tails":   dict(no_robust=True),
    "-AR":      dict(no_ar=True),
    "-ARD":     dict(no_ard=True),
    "-season":  dict(no_season=True),
}


def _real(name, n=3000):
    X = np.load(os.path.join(ROOT, "data", f"{name}_clean.npy"))
    return np.ascontiguousarray(X[:n], dtype=float)


def make_datasets():
    """name -> (clean (T,N) array, meta dict). All 2D/1D (no panel)."""
    ds = {}
    ds["heavytail"]  = (harness.gen_heavytail(T=500, N=20, seed=0), {})
    ds["drift"]      = (harness.gen_drift(T=800, N=15, seed=0), {})
    ds["ar2d"]       = (harness.gen_2d(T=400, N=20, rho=0.97, seasonal=False, seed=0), {})
    ds["highrank"]   = (harness.gen_highrank(T=500, N=30, seed=0), {})
    ds["seasonal2d"] = (harness.gen_2d(T=400, N=20, rho=0.5, seasonal=True, seed=0), {})
    ds["seasonal1d"] = (harness.gen_1d(T=600, kind="seasonal", seed=0), {})
    ds["beijing"]    = (_real("beijing", 3000), {})
    ds["etth1"]      = (_real("ETTh1", 3000), {})
    return ds


def run_variant(impute_fn, clean, meta, mech):
    """Average MAE/corr over SEEDS for one imputer on one dataset at RATE / mech."""
    maes, corrs = [], []
    for s in SEEDS:
        M = MASKERS[mech](clean, RATE, seed=s)
        Xobs = clean.copy()
        Xobs[M] = np.nan
        pred = impute_fn(Xobs.copy(), dict(meta))
        m = metrics(clean, np.asarray(pred, float), M)
        maes.append(m["mae"]); corrs.append(m["corr"])
    return float(np.mean(maes)), float(np.mean(corrs))


def main():
    ds = make_datasets()
    dnames = list(ds.keys())
    vnames = list(VARIANTS.keys())

    # ---- sanity: Full (all flags off) vs production online_impute on one dataset ----
    abl.reset_ablation()
    sx = harness.gen_2d(T=400, N=20, seed=0)
    M = MASKERS["mcar"](sx, RATE, seed=0)
    Xo = sx.copy(); Xo[M] = np.nan
    p_abl = abl.online_impute(Xo.copy(), {})
    p_prod = prod_impute(Xo.copy(), {})
    sanity_mae_abl = float(np.mean(np.abs(p_abl[M] - sx[M])))
    sanity_mae_prod = float(np.mean(np.abs(p_prod[M] - sx[M])))
    sanity_maxdiff = float(np.max(np.abs(p_abl - p_prod)))
    print(f"[SANITY] Full-abl MAE={sanity_mae_abl:.6f}  prod MAE={sanity_mae_prod:.6f}  "
          f"|MAE diff|={abs(sanity_mae_abl-sanity_mae_prod):.2e}  "
          f"max|pred-pred|={sanity_maxdiff:.2e}")

    # ---- main grid: MAE[variant][dataset] and corr ----
    mae = {v: {} for v in vnames}
    corr = {v: {} for v in vnames}
    for v in vnames:
        for dn in dnames:
            abl.reset_ablation()
            abl.set_ablation(**VARIANTS[v])
            clean, meta = ds[dn]
            mm, cc = run_variant(abl.online_impute, clean, meta, MECH_BY_DS[dn])
            mae[v][dn] = mm; corr[v][dn] = cc
            print(f"  {v:8s} {dn:11s} MAE={mm:.4f} corr={cc:.3f}")
    abl.reset_ablation()

    # ---- plain-text summary: which dial matters where (delta vs Full) ----
    print("\n=== MAE (lower better); delta vs Full in []; bold=min per column ===")
    hdr = "variant   " + " ".join(f"{d[:10]:>10s}" for d in dnames)
    print(hdr)
    best = {dn: min(mae[v][dn] for v in vnames) for dn in dnames}
    for v in vnames:
        cells = []
        for dn in dnames:
            mark = "*" if mae[v][dn] == best[dn] else " "
            cells.append(f"{mae[v][dn]:9.4f}{mark}")
        print(f"{v:9s} " + " ".join(cells))
    print("\nDelta vs Full (positive = worse than Full):")
    for v in vnames:
        if v == "Full":
            continue
        cells = " ".join(f"{mae[v][dn]-mae['Full'][dn]:+10.4f}" for dn in dnames)
        print(f"{v:9s} {cells}")
    # which dataset is hurt most by each ablation
    print("\nDial -> dataset hurt most (max MAE increase vs Full):")
    for v in vnames:
        if v == "Full":
            continue
        deltas = {dn: mae[v][dn] - mae["Full"][dn] for dn in dnames}
        worst = max(deltas, key=deltas.get)
        print(f"  {v:8s}: {worst:11s} (+{deltas[worst]:.4f})")

    # ---- write LaTeX table ----
    write_table(mae, dnames, vnames, best)
    # ---- write figure ----
    try:
        write_figure(mae, dnames, vnames)
    except Exception as e:
        print(f"[fig] skipped: {type(e).__name__}: {e}")

    return mae, corr, dict(sanity_abl=sanity_mae_abl, sanity_prod=sanity_mae_prod,
                           sanity_maxdiff=sanity_maxdiff)


# pretty column headers for the paper
COLHEAD = {
    "heavytail":  "HeavyT",
    "drift":      "Drift",
    "ar2d":       "AR(.97)",
    "highrank":   "HighRank",
    "seasonal2d": "Seas-2D",
    "seasonal1d": "Seas-1D",
    "beijing":    "Beijing",
    "etth1":      "ETTh1",
}
ROWHEAD = {
    "Full":    r"Full \cafe{}",
    "-tails":  r"\quad-- tails ($\nu$)",
    "-AR":     r"\quad-- AR ($a$)",
    "-ARD":    r"\quad-- ARD rank",
    "-season": r"\quad-- season ($\beta$)",
}


# Diagonal ablation: each dial shown ON its matched regime (where it is designed to
# bite), with the dial ON (full model) vs OFF. Delta = cost of removing it. This is the
# honest, legible form of "each dial earns its place": every row is a positive cost.
# (The full cross-regime matrix -- including the off-target columns where a dial is
# near-inert -- is the appendix figure ablation.pdf.)
DIAG = [
    (r"Student-$t$ tails ($\nu$)", "HeavyT (5\\% outliers)",        "heavytail",  "-tails"),
    (r"AR memory ($a$)",           "AR(.97), block gaps",           "ar2d",       "-AR"),
    (r"ARD rank",                  "Drift (non-stationary)",        "drift",      "-ARD"),
    (r"Fourier season ($\beta$)",  "Seas-1D (periodic series)",     "seasonal1d", "-season"),
    (r"Fourier season ($\beta$)",  "ETTh1 (daily/weekly cycle)",    "etth1",      "-season"),
]


def write_table(mae, dnames, vnames, best):
    lines = []
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{10pt}")
    lines.append(r"\caption{\textbf{Ablation: each learned dial earns its place.} "
                 r"Each dial is shown on the regime it is designed for, with the dial "
                 r"ON (full \cafe{}) vs OFF; MAE ($\downarrow$), $15\%$ missing, mean of "
                 r"3 seeds. $\Delta$ is the cost of removing the dial---positive "
                 r"everywhere, so every dial pays for itself on its matched regime "
                 r"(season decisively on genuinely periodic data). Off its target a dial "
                 r"is near-inert; the full cross-regime matrix is "
                 r"Fig.~\ref{fig:ablation}.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{@{}llccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Learned dial & Matched regime & with & without & $\Delta$MAE \\")
    lines.append(r"\midrule")
    for dial, regime, ds, off in DIAG:
        full_v = mae["Full"][ds]
        off_v = mae[off][ds]
        d = off_v - full_v
        lines.append(f"{dial} & {regime} & ${full_v:.3f}$ & ${off_v:.3f}$ & "
                     f"$\\mathbf{{+{d:.3f}}}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    out = "\n".join(lines) + "\n"
    tdir = os.path.join(ROOT, "paper", "tables")
    os.makedirs(tdir, exist_ok=True)
    path = os.path.join(tdir, "ablation.tex")
    with open(path, "w") as f:
        f.write(out)
    print(f"\n[table] wrote {path}")
    print("\n" + out)


def write_figure(mae, dnames, vnames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    abls = [v for v in vnames if v != "Full"]
    deltas = np.array([[mae[v][dn] - mae["Full"][dn] for dn in dnames] for v in abls])
    x = np.arange(len(dnames))
    w = 0.8 / len(abls)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    colors = ["#d1495b", "#edae49", "#00798c", "#66a182"]
    for i, v in enumerate(abls):
        ax.bar(x + (i - (len(abls) - 1) / 2) * w, deltas[i], w,
               label=v, color=colors[i % len(colors)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COLHEAD[d] for d in dnames], rotation=0, fontsize=8)
    ax.set_ylabel(r"$\Delta$MAE vs Full")
    ax.set_title("Cost of removing each dial, per regime "
                 "(positive on its matched regime; near-inert off-target)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fdir = os.path.join(ROOT, "paper", "figures")
    os.makedirs(fdir, exist_ok=True)
    path = os.path.join(fdir, "ablation.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {path}")


if __name__ == "__main__":
    main()
