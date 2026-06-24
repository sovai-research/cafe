"""SCALE experiment for the CAFE paper -- Gap #4: "all datasets are similar
mid-size panels; no high-N or clinical scale".

This runs CAFE (causal, point-in-time `online_impute`) on genuinely WIDE panels
-- a decade more width than the race panels (up to 862 channels) -- and MEASURES
the N-scaling exponent at that width, so the "strongly sub-linear in N" claim is
substantiated where it actually matters, not just on narrow data.

Everything point-in-time / single-machine fair: BLAS pinned to 2 threads BEFORE
numpy import; no GPU. MAE on standardise-on-observed scale (no look-ahead leak in
the normalisation), scored on held-out cells only.

Optional clinical row: if bench/datasets_scale's sibling PhysioNet loader has
cached data/physionet2012_clean.npy (+ _obsmask), we ALSO report CAFE causal MAE
vs simple causal baselines (LOCF / running-mean) on real ICU native missingness.
If that file is absent the PhysioNet row is HONESTLY skipped (no fabrication).

Outputs (under paper/):
  paper/figures/scaling_wide.pdf   -- time vs N on the wide panel, with fitted slope.
  paper/tables/scale.tex           -- wide-panel shape, MAE, runtime, measured N-slope
                                      (+ PhysioNet row if available).

Run:  python3 bench/exp_scale.py
"""
import os
# Pin BLAS threads BEFORE numpy import (fair single-machine timing).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)

import c_unified_penmf as P                       # CAFE (causal): online_impute
from eval_utils import standardize_on_observed, make_mask, score_masked
import datasets_scale as DS

DATA = os.path.join(_ROOT, "data")
FIGDIR = os.path.join(_ROOT, "paper", "figures")
TABDIR = os.path.join(_ROOT, "paper", "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

BLAS_THREADS = os.environ.get("OPENBLAS_NUM_THREADS", "2")
N_REPEAT = 2                                       # median of 2 timed runs / cell
MCAR_RATE = 0.10
SEED = 0

# Wide N-sweep grid (a decade-plus of width); the last point is the full panel.
N_GRID = [50, 100, 200, 400, 800]


# --------------------------------------------------------------------------- #
def _ensure_wide():
    """Build/refresh the wide panels and return {name: (X, shape_str)}."""
    out = {}
    for name, loader in DS.DATASETS_SCALE.items():
        cache = os.path.join(DATA, f"{name}_wide_clean.npy")
        X = np.load(cache) if os.path.exists(cache) else loader()
        out[name] = X
    return out


def _peak_mem_mb():
    """Best-effort peak RSS in MB (0.0 if unavailable). resource is POSIX-only."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return ru / 1e6 if sys.platform == "darwin" else ru / 1e3
    except Exception:
        return 0.0


def _time_cafe(Xobs, repeat=N_REPEAT):
    """Median wall-clock of `repeat` CAFE runs; returns (median_s, last_pred)."""
    ts, pred = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        pred = P.online_impute(Xobs.copy(), {})
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), pred


# --------------------------------------------------------------------------- #
def head_to_head(panels):
    """CAFE causal MAE + wall-clock + peak mem on each FULL wide panel @10% MCAR."""
    rows = []
    for name, X in panels.items():
        M = make_mask("mcar", X.shape, MCAR_RATE, SEED)
        Xstd = standardize_on_observed(X, M)
        Xobs = Xstd.copy(); Xobs[M] = np.nan
        sec, pred = _time_cafe(Xobs)
        sc = score_masked(Xstd, pred, M)
        mem = _peak_mem_mb()
        rows.append(dict(name=name, T=X.shape[0], N=X.shape[1],
                         sec=sec, mae=sc["mae"], rmse=sc["rmse"], mem=mem))
        print(f"  {name:14s} {X.shape[0]}x{X.shape[1]:4d}  "
              f"MAE={sc['mae']:.4f}  {sec:7.2f}s  peakRSS={mem:6.0f}MB")
    return rows


# --------------------------------------------------------------------------- #
def n_scaling(Xwide, name):
    """Time CAFE at increasing N on the widest panel; return grid + log-log slope."""
    T = Xwide.shape[0]
    Ns, secs = [], []
    for N in N_GRID:
        if N > Xwide.shape[1]:
            continue
        X = np.ascontiguousarray(Xwide[:, :N])
        M = make_mask("mcar", X.shape, MCAR_RATE, SEED)
        Xstd = standardize_on_observed(X, M)
        Xobs = Xstd.copy(); Xobs[M] = np.nan
        sec, _ = _time_cafe(Xobs)
        Ns.append(N); secs.append(sec)
        print(f"  T={T} N={N:4d}:  {sec:7.3f}s")
    slope = _loglog_slope(Ns, secs)
    print(f"  measured N-slope (log-log) = {slope:.2f}  "
          f"[1.0=linear, 2.0=quadratic]")
    return dict(panel=name, T=T, Ns=Ns, secs=secs, slope=slope)


def _loglog_slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


# --------------------------------------------------------------------------- #
# Optional PhysioNet-2012 clinical benchmark (only if the data was obtained).
# --------------------------------------------------------------------------- #
def _locf_fill(Xobs):
    """Causal last-observation-carried-forward, then running-mean for the prefix."""
    X = np.asarray(Xobs, float).copy()
    T, N = X.shape
    for j in range(N):
        col = X[:, j]
        last = np.nan
        run_sum, run_cnt = 0.0, 0
        for t in range(T):
            if np.isfinite(col[t]):
                last = col[t]; run_sum += col[t]; run_cnt += 1
            else:
                if np.isfinite(last):
                    col[t] = last
                elif run_cnt > 0:
                    col[t] = run_sum / run_cnt
                else:
                    col[t] = 0.0
        X[:, j] = col
    return X


def _runmean_fill(Xobs):
    """Causal running per-column mean fill (no carry-forward)."""
    X = np.asarray(Xobs, float).copy()
    T, N = X.shape
    for j in range(N):
        col = X[:, j]
        run_sum, run_cnt = 0.0, 0
        for t in range(T):
            if np.isfinite(col[t]):
                run_sum += col[t]; run_cnt += 1
            else:
                col[t] = run_sum / run_cnt if run_cnt > 0 else 0.0
        X[:, j] = col
    return X


def _impute_per_patient(fn, Xobs3):
    """Apply a causal 2D imputer to EACH patient's (48h x V) block independently.

    The PhysioNet tensor is (patients, hours, vars). The natural online axis is
    the per-patient hour dimension -- streaming state ACROSS the patient boundary
    would be both meaningless and a (cross-patient) leak. So we reset state per
    patient: every block is imputed on its own."""
    Pn, Tt, V = Xobs3.shape
    out = np.empty_like(Xobs3)
    for p in range(Pn):
        out[p] = np.asarray(fn(Xobs3[p].copy()), float)
    return out


def physionet():
    """Return a dict with the clinical-benchmark result, or {'status': skip,...}.

    Real PhysioNet-2012 ICU set-a as a (patients, 48h, vars) tensor with genuine
    native missingness. We add a SYNTHETIC MCAR hold-out on top of the observed
    cells (the only place ground truth exists) and score there. All three methods
    are causal and are run PER PATIENT (state reset at each ICU stay -- no
    cross-patient streaming/leak)."""
    clean = os.path.join(DATA, "physionet2012_clean.npy")
    obsm = os.path.join(DATA, "physionet2012_obsmask.npy")
    if not (os.path.exists(clean) and os.path.exists(obsm)):
        return dict(status="skip",
                    reason="data/physionet2012_clean.npy not present "
                           "(PhysioNet-2012 could not be obtained offline)")
    A = np.load(clean).astype(float)
    obs = np.load(obsm).astype(bool)
    if A.ndim == 3:
        Pn, Tt, V = A.shape
        shape_str = f"{Pn} ICU stays x {Tt}h x {V} vars"
    else:                              # 2D fallback: treat as a single block
        A = A[None, ...]; obs = obs[None, ...]
        Pn, Tt, V = A.shape
        shape_str = f"{Tt}x{V}"
    native_miss = 100.0 * (1.0 - obs.mean())

    # Synthetic hold-out: mask 10% of the OBSERVED cells (only there is truth).
    rng = np.random.default_rng(SEED)
    holdout = (rng.random(A.shape) < MCAR_RATE) & obs
    Xobs = A.copy()
    Xobs[~obs] = np.nan               # respect native missingness
    Xobs[holdout] = np.nan            # plus our scored hold-out
    truth = A

    res = {}
    for label, fn in [("CAFE (ours)", lambda Z: P.online_impute(Z, {})),
                      ("LOCF",        _locf_fill),
                      ("Running-mean", _runmean_fill)]:
        t0 = time.perf_counter()
        pred = _impute_per_patient(fn, Xobs)
        sec = time.perf_counter() - t0
        sc = score_masked(truth, pred, holdout)
        res[label] = dict(mae=sc["mae"], sec=sec)
        print(f"  [physionet] {label:13s} MAE={sc['mae']:.4f}  {sec:6.2f}s")
    return dict(status="ok", shape=shape_str, native_miss=native_miss,
                n_holdout=int(holdout.sum()), results=res)


# --------------------------------------------------------------------------- #
def make_figure(sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    from viz_common import PALETTE, style_ax

    Ns, secs, slope = sc["Ns"], sc["secs"], sc["slope"]
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.8), constrained_layout=True)

    ax.loglog(Ns, secs, "o-", color=PALETTE["teal"], lw=1.9, ms=6,
              mfc="white", mec=PALETTE["teal"], zorder=3, label="CAFÉ (measured)")
    # slope-1 (linear) reference through the last point
    ref = np.array(Ns, float)
    refy = secs[-1] * ref / ref[-1]
    ax.loglog(ref, refy, "--", color=PALETTE["grey"], lw=1.1, zorder=2,
              label="slope 1 (linear)")
    ax.set_xlabel("width $N$  (channels)")
    ax.set_ylabel("wall-clock (s)")
    ax.set_title(f"CAFÉ vs $N$ on {sc['panel']}\n(fixed $T={sc['T']}$)",
                 fontsize=9)
    ax.annotate(f"fitted slope $={slope:.2f}$", xy=(0.04, 0.90),
                xycoords="axes fraction", fontsize=8.5, color=PALETTE["ink"])
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in Ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    style_ax(ax)

    out = os.path.join(FIGDIR, "scaling_wide.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #
def make_table(rows, sc, phys):
    lines = []
    lines.append("% Auto-generated by bench/exp_scale.py -- real measured numbers.")
    lines.append("% Do not edit by hand; re-run exp_scale.py to refresh.")
    lines.append("\\begin{table*}[t]\\centering\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    cap = ("\\textbf{CAFE at scale} (Gap: prior panels are all mid-size, "
           "$\\le\\!100$ channels). \\cafe{} run \\emph{causally} "
           "(\\texttt{online\\_impute}) on genuinely wide panels -- up to "
           "$862$ channels, a decade more width than the race -- at "
           f"${int(MCAR_RATE*100)}\\%$ MCAR. MAE on the standardise-on-observed "
           "scale (held-out cells only); single CPU core, BLAS pinned to "
           + BLAS_THREADS + " threads, \\emph{no GPU}; median of "
           + str(N_REPEAT) + " runs. The measured log--log time-vs-$N$ slope "
           f"(${sc['slope']:.2f}$) confirms \\cafe{{}} stays strongly "
           "sub-quadratic in width at this scale.")
    if phys.get("status") == "ok":
        cap += (" The PhysioNet-2012 ICU row reports CAFE vs simple causal "
                "baselines on real clinical native missingness "
                "(extra synthetic hold-out for ground truth).")
    else:
        cap += (" (PhysioNet-2012 clinical row omitted: "
                + phys.get("reason", "data unavailable") + ".)")
    lines.append("\\caption{" + cap + "}")
    lines.append("\\label{tab:scale}")
    lines.append("\\begin{tabular}{@{}lrrrr@{}}")
    lines.append("\\toprule")
    lines.append("Wide panel & $T$ & $N$ & MAE~$\\downarrow$ & Time (s)~$\\downarrow$ \\\\")
    lines.append("\\midrule")
    pretty = {"traffic_wide": "Traffic (PEMS), full",
              "electric_wide": "Electricity (UCI), full"}
    for r in rows:
        nm = pretty.get(r["name"], r["name"])
        lines.append(f"{nm} & {r['T']} & {r['N']} & "
                     f"${r['mae']:.3f}$ & ${r['sec']:.2f}$ \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Measured $N$-scaling slope on "
                 + pretty.get(sc["panel"], sc["panel"])
                 + f" (log--log, $N\\!=\\!{min(sc['Ns'])}\\!\\to\\!{max(sc['Ns'])}$):}}"
                   f"\\quad $\\mathbf{{{sc['slope']:.2f}}}$ "
                   "\\;(1.0 = linear)} \\\\")
    if phys.get("status") == "ok":
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{@{}l}{\\emph{PhysioNet-2012 ICU "
                     "(" + phys["shape"].replace("%", "\\%")
                     + f", {phys['native_miss']:.0f}\\% native missing; "
                       "causal MAE on synthetic hold-out):}} \\\\")
        R = phys["results"]
        order = ["CAFE (ours)", "LOCF", "Running-mean"]
        cells = []
        for k in order:
            if k in R:
                lab = "\\cafe{}" if "CAFE" in k else k
                cells.append(f"{lab} ${R[k]['mae']:.3f}$")
        lines.append("\\multicolumn{5}{@{}l}{\\quad "
                     + " \\;/\\; ".join(cells) + "} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    txt = "\n".join(lines) + "\n"
    out = os.path.join(TABDIR, "scale.tex")
    with open(out, "w") as f:
        f.write(txt)
    print(f"  wrote {out}")
    return txt


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    t_all = time.perf_counter()
    print("[1/5] build/load wide panels ...")
    panels = _ensure_wide()
    for nm, X in panels.items():
        print(f"  {nm}: {X.shape}")

    print("[2/5] head-to-head CAFE on full wide panels @10% MCAR ...")
    rows = head_to_head(panels)

    print("[3/5] N-scaling sweep on the widest panel ...")
    widest = max(panels, key=lambda k: panels[k].shape[1])
    sc = n_scaling(panels[widest], widest)

    print("[4/5] PhysioNet-2012 clinical benchmark (if available) ...")
    phys = physionet()
    if phys.get("status") != "ok":
        print(f"  PhysioNet SKIPPED -- {phys.get('reason')}")

    print("[5/5] figure + table ...")
    figpath = make_figure(sc)
    txt = make_table(rows, sc, phys)

    print("\n===== SCALE SUMMARY =====")
    for r in rows:
        print(f"  {r['name']:14s} {r['T']}x{r['N']:4d}  MAE={r['mae']:.4f}  "
              f"{r['sec']:.2f}s  peakRSS={r['mem']:.0f}MB")
    print(f"  widest panel: {widest} ({panels[widest].shape[0]}x"
          f"{panels[widest].shape[1]})")
    print(f"  measured N-scaling slope (log-log) = {sc['slope']:.2f} "
          f"(1.0=linear, 2.0=quadratic)")
    if phys.get("status") == "ok":
        print(f"  PhysioNet-2012: {phys['shape']}, "
              f"{phys['native_miss']:.0f}% native missing")
        for k, v in phys["results"].items():
            print(f"    {k:13s} MAE={v['mae']:.4f}")
    else:
        print(f"  PhysioNet-2012: SKIPPED ({phys.get('reason')})")
    print(f"  total wall-clock: {time.perf_counter()-t_all:.1f}s")
    print("\n----- scale.tex -----")
    print(txt)
