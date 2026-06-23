"""Experiment MASK-GRID: CAFE vs classical CPU baselines across a mask grid.

Moves the benchmark off correlation onto MAE / RMSE as the PRIMARY metrics, and
sweeps a full grid of missingness PATTERN x RATE so no method can win by a single
lucky configuration:

  Pattern in {Point/MCAR, Subsequence, Block}
  Rate    in {0.1, 0.3, 0.5}

evaluated on the available real datasets (data/*.npy and the VLDB *_normal.txt
files, loaded via bench/harness.py conventions) plus 1-2 synthetic cases, and
additionally STRATIFIES one block-missing run by gap length (short / medium /
long contiguous gaps) to show where each method degrades.

Patterns
  Point/MCAR  : harness.mask_mcar  -- entries dropped independently/uniformly.
  Subsequence : per-column random contiguous RUNS of fixed-ish target length
                (the SAITS/TSI-Bench "subseq" scenario). Defined locally because
                harness has no subseq masker; it returns the per-cell gap length
                so we can stratify.
  Block       : harness.mask_block -- per-column contiguous blackouts with a
                wide range of block lengths (good for gap-length stratification).

Causality is LABELLED via bench/causal.py conventions: CAFE and LOCF are strictly
causal (point-in-time); LinInterp (np.interp reads the NEXT observed value) and
SoftImpute (batch, two-sided) are NON-causal references shown for context in
italics -- never bolded as winners. Bold = best CAUSAL method per cell.

Small caps (capped rows, few seeds) so the whole grid runs in seconds.

Outputs
  paper/tables/maskgrid.tex   (booktabs, \\input-ready, label tab:maskgrid)
                              ALWAYS written -- a valid stub even if a run fails,
                              so \\input{tables/maskgrid} never breaks the build.
Only numbers that were actually computed are written; failed cells render as "--".
"""
import os
import sys
import time

# keep BLAS threads modest so timings are stable and the grid stays fast
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from harness import RNG_SEED, mask_mcar, mask_block, metrics, gen_2d, gen_1d
from c_unified_penmf import online_impute as cafe_impute
from c_baselines import locf_impute
from m_baselines import linear_interp
import m_softimpute

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
TAB_OUT = os.path.join(ROOT, "paper", "tables", "maskgrid.tex")

# --- grid -----------------------------------------------------------------
RATES = (0.1, 0.3, 0.5)
PATTERNS = ("mcar", "subseq", "block")
PATTERN_LABEL = {"mcar": "Point/MCAR", "subseq": "Subsequence", "block": "Block"}
SEEDS = (0, 1, 2)
ROWS_CAP = 1500            # cap real series length so the grid runs in seconds

# method label -> (callable, is_causal)
METHODS = [
    ("CAFE", cafe_impute, True),
    ("LOCF", locf_impute, True),
    ("LinInterp", linear_interp, False),     # reads the NEXT obs -> non-causal
    ("SoftImpute", m_softimpute.impute, False),
]
CAUSAL_LABELS = [m[0] for m in METHODS if m[2]]


# --------------------------------------------------------------------------- #
# Subsequence masker: per-column contiguous runs (with gap-length labels)
# --------------------------------------------------------------------------- #
def mask_subseq(X, rate, seed=0, gap=None):
    """Per-column contiguous RUNS of missing values (subsequence scenario).

    Unlike harness.mask_block (which draws a wide range of block lengths), this
    targets a roughly FIXED run length `gap` so the gap-length stratification is
    meaningful. Returns (M, glen) where glen[t,j] is the length of the contiguous
    missing run that cell (t,j) belongs to (0 where observed).
    """
    rng = np.random.default_rng(RNG_SEED + 23 + seed)
    T, N = X.shape
    if gap is None:
        gap = max(2, T // 40)
    M = np.zeros((T, N), dtype=bool)
    target = int(rate * T * N)
    placed = 0
    guard = 0
    while placed < target and guard < target * 50 + 1000:
        guard += 1
        j = int(rng.integers(N))
        blen = int(max(2, rng.integers(max(2, gap - 1), gap + 2)))
        start = int(rng.integers(0, max(1, T - blen)))
        seg = slice(start, start + blen)
        newly = int(np.sum(~M[seg, j]))
        M[seg, j] = True
        placed += newly
    glen = _gap_lengths(M)
    return M, glen


def _gap_lengths(M):
    """For each True cell, length of its contiguous (per-column) missing run."""
    T, N = M.shape
    glen = np.zeros((T, N), dtype=int)
    for j in range(N):
        t = 0
        col = M[:, j]
        while t < T:
            if col[t]:
                s = t
                while t < T and col[t]:
                    t += 1
                glen[s:t, j] = t - s
            else:
                t += 1
    return glen


def make_mask(pattern, X, rate, seed):
    """Return (M, glen_or_None). glen only produced for gap-style patterns."""
    if pattern == "mcar":
        return mask_mcar(X, rate, seed=seed), None
    if pattern == "block":
        M = mask_block(X, rate, seed=seed)
        return M, _gap_lengths(M)
    if pattern == "subseq":
        return mask_subseq(X, rate, seed=seed)
    raise ValueError(pattern)


# --------------------------------------------------------------------------- #
# Datasets: real (loaded via harness conventions) + synthetic
# --------------------------------------------------------------------------- #
def _load_real():
    """Real published datasets used by the repo, loaded lazily; skip if absent."""
    out = []
    # point-protocol reals saved by prep_real.py as clean (T,N) z-scored arrays
    for k in ("beijing", "ETTh1"):
        p = os.path.join(DATA, f"{k}_clean.npy")
        if os.path.exists(p):
            X = np.load(p)[:ROWS_CAP]
            out.append((k.capitalize() if k == "beijing" else k,
                        np.ascontiguousarray(X, float)))
    # VLDB *_normal.txt reals (already normalized in those files)
    vldb = {"AirQ": "airq_normal.txt", "Chlorine": "chlorine_normal.txt"}
    for label, fn in vldb.items():
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            X = np.loadtxt(p)[:ROWS_CAP]
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            out.append((label, np.ascontiguousarray(X, float)))
    return out


def _datasets():
    ds = _load_real()
    # 1-2 synthetic cases: a 2D low-rank+AR panel and a strong-seasonal 1D series
    ds.append(("Syn2D", np.ascontiguousarray(gen_2d(T=800, N=20, seed=7), float)))
    ds.append(("Syn1Dseas",
               np.ascontiguousarray(gen_1d(T=800, kind="seasonal", seed=7), float)))
    return ds


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_cell(X, pattern, rate, fn):
    """Mean MAE / RMSE over SEEDS for one (dataset, pattern, rate, method)."""
    maes, rmses = [], []
    for s in SEEDS:
        M, _ = make_mask(pattern, X, rate, s)
        if not M.any():
            continue
        Xobs = X.copy()
        Xobs[M] = np.nan
        try:
            pred = np.asarray(fn(Xobs.copy(), {}), float)
            m = metrics(X, pred, M)
            maes.append(m["mae"]); rmses.append(m["rmse"])
        except Exception as e:                       # noqa: BLE001
            print(f"      FAIL {pattern} r={rate}: {type(e).__name__}: {e}",
                  flush=True)
    if not maes:
        return float("nan"), float("nan")
    return float(np.nanmean(maes)), float(np.nanmean(rmses))


def run_grid(datasets):
    """grid[(pattern,rate)][method] = (mae, rmse) averaged over datasets+seeds."""
    # accumulate per (pattern,rate,method): lists of (mae,rmse) across datasets
    acc = {(p, r): {m[0]: [] for m in METHODS} for p in PATTERNS for r in RATES}
    for dname, X in datasets:
        print(f"[{dname}] X={X.shape}", flush=True)
        for pattern in PATTERNS:
            for rate in RATES:
                line = f"  {PATTERN_LABEL[pattern]:11s} r={rate:.1f}"
                for mlabel, fn, _ in METHODS:
                    mae, rmse = eval_cell(X, pattern, rate, fn)
                    acc[(pattern, rate)][mlabel].append((mae, rmse))
                    line += f"  {mlabel}={mae:.3f}"
                print(line, flush=True)
    grid = {}
    for key, bym in acc.items():
        grid[key] = {}
        for mlabel in (m[0] for m in METHODS):
            pairs = bym[mlabel]
            maes = [p[0] for p in pairs]
            rmses = [p[1] for p in pairs]
            grid[key][mlabel] = (float(np.nanmean(maes)) if maes else float("nan"),
                                 float(np.nanmean(rmses)) if rmses else float("nan"))
    return grid


def run_gap_stratified(X, rate=0.3, seed=0):
    """Stratify ONE block-missing run by gap length (short/medium/long).

    Returns dict: {(band_label, 'mae'/'rmse'/'n'): {method: value}} plus the
    band edges, all computed on a single shared block mask so the bands are a
    true partition of the same masked cells.
    """
    M = mask_block(X, rate, seed=seed)
    glen = _gap_lengths(M)
    g = glen[M]                                   # gap length of every masked cell
    if g.size == 0:
        return None
    # three roughly equal-population bands by gap length
    q1, q2 = np.quantile(g, [1 / 3, 2 / 3])
    bands = [("short", g <= q1),
             ("medium", (g > q1) & (g <= q2)),
             ("long", g > q2)]
    edges = (float(q1), float(q2), int(g.min()), int(g.max()))

    Xobs = X.copy(); Xobs[M] = np.nan
    preds = {}
    for mlabel, fn, _ in METHODS:
        try:
            preds[mlabel] = np.asarray(fn(Xobs.copy(), {}), float)
        except Exception as e:                    # noqa: BLE001
            print(f"  gap-strat FAIL {mlabel}: {type(e).__name__}: {e}", flush=True)
            preds[mlabel] = None

    true_masked = X[M]
    res = {}
    for blabel, sel in bands:
        n = int(sel.sum())
        res[(blabel, "n")] = {"_": n}
        for mlabel, _, _ in METHODS:
            p = preds[mlabel]
            if p is None or n == 0:
                res.setdefault((blabel, "mae"), {})[mlabel] = float("nan")
                res.setdefault((blabel, "rmse"), {})[mlabel] = float("nan")
                continue
            pm = p[M][sel]
            tm = true_masked[sel]
            fin = np.isfinite(pm)
            if not fin.all():
                pm = np.where(fin, pm, np.nanmean(X))
            err = pm - tm
            res.setdefault((blabel, "mae"), {})[mlabel] = float(np.mean(np.abs(err)))
            res.setdefault((blabel, "rmse"), {})[mlabel] = float(
                np.sqrt(np.mean(err ** 2)))
    return {"res": res, "edges": edges, "bands": [b[0] for b in bands]}


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "--" if (v is None or not np.isfinite(v)) else f"{v:.3f}"


def _cell(mlabel, causal, mae, best_causal):
    s = _fmt(mae)
    if s == "--":
        return s
    if not causal:
        return r"\textit{" + s + "}"
    if best_causal is not None and np.isfinite(mae) and abs(mae - best_causal) < 1e-9:
        return r"\textbf{" + s + "}"
    return s


def make_table(grid, datasets, gap):
    """Full table when results exist; otherwise a valid STUB so \\input is safe."""
    dnames = ", ".join(d[0] for d in datasets) if datasets else "(no datasets)"
    have = grid is not None and any(
        np.isfinite(grid[k][m[0]][0]) for k in grid for m in METHODS)
    lines = []
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")

    if not have:
        # STUB: structurally valid float so the LaTeX build never breaks.
        lines.append(
            r"\caption{Mask grid (pattern $\times$ rate). \emph{Stub: run "
            r"\texttt{bench/exp\_maskgrid.py} to populate.} No numbers were "
            r"computed in this build.}")
        lines.append(r"\label{tab:maskgrid}")
        lines.append(r"\begin{tabular}{@{}lc@{}}")
        lines.append(r"\toprule")
        lines.append(r"Configuration & MAE \\")
        lines.append(r"\midrule")
        lines.append(r"(pending run) & -- \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")
        return "\n".join(lines) + "\n"

    nmeth = len(METHODS)
    lines.append(
        r"\caption{\textbf{Mask grid: MAE (and RMSE) across missingness "
        r"pattern $\times$ rate.} Primary metrics are MAE / RMSE (lower better), "
        r"\emph{not} correlation. Each cell is the mean over " + dnames +
        r" and " + str(len(SEEDS)) + r" seeds; real series capped to the first "
        + str(ROWS_CAP) + r" rows for speed. \textbf{Point/MCAR} drops cells "
        r"independently; \textbf{Subsequence} removes per-column contiguous runs "
        r"of fixed-ish length; \textbf{Block} removes per-column blackouts of "
        r"widely varying length. \cafe{} and LOCF are strictly causal "
        r"(point-in-time); LinInterp (reads the next observed value) and "
        r"SoftImpute (batch, two-sided) are non-causal references "
        r"(\emph{italic}), shown for context only. \textbf{Bold} $=$ best "
        r"\emph{causal} method per row. Format: MAE\,/\,RMSE.}")
    lines.append(r"\label{tab:maskgrid}")
    lines.append(r"\begin{tabular}{@{}ll" + "c" * nmeth + r"@{}}")
    lines.append(r"\toprule")
    hdr = "Pattern & Rate"
    for mlabel, _, causal in METHODS:
        disp = r"\cafe{}" if mlabel == "CAFE" else mlabel
        hdr += " & " + (disp if causal else r"\textit{" + disp + "}")
    lines.append(hdr + r" \\")
    lines.append(r"\midrule")
    for pi, pattern in enumerate(PATTERNS):
        for ri, rate in enumerate(RATES):
            key = (pattern, rate)
            causal_maes = [grid[key][c][0] for c in CAUSAL_LABELS
                           if np.isfinite(grid[key][c][0])]
            best_causal = min(causal_maes) if causal_maes else None
            cells = []
            for mlabel, _, causal in METHODS:
                mae, rmse = grid[key][mlabel]
                maecell = _cell(mlabel, causal, mae, best_causal)
                rmsecell = _fmt(rmse)
                cells.append(f"{maecell}\\,/\\,{rmsecell}")
            pcol = PATTERN_LABEL[pattern] if ri == 0 else ""
            lines.append(f"{pcol} & {rate:.1f} & " + " & ".join(cells) + r" \\")
        if pi < len(PATTERNS) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # gap-length stratification sub-panel (block missing, single run)
    if gap is not None:
        res, edges, bands = gap["res"], gap["edges"], gap["bands"]
        q1, q2, gmin, gmax = edges
        lines.append(r"\\[6pt]")
        lines.append(r"{\small")
        # NOTE: plain bold line (not \caption*) -- the caption package is not
        # loaded in cafe.tex, so \caption* would break the build.
        lines.append(
            r"\textbf{Gap-length stratification} (Block missing, "
            + f"rate {0.3:.1f}, single run on " + datasets[0][0] +
            r"): MAE by contiguous-gap-length band. Short $=$ gap $\le "
            + f"{q1:.0f}" + r"$, Medium $\le " + f"{q2:.0f}" +
            r"$, Long $>" + f"{q2:.0f}" + r"$ (gaps span "
            + f"{gmin}--{gmax}" + r" steps). Bold $=$ best causal.\par}")
        lines.append(r"\begin{tabular}{@{}lr" + "c" * nmeth + r"@{}}")
        lines.append(r"\toprule")
        ghdr = "Gap band & $n$"
        for mlabel, _, causal in METHODS:
            disp = r"\cafe{}" if mlabel == "CAFE" else mlabel
            ghdr += " & " + (disp if causal else r"\textit{" + disp + "}")
        lines.append(ghdr + r" \\")
        lines.append(r"\midrule")
        for blabel in bands:
            n = res[(blabel, "n")]["_"]
            maes = res[(blabel, "mae")]
            causal_vals = [maes[c] for c in CAUSAL_LABELS
                           if np.isfinite(maes.get(c, float("nan")))]
            best_causal = min(causal_vals) if causal_vals else None
            cells = []
            for mlabel, _, causal in METHODS:
                cells.append(_cell(mlabel, causal, maes.get(mlabel), best_causal))
            lines.append(f"{blabel.capitalize()} & {n} & "
                         + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def write_stub():
    """Write a minimal valid table immediately so \\input never breaks."""
    os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)
    with open(TAB_OUT, "w") as f:
        f.write(make_table(None, [], None))


def main():
    t0 = time.perf_counter()
    # Write a valid stub FIRST -- if anything below crashes, \\input still works.
    write_stub()

    datasets = _datasets()
    if not datasets:
        print("no datasets available; left stub table.", flush=True)
        return

    grid = run_grid(datasets)

    # gap-length stratification on the first (largest real) dataset, block @ 0.3
    print(f"\n[gap-stratify] block@0.3 on {datasets[0][0]}", flush=True)
    gap = run_gap_stratified(datasets[0][1], rate=0.3, seed=0)

    tex = make_table(grid, datasets, gap)
    with open(TAB_OUT, "w") as f:
        f.write(tex)

    # console summary (MAE)
    print("\n=== MAE (avg over datasets, seeds) ===", flush=True)
    hdr = f"{'pattern':12s} {'rate':>4s}" + "".join(f"{m[0]:>12s}" for m in METHODS)
    print(hdr)
    for pattern in PATTERNS:
        for rate in RATES:
            row = f"{PATTERN_LABEL[pattern]:12s} {rate:>4.1f}"
            for mlabel, _, _ in METHODS:
                row += f"{grid[(pattern, rate)][mlabel][0]:12.3f}"
            print(row)

    if gap is not None:
        print("\n=== gap-length stratification (block@0.3) MAE ===", flush=True)
        ghdr = f"{'band':8s} {'n':>7s}" + "".join(f"{m[0]:>12s}" for m in METHODS)
        print(ghdr)
        for blabel in gap["bands"]:
            n = gap["res"][(blabel, "n")]["_"]
            row = f"{blabel:8s} {n:>7d}"
            for mlabel, _, _ in METHODS:
                row += f"{gap['res'][(blabel, 'mae')][mlabel]:12.3f}"
            print(row)

    print(f"\nsaved table {TAB_OUT}", flush=True)
    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
