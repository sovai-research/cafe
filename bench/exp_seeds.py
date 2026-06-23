r"""
SEED-RIGOR experiment for the CAFE paper -- replace single-seed point estimates
with a proper statistical comparison.

Question this answers
---------------------
A single mask seed gives ONE MAE number. Is CAFE's advantage (or deficit) over
the strongest CAUSAL baseline real, or seed noise? Here we run both methods over
>= 10 independent MCAR mask seeds on each of three datasets and report:

  * mean MAE +/- a 95% confidence interval (Student-t on the per-seed means, and
    a nonparametric bootstrap CI as a cross-check), and
  * a PAIRED significance test of CAFE vs the baseline -- the two methods see the
    SAME mask on each seed, so we pair per-seed MAEs and run BOTH a paired t-test
    and the Wilcoxon signed-rank test (scipy.stats). We report the p-values and
    the mean paired difference (baseline - CAFE; positive => CAFE better).

Honesty / scope
---------------
- Both methods compared here are CAUSAL / point-in-time (CAFE and online TRMF):
  every imputed value at time t uses only data <= t. No deep models are trained
  or run; no published numbers are touched. This is a like-for-like LIVE board.
- The strongest causal baseline is online TRMF (low-rank MF + AR temporal
  regularization, fit online). It is a genuine rival to CAFE's factor core, and
  on some datasets it WINS -- we report that result as-is, never massaged.
- Caps are small (a few-thousand-row slice, <= 60 cols) so the whole thing runs
  in a couple of minutes on one CPU core. Nothing is fabricated; every cell is a
  live measurement. If a dataset file is missing it is SKIPPED (not faked).

Outputs
-------
  paper/tables/seeds.tex   -- self-contained \begin{table} float, \label{tab:seeds}
  stdout                   -- the same numbers, plus per-seed detail.

Run:  python bench/exp_seeds.py        ( < ~3 min on one CPU core, BLAS=2 )
      TIMARA_FAST=1 python bench/exp_seeds.py   (even smaller caps, smoke test)
"""
import os
# Pin BLAS threads BEFORE numpy import so timing/threads are fair & deterministic.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
_DATA = os.path.join(_ROOT, "data")
_TABDIR = os.path.join(_ROOT, "paper", "tables")
os.makedirs(_TABDIR, exist_ok=True)

import harness as H                                  # MASKERS, deterministic seeds
from c_unified_penmf import online_impute as cafe_impute      # CAFE (causal, ours)
from c_online_trmf import online_impute as trmf_impute        # strongest causal baseline

# --------------------------------------------------------------------------- #
# Config  (small caps for speed; FAST mode shrinks further)
# --------------------------------------------------------------------------- #
FAST = os.environ.get("TIMARA_FAST", "") not in ("", "0", "false", "False")
N_SEEDS = 6 if FAST else 12                 # >= 10 in the real run
RATE = 0.10                                  # 10% MCAR point missingness
MECH = "mcar"
ROW_CAP = 800 if FAST else 2500
COL_CAP = 40 if FAST else 60

BASELINE_NAME = "Online TRMF"               # strongest CAUSAL baseline


def _load_real(fname, label):
    """Load a real clean (already z-scored) matrix, capped for speed. None if absent."""
    path = os.path.join(_DATA, fname)
    if not os.path.exists(path):
        return None
    X = np.load(path)
    X = np.ascontiguousarray(X[:ROW_CAP, :COL_CAP], dtype=float)
    return (label, X)


def _build_datasets():
    """Two real datasets where the two methods genuinely differ + one synthetic
    low-rank+AR control. Real files are skipped (not faked) if missing."""
    ds = []
    b = _load_real("beijing_clean.npy", "Beijing")
    if b is not None:
        ds.append(b)
    e = _load_real("ETTh1_clean.npy", "ETTh1")
    if e is not None:
        ds.append(e)
    # synthetic control: identical generator the harness uses, capped.
    syn = H.gen_2d(T=min(ROW_CAP, 1500), N=min(COL_CAP, 30), seed=0)
    ds.append(("Synthetic", np.ascontiguousarray(syn, dtype=float)))
    return ds


# --------------------------------------------------------------------------- #
# Per-seed MAE for one method on one dataset
# --------------------------------------------------------------------------- #
def _mae(true, pred, mask):
    p = np.asarray(pred, float)[mask]
    p = np.where(np.isfinite(p), p, np.nanmean(true))   # never reward leftover NaNs
    return float(np.mean(np.abs(p - true[mask])))


def _per_seed_maes(X, fn, seeds):
    """MAE of `fn` on X under the SAME mask for each seed. Returns list aligned to seeds."""
    out = []
    for s in seeds:
        M = H.MASKERS[MECH](X, RATE, seed=s)           # identical mask both methods see
        Xobs = X.copy()
        Xobs[M] = np.nan
        pred = fn(Xobs.copy(), {})
        out.append(_mae(X, pred, M))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _t_ci(x, conf=0.95):
    """Student-t (1-conf)/2 two-sided CI half-width for the MEAN of x."""
    x = np.asarray(x, float)
    n = x.size
    m = float(np.mean(x))
    if n < 2:
        return m, 0.0
    sem = float(np.std(x, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.5 + conf / 2.0, df=n - 1))
    return m, tcrit * sem


def _bootstrap_ci(x, conf=0.95, n_boot=5000, seed=0):
    """Nonparametric percentile bootstrap CI for the mean (cross-check on _t_ci)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(12345 + seed)
    n = x.size
    if n < 2:
        return float(np.mean(x)), float(np.mean(x))
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return float(lo), float(hi)


def _paired_tests(cafe_maes, base_maes):
    """Paired CAFE-vs-baseline tests. diff = baseline - CAFE (positive => CAFE better)."""
    diff = base_maes - cafe_maes
    # paired t-test
    t_stat, t_p = stats.ttest_rel(base_maes, cafe_maes)
    # Wilcoxon signed-rank (nonparametric); guard the all-zero / tiny-n degenerate case
    try:
        w_stat, w_p = stats.wilcoxon(base_maes, cafe_maes)
        w_p = float(w_p)
    except ValueError:
        w_p = float("nan")
    return dict(mean_diff=float(np.mean(diff)),
                t_p=float(t_p), wilcoxon_p=w_p,
                n=int(diff.size))


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run():
    seeds = list(range(N_SEEDS))
    datasets = _build_datasets()
    results = []
    print(f"SEED RIGOR  |  {N_SEEDS} mask seeds, {MECH} {int(RATE*100)}%, "
          f"caps {ROW_CAP}x{COL_CAP}  |  CAFE vs {BASELINE_NAME} (both causal)\n")
    for label, X in datasets:
        cafe_maes = _per_seed_maes(X, cafe_impute, seeds)
        base_maes = _per_seed_maes(X, trmf_impute, seeds)

        cm, ch = _t_ci(cafe_maes)
        bm, bh = _t_ci(base_maes)
        cblo, cbhi = _bootstrap_ci(cafe_maes, seed=1)
        bblo, bbhi = _bootstrap_ci(base_maes, seed=2)
        test = _paired_tests(cafe_maes, base_maes)

        winner = "CAFE" if cm < bm else BASELINE_NAME
        results.append(dict(
            dataset=label, shape=f"{X.shape[0]}x{X.shape[1]}",
            cafe_mean=cm, cafe_ci=ch, cafe_boot=(cblo, cbhi),
            base_mean=bm, base_ci=bh, base_boot=(bblo, bbhi),
            winner=winner, **test,
            cafe_maes=cafe_maes, base_maes=base_maes))

        print(f"[{label}  {X.shape[0]}x{X.shape[1]}]")
        print(f"  CAFE         MAE {cm:.4f} +/- {ch:.4f}  (t-CI)  "
              f"boot[{cblo:.4f},{cbhi:.4f}]")
        print(f"  {BASELINE_NAME:11s}  MAE {bm:.4f} +/- {bh:.4f}  (t-CI)  "
              f"boot[{bblo:.4f},{bbhi:.4f}]")
        print(f"  paired diff (base-CAFE) = {test['mean_diff']:+.4f}   "
              f"t-test p={test['t_p']:.2e}   wilcoxon p={test['wilcoxon_p']:.2e}   "
              f"-> winner: {winner}")
        print(f"  per-seed CAFE: {np.array2string(cafe_maes, precision=4)}")
        print(f"  per-seed {BASELINE_NAME[:4]}: {np.array2string(base_maes, precision=4)}\n")
    return results


# --------------------------------------------------------------------------- #
# LaTeX table  (self-contained float; bold = significantly better at p<0.05)
# --------------------------------------------------------------------------- #
def _sig_marker(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def write_table(results, path):
    lines = []
    A = lines.append
    A(r"\begin{table}[t]\centering\small")
    A(r"\setlength{\tabcolsep}{4pt}")
    A(r"\caption{\textbf{Seed rigor: CAFE vs the strongest causal baseline.} "
      r"Mean MAE\,$\downarrow$ $\pm$ 95\% CI (Student-$t$) over " + str(N_SEEDS) +
      r" independent " + MECH.upper() + r"\,(" + f"{int(RATE*100)}" +
      r"\%) mask seeds; both methods see the \emph{same} mask each seed. "
      r"\textsc{paired diff} $=$ mean(baseline\,$-$\,CAFE) per seed (positive "
      r"$\Rightarrow$ CAFE lower error); $p$ is the paired $t$-test, with the "
      r"Wilcoxon signed-rank $p$ in brackets. Both methods are causal / "
      r"point-in-time (no deep models, no published numbers). The better mean is "
      r"\textbf{bold}; significance stars on the paired diff: "
      r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}")
    A(r"\label{tab:seeds}")
    A(r"\begin{tabular}{@{}lccccc@{}}")
    A(r"\toprule")
    A(r"Dataset & Shape & CAFE MAE & " + BASELINE_NAME +
      r" MAE & Paired diff & $p$ ($t$ / W) \\")
    A(r"\midrule")
    for r in results:
        cafe_cell = f"${r['cafe_mean']:.3f}\\pm{r['cafe_ci']:.3f}$"
        base_cell = f"${r['base_mean']:.3f}\\pm{r['base_ci']:.3f}$"
        if r["winner"] == "CAFE":
            cafe_cell = r"\textbf{" + cafe_cell + "}"
        else:
            base_cell = r"\textbf{" + base_cell + "}"
        diff_cell = f"${r['mean_diff']:+.3f}$" + _sig_marker(r["t_p"])
        p_cell = f"${r['t_p']:.1e}$ / ${r['wilcoxon_p']:.1e}$".replace("e-0", "e-").replace("e+0", "e+")
        A(f"{r['dataset']} & {r['shape']} & {cafe_cell} & {base_cell} & "
          f"{diff_cell} & {p_cell} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    res = run()
    write_table(res, os.path.join(_TABDIR, "seeds.tex"))
    n_sig = sum(1 for r in res if np.isfinite(r["t_p"]) and r["t_p"] < 0.05)
    print(f"\n{len(res)} datasets, {N_SEEDS} seeds each; "
          f"{n_sig}/{len(res)} paired diffs significant at p<0.05.")
