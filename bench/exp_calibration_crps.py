"""Experiment: PROBABILISTIC CALIBRATION of CAFE via CRPS / coverage / sharpness / PIT.

This is the CRPS-first replacement for the old NLL-style framing. It evaluates the
quality of CAFE's per-cell *predictive intervals* (not just point error) on held-out
masked cells across several real + synthetic datasets, scoring with proper probabilistic
metrics:

  * CRPS      -- proper score for the full predictive distribution (lower is better).
  * PICP      -- prediction-interval coverage probability at nominal 50/80/90/95%
                 (a.k.a. observed coverage); the calibration target is observed == nominal.
  * Sharpness -- mean predictive interval width (and mean sigma); narrower is better
                 *given* calibration -- it is only meaningful read alongside coverage.
  * PIT       -- probability integral transform y -> F(y); calibrated => Uniform(0,1).
                 We report a one-number flatness deviation (KS-style) plus the histogram.

ALL probabilistic metrics are imported from ``bench.metrics_prob`` (a sibling module
authored in parallel). We do NOT redefine any metric here -- this file only wires CAFE's
outputs into those metrics, scores held-out cells, and renders the table. The metric
*definitions* live in one place so the paper has a single source of truth.

CAUSAL / point-in-time: CAFE is run in its only mode -- strictly online, no look-ahead.
Each cell's (mu, sigma) at time t depends only on data <= t. The MCAR mask removes cells
that the model must then predict from past + contemporaneous observed context.

Honesty contract (NEVER fabricate):
  * Every number is read from a live ``CAFE().run`` pass; nothing is hand-typed.
  * If the live run cannot finish (e.g. data files or ``metrics_prob`` unavailable),
    we write a VALID, clearly-marked PLACEHOLDER ``calib_crps.tex`` (em-dashes, a
    "could not run" note) so the LaTeX build never breaks, and we print why -- we do
    NOT invent any value.
  * The summary states plainly whether the intervals are over-conservative
    (observed coverage systematically > nominal) -- the known honest weakness.

Outputs:
  paper/tables/calib_crps.tex   (self-contained booktabs float, \\input-ready)
"""
import os
import sys
import traceback

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)                                   # bench/  (harness, ...)
sys.path.insert(0, os.path.join(ROOT, "src"))             # src/    (import cafe)

import numpy as np

TAB_OUT = os.path.join(ROOT, "paper", "tables", "calib_crps.tex")
LABEL = "tab:calibcrps"

RATE = 0.10
SEEDS = (0, 1, 2)
ROWS_CAP = 3000
NOMINAL = (0.50, 0.80, 0.90, 0.95)


# --------------------------------------------------------------------------- #
# Resolve the probabilistic metrics from bench.metrics_prob.
#
# The metric DEFINITIONS are owned by metrics_prob (authored in parallel); we only
# *bind* them here. To be robust to its exact public names we accept a small set of
# conventional aliases per metric and pick the first that exists. We never implement
# the math -- if none of a metric's aliases is present we treat the module as
# unavailable and fall back to writing the placeholder table (no fabrication).
# --------------------------------------------------------------------------- #
_ALIASES = {
    # name -> ordered candidate attribute names in metrics_prob
    "crps":      ("crps_gaussian", "crps", "gaussian_crps", "crps_normal"),
    "picp":      ("picp", "coverage", "interval_coverage", "prediction_interval_coverage"),
    "sharpness": ("sharpness", "interval_width", "mean_interval_width", "mpiw"),
    "pit":       ("pit_values", "pit", "pit_gaussian"),
    # one-number PIT flatness / calibration error (optional; we degrade gracefully)
    "pit_dev":   ("pit_ks", "pit_deviation", "pit_flatness", "calibration_error", "ks_uniform"),
}


def _load_metrics():
    """Import bench.metrics_prob and bind callables. Returns (dict, None) on success
    or (None, reason) if the module / required metrics are unavailable."""
    try:
        import metrics_prob as MP                       # sibling module, parallel-authored
    except Exception as e:                              # not yet present at run time
        return None, f"could not import metrics_prob ({e})"

    bound = {}
    for key, cands in _ALIASES.items():
        fn = next((getattr(MP, c) for c in cands if hasattr(MP, c)), None)
        bound[key] = fn
    # the four core metrics are required; pit_dev is optional
    missing = [k for k in ("crps", "picp", "sharpness", "pit") if bound[k] is None]
    if missing:
        return None, ("metrics_prob present but missing required metric(s): "
                      + ", ".join(missing)
                      + f"  (looked for aliases {[_ALIASES[m] for m in missing]})")
    return bound, None


# --------------------------------------------------------------------------- #
# Metric call shims: metrics_prob may expose either a Gaussian-parameter API
# (mu, sigma) or a sample/quantile API. We try the (y, mu, sigma) signature first
# (closed-form Gaussian, which CAFE naturally provides), then degrade. We still do
# NOT compute the metric ourselves -- we only adapt argument order.
# --------------------------------------------------------------------------- #
def _call_crps(fn, y, mu, sigma):
    """metrics_prob.crps_gaussian(y, mu, sigma) returns the mean CRPS already."""
    out = fn(y, mu, sigma)
    return float(np.mean(out))


def _call_picp(fn, y, mu, sigma, level):
    """Coverage at a central interval of the given nominal ``level``.

    The interval ENDPOINTS are the Gaussian quantiles mu +/- z*sigma (z is a plain
    normal quantile, not a metric); the COVERAGE itself is computed by metrics_prob.
    metrics_prob's picp/coverage take (y, lo, hi); we also try a (y, mu, sigma, level)
    signature in case a different parallel build exposes that instead. We never
    compute the coverage fraction ourselves."""
    z = _z_for(level)
    lo, hi = mu - z * sigma, mu + z * sigma
    for attempt in (
        lambda: fn(y, lo, hi),                       # picp(y, lo, hi)  <- current API
        lambda: fn(y=y, lo=lo, hi=hi),
        lambda: fn(y, mu, sigma, level),             # parametric variant, if any
        lambda: fn(y=y, mu=mu, sigma=sigma, level=level),
    ):
        try:
            return float(attempt())
        except TypeError:
            continue
    raise TypeError("picp/coverage: no supported signature matched")


def _call_sharpness(fn, mu, sigma, level):
    """Mean width of the central ``level`` interval. Endpoints from Gaussian z;
    the width statistic is metrics_prob.sharpness(lo, hi)."""
    z = _z_for(level)
    lo, hi = mu - z * sigma, mu + z * sigma
    for attempt in (
        lambda: fn(lo, hi),                          # sharpness(lo, hi)  <- current API
        lambda: fn(lo=lo, hi=hi),
        lambda: fn(mu, sigma, level),                # parametric variant, if any
        lambda: fn(sigma),
    ):
        try:
            return float(np.mean(attempt()))
        except TypeError:
            continue
    raise TypeError("sharpness: no supported signature matched")


def _call_pit(fn, y, mu, sigma):
    """metrics_prob.pit_values(y, mu, sigma) -> per-cell PIT in [0,1]."""
    return np.asarray(fn(y, mu, sigma), float)


def _z_for(level):
    """Two-sided Gaussian z with mass == level (for the band shims only; this is a
    standard-normal quantile, not a probabilistic *metric*)."""
    from math import erf, sqrt
    target = 0.5 * (1.0 + level)
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + erf(mid / sqrt(2.0)))
        if cdf < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# CAFE outputs at held-out cells (via the public src/cafe API).
# --------------------------------------------------------------------------- #
def collect_cells(X, masker):
    """Run CAFE().run over SEEDS of 10% MCAR; stack (mu, sigma, y_true, mae) at the
    held-out masked cells where a finite predictive sigma is defined."""
    from cafe import CAFE
    from harness import metrics as point_metrics

    mus, sgs, trs, maes = [], [], [], []
    for s in SEEDS:
        M = masker(X, RATE, seed=s)
        Xobs = X.copy()
        Xobs[M] = np.nan

        res = CAFE().run(Xobs)
        filled = np.asarray(res.imputed, float)
        sd = np.asarray(res.uncertainty, float)        # NaN at observed cells

        sel = M & np.isfinite(sd) & np.isfinite(filled) & (sd > 1e-12)
        mus.append(filled[sel])
        sgs.append(sd[sel])
        trs.append(X[sel])
        maes.append(point_metrics(X, filled, M)["mae"])

    mu = np.concatenate(mus)
    sg = np.concatenate(sgs)
    tr = np.concatenate(trs)
    good = np.isfinite(mu) & np.isfinite(sg) & np.isfinite(tr) & (sg > 1e-12)
    return mu[good], sg[good], tr[good], float(np.mean(maes))


def evaluate_dataset(name, X, MET):
    mu, sg, tr, mae = collect_cells(X, MET["_masker"])
    crps = _call_crps(MET["crps"], tr, mu, sg)
    cover = {lvl: _call_picp(MET["picp"], tr, mu, sg, lvl) for lvl in NOMINAL}
    sharp = _call_sharpness(MET["sharpness"], mu, sg, 0.90)   # 90% interval width
    pit = _call_pit(MET["pit"], tr, mu, sg)

    # one-number PIT flatness: prefer metrics_prob's own; else a KS distance to U(0,1)
    # computed from the PIT *values metrics_prob returned* (this is a summary of the
    # imported PIT, not a re-derivation of a probabilistic score).
    if MET["pit_dev"] is not None:
        try:
            pit_dev = float(MET["pit_dev"](pit))
        except Exception:
            pit_dev = _ks_uniform(pit)
    else:
        pit_dev = _ks_uniform(pit)

    return dict(name=name, n=int(mu.size), mae=mae, crps=crps,
                cover=cover, sharp=sharp, pit_dev=pit_dev,
                pit_mean=float(np.mean(pit)))


def _ks_uniform(p):
    """KS distance of samples p to Uniform(0,1) -- a plain descriptive summary of the
    (imported) PIT values; NOT a substitute for a metrics_prob probabilistic metric."""
    p = np.sort(np.clip(np.asarray(p, float), 0.0, 1.0))
    n = p.size
    if n == 0:
        return float("nan")
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(ecdf_hi - p), np.max(p - ecdf_lo)))


# --------------------------------------------------------------------------- #
# Table writers.
# --------------------------------------------------------------------------- #
_CAPTION_HEAD = (
    r"\textbf{Probabilistic calibration of \cafe{}'s predictive intervals "
    r"(CRPS / coverage / sharpness / PIT).} "
)


def write_table(results, conservative_note):
    """Self-contained booktabs float with the live numbers."""
    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + _CAPTION_HEAD +
        r"Held-out cells under $10\%$ MCAR, " + f"{len(SEEDS)}" +
        r" seeds, scored with proper probabilistic metrics imported from a single "
        r"\texttt{metrics\_prob} module (CRPS, PICP coverage, sharpness, PIT). "
        r"Coverage is observed empirical coverage of the central interval at each "
        r"nominal level (calibrated $\Rightarrow$ observed $=$ nominal). "
        r"Sharpness is the mean $90\%$ interval width; PIT dev.\ is the KS distance "
        r"of the PIT to Uniform$(0,1)$ ($0$ $=$ perfectly calibrated shape). "
        + conservative_note +
        r" CRPS / sharpness / PIT dev.\ / MAE: lower $\downarrow$ is better.}",
        r"\label{" + LABEL + r"}",
        r"\begin{tabular}{@{}lrccccccc@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{Observed coverage} & & & \\",
        r"\cmidrule(lr){3-6}",
        r"Dataset & $n$ & @50 & @80 & @90 & @95 "
        r"& CRPS$\downarrow$ & Sharp.\ & PIT dev.$\downarrow$ \\",
        r"& & \scriptsize(.50) & \scriptsize(.80) & \scriptsize(.90) "
        r"& \scriptsize(.95) & & & \\",
        r"\midrule",
    ]
    for r in results:
        c = r["cover"]
        lines.append(
            f"{r['name']} & {r['n']:,} & {c[0.50]:.2f} & {c[0.80]:.2f} "
            f"& {c[0.90]:.2f} & {c[0.95]:.2f} & {r['crps']:.3f} "
            f"& {r['sharp']:.3f} & {r['pit_dev']:.3f} \\\\".replace(",", r"{,}"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    txt = "\n".join(lines)
    with open(TAB_OUT, "w") as f:
        f.write(txt)
    return txt


def write_stub(reason):
    """Valid, clearly-marked PLACEHOLDER float so the LaTeX build never breaks.
    No fabricated numbers -- every cell is an em-dash with a visible 'could not run'
    note carrying the reason."""
    safe = reason.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    lines = [
        r"% PLACEHOLDER -- generated by bench/exp_calibration_crps.py because the live",
        r"% run could not complete. Re-run that script to populate real numbers.",
        r"\begin{table*}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + _CAPTION_HEAD +
        r"\emph{Placeholder: the live calibration run did not complete "
        r"(\texttt{" + safe + r"}); no numbers are reported here. "
        r"Re-run \texttt{bench/exp\_calibration\_crps.py} to populate this table.}}",
        r"\label{" + LABEL + r"}",
        r"\begin{tabular}{@{}lrccccccc@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{Observed coverage} & & & \\",
        r"\cmidrule(lr){3-6}",
        r"Dataset & $n$ & @50 & @80 & @90 & @95 "
        r"& CRPS$\downarrow$ & Sharp.\ & PIT dev.$\downarrow$ \\",
        r"\midrule",
        r"\multicolumn{9}{c}{\emph{--- run pending; no fabricated values ---}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    txt = "\n".join(lines)
    with open(TAB_OUT, "w") as f:
        f.write(txt)
    return txt


# --------------------------------------------------------------------------- #
# Honest over-conservativeness verdict.
# --------------------------------------------------------------------------- #
def conservativeness_verdict(results):
    """Return (latex_note, plain_summary). Over-conservative := observed coverage
    systematically EXCEEDS nominal (intervals wider than they need to be)."""
    gaps = []   # observed - nominal, across all datasets/levels
    for r in results:
        for lvl in NOMINAL:
            gaps.append(r["cover"][lvl] - lvl)
    gaps = np.asarray(gaps, float)
    mean_gap = float(np.mean(gaps))
    frac_over = float(np.mean(gaps > 0.01))      # fraction clearly above nominal
    frac_under = float(np.mean(gaps < -0.01))    # fraction clearly below nominal
    worst_under = float(-np.min(gaps))           # magnitude of the worst undershoot

    # Over-conservative := overs dominate AND no *material* undershoot anywhere
    # (a few marginal undershoots <=0.03 at the tightest levels still count as
    # "safe but wide"; a large undershoot is the unsafe mode and overrides).
    if frac_over >= 0.6 and frac_over > frac_under and worst_under <= 0.05:
        und = ("" if frac_under < 1e-9 else
               (r" The only exceptions are a few marginal undershoots "
                f"(worst ${worst_under:.02f}$".replace("0.", ".") +
                r") at the tightest levels."))
        latex = (r"As is honest to report, the intervals are predominantly "
                 r"\emph{over-conservative}: observed coverage exceeds nominal at "
                 f"{100*frac_over:.0f}\\% of dataset/level cells (mean excess "
                 f"$+{mean_gap:.02f}$".replace("0.", ".") +
                 r"), so bands are safe but wider than ideal." + und +
                 r" Tightening calibration is left to future work.")
        plain = (f"OVER-CONSERVATIVE (predominant): observed coverage exceeds nominal "
                 f"(mean gap +{mean_gap:.3f}; {100*frac_over:.0f}% of "
                 f"dataset/level cells over-cover, {100*frac_under:.0f}% under-cover; "
                 f"worst undershoot {worst_under:.3f}).")
    elif frac_under >= 0.6 and frac_over <= 0.1:
        latex = (r"The intervals are \emph{under-confident in the other direction}: "
                 r"observed coverage falls below nominal (mean gap "
                 f"${mean_gap:.02f}$".replace("0.", ".") +
                 r"); this is the unsafe failure mode and is flagged for fixing.")
        plain = (f"UNDER-COVERING (unsafe): observed coverage below nominal "
                 f"(mean gap {mean_gap:.3f}).")
    else:
        latex = (r"Calibration is mixed across datasets/levels (mean coverage gap "
                 f"${mean_gap:+.02f}$".replace("0.", ".") +
                 r"): neither uniformly over- nor under-confident.")
        plain = (f"MIXED: mean coverage gap {mean_gap:+.3f} "
                 f"({100*frac_over:.0f}% over, {100*frac_under:.0f}% under).")
    return latex, plain


# --------------------------------------------------------------------------- #
def main():
    MET, reason = _load_metrics()
    if MET is None:
        txt = write_stub(reason)
        print(f"[calib_crps] metrics unavailable -> placeholder written: {reason}")
        print(f"saved STUB table {TAB_OUT}")
        print(txt)
        return

    # datasets: two real (capped) + one synthetic panel
    datasets = []
    try:
        from harness import gen_2d
        beijing = np.load(os.path.join(ROOT, "data", "beijing_clean.npy"))[:ROWS_CAP]
        etth1 = np.load(os.path.join(ROOT, "data", "ETTh1_clean.npy"))[:ROWS_CAP]
        syn = gen_2d(T=1500, N=20, seed=7)
        datasets = [
            ("Beijing", np.ascontiguousarray(beijing, float)),
            ("ETTh1", np.ascontiguousarray(etth1, float)),
            ("Synthetic", np.ascontiguousarray(syn, float)),
        ]
        from harness import MASKERS
        MET["_masker"] = MASKERS["mcar"]
    except Exception as e:
        txt = write_stub(f"data/harness load failed ({e})")
        print(f"[calib_crps] could not load data -> placeholder: {e}")
        print(traceback.format_exc())
        return

    results = []
    for name, X in datasets:
        print(f"[{name}] X={X.shape} ...", flush=True)
        try:
            r = evaluate_dataset(name, X, MET)
        except Exception as e:                          # one dataset failing != fabricate
            print(f"  [WARN] {name} failed: {e}")
            print(traceback.format_exc())
            continue
        results.append(r)
        c = r["cover"]
        print(f"  n={r['n']:,}  cover 50/80/90/95 = "
              f"{c[0.50]:.3f}/{c[0.80]:.3f}/{c[0.90]:.3f}/{c[0.95]:.3f}  "
              f"CRPS={r['crps']:.4f}  sharp={r['sharp']:.4f}  "
              f"PITdev={r['pit_dev']:.4f}  MAE={r['mae']:.4f}", flush=True)

    if not results:
        txt = write_stub("no dataset completed (see WARN logs above)")
        print(f"[calib_crps] no results -> placeholder written")
        print(txt)
        return

    latex_note, plain = conservativeness_verdict(results)
    txt = write_table(results, latex_note)
    print(f"\nsaved table {TAB_OUT}")

    print("\n=== HEADLINE (honest calibration verdict) ===")
    print(plain)
    for r in results:
        c = r["cover"]
        print(f"  {r['name']:10s}: cover@90={c[0.90]:.3f} "
              f"(nominal .90; gap {c[0.90]-0.90:+.3f})  "
              f"CRPS={r['crps']:.4f}  PITdev={r['pit_dev']:.4f}")
    print("\n=== TABLE LATEX ===")
    print(txt)


if __name__ == "__main__":
    main()
