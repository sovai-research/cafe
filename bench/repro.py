"""
repro.py -- ONE-command, reproducibility entrypoint for the CAFE paper.

The point of this file is transparency: a reader who suspects a "beats everything"
result can run a single command, see EXACTLY which script produces each table and
figure in the paper, and (with --run) regenerate every data-derived artifact from a
live run of the model. Nothing in the evidence suite is hand-typed except the three
clearly-labelled inline tables noted below.

WHAT THIS DOES
  * Discovers every bench/exp_*.py and bench/fig_*.py generator on disk.
  * Prints a MANIFEST mapping each generator -> the paper table(s)/figure(s) it
    writes (paper/tables/*.tex, paper/figures/*.pdf) and the cafe.tex \\label / float
    it feeds. Flags any generator with no known mapping (so new scripts surface).
  * Flags artifacts that are HAND-MAINTAINED in cafe.tex (no generating script) so a
    reviewer knows precisely which numbers are auto-derived vs hand-transcribed.
  * Reports environment provenance (python, numpy, blas threads, git commit).

MODES
    python3 bench/repro.py             # default = DRY/discovery: print manifest only,
                                       #   run nothing. Safe, fast, side-effect free.
    python3 bench/repro.py --run       # execute every discovered generator (parallel)
                                       #   then re-print the manifest with OK/FAIL +
                                       #   whether each declared output now exists.
    python3 bench/repro.py --run --serial   # same, one at a time (easier to debug)
    python3 bench/repro.py --json      # emit the manifest as JSON (machine-checkable)

This is a thin, additive sibling to make_paper.py: make_paper.py *regenerates*; repro.py
*documents the wiring and verifies it*. Both call the same generator scripts, so they
can never disagree about what produces what.

CAUSAL NOTE: every generator imports the production causal core (bench/c_unified_penmf.py
or src/cafe); none of them train or call any competitor deep model. Published deep-SOTA
numbers are NOT regenerated here -- they live in a sourced registry (see HP_DISCLOSURE.md
and src/cafe/benchmark.py) and are shown in a SEPARATE, clearly-labelled reference column,
never on a single ranked leaderboard with CAFE.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, "paper", "tables")
FIGURES = os.path.join(ROOT, "paper", "figures")

# Keep BLAS deterministic-ish and polite, matching make_paper.py.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

# --------------------------------------------------------------------------- #
# THE WIRING MAP.  generator script -> declared outputs + the cafe.tex consumer.
# This is the single source of truth for "which experiment produces which paper
# artifact".  Outputs are paths RELATIVE to the repo root.  `tex_label` is the
# \label{...} / \includegraphics target inside paper/cafe.tex that consumes it.
# Verified against bench/make_paper.py and a grep of cafe.tex (June 2026).
# --------------------------------------------------------------------------- #
MANIFEST = {
    # ---- experiment tables + their companion figures ----
    "exp_ablation.py": {
        "tables":  ["paper/tables/ablation.tex"],
        "figures": ["paper/figures/ablation.pdf"],
        "tex_label": ["tab:ablation", "fig:ablation"],
        "what": "Component ablation (MAE delta vs full model per toggle).",
    },
    "exp_longgap.py": {
        "tables":  ["paper/tables/longgap.tex"],
        "figures": ["paper/figures/longgap.pdf"],
        "tex_label": ["tab:longgap", "fig:longgap"],
        "what": "Accuracy vs gap length (long contiguous missing runs).",
    },
    "exp_mnar.py": {
        "tables":  ["paper/tables/mnar.tex"],
        # mnar.pdf IS generated but currently has NO \includegraphics in cafe.tex.
        "figures": ["paper/figures/mnar.pdf"],
        "tex_label": ["tab:mnar", "(mnar.pdf generated but UNUSED in cafe.tex)"],
        "what": "Robustness across missingness mechanisms (MCAR/MAR/MNAR).",
    },
    "exp_runtime.py": {
        "tables":  ["paper/tables/runtime.tex"],
        # NB: exp_runtime.py writes figures/scaling.pdf (not runtime.pdf).
        "figures": ["paper/figures/scaling.pdf"],
        "tex_label": ["tab:runtime", "fig:scaling"],
        "what": "Head-to-head wall-clock + MAE on one fixed real task; scaling curves.",
    },
    "exp_leakage.py": {
        "tables":  ["paper/tables/leakage.tex"],
        "figures": ["paper/figures/leakage.pdf"],
        "tex_label": ["tab:leakage", "fig:leakage"],
        "what": "Look-ahead leakage test (truncation invariance of causal imputes).",
    },
    "exp_calibration.py": {
        "tables":  ["paper/tables/calibration.tex"],
        "figures": ["paper/figures/calibration.pdf"],
        "tex_label": ["tab:calibration", "fig:calib"],
        "what": "Uncertainty calibration: coverage / sharpness / reliability + PIT.",
    },
    "exp_causalverify.py": {
        "tables":  ["paper/tables/causal_verify.tex"],
        "figures": [],
        "tex_label": ["tab:causalverify"],
        "what": "Direct verification that fitted state at t uses only data <= t.",
    },
    "exp_mnar_scope.py": {
        "tables":  ["paper/tables/mnar_scope.tex"],
        "figures": [],
        "tex_label": ["tab:mnarscope"],
        "what": "Scoped MNAR sensitivity (where/when the season dial helps vs harms).",
    },
    "exp_seeds.py": {
        "tables":  ["paper/tables/seeds.tex"],
        "figures": [],
        "tex_label": ["tab:seeds"],
        "what": "Multi-seed variance of CAFE accuracy (stability across mask seeds).",
    },
    "exp_maskgrid.py": {
        "tables":  ["paper/tables/maskgrid.tex"],
        "figures": [],
        "tex_label": ["tab:maskgrid"],
        "what": "Accuracy across a grid of missing-rates x mask patterns.",
    },
    "exp_calibration_crps.py": {
        "tables":  ["paper/tables/calib_crps.tex"],
        "figures": [],
        "tex_label": ["tab:calibcrps"],
        "what": "CRPS + coverage + sharpness uncertainty scoring (NLL dropped).",
    },
    # ---- gap-closing experiments added by the related-work review (2026) ----
    "exp_causal_rivals.py": {
        "tables":  ["paper/tables/causal_rivals.tex"],
        "figures": [],
        "tex_label": ["tab:causalrivals"],
        "what": "CAFE vs the closest causal/online statistical rivals (NoTMF, "
                "SHASTA-PCA, rGROUSE, OSW-Net/MissNet stand-in) raced live, "
                "point-in-time, on the 8 structured panels.",
    },
    "exp_conformal.py": {
        "tables":  ["paper/tables/conformal.tex"],
        "figures": [],
        "tex_label": ["tab:conformal"],
        "what": "Causal split-conformal recalibration: raw vs conformal PICP/sharpness "
                "at 50/80/90/95, MAE unchanged, point-in-time preserved.",
    },
    "exp_sensitivity.py": {
        "tables":  ["paper/tables/sensitivity.tex"],
        "figures": ["paper/figures/sensitivity.pdf"],
        "tex_label": ["tab:sensitivity", "fig:sensitivity"],
        "what": "Sensitivity of causal MAE to each fixed internal constant (flat in a "
                "wide band) + harmonic-menu aliasing penalty.",
    },
    "exp_scale.py": {
        "tables":  ["paper/tables/scale.tex"],
        "figures": ["paper/figures/scaling_wide.pdf"],
        "tex_label": ["tab:scale", "fig:scaling_wide"],
        "what": "Scale: CAFE on wide panels (up to 862 ch) + measured N-slope at "
                "width; PhysioNet-2012 ICU clinical row.",
    },
    "exp_byproduct_validation.py": {
        "tables":  ["paper/tables/byproduct.tex"],
        "figures": [],
        "tex_label": ["tab:byproduct"],
        "what": "Quantitative validation of the one-pass by-products (anomaly/"
                "dependency-net/factors/forecast) vs ground truth + a baseline each.",
    },
    "exp_seeds_full.py": {
        "tables":  ["paper/tables/seeds_full.tex"],
        "figures": [],
        "tex_label": ["tab:seedsfull"],
        "what": "Paired multi-seed CAFE vs online TRMF & BayOTIDE on all 8 panels "
                "+ mean-rank/Friedman.",
    },
    "exp_baseline_fairness.py": {
        "tables":  ["paper/tables/baseline_fairness.tex"],
        "figures": [],
        "tex_label": ["tab:fairness"],
        "what": "Deep baselines adequately trained (bidir adequacy band); the causal "
                "collapse is a protocol effect, not under-training.",
    },
    "exp_extreme_missing.py": {
        "tables":  ["paper/tables/extreme_missing.tex"],
        "figures": ["paper/figures/extreme_missing.pdf"],
        "tex_label": ["tab:extrememissing"],
        "what": "Extreme missingness: graceful degradation to 99% MCAR + near-empty-"
                "feature recoverability frontier (factor-spanned vs idiosyncratic).",
    },
    "exp_router.py": {
        "tables":  ["paper/tables/router.tex"],
        "figures": [],
        "tex_label": ["tab:router"],
        "what": "Learned (a, nu, effective-rank) signature as an automatic structure "
                "detector: FX control vs structured panels.",
    },
    "exp_backtest_lookahead.py": {
        "tables":  ["paper/tables/backtest.tex"],
        "figures": [],
        "tex_label": ["tab:backtest"],
        "what": "Rolling-origin backtest confirming no look-ahead across cut points.",
    },
    "exp_downstream.py": {
        "tables":  ["paper/tables/downstream.tex"],
        "figures": [],
        "tex_label": ["tab:downstream"],
        "what": "Downstream task utility of imputed data (vs naive fills).",
    },
    # ---- capability / illustrative figures (no table) ----
    "fig_decomposition.py": {
        "tables": [], "figures": ["paper/figures/decomposition.pdf"],
        "tex_label": ["fig:decomp"], "what": "Trend/factor/residual decomposition.",
    },
    "fig_uncertainty.py": {
        "tables": [], "figures": ["paper/figures/uncertainty.pdf"],
        "tex_label": ["fig:unc"], "what": "Predictive intervals around imputed cells.",
    },
    "fig_factors.py": {
        "tables": [], "figures": ["paper/figures/factors.pdf"],
        "tex_label": ["fig:fac"], "what": "Recovered low-rank factors.",
    },
    "fig_dependency_net.py": {
        "tables": [], "figures": ["paper/figures/dependency_net.pdf"],
        "tex_label": ["fig:net"], "what": "Cross-series dependency network.",
    },
    "fig_anomaly.py": {
        "tables": [], "figures": ["paper/figures/anomaly.pdf"],
        "tex_label": ["fig:anom"], "what": "Anomaly scores from residual stream.",
    },
    "fig_adaptation.py": {
        "tables": [], "figures": ["paper/figures/adaptation.pdf"],
        "tex_label": ["fig:adapt"], "what": "Online adaptation to regime change.",
    },
    "fig_forecasting.py": {
        "tables": [], "figures": ["paper/figures/forecasting.pdf"],
        "tex_label": ["fig:fc"], "what": "Forecasting as imputing future rows.",
    },
    "fig_benchmark.py": {
        "tables": [], "figures": ["paper/figures/benchmark.pdf"],
        "tex_label": ["fig:bench"],
        "what": "Beijing benchmark bar chart (CAFE vs published refs, separate cols).",
    },
    "fig_causal_moat.py": {
        "tables": [], "figures": ["paper/figures/causal_moat.pdf"],
        "tex_label": ["fig:moat"],
        "what": "The causal + CPU + competitive 'two-of-three' positioning figure.",
    },
}

# --------------------------------------------------------------------------- #
# HAND-MAINTAINED artifacts: numbers that live directly in cafe.tex with NO
# generating script.  Listed here so a reviewer knows EXACTLY what is auto-derived
# vs hand-transcribed.  (Audited June 2026.)
# --------------------------------------------------------------------------- #
HAND_MAINTAINED = [
    {
        "label": "tab:sota (inline, cafe.tex ~L327-352)",
        "note": "Beijing headline MAE cell is PRINTED by make_paper.py::_beijing_sota_mae "
                "and HAND-PASTED. Protocol: data/beijing_clean.npy full 17117x132 slice, "
                "per-column z-score once, 10%% MCAR point, np.random.default_rng(seed) for "
                "seed in {0,1,2}, mean MAE over 3 seeds, full causal/online impute. Deep "
                "baselines in that table use a DIFFERENT windowed/train-test protocol from "
                "a single cited source -> must be a SEPARATE labelled column, not a single "
                "ranked board.",
    },
    {
        "label": "tab:results (inline, cafe.tex ~L362+)",
        "note": "Six-dataset breadth Pearson table. NO committed generating script -- "
                "hand-typed. Reproduce-able conceptually via bench/arena.py per-group corr "
                "but NOT currently tied to a run. Flagged for code-backing.",
    },
    {
        "label": "tab:special (inline, cafe.tex ~L250-253)",
        "note": "Hand-maintained capability summary table; illustrative, not a metric run.",
    },
]


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "(unknown)"
    except Exception:
        return "(unknown)"


def _env_provenance():
    try:
        import numpy as np
        npv = np.__version__
    except Exception:
        npv = "(not importable)"
    lines = [
        f"python      : {sys.version.split()[0]}",
        f"numpy       : {npv}",
        f"git commit  : {_git_commit()}",
        "blas threads: " + ", ".join(
            f"{v}={os.environ.get(v, '-')}"
            for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")),
    ]
    return lines


def discover():
    """Return sorted list of (script, mapped?) for every exp_*.py / fig_*.py on disk."""
    found = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(HERE, "exp_*.py")) + glob.glob(os.path.join(HERE, "fig_*.py"))
    )
    return found


def _exists(rel):
    return os.path.exists(os.path.join(ROOT, rel))


def print_manifest(check_outputs=False):
    found = discover()
    mapped = [s for s in found if s in MANIFEST]
    unmapped = [s for s in found if s not in MANIFEST]
    declared_but_missing = [s for s in MANIFEST if s not in found]

    print("=" * 78)
    print("CAFE REPRODUCTION MANIFEST  (experiment -> paper artifact)")
    print("=" * 78)
    for line in _env_provenance():
        print("  " + line)
    print()
    print(f"discovered {len(found)} generator scripts in bench/ "
          f"({len(mapped)} mapped, {len(unmapped)} unmapped)")
    print()

    hdr = f"  {'GENERATOR':24s} -> ARTIFACT(s)  [cafe.tex consumer]"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in found:
        info = MANIFEST.get(s)
        if info is None:
            print(f"  {s:24s} -> (UNMAPPED -- add to repro.py MANIFEST)")
            continue
        outs = list(info["tables"]) + list(info["figures"])
        labels = ", ".join(info["tex_label"])
        if not outs:
            print(f"  {s:24s} -> (no file output)  [{labels}]")
        for i, o in enumerate(outs):
            tag = ""
            if check_outputs:
                tag = "  [exists]" if _exists(o) else "  [MISSING]"
            prefix = f"  {s:24s} ->" if i == 0 else f"  {'':24s}   "
            cons = f"  [{labels}]" if i == 0 else ""
            print(f"{prefix} {o}{tag}{cons}")
        print(f"  {'':24s}      ~ {info['what']}")
    print()

    if unmapped:
        print("  !! UNMAPPED generators (present on disk, not in MANIFEST):")
        for s in unmapped:
            print(f"       {s}")
        print()
    if declared_but_missing:
        print("  !! MANIFEST entries with NO script on disk:")
        for s in declared_but_missing:
            print(f"       {s}")
        print()

    print("  HAND-MAINTAINED in cafe.tex (NO generating script -- not auto-derived):")
    for h in HAND_MAINTAINED:
        print(f"    - {h['label']}")
    print()
    print("  Notes on the headline number, protocol mismatch, and zero-config claim:")
    print("    see paper/HP_DISCLOSURE.md  (regenerated alongside this manifest).")
    print("=" * 78)
    return found, unmapped, declared_but_missing


def manifest_as_dict(check_outputs=False):
    found = discover()
    entries = {}
    for s in found:
        info = MANIFEST.get(s)
        if info is None:
            entries[s] = {"mapped": False}
            continue
        e = {
            "mapped": True,
            "tables": info["tables"],
            "figures": info["figures"],
            "tex_label": info["tex_label"],
            "what": info["what"],
        }
        if check_outputs:
            e["outputs_exist"] = {
                o: _exists(o) for o in info["tables"] + info["figures"]
            }
        entries[s] = e
    return {
        "git_commit": _git_commit(),
        "discovered": found,
        "unmapped": [s for s in found if s not in MANIFEST],
        "manifest_only": [s for s in MANIFEST if s not in found],
        "hand_maintained": HAND_MAINTAINED,
        "generators": entries,
    }


def _run_one(script):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    ok = p.returncode == 0
    tail = (p.stdout.strip().splitlines() or [""])[-1] if ok else \
        (p.stderr.strip().splitlines() or [""])[-1]
    return script, ok, dt, tail[:90]


def run_all(serial=False):
    scripts = discover()
    print(f"running {len(scripts)} generators{' (serial)' if serial else ' (parallel)'} ...\n")
    t0 = time.perf_counter()
    if serial:
        results = [_run_one(s) for s in scripts]
    else:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            results = list(ex.map(_run_one, scripts))
    nfail = 0
    for script, ok, dt, tail in sorted(results):
        flag = "OK  " if ok else "FAIL"
        if not ok:
            nfail += 1
        print(f"  {flag} {script:24s} {dt:6.1f}s  {tail}")
    print(f"\ntotal {time.perf_counter() - t0:.1f}s; {nfail} failures\n")
    print("verifying declared outputs now exist on disk:\n")
    print_manifest(check_outputs=True)
    print("\nnext:  cd paper && tectonic cafe.tex")
    return nfail


def main(argv):
    if "--json" in argv:
        print(json.dumps(manifest_as_dict(check_outputs="--run" in argv), indent=2))
        return 0
    if "--run" in argv:
        return 1 if run_all(serial="--serial" in argv) else 0
    # default: dry / discovery
    print_manifest(check_outputs=False)
    print("\n(dry run -- nothing executed. use --run to regenerate all artifacts.)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
