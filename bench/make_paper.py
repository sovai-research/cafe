"""
make_paper.py -- ONE command to repopulate every data-derived paper artifact from
the current CAFE model, in parallel. Regenerates:
  * experiment tables  -> paper/tables/*.tex   (ablation, longgap, mnar, runtime, leakage,
                          calibration, causal_verify)
  * capability figures -> paper/figures/*.pdf  (decomposition, uncertainty, factors,
                          dependency_net, anomaly, adaptation, forecasting, benchmark,
                          causal_moat, scaling, ablation, longgap, mnar, leakage)
  * prints the Beijing SAITS-protocol MAE for tab:sota (hand-maintained in cafe.tex)

Every number is read straight from a live run of bench/c_unified_penmf.py, so the paper
can never drift from the model. After this finishes:  cd paper && tectonic cafe.tex

Usage:
    python3 bench/make_paper.py            # regenerate everything (parallel)
    python3 bench/make_paper.py --serial   # one at a time (easier to debug)
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

# generator scripts -> the artifacts they (re)write. Order is irrelevant; outputs are
# disjoint so they run concurrently. exp_ablation depends on c_unified_abl.py being in
# sync with the production core (it carries the same season/dial code + ablation toggles).
GENERATORS = [
    "exp_ablation.py", "exp_longgap.py", "exp_mnar.py", "exp_runtime.py", "exp_leakage.py",
    "exp_calibration.py", "exp_causalverify.py",
    "fig_decomposition.py", "fig_uncertainty.py", "fig_factors.py", "fig_dependency_net.py",
    "fig_anomaly.py", "fig_adaptation.py", "fig_forecasting.py", "fig_benchmark.py",
    "fig_causal_moat.py",
]


def _run(script):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    ok = p.returncode == 0
    tail = (p.stdout.strip().splitlines() or [""])[-1] if ok else \
        (p.stderr.strip().splitlines() or [""])[-1]
    return script, ok, dt, tail[:90]


def _beijing_sota_mae():
    """Reproduce the SAITS-protocol headline number for tab:sota (not auto-written)."""
    import numpy as np

    from c_unified_penmf import online_impute
    X = np.load(os.path.join(HERE, "..", "data", "beijing_clean.npy"))
    maes = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        M = rng.random(X.shape) < 0.10
        Xo = X.copy(); Xo[M] = np.nan
        P = np.asarray(online_impute(Xo.copy(), {}), float)
        e = np.abs(P[M] - X[M]); maes.append(float(np.mean(e[np.isfinite(e)])))
    return float(np.mean(maes))


def main(serial=False):
    scripts = [g for g in GENERATORS if os.path.exists(os.path.join(HERE, g))]
    missing = [g for g in GENERATORS if not os.path.exists(os.path.join(HERE, g))]
    print(f"regenerating {len(scripts)} artifacts"
          f"{' (serial)' if serial else ' (parallel)'} ...")
    t0 = time.perf_counter()
    if serial:
        results = [_run(s) for s in scripts]
    else:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            results = list(ex.map(_run, scripts))

    nfail = 0
    for script, ok, dt, tail in sorted(results):
        flag = "OK " if ok else "FAIL"
        if not ok:
            nfail += 1
        print(f"  {flag} {script:24s} {dt:5.1f}s  {tail}")
    for g in missing:
        print(f"  SKIP {g:24s} (not found)")

    print(f"\nBeijing SAITS-protocol MAE (tab:sota): {_beijing_sota_mae():.4f}")
    print(f"total {time.perf_counter()-t0:.1f}s; {nfail} failures")
    print("next:  cd paper && tectonic cafe.tex")
    return nfail


if __name__ == "__main__":
    sys.exit(1 if main(serial="--serial" in sys.argv) else 0)
