"""
Fast local leaderboard for the TIMARA imputation prototypes.

Re-run this anytime to get feedback in seconds -- no agents, no workflows.
Imports the impute functions from the prototype files directly and scores them
all on the shared harness.

  python3 run_all.py            # fast: scores all methods, NO causality verifier
  python3 run_all.py --verify   # also run the no-look-ahead verifier (slower)
  python3 run_all.py --only OnlineTRMF,CausalFE   # subset
  python3 run_all.py --datasets panel_mcar20,2d_small_block20

Add a new method by appending one line to REGISTRY below.
"""
import os, sys, time, argparse
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from harness import DATASETS, MASKERS, metrics
from causal import assert_causal


# --------------------------------------------------------------------------- #
# Registry: (display_name, "module:callable_or_factory", is_causal)
# factory form "module:make_x()" is called once to get the impute fn.
# --------------------------------------------------------------------------- #
def _load(spec):
    mod, _, attr = spec.partition(":")
    m = __import__(mod)
    if attr.endswith("()"):
        return getattr(m, attr[:-2])()
    return getattr(m, attr)


REGISTRY = [
    # unified routed causal core
    ("CausalRouter",   "c_router:online_impute",      True),
    # causal candidates
    ("OnlineTRMF",     "c_online_trmf:online_impute", True),
    ("CausalFE",       "c_fe_lowrank:online_impute",  True),
    ("FilterSSM",      "c_filter_ssm:online_impute",  True),
    ("CausalXSecReg",  "c_xsec_reg:online_impute",    True),
    # causal baselines (floor)
    ("LOCF",           "c_baselines:locf_impute",     True),
    ("CausalEWMA",     "c_baselines:make_ewma()",     True),
    ("CausalXSecMean", "c_baselines:xsecmean_impute", True),
    # non-causal oracle ceiling (allowed to look ahead)
    ("TRMF-oracle",    "c_oracle_ref:trmf_oracle",    False),
    ("MCNNM-oracle",   "c_oracle_ref:mcnnm_oracle",   False),
]


def score_one(fn, dataset, verify, seed=0):
    clean, meta, mech, rate = dataset
    M = MASKERS[mech](clean, rate, seed=seed)
    Xobs = clean.copy(); Xobs[M] = np.nan
    causal_ok, detail = None, ""
    if verify:
        try:
            causal_ok, detail = assert_causal(fn, Xobs, meta)
        except Exception as e:                      # noqa: BLE001
            causal_ok, detail = False, f"verify-err:{type(e).__name__}"
    t0 = time.perf_counter()
    try:
        pred = np.asarray(fn(Xobs.copy(), dict(meta)), dtype=float)
        dt = time.perf_counter() - t0
        m = metrics(clean, pred, M); ok, err = True, ""
    except Exception as e:                          # noqa: BLE001
        dt = time.perf_counter() - t0
        m = dict(mae=np.nan, rmse=np.nan, corr=np.nan, mre=np.nan, n=int(M.sum()))
        ok, err = False, f"{type(e).__name__}: {e}"
    return dict(time_s=dt, success=ok, error=err, causal=causal_ok, detail=detail, **m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="run the no-look-ahead verifier")
    ap.add_argument("--only", default="", help="comma-list of method names")
    ap.add_argument("--datasets", default="", help="comma-list of dataset names")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    methods = [r for r in REGISTRY if not only or r[0] in only]
    dnames = [s.strip() for s in args.datasets.split(",") if s.strip()] or list(DATASETS)

    # load fns (skip any that fail to import)
    loaded = []
    for name, spec, is_causal in methods:
        try:
            loaded.append((name, _load(spec), is_causal))
        except Exception as e:                      # noqa: BLE001
            print(f"[skip] {name}: import failed ({type(e).__name__}: {e})")

    results = {}   # (method, dataset) -> row
    t_start = time.perf_counter()
    for name, fn, is_causal in loaded:
        for dn in dnames:
            results[(name, dn)] = score_one(fn, DATASETS[dn], verify=args.verify and is_causal)
    wall = time.perf_counter() - t_start

    # ---- table: corr + time per method x dataset ----
    print(f"\n{'='*100}\nCORR  (time_s)   — by method × dataset" + ("  [verified]" if args.verify else "") + f"\n{'='*100}")
    hdr = f"{'method':16s} " + " ".join(f"{dn[:13]:>14s}" for dn in dnames)
    print(hdr); print("-" * len(hdr))
    for name, _, _ in loaded:
        cells = []
        for dn in dnames:
            r = results[(name, dn)]
            if not r["success"]:
                cells.append(f"{'FAIL':>14s}")
            else:
                flag = "" if r["causal"] in (None, True) else "!"  # ! = leaked
                cells.append(f"{r['corr']:.3f}{flag}({r['time_s']*1000:4.0f}ms)"[:14].rjust(14))
        print(f"{name:16s} " + " ".join(cells))

    # ---- per-dataset best CAUSAL method + cost-of-causality vs best oracle ----
    causal_names = {n for n, _, c in loaded if c}
    oracle_names = {n for n, _, c in loaded if not c}
    print(f"\n{'='*100}\nBEST CAUSAL vs ORACLE CEILING (corr; higher=better)\n{'='*100}")
    print(f"{'dataset':18s} {'best_causal':>22s} {'oracle_best':>22s} {'gap':>8s}")
    print("-" * 74)
    for dn in dnames:
        cz = [(n, results[(n, dn)]) for n in causal_names if results[(n, dn)]["success"]]
        oz = [(n, results[(n, dn)]) for n in oracle_names if results[(n, dn)]["success"]]
        cz = [(n, r) for n, r in cz if np.isfinite(r["corr"])]
        bc = max(cz, key=lambda x: x[1]["corr"], default=None)
        bo = max(oz, key=lambda x: x[1]["corr"], default=None)
        bc_s = f"{bc[0]} {bc[1]['corr']:.3f}" if bc else "-"
        bo_s = f"{bo[0]} {bo[1]['corr']:.3f}" if bo else "-"
        gap = f"{bo[1]['corr']-bc[1]['corr']:+.3f}" if (bc and bo) else "-"
        print(f"{dn:18s} {bc_s:>22s} {bo_s:>22s} {gap:>8s}")

    print(f"\nTotal wall: {wall:.2f}s over {len(loaded)} methods × {len(dnames)} datasets"
          + ("  (with verifier)" if args.verify else "  (no verifier)"))


if __name__ == "__main__":
    main()
