"""Position-paper field-survey driver: run cafe.audit.leakage_report across the
full method panel and emit a JSON + a LaTeX table INTO THIS PAPER'S OWN DIR.

Standalone copy of the bench panel-build logic (so we never write into paper/).
Run under .venv-bench so SAITS/BRITS are included; falls back gracefully if not.

  cd /Users/dereksnow/Sovai/Github/TIMARA
  .venv-bench/bin/python experimental/papers/position/scripts/run_audit.py
"""
from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import json, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "/Users/dereksnow/Sovai/Github/TIMARA"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "data")
OUTDIR = os.path.abspath(OUTDIR)
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import eval_utils as EU              # noqa
import cafe.audit as AUDIT           # noqa  (the tool under test)

ROWS = int(os.environ.get("LA_ROWS", 1000))
RATE = float(os.environ.get("LA_RATE", 0.10))
SEEDS = list(range(int(os.environ.get("LA_SEEDS", 2))))
PATTERNS = os.environ.get("LA_PATTERNS", "block,mcar").split(",")
WINDOW = int(os.environ.get("LA_WINDOW", 24))
DATASETS = os.environ.get("LA_DATASETS", "beijing,airquality,fredmd,etth").split(",")

_FILES = {
    "fredmd": "fredmd_clean.npy", "exchange": "exchange_clean.npy",
    "airquality": "airquality_clean.npy", "appliances": "appliances_clean.npy",
    "traffic2": "traffic2_clean.npy", "beijing": "beijing_clean.npy",
    "etth": "etth_clean.npy", "solar": "solar_clean.npy",
    "electric": "electric_clean.npy",
}


def _load(name):
    X = np.load(os.path.join(ROOT, "data", _FILES[name])).astype(float)
    if X.shape[0] > ROWS:
        X = X[:ROWS]
    if X.shape[1] > 40:
        X = X[:, :40]
    return np.ascontiguousarray(X)


def _deep_bidir(model, X):
    model.fit(X)
    return model.fill_bidir(X)


def build_panel():
    panel = {}
    try:
        import c_unified_penmf as P
        panel["CAFE"] = (lambda X: P.online_impute(X, {}), True)
    except Exception as e:
        print("  [skip] CAFE:", repr(e)[:120])
    try:
        import causal_simple as CS
        for nm, fn in CS.CAUSAL_SIMPLE.items():
            panel[nm] = ((lambda f: (lambda X: f(X, {})))(fn), True)
    except Exception as e:
        print("  [skip] simple causal suite:", repr(e)[:120])
    try:
        import m_naive as Nai
        panel["LinearInterp"] = (lambda X: Nai.linear_interp(X, {}), False)
        panel["SplineInterp"] = (lambda X: Nai.spline_interp(X, {}), False)
        panel["NOCB"] = (lambda X: Nai.nocb(X, {}), False)
    except Exception as e:
        print("  [skip] m_naive leaky baselines:", repr(e)[:120])
    try:
        import m_softimpute as S
        panel["SoftImpute"] = (lambda X: S.impute(X, {}), False)
    except Exception as e:
        print("  [skip] SoftImpute:", repr(e)[:120])
    try:
        import m_trmf as Tr
        panel["TRMF"] = (lambda X: Tr.impute(X, {}), False)
    except Exception as e:
        print("  [skip] TRMF:", repr(e)[:120])
    try:
        import causal_rivals as RV
        for nm, fn in RV.CAUSAL_RIVALS.items():
            panel[nm] = ((lambda f: (lambda X: f(X, {})))(fn), True)
    except Exception as e:
        print("  [skip] causal rivals:", repr(e)[:120])
    try:
        import online_competitors as OC
        if getattr(OC, "HAVE_GCIMPUTE", False):
            panel["gcimpute"] = (lambda X: OC.ONLINE_COMPETITORS["gcimpute"](X, {}), True)
        if getattr(OC, "HAVE_BAYOTIDE", False):
            panel["BayOTIDE"] = (lambda X: OC.ONLINE_COMPETITORS["BayOTIDE"](X, {}), True)
    except Exception as e:
        print("  [skip] online competitors:", repr(e)[:120])
    try:
        import deep_baselines as D
        if getattr(D, "HAVE_PYPOTS", False):
            for nm in ["SAITS", "BRITS", "Transformer"]:
                try:
                    m = D.build(nm, L=WINDOW, epochs=int(os.environ.get("LA_EPOCHS", 8)))
                    panel[nm] = ((lambda mm: (lambda X: _deep_bidir(mm, X)))(m), False)
                except Exception as e:
                    print(f"  [skip] {nm}:", repr(e)[:120])
        else:
            print("  [note] deep models skipped (HAVE_PYPOTS=False).")
    except Exception as e:
        print("  [note] deep baselines unavailable:", repr(e)[:120])
    return panel


def audit_dataset(panel, name, seed, pattern):
    clean = _load(name)
    T, N = clean.shape
    t0 = int(T * 0.6)
    mask = np.zeros((T, N), bool)
    mask[t0:] = EU.make_mask(pattern, (T - t0, N), RATE, seed)
    Xstd = EU.standardize_on_observed(clean, mask)
    rows = []
    for mname, (fn, native) in panel.items():
        t0w = time.time()
        try:
            rep = AUDIT.leakage_report(fn, Xstd, mask=mask, native_causal=native,
                                       L=WINDOW, n_prefixes=6)
        except Exception as e:
            print(f"    {mname:<16} FAILED {repr(e)[:100]}")
            continue
        rep["sec"] = time.time() - t0w
        rep.update(method=mname, dataset=name, pattern=pattern, seed=seed, native=native)
        rows.append(rep)
        cert = "Y" if rep["causal"] else "N"
        print(f"    {mname:<16} cert={cert} max|rev|={rep['max_revision']:.2e} "
              f"bidir={rep['bidir_mae']:.3f} causal={rep['causal_mae']:.3f} "
              f"Delta={rep['leakage_delta']:+.3f}")
    return rows


def aggregate(results):
    keyed = {}
    for r in results:
        keyed.setdefault(r["method"], []).append(r)
    agg = {}
    for m, rs in keyed.items():
        def mean(k):
            v = [x[k] for x in rs if x[k] == x[k]]
            return float(np.mean(v)) if v else float("nan")
        causal_all = all(bool(x["causal"]) for x in rs)
        agg[m] = {"causal": causal_all,
                  "max_revision": float(np.max([x["max_revision"] for x in rs])),
                  "bidir_mae": mean("bidir_mae"), "causal_mae": mean("causal_mae"),
                  "leakage_delta": mean("leakage_delta"),
                  "native": bool(rs[0]["native"]), "n": len(rs)}
    return agg


def main():
    t_start = time.time()
    print(f"FIELD-SURVEY AUDIT | datasets={DATASETS} rows<={ROWS} rate={RATE} "
          f"patterns={PATTERNS} window={WINDOW} seeds={len(SEEDS)}")
    panel = build_panel()
    print(f"panel ({len(panel)} methods): {list(panel)}\n")
    results = []
    for pattern in PATTERNS:
        for ds in DATASETS:
            for seed in SEEDS:
                print(f"=== [{pattern}] {ds} seed={seed} ===")
                try:
                    results.extend(audit_dataset(panel, ds, seed, pattern))
                except Exception as e:
                    print(f"  dataset {ds} failed:", repr(e)[:160])
                print()
    agg = aggregate(results)
    n_tasks = len({(r["dataset"], r["pattern"], r["seed"]) for r in results})
    print("=" * 78)
    print("LEAKAGE FIELD SURVEY (sorted by Delta, most leaky first)")
    print(f"{'method':<16}{'cert':>5}{'max|rev|':>12}{'Delta':>9}{'bidir':>8}{'causal':>8}")
    print("-" * 78)
    for m, a in sorted(agg.items(), key=lambda kv: -kv[1]["leakage_delta"]):
        cert = "Y" if a["causal"] else "N"
        print(f"{m:<16}{cert:>5}{a['max_revision']:>12.2e}{a['leakage_delta']:>+9.3f}"
              f"{a['bidir_mae']:>8.3f}{a['causal_mae']:>8.3f}")
    print("=" * 78)
    causal_methods = [m for m, a in agg.items() if a["causal"]]
    leaky_methods = [m for m, a in agg.items() if not a["causal"]]
    print(f"certified causal ({len(causal_methods)}): {causal_methods}")
    print(f"leaky / uncertified ({len(leaky_methods)}): {leaky_methods}")
    os.makedirs(OUTDIR, exist_ok=True)
    out = {"agg": agg, "results": results,
           "meta": {"datasets": DATASETS, "rows": ROWS, "rate": RATE,
                    "patterns": PATTERNS, "window": WINDOW, "n_tasks": n_tasks,
                    "n_methods": len(agg)}}
    with open(os.path.join(OUTDIR, "field_survey.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[json] wrote {os.path.join(OUTDIR, 'field_survey.json')}")
    print(f"Elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
