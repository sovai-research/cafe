"""Gap #1 experiment: race CAFE vs the closest CAUSAL/online STATISTICAL rivals.

CAFE claims to subsume the causal/online cluster. This experiment makes the claim
falsifiable: it runs CAFE against faithful, strictly point-in-time implementations
of the four nearest statistical rivals (bench/causal_rivals.py) -- plus online TRMF
and, when installed, gcimpute / BayOTIDE as published references -- on the 8
STRUCTURED real panels under BOTH contiguous-block and scattered (MCAR) gaps.

Protocol (identical to exp_horserace.py, reusing eval_utils + the _load pattern):
  * temporal 60/40 split; hold out RATE of LIVE-segment cells only (a deployed model
    would impute exactly there); history stays observed.
  * standardize_on_observed -> leak-free (normalisation stats from visible cells).
  * every method is run strictly point-in-time (no future used).
  * score MAE on the identical held-out live cells.

Output: paper/tables/causal_rivals.tex (mean causal MAE over the 8 structured panels
per method, CAFE bold + sorted; caption notes which CAFE by-products each rival
LACKS). Full ranking is printed to stdout.

Run:  python3 bench/exp_causal_rivals.py
Env:  CR_ROWS (1200), CR_RATE (0.10), CR_SEEDS (1), CR_PATTERNS ("block,mcar").
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import eval_utils as EU                 # noqa: E402
import causal_rivals as CR              # noqa: E402

ROWS = int(os.environ.get("CR_ROWS", 1200))
RATE = float(os.environ.get("CR_RATE", 0.10))
SEEDS = list(range(int(os.environ.get("CR_SEEDS", 1))))
PATTERNS = os.environ.get("CR_PATTERNS", "block,mcar").split(",")

# The 8 STRUCTURED panels (exchange/FX is the no-structure control, excluded from
# the headline mean -- see exp_horserace.py).
DATASETS = ["fredmd", "airquality", "appliances", "beijing",
            "traffic2", "etth", "solar", "electric"]

DATASET_DESC = {
    "fredmd": "US macro (FRED-MD)", "airquality": "Air-quality sensors",
    "appliances": "Appliance energy", "beijing": "Beijing air-quality",
    "traffic2": "Road traffic (PEMS)", "etth": "Transformer temp/load",
    "solar": "Solar PV power", "electric": "Electricity demand",
}


def _load(name):
    """Same loader as exp_horserace._load (small slices for a fast real-world race)."""
    files = {
        "fredmd": "fredmd_clean.npy", "airquality": "airquality_clean.npy",
        "appliances": "appliances_clean.npy", "beijing": "beijing_clean.npy",
        "traffic2": "traffic2_clean.npy", "etth": "etth_clean.npy",
        "solar": "solar_clean.npy", "electric": "electric_clean.npy",
    }
    X = np.load(os.path.join(ROOT, "data", files[name])).astype(float)
    if X.shape[0] > ROWS:
        X = X[:ROWS]
    if X.shape[1] > 100:
        X = X[:, :100]
    return np.ascontiguousarray(X)


# --------------------------------------------------------------------------- #
# Methods: CAFE + the four causal_rivals + online TRMF + (optional) refs.
# Each entry: name -> (family, fn(X)->filled).
# --------------------------------------------------------------------------- #
def _methods():
    out = {}
    # CAFE (the subject)
    try:
        import c_unified_penmf as P
        out["CAFE"] = ("cafe", lambda X: P.online_impute(X, {}))
    except Exception as e:
        print("  [skip] CAFE:", repr(e)[:120])

    # The four implemented causal/online statistical rivals
    for name, fn in CR.CAUSAL_RIVALS.items():
        out[name] = ("rival", (lambda f: (lambda X: f(X, {})))(fn))

    # Online TRMF (already in the repo; a temporal-matrix-factorization reference)
    try:
        import c_online_trmf as TR
        out["TRMF (online)"] = ("ref", lambda X: TR.online_impute(X, {}))
    except Exception as e:
        print("  [skip] online TRMF:", repr(e)[:120])

    # Published online references, if installed
    try:
        import online_competitors as OC
        if getattr(OC, "HAVE_GCIMPUTE", False):
            out["gcimpute"] = ("ref", lambda X: OC.ONLINE_COMPETITORS["gcimpute"](X, {}))
        if getattr(OC, "HAVE_BAYOTIDE", False):
            out["BayOTIDE"] = ("ref", lambda X: OC.ONLINE_COMPETITORS["BayOTIDE"](X, {}))
    except Exception as e:
        print("  [skip] online_competitors refs:", repr(e)[:120])

    return out


def race_dataset(name, seed, pattern):
    clean = _load(name)
    T, N = clean.shape
    t0 = int(T * 0.6)                             # history / live (deployment) split
    mask = np.zeros((T, N), bool)
    mask[t0:] = EU.make_mask(pattern, (T - t0, N), RATE, seed)
    Xstd = EU.standardize_on_observed(clean, mask)   # LEAK-FREE
    Xobs = Xstd.copy()
    Xobs[mask] = np.nan
    truth = Xstd

    rows = []
    for mname, (family, fn) in _methods().items():
        rec = {"dataset": name, "method": mname, "family": family,
               "pattern": pattern, "seed": seed, "mae": None, "sec": None}
        try:
            t = time.time()
            pred = fn(np.array(Xobs, copy=True))
            rec["sec"] = time.time() - t
            rec["mae"] = EU.score_masked(truth, pred, mask)["mae"]
        except Exception as e:
            print(f"    {mname} failed:", repr(e)[:140])
        rows.append(rec)
        mae = f"{rec['mae']:.3f}" if rec["mae"] is not None else "  -  "
        print(f"    {mname:<14} causal MAE {mae}")
    return rows


# --------------------------------------------------------------------------- #
# Aggregate + table
# --------------------------------------------------------------------------- #
def _mean_over(results, method, datasets):
    v = [r["mae"] for r in results
         if r["method"] == method and r["dataset"] in datasets and r["mae"] is not None]
    return float(np.mean(v)) if v else None


def write_table(results, meta):
    methods = []
    for r in results:
        if r["method"] not in methods:
            methods.append(r["method"])
    fam = {r["method"]: r["family"] for r in results}

    # mean causal MAE over the 8 structured panels (pooled over patterns+seeds)
    ranking = [(m, _mean_over(results, m, DATASETS)) for m in methods]
    ranking = sorted([x for x in ranking if x[1] is not None], key=lambda x: x[1])

    # caption: which CAFE by-products each rival LACKS
    lack_bits = []
    for m, _ in ranking:
        if fam.get(m) == "rival" and m in CR.RIVAL_LACKS:
            lack_bits.append(r"\emph{%s} lacks %s" %
                             (m, ", ".join(CR.RIVAL_LACKS[m])))
    lack_note = (" CAFE additionally emits six by-products (uncertainty, factors, "
                 "anomaly scores, a dependency network, a trend/season decomposition, "
                 "and a forecast) that the statistical rivals do not: "
                 + "; ".join(lack_bits) + "."
                 ) if lack_bits else ""

    pats = ", ".join(PATTERNS)
    cap = (r"\textbf{CAFE vs the closest causal/online statistical rivals "
           r"(mean point-in-time MAE $\downarrow$ over the 8 structured real panels, "
           r"%d\%% missing, %s gaps).} Every method is run strictly point-in-time "
           r"(temporal 60/40 split, held-out live-segment cells, leak-free "
           r"standardisation). The four rivals are faithful online filters in CAFE's "
           r"own lane: NoTMF (AR-factor), SHASTA-PCA (streaming heteroscedastic PPCA), "
           r"rGROUSE (robust subspace tracking) and OSW-Net (switching-network SLDS, a "
           r"compact MissNet stand-in); TRMF/gcimpute/BayOTIDE are published references "
           r"where available.%s" % (int(RATE * 100), pats, lack_note))

    lines = [r"\begin{table}[tbp]\centering\small",
             r"\setlength{\tabcolsep}{6pt}",
             r"\caption{%s}" % cap,
             r"\label{tab:causalrivals}",
             r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Method & Causal MAE $\downarrow$ & Family \\", r"\midrule"]
    famlabel = {"cafe": "factor (ours)", "rival": "online rival",
                "ref": "published ref"}
    for m, mae in ranking:
        f = fam.get(m, "ref")
        nm = r"\textbf{%s}" % m if f == "cafe" else m
        val = r"\textbf{%.3f}" % mae if f == "cafe" else "%.3f" % mae
        lines.append(f"{nm} & {val} & {famlabel.get(f, f)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(ROOT, "paper", "tables", "causal_rivals.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[table] wrote {path}")
    return ranking


def main():
    print(f"CAUSAL RIVALS RACE | datasets={DATASETS} rows<={ROWS} rate={RATE} "
          f"seeds={len(SEEDS)} patterns={PATTERNS}")
    print(f"  rivals available: {list(CR.CAUSAL_RIVALS)}")
    results = []
    for pattern in PATTERNS:
        for ds in DATASETS:
            for seed in SEEDS:
                print(f"\n=== [{pattern}] {ds} ({DATASET_DESC.get(ds, ds)}) seed={seed} ===")
                try:
                    results.extend(race_dataset(ds, seed, pattern))
                except Exception as e:
                    print(f"  dataset {ds} failed:", repr(e)[:160])

    meta = {"datasets": DATASETS, "rows_cap": ROWS, "rate": RATE,
            "seeds": len(SEEDS), "patterns": PATTERNS,
            "protocol": "temporal 60/40 split; hold out RATE of live-segment cells; "
                        "all methods point-in-time; leak-free standardize_on_observed",
            "rivals": list(CR.CAUSAL_RIVALS), "causality": CR.CAUSALITY}
    ranking = write_table(results, meta)

    print("\n================ FINAL RANKING (mean causal MAE, 8 structured panels) ================")
    for i, (m, mae) in enumerate(ranking, 1):
        print(f"  {i:>2}. {m:<16} {mae:.4f}")
    print("======================================================================================")


if __name__ == "__main__":
    main()
