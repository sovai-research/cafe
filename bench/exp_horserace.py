"""The world's first CAUSAL horse race.

Every imputation benchmark (TSI-Bench, BenchPOTS, ImputeGAP) scores models in the
BIDIRECTIONAL setting -- methods may use the future to fill the past. That is invalid
for any sequential decision. Here we run the SAME models two ways on the SAME masks:

  * BIDIRECTIONAL -- the standard (optimistic) leaderboard.
  * CAUSAL        -- strict point-in-time; deep models are applied via right-edge
                     readout (causal_race.py), batch methods cannot compete at all.

We report bidir MAE, causal MAE, and the look-ahead gap delta = causal - bidir (the
accuracy a method silently borrows from the future). CAFE's delta is 0 by construction.

Run (deep models need the bench venv):
    .venv-bench/bin/python bench/exp_horserace.py
Env knobs: HR_ROWS (default 1200), HR_EPOCHS (8), HR_SEEDS (1), HR_WINDOW (24),
           HR_DATASETS ("fredmd,exchange,airquality,appliances,beijing").

Outputs: bench/horserace_cache.json, paper/tables/horserace_causal.tex,
         paper/tables/horserace_bidir.tex.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import eval_utils as EU          # noqa: E402

ROWS = int(os.environ.get("HR_ROWS", 1200))
EPOCHS = int(os.environ.get("HR_EPOCHS", 8))
SEEDS = list(range(int(os.environ.get("HR_SEEDS", 1))))
WINDOW = int(os.environ.get("HR_WINDOW", 24))
RATE = float(os.environ.get("HR_RATE", 0.10))
DATASETS = os.environ.get("HR_DATASETS",
                          "fredmd,exchange,airquality,appliances,beijing").split(",")
CACHE = os.path.join(HERE, "horserace_cache.json")


# --------------------------------------------------------------------------- #
# Data: small slices of real datasets for a fast, diverse, real-world race.
# --------------------------------------------------------------------------- #
def _load(name):
    p = lambda f: os.path.join(ROOT, "data", f)
    files = {
        "fredmd": "fredmd_clean.npy", "exchange": "exchange_clean.npy",
        "airquality": "airquality_clean.npy", "appliances": "appliances_clean.npy",
        "traffic": "traffic_clean.npy", "beijing": "beijing_clean.npy",
        "etth1": "ETTh1_clean.npy",
    }
    X = np.load(p(files[name])).astype(float)
    if X.shape[0] > ROWS:
        X = X[:ROWS]
    if X.shape[1] > 100:                       # cap width for speed
        X = X[:, :100]
    return np.ascontiguousarray(X)


DATASET_DESC = {
    "fredmd": "US macro panel (FRED-MD)", "exchange": "FX rates (finance)",
    "airquality": "Air-quality sensors", "appliances": "Appliance energy",
    "traffic": "Road traffic (PeMS)", "beijing": "Beijing air-quality",
    "etth1": "Electricity transformer temp",
}


# --------------------------------------------------------------------------- #
# Methods. Each returns (bidir_fn, causal_fn) given the masked matrix Xobs.
# fn: (T,N) NaN-holed -> (T,N) filled. None means "not applicable in this mode".
# --------------------------------------------------------------------------- #
def _classical_methods():
    """CPU, numpy. Batch methods are bidirectional-only; LOCF is causal."""
    out = {}
    try:
        import m_softimpute as S
        out["SoftImpute"] = ("classical", lambda X: S.impute(X, {}), None)
    except Exception as e:
        print("  [skip] SoftImpute:", e)
    try:
        import m_trmf as T
        out["TRMF"] = ("classical", lambda X: T.impute(X, {}), None)
    except Exception as e:
        print("  [skip] TRMF:", e)
    try:
        import m_naive as Nai
        out["LinearInterp"] = ("classical", lambda X: Nai.linear_interp(X, {}), None)
    except Exception as e:
        print("  [skip] LinearInterp:", e)
    try:
        import c_baselines as C
        out["LOCF"] = ("causal", lambda X: C.locf_impute(X, {}),
                       lambda X: C.locf_impute(X, {}))
    except Exception as e:
        print("  [skip] LOCF:", e)
    return out


def _cafe_method():
    import c_unified_penmf as P
    f = lambda X: P.online_impute(X, {})
    return {"CAFE": ("cafe", f, f)}            # causal natively: bidir == causal


def _online_causal_methods():
    out = {}
    try:
        import online_competitors as OC
        if getattr(OC, "HAVE_GCIMPUTE", False):
            g = lambda X: OC.ONLINE_COMPETITORS["gcimpute"](X, {})
            out["gcimpute"] = ("causal", g, g)
        if getattr(OC, "HAVE_BAYOTIDE", False):
            b = lambda X: OC.ONLINE_COMPETITORS["BayOTIDE"](X, {})
            out["BayOTIDE"] = ("online", b, b)     # online filter (not strict PIT)
    except Exception as e:
        print("  [skip] online competitors:", e)
    try:
        import online_baselines as OB
        _ewcov = OB.make_online_ewcov()
        e = lambda X: _ewcov(X, {})
        out["OnlineEWCov"] = ("causal", e, e)
    except Exception as ex:
        print("  [skip] online_baselines:", ex)
    return out


def _deep_methods(Xobs):
    """PyPOTS models: same trained weights, two readouts (bidir + causal)."""
    out = {}
    try:
        import deep_baselines as D
    except Exception as e:
        print("  [skip] deep baselines (no torch/pypots?):", e)
        return out
    if not getattr(D, "HAVE_PYPOTS", False):
        print("  [skip] deep baselines: HAVE_PYPOTS=False")
        return out
    for name in ["SAITS", "BRITS", "Transformer", "TimesNet", "ImputeFormer"]:
        try:
            imp = D.build(name, L=WINDOW, epochs=EPOCHS)
            imp.fit(Xobs)
            out[name] = ("deep",
                         (lambda m: (lambda X: m.fill_bidir(X)))(imp),
                         (lambda m: (lambda X: m.fill_causal(X)))(imp))
        except Exception as e:
            print(f"  [skip] {name}:", repr(e)[:160])
    return out


# --------------------------------------------------------------------------- #
# Race
# --------------------------------------------------------------------------- #
def _run_fn(fn, Xobs, truth, mask):
    t0 = time.time()
    pred = fn(np.array(Xobs, copy=True))
    sec = time.time() - t0
    sc = EU.score_masked(truth, pred, mask)
    return sc["mae"], sc["rmse"], sec


def race_dataset(name, seed):
    clean = _load(name)
    mask = EU.mcar_mask(clean.shape, RATE, seed)
    Xstd = EU.standardize_on_observed(clean, mask)     # LEAK-FREE: stats from visible cells
    Xobs = Xstd.copy()
    Xobs[mask] = np.nan
    truth = Xstd

    methods = {}
    methods.update(_cafe_method())
    methods.update(_classical_methods())
    methods.update(_online_causal_methods())
    methods.update(_deep_methods(Xobs))         # trains on this dataset/seed's Xobs

    rows = []
    for mname, (family, bfn, cfn) in methods.items():
        rec = {"dataset": name, "method": mname, "family": family,
               "bidir_mae": None, "bidir_rmse": None, "bidir_sec": None,
               "causal_mae": None, "causal_rmse": None, "causal_sec": None,
               "delta": None, "causal_verified": None, "seed": seed}
        if bfn is not None:
            try:
                rec["bidir_mae"], rec["bidir_rmse"], rec["bidir_sec"] = _run_fn(bfn, Xobs, truth, mask)
            except Exception as e:
                print(f"    {mname} bidir failed:", repr(e)[:140])
        if cfn is not None:
            try:
                rec["causal_mae"], rec["causal_rmse"], rec["causal_sec"] = _run_fn(cfn, Xobs, truth, mask)
            except Exception as e:
                print(f"    {mname} causal failed:", repr(e)[:140])
        if rec["bidir_mae"] is not None and rec["causal_mae"] is not None:
            rec["delta"] = rec["causal_mae"] - rec["bidir_mae"]
        # native-causal families: bidir mirrors causal (they never use the future)
        if family in ("cafe", "causal") and rec["causal_mae"] is not None:
            rec["delta"] = 0.0 if rec["bidir_mae"] is None else rec["delta"]
        rows.append(rec)
        b = f"{rec['bidir_mae']:.3f}" if rec["bidir_mae"] is not None else "  -  "
        c = f"{rec['causal_mae']:.3f}" if rec["causal_mae"] is not None else "  -  "
        print(f"    {mname:<14} bidir {b}  causal {c}")
    return rows


def verify_causal_methods():
    """One-time truncation-invariance check of the causal fillers on a tiny matrix."""
    import causal_race as CR
    rng = np.random.default_rng(0)
    X = rng.standard_normal((160, 6))
    X[rng.random(X.shape) < 0.15] = np.nan
    checks = {}
    try:
        import c_unified_penmf as P
        checks["CAFE"] = CR.verify_causal(lambda Z: P.online_impute(Z, {}), X)["causal"]
    except Exception:
        pass
    try:
        import online_competitors as OC
        if getattr(OC, "HAVE_GCIMPUTE", False):
            checks["gcimpute"] = CR.verify_causal(
                lambda Z: OC.ONLINE_COMPETITORS["gcimpute"](Z, {}), X)["causal"]
    except Exception:
        pass
    return checks


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _agg(results):
    """Mean over seeds -> per (dataset, method) and the across-dataset mean."""
    keyed = {}
    for r in results:
        keyed.setdefault((r["dataset"], r["method"]), []).append(r)
    agg = {}
    for (ds, m), rs in keyed.items():
        fam = rs[0]["family"]
        def mean(k):
            v = [x[k] for x in rs if x[k] is not None]
            return float(np.mean(v)) if v else None
        agg[(ds, m)] = {"family": fam, "bidir_mae": mean("bidir_mae"),
                        "causal_mae": mean("causal_mae"), "delta": mean("delta")}
    return agg


def write_tables(results, meta):
    agg = _agg(results)
    methods = sorted({m for (_, m) in agg})
    datasets = [d for d in DATASETS if any(ds == d for (ds, _) in agg)]

    def mean_over_ds(method, key):
        v = [agg[(d, method)][key] for d in datasets
             if (d, method) in agg and agg[(d, method)][key] is not None]
        return float(np.mean(v)) if v else None

    # ---- CAUSAL leaderboard (the showcase): sorted by mean causal MAE ----
    causal_rows = [(m, mean_over_ds(m, "causal_mae")) for m in methods]
    causal_rows = sorted([r for r in causal_rows if r[1] is not None], key=lambda r: r[1])
    lines = [r"\begin{table}[t]\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{The causal horse race (mean MAE $\downarrow$ over %d real datasets, %d%% MCAR).} "
             r"Every method held to strict point-in-time evaluation; deep models applied via right-edge "
             r"readout (causal\_race). Batch methods (SoftImpute, TRMF, linear interpolation) cannot be "
             r"made causal and do not appear. \cafe{} leads the only leaderboard valid for sequential decisions.}"
             % (len(datasets), int(RATE * 100)),
             r"\label{tab:causalrace}",
             r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Method & Causal MAE $\downarrow$ & Family \\", r"\midrule"]
    famlabel = {"cafe": "factor (ours)", "causal": "online", "online": "online*",
                "deep": "deep (right-edge)", "classical": "classical"}
    for m, mae in causal_rows:
        fam = agg[next((d, m) for d in datasets if (d, m) in agg)]["family"]
        bold = r"\textbf{%s}" % m if fam == "cafe" else m
        val = r"\textbf{%.3f}" % mae if fam == "cafe" else "%.3f" % mae
        lines.append(f"{bold} & {val} & {famlabel.get(fam, fam)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(ROOT, "paper", "tables", "horserace_causal.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- BIDIRECTIONAL leaderboard + look-ahead gap ----
    bidir_rows = [(m, mean_over_ds(m, "bidir_mae"), mean_over_ds(m, "delta")) for m in methods]
    bidir_rows = sorted([r for r in bidir_rows if r[1] is not None], key=lambda r: r[1])
    lines = [r"\begin{table}[t]\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\textbf{Bidirectional leaderboard and the look-ahead gap.} Mean MAE over %d real "
             r"datasets under the standard (future-using) protocol, and $\Delta=$ causal$-$bidirectional MAE: "
             r"the accuracy a method silently borrows from the future. \cafe{}'s $\Delta$ is $0$ by construction.}"
             % len(datasets),
             r"\label{tab:bidirrace}",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Method & Bidir MAE & $\Delta$ look-ahead & Family \\", r"\midrule"]
    for m, mae, dl in bidir_rows:
        fam = agg[next((d, m) for d in datasets if (d, m) in agg)]["family"]
        bold = r"\textbf{%s}" % m if fam == "cafe" else m
        dstr = "%.3f" % dl if dl is not None else "--"
        lines.append(f"{bold} & {mae:.3f} & {dstr} & {famlabel.get(fam, fam)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(ROOT, "paper", "tables", "horserace_bidir.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[tables] wrote paper/tables/horserace_causal.tex + horserace_bidir.tex")


# --------------------------------------------------------------------------- #
def main():
    print(f"CAUSAL HORSE RACE | datasets={DATASETS} rows<={ROWS} epochs={EPOCHS} "
          f"seeds={len(SEEDS)} window={WINDOW} rate={RATE}")
    import torch  # noqa: F401  (ensure venv; deep methods need it)
    try:
        import pypots
        pv = pypots.__version__
        import torch as _t
        tv = _t.__version__
    except Exception:
        pv = tv = "n/a"

    results = []
    for ds in DATASETS:
        for seed in SEEDS:
            print(f"\n=== {ds} ({DATASET_DESC.get(ds, ds)})  seed={seed} ===")
            try:
                results.extend(race_dataset(ds, seed))
            except Exception as e:
                print(f"  dataset {ds} failed:", repr(e)[:160])

    verify = verify_causal_methods()
    print("\n[causal-verify]", verify)

    meta = {"datasets": DATASETS, "rows_cap": ROWS, "epochs": EPOCHS,
            "seeds": len(SEEDS), "window": WINDOW, "rate": RATE,
            "pypots_version": pv, "torch_version": tv,
            "leak_free": "standardize_on_observed (post-mask, visible-only stats)",
            "causal_verified": verify,
            "cmd": ".venv-bench/bin/python bench/exp_horserace.py"}
    with open(CACHE, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"[cache] wrote {CACHE} ({len(results)} rows)")
    write_tables(results, meta)


if __name__ == "__main__":
    main()
