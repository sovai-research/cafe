"""LEAKAGE AUDIT experiment -- the leakage leaderboard ("the epsilon of imputation").

Generalises the CAFE-specific causality check into a model-agnostic STANDARD and runs
it across the whole method panel. For every method we emit, on equal terms:

  * a binary CAUSALITY CERTIFICATE  -- max_revision under truncation invariance
    (re-impute on growing time-prefixes; an early cell that never changes => causal).
  * a continuous LEAKAGE SCORE      -- Delta = causal_MAE - bidir_MAE, the accuracy a
    method silently borrows from the future (its bidirectional fill vs the same method
    honestly applied as a strict point-in-time filter via trailing-window right-edge).

Uses src/cafe/audit.py (cafe.audit.leakage_report / audit_panel) -- the library API --
so the experiment is a thin driver over the reusable tool.

Panel: CAFE + the full simple causal suite (LOCF/mean/rolling/EWMA/Kalman/...) +
SoftImpute / TRMF / linear-interp (batch, leaky) + the causal rivals (NoTMF/SHASTA/
rGROUSE/OSW-Net) + gcimpute/BayOTIDE when installed. Deep models (SAITS/BRITS/...) are
audited only if the bench venv (torch+pypots) is active; otherwise noted, not faked.

Protocol (matches exp_horserace.py / exp_causal_rivals.py):
  * temporal 60/40 split; hold out RATE of LIVE-segment cells only; history stays
    observed (a deployed model imputes exactly there).
  * standardize_on_observed -> leak-free normalisation (stats from visible cells).
  * the certificate is computed on the same masked matrix.

Outputs:
  paper/tables/leakage_audit.tex     -- the leaderboard (sorted by Delta, desc)
  paper/figures/leakage_audit.pdf    -- Delta bars + certificate markers
Everything is printed to stdout.

Run:  python3 bench/exp_leakage_audit.py
Env:  LA_ROWS (1000), LA_RATE (0.10), LA_SEEDS (1), LA_PATTERNS ("block,mcar"),
      LA_DATASETS ("beijing,airquality,fredmd,etth"), LA_WINDOW (24).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import eval_utils as EU            # noqa: E402
import cafe.audit as AUDIT         # noqa: E402  (the model-agnostic tool under test)

ROWS = int(os.environ.get("LA_ROWS", 1000))
RATE = float(os.environ.get("LA_RATE", 0.10))
SEEDS = list(range(int(os.environ.get("LA_SEEDS", 1))))
PATTERNS = os.environ.get("LA_PATTERNS", "block,mcar").split(",")
WINDOW = int(os.environ.get("LA_WINDOW", 24))
DATASETS = os.environ.get("LA_DATASETS", "beijing,airquality,fredmd,etth").split(",")
CACHE = os.path.join(HERE, "leakage_audit_cache.json")

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
    if X.shape[1] > 40:                       # cap width for a fast audit
        X = X[:, :40]
    return np.ascontiguousarray(X)


# --------------------------------------------------------------------------- #
# Build the method panel: name -> (impute_fn, native_causal_bool).
# native_causal=True means the method is already strictly point-in-time, so the
# audit's causal variant == the method itself (no trailing-window readout needed).
# native_causal=False means we DERIVE its honest causal variant and measure Delta.
# --------------------------------------------------------------------------- #
def build_panel():
    panel = {}

    # CAFE (natively causal)
    try:
        import c_unified_penmf as P
        panel["CAFE"] = (lambda X: P.online_impute(X, {}), True)
    except Exception as e:
        print("  [skip] CAFE:", e)

    # Simple causal suite -- but causality is MEASURED, not assumed. linear_interp /
    # spline / nocb in m_naive are LEAKY; LOCF / seasonal / Kalman etc. are causal.
    try:
        import causal_simple as CS
        for nm, fn in CS.CAUSAL_SIMPLE.items():
            panel[nm] = ((lambda f: (lambda X: f(X, {})))(fn), True)
    except Exception as e:
        print("  [skip] simple causal suite:", e)

    # Explicitly LEAKY baselines (the negative controls). Audited as non-native so
    # their derived causal variant exposes the borrowed future.
    try:
        import m_naive as Nai
        panel["LinearInterp"] = (lambda X: Nai.linear_interp(X, {}), False)
        panel["SplineInterp"] = (lambda X: Nai.spline_interp(X, {}), False)
        panel["NOCB"] = (lambda X: Nai.nocb(X, {}), False)
    except Exception as e:
        print("  [skip] m_naive leaky baselines:", e)

    # Batch low-rank / temporal-MF (bidirectional, leaky) -> derive causal variant.
    try:
        import m_softimpute as S
        panel["SoftImpute"] = (lambda X: S.impute(X, {}), False)
    except Exception as e:
        print("  [skip] SoftImpute:", e)
    try:
        import m_trmf as Tr
        panel["TRMF"] = (lambda X: Tr.impute(X, {}), False)
    except Exception as e:
        print("  [skip] TRMF:", e)

    # Causal rivals (online filters, claimed causal -> MEASURED).
    try:
        import causal_rivals as RV
        for nm, fn in RV.CAUSAL_RIVALS.items():
            panel[nm] = ((lambda f: (lambda X: f(X, {})))(fn), True)
    except Exception as e:
        print("  [skip] causal rivals:", e)

    # Online competitors (gcimpute / BayOTIDE) if installed.
    try:
        import online_competitors as OC
        if getattr(OC, "HAVE_GCIMPUTE", False):
            panel["gcimpute"] = (lambda X: OC.ONLINE_COMPETITORS["gcimpute"](X, {}), True)
        if getattr(OC, "HAVE_BAYOTIDE", False):
            panel["BayOTIDE"] = (lambda X: OC.ONLINE_COMPETITORS["BayOTIDE"](X, {}), True)
    except Exception as e:
        print("  [skip] online competitors:", e)

    # Deep models (only with torch+pypots in the bench venv). Audited bidir->causal.
    try:
        import deep_baselines as D
        if getattr(D, "HAVE_PYPOTS", False):
            for nm in ["SAITS", "BRITS"]:
                try:
                    m = D.build(nm, L=WINDOW, epochs=int(os.environ.get("LA_EPOCHS", 6)))
                    panel[nm] = ((lambda mm: (lambda X: _deep_bidir(mm, X)))(m), False)
                except Exception as e:
                    print(f"  [skip] {nm}:", repr(e)[:120])
        else:
            print("  [note] deep models skipped (HAVE_PYPOTS=False; run under "
                  ".venv-bench for SAITS/BRITS).")
    except Exception as e:
        print("  [note] deep baselines unavailable (no torch/pypots):", repr(e)[:120])

    return panel


def _deep_bidir(model, X):
    """Fit the deep model transductively on the given (NaN-holed) matrix and return
    its bidirectional fill. The audit then derives the causal (right-edge) variant."""
    model.fit(X)
    return model.fill_bidir(X)


# --------------------------------------------------------------------------- #
# One dataset: build the masked task, run the model-agnostic audit on every method.
# --------------------------------------------------------------------------- #
def audit_dataset(name, seed, pattern):
    clean = _load(name)
    T, N = clean.shape
    t0 = int(T * 0.6)
    mask = np.zeros((T, N), bool)
    mask[t0:] = EU.make_mask(pattern, (T - t0, N), RATE, seed)
    Xstd = EU.standardize_on_observed(clean, mask)        # leak-free truth
    # cafe.audit.leakage_report with mask=mask: certificate on the masked matrix,
    # accuracy Delta scored on the held-out cells, all on the leak-free scale.
    rows = []
    for mname, (fn, native) in PANEL.items():
        t0w = time.time()
        rep = AUDIT.leakage_report(fn, Xstd, mask=mask, native_causal=native,
                                   L=WINDOW, n_prefixes=6)
        rep["sec"] = time.time() - t0w
        rep.update(method=mname, dataset=name, pattern=pattern, seed=seed,
                   native=native)
        rows.append(rep)
        cert = "Y" if rep["causal"] else "N"
        print(f"    {mname:<16} cert={cert} max|rev|={rep['max_revision']:.2e} "
              f"bidir={rep['bidir_mae']:.3f} causal={rep['causal_mae']:.3f} "
              f"Delta={rep['leakage_delta']:+.3f}")
    return rows


# --------------------------------------------------------------------------- #
# Aggregate + table + figure.
# --------------------------------------------------------------------------- #
def aggregate(results):
    keyed = {}
    for r in results:
        keyed.setdefault(r["method"], []).append(r)
    agg = {}
    for m, rs in keyed.items():
        def mean(k):
            v = [x[k] for x in rs if x[k] == x[k]]   # drop NaN
            return float(np.mean(v)) if v else float("nan")
        # a method is certified causal iff it certifies on EVERY task (worst case)
        causal_all = all(bool(x["causal"]) for x in rs)
        agg[m] = {"causal": causal_all,
                  "max_revision": float(np.max([x["max_revision"] for x in rs])),
                  "bidir_mae": mean("bidir_mae"), "causal_mae": mean("causal_mae"),
                  "leakage_delta": mean("leakage_delta"),
                  "native": bool(rs[0]["native"])}
    return agg


def _fmt_rev(d):
    if d <= 0.0:
        return r"$0.0$"
    if d < 1e-6:
        return r"$0.0$"
    mant, exp = ("%.1e" % d).split("e")
    return r"$%s\mathrm{e}{%d}$" % (mant, int(exp))


def write_table(agg, n_tasks):
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["leakage_delta"])
    path = os.path.join(ROOT, "paper", "tables", "leakage_audit.tex")
    lines = [
        r"\begin{table*}[tbp]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{The leakage leaderboard --- a model-agnostic look-ahead "
        r"audit (``the $\epsilon$ of imputation'').} Every method, wrapped through the "
        r"same neutral standard (\texttt{cafe.audit.leakage\_report}), receives a "
        r"binary \emph{causality certificate} (max revision of an early imputed cell "
        r"under truncation invariance: re-impute on growing time-prefixes; a strictly "
        r"point-in-time method never revises an early fill) and a continuous "
        r"\emph{leakage score} $\Delta=$ causal$-$bidirectional MAE (the accuracy a "
        r"method silently borrows from the future, measured by applying the same model "
        r"as an honest right-edge filter). Sorted by $\Delta$ (most leaky first), "
        r"mean over " + str(n_tasks) + r" real tasks. Natively causal methods (\cafe{}, "
        r"LOCF, Kalman, the online filters) certify with $\Delta\!\approx\!0$; batch "
        r"and interpolation methods borrow large $\Delta$ they cannot keep in a "
        r"backtest.}",
        r"\label{tab:leakageaudit}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Causal? & max$|$rev$|$ & $\Delta$ leak & Bidir MAE & Causal MAE \\",
        r"\midrule",
    ]
    for m, a in rows:
        cert = r"\checkmark" if a["causal"] else r"$\times$"
        mname = r"\textbf{%s}" % m if m == "CAFE" else m
        d = a["leakage_delta"]
        dstr = ("%+.3f" % d) if d == d else "--"
        lines.append(f"{mname} & {cert} & {_fmt_rev(a['max_revision'])} & {dstr} & "
                     f"{a['bidir_mae']:.3f} & {a['causal_mae']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    txt = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(txt)
    print(f"\n[table] wrote {path}\n")
    print(txt)


def write_figure(agg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[fig] matplotlib unavailable, skipping:", e)
        return
    rows = sorted(agg.items(), key=lambda kv: kv[1]["leakage_delta"])
    names = [m for m, _ in rows]
    deltas = [a["leakage_delta"] for _, a in rows]
    causal = [a["causal"] for _, a in rows]
    colors = ["#2a9d8f" if c else "#e76f51" for c in causal]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"leakage $\Delta$ = causal MAE $-$ bidirectional MAE "
                  r"(future borrowed)")
    ax.set_title("Leakage leaderboard: certified causal (green) vs leaky (red)")
    for yi, (m, a) in zip(y, rows):
        mk = "o" if a["causal"] else "x"
        ax.scatter([deltas[yi] if False else max(deltas) * 1.02], [yi],
                   marker=mk, color="black", s=18, zorder=3)
    fig.tight_layout()
    path = os.path.join(ROOT, "paper", "figures", "leakage_audit.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def main():
    global PANEL
    t_start = time.time()
    print(f"LEAKAGE AUDIT | datasets={DATASETS} rows<={ROWS} rate={RATE} "
          f"patterns={PATTERNS} window={WINDOW} seeds={len(SEEDS)}")
    PANEL = build_panel()
    print(f"panel ({len(PANEL)} methods): {list(PANEL)}\n")

    results = []
    for pattern in PATTERNS:
        for ds in DATASETS:
            for seed in SEEDS:
                print(f"=== [{pattern}] {ds} seed={seed} ===")
                try:
                    results.extend(audit_dataset(ds, seed, pattern))
                except Exception as e:
                    print(f"  dataset {ds} failed:", repr(e)[:160])
                print()

    agg = aggregate(results)
    n_tasks = len({(r["dataset"], r["pattern"], r["seed"]) for r in results})

    print("=" * 78)
    print("LEAKAGE LEADERBOARD (sorted by Delta, most leaky first)")
    print(f"{'method':<16}{'cert':>5}{'max|rev|':>12}{'Delta':>9}"
          f"{'bidir':>8}{'causal':>8}")
    print("-" * 78)
    for m, a in sorted(agg.items(), key=lambda kv: -kv[1]["leakage_delta"]):
        cert = "Y" if a["causal"] else "N"
        print(f"{m:<16}{cert:>5}{a['max_revision']:>12.2e}"
              f"{a['leakage_delta']:>+9.3f}{a['bidir_mae']:>8.3f}"
              f"{a['causal_mae']:>8.3f}")
    print("=" * 78)

    # Separation summary (the honest verdict, quantified).
    causal_methods = [m for m, a in agg.items() if a["causal"]]
    leaky_methods = [m for m, a in agg.items() if not a["causal"]]
    causal_delta = [agg[m]["leakage_delta"] for m in causal_methods]
    leaky_delta = [agg[m]["leakage_delta"] for m in leaky_methods]
    print(f"certified causal ({len(causal_methods)}): {causal_methods}")
    print(f"  -> max |Delta| among certified-causal: "
          f"{max(abs(x) for x in causal_delta) if causal_delta else float('nan'):.4f}")
    print(f"leaky / uncertified ({len(leaky_methods)}): {leaky_methods}")
    print(f"  -> mean Delta among leaky: "
          f"{np.mean(leaky_delta) if leaky_delta else float('nan'):.4f}")

    with open(CACHE, "w") as f:
        json.dump({"agg": agg, "results": results,
                   "meta": {"datasets": DATASETS, "rows": ROWS, "rate": RATE,
                            "patterns": PATTERNS, "window": WINDOW,
                            "n_tasks": n_tasks}}, f, indent=2, default=float)
    print(f"[cache] wrote {CACHE}")

    write_table(agg, n_tasks)
    write_figure(agg)
    print(f"Elapsed: {time.time()-t_start:.1f}s")
    return agg


if __name__ == "__main__":
    main()
