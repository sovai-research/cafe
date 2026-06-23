"""Post-process the horse-race cache: per-dataset block table + block-only cache for
the flip figure. Reads bench/horserace_cache.json; does not re-run any model."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_cache = json.load(open(os.path.join(HERE, "horserace_cache.json")))
R, META = _cache["results"], _cache["meta"]
ds = sorted(set(r["dataset"] for r in R))
dn = {"fredmd": "FRED-MD", "exchange": "Exchange", "airquality": "AirQual",
      "appliances": "Applnce", "beijing": "Beijing"}
methods = [("CAFE", "cafe"), ("BayOTIDE", "online"), ("OnlineEWCov", "causal"),
           ("KalmanLL", "causal"), ("LOCF", "causal"), ("SAITS", "deep"),
           ("ImputeFormer", "deep")]


def get(pat, m, d):
    rr = [r for r in R if r["pattern"] == pat and r["dataset"] == d and r["method"] == m]
    return rr[0]["causal_mae"] if rr and rr[0]["causal_mae"] is not None else None


best = {d: min(v for m, _ in methods if (v := get("block", m, d)) is not None) for d in ds}

caption = (
    r"\caption{\textbf{Causal horse race, per dataset (block-missing, causal MAE $\downarrow$).} "
    r"Strictly point-in-time evaluation on five real datasets. \cafe{} is the best causal imputer "
    r"on every dataset with genuine factor structure (macro, air-quality, energy, Beijing); on the "
    r"near-random-walk FX series a last-value / Kalman model is correctly better (a factor prior is "
    r"the wrong model there). Deep imputers, trained on history and applied strictly point-in-time, "
    r"collapse throughout. Best per column in \textbf{bold}.}"
)
L = [r"\begin{table}[t]\centering\small", r"\setlength{\tabcolsep}{4pt}", caption,
     r"\label{tab:horseraceperds}",
     r"\begin{tabular}{@{}l" + "c" * len(ds) + r"@{}}", r"\toprule",
     "Method (causal) & " + " & ".join(dn.get(d, d) for d in ds) + r" \\", r"\midrule"]
for m, fam in methods:
    cells = []
    for d in ds:
        v = get("block", m, d)
        if v is None:
            cells.append("--")
        elif abs(v - best[d]) < 1e-9:
            cells.append(r"\textbf{%.3f}" % v)
        else:
            cells.append("%.3f" % v)
    nm = (r"\textbf{%s}" % m) if fam == "cafe" else (m + r"$^\dagger$" if fam == "deep" else m)
    L.append(nm + " & " + " & ".join(cells) + r" \\")
    if m == "LOCF":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}",
      r" \\[2pt]{\footnotesize $^\dagger$deep models, applied strictly point-in-time "
      r"(right-edge readout).}", r"\end{table}"]
open(os.path.join(ROOT, "paper", "tables", "horserace_perdataset.tex"), "w").write("\n".join(L) + "\n")
print("wrote paper/tables/horserace_perdataset.tex")

blk = {"meta": META, "results": [r for r in R if r["pattern"] == "block"]}
json.dump(blk, open(os.path.join(HERE, "horserace_cache_block.json"), "w"))
print("wrote bench/horserace_cache_block.json", len(blk["results"]), "rows")
