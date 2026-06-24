"""Build the field-survey figure (leakage bar chart) and the LaTeX table fragment
for the position paper, straight from data/field_survey.json. Writes into THIS DIR.

  python3 experimental/papers/position/scripts/make_figs_tables.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))

d = json.load(open(os.path.join(DATA, "field_survey.json")))
agg = d["agg"]; meta = d["meta"]

# stable display order: leaky (by Delta desc) on top, then causal block
rows = sorted(agg.items(), key=lambda kv: (kv[1]["causal"], -kv[1]["leakage_delta"]))


# ---------------- figure: leakage bars + certificate markers ---------------- #
def make_fig():
    order = sorted(agg.items(), key=lambda kv: kv[1]["leakage_delta"])
    names = [m for m, _ in order]
    deltas = [a["leakage_delta"] for _, a in order]
    causal = [a["causal"] for _, a in order]
    colors = ["#2a9d8f" if c else "#e76f51" for c in causal]
    fig, ax = plt.subplots(figsize=(6.6, max(3.5, 0.30 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel(r"leakage $\Delta$ = causal MAE $-$ bidirectional MAE  (future borrowed)")
    ax.set_title("Field survey: certified causal (teal, $\\Delta\\!=\\!0$) vs uncertified (orange)",
                 fontsize=9)
    # certificate marker at left margin
    xpos = min(deltas) - 0.03
    for yi, (m, a) in zip(y, order):
        mk = r"$\checkmark$" if a["causal"] else r"$\times$"
        ax.text(xpos, yi, mk, va="center", ha="center", fontsize=8,
                color="#2a9d8f" if a["causal"] else "#e76f51")
    ax.set_xlim(xpos - 0.02, max(deltas) + 0.04)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(FIG, "field_survey.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("[fig]", p)


# ---------------- LaTeX table fragment -------------------------------------- #
def fmt_rev(d):
    if d <= 1e-6:
        return r"$0$"
    mant, exp = ("%.1e" % d).split("e")
    return r"$%s{\times}10^{%d}$" % (mant, int(exp))


def make_table():
    # group: leaky first (Delta desc), then causal (MAE asc within)
    leaky = sorted([kv for kv in agg.items() if not kv[1]["causal"]],
                   key=lambda kv: -kv[1]["leakage_delta"])
    causal = sorted([kv for kv in agg.items() if kv[1]["causal"]],
                    key=lambda kv: kv[1]["causal_mae"])
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\caption{\textbf{The leakage field survey.} Every imputer, wrapped through the "
        r"\emph{same} neutral standard (\texttt{cafe.audit.leakage\_report}), receives a "
        r"binary causality certificate (max revision of an early fill under prefix "
        r"truncation) and a leakage score $\Delta=\mathrm{MAE}_{\text{causal}}-"
        r"\mathrm{MAE}_{\text{bidir}}$. Mean over " + str(meta["n_tasks"]) +
        r" real tasks (" + ",".join(meta["datasets"]) + r"; block+MCAR; "
        r"window $L{=}" + str(meta["window"]) + r"$). \textbf{Top:} uncertified "
        r"(leaky); \textbf{bottom:} certified strictly point-in-time. \cafe{} is the "
        r"existence proof: certified, $\Delta\!=\!0$, and the lowest causal MAE.}",
        r"\label{tab:fieldsurvey}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Method & Cert. & max$|$rev$|$ & $\Delta$ & Bidir & Causal \\",
        r"\midrule",
        r"\multicolumn{6}{@{}l}{\emph{Uncertified (leaky): future is borrowed or fills revise}}\\",
    ]
    for m, a in leaky:
        cert = r"$\times$"
        lines.append(f"{m} & {cert} & {fmt_rev(a['max_revision'])} & "
                     f"{a['leakage_delta']:+.3f} & {a['bidir_mae']:.3f} & {a['causal_mae']:.3f} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{@{}l}{\emph{Certified strictly point-in-time ($\Delta\!=\!0$, max$|$rev$|\!=\!0$)}}\\")
    for m, a in causal:
        nm = r"\textbf{\cafe{}}" if m == "CAFE" else m
        lines.append(f"{nm} & $\\checkmark$ & $0$ & "
                     f"{a['leakage_delta']:+.3f} & {a['bidir_mae']:.3f} & {a['causal_mae']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(DATA, "field_survey_table.tex"), "w") as f:
        f.write(txt)
    print("[tex]", os.path.join(DATA, "field_survey_table.tex"))


# ---------------- gap-theory real-panel band table -------------------------- #
def make_gap_table():
    g = json.load(open(os.path.join(DATA, "gap_theory.json")))
    band = g["ar_band"]
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{When the leak bites, per panel.} Lag-1 memory $\hat a$ "
        r"(point-in-time estimate) and the closed-form predicted look-ahead gap "
        r"$\Delta(\hat a,g)$ from Eq.~\eqref{eq:deltaclosed} at gap lengths $g{=}3,10$. "
        r"High-memory panels (FRED-MD, Beijing, AirQuality) have a large exploitable "
        r"gap; near-memoryless panels (Traffic, Electric) give the future almost "
        r"nothing. Closed form matches exact filter/smoother runs at "
        r"$R^2={" + ("%.4f" % g["r2"]) + r"}$ (median rel.\ err.\ "
        r"${" + ("%.1f" % (100 * g["median_rel_err"])) + r"}\%$).}",
        r"\label{tab:gapband}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Panel & $\hat a$ & $\Delta(\hat a,3)$ & $\Delta(\hat a,10)$ \\",
        r"\midrule",
    ]
    nice = {"fredmd": "FRED-MD", "beijing": "Beijing", "airquality": "AirQuality",
            "appliances": "Appliances", "solar": "Solar", "traffic2": "Traffic",
            "etth": "ETTh", "electric": "Electric", "exchange": "Exchange (FX)"}
    for k in ["fredmd", "exchange", "appliances", "beijing", "airquality", "solar",
              "etth", "traffic2", "electric"]:
        if k not in band:
            continue
        b = band[k]
        lines.append(f"{nice.get(k,k)} & {b['a_hat']:.3f} & {b['Dpred_g3']:.4f} & "
                     f"{b['Dpred_g10']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(DATA, "gap_band_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[tex]", os.path.join(DATA, "gap_band_table.tex"))


if __name__ == "__main__":
    make_fig(); make_table(); make_gap_table()
    # quick separation stats for the README / paper text
    cau = [v for v in agg.values() if v["causal"]]
    leak = [v for v in agg.values() if not v["causal"]]
    print(f"causal n={len(cau)} maxAbsDelta={max(abs(v['leakage_delta']) for v in cau):.4f}")
    print(f"leaky  n={len(leak)} meanDelta={np.mean([v['leakage_delta'] for v in leak]):+.4f}")
    print("CAFE causal MAE:", round(agg['CAFE']['causal_mae'], 3),
          "best causal rival:",
          min(((k, v['causal_mae']) for k, v in agg.items() if v['causal'] and k != 'CAFE'),
              key=lambda kv: kv[1]))
