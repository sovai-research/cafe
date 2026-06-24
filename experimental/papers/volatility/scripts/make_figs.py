"""Render the figures for the CAFE-as-volatility-model investigation from the saved
experiment JSON (run exp1/exp2/exp3 first). Produces three PDFs in ../figures/."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def fig1_univariate():
    e1 = load("exp1_univariate.json")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    # (a) half-life sweep
    sw = e1["halflife_sweep"]["t"]
    hls = [int(k) for k in sw["curve"]]
    ql = list(sw["curve"].values())
    ax[0].plot(hls, ql, "o-", ms=3)
    ax[0].axvline(sw["best_hl"], color="g", ls="--", label=f"optimal HL={sw['best_hl']}")
    ax[0].axvline(sw["cafe_hl"], color="r", ls="--", label=f"CAFE HL={int(sw['cafe_hl'])}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("EW half-life (steps)")
    ax[0].set_ylabel("QLIKE vs true variance")
    ax[0].set_title("(a) CAFE's scale half-life is\nfar slower than vol clustering needs")
    ax[0].set_ylim(min(ql) - 0.02, np.percentile(ql, 80) + 0.05)
    ax[0].legend(fontsize=7)
    # (b) spike response
    sp = e1["spike"]
    h = np.array(sp["h"]); cafe = np.array(sp["cafe"]); ewma = np.array(sp["ewma"])
    t = np.arange(len(h)); win = slice(450, 800)
    ax[1].plot(t[win], h[win], "k-", lw=1.5, label="true variance")
    ax[1].plot(t[win], ewma[win], "-", color="C0", label="EWMA(0.94)")
    ax[1].plot(t[win], cafe[win], "-", color="C3", label="CAFE scale (HL=200)")
    ax[1].set_xlabel("time"); ax[1].set_ylabel("conditional variance")
    ax[1].set_title("(b) CAFE's scale lags the spike\nand barely reverts")
    ax[1].legend(fontsize=7)
    # (c) QLIKE bars vs GARCH MLE (t innovations)
    agg = e1["dists"]["t"]
    names = ["GARCH_t_MLE", "GARCH_normal_MLE", "EWMA_0.94", "CAFE_scale_HL200"]
    labels = ["GARCH-t\nMLE", "GARCH-N\nMLE", "EWMA\n0.94", "CAFE\nHL200"]
    vals = [agg[n]["qlike_true_mean"] for n in names]
    base = vals[0]
    bars = ax[2].bar(labels, vals, color=["C2", "C2", "C0", "C3"])
    ax[2].axhline(base, color="C2", ls="--", lw=0.8)
    ax[2].set_ylim(base - 0.04, max(vals) + 0.05)
    ax[2].set_ylabel("QLIKE vs true variance")
    ax[2].set_title("(c) As-is, CAFE is the worst\nunivariate vol forecaster")
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v, f"+{v-base:.3f}",
                   ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_univariate.pdf"))
    plt.close(fig)


def fig2_betat():
    e2 = load("exp2_betatgarch.json")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    # (a) clean vs robust QLIKE
    clean = e2["clean"]; rob = e2["robust"]
    names = ["BetaT_GARCH", "GARCH_t_MLE", "CAFE_scale_HL200"]
    lab = ["Beta-t\nGARCH", "GARCH-t\nMLE", "CAFE\nHL200"]
    cv = [clean[n]["qlike_true_mean"] for n in names]
    x = np.arange(len(names))
    ax[0].bar(x, cv, color=["C4", "C2", "C3"])
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab)
    ax[0].set_ylim(min(cv) - 0.04, max(cv) + 0.04)
    ax[0].set_ylabel("QLIKE vs true variance")
    ax[0].set_title("(a) Clean data: the upgrade\nmatches GARCH-MLE")
    # (b) robustness: overshoot
    infl = e2["robust_spike_inflation"]
    ax[1].bar(["Beta-t\nGARCH", "Gaussian\nGARCH"],
              [infl["BetaT_GARCH"], infl["GARCH_normal_MLE"]], color=["C4", "C1"])
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8, label="true variance")
    ax[1].set_ylabel("post-outlier variance overshoot (x)")
    ax[1].set_title("(b) Outliers: Beta-t down-weights,\nGaussian GARCH over-reacts")
    ax[1].legend(fontsize=7)
    # (c) spike tracking
    sp = e2["spike"]
    h = np.array(sp["h"]); hb = np.array(sp["betat"]); hc = np.array(sp["cafe"])
    t = np.arange(len(h)); win = slice(450, 800)
    ax[2].plot(t[win], h[win], "k-", lw=1.5, label="true variance")
    ax[2].plot(t[win], hb[win], "-", color="C4", label="Beta-t-GARCH")
    ax[2].plot(t[win], hc[win], "-", color="C3", label="CAFE HL200")
    ax[2].set_xlabel("time"); ax[2].set_ylabel("conditional variance")
    ax[2].set_title("(c) Same machinery + score recursion\nnow tracks the spike")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_betatgarch.pdf"))
    plt.close(fig)


def fig3_panel():
    e3 = load("exp3_panel.json")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    # (a) curse of dimensionality
    B = e3["B_curse"]
    Ns = sorted(int(k) for k in B)
    fv = [B[str(n)]["CAFE_FV"]["mvport"] for n in Ns]
    cc = [B[str(n)]["CCC"]["mvport"] for n in Ns]
    dc = [B[str(n)]["DCC"]["mvport"] for n in Ns]
    orc = [B[str(n)]["ORACLE"]["mvport"] for n in Ns]
    ax[0].plot(Ns, fv, "o-", color="C4", label="CAFE-FV (rank K)")
    ax[0].plot(Ns, cc, "s-", color="C1", label="CCC-GARCH")
    dcn = [n for n, d in zip(Ns, dc) if d is not None]
    dcv = [d for d in dc if d is not None]
    ax[0].plot(dcn, dcv, "^-", color="C3", label="DCC-GARCH")
    ax[0].plot(Ns, orc, "k--", lw=0.8, label="oracle")
    ax[0].set_xlabel("N assets (train T=350)"); ax[0].set_ylabel("MV-portfolio variance")
    ax[0].set_yscale("log")
    ax[0].set_title("(a) Curse of dimensionality:\nDCC/CCC blow up, CAFE-FV improves")
    ax[0].legend(fontsize=7)
    # (b) gaps
    C = e3["C_gaps"]
    order = ["CAFE_FV_native", "DCC_cafeimpute", "DCC_meanimpute", "CCC_meanimpute"]
    lab = ["CAFE-FV\nnative", "DCC +\nCAFE-impute", "DCC +\nmean-impute",
           "CCC +\nmean-impute"]
    vals = [C[k]["mvport_mean"] for k in order]
    err = [C[k]["mvport_std"] for k in order]
    ax[1].bar(lab, vals, yerr=err, color=["C4", "C0", "C1", "C3"], capsize=3)
    ax[1].set_ylabel("MV-portfolio variance")
    ax[1].set_title(f"(b) {C['avg_missing_frac']*100:.0f}% async gaps: native CAFE-FV"
                    "\nwins; naive impute is catastrophic")
    # (c) runtime
    D = e3["D_runtime"]
    Nd = sorted(int(k) for k in D)
    fvfit = [D[str(n)]["CAFE_FV"]["fit_sec"] for n in Nd]
    cccfit = [D[str(n)]["CCC"]["fit_sec"] for n in Nd]
    dccfit = [(D[str(n)]["DCC"].get("fit_sec")) for n in Nd]
    ax[2].plot(Nd, fvfit, "o-", color="C4", label="CAFE-FV (K MLE fits)")
    ax[2].plot(Nd, cccfit, "s-", color="C1", label="CCC (N MLE fits)")
    dn = [n for n, d in zip(Nd, dccfit) if d is not None]
    dv = [d for d in dccfit if d is not None]
    ax[2].plot(dn, dv, "^-", color="C3", label="DCC (N + N^2/step)")
    ax[2].set_xlabel("N assets"); ax[2].set_ylabel("fit wall-clock (s)")
    ax[2].set_title("(c) Runtime: CAFE-FV flat in N,\nDCC super-linear / infeasible")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_panel.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_univariate()
    fig2_betat()
    fig3_panel()
    print("wrote fig1_univariate.pdf, fig2_betatgarch.pdf, fig3_panel.pdf to figures/")
