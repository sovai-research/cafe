"""Emit LaTeX tables (data/*.tex) for volatility.tex from the saved experiment JSON."""
from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))


def load(n):
    with open(os.path.join(DATA, n)) as f:
        return json.load(f)


def w(name, s):
    with open(os.path.join(DATA, name), "w") as f:
        f.write(s)
    print("wrote", name)


def tab_univariate():
    e = load("exp1_univariate.json")  # tight cols
    pretty = {"GARCH_t_MLE": "GARCH-$t$ MLE",
              "GARCH_normal_MLE": "GARCH-$\\mathcal{N}$ MLE",
              "EWMA_0.94": "EWMA(0.94)",
              "CAFE_scale_HL200": "\\textbf{\\cafe{} scale (HL200)}"}
    order = ["GARCH_t_MLE", "GARCH_normal_MLE", "EWMA_0.94", "CAFE_scale_HL200"]
    rows = []
    for dist in ("normal", "t"):
        agg = e["dists"][dist]
        base = agg["GARCH_t_MLE"]["qlike_true_mean"]
        cells = []
        for k in order:
            m = agg[k]
            cells.append((pretty[k], m["qlike_true_mean"], m["qlike_true_mean"] - base,
                          m["var_mse_mean"], m["var_viol_mean"]))
        rows.append((dist, cells))
    sw_t = e["halflife_sweep"]["t"]
    s = []
    s.append("\\begin{table}[t]\\centering\\small")
    s.append("\\setlength{\\tabcolsep}{4pt}")
    s.append("\\caption{\\textbf{\\cafe{} as-is is the worst univariate volatility "
             "forecaster.} One-step variance forecasts on simulated GARCH(1,1) "
             "(8 seeds), scored by QLIKE against the \\emph{true} latent variance "
             "(lower better), variance-MSE, and 1\\% VaR violation rate (target "
             "0.010). \\cafe{}'s scale recursion is verified bit-identical "
             "($0.0$e$+$0) to the library's \\texttt{\\_update\\_robust\\_scale}. "
             "Its optimal EW half-life is $\\approx\\!22$; it uses $200$.}")
    s.append("\\label{tab:univariate}")
    s.append("\\begin{tabular}{@{}lcccc@{}}\\toprule")
    s.append("Method & QLIKE & $\\Delta$MLE & MSE & VaR \\\\\\midrule")
    for dist, cells in rows:
        lab = "Gaussian innov." if dist == "normal" else "Student-$t$ innov."
        s.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{lab}}}}}\\\\")
        for name, ql, gap, mse, viol in cells:
            s.append(f"\\quad {name} & {ql:.3f} & ${gap:+.3f}$ & {mse:.3f} & "
                     f"{viol:.3f} \\\\")
        s.append("\\addlinespace")
    s.append("\\bottomrule\\end{tabular}")
    s.append("\\end{table}")
    w("tab_univariate.tex", "\n".join(s))


def tab_betat():
    e = load("exp2_betatgarch.json")
    cl, rb, infl = e["clean"], e["robust"], e["robust_spike_inflation"]
    pretty = {"BetaT_GARCH": "\\textbf{Beta-$t$-GARCH}",
              "GARCH_t_MLE": "GARCH-$t$ MLE",
              "GARCH_normal_MLE": "GARCH-$\\mathcal{N}$ MLE",
              "CAFE_scale_HL200": "\\cafe{} scale, HL$=$200"}
    s = []
    s.append("\\begin{table}[t]\\centering\\small")
    s.append("\\setlength{\\tabcolsep}{4pt}")
    s.append("\\caption{\\textbf{The Beta-$t$-GARCH upgrade, built from \\cafe{}'s own "
             "IRLS weight, is competitive and robust.} QLIKE vs.\\ true variance "
             "(lower better), 8 seeds. \\emph{Clean}: true GARCH-$t$ returns. "
             "\\emph{Outliers}: $1\\%$ additive jumps not part of the vol process. "
             "Last column: median variance over-shoot the step after a jump "
             "($1.0$ ideal). The Beta-$t$ innovation $w\\,r^2$ is bit-identical "
             "($0.0$e$+$0) to \\cafe{}'s weight.}")
    s.append("\\label{tab:betat}")
    s.append("\\begin{tabular}{@{}lccc@{}}\\toprule")
    s.append("Method & clean & outliers & overshoot \\\\\\midrule")
    for k in ["BetaT_GARCH", "GARCH_t_MLE", "GARCH_normal_MLE", "CAFE_scale_HL200"]:
        c = cl.get(k, {}).get("qlike_true_mean")
        r = rb.get(k, {}).get("qlike_true_mean")
        cstr = f"{c:.3f}" if c is not None else "--"
        rstr = f"{r:.3f}" if r is not None else "--"
        iv = infl.get(k)
        istr = f"$\\times{iv:.2f}$" if iv is not None else "--"
        s.append(f"{pretty[k]} & {cstr} & {rstr} & {istr} \\\\")
    s.append("\\bottomrule\\end{tabular}")
    s.append("\\end{table}")
    w("tab_betat.tex", "\n".join(s))


def tab_panel():
    e = load("exp3_panel.json")
    A = e["A_quality"]; C = e["C_gaps"]
    pa = {"CAFE_FV": "\\textbf{\\cafe{}-FV} \\emph{(proposed)}", "DCC": "DCC-GARCH",
          "CCC": "CCC-GARCH", "EWMA": "EWMA cov.", "Sample": "Sample cov."}
    s = []
    s.append("\\begin{table}[t]\\centering\\small")
    s.append("\\setlength{\\tabcolsep}{4pt}")
    s.append("\\caption{\\textbf{The factor-volatility niche.} Out-of-sample "
             "minimum-variance-portfolio realized variance (the standard MGARCH "
             "economic loss, lower better). \\emph{Top}: complete data, $N{=}20$, "
             "6 seeds (oracle $=" + f"{A['ORACLE']['mvport_mean']:.3f}" + "$); DCC "
             "wins statistical covariance-QLIKE but \\cafe{}-FV's well-conditioned "
             "low-rank inverse wins the portfolio. \\emph{Bottom}: $16\\%$ "
             "asynchronous block gaps, $N{=}30$, 4 seeds; classical MGARCH needs a "
             "complete-data front-end.}")
    s.append("\\label{tab:panel}")
    s.append("\\begin{tabular}{@{}lcc@{}}\\toprule")
    s.append("Method & MV-port.\\ var. & cov-QLIKE \\\\\\midrule")
    s.append("\\multicolumn{3}{@{}l}{\\emph{Complete data ($N{=}20$)}}\\\\")
    items = sorted([(k, v) for k, v in A.items() if k != "ORACLE"],
                   key=lambda kv: kv[1]["mvport_mean"])
    for k, v in items:
        s.append(f"\\quad {pa[k]} & {v['mvport_mean']:.3f}\\,$\\pm$\\,"
                 f"{v['mvport_std']:.3f} & {v['covqlike_mean']:.2f} \\\\")
    s.append("\\addlinespace")
    miss = f"{C['avg_missing_frac']*100:.0f}"
    s.append("\\multicolumn{3}{@{}l}{\\emph{Asynchronous gaps (" + miss +
             "\\% missing, $N{=}30$)}}\\\\")
    pg = {"CAFE_FV_native": "\\textbf{\\cafe{}-FV} \\emph{(native gaps)}",
          "DCC_cafeimpute": "DCC $+$ \\cafe{}-impute",
          "DCC_meanimpute": "DCC $+$ mean-impute",
          "CCC_meanimpute": "CCC $+$ mean-impute"}
    gi = sorted([(k, v) for k, v in C.items() if k != "avg_missing_frac"],
                key=lambda kv: kv[1]["mvport_mean"])
    for k, v in gi:
        s.append(f"\\quad {pg[k]} & {v['mvport_mean']:.3f}\\,$\\pm$\\,"
                 f"{v['mvport_std']:.3f} & -- \\\\")
    s.append("\\bottomrule\\end{tabular}")
    s.append("\\end{table}")
    w("tab_panel.tex", "\n".join(s))


def tab_curse():
    e = load("exp3_panel.json")["B_curse"]
    Ns = sorted(int(k) for k in e)
    s = []
    s.append("\\begin{table}[t]\\centering\\small")
    s.append("\\setlength{\\tabcolsep}{4pt}")
    s.append("\\caption{\\textbf{Curse of dimensionality, fixed train $T{=}350$.} "
             "Sample-correlation condition number and MV-portfolio variance as $N$ "
             "grows. CCC/DCC degrade as the $N\\times N$ correlation becomes "
             "ill-conditioned (DCC's $O(N^2)$/step fit becomes infeasible past "
             "$N{=}80$); \\cafe{}-FV (rank $K{=}3$) keeps improving toward the "
             "oracle.}")
    s.append("\\label{tab:curse}")
    s.append("\\begin{tabular}{@{}lccccc@{}}\\toprule")
    s.append("$N$ & cond(corr) & \\cafe{}-FV & CCC & DCC & oracle \\\\\\midrule")
    for n in Ns:
        r = e[str(n)]
        dcc = r["DCC"]["mvport"]
        dccs = f"{dcc:.3f}" if dcc is not None else "\\emph{infeas.}"
        s.append(f"{n} & {r['corr_cond']:.0f} & {r['CAFE_FV']['mvport']:.3f} & "
                 f"{r['CCC']['mvport']:.3f} & {dccs} & "
                 f"{r['ORACLE']['mvport']:.3f} \\\\")
    s.append("\\bottomrule\\end{tabular}")
    s.append("\\end{table}")
    w("tab_curse.tex", "\n".join(s))


if __name__ == "__main__":
    tab_univariate(); tab_betat(); tab_panel(); tab_curse()
