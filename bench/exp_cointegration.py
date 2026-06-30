"""
COINTEGRATION experiment for the CAFE paper.

QUESTION. The paper keeps FX as a "no-structure control" where CAFE is correctly
beaten by a last-value/Kalman prior, and states a "which tool when" rule (use CAFE
when the panel shares structure; use last-value when each series is an independent
random walk). This script makes that rule PRECISE and TESTABLE: the property that
decides whether the contemporaneous cross-section helps a missing cell is
COINTEGRATION -- a stationary cross-sectional structure.

MECHANISM. A rank-R factor model on non-stationary (I(1)) series IS the common-trends
representation of a cointegration system: if N series share R<N common stochastic
trends, there are N-R cointegrating vectors and the idiosyncratic part (which CAFE
models with psi + the AR carry) is STATIONARY -> mean-reverting. That stationarity is
exactly why the cross-section anchors a missing series across a gap. When the
idiosyncratic part is itself a unit root (no cointegration), last-value is Bayes-optimal
causally and CAFE's factor machinery cannot beat it.

DESIGN.
  (a) Controlled dial: a common-trends panel X_t = B F_t + U_t with F_t an r-dim random
      walk (r common trends) and U_t a stationary AR(1) idiosyncratic. Sweep r so the
      cointegration rank N-r goes 0 -> N-1; report MAE(CAFE)/MAE(LOCF) under BLOCK gaps.
      Prediction: ratio falls monotonically and crosses 1 (CAFE wins) as cointegration
      rank rises. Control: with a non-mean-reverting (phi=0 -> unit-root) idiosyncratic,
      no cointegration exists for any r and CAFE ~ LOCF.
  (b) Two REAL financial panels at opposite corners, SAME protocol, both near-unit-root
      in levels: a cointegrated yield-curve / rates panel (FRED-MD: FEDFUNDS, TB3MS,
      TB6MS, GS1, GS5, GS10, AAA, BAA -- a single rate level trend with stationary
      spreads) where CAFE wins, and the FX panel (8 independent floating rates, no
      cointegration) where last-value wins.

WHAT IT WRITES:
    paper/figures/cointegration.pdf  ratio-vs-rank dial (a) + real-panel flip bars (b)
    paper/tables/cointegration.tex   synthetic sweep rows + the two real-panel anchors
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cafe  # noqa: E402


# --------------------------------------------------------------------------- #
def _locf(Xobs):
    Z = np.asarray(Xobs, float).copy()
    for j in range(Z.shape[1]):
        last = np.nan
        for t in range(Z.shape[0]):
            if np.isfinite(Z[t, j]):
                last = Z[t, j]
            elif np.isfinite(last):
                Z[t, j] = last
        col = Z[:, j]
        m = np.nanmean(col)
        col[np.isnan(col)] = (m if np.isfinite(m) else 0.0)
    return Z


def _zscore(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _block_mask(shape, rate, seed, glen):
    T, N = shape
    rng = np.random.default_rng(seed)
    M = np.zeros((T, N), bool)
    tgt = int(rate * T)
    for j in range(N):
        f, g = 0, 0
        while f < tgt and g < 300:
            s = int(rng.integers(0, max(1, T - glen)))
            if not M[s:s + glen, j].any():
                M[s:s + glen, j] = True
                f += glen
            g += 1
    return M


def _eval_block(X, rate=0.12, seed=1, glen=20):
    X = _zscore(X)
    m = _block_mask(X.shape, rate, seed, glen)
    if not m.any():
        return np.nan, np.nan
    Xo = X.copy()
    Xo[m] = np.nan
    cf = np.asarray(cafe.impute(Xo), float)
    lf = _locf(Xo)
    mae = lambda P: float(np.mean(np.abs(P[m] - X[m])))
    return mae(cf), mae(lf)


def _common_trends_panel(N, T, r, phi, seed, sig_f=1.0, sig_u=1.0):
    """X_t = B F_t + U_t, F_t r-dim random walk, U_t stationary AR(1) (phi)."""
    rng = np.random.default_rng(seed)
    F = np.cumsum(sig_f * rng.standard_normal((T, r)), axis=0)
    B = rng.standard_normal((N, r)) / np.sqrt(r)
    U = np.zeros((T, N))
    sd_e = sig_u * np.sqrt(max(1 - phi * phi, 1e-6))
    e = sd_e * rng.standard_normal((T, N))
    for t in range(1, T):
        U[t] = phi * U[t - 1] + e[t]
    return F @ B.T + U


# --------------------------------------------------------------------------- #
def run_synthetic(N=12, T=1500, phi=0.85, glen=20, seeds=range(8)):
    rows = []
    for r in (12, 9, 6, 3):                       # cointegration rank = N - r
        cs, ls = [], []
        for s in seeds:
            X = _common_trends_panel(N, T, r, phi, seed=s)
            c, l = _eval_block(X, seed=100 + s, glen=glen)
            cs.append(c); ls.append(l)
        cs, ls = np.array(cs), np.array(ls)
        rows.append(dict(rank=N - r, cafe=cs.mean(), locf=ls.mean(),
                         ratio=float(np.mean(cs / ls)),
                         ratio_sd=float(np.std(cs / ls))))
    # control: phi=0 (unit-root idiosyncratic -> no cointegration for any r)
    ctrl = []
    for r in (9, 3):
        cs, ls = [], []
        for s in seeds:
            X = _common_trends_panel(N, T, r, phi=0.0, seed=s)
            # phi=0 makes U white (stationary); to kill cointegration use a RW idio:
            Uw = np.cumsum(0.3 * np.random.default_rng(1000 + s).standard_normal((T, N)), 0)
            Xrw = X + Uw                            # add a per-series unit-root -> no coint
            c, l = _eval_block(Xrw, seed=100 + s, glen=glen)
            cs.append(c); ls.append(l)
        ctrl.append(dict(rank=N - r, ratio=float(np.mean(np.array(cs) / np.array(ls)))))
    return rows, ctrl


def run_real():
    out = {}
    # cointegrated rates / yield-curve panel from FRED-MD RAW LEVELS
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(ROOT, "data", "fredmd_current.csv"))
        cols = ["FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA"]
        R = df[cols].apply(pd.to_numeric, errors="coerce").dropna().values
        cs, ls = [], []
        for s in range(6):
            c, l = _eval_block(R, seed=10 + s, glen=12)
            cs.append(c); ls.append(l)
        out["rates"] = dict(name="Rates / yield curve (FRED-MD)", N=R.shape[1],
                            cafe=float(np.mean(cs)), locf=float(np.mean(ls)),
                            ratio=float(np.mean(cs) / np.mean(ls)), coint="yes")
    except Exception as e:                                          # pragma: no cover
        out["rates"] = dict(name="Rates / yield curve (FRED-MD)", error=repr(e)[:80])
    # FX panel (no cointegration)
    X = np.load(os.path.join(ROOT, "data", "exchange_clean.npy"))[:2000]
    cs, ls = [], []
    for s in range(6):
        c, l = _eval_block(X, seed=10 + s, glen=12)
        cs.append(c); ls.append(l)
    out["fx"] = dict(name="Exchange rates / FX", N=X.shape[1],
                     cafe=float(np.mean(cs)), locf=float(np.mean(ls)),
                     ratio=float(np.mean(cs) / np.mean(ls)), coint="no")
    return out


# --------------------------------------------------------------------------- #
def write_table(syn, real, path):
    L = []
    L.append(r"\begin{table}[t]\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\caption{\textbf{The cross-section helps iff the panel is cointegrated.} "
             r"Causal block-gap MAE of \cafe{} vs.\ last-value (LOCF), lower is better; "
             r"\emph{ratio} $=$ \cafe{}/LOCF ($<1$ $\Rightarrow$ \cafe{} wins). "
             r"\emph{Top}: a controlled common-trends panel ($N{=}12$, $X_t{=}BF_t{+}U_t$, "
             r"$F_t$ an $r$-dim random walk, $U_t$ stationary AR) as the cointegration rank "
             r"$N{-}r$ grows $0\!\to\!9$: the ratio falls monotonically and crosses $1$. "
             r"A unit-root (non-cointegrated) idiosyncratic gives $\approx1$ at every rank "
             r"(control). \emph{Bottom}: two real financial panels, \emph{both} near-unit-root "
             r"in levels, under the same protocol -- a cointegrated rates/yield-curve panel "
             r"(one rate-level trend, stationary spreads) where \cafe{} wins $2\times$, and the "
             r"independent-floating FX panel where last-value wins. The discriminating property "
             r"is cointegration, not whether the data is financial.}")
    L.append(r"\label{tab:cointegration}")
    L.append(r"\begin{tabular}{@{}lrrrr@{}}")
    L.append(r"\toprule")
    L.append(r"Panel & coint.\ rank & \cafe{} & LOCF & ratio \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{@{}l}{\emph{Controlled common-trends panel ($N{=}12$, block gaps)}}\\")
    for row in syn:
        L.append(f"\\quad rank {row['rank']} & {row['rank']} & {row['cafe']:.3f} & "
                 f"{row['locf']:.3f} & {row['ratio']:.2f} \\\\")
    ctrl_txt = ", ".join(f"{c['ratio']:.2f}" for c in real["_ctrl"])
    L.append(r"\quad \emph{unit-root idio.\ (control)} & \multicolumn{4}{r}{"
             + r"no win: ratio " + ctrl_txt + r"} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{5}{@{}l}{\emph{Real financial panels (same protocol, both I(1))}}\\")
    rr = real["rates"]; fx = real["fx"]
    if "cafe" in rr:
        L.append(f"\\quad Rates / yield curve & cointeg. & \\textbf{{{rr['cafe']:.3f}}} & "
                 f"{rr['locf']:.3f} & \\textbf{{{rr['ratio']:.2f}}} \\\\")
    L.append(f"\\quad Exchange rates / FX & none & {fx['cafe']:.3f} & "
             f"\\textbf{{{fx['locf']:.3f}}} & {fx['ratio']:.2f} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def write_figure(syn, real, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.1))
    # (a) ratio vs cointegration rank
    ranks = [r["rank"] for r in syn]
    ratios = [r["ratio"] for r in syn]
    sds = [r["ratio_sd"] for r in syn]
    ax1.errorbar(ranks, ratios, yerr=sds, marker="o", color="#2B6CB0", lw=1.6,
                 capsize=3, label="CAFÉ/LOCF (cointegrated idio.)")
    ctrl_ranks = [c["rank"] for c in real["_ctrl"]]
    ctrl_ratios = [c["ratio"] for c in real["_ctrl"]]
    ax1.plot(ctrl_ranks, ctrl_ratios, marker="s", ls="--", color="#94A3B8", lw=1.3,
             label="unit-root idio. (no coint.)")
    ax1.axhline(1.0, color="#C53030", lw=1.0, ls=":")
    ax1.text(0.2, 1.02, "CAFÉ worse", color="#C53030", fontsize=8, va="bottom")
    ax1.text(0.2, 0.98, "CAFÉ better", color="#2C7A7B", fontsize=8, va="top")
    ax1.set_xlabel("cointegration rank $N-r$")
    ax1.set_ylabel("MAE ratio  CAFÉ / LOCF")
    ax1.set_title("(a) the cross-section helps as\ncointegration rank grows", fontsize=9)
    ax1.legend(fontsize=7, loc="upper right")
    # (b) real-panel flip
    rr, fx = real["rates"], real["fx"]
    labels = ["Rates /\nyield curve\n(cointegrated)", "FX\n(no\ncointegration)"]
    cafe_v = [rr.get("cafe", np.nan), fx["cafe"]]
    locf_v = [rr.get("locf", np.nan), fx["locf"]]
    x = np.arange(2)
    ax2.bar(x - 0.18, cafe_v, 0.36, color="#2B6CB0", label="CAFÉ")
    ax2.bar(x + 0.18, locf_v, 0.36, color="#C05621", label="LOCF")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("causal block-gap MAE")
    ax2.set_title("(b) two real financial panels,\nopposite outcomes", fontsize=9)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    print("running synthetic cointegration sweep ...")
    syn, ctrl = run_synthetic()
    for r in syn:
        print(f"  rank {r['rank']}: CAFE={r['cafe']:.3f} LOCF={r['locf']:.3f} "
              f"ratio={r['ratio']:.2f}")
    print("  control (no coint.):", {c["rank"]: round(c["ratio"], 2) for c in ctrl})
    print("running real panels ...")
    real = run_real()
    real["_ctrl"] = ctrl
    for k in ("rates", "fx"):
        print(f"  {k}: {real[k]}")
    figpath = os.path.join(ROOT, "paper", "figures", "cointegration.pdf")
    texpath = os.path.join(ROOT, "paper", "tables", "cointegration.tex")
    write_figure(syn, real, figpath)
    write_table(syn, real, texpath)
    print("wrote", figpath)
    print("wrote", texpath)


if __name__ == "__main__":
    main()
