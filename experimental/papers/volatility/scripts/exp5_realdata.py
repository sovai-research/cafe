"""
Experiment 5: do the panel results hold on REAL equity returns?

Data: Ken French daily VALUE-WEIGHTED portfolio returns, 1990-2026 (real volatility
clustering, fat tails, leverage -- none of which the sim had). 49 Industry portfolios
(N=49) and 100 size/BM portfolios (N=95 after the dense-window filter).

There is no true Sigma_t on real data, so cov-QLIKE is unavailable. We use the two
honest real-data metrics:
  * GMV: out-of-sample realized variance of the global minimum-variance portfolio
    w_t = Sigma_t^{-1}1 / (1' Sigma_t^{-1}1)  -- the standard MGARCH economic loss
    (Engle; Ledoit-Wolf). Lower = a better covariance forecast. Reported as annualised
    volatility, sqrt(252 * var) in %.
  * NLL: one-step predictive Gaussian negative log-likelihood of the realized return
    vector under Sigma_t (the density / rank-1-proxy QLIKE). Lower = better.

Part A : rolling-window quality, N=49, all models incl. DCC/CCC.
Part B : curse of dimensionality on real data -- short train, sweep N (sub-sampled
         from the N=95 panel); DCC capped.
Part C : asynchronous gaps on real returns -- CAFE-FV native vs DCC + (mean|CAFE) impute.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garch_lib as gl
import mvol_lib as mv

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def load(tag):
    z = np.load(os.path.join(OUT, f"{tag}_real.npz"), allow_pickle=True)
    return z["R"].astype(float)


def gmv_annvol(Sig, R_eval):
    rv, ok = mv.mv_portfolio_realized_var(Sig, R_eval)
    return float(np.sqrt(252.0 * rv) * 100.0), ok       # annualised vol in %


def pred_nll(Sig, R_eval):
    """Mean one-step Gaussian predictive NLL per observation (lower=better)."""
    Te, N = R_eval.shape
    vals = []
    for t in range(Te):
        if not np.all(np.isfinite(R_eval[t])):
            continue
        S = Sig[t] + 1e-10 * np.eye(N)
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(ld):
            continue
        q = R_eval[t] @ np.linalg.solve(S, R_eval[t])
        vals.append(0.5 * (ld + q + N * np.log(2 * np.pi)))
    return float(np.mean(vals)) if vals else np.nan


def _models(N, dcc_max=50):
    m = {"CAFE_FV": lambda: mv.CafeFV(3),
         "StaticFactor": lambda: mv.StaticFactorCov(3),
         "NLS": lambda: mv.NLSCov(),
         "LW_linear": lambda: mv.LWLinearCov(),
         "EWMA": lambda: mv.EWMACov(),
         "Sample": lambda: mv.SampleCov()}
    if N <= dcc_max:
        m["DCC"] = lambda: mv.DCCModel(True)
        m["CCC"] = lambda: mv.DCCModel(False)
    return m


def part_A(R, Ttr=1250, Tev=500, step=500):
    starts = list(range(0, len(R) - Ttr - Tev, step))
    agg = {k: {"gmv": [], "nll": []} for k in _models(R.shape[1])}
    for s in starts:
        Rtr = R[s:s + Ttr]; Rev = R[s + Ttr:s + Ttr + Tev]
        mu = Rtr.mean(0)
        for name, ctor in _models(R.shape[1]).items():
            M = ctor().fit(Rtr - mu)
            S = M.filter_eval(Rev - mu)
            g, _ = gmv_annvol(S, Rev - mu)
            agg[name]["gmv"].append(g)
            agg[name]["nll"].append(pred_nll(S, Rev - mu))
    out = {k: dict(gmv_mean=float(np.mean(v["gmv"])), gmv_std=float(np.std(v["gmv"])),
                   nll_mean=float(np.mean(v["nll"]))) for k, v in agg.items()}
    out["_n_windows"] = len(starts)
    return out


def part_B(R, Ns=(20, 40, 80, 95), Ttr=350, Tev=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = {}
    for N in Ns:
        cols = rng.choice(R.shape[1], size=min(N, R.shape[1]), replace=False)
        X = R[:, cols]
        # one mid-sample window with a short train (the ill-conditioned regime)
        s = (len(X) - Ttr - Tev) // 2
        Rtr, Rev = X[s:s + Ttr], X[s + Ttr:s + Ttr + Tev]
        mu = Rtr.mean(0)
        cond = float(np.linalg.cond(np.corrcoef(Rtr, rowvar=False)))
        ent = {"N": int(N), "corr_cond": cond}
        for name, ctor in _models(N, dcc_max=50).items():
            M = ctor().fit(Rtr - mu); S = M.filter_eval(Rev - mu)
            ent[name] = gmv_annvol(S, Rev - mu)[0]
        rows[int(N)] = ent
        print(f"  [B] N={N:3d} cond={cond:8.1f} | CAFE_FV={ent['CAFE_FV']:.2f} "
              f"statFac={ent['StaticFactor']:.2f} NLS={ent['NLS']:.2f} "
              f"DCC={ent.get('DCC','infeas')}")
    return rows


def part_C(R, N=49, Ttr=1250, Tev=500, seeds=range(4), gap_rate=0.20, mean_gap=25):
    import cafe
    X = R[:, :N]
    s0 = len(X) - Ttr - Tev
    agg = {k: [] for k in ("CAFE_FV_native", "DCC_meanimpute", "DCC_cafeimpute",
                           "CCC_meanimpute")}
    cover = []
    for sd in seeds:
        Rg, obs = gl.induce_async_gaps(X, gap_rate=gap_rate, mean_gap=mean_gap, seed=sd)
        cover.append(1.0 - obs.mean())
        Rtr, Rev = Rg[s0:s0 + Ttr], Rg[s0 + Ttr:s0 + Ttr + Tev]
        otr, oev = obs[s0:s0 + Ttr], obs[s0 + Ttr:s0 + Ttr + Tev]
        Rev_full = X[s0 + Ttr:s0 + Ttr + Tev]            # realized (no NaN) for scoring
        mu = np.nanmean(Rtr, 0)

        fv = mv.CafeFV(3).fit(Rtr - mu, otr)
        Sf = fv.filter_eval(Rev - mu, oev)
        agg["CAFE_FV_native"].append(gmv_annvol(Sf, Rev_full - mu)[0])

        def mean_imp(A, ref):
            A = A.copy(); col = np.nanmean(ref, 0); col = np.where(np.isfinite(col), col, 0)
            idx = np.where(np.isnan(A)); A[idx] = np.take(col, idx[1]); return A
        Rtr_m = mean_imp(Rtr, Rtr); Rev_m = mean_imp(Rev, Rtr)
        for nm, dyn in [("DCC_meanimpute", True), ("CCC_meanimpute", False)]:
            M = mv.DCCModel(dyn).fit(Rtr_m - mu); S = M.filter_eval(Rev_m - mu)
            agg[nm].append(gmv_annvol(S, Rev_full - mu)[0])

        filled = np.asarray(cafe.impute(np.vstack([Rtr, Rev])))
        Rtr_c, Rev_c = filled[:Ttr] - mu, filled[Ttr:] - mu
        M = mv.DCCModel(True).fit(Rtr_c); S = M.filter_eval(Rev_c)
        agg["DCC_cafeimpute"].append(gmv_annvol(S, Rev_full - mu)[0])
    out = {k: dict(gmv_mean=float(np.mean(v)), gmv_std=float(np.std(v)))
           for k, v in agg.items()}
    out["avg_missing_frac"] = float(np.mean(cover))
    return out


def main():
    R49 = load("ff49"); R95 = load("ff100")
    res = {}
    print(f"[A] rolling quality, N=49 (real, {R49.shape[0]} days)...")
    res["A_quality_N49"] = part_A(R49)
    a = res["A_quality_N49"]
    print(f"   ({a['_n_windows']} windows) GMV ann.vol % (lower=better) | NLL:")
    for k, v in sorted([(k, v) for k, v in a.items() if not k.startswith("_")],
                       key=lambda kv: kv[1]["gmv_mean"]):
        print(f"     {k:13s} GMV={v['gmv_mean']:6.2f}±{v['gmv_std']:.2f}  NLL={v['nll_mean']:.3f}")
    print("\n[B] curse of dimensionality on real data (short train T=350)...")
    res["B_curse_real"] = part_B(R95)
    print("\n[C] asynchronous gaps on real returns...")
    res["C_gaps_real"] = part_C(R49)
    c = res["C_gaps_real"]
    print(f"   (avg missing {c['avg_missing_frac']*100:.0f}%) GMV ann.vol %:")
    for k, v in sorted([(k, v) for k, v in c.items() if k != "avg_missing_frac"],
                       key=lambda kv: kv[1]["gmv_mean"]):
        print(f"     {k:18s} GMV={v['gmv_mean']:6.2f}±{v['gmv_std']:.2f}")
    with open(os.path.join(OUT, "exp5_realdata.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nsaved -> data/exp5_realdata.json")


if __name__ == "__main__":
    main()
