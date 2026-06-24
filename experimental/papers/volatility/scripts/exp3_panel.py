"""
Experiment 3 (H3): The real niche -- scalable factor volatility on HIGH-DIMENSIONAL
and INCOMPLETE panels, where classical multivariate GARCH (DCC/BEKK) struggles.

Writeup claim under test: "Classic multivariate GARCH dies of the curse of
dimensionality past a handful of series; GARCH chokes on gaps and non-synchronous
trading. CAFE already produces a low-rank-plus-diagonal conditional covariance,
scalable to hundreds of assets essentially for free, and handles ragged panels
natively. That combination -- scalable factor volatility on incomplete panels -- is
a real gap in the literature."

Four sub-tests on a factor-GARCH panel (true Sigma_t known):
  A. QUALITY (complete data, N=20, multi-seed): minimum-variance-portfolio realized
     variance + covariance QLIKE for CAFE-FV vs DCC / CCC / EWMA / Sample.
  B. CURSE OF DIMENSIONALITY: short training window, N swept up toward/over T. DCC/CCC
     rely on an N-by-N sample correlation that degrades as N -> T; CAFE-FV (rank K)
     stays well-conditioned. Track MV-portfolio variance + correlation condition number.
  C. GAPS: asynchronous block missingness. CAFE-FV ingests it natively; DCC/CCC need a
     complete-data front-end (naive mean-impute vs CAFE-impute). Compare MV variance.
  D. RUNTIME: fit + per-step filter wall-clock vs N (the O(N*K) vs O(N^2) story).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garch_lib as gl
import mvol_lib as mv

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(OUT, exist_ok=True)


def _score(M, Rtr, Rev, Sev, obs_tr=None, obs_ev=None):
    t = time.time()
    M.fit(Rtr, obs_tr) if obs_tr is not None else M.fit(Rtr)
    S = M.filter_eval(Rev, obs_ev) if obs_ev is not None else M.filter_eval(Rev)
    el = time.time() - t
    rv, ok = mv.mv_portfolio_realized_var(S, Rev)
    ql = mv.cov_qlike(S, Sev)
    return dict(mvport=rv, covqlike=ql, ok=ok, sec=el)


# --------------------------------------------------------------------------- #
def part_A_quality(seeds=range(6), N=20, K=3, Ttr=1200, Tev=800):
    models = lambda: {"CAFE_FV": mv.CafeFV(K), "DCC": mv.DCCModel(True),
                      "CCC": mv.DCCModel(False), "EWMA": mv.EWMACov(),
                      "Sample": mv.SampleCov()}
    agg = {k: {"mvport": [], "covqlike": []} for k in models()}
    oracle = []
    for s in seeds:
        R, Sig, _ = gl.sim_factor_garch_panel(Ttr + Tev, N=N, K=K, seed=s)
        Rtr, Rev, Sev = R[:Ttr], R[Ttr:], Sig[Ttr:]
        for name, M in models().items():
            r = _score(M, Rtr, Rev, Sev)
            agg[name]["mvport"].append(r["mvport"])
            agg[name]["covqlike"].append(r["covqlike"])
        ov, _ = mv.mv_portfolio_realized_var(Sev, Rev)
        oracle.append(ov)
    out = {k: dict(mvport_mean=float(np.mean(v["mvport"])),
                   mvport_std=float(np.std(v["mvport"])),
                   covqlike_mean=float(np.mean(v["covqlike"])))
           for k, v in agg.items()}
    out["ORACLE"] = dict(mvport_mean=float(np.mean(oracle)))
    return out


def part_B_curse(Ns=(10, 20, 40, 80, 150, 250), K=3, Ttr=350, Tev=400, seed=0):
    rows = {}
    for N in Ns:
        R, Sig, _ = gl.sim_factor_garch_panel(Ttr + Tev, N=N, K=K, seed=seed)
        Rtr, Rev, Sev = R[:Ttr], R[Ttr:], Sig[Ttr:]
        # condition number of the training sample correlation (DCC/CCC backbone)
        Cr = np.corrcoef(Rtr, rowvar=False)
        cond = float(np.linalg.cond(Cr))
        entry = dict(N=N, Ttr=Ttr, corr_cond=cond)
        fv = _score(mv.CafeFV(K), Rtr, Rev, Sev)
        entry["CAFE_FV"] = dict(mvport=fv["mvport"], ok=fv["ok"])
        cc = _score(mv.DCCModel(False), Rtr, Rev, Sev)
        entry["CCC"] = dict(mvport=cc["mvport"], ok=cc["ok"])
        if N <= 80:                              # DCC param fit is O(N^2)/step -> cap it
            dc = _score(mv.DCCModel(True), Rtr, Rev, Sev)
            entry["DCC"] = dict(mvport=dc["mvport"], ok=dc["ok"])
        else:
            entry["DCC"] = dict(mvport=None, ok=0.0, note="infeasible at this N (cost)")
        sa = _score(mv.SampleCov(), Rtr, Rev, Sev)
        entry["Sample"] = dict(mvport=sa["mvport"], ok=sa["ok"])
        ov, _ = mv.mv_portfolio_realized_var(Sev, Rev)
        entry["ORACLE"] = dict(mvport=float(ov))
        rows[N] = entry
        print(f"  [B] N={N:4d} cond(corr)={cond:9.1f}  "
              f"CAFE_FV={fv['mvport']:.4f}  CCC={cc['mvport']:.4f}  "
              f"DCC={entry['DCC']['mvport']}")
    return rows


def part_C_gaps(seeds=range(4), N=30, K=3, Ttr=1100, Tev=700,
                gap_rate=0.20, mean_gap=25):
    import cafe
    agg = {k: [] for k in ("CAFE_FV_native", "DCC_meanimpute", "DCC_cafeimpute",
                           "CCC_meanimpute")}
    cover = []
    for s in seeds:
        R, Sig, _ = gl.sim_factor_garch_panel(Ttr + Tev, N=N, K=K, seed=s)
        Rg, obs = gl.induce_async_gaps(R, gap_rate=gap_rate, mean_gap=mean_gap, seed=s)
        cover.append(1.0 - obs.mean())
        Rtr, Rev = Rg[:Ttr], Rg[Ttr:]
        otr, oev = obs[:Ttr], obs[Ttr:]
        Sev = Sig[Ttr:]
        Rev_full = R[Ttr:]                       # realized returns for scoring (no NaN)

        # CAFE-FV native (gap-aware fit + filter)
        fv = mv.CafeFV(K)
        fv.fit(Rtr, otr); Sf = fv.filter_eval(Rev, oev)
        agg["CAFE_FV_native"].append(mv.mv_portfolio_realized_var(Sf, Rev_full)[0])

        # front-end 1: naive mean (rolling) imputation, then DCC/CCC on complete data
        Rtr_m = _mean_impute(Rtr); Rev_m = _mean_impute(Rev, ref=Rtr_m)
        dccm = mv.DCCModel(True); dccm.fit(Rtr_m); Sdm = dccm.filter_eval(Rev_m)
        agg["DCC_meanimpute"].append(mv.mv_portfolio_realized_var(Sdm, Rev_full)[0])
        cccm = mv.DCCModel(False); cccm.fit(Rtr_m); Scm = cccm.filter_eval(Rev_m)
        agg["CCC_meanimpute"].append(mv.mv_portfolio_realized_var(Scm, Rev_full)[0])

        # front-end 2: CAFE imputation (causal), then DCC -- CAFE as a gap front-end
        full = np.vstack([Rtr, Rev])
        filled = np.asarray(cafe.impute(full))
        Rtr_c, Rev_c = filled[:Ttr], filled[Ttr:]
        dccc = mv.DCCModel(True); dccc.fit(Rtr_c); Sdc = dccc.filter_eval(Rev_c)
        agg["DCC_cafeimpute"].append(mv.mv_portfolio_realized_var(Sdc, Rev_full)[0])

    out = {k: dict(mvport_mean=float(np.mean(v)), mvport_std=float(np.std(v)))
           for k, v in agg.items()}
    out["avg_missing_frac"] = float(np.mean(cover))
    return out


def _mean_impute(X, ref=None):
    """Causal-ish fill: replace NaN with the column mean of the reference block
    (the training block, or the array itself). Standard naive MGARCH front-end."""
    X = X.copy()
    src = ref if ref is not None else X
    col = np.nanmean(np.where(np.isnan(src), np.nan, src), axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col, idx[1])
    return X


def part_D_runtime(Ns=(20, 50, 100, 200), K=3, Ttr=600, Tev=300, seed=0):
    rows = {}
    for N in Ns:
        R, Sig, _ = gl.sim_factor_garch_panel(Ttr + Tev, N=N, K=K, seed=seed)
        Rtr, Rev = R[:Ttr], R[Ttr:]
        ent = {}
        for name, M in [("CAFE_FV", mv.CafeFV(K)), ("CCC", mv.DCCModel(False))]:
            t = time.time(); M.fit(Rtr); fit = time.time() - t
            t = time.time(); M.filter_eval(Rev); filt = time.time() - t
            ent[name] = dict(fit_sec=fit, filter_sec=filt,
                             filter_per_step_ms=1000 * filt / Tev)
        if N <= 100:
            M = mv.DCCModel(True)
            t = time.time(); M.fit(Rtr); fit = time.time() - t
            t = time.time(); M.filter_eval(Rev); filt = time.time() - t
            ent["DCC"] = dict(fit_sec=fit, filter_sec=filt,
                              filter_per_step_ms=1000 * filt / Tev)
        else:
            ent["DCC"] = dict(fit_sec=None, note="not run (cost)")
        rows[N] = ent
        print(f"  [D] N={N:4d}  CAFE_FV fit={ent['CAFE_FV']['fit_sec']:.2f}s "
              f"filt/step={ent['CAFE_FV']['filter_per_step_ms']:.2f}ms | "
              f"CCC fit={ent['CCC']['fit_sec']:.2f}s | "
              f"DCC fit={ent['DCC'].get('fit_sec')}")
    return rows


def main():
    t0 = time.time()
    res = {}
    print("\n[A] complete-data quality (N=20, 6 seeds)...")
    res["A_quality"] = part_A_quality()
    print("\n[B] curse of dimensionality (sweep N)...")
    res["B_curse"] = part_B_curse()
    print("\n[C] asynchronous gaps...")
    res["C_gaps"] = part_C_gaps()
    print("\n[D] runtime scaling...")
    res["D_runtime"] = part_D_runtime()
    with open(os.path.join(OUT, "exp3_panel.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== EXP3 panel summary (elapsed {time.time()-t0:.1f}s) ===")
    A = res["A_quality"]
    print("\n[A] Complete data, MV-portfolio realized variance (lower=better; "
          f"oracle={A['ORACLE']['mvport_mean']:.4f}):")
    for k, v in sorted([(k, v) for k, v in A.items() if k != "ORACLE"],
                       key=lambda kv: kv[1]["mvport_mean"]):
        print(f"    {k:10s} MVport={v['mvport_mean']:.4f} +/- {v['mvport_std']:.4f}"
              f"   covQLIKE={v['covqlike_mean']:.3f}")
    print("\n[C] Gaps (avg missing "
          f"{res['C_gaps']['avg_missing_frac']*100:.0f}%), MV-portfolio variance:")
    for k, v in sorted([(k, v) for k, v in res["C_gaps"].items()
                        if k != "avg_missing_frac"],
                       key=lambda kv: kv[1]["mvport_mean"]):
        print(f"    {k:18s} MVport={v['mvport_mean']:.4f} +/- {v['mvport_std']:.4f}")
    print("\nsaved -> data/exp3_panel.json")


if __name__ == "__main__":
    main()
