"""
Experiment 1 (H1): Is CAFE's variance machinery a competitive UNIVARIATE vol model?

Writeup claim under test: "As written, CAFE is not a better univariate GARCH. Its
scale moves too slowly (HALFLIFE=200 vs the days-to-weeks clustering needs); it has
no mean reversion; causal buys it nothing because GARCH is already point-in-time."

We simulate true GARCH(1,1) returns (we therefore KNOW the latent conditional
variance) and compare one-step-ahead variance forecasts of:
  * CAFE-as-is        : the library's robust EW residual scale^2 (HALFLIFE=200)
  * EWMA(0.94)        : RiskMetrics daily default
  * GARCH(1,1)-MLE    : the gold standard (arch), normal and Student-t
  * EWMA halflife sweep: locates the optimal forgetting and shows where 200 sits

Metrics: QLIKE (vs r^2 proxy AND vs true h), variance-MSE vs true h, 1% VaR back-test.
We also quantify the two structural defects directly: (a) lag of the scale behind a
vol spike, (b) absence of mean reversion to a long-run level.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garch_lib as gl

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(OUT, exist_ok=True)

WARM = 300        # drop warm-up rows from scoring (all methods need spin-up)


def eval_forecasts(r, h_true, forecasts, nu_for_var=None):
    rows = {}
    rr = r[WARM:]
    ht = h_true[WARM:]
    proxy = rr ** 2
    for name, (h, dist, nu) in forecasts.items():
        hh = np.asarray(h)[WARM:]
        pi, lr, pval = gl.var_backtest(rr, hh, p=0.01, dist=dist,
                                       nu=(nu if nu else 6.0))
        rows[name] = dict(
            qlike_proxy=gl.qlike(hh, proxy),
            qlike_true=gl.qlike(hh, ht),
            var_mse=gl.var_mse(hh, ht),
            var_viol=pi, var_target=0.01, kupiec_p=pval,
        )
    return rows


def run_one(dist, seed):
    # persistent, clearly mean-reverting GARCH; t-innovations stress the tails
    r, h = gl.sim_garch(4000, omega=0.05, alpha=0.08, beta=0.90,
                        dist=dist, nu=6.0, seed=seed)
    forecasts = {}
    # CAFE as-is (faithful isolation, robust, HL=200)
    forecasts["CAFE_scale_HL200"] = (gl.cafe_scale_var(r), "t", None)
    # EWMA RiskMetrics
    forecasts["EWMA_0.94"] = (gl.ewma_var(r, lam=0.94), "normal", None)
    # GARCH MLE (gold standard) -- normal and Student-t
    hg, _ = gl.arch_garch_var(r, dist="normal")
    forecasts["GARCH_normal_MLE"] = (hg, "normal", None)
    try:
        ht_, rest = gl.arch_garch_var(r, dist="t")
        nu_t = float(rest.params.get("nu", 6.0))
        forecasts["GARCH_t_MLE"] = (ht_, "t", nu_t)
    except Exception:
        pass
    res = eval_forecasts(r, h, forecasts)
    return res


def halflife_sweep(dist="t", seeds=(0, 1, 2, 3, 4)):
    """QLIKE(true) of the robust EW scale as a function of half-life -> shows the
    optimum is ~10-30 and HL=200 is far up the wrong side of the curve."""
    hls = [3, 5, 8, 11, 16, 22, 30, 45, 70, 110, 160, 200, 300, 500]
    agg = {hl: [] for hl in hls}
    for seed in seeds:
        r, h = gl.sim_garch(4000, dist=dist, nu=6.0, seed=seed)
        for hl in hls:
            hh = gl.cafe_scale_var(r, halflife=hl)
            agg[hl].append(gl.qlike(hh[WARM:], h[WARM:]))
    out = {hl: float(np.mean(v)) for hl, v in agg.items()}
    best_hl = min(out, key=out.get)
    return out, best_hl


def spike_response(seed=7):
    """Quantify lag + no-mean-reversion: drop a deterministic vol regime (low->high
    ->low) into the variance path and measure how fast each forecast reacts/reverts."""
    rng = np.random.default_rng(seed)
    T = 1500
    h = np.full(T, 0.5)
    h[500:600] = 5.0                      # sharp 100-step high-vol burst
    z = rng.standard_normal(T)
    r = np.sqrt(h) * z
    cafe = gl.cafe_scale_var(r)
    ewma = gl.ewma_var(r, 0.94)
    # time for each to cover half the up-jump after t=500, and to revert after t=600
    def react_time(series, t0, target_frac=0.5, up=True):
        base = series[t0 - 1]
        tgt = base + target_frac * (h[t0] - base) if up else \
              base - target_frac * (base - h[t0 + 50])
        for k in range(t0, min(t0 + 400, T)):
            if (up and series[k] >= tgt) or (not up and series[k] <= tgt):
                return k - t0
        return None
    return dict(
        cafe_rise=react_time(cafe, 500, up=True),
        ewma_rise=react_time(ewma, 500, up=True),
        cafe_revert=react_time(cafe, 600, up=False),
        ewma_revert=react_time(ewma, 600, up=False),
        h=h.tolist(), r=r.tolist(), cafe=cafe.tolist(), ewma=ewma.tolist(),
    )


def main():
    t0 = time.time()
    results = {"dists": {}, "halflife_sweep": {}, "spike": {}}
    for dist in ("normal", "t"):
        seeds = range(8)
        per = [run_one(dist, s) for s in seeds]
        names = per[0].keys()
        agg = {}
        for name in names:
            agg[name] = {}
            for metric in per[0][name]:
                vals = [p[name][metric] for p in per if name in p]
                agg[name][metric + "_mean"] = float(np.mean(vals))
                agg[name][metric + "_std"] = float(np.std(vals))
        results["dists"][dist] = agg
    for dist in ("normal", "t"):
        sweep, best = halflife_sweep(dist=dist)
        results["halflife_sweep"][dist] = dict(curve=sweep, best_hl=best,
                                               cafe_hl=gl.HALFLIFE)
    results["spike"] = spike_response()
    with open(os.path.join(OUT, "exp1_univariate.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ----
    print(f"\n=== EXP1 univariate (elapsed {time.time()-t0:.1f}s) ===")
    for dist in ("normal", "t"):
        print(f"\n--- GARCH({dist}) innovations, QLIKE(true h) lower=better ---")
        agg = results["dists"][dist]
        base = agg["GARCH_t_MLE"]["qlike_true_mean"] if "GARCH_t_MLE" in agg \
            else agg["GARCH_normal_MLE"]["qlike_true_mean"]
        for name, m in sorted(agg.items(), key=lambda kv: kv[1]["qlike_true_mean"]):
            gap = m["qlike_true_mean"] - base
            print(f"  {name:20s} QLIKE_true={m['qlike_true_mean']:.4f} "
                  f"(+{gap:+.4f})  VaR1%_viol={m['var_viol_mean']:.3f} "
                  f"varMSE={m['var_mse_mean']:.3f}")
    for dist in ("normal", "t"):
        sw = results["halflife_sweep"][dist]
        print(f"\nHalf-life sweep ({dist}): optimal HL={sw['best_hl']}, "
              f"CAFE uses HL={sw['cafe_hl']:.0f}")
        print("  HL :", "  ".join(f"{int(k)}" for k in sw["curve"]))
        print("  QL :", "  ".join(f"{v:.3f}" for v in sw["curve"].values()))
    sp = results["spike"]
    print(f"\nSpike response (steps to react): CAFE rise={sp['cafe_rise']} vs "
          f"EWMA rise={sp['ewma_rise']}; CAFE revert={sp['cafe_revert']} vs "
          f"EWMA revert={sp['ewma_revert']}")
    print("\nsaved -> data/exp1_univariate.json")


if __name__ == "__main__":
    main()
