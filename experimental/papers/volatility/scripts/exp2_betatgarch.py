"""
Experiment 2 (H4): Does the Beta-t-GARCH upgrade turn CAFE's machinery into a
competitive -- and robust -- volatility model, using parts CAFE already computes?

Writeup claim under test: "Replace the fixed-decay EW scale with a score-driven
(GAS/DCS) recursion -- Beta-t-GARCH. Its variance innovation is the Student-t score
w*r^2 -- EXACTLY the IRLS weight CAFE already computes. You add only a learned
reaction coefficient, persistence and a long-run level."

Three tests:
  (A) IDENTITY: the Beta-t innovation equals CAFE's own IRLS weight times r^2,
      verified against the library's _UnifiedCore._wt -- the upgrade really is built
      from existing machinery (max|dev| reported).
  (B) CLEAN: on true GARCH-t data, Beta-t-GARCH ~ GARCH-MLE and CRUSHES CAFE-as-is
      -- the slow HL=200 scale is fixed by swapping in the recursion.
  (C) ROBUST: with additive outliers (fat-finger prints NOT in the vol process), the
      Beta-t score self-down-weights them; Gaussian GARCH-MLE over-reacts. We score
      against the TRUE latent variance (which excludes the injected jumps).
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garch_lib as gl

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(OUT, exist_ok=True)
WARM = 300


def check_identity(seed=0):
    """Confirm Beta-t innovation u_t = w_t r_t^2 uses CAFE's exact IRLS weight."""
    from cafe._core import _UnifiedCore
    core = _UnifiedCore(1, periods=[], E=1)
    r, _ = gl.sim_garch(2000, dist="t", nu=6.0, seed=seed)
    omega, alpha, beta, nu = 0.05, 0.05, 0.92, 6.0
    h = gl.betat_garch_filter(r, omega, alpha, beta, nu)
    devs = []
    for t in range(len(r)):
        ht = max(h[t], 1e-9)
        # the library's own weight, with scale2 set to this step's h_t and nu
        core.scale2 = ht; core.nu = nu
        w_lib = float(core._wt(np.array([r[t] ** 2]))[0])
        u_lib = w_lib * r[t] ** 2
        w_loc = (nu + 1.0) / (nu + r[t] ** 2 / ht)
        u_loc = w_loc * r[t] ** 2
        devs.append(abs(u_lib - u_loc))
    return float(np.max(devs))


def clean_test(seeds=range(8)):
    rows = {k: [] for k in ("BetaT_GARCH", "GARCH_t_MLE", "CAFE_scale_HL200")}
    for s in seeds:
        r, h = gl.sim_garch(4000, omega=0.05, alpha=0.08, beta=0.90,
                            dist="t", nu=6.0, seed=s)
        fit = gl.fit_betat_garch(r)
        hg, rest = gl.arch_garch_var(r, dist="t")
        hc = gl.cafe_scale_var(r)
        ht = h[WARM:]
        rows["BetaT_GARCH"].append(gl.qlike(fit["h"][WARM:], ht))
        rows["GARCH_t_MLE"].append(gl.qlike(hg[WARM:], ht))
        rows["CAFE_scale_HL200"].append(gl.qlike(hc[WARM:], ht))
    return {k: dict(qlike_true_mean=float(np.mean(v)),
                    qlike_true_std=float(np.std(v))) for k, v in rows.items()}


def robust_test(seeds=range(8), rate=0.01, scale=8.0):
    """Additive outliers: score every model against the TRUE latent h (which the
    injected jumps are not part of). A robust vol model should ignore the jumps."""
    rows = {k: [] for k in ("BetaT_GARCH", "GARCH_normal_MLE", "GARCH_t_MLE",
                            "CAFE_scale_HL200")}
    spike_inflation = {k: [] for k in ("BetaT_GARCH", "GARCH_normal_MLE")}
    for s in seeds:
        r, h = gl.sim_garch(4000, omega=0.05, alpha=0.08, beta=0.90,
                            dist="t", nu=6.0, seed=s)
        rc, mask = gl.add_additive_outliers(r, rate=rate, scale=scale, seed=100 + s)
        fit = gl.fit_betat_garch(rc)
        hgn, _ = gl.arch_garch_var(rc, dist="normal")
        hgt, _ = gl.arch_garch_var(rc, dist="t")
        hc = gl.cafe_scale_var(rc)
        ht = h[WARM:]
        rows["BetaT_GARCH"].append(gl.qlike(fit["h"][WARM:], ht))
        rows["GARCH_normal_MLE"].append(gl.qlike(hgn[WARM:], ht))
        rows["GARCH_t_MLE"].append(gl.qlike(hgt[WARM:], ht))
        rows["CAFE_scale_HL200"].append(gl.qlike(hc[WARM:], ht))
        # how much does the forecast over-shoot the TRUE variance right after a jump?
        idx = np.where(mask)[0]
        idx = idx[(idx > WARM) & (idx < len(r) - 2)]
        nxt = idx + 1
        spike_inflation["BetaT_GARCH"].append(
            float(np.median(fit["h"][nxt] / h[nxt])))
        spike_inflation["GARCH_normal_MLE"].append(
            float(np.median(hgn[nxt] / h[nxt])))
    out = {k: dict(qlike_true_mean=float(np.mean(v)),
                   qlike_true_std=float(np.std(v))) for k, v in rows.items()}
    infl = {k: float(np.mean(v)) for k, v in spike_inflation.items()}
    return out, infl


def spike_response():
    """Same deterministic vol burst as exp1 -- show the Beta-t recursion reacts/
    reverts FAST where CAFE-as-is (HL=200) could not."""
    rng = np.random.default_rng(7)
    T = 1500
    h = np.full(T, 0.5); h[500:600] = 5.0
    r = np.sqrt(h) * rng.standard_normal(T)
    fit = gl.fit_betat_garch(r)
    cafe = gl.cafe_scale_var(r)
    hb = fit["h"]

    def react_time(series, t0, up=True):
        base = series[t0 - 1]
        tgt = base + 0.5 * (h[t0] - base) if up else base - 0.5 * (base - h[t0 + 50])
        for k in range(t0, min(t0 + 400, T)):
            if (up and series[k] >= tgt) or (not up and series[k] <= tgt):
                return k - t0
        return None
    return dict(betat_rise=react_time(hb, 500, True),
                cafe_rise=react_time(cafe, 500, True),
                betat_revert=react_time(hb, 600, False),
                cafe_revert=react_time(cafe, 600, False),
                params=dict(omega=fit["omega"], alpha=fit["alpha"],
                            beta=fit["beta"], nu=fit["nu"]),
                h=h.tolist(), betat=hb.tolist(), cafe=cafe.tolist())


def main():
    t0 = time.time()
    res = {}
    res["identity_max_dev"] = check_identity()
    res["clean"] = clean_test()
    rob, infl = robust_test()
    res["robust"] = rob
    res["robust_spike_inflation"] = infl
    res["spike"] = spike_response()
    with open(os.path.join(OUT, "exp2_betatgarch.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n=== EXP2 Beta-t-GARCH (elapsed {time.time()-t0:.1f}s) ===")
    print(f"\n(A) Innovation identity  max|u_lib - w*r^2| = {res['identity_max_dev']:.2e}"
          "  (Beta-t innovation IS CAFE's IRLS weight)")
    print("\n(B) CLEAN GARCH-t data, QLIKE(true h) lower=better:")
    for k, v in sorted(res["clean"].items(), key=lambda kv: kv[1]["qlike_true_mean"]):
        print(f"    {k:20s} {v['qlike_true_mean']:.4f} +/- {v['qlike_true_std']:.4f}")
    print("\n(C) ADDITIVE-OUTLIER data, QLIKE(true h) lower=better:")
    for k, v in sorted(res["robust"].items(), key=lambda kv: kv[1]["qlike_true_mean"]):
        print(f"    {k:20s} {v['qlike_true_mean']:.4f} +/- {v['qlike_true_std']:.4f}")
    print("    post-jump variance over-shoot (forecast/true, 1=perfect):")
    for k, v in res["robust_spike_inflation"].items():
        print(f"      {k:20s} x{v:.2f}")
    sp = res["spike"]
    print(f"\n    Spike reaction: Beta-t rise={sp['betat_rise']} revert={sp['betat_revert']}"
          f"  vs  CAFE-as-is rise={sp['cafe_rise']} revert={sp['cafe_revert']}")
    print(f"    fitted Beta-t params: {sp['params']}")
    print("\nsaved -> data/exp2_betatgarch.json")


if __name__ == "__main__":
    main()
