"""Gap-theory validation for the position paper: closed form Delta(a,g) vs exact
Kalman-filter/RTS-smoother Monte-Carlo, plus real-panel a-hat band. Pure numpy.
Writes JSON + a self-contained figure into THIS PAPER'S DIR.

  cd /Users/dereksnow/Sovai/Github/TIMARA
  python3 experimental/papers/position/scripts/run_gap_theory.py
"""
import os, sys, json, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/dereksnow/Sovai/Github/TIMARA"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "data"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))


def Vf(a, k, s2=1.0):
    return s2 * (1.0 - a ** (2 * k))


def Vs(a, k, g, s2=1.0):
    num = (1.0 - a ** (2 * k)) * (1.0 - a ** (2 * (g + 1 - k)))
    den = (1.0 - a ** (2 * (g + 1)))
    return s2 * num / den


def delta_pred(a, g, s2=1.0):
    if g < 1:
        return 0.0
    ks = np.arange(1, g + 1)
    return float(np.sqrt(2.0 / np.pi) * np.mean(np.sqrt(Vf(a, ks, s2)) - np.sqrt(Vs(a, ks, g, s2))))


def filter_pred_gap(a, x0, g):
    ks = np.arange(1, g + 1)
    return (a ** ks) * x0


def smoother_pred_gap(a, x0, xR, g):
    ks = np.arange(1, g + 1)
    den = 1.0 - a ** (2 * (g + 1))
    if den <= 0:
        return 0.5 * (x0 + xR) * np.ones(g)
    w0 = a ** ks * (1.0 - a ** (2 * (g + 1 - ks))) / den
    wR = a ** (g + 1 - ks) * (1.0 - a ** (2 * ks)) / den
    return w0 * x0 + wR * xR


def measure_delta(a, g, ntr=20000, s2=1.0, seed=0):
    rng = np.random.default_rng(seed)
    L = g + 2
    q = s2 * (1.0 - a * a) if a < 1 else s2 * 1e-6
    x = np.empty((ntr, L))
    x[:, 0] = rng.normal(0.0, np.sqrt(s2), ntr)
    sw = np.sqrt(max(q, 0.0))
    for t in range(1, L):
        x[:, t] = a * x[:, t - 1] + rng.normal(0.0, sw, ntr)
    af = np.zeros(ntr); as_ = np.zeros(ntr)
    for i in range(ntr):
        truth = x[i, 1:g + 1]
        af[i] = np.mean(np.abs(filter_pred_gap(a, x[i, 0], g) - truth))
        as_[i] = np.mean(np.abs(smoother_pred_gap(a, x[i, 0], x[i, g + 1], g) - truth))
    return af.mean() - as_.mean(), af.mean(), as_.mean()


def estimate_ar1(col):
    c = col[np.isfinite(col)]
    if c.size < 10:
        return np.nan
    c = c - c.mean()
    d = np.dot(c, c)
    if d <= 0:
        return np.nan
    return float(np.clip(np.dot(c[1:], c[:-1]) / d, 0.0, 0.999))


def dataset_ar(name):
    p = os.path.join(ROOT, "data", f"{name}_clean.npy")
    if not os.path.exists(p):
        return None
    X = np.load(p)
    if X.ndim != 2:
        return None
    aas = [estimate_ar1(X[:, j]) for j in range(min(X.shape[1], 200))]
    aas = [a for a in aas if np.isfinite(a)]
    return float(np.median(aas)) if aas else None


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    A_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    G_grid = [1, 2, 3, 5, 8, 12, 20, 32]
    cells, preds, meas = [], [], []
    print(f"{'a':>5} {'g':>4} {'Dpred':>9} {'Dmeas':>9} {'rel.err':>9}")
    for a in A_grid:
        for g in G_grid:
            dp = delta_pred(a, g)
            dm, _, _ = measure_delta(a, g, ntr=12000, seed=0)
            rel = abs(dp - dm) / (abs(dm) + 1e-9)
            cells.append({"a": a, "g": g, "Dpred": dp, "Dmeas": dm, "rel": rel})
            preds.append(dp); meas.append(dm)
            print(f"{a:5.2f} {g:4d} {dp:9.4f} {dm:9.4f} {rel:9.3f}")
    preds = np.array(preds); meas = np.array(meas)
    ss_res = float(np.sum((meas - preds) ** 2))
    ss_tot = float(np.sum((meas - meas.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rels = np.array([c["rel"] for c in cells if c["Dmeas"] > 1e-3])
    medrel = float(np.median(rels)) if rels.size else float("nan")
    print(f"\nR^2(pred,meas)={r2:.4f}  median rel.err (Dmeas>1e-3)={medrel:.3f}")

    datasets = ["fredmd", "beijing", "airquality", "appliances", "solar",
                "traffic2", "etth", "electric", "exchange"]
    ar_band = {}
    print(f"\n{'dataset':>12} {'a-hat':>7} {'Dpred(g=3)':>11} {'Dpred(g=10)':>12}")
    for nm in datasets:
        ah = dataset_ar(nm)
        if ah is None:
            continue
        d3, d10 = delta_pred(ah, 3), delta_pred(ah, 10)
        ar_band[nm] = {"a_hat": ah, "Dpred_g3": d3, "Dpred_g10": d10}
        print(f"{nm:>12} {ah:7.3f} {d3:11.4f} {d10:12.4f}")

    out = {"cells": cells, "r2": r2, "median_rel_err": medrel,
           "ar_band": ar_band, "A_grid": A_grid, "G_grid": G_grid}
    with open(os.path.join(OUT, "gap_theory.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- figure: (a) curves per a; (b) measured vs predicted scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(A_grid)))
    for a, c in zip(A_grid, cmap):
        gs = np.array(G_grid)
        ax.plot(gs, [delta_pred(a, g) for g in gs], color=c, lw=1.6, label=f"a={a}")
        dm = [measure_delta(a, g, ntr=8000, seed=1)[0] for g in gs]
        ax.scatter(gs, dm, color=c, s=14, zorder=3)
    ax.set_xlabel("gap length $g$"); ax.set_ylabel(r"look-ahead gap $\Delta(a,g)$")
    ax.set_title("(a) closed form (lines) vs exact filter/smoother (points)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(preds, meas, s=18, color="#2C7A7B", edgecolor="black", linewidth=0.3)
    lim = max(preds.max(), meas.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel(r"predicted $\Delta$ (closed form)")
    ax.set_ylabel(r"measured $\Delta$ (exact runs)")
    ax.set_title(f"(b) all $(a,g)$ cells: $R^2={r2:.3f}$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gap_theory.pdf"), bbox_inches="tight")
    print(f"\n[wrote] {os.path.join(FIG, 'gap_theory.pdf')}")
    print(f"[wrote] {os.path.join(OUT, 'gap_theory.json')}")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
