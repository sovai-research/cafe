"""CAFE adaptation figure: the model self-tunes its dials online.

Single-column. Reads the REAL c_unified_penmf trace via viz_common:
  nu   -- Student-t degrees of freedom (robustness / tail dial)  [left y]
  a    -- AR(1) coefficient (memory dial)                         [right twin y]
  eff  -- effective rank: per-row count of active (below-median) ARD columns
Everything is read from the running estimator; no tuned constants.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import viz_common as V

P = V.PALETTE

# --- run the real model ------------------------------------------------------
Xobs, Xtrue, M, parts = V.demo_series(T=500, outlier_frac=0.02, seed=8)
filled, trace, core = V.run_traced(Xobs)
C = V.components(trace, Xobs.shape[1])

t = np.arange(len(trace))
nu = C["nu"]
a = C["a"]
alpha = C["alpha"]                                   # (T, R) ARD precisions
# effective rank = active factors per row = columns with precision below the
# row median (low precision == relaxed prior == factor in use).
med = np.median(alpha, axis=1, keepdims=True)
eff = (alpha < med).sum(axis=1)

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.5))

# left axis: nu (robustness dial)
l_nu, = ax.plot(t, nu, color=P["red"], lw=1.3, label=r"$\nu$  tails (dof)")
ax.set_xlabel("time  $t$", fontsize=9)
ax.set_ylabel(r"Student-$t$ dof  $\nu$", color=P["red"], fontsize=9)
ax.tick_params(axis="y", labelcolor=P["red"])
V.style_ax(ax)

# right twin axis: a (memory dial)
ax_a = ax.twinx()
l_a, = ax_a.plot(t, a, color=P["blue"], lw=1.3, label=r"$a$  memory (AR)")
ax_a.set_ylabel(r"AR coef.  $a$", color=P["blue"], fontsize=9)
ax_a.tick_params(axis="y", labelcolor=P["blue"], labelsize=8)
ax_a.spines["top"].set_visible(False)

# effective rank as a step trace, scaled onto the AR axis for shared display
amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
emax = max(int(eff.max()), 1)
eff_scaled = amin + (eff / emax) * (amax - amin)
l_r, = ax_a.step(t, eff_scaled, where="post", color=P["teal"], lw=1.3,
                 alpha=0.85, label=r"eff. rank ($\times$%d)" % emax)

# legend combining all three traces
lines = [l_nu, l_a, l_r]
ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.5,
          loc="center right", frameon=False)

ax.set_title("CAFE learns its own dials online:\ntails ($\\nu$), memory ($a$), rank",
             fontsize=9)

fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figures", "adaptation.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("saved", out, "nu range", nu.min(), nu.max(),
      "a range", a.min(), a.max(), "eff range", eff.min(), eff.max())
