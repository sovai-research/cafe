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
# Two stacked panels: continuous dials (nu, a) on top with twin axes; the
# integer effective rank on its own panel below (avoids misleading rescaling
# and the overlap between the rank step and the AR line).
fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(3.4, 3.0), sharex=True,
    gridspec_kw=dict(height_ratios=[2.4, 1.0]), constrained_layout=True)

# --- top panel: nu (left) and a (right twin) -------------------------------
l_nu, = ax.plot(t, nu, color=P["red"], lw=1.3, label=r"$\nu$  tails (dof)")
ax.set_ylabel(r"Student-$t$ dof  $\nu$", color=P["red"], fontsize=9)
ax.tick_params(axis="y", labelcolor=P["red"])
V.style_ax(ax)

ax_a = ax.twinx()
l_a, = ax_a.plot(t, a, color=P["blue"], lw=1.3, label=r"$a$  memory (AR)")
ax_a.set_ylabel(r"AR coef.  $a$", color=P["blue"], fontsize=9)
ax_a.tick_params(axis="y", labelcolor=P["blue"], labelsize=8)
ax_a.spines["top"].set_visible(False)

# compact stacked legend in the empty mid-right band (there nu has fallen to
# ~4.2 and a sits near 0.8, leaving the mid-height clear), so it covers no data
lines = [l_nu, l_a]
ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.5,
          loc="center right", bbox_to_anchor=(0.995, 0.62),
          frameon=True, framealpha=0.9, ncol=1, handlelength=1.5,
          borderpad=0.35, handletextpad=0.5, labelspacing=0.35)

# --- bottom panel: integer effective rank ----------------------------------
axr.step(t, eff, where="post", color=P["teal"], lw=1.3, alpha=0.9)
axr.fill_between(t, eff, step="post", color=P["teal"], alpha=0.12)
axr.set_ylabel("eff.\nrank", color=P["teal"], fontsize=9)
axr.tick_params(axis="y", labelcolor=P["teal"])
axr.set_xlabel("time  $t$", fontsize=9)
emax = max(int(eff.max()), 1)
axr.set_ylim(0, emax + 0.5)
axr.set_yticks(range(0, emax + 1))
V.style_ax(axr)

fig.suptitle("CAFÉ learns its own dials online:\ntails ($\\nu$), memory ($a$), rank",
             fontsize=9)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "figures", "adaptation.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
print("saved", out, "nu range", nu.min(), nu.max(),
      "a range", a.min(), a.max(), "eff range", eff.min(), eff.max())
