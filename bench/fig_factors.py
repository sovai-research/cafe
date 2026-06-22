"""Figure: latent common factors + ARD rank selection (single-column).

Reads the REAL c_unified_penmf trace via viz_common: the latent factor paths
C["Z"] (T,R) and the per-step ARD precisions C["alpha"] (T,R). Factors that ARD
has pruned have a large final precision (-> tiny contribution, flat near zero);
the surviving few carry the signal. Nothing here is hand-drawn.
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import viz_common as V

PAL = V.PALETTE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "paper", "figures", "factors.pdf")

# --- real model run -------------------------------------------------------
Xo, Xt, M, parts = V.demo_series(rank=2, N=10, seed=4)
filled, trace, core = V.run_traced(Xo)
C = V.components(trace, Xo.shape[1])
Z = C["Z"]                       # (T, R) latent factor paths
alpha_final = C["alpha"][-1]     # (R,) final ARD precisions
T, R = Z.shape
t = np.arange(T)

# ARD prunes by driving alpha (precision) large; active factors keep alpha small.
# Use the log-gap in the final precisions to separate "active" from "killed".
order = np.argsort(alpha_final)
la = np.log10(alpha_final + 1e-12)
gaps = np.diff(la[order])
split = int(np.argmax(gaps)) + 1 if R > 1 else R   # boundary in sorted order
active_mask = np.zeros(R, dtype=bool)
active_mask[order[:split]] = True
eff_rank = int(active_mask.sum())

# --- plot -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.5))
active_cols = [PAL["blue"], PAL["amber"], PAL["teal"], PAL["green"],
               PAL["purple"], PAL["red"]]
ci = 0
# draw killed first (grey, thin, behind), then active on top
for r in range(R):
    if not active_mask[r]:
        ax.plot(t, Z[:, r], color=PAL["grey"], lw=0.8, alpha=0.7, zorder=1)
for r in range(R):
    if active_mask[r]:
        col = active_cols[ci % len(active_cols)]; ci += 1
        ax.plot(t, Z[:, r], color=col, lw=1.3, zorder=3,
                label=f"factor {ci} (alpha={alpha_final[r]:.2f})")

# one grey legend proxy for the pruned bundle
ax.plot([], [], color=PAL["grey"], lw=0.8,
        label=f"pruned x{R - eff_rank} (alpha={alpha_final[~active_mask].min():.0f}+)")

V.style_ax(ax)
ax.axhline(0.0, color=PAL["ink"], lw=0.5, alpha=0.4, zorder=0)
ax.set_xlabel("time t", fontsize=9)
ax.set_ylabel("latent factor value $Z_{t,r}$", fontsize=9)
ax.set_title("Latent factors emerge; ARD prunes the rest (rank is learned)",
             fontsize=8.2)
ax.legend(fontsize=6.5, loc="upper left", frameon=False, handlelength=1.6,
          labelspacing=0.3)
ax.text(0.97, 0.05, f"effective rank = {eff_rank}", transform=ax.transAxes,
        fontsize=8.5, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL["slate"], lw=0.8))

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("saved", os.path.abspath(OUT))
print(f"R={R} eff_rank={eff_rank} alpha_final={np.round(alpha_final,3)}")
