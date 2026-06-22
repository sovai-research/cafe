"""Forecasting = imputing future rows. Single-column figure.

We mask the LAST H rows of every series entirely and ask the REAL c_unified_penmf
model (online_impute) to fill them. Because the model is point-in-time/causal, the
fill of those future rows IS a forecast: the AR/Kalman latent state z_t = a z_{t-1}
continues with no new observations. We plot one series' observed history, the
model's forecast in the masked region (dashed), and the held-out truth (faint).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from viz_common import demo_series, style_ax, PALETTE
import c_unified_penmf as P

# --- build the panel and the forecasting task -------------------------------
X, Xtrue, M, parts = demo_series(T=300, N=6, seed=6)
T, N = X.shape
H = 40
origin = T - H

Xf = X.copy()
Xf[origin:, :] = np.nan          # blank the entire future horizon, all series

filled = P.online_impute(Xf, {})  # the REAL model: forecast = filled future rows

j = 0                            # series to display
hist_t = np.arange(origin)
fut_t = np.arange(origin, T)

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)

# observed history (truth that the model actually saw)
ax.plot(hist_t, Xtrue[:origin, j], color=PALETTE["ink"], lw=1.3,
        label="observed history")

# held-out truth in the forecast region, for reference (faint)
ax.plot(fut_t, Xtrue[origin:, j], color=PALETTE["grey"], lw=1.3, alpha=0.7,
        label="held-out truth")

# the model's forecast = its fill of the future rows (dashed)
ax.plot(fut_t, filled[origin:, j], color=PALETTE["red"], lw=1.3, ls="--",
        label="model forecast (fill)")

# connect the last observed point to the first forecast point
ax.plot([origin - 1, origin], [Xtrue[origin - 1, j], filled[origin, j]],
        color=PALETTE["red"], lw=1.3, ls="--")

# forecast-origin marker
ax.axvline(origin, color=PALETTE["slate"], lw=0.9, ls=":")
ax.axvspan(origin, T - 1, color=PALETTE["amber"], alpha=0.08, lw=0)

# annotate the origin in whitespace just left of the line, inside the data area
y0, y1 = ax.get_ylim()
ax.text(origin - 4, y0 + 0.06 * (y1 - y0), "forecast\norigin", fontsize=6.5,
        color=PALETTE["slate"], va="bottom", ha="right", linespacing=0.95)

style_ax(ax)
ax.set_xlabel("time step $t$", fontsize=9)
ax.set_ylabel(f"series {j} value", fontsize=9)
ax.set_title("Forecasting is just imputing future rows (one model)", fontsize=9,
             pad=14)
ax.legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=3, columnspacing=1.0, handlelength=1.6, handletextpad=0.5,
          borderpad=0.3, framealpha=0.0, borderaxespad=0.2)

out = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   "..", "paper", "figures", "forecasting.pdf"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)

mae = np.abs(filled[origin:, j] - Xtrue[origin:, j]).mean()
print(f"saved {out}  H={H} origin={origin}  series {j} forecast MAE={mae:.3f}")
