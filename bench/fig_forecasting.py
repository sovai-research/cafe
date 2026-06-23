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
from viz_common import demo_series, style_ax, PALETTE, run_traced

# --- build the panel and the forecasting task -------------------------------
X, Xtrue, M, parts = demo_series(T=300, N=6, seed=6)
T, N = X.shape
H = 24                           # match the library's default forecast horizon
origin = T - H

Xf = X.copy()
Xf[origin:, :] = np.nan          # blank the entire future horizon, all series

# the REAL model: forecast = filled future rows. Trace it so we can also read
# the model's own AR/Kalman parameters for the predictive cone below.
filled, _trace, core = run_traced(Xf)

# Display the series with the most forecastable STRUCTURE: high seasonal signal
# relative to the random-walk factor + idiosyncratic noise. Season and level are
# deterministic in t, so the model extrapolates them through the blackout; this
# picks the panel's clearest forecast by an intrinsic property, not by outcome.
struct = np.array([parts["season"][:, c].var()
                   / (parts["factor"][:, c].var() + parts["noise"][:, c].var() + 1e-9)
                   for c in range(N)])
j = int(struct.argmax())
hist_t = np.arange(origin)
fut_t = np.arange(origin, T)

# --- predictive-uncertainty cone (a faithful read-out, not a re-fit) ---------
# In a blackout the model auto-regresses z_t = a z_{t-1} with per-step prior
# variance diag(s_fac) (its own generative law, c_unified_penmf l.409), so the
# h-step state variance is s_fac * (1-a^{2h})/(1-a^2) and the value variance for
# series j is sum_r W[j,r]^2 * that + psi[j]. The cone widens with horizon.
a    = float(core.a)
Wj2  = (np.asarray(core.W)[j] ** 2 * np.asarray(core.s_fac)).sum()
psij = float(np.asarray(core.psi)[j])
h    = np.arange(1, H + 1)
g    = (1.0 - a ** (2 * h)) / (1.0 - a * a) if a * a < 1.0 else h.astype(float)
sd_fut = np.sqrt(Wj2 * g + psij)
band_t  = np.concatenate([[origin - 1], fut_t])
band_mu = np.concatenate([[Xtrue[origin - 1, j]], filled[origin:, j]])
band_sd = np.concatenate([[0.0], sd_fut])

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)

# observed history (truth that the model actually saw)
ax.plot(hist_t, Xtrue[:origin, j], color=PALETTE["ink"], lw=1.3,
        label="observed")

# predictive uncertainty cone (+/-2 sigma), widening into the horizon
ax.fill_between(band_t, band_mu - 2 * band_sd, band_mu + 2 * band_sd,
                color=PALETTE["red"], alpha=0.13, lw=0,
                label=r"$\pm2\sigma$")

# held-out truth in the forecast region, for reference (faint)
ax.plot(fut_t, Xtrue[origin:, j], color=PALETTE["grey"], lw=1.3, alpha=0.7,
        label="held-out truth")

# the model's forecast = its fill of the future rows (dashed)
ax.plot(fut_t, filled[origin:, j], color=PALETTE["red"], lw=1.3, ls="--",
        label="forecast")

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
ax.legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=4, columnspacing=0.9, handlelength=1.4, handletextpad=0.4,
          borderpad=0.3, framealpha=0.0, borderaxespad=0.2)

out = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   "..", "paper", "figures", "forecasting.pdf"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)

mae = np.abs(filled[origin:, j] - Xtrue[origin:, j]).mean()
cov2 = float((np.abs(filled[origin:, j] - Xtrue[origin:, j]) / sd_fut <= 2).mean())
print(f"saved {out}  H={H} origin={origin}  series {j} (struct={struct[j]:.1f}) "
      f"MAE={mae:.3f}  a={a:.3f}  2sigma-cover={cov2:.0%}")
