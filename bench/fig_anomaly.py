"""CAFE figure: free anomaly detection via Student-t row weights.

Top panel: one observed series with the injected outlier rows circled in red.
Bottom panel: the per-row Student-t anomaly weight C["row_w"] (low = flagged),
shaded where it drops below a small threshold. The low-weight rows line up with
the true injected outliers -- the model gives a point-in-time anomaly score for
free, as a by-product of its heavy-tailed (Student-t) observation likelihood.

Every number/curve is read straight from the REAL c_unified_penmf estimator via
viz_common (run_traced + components). Nothing is hand-drawn.
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import viz_common as V

P = V.PALETTE

# --- run the REAL model with tracing -------------------------------------
Xo, Xt, M, parts = V.demo_series(outlier_frac=0.03, seed=5)
filled, trace, core = V.run_traced(Xo)
C = V.components(trace, Xo.shape[1])

T, N = Xo.shape
t = np.arange(T)
row_w = C["row_w"]                  # (T,) Student-t per-row anomaly weight
om = parts["outliers"]             # (T,N) injected-outlier mask
obs = ~np.isnan(Xo)

# pick the series with the most VISIBLE (observed) injected outliers
j = int((om & obs).sum(axis=0).argmax())

xj = Xo[:, j]
obs_j = obs[:, j]
out_j = om[:, j] & obs_j           # injected outliers actually observed in series j
out_rows = om.any(axis=1)         # any-series outlier rows (what the row weight sees)

# small threshold for "flagged": low quantile of the weights
thr = np.quantile(row_w, 0.15)
flagged = row_w < thr

# --- figure ---------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(3.4, 2.5), sharex=True,
    gridspec_kw=dict(height_ratios=[1.45, 1.0], hspace=0.18))

# top: observed series + injected outliers circled
ax0.plot(t[obs_j], xj[obs_j], color=P["slate"], lw=1.0, alpha=0.55, zorder=1)
ax0.scatter(t[obs_j], xj[obs_j], s=5, color=P["blue"], zorder=2,
            label="observed")
ax0.scatter(t[out_j], xj[out_j], s=46, facecolors="none",
            edgecolors=P["red"], linewidths=1.3, zorder=3,
            label="injected outlier")
ax0.set_ylabel(f"series {j}", fontsize=8)
ax0.set_title("Student-t weights flag outliers point-in-time\n(free anomaly score)",
              fontsize=9)
ax0.legend(fontsize=6.5, loc="upper left", frameon=False, ncol=2,
           handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)
V.style_ax(ax0)

# bottom: per-row Student-t weight, shade where flagged
ax1.axhline(thr, color=P["grey"], lw=0.8, ls="--", zorder=1)
ax1.fill_between(t, 0, row_w, where=flagged, step="mid",
                 color=P["red"], alpha=0.20, zorder=1,
                 label=f"flagged (w < {thr:.2f})")
ax1.plot(t, row_w, color=P["teal"], lw=1.3, zorder=2)
# mark where true outlier rows are, to show alignment
ymax = max(1.05, float(row_w.max()) * 1.02)
ax1.plot(t[out_rows], np.full(out_rows.sum(), -0.04 * ymax),
         marker="|", ls="none", ms=5, color=P["red"], alpha=0.8,
         label="true outlier row")
ax1.set_ylim(-0.08 * ymax, ymax)
ax1.set_ylabel("row weight", fontsize=8)
ax1.set_xlabel("time t", fontsize=8)
ax1.legend(fontsize=6.5, loc="lower center", frameon=True, framealpha=0.85,
           edgecolor="none", ncol=2, handletextpad=0.3, columnspacing=0.9,
           borderaxespad=0.2)
V.style_ax(ax1)

out = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "paper", "figures", "anomaly.pdf"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

# --- honesty check: do low weights line up with true outliers? -----------
det = flagged[out_rows].mean()
fp = flagged[~out_rows].mean()
print(f"saved {out} ({os.path.getsize(out)} bytes)")
print(f"series j={j}, visible outliers={out_j.sum()}, thr={thr:.3f}")
print(f"recall(flagged|true outlier row)={det:.2f}, "
      f"false-flag rate(clean row)={fp:.2f}")
print(f"mean weight: outlier rows={row_w[out_rows].mean():.3f}, "
      f"clean rows={row_w[~out_rows].mean():.3f}")
