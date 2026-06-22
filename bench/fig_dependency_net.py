"""CAFE paper figure: dependency_net (single-column).

Shows the cross-sectional dependency network CAFE recovers. We run the REAL
c_unified_penmf core (via viz_common.run_traced) on a demo panel with block
structure (rank-3 common factors -> correlated groups), read the pooled EW
residual covariance straight out of the running estimator, convert it to a
correlation matrix, reorder series by the leading eigenvector to expose blocks,
and plot it as a diverging heatmap. Nothing is fabricated: every entry comes
from the model's learned residual covariance.
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import viz_common as V

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "paper", "figures", "dependency_net.pdf")
OUT = os.path.abspath(OUT)

# --- run the REAL model -----------------------------------------------------
Xobs, Xtrue, M, parts = V.demo_series(N=12, rank=3, seed=7)
filled, trace, core = V.run_traced(Xobs)
cov = V.residual_cov(core)
assert cov is not None, "residual_cov returned None"
N = cov.shape[0]

# --- covariance -> correlation ---------------------------------------------
d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
R = cov / np.outer(d, d)
R = np.clip(R, -1.0, 1.0)

# --- reorder by leading eigenvector to reveal blocks ------------------------
w, Vec = np.linalg.eigh(R)
lead = Vec[:, np.argmax(w)]
order = np.argsort(lead)
Rord = R[np.ix_(order, order)]

# --- plot -------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.5))

im = ax.imshow(Rord, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")

labels = [f"s{j}" for j in order]
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(labels, fontsize=6, rotation=90)
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel("series (reordered by leading eigenvector)", fontsize=8)
ax.set_ylabel("series", fontsize=8)
ax.set_title("Residual covariance = learned dependency\nnetwork between series",
             fontsize=9)

# thin grid between cells for readability
ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
ax.grid(which="minor", color="white", lw=0.5)
ax.tick_params(which="minor", length=0)
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("residual correlation", fontsize=8)
cbar.ax.tick_params(labelsize=7)
cbar.outline.set_visible(False)

fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)

# off-diagonal correlation magnitude as a sanity readout
off = R[~np.eye(N, dtype=bool)]
print(f"saved {OUT}")
print(f"N={N}  max|off-diag corr|={np.abs(off).max():.2f}  "
      f"mean|off-diag|={np.abs(off).mean():.2f}")
print(f"file exists={os.path.exists(OUT)}  "
      f"size={os.path.getsize(OUT) if os.path.exists(OUT) else 0} bytes")
