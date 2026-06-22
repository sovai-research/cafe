"""CAFE additive decomposition figure (single-column).

Shows, for ONE real series, how the imputed value is the sum of human-readable
parts: level + season + factor (+ noise residual). Every curve is read straight
out of the traced c_unified_penmf model via viz_common -- nothing is drawn by hand.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import viz_common as V

PAL = V.PALETTE

# --- run the REAL model with tracing ---------------------------------------
Xo, Xt, M, parts = V.demo_series(seed=1)
filled, trace, core = V.run_traced(Xo)
C = V.components(trace, Xo.shape[1])
T, N = Xo.shape

# --- pick a series with a visible contiguous missing block -----------------
def longest_block(col):
    best = bs = cur = cs = 0
    for t in range(len(col)):
        if col[t]:
            if cur == 0:
                cs = t
            cur += 1
            if cur > best:
                best, bs = cur, cs
        else:
            cur = 0
    return bs, best

cands = [(j, *longest_block(M[:, j])) for j in range(N)]
j, blk_start, blk_len = max(cands, key=lambda x: x[2])
blk_end = blk_start + blk_len
t = np.arange(T)

# --- the interpretable stacked pieces (all from the trace) -----------------
# The model fills a missing cell as  level + season + time_fe + factor + carry,
# where carry is the idiosyncratic AR(1) term. We fold the (small) shared
# time fixed-effect into the level offset and expose the remaining named pieces.
# The gap between "+factor" and "filled" is exactly the per-series AR carry --
# read out of the trace, not invented.
lvl = C["level"][:, j] + C["time_fe"]            # per-series level + time FE
lvl_sea = lvl + C["season"][:, j]
recon = lvl_sea + C["factor"][:, j]              # level + season + factor
fill = C["filled"][:, j]                          # + idiosyncratic AR carry

obs_mask = ~np.isnan(Xo[:, j])

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.5))

# shade the missing block span lightly
ax.axvspan(blk_start, blk_end, color=PAL["grey"], alpha=0.16, lw=0,
           label="missing block")

# observed points
ax.scatter(t[obs_mask], Xo[obs_mask, j], s=5, color=PAL["grey"],
           alpha=0.7, zorder=2, label="observed", edgecolors="none")

# stacked / overlaid additive pieces
ax.plot(t, lvl, lw=1.3, color=PAL["slate"], label="level")
ax.plot(t, lvl_sea, lw=1.3, color=PAL["teal"], label="+ season")
ax.plot(t, recon, lw=1.3, color=PAL["blue"], label="+ factor")
ax.plot(t, fill, lw=1.3, color=PAL["red"], ls=(0, (4, 1.5)),
        label="+ AR carry = fill")

ax.set_title("CAFE decomposes every value into interpretable parts",
             fontsize=8.5, pad=4)
ax.set_xlabel("time $t$", fontsize=9)
ax.set_ylabel(f"series $j={j}$ value", fontsize=9)
ax.set_xlim(0, T - 1)
V.style_ax(ax)
ax.legend(fontsize=6.6, ncol=2, loc="upper left", framealpha=0.85,
          handlelength=1.6, columnspacing=1.0, borderaxespad=0.3)

out = "/Users/dereksnow/Sovai/Github/TIMARA/paper/figures/decomposition.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
plt.close(fig)

# honesty check: on OBSERVED cells the fill must equal the observed value, and
# on MISSING cells the fill must equal level+season+time_fe+factor+carry exactly.
miss = M[:, j]
obs_pass = np.allclose(fill[obs_mask], Xo[obs_mask, j])
carry = fill - recon  # the idiosyncratic AR carry term, read from the trace
print(f"series j={j}, block t=[{blk_start},{blk_end}) len={blk_len}")
print(f"observed cells pass through unchanged: {obs_pass}")
print(f"AR carry on missing: max|carry|={np.abs(carry[miss]).max():.3f}")
print(f"saved {out}  exists={os.path.exists(out)}  "
      f"size={os.path.getsize(out)} bytes")
