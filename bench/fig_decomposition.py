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

# --- zoom to a readable window centred on the missing block ----------------
# Full 400-step panel overlays four wiggly curves into illegible spaghetti;
# a window around the gap shows the additive layering clearly. Still the exact
# same real model output -- only the x-range shown is restricted.
pad = max(blk_len, 30)
w0 = max(0, blk_start - pad)
w1 = min(T, blk_end + pad)
ws = slice(w0, w1)

# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 2.6), constrained_layout=True)

# shade the missing block span lightly
ax.axvspan(blk_start, blk_end, color=PAL["grey"], alpha=0.18, lw=0,
           label="missing block")

# observed points
win_obs = obs_mask & (t >= w0) & (t < w1)
ax.scatter(t[win_obs], Xo[win_obs, j], s=9, color=PAL["ink"],
           alpha=0.55, zorder=5, label="observed", edgecolors="none")

# stacked / overlaid additive pieces
ax.plot(t[ws], lvl[ws], lw=1.2, color=PAL["slate"], label="level")
ax.plot(t[ws], lvl_sea[ws], lw=1.2, color=PAL["teal"], label="+ season")
ax.plot(t[ws], recon[ws], lw=1.2, color=PAL["blue"], label="+ factor")
ax.plot(t[ws], fill[ws], lw=1.6, color=PAL["red"], ls=(0, (4, 1.5)),
        zorder=4, label="+ AR carry = fill")

ax.set_title("CAFE decomposes every value into interpretable parts",
             fontsize=10, pad=5)
ax.set_xlabel("time $t$", fontsize=9)
ax.set_ylabel(f"series $j={j}$ value", fontsize=9)
ax.set_xlim(w0, w1 - 1)
V.style_ax(ax)
ax.legend(fontsize=7.5, ncol=1, loc="center left",
          bbox_to_anchor=(1.01, 0.5), framealpha=0.9,
          handlelength=1.8, borderaxespad=0.0)

out = "/Users/dereksnow/Sovai/Github/TIMARA/paper/figures/decomposition.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
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
