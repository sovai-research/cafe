"""Figure: per-cell predictive uncertainty band from the REAL CAFE model.

Everything plotted is read straight out of the traced c_unified_penmf core via
viz_common: the filled values, and the per-cell posterior predictive variance
`cvar` (finite only where a value was imputed). No curve is hand-drawn.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viz_common as V

PAL = V.PALETTE
OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "paper", "figures", "uncertainty.pdf"))


def longest_block(mask_col):
    """Return (start, end) of the longest contiguous run of True."""
    T = len(mask_col); runs = []; s = None
    for t in range(T):
        if mask_col[t] and s is None:
            s = t
        if not mask_col[t] and s is not None:
            runs.append((s, t)); s = None
    if s is not None:
        runs.append((s, T))
    return max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)


def main():
    # REAL model run -------------------------------------------------------
    Xobs, Xtrue, M, _ = V.demo_series(seed=2)
    filled, trace, core = V.run_traced(Xobs)
    C = V.components(trace, Xobs.shape[1])
    T, N = Xobs.shape

    # pick a series that owns a real missing block ------------------------
    j = max(range(N), key=lambda c: longest_block(M[:, c])[1]
            - longest_block(M[:, c])[0])
    a, b = longest_block(M[:, j])

    t = np.arange(T)
    obs = ~np.isnan(Xobs[:, j])
    cvar = C["cvar"][:, j]                 # NaN where observed / no estimate
    sd = np.sqrt(cvar)
    band = np.isfinite(sd)                 # cells that carry an uncertainty band
    fj = filled[:, j]

    # +/-2 sigma band, only where defined; masked array => no spurious fill
    lo = np.ma.masked_array(fj - 2 * sd, mask=~band)
    hi = np.ma.masked_array(fj + 2 * sd, mask=~band)

    # honest calibration number, this whole series ------------------------
    mj = M[:, j]
    cover = (np.abs(fj[mj] - Xtrue[mj, j]) <= 2 * sd[mj]).mean()

    # window the view tightly around the block ----------------------------
    lo_x = max(0, a - 60); hi_x = min(T, b + 60)
    vsl = slice(lo_x, hi_x)
    # robust y-range from the visible window (truth + band), ignore outliers
    yvals = np.concatenate([
        Xtrue[vsl, j], fj[vsl],
        np.asarray(hi[vsl].filled(np.nan)), np.asarray(lo[vsl].filled(np.nan))])
    yvals = yvals[np.isfinite(yvals)]
    ylo = np.percentile(yvals, 1); yhi_v = np.percentile(yvals, 99)
    pad = 0.18 * (yhi_v - ylo)

    # plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)

    ax.fill_between(t, lo, hi, color=PAL["blue"], alpha=0.20, lw=0,
                    label=r"$\pm 2\sqrt{\mathrm{cvar}}$")
    ax.plot(t, Xtrue[:, j], color=PAL["grey"], lw=1.0, alpha=0.85,
            label="ground truth", zorder=2)
    ax.plot(t, fj, color=PAL["blue"], lw=1.3, label="CAFÉ fill", zorder=3)
    ax.plot(t[obs], Xobs[obs, j], ls="none", marker="o", ms=2.4,
            mfc=PAL["ink"], mec="none", label="observed", zorder=4)

    # mark the missing block
    ax.axvspan(a, b, color=PAL["amber"], alpha=0.07, lw=0, zorder=0)

    # extra headroom at top for the in-block annotation so it clears the title
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(ylo - pad, yhi_v + pad * 2.4)
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel(f"series {j}", fontsize=9)
    ax.set_title("Every fill carries an uncertainty band (widens in gaps)",
                 fontsize=9.5, pad=6)

    # annotate the widening of the band inside the block ------------------
    mid = (a + b) // 2
    y_top = float((fj + 2 * sd)[mid])
    ax.annotate("band widens:\nno data in block",
                xy=(mid, y_top), xytext=(mid, yhi_v + pad * 2.2),
                fontsize=7.0, ha="center", va="top", color=PAL["amber"],
                arrowprops=dict(arrowstyle="->", color=PAL["amber"], lw=1.0))

    # honest calibration number, parked in clear space at lower left
    ax.text(0.02, 0.04, f"2$\\sigma$ coverage = {cover:.0%}",
            transform=ax.transAxes, fontsize=7.2, color=PAL["slate"],
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PAL["grey"],
                      lw=0.5, alpha=0.85))
    V.style_ax(ax)
    # legend below the axes so it never covers data
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, ncol=4, handlelength=1.2, columnspacing=1.1,
              handletextpad=0.5, borderaxespad=0.0)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    sz = os.path.getsize(OUT)
    print(f"saved {OUT} ({sz} bytes); series j={j}, block=[{a},{b}], "
          f"coverage={cover:.3f}")


if __name__ == "__main__":
    main()
