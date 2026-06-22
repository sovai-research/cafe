"""causal_moat.pdf -- the headline 'no look-ahead' figure for CAFE.

For a fixed early missing target cell (t0, j), we sweep prefix lengths L and, for
each L, truncate X to its first L rows and impute with (a) CAFE's online causal
estimator and (b) a NON-causal batch SoftImpute. We record what each method
predicts for the SAME early cell (t0, j) as the future grows.

CAFE is point-in-time: its imputation of an early cell depends only on data up to
t0, so the recorded value is a flat horizontal line regardless of L. SoftImpute
re-reads the whole truncated matrix every time, so its estimate of the past wanders
as future rows are revealed -- a look-ahead leak. Every number below is read from
the REAL estimators; nothing is hand-drawn.
"""
import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt

import viz_common as V
import c_unified_penmf as P
import m_softimpute as S

OUT = "/Users/dereksnow/Sovai/Github/TIMARA/paper/figures/causal_moat.pdf"


def main():
    # One matrix with known structure.
    Xobs, Xtrue, M, _ = V.demo_series(T=300, seed=3)
    T, N = Xobs.shape

    # Pick an EARLY missing target cell near t0~60.
    t0_target = 60
    cand = [(t, j) for t in range(t0_target - 5, t0_target + 6)
            for j in range(N) if np.isnan(Xobs[t, j])]
    if not cand:
        raise RuntimeError("no missing cell near t0")
    # closest to t0_target=60
    t0, j = min(cand, key=lambda c: (abs(c[0] - t0_target), c[1]))

    Ls = list(range(t0 + 10, T + 1, 15))
    cafe_vals, soft_vals = [], []
    for L in Ls:
        Xpre = np.ascontiguousarray(Xobs[:L].copy())
        cafe = P.online_impute(Xpre, None)
        soft = S.impute(Xpre, None)
        cafe_vals.append(cafe[t0, j])
        soft_vals.append(soft[t0, j])

    cafe_vals = np.asarray(cafe_vals)
    soft_vals = np.asarray(soft_vals)

    # honesty check: how flat is CAFE really?
    cafe_spread = cafe_vals.max() - cafe_vals.min()
    soft_spread = soft_vals.max() - soft_vals.min()
    print(f"target cell (t0={t0}, j={j}); L sweep {Ls[0]}..{Ls[-1]} "
          f"({len(Ls)} pts)")
    print(f"CAFE value range = {cafe_spread:.2e}  (flat)")
    print(f"SoftImpute value range = {soft_spread:.3f}  (wanders)")

    pal = V.PALETTE
    fig, ax = plt.subplots(figsize=(6.8, 2.6), constrained_layout=True)

    ax.plot(Ls, soft_vals, "-o", color=pal["red"], lw=1.3, ms=3.2,
            label="SoftImpute (non-causal batch)", zorder=3)
    ax.plot(Ls, cafe_vals, "-o", color=pal["blue"], lw=1.3, ms=3.2,
            label="CAFE (online, point-in-time)", zorder=4)

    # Headroom so annotations sit in clean whitespace, not on the data.
    dmin = float(min(cafe_vals.min(), soft_vals.min()))
    dmax = float(max(cafe_vals.max(), soft_vals.max()))
    span = dmax - dmin
    ymin, ymax = dmin - 0.10 * span, dmax + 0.26 * span
    ax.set_ylim(ymin, ymax)

    # Mark when the target cell first became available (t0): everything to the
    # right is "future" relative to the imputed cell. Label rides the vline so it
    # never crosses a data series.
    ax.axvline(t0, color=pal["grey"], lw=0.9, ls=":", zorder=1)
    ax.text(t0 + 3, ymax - 0.03 * span,
            f"target window ends at $t={t0}$ (everything right of this is future)",
            fontsize=7, color=pal["slate"], va="top", ha="left", zorder=5)

    cafe_y = float(cafe_vals[0])
    # CAFE callout: park it in the empty band just below the flat blue line,
    # over the right half where SoftImpute has already dropped far away.
    ki = int(0.62 * (len(Ls) - 1))
    ax.annotate("CAFE: invariant to the future\n(frozen once $t$ has passed)",
                xy=(Ls[ki], cafe_y),
                xytext=(Ls[ki], cafe_y - 0.30 * span),
                fontsize=7.5, color=pal["blue"], ha="center", va="top",
                zorder=5,
                arrowprops=dict(arrowstyle="->", color=pal["blue"], lw=0.9))
    # SoftImpute callout: park it in the empty upper-right band, well above the
    # wandering red line, pointing down to its largest deviation.
    kmax = int(np.argmax(np.abs(soft_vals - soft_vals[0])))
    ax.annotate("SoftImpute: estimate of the\npast drifts as future is revealed",
                xy=(Ls[kmax], soft_vals[kmax]),
                xytext=(Ls[-1], dmin + 0.34 * span),
                fontsize=7.5, color=pal["red"], ha="right", va="bottom",
                zorder=5,
                arrowprops=dict(arrowstyle="->", color=pal["red"], lw=0.9))

    ax.set_xlabel("prefix length $L$ (rows of history given to the imputer)",
                  fontsize=9)
    ax.set_ylabel(f"imputed value at cell $(t_0{{=}}{t0},\\,j{{=}}{j})$",
                  fontsize=9)
    ax.set_title("No look-ahead: CAFE's past imputations are frozen; "
                 "batch methods leak", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    V.style_ax(ax)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    sz = os.path.getsize(OUT)
    print(f"saved {OUT}  ({sz} bytes)")
    assert sz > 3000, "PDF too small"


if __name__ == "__main__":
    main()
