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
    fig, ax = plt.subplots(figsize=(6.8, 2.6))

    ax.plot(Ls, soft_vals, "-o", color=pal["red"], lw=1.3, ms=3.2,
            label="SoftImpute (non-causal batch)", zorder=3)
    ax.plot(Ls, cafe_vals, "-o", color=pal["blue"], lw=1.3, ms=3.2,
            label="CAFE (online, point-in-time)", zorder=4)

    # mark when the target cell first became available (t0) -- everything to the
    # right is "future" relative to the imputed cell.
    ax.axvline(t0, color=pal["grey"], lw=0.9, ls=":", zorder=1)
    ymin, ymax = ax.get_ylim()
    ax.text(t0 + 2, ymax - 0.04 * (ymax - ymin),
            f"target cell observed window ends at t={t0}",
            fontsize=7, color=pal["slate"], va="top")

    # Annotations.
    cy = cafe_vals.mean()
    ax.annotate("CAFE: invariant to the future\n(frozen once t passes)",
                xy=(Ls[len(Ls) // 2], cafe_vals[len(Ls) // 2]),
                xytext=(Ls[len(Ls) // 3], cy - 0.55 * (ymax - ymin) * 0.6
                        if cy > (ymin + ymax) / 2 else cy + 0.30 * (ymax - ymin)),
                fontsize=7.5, color=pal["blue"], ha="center",
                arrowprops=dict(arrowstyle="->", color=pal["blue"], lw=0.9))
    kmax = int(np.argmax(np.abs(soft_vals - soft_vals[0])))
    ax.annotate("SoftImpute: past estimate\nchanges as future is revealed",
                xy=(Ls[kmax], soft_vals[kmax]),
                xytext=(Ls[max(1, kmax - 3)],
                        soft_vals[kmax] + 0.30 * (ymax - ymin)
                        * (1 if soft_vals[kmax] < (ymin + ymax) / 2 else -1)),
                fontsize=7.5, color=pal["red"], ha="center",
                arrowprops=dict(arrowstyle="->", color=pal["red"], lw=0.9))

    ax.set_xlabel("prefix length $L$ (rows of history given to the imputer)",
                  fontsize=9)
    ax.set_ylabel(f"imputed value at cell $(t_0{{=}}{t0},\\,j{{=}}{j})$",
                  fontsize=9)
    ax.set_title("No look-ahead: CAFE's past imputations are frozen; "
                 "batch methods leak", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="best", frameon=False)
    V.style_ax(ax)

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    sz = os.path.getsize(OUT)
    print(f"saved {OUT}  ({sz} bytes)")
    assert sz > 3000, "PDF too small"


if __name__ == "__main__":
    main()
