"""CAFE paper figure: headline accuracy bar chart on Beijing Air-Quality.

Uses the REAL published values from the paper's Table 2 (SAITS protocol,
standardized, 10% MCAR, MAE; lower is better). These are fixed literature
numbers -- nothing is fabricated. CAFE is our model, highlighted.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_common import PALETTE, style_ax

OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "paper", "figures", "benchmark.pdf"))

# Published Table 2 values (MAE, lower is better). Do NOT invent others.
results = [
    ("Median",      0.763),
    ("M-RNN",       0.294),
    ("GP-VAE",      0.268),
    ("Transformer", 0.158),
    ("BRITS",       0.153),
    ("SAITS",       0.137),
    ("CAFÉ (ours)", 0.108),
]

# Sort worst -> best so best ends up at the top of a horizontal bar chart.
results = sorted(results, key=lambda kv: kv[1], reverse=True)
labels = [r[0] for r in results]
vals = [r[1] for r in results]
ypos = list(range(len(results)))

fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)

colors = [PALETTE["teal"] if "CAFÉ" in lab else PALETTE["grey"] for lab in labels]
bars = ax.barh(ypos, vals, color=colors, height=0.68,
               edgecolor="white", linewidth=0.4, zorder=3)

ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8)
for tick, lab in zip(ax.get_yticklabels(), labels):
    if "CAFÉ" in lab:
        tick.set_color(PALETTE["teal"])
        tick.set_fontweight("bold")

ax.set_xlabel("MAE on missing entries (lower is better)", fontsize=8.5)
ax.set_xlim(0, max(vals) * 1.18)

# Value labels at bar ends.
for y, v, lab in zip(ypos, vals, labels):
    is_cafe = "CAFÉ" in lab
    ax.text(v + max(vals) * 0.012, y, f"{v:.3f}",
            va="center", ha="left", fontsize=7.6,
            color=PALETTE["teal"] if is_cafe else PALETTE["ink"],
            fontweight="bold" if is_cafe else "normal", zorder=4)

# Annotation on the CAFE bar, placed in the empty right-hand whitespace.
cafe_y = labels.index("CAFÉ (ours)")
ax.annotate("causal + CPU\n(others bidirectional + GPU)",
            xy=(vals[cafe_y], cafe_y),
            xytext=(max(vals) * 0.40, cafe_y - 0.55),
            fontsize=6.6, color=PALETTE["teal"], ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=PALETTE["teal"],
                            lw=0.8, shrinkA=2, shrinkB=2,
                            connectionstyle="arc3,rad=0.2"))

ax.set_title("Lowest MAE on Beijing Air-Quality --\nat less information and ~1000x less compute",
             fontsize=8.8, fontweight="bold", pad=6)

style_ax(ax)
ax.grid(True, axis="x", alpha=0.18, lw=0.6)
ax.grid(False, axis="y")
# labels sorted worst->best (CAFE last/highest index); place best at the top.
ax.set_ylim(len(results) - 0.5, -0.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
print("saved", OUT, "size", os.path.getsize(OUT))
