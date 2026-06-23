"""Builds notebooks/cafe_benchmark.ipynb. Run, then execute to validate.

Honest-framing contract (kept in sync with README.md, paper/cafe.tex and the
single source of truth bench/refs_published.py):

  * Published competitor numbers are CONTEXT, never a head-to-head leaderboard.
    CAFE is never ranked into the same column as FGTI/CSDI/SAITS.
  * Where two sources disagree on a cell (Beijing SAITS .137 [du2023] vs .155
    [tsibench]; BRITS .153 vs .127; Transformer .158 vs .142) BOTH are shown.
    We do NOT cherry-pick the flattering one. Under the TSI-Bench source CSDI
    (.102) is the lowest published Beijing MAE, so NO protocol-independent
    "lowest MAE" claim is made.
  * Wherever CAFE's MAE appears the exact mask protocol is named:
    10% MCAR / point masking on the standardized dense slice.
  * Message is the "two-of-three": causal (point-in-time) + CPU-only + in the
    same accuracy band as bidirectional deep SOTA.

cafe.benchmark() (src/cafe/benchmark.py) already encodes this honest comparison
(causal vs bidirectional blocks, both-source rows, CAFE not ranked among the
published rows) and mirrors bench/refs_published.py, so the notebook leans on it
rather than re-deriving the numbers here.
"""
import pathlib

import nbformat as nbf
from nbformat.v4 import new_code_cell as code, new_markdown_cell as md, new_notebook

cells = []

cells.append(md(r"""# CAFÉ — one line, a full benchmark

> **The whole thesis of this library: one line of code runs a complete, *honest*
> analysis.** No config, no fitting, no plumbing.

This notebook benchmarks CAFÉ's imputation against live baselines and the published
state-of-the-art — with a single call, `cafe.benchmark(...)`.

**The honest framing up front — the "two-of-three".** Prior strong imputers pick at most
two of three properties: **causal / point-in-time**, **CPU-only**, and **competitive with
bidirectional deep SOTA**. The published front-runners — SAITS, BRITS, Transformer, CSDI,
ImputeFormer, FGTI — are all **bidirectional** (to fill `X[t]` they look at the *whole
window, including the future*) **and GPU-trained**. Bidirectional filling is a *smoothing*
task; in quantitative finance it is literally *forbidden look-ahead bias*
([Blanchet–Pelger 2022](https://arxiv.org/abs/2202.00871)). CAFÉ is **causal** — it fills
`X[t]` from data `≤ t` only, so it is backtest-safe by construction — **CPU-only**
(`numpy`, no GPU, no training), and lands **in the same accuracy band** as those
bidirectional deep models. That combination is, as of today, essentially unoccupied.

**What this notebook will and will not claim.** Every baseline below is *run live* on the
**same mask, same seed, same held-out cells** — apples-to-apples. The published deep
numbers are shown only as **clearly-labelled, different-protocol context**, never merged
with the live results and **CAFÉ is never ranked among them**. They come from one
reconciled registry, [`bench/refs_published.py`](../bench/refs_published.py); where two
sources disagree on a cell, **both values are kept** (we do not cherry-pick the flattering
one). Under the TSI-Bench source CSDI (0.102) is the *lowest* published Beijing MAE, so we
make **no protocol-independent "lowest MAE" claim**."""))

cells.append(md("## 0 · Setup"))
cells.append(code(r"""import sys, pathlib
_src = pathlib.Path.cwd().parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import matplotlib.pyplot as plt
import cafe
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print("CAFÉ", cafe.__version__)"""))

cells.append(md(r"""## 1 · The one-liner

`cafe.benchmark()` with no arguments runs on a self-contained synthetic dataset — CAFÉ
against several baselines, **same held-out cells, same seed, scored identically**. That's
the entire API."""))
cells.append(code(r"""cafe.benchmark()"""))

cells.append(md(r"""Read the table top to bottom: it is **grouped by kind** — *causal* methods
(fill `X[t]` from data `≤ t`, backtest-safe) first, *bidirectional* methods (use the whole
series incl. the future — smoothing) below, each block sorted best-first. CAFÉ tops the
causal block (★). The `global mean` row is the **predict-the-mean trap** detector: anything
near it has no real skill. The baselines are themselves `numpy`-only, matching the
library."""))

cells.append(md(r"""## 2 · The headline — real data, in context with the published SOTA

`cafe.benchmark("beijing")` runs on **Beijing Multi-Site Air-Quality** (17,117 × 132) — the
dataset SAITS / CSDI / BRITS report on. **Live baselines are run here on the same mask.**
The published deep-learning numbers below are printed only as **labelled context**, drawn
from the single reconciled registry `bench/refs_published.py`.

**Name the mask, every time.** CAFÉ's score here is its MAE under a **10% MCAR / point mask
on the standardized dense slice** — i.e. 10% of observed cells dropped independently at
random (`np.random.default_rng`), per-column z-scored, MAE on the held-out cells. CAFÉ
imputes the **full series causally / online**.

**Why this is *not* a head-to-head leaderboard.** The published numbers come from a
*different, windowed* train/val/test protocol on a different Beijing preprocessing. They are
context, not a ranking — so CAFÉ is deliberately **not** placed in the same column as
FGTI / CSDI / SAITS. And because two sources disagree, the table shows **both** values
(e.g. SAITS 0.137 [du2023] *and* 0.155 [tsibench]; BRITS 0.153 *and* 0.127; Transformer
0.158 *and* 0.142). Under TSI-Bench, **CSDI (0.102) is the lowest** published Beijing MAE —
we claim only that CAFÉ lands *in the band*, not that it is lowest."""))
cells.append(code(r"""beijing = cafe.benchmark("beijing")"""))

cells.append(md(r"""**What this shows — stated honestly.** CAFÉ — *causal, `numpy`-only, ~7 s on one
CPU core, no training* — beats every **live** baseline run here (linear interp, SoftImpute,
SVDImpute, NOCB, LOCF, Kalman filter, …), causal and bidirectional alike. Its MAE under the
**10% MCAR / point mask on the standardized slice** lands **inside the published
bidirectional band**: in the same neighbourhood as SAITS (0.137 [du2023] / 0.155
[tsibench]), Transformer (0.142–0.158) and BRITS (0.127–0.153), with the diffusion model
CSDI (0.102, [tsibench]) and iTransformer (0.123) reported lower — **both of which see the
future**. CAFÉ is the **only causal method in the comparison**.

Because those rows use a *different protocol*, read this as **"CAFÉ sits in the band"**, not
as a win or a loss against any single number. The defensible claim is the **two-of-three**:
a causal, CPU-only method landing in the deep-SOTA accuracy band *at all*."""))
cells.append(code(r"""beijing.plot(); plt.tight_layout(); plt.show()"""))

cells.append(md(r"""Blue = causal (top block), grey = bidirectional (below the dotted divider). The
dashed red line is the **best published bidirectional** score (CSDI 0.102, TSI-Bench) — drawn
as a *different-protocol context line*, not a like-for-like target. CAFÉ sits near it while
being the only backtest-safe option in the figure."""))

cells.append(md(r"""## 3 · The honest hard case — contiguous gaps

Scattered (MCAR) gaps are the easy case: a missing cell usually has observed neighbours in
its own row. The hard, realistic case is **block** missingness — a sensor goes dark for a
stretch. We don't hide from it; `pattern="block"` runs it on the **same Beijing slice,
10% held out**. Causal methods can't borrow from the future across a blackout, so the gap
is genuinely harder for everyone — but CAFÉ's factor + AR state still carries it, and it
still tops the live causal block."""))
cells.append(code(r"""cafe.benchmark("beijing", pattern="block", missing=0.1)"""))

cells.append(md(r"""### Gap-length stratification

The research harness (`bench/exp_maskgrid.py`) sweeps a full **pattern × rate** grid
(Point/MCAR, Subsequence, Block × 10/30/50%) and stratifies one block run by **gap length**
(short / medium / long contiguous gaps), writing `paper/tables/maskgrid.tex`. Every cell
there is run *live on the same mask* — no competitor deep model is ever executed. We
reproduce the spirit of it here, **honestly and self-contained**, by scoring CAFÉ against
the live baselines as the block fraction grows: longer / denser blackouts are where any
*causal* method must give up the most, since it cannot look across the gap from the
future."""))
cells.append(code(r"""rates = [0.1, 0.2, 0.3, 0.4]
grid = {}
for rt in rates:
    r = cafe.benchmark("beijing", pattern="block", missing=rt, verbose=False)
    grid[rt] = {row["method"]: row["mae"] for row in r.rows if row["ok"]}

methods = ["CAFÉ", "LOCF", "linear interp", "SoftImpute"]
print(f"{'block %':>8s}" + "".join(f"{m:>15s}" for m in methods))
for rt in rates:
    print(f"{int(rt*100):>7d}%" + "".join(f"{grid[rt].get(m, float('nan')):>15.3f}" for m in methods))
print("\n(MAE, standardised, 10–40% Block mask on the Beijing slice; lower is better.")
print(" CAFÉ & LOCF are strictly causal; linear interp & SoftImpute are bidirectional.)")"""))

cells.append(code(r"""fig, ax = plt.subplots(figsize=(7, 4))
for m in methods:
    ys = [grid[rt].get(m, np.nan) for rt in rates]
    style = dict(marker="o", lw=2) if m == "CAFÉ" else dict(marker=".", lw=1, alpha=0.8)
    ax.plot([int(rt*100) for rt in rates], ys, label=m, **style)
ax.set_xlabel("Block-missing fraction (%)")
ax.set_ylabel("MAE (standardised)")
ax.set_title("Beijing — degradation under growing contiguous gaps (all run live, same mask)")
ax.legend()
plt.tight_layout(); plt.show()"""))

cells.append(md(r"""## 4 · Breadth — another real dataset, one line

`cafe.benchmark("etth1")` on the Electricity Transformer Temperature set (17,420 × 7), under
the same **10% MCAR / point mask on the standardized slice**. No published reference rows
here on purpose: the public ETT *imputation* numbers use a different protocol, so quoting
them would be apples-to-oranges — and this notebook never does that."""))
cells.append(code(r"""etth1 = cafe.benchmark("etth1", missing=0.1)
etth1.plot(); plt.tight_layout(); plt.show()"""))

cells.append(md(r"""## 5 · It's the same one model behind everything

The benchmark imputer is the *same* `cafe.impute` you'd use in one line on your own data —
and the same forward pass yields uncertainty, factors, anomalies, a decomposition, a
dependency network and forecasts (see **`cafe_tutorial.ipynb`**). One model, every view of
its own posterior — all of it causal."""))
cells.append(code(r"""X = np.load("../data/ETTh1_clean.npy")[:800]          # a real matrix
X[np.random.default_rng(0).random(X.shape) < 0.1] = np.nan
res = cafe.CAFE().run(X)
print("imputed:", np.asarray(res.imputed).shape,
      "| factors:", res.factors().shape,
      "| anomaly:", np.asarray(res.anomaly_scores()).shape,
      "| params:", res.params)"""))

cells.append(md(r"""## Recap

| One line | What it does |
|---|---|
| `cafe.impute(df)` | fill gaps, any container, **no look-ahead** |
| `cafe.benchmark(df)` | CAFÉ vs **live** baselines + cited SOTA *context*, scored honestly |
| `cafe.benchmark("beijing")` | reproduce the headline: causal CAFÉ **in the bidirectional band** |
| `cafe.CAFE().run(df)` | uncertainty, factors, anomalies, decomposition, network, forecast |

**The claim, stated precisely.** CAFÉ is **causal (point-in-time), CPU-only, zero-config**,
and lands in the **same MAE band** as GPU-trained bidirectional SOTA on the very benchmark
those methods report — measured under a **10% MCAR / point mask on the standardized dense
slice**. That is the **two-of-three**: causal + CPU + competitive. We do **not** claim the
single lowest MAE — under TSI-Bench, CSDI (0.102) is lower, and CAFÉ is never ranked
head-to-head against the different-protocol published numbers. The published numbers come
from one reconciled registry, [`bench/refs_published.py`](../bench/refs_published.py), which
keeps both values wherever two sources disagree.

```bash
pip install cafe-impute
```"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
out = pathlib.Path(__file__).parent / "cafe_benchmark.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
