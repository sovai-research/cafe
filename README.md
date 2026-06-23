<h1 align="center">CAFÉ</h1>
<p align="center"><b>Causal Adaptive Factor Estimation</b><br>
Zero-config, CPU-first, <b>point-in-time</b> missing-value imputation —
with uncertainty, factors, anomalies and forecasts from a single forward pass.</p>

<p align="center">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT">
<img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg" alt="py">
<img src="https://img.shields.io/badge/deps-numpy--only-informational.svg" alt="deps">
<img src="https://img.shields.io/badge/no-GPU%20%7C%20no%20training-success.svg" alt="cpu">
</p>

> **CAFÉ** is the model formerly developed in this repo under the name **TIMARA**.
> It is a *mechanistic statistical model — not a neural network*: one penalised
> objective whose learned parameters make SoftImpute, TRMF, the Kalman filter,
> MC-NNM and Gaussian conditional-mean imputation all **special cases**.

---

## Why CAFÉ

Almost every imputer fills `X[t]` using the **entire** series — including the future.
That silently leaks look-ahead into any sequential pipeline (a trading backtest, an
online controller, an early-warning monitor) and inflates measured performance.

CAFÉ fills `X[t]` using **only data up to time `t`** (past + the contemporaneous
cross-section), and a mechanical verifier *proves* no past imputation changes when
the future arrives. It is:

- **Causal / point-in-time** — backtest-safe by construction (the moat).
- **Zero-config** — `cafe.impute(data)`; rank, memory, tail-robustness and seasonality
  are learned from the data (ARD / empirical-Bayes / EM), not set by you.
- **CPU-first, `numpy`-only** — the entire estimator runs on `numpy` alone (no `scipy`,
  no compiled extension), no GPU, no training run. Installs in seconds. Runs the full
  benchmark suite in ~1 s.
- **Container-native** — `numpy`, `pandas`, `polars`, 1D or 2D, dtype/labels preserved.
- **More than imputation** — the same pass yields per-cell uncertainty, latent
  factors, anomaly scores, an additive decomposition, a dependency network and forecasts.

### The "two-of-three" claim

Prior strong imputers pick at most two of {**causal / point-in-time**, **CPU-only**,
**competitive with bidirectional deep SOTA**}. The published front-runners — SAITS,
BRITS, Transformer, CSDI, ImputeFormer, FGTI — are all **bidirectional** (they fill
the past using the future) **and GPU-trained**. CAFÉ is, to our knowledge, the first
method to credibly claim **all three at once**: strictly point-in-time, `numpy`-only on
a CPU, and in the same accuracy band as those bidirectional deep models.

On `data/beijing_clean.npy` (the longest fully-observed slice, 17,117 × 132,
per-column z-scored once), under a **10% point-MCAR mask** (`np.random.default_rng`,
seeds {0,1,2}, MAE on the standardised scale over held-out cells), CAFÉ imputes the
**full series causally/online** and reaches **MAE ≈ 0.108**. The published deep numbers
(SAITS, BRITS, …) come from a **different, windowed train/val/test protocol** on a
different Beijing preprocessing, so they are **context, not a head-to-head leaderboard**
— CAFÉ is *not* ranked among them. Under the TSI-Bench source, diffusion-based **CSDI
reaches 0.102**, lower than CAFÉ; we therefore make **no protocol-independent "lowest
MAE" claim**. The point is the moat: a causal, CPU-only method landing *in that band*
at all. Published numbers come from one reconciled registry
([`bench/refs_published.py`](bench/refs_published.py)); see [`paper/cafe.pdf`](paper/cafe.pdf).

## Install

```bash
git clone https://github.com/sovai-research/cafe.git
cd cafe
pip install -e .            # core (numpy only)
pip install -e ".[all]"     # + pandas, polars, matplotlib
```

## Quick start

```python
import cafe

# zero-config — same container type comes back, gaps filled, no look-ahead
filled = cafe.impute(df)            # pandas / polars DataFrame, or numpy array, 1D or 2D
```

A DataFrame may freely mix types: a `date` column, string ids and numeric sensors all
in one frame. CAFÉ imputes **only the numeric columns**, passes everything else through
untouched, and preserves column order — so `cafe.impute(raw_df)` just works, no manual
column selection.

> **Notebooks** (all runnable, executed end-to-end):
> - [`cafe_tutorial.ipynb`](notebooks/cafe_tutorial.ipynb) — polars-first deep dive on real
>   ETTh1: the one-liner, the no-look-ahead proof, accuracy, calibrated + gap-widening
>   uncertainty, factors, anomaly detection, exact decomposition, dependency net, forecast.
> - [`cafe_it_just_works.ipynb`](notebooks/cafe_it_just_works.ipynb) — every container/shape
>   (numpy/pandas/polars, 1D/2D/3D), five real datasets, the numpy-only proof, and the nasty
>   edge cases — all via one call.
> - [`cafe_benchmark.ipynb`](notebooks/cafe_benchmark.ipynb) — `cafe.benchmark()` vs causal
>   and bidirectional baselines, with cited published SOTA.

### Benchmark in one line

```python
cafe.benchmark()                 # synthetic data, CAFÉ vs baselines, printed table
cafe.benchmark(df)               # your data, scored honestly (causal vs bidirectional)
cafe.benchmark("beijing")        # real data + cited published SOTA reference rows
```

On the **Beijing Multi-Site Air-Quality** benchmark (17,117 × 132, **10% point-MCAR**,
standardised), CAFÉ — *causal, CPU-only, no training* — reaches **MAE ≈ 0.108**, in the
band of the published **bidirectional** deep models (SAITS, BRITS, Transformer) while
being the only causal one. Those deep numbers use a **different windowed train/val/test
protocol**, so the benchmark prints them as a clearly-labelled, cited **reference block —
context, not a ranked board** — and CAFÉ is not placed among them; under one source CSDI
(0.102) is lower, so no "lowest MAE" claim is made. Every deep competitor uses the
*future* to fill the past (smoothing — forbidden look-ahead in a backtest); CAFÉ does
not. The benchmark runs the simple baselines *live* on the same mask, separates **causal
vs bidirectional** tiers, and mirrors published numbers from the single registry
[`bench/refs_published.py`](bench/refs_published.py) — see
[`notebooks/cafe_benchmark.ipynb`](notebooks/cafe_benchmark.ipynb).

### Everything from one causal pass

```python
res = cafe.CAFE().run(df)

res.imputed                  # the filled data (original container)
res.uncertainty              # per-cell posterior std  (bands widen inside long gaps)
res.confidence_interval()    # (lower, upper) at 1.96 sigma
res.factors()                # latent common factors z_t  (streaming robust DFM)
res.anomaly_scores()         # per-time outlier score in [0,1] (0 = fit, 1 = outlier)
res.decompose()              # {'level','season','factor','residual'} — sums to the data
res.dependency_network()     # NxN residual-correlation network between series
res.params                   # learned dials: {'nu', 'ar', 'effective_rank'}

# forecasting == imputing future rows (AR/Kalman state), with the same model
future = cafe.CAFE().forecast(df, horizon=24)
```

### Missingness as signal (causal features)

When *where* a value is missing is itself informative (clinical panels, sensors,
financial reporting), the gap pattern is a feature — not just a hole to fill. CAFÉ
ships a **strictly forward-only** feature builder: every feature at row `t` is a
function of rows `≤ t` only (no future), so it is safe to use alongside the imputed
values in a downstream causal model.

```python
from cafe.missingness import missingness_features

# pass the original (with NaNs) OR pass mask= explicitly when the data is already filled
feats = missingness_features(df, mask=was_missing)        # same container type back
```

It emits five families per numeric column: `was_imputed` (indicator),
`time_since_obs` (BRITS-style steps since last observed), `gap_length` (current run of
missing), `missing_rate` (causal expanding fraction missing), and `selective_mim` —
indicators emitted **only for columns whose missingness is informative**, scored
leak-free by an expanding contemporaneous association test to avoid high-dimensional
MIM overfitting. Returns the same container type (`<col>__<feature>` columns), or pass
`return_meta=True` for the raw arrays plus the list of informative columns.

## More in the research harness (`bench/`)

The library is deliberately small; the empirical evidence lives in `bench/`, each
experiment self-contained, CPU-only, and run *live* (no fabricated numbers):

- **`refs_published.py`** — the single reconciled registry of *published* competitor
  numbers (one source of truth; both values kept where sources disagree).
- **`exp_seeds.py`** — multi-seed paired CAFÉ-vs-causal-baseline comparison with
  Student-t / bootstrap CIs and a paired significance test.
- **`exp_maskgrid.py`** — MAE/RMSE across mask pattern × rate (point / subsequence /
  block × 0.1/0.3/0.5), causal vs non-causal reference columns.
- **`exp_backtest_lookahead.py`** — quantifies the *decision* cost of look-ahead from
  non-causal imputation in a walk-forward backtest (CAFÉ's gap is exactly 0).
- **`exp_downstream.py`** — downstream forecasting utility under a strict temporal
  split (reconstruction MAE is neither necessary nor sufficient for downstream gain).
- **`exp_calibration_crps.py`** + **`metrics_prob.py`** — CRPS, coverage and sharpness
  for the predictive intervals (mask-aware probabilistic metrics, NLL dropped).
- **`exp_mnar_scope.py`** — MCAR→MNAR degradation and an honest scope statement of what
  self-censored values CAFÉ can and cannot recover.
- **`m_naive.py` / `online_baselines.py`** — naive and causal/online rivals (LOCF,
  seasonal-naive, GROUSE-lite, streaming EW-cov), each tagged causal / non-causal.

`bench/repro.py` lists every generator and the paper table/figure it writes;
`make repro` shows the manifest and `make repro-run` regenerates them.

## What it is (in one paragraph)

CAFÉ reads each value as **level + season + shared trend + noise**: a per-series
running level, a few Fourier waves, a handful of common factors that move many series
together, and heavy-tailed residual noise. To fill a hole it adds up the pieces it can
compute from the past and the rest of the current row — the reasoning a careful analyst
would apply, done automatically, online, and provably without peeking at the future.
The four "dials" (how many factors, how much memory, how heavy the tails, how strong
the seasonality) are learned from the data. No neural network, no training phase.

The objective and its special cases:

```
min  Σ ρ_ν( x_ti − μ_e,i − (Φ_t β)_i − (z_t Wᵀ)_i )      # robust (Student-t) fit
   + Σ_l α_l ‖W_:,l‖²       (ARD → rank)                  SoftImpute : a=0, ν→∞
   + λ_z Σ_t ‖z_t − a z_{t−1}‖²   (→ dynamics)            TRMF       : a learned
   + λ_b ‖β‖²  (→ seasonality)   + ridge(μ)  (→ FE)        Kalman/SSM : a→1
                                                          MC-NNM     : FE + low rank
   z_t = a z_{t−1} + η_t,   ε ~ t_ν(0, Ψ)                 EW-cov     : rank→0
```

## Repository layout

```
src/cafe/          the library (_core.py = the estimator, io.py = container adapters,
                   model.py = CAFE / CafeResult / impute)
src/tests/         smoke tests (container round-trip + causality verifier)
paper/             the CAFÉ paper (cafe.tex, cafe.pdf) + figures/
bench/             research harness: 22-case arena, causal verifier, robustness
                   contract, baselines, and the model under study (c_unified_penmf.py)
data/              published benchmark datasets
```

`bench/` is the research lab (benchmarks, the causal/robustness verifiers, the ablation
history); `src/cafe/` is the packaged product. Both share the same estimator.

## Guarantees

- **No look-ahead** — `src/tests/test_smoke.py::test_causality_no_lookahead` asserts past
  imputations are unchanged when the future is appended; `bench/causal.py` runs the full
  time-prefix verifier across the benchmark suite.
- **Robustness** — `bench/robustness.py` checks finite, same-shape output on every edge
  input (all-NaN, 1×1, constant, Inf, huge/tiny, wide/tall, single entity/time).

## Citation

If you use CAFÉ in your research, please cite the paper ([`paper/cafe.pdf`](paper/cafe.pdf)):

```bibtex
@misc{snow2026cafe,
  title  = {CAF\'E: Causal Adaptive Factor Estimation for Point-in-Time Imputation},
  author = {Snow, Derek},
  year   = {2026},
  note   = {https://github.com/sovai-research/cafe}
}
```

Questions or issues: [d.snow@sov.ai](mailto:d.snow@sov.ai) or open an issue.

## License

MIT.
