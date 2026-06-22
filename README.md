<h1 align="center">CAFÉ</h1>
<p align="center"><b>Causal Adaptive Factor Estimation</b><br>
Zero-config, CPU-first, <b>point-in-time</b> missing-value imputation —
with uncertainty, factors, anomalies and forecasts from a single forward pass.</p>

<p align="center">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT">
<img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg" alt="py">
<img src="https://img.shields.io/badge/deps-numpy%20%2B%20scipy-informational.svg" alt="deps">
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
- **CPU-first** — pure `numpy`/`scipy`, no GPU, no training run. Runs the full
  benchmark suite in ~1 s.
- **Container-native** — `numpy`, `pandas`, `polars`, 1D or 2D, dtype/labels preserved.
- **More than imputation** — the same pass yields per-cell uncertainty, latent
  factors, anomaly scores, an additive decomposition, a dependency network and forecasts.

On the standard Beijing Air-Quality benchmark (SAITS protocol), CAFÉ reaches the
**lowest MAE (0.111)** — below SAITS (0.137), BRITS (0.153) and the Transformer
(0.158) — while being the only **causal, CPU-only** method (the deep baselines are
bidirectional and GPU-trained). See [`paper/cafe.pdf`](paper/cafe.pdf).

## Install

```bash
pip install -e .            # core (numpy + scipy)
pip install -e ".[all]"     # + pandas, polars, matplotlib
```

## Quick start

```python
import cafe

# zero-config — same container type comes back, gaps filled, no look-ahead
filled = cafe.impute(df)            # pandas / polars DataFrame, or numpy array, 1D or 2D
```

### Everything from one causal pass

```python
res = cafe.CAFE().run(df)

res.imputed                  # the filled data (original container)
res.uncertainty              # per-cell posterior std  (bands widen inside long gaps)
res.confidence_interval()    # (lower, upper) at 1.96 sigma
res.factors()                # latent common factors z_t  (streaming robust DFM)
res.anomaly_scores()         # per-time outlier score from the Student-t weights
res.decompose()              # {'level', 'season', 'factor'} additive parts
res.dependency_network()     # NxN residual-correlation network between series
res.params                   # learned dials: {'nu', 'ar', 'effective_rank'}

# forecasting == imputing future rows (AR/Kalman state), with the same model
future = cafe.CAFE().forecast(df, horizon=24)
```

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

## License

MIT.
