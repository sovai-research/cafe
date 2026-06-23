# Hyperparameter & Configuration Disclosure

This document discloses, for every method that appears in the CAFE paper or
benchmark, exactly what was tuned and by whom. It exists to defuse the most common
reproducibility objection to imputation papers: that a "best-reported" number was
cherry-picked from a large hyperparameter sweep. The benchmark-hardening literature
(e.g. TSI-Bench, arXiv:2406.12747) documents that strong deep baselines are routinely
selected from **on the order of 100+ configurations per method per dataset**, and that
the *best* such config is reported. CAFE's entire claim rests on the opposite stance:
**the user sets nothing.**

Regenerate this disclosure's companion machine manifest with:

```
python3 bench/repro.py            # dry: print experiment -> artifact manifest
python3 bench/repro.py --run      # regenerate every table/figure from a live run
python3 bench/repro.py --json     # manifest as JSON
```

---

## 1. CAFE — user-facing hyperparameters: NONE (zero-config)

The public API is literally a single call with no tunable arguments:

```python
import cafe
filled = cafe.impute(data)          # numpy / pandas / polars, 1D or 2D
```

`cafe.impute(data, meta=None)` (src/cafe/model.py) and the underlying
`online_impute(X, meta)` (bench/c_unified_penmf.py) expose **zero accuracy
hyperparameters to the caller**. `meta` only declares panel structure (which columns
are entity/time keys); it does **not** carry any knob that trades off accuracy. There
is **no rank to pick, no learning rate, no window to choose, no regularization weight
to set, no model to select**. The same call is used unchanged across every dataset,
missing-rate, and mask pattern in the paper.

### Why there is nothing to tune (it is adaptive, not fixed-by-luck)

CAFE's structural choices **emerge from the data** via penalties rather than being
selected by the user or by a validation sweep:

- **Rank** emerges from an ARD prior over factor columns (`alpha_l ||W[:,l]||^2`):
  unneeded columns are shrunk to zero, so the effective rank is data-driven, not set.
- **Dynamics** (the AR coefficient `a`) and **noise scale** (`nu`) are estimated online
  from data `<= t`.
- **Seasonality** emerges from a Fourier ridge whose ARD shrinks unused harmonics; the
  candidate periods are fixed semantic cycles (e.g. daily/weekly), not tuned per dataset.
- **Fixed effects** (`mu`) are estimated, not configured.

### Internal constants (library-fixed, identical across all experiments)

For full transparency: the core does contain a handful of **internal constants** that
are set **once by the library author and never exposed to or varied by the user**. They
are the same value in every paper experiment. They are engineering defaults of the
online estimator, not per-dataset accuracy knobs:

| Constant       | Value (fixed) | Role |
|----------------|---------------|------|
| `WINDOW`       | 400 rows      | trailing causal window for periodic factor refits |
| `HALFLIFE`     | 200 rows      | exponential forgetting for pooled scale/covariance |
| `MU_HALFLIFE`  | ~1e5 (≈static)| forgetting for the fixed-effect / level term |
| `REFIT_EVERY`  | (fixed)       | how often column factors `W` are recomputed |

These are **not swept**. There is no script in `bench/` that grids over them to pick a
winner; changing them is a library-internal engineering decision, not a user action.
This is the qualitative difference from "best-reported": a CAFE user cannot, even in
principle, obtain a different number by tuning, because there is nothing exposed to tune.

### Causal / point-in-time guarantee

Every parameter (`W`, `a`, `nu`, scales, ARD `alpha`s, `mu`) at time `t` is fit using
**only rows at times `<= t`** plus the contemporaneous observed cells of row `t`. This
is verified by `bench/exp_leakage.py` (tab:leakage), `bench/exp_causalverify.py`
(tab:causalverify), and `bench/exp_backtest_lookahead.py` (tab:backtest), all wired into
the reproduction manifest.

---

## 2. Classical CPU baselines we run LIVE (numpy/scipy)

These are run by us on identical data/masks as CAFE, so their configs are disclosed in
full. Where a baseline has a nominal knob, we use a single fixed default for all
experiments (no per-dataset tuning), matching the zero-config spirit:

| Baseline        | Tunable? | Setting used (fixed across all experiments) |
|-----------------|----------|----------------------------------------------|
| global mean / mean-of-past | no | column mean (causal: mean of observed `<= t`) |
| Median          | no       | column median |
| LOCF / NOCB     | no       | last/next observed carry |
| linear interp   | no       | endpoint linear interpolation |
| Holt (level+trend) | implicit | standard smoothing, fixed defaults |
| Kalman filter   | implicit | local-level/trend, fixed defaults |
| SVDImpute       | rank     | fixed rank, same for all datasets |
| SoftImpute      | λ        | single fixed λ; **not** per-dataset tuned |
| TRMF (if run)   | rank, λ  | single fixed config; reimplemented in numpy |

We **never** train or call any competitor deep model (SAITS / BRITS / CSDI / FGTI /
ImputeFormer / GP-VAE / M-RNN / iTransformer / PriSTI). Those appear only as published
numbers (Section 3).

---

## 3. Published deep-SOTA numbers — sourced, NOT re-run, DIFFERENT protocol

CAFE is compared to deep methods **only via numbers published in their own papers**,
clearly labelled, and placed in a **separate reference column** — never on a single
ranked leaderboard with CAFE bolded as the winner. These methods are all
**bidirectional + GPU**, and (critically) were produced under a **best-of-many-configs**
selection: their reported numbers are the *best* of large per-dataset hyperparameter
sweeps (the regime TSI-Bench documents). CAFE's number is its *only* number.

### 3a. Protocol mismatch (must be stated wherever these numbers appear)

The published deep numbers use a **windowed protocol with a train/validation/test
split** on their own preprocessing of Beijing Air-Quality (e.g. fixed-length windows,
held-out test cells). CAFE's headline number instead imputes the **full series causally
/ online** — no windows, no train/test split. These are **different settings on
related data and must not be ranked head-to-head as a single board.**

CAFE's headline Beijing number (`tab:sota`) is produced by
`bench/make_paper.py::_beijing_sota_mae` and its exact protocol is:

- **Data**: `data/beijing_clean.npy`, the longest fully-observed contiguous slice,
  shape **17117 × 132** (`bench/prep_real.py`).
- **Normalization**: per-column z-score applied **once** (`(X-mean)/(std+1e-9)`,
  prep_real.py); CAFE does not re-standardize internally.
- **Mask**: 10% MCAR point, `np.random.default_rng(seed)` for `seed in {0,1,2}`.
- **Aggregation**: mean MAE over the 3 seeds, on the standardized scale, over masked
  cells only.
- **Model**: CAFE imputes the full series causally/online (no windows, no split).

This protocol is **not bitwise identical** to any published deep result, by construction.

### 3b. THE ONE published-number registry (reconciliation)

The repo previously carried **two contradictory** published-number sets:
`paper/cafe.tex` (sourced to Du et al. 2023 / SAITS) and `src/cafe/benchmark.py`
(sourced to TSI-Bench). They disagree on every shared method. Going forward there must
be **one source per dataset**. The two candidate registries are recorded here verbatim
so the discrepancy is documented and a single choice can be enforced across
`cafe.tex`, `bench/fig_benchmark.py`, and `src/cafe/benchmark.py`.

**Registry A — `paper/cafe.tex` tab:sota / `bench/fig_benchmark.py`**
Source cited: *Du, Côté, Liu — SAITS, 2023* (`du2023`). Beijing, standardized MAE @ 10% MCAR:

| Method      | MAE  | Dir.  | HW  |
|-------------|------|-------|-----|
| Median      | .763 | —     | —   |
| M-RNN       | .294 | bidir | GPU |
| GP-VAE      | .268 | bidir | GPU |
| Transformer | .158 | bidir | GPU |
| BRITS       | .153 | bidir | GPU |
| SAITS       | .137 | bidir | GPU |

**Registry B — `src/cafe/benchmark.py`**
Source cited: *TSI-Bench, arXiv:2406.12747 (preprint)*. Beijing, standardized MAE @ 10% MCAR:

| Method       | MAE  | Dir.  | HW  |
|--------------|------|-------|-----|
| CSDI         | .102 | bidir | GPU |
| iTransformer | .123 | bidir | GPU |
| BRITS        | .127 | bidir | GPU |
| Transformer  | .142 | bidir | GPU |
| SAITS        | .155 | bidir | GPU |

**Discrepancy (must be reconciled to ONE source):** the two registries disagree on
every shared method (BRITS .153 vs .127; SAITS .137 vs .155; Transformer .158 vs .142)
and cite different papers. Under Registry B, **CSDI (.102) is lower than CAFE (~.108)**,
so any "lowest MAE" phrasing depends entirely on which registry is chosen. Because the
protocols differ from CAFE's anyway, the honest framing is: **CAFE is competitive with
bidirectional GPU deep SOTA while being the only causal, CPU-only, zero-config method —
not "it beats everything on a shared board."**

> ACTION (owners of cafe.tex, benchmark.py, fig_benchmark.py): pick ONE registry above
> (TSI-Bench, Registry B, is the single coherent modern table and is recommended) and use
> it consistently. This file documents both only to make the contradiction explicit; it is
> not itself the live registry.

---

## 4. The defensible headline ("two-of-three")

No prior published method is simultaneously **(causal / point-in-time)** +
**(CPU-only)** + **(competitive with bidirectional deep SOTA)**. The strongest
baselines (ImputeFormer, FGTI, CSDI, PriSTI, SAITS, BRITS, GP-VAE) are all
bidirectional + GPU and best-reported from large sweeps. CAFE is the first to credibly
claim all three at once, with **zero user-set hyperparameters**. Reproducibility of
every CAFE number (Section 1 protocols, full manifest in `bench/repro.py`) is what
turns "beats everything" suspicion into a checkable claim.
