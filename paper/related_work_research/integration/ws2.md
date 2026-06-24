# WS2 integration note — Gap #5: causal split-conformal recalibration

Turns CAFE's admitted over-conservative intervals into calibrated ones. **Sigma-only**
(imputation MAE provably unchanged), **strictly point-in-time** (truncation-invariant),
distribution-free flavour. All numbers below are from a live `bench/exp_conformal.py`
run (3 seeds, 10% MCAR, datasets shared with `exp_calibration_crps.py`).

Headline (live): mean |coverage − nominal| **0.095 → 0.019**; MAE Δ = **0** on every
dataset; prefix-vs-full band max diff = **0** (point-in-time).

---

## 1. New artifact: `\input{tables/conformal}`

Add the line **immediately after** `\input{tables/calib_crps}` (cafe.tex line 619), at
the end of the **"Uncertainty (CRPS, coverage, sharpness)"** paragraph:

```latex
\input{tables/calib_crps}
\input{tables/conformal}        % <-- WS2: causal conformal recalibration
```

`paper/tables/conformal.tex` is self-contained (booktabs `table*`, `\label{tab:conformal}`,
caption states the protocol). Label to cross-reference: `tab:conformal`.

## 2. Paragraph edit — weakness → strength

The current paragraph ends by conceding the bands are "safe-but-wide … not yet sharply
calibrated … tightening … is explicit future work." Append (or replace that last clause
with) the following, turning the concession into a delivered result:

> A thin causal recalibration layer closes this gap. We rescale the per-cell $\sigma$ by
> a single split-conformal multiplier $q(\text{level})$—the $(1{-}\alpha)$ empirical
> quantile of held-out calibration residuals collected by a second strictly online pass
> (a fresh calibration mask hides a fraction of observed cells, which are then predicted
> exactly as missing cells are, so their normalized residuals are exchangeable with the
> test cells). The recalibrated band $\mu\pm q(\text{level})\,\sigma$ hits nominal
> coverage—mean $|\text{coverage}-\text{nominal}|$ drops from $0.095$ to $0.019$ across
> datasets and levels—with materially tighter intervals, while the imputation $\mu$ is
> bit-identical (the change is $\sigma$-only) and the multiplier at time $t$ uses only
> calibration residuals from rows $<t$, so the layer is as point-in-time as \cafe{}
> itself (Table~\ref{tab:conformal}). This is the causal analogue of conformalized matrix
> completion~\cite{gui2023cmc} and of conformal prediction under missing
> values~\cite{zaffran2023cpmv}, neither of which is online/point-in-time.

(If the paragraph's "future work" sentence is kept elsewhere, soften it: the *absolute
calibration* is now delivered; only per-pattern conditional calibration remains open.)

## 3. `bibitem` entries (verified arXiv IDs)

Add to the bibliography in `paper/cafe.tex` (same `\bibitem` style as `mazumder2010` etc.):

```latex
\bibitem{zaffran2023cpmv} M.~Zaffran, A.~Dieuleveut, J.~Josse, Y.~Romano.
  Conformal prediction with missing values. \emph{ICML}, 2023. arXiv:2306.02732.
\bibitem{gui2023cmc} Y.~Gui, R.~Foygel~Barber, C.~Ma. Conformalized matrix
  completion. \emph{NeurIPS}, 2023. arXiv:2305.10637.
```

## 4. `bench/repro.py` MANIFEST entry

Add to the `MANIFEST` dict (alongside the other `exp_*.py` entries):

```python
    "exp_conformal.py": {
        "tables":  ["paper/tables/conformal.tex"],
        "figures": [],
        "tex_label": ["tab:conformal"],
        "what": "Causal split-conformal recalibration: raw vs conformal PICP/sharpness "
                "at 50/80/90/95, MAE unchanged, point-in-time preserved.",
    },
```

## 5. Notebook cell snippet (calibrated intervals)

```python
import cafe
res = cafe.CAFE().run(df)               # df: 1D/2D numpy / pandas / polars with NaNs

# raw (over-conservative) band vs. causal split-conformal calibrated band
lo, hi   = res.calibrated_interval(0.90)        # mu ± q(0.90)·sigma, point-in-time
halfwidth = res.calibrated_uncertainty(0.90)    # q(0.90)·sigma per cell (NaN where observed)

# the imputation itself is untouched — calibration only reshapes the band width:
assert (res.imputed == cafe.CAFE().run(df).imputed).all().all()
```

## Public API added (backward-compatible)

- `cafe.CafeResult.calibrated_interval(level=0.9, cal_rate=0.15, cal_seed=0, window=4000, min_scores=30) -> (lo, hi)`
- `cafe.CafeResult.calibrated_uncertainty(level=0.9, ...) -> half-width container`
- `cafe.ConformalCalibrator`, `cafe.conformal_multipliers` (module `cafe.conformal`)
- New files: `src/cafe/conformal.py`, `src/tests/test_conformal.py`, `bench/exp_conformal.py`.
- Edited: `src/cafe/model.py` (added the two methods + lazy calibrator cache),
  `src/cafe/__init__.py` (exports). `src/cafe/_core.py` is **unchanged**.

## Honest caveats

- Calibration is excellent at 80/90/95; at the tightest **50%** level the conformal band
  slightly *under*-covers on the synthetic/Beijing panels (≈0.46–0.53 vs 0.50) — the
  median is the hardest quantile to pin with a finite window, and it errs tight rather
  than wide. Still far closer to nominal than the raw band (which over-covers 50% at
  ≈0.64–0.88).
- The layer needs the **traced** run (`CAFE().run`, not `impute`) and is **1D/2D only**;
  panel calibration raises `NotImplementedError` (future work).
- The calibration pass runs CAFE a second time (one extra forward pass); results are
  deterministic via `cal_seed`.
