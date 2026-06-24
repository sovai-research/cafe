# Is CAFÉ a better GARCH? — an empirical test

This folder tests, with controlled simulations where the truth is known, the
"CAFÉ-as-a-volatility-model" thesis: that CAFÉ is *not* a better univariate GARCH,
but that its factor + missing-data structure opens a genuine niche (scalable factor
volatility on incomplete high-dimensional panels), and that one score-driven
recursion — built from machinery CAFÉ already computes — turns it into a contender.

All forecasts are strictly one-step-ahead / point-in-time (the variance forecast for
return `r_t` uses only data `< t`), matching how both GARCH and CAFÉ operate.

## How to reproduce

```bash
python scripts/exp1_univariate.py    # H1: CAFÉ as-is vs GARCH/EWMA (univariate)
python scripts/exp2_betatgarch.py    # H4: the Beta-t-GARCH upgrade
python scripts/exp3_panel.py         # H3: high-dim / incomplete panels vs DCC
python scripts/make_figs.py          # figures/ from the saved data/*.json
```

`garch_lib.py` (univariate DGPs, CAFÉ-scale isolation, Beta-t-GARCH, metrics) and
`mvol_lib.py` (DCC/CCC, the CAFÉ-FV factor-vol prototype, baselines) are shared.

## Verdict — the thesis holds, with the nuance it predicted

| Claim | Result | Evidence |
|---|---|---|
| **H1** CAFÉ as-is is *not* a competitive univariate vol model | **CONFIRMED** | Worst of all methods; QLIKE +0.11 (Gaussian) / +0.28 (t) over GARCH-MLE; variance-MSE 2.2–5.3 vs ≈0.01–0.04 |
| ↳ because its scale half-life (200) is far too slow | **CONFIRMED** | Optimal EW half-life is ≈22; HL=200 sits far up the wrong side of the QLIKE curve |
| ↳ and it lags spikes / barely mean-reverts | **CONFIRMED** | Never reaches half a vol-spike within 400 steps (EWMA: 8); reverts in 140 (EWMA: 13) |
| **H2** "causal" buys nothing here (GARCH is already point-in-time) | **CONFIRMED (by construction)** | The whole comparison is one-step causal; CAFÉ's no-look-ahead moat is null against GARCH |
| **H3** the real niche: scalable factor vol on incomplete high-dim panels | **CONFIRMED, decisively** | See below — CAFÉ-FV beats DCC/CCC on the economic loss, survives the curse of dimensionality, ingests gaps natively, and is flat in N |
| **H4** the Beta-t-GARCH upgrade (innovation = CAFÉ's own IRLS weight) makes it a contender | **CONFIRMED** | Beta-t innovation `w·r²` is *byte-identical* to the library weight; matches GARCH-MLE on clean data and **beats** it under outliers |

### H1 — univariate (exp1, `data/exp1_univariate.json`, `figures/fig1_univariate.pdf`)

GARCH(1,1) sims, 8 seeds, QLIKE vs the **true** latent variance (lower = better):

| model | QLIKE (t-innovations) | gap vs GARCH-MLE | variance-MSE |
|---|---|---|---|
| GARCH-t MLE (gold standard) | 1.682 | — | 0.030 |
| GARCH-N MLE | 1.683 | +0.001 | 0.044 |
| EWMA(0.94) | 1.706 | +0.024 | 0.329 |
| **CAFÉ scale, HL=200** | **1.962** | **+0.280** | **5.269** |

CAFÉ's variance machinery — verified **bit-identical** to the library's
`_UnifiedCore._update_robust_scale` (`cross_check_cafe_scale → 0.0e+00`) — is a
robust EWMA with half-life 200. That is an excellent *level* estimator and a poor
*volatility* estimator: it cannot track clustering and has no long-run level to
revert to. This is exactly what the thesis said, measured.

### H4 — the upgrade (exp2, `figures/fig2_betatgarch.pdf`)

A Beta-t-GARCH variance recursion `h_{t+1} = ω + α·(w_t r_t²) + β·h_t`, whose
innovation `w_t = (ν+1)/(ν + r_t²/h_t)` is **exactly** the IRLS weight CAFÉ already
computes (identity verified to `0.0e+00` against the library's `_wt`):

- **Clean GARCH-t data:** Beta-t-GARCH QLIKE **1.688** vs GARCH-MLE **1.682** — it
  closes ~98% of the gap CAFÉ-as-is leaves, and tracks spikes in 17 steps (CAFÉ-as-is:
  never).
- **Additive-outlier data:** Beta-t-GARCH **1.784** *beats* GARCH-t **1.885** and
  Gaussian GARCH **1.927**. After an outlier its variance over-shoots the truth ×2.6
  vs Gaussian GARCH's ×6.0 — the score self-down-weights the jump.

The upgrade is genuinely incremental: the robust, outlier-resistant volatility
innovation is already half-built in CAFÉ; you add only a reaction `α`, a persistence
`β`, and a long-run level `ω`.

### H3 — the niche: high-dimensional & incomplete panels (exp3, `figures/fig3_panel.pdf`)

Factor-GARCH panels (true `Σ_t = B diag(h^f) Bᵀ + diag(h^ε)`). `CAFÉ-FV` is the
proposed low-rank-plus-diagonal Beta-t factor-vol model (a prototype of the upgrade,
*not* the shipped library). Loss = out-of-sample minimum-variance-portfolio realized
variance (the standard MGARCH economic loss), lower = better.

- **Complete data (N=20):** CAFÉ-FV **0.192** (oracle 0.185) < Sample 0.206 < DCC
  0.214 < CCC 0.221 < EWMA 0.304. Honest nuance: DCC wins the *statistical*
  covariance QLIKE (1.14 vs 1.24), but CAFÉ-FV's well-conditioned low-rank inverse
  wins the *economic* portfolio loss.
- **Curse of dimensionality (T_train=350, N→250):** the sample-correlation condition
  number explodes 33 → 17 200; CCC blows up (0.082 at N=80 → 1.07 at N=150) and DCC
  becomes infeasible, while CAFÉ-FV keeps *improving* with N (0.045 → 0.016), hugging
  the oracle.
- **Asynchronous gaps (16% missing):** CAFÉ-FV native **0.134** wins. DCC with a
  naive mean-impute front-end is catastrophic (0.82); DCC with a **CAFÉ-impute**
  front-end recovers to 0.154 — so CAFÉ adds value both *as* the covariance model and
  *as* a gap front-end for classical MGARCH.
- **Runtime:** CAFÉ-FV fit is flat in N (~0.55s, only K factor MLEs) with O(N·K)
  per-step filtering; DCC fit grows 0.9s → 8.3s (N=100) and is infeasible beyond.

## Bottom line

The reviewer's analysis is correct and now measured: **CAFÉ as written is a poor
univariate volatility model** (and "causal" is no advantage against GARCH), but it
sits **one score-driven recursion** — a recursion whose key quantity it already
computes — away from a **robust, scalable factor-volatility model for incomplete
high-dimensional panels**, a genuinely under-occupied niche where DCC/BEKK fail. This
is a financial-econometrics story (JFE / JAE / IJF), not a generic-imputation one.

### Caveats (honesty)
- DGPs are simulated factor-GARCH; real equity panels add leverage/asymmetry,
  intraday seasonality and microstructure not modelled here.
- `CAFÉ-FV` is a faithful prototype of the proposed upgrade (PCA loadings + Beta-t
  factor/idiosyncratic vols), not the shipped `src/cafe` code; turning it into a
  point-in-time online estimator inside `_UnifiedCore` (swapping the HL=200 EW scale
  for the score recursion) is the actual implementation step.
- Idiosyncratic vols in CAFÉ-FV use a fixed RiskMetrics-like reaction/persistence
  (method-of-moments level) so only the K factors are MLE-fit — this is what makes it
  O(K) fits and is stated, not hidden.
