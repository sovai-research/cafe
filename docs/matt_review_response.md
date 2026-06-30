# Response to Matt's CAFÉ review — and where we take it next

*Derek → Matt (with Eeshaan cc'd for the hand-off items in §C).*
Everything below is grounded in the current `src/cafe` model and `paper/cafe.tex`; the FX
numbers are fresh runs (`experimental/papers/position/scripts/coint_probe.py`).

Thanks — and yes, the favourite-graph one (the causal **leaderboard-flip**,
`figures/causal_race_flip.pdf`, and its closed-form twin `figures/gap_theory.pdf`) is the
one I'd lead the talk with too. It *is* the hook, and it's worth saying why below.

---

## A. The three questions you raised, answered

### 1. Can CAFÉ do error / outlier detection — "given a result, did it come from the series"?

**Yes, and it's already a first-class read-out, not a bolt-on.** This is the single most
natural extension of what CAFÉ is, because CAFÉ is a *generative* model of the series:
`x_{t,i} = μ + season + Wz_t + ε`, with `ε ~ Student-t_ν`. So for any value — observed or
proposed — it computes a point-in-time **conditional likelihood**: how surprising is this
cell given everything ≤ t and the contemporaneous cross-section. Two concrete forms ship
today:

- **Per-row outlier score** `res.anomaly_scores()` ∈ [0,1]: the bounded transform of the
  Student-t IRLS weight, `u/(ν+u)` with `u` the standardised squared residual — exactly 0
  at a perfect fit, → 1 as the residual blows up. Causal (row *t* uses only data ≤ t).
- **Per-cell residual / posterior σ**: the same machinery at cell granularity.

Your framing — "given a result, evaluate whether it came from the time series" — is *the
likelihood-ratio / conformal p-value reading of that score*, and it's a clean way to pitch
it: CAFÉ is a **point-in-time novelty detector** for streaming data. Feed it a candidate
value and it returns a calibrated "does this belong" score, with the same no-look-ahead
guarantee that makes it backtest-safe.

**Is it good?** We already validate it (Table `byproduct.tex`): on a synthetic panel with
injected point spikes + contiguous bursts, the *free* score scores **ROC-AUC 0.963** vs a
**dedicated** causal rolling-Hampel/MAD detector's 0.991. So it's ~97% of a purpose-built
detector's quality, *for free, from the imputation pass*, while the Hampel detector does
nothing else.

→ **On your "another paper, or just prove it can be used?"** My strong recommendation:
**not a separate paper — a hardened by-product section + a worked notebook.** The honest
scientific claim is "outlier detection falls out of the same causal pass at ~parity with a
specialist," not "we beat the anomaly-detection SOTA" (we haven't run that gauntlet, and it
has its own TSI-Bench-style minefield). The *positioning* — one causal model that imputes,
flags bad data, and tells you which fills it can't trust (the recoverability certificate) —
is the differentiator. Benchmarking it as a headline detector is a different, riskier paper;
proving it's a credible **data-quality / error-detection** read-out is a one-week hardening
job (see §C, E1).

### 2. Why we lose on FX — and are we "doomed by lack of cointegrating pairs?"

Your instinct about cointegration is **exactly right**, and it's worth making precise
because it generalises cleanly to the rest of finance.

CAFÉ's edge is a **shared low-rank factor** that lets the *contemporaneous cross-section*
fill a gap better than a series' own last value. A rank-R factor model on non-stationary
(I(1)) series **is the common-trends representation of a cointegration system**: if N series
share R < N common stochastic trends, there are N−R cointegrating vectors, and the
idiosyncratic part (what CAFÉ models with ψ + the AR carry) is **stationary → mean-reverting**.
That stationarity is *why* the cross-section anchors a missing series across a gap. When the
idiosyncratic part is itself a unit root (no cointegration), there is nothing to revert to and
**last-value is Bayes-optimal causally** — CAFÉ can't beat it and shouldn't try.

FX majors are the worst case for us: 8 floating rates, each ≈ its own unit root.
Measured on `data/exchange_clean.npy`: **level AR(1) = 0.999, return AR(1) = −0.12** (≈
unpredictable). CAFÉ's router *detects* this — it learns `a=0.96, ν=60, eff-rank 2` (Table
`router.tex`), the "shared-structure assumption does not hold, defer to per-entity carry"
signature — and the paper already reports the honest loss (FX causal MAE 0.50 vs a Kalman
local-level's 0.14).

**I ran the cointegration claim as a controlled experiment** (block gaps, len 30, N=12,
`coint_probe.py`). Sweeping the number of common trends `r` (cointegration rank = N−r):

| common trends r | cointegration rank | CAFÉ | LOCF | CAFÉ/LOCF |
|---:|---:|---:|---:|---:|
| 12 | 0 | 0.372 | 0.281 | **1.32** (LOCF wins) |
| 8 | 4 | 0.282 | 0.222 | 1.27 |
| 4 | 8 | 0.279 | 0.259 | 1.08 |
| 2 | 10 | 0.215 | 0.258 | **0.83 (CAFÉ wins)** |

CAFÉ's relative position improves **monotonically with cointegration rank**, crossing into a
win once the cross-section is genuinely informative. With a *non*-mean-reverting idiosyncratic
(φ=0), CAFÉ ≈ LOCF regardless of `r`. And **real FX sits at the no-cointegration corner**:
block-gap ratio **2.40**, scattered-gap ratio **5.69** — far at the LOCF-wins end, just like
the synthetic `rank-0` row.

So the precise answer: **we are not "doomed on finance," we are correctly beaten on
independent random-walk *levels*.** The discriminating property is *cointegration / a
stationary cross-sectional structure*, not "is it financial." That flips into a positive
claim about where we **should** win (§3 of the hook, and §C E2).

### 3. The hook — where, in a financial series, would this actually matter?

The position paper contemplates it; here's the one-sentence version I'd put on the slide:

> **The cost of look-ahead in imputation is a *property of the data's autocorrelation*, we
> derived it in closed form, and it is exactly the regimes quant finance cares about where
> the leak is largest — so a causal imputer isn't a nicety, it's the only honest way to build
> features for a backtest.**

Three layers to it, strongest first:

1. **The leakage moat is the hook, and finance is its home turf.** Almost every imputer
   (interpolation, SoftImpute, every deep model — SAITS/BRITS/CSDI) fills `x_t` using the
   future. Drop that into a trading backtest and you've leaked tomorrow into today's feature.
   Bryzgalova–Lerner–Lettau–Pelger (*Missing Financial Data*, RFS 2025) documents exactly this
   inflation for cross-sectionally imputed firm characteristics. CAFÉ is the imputation analogue
   of their fix: `max_revision = 0` by construction, and our **Online Downstream Test** shows the
   non-causal methods' apparent edge *evaporates live* while CAFÉ's reported score = its live score
   (`online_downstream.tex`). **That is the graph you liked** — the leaderboard flips when you
   forbid the future, and the only board valid for a backtest is the causal one.

2. **The closed-form gap theory says the leak is biggest precisely in financial-style data.**
   Δ(a,g) = √(2/π)·mean_k(√V_f − √V_s) with V_f, V_s the exact filter/smoother variances. The leak
   → 0 for memoryless data and peaks at moderate-to-high autocorrelation — i.e. persistent series.
   Macro/rates/credit/vol all live there. (FX is the curiosity: a≈1 but the leak is *also* small at
   short gaps because the correlation time 1/(1−a) is huge — high persistence, but nothing
   cross-sectional to steal.)

3. **The positive niche (where we WIN, not just "don't leak"):** financial panels with genuine
   *stationary cross-sectional* structure — **yield-curve tenors** (level/slope/curvature, tenors
   cointegrated), **CDS / credit curves**, **implied-vol surfaces** (level/skew/term factors; a
   missing strike pinned by its neighbours), **futures term structures**, **equity characteristics
   panels** (the Bryzgalova setting), and **cross-rates** (triangular arbitrage = exact
   cointegration). These are the matched regime; FX *spot levels* are the anti-regime. The volatility
   work already points at one concrete downstream version of this: robust, point-in-time **factor
   covariance on incomplete high-dimensional panels** (beats Ledoit–Wolf NLS on covariance-QLIKE in
   the simulated factor-GARCH study; competitive-not-dominant on real FF49 — keep it as a
   *capability demo*, not a "we beat DCC" claim).

---

## B. The sharper mechanical questions

### How do μ, season, and factor get fit — and does the factor have an intercept?

Your read is **correct: the factor carries no intercept, because that *is* μ.** The fit is
sequenced so the terms don't fight over the mean (`_core.py:process_row`):

1. **μ (level)** — per-feature fixed effect, an expanding/EW mean with a *very* long half-life
   (`MU_HALFLIFE≈1e5`, i.e. near-static). This is the only term that carries the absolute level.
2. **time-FE** — a *contemporaneous* shared level: the robust (trimmed), shrunk cross-sectional
   mean of the de-μ residual at time t. Catches a coherent regime shift the slow μ hasn't absorbed
   yet (the MC-NNM time-FE term). Point-in-time (same-t cross-section only).
3. **factor `Wz_t`** — fit on the residual **after μ and time-FE are removed**, so it is
   **mean-zero by construction**. No intercept: an intercept would be unidentified against μ +
   time-FE. ARD on the columns of W lets the effective rank emerge.
4. **season `Φ_t β`** — fit by online ridge (RLS) on the **factor residual** `x − μ − g·Wz − time_FE`
   with `g = N/(N+R)`, so a cycle the shared factors already explain is *claimed first and not
   double-counted*; ARD per harmonic shrinks unsupported cycles to ~0.

So the identification order is **level → contemporaneous level → mean-zero factor → residual
season**, each fit on the previous one's residual. That's also why "no seasonality / no factor"
are the *learned defaults*, not switches.

### FX as a random walk — "but they trend, and carry is a stable predictor"

Two distinct points, both fair:

- **"They trend"** → a trend in levels is a near-unit-root, and CAFÉ represents it as the AR
  state with `a→1` (the Kalman/random-walk corner) plus the idiosyncratic carry — *not* as
  mean-reversion. So trending FX doesn't get pulled to a mean; it gets carried forward. That's
  also why on FX we essentially *become* last-value (and a dedicated last-value does it a hair
  better — "model gap, not look-ahead," the triangles above the optimal-filter line in
  `gap_theory.pdf`).
- **"Carry is a stable predictor"** → this is the real, honest limitation: **carry is an
  exogenous covariate, and CAFÉ is covariate-free.** It absorbs static entity effects only
  through μ; it does not yet condition the state or loadings on an external regressor like the
  rate differential. So the one thing that genuinely *does* predict FX, we currently can't use.
  That's a clean, motivated extension (§C E3: covariate-conditioned state) — and notably it's the
  honest reason we lose on FX *even versus a smart prior*, distinct from the cross-section reason.

### Trends + expanding window + seasonality → "the long way around mean-reversion"?

Good worry, and we engineered against exactly it. Two guards:

- **The seasonal basis is a *fixed semantic menu* (7,12,24,48,168,365), never window
  subharmonics.** Subharmonics of the window would happily fit a smooth trend and then
  extrapolate wildly across a block gap — the failure you're gesturing at. By forbidding them
  (and gating each harmonic until ≥2 cycles of support accrue), season can't impersonate a trend.
- **μ is near-static (long half-life), so it is *not* a fast mean the series gets reeled back
  to.** Trend/drift is carried by `a→1` + the idiosyncratic AR carry, which extrapolate the last
  level rather than reverting it.

The residual tension you're sensing is real but bounded: on a *pure single trending series with
no cross-section*, CAFÉ is a deliberately-general (hence slightly sub-optimal) causal estimator and
a drift/last-value model edges it — we show this openly (ETTh, FX). Where there's a cross-section,
the factor handles the co-trending and the worry doesn't bite.

### Fat tails: will ν = 4 + 6/κ hold?

This is the one I'd flag a **genuine caveat** on, and you've put your finger on the soft spot.

The map comes from the Student-t excess-kurtosis identity κ = 6/(ν−4) ⇒ **ν = 4 + 6/κ**, valid
**only for ν > 4** (the regime where the fourth moment exists). We estimate κ online from EW
residual moments and invert it, then **clamp ν ∈ [2.5, 60]**. Implications:

- For **moderately** heavy tails (ν≈5–8 — most macro/sensor outlier data, COVID-era spikes) the
  estimator is well-behaved and the ablation shows it pays for itself (removing Student-t nearly
  doubles MAE on heavy-tailed data, 1.16→2.10).
- For **truly fat-tailed financial returns** (ν often ≈ 3–4, sometimes < 3), the *sample
  kurtosis is unstable or undefined*, so the moment-inversion is unreliable below ν≈4. We don't
  blow up — we floor at 2.5 and the **redescending IRLS weights still down-weight tails correctly**
  (robustness degrades gracefully) — but the *dof estimate itself* shouldn't be trusted as a
  tail-index in that regime, and I wouldn't market ν as a calibrated tail measure for raw daily
  returns. The volatility work confirms the *weight* is right (the Beta-t-GARCH innovation is
  byte-identical to CAFÉ's IRLS weight and matches GARCH-MLE on clean Student-t data), but that's
  the down-weighting, not the κ→ν inversion.
- Honest fix if we want a defensible tail estimate on returns: replace the kurtosis-moment
  estimator with a **profile/EM MLE for ν** (robust to the undefined-fourth-moment regime). Small,
  contained change (§C E4).

---

## C. Where this leaves us, and the hand-off to Eeshaan

**Already folded into the paper (`cafe.tex`), with reproducible generators:**
- **The cointegration dial** — `bench/exp_cointegration.py` → `tables/cointegration.tex` +
  `figures/cointegration.pdf`, new paragraph after the router table. Turns the FX loss into a
  named law (cross-section helps ⟺ cointegration) and adds a *real financial win* (FRED-MD
  yield curve: CAFÉ 0.097 vs LOCF 0.217) opposite the FX loss.
- **Fat-tail honest scope** — `bench/exp_fattail.py` → `tables/fattail.tex`, new paragraph after
  the ablation. Answers Matt's `ν=4+6/κ` directly: recovers ν for ν>4, saturates safely at ≈4
  for ν≤4, robustness preserved throughout.
- **Anomaly reframe** — one clause in the by-product paragraph: the score is a *point-in-time
  conditional-likelihood* "does this value belong" test (parity with a specialist, not a SOTA claim).

**Robustness verdict:** the core is solid and the honest-limitations story is a strength, not a
gap. Nothing here is a "the model is broken" finding — they're all either (a) already-working
capabilities we under-sell (anomaly detection), (b) correct-by-design losses we should *frame*
better (FX), or (c) clean, well-scoped extensions (covariates, ν-MLE). Remaining items for
Eeshaan, sized:

- **E1 — Harden the error-detection read-out (≈1 wk).** Promote `anomaly_scores()` to a
  documented "data-quality / novelty" API; add a real labelled benchmark (e.g. a public sensor-
  fault or known-outlier financial series) beside the synthetic ROC-AUC; add the
  "given-a-value, does-it-belong" conformal p-value form and a worked notebook. *Frame as a
  by-product at parity with a specialist, not a new SOTA claim.*
- **E2 — ✅ DONE (in paper).** The cointegration dial is now `bench/exp_cointegration.py` +
  Table~`cointegration` + Fig~`cointegration` in `cafe.tex`. Remaining stretch: add a vol-surface
  or CDS-curve slice as a third real anchor on the same axis (more financial breadth).
- **E3 — Covariate-conditioned state (≈2–3 wk, research-y).** Let μ / the state ingest an
  exogenous regressor (carry, rate differential). Directly addresses the one honest reason we lose
  on FX *even vs a prior*, and unlocks the financial panels in the hook.
- **E4 — ν via profile-EM MLE (≈few days).** The fat-tail *honest scope* is now in the paper
  (Table~`fattail`): ν is recovered for ν>4 and saturates safely below. The remaining *upgrade* is
  to replace the κ-moment inversion with a profile-likelihood ν estimator so calibration extends
  below ν=4 (returns); keep the moment estimate as a warm start.
- **E5 — (stretch) a cointegrated-finance benchmark slice.** Yield-curve tenors / CDS curve /
  vol surface as a public, reproducible panel where we expect to *win* causally — the empirical
  payoff of E2/E3.

I'd sequence **E2 → E1 → E4 → E3**, with E5 as the demonstrator once E3 lands.

— D.
