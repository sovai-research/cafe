# The $\epsilon$ of Imputation — a position paper

A standalone academic **position / standards paper**, separate from the CAFÉ
method paper (`paper/cafe.tex`). It argues:

> **Look-ahead leakage in time-series imputation is measurable, pervasive, and
> unnecessary — and the field should report a causality certificate and a leakage
> score $\Delta$ alongside MAE.**

The star is the **standard + the field survey**, not a method. CAFÉ is demoted to
a one-row *existence proof* (a causal method need not leak to be competitive).

## The three legs (and which run backs each)

**Leg 1 — Measurable.** A model-agnostic look-ahead audit promoted to a proposed
community standard. Two numbers from any imputer `f(X) -> X̂`:
- **causality certificate** `max_revision`: largest change to an early imputed cell
  as the series is grown one prefix at a time (truncation invariance). `0` ⇔
  strictly point-in-time.
- **leakage score** `Δ = MAE_causal − MAE_bidir`: accuracy borrowed from the future
  (bidirectional fill vs. the same model run as an honest right-edge filter).

Source: `src/cafe/audit.py` (`leakage_report`), presented crisply with pseudocode
(Listing 1, verbatim from the reference implementation).

**Leg 2 — Pervasive.** A field survey re-auditing **28 methods × 16 real tasks**
(beijing, airquality, fredmd, etth × block+MCAR × 2 seeds; 10% held-out; L=24)
through the *identical* standard.
- Run: `scripts/run_audit.py` → `data/field_survey.json`
- Table/figure: `scripts/make_figs_tables.py` → `data/field_survey_table.tex`,
  `figures/field_survey.pdf`
- **Real numbers obtained (mean over 16 tasks):**
  - **18 of 28 methods certify** (`max_revision=0`, `Δ=0` exactly): CAFE, LOCF,
    Kalman local-level, rolling/expanding mean–median, EWMA, drift, xsec-mean,
    online EW-cov, GROUSE, NoTMF, SHASTA-PCA, rGROUSE, OSW-Net, Zero, SeasonalNaive.
    Max `|Δ|` among certified-causal methods = **0.0000**.
  - **10 of 28 fail** the certificate or carry positive Δ:
    - Interpolation borrows accuracy: SplineInterp **Δ=+0.222** (max|rev|≈44.5),
      NOCB **+0.193**, LinearInterp **+0.150**; TRMF +0.008 (max|rev|≈2.3).
    - Online posterior-refiners borrow *no accuracy* yet are not truncation-invariant:
      gcimpute, BayOTIDE **Δ≈0** but `max_revision` ≈ 0.49 / 0.34.
    - Deep imputers post **negative Δ** — their honest right-edge filter beats their
      own bidirectional reconstruction: **SAITS Δ=−0.376, BRITS −0.340,
      Transformer −0.378** (max|rev| ≈ 1–2). The bidirectional objective is the
      wrong one even by reconstruction error on these tasks.

**Closed form — when the leak bites.** `Δ(a,g)` from exact Kalman-filter / RTS-smoother
posterior variances (Eq. 1–2), validated against exact Monte-Carlo runs.
- Run: `scripts/run_gap_theory.py` → `data/gap_theory.json`, `figures/gap_theory.pdf`
- **Real numbers:** closed form vs. exact runs over the (a,g) plane at
  **R² = 0.9996**, median rel. err **1.2%**. Real-panel lag-1 memory â places each
  dataset on the map: high-memory panels leak (FRED-MD â=0.996, FX 0.999,
  Beijing 0.929, AirQuality 0.892), near-memoryless give nothing (Traffic 0.083,
  Electric 0.048; Δ<1e-3). Table: `data/gap_band_table.tex`.

**Leg 3 — Unnecessary.** CAFÉ as a constructive existence proof. In the same survey
it certifies (`Δ=0`, `max_revision=0`) **and posts the lowest causal MAE of any
method, causal or not: 0.286** vs. 0.301 for the best certified-causal rival
(SHASTA-PCA) and 0.356 for BayOTIDE.

**Field corroboration** (from `paper/related_work_research/`): documented incidents
where leakage forced retraction/re-runs — SSSD (Solar-preprocessing leakage,
retrained with worse error), GIFT-Eval (pretraining/test overlap warning), TimeFlow
(evaluating on series seen in training). Cited in §1.

## Recommendation (§6)
- **R1.** Report `(max_revision, Δ)` alongside MAE — one extra audit call.
- **R2.** Score causal and bidirectional protocols on *separate* boards; never rank
  together.
- **R3.** Make the certificate an entry condition for "online/streaming/real-time"
  claims.

## Build
```
cd experimental/papers/position
tectonic position.tex
```
Produces `position.pdf` (**7 pages**, 0 undefined references). `pdflatex`/`xelatex`
are unavailable in this environment; `tectonic` is the build tool.

To regenerate all data + figures + table fragments from scratch:
```
# from repo root /Users/dereksnow/Sovai/Github/TIMARA
.venv-bench/bin/python experimental/papers/position/scripts/run_audit.py      # field survey (deep models need torch+pypots)
python3 experimental/papers/position/scripts/run_gap_theory.py                 # gap theory + figure
python3 experimental/papers/position/scripts/make_figs_tables.py               # survey figure + LaTeX tables
```

## Honesty contract
Every number in the paper is from a run executed here (`data/*.json`) or is
explicitly attributed to the CAFÉ method paper (the larger eight-dataset deep horse
race, baseline-fairness, and downstream results are *cited*, not re-run). The audit
core (`cafe.audit`) is numpy-only; deep models (SAITS/BRITS/Transformer) were run
live under `.venv-bench` (torch + PyPOTS), ~8.6 min for the full survey.

## Files
- `position.tex` / `position.pdf` — the paper (self-contained preamble + `\bibitem`
  bibliography; does **not** `\input` from `paper/`).
- `scripts/` — `run_audit.py`, `run_gap_theory.py`, `make_figs_tables.py`.
- `data/` — `field_survey.json`, `gap_theory.json`, and the `\input`-ed `.tex`
  table fragments.
- `figures/` — `field_survey.pdf`, `gap_theory.pdf`.
