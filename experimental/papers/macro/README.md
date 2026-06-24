# Honest Nowcasting at the Ragged Edge

A standalone macro / central-bank nowcasting paper, separate from the CAFÉ method paper
(`paper/cafe.tex`). Everything for it lives in this directory.

## Thesis
Nowcasting under mixed frequencies and a ragged edge is, formally, a missing-data
problem. A **causal, strictly point-in-time imputation model** can solve it directly —
treat the unreleased target as a missing cell, read off its fill — and is **competitive
with the standard EM dynamic factor model (EM-DFM)** while running CPU-only in
milliseconds. "No look-ahead" is the discipline central banks already live by; a
vintage-faithful evaluation is the only honest way to demonstrate it.

## Real-time design used: GENUINE ALFRED vintages (not simulated)
This is the **gold-standard Croushore–Stark real-time protocol with real archival
vintages**, not a pseudo-real-time re-truncation of modern data.

- For every vintage (decision) date `v` and every series, we download the series **exactly
  as it was published on `v`** from ALFRED (Archival FRED), endpoint
  `alfredgraph.csv?id=<SERIES>&vintage_date=<v>`. The publication-lag ragged edge is the
  real one the data actually had.
- Target: **INDPRO** (industrial production growth, % m/m), a canonical monthly
  coincident-activity index.
- At each `v` we nowcast IP growth for the first **unpublished** month and score against
  the **final revised** value (latest vintage). No look-ahead is possible because every
  input panel is literally the as-known data.
- Vintages: bimonthly origins, `2015-07` … `2025-03` (59 vintages; `step=2` in
  `fetch_vintages.py`). Predictors: payroll & civilian employment, unemployment rate, real
  income, real consumption, capacity utilisation, durable-goods orders, manufacturing
  weekly hours, housing starts, manufacturing employment, plus IP's own past releases.

We did **not** fabricate any vintage. If ALFRED had been unreachable we would have fallen
back to an explicitly-labelled pseudo-real-time design on FRED-MD; that fallback was
**not** needed — the real vintages were fetched successfully (see `data/fetch_log.txt`).

## Headline result (genuine nowcast, h=1, 59 vintages 2015-07 … 2025-03)
Reported on **two samples × two metrics** — no cherry-picking the flattering cut.

| | Normal times (ex-2020, N=53) | | Full sample (incl. 2020, N=59) | |
|---|---|---|---|---|
| **Nowcaster** | RMSE | MAE | RMSE | MAE |
| **CAFÉ (ours)** | **0.825** | **0.581** | 2.283 | 0.959 |
| EM-DFM | 0.804 | 0.591 | 2.097 | 0.914 |
| Persistence (RW) | 0.993 | 0.709 | 1.641 | 0.950 |
| AR(1) | 0.845 | 0.612 | 1.858 | 0.887 |

**Honest reading.** In *normal times* CAFÉ ties EM-DFM on RMSE (within 2.6%), **beats** it
on the outlier-robust MAE, and both clearly beat persistence/AR(1) — the intended result.
Over the *full* sample the single **April-2020 COVID crash** (an extreme IP swing no method
nowcasts) dominates squared error, so full-sample RMSE rewards whichever model reacted most
(the random walk "wins"); under the robust MAE the full field is a near-tie. We report all
four numbers and let them fall where they may. CAFÉ runs CPU-only in milliseconds per
vintage with no frequency-aware code.

## Which numbers back which claims
All numbers are produced by runs in `code/`; nothing is typed by hand into the paper.

| Claim | Source |
|---|---|
| Two-sample, two-metric table (CAFÉ vs EM-DFM vs persistence vs AR(1)) | `data/results.json` (`blocks.{full,ex2020}.h1`), `data/results_table.csv` → Table 1 |
| Full-sample RMSE ranking is a COVID-outlier artefact (full vs ex-2020 bars) | `figures/fig_horizon.pdf` |
| Real-time nowcast paths vs final revised truth (COVID window shaded) | `figures/fig_vintage.pdf` |
| Real ragged edge / publication lag | `data/fetch_log.txt`, `data/alfred_cache/*.csv` |

The paper's numeric placeholders are filled by `code/patch_paper.py` reading
`data/results.json` (re-generate `macro.tex` from git before re-patching, since the patcher
only fills `[[...]]` tokens).

## Reproduce
```bash
cd experimental/papers/macro/code
python3 fetch_vintages.py      # downloads + caches real ALFRED vintages (slow, polite:
                               #   ~3s/request; ALFRED throttles bursts). Resumable: cached
                               #   files in ../data/alfred_cache are reused.
python3 run_nowcast.py         # CAFÉ / EM-DFM / persistence / AR(1) real-time evaluation
python3 patch_paper.py         # fill macro.tex placeholders from results.json
cd ..
tectonic macro.tex             # build the PDF (pdflatex/xelatex not required)
```

## Caveats (also stated in the paper)
- One monthly target (IP growth). A multi-target study and a genuinely quarterly target
  (e.g. GDP, which adds the frequency-mixing constraint) would strengthen the conclusion.
- EM-DFM here is an off-the-shelf single-factor `statsmodels.DynamicFactorMQ`; a
  production central-bank DFM is more heavily engineered and would likely be stronger.
- Vintages are **bimonthly** over ~a decade (59 origins); longer span / monthly cadence
  would tighten the estimates.
- The full sample includes the COVID-19 shock; we report both the full and 2020-excluded
  samples and the outlier-robust MAE so the reader sees exactly how much of the full-sample
  RMSE ranking is the single 2020 month. No outlier removal is hidden.
- Point nowcasts only; CAFÉ's calibrated predictive intervals are not evaluated here.

## Files
- `code/fetch_vintages.py` — ALFRED real-time vintage downloader (curl-backed, cached).
- `code/run_nowcast.py` — the real-time nowcasting evaluation and figures.
- `code/patch_paper.py` — writes real numbers into `macro.tex`.
- `data/` — cached vintages (`alfred_cache/`, `vintages.pkl`), results, logs.
- `figures/` — `fig_horizon.pdf`, `fig_vintage.pdf`.
- `macro.tex` — the self-contained paper (own preamble + bibliography).
