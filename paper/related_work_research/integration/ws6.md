# WS6 integration note -- rigor & fairness (Gaps #13 + #7)

Branch: `gap/rigor-fairness` (worktree `TIMARA-ws6`). Two new self-contained
experiments + LaTeX floats. Nothing in the DO-NOT-EDIT set was touched; this note
records exactly where to wire the floats into `paper/cafe.tex` and `bench/repro.py`.

## New files

| File | Output | `\label` | What it proves |
|------|--------|----------|----------------|
| `bench/exp_seeds_full.py` | `paper/tables/seeds_full.tex` | `tab:seedsfull` | Significance breadth: paired multi-seed CAFE vs online TRMF **and** BayOTIDE across all 8 structured panels + mean-rank / Friedman summary. |
| `bench/exp_baseline_fairness.py` | `paper/tables/baseline_fairness.tex` | `tab:fairness` | Deep baselines are adequately trained (bidir MAE in the adequacy band); the causal collapse is a protocol effect, not under-training. |

Both exit 0 with real, live numbers (no fabrication).

## Live results (12 seeds, MCAR 10%, caps 2500x60, all causal)

`exp_seeds_full.py` -- CAFE vs online TRMF vs BayOTIDE on all 8 panels. Paired
diff = mean(rival - CAFE) per seed; positive => CAFE lower error. All paired
t/Wilcoxon tests significant at p<0.05.

| Panel | CAFE | TRMF | BayOTIDE | best | diff(TRMF) | diff(Bay) |
|-------|------|------|----------|------|-----------|-----------|
| FRED-MD     | 0.077 | 0.207 | 0.163 | CAFE | +0.130*** | +0.086*** |
| AirQuality  | 0.166 | 0.214 | 0.364 | CAFE | +0.048*** | +0.198*** |
| Appliances  | 0.232 | 0.455 | 0.311 | CAFE | +0.223*** | +0.079*** |
| Beijing     | 0.146 | 0.328 | 0.277 | CAFE | +0.182*** | +0.130*** |
| Traffic2    | 0.238 | 0.296 | 0.416 | CAFE | +0.058*** | +0.178*** |
| ETTh        | 0.292 | **0.256** | 0.403 | **TRMF** | **-0.036*** | +0.111*** |
| Solar       | 0.130 | **0.126** | 0.172 | **TRMF** | **-0.005*** | +0.042*** |
| Electric    | 0.349 | 0.419 | 0.452 | CAFE | +0.070*** | +0.103*** |

Mean rank (1=best): **CAFE 1.25**, TRMF 2.12, BayOTIDE 2.62. Friedman omnibus
p = 2.08e-2 (rankings differ). CAFE best on **6/8** panels; it honestly LOSES on
ETTh (7-col transformer, pure-AR regime -- TRMF's core fits better, -0.036 MAE,
p=3e-6) and marginally on Solar (-0.005 MAE, p=3e-4). Reported as-is.

`exp_baseline_fairness.py` -- from live `bench/horserace_cache.json` (epochs=40,
window=24, rows_cap=1000, PyPOTS 1.5). All 5 deep models adequately trained
(best bidir MAE <= 0.45 band; under-trained floor ~0.50-0.75) yet collapse causally:

| Model | bidir best/mean | causal mean | gap | registry (Beijing, diff protocol) | trained? |
|-------|-----------------|-------------|-----|-----------------------------------|----------|
| SAITS        | 0.175 / 0.449 | 0.643 | +0.194 | 0.155 | yes |
| BRITS        | 0.175 / 0.461 | 0.840 | +0.379 | 0.127 | yes |
| Transformer  | 0.209 / 0.453 | 0.710 | +0.257 | 0.142 | yes |
| TimesNet     | 0.282 / 0.483 | 0.772 | +0.289 | --    | yes |
| ImputeFormer | 0.206 / 0.499 | 0.614 | +0.115 | --    | yes |

The registry column (TSI-Bench / du2023, full-data official windowed protocol) is
CITED context only and intentionally lower than our capped-slice bidir -- shown as
context, **not** a head-to-head, and not re-run at full official scale here.

## Where to wire into `paper/cafe.tex` (DO-NOT-EDIT here; instructions only)

1. **Statistical rigor paragraph** (currently ll.495-503, the
   "Statistical rigor (paired, multi-seed)" block that ends with
   `\input{tables/seeds}`). EXTEND it with one sentence and add the new float
   right after the existing `\input{tables/seeds}`:

   > Broadening the test from three datasets to all eight structured panels and
   > adding a second causal rival (BayOTIDE, the online Bayesian imputer) confirms
   > the picture (Table~\ref{tab:seedsfull}): over $12$ MCAR$(10\%)$ seeds CAFE has
   > the best mean rank ($1.25$ vs.\ $2.12$ for online TRMF and $2.62$ for BayOTIDE;
   > Friedman $p<0.05$) and is the best causal imputer on $6$ of $8$ panels, with
   > every paired difference significant at $p<0.05$. It cedes only ETTh (the
   > $7$-series pure-AR regime, where TRMF's carry-forward core edges it by $0.036$
   > MAE) and marginally Solar -- both reported as-is.

   ```
   \input{tables/seeds_full}
   ```

2. **Baseline-protocol box near the causal horse race** (after the
   "The causal horse race" paragraph, ll.367-418, just before the per-dataset
   tables at `\input{tables/horserace_perdataset}` l.446). Add a short paragraph +
   the float so a reviewer cannot dismiss the collapse as under-training:

   > \paragraph{The deep baselines are adequately trained (the collapse is
   > protocol, not under-training).} A natural objection is that the causal
   > collapse merely reflects feeble baselines. It does not
   > (Table~\ref{tab:fairness}): each deep model's \emph{bidirectional} MAE sits in
   > the adequately-trained band (best $\le 0.45$, far under the $\sim0.50$--$0.75$
   > under-trained floor of the same convergence harness), yet its \emph{causal}
   > MAE in the SAME run is far worse (gap $+0.12$ to $+0.38$). The models are
   > competent in their native future-using setting; only the protocol breaks them.
   > The published registry (full-data official split) is cited for context, on a
   > different and easier protocol than our fast capped-slice run -- not a
   > head-to-head.

   ```
   \input{tables/baseline_fairness}
   ```

## `bench/repro.py` MANIFEST entries (add to the `MANIFEST` dict, ~l.116)

```python
"exp_seeds_full.py": {
    "tables":  ["paper/tables/seeds_full.tex"],
    "figures": [],
    "tex_label": ["tab:seedsfull"],
    "what": "Paired multi-seed CAFE vs online TRMF & BayOTIDE on all 8 panels + mean-rank/Friedman.",
},
"exp_baseline_fairness.py": {
    "tables":  ["paper/tables/baseline_fairness.tex"],
    "figures": [],
    "tex_label": ["tab:fairness"],
    "what": "Deep baselines adequately trained (bidir adequacy band); causal collapse is a protocol effect.",
},
```

(`exp_baseline_fairness.py` reads the existing `bench/horserace_cache.json`; it
needs no torch. `exp_seeds_full.py` runs CAFE+TRMF under base `python3` and adds
BayOTIDE automatically when torch is importable; both are also runnable under the
bench venv `/Users/dereksnow/Sovai/Github/TIMARA/.venv-bench/bin/python` with
`PYTHONPATH=bench:src`.)
