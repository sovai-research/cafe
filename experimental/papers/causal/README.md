# CAFÉ-CD — a short results paper

A standalone results paper on **CAFÉ-CD**: point-in-time causal discovery run on the
idiosyncratic residuals of the CAFÉ dynamic factor model. It is a spin-off of the main
method paper (`paper/cafe.tex`) in the same style as
`experimental/papers/{position,volatility,macro}` — it **cites** the main paper rather
than `\input`-ing from it, and carries its own preamble and `\bibitem` bibliography.

Nothing under `src/cafe/`, `bench/`, `paper/` or `pyproject.toml` is written by this
directory. `tables/replication.tex` is the one file here written by something else
(`bench/exp_causal_replication.py --tables-only`); this build inputs it and names its
source in the provenance map, but never regenerates or edits it.

## What the paper claims

**1 — The method (§1).** CAFÉ residualisation → engine routing → backward-only lag
screening → panel group-lasso skeleton (shared `B0` + sparse per-entity deviations) →
assumption-routed orientation with explicit abstention (`o-o`, `<->`) instead of a
manufactured DAG. numpy-only, CPU-only; every stage reads rows `<= t`, so `mode="pit"`
is truncation-invariant *by construction*, not by convention.

**2 — Scope, stated up front (§2).** On a controlled ladder in the common-variance share
χ the crossover χ\* sits at **0.83** under an unpenalised OLS-VAR, **0.68** under the
panel group lasso and **0.72** under an L1 VAR — the FNETS/Krampe mechanism made
quantitative. **Below the crossover the residual is worse than raw, and the paper says
so in the same paragraph** — but it now also says that this is a statement about the
*representation*, not the pipeline: §3 wins at χ=0.50 by 0.265 AUC where residualisation
alone loses 0.056. Table 1 and Table 2 answer different questions.

**3 — The head-to-head we win (§3), on ground truth we audit.** Synthetic panels
(E=10, T=120, P=24, R=3, K=2, Laplace): **rank 1 of 15 at every χ**, 0.985 → 0.929,
against a best baseline that never leaves 0.708–0.720. Two objections are answered in
the table rather than left for the reader: a **`+resid` fairness arm** re-runs every
baseline on the *identical* CAFÉ residual (it moves them −0.048 to +0.086, mean +0.011,
against a gap of 0.22–0.27), and the **pooling advantage is named**: the baselines are
single-series methods applied per entity, so a large part of the margin is the entity
dimension. That is the contribution, and it does not transfer to §5.

**4 — Benchmark forensics (§4).** Every criticism sits next to the reproduction that
earns it.
- Five published CausalRivers (ICLR 2025) cells reproduced; the same VAR then moves
  **0.8493 → 0.5931** on all 2 139 subgraphs by switching to a scale-free statistic, and
  below chance if rescaled. Measured varsortability of the true graphs: **0.9393**.
  **The scale-free bar is 0.593.**
- Two CausalDynamics (NeurIPS 2025 D&B) `simple` cells are broken and both published
  AUROCs reproduce to 1e-3 *by replicating the defect*.
- **`coupled`: the Default row is re-derived from the authors' own shipped
  `outputs/coupled.tar.gz`, and then a constant identity matrix scores 0.682/0.353 and
  no shipped method beats it** (paired Wilcoxon, n=100; PCMCI+ p=0.21 AUROC, p=5e-18
  *against* on AUPRC, winning on 2%). Mechanism: the protocol keeps the diagonal and
  **40.8% of the positive class is self-loops**. `Confounder` (19%) and `Lag` (17%) are
  real venues — where the bar is a plain **OLS VAR(1) at ~1 ms**.
- **The published AUPRC null is too low on both tiers**: against a random *ranker* the
  floor is 0.188 on `coupled` and **0.759** on `simple`, so **every published `simple`
  AUPRC (.67–.70) is below chance**.
- **A third upstream defect:** `grasp` feeds causal-learn's endpoint matrix (−1/0/+1)
  into `roc_auc_score` as a score; `abs()` first moves AUROC 0.442 → 0.545.

**5 — Where we lose (§5), prominently.** CausalRivers head-to-head, AUPRC: flood/R-5
published arm **11/13**; scale-free arm 4/13; the `east` configurations mid-table.
**`var_ols` beats CAFÉ-CD in every completed configuration.** CausalDynamics `simple`,
54 matched systems, **6/13** — on a tier whose measured random-ranker floor (0.759)
swallows ten of the thirteen entries. `coupled`/`climate` overlap on 7 and 11
fully-scored systems, too few to rank, and no number is drawn from them.

**6 — The diagnosis (§6), with a measured mechanism.**
- **The representation cost is partly a located library defect.** `src/cafe/_core.py`
  sets the ARD precision `alpha_l = N/(||W[:,l]||^2 + 1)`; the `+1` does not rescale
  with the data, so under `x -> c*x` the mechanism is not equivariant. Measured on our
  own generator: the factor adjustment removes 74% of the variance at unit scale, 1.5%
  at mean sd 0.375, and **exactly nothing** below that. On unrescaled CausalRivers,
  where column sds inside one five-station panel span 0.030–175 and every sample has a
  sub-unit column, it is **mis-scaled rather than idle** — the share removed is
  *negative* on 14 of 25 samples. Reported as a located defect with a measured cost and
  a re-certification requirement (the line is in the production `impute()` path;
  standardising inside `discover()` instead would make the scale prefix-dependent and
  could break the `rev = 0` certificate), **not** as a pending win. Fixing it would not
  reach `var_ols`, and no benchmark verdict changes.
- **`effective_rank` is not a usable factor-activity indicator** — it reads 0 across the
  whole of that sweep, including the rows removing 74%. Every factor-activity claim in
  the paper comes from the adjusted-vs-kept residual comparison instead.
- A **2×2** over representation × conditioning set, same estimator and same moment
  tensor throughout, identity-checked: the raw-conditional cell reproduces `var_ols` and
  the raw-bivariate cell reproduces `granger_pairwise` to **0.0000** mean absolute
  AUROC on all 40 samples. **Residualising costs 0.278 AUROC (38/40); the conditioning
  set a further 0.055.** On these near-collinear five-station panels the pervasive
  component *is* the causal transmission, so projecting it out removes the edges.
- The bivariate arm is **not** presented as a fix: a separate 40-sample measurement
  under a different protocol reverses the ordering on the residual, our own sample
  going 30 → 40 shrank the gap without flipping it, and the shipped `_scoring_router`
  abstains on our own panels (max |Δ| 0.0007 over the 15 cached h2h cells).
- A second limit, on our **own** generator (Table 9): the evidence blend's premium over
  its strongest single component is **monotone in E and negative at E=1** (−0.0101 →
  +0.0086), and every public single-series benchmark is E=1. **No gate was shipped** —
  the effect is 0.01–0.02 AUC, it changes no rank, and fitting one would be the
  benchmark-fitting §4 criticises.
- **Three hypotheses tested and falsified**, all reported: "residualisation removes the
  scale prior" (varsortability 0.99 raw → 0.76 residual over 25 subgraphs, still far
  above the 0.5 no-signal point — the ordering survives and our estimator is simply
  built not to read it, which is why it shares the exactly-+0.0000 signature with
  Granger/PC/FCI/PCMCI); the antisymmetric-share test
  (predicted sign on all 25 samples, AUC falls on 12, sign agreement 0.40 — the
  statistic saturates on near-unit-root discharge) and the path-resolution test
  (resolution *anti*-correlated with informativeness).

**7 — The point-in-time board (§7), now complete.** Eight tasks for all thirteen
methods. CAFÉ-CD 0/8 revisions, 0.000 churn — a statement about its design, not a
contest won. The informative columns are churn magnitude and the leakage delta Δ on each
rival's own metric.

**8 — Runtime (§8).** **CAFÉ-CD 2.42 s vs PCMCI+ 49.55 s (21×)** on a real 12-variable
CauseMe series at the identical dense 0.917 submission. Orientation is proved
end-to-end on CauseMe's own format and metrics: **all 13 methods beat their own
transpose**. 52 artefacts built and validated; **nothing uploaded**, and no CauseMe
leaderboard number appears anywhere in the paper.

**9 — Two engineering results (§9)**, one of them a null result on our own idea
(bit-packed screening is 8–21× *slower* than dense BLAS).

**10 — Limitations (§10)** get their own numbered section.

## Live vs pending

`scripts/make_tables.py` re-derives the provenance map and the pending list on every
build: it emits `--` for missing cells, appends a human-readable reason to a `PENDING`
list, and writes both into `tables/status.tex`, which §11 of the paper prints verbatim.
**If a run finishes, rebuild and the cells fill themselves in.**

Because several caches are still being written, **every number the running text quotes
from a live cache is emitted as a `\newcommand` into `tables/macros.tex`** and used by
name in `causal.tex`. A rebuild therefore cannot leave the prose disagreeing with the
table beside it. Hand-typed numbers in the body are only those from caches that are
complete and closed.

| number | status |
|---|---|
| χ ladder / crossovers (Table 1, Fig. 1) | **finished** — `bench/causal_sparse_gate_cache.base.json` |
| audited-truth head-to-head + `+resid` arm (Table 2) | **finished** — 240 rows / 16 cells, `bench/causal_synthetic_h2h_cache.json` |
| CausalRivers reproduction + scale ladder (Table 3) | **finished** — `bench/bench_rivers_cache.json` |
| varsortability / R²-sortability (Table 3b) | **finished** — live run, `data/varsortability.json` |
| CausalDynamics `simple` forensics (Table 5) | **finished** — `bench/bench_dynamics_cache.json` |
| CausalDynamics `coupled` forensics (Table 6) | **finished** — `data/causal/cache/cd_coupled_audit.json` |
| CausalRivers head-to-head ranks (Table 7a) | **live** — one config per rebuild as `exp_bench_rivers.py` finishes them; unfinished configs are listed in `status.tex`, never averaged |
| CausalDynamics matched board (Table 7b) | **finished** — live run, `data/dynamics_matched.json` |
| conditioning 2×2 diagnosis (Table 9) | **finished** — live run, `data/conditioning.json` |
| per-method scale-prior decomposition (Table 4) | **finished** — `bench/bench_rivers_cache.json`; grows as the harness finishes methods |
| ARD scale-equivariance sweep + rivers arms (Table 10) | **finished** — live run, `data/scale_equivariance.json` |
| evidence-blend replication (Table 9) | **finished** — `bench/exp_causal_replication.py` |
| point-in-time board, all 13 rows (Table 10) | **finished** — 8 tasks each, `bench/pit_board_cache.json` |
| CauseMe runtime + orientation (Table 11) | **finished** — `bench/bench_causeme_cache.json` |
| engine router (Table 4a) | **finished** — live run, `data/engine_router.json` |
| screening backend (Table 4b) | **finished** — live run, `data/screen_backend.json` |
| CausalDynamics `coupled`/`climate` head-to-heads | **not printed** — the baseline board and the CAFÉ-CD board overlap on 7 and 11 fully-scored systems; too few to rank, and `status.tex` says so |

## Honesty contract

Every number in the paper comes from a cache produced by a run in this repository, or
from one of the six scripts here that we executed. There are no illustrative figures
and no hand-transcribed cells: `make_tables.py` may not write a number it did not read
from a JSON file.

Three numbers in §6's running text — the 40-sample residual bivariate re-measurement
under the router's protocol, the antisymmetric-share test and the path-resolution test
— were measured in the scoring-router investigation and are **not** cached in this
directory. `status.tex` names them and says so, rather than letting them look local.

Two caveats are carried in the paper rather than smoothed away: CAFÉ-CD's AUPRC on the
point-in-time board is a **lower bound** (those rows predate the change of
`CausalGraph.scores()` from a two-tier step to continuous evidence — the certificate
columns read the support and are unaffected), and the CausalDynamics row was re-run
after that change with the support reproducing on 54/54 systems, so the effect there was
+0.001 AUPRC.

`bench/bench_dynamics_cache.json` holds CAFÉ-CD **twice**: a stale set of rows under the
cache tag `batch` (written when CAFÉ-CD was still in `METHODS`) and the current
dedicated `cafecd` tag. They disagree on the *support*, so they are two different
pipelines. `scripts/run_dynamics_matched.py` settles it by re-running the identical job
and checking the support: 54/54 identical to the `cafecd` tag. **The stale rows give
CAFÉ-CD 0.685 / 12th of 13 on an 18-system subset; the current ones give 0.729 / 6th of
13 on the 54 matched systems, and that is what the paper reports.**

## Build

```bash
# from the repo root /Users/dereksnow/Sovai/Github/TIMARA

# 1. the six live probes (write experimental/papers/causal/data/)
.venv-bench/bin/python experimental/papers/causal/scripts/run_engine_router.py         # 16 CAFE passes
.venv-bench/bin/python experimental/papers/causal/scripts/run_screen_backend.py        # ~15 s
.venv-bench/bin/python experimental/papers/causal/scripts/run_varsortability.py        # ~9 s  (needs data/causal/)
.venv-bench/bin/python experimental/papers/causal/scripts/run_dynamics_matched.py      # ~1 min (needs CausalDynamics)
.venv-bench/bin/python experimental/papers/causal/scripts/run_conditioning_diagnosis.py # ~3 min (needs CausalRivers)
.venv-bench/bin/python experimental/papers/causal/scripts/run_scale_equivariance.py    # ~2 min (needs CausalRivers)

# 2. the one table generated elsewhere
.venv-bench/bin/python bench/exp_causal_replication.py --tables-only

# 3. regenerate every other table + the figure + tables/macros.tex from the caches
.venv-bench/bin/python experimental/papers/causal/scripts/make_tables.py

# 4. the paper
cd experimental/papers/causal && tectonic causal.tex
```

Produces `causal.pdf` — **13 pages, 0 undefined references, 0 overfull hboxes**.
`pdflatex`/`xelatex` are unavailable in this environment; `tectonic` (0.16.9) is the
build tool. `make_tables.py` needs numpy + matplotlib; the probes need `cafe` on the
path (they insert `src/` themselves), `run_varsortability.py` and
`run_conditioning_diagnosis.py` need the CausalRivers download and
`run_dynamics_matched.py` the CausalDynamics one, both under `data/causal/` (see
`docs/causal_baselines.md`).

## Files

```
causal.tex          the paper (self-contained preamble + \bibitem bibliography)
causal.pdf          13 pages
README.md           this file
scripts/
  make_tables.py                  every table but replication.tex, plus the figure, the
                                  macro file and the PENDING list, all from the caches
  run_engine_router.py            live: router vs held-out MAE under each engine pinned
  run_screen_backend.py           live: bit-packed vs dense screening, + bit-identity check
  run_varsortability.py           live: varsortability / R2-sortability of CausalRivers truth
  run_dynamics_matched.py         live: CAFE-CD re-run on CausalDynamics `simple`, matched
                                  to the baseline board, with a support identity check
  run_conditioning_diagnosis.py   live: the representation x conditioning 2x2 on CausalRivers
  run_scale_equivariance.py       live: is the factor adjustment scale-equivariant, does it
                                  engage on CausalRivers, and does residualisation remove
                                  the scale prior (it does not)
data/
  engine_router.json  screen_backend.json  varsortability.json
  dynamics_matched.json  conditioning.json  scale_equivariance.json
tables/
  gate.tex  h2h.tex  rivers.tex  scaleprior.tex  dynamics.tex  coupled.tex
  losses.tex  diag.tex  scale.tex  runtime.tex  pit.tex  router.tex  screen.tex
  eng.tex  macros.tex  status.tex
  replication.tex   <- written by bench/exp_causal_replication.py, only INPUT here
figures/
  gate.pdf
```

`tables/*.tex` and `figures/gate.pdf` are **generated**; do not edit them by hand.
`eng.tex` is the full-width float that wraps `router.tex` and `screen.tex` side by side.
`macros.tex` must be `\input` **before** `\begin{document}`; macro names may not contain
digits.

## Upstream sources (read-only)

- `bench/causal_sparse_gate_cache.base.json`, `bench/exp_causal_sparse_gate.py`
- `bench/causal_synthetic_h2h_cache.json`, `bench/exp_causal_synthetic_h2h.py`
- `bench/causal_replication_cache.json`, `bench/exp_causal_replication.py`
- `bench/bench_rivers_cache.json`, `bench/exp_bench_rivers.py`
- `bench/bench_dynamics_cache.json`, `bench/exp_bench_dynamics.py`
- `bench/bench_causeme_cache.json`, `bench/exp_bench_causeme.py`
- `bench/pit_board_cache.json`, `bench/exp_pit_board.py`
- `data/causal/cache/cd_coupled_audit.json`, `bench/datasets_causal.py --audit coupled`
- `bench/gen_causal_panel.py`, `bench/causal_metrics.py`, `bench/sortability.py`
- `docs/causal_baselines.md` — install landmines, GPL isolation for `tigramite`, the
  14-convention self-test, the reproduction protocol, the self-loop trap, and §6c/§6d/§9
  behind the `coupled`, `climate` and CauseMe findings.
