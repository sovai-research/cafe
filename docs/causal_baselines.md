# External causal-discovery baselines — install, conventions, protocol

Machinery for the CAFÉ-CD "better than SOTA" claim. Two modules:

| file | role |
|---|---|
| `bench/causal_baselines.py` | one uniform `run_baseline(name, X, max_lag=8, **kw) -> BaselineResult` over 12 external / reference methods, with a self-test that PROVES every index convention |
| `bench/datasets_causal.py` | loaders for CausalRivers, CausalDynamics and the CauseMe submission format, plus verbatim re-implementations of each benchmark's own scoring code |

Everything here is **bench-only**. `src/cafe/` stays MIT + strict-numpy and must
never import any of it. `pip install -e .` still resolves to `numpy` only
(verified: `Requirement already satisfied: numpy>=1.21 … Would install
cafe-impute-0.1.0`, nothing else).

---

## 1. What was installed

Into `.venv-bench` (Python 3.13.2) only. Nothing touched system Python.

| package | version | licence | how installed |
|---|---|---|---|
| `lingam` | **1.13.0** | MIT | `pip install --no-deps lingam` (see caveat) |
| `causal-learn` | **0.1.4.8** | MIT | `pip install causal-learn` |
| `dagma` | **1.1.1** | Apache-2.0 | `pip install dagma` |
| `tigramite` | **5.2.10.1** | **GPL-3.0** | `pip install tigramite` — **subprocess-only** |
| `semopy` | 2.3.11 | MIT | undeclared runtime import of `lingam/__init__.py` |
| `psy` | 0.0.1 | MIT | undeclared runtime import of `lingam/__init__.py` |
| `autograd` | 1.9.1 | MIT | undeclared runtime import of `lingam/__init__.py` |
| `pygam` | 0.12.0 | Apache-2.0 | undeclared runtime import of `lingam/__init__.py`; `--no-deps` |
| `numdifftools` | 0.11.1 | BSD | dep of `semopy` |
| `momentchi2` | 0.1.8 | MIT | dep of `causal-learn` |
| `pydot` | 4.0.1 | MIT | dep of `causal-learn` |
| `igraph` (python-igraph) | 1.0.0 | **GPL-2.0+** | pulled in by `dagma` — see §2 |
| `texttable` | 1.7.0 | LGPL | dep of `igraph` |
| `progressbar2` | 4.6.0 | BSD | dep of `pygam` |
| `python-utils` | 4.0.1 | BSD | dep of `progressbar2` |

Pre-existing and **unchanged**: numpy 2.5.0, scipy 1.18.0, scikit-learn 1.9.0,
statsmodels 0.14.6, networkx 3.6.1, pandas 3.0.3, polars 1.42.0, h5py 3.16.0,
torch 2.12.1, pypots 1.5. **No package was downgraded.**

### Two install landmines

**(a) `lingam` pins a stale scipy.** `lingam 1.13.0` declares `scipy<=1.13.1`,
which has no cp313 wheel, so a plain `pip install lingam` tries to build scipy
from source and dies on a missing `gfortran`. lingam runs fine on scipy 1.18.
Install it with `--no-deps`. Same for `pygam` (`scipy<1.17`), which would
otherwise silently downgrade scipy to 1.16.3 for the whole bench env.

**(b) `lingam/__init__.py` eagerly imports four undeclared packages.** It
imports every estimator at package import time, so `import lingam` fails with
`ModuleNotFoundError` for `semopy`, then `pygam`, then `psy`, then `autograd`,
none of which appear in its metadata. All four must be present even if you only
want `VARLiNGAM`.

Working sequence on Python 3.13:

```bash
.venv-bench/bin/pip install causal-learn dagma tigramite semopy autograd psy
.venv-bench/bin/pip install --no-deps lingam pygam
.venv-bench/bin/pip install "progressbar2>=4.2.0,<5"
```

On Python ≤3.12 the plain `pip install -e ".[bench]"` resolves without help.

**`causalnex` is NOT installable.** It declares `Requires-Python >=3.8,<3.11`;
no distribution exists for 3.13 (`ERROR: No matching distribution found`).
DYNOTEARS is therefore **re-implemented from the paper** (Pamfil et al.,
AISTATS 2020) in numpy + scipy inside `bench/causal_baselines.py`, with
causalnex's own defaults (`lambda_w=lambda_a=0.1`, `max_iter=100`,
`h_tol=1e-8`, `w_threshold=0.0`). Augmented Lagrangian, `scipy.optimize`
L-BFGS-B on the nonnegative split `W = W⁺ − W⁻`, `A = A⁺ − A⁻`. Correctness is
covered by the same convention self-test as every wrapper, and it reproduces
the published CausalDynamics DYNOTEARS number to 0.001 (§6).

---

## 2. Licence discipline

`src/cafe/` is MIT. The rules actually enforced in code:

| package | licence | how it is used |
|---|---|---|
| `lingam` | MIT | imported in-process — fine |
| `causal-learn` | MIT | imported in-process — fine |
| `dagma` | Apache-2.0 | **`dagma.linear` only** |
| `tigramite` | **GPL-3.0** | **never imported in-process** |
| py-tetrad / causal-cmd | GPL-3.0 | same rule; not installed |

### The GPL isolation mechanism (tigramite)

`pcmci` and `pcmciplus` never import tigramite in the calling interpreter.
`_run_tigramite()` in `bench/causal_baselines.py`:

1. calls `assert_no_gpl_imported()`, which raises if `tigramite`, `pytetrad` or
   `causalcmd` is already in `sys.modules` of *this* process;
2. makes a fresh temp dir and writes the generated runner script
   `_tigramite_runner.py` (module-level string `_TIGRAMITE_RUNNER`) into it;
3. writes `X` and a JSON config to `in.npz`;
4. `subprocess.run([sys.executable, runner, in.npz, out.npz], timeout=…,
   cwd=tmp, env=_isolated_env())`. `_isolated_env()` strips the repo's `src/`
   and `bench/` from `PYTHONPATH` so `cafe` is not even importable in there;
5. the runner itself refuses to start if `"cafe" in sys.modules`;
6. results come back as a plain `.npz` (`val_matrix`, `p_matrix`, `graph`).

So the only thing crossing the boundary is numeric array data. A hard
`subprocess` timeout applies (the SIGALRM soft timeout used for in-process
baselines cannot kill a child). Verified: `run_baseline("pcmci", X,
timeout_s=0.001)` returns `ok=False, err="TIMEOUT (subprocess): …"`, and
`'tigramite' in sys.modules` is `False` in the parent afterwards.

### One licence surprise

`dagma` declares Apache-2.0 but **depends on `python-igraph`, which is
GPL-2.0+**. `dagma.utils` (DAG simulation helpers) imports it; `dagma.linear`
does **not** — confirmed by
`import dagma.linear; 'igraph' in sys.modules  ->  False`, then
`import dagma.utils; 'igraph' in sys.modules  ->  True`. The wrapper imports
`from dagma.linear import DagmaLinear` and nothing else. Do not `import dagma`
bare in any process that matters.

---

## 3. Index / sign conventions — verified, not assumed

**This repo's convention, everywhere:**

```
A_lag[k-1, i, j]  =  strength of  "j at lag k  ->  i at lag 0"
A_contemp[i, j]   =  strength of  "j at lag 0  ->  i at lag 0"
```

i.e. `A[effect, cause]`. Row = the variable being caused.

| library | its own convention | conversion applied |
|---|---|---|
| `lingam.VARLiNGAM` | `adjacency_matrices_` is `(lags+1, P, P)`; `[0]` = contemporaneous `B0`; `[k][i,j]` = j at lag k → i | **identity**; `A_contemp = [0]`, `A_lag[k-1] = [k]` (zero-padded when `criterion='bic'` picks fewer lags) |
| `lingam.DirectLiNGAM` | `adjacency_matrix_[i,j]` = coefficient of j in the equation for i | **identity** |
| statsmodels `VAR.coefs` | `coefs[k, eq, var]` = var at lag k+1 in equation eq | **identity** (this is what `var_ols` matches) |
| `dagma.linear` | `X ~ X W`, so `W[i,j] ≠ 0` means **i → j** | **transpose**: `A_contemp = W.T` |
| DYNOTEARS | `X ~ X W + Y A`, `W[i,j]` = i→j contemporaneous, `A[k·d+i, j]` = i at lag k+1 → j | **transpose**: `A_contemp = W.T`, `A_lag[k] = A[k·d:(k+1)·d].T` |
| `tigramite` | `val_matrix[i, j, tau]` = i at t−tau → j at t | **transpose**: `A_lag[k-1,i,j] = val[j,i,k]`, `A_contemp[i,j] = val[j,i,0]` (PCMCI+ only) |
| `causal-learn` | `g.graph[a,b]` is the **endpoint at node a** of the a–b edge: `-1` tail, `1` arrowhead, `2` circle. `i → j` ⟺ `graph[j,i]==1 and graph[i,j]==-1` | pairwise decoder `_decode_general_graph`; a→b sets `out[b,a]` |
| CausalRivers labels | `labels[m,n]=1` for edge `nodes[n] → nodes[m]` = `[effect, cause]` | **identity** |
| CausalDynamics (`simple`) | `adjacency_matrix[i,j]=1` ⟺ `∂ẋ_i/∂x_j ≠ 0` = `[effect, cause]` | **identity** |
| CausalDynamics (`coupled`/SCM) | `scm.py` documents the **opposite** (`A[i,j]` = edge i→j) | **transpose** — verify before scoring `coupled` |
| CauseMe | `A_ij` = score for link i → j = `[cause, effect]` | **transpose** on submission (`causeme_matrix_from_result`) |

### How each one is proven

`python bench/causal_baselines.py --self-test` builds two systems whose true
edge set is **disjoint from its own transpose**, so a flipped wrapper produces a
*negative* margin rather than a merely-worse one:

* `make_asym_var` — 4-var VAR, lag 1: `0→1 (.80)`, `1→2 (.70)`, `2→3 (.60)`;
  lag 2: `0→3 (.55)`; distinct AR terms `.60/.50/.40/.30` so even the diagonal
  is asymmetric. Laplace innovations so LiNGAM is identified.
* `make_asym_contemp` — v-structure `0→2←1` then `2→3` (orientable by PC/FCI),
  non-Gaussian **equal-variance** noise (identifiable for LiNGAM and DAGMA).

Statistic: `min(score on true directed edges) − max(score on the reversed
edges)`, over pairs where exactly one direction is true. Positive = correct.

**Result (all 14 checks pass):**

```
baseline               kind         margin  pairs       s  verdict
--------------------------------------------------------------------------
var_ols                lag         20.8262      4    0.00  PASS
granger_pairwise       lag          9.2732      4    0.01  PASS
varlingam              lag          0.2686      4    0.14  PASS
dynotears              lag          0.4886      4    0.17  PASS
pc_ts                  lag          1.0000      4    0.04  PASS
pcmci                  lag          0.4022      4    0.55  PASS
pcmciplus              lag          0.4072      4    0.51  PASS
directlingam           contemp      0.5283      3    0.01  PASS
dagma_linear           contemp      0.8664      3    0.28  PASS
dynotears              contemp      0.8323      3    0.11  PASS
pc                     contemp      1.0000      3    0.00  PASS
fci                    contemp      0.5000      3    0.00  PASS
fci_ts                 contemp      1.0000      4    0.04  PASS
var_ols[lag-index]     lag          0.2515      1    0.00  PASS

14/14 conventions verified
```

`var_ols[lag-index]` additionally checks that the lag-2-only edge `0→3` lands in
`A_lag[1]` and not `A_lag[0]`, so the lag axis is indexed correctly too.

### The self-test caught a real bug

On the first run `dagma_linear` reported `FAIL-TRANSPOSED (margin −0.8043)`.
The conversion was right; the **default `standardize=True` was wrong**. DAGMA's
`l2` score assumes equal *noise* variance, and z-scoring equalises the
*marginal* variances instead, destroying the varsortability signal it leans on —
the DAG came back reversed. Measured directly:

```
standardize=True   margin = -0.8043   (DAG returned REVERSED)
standardize=False  margin = +0.8664   (correct)
```

`dagma_linear` and `dynotears` therefore default to `standardize=False`, unlike
every other wrapper. This is the same varsortability effect that
`bench/sortability.py` measures; on real data it is not academic (§5).

---

## 4. The baselines

`run_baseline(name, X, max_lag=8, timeout_s=600.0, **kw) -> BaselineResult`.
`X` is `(T, P)`, row axis = time; NaNs are linearly interpolated (most of these
libraries reject NaN outright). It **never raises**: exceptions, SIGALRM
timeouts, subprocess timeouts and `MemoryError` all come back as
`ok=False, err=…` with zero-filled arrays that are safe to score.

| name | lagged? | contemp? | notes |
|---|---|---|---|
| `var_ols` | ✔ | ✔ | pooled OLS-VAR; `score='tstat'` (default) / `'abscoef'` / `'coef'` |
| `granger_pairwise` | ✔ | ✖ | bivariate F-test per ordered pair; per-lag `|t|` in `A_lag` |
| `varlingam` | ✔ | ✔ | `lingam.VARLiNGAM` |
| `directlingam` | opt | ✔ | `augment_lags=True` adds a time-ordered prior-knowledge matrix |
| `dynotears` | ✔ | ✔ | our numpy re-implementation |
| `dagma_linear` | ✖ | ✔ | contemporaneous only |
| `pc`, `fci` | ✖ | ✔ | causal-learn on the cross-section |
| `pc_ts`, `fci_ts` | ✔ | ✔ | same, on the lag-augmented window; lag edges oriented by time |
| `pcmci`, `pcmciplus` | ✔ | ✚ | tigramite, **out-of-process**; contemporaneous for PCMCI+ only |

`meta['scores_lagged'] / ['scores_contemp']` tell the metric layer which
channels a method can populate, so contemporaneous-only methods are not
silently scored as if they had failed on the lag edges.

### `var_ols` is first-class, deliberately

On CausalRivers a plain VAR beats every published causal-discovery method
(AUROC .85 vs VARLiNGAM .72, PCMCI .60). If CAFÉ-CD cannot beat it there is
nothing to publish, so it is a proper implementation: exact OLS on the full lag
design with one shared Gram inverse (identical regressors in every equation, so
SUR collapses to equation-by-equation OLS), classical t-statistics with the
right dof, `pinv` fallback plus a `rank_deficient` flag, optional ridge, and a
residual **partial-correlation** contemporaneous channel with a Fisher-z test
(symmetric — `meta['contemp_symmetric']=True`). Coefficients and t-statistics
are both kept in `meta` regardless of the `score` chosen.

---

## 5. Datasets

Cached under `data/causal/`, which is in `.gitignore` (≈1.4 GB total).

### CausalRivers (ICLR 2025)

* `https://github.com/CausalRivers/causalrivers`, release asset `product.zip`.
* **Download size checked before pulling: 133.2 MB compressed → 1.0 GB
  extracted.** Under the ~2 GB budget, so fetched in full.
* Contents: `rivers_ts_east_germany.csv` (619 MB, 175 296 rows × 666 stations,
  2019-2023 @ 15 min), `rivers_ts_bavaria.csv` (466 MB, 494 stations),
  `rivers_ts_flood.csv` (0.8 MB, 3 011 rows × 42, Sep/Oct-2024 window), three
  pickled NetworkX DiGraphs (east 666 nodes/651 edges, bavaria 494/490, flood
  42/42) and three metadata CSVs.
* Licence: data under *Datenlizenz Deutschland – Namensnennung – Version 2.0*
  (dl-de/by-2-0) from the German state environmental agencies. The upstream
  repo ships **no LICENSE file**; its code is used as a written reference only —
  the sampler, preprocessing and scoring are re-implemented in
  `bench/datasets_causal.py`.
* A 5.3 MB shallow clone of the repo is kept at
  `data/causal/repo_causalrivers/` purely as reference material.

**Subgraph sampler verified against the paper's Table 6** (counts on
RiversEastGermany):

| set | ours | paper | |
|---|---|---|---|
| random_3 | 1196 | 1196 | ✔ |
| random_5 | 7521 | 7521 | ✔ |
| close_3 | 636 | 636 | ✔ |
| close_5 | 637 | 637 | ✔ |
| root_cause_3 | 649 | 649 | ✔ |
| root_cause_5 | 655 | 655 | ✔ |
| confounder_3 | 24 | 24 | ✔ |
| 1_random_3 | 651 | 651 | ✔ |
| 1_random_5 | 2790 | 2790 | ✔ |
| confounder_5 | **332** | **361** | ✘ — explained |

`confounder_5` differs because upstream's `select_confounder_samples`
concatenates each confounder's candidate list without deduping *across*
confounders, so a subgraph containing two confounders is counted twice.
Summing without our final dedupe gives exactly 361. Nine of ten sets match
exactly, so the sampler is faithful.

### CausalDynamics (NeurIPS 2025 D&B, MIT)

* HuggingFace `kausable/CausalDynamics`.
* **Sizes checked before downloading**: `inputs/simple.tar.gz` **43.9 MB**,
  `inputs/climate.tar.gz` 54.2 MB, `inputs/coupled.tar.gz` **6.51 GB**;
  `outputs/{simple,climate,coupled}.tar.gz` 0.42 / 0.01 / 13.2 MB.
* **All three tiers are now fetched** — 6.61 GB downloaded → 10.2 GB extracted,
  member counts verified against the tarballs (`simple` 585/585, `climate`
  12/12, `coupled` 14 096/14 096 `.nc`). `coupled` is over the harness's ~2 GB
  budget and is opted in **by name** (`CD_OVER_BUDGET_OK`) rather than by
  raising the budget for everything; any *other* over-budget subset still needs
  `force=True`. The download is resumable via HTTP `Range`.
* Layout: `simple/<noise=…_confounder=…>/data/<System>_N10_T1000.nc` — netCDF4
  (HDF5, read with h5py, no xarray needed) holding `time_series (1000, 10, 3)`
  = 10 independent realisations of one 3-variable system, and
  `adjacency_matrix (3, 3)`. 10 noise/confounder settings × 57 systems.
* `coupled/<coupling=…_noise=…_systems=…_confounder=…_standardize=…_timelag=…_activation=…>/`
  — 720 setting directories, 14 096 systems. `time_series` is 4-D
  `(T, n_traj, n_nodes, dim)`; `eval.py` keeps only component 0 of each node, so
  a coupled system is a **partially observed** `n_nodes`-variable problem. The
  label is `adjacency_matrix_summary`, stored `[cause, effect]` — **the
  transpose of `simple`** (§6c(c)). `climate/` uses neither scheme: two
  directories, one per idealized model (§6d).
* `outputs/` ships the **authors' own per-system eval metrics** for 10 methods,
  so the leaderboard can be recovered without re-running anything —
  `causal_dynamics_published_eval(method)`.
* Ground-truth convention confirmed empirically, not from the docstring: the
  file for Rössler is `[[0,1,1],[1,1,0],[1,0,1]]`, exactly the rows of the
  Rössler Jacobian (`ẋ=−y−z`, `ẏ=x+ay`, `ż=b+z(x−c)`). So
  `adjacency_matrix[i,j]=1 ⟺ ∂ẋ_i/∂x_j ≠ 0 ⟺ j→i`.

### CauseMe — artefacts built, nothing submitted

`causeme_submission_spec()` returns the contract; `causeme_build_submission()`
writes a valid file **to disk only**. `bench/exp_bench_causeme.py` now builds
and validates those files for all 12 baselines plus CAFE-CD on real, publicly
downloadable experiments — **§9**. The zips carry no ground truth, so there is
no local accuracy number and none is invented. What an entry requires:

1. Register the method on causeme.uv.es (name, parameters, description,
   optional paper/code URL). Uploading a code zip exposing
   `my_method(data, **params) -> (val_matrix, p_matrix, lag_matrix)` is what
   earns the "Validated" badge.
2. Registration returns a `method_sha`.
3. Download an experiment zip (a few hundred `.txt` datasets, each
   `np.loadtxt` → `(T, N)`). **Results must be ordered by sorted filename.**
4. Emit one **bz2-compressed JSON** dict with required keys `method_sha`,
   `parameter_values`, `model`, `experiment`, `scores`; optional `pvalues`,
   `lags`, `runtimes`. Each of `scores`/`pvalues`/`lags` is a list over
   datasets of **row-major flattened** `(N, N)` matrices.
5. Upload on "My Results" (rate-limited per user).

`A_ij` = score for link **i → j** — the transpose of ours. Scores are
non-negative, higher = more confident. Metrics: ROC-AUC over all `N×N` entries
**excluding self-links**, pooled over the experiment's `B` datasets; FPR/TPR
from p-values at α=0.05; F-measure with **β=0.5**; TLR on lags.

---

## 6. Reproducing published numbers

### 6a. CausalRivers — VAR, AUROC (Full TS): **we get 0.8493, they report 0.85**

The published quartet (VAR .85, CDMI .83, DYNOTEARS .79, VARLiNGAM .72,
PCMCI .60) is **Figure 3a, Experiment Set 2**, not Table 3: the **flood graph**
(42 nodes), "Random 5" sampling (2 139 subgraphs), scored on the **full
five-year east-Germany series**. `bench/datasets_causal.py` exposes that pairing
as region `"flood_full"` (flood graph + east-Germany CSV).

Protocol, transcribed from `benchmark.py` / `tools/*.py`:

* metric = **mean over samples of per-sample AUROC** ("Individual AUROC"), for
  the **best hyperparameter cell** (no label-free selection rule);
* **diagonal removed** (`remove_diagonal=True`) — self-loops are *not* scored;
* summary graph = **max over lags** of the (absolute) coefficient;
* preprocessing = round the 15-min index to `resolution` → groupby-mean →
  optional min-max normalise → linear interpolate → trim each sample to the
  first..last fully-observed row;
* grid = resolution {15min,1H,6H,12H,24H} × normalize {T,F} × max_lag {3,5}
  (× `var_absolute_values` {T,F} for VAR).

Run it: `python bench/datasets_causal.py --repro rivers`.

**Three VAR cells reproduce, on three different graph sets:**

| graph set | samples | metric | ours | published |
|---|---|---|---|---|
| flood_full / Random-5 (Fig. 3a "Full TS") | 2 139 | mean per-sample AUROC | **0.8493** | **0.85** |
| east / Random-3 (Table 3) | 1 196 | mean per-sample AUROC | **0.8229** | **0.82** |
| east / Random-5 (Table 3) | 7 521 | mean per-sample AUROC | **0.8029** | **0.80** |
| east / Random-3 (Table 7) | 1 196 | mean per-sample max-F1 | **0.8292** | **0.83** |

Cell: `var_ols`, 6 h, `normalize=False`, `standardize=False`, `contemp=False`,
`max_lag=3`, `score='abscoef'`.

**"F1 max" in Table 7 is the PER-SAMPLE max-F1, not the pooled one.** Upstream's
`score()` returns both ("Max F1" pooled, "Max individual F1" per-sample-mean);
the paper's Table 7 prints magnitudes that only the per-sample variant matches.
On east/Random-3 our pooled max-F1 is **0.696** while the per-sample mean is
**0.829** against a published 0.83 — a 0.13 spread that would look like a
serious disagreement if the wrong one were quoted. `score_causal_rivers` now
returns `f1max` (pooled) and `f1max_individual` (per-sample) side by side; use
`f1max_individual` when comparing against Table 7.

**Other methods on flood_full / Random-5 (2 139 samples, same protocol):**

| method | ours | published | note |
|---|---|---|---|
| `var_ols` | **0.849** | 0.85 | 6 h, `max_lag=3`, `score='abscoef'` |
| `dynotears` | **0.816** | 0.79 | 24 h, `max_lag=3`; we land 0.026 ABOVE |
| `varlingam` | **0.740** | 0.72 | 24 h, `max_lag=5`, `prune=True`, `criterion=None`; 0.02 ABOVE |
| `granger_pairwise` | 0.758 | — | no published counterpart; 24 h, `max_lag=3` |
| `pcmci` | see below | 0.60 | out-of-process; ~26 min/cell on 2 139 samples |

**The DYNOTEARS run exposes a second rescaling failure, and it explains four
published cells.** With `normalize=True` (the benchmark's min-max step), our
DYNOTEARS returns AUROC **exactly 0.500** and finishes in 2 s instead of 25 min:
squeezed into [0, 1], every series is small enough that the L1 penalties
(`lambda_w = lambda_a = 0.1`) drive the solution to the **all-zero graph**, so
the prediction is constant. That is almost certainly what the paper's Table 3
Dynotears entries of exactly `.50` (Random-3, Close-3, Close-5, Root-cause-3)
are: a degenerate empty graph, not a measurement. Reported "0.50" from a
gradient-based method on min-max data should always be checked for this.

**Three switches all had to be right, and each one alone is worth a lot.** This
is the protocol detail that matters most:

| variant | individual AUROC |
|---|---|
| full protocol (above) | **0.849** |
| + our contemporaneous partial-correlation channel folded into the summary | 0.836 |
| `score='tstat'` instead of `'abscoef'` | 0.610 |
| `standardize=True` (wrapper z-scores) | **0.413** |
| `normalize=True` (benchmark min-max) | **0.470** |

Rescaling the series in *either* place takes VAR from 0.85 to **below chance**.
The reason is stated in the CausalRivers paper itself: discharge volume grows
downstream, so the raw coefficient **magnitude** carries the direction. A VAR
that scores `|coef|` on unrescaled data is exploiting a hydrological
scale prior, not conditional independence — and any z-scoring wrapper (ours by
default, and most published harnesses) silently destroys it. A t-statistic,
being scale-free, only reaches 0.61.

Resolution sweep on the full 2 139 subgraphs (`normalize=False`,
`standardize=False`, `score='abscoef'`, contemporaneous channel **on**, so all
values sit ~0.013 below the headline cell):

| resolution | max_lag 3 | max_lag 5 |
|---|---|---|
| 15 min | 0.795 | 0.792 |
| 1 h | 0.788 | 0.793 |
| **6 h** | **0.836** | 0.831 |
| 12 h | 0.832 | 0.835 |
| 24 h | 0.828 | 0.829 |

**Caveat on the F1 figures in the brief.** The numbers "MSSD 0.46, tsFCI 0.40,
DYNOTEARS 0.37, PCMCI 0.36, VARLiNGAM 0.17" do **not** appear anywhere in the
CausalRivers ICLR 2025 paper. Its F1-max table (Table 7, RiversEastGermany,
diagonal removed) reads VAR .83/.65, Varlingam .81/.63, PCMCI .70/.52,
Dynotears .62/.54 for Random 3/5 — VARLiNGAM 0.17 is nowhere near anything
reported, and MSSD/tsFCI are not evaluated in that paper at all. Those figures
are from some other source (possibly the TCD-Arena follow-up the repo links to)
and should be re-sourced before being used as a target. The real published
numbers are transcribed into `CR_PUBLISHED` in `bench/datasets_causal.py`.

### 6b. CausalDynamics — reproduced to 0.001, and two published cells are broken

`python bench/datasets_causal.py --repro dynamics`, on
`simple / noise=0.00_confounder=False`, all 54 scored systems (3 of 57 are
skipped because their ground truth is all-ones), 10 realisations each,
z-standardised, **self-loops included**:

| variant | Ind AUROC | Ind AUPRC | shipped |
|---|---|---|---|
| `dynotears`, binarised, **transposed** (= their wrapper) | **0.425** | 0.675 | **0.426 / 0.675** |
| `dynotears`, binarised, orientation fixed | 0.489 | 0.695 | — |
| `varlingam`, `prune=False`, binarised | **0.500** | 0.687 | **0.500 / 0.693** |
| `varlingam`, `prune=True`, binarised (= their wrapper) | 0.515 | 0.699 | — |
| `var_ols`, `|coef|` | 0.522 | 0.726 | — |
| `var_ols`, `|t|` | 0.487 | 0.736 | — |

Our from-scratch DYNOTEARS reproduces their published cell to **0.001** — but
only when we replicate their bug. Two findings:

1. **Their DYNOTEARS is transposed.** `baselines/dynotears.py` builds
   `adj_matrix[i, j]` = "i at lag τ → j at lag 0" (cause-row) and scores it
   against a ground truth that is `[effect, cause]`. That is exactly why the
   published number, 0.426, is *below* chance. Fixing the orientation moves it
   to 0.489 — from "clearly wrong" to "chance".
2. **Their VARLiNGAM is degenerate on most cells.** It binarises with
   `abs(adjacency_matrices_[tau_max]) > 0`; wherever lingam's adaptive-lasso
   prune leaves the lag matrix dense the prediction is all-ones and AUROC is
   **exactly 0.500 by construction**, not by measurement. That happens on
   **28.3%** of `simple` cells. Note `lingam.VARLiNGAM` defaults to
   `prune=True` and the shipped wrapper does not override it, so their matrix
   *is* pruned — `prune=False` also lands on 0.500 but is not the mechanism
   behind the published number; their per-system AUROC has sd 0.124 and spans
   0.283–0.875. `cutsplus` and `rcd` are also exactly 0.500 on `simple`, and
   §6c(e) measures the same rates on `coupled`, where they are very different.

More broadly: on CausalDynamics-`simple` **nothing beats chance** — the shipped
eval spans 0.426–0.518 AUROC against a 0.500 null, with a 0.687 null AUPRC that
every method sits within ±0.02 of. It is a fine stress test but not currently a
place where a win means anything. Worse than that: §6c(b) measures the floor for
a random *ranker* rather than the all-zeros constant the benchmark uses, and it
is **0.493 AUROC / 0.759 AUPRC** — so on `simple` every published AUPRC (.67–.70)
is *below chance*, not merely near it.

The `coupled` split is where the discrimination is. The 6.51 GB download has now
been taken in full; **§6c** is the audit of it. Short version: `coupled` does
separate methods, but on three of its five published rows — including the
headline Default row — a constant identity matrix beats every shipped method.
The **Confounder** row is the one where a win still means something.


### 6c. CausalDynamics `coupled` — reproduced; three of its five rows are beaten by a constant

`.venv-bench/bin/python bench/datasets_causal.py --audit coupled`
(`--tables-only` re-prints everything from `data/causal/cache/cd_coupled_audit.json`
without re-reading 10 GB of netCDF or re-running a single fit).

`coupled` support was added to `load_causal_dynamics` while a detached job was
importing it, so **the `simple` path is checked, not assumed, to be unchanged**:
`--selftest` hashes every system of both default cells (X, A, `A_native`, and
the orientation flags) against `CD_CONTRACT_DIGESTS` — `simple` n=57 digest
`35a2cb4d…`, `coupled` n=100 digest `4853c757…`. Changing the loader without
updating those constants is a hard failure. Confirmed at the *results* level
too: after every change here, `--repro dynamics` still returns the §6b table
bit for bit (`varlingam` 0.500 / 0.687 and 0.515 / 0.699; `dynotears`
transposed 0.425 / 0.675, orientation fixed 0.489 / 0.695).

#### The download, measured before pulling

`HEAD` on `huggingface.co/datasets/kausable/CausalDynamics/resolve/main/{inputs,outputs}/<subset>.tar.gz`,
re-checked 2026-09-08:

| subset | inputs | outputs | extracted | `.nc` in tar → on disk |
|---|---|---|---|---|
| `simple` | 43 921 694 B (43.9 MB) | 420 473 B | 109 MB | 585 → 585 |
| `climate` | 54 233 197 B (54.2 MB) | 14 954 B | 62 MB | 12 → 12 |
| `coupled` | **6 514 628 417 B (6.51 GB)** | 13 167 388 B | 10 GB | 14 096 → 14 096 |

6.51 GB is what was expected, so **nothing was subsetted**: all three tiers are
complete, 6.61 GB downloaded → 10.2 GB on disk. The member counts are
`tar tzf | wc -l` against the extracted tree, not an assumption that extraction
succeeded. The download is resumable (`_download_resumable` continues a partial
file with an HTTP `Range` request) because restarting 6.5 GB after a dropped
connection is not acceptable.

#### The cell

The published coupled **Default** row is `noise=0.00, systems=10,
confounder=False, standardize=False, timelag=0`, with `coupling` ∈ {linear,
nonlinear, periodic} and `activation` ∈ {none, mixed} **free** — 6 setting
directories, **100 systems**, 10 trajectories each. `time_series` is
`(T, n_traj, n_nodes, dim)` and `eval.py` keeps only component 0 of every node,
so each system is a **partially observed** 10-variable problem, not a 30-variable
one; `load_causal_dynamics` does the same.

One discrepancy worth recording: the site's prose says this row covers "4745
graphs". No pinning of the released artefact contains 4745 systems (the closest
is 4729 for `systems=3`; the whole coupled tier is 14 096), whereas this
100-system pinning reproduces 5 of the 7 published cells to ≤ 0.026 (below).
The pinning is right; the count is prose.

#### (a) The published row, re-derived from the shipped `outputs/coupled.tar.gz`

No re-running: `outputs/coupled.tar.gz` (13 MB) already contains the authors'
own per-system Individual/Joint AUROC, AUPRC, SHD and null for every method.
"constant" is the fraction of cells whose AUROC is **exactly** 0.500 — a
constant prediction, where 0.5 is true by construction rather than by
measurement.

| method | Ind AUROC | sd | Ind AUPRC | published | n | > 0.5 on | constant cells | *d* |
|---|---|---|---|---|---|---|---|---|
| `pcmciplus` | 0.676 | 0.032 | 0.264 | .68 / .27 | 100 | 100% | 0% | 5.6 |
| `tsci` | 0.654 | 0.043 | 0.323 | .69 / .36 | 100 | 100% | 0% | 3.6 |
| `fpcmci` | 0.646 | 0.086 | 0.241 | .67 / .24 | 100 | 77% | **23%** | 1.7 |
| `dynotears` | 0.621 | 0.049 | 0.259 | .66 / .32 | 100 | 100% | 0% | 2.5 |
| `varlingam` | 0.588 | 0.032 | 0.187 | .57 / .18 | 100 | 96% | 4% | 2.7 |
| `cutsplus` | 0.504 | 0.013 | 0.159 | .50 / .16 | 100 | 31% | **56%** | 0.3 |
| `rcd` | 0.500 | 0.005 | 0.155 | — | 100 | 40% | **28%** | −0.0 |
| `ngc_lstm` | 0.500 | 0.005 | 0.153 | .50 / .16 | 100 | 56% | 0% | −0.0 |
| `tcdf` | 0.497 | 0.022 | 0.156 | — | **3** | 33% | 0% | −0.1 |
| `grasp` | 0.489 | 0.021 | 0.155 | — | **31** | 35% | 3% | −0.5 |

`sd`, `> 0.5 on` and *d* = (mean − 0.5)/sd are per-**system**, not per-cell
averages: "0.676 on average" and "above 0.5 on 100 of 100 systems" are different
claims and only the second one shows the separation is not an artefact of
averaging.

Five of seven published cells land within 0.026; `dynotears` (−0.039/−0.061) and
`tsci` (−0.036/−0.037) are further off, both in the same direction, which points
at a data revision rather than at the pinning. Averaging any *other* pinning is
much worse (mean absolute error 0.018 for this row vs 0.067–0.088 for every
looser pinning tried, including all 720 cells).

Note `tcdf` produced eval files for **3 of 100** systems and `grasp` for 31 —
both crash on most of the tier and neither appears in the published table.

#### (b) The chance floor, measured rather than assumed

The benchmark's own `Null_AUROC`/`Null_AUPRC` come from an **all-zeros constant**
prediction. For a ranking metric that is the wrong reference: with a constant
score AUROC is 0.5 by tie-averaging and average precision collapses to the
positive rate. `causal_dynamics_trivial_baselines` instead pushes label-free
predictors through eval.py's own scorer — 20 draws per system, benchmark
protocol (**diagonal kept**), and again with the diagonal masked:

| tier | predictor | AUROC | AUPRC | off-diag AUROC | off-diag AUPRC |
|---|---|---|---|---|---|
| coupled (100 systems, positive rate 0.153, off-diagonal 0.100) | uniform-random ranker | 0.499 | **0.188** | 0.495 | **0.140** |
| | all-zeros (their `Null`) | 0.500 | 0.153 | 0.500 | 0.100 |
| | **identity matrix** | **0.682** | **0.353** | 0.500 | 0.100 |
| | identity + random off-diagonal | 0.685 | 0.439 | 0.505 | 0.143 |
| simple (54 systems, positive rate 0.687, off-diagonal 0.728) | uniform-random ranker | 0.493 | **0.759** | 0.500 | 0.806 |
| | all-zeros (their `Null`) | 0.500 | 0.687 | 0.500 | 0.723 |
| | identity matrix | 0.453 | 0.681 | 0.500 | 0.723 |

Two things follow, and both matter more than any single method's score.

**1. The published AUPRC null is too low, on both tiers.** A random *ranker*
scores 0.188 on coupled and 0.759 on simple, not 0.153 and 0.687 — average
precision is upward-biased at these tiny sample sizes (9 decisions per simple
system, 100 per coupled system). Consequences: on `simple` **every published
AUPRC (.67–.70) is below the random-ranker floor**; on `coupled` NGC (.16) and
CUTS+ (.16) are below it and VARLiNGAM (.18) is at it. (The off-diagonal
floors, for §6c(d): 0.495 / 0.140 on coupled.)

**2. On `coupled`, a constant identity matrix scores 0.682 / 0.353.** No
parameters, no data, no fit. On the published numbers that is above five of the
seven AUROCs, level with PCMCI+ (.68), and below only TSCI's .69; on AUPRC it is
above six of seven and just below TSCI's .36. Re-derived from the shipped eval
(§6c(a)) it is above **all ten** methods on both metrics, and paired system by
system (§6c(b2)) not one of them beats it. The mechanism is arithmetic: CausalDynamics **keeps the diagonal** (§7), the coupled
ground truth's diagonal is 63% dense against a 10% dense off-diagonal, and the
diagonal therefore carries **40.8% of the entire positive class**. Anything that
reliably fires on autocorrelation — trivial in an ODE system — banks 41% of the
positives before looking at a single cross-variable relation. On `simple` the
same predictor scores 0.453, *below* chance, so this is specific to `coupled`.

#### (b2) Paired against the identity matrix — the Default row

Comparing published means to a floor is weak. The shipped eval stores a
**per-system** Individual AUROC/AUPRC, and the identity predictor can be scored
on exactly those systems, so the comparison is paired system by system
(`causal_dynamics_identity_paired`, Wilcoxon signed-rank, n = 100):

| method | AUROC | identity | p | AUPRC | identity | p | method wins on |
|---|---|---|---|---|---|---|---|
| `pcmciplus` | 0.676 | 0.682 | 0.21 | 0.264 | 0.353 | 5e-18 | 47% / **2%** of systems |
| `tsci` | 0.654 | 0.682 | 9e-09 | 0.323 | 0.353 | 4e-04 | 29% / 33% |
| `fpcmci` | 0.646 | 0.682 | 5e-03 | 0.241 | 0.353 | 5e-18 | 43% / 2% |
| `dynotears` | 0.621 | 0.682 | 7e-15 | 0.259 | 0.353 | 9e-17 | 17% / 9% |
| `varlingam` | 0.588 | 0.682 | 5e-18 | 0.187 | 0.353 | 4e-18 | 2% / 0% |
| `cutsplus` | 0.504 | 0.682 | 4e-18 | 0.159 | 0.353 | 4e-18 | 0% / 0% |
| `ngc_lstm` | 0.500 | 0.682 | 4e-18 | 0.153 | 0.353 | 4e-18 | 0% / 0% |
| `rcd` | 0.500 | 0.682 | 4e-18 | 0.155 | 0.353 | 4e-18 | 0% / 0% |
| `grasp` (n=31) | 0.489 | 0.677 | 9e-10 | 0.155 | 0.339 | 9e-10 | 0% / 0% |

**Not one of the nine methods with enough eval files to test beats a constant
identity matrix** (`tcdf`, at 3 systems, is too small to test at all). The best
of them, PCMCI+, is statistically indistinguishable from it on AUROC (p = 0.21)
and significantly *worse* on AUPRC, winning on 2% of systems. TSCI — the
published AUROC leader at .69 — is significantly worse than the identity matrix
on both metrics.


#### (b3) Every published `coupled` row against the identity matrix — where to compete

The Default row is not the whole tier. The leaderboard publishes five coupled
rows, each varying one axis, and **the diagonal's grip on the metric is not
constant across them**. `causal_dynamics_all_rows_vs_identity` runs the same
paired test on all five:

| row | cells | systems | identity AUROC | identity AUPRC | beats identity on AUROC | on **both** metrics |
|---|---|---|---|---|---|---|
| Default | 6 | 100 | 0.682 | 0.353 | — | — |
| Noise | 24 | 480 | 0.685 | 0.357 | — | — |
| **Confounder** | 6 | 101 | **0.546** | **0.183** | `pcmciplus` | **`pcmciplus`** |
| Lag | 6 | 103 | 0.544 | 0.236 | `pcmciplus`, `fpcmci` | — |
| Standardize | 6 | 99 | 0.683 | 0.354 | — | — |

On **Default, Noise and Standardize** the identity matrix sits at ≈ 0.68 / 0.35
and **nothing in the shipped suite beats it on either metric** — PCMCI+ merely
ties on AUROC (p = 0.21 and 0.33) and loses on AUPRC by 0.09.

On **Confounder** the identity predictor drops to 0.546 / 0.183 — the
confounded graphs dilute the diagonal — and **PCMCI+ genuinely clears it on
both metrics**: 0.593 vs 0.546 AUROC (p = 4e-12) and 0.208 vs 0.183 AUPRC
(p = 3e-13). That is a real result on a real benchmark.

On **Lag**, PCMCI+ (0.560) and F-PCMCI (0.552) clear the 0.544 identity AUROC,
but neither clears it on AUPRC (0.236 vs 0.236) — a half win among the shipped
suite, though our own baselines clear it comfortably on both (below).

**What those two rows actually cost to win** (`--audit coupled` block (b4)).
Our own baselines, same protocol, aligned orientation, diagonal kept —
`causal_dynamics_coupled_repro(causal_dynamics_row_cells(row))`:

**Confounder** (101 systems, positive rate 0.156, diagonal share of positives
**19.0%**):

| variant | AUROC | AUPRC | off-diag AUROC | off-diag AUPRC |
|---|---|---|---|---|
| random ranker | 0.499 | 0.197 | 0.500 | 0.189 |
| identity matrix | 0.546 | 0.183 | 0.500 | 0.143 |
| `pcmciplus` (shipped eval, best of the suite) | **0.593** | 0.208 | — | — |
| `var_ols` (`score='tstat'`) | 0.590 | **0.312** | **0.556** | **0.227** |
| `var_ols` (`score='abscoef'`) | 0.543 | 0.244 | 0.536 | 0.219 |
| `varlingam` | 0.551 | 0.213 | 0.533 | 0.186 |
| `dynotears` | 0.538 | 0.240 | 0.499 | 0.169 |

**Lag** (103 systems, positive rate 0.202, diagonal share **17.0%**):

| variant | AUROC | AUPRC | off-diag AUROC | off-diag AUPRC |
|---|---|---|---|---|
| random ranker | 0.502 | 0.237 | 0.502 | 0.225 |
| identity matrix | 0.544 | 0.236 | 0.500 | 0.187 |
| `pcmciplus` (shipped eval, best of the suite) | 0.560 | 0.236 | — | — |
| `var_ols` (`score='abscoef'`) | **0.607** | 0.300 | **0.597** | **0.282** |
| `varlingam` | 0.585 | 0.298 | 0.565 | 0.277 |
| `var_ols` (`score='tstat'`) | 0.545 | **0.306** | 0.515 | 0.227 |
| `dynotears` | 0.540 | 0.265 | 0.515 | 0.216 |

On **Confounder** a **plain OLS VAR(1), ~1 ms per fit**, matches PCMCI+ on AUROC
(0.590 vs 0.593) and beats it by 50% on AUPRC (0.312 vs 0.208). On **Lag** it
beats PCMCI+ on both (0.607 / 0.300 vs 0.560 / 0.236) and is far above the floor
with the diagonal masked (0.597 / 0.282 vs 0.502 / 0.225). So both rows *are*
venues where clearing the trivial floor means something — but the bar to clear
is a linear VAR, not the published leaderboard. (Also worth recording:
`lingam.VARLiNGAM` fails outright on **561 of 1 010** confounded trajectories —
55% — which the harness scores as all-zero predictions, exactly as `eval.py`
does.)

**This is the actionable form of the finding.** The coupled tier is not
uniformly worthless; the rows the leaderboard leads with are. If CAFE-CD is to
be measured here, the row to compete on is **Confounder** (and secondarily
Lag), with `var_ols` `|t|` at 0.590 / 0.312 as the bar and the identity matrix
in the table. Reporting a Default-row number without the identity row beside it
is reporting a self-loop detector.

#### (c) Orientation of the stored label, measured on all 100 systems

`simple` stores the Jacobian pattern in `adjacency_matrix` (`[effect, cause]`,
this repo's convention); `coupled` stores an SCM summary in
`adjacency_matrix_summary` (`[cause, effect]`, the transpose). **The two subsets
of the same benchmark disagree.** Verified from the data, not the docstring:
`scm.get_root_nodes_mask` is `~A.T.bool().any(dim=1)`, i.e. a root node is one
with **no incoming edges**, so in `[cause, effect]` storage its off-diagonal
*column* is empty. Across the 100 systems / 630 root nodes of the Default row:

* off-diagonal **column** empty: **100.0%**
* off-diagonal **row** empty: **0.0%**

`load_causal_dynamics` therefore transposes `coupled` into this repo's
`[effect, cause]` convention and keeps the as-stored matrix in
`meta['A_native']`, which is what every reproduction scores against because that
is what `eval.py` scores against.

#### (d) Our own runs: reproduction, and what each defect costs

`causal_dynamics_coupled_repro` fits each baseline **once** per trajectory and
then scores it under every (orientation × binarisation × diagonal) variant, so
the columns differ only in the scoring. Protocol is `eval.py`'s exactly:
`np.nan_to_num`, z-standardised over (trajectory, time), 10 trajectories per
system, scored against `meta['A_native']`, diagonal kept, all-ones truths
skipped. `T = 1` transposes our `[effect, cause]` estimate into the coupled
`[cause, effect]` storage order, i.e. **aligned with the truth**.

| variant | T | bin | AUROC | AUPRC | off-diag AUROC | off-diag AUPRC | |
|---|---|---|---|---|---|---|---|
| `varlingam` | 0 | 1 | **0.597** | **0.191** | 0.531 | 0.111 | **= their wrapper** (shipped 0.588 / 0.187, published .57 / .18) |
| `varlingam` | 1 | 1 | 0.612 | 0.198 | 0.555 | 0.120 | orientation fixed |
| `varlingam` | 1 | 0 | 0.706 | 0.380 | 0.603 | 0.218 | + binarisation removed |
| `dynotears` | 1 | 1 | **0.629** | **0.291** | 0.528 | 0.135 | **= their wrapper** (shipped 0.621 / 0.259, published .66 / .32) |
| `dynotears` | 0 | 1 | 0.623 | 0.281 | 0.517 | 0.125 | mis-oriented |
| `dynotears` | 1 | 0 | 0.633 | 0.349 | 0.528 | 0.147 | binarisation removed |
| `var_ols` (`score='abscoef'`) | 1 | 0 | **0.730** | **0.390** | **0.656** | **0.228** | not in their suite |
| `var_ols` (`score='tstat'`) | 1 | 0 | 0.665 | 0.449 | 0.545 | 0.158 | not in their suite |

**Reproduction.** Our from-scratch VARLiNGAM in their wrapper's exact
configuration lands 0.009 AUROC / 0.004 AUPRC from the shipped eval; DYNOTEARS
lands 0.008 / 0.032. Both are within the gap the shipped eval itself has to the
website (§6c(a)), so the cell is reproduced.

**And a plain OLS VAR(1) ranked by coefficient magnitude — ~1 ms per fit,
not in their suite — scores 0.730 / 0.390, above every published method** (best published .69 /
.36), above the identity matrix, and it is the only entry here that is clearly
above the floor once the diagonal is removed (0.656 / 0.228 against an
off-diagonal floor of 0.495 / 0.140).

#### (e) The two `simple` defects, re-checked on `coupled` — plus a third

`CD_WRAPPER_AUDIT` records what every wrapper's last line actually emits, read
off `CausalDynamics/src/causaldynamics/baselines/*.py`.

**1. The transposition defect is real on `coupled` too, but it hits the *other*
half of the suite.** Because the two subsets store opposite conventions, the
same unchanged wrapper code is mis-oriented on different methods in each:

| wrapper | emits | on `coupled` (truth `[cause, effect]`) | on `simple` (truth `[effect, cause]`) |
|---|---|---|---|
| `dynotears`, `tsci`, `pcmciplus`, `fpcmci`, `ngc_lstm`, `cutsplus`, `tcdf` | `[cause, effect]` | **correct** | **transposed** |
| `varlingam`, `rcd` | `[effect, cause]` | **transposed** | correct |

So `DYNOTEARS`' published coupled .66 is *not* affected by the bug that produced
its below-chance .43 on `simple` — and `VARLiNGAM`'s coupled .57 *is*. Fixing
VARLiNGAM's orientation is worth +0.015 AUROC binarised (+0.039 unbinarised);
it does not change the ranking. This is a smaller effect than on `simple`, and
that is the honest finding: **on `coupled` the transposition is real but not
what is wrong with the leaderboard.**

**2. The VARLiNGAM all-ones defect is largely absent on `coupled`.**
`abs(adjacency_matrices_[tau_max]) > 0` is all-ones — AUROC 0.500 by
construction — on **28.3% of `simple` cells but only 0–4% of `coupled` cells**
(4% in the shipped eval, 2% across our 1 000-fit re-run, 0 of 10 in the live
probe), because the coupled series make lingam's adaptive-lasso prune bite. The binarisation is still costly where it is
*not* degenerate: it takes our VARLiNGAM from 0.706 / 0.380 to 0.612 / 0.198,
i.e. **−0.094 AUROC and −0.182 AUPRC purely from discarding the ranking that the
metric needs**. The methods that *are* structurally degenerate on `coupled` are
different ones: `cutsplus` (56% of cells constant), `rcd` (28%) and `fpcmci`
(23% — the published #3 method returns a constant prediction on nearly a quarter
of the row).

**3. A third defect, previously unrecorded, in `grasp` — and it affects both
subsets.** `baselines/perm.py` is two lines: `G = CL_GRASP(X); self.adj_matrix =
G.graph`. But causal-learn's `GeneralGraph.graph` is an **endpoint** matrix, not
an adjacency: `graph[j,i] = 1` with `graph[i,j] = -1` encodes `i → j`. That
matrix goes straight into `roc_auc_score` **as a score**, so a `-1` ranks
*below* the `0` that means "no edge" — every detected edge's tail endpoint is
scored as more confidently absent than an untested pair. Measured live
(`causal_dynamics_wrapper_probe`, 10 coupled systems): **19% of the submitted
matrix is `-1`**, and taking `abs()` first moves AUROC from **0.442 to 0.545** —
the encoding bug alone costs 0.103 AUROC. That is why `grasp` sits *below*
chance (0.489) on the row, and paired against the identity matrix it loses on
31 of 31 systems (p = 9e-10).
`grasp` and `tcdf` are run by `eval.py` but do not appear in the published
table; `tcdf` additionally crashes on 97 of the 100 Default systems.

#### Verdict on `coupled` as a venue

`coupled` is **not** degenerate the way `simple` is. `simple` spans 0.426–0.518
against a 0.500 null and nothing in it separates. On `coupled`, four of the ten
shipped methods are above 0.5 on 96–100% of the 100 systems individually
(`pcmciplus` 100%, `tsci` 100%, `dynotears` 100%, `varlingam` 96%) and `fpcmci`
on 77%, with per-system effect sizes of 1.7–5.6 standard deviations. The
discrimination is real.

But the thing it discriminates is **mostly autocorrelation, not causal
structure**:

* on the Default row 40.8% of the positive class is the diagonal, which the
  protocol keeps;
* a constant identity matrix scores 0.682 / 0.353 there and **no shipped method
  beats it** (§6c(b2)) — nor on `Noise` or `Standardize` (§6c(b3));
* with the diagonal masked, the published-style methods we can re-run collapse
  to 0.528–0.555 AUROC against a 0.495 floor, while a 1 ms OLS VAR reaches
  0.656;
* five of the ten shipped methods are broken in identifiable ways — `grasp`
  (endpoint matrix fed to `roc_auc_score` as a score), `varlingam` and `rcd`
  (orientation), `cutsplus`, `rcd` and `fpcmci` (constant predictions on 56 /
  28 / 23% of cells) — and `tcdf` runs on 3 of 100 systems.

**So: a win on the coupled *Default* number means very little on its own — it
can be bought with a constant.** But the tier is not uniformly degenerate, and
§6c(b3) says exactly where it is not: on **Confounder** and **Lag** the
diagonal carries only 19% and 17% of the positives, the identity floor falls to
≈ 0.545, and methods separate from it — so a win there is a real win. The
sobering part is that the strongest thing on both rows is a 1 ms OLS VAR, not
anything on the leaderboard. Practical rules for using this benchmark:

1. **Compete on `Confounder` and `Lag`**, against a plain OLS VAR — 0.590 /
   0.312 and 0.607 / 0.300 respectively — not against the published PCMCI+
   .56 / .20 and .55 / .23. Report `Default`, `Noise` and `Standardize` only
   alongside the identity row, or not at all.
2. **Always print the identity-matrix row** in any coupled table. It costs
   nothing and it is the difference between a result and an artefact.
3. **Always print the off-diagonal pair** next to the benchmark-native one.
4. **Use the measured random-ranker floor** (0.499 / 0.188 full, 0.495 / 0.140
   off-diagonal), not the shipped all-zeros `Null`.
5. `simple` remains unusable at any presentation, and `climate` (§6d) is one
   graph plus a row that does not reproduce.


### 6d. CausalDynamics `climate` — completed for coverage; one row is a single graph

The third tier (54.2 MB inputs, 15 kB outputs) is now fetched and loadable —
`CD_DEFAULT_SETTING['climate'] = 'coupled_atmos_ocean'`. It does **not** use the
`key=value` directory scheme: its two "settings" are the two idealized models,
and they are exactly the leaderboard's two climate rows.

| row | directory | systems | vars | node `dim` | positive rate |
|---|---|---|---|---|---|
| Atmos-ocean | `coupled_atmos_ocean` | **1** | 4 | 374 | 0.812 |
| ENSO modes | `coupled_enso_modes` | 11 | 10 | 1 | 0.666 |

Re-derived from the shipped eval, against the measured floor (20 random draws
per system, benchmark protocol):

| Atmos-ocean | AUROC | AUPRC | published | constant |
|---|---|---|---|---|
| uniform-random ranker | 0.481 | **0.831** | — | — |
| all-zeros (their `Null`) | 0.500 | 0.812 | — | — |
| **identity matrix** | 0.654 | 0.870 | — | — |
| `pcmciplus` | 0.692 | 0.885 | .69 / .88 | 0% |
| `dynotears` | 0.638 | 0.864 | .62 / .86 | 0% |
| `tsci` | 0.585 | 0.844 | .58 / .84 | 0% |
| `varlingam`, `fpcmci`, `cutsplus`, `rcd`, `tcdf` | 0.500 | 0.812 | .50 / .81 | **100%** |
| `ngc_lstm` | 0.496 | 0.811 | .49 / .81 | 0% |
| `grasp` | 0.485 | 0.810 | — | 0% |

Atmos-ocean **reproduces to ≤ 0.02** — but the row is *one graph*. Five of the
ten methods return a constant matrix on it (AUROC 0.500 by construction), the
published .81 AUPRCs are below the 0.831 random-ranker floor, and the identity
matrix again beats everything except PCMCI+.

**ENSO modes does not reproduce.** The shipped eval over its 11 systems gives
`fpcmci` 0.570, `pcmciplus` 0.569, `varlingam` 0.560, `dynotears` 0.552 AUROC,
against a published row of .50 / .51 / .51 / .50 — the published row is *lower*
than the artefact it ships with. And its published AUPRC column is a constant
**.81 for all seven methods**, which is the *Atmos-ocean* positive rate (0.812),
not ENSO's (0.666, and the shipped eval gives 0.666–0.701). That column looks
carried over from the row above it. With 11 systems this is a small sample, but
the AUPRC coincidence is not a sampling artefact.

Either way the tier is not a venue: on ENSO everything sits in 0.498–0.570
against a 0.502 floor, and Atmos-ocean is a single graph.

---

## 7. Evaluation protocol — and the self-loop trap

**The two benchmarks disagree about self-loops, and the mismatch alone moves the
score a lot.**

| | CausalRivers | CausalDynamics | CauseMe | this repo (internal) |
|---|---|---|---|---|
| self-loops `i→i` scored? | **No** — diagonal removed | **Yes** — diagonal is part of the Jacobian truth | **No** — "excluding self-links" | **Yes** — AR terms are usually the strongest true edges |
| decisions per P-node sample | `P(P−1)` | `P²` | `P(P−1)` | `P²` |
| headline metric | mean per-sample AUROC | mean per-realisation AUROC + AUPRC | pooled ROC-AUC | — |
| also reported | pooled AUROC, max-F1, max-accuracy | joint AUROC/AUPRC, SHD, null | FPR/TPR, F(β=0.5), TLR | — |

CausalRivers removes the diagonal because "rivers are highly autocorrelated and
the causal links here are not relevant". Our internal convention keeps it
because AR terms *are* real edges. **Scoring a CausalRivers sample with the
diagonal left in inflates every method**, since `i→i` is trivially the largest
VAR coefficient and would be counted as a true positive. `score_causal_rivers`
defaults to `remove_diagonal=True`; `score_causal_dynamics` never masks.

Other protocol points:

* **Adjacency scoring.** Both benchmarks score a `(P,P)` *summary* graph, not
  the lag-resolved tensor. `causal_baselines.summary_graph(res, how='max')`
  collapses `|A_lag|` over lags and (optionally) folds in `|A_contemp|`.
  `include_contemp` matters: CausalRivers' VAR has no contemporaneous term, so
  including ours costs 0.013 AUROC (§6a).
* **Ties.** AUROC via `sklearn.roc_auc_score` handles ties by averaging ranks,
  so a constant prediction scores exactly 0.5 — which is how the degenerate
  CausalDynamics baselines land on 0.500. For CPDAG/PAG outputs, undirected and
  bidirected edges are split 0.5/0.5 across both directions and `o→` scores
  0.75/0.25 (`_decode_general_graph`, tunable). Any such convention is a tie
  policy, and it should be stated whenever a constraint-based method is ranked
  against a score-based one.
* **F1 max** is the best F1 over *all* thresholds of the pooled prediction — an
  oracle-threshold number. It is not achievable without labels, and neither is
  the "best hyperparameter cell" AUROC. Both benchmarks report oracle numbers;
  our comparisons must either do the same for everyone, or report a label-free
  selection rule for everyone. Do not mix.
* **Failures** count. `reproduce_causal_rivers` scores a failed fit as an
  all-zero prediction (upstream does the same via `except: pred = zeros`) and
  reports `fails` separately, so a method cannot win by declining hard samples.

---

## 8. Re-running everything

```bash
# conventions (add --fast to skip the out-of-process tigramite checks)
.venv-bench/bin/python bench/causal_baselines.py --self-test
.venv-bench/bin/python bench/causal_baselines.py --list

# dataset inventory / fetch
.venv-bench/bin/python bench/datasets_causal.py
.venv-bench/bin/python bench/datasets_causal.py --fetch all

# published-number reproduction
.venv-bench/bin/python bench/datasets_causal.py --repro rivers
.venv-bench/bin/python bench/datasets_causal.py --repro dynamics

# loader contract: hashes BOTH CausalDynamics default cells and checks them
# against CD_CONTRACT_DIGESTS, so "coupled support was added without changing
# simple" is checked rather than asserted
.venv-bench/bin/python bench/datasets_causal.py --selftest

# CausalDynamics `coupled` audit (Sec. 6c): shipped leaderboard re-average,
# measured chance floor, the identity-matrix paired test on ALL FIVE published
# rows, the wrapper source audit + live defect probe, and our own reproduction.
# ~20 min the first time; the raw prediction cubes and every table are cached,
# and blocks added later are back-filled into an existing cache in place.
.venv-bench/bin/python bench/datasets_causal.py --audit coupled
.venv-bench/bin/python bench/datasets_causal.py --tables-only   # from cache only

# CauseMe (Sec. 9): artefacts only, nothing is uploaded
.venv-bench/bin/python bench/exp_bench_causeme.py --inventory
.venv-bench/bin/python bench/exp_bench_causeme.py --roundtrip
.venv-bench/bin/python bench/exp_bench_causeme.py --run --validate --agreement
.venv-bench/bin/python bench/exp_bench_causeme.py --tables-only
```

Measured costs on the full CausalRivers flood Random-5 set (2 139 subgraphs,
`max_lag=3`):

| method | 6 h (T≈7 000) | 24 h (T≈1 760) |
|---|---|---|
| `var_ols` | 2 s | 2 s |
| `granger_pairwise` | 40 s | 13 s |
| `varlingam` (`prune=False`) | ~1 min | ~35 s |
| `varlingam` (`prune=True`) | ~29 min | 2.5 min (`max_lag=3`: 11 min) |
| `dynotears` | ~15 min | 25 min |
| `pcmci` (subprocess per sample) | ~44 min | ~26 min |

Use `limit=` for smoke tests. The east-Germany sets are larger still
(Random-3 = 1 196 samples, Random-5 = 7 521); `var_ols` handles them in 8 s and
16 s respectively, everything else scales linearly from the table above.

---

## 9. CauseMe — 13 methods, submittable artefacts, nothing uploaded

`bench/exp_bench_causeme.py`. Cache `bench/bench_causeme_cache.json`,
`--tables-only` re-prints every table from it.

### 9a. What can and cannot be measured locally

The experiment zips are public — `https://causeme.uv.es/static/datasets/<experiment>.zip`,
no account needed — and `--inventory` prints their contents rather than trusting
the description:

| experiment | zip | members | extensions | shape | ground truth in the zip |
|---|---|---|---|---|---|
| `linear-VAR_N-5_T-300` | 1 183 kB | 200 | `.txt` only | (300, 5) | **no** |
| `nonlinear-VAR_N-5_T-300` | 1 168 kB | 200 | `.txt` only | (300, 5) | **no** |
| `logistic-deterministic_N-5_T-300_weak` | 999 kB | 200 | `.txt` only | (300, 5) | **no** |
| `river-runoff_N-12_T-4600` | 106 kB | 1 | `.txt` only | (4600, 12) | **no** |

2 662 experiment zips are reachable this way across 56 models. **None of them
ships a ground-truth graph** — that is the design of the platform: scoring
happens server-side after an upload, and `/rank/` is behind a login (it returns
the login page to an anonymous GET). So this harness cannot print a CauseMe AUC,
and any number presented as one would be fabricated. What it produces instead is
the artefact, validated, plus the three things that *are* locally measurable:
**orientation** (§9b), **runtime** and **cross-method agreement** (§9c).

### 9b. Orientation round-trip — the thing that silently inverts a submission

CauseMe is `A_ij = score for i → j` (**cause row, effect column**) — the
transpose of this repo's `A[effect, cause]` (§3). A transposed submission does
not score badly-but-plausibly, it scores *inverted*, and there is no local
signal that would catch it. `--roundtrip` therefore generates a sparse stable
VAR with a known graph, **writes it to disk in CauseMe's own on-disk format**,
reads it back through the identical loader → method → `causeme_matrix_from_result`
path, and scores it with CauseMe's own metric definitions (ROC-AUC over all `N×N`
entries **excluding self-links**, pooled over the B datasets; TPR/FPR and
F(β=0.5) from the p-values at α=0.05; TLR among correctly predicted links).

12 datasets, N=5, T=300, max lag 3, link density 0.254. **This is a synthetic
format test, not a CauseMe score.**

| method | AUC | AUC transposed | margin | TPR | FPR | F(β=0.5) | TLR | s/dataset |
|---|---|---|---|---|---|---|---|---|
| `pcmci` | **1.000** | 0.549 | +0.451 | 1.000 | 0.134 | 0.761 | 1.000 | 1.954 |
| `pcmciplus` | **1.000** | 0.514 | +0.486 | 1.000 | 0.056 | 0.884 | 0.984 | 1.284 |
| `var_ols` | **1.000** | 0.565 | +0.435 | 1.000 | 0.179 | 0.704 | 1.000 | 0.000 |
| `dynotears` | **1.000** | 0.551 | +0.449 | 1.000 | 0.425 | 0.501 | 1.000 | 0.157 |
| `granger_pairwise` | 0.990 | 0.590 | +0.400 | 1.000 | 0.268 | 0.614 | 1.000 | 0.004 |
| `pc_ts` | 0.975 | 0.557 | +0.418 | 0.984 | 0.034 | 0.923 | 0.983 | 0.149 |
| `fci_ts` | 0.972 | 0.562 | +0.410 | 0.967 | 0.022 | **0.942** | 1.000 | 0.203 |
| `varlingam` | 0.971 | 0.573 | +0.398 | 0.951 | 0.073 | 0.841 | 0.862 | 0.040 |
| **`cafe_cd`** | 0.921 | 0.554 | +0.367 | 0.984 | 0.564 | 0.426 | 0.900 | 0.419 |
| `pc` | 0.609 | 0.486 | +0.123 | 0.213 | 0.034 | 0.474 | — | 0.007 |
| `fci` | 0.603 | 0.555 | +0.048 | 0.016 | 0.006 | 0.072 | — | 0.002 |
| `directlingam` | 0.554 | 0.531 | +0.022 | 0.213 | 0.106 | 0.344 | — | 0.016 |
| `dagma_linear` | 0.513 | 0.502 | +0.011 | 0.049 | 0.022 | 0.169 | — | 0.642 |

**All 13 beat their own transpose**, so the adapter's orientation is right for
every method including CAFE-CD. Replicated on an independent draw (B=24,
seed=7) for the five fastest lag-aware methods — margins +0.385 to +0.465, all
`OK`, CAFE-CD 0.925 vs 0.496 transposed — so the margin is not a 12-dataset
fluke. The bottom four are the `CONTEMP_ONLY` set
(`pc`, `fci`, `directlingam`, `dagma_linear`): the round-trip truth is purely
lagged, so near-chance is the *correct* result for them and it is evidence that
the harness is not leaking lagged information into a contemporaneous method.
CAFE-CD's FPR of 0.564 is the honest reading of its `.adjacency()` support at
default settings — a high-recall, low-precision decision rule; its **ranking**
(AUC 0.921, TLR 0.900) is far better than its **support**, which is exactly the
score-vs-adjacency split `CausalGraph.scores()` documents.

### 9c. The artefacts

One bz2-compressed JSON per (experiment, method) in
`data/causal/causeme/submissions/` — **52 files, 13 methods × the 4 experiments
above, all built and all validated (0 problems)** — by `causeme_build_submission`,
which writes to disk and never opens a socket. The run is checkpointed per
(experiment, method) in the cache, so `--run` resumes and only `--force`
rebuilds. `--validate` checks each artefact against the platform's stated
contract:

* required keys `method_sha`, `parameter_values`, `model`, `experiment`, `scores`;
* one entry per dataset, ordered by `sorted(zipfile.namelist())`;
* each entry a **row-major (C-order)** flatten of an `(N, N)` matrix;
* scores non-negative and finite, **self-links zeroed**;
* `pvalues` in [0, 1], `lags` non-negative integers;
* a bz2 → JSON → latin1 round-trip that reconstructs the dict exactly.

`parameter_values` names the statistic each method actually ranked by rather
than pretending they all rank by `|coef|` — it is the only provenance a
submission carries.

**Runtime**, which CauseMe accepts as an optional `runtimes` field and uses for
inter-method comparison, is the one number that *is* locally measurable and
locally meaningful. `linear-VAR_N-5_T-300`, 200 datasets, N=5, T=300,
`max_lag=5`, one process, BLAS pinned to 2 threads (`nonzero` is the fraction of
off-diagonal entries the method gives a non-zero score, i.e. how dense its
ranking is):

| method | s/dataset | total s | fails | nonzero | artefact |
|---|---|---|---|---|---|
| `var_ols` | 0.000 | 0.1 | 0 | 0.800 | 37 kB |
| `pc` | 0.004 | 0.8 | 0 | 0.193 | 3 kB |
| `fci` | 0.005 | 0.9 | 0 | 0.252 | 3 kB |
| `granger_pairwise` | 0.008 | 1.5 | 0 | 0.800 | 37 kB |
| `directlingam` | 0.020 | 4.0 | 0 | 0.143 | 9 kB |
| `varlingam` | 0.054 | 10.8 | 0 | 0.152 | 11 kB |
| `pc_ts` | 0.146 | 29.1 | 0 | 0.130 | 5 kB |
| `fci_ts` | 0.152 | 30.4 | 0 | 0.131 | 5 kB |
| `dynotears` | 0.279 | 55.8 | 0 | 0.590 | 30 kB |
| **`cafe_cd`** | **0.352** | 70.4 | 0 | 0.800 | 12 kB |
| `dagma_linear` | 0.776 | 155.1 | 0 | 0.114 | 8 kB |
| `pcmciplus` | 1.283 | 256.6 | 0 | 0.800 | 37 kB |
| `pcmci` | 1.298 | 259.6 | 0 | 0.800 | 37 kB |

No method failed on any of the 200 datasets. CAFE-CD is **3.6× faster than
PCMCI+** on the same CPU with a dense (fully ranked) submission — which is the
comparison the CPU/causal niche actually turns on. Note the two tigramite
methods pay a subprocess spawn per dataset (GPL isolation, §2), so their
absolute numbers are a per-dataset-invocation cost, not a pure algorithm cost.

The gap widens with size. On the only **real-data** experiment,
`river-runoff_N-12_T-4600` (N=12, T=4600, one dataset), where the spawn cost is
amortised over a much larger problem:

| method | s | | method | s | | method | s |
|---|---|---|---|---|---|---|---|
| `var_ols` | 0.003 | | `dagma_linear` | 1.51 | | `pc_ts` | 10.66 |
| `granger_pairwise` | 0.15 | | `varlingam` | 1.85 | | `dynotears` | 11.10 |
| `pc` | 0.19 | | **`cafe_cd`** | **2.42** | | `fci_ts` | 12.41 |
| `fci` | 0.25 | | `pcmci` | 8.31 | | **`pcmciplus`** | **49.55** |
| `directlingam` | 0.35 | | | | | | |

**CAFE-CD 2.4 s vs PCMCI+ 49.6 s — 20× — on the same CPU**, with the same dense
0.917 non-zero fraction. No method failed. That is the number worth carrying
into the paper's runtime claim, because it is a real 12-variable series rather
than a 5-variable toy.

**Method agreement** (`--agreement`), pooled Spearman between every pair of
methods' off-diagonal scores. On `linear-VAR_N-5_T-300` (4 000 pooled links):

| method | mean ρ vs the other 12 | | method | mean ρ |
|---|---|---|---|---|
| `dynotears` | +0.459 | | `varlingam` | +0.385 |
| `pcmci` | +0.444 | | `fci` | +0.288 |
| `granger_pairwise` | +0.437 | | `pc` | +0.263 |
| `var_ols` | +0.426 | | `directlingam` | +0.196 |
| `fci_ts` | +0.420 | | **`cafe_cd`** | **+0.152** |
| `pc_ts` | +0.420 | | `dagma_linear` | +0.138 |
| `pcmciplus` | +0.405 | | | |

Nothing is negative, so no submission is inverted — the failure mode this is
here to catch. The tightest pair is `fci_ts | pc_ts` at ρ = 0.995 (same
algorithm family, same conditioning). CAFE-CD's ranking is the most distinctive
of the lag-aware methods (also +0.134 on `nonlinear-VAR`). **Without ground
truth that is not evidence either way**: on the round-trip, where truth *is*
known, CAFE-CD scores AUC 0.921. It is simply the number to look at first if a
real CauseMe submission ever comes back poor.

The check earned its keep immediately: on
`logistic-deterministic_N-5_T-300_weak` — chaotic logistic maps, weak coupling —
**our own DYNOTEARS returns an all-zero matrix on all 200 datasets**, a constant
submission whose AUC would be exactly 0.5 by construction, and `--agreement`
flags it (`constant_submissions: ['dynotears']`, ρ undefined). That is the same
degeneracy class as the shipped CausalDynamics wrappers in §6c(e), now in our
own suite; it is reported rather than averaged away. `pc`/`fci`/`pc_ts`/`fci_ts`
also hit singular correlation matrices on 2–7 of those 200 datasets, which the
harness scores as all-zero predictions and counts in `fails`.

**One honest limitation.** `BaselineResult` carries a binary *support*, not
calibrated p-values, so the `pvalues` channel we submit is `0` where the method
selected the edge and `1` where it did not. CauseMe's **AUC** — the headline
metric — reads the `scores` matrix and is unaffected. Its **FPR / TPR /
F(β=0.5)** columns are computed from `pvalues ≤ 0.05`, so for these artefacts
they would measure the *support rule*, not calibration, and an FPR near 0.05
would be a coincidence rather than evidence of a well-calibrated test.
`parameter_values` says so explicitly. Methods that do have real p-values
(`pcmci`, `pcmciplus`, `var_ols`) would need `BaselineResult` extended before
that column means anything.

### 9d. What a submission would still require

Nothing in this repo has been uploaded, and no code path can upload. To submit:

1. an account on `causeme.uv.es/signup/`;
2. the method registered under **My Results**, ideally with a code zip whose
   module exposes `my_method(data, **params) -> (val_matrix, p_matrix, lag_matrix)`
   — that is what earns the *Validated* badge and lets the platform time the
   method itself;
3. the `method_sha` registration returns, substituted for the placeholder
   (`--method-sha <sha>`; the artefacts currently carry the deliberately
   invalid `UNREGISTERED-PLACEHOLDER-NOT-A-REAL-METHOD-SHA`, so an accidental
   upload cannot be attributed to a real method);
4. a manual upload of the `.json.bz2` on **My Results** — rate limited per user.

Only then does a CauseMe number exist. Until it does, this file's claim is
"a valid, orientation-verified artefact for 13 methods", not "a CauseMe result".
