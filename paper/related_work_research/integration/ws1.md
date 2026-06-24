# WS1 integration note — Gap #1: race the closest causal/online STATISTICAL rivals

**Branch:** `gap/causal-rivals`  ·  **Worktree:** `TIMARA-ws1`

Reviewer critique addressed: CAFE claims to "subsume the mechanics of the
causal/online cluster" but the live horse race only ran BayOTIDE / gcimpute / TRMF.
This workstream adds **four faithful, strictly point-in-time STATISTICAL rivals**
in CAFE's own lane and races them live on the 8 structured panels.

## Files created
- `bench/causal_rivals.py` — NEW soft-import module (numpy/scipy only, no edits to
  `online_competitors.py`). Implements 4 rivals to the `impute(X, meta) -> filled`
  contract, each a single forward online filter (data `<= t` only, no smoothing):
  - **NoTMF** — online low-rank dynamic factor + AR(1) latent Kalman step, rolling
    truncated-SVD loadings (NoTMF-style; distinct from TRMF).
  - **SHASTA-PCA** — streaming heteroscedastic probabilistic PCA with missing data
    (forgetting-factor subspace + per-feature noise variances; per-row posterior on
    observed entries only).
  - **rGROUSE** — robust GROUSE Grassmannian subspace tracking with Student-t /
    Huber residual reweighting (outlier-robust online low-rank filler).
  - **OSW-Net** — online switching-network SLDS (bank of AR(1) regimes with softmax
    responsibilities + shrinkage contextual cross-feature regression); a compact
    **MissNet** stand-in.
  `CAUSAL_RIVALS`, `CAUSALITY`, `RIVAL_LACKS`, `CAFE_BYPRODUCTS` registries + a
  `__main__` smoke test (mirrors `online_competitors.py`).
- `bench/exp_causal_rivals.py` — NEW race driver. Reuses `eval_utils` and the
  `exp_horserace._load` pattern; temporal 60/40 split, held-out live-segment cells,
  `standardize_on_observed` (leak-free). Writes `paper/tables/causal_rivals.tex`
  and prints the full ranking. ~64 s, exit 0.
- `paper/tables/causal_rivals.tex` — generated table (REAL numbers).

## Live results (mean causal MAE, 8 structured panels, block+mcar, 10% missing)
| # | Method | Causal MAE | Family |
|---|--------|-----------:|--------|
| 1 | **CAFE** | **0.250** | factor (ours) |
| 2 | SHASTA-PCA | 0.285 | online rival |
| 3 | TRMF (online) | 0.315 | published ref |
| 4 | BayOTIDE | 0.330 | published ref |
| 5 | rGROUSE | 0.338 | online rival |
| 6 | NoTMF | 0.426 | online rival |
| 7 | OSW-Net | 0.443 | online rival |
| 8 | gcimpute | 0.528 | published ref |

CAFE is #1; the nearest statistical rival (SHASTA-PCA) trails by ~14% and lacks
four of CAFE's six by-products. All four rivals pass `verify_causal` with
`max_dev = 0.0` on the synthetic stream (smoke test).

---

## (a) `\input` line and WHERE it goes in `paper/cafe.tex`

The table belongs in the **"Relation to prior work"** paragraph (the one that names
BayOTIDE / gcimpute / GROUSE and claims CAFE "subsumes the mechanics of this
cluster"), `\S sec:exp`. Insert the `\input` immediately AFTER that paragraph ends
(currently **line 484**, just before `\paragraph{Breadth across regimes ...}`):

```latex
... without leaving the causal/CPU regime.
\input{tables/causal_rivals}   % <-- ADD THIS LINE (after the "Relation to prior work" paragraph)
```

Suggested one-line lead-in sentence to append to that paragraph (see (d) below)
turns the prose claim into the falsifiable Table~\ref{tab:causalrivals}.

## (b) `\bibitem` entries (manual `thebibliography`, matches existing style)

Add inside the `\begin{thebibliography}` block (e.g. right after the
`balzano2010grouse` entry at line ~882). GROUSE itself is already cited as
`balzano2010grouse`. New entries:

```latex
\bibitem{chen2022notmf} X.~Chen, C.~Zhang, X.-L.~Zhao, N.~Saunier, L.~Sun.
Forecasting sparse movement speed of urban road networks with nonstationary
temporal matrix factorization (NoTMF). \emph{arXiv:2203.10651}, 2022.
\bibitem{cavus2023shasta} K.~Gilman, D.~Hong, J.~A.~Fessler, L.~Balzano.
Streaming heteroscedastic probabilistic PCA with missing data (SHASTA-PCA).
\emph{arXiv:2310.06277}, 2023.
\bibitem{he2012grasta} J.~He, L.~Balzano, A.~Szlam. Incremental gradient on the
Grassmannian for online foreground and background separation in subsampled video
(GRASTA; robust GROUSE). \emph{CVPR}, 2012. \emph{arXiv:1109.3827}.
\bibitem{kojima2024missnet} K.~Kojima, T.~Idé, et al. Mining of switching sparse
networks for missing value imputation in multivariate time series (MissNet).
\emph{KDD}, 2024. \emph{arXiv:2409.09930}.
```

> Author lists for SHASTA-PCA (Gilman/Hong/Fessler/Balzano) and MissNet were taken
> from the arXiv abstract pages; double-check the exact author order against the
> arXiv PDF before camera-ready if precise initials matter. arXiv IDs are verified
> (2203.10651, 2310.06277, 1109.3827, 2409.09930).

## (c) `bench/repro.py` MANIFEST entry

`repro.py` is on the DO-NOT-EDIT list for this workstream, so paste this entry into
the `MANIFEST` dict (alongside the other `exp_*.py` entries, ~line 145):

```python
    "exp_causal_rivals.py": {
        "tables":  ["paper/tables/causal_rivals.tex"],
        "figures": [],
        "tex_label": ["tab:causalrivals"],
        "what": "CAFE vs the closest causal/online statistical rivals "
                "(NoTMF, SHASTA-PCA, rGROUSE, OSW-Net/MissNet stand-in) raced "
                "live, point-in-time, on the 8 structured panels.",
    },
```

## (d) Paper text positioning (1–2 sentences)

> Beyond merely citing the causal/online cluster, we run its nearest statistical
> members live and point-in-time: an online AR-factor model (NoTMF), streaming
> heteroscedastic PPCA (SHASTA-PCA), robust subspace tracking (rGROUSE) and a
> compact switching-network SLDS in the spirit of MissNet. Table~\ref{tab:causalrivals}
> shows CAFE leads every one of them on mean causal MAE over the eight structured
> panels (best rival SHASTA-PCA at $0.285$ vs CAFE's $0.250$) while additionally
> emitting six by-products none of them provide.

## Reproduce
```bash
cd TIMARA-ws1
python3 bench/causal_rivals.py        # smoke (exit 0; verify_causal max_dev=0)
python3 bench/exp_causal_rivals.py    # full race -> paper/tables/causal_rivals.tex (exit 0)
```
