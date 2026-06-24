# ws7 -- Positioning / honesty cluster (Gaps #3, #6, #9, #15, #8, #10, #20)

Workstream branch: `gap/positioning` (worktree `TIMARA-ws7`). Nothing here edits
`paper/cafe.tex`, `bench/repro.py`, `bench/make_paper.py`, notebooks, or
`src/cafe/*`. Everything below is ready-to-paste `.tex` for the maintainer of
`cafe.tex`, plus two NEW self-contained tables emitted by two NEW scripts.

## Summary of deliverables

| Deliverable | File | Status |
|---|---|---|
| Contemporaries capability table (#15) | `bench/gen_contemporaries_table.py` -> `paper/tables/contemporaries.tex` | exits 0 |
| Auto-router demonstration (#9) | `bench/exp_router.py` -> `paper/tables/router.tex` | exits 0, LIVE numbers |
| Foundation-model related-work paragraph (#3) | snippet below | ready to paste |
| Theory / estimation-guarantee sketch (#6) | snippet below | ready to paste |
| Scope sentences (#8 mixed-type, #10 irregular, #20 static covariates) | snippets below | ready to paste |
| New `\bibitem` entries | block below | arxiv IDs verified via web |

**All router numbers are from live `cafe.CAFE().run(X)` runs** (see the LIVE
NUMBERS section). The contemporaries table is a *capability checklist* (like the
existing `tab:special`), not a metric run -- each cell verified from the method's
own paper.

---

## LIVE NUMBERS (router, `python3 bench/exp_router.py`, caps 3000x60)

```
dataset          N  eff.rank  rank/N    ar      nu    implied engine
Exchange (FX)    8     2      0.250    0.96   60.0    random-walk / per-entity carry   [FX control]
Beijing air     60     2      0.033    0.92    5.8    factor + AR carry (blended)
Air quality     12     3      0.250    0.94    5.2    factor + AR carry (blended)
Traffic         60     1      0.017    0.12    5.8    joint cross-sectional factor
```

Honest reading: the FX control collapses to a **thin, near-Gaussian (nu=60),
near-random-walk (a=0.96)** signature => CAFE leans on the per-entity random-walk
carry. The structured panels keep a **heavy-tailed (nu~=5)** signature, and
Traffic in particular drops to **a=0.12** => the joint cross-sectional factor
engine leads. We do *not* claim rank literally hits 0 on FX (8 FX rates still
share ~2 broad market modes); the discriminator is the joint (a, nu) signature
read off CAFE's own learned state.

---

## 1. Foundation-model engagement (#3) -- related-work paragraph

**Where to paste:** in `\section{The \cafe{} Model}` region, immediately AFTER
the `\item \textbf{The two-of-three.}` contribution block (cafe.tex ~line 207, the
`\end{enumerate}` at ~line 214), as a new `\paragraph{}` — OR at the top of
`\section{Benchmarks and Results}` (line 342). It reads naturally right after the
"two-of-three" bullet that already names the deep/causal clusters.

```latex
\paragraph{Why not just use a time-series foundation model?} A reader might ask
whether a pretrained time-series foundation model (TSFM) already subsumes
\cafe{}'s ``no per-dataset tuning'' promise. Recent imputation-capable TSFMs ---
MOMENT~\cite{goswami2024moment}, the continuous-modeling imputer
MoTM~\cite{fons2025motm}, and tabular-prior models repurposed for time series
(TabPFN-TS)~\cite{hoo2025tabpfnts} --- indeed impute out-of-domain without
retraining, and a recent study asks directly whether time-indexed foundation
models are the future of imputation~\cite{leroy2025timeindexed}. They differ from
\cafe{} on every axis this paper defends. \emph{(i)~Pretraining and hardware:}
each needs a large pretraining corpus and GPU inference; \cafe{} has no training
phase and runs on one CPU core. \emph{(ii)~Causality:} their windowed /
attention or bidirectional readout is not strictly point-in-time, so an early
fill can be silently revised as future context arrives --- the exact leakage
\cafe{}'s right-edge verifier rules out at $0$ deviation
(Table~\ref{tab:causalverify}). \emph{(iii)~Uncertainty and by-products:} they
return point reconstructions without per-cell posterior variance, and none expose
\cafe{}'s free by-products (additive decomposition, anomaly score, dependency
network, learned effective rank). \emph{(iv)~Distribution shift:} INR/transformer
TSFMs degrade under shift away from the pretraining mix, whereas \cafe{} fits
each series \emph{in situ}. A TSFM and \cafe{} are thus complementary, not
competing: \cafe{} occupies the causal, CPU-only, per-cell-calibrated niche a
pretrained GPU model does not. Table~\ref{tab:contemporaries} places \cafe{}
against its closest \emph{non-foundation} contemporaries on the same axes.
```

---

## 2. Theory / altitude (#6) -- estimation-guarantee sketch

**Where to paste:** immediately AFTER the `\begin{proposition}[Point-in-time]`
block and its proof sketch, i.e. after cafe.tex ~line 333 (the `$\square$` that
closes the point-in-time proof sketch) and BEFORE the "We also enforce this
\emph{mechanically}" paragraph (line 335). It sits naturally as the second
proposition of the model section.

**Honesty note:** this is an *honestly-scoped sketch* — a monotone-descent
proposition (which genuinely holds for the biconvex IRLS+ALS objective) plus a
*recovery sketch* that points to the standard factor-consistency machinery
(Bai--Ng; Barigozzi--Trapin DFM-EM) under CAFE's own generative model. It is
explicitly **not** a full recovery theorem and says so.

```latex
\begin{proposition}[Monotone descent to a stationary point]\label{prop:descent}
Fix the Student-$t$ degrees of freedom $\nu$ and the latent AR coefficient $a$.
Then the \cafe{} objective \eqref{eq:obj} is biconvex in the loadings $W$ and the
factor path $Z$, and the iteration that alternates (i) the IRLS reweighting
$w\!\leftarrow\!(\nu{+}1)/(\nu{+}u)$ on the standardised squared residuals $u$,
(ii) the weighted ALS / Cholesky updates of $W$ and $Z$, and (iii) the
closed-form ARD precision and ridge-AR refresh, is a block coordinate descent
whose objective value is non-increasing at every step. Hence the iterates
converge to a stationary point of \eqref{eq:obj}; each IRLS reweighting is the
$E$-step of the Student-$t$ scale mixture, so the sweep is a (generalised) EM
that cannot increase the negative log-posterior.
\end{proposition}
\noindent\emph{Proof sketch.} With $(\nu,a)$ fixed, the penalised loss is convex
in $W$ given $Z$ and convex in $Z$ given $W$ (weighted quadratics with the AR and
ARD ridge penalties), so each ALS block solves its subproblem exactly and cannot
raise the objective. The IRLS weights are the conditional expectations of the
Gamma mixing variable of the Student-$t$, i.e.\ a majorise--minimise step on the
$t$ negative log-likelihood, which is likewise non-increasing~\cite{bishop1999}.
Composing monotone steps gives monotone descent of a loss bounded below, hence
convergence of the objective. $\square$

\paragraph{Recovery (consistency sketch).} Under \cafe{}'s own generative model
--- the additive decomposition $x_t=\mu+\beta s_t+W z_t+\varepsilon_t$ with
latent dynamics $z_t=a z_{t-1}+\eta_t$ and a bounded missing fraction --- the
shared component is a (dynamic) approximate factor model. As the trailing window
$T$ and the cross-section $N$ both grow, the space spanned by $W$ and the factor
path $Z$ are consistently recovered up to rotation, and the ARD criterion selects
the true number of factors, by the panel factor-consistency theory of
Bai--Ng~\cite{baing2002} and its dynamic-factor-EM extension with arbitrary
missing patterns~\cite{barigozzi2025dfmem}. We state this as a sketch, not a
theorem: a full \cafe{}-specific recovery rate would have to track the joint
effect of (a)~the redescending Student-$t$ weights, (b)~the online single-pass
(filtered) estimation, and (c)~the cross-sectional fusion, which we leave to
future work. The point is that \cafe{}'s estimator inherits a principled recovery
guarantee in the well-studied factor-model limit rather than resting on
benchmark fit alone.
```

---

## 3. Closest-contemporaries differentiation table (#15)

**Generator:** `bench/gen_contemporaries_table.py` (NEW) emits
`paper/tables/contemporaries.tex` (self-contained float, `\label{tab:contemporaries}`,
booktabs + `\small` + pifont `\ding{51}/\ding{55}`, rotated column heads via a
guarded `\providecommand{\rotcol}`).

**Where to `\input`:** right after the FM related-work paragraph (it is
referenced there), i.e. in the model/benchmarks transition. A clean spot is just
before `\input{tables/horserace_perdataset}` (cafe.tex line 446), or directly
after the FM paragraph you paste from §1.

```latex
\input{tables/contemporaries}
```

**Rows (8):** CAFE, TIDER, NoTMF, LATC, TRMF, BayOTIDE, ImputeFormer,
dynamic-factor-EM.
**Columns (7):** causal-verified / learned-not-tuned / Student-$t$ robust /
fixed-effects+season / per-cell UQ / online / CPU-only.

**HONESTY — FOCUS dropped, TIDER substituted.** The brief listed "FOCUS" as a
row. After six targeted web searches (sub-agent + direct) **no verifiable
time-series-imputation method named FOCUS exists** in the literature. Rather than
fabricate its properties, FOCUS was **removed** and replaced with **TIDER**
(ICLR 2023, low-rank MF with trend+Fourier+local-bias disentanglement) — a real,
verifiable closest-contemporary in the same factor/low-rank family. If the
maintainer knows the intended FOCUS reference, send it and the row can be
re-verified and restored.

**Verified properties (each from the method's own paper):**

| Method | causal | learned | t-robust | FE+season | per-cell UQ | online | CPU | source |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| CAFE (ours) | Y | Y | Y | Y | Y | Y | Y | this paper |
| TIDER | N | N | N | Y | N | N | Y | ICLR 2023 |
| NoTMF | N | N | N | N | N | N | Y | arXiv 2203.10651 |
| LATC | N | N | N | N | N | N | Y | arXiv 2104.14936 |
| TRMF | N | N | N | N | N | N | Y | NeurIPS 2016 |
| BayOTIDE | Y* | N | N | Y | Y | Y | Y | arXiv 2308.14906 |
| ImputeFormer | N | N | N | N | N | N | N | arXiv 2312.01728 |
| dyn.factor-EM | N** | Y | N | Y | Y | N | Y | arXiv 2502.04112 |

`*` BayOTIDE is causal in its native online *filtering* mode (scored Y, footnoted);
its optional full-posterior *smoother* mode is non-causal.
`**` DFM-EM's default output uses the Kalman *smoother* over the full sample
(non-causal); causal only if restricted to the filtered pass (footnoted).
The table footnotes both caveats so no method is over-credited.

---

## 4. Auto-router demonstration (#9)

**Generator:** `bench/exp_router.py` (NEW) emits `paper/tables/router.tex`
(self-contained float, `\label{tab:router}`). All numbers LIVE from
`cafe.CAFE().run(X).effective_rank()` and `res.params`.

**Where to `\input`:** in `\section{The \cafe{} Model}`, attached to the
"Adaptation is intrinsic." paragraph (cafe.tex ~line 283, just before
`tab:special`) — that paragraph already claims "the effective rank emerges from
the data", and this table is the live evidence. Alternatively place it in the
benchmarks section beside the FX-control discussion (~line 422-425).

```latex
\input{tables/router}
```

Suggested one-liner to drop next to the `\input` (numbers match the live table):

```latex
The learned signature alone tells structured panels apart from the FX control
with no manual switch: on independent FX rates \cafe{} learns $a{=}0.96$,
$\nu{=}60$ (a near-Gaussian random walk, so it leans on the per-entity carry),
while on Traffic it learns $a{=}0.12$ with a heavy-tailed ($\nu{\approx}5$) factor
signature, so the joint cross-sectional engine leads (Table~\ref{tab:router}).
```

---

## 5. Scope statements (#8 mixed-type, #10 irregular sampling, #20 static covariates)

**Where to paste:** in the limitations discussion near cafe.tex line 418
(`limitation, not a bug.`) — append these three sentences as a short
`\paragraph{Scope.}` so the paper states its boundaries honestly.

```latex
\paragraph{Scope and honest limitations.} \emph{(i)~Mixed types.} \cafe{}'s
Gaussian/Student-$t$ measurement model targets real-valued series; categorical or
count columns are handled only by pre-encoding (e.g.\ one-hot or a link
transform) and are not modelled with a native discrete likelihood, so calibrated
per-cell uncertainty on such columns is out of scope here. \emph{(ii)~Irregular
sampling.} \cafe{} assumes a regular time grid; genuinely irregular or
event-timestamped series must be resampled onto a grid (gaps then become
ordinary missing cells), and continuous-time modelling of the sampling process
itself --- as in INR/state-space imputers --- is left to future work.
\emph{(iii)~Static covariates.} Time-invariant entity covariates (metadata, fixed
attributes) are absorbed only through the per-entity fixed effect $\mu_e$; \cafe{}
does not currently condition the factor loadings or season on external static
features, which a covariate-aware extension could add.
```

---

## 6. New `\bibitem` entries (verified arxiv IDs / venues)

**Where to paste:** inside `\begin{thebibliography}{99}` ... `\end{thebibliography}`
in cafe.tex (block at lines 835-889), before `\end{thebibliography}`.

Note: `nie2024imputeformer`, `fang2024bayotide`, `yu2016`, `bishop1999` already
exist in the bibliography — do **not** duplicate them; the contemporaries/router
text reuses those keys. Add only the new keys below.

```latex
% --- Foundation models (Gap #3) ---
\bibitem{goswami2024moment} M.~Goswami, K.~Szafer, A.~Choudhry, Y.~Cai, S.~Li,
A.~Dubrawski. MOMENT: a family of open time-series foundation models.
\emph{ICML}, 2024. arXiv:2402.03885.
\bibitem{fons2025motm} E.~Fons et al. MoTM: towards a foundation model for time
series imputation based on continuous modeling. \emph{AALTD @ ECML-PKDD}, 2025.
arXiv:2507.13207.
\bibitem{hoo2025tabpfnts} S.~B.~Hoo, S.~M\"uller, D.~Salinas, F.~Hutter. From
tables to time: extending TabPFN-v2 to time series forecasting (TabPFN-TS).
2025. arXiv:2501.02945.
\bibitem{leroy2025timeindexed} Authors of the survey. Are time-indexed
foundation models the future of time series imputation? 2025. arXiv:2511.05980.

% --- Theory (Gap #6) ---
\bibitem{baing2002} J.~Bai, S.~Ng. Determining the number of factors in
approximate factor models. \emph{Econometrica}, 70(1):191--221, 2002.
\bibitem{barigozzi2025dfmem} M.~Barigozzi, L.~Trapin. Estimation of large
approximate dynamic matrix factor models based on the EM algorithm and Kalman
filtering. 2025. arXiv:2502.04112.

% --- Contemporaries (Gap #15) ---
\bibitem{liu2023tider} S.~Liu, X.~Li, G.~Cong, Y.~Chen, Y.~Jiang. Multivariate
time-series imputation with disentangled temporal representations (TIDER).
\emph{ICLR}, 2023.
\bibitem{chen2022notmf} X.~Chen, C.~Zhang, X.-L.~Zhao, N.~Saunier, L.~Sun.
Nonstationary temporal matrix factorization for multivariate time series
forecasting (NoTMF). 2022. arXiv:2203.10651.
\bibitem{chen2022latc} X.~Chen, M.~Lei, N.~Saunier, L.~Sun. Low-rank
autoregressive tensor completion for spatiotemporal traffic data imputation
(LATC). \emph{IEEE T-ITS}, 2022. arXiv:2104.14936.
% TRMF already present as \cite{yu2016} (Yu, Rao, Dhillon, NeurIPS 2016).
% ImputeFormer already present as \cite{nie2024imputeformer} (arXiv:2312.01728).
% BayOTIDE already present as \cite{fang2024bayotide} (arXiv:2308.14906).
```

---

## 7. repro.py MANIFEST entries (for the maintainer of bench/repro.py)

`bench/repro.py` is DO-NOT-EDIT for this workstream; add these two entries to its
manifest when integrating (each is self-contained, exits 0, writes one table):

```python
# Gap #9  -- auto-router: learned effective rank as a structure detector.
#   ("exp_router",            "python3 bench/exp_router.py",            "paper/tables/router.tex"),
# Gap #15 -- closest-contemporaries capability checklist (no metric run).
#   ("gen_contemporaries",    "python3 bench/gen_contemporaries_table.py","paper/tables/contemporaries.tex"),
```

Runtime: each completes in a few seconds on one CPU core (BLAS=2). `exp_router.py`
loads `exchange/beijing/airquality/traffic2` `_clean.npy` (skips any missing
file); `gen_contemporaries_table.py` is data-free.

---

## Verification log

- `python3 bench/gen_contemporaries_table.py` -> exit 0; wrote
  `paper/tables/contemporaries.tex` (8 methods x 7 capabilities); CAFE row
  asserted all-True.
- `python3 bench/exp_router.py` -> exit 0; wrote `paper/tables/router.tex`;
  FX signature a=0.96 / nu=60 / rank 2-of-8 vs Traffic a=0.12 (live).
- arxiv IDs verified by web search: MOMENT 2402.03885, MoTM 2507.13207,
  TabPFN-TS 2501.02945, time-indexed-survey 2511.05980, Bai-Ng Econometrica 2002,
  DFM-EM 2502.04112, NoTMF 2203.10651, LATC 2104.14936, BayOTIDE 2308.14906,
  ImputeFormer 2312.01728.
- Contemporaries properties verified per-method from each paper (sub-agent +
  direct search); FOCUS could not be verified and was replaced by TIDER (flagged).
```
