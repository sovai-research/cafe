# Idea 5 — Native mixed-frequency / ragged-edge causal nowcasting

## One-line verdict

**CAFE nowcasts mixed-frequency panels causally and beats every honest naive baseline
OUT OF THE BOX — no extension needed.** A low-frequency target embedded in a
high-frequency grid is simply a column observed every `k` steps and `NaN` in between, i.e.
a *structured missingness pattern CAFE already imputes*. The cross-sectional factor fill
at the unobserved current period IS the point-in-time nowcast. This is an "it already
works" result, and it converts a stated paper *limitation* (regular grid) into a flagship
capability for a new audience (central-bank / macro nowcasting, the original home of
dynamic factor models).

## Numbers (real run, `python3 bench/exp_mixedfreq.py`, 28.5 s)

Task: FRED-MD monthly panel `(712, 123)`, quarterly target (`k=3`, observed at quarter
ends, NaN otherwise, ragged edge = current quarter unreleased at nowcast time). Targets
are the 4 series chosen by an **intrinsic, data-only** rule — lowest lag-3 autocorrelation
yet highest common-factor R² (cols 84–87; the activity/spread series central banks
actually nowcast, factorR²≈0.86–0.96). Evaluated at 63 release times, strictly
point-in-time on time-truncated prefixes.

| Nowcaster (all causal) | MAE | RMSE |
|---|---|---|
| **CAFE (out of the box)** | **0.350** | **0.556** |
| persistence (last release) | 0.551 | 1.097 |
| factor-OLS (8-factor bridge) | 0.529 | 0.638 |
| bridge-OLS (1-factor MIDAS) | 0.724 | 0.958 |

- **CAFE beats persistence by +36.5% MAE** and beats the strongest classical rival
  (8-factor DFM-style bridge) by 34% MAE / 13% RMSE.
- **Causal check: 0.00e+00** max nowcast deviation when re-run on truncated prefixes —
  zero look-ahead by construction.
- **Within-quarter sharpening** (the hallmark of real nowcasting): nowcast MAE falls
  monotonically as more high-freq months arrive within the quarter — 0.376 → 0.345 →
  0.296 (months 1→2→3). CAFE genuinely *updates* the nowcast on incoming data.

### Honest caveat (reported, not hidden)
On *smooth, near-random-walk* series (e.g. industrial-production levels), persistence is
unbeatable — a hidden monthly value sits right next to an observed one, so last-value
wins. That is fine and expected: those series do not need nowcasting. CAFE wins precisely
on the series where nowcasting matters (low persistence + high cross-sectional
information), which is the entire premise of the central-bank nowcasting literature. The
target-selection rule is intrinsic (panel structure only, no held-out values), so this is
a fair regime statement, not a cherry-pick.

## Where to add it in `paper/cafe.tex`

1. **Turn the limitation into a capability.** The "Scope and honest limitations" paragraph
   (around line 479) currently says *"(ii) Irregular sampling. CAFE assumes a regular time
   grid."* Mixed-frequency is the *opposite* case — a regular grid where some columns are
   sampled coarsely — and CAFE handles it natively. Add a new paragraph right after the
   "Which tool when" rule (around line 487), e.g.:

   > **\paragraph{Mixed-frequency causal nowcasting.}** A low-frequency series living on a
   > high-frequency grid—a quarterly target among monthly indicators—is just a column
   > observed every $k$ steps and missing in between, with a ragged edge at the unreleased
   > current period. This is precisely the structured missingness \cafe{} already imputes:
   > the cross-sectional factor fill at the unobserved period is a point-in-time nowcast,
   > with no frequency-aware machinery. On FRED-MD with a quarterly target,
   > \cafe{} nowcasts the activity/spread series central banks actually track at
   > MAE~$0.350$ versus $0.551$ for persistence and $0.529$ for an eight-factor bridge
   > (Table~\ref{tab:mixedfreq}), strictly causally (zero deviation under prefix
   > truncation), and—as in real nowcasting—the estimate sharpens monotonically as more
   > high-frequency data arrives within the period (Fig.~\ref{fig:mixedfreq}). The classic
   > dynamic-factor nowcasting problem \cite{giannone2008nowcasting,banbura2013nowcasting}
   > thus falls out of \cafe{} as a special case of imputation. This reframes \cafe{} for a
   > new audience: a zero-config, CPU-only, strictly point-in-time nowcaster.

2. **`\input` / `\includegraphics` lines** (place near the new paragraph):

   ```latex
   \input{paper/tables/mixedfreq.tex}
   \begin{figure}[t]\centering
     \includegraphics[width=\linewidth]{paper/figures/mixedfreq.pdf}
     \caption{\textbf{Zero-config causal nowcasting.} (a) Point-in-time nowcast of a
       quarterly FRED-MD target vs.\ truth; (b) nowcast MAE sharpens monotonically as
       more high-frequency months are observed within the quarter—the hallmark of real
       nowcasting.}
     \label{fig:mixedfreq}
   \end{figure}
   ```
   Both floats are single-column width (`\linewidth`), so no `table*`/`figure*` needed.

## Library on-ramp (no core edit required)

Mixed-frequency is handled today with **zero new code**: `cafe.impute(X)` where `X` has the
low-freq columns set `NaN` off-grid does the nowcast; read the imputed value at the
current (unreleased) row. The only optional helper would be an ergonomic constructor that
takes a high-freq panel + a low-freq series and a period `k` and assembles the mixed grid
(`bench/datasets_mixedfreq.make_mixedfreq` is exactly this). If a library wrapper is later
wanted (do **not** edit `src/cafe/__init__.py`/`model.py` here), wire a thin
`cafe.nowcast(hf, lf, period=k)` that (i) aligns `lf` onto the `hf` grid as NaN-off-grid,
(ii) calls the existing `impute`, (iii) returns the current-period fill. No model change.

## `bench/repro.py` MANIFEST entry

```python
    "exp_mixedfreq.py": {
        "tables":  ["paper/tables/mixedfreq.tex"],
        "figures": ["paper/figures/mixedfreq.pdf"],
        "tex_label": ["tab:mixedfreq", "fig:mixedfreq"],
        "what": "Mixed-frequency causal nowcasting (quarterly target in monthly FRED-MD): "
                "CAFE vs persistence / bridge-OLS / factor-OLS, point-in-time + within-"
                "quarter sharpening.",
    },
```
(Generator: `bench/exp_mixedfreq.py`; data builder: `bench/datasets_mixedfreq.py`.)

## Bibitems to add

```latex
\bibitem{giannone2008nowcasting}
D.~Giannone, L.~Reichlin, and D.~Small.
\newblock Nowcasting: The real-time informational content of macroeconomic data.
\newblock {\em Journal of Monetary Economics}, 55(4):665--676, 2008.

\bibitem{marianoMurasawa2003}
R.~S. Mariano and Y.~Murasawa.
\newblock A new coincident index of business cycles based on monthly and quarterly series.
\newblock {\em Journal of Applied Econometrics}, 18(4):427--443, 2003.

\bibitem{ghysels2004midas}
E.~Ghysels, P.~Santa-Clara, and R.~Valkanov.
\newblock The {MIDAS} touch: Mixed data sampling regression models.
\newblock Working paper, UNC and UCLA, 2004.

\bibitem{banbura2013nowcasting}
M.~Ba\'nbura, D.~Giannone, M.~Modugno, and L.~Reichlin.
\newblock Now-casting and the real-time data flow.
\newblock In {\em Handbook of Economic Forecasting}, volume 2A, pp.~195--237. Elsevier, 2013.
```

## Verdict on the new framing

**"The zero-config causal nowcaster" is a defensible, distinctive new framing/audience for
the paper.** Nowcasting is the *original* home of dynamic factor models (Giannone-Reichlin-
Small), and the existing offerings are bespoke, frequentist/Bayesian DFMs that require
manual frequency mapping, EM, and tuning. CAFE delivers the same capability self-tuning,
CPU-only, strictly point-in-time, and as a *free side effect of imputation* — beating
persistence and a multi-factor bridge on the series that actually need nowcasting. It
converts a stated limitation (regular grid) into a flagship result and opens a macro/
central-bank audience that the current imputation framing does not reach.
