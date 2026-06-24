# ws3 — Gap #2: sensitivity to fixed internal constants ("no-hyperparameter" stress test)

## What this delivers
A live sweep of every fixed internal constant CAFE carries (trailing window, the
factor-vs-season gate `g=N/(N+R)`, the kurtosis→ν map `ν=4+6/κ` base & slope, the
season winsor band, the inverse-variance fusion weight) over a wide band around each
default, plus a deliberately-wrong harmonic-menu **aliasing** check. Message: results are
**flat in a wide band around every default** (the defaults are principled, not tuned),
and the one real sensitivity (the harmonic menu, on a lone periodic series) is reported
openly rather than hidden.

## Headline numbers (live run, 4 structured panels × 3 seeds, 10% MCAR)
Max % MAE change across each swept band (small ⇒ insensitive):

| Constant | swept range | max ΔMAE |
|---|---|---|
| Trailing window W | 100–800 | 1.8% |
| Season gate R in g=N/(N+R) | 2–16 | 0.2% |
| ν-map base (4+6/κ) | 2–6 | 0.6% |
| ν-map slope (4+6/κ) | 3–12 | 0.2% |
| Season winsor band k | 0.5–4 | 3.3% |
| Inverse-var fusion tilt | 0.25–4 | 7.3% |

**Aliasing penalty:** on a lone strongly-periodic series (true periods 24,168) under
block gaps, a grossly wrong harmonic menu costs **+75.6%** MAE vs the canonical menu —
the single real sensitivity, surfaced not hidden. (On wide panels the ARD/factor gate
absorbs the cycle so the menu is near-inert there; the lone-series block-gap regime is
where it actually bites.)

## LaTeX include lines
```latex
\input{tables/sensitivity}                 % Table tab:sensitivity
```
```latex
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figures/sensitivity}
  \caption{...}\label{fig:sensitivity}
\end{figure}
```

## Where in cafe.tex
- **Primary anchor:** the `\paragraph{Adaptation is intrinsic.}` block (cafe.tex
  ~L270–282), which ends "No quantity in \eqref{eq:obj} is a constant fit to a
  benchmark." Add `\input{tables/sensitivity}` and the `fig:sensitivity` float
  immediately after that paragraph — the table is the direct empirical backing for
  that exact claim.
- **Secondary anchor:** the `\paragraph{Ablation.}` discussion (cafe.tex ~L564–575)
  and the no-hyperparameter claims at L93 / L195–197 may cross-reference
  `Table~\ref{tab:sensitivity}` / `Fig.~\ref{fig:sensitivity}`.

## Suggested 2–3 sentence paragraph (drop after "Adaptation is intrinsic")
> \paragraph{No tuned constants.} CAFE still carries a handful of \emph{fixed} internal
> defaults (the trailing window, the season gate $g{=}N/(N{+}R)$, the kurtosis$\to\nu$
> map $\nu{=}4{+}6/\kappa$, the winsor band, the inverse-variance fusion). To show these
> are principled rather than tuned, Table~\ref{tab:sensitivity} and
> Fig.~\ref{fig:sensitivity} sweep each over a wide band: the mean causal MAE moves by at
> most $7.3\%$ (most by $<3\%$) anywhere in the band, so accuracy is flat in a broad
> neighbourhood of every default. The one genuine sensitivity is the seasonal
> \emph{harmonic menu}: on a lone periodic series under block gaps, where the cycle is the
> only way to fill a gap, a grossly wrong menu costs $+75.6\%$ MAE---reported here openly,
> and inert on wide panels where the shared factors absorb the cycle.

## repro.py MANIFEST entry
Add to `MANIFEST` in `bench/repro.py` (DO-NOT-EDIT in this workstream; apply at merge):
```python
"exp_sensitivity.py": {
    "tables":  ["paper/tables/sensitivity.tex"],
    "figures": ["paper/figures/sensitivity.pdf"],
    "tex_label": ["tab:sensitivity", "fig:sensitivity"],
    "what": "Sensitivity of causal MAE to each fixed internal constant (flat in a wide "
            "band) + harmonic-menu aliasing penalty.",
},
```

## Core edit (additive, defaults bit-identical) + invariance evidence
`bench/c_unified_penmf.py` was edited **additively only**: a `_SENS_DEFAULTS` dict, a
`_sens(key)` getter (reads an override from the per-call `meta` dict or a `CAFE…` env
var, else returns the original hard-coded default), and a save/restore of the active
override snapshot inside `online_impute`. Each swept constant's literal in the hot path
was replaced by `_sens("…")`, whose default equals the original literal. The trailing
`WINDOW` global is now stored per-core as `self.WINDOW` (default 400). **With no override
supplied, behaviour is bit-identical.**

Invariance proof (make_paper.py `_beijing_sota_mae` recipe: `data/beijing_clean.npy`,
3 seeds, 10% MCAR, `online_impute`):

- BEFORE edits: `0.1082740074`
- AFTER edits, no overrides: `0.1082740074`  → identical to 10 digits.

`exp_sensitivity.py` re-runs this check at startup and prints the result every run.

## Run
`python3 bench/exp_sensitivity.py` (BLAS=2, base python3) → exit 0, ~3 min on one core.
Writes `paper/tables/sensitivity.tex` (real numbers) and `paper/figures/sensitivity.pdf`,
and prints all numbers + the Beijing-MAE invariance line to stdout.
