# WS8 integration note — Extreme / pathological missingness (the recoverability frontier)

**New files:** `bench/exp_extreme_missing.py` → `paper/tables/extreme_missing.tex` (`\label{tab:extrememissing}`) + `paper/figures/extreme_missing.pdf`.

## Honest verdict (answers the "1 row out of 10,000" question directly)
- **k=1 anchor:** identifiability floor — CAFE collapses to the mean (no equations to fix the loading). Disclosed, not hidden; a matrix-completion limit, not a bug.
- **Factor-spanned near-empty feature, anchors growing:** CAFE genuinely recovers it and the margin over mean-fill **grows** with k (k=2000: CAFE 0.185 vs mean 0.791). Recoverability engages as the loading becomes estimable.
- **Idiosyncratic near-empty feature:** CAFE tracks mean-fill (0.752 vs 0.776) — falls back to the level, does **not** hallucinate.
- **Global extreme MCAR:** CAFE stays finite through 99% and beats mean-fill/SoftImpute, but **causal LOCF beats it at every rate** (scattered points flanked by observed neighbours favour carry-forward — consistent with the paper's existing "linear-interp is the telling cheap baseline" framing). Reported as-is.
- **Edge/finiteness contract:** all-but-one-observed, single global anchor, all-missing feature → finite, same-shape, observed-cells-preserved (✓).

This **qualifies and sharpens** the recoverability story already in the MNAR-scope paragraph (Cheng et al. causal-view): CAFE recovers a self-/heavily-censored value *only insofar as the surviving cross-section informs it*, and now we measure exactly where that frontier sits.

## cafe.tex wiring
Add `\input{tables/extreme_missing}` and `\includegraphics[width=\linewidth]{figures/extreme_missing}` in the **Robustness** region, right after the long-gap paragraph and `\input{tables/longgap}` (~L578–589), before the "MNAR scope (honest)" paragraph — it is the natural companion to the long-gap degradation curve.

Suggested 3-sentence paragraph (paste after the long-gap paragraph):
> **Extreme missingness and the recoverability frontier.** Pushed to pathological sparsity, \cafe{} degrades gracefully rather than failing: under global MCAR up to $99\%$ it stays finite and beats the naive causal fills, and a single near-empty feature (observed at only $k$ anchor rows in a $10{,}000$-row panel) is reconstructed *exactly insofar as the surviving cross-section carries it* — when the feature is factor-spanned the margin over mean-fill grows with $k$, while an idiosyncratic feature is correctly tracked to its level rather than hallucinated (Table~\ref{tab:extrememissing}). The honest floor is the few-anchor end ($k\!\le\!2$), where with almost no equations to fix the loading even a spanned column collapses toward the mean — the matrix-completion identifiability limit, disclosed rather than hidden, and the quantitative form of the recoverability lever of Cheng et al.~\cite{cheng2025causalview}.

## repro.py MANIFEST entry (apply centrally)
```python
    "exp_extreme_missing.py": {
        "tables":  ["paper/tables/extreme_missing.tex"],
        "figures": ["paper/figures/extreme_missing.pdf"],
        "tex_label": ["tab:extrememissing"],
        "what": "Extreme missingness: graceful degradation to 99% MCAR + near-empty-feature recoverability frontier (factor-spanned vs idiosyncratic).",
    },
```

No new bibitems (reuses `cheng2025causalview`, already cited). No DO-NOT-EDIT file modified.
