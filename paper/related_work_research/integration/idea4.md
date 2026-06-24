# Idea #4 — Recoverability certificate / risk-controlled (selective) imputation

**Status:** validated, ready to wire centrally. New code only; no shipped file edited.

## What was built

- `src/cafe/recoverability.py` — numpy-only certificate. Public API:
  - `recoverability_score(*, sigma, miss, obs_mask, q_grid, row_w, loading_energy, active_rank, weights=None, return_components=False)` — per-cell score in `[0,1]` (1 = trustworthy), weighted geometric mean of five named channels (`c_scale` calibrated-σ, `c_anchor` cross-sectional anchors, `c_recency` time-since-observed, `c_span` factor/loading support, `c_robust` Student-t row weight). NaN at observed cells.
  - `certificate_components(...)` — the five sub-signals (each in `[0,1]`), for introspection.
  - `score_from_result(res, conformal=True, weights=None, return_components=False, **conformal_kwargs)` — extracts every signal from a **traced** `CafeResult` and returns the score. This is the single mapping from CAFE state → certificate; the `CafeResult` methods just wrap it.
  - `selective_impute(filled, score, min_confidence)` — `filled` with NaN where `score < tau`.
- `bench/exp_recoverability.py` — validation; writes `paper/tables/recoverability.tex` + `paper/figures/recoverability.pdf`. Exits 0.
- `src/tests/test_recoverability.py` — score in `[0,1]`/finite/NaN-at-observed; selective abstains monotonically more as `tau` rises; pure-function path. 4 tests pass.

## Central wiring (for you to add to the DO-NOT-EDIT files)

### `src/cafe/__init__.py`
Add the import + exports:
```python
from .recoverability import recoverability_score, selective_impute, score_from_result
# ... append to __all__:
"recoverability_score", "selective_impute", "score_from_result",
```

### `src/cafe/model.py` — two `CafeResult` methods (paste in the class, near `calibrated_uncertainty`)
```python
    # ---- per-cell recoverability certificate (selective / risk-controlled imputation) ----
    def recoverability_score(self, conformal=True, weights=None, return_components=False,
                             **conformal_kwargs):
        """Per-cell recoverability certificate in [0,1] (1 = trustworthy fill, 0 = CAFE
        cannot recover this cell), NaN where observed. Built from this run's own state
        (posterior sigma, the causal conformal scale, cross-sectional anchor support,
        factor/loading support, Student-t robustness) -- no held-out truth is used.
        Needs the traced run (CAFE().run on 1D/2D). ``conformal=True`` expresses the
        certificate on the calibrated q*sigma scale; extra kwargs go to the calibrator."""
        from .recoverability import score_from_result
        s = score_from_result(self, conformal=conformal, weights=weights,
                              return_components=return_components, **conformal_kwargs)
        if return_components:
            score, comp = s
            return from_matrix(score, self._ctx), comp
        return from_matrix(s, self._ctx)

    def selective_imputed(self, min_confidence=0.5, **kwargs):
        """The imputation with NaN wherever the recoverability certificate < ``min_confidence``
        (abstain on cells CAFE cannot recover, rather than returning a confident-but-wrong
        value). Observed cells are always kept. ``kwargs`` forwarded to
        :meth:`recoverability_score`."""
        from .recoverability import selective_impute
        import numpy as _np
        score = _np.asarray(self.recoverability_score(**kwargs)
                            if self._ctx.kind == "numpy"
                            else from_matrix.__self__ and score_from_result(self, **kwargs))
        # simplest robust form: compute on the raw matrix grid
        from .recoverability import score_from_result
        sc = score_from_result(self, **kwargs)
        out = selective_impute(self._filled, sc, min_confidence)
        return from_matrix(out, self._ctx)
```
(Use the clean `selective_imputed` body — it calls `score_from_result` on the raw matrix and `selective_impute`; the first three lines above are illustrative and can be dropped. Net: `sc = score_from_result(self, **kwargs); out = selective_impute(self._filled, sc, min_confidence); return from_matrix(out, self._ctx)`.)

Panel runs (untraced) raise `NotImplementedError` from `score_from_result` — same contract as `calibrated_uncertainty`.

## Paper (`paper/cafe.tex`)

- **Figure** `\label{fig:recoverability}` (file `paper/figures/recoverability.pdf`): (a) calibration curve, (b) risk-coverage cert-vs-random.
- **Table** `\input{tables/recoverability}` (`\label{tab:recoverability}`).
- **Placement:** add a paragraph + the table right after the extreme-missingness paragraph (currently ends line ~731, `\input{tables/extreme_missing}`), since the certificate is the *operational* turn of the recoverability frontier; cross-reference the conformal paragraph (line ~756). Suggested paragraph:

> **Selective imputation: a recoverability certificate.** The recoverability frontier above is descriptive; we turn it into an operational, per-cell *certificate* in $[0,1]$ that a downstream user can act on. From the same causal pass we combine the posterior $\sigma$, the split-conformal scale $q(\text{level})$, the cross-sectional anchor support behind a cell, the feature's loading energy (factor-spanned vs idiosyncratic), and the Student-$t$ row weight into a single confidence score (Table~\ref{tab:recoverability}, Fig.~\ref{fig:recoverability}). It is genuinely calibrated on real panels: pooled across Beijing/ETTh1/FRED-MD the certificate ranks held-out error with Spearman $\rho=+0.45$ and a monotone calibration curve, so gating fills by the certificate---abstaining (returning NaN) on the least-confident cells---drops retained MAE by $\sim$\,$37\%$ at $50\%$ coverage versus $\approx0\%$ for random abstention. On the spanned-vs-idiosyncratic frontier the certificate ranks the idiosyncratic (no cross-sectional evidence) column well below the factor-spanned one, i.e.\ \cafe{} flags the cells it cannot recover. \emph{Honestly, the calibration is dataset-dependent: it is sharp on FRED-MD ($\rho=+0.54$) and drives the pooled win and the cross-regime separation, but is nearly flat \emph{within} Beijing and ETTh1 ($\rho\approx0.03$–$0.05$)}---the certificate reliably separates recoverable from unrecoverable \emph{regimes} but is a weaker per-cell error rank inside an already-homogeneous panel. As a risk control (abstain rather than emit a confident-but-wrong fill) it is deployable; as a fine-grained per-cell error predictor it is partial and reported as such.

This honesty is required by the contract: do **not** state per-cell calibration is uniform.

## `bench/repro.py` MANIFEST entry

Add to the `MANIFEST` dict:
```python
    "exp_recoverability.py": {
        "tables":  ["paper/tables/recoverability.tex"],
        "figures": ["paper/figures/recoverability.pdf"],
        "tex_label": ["tab:recoverability", "fig:recoverability"],
        "what": "Recoverability certificate / selective imputation: calibration (cert vs "
                "realized error) + risk-coverage (cert-gated vs random) on real panels and "
                "the factor-spanned vs idiosyncratic frontier.",
    },
```

## Bibitems (selective / risk-controlled / conformal-risk prediction) — add to `thebibliography`

```latex
\bibitem{elyaniv2010selective} R.~El-Yaniv, Y.~Wiener. On the foundations of
noise-free selective classification. \emph{JMLR}, 11:1605--1641, 2010.
\bibitem{geifman2017selectivenet} Y.~Geifman, R.~El-Yaniv. Selective prediction
with deep neural networks. \emph{NeurIPS}, 2017. \emph{arXiv:1705.08500}.
\bibitem{angelopoulos2024conformalrisk} A.~N.~Angelopoulos, S.~Bates, A.~Fisch,
L.~Lei, T.~Schuster. Conformal risk control. \emph{ICLR}, 2024. \emph{arXiv:2208.02814}.
```
(`zaffran2023cpmv` and `gui2023cmc` already present cover conformal-under-missingness / conformalized matrix completion; cite alongside.)

## Honest verdict (from a live run, `python3 bench/exp_recoverability.py`)

- **Calibrated?** YES in aggregate: pooled real ρ(cert, −|err|) = **+0.45**, calibration curve monotone-down fraction **0.86** (bin mean |err| falls 0.28 → 0.03 as cert rises). **Caveat:** per-dataset is heterogeneous — FRED-MD ρ=+0.54, but Beijing ρ=+0.03 and ETTh1 ρ=+0.05 are nearly flat. Strong across *regimes*, weak *within* a homogeneous panel.
- **Beats random by a wide margin?** YES. Real panels: cert-gated MAE 0.215 (100%) → 0.196 (80%) → 0.135 (50%); random stays flat at 0.215. 50%-coverage drop: **37% (cert) vs −0% (random)**.
- **Flags idiosyncratic-unrecoverable cells?** YES. Frontier: idiosyncratic mean cert **0.42** (|err| 0.69) vs factor-spanned **0.61** (|err| 0.13); gap +0.20.
- **Deployable claim "CAFE knows what it cannot recover"?** YES as a **risk control / abstention** mechanism (abstain on the unrecoverable rather than emit a confident-but-wrong fill, and cleanly separate recoverable from unrecoverable regimes). PARTIAL as a fine-grained per-cell error predictor inside an already-homogeneous panel — report both halves.
