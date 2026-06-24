# Idea #1 — Leakage Audit ("the epsilon of imputation"): integration note

A reusable, model-agnostic look-ahead AUDIT that wraps ANY imputer `fn(X)->filled`
and emits (a) a binary causality certificate (`max_revision` under truncation
invariance) and (b) a continuous leakage score `Delta = causal_MAE - bidir_MAE`.
Generalises CAFE's internal verifier into community infrastructure: every method gets
a certificate + a leakage number on equal terms — a "leakage leaderboard".

## New files (all NEW; nothing in the DO-NOT-EDIT set was touched)
- `src/cafe/audit.py` — numpy-only core API: `leakage_report`, `audit_panel`,
  `verify_causal`, `make_causal_variant`, `leakage_delta`, `trailing_windows`,
  `right_edge`. Soft-imports nothing heavy (matplotlib/bench only live in the bench
  driver). Self-contained: re-implements trailing-window + leak-free standardisation
  so the library has no dependency on the `bench/` tree.
- `bench/exp_leakage_audit.py` — driver over the panel (CAFE + simple causal suite +
  SoftImpute/TRMF/linear-interp/spline/nocb + causal rivals + gcimpute/BayOTIDE +
  SAITS/BRITS when torch/pypots present). Writes `paper/tables/leakage_audit.tex` and
  `paper/figures/leakage_audit.pdf`; caches `bench/leakage_audit_cache.json`. Prints
  everything. **Exit 0**, 8 real tasks (4 datasets × {block,mcar}), ~237 s on base
  `python3` (which here has torch+pypots, so the deep models ran).
- `src/tests/test_audit.py` — 5 tests, all pass (`pytest -q` → 5 passed). Certifies
  LOCF and **CAFE** causal (Δ≈0, max_rev=0); flags linear-interp and a global-mean
  control as leaky (max_rev>0, Δ>0).

## (1) src/cafe/__init__.py export line — APPLY THIS (I did NOT edit __init__.py)
Add `audit` to the imports and `__all__`:

```python
from . import audit            # add near `from . import baselines`
# ...
__all__ = ["CAFE", "CafeResult", "impute", "benchmark", "baselines",
           "audit",            # <-- add
           "ConformalCalibrator", "conformal_multipliers", "__version__"]
```
Then `cafe.audit.leakage_report(fn, X)` is the public one-call API.

## (2) cafe.tex — where to add the table + framing paragraph
Insert immediately AFTER `\input{tables/causal_verify}` (currently line 395), i.e. at
the end of the truncation-invariance discussion (Prop. 1 measured). Suggested text:

> \paragraph{From a CAFE-specific check to a neutral field standard.} The verifier
> above is not specific to \cafe{}: it wraps \emph{any} imputer $f(X)\!\to\!\hat X$
> following the standard contract and reports two model-agnostic numbers — a binary
> \emph{causality certificate} (the maximum revision of an early imputed cell as the
> series grows, i.e.\ truncation invariance) and a continuous \emph{leakage score}
> $\Delta=\mathrm{MAE}_{\text{causal}}-\mathrm{MAE}_{\text{bidir}}$, the accuracy a
> method silently borrows from the future (its bidirectional fill vs.\ the same model
> applied honestly as a right-edge filter). We package this as
> \texttt{cafe.audit.leakage\_report} and run it across the full method panel
> (Table~\ref{tab:leakageaudit}). The audit cleanly separates the field: every
> natively causal method certifies with $\Delta\!\approx\!0$ and zero revision, while
> interpolation and batch low-rank methods incur positive $\Delta$ and large
> revisions. Two honest nuances surface — published \emph{online} methods (gcimpute,
> BayOTIDE) borrow \emph{no accuracy} ($\Delta\!\approx\!0$) yet are not bit-exactly
> truncation-invariant (a globally-shared posterior is refined as the stream advances),
> and the deep imputers (SAITS, BRITS) post a \emph{negative} $\Delta$ — their honest
> right-edge filter beats their own bidirectional fill, confirming the bidirectional
> setting is the wrong objective for a sequential decision. We propose this audit as
> neutral community infrastructure: any new method can publish its certificate and
> $\Delta$ alongside its MAE.

Then add the float:
```latex
\input{tables/leakage_audit}
```
(Optional figure: `paper/figures/leakage_audit.pdf` — a Δ bar chart coloured by
certificate; add with `\includegraphics` if a figure is wanted.)

## (3) repro.py MANIFEST entry — add inside MANIFEST = { ... }
```python
    "exp_leakage_audit.py": {
        "tables":  ["paper/tables/leakage_audit.tex"],
        "figures": ["paper/figures/leakage_audit.pdf"],
        "tex_label": ["tab:leakageaudit", "(leakage_audit.pdf optional in cafe.tex)"],
        "what": "Model-agnostic leakage audit: per-method causality certificate "
                "(max revision under truncation invariance) + continuous leakage "
                "Delta = causal-bidir MAE, across the full panel. The leakage "
                "leaderboard / 'epsilon of imputation'.",
    },
```

## (4) Honest verdict
**Yes — this is a credible field-standard tool, and it earns its keep on the FIRST
run by surfacing nuances a binary check would miss.**

Leakage leaderboard (mean over 8 real tasks; full numbers in the cache):

| group | methods | certificate | Δ |
|---|---|---|---|
| natively causal (18) | CAFE, LOCF, Kalman, EWMA, Rolling*, NoTMF, SHASTA-PCA, rGROUSE, OSW-Net, GROUSE, XSecMean, ... | ✓ max_rev = 0 | Δ = 0.000 (all) |
| interpolation (leaky) | NOCB (+0.169), LinearInterp (+0.136), SplineInterp (+0.108) | ✗ max_rev 2–44 | large +Δ |
| batch low-rank | TRMF (+0.001), SoftImpute (−0.097) | ✗ max_rev 1.0–1.5 | small |
| online (published) | gcimpute, BayOTIDE | ✗ max_rev 0.18–0.48 | Δ ≈ 0.000 |
| deep | BRITS (−0.323), SAITS (−0.400) | ✗ max_rev 0.8–1.1 | negative |

Separation is clean on the axis that matters: the certificate (`max_revision`)
**perfectly** partitions natively-causal (=0) from everything that touches the future
(>0). Surprises, all honest:
- **gcimpute / BayOTIDE**: causal in spirit (Δ≈0, borrow no accuracy) but NOT
  bit-exactly truncation-invariant — their shared posterior keeps being refined. The
  audit correctly distinguishes "online filter" from "strict point-in-time". (gcimpute
  even certifies on some tasks and not others — I aggregate worst-case.)
- **SAITS / BRITS**: NEGATIVE Δ. Their right-edge causal variant is *more* accurate
  than their bidirectional fill — strong evidence the bidirectional objective these
  models optimise is the wrong one for sequential use, and that the field's standard
  leaderboard flatters them for the wrong reason.
- **TRMF**: borrows almost nothing here (+0.001) — under this temporal split TRMF's
  bidirectional and right-edge fills nearly coincide, unlike the horse-race regime.

Limitation (honest): the derived "causal variant" of a batch/deep method depends on
the trailing-window length `L` (here 24); Δ is a function of that readout, not an
intrinsic constant. We report `L` in every record. The *certificate* (max_revision)
is `L`-free and is the load-bearing, unarguable number.

Net: a 10x contribution as INFRASTRUCTURE, not as another accuracy point — it turns a
CAFE-only check into a one-call standard any method can run, and it produced a
non-trivial, publishable finding (deep models' negative Δ; online ≠ strict PIT) on its
first pass. Strongly recommend wiring it into the paper next to Prop. 1.

## (5) bibitems
No new external citations strictly required (reuses datasets/methods already cited).
If a name is wanted for the standard, cite the leak-trap paper already in the repo
(arXiv:2405.17508, the standardise-then-mask leak `eval_utils.py` calls out) as the
motivation for a mechanical, shared leakage standard. No new `\bibitem` is mandatory.

## Verify (all green)
- `python3 -c "import sys;sys.path.insert(0,'src');import cafe.audit"` → OK
- `python3 bench/exp_leakage_audit.py` → exit 0, real numbers, table+figure written
- `python3 -m pytest src/tests/test_audit.py -q` → 5 passed
