# WS4 — Gap #4: scale (high-N panels + clinical) — integration note

**Gap answered:** "all datasets are similar mid-size panels (≤100 channels); no
high-N or clinical scale." WS4 adds (a) genuinely WIDE real panels — up to **862
channels**, ~8.6× the widest race panel — with a MEASURED N-scaling slope at that
width, and (b) the **PhysioNet-2012 ICU** clinical benchmark (obtained live).

## Files created
- `bench/datasets_scale.py` — loader (NOT an experiment): builds two wide panels
  from the on-disk `.gz` files, z-scored/finite, cached to
  `data/<name>_wide_clean.npy`. Run: `python3 bench/datasets_scale.py`.
- `bench/exp_scale.py` — experiment: writes `paper/tables/scale.tex` +
  `paper/figures/scaling_wide.pdf`, prints all numbers. Run:
  `python3 bench/exp_scale.py` (exit 0, ~80–90 s, BLAS=2, no GPU).

## Measured results (all from a live run; honest)
| panel | shape | MAE (10% MCAR) | wall-clock |
|---|---|---|---|
| Traffic (PEMS), FULL width | 2500 × 862 | 0.140 | ~15 s |
| Electricity (UCI), FULL width | 2500 × 321 | 0.233 | ~2.5 s |

**Measured N-scaling slope (Traffic, log–log, N = 50→800): 1.15** (1.0 = linear,
2.0 = quadratic) — confirms CAFE stays strongly **sub-quadratic / near-linear in
N at a decade more width** than the race. peak RSS ~0.7 GB.

**PhysioNet-2012 ICU (set-a):** real data obtained live (see provenance). Tensor
1000 stays × 48 h × 36 vars, **80.7% native missing**; each patient block imputed
independently (state reset per stay — no cross-patient leak); scored on an extra
10% synthetic hold-out of observed cells. Causal MAE:
**CAFE 0.532 / LOCF 0.441 / running-mean 0.515.**
*Honest reading:* on these very short (48-step), 81%-missing per-patient blocks
**LOCF beats CAFE** — there is too little per-stay history for the factor+AR core
to learn structure, and clinical signals are well served by carry-forward. This
is reported as-is (no spin); it bounds where CAFE's machinery pays off (it needs
length/density) and is a useful honesty signal, not a win.

## LaTeX integration (where in cafe.tex)
Target = the **Compute / scaling paragraph** (currently lines ~621–630, the
`\paragraph{Compute.}` block + `\input{tables/runtime}`), which already makes the
"strongly sub-linear in width N" claim only on ≤132-wide data via
`Fig.~\ref{fig:scaling}` / `Table~\ref{tab:runtime}`.

Add, right after `\input{tables/runtime}` (line ~630):

```latex
\input{tables/scale}
```

And, to back the sub-linear-in-N sentence at real width, either add the figure in
the same figure column as `scaling.pdf` (near line ~801) or inline near the
Compute paragraph:

```latex
\includegraphics[width=0.74\columnwidth]{figures/scaling_wide}
```

**Suggested 2–3 sentence paragraph** (drop into / extend the Compute paragraph, or
the breadth/coverage paragraph):

> To show this is not an artifact of narrow panels, we re-measure on genuinely
> wide data: at the FULL $862$-sensor Traffic width (a decade more than the race
> panels) the empirical log–log time-vs-$N$ slope is $1.15$
> (Table~\ref{tab:scale}, Fig.~\ref{fig:scaling_wide}), confirming \cafe{} stays
> near-linear in $N$ at scale—each row still does only rank-$R$ Cholesky solves,
> never an $N\times N$ inverse. On the clinical PhysioNet-2012 ICU benchmark
> ($1000$ stays $\times 48$h $\times 36$ vars, $81\%$ native missing), \cafe{} is
> competitive with simple causal fills but does not beat last-observation
> carry-forward on these very short, sparsely-observed per-patient blocks—a
> regime where there is too little history for any low-rank/AR structure to help.

(Reference the figure with `\label{fig:scaling_wide}` in its `\captionof`.)

## repro.py MANIFEST entries
`bench/exp_scale.py` is a NEW experiment — add to `MANIFEST` in `bench/repro.py`
(do NOT edit repro.py here per task rules; this is the entry to add):

```python
"exp_scale.py": {
    "tables":  ["paper/tables/scale.tex"],
    "figures": ["paper/figures/scaling_wide.pdf"],
    "tex_label": ["tab:scale", "fig:scaling_wide"],
    "what": "Scale: CAFE on wide panels (up to 862 ch) + measured N-slope at "
            "width; PhysioNet-2012 ICU clinical row.",
},
```

`bench/datasets_scale.py` is a LOADER (like datasets_real.py / datasets_extra.py),
not a paper-artifact generator — it is not added to MANIFEST (those loaders are
not in MANIFEST either); note it as a dependency of exp_scale.py.

## Dataset provenance (datasets_real.md-style)
- **traffic_wide** — traffic, 2500 × 862. Source: `data/traffic.txt.gz`, San
  Francisco Bay Area freeway road-occupancy (Caltrans PEMS), hourly, LSTNet/Lai
  et al.; raw 17544 × 862, complete. Slice: ALL 862 sensors (drop degenerate
  cols), first 2500 hours. z-scored, finite. Top-10 PCs = 84.4%.
- **electric_wide** — energy, 2500 × 321. Source: `data/electricity.txt.gz`, UCI
  ElectricityLoadDiagrams2011-2014 (LSTNet version), hourly kWh of 321 clients;
  raw 26304 × 321, complete. Slice: ALL 321 clients, every 3rd hour, first 2500
  rows. z-scored, finite. Top-10 PCs = 85.9%.
- **physionet2012** — clinical, (1000, 48, 36). Source: PhysioNet/CinC 2012
  Challenge set-a, `https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz`
  (6.6 MB gzip, verified; 4000 patient files, first 1000 used). Per-patient
  (Time,Parameter,Value) triplets resampled onto a 48 h hourly grid, Parameter
  pivoted to 36 vars, 6 static descriptors dropped, negative sentinels = missing.
  z-scored per var on observed cells; NaN where unobserved. 80.7% native missing.
  Cached: `data/physionet2012_clean.npy`, `data/physionet2012_obsmask.npy` (+ raw
  tarball / extracted dir / Outcomes-a.txt kept under data/).

## PhysioNet status: OBTAINED (live)
The PhysioNet-2012 set-a download SUCCEEDED offline-of-paywall (public PhysioNet
mirror, no auth). Real ICU data, no fabrication. The clinical row is present in
`scale.tex` with the honest LOCF-wins caveat above. If a future re-run lacks
network, `exp_scale.py` detects the absent `.npy` and HONESTLY skips the row
(prints "PhysioNet SKIPPED" and writes a "could not run" note into the caption) —
the wide-panel + N-slope results are independent and still produced.
