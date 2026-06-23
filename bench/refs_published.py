#!/usr/bin/env python3
"""refs_published.py — SINGLE SOURCE OF TRUTH for published competitor numbers.

WHY THIS FILE EXISTS
--------------------
The repo previously carried TWO contradictory registries of published
competitor MAEs that disagreed on every shared method, e.g.:
    paper/cafe.tex tab:sota  : BRITS .153, SAITS .137, Transformer .158  (du2023)
    src/cafe/benchmark.py     : BRITS .127, SAITS .155, Transformer .142  (TSI-Bench)
Because they were never reconciled, CAFE's headline ".108" was implicitly
claimed to beat BOTH sets — yet under the TSI-Bench source CSDI (.102) actually
beats CAFE. "Lowest MAE" therefore depended on which un-reconciled table you
picked. That is exactly the kind of citation drift that gets caught in review.

This module is the ONE registry. Every published number lives here as a
structured record with its own source. Where two sources disagree on the same
(method, dataset, mask, rate, metric) cell we record BOTH entries, each tagged
with its source — we never silently pick one.

NON-NEGOTIABLES honoured here
-----------------------------
* These are PUBLISHED numbers only (we never run competitor deep models).
* They come from DIFFERENT preprocessing / windowed train-val-test protocols
  than CAFE's full-series causal online imputation. They are CONTEXT, NEVER a
  like-for-like head-to-head leaderboard. render_latex() makes that explicit and
  SEPARATES causal vs bidirectional methods into their own blocks.
* Anything we could not pin to a specific paper/table is marked source/value as
  "unverified" rather than fabricated.

USAGE
-----
    python3 bench/refs_published.py          # writes paper/tables/published_refs.tex
    from refs_published import REFS, get, render_latex
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Output path — derived relative to THIS file so the repo can be renamed or
# moved without breaking (no hardcoded /Users/... absolute path; that is the
# leak class that embarrassed TSI-Bench).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
TAB_OUT = os.path.join(_REPO, "paper", "tables", "published_refs.tex")


# --------------------------------------------------------------------------- #
# Source registry — every literature citation used below, keyed by a short id so
# the table can show a compact tag and the full reference is auditable in ONE
# place. Keep these strings verbatim-checkable against the cited papers.
# --------------------------------------------------------------------------- #
SOURCES: dict[str, str] = {
    "du2023":      "Du et al. 2023, SAITS (ESWA), Table 2",
    "tsibench":    "TSI-Bench, arXiv:2406.12747 (NeurIPS'24 D&B; withdrawn from ICLR'25)",
    "fgti2024":    "FGTI, NeurIPS'24 (frequency-domain diffusion)",
    "csdi2021":    "CSDI, NeurIPS'21, arXiv:2107.03502",
    "pristi2023":  "PriSTI, ICDE'23",
    "impf2024":    "ImputeFormer, KDD'24, arXiv:2312.01728",
    "gpvae2020":   "GP-VAE, AISTATS'20 (as reported by CSDI/SAITS)",
    "unverified":  "UNVERIFIED — not pinned to a specific paper/table",
}


@dataclass(frozen=True)
class Ref:
    """One published competitor number, fully self-describing.

    Two records may share (method, dataset, mask_type, rate, metric) but differ
    in `value`/`source`: that is an INTENTIONAL recorded disagreement, not a bug.
    """

    method: str
    dataset: str
    variant: str            # preprocessing / split protocol, e.g. "windowed train/val/test"
    mask_type: str          # mcar-point / block / etc.
    rate: float             # missing fraction, e.g. 0.10
    metric: str             # "MAE" / "CRPS" / "RMSE" ...
    value: object           # float, or the string "unverified"
    source: str             # key into SOURCES
    hardware: str           # "GPU" / "CPU"
    causal: bool
    bidirectional: bool
    note: str = field(default="")

    @property
    def disagree_key(self) -> tuple:
        """Cells that, if shared by 2+ records, mark a cross-source disagreement."""
        return (self.method, self.dataset, self.mask_type, self.rate, self.metric)


# --------------------------------------------------------------------------- #
# THE REGISTRY.
# Only numbers present in the benchmark-hardening notes or pinned to a known
# paper/table are given as floats; everything else is value="unverified".
#
# Standardised MAE @ 10% point/MCAR unless stated. All deep methods below are
# BIDIRECTIONAL + GPU (each fill sees the whole series) under a WINDOWED
# train/val/test protocol — a different setting from CAFE's full-series causal
# online impute. Hence: context column, never a ranked board with CAFE inside.
# --------------------------------------------------------------------------- #
REFS: list[Ref] = [
    # ---- Beijing Air-Quality, MAE @ 10% MCAR-point ----
    # Two sources disagree on every shared method: record BOTH (du2023 vs tsibench).
    Ref("SAITS", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.137, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("SAITS", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.155, "tsibench", "GPU", causal=False, bidirectional=True,
        note="disagrees with du2023 (.137) — different preprocessing/window"),
    Ref("BRITS", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.153, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("BRITS", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.127, "tsibench", "GPU", causal=False, bidirectional=True,
        note="disagrees with du2023 (.153)"),
    Ref("Transformer", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.158, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("Transformer", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.142, "tsibench", "GPU", causal=False, bidirectional=True,
        note="disagrees with du2023 (.158)"),
    Ref("iTransformer", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.123, "tsibench", "GPU", causal=False, bidirectional=True),
    Ref("CSDI", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.102, "tsibench", "GPU", causal=False, bidirectional=True,
        note="under THIS source CSDI beats CAFE's .108 — do not claim global 'lowest MAE'"),
    Ref("GP-VAE", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.268, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("M-RNN", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.294, "du2023", "GPU", causal=False, bidirectional=True),
    # FGTI — Beijing MAE 0.149 @10% (benchmark-hardening notes, line "Bar: MAE 0.149@10% KDD").
    # MUST stay in its own labeled context cell; NEVER head-to-head vs our .108
    # (different Beijing variant + masking — explicit punch-list item #2).
    Ref("FGTI", "Beijing Air-Quality", "KDD/windowed (FGTI paper)", "mcar-point", 0.10,
        "MAE", 0.149, "fgti2024", "GPU", causal=False, bidirectional=True,
        note="FGTI paper reports SAITS .304 on its own variant — protocol-specific, "
             "NOT comparable to du2023/tsibench SAITS or to CAFE .108"),
    # ImputeFormer — closest published low-rank rival; same turf (Beijing/traffic/solar).
    # No single MAE value pinned to a table in our notes -> mark unverified, do NOT invent.
    Ref("ImputeFormer", "Beijing Air-Quality", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", "unverified", "impf2024", "GPU", causal=False, bidirectional=True,
        note="closest low-rank competitor; exact Beijing MAE cell not pinned in our "
             "notes — fill from the KDD'24 table before citing a number"),

    # ---- PhysioNet-2012, MAE @ 10% (SAITS Table 2) ----
    Ref("SAITS", "PhysioNet-2012", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.186, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("Transformer", "PhysioNet-2012", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.190, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("BRITS", "PhysioNet-2012", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.256, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("M-RNN", "PhysioNet-2012", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.533, "du2023", "GPU", causal=False, bidirectional=True),

    # ---- Electricity, MAE @ 10% (SAITS Table 2) ----
    Ref("SAITS", "Electricity", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.735, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("Transformer", "Electricity", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.823, "du2023", "GPU", causal=False, bidirectional=True),
    Ref("BRITS", "Electricity", "windowed train/val/test", "mcar-point", 0.10,
        "MAE", 0.847, "du2023", "GPU", causal=False, bidirectional=True),

    # ---- AQI-36, CRPS (probabilistic) — PriSTI beats CSDI here (notes, line 12) ----
    Ref("PriSTI", "AQI-36", "block (CSDI/PriSTI protocol)", "block", 0.10,
        "CRPS", 0.0997, "pristi2023", "GPU", causal=False, bidirectional=True,
        note="PriSTI reports 0.0997 vs CSDI 0.1056 on AQI-36"),
    Ref("CSDI", "AQI-36", "block (CSDI/PriSTI protocol)", "block", 0.10,
        "CRPS", 0.1056, "pristi2023", "GPU", causal=False, bidirectional=True),
    Ref("GP-VAE", "AQI-36", "block (CSDI/PriSTI protocol)", "block", 0.10,
        "CRPS", 0.3377, "csdi2021", "GPU", causal=False, bidirectional=True,
        note="weakest probabilistic baseline; cite, don't fear"),
]


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #
def get(method: Optional[str] = None, dataset: Optional[str] = None,
        metric: Optional[str] = None) -> list[Ref]:
    """Filter the registry. None = wildcard. Case-insensitive on method/dataset."""
    out = []
    for r in REFS:
        if method is not None and r.method.lower() != method.lower():
            continue
        if dataset is not None and r.dataset.lower() != dataset.lower():
            continue
        if metric is not None and r.metric.lower() != metric.lower():
            continue
        out.append(r)
    return out


def disagreements() -> dict[tuple, list[Ref]]:
    """Return cells where >=2 sources report DIFFERENT values for the same key."""
    by_key: dict[tuple, list[Ref]] = {}
    for r in REFS:
        by_key.setdefault(r.disagree_key, []).append(r)
    out = {}
    for k, grp in by_key.items():
        vals = {x.value for x in grp if x.value != "unverified"}
        if len(vals) >= 2:
            out[k] = grp
    return out


# --------------------------------------------------------------------------- #
# LaTeX rendering
# --------------------------------------------------------------------------- #
def _esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def _fmt_val(v) -> str:
    if v == "unverified":
        return r"\textit{unver.}"
    return f"{float(v):.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else _esc(v)


def _src_tag(key: str) -> str:
    """Short bracketed source tag for the table body."""
    return _esc(key)


def render_latex(path: str = TAB_OUT) -> str:
    """Write a self-contained LaTeX float to `path` and return the LaTeX string.

    Layout: ONE table, split into a CAUSAL block and a BIDIRECTIONAL block, with
    a caption stating in plain terms that these are PUBLISHED numbers under
    DIFFERENT protocols (context, not head-to-head). Rows are grouped by dataset;
    cross-source disagreements appear as separate rows, each carrying its source.
    """
    # Order rows: causal first (currently none published in our scope — that IS
    # the moat), then bidirectional, grouped by dataset then metric.
    causal_rows = [r for r in REFS if r.causal]
    bidir_rows = [r for r in REFS if not r.causal]

    def _row(r: Ref) -> str:
        cells = [
            _esc(r.method),
            _esc(r.dataset),
            _esc(r.metric),
            f"{r.rate:.2f}",
            _fmt_val(r.value),
            _src_tag(r.source),
            r.hardware,
        ]
        return "  " + " & ".join(cells) + r" \\"

    def _block(rows: list[Ref], title: str) -> list[str]:
        if not rows:
            return [r"  \multicolumn{7}{@{}l}{\emph{" + _esc(title) +
                    r" --- none published (this is the CAFE moat)}}\\"]
        # group by dataset for readability
        rows = sorted(rows, key=lambda x: (x.dataset, x.metric, str(x.value)))
        lines = [r"  \multicolumn{7}{@{}l}{\textbf{" + _esc(title) + r"}}\\",
                 r"  \midrule"]
        prev_ds = None
        for r in rows:
            if r.dataset != prev_ds:
                if prev_ds is not None:
                    lines.append(r"  \addlinespace[2pt]")
                prev_ds = r.dataset
            lines.append(_row(r))
        return lines

    # Build the unique source legend actually used in the table.
    used = sorted({r.source for r in REFS})
    legend = "; ".join(f"\\textbf{{{_esc(k)}}}: {_esc(SOURCES[k])}" for k in used)

    body = []
    body += _block(causal_rows, "Causal / point-in-time (no future access)")
    body.append(r"  \midrule")
    body += _block(bidir_rows, "Bidirectional (each fill sees the whole series; GPU)")

    tex = r"""% AUTO-GENERATED by bench/refs_published.py -- DO NOT EDIT BY HAND.
% Single source of truth for PUBLISHED competitor numbers.
\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{6pt}
\caption{\textbf{Published competitor numbers --- context, not a head-to-head
leaderboard.} Every value below is taken \emph{verbatim from the cited paper};
we never re-run competitor deep models. \textbf{These numbers were produced under
DIFFERENT data variants, masks, and (windowed train/val/test) protocols from
\cafe{}'s full-series causal online imputation, so they are NOT like-for-like and
\cafe{} is deliberately \emph{not} ranked into this table.} All listed deep
methods are \emph{bidirectional} (each fill sees the whole series) and GPU-trained;
no prior method here is causal/point-in-time, which is the gap \cafe{} fills. Where
two sources report different values for the same cell, BOTH are listed with their
source (e.g.\ Beijing SAITS .137 vs .155, BRITS .153 vs .127) --- under the
TSI-Bench source CSDI (.102) is the strongest Beijing MAE, so no single ``lowest
MAE'' claim is protocol-independent. Cells we could not pin to a specific table are
marked \textit{unver.}\ rather than guessed.}
\label{tab:publishedrefs}
\begin{tabular}{@{}lllccll@{}}
\toprule
Method & Dataset & Metric & Rate & Value$\downarrow$ & Source & HW \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\vspace{2pt}

{\footnotesize\emph{Sources.} """ + legend + r"""}
\end{table*}
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(tex)
    return tex


if __name__ == "__main__":
    tex = render_latex()
    print(f"[refs_published] wrote {TAB_OUT}  ({len(REFS)} records, "
          f"{sum(1 for r in REFS if r.value == 'unverified')} unverified)")
    dis = disagreements()
    if dis:
        print(f"[refs_published] {len(dis)} cross-source disagreement(s) recorded "
              f"(BOTH kept, never silently resolved):")
        for k, grp in dis.items():
            vals = ", ".join(f"{x.value} [{x.source}]" for x in grp
                             if x.value != "unverified")
            print(f"  - {k[0]} / {k[1]} / {k[2]} @ {k[3]:.2f} {k[4]}: {vals}")
