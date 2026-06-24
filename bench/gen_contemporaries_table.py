r"""
CLOSEST-CONTEMPORARIES capability table for the CAFE paper (Gap #15).

What this is (and is NOT)
-------------------------
This emits a *capability / feature checklist* -- exactly in the spirit of the
paper's existing Table~\ref{tab:special} (classical imputers as CAFE special
cases). It is NOT a metric run: no MAE, no live numbers, no leaderboard. It is a
qualitative comparison of seven DESIGN properties across CAFE and the closest
low-rank / factor / online imputation contemporaries, so a reader can see at a
glance which capabilities co-occur in one method. The caption says so plainly.

Honesty
-------
Every property below was verified against the METHOD'S OWN paper (arxiv IDs in
the inline comments and in ws7.md). Where a method is causal only in a
restricted (filtering) mode but non-causal in its default (smoothing) output, we
score it on the DEFAULT/headline output and footnote the caveat -- we do not give
a method a checkmark for a mode that is not its advertised result.

Columns (7 boolean capabilities)
  C1 causal-verified     strictly point-in-time: cell at t uses only data <= t
  C2 learned-not-tuned   structure/rank/dials learned from data, no per-dataset HP tuning
  C3 Student-t robust    heavy-tailed / explicitly robust likelihood (not Gaussian)
  C4 FE + season         explicit fixed-effects (entity/feature means) AND seasonality
  C5 per-cell UQ         per-imputed-cell posterior variance / predictive interval
  C6 online              streaming / forward-update capable
  C7 CPU-only            classical, no GPU training required

Output: paper/tables/contemporaries.tex  (self-contained float, \label{tab:contemporaries})
Run:    python3 bench/gen_contemporaries_table.py
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TABDIR = os.path.join(_ROOT, "paper", "tables")
os.makedirs(_TABDIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Verified capability matrix.  Each row: (display name, [C1..C7], footnote-key)
# Booleans are True/False ONLY (verified from each method's own paper).
#
# Verification provenance (arxiv / venue) -- see ws7.md for the evidence lines:
#   CAFE              : this paper (causal by construction; ARD-learned rank;
#                       learned nu (Student-t); FE+Fourier season; per-cell post.
#                       variance; strictly online; pure NumPy/CPU).
#   TIDER             : ICLR 2023 (Liu et al.) low-rank MF, trend+Fourier+local
#                       bias; batch; Gaussian; point estimate; CPU.
#   NoTMF             : arXiv 2203.10651 (Chen et al.) nonstationary temporal MF,
#                       VAR on factors + seasonal differencing; batch; Gaussian;
#                       point estimate; CPU.
#   LATC              : arXiv 2104.14936 (Chen, Lei, Saunier, Sun) low-rank AR
#                       tensor completion; global/batch; Gaussian; point; CPU.
#   TRMF              : NeurIPS 2016 (Yu, Rao, Dhillon) temporal regularized MF;
#                       batch; Gaussian; point; CPU.
#   BayOTIDE          : arXiv 2308.14906 (ICML 2024) Bayesian online functional
#                       decomposition; trend(Matern)+periodic kernels; per-cell
#                       posterior; ONLINE/filtering (causal in its native online
#                       mode); Gaussian likelihood; CPU state-space. HP tuned.
#   ImputeFormer      : arXiv 2312.01728 (KDD 2024) low-rank-induced Transformer;
#                       bidirectional attention (non-causal); deep/GPU; point;
#                       trained per dataset.
#   dynamic-factor-EM : arXiv 2502.04112 / Banbura-Modugno 2014; DFM via EM +
#                       Kalman SMOOTHER (default output uses full sample ->
#                       non-causal); MLE-learned; Gaussian; per-cell smoother
#                       variance; CPU. (Causal only if restricted to the filter.)
# --------------------------------------------------------------------------- #
T, F = True, False

ROWS = [
    # name                 C1     C2     C3     C4     C5     C6     C7    note
    ("\\cafe{} (ours)",  [ T,    T,     T,     T,     T,     T,     T ],  None),
    ("TIDER",            [ F,    F,     F,     T,     F,     F,     T ],  None),
    ("NoTMF",            [ F,    F,     F,     F,     F,     F,     T ],  None),
    ("LATC",             [ F,    F,     F,     F,     F,     F,     T ],  None),
    ("TRMF",             [ F,    F,     F,     F,     F,     F,     T ],  None),
    ("BayOTIDE",         [ T,    F,     F,     T,     T,     T,     T ],  "a"),
    ("ImputeFormer",     [ F,    F,     F,     F,     F,     F,     F ],  None),
    ("Dyn.\\,factor-EM", [ F,    T,     F,     T,     T,     F,     T ],  "b"),
]

COL_HEADS = [
    r"\rotcol{Causal-verified}",
    r"\rotcol{Learned, not tuned}",
    r"\rotcol{Student-$t$ robust}",
    r"\rotcol{Fixed-effects $+$ season}",
    r"\rotcol{Per-cell UQ}",
    r"\rotcol{Online}",
    r"\rotcol{CPU-only}",
]

FOOTNOTES = {
    "a": r"Causal in its native online \emph{filtering} mode (scored here); the "
         r"optional full-posterior \emph{smoother} mode is non-causal.",
    "b": r"Default output uses the Kalman \emph{smoother} over the full sample "
         r"(non-causal); strictly causal only if restricted to the filtered pass.",
}


def _mark(b):
    # pifont is loaded in cafe.tex: \ding{51}=check, \ding{55}=cross.
    return r"\ding{51}" if b else r"\ding{55}"


def write_table(path):
    L = []
    A = L.append
    A(r"\begin{table}[t]\centering\small")
    A(r"\setlength{\tabcolsep}{4pt}")
    A(r"% \rotcol: vertical column header; falls back gracefully if rotating "
      r"is unavailable.")
    A(r"\providecommand{\rotcol}[1]{\rotatebox{90}{#1}}")
    A(r"\caption{\textbf{Closest contemporaries: a capability comparison.} "
      r"Like Table~\ref{tab:special}, this is a \emph{qualitative feature "
      r"checklist}, \emph{not} a metric run: each entry is a design property "
      r"verified from the method's own paper, not a measured score. "
      r"\ding{51}\,$=$\,has the capability, \ding{55}\,$=$\,does not. \cafe{} is "
      r"the only method that simultaneously is causal-verified, learns its "
      r"structure without per-dataset tuning, uses a heavy-tailed (Student-$t$) "
      r"likelihood, carries explicit fixed-effects and seasonality, emits "
      r"per-cell uncertainty, runs online, and stays CPU-only. Properties are "
      r"scored on each method's \emph{default/headline} output.}")
    A(r"\label{tab:contemporaries}")
    A(r"\begin{tabular}{@{}l" + "c" * len(COL_HEADS) + r"@{}}")
    A(r"\toprule")
    A(r"Method & " + " & ".join(COL_HEADS) + r" \\")
    A(r"\midrule")
    for name, bits, note in ROWS:
        cells = " & ".join(_mark(b) for b in bits)
        tag = (r"$^{" + note + r"}$") if note else ""
        A(f"{name}{tag} & {cells} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    # footnotes block
    fn = [f"$^{{{k}}}$ {v}" for k, v in FOOTNOTES.items()]
    A(r"\\[2pt]{\footnotesize " + " \\quad ".join(fn) + r"}")
    A(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path}  ({len(ROWS)} methods x {len(COL_HEADS)} capabilities)")


if __name__ == "__main__":
    out = os.path.join(_TABDIR, "contemporaries.tex")
    write_table(out)
    # quick self-check: CAFE row must be all-True; print the capability spread.
    cafe_bits = ROWS[0][1]
    assert all(cafe_bits), "CAFE row must hold all 7 capabilities"
    for name, bits, _ in ROWS:
        print(f"  {name:22s} {''.join('1' if b else '0' for b in bits)}")
    print("OK: contemporaries capability table emitted.")
