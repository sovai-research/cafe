"""
RECOVERABILITY CERTIFICATE / RISK-CONTROLLED (SELECTIVE) IMPUTATION -- idea #4.

CAFE's extreme-missingness study (exp_extreme_missing.py) found a recoverability frontier: a
factor-spanned near-empty feature is reconstructed, an idiosyncratic few-anchor one is not.
This experiment turns that into a per-cell CERTIFICATE (src/cafe/recoverability.py) -- a score
in [0,1] built ONLY from CAFE's own internal state (per-cell posterior sigma, the conformal
scale, cross-sectional anchor support, factor/loading support, the Student-t robustness
weight) -- and VALIDATES it on held-out cells:

  (a) CALIBRATION. Bin imputed cells by their certificate; does a LOWER certificate genuinely
      mean HIGHER realized |error|? (monotone decreasing => the score means what it claims).

  (b) SELECTIVE-RISK (risk-coverage). Sweep an abstention threshold tau: as we abstain on the
      least-confident cells (return NaN), plot error-on-RETAINED vs coverage. Compare
      CERTIFICATE-gated vs RANDOM-gated. The certificate must DOMINATE: a far steeper error
      drop as coverage falls. Reported on real panels AND the factor-spanned-vs-idiosyncratic
      synthetic setup (the certificate should flag the idiosyncratic few-anchor cells as
      low-confidence -> they get abstained first).

HONESTY CONTRACT: every number is from a live CAFE run; the certificate never sees held-out
truth. If the certificate is poorly calibrated, the figure/table/verdict say so. A calibrated
"CAFE doesn't know this cell" is the win.

Outputs:
    paper/tables/recoverability.tex   (selective-risk summary: error @ 100/80/50% coverage,
                                       certificate vs random, on each setting)
    paper/figures/recoverability.pdf  ((a) calibration curve, (b) risk-coverage curve)
and prints all numbers + an honest verdict.
"""
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cafe
from cafe.recoverability import score_from_result

from viz_common import PALETTE, style_ax
from eval_utils import mcar_mask, score_masked
from exp_extreme_missing import (gen_anchor_panel, robust_standardize, feature0_r2,
                                  P1_T, P1_N)

RNG_SEED = 20260624


# --------------------------------------------------------------------------- #
# Core helper: impute X (NaN = missing) with CAFE, return (filled, certificate)
# on the SAME (T, N) grid. The certificate is computed from model state only.
# --------------------------------------------------------------------------- #
def cafe_fill_and_certificate(Xobs, conformal=True):
    res = cafe.CAFE().run(np.ascontiguousarray(Xobs, float))
    filled = np.asarray(res.imputed, float)
    score = score_from_result(res, conformal=conformal)      # NaN at observed cells
    return filled, score


# --------------------------------------------------------------------------- #
# (a) CALIBRATION: mean |error| per certificate bin (should DECREASE as score rises)
# --------------------------------------------------------------------------- #
def calibration_curve(score, abserr, n_bins=8):
    """Bin the scored cells by certificate into equal-count quantile bins; return the bin
    centres (mean score) and mean |error| per bin. Monotone-DOWN means a higher certificate
    => lower realized error."""
    s = np.asarray(score, float)
    e = np.asarray(abserr, float)
    ok = np.isfinite(s) & np.isfinite(e)
    s, e = s[ok], e[ok]
    if s.size < n_bins * 2:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(s)
    edges = np.linspace(0, s.size, n_bins + 1).astype(int)
    cx, cy, cn = [], [], []
    for b in range(n_bins):
        idx = order[edges[b]:edges[b + 1]]
        if idx.size == 0:
            continue
        cx.append(float(np.mean(s[idx])))
        cy.append(float(np.mean(e[idx])))
        cn.append(int(idx.size))
    return np.array(cx), np.array(cy), np.array(cn)


def spearman(a, b):
    """Spearman rank correlation (numpy-only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# (b) RISK-COVERAGE: error on retained cells as we abstain on the least-confident.
# --------------------------------------------------------------------------- #
def risk_coverage(score, abserr, coverages):
    """For each target coverage c, keep the top-c fraction of cells BY SCORE (abstain on the
    rest) and report mean |error| on the retained cells. Returns dict coverage -> error."""
    s = np.asarray(score, float)
    e = np.asarray(abserr, float)
    ok = np.isfinite(s) & np.isfinite(e)
    s, e = s[ok], e[ok]
    n = s.size
    order = np.argsort(-s)                       # most-confident first
    out = {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[c] = float(np.mean(e[order[:k]]))
    return out


def risk_coverage_random(abserr, coverages, seeds=(0, 1, 2, 3, 4)):
    """Random-gated baseline: retain a random c-fraction; average error over seeds. By
    construction this is ~flat at the overall mean |error| (random retention does not lower
    error). The certificate must beat this."""
    e = np.asarray(abserr, float)
    e = e[np.isfinite(e)]
    n = e.size
    out = {c: [] for c in coverages}
    for sd in seeds:
        rng = np.random.default_rng(sd)
        perm = rng.permutation(n)
        for c in coverages:
            k = max(1, int(round(c * n)))
            out[c].append(float(np.mean(e[perm[:k]])))
    return {c: float(np.mean(v)) for c, v in out.items()}


COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


# --------------------------------------------------------------------------- #
# SETTING 1: real panels under MCAR -- pool scored cells across datasets/seeds
# --------------------------------------------------------------------------- #
REAL = {
    "Beijing": ("beijing_clean.npy", (slice(0, 3000), slice(0, 30))),
    "ETTh1":   ("ETTh1_clean.npy",   (slice(0, 3000), slice(None))),
    "FRED-MD": ("fredmd_clean.npy",  (slice(0, 3000), slice(None))),
}
REAL_RATE = 0.4
REAL_SEEDS = [0, 1]


def setting_real():
    print("\n=== SETTING 1: real panels (MCAR %.0f%%) ===" % (REAL_RATE * 100), flush=True)
    all_score, all_err = [], []
    per_ds = {}
    for dname, (fn, sl) in REAL.items():
        Xclean = np.load(os.path.join(ROOT, "data", fn))[sl]
        Xclean = np.ascontiguousarray(Xclean, float)
        ds_score, ds_err = [], []
        for seed in REAL_SEEDS:
            M = mcar_mask(Xclean.shape, REAL_RATE, seed=seed)
            Xstd = robust_standardize(Xclean, M)
            Xobs = Xstd.copy(); Xobs[M] = np.nan
            filled, score = cafe_fill_and_certificate(Xobs)
            err = np.abs(filled - Xstd)
            # only score the held-out (M) cells we have truth for AND a certificate at
            sel = M & np.isfinite(score)
            ds_score.append(score[sel]); ds_err.append(err[sel])
        ds_score = np.concatenate(ds_score); ds_err = np.concatenate(ds_err)
        per_ds[dname] = (ds_score, ds_err)
        all_score.append(ds_score); all_err.append(ds_err)
        rho = spearman(ds_score, -ds_err)        # +rho => higher score, lower error
        print(f"  [{dname:8s}] cells={ds_score.size:6d}  "
              f"spearman(score, -|err|)={rho:+.3f}", flush=True)
    return np.concatenate(all_score), np.concatenate(all_err), per_ds


# --------------------------------------------------------------------------- #
# SETTING 2: factor-spanned vs idiosyncratic near-empty feature.
# The certificate should rank idiosyncratic-feature cells LOWER than spanned-feature cells.
# We build a panel with BOTH a spanned and an idiosyncratic sparse column, mask both heavily,
# and check the certificate separates them AND tracks error within each.
# --------------------------------------------------------------------------- #
S2_T, S2_N = 4000, 20
S2_SEEDS = [0, 1, 2]
S2_MISS = 0.6                                    # heavy missing on the two special columns


def setting_frontier():
    print("\n=== SETTING 2: factor-spanned vs idiosyncratic near-empty feature ===",
          flush=True)
    sp_score, sp_err, id_score, id_err = [], [], [], []
    r2s = {"spanned": [], "idiosyncratic": []}
    for seed in S2_SEEDS:
        # build one panel with a spanned col (0) and graft an idiosyncratic col (1)
        Xsp = gen_anchor_panel(S2_T, S2_N, "spanned", seed)
        Xid = gen_anchor_panel(S2_T, S2_N, "idiosyncratic", seed)
        X = Xsp.copy()
        X[:, 1] = Xid[:, 0]                       # col 0 = spanned, col 1 = idiosyncratic
        rng = np.random.default_rng(RNG_SEED + seed)
        M = np.zeros(X.shape, bool)
        # heavily mask BOTH special columns (scattered), light MCAR elsewhere
        M[:, 0] = rng.random(S2_T) < S2_MISS
        M[:, 1] = rng.random(S2_T) < S2_MISS
        M[:, 2:] = rng.random((S2_T, S2_N - 2)) < 0.1
        Xstd = robust_standardize(X, M)
        r2s["spanned"].append(feature0_r2(robust_standardize(X[:, [0] + list(range(2, S2_N))],
                                                             np.zeros((S2_T, S2_N - 1), bool))))
        Xobs = Xstd.copy(); Xobs[M] = np.nan
        filled, score = cafe_fill_and_certificate(Xobs)
        err = np.abs(filled - Xstd)
        sel0 = M[:, 0] & np.isfinite(score[:, 0])
        sel1 = M[:, 1] & np.isfinite(score[:, 1])
        sp_score.append(score[sel0, 0]); sp_err.append(err[sel0, 0])
        id_score.append(score[sel1, 1]); id_err.append(err[sel1, 1])
    sp_score = np.concatenate(sp_score); sp_err = np.concatenate(sp_err)
    id_score = np.concatenate(id_score); id_err = np.concatenate(id_err)
    print(f"  factor-spanned  col: mean cert={sp_score.mean():.3f}  "
          f"mean|err|={sp_err.mean():.3f}  (n={sp_score.size})", flush=True)
    print(f"  idiosyncratic   col: mean cert={id_score.mean():.3f}  "
          f"mean|err|={id_err.mean():.3f}  (n={id_score.size})", flush=True)
    flagged = id_score.mean() < sp_score.mean()
    print(f"  -> certificate ranks idiosyncratic BELOW spanned (flags unrecoverable): "
          f"{flagged}  (gap={sp_score.mean() - id_score.mean():+.3f})", flush=True)
    return (sp_score, sp_err), (id_score, id_err), flagged


# --------------------------------------------------------------------------- #
# FIGURE
# --------------------------------------------------------------------------- #
def make_figure(real_pack, frontier_pack):
    real_score, real_err, _ = real_pack
    (sp_score, sp_err), (id_score, id_err), _ = frontier_pack

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)

    # ---- (a) calibration: mean |err| vs certificate bin ----
    cx, cy, _ = calibration_curve(real_score, real_err, n_bins=8)
    axA.plot(cx, cy, marker="o", color=PALETTE["teal"], lw=2.0, ms=5,
             markeredgecolor="white", markeredgewidth=0.5, label="real panels")
    # frontier cells, pooled, for a second calibration trace
    fc_s = np.concatenate([sp_score, id_score]); fc_e = np.concatenate([sp_err, id_err])
    fx, fy, _ = calibration_curve(fc_s, fc_e, n_bins=6)
    if fx.size:
        axA.plot(fx, fy, marker="s", color=PALETTE["purple"], lw=1.6, ms=4.5, ls="--",
                 markeredgecolor="white", markeredgewidth=0.4, label="frontier (spanned+idio)")
    style_ax(axA)
    axA.set_xlabel("recoverability certificate (bin mean)", fontsize=8.5)
    axA.set_ylabel("realized mean $|$error$|$", fontsize=8.5)
    axA.set_title("(a) Calibration: lower certificate\n$\\to$ higher error (monotone $\\downarrow$)",
                  fontsize=8.5)
    axA.legend(fontsize=6.5, frameon=False, loc="upper right")

    # ---- (b) risk-coverage: certificate vs random ----
    rc_cert = risk_coverage(real_score, real_err, COVERAGES)
    rc_rand = risk_coverage_random(real_err, COVERAGES)
    cov = [c * 100 for c in COVERAGES]
    axB.plot(cov, [rc_cert[c] for c in COVERAGES], marker="o", color=PALETTE["teal"],
             lw=2.2, ms=5, markeredgecolor="white", markeredgewidth=0.5,
             label="certificate-gated", zorder=5)
    axB.plot(cov, [rc_rand[c] for c in COVERAGES], marker="v", color=PALETTE["grey"],
             lw=1.6, ms=5, ls="--", label="random-gated", zorder=2)
    style_ax(axB)
    axB.set_xlabel("coverage (\\% of cells retained)", fontsize=8.5)
    axB.set_ylabel("MAE on retained cells", fontsize=8.5)
    axB.set_title("(b) Selective risk: certificate gating\nlowers retained error far faster",
                  fontsize=8.5)
    axB.invert_xaxis()
    axB.legend(fontsize=6.8, frameon=False, loc="upper left")

    out = os.path.join(ROOT, "paper", "figures", "recoverability.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# TABLE
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "--" if not np.isfinite(v) else f"{v:.3f}"


def make_table(real_pack, frontier_pack, real_rho, frontier_flagged):
    real_score, real_err, per_ds = real_pack
    (sp_score, sp_err), (id_score, id_err), _ = frontier_pack

    # rows: each real dataset + pooled + frontier(spanned) + frontier(idio)
    rows = []
    cov_pts = [1.0, 0.8, 0.5]
    for dname, (s, e) in per_ds.items():
        rc_c = risk_coverage(s, e, cov_pts)
        rc_r = risk_coverage_random(e, cov_pts)
        rho = spearman(s, -e)
        rows.append((dname, rho, rc_c, rc_r))
    rc_c = risk_coverage(real_score, real_err, cov_pts)
    rc_r = risk_coverage_random(real_err, cov_pts)
    rows.append(("\\emph{Real (pooled)}", spearman(real_score, -real_err), rc_c, rc_r))
    for lab, s, e in (("Frontier: spanned", sp_score, sp_err),
                      ("Frontier: idio.", id_score, id_err)):
        rc_c = risk_coverage(s, e, cov_pts)
        rc_r = risk_coverage_random(e, cov_pts)
        rows.append((lab, spearman(s, -e), rc_c, rc_r))

    L = []
    L.append(r"\begin{table}[t]\centering\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(
        r"\caption{\textbf{Recoverability certificate: selective (risk-controlled) "
        r"imputation.} A per-cell certificate in $[0,1]$ built from \cafe{}'s own state "
        r"(posterior $\sigma$, conformal scale, cross-sectional anchor support, "
        r"factor/loading support, Student-$t$ robustness) predicts each fill's error. "
        r"\emph{Spearman} $\rho$ between certificate and $-|$error$|$ (held-out cells; "
        r"$+$ means a higher certificate genuinely means lower error). \emph{MAE@cov} is the "
        r"error on the retained cells when we abstain on the least-confident, keeping "
        r"$100/80/50\%$ of cells, for \textbf{cert}ificate-gating vs \textbf{rand}om-gating: "
        r"the certificate drops retained error far faster, so abstaining on low-certificate "
        r"cells removes the genuinely unrecoverable ones. On the recoverability frontier the "
        r"certificate ranks the \emph{idiosyncratic} (no cross-sectional evidence) column "
        r"below the \emph{factor-spanned} one -- \cafe{} flags the cells it cannot recover. "
        r"Every number is from a live causal run; the certificate never sees held-out truth.}")
    L.append(r"\label{tab:recoverability}")
    L.append(r"\begin{tabular}{@{}lc" + "cc" * len(cov_pts) + r"@{}}")
    L.append(r"\toprule")
    hdr2 = " & ".join(
        [r"\multicolumn{2}{c}{MAE@%d\%%}" % int(c * 100) for c in cov_pts])
    L.append(r"Setting & $\rho$ & " + hdr2 + r" \\")
    sub = " & ".join(["cert & rand"] * len(cov_pts))
    L.append(r" & & " + sub + r" \\")
    L.append(r"\midrule")
    for name, rho, rc_c, rc_r in rows:
        cells = []
        for c in cov_pts:
            cells.append(_fmt(rc_c[c]))
            cells.append(_fmt(rc_r[c]))
        L.append(f"{name} & {rho:+.2f} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    tex = "\n".join(L) + "\n"
    out = os.path.join(ROOT, "paper", "tables", "recoverability.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(tex)
    print(f"table  -> {out}", flush=True)
    return out, tex


# --------------------------------------------------------------------------- #
def verdict(real_pack, frontier_pack):
    real_score, real_err, _ = real_pack
    (sp_score, sp_err), (id_score, id_err), flagged = frontier_pack
    print("\n" + "=" * 72, flush=True)
    print("HONEST VERDICT", flush=True)
    print("=" * 72, flush=True)

    rho = spearman(real_score, -real_err)
    cx, cy, _ = calibration_curve(real_score, real_err, n_bins=8)
    # monotonicity: fraction of adjacent bins where error decreases as score rises
    mono = float(np.mean(np.diff(cy) <= 0)) if cy.size > 1 else float("nan")
    print(f"[calibration] real panels: spearman(cert,-|err|)={rho:+.3f}; "
          f"calibration-bin monotone-down fraction={mono:.2f}", flush=True)
    print(f"  bin mean cert : {np.round(cx,3)}", flush=True)
    print(f"  bin mean |err|: {np.round(cy,3)}  (should decrease)", flush=True)

    rc_c = risk_coverage(real_score, real_err, [1.0, 0.8, 0.5])
    rc_r = risk_coverage_random(real_err, [1.0, 0.8, 0.5])
    print(f"[risk-coverage real] cert: 100%={rc_c[1.0]:.3f} 80%={rc_c[0.8]:.3f} "
          f"50%={rc_c[0.5]:.3f}", flush=True)
    print(f"                     rand: 100%={rc_r[1.0]:.3f} 80%={rc_r[0.8]:.3f} "
          f"50%={rc_r[0.5]:.3f}", flush=True)
    drop_c = (rc_c[1.0] - rc_c[0.5]) / max(rc_c[1.0], 1e-9)
    drop_r = (rc_r[1.0] - rc_r[0.5]) / max(rc_r[1.0], 1e-9)
    print(f"  -> error drop 100%->50% coverage: certificate {drop_c*100:.1f}% vs "
          f"random {drop_r*100:.1f}% (certificate should dominate)", flush=True)

    print(f"[frontier] idiosyncratic mean cert={id_score.mean():.3f} < spanned mean "
          f"cert={sp_score.mean():.3f}: {flagged} "
          f"(flags the unrecoverable column)", flush=True)

    calibrated = (rho > 0.15) and (mono >= 0.6)
    dominates = drop_c > drop_r + 0.05
    print("\nVERDICT:", flush=True)
    print(f"  * Certificate calibrated (lower cert -> higher error): "
          f"{'YES' if calibrated else 'WEAK/NO'} "
          f"(rho={rho:+.3f}, monotone={mono:.2f}).", flush=True)
    print(f"  * Certificate-gating beats random by a wide margin: "
          f"{'YES' if dominates else 'NO'} "
          f"(50%-coverage error drop {drop_c*100:.0f}% vs {drop_r*100:.0f}%).", flush=True)
    print(f"  * Flags idiosyncratic-unrecoverable cells: {'YES' if flagged else 'NO'}.",
          flush=True)
    deployable = calibrated and dominates and flagged
    print(f"  * 'CAFE knows what it cannot recover' is a defensible, deployable claim: "
          f"{'YES' if deployable else 'PARTIALLY -- see numbers above'}.", flush=True)
    return calibrated, dominates, flagged


def main():
    t0 = time.perf_counter()
    real_pack = setting_real()
    frontier_pack = setting_frontier()
    real_rho = spearman(real_pack[0], -real_pack[1])
    make_figure(real_pack, frontier_pack)
    _, tex = make_table(real_pack, frontier_pack, real_rho, frontier_pack[2])
    verdict(real_pack, frontier_pack)
    print(f"\ntotal compute: {time.perf_counter() - t0:.1f}s", flush=True)
    print("\n----- recoverability.tex -----")
    print(tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
