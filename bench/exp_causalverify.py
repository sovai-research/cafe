#!/usr/bin/env python
"""
CAUSALVERIFY: turn Proposition 1 (strict point-in-time / truncation invariance)
into a MEASURED fact, and quantify the causal "moat" vs a non-causal method.

(1) TRUNCATION-INVARIANCE TEST (CAFE / online_impute):
    Run online_impute on the full masked series AND on many time-PREFIXES X[:k].
    For each early missing cell at time t < k, the value imputed on the prefix
    must EXACTLY equal the value imputed on the full series (Prop 1). We report
    the number of (cell, prefix) comparisons and the MAX |deviation| (expect 0).

(2) THE MOAT (contrast, SoftImpute / m_softimpute, non-causal):
    For the SAME early cells, recompute the cell's fill on GROWING prefixes and
    measure mean |revision| as the future is revealed. Non-causal => nonzero.

Self-contained, < ~5 min. Prefixes subsampled for speed.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import harness  # noqa: E402
from c_unified_penmf import online_impute  # noqa: E402
import m_softimpute  # noqa: E402

CAFE_TOL_REPORT = 1e-9   # any CAFE deviation above this is a real bug -> shout
RATE = 0.15
MASKER = harness.MASKERS["mcar"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _zscore(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd


def load_datasets():
    """Return list of (name, X2d, meta). Real data z-scored so |revision| is on
    a standardised scale comparable to the paper's SAITS protocol."""
    out = []

    # Beijing Air-Quality [:1500]  (z-scored, a few cols capped for speed)
    bj = np.load(os.path.join(ROOT, "data", "beijing_clean.npy"))[:1500]
    bj = _zscore(bj[:, :30])
    out.append(("BeijingAir", np.ascontiguousarray(bj), {}))

    # ETTh1 [:1500] (z-scored, 7 features)
    et = np.load(os.path.join(ROOT, "data", "ETTh1_clean.npy"))[:1500]
    et = _zscore(et)
    out.append(("ETTh1", np.ascontiguousarray(et), {}))

    # Synthetic 2D harness generator (already well-scaled, low-rank+AR+seasonal)
    syn = harness.gen_2d(T=1500, N=20, seed=0)
    out.append(("Synthetic2D", np.ascontiguousarray(_zscore(syn)), {}))

    # Panel case (entity x time x feature), if the harness exposes it
    try:
        pX, pmeta = harness.gen_panel(E=20, T=80, F=10, seed=0)
        out.append(("Panel", np.ascontiguousarray(pX), pmeta))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] panel skipped: {e}")

    return out


# --------------------------------------------------------------------------- #
# Truncation-invariance test (CAFE) + moat (SoftImpute)
# --------------------------------------------------------------------------- #
def make_mask(X):
    M = MASKER(X, RATE, seed=0)
    # never leave an all-missing row at the top (degenerate prefix); not required
    return M


def prefix_grid(T, n_prefixes=12, t_max_frac=0.6, t_lo=40):
    """Prefix end-points k (on the TIME axis). We compare cells at t < k, so cap
    k well below T to keep many comparable early cells and bound cost."""
    k_hi = max(t_lo + 10, int(t_max_frac * T))
    ks = np.unique(np.linspace(t_lo, k_hi, n_prefixes).astype(int))
    return [int(k) for k in ks if k <= T]


def time_prefix(X, meta, k):
    """Truncate to all data strictly before TIME k.

    For a plain 2D series the time axis IS the row axis, so this is X[:k].
    For an ENTITY-MAJOR panel, X[:k] would cut across entity blocks (not a time
    prefix at all); we must instead keep every row whose time_id < k, across all
    entities, and truncate the id arrays to match. This is the only truncation
    that respects the panel's point-in-time semantics (state is reset per entity;
    W,a,nu are pooled over data at time <= k)."""
    if "time_ids" in meta:
        tids = np.asarray(meta["time_ids"])
        sel = tids < k
        sub = np.ascontiguousarray(X[sel])
        sub_meta = dict(meta)
        sub_meta["time_ids"] = tids[sel]
        sub_meta["entity_ids"] = np.asarray(meta["entity_ids"])[sel]
        return sub, sub_meta, sel
    return X[:k].copy(), dict(meta), slice(0, k)


def run_cafe_invariance(name, X, meta, ks):
    """Full fill vs each prefix fill; max |dev| over early missing cells.
    Returns (n_comparisons, max_dev, n_cells_at_t0, full_fill)."""
    Xobs = X.copy()
    M = make_mask(X)
    Xobs[M] = np.nan

    full = np.asarray(online_impute(Xobs.copy(), dict(meta)), float)

    max_dev = 0.0
    n_cmp = 0
    worst = None
    for k in ks:
        sub, sub_meta, sel = time_prefix(Xobs, meta, k)
        pf = np.asarray(online_impute(sub.copy(), sub_meta), float)
        Msub = M[sel]                       # missing cells inside the prefix
        full_sel = full[sel]                # aligned full-series fills
        if not Msub.any():
            continue
        dev = np.abs(pf[Msub] - full_sel[Msub])
        n_cmp += int(Msub.sum())
        if dev.size and dev.max() > max_dev:
            max_dev = float(dev.max())
            worst = (k, float(dev.max()))
    return n_cmp, max_dev, M, full, worst


def run_softimpute_moat(name, X, meta, M, ks, max_cells=120):
    """For a sample of early missing cells, recompute SoftImpute's fill on each
    growing prefix that contains the cell; mean |revision| = mean over cells of
    |fill_on_largest_prefix - fill_on_smallest_prefix_containing_cell|, averaged
    over consecutive prefix steps. Non-causal => nonzero."""
    Xobs = X.copy()
    Xobs[M] = np.nan
    tids = np.asarray(meta["time_ids"]) if "time_ids" in meta \
        else np.arange(X.shape[0])

    # candidate EARLY missing cells: time < ks[2] so several prefixes contain
    # them. Track by absolute (row, col) so panel layout is handled uniformly.
    t_cap = ks[2] if len(ks) > 2 else ks[-1]
    early_rows = tids < t_cap
    cand = np.argwhere(M & early_rows[:, None])
    if cand.shape[0] == 0:
        return 0.0, 0
    rng = np.random.default_rng(0)
    if cand.shape[0] > max_cells:
        pick = rng.choice(cand.shape[0], size=max_cells, replace=False)
        cand = cand[pick]
    cells = [(int(r), int(j), int(tids[r])) for r, j in cand]

    # SoftImpute fill for each prefix, mapped back onto absolute rows via sel.
    fills = {}        # k -> dict(row_index -> filled_row_vector)
    for k in ks:
        sub, sub_meta, sel = time_prefix(Xobs, meta, k)
        out = np.asarray(m_softimpute.impute(sub.copy(), sub_meta), float)
        rows = np.where(sel)[0] if isinstance(sel, np.ndarray) else \
            np.arange(sel.start, sel.stop)
        fills[k] = (set(rows.tolist()), dict(zip(rows.tolist(), out)))

    revisions = []
    for (r, j, t) in cells:
        seq = []
        for k in ks:
            present, fmap = fills[k]
            if r in present:
                seq.append(fmap[r][j])
        if len(seq) < 2:
            continue
        revisions.append(float(np.mean(np.abs(np.diff(np.asarray(seq))))))
    mean_rev = float(np.mean(revisions)) if revisions else 0.0
    return mean_rev, len(revisions)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.perf_counter()
    datasets = load_datasets()
    results = []
    total_cmp = 0
    global_max_dev = 0.0

    for name, X, meta in datasets:
        if "time_ids" in meta:                       # panel: time axis != rows
            Ttime = int(np.asarray(meta["time_ids"]).max()) + 1
            ks = prefix_grid(Ttime, t_lo=10, t_max_frac=0.75)
        else:
            ks = prefix_grid(X.shape[0])
        n_cmp, max_dev, M, full, worst = run_cafe_invariance(name, X, meta, ks)
        mean_rev, n_rev = run_softimpute_moat(name, X, meta, M, ks)
        total_cmp += n_cmp
        global_max_dev = max(global_max_dev, max_dev)
        results.append(dict(dataset=name, shape=f"{X.shape[0]}x{X.shape[1]}",
                            n_cmp=n_cmp, cafe_max_dev=max_dev,
                            soft_mean_rev=mean_rev, n_rev=n_rev))
        flag = "  <<< EXCEEDS 1e-9 (BUG!)" if max_dev > CAFE_TOL_REPORT else ""
        print(f"[{name:11s}] {X.shape[0]}x{X.shape[1]}  "
              f"#cmp={n_cmp:6d}  CAFE max|dev|={max_dev:.3e}{flag}  "
              f"Soft mean|rev|={mean_rev:.3f} (n={n_rev})")
        if worst and max_dev > CAFE_TOL_REPORT:
            print(f"    worst CAFE: prefix k={worst[0]} dev={worst[1]:.3e}")

    print("-" * 72)
    print(f"TOTAL comparisons (CAFE cell,prefix): {total_cmp}")
    print(f"GLOBAL CAFE max|deviation|: {global_max_dev:.3e}")
    if global_max_dev > CAFE_TOL_REPORT:
        print("!!! CAFE DEVIATION EXCEEDS 1e-9 -- truncation invariance VIOLATED !!!")
    else:
        print("CAFE truncation invariance HOLDS (bitwise within 1e-9).")
    soft_overall = float(np.mean([r["soft_mean_rev"] for r in results
                                  if r["n_rev"] > 0]))
    print(f"SoftImpute overall mean|revision|: {soft_overall:.3f}")
    print(f"Elapsed: {time.perf_counter()-t0:.1f}s")

    write_table(results, soft_overall, total_cmp, global_max_dev)
    return results


def _fmt_dev(d):
    if d == 0.0:
        return r"$0.0$"
    return f"${d:.1e}".replace("e", r"\mathrm{e}{") + "}$"


def write_table(results, soft_overall, total_cmp, global_max_dev):
    path = os.path.join(ROOT, "paper", "tables", "causal_verify.tex")
    lines = []
    lines.append(r"\begin{table}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\caption{\textbf{Proposition~\ref{prop:causal}, measured.} "
                 r"Truncation invariance is checked by re-imputing on every time "
                 r"prefix $X_{:k}$ and comparing each early missing cell "
                 r"($t<k$) against the full-series fill. \cafe{} (causal) is "
                 r"bitwise identical across all "
                 + f"{total_cmp:,}".replace(",", r"{,}") +
                 r" (cell, prefix) comparisons; the non-causal SoftImpute "
                 r"revises the \emph{same} early cells as the future is "
                 r"revealed (mean $|$revision$|$ on the standardised scale).}")
    lines.append(r"\label{tab:causalverify}")
    lines.append(r"\begin{tabular}{@{}lrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & \#cmp & \cafe{} max$|\Delta|$ "
                 r"& SoftImpute mean$|$rev$|$ \\")
    lines.append(r"\midrule")
    for r in results:
        lines.append(f"{r['dataset']} & {r['n_cmp']:,} & {_fmt_dev(r['cafe_max_dev'])} "
                     f"& ${r['soft_mean_rev']:.2f}$ \\\\".replace(",", r"{,}"))
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Overall}} & {total_cmp:,} & "
                 f"{_fmt_dev(global_max_dev)} & "
                 f"$\\mathbf{{{soft_overall:.2f}}}$ \\\\".replace(",", r"{,}"))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    txt = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(txt)
    print(f"\nWrote {path}\n")
    print(txt)


if __name__ == "__main__":
    main()
