"""Money-shot figures for the world's first CAUSAL imputation horse race.

These read a JSON cache produced by ``bench/exp_horserace.py`` (schema below) and
render the paper's headline "leaderboard flip": under the standard BIDIRECTIONAL
(smoothing) protocol deep/classical models look strong, but under the strict
CAUSAL (point-in-time) protocol they collapse -- they were silently borrowing
accuracy from the future -- while CAFE, causal by construction, rises to #1.

Cache schema (``bench/horserace_cache.json``)::

    {
      "meta": {"datasets": [...], "window": 24, "pypots_version": "...", ...},
      "results": [
        {"dataset": "fredmd", "method": "CAFE", "family": "causal",
         "bidir_mae": 0.41, "causal_mae": 0.41, "delta": 0.0,
         "causal_verified": true},
        ...
      ]
    }

``delta = causal_mae - bidir_mae`` is the look-ahead dependence (CAFE = 0 by
construction). ``family in {causal, deep, classical}``.

Outputs (paper/figures/):
  * causal_race_flip.pdf  -- slope/bump chart: rank-by-bidir -> rank-by-causal.
  * causal_race_gap.pdf   -- look-ahead gap (delta) bar chart, sorted.
  * causal_race_panels.pdf -- per-dataset bidir-vs-causal grouped bars (optional).

Everything degrades gracefully: missing fields, NaNs, variable method sets, and
methods that lack causal numbers are all handled without crashing.
"""
from __future__ import annotations

import os
import sys
import json
import math
import tempfile

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_common import PALETTE, style_ax

# ---------------------------------------------------------------------------- #
# Paths / constants
# ---------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "paper", "figures"))
DEFAULT_CACHE = os.path.join(HERE, "horserace_cache.json")

# Family -> colour. CAFE (the causal family) is the highlighted teal; deep models
# are amber (the ones that collapse); classical baselines are a muted slate.
FAMILY_COLOR = {
    "cafe": PALETTE["teal"],       # CAFE itself
    "causal": PALETTE["teal"],     # causal-native baselines (also Delta=0)
    "online": PALETTE["teal"],     # online/causal-native (also Delta=0)
    "deep": PALETTE["amber"],      # the deep models that collapse
    "classical": PALETTE["slate"], # batch classical (no causal variant)
}
DEFAULT_COLOR = PALETTE["grey"]
CAFE_NAMES = {"cafe", "café", "cafe (ours)", "café (ours)"}

# Families that are causal-native: their causal MAE EQUALS their bidir MAE
# (Delta = 0). On the value axis these render as perfectly horizontal lines.
FLAT_FAMILIES = {"cafe", "causal", "online"}


def _is_cafe(method: str) -> bool:
    return str(method).strip().lower() in CAFE_NAMES


def _fam_color(family: str) -> str:
    return FAMILY_COLOR.get(str(family).strip().lower(), DEFAULT_COLOR)


def _num(x):
    """Coerce to float, mapping None/missing/non-finite to NaN."""
    if x is None:
        return float("nan")
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


# ---------------------------------------------------------------------------- #
# Loading / aggregation
# ---------------------------------------------------------------------------- #
def load_cache(path: str = DEFAULT_CACHE) -> dict:
    """Read the JSON cache. Returns ``{"meta": {...}, "results": [...]}``.

    Robust to a missing file (raises a clear error) and to absent keys."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"cache root must be an object, got {type(raw)}")
    meta = raw.get("meta", {}) or {}
    results = raw.get("results", []) or []
    clean = []
    for r in results:
        if not isinstance(r, dict):
            continue
        method = r.get("method")
        if method is None:
            continue
        clean.append({
            "dataset": r.get("dataset", "?"),
            "method": str(method),
            "family": r.get("family", "classical"),
            "bidir_mae": _num(r.get("bidir_mae")),
            "causal_mae": _num(r.get("causal_mae")),
            "delta": _num(r.get("delta")),
            "causal_verified": bool(r.get("causal_verified", False)),
        })
    return {"meta": meta, "results": clean}


def aggregate(results: list) -> list:
    """Mean over datasets per method -> one row per method.

    Returns a list of dicts with keys: method, family, bidir_mae, causal_mae,
    delta, n, causal_verified (all-true across datasets). NaNs are ignored in the
    means (``nanmean``); a method with no finite values in a column yields NaN
    there so downstream plotting can skip it."""
    by_method: dict[str, dict] = {}
    for r in results:
        m = r["method"]
        slot = by_method.setdefault(m, {
            "method": m, "family": r["family"],
            "bidir": [], "causal": [], "delta": [], "verified": [],
        })
        slot["bidir"].append(r["bidir_mae"])
        slot["causal"].append(r["causal_mae"])
        slot["delta"].append(r["delta"])
        slot["verified"].append(r["causal_verified"])
        # keep the first non-trivial family label seen
        if slot["family"] in (None, "", "classical") and r["family"]:
            slot["family"] = r["family"]

    out = []
    for m, slot in by_method.items():
        bidir = np.asarray(slot["bidir"], float)
        causal = np.asarray(slot["causal"], float)
        delta = np.asarray(slot["delta"], float)
        # delta defaults to causal-bidir per row where it was NaN
        for i in range(len(delta)):
            if math.isnan(delta[i]):
                delta[i] = causal[i] - bidir[i]
        out.append({
            "method": m,
            "family": slot["family"],
            "bidir_mae": _safe_nanmean(bidir),
            "causal_mae": _safe_nanmean(causal),
            "delta": _safe_nanmean(delta),
            "n": int(np.isfinite(bidir).sum() | np.isfinite(causal).sum()
                     if False else len(bidir)),
            "causal_verified": all(bool(v) for v in slot["verified"])
            if slot["verified"] else False,
        })
    return out


def _safe_nanmean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _rank(vals: dict) -> dict:
    """Rank methods 1..k by ascending MAE (1 = best). Methods with NaN are
    dropped from the ranking (returned only for finite entries)."""
    finite = {m: v for m, v in vals.items() if math.isfinite(v)}
    order = sorted(finite, key=lambda m: finite[m])
    return {m: i + 1 for i, m in enumerate(order)}


# ---------------------------------------------------------------------------- #
# Figure 1: the leaderboard flip (slope / bump chart)
# ---------------------------------------------------------------------------- #
def _is_flat_family(family: str) -> bool:
    """True for causal-native families whose causal MAE == bidir MAE (Delta=0)."""
    return str(family).strip().lower() in FLAT_FAMILIES


def fig_flip(agg: list, out_path: str, meta: dict | None = None) -> str:
    """Value-anchored slope chart: y = MAE on a *shared* scale for both protocols.

    Left column = methods placed by their BIDIRECTIONAL MAE; right column = placed
    by their CAUSAL MAE. Because the y-axis is the actual MAE value (not an integer
    rank), a method whose score is unchanged by the protocol (Delta = causal-bidir
    = 0; every causal-native / CAFE / online method) draws as a PERFECTLY
    HORIZONTAL line. Deep models slope upward (their MAE worsens left->right) and
    visibly cross above CAFE -- the "flip" reads as honest line crossings. The
    crowded 1..24 integer-rank axis disappears; small rank numbers survive as
    endpoint annotations. Lower MAE = better, so the axis is inverted (best on top).

    Methods lacking a causal number (batch classical: TRMF/LinearInterp/SoftImpute)
    enter from the left and stub off with a "no causal variant" note. CAFE stays
    bold teal and on top either way."""
    bidir = {r["method"]: r["bidir_mae"] for r in agg}
    causal = {r["method"]: r["causal_mae"] for r in agg}
    fam = {r["method"]: r["family"] for r in agg}

    rb = _rank(bidir)                       # 1 = best bidir MAE
    rc = _rank(causal)                      # 1 = best causal MAE
    methods = [m for m in bidir if m in rb]  # need at least a bidir number
    if not methods:
        raise ValueError("no method has a bidirectional MAE to plot")

    # ---- shared value axis ------------------------------------------------- #
    # Cap the visible range so a single blow-up method (e.g. Drift ~2.8) does not
    # crush everyone into the bottom strip. Points above the cap are clipped to a
    # dashed exit line + annotation.
    finite_vals = [v for v in list(bidir.values()) + list(causal.values())
                   if math.isfinite(v)]
    vlo = min(finite_vals)
    # robust cap: keep the dense band, clip the long tail
    body = sorted(v for v in finite_vals if math.isfinite(v))
    q_hi = body[int(0.92 * (len(body) - 1))] if len(body) > 1 else body[0]
    vcap = max(q_hi * 1.06, vlo + 0.15)
    has_clip = any(v > vcap for v in finite_vals)
    pad = 0.05 * (vcap - vlo)
    ytop, ybot = vlo - pad, vcap + pad      # data coords (pre-inversion)

    def _y(v):
        """Map a value to plotted y, clipping above the cap."""
        return min(v, vcap)

    x_left, x_right = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)

    # Collect label requests per side so we can de-collide them vertically
    # (many methods tie within rounding, e.g. 0.545, and would overprint).
    left_lbls, right_lbls = [], []   # each: [y, text, color, fontsize, bold]

    for m in methods:
        cafe = _is_cafe(m)
        col = PALETTE["teal"] if cafe else _fam_color(fam[m])
        flat = _is_flat_family(fam[m]) or cafe
        b, c = bidir[m], causal[m]
        has_causal = math.isfinite(c)
        y0 = _y(b)

        lw = 3.4 if cafe else (1.8 if has_causal and not flat else 1.5)
        alpha = 0.97 if cafe else (0.85 if not flat else 0.55)
        zbase = 6 if cafe else (4 if not flat else 3)

        if has_causal:
            y1 = _y(c)
            ax.plot([x_left, x_right], [y0, y1], color=col, lw=lw, alpha=alpha,
                    solid_capstyle="round", zorder=zbase)
            ax.scatter([x_left, x_right], [y0, y1], s=52 if cafe else 24,
                       color=col, zorder=zbase + 2, edgecolor="white",
                       linewidth=0.7)
        else:
            # no causal variant: short dashed stub leaving the left column
            ax.plot([x_left, x_left + 0.18], [y0, y0], color=col, lw=lw,
                    alpha=0.8, solid_capstyle="round", zorder=zbase)
            ax.scatter([x_left], [y0], s=24, color=col, zorder=zbase + 2,
                       edgecolor="white", linewidth=0.7)
            ax.plot([x_left + 0.18, x_left + 0.34], [y0, y0], color=col,
                    lw=lw, alpha=0.8, ls=(0, (2, 2)), zorder=zbase)

        fs = 8.4 if cafe else 7.3
        rl = rb.get(m)
        left_lbls.append([y0, (f"{m}  ·{rl}" if rl else m), col, fs, cafe])
        if has_causal:
            rr = rc.get(m)
            right_lbls.append([_y(c), (f"·{rr}  {m}" if rr else m),
                               col, fs, cafe])
        else:
            # stub annotation rides with the left side but offset to the right
            ax.text(x_left + 0.36, y0, "no causal variant", ha="left",
                    va="center", fontsize=6.6, color=col, style="italic",
                    alpha=0.9, zorder=8)

    # ---- de-collide labels vertically (greedy, in data coords) ------------- #
    span = abs(ybot - ytop)
    min_gap = 0.020 * span               # minimum vertical separation

    def _place(lbls, x, ha):
        # sort by y ascending in *data* coords; ytop<ybot here may be inverted,
        # so sort by raw value then enforce spacing in display order.
        order = sorted(range(len(lbls)), key=lambda i: lbls[i][0])
        ys = [lbls[i][0] for i in order]
        # push apart so neighbours differ by >= min_gap
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < min_gap:
                ys[k] = ys[k - 1] + min_gap
        # gentle back-pass to keep the cluster near its origin (cosmetic)
        for k in range(len(ys) - 2, -1, -1):
            if ys[k + 1] - ys[k] < min_gap:
                ys[k] = ys[k + 1] - min_gap
        for slot, k in enumerate(order):
            y, text, col, fs, bold = lbls[k]
            ax.text(x, ys[slot], text, ha=ha, va="center", fontsize=fs,
                    color=col, fontweight="bold" if bold else "normal",
                    zorder=8)

    _place(left_lbls, x_left - 0.025, "right")
    _place(right_lbls, x_right + 0.025, "left")

    # ---- axes -------------------------------------------------------------- #
    ax.set_ylim(ybot, ytop)                 # inverted: best (low MAE) on top
    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["BIDIRECTIONAL\n(uses the future)",
                        "CAUSAL\n(point-in-time)"], fontsize=11)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel("imputation error  (mean absolute error · lower is better)",
                  fontsize=10)
    ax.tick_params(axis="y", labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.grid(True, axis="y", alpha=0.13, lw=0.6)

    # mark the clip cap so a clipped (e.g. Drift) line reads as "off the chart"
    if has_clip:
        ax.axhline(vcap, color=PALETTE["grey"], lw=0.8, ls=(0, (2, 3)),
                   alpha=0.6, zorder=1)
        ax.text(0.5, vcap, "worse-scoring methods clipped above this line",
                ha="center", va="bottom", fontsize=6.6, color=PALETTE["grey"],
                style="italic", zorder=8)

    # subtle column guide lines
    for xc in (x_left, x_right):
        ax.axvline(xc, color=PALETTE["grey"], lw=0.6, alpha=0.18, zorder=0)

    ax.set_title("Forbid the future and the board collapses:  "
                 "deep & interpolation methods fall, CAFÉ (causal) leads either way",
                 fontsize=12, fontweight="bold", pad=30)

    # ---- legend: horizontal, under the title, clear of the x tick labels --- #
    present = []
    for famkey, lab in (("deep", "deep models (PyPOTS) — borrow the future"),
                        ("classical", "batch classical — no causal variant"),
                        ("causal", "CAFÉ & causal-native — score unchanged (flat)")):
        if any(str(r["family"]).strip().lower() in (
                ({"causal", "cafe", "online"} if famkey == "causal" else {famkey}))
               for r in agg):
            present.append(plt.Line2D([0], [0], color=FAMILY_COLOR[famkey],
                                      lw=3.0, label=lab))
    if present:
        ax.legend(handles=present, fontsize=8.6, loc="lower center",
                  ncol=len(present), frameon=False,
                  bbox_to_anchor=(0.5, 1.005), handlelength=1.6,
                  columnspacing=1.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------- #
# Figure 2: the look-ahead gap (delta) bar chart
# ---------------------------------------------------------------------------- #
def fig_gap(agg: list, out_path: str, meta: dict | None = None) -> str:
    """Horizontal bar chart of the look-ahead gap ``delta`` per method, sorted
    largest-first. CAFE sits at ~0 and is highlighted; annotate the borrowed
    accuracy."""
    rows = [r for r in agg if math.isfinite(r["delta"])]
    if not rows:
        raise ValueError("no finite delta values to plot")
    rows = sorted(rows, key=lambda r: r["delta"])  # smallest at top after invert

    labels = [r["method"] for r in rows]
    vals = [r["delta"] for r in rows]
    cols = [PALETTE["teal"] if _is_cafe(r["method"]) else _fam_color(r["family"])
            for r in rows]
    ypos = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(4.4, 2.8), constrained_layout=True)
    ax.barh(ypos, vals, color=cols, height=0.68, edgecolor="white",
            linewidth=0.4, zorder=3)
    ax.axvline(0.0, color=PALETTE["ink"], lw=0.8, zorder=2)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if _is_cafe(r["method"]):
            tick.set_color(PALETTE["teal"])
            tick.set_fontweight("bold")

    vmax = max(vals + [0.0])
    vmin = min(vals + [0.0])
    span = (vmax - vmin) or 1.0
    ax.set_xlim(min(vmin, 0) - 0.04 * span, vmax + 0.20 * span)
    for y, v, r in zip(ypos, vals, rows):
        cafe = _is_cafe(r["method"])
        ax.text(v + 0.012 * span if v >= 0 else v - 0.012 * span, y,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.4,
                color=PALETTE["teal"] if cafe else PALETTE["ink"],
                fontweight="bold" if cafe else "normal", zorder=4)

    ax.set_xlabel("look-ahead gap  $\\Delta$ = causal MAE $-$ bidir MAE",
                  fontsize=8.5)
    ax.set_title("Accuracy borrowed from the future\n"
                 "(CAFÉ borrows none: $\\Delta = 0$ by construction)",
                 fontsize=9.2, fontweight="bold", pad=6)

    # callout on the worst offender
    worst = max(rows, key=lambda r: r["delta"])
    wy = labels.index(worst["method"])
    if worst["delta"] > 0:
        ax.annotate("accuracy borrowed\nfrom the future",
                    xy=(worst["delta"], wy),
                    xytext=(worst["delta"] - 0.34 * span, wy + 0.9),
                    fontsize=7.0, color=PALETTE["amber"], ha="center",
                    va="center",
                    arrowprops=dict(arrowstyle="->", color=PALETTE["amber"],
                                    lw=0.8, connectionstyle="arc3,rad=0.2"))

    style_ax(ax)
    ax.grid(True, axis="x", alpha=0.18, lw=0.6)
    ax.grid(False, axis="y")
    ax.set_ylim(-0.6, len(rows) - 0.4)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------- #
# Figure 3 (optional): per-dataset small multiples
# ---------------------------------------------------------------------------- #
def fig_panels(results: list, out_path: str, meta: dict | None = None) -> str:
    """Small multiples: one panel per dataset, grouped bars bidir vs causal MAE
    per method. Returns the path, or "" if there is nothing to plot."""
    datasets = []
    for r in results:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
    if not datasets:
        return ""

    nd = len(datasets)
    ncol = min(3, nd)
    nrow = math.ceil(nd / ncol)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.0 * ncol, 2.4 * nrow),
                             constrained_layout=True, squeeze=False)

    handles = None
    for k, ds in enumerate(datasets):
        ax = axes[k // ncol][k % ncol]
        rows = [r for r in results if r["dataset"] == ds]
        # order methods by bidir MAE within the dataset
        rows = sorted(rows, key=lambda r: (not math.isfinite(r["bidir_mae"]),
                                           r["bidir_mae"]))
        labels = [r["method"] for r in rows]
        bidir = [r["bidir_mae"] for r in rows]
        causal = [r["causal_mae"] for r in rows]
        x = np.arange(len(labels))
        w = 0.38
        bb = ax.bar(x - w / 2, np.nan_to_num(bidir, nan=0.0), w,
                    color=PALETTE["grey"], edgecolor="white", linewidth=0.3,
                    label="bidirectional", zorder=3)
        bc = ax.bar(x + w / 2, np.nan_to_num(causal, nan=0.0), w,
                    color=PALETTE["amber"], edgecolor="white", linewidth=0.3,
                    label="causal", zorder=3)
        # recolour CAFE's causal bar to teal so it pops
        for rect, r in zip(bc, rows):
            if _is_cafe(r["method"]):
                rect.set_color(PALETTE["teal"])
        handles = [bb, bc]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.6, rotation=35, ha="right")
        ax.set_title(str(ds), fontsize=8.5)
        ax.set_ylabel("MAE", fontsize=7.5)
        style_ax(ax)
        ax.grid(True, axis="y", alpha=0.18, lw=0.6)
        ax.grid(False, axis="x")

    # blank any unused panels
    for k in range(nd, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    if handles is not None:
        fig.legend(handles=handles, fontsize=7.6, loc="upper right",
                   frameon=False, ncol=2)
    fig.suptitle("Per-dataset MAE: bidirectional vs. causal protocol",
                 fontsize=10, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------------- #
def make_all(cache_path: str = DEFAULT_CACHE, fig_dir: str = FIG_DIR) -> list:
    """Build all figures from a cache. Returns the list of paths written."""
    cache = load_cache(cache_path)
    meta, results = cache["meta"], cache["results"]
    if not results:
        raise ValueError("cache has no usable results")
    agg = aggregate(results)

    written = []
    written.append(fig_flip(agg, os.path.join(fig_dir, "causal_race_flip.pdf"),
                            meta))
    written.append(fig_gap(agg, os.path.join(fig_dir, "causal_race_gap.pdf"),
                           meta))
    p = fig_panels(results, os.path.join(fig_dir, "causal_race_panels.pdf"),
                   meta)
    if p:
        written.append(p)
    return written


# ---------------------------------------------------------------------------- #
# Self-test: build a MOCK cache, render against it, assert PDFs are non-trivial.
# (The real cache from exp_horserace.py does not exist yet.)
# ---------------------------------------------------------------------------- #
def _mock_cache() -> dict:
    """A few datasets x methods with realistic numbers: deep models win the bidir
    race but collapse under causal; CAFE is flat and ends up #1."""
    datasets = ["fredmd", "beijing_air", "electricity"]
    # (method, family, bidir base, causal base)
    base = [
        ("CAFE",       "causal",    0.41, 0.41),  # delta 0
        ("SAITS",      "deep",      0.33, 0.71),
        ("BRITS",      "deep",      0.36, 0.66),
        ("Transformer","deep",      0.35, 0.69),
        ("SoftImpute", "classical", 0.55, 0.95),
        ("TRMF",       "classical", 0.49, 0.62),
    ]
    rng = np.random.default_rng(0)
    results = []
    for ds in datasets:
        for name, fam, bb, cc in base:
            jb = bb + rng.normal(0, 0.02)
            jc = (jb if name == "CAFE" else cc + rng.normal(0, 0.03))
            results.append({
                "dataset": ds, "method": name, "family": fam,
                "bidir_mae": round(float(jb), 3),
                "causal_mae": round(float(jc), 3),
                "delta": round(float(jc - jb), 3),
                "causal_verified": name == "CAFE" or fam == "deep",
            })
    # inject robustness stressors: a NaN, a missing field, an unknown family
    results.append({"dataset": "fredmd", "method": "GhostNet",
                    "family": "mystery", "bidir_mae": float("nan"),
                    "causal_mae": 0.8})  # no delta / no verified
    return {
        "meta": {"datasets": datasets, "window": 24,
                 "pypots_version": "1.5", "torch_version": "2.12.1",
                 "cmd": "python bench/exp_horserace.py"},
        "results": results,
    }


def _selftest() -> None:
    tmp = tempfile.mkdtemp(prefix="horserace_")
    cache_path = os.path.join(tmp, "horserace_cache.json")
    with open(cache_path, "w") as fh:
        json.dump(_mock_cache(), fh)
    out_dir = os.path.join(tmp, "figures")
    paths = make_all(cache_path, out_dir)
    print(f"mock cache: {cache_path}")
    for p in paths:
        sz = os.path.getsize(p)
        print(f"  rendered {os.path.basename(p)}  ({sz} bytes)")
        assert sz > 2000, f"PDF too small / empty: {p}"
    assert len(paths) >= 2, "expected at least flip + gap figures"
    print("fig_causal_race self-test PASSED")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help="path to horserace JSON cache")
    ap.add_argument("--figdir", default=FIG_DIR, help="output figure directory")
    ap.add_argument("--selftest", action="store_true",
                    help="render against an internal mock cache (no real data)")
    args = ap.parse_args()

    if args.selftest or not os.path.exists(args.cache):
        if not args.selftest:
            print(f"[cache not found: {args.cache}] running self-test on mock data")
        _selftest()
    else:
        for p in make_all(args.cache, args.figdir):
            print("saved", p, os.path.getsize(p))
