"""Public causal-discovery benchmarks WITH GROUND-TRUTH GRAPHS.

Companion to ``bench/causal_baselines.py``.  Follows the provenance convention
of ``bench/datasets_real.py`` / ``paper/datasets_real.md``: every loader states
its source, licence, on-disk size and the exact preprocessing it applies, and
everything downloaded is cached under ``data/causal/`` (git-ignored).

Every loader returns ``(X, A_true, meta)``:

    X       (T, P) float64, row axis = TIME, NaNs preserved where the source
            has genuine gaps (each loader documents what it fills).
    A_true  (P, P) float64/bool in THIS REPO'S CONVENTION:
                A_true[i, j] != 0  <=>  "j causes i"
            (effect row, cause column -- the same convention as
            causal_baselines.BaselineResult.A_contemp / .A_lag[k]).
    meta    dict: at least 'name', 'source', 'licence', 'nodes', 'self_loops'
            (whether the benchmark's own protocol SCORES i->i), 'protocol'.

--------------------------------------------------------------------------
DATASETS
--------------------------------------------------------------------------
1. CausalRivers   ICLR 2025.  River discharge, eastern Germany (666 stations)
   + Bavaria (494) + Elbe-flood (42), 2019-2023 @ 15 min, with two (three)
   NetworkX ground-truth graphs that sample into thousands of subgraphs.
   Download: one 133 MB zip -> 1.0 GB extracted (well under the 2 GB budget,
   so it is fetched in full).  Data licence: "Datenlizenz Deutschland -
   Namensnennung - Version 2.0" (dl-de/by-2-0) from the German state
   environmental agencies.  Code in the repo is used only as a REFERENCE --
   the sampling/preprocessing below is re-implemented here.

2. CausalDynamics NeurIPS 2025 D&B, MIT.  Thousands of coupled ODE/SDE systems
   with Jacobian-derived ground-truth graphs, plus two idealized climate
   models.  HuggingFace ``kausable/CausalDynamics``.
   ``simple`` (inputs 43.9 MB + outputs 0.4 MB -> 109 MB extracted) AND
   ``coupled`` (inputs 6.51 GB + outputs 13.2 MB -> 10 GB extracted) are both
   fetched; ``coupled`` is the only tier where the benchmark discriminates at
   all, so it is worth the disk -- see CAUSALDYNAMICS_SUBSETS and
   docs/causal_baselines.md Sec. 6c.

   WHAT THE COUPLED TIER ACTUALLY MEASURES.  ``causal_dynamics_coupled_audit``
   (CLI: ``--audit coupled``, ``--tables-only``) re-derives the published
   Default row from the shipped eval, MEASURES the chance floor with a random
   ranker instead of the benchmark's all-zeros null, and pairs every shipped
   method against a constant identity matrix.  The identity matrix scores
   0.682/0.353 and no shipped method beats it, because the protocol keeps the
   diagonal and the diagonal carries 40.8% of the positive class.  Report the
   off-diagonal pair alongside anything from this tier -- see
   docs/causal_baselines.md Sec. 6c.

   THE TWO SUBSETS USE OPPOSITE EDGE CONVENTIONS.  ``simple`` stores the
   Jacobian sparsity pattern in ``adjacency_matrix`` ([effect, cause], this
   repo's convention); ``coupled`` stores an SCM summary graph in
   ``adjacency_matrix_summary`` ([cause, effect], the transpose).  Verified
   from the data, not the docs: on the 276 lag-free coupled systems, the nodes
   flagged in ``root_nodes`` have an empty off-diagonal COLUMN in 100.0% of
   files and an empty off-diagonal ROW in 0.0%.  ``load_causal_dynamics``
   therefore transposes ``coupled`` into this repo's convention and keeps the
   as-stored matrix in ``meta['A_native']``.

3. CauseMe        https://causeme.uv.es/ -- submission FORMAT is recorded and a
   submission file can be BUILT LOCALLY, but nothing is ever uploaded.
"""
from __future__ import annotations

import bz2
import json
import os
import pickle
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_BENCH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(_BENCH, "..", "data"))
CDATA = os.path.join(DATA, "causal")
CACHE = os.path.join(CDATA, "cache")

CR_PRODUCT = os.path.join(CDATA, "product")
CR_ZIP_URL = ("https://github.com/CausalRivers/causalrivers/releases/download/"
              "First_release/product.zip")
CR_REPO_URL = "https://github.com/CausalRivers/causalrivers"

CD_ROOT = os.path.join(CDATA, "causaldynamics", "extracted")
CD_HF = "https://huggingface.co/datasets/kausable/CausalDynamics/resolve/main"

CAUSEME_URL = "https://causeme.uv.es/"


def _ensure_dirs():
    os.makedirs(CDATA, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)


# =========================================================================== #
# 1.  CausalRivers
# =========================================================================== #
CR_REGIONS = {
    "east": ("rivers_east_germany.p", "rivers_ts_east_germany.csv",
             "rivers_meta_east_germany.csv"),
    "bavaria": ("rivers_bavaria.p", "rivers_ts_bavaria.csv",
                "rivers_meta_bavaria.csv"),
    "flood": ("rivers_flood.p", "rivers_ts_flood.csv",
              "rivers_meta_flood.csv"),
}

# The flood GRAPH is a 42-node subset of RiversEastGermany, but the flood CSV
# only covers the four-week Sep/Oct-2024 flood window.  The paper's "Full TS"
# column of Figure 3a uses the flood GRAPH against the FIVE-YEAR east-Germany
# series, so that pairing gets its own key.
CR_TS_OVERRIDE = {"flood_full": ("rivers_flood.p",
                                 "rivers_ts_east_germany.csv",
                                 "rivers_meta_east_germany.csv")}
CR_REGIONS.update(CR_TS_OVERRIDE)

CR_LICENCE = ("data: dl-de/by-2-0 (Datenlizenz Deutschland Namensnennung 2.0), "
              "German state environmental agencies; benchmark: CausalRivers, "
              "ICLR 2025")

# Published reference numbers, transcribed from the ICLR 2025 paper.
# Table 3 = AUROC, Table 7 = 'F1 max', both on RiversEastGermany with the
# diagonal REMOVED and the best hyperparameter combination per method.
CR_PUBLISHED = {
    # (metric, graph_set): {method: value}
    ("auroc", "random_3"): {"var": .82, "varlingam": .79, "dynotears": .50,
                            "pcmci": .65, "cdmi": .80},
    ("auroc", "random_5"): {"var": .80, "varlingam": .75, "dynotears": .61,
                            "pcmci": .65, "cdmi": .78},
    ("auroc", "close_3"): {"var": .81, "varlingam": .79, "dynotears": .50,
                           "pcmci": .64, "cdmi": .81},
    ("auroc", "close_5"): {"var": .81, "varlingam": .77, "dynotears": .50,
                           "pcmci": .62, "cdmi": .81},
    ("auroc", "root_cause_3"): {"var": .79, "varlingam": .77, "dynotears": .50,
                                "pcmci": .70, "cdmi": .72},
    ("auroc", "confounder_3"): {"var": .71, "varlingam": .68, "dynotears": .53,
                                "pcmci": .66, "cdmi": .63},
    ("f1max", "random_3"): {"var": .83, "varlingam": .81, "dynotears": .62,
                            "pcmci": .70, "cdmi": .82},
    ("f1max", "random_5"): {"var": .65, "varlingam": .63, "dynotears": .54,
                            "pcmci": .52, "cdmi": .63},
    ("f1max", "close_5"): {"var": .68, "varlingam": .65, "dynotears": .37,
                           "pcmci": .48, "cdmi": .67},
    # Figure 3a, Experiment Set 2: flood GRAPH (42 nodes), "Random 5" subgraph
    # sampling, evaluated on the FULL five-year time series.  These are the
    # numbers quoted as "AUROC (Full TS)".
    ("auroc", "flood_full_random_5"): {"var": .85, "cdmi": .83, "dynotears": .79,
                                       "varlingam": .72, "pcmci": .60,
                                       "rp": .78, "cc": .81, "rpcc": .81,
                                       "cp": .65},
}

CR_PROTOCOL = dict(
    metric="mean over subgraph samples of per-sample AUROC ('Individual AUROC'); "
           "'F1 max' = best F1 over all thresholds of the POOLED prediction",
    self_loops=False,
    self_loops_note="CausalRivers REMOVES the diagonal before scoring "
                    "(benchmark.py remove_diagonal=True): 'rivers are highly "
                    "autocorrelated and the causal links here are not relevant'. "
                    "An n-node sample therefore contributes n*(n-1) decisions.",
    summary_graph="max over lags of the (absolute) coefficient "
                  "(config/method/var.yaml: map_to_summary_graph=max)",
    hyperparameters="best over resolution {15min,1H,6H,12H,24H} x normalize "
                    "{True,False} x max_lag {3,5} (+ var_absolute_values "
                    "{True,False} for VAR); the paper reports the BEST cell, "
                    "with no label-free selection rule",
    preprocessing="round the 15-min datetime index to `resolution`, groupby-mean, "
                  "min-max normalise per column (optional), then linear "
                  "interpolate; finally trim each sample to the first..last row "
                  "that is fully observed",
    label_convention="graph_to_label_tensor sets labels[m, n] = 1 for edge "
                     "(nodes[n] -> nodes[m]), i.e. labels[effect, cause] -- "
                     "the same convention as this repo",
)


def causal_rivers_available() -> bool:
    return os.path.isdir(CR_PRODUCT) and os.path.exists(
        os.path.join(CR_PRODUCT, "rivers_east_germany.p"))


def fetch_causal_rivers(force: bool = False) -> str:
    """Download + unzip the 133 MB CausalRivers ``product.zip`` (1.0 GB on
    disk).  Returns the product directory."""
    _ensure_dirs()
    if causal_rivers_available() and not force:
        return CR_PRODUCT
    import urllib.request
    import zipfile
    zpath = os.path.join(CDATA, "product.zip")
    if not os.path.exists(zpath) or force:
        print(f"[causal_rivers] downloading {CR_ZIP_URL} (~133 MB)")
        urllib.request.urlretrieve(CR_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(CDATA)
    return CR_PRODUCT


def causal_rivers_graph(region: str = "east"):
    """The full NetworkX DiGraph ground truth for a region."""
    import networkx as nx  # noqa: F401  (needed to unpickle)
    if region not in CR_REGIONS:
        raise ValueError(f"region must be one of {sorted(CR_REGIONS)}")
    fetch_causal_rivers()
    gp, _, _ = CR_REGIONS[region]
    with open(os.path.join(CR_PRODUCT, gp), "rb") as fh:
        return pickle.load(fh)


# --- subgraph sampling (re-implementation of tools/graph_sampling_tools.py) -- #
def _all_connected_subsets(G, n_vars: int) -> List[Tuple[int, ...]]:
    """Every weakly-connected node subset of size n_vars, as sorted tuples.

    Mirrors CausalRivers' ``get_all_subgraphs``: grow from each start node by
    adding any predecessor OR successor of any node already in the set, n_vars-1
    times, then dedupe by sorted tuple.  (They dedupe only at the end; deduping
    at every step yields the identical final set and is far cheaper.)
    """
    out: set = set()
    for s in G.nodes:
        stack = {(s,)}
        for _ in range(n_vars - 1):
            nxt = set()
            for g in stack:
                ext = set()
                for node in g:
                    ext.update(x for x in G.predecessors(node) if x not in g)
                    ext.update(x for x in G.successors(node) if x not in g)
                for e in ext:
                    nxt.add(tuple(sorted(g + (e,))))
            if not nxt:
                stack = set()
                break
            stack = nxt
        out |= stack
    return sorted(out)


def _longest_path_measure(G, nodes, measure="geo_distance"):
    import networkx as nx
    sub = G.subgraph(nodes)
    lengths = []
    for a in sub.nodes:
        for b in sub.nodes:
            for path in nx.all_simple_edge_paths(sub, a, b):
                d = [sub.edges[u, v][measure] for u, v in path]
                if any(x is None for x in d):
                    return np.inf
                lengths.append(sum(d))
    return max(lengths) if lengths else np.inf


def causal_rivers_subgraphs(region: str = "east", n_vars: int = 3,
                            strategy: str = "random",
                            max_distance: float = 0.3,
                            limit: Optional[int] = None,
                            seed: int = 0) -> List[Tuple[int, ...]]:
    """Node sets for one of the paper's graph SETS.

    VERIFIED against Table 6 of the paper (counts on RiversEastGermany):
    random_3 1196, random_5 7521, close_3 636, close_5 637, root_cause_3 649,
    root_cause_5 655, confounder_3 24, 1_random_3 651, 1_random_5 2790 -- all
    EXACT.  The single exception is confounder_5: we return 332, the paper says
    361, because their ``select_confounder_samples`` concatenates the candidate
    list of every confounder without deduping ACROSS confounders, so a subgraph
    containing two confounders is counted twice.  Summing without the final
    dedupe reproduces 361 exactly.

    strategy: 'random' (all connected subsets -- the paper's "Random"),
              'root_cause' (chains: longest path == n_vars-1),
              'close' (max geographic edge distance < max_distance),
              'confounder' (subsets containing a multi-successor node AND all
                            of its successors, with that node then REMOVED),
              '1_random' (a connected (n-1)-set plus one isolated node).
    """
    import networkx as nx
    G = causal_rivers_graph(region)
    rng = np.random.default_rng(seed)
    if strategy == "confounder":
        conf = [x for x in G.nodes if len(list(G.successors(x))) > 1]
        pool = _all_connected_subsets(G, n_vars)
        samples: List[Tuple[int, ...]] = []
        for c in conf:
            cand = [x for x in pool if c in x]
            for succ in G.successors(c):
                cand = [x for x in cand if succ in x]
            samples.extend(cand)
        subs = sorted(set(samples))
    elif strategy == "1_random":
        base = _all_connected_subsets(G, n_vars - 1)
        iso = list(nx.isolates(G))
        if not iso:
            raise RuntimeError(f"region {region!r} has no isolated nodes")
        subs = [tuple(sorted(list(b) + [iso[int(rng.integers(len(iso)))]]))
                for b in base]
    else:
        subs = _all_connected_subsets(G, n_vars)
        if strategy == "root_cause":
            subs = [x for x in subs
                    if nx.dag_longest_path_length(G.subgraph(x)) == n_vars - 1]
        elif strategy == "close":
            subs = [x for x in subs
                    if _longest_path_measure(G, x) < max_distance]
        elif strategy != "random":
            raise ValueError(f"unknown strategy {strategy!r}")
    if limit is not None:
        subs = subs[:limit]
    return subs


def causal_rivers_labels(region: str, nodes: Sequence[int]) -> np.ndarray:
    """(P, P) ground truth for a node subset, OUR convention A[i, j] = j -> i.

    Node ORDER is ``sorted(nodes)`` -- exactly what CausalRivers does, and what
    the time-series column order below matches.
    """
    G = causal_rivers_graph(region)
    ns = sorted(nodes)
    P = len(ns)
    A = np.zeros((P, P))
    for n, x in enumerate(ns):
        for m, y in enumerate(ns):
            if G.has_edge(x, y):        # x -> y
                A[m, n] = 1.0           # A[effect, cause]
    return A


# --- time series ----------------------------------------------------------- #
def _cr_preprocess(df, resolution="6h", interpolate=True, normalize=True,
                   subsample=1):
    """CausalRivers' ``preprocess_data`` (tools/tools.py), re-implemented.

    Order matters and is preserved verbatim: round -> groupby-mean ->
    subsample -> min-max normalise -> linear interpolate.  Normalising BEFORE
    interpolating is what the reference does; swapping them changes nothing for
    min-max but is noted so the protocol can be audited.
    """
    import pandas as pd
    d = df.copy()
    d["dt"] = pd.to_datetime(d.index).round(resolution).values
    d = d.groupby("dt").mean()
    if subsample and subsample > 1:
        d = d.iloc[::subsample, :]
    if normalize:
        d = (d - d.min()) / (d.max() - d.min())
    if interpolate:
        d = d.interpolate()
    return d


def _cr_trim_trailing_nans(A: np.ndarray) -> np.ndarray:
    """CausalRivers' ``remove_trailing_nans``: keep rows first..last where the
    sample is fully observed."""
    good = np.where(np.isfinite(A).all(axis=1))[0]
    if good.size == 0:
        return A[:0]
    return A[good.min(): good.max() + 1]


def _cr_ts_cache_path(region, nodes_key, resolution, normalize, interpolate,
                      subsample):
    return os.path.join(
        CACHE, f"cr_{region}_{nodes_key}_{resolution}_n{int(normalize)}"
               f"_i{int(interpolate)}_s{subsample}.npz")


def causal_rivers_series(region: str = "east", nodes: Optional[Sequence[int]] = None,
                         resolution: str = "6h", normalize: bool = True,
                         interpolate: bool = True, subsample: int = 1,
                         cache: bool = True):
    """Preprocessed (T, P) matrix for ``sorted(nodes)`` of a region.

    The raw east/bavaria CSVs are 619/466 MB, so the preprocessed matrix is
    cached to ``data/causal/cache/``.  When ``nodes`` is None every station of
    the region's graph is loaded (heavy; prefer passing a node subset).
    """
    import pandas as pd
    fetch_causal_rivers()
    gp, tsp, _ = CR_REGIONS[region]
    G = causal_rivers_graph(region)
    ns = sorted(G.nodes) if nodes is None else sorted(nodes)
    key = ("all" if nodes is None
           else f"{len(ns)}x{abs(hash(tuple(ns))) % (10 ** 10)}")
    cp = _cr_ts_cache_path(region, key, resolution, normalize, interpolate,
                           subsample)
    if cache and os.path.exists(cp):
        with np.load(cp, allow_pickle=False) as z:
            if list(z["nodes"]) == list(ns):
                return z["X"], list(z["nodes"])
    usecols = ["datetime"] + [str(x) for x in ns]
    df = pd.read_csv(os.path.join(CR_PRODUCT, tsp), index_col="datetime",
                     usecols=usecols)
    df = df[[str(x) for x in ns]]
    d = _cr_preprocess(df, resolution=resolution, interpolate=interpolate,
                       normalize=normalize, subsample=subsample)
    X = d.to_numpy(dtype=float)
    if cache:
        _ensure_dirs()
        np.savez_compressed(cp, X=X, nodes=np.array(ns))
    return X, ns


def load_causal_rivers(region: str = "flood_full", n_vars: int = 5,
                       strategy: str = "random", index: int = 0,
                       resolution: str = "6h", normalize: bool = True,
                       interpolate: bool = True, subsample: int = 1,
                       trim: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """ONE CausalRivers subgraph sample as ``(X, A_true, meta)``.

    Use :func:`iter_causal_rivers` for the whole graph set (that is what the
    published numbers average over).
    """
    subs = causal_rivers_subgraphs(region, n_vars=n_vars, strategy=strategy)
    if not subs:
        raise RuntimeError(f"no subgraphs for {region}/{strategy}/{n_vars}")
    nodes = subs[index % len(subs)]
    return _cr_sample(region, nodes, resolution, normalize, interpolate,
                      subsample, trim, strategy, n_vars, index, len(subs))


def _cr_sample(region, nodes, resolution, normalize, interpolate, subsample,
               trim, strategy, n_vars, index, n_samples,
               joint: Optional[Tuple[np.ndarray, List[int]]] = None):
    ns = sorted(nodes)
    if joint is not None:
        Xall, all_nodes = joint
        pos = [all_nodes.index(x) for x in ns]
        X = Xall[:, pos]
    else:
        X, _ = causal_rivers_series(region, ns, resolution=resolution,
                                    normalize=normalize,
                                    interpolate=interpolate,
                                    subsample=subsample)
    if trim:
        X = _cr_trim_trailing_nans(X)
    gregion = "flood" if region == "flood_full" else region
    A = causal_rivers_labels(gregion, ns)
    meta = dict(name=f"causal_rivers/{region}/{strategy}_{n_vars}",
                source=CR_REPO_URL, licence=CR_LICENCE, nodes=ns,
                region=region, graph_region=gregion, strategy=strategy,
                n_vars=n_vars, index=index, n_samples=n_samples,
                resolution=resolution, normalize=normalize,
                interpolate=interpolate, subsample=subsample,
                self_loops=False, protocol=dict(CR_PROTOCOL),
                T=int(X.shape[0]))
    return X, A, meta


def iter_causal_rivers(region: str = "flood_full", n_vars: int = 5,
                       strategy: str = "random", resolution: str = "6h",
                       normalize: bool = True, interpolate: bool = True,
                       subsample: int = 1, trim: bool = True,
                       limit: Optional[int] = None, seed: int = 0
                       ) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
    """Stream every sample of a CausalRivers graph set.

    The whole region's series is preprocessed ONCE and then sliced per sample
    -- identical to ``load_joint_samples`` in the reference, because every step
    of the preprocessing is column-wise.
    """
    subs = causal_rivers_subgraphs(region, n_vars=n_vars, strategy=strategy,
                                   limit=limit, seed=seed)
    needed = sorted({n for s in subs for n in s})
    joint = causal_rivers_series(region, needed, resolution=resolution,
                                 normalize=normalize, interpolate=interpolate,
                                 subsample=subsample)
    joint = (joint[0], list(joint[1]))
    for i, nodes in enumerate(subs):
        yield _cr_sample(region, nodes, resolution, normalize, interpolate,
                         subsample, trim, strategy, n_vars, i, len(subs),
                         joint=joint)


# =========================================================================== #
# 2.  CausalDynamics
# =========================================================================== #
CD_LICENCE = "MIT (kausable/CausalDynamics, NeurIPS 2025 D&B)"
CD_REPO = "https://github.com/kausable/CausalDynamics"

CAUSALDYNAMICS_SUBSETS = {
    # subset:   (inputs bytes, outputs bytes, fetched?)
    # Sizes are the HTTP content-length of
    # {CD_HF}/{inputs,outputs}/<subset>.tar.gz, re-checked 2026-09-08.
    # All three are FETCHED: 6.61 GB downloaded -> 10.2 GB extracted.  Member
    # counts verified with `tar tzf | wc -l` against the extracted tree
    # (simple 585/585, climate 12/12, coupled 14 096/14 096 .nc), so nothing
    # was subsetted.
    "simple": (43_921_694, 420_473, True),
    "climate": (54_233_197, 14_954, True),
    "coupled": (6_514_628_417, 13_167_388, True),    # 6.51 GB -> 10 GB
}

# ``coupled`` is over the ~2 GB default budget but is the ONLY tier with real
# discrimination, so it is opted in explicitly rather than by raising the
# budget for everything.
CD_BUDGET_BYTES = 2_000_000_000
CD_OVER_BUDGET_OK = ("coupled",)

CD_PROTOCOL = dict(
    metric="AUROC and AUPRC over the flattened (P, P) summary adjacency; "
           "'Individual' = mean over the batch dimension, 'Joint' = pooled",
    self_loops=True,
    self_loops_note="CausalDynamics KEEPS the diagonal: the ground truth is the "
                    "Jacobian sparsity pattern, whose diagonal is usually 1, and "
                    "score.py never masks it.  A P-node system therefore "
                    "contributes P*P decisions -- the opposite of CausalRivers.",
    label_convention="simple subset: adjacency_matrix[i, j] = 1 iff "
                     "d(xdot_i)/d(x_j) != 0, i.e. 'j -> i' (effect, cause) -- the "
                     "same convention as this repo. coupled subset: "
                     "adjacency_matrix_summary[i, j] = 1 iff 'i -> j' "
                     "(cause, effect) -- the TRANSPOSE. scm.py documents this "
                     "('A[i,j] indicates an edge from i to j') and the data "
                     "confirms it: on the 276 lag-free coupled systems the "
                     "root_nodes have an empty off-diagonal COLUMN 100.0% of "
                     "the time and an empty off-diagonal ROW 0.0% of the time. "
                     "The two subsets of the SAME benchmark therefore disagree.",
    labels_used_by_eval="eval.py scores against adjacency_matrix_summary when "
                        "it exists (all coupled files) and adjacency_matrix "
                        "otherwise (all simple files) -- NOT always "
                        "adjacency_matrix",
    observation_note="coupled time_series is 4-D (T, n_traj, n_nodes, dim); "
                     "eval.py keeps ONLY component 0 of every node "
                     "(timeseries[..., 0]), so each coupled system is a "
                     "PARTIALLY OBSERVED n_nodes-variable problem, not an "
                     "n_nodes*dim one",
    baseline_note="two defects in the shipped wrapper suite, both in code "
                  "shared by every subset. (1) VARLiNGAM binarises with "
                  "abs(adjacency_matrices_[tau_max]) > 0; whenever lingam's "
                  "adaptive-lasso prune leaves the lag matrix dense this is "
                  "all-ones and AUROC is exactly 0.500 by construction -- it "
                  "happens in 28.3% of simple cells and 22.9% of coupled "
                  "cells, NOT always. (2) ORIENTATION: the wrappers are a "
                  "mix of conventions -- DYNOTEARS, TSCI, TCDF, NGC, PCMCI+ "
                  "and FPCMCI emit [cause, effect] while VARLiNGAM and RCD "
                  "emit [effect, cause] (lingam's adjacency_matrices_[k][i, j] "
                  "is the effect of x_j(t-k) on x_i(t), verified directly). "
                  "Because the two subsets store OPPOSITE ground truths, a "
                  "DIFFERENT half of the suite is transposed on each: on "
                  "simple DYNOTEARS/TSCI are transposed (hence their "
                  "below-chance .43/.46), on coupled VARLiNGAM/RCD are "
                  "transposed instead. DYNOTEARS' coupled .66 is NOT affected "
                  "by the transposition bug; VARLiNGAM's coupled .57 IS.",
)

# The leaderboard's "Default" rows (website/leaderboard.md).  Every other row
# varies ONE axis of these and holds the rest fixed.
CD_DEFAULT_SETTING = {
    "simple": "noise=0.00_confounder=False",
    # climate does not use the key=value directory scheme at all; its two
    # "settings" are the two idealized models, and the leaderboard's
    # "Atmos-ocean" / "ENSO modes" rows are exactly these two directories.
    "climate": "coupled_atmos_ocean",
    "coupled": ("coupling=linear_noise=0.00_systems=10_confounder=False"
                "_standardize=False_timelag=0_activation=none"),
}

# "Default" is not one directory: coupling and activation are FREE, everything
# else pinned.  These 6 cells are what the published Default row averages.
CD_COUPLED_DEFAULT_CELL = dict(noise="0.00", systems="10", confounder="False",
                               standardize="False", timelag="0")

CD_PUBLISHED = {   # kausable.github.io/CausalDynamics leaderboard, AUROC/AUPRC
    "simple": {"pcmciplus": (.47, .68), "fpcmci": (.51, .70),
               "varlingam": (.50, .69), "dynotears": (.43, .67),
               "ngc": (.50, .68), "tsci": (.46, .68), "cutsplus": (.48, .68)},
    "coupled": {"pcmciplus": (.68, .27), "fpcmci": (.67, .24),
                "varlingam": (.57, .18), "dynotears": (.66, .32),
                "ngc": (.50, .16), "tsci": (.69, .36), "cutsplus": (.50, .16)},
    "climate": {"pcmciplus": (.69, .88), "fpcmci": (.50, .81),
                "varlingam": (.50, .81), "dynotears": (.62, .86),
                "ngc": (.49, .81), "tsci": (.58, .84), "cutsplus": (.50, .81)},
}


# Where the coupled audit (leaderboard re-average + measured chance floor +
# our own reproduction) is cached, so --tables-only can regenerate every table
# without touching the 10 GB of netCDF again.
CD_AUDIT_CACHE = os.path.join(CACHE, "cd_coupled_audit.json")

# The leaderboard row this audit targets, transcribed from the shipped
# website/leaderboard.md (NOT from the rendered site), together with the paper's
# own description of what it averages over.
CD_LEADERBOARD_SRC = ("kausable/CausalDynamics website/leaderboard.md, Table 1, "
                      "'Default' rows; coupled Default = n=10 coupled ODEs, "
                      "delta=0, no confounder, tau=0, no internal standardization")


def causal_dynamics_available(subset: str = "simple") -> bool:
    return os.path.isdir(os.path.join(CD_ROOT, subset))


def fetch_causal_dynamics(subset: str = "simple", force: bool = False) -> str:
    """Download + extract one CausalDynamics subset.

    'coupled' inputs are 6.51 GB, over the ~2 GB budget, but it is the only
    tier where the benchmark discriminates, so it is opted in by name via
    CD_OVER_BUDGET_OK.  Any OTHER over-budget subset still needs force=True.

    The download is resumable: a partial ``inputs_<subset>.tar.gz`` in the
    cache is continued with an HTTP Range request rather than restarted, so an
    interrupted 6.5 GB pull does not have to be repeated.
    """
    if subset not in CAUSALDYNAMICS_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(CAUSALDYNAMICS_SUBSETS)}")
    nin, nout, _ = CAUSALDYNAMICS_SUBSETS[subset]
    if nin > CD_BUDGET_BYTES and subset not in CD_OVER_BUDGET_OK and not force:
        raise RuntimeError(
            f"CausalDynamics '{subset}' inputs are {nin / 1e9:.2f} GB, above the "
            "~2 GB budget for this harness. Pass force=True to override.")
    if causal_dynamics_available(subset) and not force:
        return os.path.join(CD_ROOT, subset)
    import tarfile
    root = os.path.join(CDATA, "causaldynamics")
    os.makedirs(CD_ROOT, exist_ok=True)
    want = {"inputs": nin, "outputs": nout}
    for stage in ("inputs", "outputs"):
        tgz = os.path.join(root, f"{stage}_{subset}.tar.gz")
        url = f"{CD_HF}/{stage}/{subset}.tar.gz"
        if force and os.path.exists(tgz):
            os.remove(tgz)
        _download_resumable(url, tgz, want[stage])
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(path=CD_ROOT)
    return os.path.join(CD_ROOT, subset)


def _download_resumable(url: str, dest: str, expect: Optional[int] = None,
                        chunk: int = 1 << 20) -> str:
    """GET ``url`` -> ``dest``, continuing a partial file with a Range request.

    Needed because the coupled inputs are 6.51 GB: restarting from zero after a
    dropped connection is not acceptable.  ``expect`` is the known
    content-length; when the local file already has that many bytes nothing is
    transferred.
    """
    import urllib.request
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if expect is not None and have == expect:
        return dest
    if expect is not None and have > expect:      # corrupt/truncated-to-longer
        os.remove(dest)
        have = 0
    hdr = {"User-Agent": "curl/8"}
    if have:
        hdr["Range"] = f"bytes={have}-"
        print(f"[causal_dynamics] resuming {url} at {have / 1e9:.2f} GB")
    else:
        print(f"[causal_dynamics] downloading {url}")
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=120) as r:
        # 200 on a Range request means the server ignored it -> start over.
        mode = "ab" if (have and r.status == 206) else "wb"
        with open(dest, mode) as fh:
            while True:
                b = r.read(chunk)
                if not b:
                    break
                fh.write(b)
    return dest


def causal_dynamics_settings(subset: str = "simple") -> List[str]:
    """The available noise/confounder cells, e.g. 'noise=0.00_confounder=False'."""
    root = os.path.join(CD_ROOT, subset)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def causal_dynamics_systems(subset: str = "simple",
                            setting: Optional[str] = None) -> List[str]:
    if setting is None:
        setting = CD_DEFAULT_SETTING.get(subset, "noise=0.00_confounder=False")
    d = os.path.join(CD_ROOT, subset, setting, "data")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".nc"))


def parse_cd_setting(setting: str) -> Dict[str, str]:
    """'coupling=linear_noise=0.00_systems=10_...' -> {'coupling': 'linear', ...}.

    The keys are separated by '_' and every key contains '=', so splitting on
    '_' and keeping the '='-bearing tokens is unambiguous for both subsets.
    """
    return dict(kv.split("=", 1) for kv in setting.split("_") if "=" in kv)


def causal_dynamics_cells(subset: str = "coupled", **pins: str) -> List[str]:
    """Every setting directory whose parsed keys match all ``pins``.

    ``causal_dynamics_cells('coupled', **CD_COUPLED_DEFAULT_CELL)`` returns the
    6 directories (3 couplings x 2 activations) that the published "Default"
    row averages over.
    """
    out = []
    for s in causal_dynamics_settings(subset):
        d = parse_cd_setting(s)
        if all(d.get(k) == v for k, v in pins.items()):
            out.append(s)
    return out


def load_causal_dynamics(system: str = "Lorenz_N10_T1000",
                         subset: str = "simple",
                         setting: Optional[str] = None,
                         realisation: Optional[int] = 0,
                         standardize: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """One CausalDynamics system as ``(X, A_true, meta)``.

    Files are netCDF4 (= HDF5), read with h5py so no xarray dependency is
    needed.

    simple
        ``time_series`` is (T, n_traj, P): ``n_traj`` independent realisations
        of the SAME P-variable system (same ground-truth graph).  Label is
        ``adjacency_matrix``.
    coupled
        ``time_series`` is (T, n_traj, n_nodes, dim): each "node" is a whole
        sub-system (a Lorenz, a Rossler, ...) with ``dim`` internal states, and
        the coupling graph is between NODES.  eval.py keeps only component 0 of
        every node -- ``timeseries[..., 0]`` -- so the task is a PARTIALLY
        OBSERVED n_nodes-variable problem; this loader does the same.  Label is
        ``adjacency_matrix_summary`` (regular edges + time edges), which is
        what eval.py scores against whenever it is present.

    ``realisation=None`` still returns the first realisation in X; the whole
    (T, n_traj, P) cube is in ``meta['cube']`` either way.

    ORIENTATION.  A_true is always returned in THIS REPO's convention,
    ``A_true[i, j] != 0  <=>  "j causes i"``:

      * simple  -- ``adjacency_matrix`` is the Jacobian sparsity pattern and is
        ALREADY [effect, cause]; returned unchanged.  Verified on Rossler: the
        file gives [[0,1,1],[1,1,0],[1,0,1]], exactly the rows of the Rossler
        Jacobian (xdot = -y - z; ydot = x + a y; zdot = b + z(x - c)).
      * coupled -- ``adjacency_matrix_summary`` is [cause, effect] (scm.py:
        "A[i,j] indicates an edge from i to j"), so it is TRANSPOSED here.
        Verified from the data on the 276 lag-free coupled systems: the nodes
        flagged in ``root_nodes`` (which by definition have no parents) have an
        empty off-diagonal COLUMN in 100.0% of files and an empty off-diagonal
        ROW in 0.0%.

    ``meta['A_native']`` is the matrix EXACTLY as stored, in the benchmark's own
    orientation -- that is what ``reproduce_causal_dynamics`` scores against so
    it can match the shipped numbers.  For simple the two are identical.
    """
    import h5py
    if setting is None:
        setting = CD_DEFAULT_SETTING.get(subset, "noise=0.00_confounder=False")
    p = os.path.join(CD_ROOT, subset, setting, "data", f"{system}.nc")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found; run fetch_causal_dynamics({subset!r}) first")
    with h5py.File(p, "r") as f:
        ts = np.asarray(f["time_series"][...], dtype=float)
        # eval.py: adjacency_matrix_summary when present, else adjacency_matrix.
        label_var = ("adjacency_matrix_summary"
                     if "adjacency_matrix_summary" in f else "adjacency_matrix")
        A_native = np.asarray(f[label_var][...], dtype=float)
        root_nodes = (np.asarray(f["root_nodes"][...]).astype(bool)
                      if "root_nodes" in f else None)
    # Partially observed coupled systems: keep component 0 of every node.
    dim = None
    if ts.ndim == 4:
        dim = int(ts.shape[3])
        ts = ts[..., 0]
    if ts.ndim == 3:
        cube = ts
        X = ts[:, realisation or 0, :]
    else:
        cube = ts[:, None, :]
        X = ts
    # Into this repo's [effect, cause] convention.
    transposed = label_var == "adjacency_matrix_summary"
    A = A_native.T.copy() if transposed else A_native
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    meta = dict(name=f"causal_dynamics/{subset}/{setting}/{system}",
                source=CD_REPO, licence=CD_LICENCE, system=system,
                subset=subset, setting=setting,
                n_realisations=int(cube.shape[1]), realisation=realisation,
                cube=cube, self_loops=True, protocol=dict(CD_PROTOCOL),
                T=int(X.shape[0]), P=int(X.shape[1]),
                A_native=A_native, label_var=label_var,
                native_orientation="[cause, effect]" if transposed
                                   else "[effect, cause]",
                transposed_from_native=bool(transposed),
                node_dim=dim, partially_observed=dim is not None,
                root_nodes=root_nodes)
    return X, A, meta


def iter_causal_dynamics(subset: str = "simple",
                         setting: Optional[str] = None,
                         systems: Optional[Iterable[str]] = None,
                         realisation: Optional[int] = 0, limit=None
                         ) -> Iterator[Tuple[np.ndarray, np.ndarray, dict]]:
    if setting is None:
        setting = CD_DEFAULT_SETTING.get(subset, "noise=0.00_confounder=False")
    names = list(systems) if systems else causal_dynamics_systems(subset, setting)
    if limit:
        names = names[:limit]
    for s in names:
        yield load_causal_dynamics(s, subset=subset, setting=setting,
                                   realisation=realisation)


def causal_dynamics_published_eval(method: str, subset: str = "simple",
                                   setting: Optional[str] = None
                                   ) -> Dict[str, float]:
    """The benchmark's OWN shipped per-system metrics for a baseline.

    ``outputs/<subset>.tar.gz`` contains, for each method and system, an .nc
    holding Individual/Joint AUROC, AUPRC, SHD and the null model.  Averaging
    these reproduces the leaderboard exactly, without re-running anything.
    """
    import h5py
    if setting is None:
        setting = CD_DEFAULT_SETTING.get(subset, "noise=0.00_confounder=False")
    d = os.path.join(CD_ROOT, subset, setting, "eval", method)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} not found (methods: "
                                f"{causal_dynamics_eval_methods(subset, setting)})")
    acc: Dict[str, List[float]] = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".nc"):
            continue
        with h5py.File(os.path.join(d, fn), "r") as f:
            for k in f.keys():
                acc.setdefault(k, []).append(float(f[k][()]))
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out["n_systems"] = float(len(next(iter(acc.values()), [])))
    return out


def causal_dynamics_eval_methods(subset: str = "simple",
                                 setting: Optional[str] = None) -> List[str]:
    if setting is None:
        setting = CD_DEFAULT_SETTING.get(subset, "noise=0.00_confounder=False")
    d = os.path.join(CD_ROOT, subset, setting, "eval")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def causal_dynamics_leaderboard(subset: str = "coupled",
                                settings: Optional[Iterable[str]] = None,
                                methods: Optional[Iterable[str]] = None,
                                ) -> Dict[str, Dict[str, float]]:
    """Re-derive a leaderboard ROW by averaging the SHIPPED eval .nc files.

    This needs only ``outputs/<subset>.tar.gz`` (13 MB for coupled) -- no time
    series and no re-running of any baseline.  ``settings=None`` uses the
    published "Default" cell for the subset.

    Also returns, per method, the fraction of cells that are DEGENERATE:
    ``auroc_exactly_half`` (a constant prediction -- AUROC is 0.5 by
    construction, not by measurement) and ``auprc_equals_null`` (the method's
    AUPRC is bit-identical to the shipped null model's).
    """
    import h5py
    if settings is None:
        settings = (causal_dynamics_cells(subset, **CD_COUPLED_DEFAULT_CELL)
                    if subset == "coupled"
                    else [CD_DEFAULT_SETTING.get(subset)])
    settings = list(settings)
    acc: Dict[str, Dict[str, List[float]]] = {}
    for s in settings:
        ev = os.path.join(CD_ROOT, subset, s, "eval")
        if not os.path.isdir(ev):
            continue
        for m in sorted(os.listdir(ev)):
            if methods and m not in methods:
                continue
            d = os.path.join(ev, m)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".nc"):
                    continue
                with h5py.File(os.path.join(d, fn), "r") as f:
                    rec = {k: float(f[k][()]) for k in f.keys()}
                a = acc.setdefault(m, {})
                for k, v in rec.items():
                    a.setdefault(k, []).append(v)
                a.setdefault("_half", []).append(
                    float(abs(rec.get("Individual_AUROC", np.nan) - .5) < 1e-9))
                a.setdefault("_null", []).append(float(
                    abs(rec.get("Individual_AUPRC", np.nan)
                        - rec.get("Null_AUPRC", np.nan)) < 1e-9))
                a.setdefault("_above", []).append(
                    float(rec.get("Individual_AUROC", np.nan) > 0.5 + 1e-9))
                a.setdefault("_spread", []).append(
                    rec.get("Individual_AUROC", np.nan))
    out = {}
    for m, a in acc.items():
        o = {k: float(np.nanmean(v)) for k, v in a.items() if not k.startswith("_")}
        o["auroc_exactly_half"] = float(np.mean(a["_half"]))
        o["auprc_equals_null"] = float(np.mean(a["_null"]))
        # Per-system spread, not just the mean: "0.676 on average" and "above
        # 0.5 on 100 of 100 systems" are different claims and only the second
        # one says the separation is not an artefact of averaging.
        o["auroc_above_half_frac"] = float(np.mean(a["_above"]))
        o["auroc_sd"] = float(np.nanstd(a["_spread"]))
        o["auroc_min"] = float(np.nanmin(a["_spread"]))
        o["auroc_max"] = float(np.nanmax(a["_spread"]))
        o["effect_size_vs_half"] = float(
            (np.nanmean(a["_spread"]) - 0.5) / (np.nanstd(a["_spread"]) + 1e-12))
        o["n_cells"] = int(len(a["_half"]))
        out[m] = o
    return out


def causal_dynamics_chance_floor(subset: str = "coupled",
                                 settings: Optional[Iterable[str]] = None,
                                 n_draws: int = 20, seed: int = 0,
                                 limit: Optional[int] = None) -> Dict[str, float]:
    """MEASURE the chance floor by scoring uniform-random graphs.

    The benchmark's own ``Null_AUROC``/``Null_AUPRC`` come from an all-ZEROS
    prediction, which is a degenerate control: with a constant score AUROC is
    0.5 and AUPRC collapses to the positive rate.  This instead draws
    U(0,1) scores -- a real random ranker -- and runs them through the
    benchmark's own scoring path, so "beats chance" is measured.

    AUPRC's null is the EDGE DENSITY of the ground truth, which varies a lot
    across coupled cells, so the density is reported alongside.
    """
    if settings is None:
        settings = (causal_dynamics_cells(subset, **CD_COUPLED_DEFAULT_CELL)
                    if subset == "coupled"
                    else [CD_DEFAULT_SETTING.get(subset)])
    rng = np.random.default_rng(seed)
    au, ap, dens, shipped_au, shipped_ap = [], [], [], [], []
    for s in settings:
        for X, A, meta in iter_causal_dynamics(subset, s, realisation=None,
                                               limit=limit):
            L = meta["A_native"]
            if np.all(L == 1) or np.all(L == 0):
                continue                       # exactly what eval.py skips
            n = int(meta["cube"].shape[1])
            for _ in range(n_draws):
                preds = rng.random((n, *L.shape))
                r = score_causal_dynamics(preds, L)
                au.append(r["individual_auroc"])
                ap.append(r["individual_auprc"])
            dens.append(float(np.mean(L != 0)))
            z = score_causal_dynamics(np.zeros((n, *L.shape)), L)
            shipped_au.append(z["individual_auroc"])
            shipped_ap.append(z["individual_auprc"])
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(subset=subset, n_systems=len(dens), n_draws=n_draws,
                random_auroc=f(au), random_auroc_sd=float(np.std(au)) if au else float("nan"),
                random_auprc=f(ap), random_auprc_sd=float(np.std(ap)) if ap else float("nan"),
                allzero_auroc=f(shipped_au), allzero_auprc=f(shipped_ap),
                edge_density=f(dens),
                density_min=float(np.min(dens)) if dens else float("nan"),
                density_max=float(np.max(dens)) if dens else float("nan"))



# --------------------------------------------------------------------------- #
# Trivial predictors.  These are the reference points a published number has to
# clear before it means anything, and on the coupled split TWO of them clear the
# published leaderboard.
# --------------------------------------------------------------------------- #
CD_TRIVIAL = ("random", "zeros", "identity", "identity_plus_noise")


def _cd_trivial_preds(kind: str, P: int, n: int, rng) -> np.ndarray:
    if kind == "random":
        return rng.random((n, P, P))
    if kind == "zeros":
        return np.zeros((n, P, P))
    if kind == "identity":
        return np.tile(np.eye(P)[None], (n, 1, 1))
    if kind == "identity_plus_noise":
        return rng.random((n, P, P)) * 0.5 + np.eye(P)[None]
    raise ValueError(kind)


# Golden digests of the two loader contracts.  The `simple` split is read by a
# detached pipeline while this file is edited, and "coupled support was added
# without touching simple" is a claim that has to be CHECKED, not asserted, so
# the whole default cell of each subset is hashed.  Regenerate deliberately
# (and say why) if the loader is ever meant to change.
CD_CONTRACT_DIGESTS = {
    "simple": "35a2cb4d0e9391952bd8d59765252170a81b14bd043b8939b0c0152437d7f6af",
    "coupled": "4853c757fa6850ae55abc6c591890821176356d3469cdf8c4a3588cb7a98ca04",
}


def causal_dynamics_contract_selftest(subsets=("simple", "coupled"),
                                      strict: bool = True) -> dict:
    """Hash every system of each subset's default cell and check the invariants.

    Catches a silent behaviour change in ``load_causal_dynamics`` -- the failure
    mode that matters most here, because ``coupled`` support was added to a file
    that a running job is importing.
    """
    import hashlib
    out = {}
    for sub in subsets:
        h = hashlib.sha256()
        if sub == "coupled":
            pairs = [(c, s) for c in causal_dynamics_cells(
                "coupled", **CD_COUPLED_DEFAULT_CELL)
                for s in causal_dynamics_systems("coupled", c)]
        else:
            pairs = [(None, s) for s in causal_dynamics_systems(sub)]
        n = 0
        for setting, s in pairs:
            X, A, m = load_causal_dynamics(s, subset=sub, setting=setting)
            n += 1
            assert X.ndim == 2 and A.shape == (X.shape[1], X.shape[1])
            assert m["self_loops"] is True                 # both subsets score i->i
            if sub == "coupled":
                assert m["label_var"] == "adjacency_matrix_summary"
                assert m["transposed_from_native"] is True
                assert np.array_equal(A, m["A_native"].T)
                assert m["partially_observed"] and m["node_dim"] == 3
                h.update((setting + s).encode())
                h.update(np.ascontiguousarray(X, dtype="<f8").tobytes())
                h.update(np.ascontiguousarray(A, dtype="<f8").tobytes())
            else:
                assert m["label_var"] == "adjacency_matrix"
                assert m["transposed_from_native"] is False
                assert np.array_equal(A, m["A_native"])
                assert not m["partially_observed"]
                h.update(s.encode())
                h.update(np.ascontiguousarray(X, dtype="<f8").tobytes())
                h.update(np.ascontiguousarray(A, dtype="<f8").tobytes())
                h.update(np.ascontiguousarray(m["A_native"], dtype="<f8").tobytes())
                h.update(str((m["label_var"], m["self_loops"],
                              m["transposed_from_native"],
                              m["partially_observed"])).encode())
        got = h.hexdigest()
        want = CD_CONTRACT_DIGESTS.get(sub)
        out[sub] = dict(n_systems=n, digest=got, expected=want,
                        ok=(want is None or got == want))
        if strict and want and got != want:
            raise AssertionError(
                f"load_causal_dynamics({sub!r}) CHANGED: digest {got} != {want}. "
                "If the change is intended, update CD_CONTRACT_DIGESTS and say "
                "why in docs/causal_baselines.md.")
    return out


def causal_dynamics_trivial_baselines(subset: str = "coupled",
                                      settings: Optional[Iterable[str]] = None,
                                      n_draws: int = 20, seed: int = 0,
                                      limit: Optional[int] = None,
                                      kinds: Iterable[str] = CD_TRIVIAL) -> dict:
    """MEASURE the floor: label-free predictors run through eval.py's own scorer.

    Four of them, each answering a different question:

    ``random``    uniform U(0,1) scores -- a real random RANKER.  This, not the
                  benchmark's ``Null``, is the chance level for a ranking
                  metric.  The shipped ``Null_AUPRC`` comes from an all-ZEROS
                  constant, whose average precision collapses to the positive
                  rate; a random ranker scores ABOVE that whenever the graph is
                  small, because average precision is upward-biased in small
                  samples.  On ``simple`` (9 decisions per system, 68.7%
                  positive) the gap is +0.073 and every published AUPRC sits
                  below the random floor.
    ``zeros``     the benchmark's own ``Null``, recomputed, as a cross-check.
    ``identity``  predict every self-loop and NOTHING else.  A constant matrix
                  with no parameters and no data access.
    ``identity_plus_noise``  self-loops plus a random ranking off the diagonal.

    Everything is reported twice: under the benchmark's protocol (diagonal
    KEPT) and off-diagonal-only, because the split between the two is the whole
    story on ``coupled``.
    """
    if settings is None:
        settings = (causal_dynamics_cells(subset, **CD_COUPLED_DEFAULT_CELL)
                    if subset == "coupled"
                    else [CD_DEFAULT_SETTING.get(subset)])
    settings = list(settings)
    rng = np.random.default_rng(seed)
    acc: Dict[str, Dict[str, List[float]]] = {}
    dens: Dict[str, List[float]] = {"full": [], "offdiag": []}
    diag_share, n_sys, n_skip = [], 0, 0
    for s in settings:
        for X, A, meta in iter_causal_dynamics(subset, s, realisation=None,
                                               limit=limit):
            L = np.asarray(meta["A_native"], dtype=float)
            if np.all(L == 1) or np.all(L == 0):
                n_skip += 1
                continue                       # exactly what eval.py skips
            n_sys += 1
            P = L.shape[0]
            off = ~np.eye(P, dtype=bool)
            dens["full"].append(float(L.mean()))
            dens["offdiag"].append(float(L[off].mean()))
            diag_share.append(float(np.diag(L).sum() / max(L.sum(), 1.0)))
            for kind in kinds:
                n = n_draws if kind in ("random", "identity_plus_noise") else 1
                preds = _cd_trivial_preds(kind, P, n, rng)
                for tag, od in (("full", False), ("offdiag", True)):
                    if od and L[off].min() == L[off].max():
                        continue
                    r = score_causal_dynamics(preds, L, offdiag=od)
                    a = acc.setdefault(f"{kind}/{tag}", {})
                    a.setdefault("auroc", []).append(r["individual_auroc"])
                    a.setdefault("auprc", []).append(r["individual_auprc"])
    out = {"subset": subset, "n_settings": len(settings), "n_systems": n_sys,
           "n_skipped_all_ones_or_zero": n_skip, "n_draws": n_draws, "seed": seed,
           "positive_rate_full": float(np.mean(dens["full"])) if dens["full"] else float("nan"),
           "positive_rate_offdiag": float(np.mean(dens["offdiag"])) if dens["offdiag"] else float("nan"),
           "positive_rate_full_min": float(np.min(dens["full"])) if dens["full"] else float("nan"),
           "positive_rate_full_max": float(np.max(dens["full"])) if dens["full"] else float("nan"),
           "diagonal_share_of_positives": float(np.mean(diag_share)) if diag_share else float("nan"),
           "rows": {}}
    for k, a in acc.items():
        out["rows"][k] = {"auroc": float(np.mean(a["auroc"])),
                          "auroc_sd": float(np.std(a["auroc"])),
                          "auprc": float(np.mean(a["auprc"])),
                          "auprc_sd": float(np.std(a["auprc"])),
                          "n": len(a["auroc"])}
    return out


# --------------------------------------------------------------------------- #
# The audit itself.  Cached, so --tables-only regenerates every table without
# re-reading 10 GB of netCDF or re-running a single fit.
# --------------------------------------------------------------------------- #
# Each entry is fitted ONCE over the whole row and then scored under every
# (orientation x binarisation x diagonal) variant, because refitting per variant
# costs 8x for identical numbers.  `their_orientation` records which transpose
# reproduces the shipped wrapper: DYNOTEARS emits [cause, effect] and the coupled
# truth is stored [cause, effect], so THEIR dynotears is aligned here; lingam
# emits [effect, cause], so THEIR varlingam is the transposed one -- the exact
# mirror image of the `simple` split, where the same code has the same bug on the
# opposite half of the suite.
CD_AUDIT_REPRO = (
    # (label, baseline, kwargs, transpose_that_reproduces_their_wrapper)
    ("varlingam", "varlingam", {}, False),
    ("dynotears", "dynotears", dict(lambda_w=0.2, lambda_a=0.1), True),
    ("var_ols|coef|", "var_ols", dict(score="abscoef"), None),
    ("var_ols|t|", "var_ols", dict(score="tstat"), None),
)


def causal_dynamics_coupled_repro(settings: Optional[Iterable[str]] = None,
                                  entries=CD_AUDIT_REPRO, max_lag: int = 1,
                                  limit: Optional[int] = None,
                                  verbose: bool = True,
                                  cache_dir: Optional[str] = None) -> List[dict]:
    """Fit each baseline ONCE per trajectory, score it under every variant.

    Variants: orientation (as-emitted vs transposed), the shipped wrappers'
    ``abs(A) > 0`` binarisation on/off, and diagonal kept (their protocol) vs
    masked (the cross-variable problem underneath it).  Everything is scored
    against ``meta['A_native']`` -- the label exactly as stored -- because that
    is what ``eval.py`` scores against.

    The raw (n_systems, n_traj, P, P) prediction cube is cached per baseline
    under ``data/causal/cache/``, so a change to the SCORING never costs the
    fits again -- VARLiNGAM alone is ~1000 fits.
    """
    import causal_baselines as cb
    if settings is None:
        settings = causal_dynamics_cells("coupled", **CD_COUPLED_DEFAULT_CELL)
    settings = list(settings)
    cache_dir = cache_dir or CACHE
    systems = []
    for st in settings:
        for X, A, meta in iter_causal_dynamics("coupled", st, realisation=None,
                                               limit=limit):
            L = np.asarray(meta["A_native"], dtype=float)
            if np.all(L == 1) or np.all(L == 0):
                continue                       # exactly what eval.py skips
            cube = np.nan_to_num(meta["cube"].transpose(1, 0, 2))   # (N, T, P)
            cube = (cube - cube.mean(axis=(0, 1), keepdims=True)) / (
                cube.std(axis=(0, 1), keepdims=True) + 1e-8)
            systems.append((cube, L))
    labs = np.asarray([L for _, L in systems])
    out = []
    for label, base, kw, their_T in entries:
        tag = "".join(ch if ch.isalnum() else "_" for ch in label)
        npz = os.path.join(cache_dir, f"cd_coupled_repro_{tag}_lag{max_lag}"
                                      f"_n{len(systems)}.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            preds_all, fails, secs = z["preds"], int(z["fails"]), float(z["seconds"])
        else:
            t0 = time.time()
            per_system, fails = [], 0
            for cube, L in systems:
                preds = []
                for x in cube:
                    r = cb.run_baseline(base, x, max_lag=max_lag, **kw)
                    if not r.ok:
                        fails += 1
                        preds.append(np.zeros_like(L))
                    else:
                        preds.append(np.abs(np.asarray(r.A_lag[max_lag - 1], float)))
                per_system.append(np.asarray(preds))
            preds_all = np.asarray(per_system, dtype=np.float32)
            secs = float(time.time() - t0)
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(npz, preds=preds_all, labs=labs,
                                fails=np.int64(fails), seconds=np.float64(secs))
        preds_all = np.nan_to_num(np.asarray(preds_all, dtype=float),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        rec = dict(label=label, baseline=base,
                   kwargs={k: str(v) for k, v in kw.items()},
                   n_systems=int(preds_all.shape[0]),
                   n_fits=int(preds_all.shape[0] * preds_all.shape[1]),
                   fails=int(fails), seconds=secs, cache=os.path.basename(npz),
                   their_orientation=their_T, variants={})
        for tr in (False, True):
            for binz in (False, True):
                variants = []
                for i in range(preds_all.shape[0]):
                    S = preds_all[i]
                    if tr:
                        S = np.transpose(S, (0, 2, 1))
                    if binz:
                        S = (S > 0).astype(float)
                    variants.append(S)
                const = float(np.mean([float(np.mean([s.max() == s.min()
                                                      for s in S]))
                                       for S in variants]))
                for od in (False, True):
                    acc = []
                    for S, L in zip(variants, labs):
                        off = ~np.eye(L.shape[0], dtype=bool)
                        if od and L[off].max() == L[off].min():
                            continue
                        acc.append(score_causal_dynamics(S, L, offdiag=od))
                    key = (f"transpose={int(tr)}_binarise={int(binz)}"
                           f"_{'offdiag' if od else 'full'}")
                    rec["variants"][key] = {
                        k: float(np.mean([a[k] for a in acc]))
                        for k in ("individual_auroc", "individual_auprc",
                                  "joint_auroc", "joint_auprc")}
                    rec["variants"][key]["constant_prediction_frac"] = const
        out.append(rec)
        if verbose:
            # Entries with no shipped wrapper (their_orientation is None) have no
            # "their variant" to show; report the ALIGNED, unbinarised one, which
            # is the only variant that means anything for them.
            key = (f"transpose={int(bool(their_T))}_binarise=1_full"
                   if their_T is not None else "transpose=1_binarise=0_full")
            what = "their-wrapper" if their_T is not None else "aligned |coef|"
            v = rec["variants"][key]
            print(f"    {label:16s} {what:14s} AUROC "
                  f"{v['individual_auroc']:.3f} AUPRC {v['individual_auprc']:.3f} "
                  f"({rec['seconds']:.0f}s, {rec['fails']} failed fits)", flush=True)
    return out


def causal_dynamics_coupled_audit(force: bool = False, n_draws: int = 20,
                                  limit: Optional[int] = None,
                                  with_repro: bool = True,
                                  cache_path: Optional[str] = None) -> dict:
    """Everything the coupled split needs before a win on it can be believed.

    Four blocks, all cached to ``CD_AUDIT_CACHE`` as JSON:

    ``leaderboard``  the published 'Default' row RE-DERIVED from the shipped
                     ``outputs/coupled.tar.gz`` eval files, with the degeneracy
                     rate per method (what fraction of cells are a constant
                     prediction, i.e. AUROC exactly 0.5 by construction).
    ``floor``        the MEASURED chance floor per tier, full and off-diagonal,
                     plus the trivial identity predictor.
    ``repro``        our own from-scratch runs under eval.py's protocol, in
                     their orientation and in the fixed one, with and without
                     their ``abs(A) > 0`` binarisation.
    ``orientation``  the evidence that the coupled ground truth is stored
                     [cause, effect] -- the root-node column/row test, run on
                     every system in the row rather than asserted from a
                     docstring.
    """
    path = cache_path or CD_AUDIT_CACHE
    if os.path.exists(path) and not force:
        with open(path) as fh:
            d = json.load(fh)
        # Blocks added after a cache was written are back-filled in place rather
        # than forcing a 20-minute re-run of everything else.
        changed = False
        lb = d.get("leaderboard", {})
        if lb and "auroc_above_half_frac" not in next(iter(lb.values()), {}):
            # per-system spread was added after this cache was written; it is
            # ~10 s of small netCDF reads, so refresh it rather than print nan
            d["leaderboard"] = causal_dynamics_leaderboard(
                "coupled", settings=d.get("cells"))
            changed = True
        if "wrappers" not in d:
            d["wrappers"] = dict(source_audit=CD_WRAPPER_AUDIT,
                                 probe=causal_dynamics_wrapper_probe())
            changed = True
        if "identity_paired" not in d:
            d["identity_paired"] = causal_dynamics_identity_paired("coupled")
            changed = True
        if "rows_vs_identity" not in d:
            d["rows_vs_identity"] = causal_dynamics_all_rows_vs_identity()
            changed = True
        if "repro_rows" not in d:
            # the two coupled rows where the diagonal loosens its grip, so our
            # own bar on them is what a CAFE-CD number would be measured
            # against.  Re-uses the cached fit cubes when they exist.
            d["repro_rows"] = {r: dict(
                repro=causal_dynamics_coupled_repro(
                    causal_dynamics_row_cells(r), verbose=False),
                floor=causal_dynamics_trivial_baselines(
                    "coupled", causal_dynamics_row_cells(r)))
                for r in CD_AUDIT_ROWS}
            changed = True
        if changed:
            _cd_write_json(path, d)
        return d
    cells = causal_dynamics_cells("coupled", **CD_COUPLED_DEFAULT_CELL)
    out: Dict[str, Any] = {
        "subset": "coupled", "cells": cells,
        "published": CD_PUBLISHED["coupled"],
        "published_source": CD_LEADERBOARD_SRC,
        "n_systems_per_cell": {c: len(causal_dynamics_systems("coupled", c))
                               for c in cells},
    }
    out["leaderboard"] = causal_dynamics_leaderboard("coupled", settings=cells)
    out["floor"] = {
        "coupled": causal_dynamics_trivial_baselines("coupled", cells,
                                                     n_draws=n_draws, limit=limit),
        "simple": causal_dynamics_trivial_baselines(
            "simple", [CD_DEFAULT_SETTING["simple"]], n_draws=n_draws, limit=limit),
    }
    out["orientation"] = causal_dynamics_orientation_evidence("coupled", cells)
    out["wrappers"] = dict(source_audit=CD_WRAPPER_AUDIT,
                           probe=causal_dynamics_wrapper_probe())
    out["identity_paired"] = causal_dynamics_identity_paired("coupled")
    out["rows_vs_identity"] = causal_dynamics_all_rows_vs_identity()
    out["repro_rows"] = {r: dict(
        repro=causal_dynamics_coupled_repro(causal_dynamics_row_cells(r),
                                            verbose=False),
        floor=causal_dynamics_trivial_baselines("coupled",
                                                causal_dynamics_row_cells(r)))
        for r in CD_AUDIT_ROWS}
    if with_repro:
        out["repro"] = causal_dynamics_coupled_repro(cells, limit=limit)
    _cd_write_json(path, out)
    return out


def _cd_write_json(path: str, obj) -> None:
    """Atomic: temp file in the same dir, then rename over the target."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# The five published COUPLED rows.  Each varies ONE axis of the Default cell and
# holds the rest fixed; "Noise" is every non-zero noise level, so it is a pin on
# everything EXCEPT noise.  Transcribed from website/leaderboard.md.
# The rows worth re-running our own baselines on: the two where the identity
# matrix does NOT already win, i.e. the only ones where a result can mean
# something (see causal_dynamics_all_rows_vs_identity).
CD_AUDIT_ROWS = ("Confounder", "Lag")

CD_COUPLED_ROWS = {
    "Default":     dict(noise="0.00", systems="10", confounder="False",
                        standardize="False", timelag="0"),
    "Noise":       dict(systems="10", confounder="False",
                        standardize="False", timelag="0"),   # noise != 0.00
    "Confounder":  dict(noise="0.00", systems="10", confounder="True",
                        standardize="False", timelag="0"),
    "Lag":         dict(noise="0.00", systems="10", confounder="False",
                        standardize="False", timelag="1"),
    "Standardize": dict(noise="0.00", systems="10", confounder="False",
                        standardize="True", timelag="0"),
}

CD_PUBLISHED_COUPLED_ROWS = {   # AUROC / AUPRC, website/leaderboard.md Table 1
    "Default":     dict(pcmciplus=(.68, .27), fpcmci=(.67, .24), varlingam=(.57, .18),
                        dynotears=(.66, .32), ngc_lstm=(.50, .16), tsci=(.69, .36),
                        cutsplus=(.50, .16)),
    "Noise":       dict(pcmciplus=(.66, .30), fpcmci=(.62, .25), varlingam=(.58, .19),
                        dynotears=(.62, .29), ngc_lstm=(.50, .16), tsci=(.52, .18),
                        cutsplus=(.50, .16)),
    "Confounder":  dict(pcmciplus=(.56, .20), fpcmci=(.57, .19), varlingam=(.51, .17),
                        dynotears=(.49, .17), ngc_lstm=(.50, .17), tsci=(.49, .18),
                        cutsplus=(.50, .17)),
    "Lag":         dict(pcmciplus=(.55, .23), fpcmci=(.55, .23), varlingam=(.52, .21),
                        dynotears=(.52, .22), ngc_lstm=(.50, .20), tsci=(.53, .23),
                        cutsplus=(.50, .20)),
    "Standardize": dict(pcmciplus=(.69, .27), fpcmci=(.67, .23), varlingam=(.57, .18),
                        dynotears=(.68, .34), ngc_lstm=(.50, .15), tsci=(.69, .35),
                        cutsplus=(.51, .16)),
}


def causal_dynamics_row_cells(row: str) -> List[str]:
    """The setting directories a published coupled row averages over."""
    pin = CD_COUPLED_ROWS[row]
    out = []
    for s in causal_dynamics_settings("coupled"):
        d = parse_cd_setting(s)
        if not all(d.get(k) == v for k, v in pin.items()):
            continue
        if row == "Noise" and d.get("noise") == "0.00":
            continue                       # the Noise row is the non-zero levels
        out.append(s)
    return out


def causal_dynamics_all_rows_vs_identity() -> Dict[str, dict]:
    """Run the identity-matrix paired test on EVERY published coupled row.

    This is what turns "the coupled leaderboard is beaten by a constant" from a
    slogan into a usable instruction.  The diagonal's share of the positive
    class is not constant across rows: the Confounder and Lag rows dilute it,
    the identity predictor drops from ~0.68 to ~0.55 there, and PCMCI+ then
    clears it on BOTH metrics.  Those are the rows worth competing on.
    """
    out = {}
    for row in CD_COUPLED_ROWS:
        cells = causal_dynamics_row_cells(row)
        paired = causal_dynamics_identity_paired("coupled", cells)
        best = None
        for m, r in paired.items():
            if r["auroc"] > r["identity_auroc"] and r["p_auroc"] < 0.05:
                if best is None or r["auroc"] > paired[best]["auroc"]:
                    best = m
        out[row] = dict(cells=cells, n_cells=len(cells), paired=paired,
                        identity_auroc=(list(paired.values())[0]["identity_auroc"]
                                        if paired else float("nan")),
                        identity_auprc=(list(paired.values())[0]["identity_auprc"]
                                        if paired else float("nan")),
                        beats_identity_on_auroc=sorted(
                            m for m, r in paired.items()
                            if r["auroc"] > r["identity_auroc"] and r["p_auroc"] < 0.05),
                        beats_identity_on_both=sorted(
                            m for m, r in paired.items()
                            if r["auroc"] > r["identity_auroc"] and r["p_auroc"] < 0.05
                            and r["auprc"] > r["identity_auprc"] and r["p_auprc"] < 0.05),
                        best=best,
                        published=CD_PUBLISHED_COUPLED_ROWS.get(row, {}))
    return out


def causal_dynamics_identity_paired(subset: str = "coupled",
                                    settings: Optional[Iterable[str]] = None
                                    ) -> Dict[str, dict]:
    """PAIRED test: does each SHIPPED method beat a constant identity matrix?

    Comparing published means to a floor is weak; the shipped eval stores a
    per-system Individual AUROC/AUPRC, and the identity predictor can be scored
    on exactly the same systems, so the comparison can be paired system by
    system and tested (Wilcoxon signed-rank).  That is the difference between
    "the floor is close" and "no published method beats the floor".

    Returns, per method: the two paired means, the two-sided p-values, and the
    fraction of systems on which the method actually wins.
    """
    import h5py
    from scipy.stats import wilcoxon
    from sklearn.metrics import average_precision_score, roc_auc_score
    if settings is None:
        settings = (causal_dynamics_cells(subset, **CD_COUPLED_DEFAULT_CELL)
                    if subset == "coupled"
                    else [CD_DEFAULT_SETTING.get(subset)])
    ident: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for s in settings:
        d = os.path.join(CD_ROOT, subset, s, "data")
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".nc"):
                continue
            with h5py.File(os.path.join(d, fn), "r") as f:
                var = ("adjacency_matrix_summary"
                       if "adjacency_matrix_summary" in f else "adjacency_matrix")
                L = np.asarray(f[var][...], dtype=float)
            if np.all(L == 1) or np.all(L == 0):
                continue
            E = np.eye(L.shape[0]).ravel()
            ident[(s, fn[:-3])] = (float(roc_auc_score(L.ravel(), E)),
                                   float(average_precision_score(L.ravel(), E)))
    acc: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for s in settings:
        ev = os.path.join(CD_ROOT, subset, s, "eval")
        if not os.path.isdir(ev):
            continue
        for m in sorted(os.listdir(ev)):
            dm = os.path.join(ev, m)
            if not os.path.isdir(dm):
                continue
            for fn in sorted(os.listdir(dm)):
                if not fn.endswith(".nc") or (s, fn[:-3]) not in ident:
                    continue
                with h5py.File(os.path.join(dm, fn), "r") as f:
                    a = float(f["Individual_AUROC"][()])
                    p = float(f["Individual_AUPRC"][()])
                ia, ip = ident[(s, fn[:-3])]
                acc.setdefault(m, []).append((a, p, ia, ip))
    out = {}
    for m, rows in acc.items():
        r = np.asarray(rows, dtype=float)
        if len(r) < 5:
            continue
        out[m] = dict(
            n=len(r), auroc=float(r[:, 0].mean()), identity_auroc=float(r[:, 2].mean()),
            auprc=float(r[:, 1].mean()), identity_auprc=float(r[:, 3].mean()),
            p_auroc=float(wilcoxon(r[:, 0], r[:, 2]).pvalue),
            p_auprc=float(wilcoxon(r[:, 1], r[:, 3]).pvalue),
            win_rate_auroc=float(np.mean(r[:, 0] > r[:, 2])),
            win_rate_auprc=float(np.mean(r[:, 1] > r[:, 3])),
            beats_identity=bool(r[:, 0].mean() > r[:, 2].mean()
                                and wilcoxon(r[:, 0], r[:, 2]).pvalue < 0.05))
    return out


# --------------------------------------------------------------------------- #
# The shipped wrapper suite, audited against its own source.  ORIENTATION is
# read off each wrapper's last line; the coupled ground truth is [cause, effect]
# (see causal_dynamics_orientation_evidence), so a wrapper that emits
# [effect, cause] is TRANSPOSED on coupled -- and, because `simple` stores the
# opposite convention, the SAME code is transposed on the OTHER half of the
# suite there.  This is a source audit, not a guess: every quote is from
# CausalDynamics src/causaldynamics/baselines/*.py.
# --------------------------------------------------------------------------- #
CD_WRAPPER_AUDIT = {
    "pcmciplus": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="results['p_matrix'][..., -1] < pc_alpha; tigramite p_matrix[i,j,tau] "
            "is the p-value for X_i(t-tau) -> X_j(t)",
        note="score is a BOOLEAN, so AUROC sees two tiers, not a ranking"),
    "fpcmci": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="results.get_skeleton()[-1]",
        note="returns a CONSTANT matrix on 23% of coupled Default cells "
             "(AUROC exactly 0.500 there, by construction)"),
    "varlingam": dict(
        emits="[effect, cause]", ok_on_coupled=False, ok_on_simple=True,
        src="lingam.VARLiNGAM.adjacency_matrices_[tau_max]; abs(A) > 0. "
            "lingam's adjacency_matrices_[k][i, j] is the effect of x_j(t-k) "
            "on x_i(t)",
        note="TRANSPOSED on coupled.  The abs(A)>0 binarisation is also "
             "lossy: it discards the ranking that AUROC/AUPRC need"),
    "rcd": dict(
        emits="[effect, cause]", ok_on_coupled=False, ok_on_simple=True,
        src="causallearn RCD.adjacency_matrix_ (lingam convention), "
            "nan_to_num only",
        note="TRANSPOSED on coupled; constant on 28% of coupled Default cells"),
    "dynotears": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="adj_matrix[i, j] = weight for i at lag tau -> j at lag 0; "
            "abs(adj) > 0",
        note="CORRECT on coupled.  Its below-chance .43 on simple is the "
             "SAME code scored against simple's [effect, cause] truth"),
    "tsci": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="adj_matrix[source, target] = corr; > corr_thres",
        note="np.random.randn init is harmless -- every (source, target) is "
             "overwritten by the double loop"),
    "ngc_lstm": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="cLSTM.GC().T -- GC()[i, j] is 'j Granger-causes i', so .T is "
            "[cause, effect]",
        note="at chance on coupled (0.500 +- 0.005) without being constant"),
    "cutsplus": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="min-max normalise, then > corr_thres",
        note="constant on 56% of coupled Default cells"),
    "tcdf": dict(
        emits="[cause, effect]", ok_on_coupled=True, ok_on_simple=False,
        src="adj[j, i] = True means j -> i",
        note="crashes on most systems: 3 of 100 eval files on the coupled "
             "Default row, 9 429 of 14 089 tier-wide"),
    "grasp": dict(
        emits="ENDPOINT MATRIX, not an adjacency", ok_on_coupled=False,
        ok_on_simple=False,
        src="self.adj_matrix = G.graph, where G is a causal-learn GeneralGraph",
        note="THIRD DEFECT, previously unreported and affecting BOTH subsets: "
             "GeneralGraph.graph is the endpoint encoding (graph[j,i]=1 and "
             "graph[i,j]=-1 means i -> j), so it contains -1 entries.  Fed "
             "straight to roc_auc_score as a SCORE, a -1 ranks BELOW the 0 "
             "that means 'no edge': every detected edge's tail endpoint is "
             "scored as more confidently absent than an untested pair.  See "
             "causal_dynamics_wrapper_probe for the measured cost"),
}


def causal_dynamics_wrapper_probe(n_systems: int = 10, setting: Optional[str] = None,
                                  subset: str = "coupled") -> dict:
    """LIVE probe of the two defects that can be measured without re-running the suite.

    ``grasp`` -- run causal-learn's GRaSP exactly as the shipped wrapper does
    and score (i) its raw ``G.graph`` endpoint matrix, which is what upstream
    submits, against (ii) ``abs(G.graph)``, which is what it presumably meant.
    The difference is the cost of the encoding bug.

    ``varlingam`` -- the rate at which ``abs(adjacency_matrices_[tau_max]) > 0``
    is ALL-ONES, i.e. a constant prediction whose AUROC is 0.500 by
    construction.  On ``simple`` this is 28.3% of cells; the coupled rate is
    measured here.
    """
    from sklearn.metrics import roc_auc_score
    if setting is None:
        setting = (causal_dynamics_cells(subset, **CD_COUPLED_DEFAULT_CELL)[1]
                   if subset == "coupled" else CD_DEFAULT_SETTING[subset])
    raw, absd, frac_neg, ones = [], [], [], []
    n = 0
    for X, A, meta in iter_causal_dynamics(subset, setting, realisation=None,
                                           limit=n_systems):
        L = np.asarray(meta["A_native"], dtype=float)
        if np.all(L == 1) or np.all(L == 0):
            continue
        cube = np.nan_to_num(meta["cube"].transpose(1, 0, 2))
        cube = (cube - cube.mean(axis=(0, 1), keepdims=True)) / (
            cube.std(axis=(0, 1), keepdims=True) + 1e-8)
        x = cube[0]
        n += 1
        try:                                    # MIT, in-process
            from causallearn.search.PermutationBased.GRaSP import grasp as _grasp
            g = np.asarray(_grasp(x).graph, dtype=float)
            raw.append(roc_auc_score(L.ravel(), g.ravel()))
            absd.append(roc_auc_score(L.ravel(), np.abs(g).ravel()))
            frac_neg.append(float((g < 0).mean()))
        except Exception:
            pass
        try:                                    # MIT, in-process
            import lingam
            m = lingam.VARLiNGAM(lags=1)
            m.fit(x)
            ones.append(float(np.all(np.abs(m.adjacency_matrices_[1]) > 0)))
        except Exception:
            pass
    f = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(subset=subset, setting=setting, n_systems=n,
                grasp_auroc_as_shipped=f(raw), grasp_auroc_abs=f(absd),
                grasp_cost=f(absd) - f(raw) if raw and absd else float("nan"),
                grasp_frac_negative_entries=f(frac_neg),
                varlingam_all_ones_rate=f(ones), n_grasp=len(raw), n_varlingam=len(ones),
                note="grasp_auroc_as_shipped feeds the raw -1/0/1 endpoint "
                     "matrix to roc_auc_score, exactly as baselines/perm.py "
                     "does; grasp_auroc_abs takes abs() first")


def causal_dynamics_orientation_evidence(subset: str = "coupled",
                                         settings: Optional[Iterable[str]] = None
                                         ) -> dict:
    """Is the stored label [cause, effect] or [effect, cause]?  MEASURE it.

    ``root_nodes`` in every CausalDynamics file marks the nodes with NO
    INCOMING edges (``scm.get_root_nodes_mask`` is ``~A.T.bool().any(dim=1)``,
    i.e. an empty COLUMN).  So in [cause, effect] storage a root node has an
    empty off-diagonal COLUMN and a possibly-full ROW; in [effect, cause] it is
    the other way round.  Counting both across every system in the row settles
    it from the data instead of from a docstring.
    """
    if settings is None:
        settings = (causal_dynamics_cells(subset, **CD_COUPLED_DEFAULT_CELL)
                    if subset == "coupled"
                    else [CD_DEFAULT_SETTING.get(subset)])
    col_empty = row_empty = n_roots = n_files = 0
    diag_sum = 0.0
    for s in settings:
        for X, A, meta in iter_causal_dynamics(subset, s, realisation=None):
            L = np.asarray(meta["A_native"], dtype=float)
            rn = meta.get("root_nodes")
            if rn is None:
                continue
            n_files += 1
            diag_sum += float(np.diag(L).sum()) / max(L.shape[0], 1)
            off = L.copy()
            np.fill_diagonal(off, 0.0)
            for i in np.flatnonzero(np.asarray(rn, dtype=bool)):
                n_roots += 1
                col_empty += int(off[:, i].sum() == 0)
                row_empty += int(off[i, :].sum() == 0)
    f = lambda v: (v / n_roots) if n_roots else float("nan")
    return dict(subset=subset, n_files=n_files, n_root_nodes=n_roots,
                frac_root_column_empty=f(col_empty),
                frac_root_row_empty=f(row_empty),
                verdict=("[cause, effect]" if col_empty > row_empty
                         else "[effect, cause]"),
                mean_diagonal_density=(diag_sum / n_files) if n_files else float("nan"),
                rule="scm.get_root_nodes_mask = ~A.T.bool().any(dim=1): a root "
                     "node has NO INCOMING edges, so in [cause, effect] its "
                     "off-diagonal COLUMN is empty")


# =========================================================================== #
# 3.  CauseMe -- FORMAT ONLY.  Nothing here uploads anything.
# =========================================================================== #
CAUSEME_SPEC = dict(
    url=CAUSEME_URL,
    what_an_entry_requires=[
        "1. Register the METHOD on causeme.uv.es (name, parameters, "
        "description, optional paper/code URL). Uploading a code zip whose "
        "module exposes `my_method(data, **params) -> (val_matrix, p_matrix, "
        "lag_matrix)` is what lets CauseMe mark a result 'Validated'.",
        "2. Registration returns a `method_sha` that identifies the method.",
        "3. Download an experiment zip from Data and Models. It contains a few "
        "hundred .txt datasets, each loadable with np.loadtxt -> (T, N). "
        "Results MUST be ordered by sorted(filename).",
        "4. Run the method on every dataset, collect the matrices, and write "
        "ONE bz2-compressed JSON dict.",
        "5. Upload that .json.bz2 on 'My Results'. Uploads per user are rate "
        "limited.",
    ],
    required_fields=["method_sha", "parameter_values", "model", "experiment",
                     "scores"],
    optional_fields=["pvalues", "lags", "runtimes"],
    orientation="A_ij is the score for the link i -> j (CAUSE row, EFFECT "
                "column) -- the TRANSPOSE of this repo's convention, so a "
                "BaselineResult must be transposed before submission (see "
                "causeme_matrix_from_result).",
    flattening="row-major (C-order) flatten of each (N, N) matrix; the top "
               "level of 'scores' is a list over datasets",
    scores="non-negative reals, higher = more confident; 0 = certain absence",
    pvalues="optional, same shape; thresholded at 5% for the binary metrics",
    lags="optional, same shape; positive int lag, 0 for instantaneous. Only "
         "read where the link is predicted present.",
    metrics=["AUC (ROC) over all N*N entries EXCLUDING SELF-LINKS, pooled over "
             "the B datasets of an experiment",
             "FPR / TPR from the p-values at alpha=0.05",
             "F-measure with beta=0.5 (precision-weighted) from the p-values",
             "TLR (true lag rate) among correctly predicted links"],
    self_loops=False,
    file_format="bz2-compressed JSON: "
                "bz2.BZ2File(path,'w').write(json.dumps(results).encode('latin1'))",
    reference_baseline="the shipped example method is a statsmodels VAR "
                       "(trend='nc') on z-scored data that stores, per ordered "
                       "pair, |coef| at the lag with the smallest p-value, with "
                       "p-values Bonferroni-multiplied by maxlags",
    note="This harness NEVER uploads. causeme_build_submission writes a local "
         "file only.",
)


def causeme_submission_spec() -> dict:
    """The CauseMe submission contract, recorded for reference."""
    return dict(CAUSEME_SPEC)


def causeme_matrix_from_result(res, how: str = "max"):
    """(scores, pvalues, lags) in CAUSEME orientation (A_ij = i -> j).

    ``res`` is a ``causal_baselines.BaselineResult``.  Our arrays are
    [effect, cause]; CauseMe wants [cause, effect], so everything is
    transposed.  Self-links are zeroed (CauseMe excludes them from the AUC
    anyway) and the lag matrix reports the argmax lag of the score.
    """
    A = np.abs(np.asarray(res.A_lag, dtype=float))
    C = np.abs(np.asarray(res.A_contemp, dtype=float))
    stack = np.concatenate([C[None, ...], A], axis=0)       # lag 0, 1, ..., K
    if how == "max":
        best = stack.max(axis=0)
        lag = stack.argmax(axis=0)
    elif how == "sum":
        best = stack.sum(axis=0)
        lag = stack.argmax(axis=0)
    else:
        raise ValueError(how)
    scores = best.T.copy()                                  # -> [cause, effect]
    lags = lag.T.astype(np.uint8).copy()
    np.fill_diagonal(scores, 0.0)
    pv = None
    sup = np.concatenate([np.asarray(res.support_contemp)[None, ...],
                          np.asarray(res.support_lag)], axis=0).any(axis=0)
    pv = np.where(sup.T, 0.0, 1.0)
    np.fill_diagonal(pv, 1.0)
    return scores, pv, lags


def causeme_build_submission(scores: List[np.ndarray], model: str,
                             experiment: str, method_sha: str,
                             parameter_values: str,
                             pvalues: Optional[List[np.ndarray]] = None,
                             lags: Optional[List[np.ndarray]] = None,
                             runtimes: Optional[List[float]] = None,
                             out_path: Optional[str] = None) -> dict:
    """Build (and optionally write) a CauseMe results file LOCALLY.

    Does NOT contact causeme.uv.es.  Matrices must already be in CauseMe
    orientation (see causeme_matrix_from_result) and ordered exactly like
    ``sorted(zip_ref.namelist())`` of the experiment archive.
    """
    results: Dict[str, Any] = {
        "method_sha": method_sha,
        "parameter_values": parameter_values,
        "model": model,
        "experiment": experiment,
        "scores": [np.asarray(s, dtype=float).ravel().tolist() for s in scores],
    }
    if pvalues is not None:
        results["pvalues"] = [np.asarray(p, dtype=float).ravel().tolist()
                              for p in pvalues]
    if lags is not None:
        results["lags"] = [np.asarray(l, dtype=int).ravel().tolist() for l in lags]
    if runtimes is not None:
        results["runtimes"] = [float(r) for r in runtimes]
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with bz2.BZ2File(out_path, "w") as fh:
            fh.write(bytes(json.dumps(results), encoding="latin1"))
    return results


# =========================================================================== #
# 4.  Benchmark-native SCORING + published-number reproduction
# =========================================================================== #
# These are verbatim re-implementations of each BENCHMARK'S OWN scoring code
# (CausalRivers tools/scoring_tools.py, CausalDynamics src/causaldynamics/
# score.py), kept here rather than in the general metric layer precisely so the
# protocol -- diagonal handling, pooling, threshold sweep -- can be audited
# against the upstream source line by line.
# =========================================================================== #
def _offdiag(M):
    P = M.shape[0]
    return M[~np.eye(P, dtype=bool)]


def score_causal_rivers(preds, labels, remove_diagonal: bool = True) -> dict:
    """CausalRivers ``tools.scoring_tools.score``.

    preds/labels: iterables of (P, P) arrays in OUR convention [effect, cause].
    ``remove_diagonal=True`` (their default) drops the self-loops before
    scoring, so an n-node sample contributes n*(n-1) decisions.

    Returns
      individual_auroc  mean over samples of per-sample AUROC -- THE headline
                        number of Table 3 / Figure 3a
      joint_auroc       pooled AUROC over every decision
      f1max             best F1 over all thresholds of the POOLED prediction
                        (their 'Max F1')
      f1max_individual  mean over samples of each sample's own best F1
                        (their 'Max individual F1').  This is systematically
                        HIGHER than `f1max` because every sample gets its own
                        oracle threshold, and it is the variant that matches
                        the magnitudes printed in Table 7 -- see
                        docs/causal_baselines.md.
    """
    from sklearn.metrics import precision_recall_curve, roc_auc_score

    def _best_f1(y, p):
        pr, rc, _ = precision_recall_curve(y, p)
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = 2 * rc * pr / (rc + pr)
        return float(np.nanmax(f1))

    aur, f1i, yl, pl = [], [], [], []
    for S, A in zip(preds, labels):
        y = (_offdiag(A) if remove_diagonal else np.asarray(A).ravel())
        p = (_offdiag(S) if remove_diagonal else np.asarray(S).ravel())
        y = np.asarray(y, dtype=int)
        p = np.nan_to_num(np.asarray(p, dtype=float))
        yl.append(y)
        pl.append(p)
        if len(np.unique(y)) > 1:           # AUROC undefined for empty samples
            aur.append(roc_auc_score(y, p))
        if y.sum() > 0:                     # F1 undefined with no positives
            f1i.append(_best_f1(y, p))
    Y, P_ = np.concatenate(yl), np.concatenate(pl)
    return dict(individual_auroc=float(np.mean(aur)) if aur else float("nan"),
                joint_auroc=float(roc_auc_score(Y, P_)),
                f1max=_best_f1(Y, P_),
                f1max_individual=float(np.mean(f1i)) if f1i else float("nan"),
                n_samples=len(yl), n_scored=len(aur))


def score_causal_dynamics(preds, labels, offdiag: bool = False) -> dict:
    """CausalDynamics ``src/causaldynamics/score.py``.

    preds: (N, P, P) -- one estimated graph per realisation of ONE system.
    labels: (P, P).  The diagonal is KEPT (their ground truth is the Jacobian
    sparsity pattern, whose diagonal is usually 1).

    ``offdiag=True`` is NOT their protocol.  It masks the diagonal and scores
    only the P*(P-1) cross-variable decisions.  It exists because on the
    coupled split the diagonal is 63% dense and carries 40.8% of the whole
    positive class, so the benchmark's headline number is dominated by
    self-loop detection -- a constant identity matrix scores 0.682/0.353 on it.
    The default is FALSE and reproduces the shipped numbers exactly; anything
    comparing against a published CausalDynamics figure must leave it off.

    Non-finite predictions are mapped to 0, matching eval.py's behaviour: it
    wraps ``score()`` in a bare ``except`` that falls back to an all-zero
    prediction, so a method that emits inf is scored as if it had predicted
    nothing rather than crashing the cell.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    preds = np.nan_to_num(np.asarray(preds, dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
    L = np.repeat(np.asarray(labels, dtype=float)[None], preds.shape[0], 0)
    if offdiag:
        m = ~np.eye(L.shape[-1], dtype=bool)
        L = L[:, m]
        preds = preds[:, m]
    else:
        L = L.reshape(len(L), -1)
        preds = preds.reshape(len(preds), -1)
    ia = [roc_auc_score(L[i], preds[i]) for i in range(len(L))]
    ip = [average_precision_score(L[i], preds[i]) for i in range(len(L))]
    return dict(individual_auroc=float(np.mean(ia)),
                individual_auprc=float(np.mean(ip)),
                joint_auroc=float(roc_auc_score(L.ravel(), preds.ravel())),
                joint_auprc=float(average_precision_score(L.ravel(), preds.ravel())),
                null_auroc=float(roc_auc_score(L.ravel(), np.zeros(L.size))),
                n_realisations=int(preds.shape[0]))


# The protocol that actually reproduces the published CausalRivers VAR number.
# NOTE the two rescaling switches: `normalize` is the BENCHMARK's min-max step
# and `standardize` is the WRAPPER's z-scoring.  BOTH must be off -- see
# docs/causal_baselines.md; turning either on takes VAR from ~0.85 to ~0.42
# (below chance), because the raw discharge magnitude is the signal.
# Reproduces 0.8493 against the published 0.85 (see docs/causal_baselines.md).
# THREE switches all have to be right, and each one alone costs a lot:
#   normalize=False   the benchmark's min-max step OFF
#   standardize=False the wrapper's z-scoring OFF   (either one on -> ~0.42)
#   contemp=False     no contemporaneous channel    (on -> 0.836)
# because CausalRivers' VAR baseline is coefficients-only, and the raw
# discharge MAGNITUDE (water volume grows downstream) is the signal.
CR_REPRO_CELL = dict(region="flood_full", n_vars=5, strategy="random",
                     resolution="6h", normalize=False, max_lag=3,
                     baseline="var_ols",
                     baseline_kw=dict(score="abscoef", standardize=False,
                                      contemp=False))


def reproduce_causal_rivers(baseline: str = "var_ols", *, region="flood_full",
                            n_vars=5, strategy="random", resolution="6h",
                            normalize=False, max_lag=3, limit=None,
                            summary_how="max", timeout_s=120.0,
                            verbose=True, **baseline_kw) -> dict:
    """Run ONE baseline over a whole CausalRivers graph set under the paper's
    protocol and return its scores.  Defaults are CR_REPRO_CELL."""
    import causal_baselines as cb
    preds, labs = [], []
    fails = 0
    for X, A, meta in iter_causal_rivers(region, n_vars=n_vars, strategy=strategy,
                                         resolution=resolution,
                                         normalize=normalize, limit=limit):
        if X.shape[0] < max_lag * 5 + 10:
            fails += 1
            S = np.zeros_like(A)
        else:
            r = cb.run_baseline(baseline, X, max_lag=max_lag,
                                timeout_s=timeout_s, **baseline_kw)
            if not r.ok:
                fails += 1
                S = np.zeros_like(A)
            else:
                S = cb.summary_graph(r, how=summary_how, include_contemp=True)
        preds.append(S)
        labs.append(A)
    out = score_causal_rivers(preds, labs, remove_diagonal=True)
    out.update(baseline=baseline, region=region, n_vars=n_vars,
               strategy=strategy, resolution=resolution, normalize=normalize,
               max_lag=max_lag, fails=fails, **{
                   k: v for k, v in baseline_kw.items()
                   if isinstance(v, (str, bool, int, float, type(None)))})
    key = ("auroc", f"{region}_{strategy}_{n_vars}"
           if region == "flood_full" else f"{strategy}_{n_vars}")
    pub = CR_PUBLISHED.get(key, {})
    out["published_auroc"] = pub.get(
        {"var_ols": "var", "varlingam": "varlingam", "dynotears": "dynotears",
         "pcmci": "pcmci", "pcmciplus": "pcmci"}.get(baseline, baseline))
    if verbose:
        p = out["published_auroc"]
        print(f"{baseline:16s} individual_auroc={out['individual_auroc']:.3f} "
              f"joint={out['joint_auroc']:.3f} f1max={out['f1max']:.3f} "
              f"f1max_ind={out['f1max_individual']:.3f} "
              f"n={out['n_samples']} fails={fails}"
              + (f"   published={p:.2f}" if p else "   published=?"))
    return out


def reproduce_causal_dynamics(baseline: str = "varlingam", *, subset="simple",
                              setting=None, settings=None,
                              max_lag=1, binarise=True, transpose=False,
                              limit=None, verbose=True, offdiag=False,
                              **baseline_kw) -> dict:
    """Run ONE baseline over a CausalDynamics setting under eval.py's protocol.

    ``binarise`` reproduces the shipped baselines' ``abs(A) > 0`` step;
    ``transpose`` flips our [effect, cause] estimate into [cause, effect].

    Scoring is against ``meta['A_native']`` -- the label EXACTLY as stored --
    because that is what eval.py scores against.  Our own baselines emit
    [effect, cause], so on the coupled subset (native labels are
    [cause, effect]) ``transpose=True`` is the ALIGNED setting and
    ``transpose=False`` is the mis-oriented one; on simple it is the reverse.
    ``settings`` runs several setting directories and pools them, which is how
    a published leaderboard ROW (e.g. the 6-cell coupled "Default") is
    reproduced.
    """
    import causal_baselines as cb
    if settings is None:
        settings = [setting if setting is not None
                    else CD_DEFAULT_SETTING.get(subset)]
    rows = []
    for st in settings:
        for X, A, meta in iter_causal_dynamics(subset, st, realisation=None,
                                               limit=limit):
            L = meta["A_native"]           # score against the stored label
            if np.all(L == 1) or np.all(L == 0):
                continue                   # exactly what eval.py skips
            cube = np.nan_to_num(meta["cube"].transpose(1, 0, 2))   # (N, T, P)
            cube = (cube - cube.mean(axis=(0, 1), keepdims=True)) / (
                cube.std(axis=(0, 1), keepdims=True) + 1e-8)
            preds = []
            for x in cube:
                r = cb.run_baseline(baseline, x, max_lag=max_lag, **baseline_kw)
                S = (np.zeros_like(L) if not r.ok
                     else np.abs(r.A_lag[max_lag - 1]))
                if binarise:
                    S = (S > 0).astype(float)
                if transpose:
                    S = S.T
                preds.append(S)
            if offdiag and L[~np.eye(L.shape[0], dtype=bool)].ptp() == 0:
                continue        # no off-diagonal contrast left to score
            rows.append(score_causal_dynamics(np.array(preds), L,
                                              offdiag=offdiag))
    out = {k: float(np.mean([r[k] for r in rows]))
           for k in ("individual_auroc", "individual_auprc", "joint_auroc",
                     "joint_auprc")}
    out.update(baseline=baseline, subset=subset, offdiag=bool(offdiag),
               setting=settings[0] if len(settings) == 1 else f"{len(settings)} cells",
               n_systems=len(rows), binarise=binarise, transpose=transpose,
               **{k: v for k, v in baseline_kw.items()
                  if isinstance(v, (str, bool, int, float, type(None)))})
    if verbose:
        pub = CD_PUBLISHED.get(subset, {}).get(
            {"var_ols": None}.get(baseline, baseline))
        extra = " ".join(f"{k}={v}" for k, v in baseline_kw.items()
                         if isinstance(v, (str, bool, int, float, type(None))))
        print(f"{baseline:11s} bin={binarise} T={transpose} {extra:32s} "
              f"AUROC={out['individual_auroc']:.3f} "
              f"AUPRC={out['individual_auprc']:.3f} n={len(rows)}"
              + (f"   published={pub}" if pub else ""))
    return out


# =========================================================================== #
# registry + CLI
# =========================================================================== #
DATASETS_CAUSAL = {
    "causal_rivers": load_causal_rivers,
    "causal_dynamics": load_causal_dynamics,
}

_DESC = {
    "causal_rivers": "River discharge, E-Germany/Bavaria/Elbe-flood; ICLR 2025; "
                     "thousands of ground-truth subgraphs; self-loops NOT scored",
    "causal_dynamics": "Coupled ODE/SDE dynamical systems; NeurIPS 2025 D&B; "
                       "Jacobian ground truth; self-loops ARE scored",
}


def main():
    _ensure_dirs()
    print("CausalRivers   :", "present" if causal_rivers_available()
          else f"absent (fetch_causal_rivers(); {CR_ZIP_URL})")
    if causal_rivers_available():
        for r in ("east", "bavaria", "flood"):
            G = causal_rivers_graph(r)
            print(f"  {r:9s} nodes={G.number_of_nodes():4d} "
                  f"edges={G.number_of_edges():4d}")
        for r, n in (("flood", 3), ("flood", 5)):
            print(f"  subgraphs {r}/random_{n}: "
                  f"{len(causal_rivers_subgraphs(r, n_vars=n))}")
    print()
    for s, (nin, nout, want) in CAUSALDYNAMICS_SUBSETS.items():
        state = "present" if causal_dynamics_available(s) else "absent"
        print(f"CausalDynamics {s:8s} inputs={nin / 1e6:8.1f} MB "
              f"outputs={nout / 1e6:6.2f} MB  fetch={want}  [{state}]")
    for sub in ("simple", "coupled"):
        if not causal_dynamics_available(sub):
            continue
        st = causal_dynamics_settings(sub)
        print(f"  {sub}: settings ({len(st)}), "
              f"systems in default setting: "
              f"{len(causal_dynamics_systems(sub))}")
        if sub == "coupled":
            cells = causal_dynamics_cells("coupled", **CD_COUPLED_DEFAULT_CELL)
            n = sum(len(causal_dynamics_systems("coupled", c)) for c in cells)
            print(f"    published 'Default' row = {len(cells)} setting dirs, "
                  f"{n} systems")
        print(f"    shipped eval methods: {causal_dynamics_eval_methods(sub)}")
    print()
    print("CauseMe        : format recorded in CAUSEME_SPEC; nothing is uploaded")


def _repro_cli(which: str, limit: Optional[int] = None):
    if which in ("rivers", "all"):
        print("=== CausalRivers: flood graph, Random-5, FULL 5-year series ===")
        print("    published (ICLR'25 Fig. 3a, 'Full TS'): VAR .85  Dynotears .79  "
              "Varlingam .72  PCMCI .60")
        c = dict(CR_REPRO_CELL)
        kw = c.pop("baseline_kw")
        c.pop("baseline")
        reproduce_causal_rivers("var_ols", limit=limit, **c, **kw)
    if which in ("dynamics", "all"):
        print("\n=== CausalDynamics simple, noise=0, no confounder ===")
        print("    leaderboard: Varlingam .50/.69   Dynotears .43/.67")
        # `prune=False` + binarise also lands on 0.500, but do NOT read that as
        # the mechanism behind the published number: lingam.VARLiNGAM defaults
        # to prune=True and the shipped wrapper does not override it, so their
        # matrix IS pruned.  Their per-system AUROC has sd 0.124 and spans
        # 0.283-0.875 with only 3/54 cells exactly 0.5 -- the 0.500 is a real
        # average, not an all-ones artefact.  See docs Sec. 6c.
        reproduce_causal_dynamics("varlingam", limit=limit, criterion=None,
                                  prune=False)
        reproduce_causal_dynamics("varlingam", limit=limit, criterion="bic")
        # transpose=True reproduces their DYNOTEARS wrapper, which stores
        # adj[cause, effect] and scores it against an [effect, cause] truth.
        reproduce_causal_dynamics("dynotears", limit=limit, transpose=True,
                                  lambda_w=0.2, lambda_a=0.1)
        reproduce_causal_dynamics("dynotears", limit=limit, transpose=False,
                                  lambda_w=0.2, lambda_a=0.1)
        print("    shipped eval bundled WITH the dataset (no re-running needed):")
        for m in causal_dynamics_eval_methods("simple"):
            e = causal_dynamics_published_eval(m)
            print(f"      {m:12s} AUROC={e['Individual_AUROC']:.3f} "
                  f"AUPRC={e['Individual_AUPRC']:.3f} "
                  f"n={int(e['n_systems'])}")
    if which in ("dynamics-coupled", "all"):
        _repro_cli_coupled(limit)


def _repro_cli_coupled(limit: Optional[int] = None, force: bool = False,
                       tables_only: bool = False):
    """The coupled 'Default' row: 6 setting dirs, 100 systems, 10 trajectories.

    ``tables_only=True`` re-prints every table from ``CD_AUDIT_CACHE`` without
    touching a single .nc file.
    """
    if tables_only and not os.path.exists(CD_AUDIT_CACHE):
        raise SystemExit(f"no cache at {CD_AUDIT_CACHE}; run without "
                         "--tables-only first")
    a = causal_dynamics_coupled_audit(force=force and not tables_only,
                                      limit=limit,
                                      with_repro=True)
    cells = a["cells"]
    print("\n=== CausalDynamics coupled, published 'Default' row ===")
    print(f"    {len(cells)} setting dirs (coupling x activation free; noise=0, "
          "systems=10, no confounder, timelag=0, no standardize)")
    print(f"    {sum(a['n_systems_per_cell'].values())} systems x "
          f"10 trajectories.  Source: {a['published_source']}")
    print("    published: " + "  ".join(
        f"{m} {v[0]:.2f}/{v[1]:.2f}" for m, v in sorted(a["published"].items())))

    print("\n    (a) shipped eval, re-averaged from outputs/coupled.tar.gz")
    print("        'constant' = cells whose AUROC is EXACTLY 0.5, i.e. the "
          "method emitted a constant matrix and 0.5 is true by construction.")
    print(f"        {'method':12s} {'AUROC':>6s} {'sd':>5s} {'AUPRC':>6s} "
          f"{'n':>4s} {'>0.5':>6s} {'constant':>9s} {'d':>5s}   published")
    for m in sorted(a["leaderboard"], key=lambda k: -a["leaderboard"][k]["Individual_AUROC"]):
        o = a["leaderboard"][m]
        p = a["published"].get({"ngc_lstm": "ngc"}.get(m, m))
        print(f"        {m:12s} {o['Individual_AUROC']:6.3f} "
              f"{o.get('auroc_sd', float('nan')):5.3f} "
              f"{o['Individual_AUPRC']:6.3f} "
              f"{o['n_cells']:4d} {o.get('auroc_above_half_frac', float('nan')) * 100:5.0f}% "
              f"{o['auroc_exactly_half'] * 100:8.0f}% "
              f"{o.get('effect_size_vs_half', float('nan')):5.1f}   "
              + (f"{p[0]:.2f}/{p[1]:.2f}" if p else "-"))

    print("\n    (b) MEASURED floor -- label-free predictors through the SAME scorer")
    for sub in ("coupled", "simple"):
        f = a["floor"][sub]
        print(f"        [{sub}] {f['n_systems']} systems, positive rate "
              f"{f['positive_rate_full']:.3f} (off-diagonal "
              f"{f['positive_rate_offdiag']:.3f}); the diagonal carries "
              f"{f['diagonal_share_of_positives'] * 100:.1f}% of all positives")
        for kind in CD_TRIVIAL:
            r = f["rows"].get(f"{kind}/full")
            o = f["rows"].get(f"{kind}/offdiag")
            if not r:
                continue
            print(f"            {kind:20s} full AUROC {r['auroc']:.3f} "
                  f"AUPRC {r['auprc']:.3f}"
                  + (f"   off-diagonal AUROC {o['auroc']:.3f} "
                     f"AUPRC {o['auprc']:.3f}" if o else ""))

    ip = a.get("identity_paired")
    if ip:
        print("\n    (b2) PAIRED against the identity matrix, system by system "
              "(Wilcoxon signed-rank)")
        print(f"        {'method':11s} {'n':>4s} {'AUROC':>6s} {'ident':>6s} "
              f"{'p':>9s} | {'AUPRC':>6s} {'ident':>6s} {'p':>9s} | wins")
        for m in sorted(ip, key=lambda k: -ip[k]["auroc"]):
            r = ip[m]
            print(f"        {m:11s} {r['n']:4d} {r['auroc']:6.3f} "
                  f"{r['identity_auroc']:6.3f} {r['p_auroc']:9.1e} | "
                  f"{r['auprc']:6.3f} {r['identity_auprc']:6.3f} "
                  f"{r['p_auprc']:9.1e} | AUROC on "
                  f"{r['win_rate_auroc'] * 100:3.0f}% of systems, AUPRC on "
                  f"{r['win_rate_auprc'] * 100:3.0f}%")
    rv = a.get("rows_vs_identity")
    if rv:
        print("\n    (b3) EVERY published coupled row vs the identity matrix")
        print("        the diagonal's grip is not constant across rows -- where it "
              "loosens, a method can win")
        print(f"        {'row':12s} {'cells':>5s} {'sys':>4s} {'ident AUROC':>11s} "
              f"{'ident AUPRC':>11s}  beats it on AUROC (p<.05) / on BOTH")
        for row in ("Default", "Noise", "Confounder", "Lag", "Standardize"):
            r = rv.get(row)
            if not r:
                continue
            n = max((v["n"] for v in r["paired"].values()), default=0)
            print(f"        {row:12s} {r['n_cells']:5d} {n:4d} "
                  f"{r['identity_auroc']:11.3f} {r['identity_auprc']:11.3f}  "
                  f"{', '.join(r['beats_identity_on_auroc']) or 'NOTHING'}"
                  f"  /  {', '.join(r['beats_identity_on_both']) or 'NOTHING'}")
    rr = a.get("repro_rows")
    if rr:
        print("\n    (b4) the two rows the identity matrix does NOT already win, "
              "with our own baselines")
        pub_best = {"Confounder": ("pcmciplus", 0.593, 0.208),
                    "Lag": ("pcmciplus", 0.560, 0.236)}
        for row, blk in rr.items():
            fc = blk["floor"]
            print(f"\n        [{row}]  floor: random "
                  f"{fc['rows']['random/full']['auroc']:.3f}/"
                  f"{fc['rows']['random/full']['auprc']:.3f}   identity "
                  f"{fc['rows']['identity/full']['auroc']:.3f}/"
                  f"{fc['rows']['identity/full']['auprc']:.3f}   diagonal share "
                  f"of positives {fc['diagonal_share_of_positives'] * 100:.1f}% "
                  f"(40.8% on Default)")
            print(f"        {'variant':16s} {'AUROC':>6s} {'AUPRC':>6s} | "
                  f"{'offAUROC':>8s} {'offAUPRC':>8s}  fits failed")
            for r in blk["repro"]:
                f = r["variants"]["transpose=1_binarise=0_full"]
                o = r["variants"]["transpose=1_binarise=0_offdiag"]
                print(f"        {r['label']:16s} {f['individual_auroc']:6.3f} "
                      f"{f['individual_auprc']:6.3f} | {o['individual_auroc']:8.3f} "
                      f"{o['individual_auprc']:8.3f}  {r['fails']:5d}/{r['n_fits']}")
            b = pub_best.get(row)
            if b:
                print(f"        best SHIPPED method on this row: {b[0]} "
                      f"{b[1]:.3f} / {b[2]:.3f}")
    orient = a["orientation"]
    print(f"\n    (c) orientation of the stored label, measured on "
          f"{orient['n_files']} systems / {orient['n_root_nodes']} root nodes:")
    print(f"        root-node off-diagonal COLUMN empty "
          f"{orient['frac_root_column_empty'] * 100:.1f}%   ROW empty "
          f"{orient['frac_root_row_empty'] * 100:.1f}%  ->  {orient['verdict']}")
    print(f"        mean diagonal density of the truth: "
          f"{orient['mean_diagonal_density']:.2f}")

    if a.get("repro"):
        print("\n    (d) our own runs, eval.py's protocol "
              "(z-standardised, 10 trajectories per system, label = A_native)")
        print("        'T' = our [effect, cause] estimate transposed into the "
              "coupled [cause, effect] storage order (= ALIGNED with the truth);")
        print("        'bin' = the shipped wrappers' abs(A) > 0 step; "
              "'off' masks the diagonal.")
        print(f"        {'variant':16s} {'T':>2s} {'bin':>4s} {'AUROC':>6s} "
              f"{'AUPRC':>6s} | {'offAUROC':>8s} {'offAUPRC':>8s} "
              f"{'const':>6s}  note")
        for r in a["repro"]:
            for tr in (False, True):
                for binz in (True, False):
                    f = r["variants"][f"transpose={int(tr)}_binarise={int(binz)}_full"]
                    o = r["variants"][f"transpose={int(tr)}_binarise={int(binz)}_offdiag"]
                    # On coupled the truth is [cause, effect], so T=1 is the
                    # ALIGNED orientation for every one of our estimators; a
                    # wrapper is "theirs" when its transpose matches ours.
                    tag = "   aligned" if tr else "   mis-oriented"
                    if r["their_orientation"] is not None and binz and \
                            bool(tr) == bool(r["their_orientation"]):
                        tag = "<- THEIR wrapper" + (" (already aligned)" if tr
                                                    else " (mis-oriented)")
                    print(f"        {r['label']:16s} {int(tr):>2d} {int(binz):>4d} "
                          f"{f['individual_auroc']:6.3f} {f['individual_auprc']:6.3f} | "
                          f"{o['individual_auroc']:8.3f} {o['individual_auprc']:8.3f} "
                          f"{f['constant_prediction_frac'] * 100:5.0f}%  {tag}")
    w = a.get("wrappers")
    if w:
        print("\n    (e) the shipped wrapper suite, audited against its own source")
        print("        coupled truth is [cause, effect]; a wrapper that emits "
              "[effect, cause] is TRANSPOSED here")
        print(f"        {'method':11s} {'emits':22s} {'coupled':>8s} {'simple':>7s}  note")
        for m, d in sorted(w["source_audit"].items()):
            print(f"        {m:11s} {d['emits']:22s} "
                  f"{'OK' if d['ok_on_coupled'] else 'TRANSPOSED':>8s} "
                  f"{'OK' if d['ok_on_simple'] else 'TRANSPOSED':>7s}  "
                  f"{d['note'][:74]}")
        p = w["probe"]
        print(f"\n        live probe on {p['n_systems']} coupled systems:")
        print(f"          grasp: AUROC {p['grasp_auroc_as_shipped']:.3f} as shipped "
              f"(raw -1/0/1 endpoint matrix) vs {p['grasp_auroc_abs']:.3f} with "
              f"abs() -- the encoding bug costs {p['grasp_cost']:.3f} AUROC; "
              f"{p['grasp_frac_negative_entries'] * 100:.0f}% of the submitted "
              "matrix is -1, i.e. ranked BELOW 'no edge'")
        print(f"          varlingam: abs(A) > 0 is ALL-ONES on "
              f"{p['varlingam_all_ones_rate'] * 100:.0f}% of coupled systems "
              "(28.3% on simple) -- where it is, AUROC is 0.500 by construction")
    print(f"\n    cache: {os.path.relpath(CD_AUDIT_CACHE, os.path.dirname(_BENCH))}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repro",
                    choices=["rivers", "dynamics", "dynamics-coupled", "all"],
                    help="reproduce a published cell")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fetch",
                    choices=["rivers", "dynamics", "dynamics-coupled", "all"])
    ap.add_argument("--audit", choices=["coupled"],
                    help="cached coupled audit: shipped leaderboard + measured "
                         "chance floor + trivial predictors + our reproduction")
    ap.add_argument("--tables-only", action="store_true", dest="tables_only",
                    help="re-print the audit tables from the JSON cache only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="hash both CausalDynamics loader contracts and check "
                         "them against CD_CONTRACT_DIGESTS")
    _a = ap.parse_args()
    if _a.selftest:
        for k, v in causal_dynamics_contract_selftest().items():
            print(f"{k:8s} n={v['n_systems']:4d} digest={v['digest'][:16]}... "
                  f"{'OK' if v['ok'] else 'CHANGED'}")
        raise SystemExit(0)
    if _a.fetch in ("rivers", "all"):
        fetch_causal_rivers()
    if _a.fetch in ("dynamics", "all"):
        fetch_causal_dynamics("simple")
    if _a.fetch in ("dynamics-coupled", "all"):
        fetch_causal_dynamics("coupled")     # 6.51 GB -> 10 GB extracted
    if _a.audit or _a.tables_only:
        _repro_cli_coupled(_a.limit, force=_a.force, tables_only=_a.tables_only)
    elif _a.repro:
        _repro_cli(_a.repro, _a.limit)
    else:
        main()
