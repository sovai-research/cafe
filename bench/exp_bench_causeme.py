#!/usr/bin/env python
r"""CauseMe: 12 external baselines + CAFE-CD, in the platform's own submission format.

NOTHING HERE CONTACTS causeme.uv.es EXCEPT TO DOWNLOAD PUBLIC EXPERIMENT DATA.
No result is ever uploaded.  ``causeme_build_submission`` writes to disk only.

WHY THERE IS NO ACCURACY COLUMN FOR THE REAL EXPERIMENTS
-------------------------------------------------------
CauseMe deliberately withholds ground truth: an experiment zip is a few hundred
whitespace-separated ``.txt`` files and NOTHING else -- verified, not assumed
(``--inventory`` prints the member list of every cached zip).  Scoring happens
server-side after an upload, and the ranking pages (``/rank/``) are behind a
login.  So this file CANNOT print a CauseMe AUC, and any number claiming to be
one would be fabricated.

What it can do, and does:

  1. **Build a valid submission artefact** for every (method, experiment) pair --
     one bz2-compressed JSON per pair, in ``data/causal/causeme/submissions/``.
  2. **Validate that artefact** against the platform's stated contract
     (``--validate``): required keys, one flattened (N,N) matrix per dataset,
     dataset ORDER = ``sorted(zip.namelist())``, row-major flatten, non-negative
     scores, zeroed self-links, p-values in [0,1], integer lags, and a
     bz2+JSON+latin1 round-trip that reconstructs every matrix bit-identically.
  3. **Prove the orientation** end to end (``--roundtrip``).  CauseMe is
     ``A_ij = score for i -> j`` (CAUSE row, EFFECT column) -- the TRANSPOSE of
     this repo's ``A[effect, cause]``.  A transposed submission does not merely
     score worse, it scores *inverted*.  ``--roundtrip`` generates a VAR with a
     known graph, writes it to disk in CauseMe's own on-disk format, pushes it
     through the identical loader -> method -> ``causeme_matrix_from_result``
     path, and scores it with CauseMe's own metric definitions (AUC over all
     N*N entries EXCLUDING self-links, pooled over the B datasets; TPR/FPR and
     F(beta=0.5) from the p-values at alpha=0.05).  Every method must beat its
     own transpose.  This is a FORMAT test on synthetic data -- it is NOT a
     CauseMe score and is never reported as one.
  4. **Measure runtime**, which CauseMe accepts as an optional ``runtimes``
     field and uses for inter-comparison.  This is a real, locally measurable
     result and the one place a CPU-only method can be compared honestly
     without the platform.
  5. **Measure method agreement** (``--agreement``) -- pooled Spearman rank
     correlation between every pair of methods' off-diagonal score vectors, read
     back out of the built artefacts.  With no ground truth this is the only
     cross-method signal available locally: it says which methods would be
     ranked similarly by an AUC, and it catches a method whose submission is
     anti-correlated with everything else (the signature of an inverted
     orientation).  Agreement is NOT accuracy and is never reported as such.

SELF-LOOPS.  CauseMe EXCLUDES self-links from every metric ("computed across
all (N, N) matrices (excluding self-links)").  Submitted matrices therefore
carry a zeroed diagonal, and every local metric in this file masks it too.
That is the opposite of CausalDynamics (which keeps the diagonal as part of the
Jacobian truth) and the same as CausalRivers.  See docs/causal_baselines.md s7.

WHAT AN ACTUAL SUBMISSION STILL NEEDS  (none of it is done here)
---------------------------------------------------------------
  * a causeme.uv.es account (``/signup/``);
  * the method registered under ``/my_uploads/``, ideally with a code zip whose
    module exposes ``my_method(data, **params) -> (val_matrix, p_matrix,
    lag_matrix)`` -- that is what earns the "Validated" badge and lets the
    platform time the method itself;
  * the ``method_sha`` that registration returns, substituted for the
    placeholder in every artefact (``--method-sha``);
  * a manual upload on "My Results", which is rate limited per user.

Licence discipline is unchanged: ``tigramite`` (GPL-3.0) stays out of process,
``dagma.linear`` is imported without ``dagma.utils`` (which would drag in
GPL-2.0+ python-igraph), and ``src/cafe`` never imports anything in bench/.

Usage
-----
    .venv-bench/bin/python bench/exp_bench_causeme.py --inventory
    .venv-bench/bin/python bench/exp_bench_causeme.py --roundtrip
    .venv-bench/bin/python bench/exp_bench_causeme.py --run
    .venv-bench/bin/python bench/exp_bench_causeme.py --validate
    .venv-bench/bin/python bench/exp_bench_causeme.py --agreement
    .venv-bench/bin/python bench/exp_bench_causeme.py --tables-only
"""
from __future__ import annotations

import argparse
import bz2
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

# 12 baselines, some of which fan out to their own BLAS.  Pin BEFORE numpy is
# imported -- numpy reads these exactly once, at import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np                                              # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import causal_baselines as cb                                   # noqa: E402
import datasets_causal as dc                                    # noqa: E402

CACHE_PATH = os.path.join(_HERE, "bench_causeme_cache.json")
CM_ROOT = os.path.join(_ROOT, "data", "causal", "causeme")
ZIP_DIR = os.path.join(CM_ROOT, "zips")
SUB_DIR = os.path.join(CM_ROOT, "submissions")
CM_BASE = "https://causeme.uv.es/static/datasets"

# The sha is a per-user identifier issued at method registration.  It is a
# REQUIRED field, so the artefact carries an obviously-fake placeholder rather
# than a plausible-looking hex string that might be mistaken for a real one.
PLACEHOLDER_SHA = "UNREGISTERED-PLACEHOLDER-NOT-A-REAL-METHOD-SHA"

# --------------------------------------------------------------------------- #
# Experiments.  Every one of these is downloadable anonymously from
# /static/datasets/<name>.zip; sizes are the measured content-length.
# `max_lag` follows the platform's own model description ("The maximum time
# delay is 5" for the VAR families); it is recorded in `parameter_values` so a
# submission states the lag budget it used.
# --------------------------------------------------------------------------- #
EXPERIMENTS: Dict[str, dict] = {
    "linear-VAR_N-5_T-300": dict(
        model="linear-VAR", N=5, T=300, max_lag=5, bytes=1_182_999,
        note="autocorrelation + time delays, linear; max true delay 5"),
    "nonlinear-VAR_N-5_T-300": dict(
        model="nonlinear-VAR", N=5, T=300, max_lag=5, bytes=1_168_457,
        note="nonlinear functional dependencies on the same graph family"),
    "logistic-deterministic_N-5_T-300_weak": dict(
        model="logistic-deterministic", N=5, T=300, max_lag=5, bytes=998_865,
        note="chaotic logistic map, weak coupling -- deterministic, the case "
             "Granger-type methods are expected to fail"),
    "river-runoff_N-12_T-4600": dict(
        model="river-runoff", N=12, T=4600, max_lag=5, bytes=105_807,
        note="REAL river runoff, a single long dataset (B=1)"),
}

DEFAULT_EXPERIMENTS = ("linear-VAR_N-5_T-300", "nonlinear-VAR_N-5_T-300",
                       "logistic-deterministic_N-5_T-300_weak",
                       "river-runoff_N-12_T-4600")

# The 12 externals, in the order docs/causal_baselines.md s4 lists them, plus
# CAFE-CD.  `pcmci`/`pcmciplus` run OUT OF PROCESS (tigramite is GPL-3.0).
METHODS = ("var_ols", "granger_pairwise", "varlingam", "directlingam",
           "dynotears", "dagma_linear", "pc", "fci", "pc_ts", "fci_ts",
           "pcmci", "pcmciplus", "cafe_cd")

METHOD_KW: Dict[str, dict] = {
    "dynotears": dict(lambda_w=0.2, lambda_a=0.1),
}

# Per-dataset wall-clock ceiling.  A method that blows it is scored as an
# all-zero matrix, exactly as the CausalDynamics and CausalRivers harnesses do,
# so a method cannot win by declining hard datasets.
TIMEOUT_S = 300.0


# =========================================================================== #
# cache
# =========================================================================== #
def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def cache_load() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {"cells": {}, "meta": {}}
    try:
        with open(CACHE_PATH) as fh:
            d = json.load(fh)
        d.setdefault("cells", {})
        d.setdefault("meta", {})
        return d
    except Exception as e:                                  # corrupt -> keep a copy
        os.replace(CACHE_PATH, CACHE_PATH + ".corrupt")
        print(f"  ! cache unreadable ({e}); moved aside")
        return {"cells": {}, "meta": {}}


def cache_save(cache: dict) -> None:
    """Atomic: temp file in the same dir, then rename over the target."""
    cache["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cache["meta"]["spec"] = dc.causeme_submission_spec()
    cache["meta"]["experiments"] = EXPERIMENTS
    cache["meta"]["methods"] = list(METHODS)
    cache["meta"]["nothing_uploaded"] = True
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True, default=_jsonable)
    os.replace(tmp, CACHE_PATH)


def cache_put(cache: dict, key: str, value) -> None:
    cache["cells"][key] = value
    cache_save(cache)


# =========================================================================== #
# data
# =========================================================================== #
def fetch_experiment(name: str, force: bool = False) -> str:
    """Download one experiment zip into data/causal/causeme/zips (public URL)."""
    os.makedirs(ZIP_DIR, exist_ok=True)
    dest = os.path.join(ZIP_DIR, f"{name}.zip")
    if os.path.exists(dest) and not force:
        return dest
    url = f"{CM_BASE}/{name}.zip"
    print(f"  [causeme] GET {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
    os.replace(tmp, dest)
    return dest


def load_experiment(name: str) -> Tuple[List[str], List[np.ndarray]]:
    """(names, arrays) for one experiment, ORDERED BY SORTED FILENAME.

    The platform is explicit that "the order of the matrices in the results
    list should be the same as the one found in the zipped downloaded data
    file"; sorting the namelist is the deterministic reading of that, and it is
    what the shipped ``causeme_example`` script does.  Each member is
    whitespace-separated with N series in COLUMNS, so ``np.loadtxt`` gives
    ``(T, N)`` directly.
    """
    path = fetch_experiment(name)
    out_names, out_arrays = [], []
    with zipfile.ZipFile(path) as z:
        for member in sorted(z.namelist()):
            if not member.endswith(".txt"):
                continue
            X = np.loadtxt(io.BytesIO(z.read(member)))
            if X.ndim == 1:
                X = X[:, None]
            out_names.append(member)
            out_arrays.append(np.asarray(X, dtype=float))
    return out_names, out_arrays


def inventory(names: Sequence[str]) -> dict:
    """What is actually inside each zip -- the evidence that no truth ships."""
    rows = {}
    for n in names:
        path = fetch_experiment(n)
        with zipfile.ZipFile(path) as z:
            members = sorted(z.namelist())
        exts = sorted({os.path.splitext(m)[1] or "<none>" for m in members})
        X = None
        with zipfile.ZipFile(path) as z:
            if members:
                X = np.loadtxt(io.BytesIO(z.read(members[0])))
        rows[n] = dict(
            zip_bytes=os.path.getsize(path), n_members=len(members),
            extensions=exts, first=members[0] if members else None,
            shape_T_N=list(np.shape(X)) if X is not None else None,
            ground_truth_present=bool(
                [m for m in members
                 if not m.endswith(".txt")
                 or any(k in m.lower() for k in ("truth", "graph", "adj", "link"))]),
        )
    return rows


# =========================================================================== #
# methods -> CauseMe matrices
# =========================================================================== #
def _cafe_known_kwargs() -> set:
    try:
        from cafe.discovery import _DEFAULTS
        return set(_DEFAULTS)
    except Exception:
        try:
            import inspect
            from cafe.discovery import discover
            return set(inspect.signature(discover).parameters)
        except Exception:
            return {"max_lag"}


def run_cafe_cd(X: np.ndarray, max_lag: int):
    """CAFE-CD -> a BaselineResult-shaped object, so ONE adapter serves all 13.

    CAFE's convention is ``A[target, source]`` = ``[effect, cause]``, the same
    as ``BaselineResult``, so nothing is transposed here; the single transpose
    into CauseMe orientation happens in ``causeme_matrix_from_result``.
    """
    from cafe.discovery import discover
    P = int(np.asarray(X).shape[1])
    known = _cafe_known_kwargs()
    kw = {k: v for k, v in dict(max_lag=max_lag).items() if k in known}
    t0 = time.time()
    g = discover(np.asarray(X, dtype=float), **kw)
    dt = time.time() - t0
    if hasattr(g, "graphs") and hasattr(g, "checkpoints"):      # PIT sequence
        g = g.graphs[-1]
    S = np.abs(np.nan_to_num(np.asarray(g.scores(lag="all"), dtype=float)))
    A = np.abs(np.nan_to_num(np.asarray(g.adjacency(lag="all"), dtype=float)))
    if S.ndim != 3 or S.shape[1:] != (P, P):
        raise RuntimeError(f"CAFE-CD scores(lag='all') gave {S.shape}, want (K+1,{P},{P})")
    K = S.shape[0] - 1
    return cb.BaselineResult(
        A_lag=S[1:] if K else np.zeros((0, P, P)),
        A_contemp=S[0],
        support_lag=(A[1:] != 0) if K else np.zeros((0, P, P), bool),
        support_contemp=(A[0] != 0),
        runtime_s=dt, ok=True, err="",
        meta=dict(name="cafe_cd", max_lag=K, P=P,
                  scores_lagged=True, scores_contemp=True,
                  contemp_symmetric=False, call=f"discover(X, {kw})"))


# The `parameter_values` string is free text on the platform but it is the ONLY
# record of what was actually run, so it names the statistic each method ranks
# by rather than pretending they all rank by |coef|.
METHOD_STAT = {
    "var_ols": "|t| of the OLS VAR coefficient (causal_baselines default score='tstat')",
    "granger_pairwise": "pairwise Granger F statistic",
    "varlingam": "|adjacency_matrices_[k]| from lingam.VARLiNGAM (adaptive-lasso pruned)",
    "directlingam": "|adjacency_matrix_| from lingam.DirectLiNGAM (contemporaneous only)",
    "dynotears": "|W|,|A| from a from-scratch DYNOTEARS (lambda_w=0.2, lambda_a=0.1)",
    "dagma_linear": "|W| from dagma.linear.DagmaLinear (contemporaneous only)",
    "pc": "causal-learn PC, CPDAG decoded to a score (contemporaneous only)",
    "fci": "causal-learn FCI, PAG decoded to a score (contemporaneous only)",
    "pc_ts": "causal-learn PC on the time-expanded design",
    "fci_ts": "causal-learn FCI on the time-expanded design",
    "pcmci": "tigramite PCMCI, 1 - p_matrix, run OUT OF PROCESS (GPL-3.0)",
    "pcmciplus": "tigramite PCMCI+, 1 - p_matrix, run OUT OF PROCESS (GPL-3.0)",
    "cafe_cd": "CAFE-CD cafe.discovery.discover(...).scores(lag='all')",
}


def describe_parameters(method: str, max_lag: int) -> str:
    """What actually produced the numbers -- the only provenance a submission carries."""
    kw = METHOD_KW.get(method, {})
    return (f"max_lag={max_lag}; "
            f"edge statistic = {METHOD_STAT.get(method, 'method default')}; "
            f"submitted score = max over lags 0..{max_lag} of |statistic|; "
            f"pvalues = binary support (0 = method selected the edge, 1 = not) "
            f"-- NOT calibrated p-values; lags = argmax-lag of the score; "
            f"self-links zeroed (CauseMe excludes them); "
            + (f"kwargs={kw}; " if kw else "")
            + "harness = TIMARA bench/exp_bench_causeme.py, nothing uploaded")


def run_one(method: str, X: np.ndarray, max_lag: int):
    """One dataset, one method -> (scores, pvalues, lags, runtime_s, ok, err).

    Matrices come back in CAUSEME orientation (``A_ij`` = score for ``i -> j``)
    with a zeroed diagonal, via ``dc.causeme_matrix_from_result``.
    """
    t0 = time.time()
    if method == "cafe_cd":
        try:
            res = run_cafe_cd(X, max_lag)
        except Exception as e:
            P = int(X.shape[1])
            return (np.zeros((P, P)), np.ones((P, P)), np.zeros((P, P), int),
                    time.time() - t0, False, f"{type(e).__name__}: {e}")
    else:
        res = cb.run_baseline(method, X, max_lag=max_lag, timeout_s=TIMEOUT_S,
                              **METHOD_KW.get(method, {}))
        if not res.ok:
            P = int(X.shape[1])
            return (np.zeros((P, P)), np.ones((P, P)), np.zeros((P, P), int),
                    time.time() - t0, False, res.err)
    s, p, l = dc.causeme_matrix_from_result(res, how="max")
    return s, p, l, float(time.time() - t0), True, ""


# =========================================================================== #
# CauseMe's own metrics -- used ONLY on synthetic round-trip data
# =========================================================================== #
def causeme_metrics(scores: Sequence[np.ndarray], truth: Sequence[np.ndarray],
                    pvalues: Optional[Sequence[np.ndarray]] = None,
                    lags: Optional[Sequence[np.ndarray]] = None,
                    true_lags: Optional[Sequence[np.ndarray]] = None,
                    alpha: float = 0.05, beta: float = 0.5) -> dict:
    """The platform's stated metrics, re-implemented from its own description.

    * AUC  -- ROC-AUC over every ``(N, N)`` entry EXCLUDING self-links, POOLED
      over the B datasets of the experiment.
    * FPR / TPR -- from the p-values thresholded at ``alpha``, over the same
      ``N*(N-1)*B`` links.
    * F-measure -- ``beta=0.5`` (precision-weighted) on the same binary
      predictions.
    * TLR -- true-lag rate among CORRECTLY predicted links only.

    Everything is in CAUSEME orientation; ``truth[b][i, j] = 1`` means i -> j.
    """
    from sklearn.metrics import roc_auc_score
    ys, ss, ps = [], [], []
    for b, (S, Y) in enumerate(zip(scores, truth)):
        off = ~np.eye(Y.shape[0], dtype=bool)
        ys.append(np.asarray(Y, float)[off])
        ss.append(np.asarray(S, float)[off])
        if pvalues is not None:
            ps.append(np.asarray(pvalues[b], float)[off])
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    out = dict(n_links=int(y.size), n_true=int(y.sum()),
               density=float(y.mean()),
               auc=float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else float("nan"))
    if ps:
        pred = (np.concatenate(ps) <= alpha).astype(float)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        b2 = beta * beta
        out.update(tpr=rec, fpr=fp / (fp + tn) if fp + tn else 0.0,
                   precision=prec, recall=rec,
                   f_beta=((1 + b2) * prec * rec / (b2 * prec + rec)
                           if (b2 * prec + rec) else 0.0), beta=beta, alpha=alpha)
        if lags is not None and true_lags is not None:
            hit = miss = 0
            for b_, (L, TL, Y, P_) in enumerate(zip(lags, true_lags, truth, pvalues)):
                off = ~np.eye(Y.shape[0], dtype=bool)
                ok = (np.asarray(P_, float) <= alpha) & (np.asarray(Y) == 1) & off
                hit += int((np.asarray(L)[ok] == np.asarray(TL)[ok]).sum())
                miss += int(ok.sum())
            out["tlr"] = hit / miss if miss else float("nan")
    return out


# =========================================================================== #
# synthetic round-trip -- the orientation proof
# =========================================================================== #
def make_var_experiment(B: int = 12, N: int = 5, T: int = 300, max_lag: int = 3,
                        density: float = 0.25, seed: int = 0,
                        out_dir: Optional[str] = None):
    """A CauseMe-SHAPED experiment with known truth, written in their on-disk format.

    Sparse stable VAR, coefficients drawn away from zero so the graph is
    identifiable, plus the self-lag term every CauseMe model has (excluded from
    scoring anyway).  Returned truth is in CAUSEME orientation:
    ``Y[i, j] = 1`` iff ``i -> j`` at some lag >= 1.
    """
    rng = np.random.default_rng(seed)
    names, Xs, Ys, TLs = [], [], [], []
    for b in range(B):
        A = np.zeros((max_lag, N, N))                       # A[k-1, effect, cause]
        Y = np.zeros((N, N))
        TL = np.zeros((N, N), dtype=int)
        for i in range(N):
            A[0, i, i] = rng.uniform(0.25, 0.45)            # own AR term
            for j in range(N):
                if i == j or rng.random() > density:
                    continue
                k = int(rng.integers(1, max_lag + 1))
                A[k - 1, i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 0.6)
                Y[j, i] = 1.0                               # CauseMe: cause row
                TL[j, i] = k
        # rescale to a spectral radius that is comfortably stable
        comp = np.zeros((N * max_lag, N * max_lag))
        comp[:N] = np.concatenate([A[k] for k in range(max_lag)], axis=1)
        if max_lag > 1:
            comp[N:, :-N] = np.eye(N * (max_lag - 1))
        r = np.max(np.abs(np.linalg.eigvals(comp)))
        if r >= 0.9:
            A *= 0.9 / r
        burn = 200
        X = np.zeros((T + burn + max_lag, N))
        E = rng.standard_normal((T + burn + max_lag, N))
        for t in range(max_lag, T + burn + max_lag):
            X[t] = E[t]
            for k in range(1, max_lag + 1):
                X[t] += A[k - 1] @ X[t - k]
        Xs.append(X[burn + max_lag:])
        Ys.append(Y)
        TLs.append(TL)
        names.append(f"roundtrip-VAR_N-{N}_T-{T}_{b + 1:04d}.txt")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for n, X in zip(names, Xs):
            np.savetxt(os.path.join(out_dir, n), X)
        # read them back through the SAME path the real loader uses
        Xs = [np.loadtxt(os.path.join(out_dir, n)) for n in sorted(names)]
        order = np.argsort(names)
        Ys = [Ys[i] for i in order]
        TLs = [TLs[i] for i in order]
        names = sorted(names)
    return names, Xs, Ys, TLs


def roundtrip(methods: Sequence[str], B: int = 12, N: int = 5, T: int = 300,
              max_lag: int = 3, seed: int = 0) -> dict:
    """Every method must beat its own transpose on data with a known graph."""
    tmp = os.path.join(CM_ROOT, "roundtrip")
    names, Xs, Ys, TLs = make_var_experiment(B, N, T, max_lag, seed=seed,
                                             out_dir=tmp)
    rows = {}
    for m in methods:
        S, PV, LG, rt, fails = [], [], [], [], 0
        for X in Xs:
            s, p, l, dt, ok, err = run_one(m, X, max_lag)
            S.append(s); PV.append(p); LG.append(l); rt.append(dt)
            fails += (not ok)
        good = causeme_metrics(S, Ys, PV, LG, TLs)
        flip = causeme_metrics([x.T for x in S], Ys)
        rows[m] = dict(auc=good["auc"], auc_transposed=flip["auc"],
                       margin=good["auc"] - flip["auc"],
                       oriented_correctly=bool(good["auc"] > flip["auc"]),
                       tpr=good.get("tpr"), fpr=good.get("fpr"),
                       f_beta=good.get("f_beta"), tlr=good.get("tlr"),
                       density=good["density"], n_links=good["n_links"],
                       fails=fails, runtime_s_mean=float(np.mean(rt)))
    return dict(B=B, N=N, T=T, max_lag=max_lag, seed=seed, rows=rows,
                note="SYNTHETIC format/orientation test. NOT a CauseMe score.")


# =========================================================================== #
# the real run -- artefacts only
# =========================================================================== #
def sub_path(method: str, experiment: str) -> str:
    return os.path.join(SUB_DIR, f"{experiment}__{method}.json.bz2")


def run_experiment(cache: dict, experiment: str, methods: Sequence[str],
                   method_sha: str, limit: Optional[int] = None,
                   force: bool = False) -> None:
    cfg = EXPERIMENTS[experiment]
    names, Xs = load_experiment(experiment)
    if limit:
        names, Xs = names[:limit], Xs[:limit]
    model = cfg["model"]
    max_lag = cfg["max_lag"]
    print(f"\n=== {experiment}  B={len(Xs)}  (T,N)={Xs[0].shape}  max_lag={max_lag} ===")
    os.makedirs(SUB_DIR, exist_ok=True)
    for m in methods:
        key = f"run/{experiment}/{m}"
        if key in cache["cells"] and not force and os.path.exists(sub_path(m, experiment)):
            r = cache["cells"][key]
            print(f"  {m:18s} cached  {r['runtime_s_total']:8.1f}s  fails={r['fails']}")
            continue
        S, PV, LG, RT, fails, errs = [], [], [], [], 0, {}
        t0 = time.time()
        for X in Xs:
            s, p, l, dt, ok, err = run_one(m, X, max_lag)
            S.append(s); PV.append(p); LG.append(l); RT.append(dt)
            if not ok:
                fails += 1
                errs[err[:120]] = errs.get(err[:120], 0) + 1
        total = time.time() - t0
        params = describe_parameters(m, max_lag)
        out = sub_path(m, experiment)
        dc.causeme_build_submission(
            S, model=model, experiment=experiment, method_sha=method_sha,
            parameter_values=params, pvalues=PV, lags=LG, runtimes=RT,
            out_path=out)
        rec = dict(experiment=experiment, model=model, method=m, B=len(Xs),
                   N=int(Xs[0].shape[1]), T=int(Xs[0].shape[0]),
                   max_lag=max_lag, fails=fails, errors=errs,
                   runtime_s_total=float(total),
                   runtime_s_per_dataset=float(np.mean(RT)),
                   runtime_s_median=float(np.median(RT)),
                   artefact=os.path.relpath(out, _ROOT),
                   artefact_bytes=os.path.getsize(out),
                   nonzero_frac=float(np.mean([np.mean(s > 0) for s in S])),
                   method_sha=method_sha, uploaded=False)
        cache_put(cache, key, rec)
        print(f"  {m:18s} {total:8.1f}s  {np.mean(RT):7.3f}s/dataset  "
              f"fails={fails:3d}  -> {os.path.basename(out)} "
              f"({os.path.getsize(out) / 1e3:.0f} kB)")
        if errs:
            for e, c in list(errs.items())[:2]:
                print(f"      ! {c}x {e}")


# =========================================================================== #
# artefact validation
# =========================================================================== #
def validate_artefact(path: str, experiment: str) -> dict:
    """Check ONE built file against the platform's stated contract."""
    spec = dc.causeme_submission_spec()
    problems: List[str] = []
    with bz2.BZ2File(path, "r") as fh:
        raw = fh.read().decode("latin1")
    d = json.loads(raw)
    for k in spec["required_fields"]:
        if k not in d:
            problems.append(f"missing required field {k!r}")
    # Only the COUNT and ORDER are needed here, so read the namelist rather
    # than np.loadtxt-ing 200 datasets per artefact.
    with zipfile.ZipFile(fetch_experiment(experiment)) as z:
        names = [m for m in sorted(z.namelist()) if m.endswith(".txt")]
    B = len(names)
    S = d.get("scores", [])
    if len(S) != B:
        problems.append(f"scores has {len(S)} entries, experiment has {B} datasets")
    N = EXPERIMENTS[experiment]["N"]
    for i, v in enumerate(S[:B]):
        a = np.asarray(v, dtype=float)
        if a.size != N * N:
            problems.append(f"scores[{i}] has {a.size} entries, want {N * N}")
            break
        M = a.reshape(N, N)
        if np.any(a < 0):
            problems.append(f"scores[{i}] has negative entries")
            break
        if not np.all(np.diag(M) == 0):
            problems.append(f"scores[{i}] has a non-zero self-link")
            break
        if not np.all(np.isfinite(a)):
            problems.append(f"scores[{i}] is not finite")
            break
    for opt, lo, hi in (("pvalues", 0.0, 1.0),):
        if opt in d:
            if len(d[opt]) != B:
                problems.append(f"{opt} has {len(d[opt])} entries, want {B}")
            else:
                a = np.asarray(d[opt][0], dtype=float)
                if a.size != N * N or a.min() < lo or a.max() > hi:
                    problems.append(f"{opt}[0] out of contract")
    if "lags" in d:
        a = np.asarray(d["lags"][0])
        if a.size != N * N or a.min() < 0 or not np.all(a == a.astype(int)):
            problems.append("lags[0] is not a non-negative integer matrix")
    # bz2 + json + latin1 round-trip must be lossless
    tmp = path + ".rt"
    with bz2.BZ2File(tmp, "w") as fh:
        fh.write(bytes(json.dumps(d), encoding="latin1"))
    with bz2.BZ2File(tmp, "r") as fh:
        d2 = json.loads(fh.read().decode("latin1"))
    os.remove(tmp)
    if d2 != d:
        problems.append("bz2/json/latin1 round-trip is not lossless")
    return dict(path=os.path.relpath(path, _ROOT), ok=not problems,
                problems=problems, B=B, N=N,
                order="sorted(zipfile.namelist())", flatten="row-major (C)",
                orientation="A_ij = score for i -> j (cause row, effect column)",
                method_sha=d.get("method_sha"),
                is_placeholder_sha=d.get("method_sha") == PLACEHOLDER_SHA,
                bytes=os.path.getsize(path))


# =========================================================================== #
# method agreement -- the only cross-method signal available without truth
# =========================================================================== #
def agreement(experiment: str, methods: Sequence[str]) -> dict:
    """Pooled Spearman between every pair of methods' off-diagonal scores.

    Read back out of the BUILT ARTEFACTS, so it measures exactly what would be
    submitted.  Self-links are already zero in the artefacts and are dropped
    here as well, because CauseMe excludes them from every metric.
    """
    from scipy.stats import spearmanr
    N = EXPERIMENTS[experiment]["N"]
    off = ~np.eye(N, dtype=bool)
    vecs: Dict[str, np.ndarray] = {}
    for m in methods:
        p = sub_path(m, experiment)
        if not os.path.exists(p):
            continue
        with bz2.BZ2File(p, "r") as fh:
            d = json.loads(fh.read().decode("latin1"))
        vecs[m] = np.concatenate([np.asarray(v, float).reshape(N, N)[off]
                                  for v in d["scores"]])
    names = sorted(vecs)
    rho = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = vecs[a], vecs[b]
            if np.ptp(va) == 0 or np.ptp(vb) == 0:
                r = float("nan")
            else:
                r = float(spearmanr(va, vb).statistic)
            rho[f"{a}|{b}"] = r
    mean_rho = {m: float(np.nanmean([v for k, v in rho.items()
                                     if m in k.split("|")])) for m in names}
    return dict(experiment=experiment, n_methods=len(names), methods=names,
                n_links=int(len(next(iter(vecs.values()), []))),
                pairwise=rho, mean_agreement=mean_rho,
                constant_submissions=[m for m in names if np.ptp(vecs[m]) == 0],
                note="Spearman over the pooled off-diagonal scores of the built "
                     "artefacts. Agreement is not accuracy; a strongly NEGATIVE "
                     "row would indicate an inverted orientation.")


# =========================================================================== #
# reporting
# =========================================================================== #
def report(cache: dict) -> None:
    cells = cache["cells"]
    rt = cells.get("roundtrip")
    print("\n=== A. Orientation round-trip (SYNTHETIC VAR, not a CauseMe score) ===")
    if not rt:
        print("    (not run -- bench/exp_bench_causeme.py --roundtrip)")
    else:
        print(f"    B={rt['B']} datasets, N={rt['N']}, T={rt['T']}, "
              f"max_lag={rt['max_lag']}, link density {list(rt['rows'].values())[0]['density']:.3f}")
        print(f"    {'method':18s} {'AUC':>6s} {'AUC(T)':>7s} {'margin':>7s} "
              f"{'TPR':>6s} {'FPR':>6s} {'F0.5':>6s} {'TLR':>6s} {'s/ds':>7s}  oriented")
        for m, r in rt["rows"].items():
            f = lambda v: "  --  " if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:6.3f}"
            print(f"    {m:18s} {r['auc']:6.3f} {r['auc_transposed']:7.3f} "
                  f"{r['margin']:+7.3f} {f(r['tpr'])} {f(r['fpr'])} {f(r['f_beta'])} "
                  f"{f(r['tlr'])} {r['runtime_s_mean']:7.3f}  "
                  f"{'OK' if r['oriented_correctly'] else 'INVERTED'}")
    print("\n=== B. CauseMe artefacts built (nothing uploaded) ===")
    runs = {k: v for k, v in cells.items() if k.startswith("run/")}
    if not runs:
        print("    (not run -- bench/exp_bench_causeme.py --run)")
    else:
        exps = sorted({v["experiment"] for v in runs.values()})
        for e in exps:
            rows = sorted((v for v in runs.values() if v["experiment"] == e),
                          key=lambda r: r["runtime_s_per_dataset"])
            r0 = rows[0]
            print(f"\n    {e}  B={r0['B']}  N={r0['N']}  T={r0['T']}  "
                  f"max_lag={r0['max_lag']}")
            print(f"    {'method':18s} {'s/dataset':>10s} {'total s':>9s} "
                  f"{'fails':>6s} {'nonzero':>8s}  artefact")
            for r in rows:
                print(f"    {r['method']:18s} {r['runtime_s_per_dataset']:10.3f} "
                      f"{r['runtime_s_total']:9.1f} {r['fails']:6d} "
                      f"{r['nonzero_frac']:8.3f}  "
                      f"{os.path.basename(r['artefact'])} "
                      f"({r['artefact_bytes'] / 1e3:.0f} kB)")
    print("\n=== C. Artefact validation ===")
    v = cells.get("validate")
    if not v:
        print("    (not run -- bench/exp_bench_causeme.py --validate)")
    else:
        bad = [r for r in v["files"] if not r["ok"]]
        print(f"    {v['n_files']} artefacts checked, {len(bad)} with problems")
        print(f"    order={v['files'][0]['order']}  flatten={v['files'][0]['flatten']}")
        print(f"    orientation={v['files'][0]['orientation']}")
        for r in bad[:10]:
            print(f"    ! {r['path']}: {'; '.join(r['problems'])}")
        if v.get("placeholder_sha"):
            print("    NOTE: every artefact carries the placeholder method_sha; "
                  "re-run with --method-sha <real sha> before any upload.")
    print("\n=== D. Method agreement (no ground truth is available locally) ===")
    ag = {k: v for k, v in cells.items() if k.startswith("agreement/")}
    if not ag:
        print("    (not run -- bench/exp_bench_causeme.py --agreement)")
    for k in sorted(ag):
        a = ag[k]
        print(f"\n    {a['experiment']}  ({a['n_methods']} methods, "
              f"{a['n_links']} pooled off-diagonal links)")
        # A constant submission has an undefined Spearman -- print it as such
        # rather than crashing or, worse, silently sorting it to the top.
        def _key(kv):
            return 1.0 if not np.isfinite(kv[1]) else -kv[1]
        for m, r in sorted(a["mean_agreement"].items(), key=_key):
            if not np.isfinite(r):
                print(f"      {m:18s} mean rho   n/a   CONSTANT SUBMISSION "
                      "(AUC would be exactly 0.5 by construction)")
                continue
            bar = "#" * max(0, int(round(r * 20)))
            print(f"      {m:18s} mean rho {r:+.3f}  {bar}")
        if a["constant_submissions"]:
            print(f"      ! constant submissions (rho undefined): "
                  f"{a['constant_submissions']}")

    print("\n=== E. What a submission would still require ===")
    for i, s in enumerate(dc.causeme_submission_spec()["what_an_entry_requires"], 1):
        print(f"    {s}")
    print("    STATUS: nothing in this repo has been uploaded to causeme.uv.es.")


# =========================================================================== #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("inventory", "roundtrip", "run", "validate", "agreement",
              "report", "all", "force", "tables_only"):
        ap.add_argument(f"--{f.replace('_', '-')}", action="store_true",
                        dest=f)
    ap.add_argument("--experiments", type=str, default=None)
    ap.add_argument("--methods", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first LIMIT datasets of each experiment")
    ap.add_argument("--method-sha", type=str, default=PLACEHOLDER_SHA)
    args = ap.parse_args(argv)

    cache = cache_load()
    exps = (args.experiments.split(",") if args.experiments
            else list(DEFAULT_EXPERIMENTS))
    meths = args.methods.split(",") if args.methods else list(METHODS)
    for e in exps:
        if e not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {e!r}; known: {sorted(EXPERIMENTS)}")

    if args.tables_only:
        report(cache)
        return
    if args.inventory or args.all:
        inv = inventory(exps)
        cache_put(cache, "inventory", inv)
        print("\n=== CauseMe experiment inventory (what ships in each zip) ===")
        for n, r in inv.items():
            print(f"  {n}\n    {r['zip_bytes'] / 1e3:8.0f} kB  {r['n_members']:4d} members "
                  f"{r['extensions']}  first={r['first']}  (T,N)={r['shape_T_N']}  "
                  f"ground_truth_present={r['ground_truth_present']}")
    if args.roundtrip or args.all:
        print("\n=== Orientation round-trip (synthetic; NOT a CauseMe score) ===")
        r = roundtrip(meths)
        cache_put(cache, "roundtrip", r)
    if args.run or args.all:
        for e in exps:
            run_experiment(cache, e, meths, args.method_sha, limit=args.limit,
                           force=args.force)
    if args.validate or args.all:
        files = []
        for e in exps:
            for m in meths:
                p = sub_path(m, e)
                if os.path.exists(p):
                    files.append(validate_artefact(p, e))
        rec = dict(n_files=len(files), files=files,
                   placeholder_sha=any(f["is_placeholder_sha"] for f in files),
                   all_ok=all(f["ok"] for f in files))
        cache_put(cache, "validate", rec)
    if args.agreement or args.all:
        for e in exps:
            if any(os.path.exists(sub_path(m, e)) for m in meths):
                cache_put(cache, f"agreement/{e}", agreement(e, meths))
    if args.report or args.all or not any(
            (args.inventory, args.roundtrip, args.run, args.validate,
             args.agreement)):
        report(cache)


if __name__ == "__main__":
    main()
