#!/usr/bin/env python
r"""THE POINT-IN-TIME BOARD --- a look-ahead audit for causal DISCOVERY.

This is the graph analogue of ``bench/exp_leakage_audit.py`` ("the epsilon of
imputation").  That experiment showed that the imputation field benchmarks in the
*bidirectional* setting and gave every imputer two numbers on one neutral standard: a
binary causality certificate (``max_revision`` under truncation invariance) and a
continuous leakage score (the accuracy a method borrows from the future).

**Causal discovery has the identical pathology and nobody has audited it.**  Every
published time-series causal-discovery method --- PCMCI/PCMCI+, DYNOTEARS, VARLiNGAM,
DAGMA, PC/FCI, Granger --- is *batch*: it consumes the whole series and emits one
graph.  If that graph is used in a backtest, or to decide anything at time ``t``, it
carries information from after ``t``.  The literature does not measure this.

--------------------------------------------------------------------------------
THE TWO NUMBERS  (defined once, applied identically to every method)
--------------------------------------------------------------------------------
Notation: ``G(Z)`` is the graph a method emits when fitted on the block of rows ``Z``;
``S(Z)`` is its own binary support (its published thresholding decision, never one we
impose).  ``X`` is ``(T, P)``, row axis = time.

**(1) max_edge_revision --- the binary certificate.**
    A method run at analysis-time ``tau`` asserts ``S(X[:tau+1])``; that graph is what
    you would act on at every ``t <= tau``.  The audit re-runs the method on growing
    prefixes and asks whether an already-asserted edge is ever rewritten once later
    data arrives.  This is exactly ``cafe.audit.verify_causal_series`` with the graph
    support passed as a 0/1 indicator, which is what that function was written for ---
    we call the library function, we do not reimplement it.  For a batch method the
    reporting function is degenerate in ``t`` (its answer at time ``t`` is whatever it
    fitted on everything it was given), and that degeneracy IS the pathology.
    ``max_edge_revision`` is 0.0 (no asserted edge was ever revised) or 1.0 (some edge
    appeared, vanished or flipped).

    A vacuously-empty graph never revises.  Every row therefore carries ``n_edges``
    and a ``degenerate`` flag, so a zero that means "emits nothing" is never confused
    with a zero that means "point-in-time".

**(2) edge_churn --- how BIG the revision is.**
    The certificate is binary, so we also report the size of the revision.  Write
    ``report(Z, t)`` for the graph a method asserts FOR TIME ``t`` when the analysis is
    run on rows ``Z``.  Then

        ``churn(tau) = Jaccard-distance( report(X[:tau+1], tau), report(X, tau) )``

    --- the graph you hold for time ``tau`` if you run the analysis at ``tau``, versus
    the graph you hold for that same time ``tau`` if you run it after seeing everything.
    For a batch method those are ``G(X[:tau+1])`` and ``G(X)``, so churn is exactly the
    magnitude of the revision.  For a point-in-time method they are the same object and
    churn is 0.  This is NOT "how much the graph changes as data arrives" (legitimate
    learning, which every method does); it is "how much of what you already acted on
    gets rewritten".  Raw counts of edges that appear, vanish or flip orientation
    accompany it, and sweeping ``tau`` answers how revision accumulates with series
    length.

**(3) leakage_delta --- what look-ahead is worth, on the rival's own terms.**
    Graph quality is measured against ground truth two ways, with the method's own
    scores and its own threshold, on the benchmark's own metric:

      * ``Q_batch``   fitted on the whole series --- the published protocol.
      * ``Q_honest``  refitted on each prefix and read out at the right edge, i.e. the
                      graph you would actually have held at decision time ``t``,
                      averaged over ``t``.  This is the discovery analogue of
                      ``cafe.audit.make_causal_variant``'s trailing-window/right-edge
                      wrapping of a bidirectional imputer.

    ``leakage_delta = Q_batch - Q_honest``.  Positive => the published number owes
    something to information the method would not have had.  The imputation audit
    found *negative* deltas for some methods, so no sign is assumed.

    ``Q_honest`` uses a shorter series than ``Q_batch``, so ``leakage_delta`` mixes
    "less data" with "no future data".  We therefore also report a **sample-size
    matched** control, which is the strict analogue of the imputation wrapper:

      * ``Q_trail``  fit on the TRAILING window ``X[t-L+1 : t+1]`` (L rows, all <= t)
      * ``Q_centr``  fit on a window of the SAME LENGTH L straddling ``t``
      * ``delta_matched = Q_centr - Q_trail`` --- identical sample size, the only
        difference is whether the window is allowed to reach past ``t``.  This is
        pure look-ahead with the sample-size effect removed.

--------------------------------------------------------------------------------
FRAMING (read before quoting any number from this file)
--------------------------------------------------------------------------------
This is not "our benchmark, which we win".  (1) The property is independently
motivated: point-in-time validity is what a backtest or any decision at time ``t``
requires, and it is the standard CAFE already holds itself to for imputation.
(2) The ``leakage_delta`` half is measured entirely on the rivals' own terms --- their
metric, their threshold, their data.  A rival with near-zero revision is a real and
publishable finding, and is reported as such rather than buried.

--------------------------------------------------------------------------------
KNOWN LIMITATION (stated up front, not in a footnote)
--------------------------------------------------------------------------------
``cafe.discovery.CausalGraph.scores()`` currently returns a hard two-tier step
(selected edges in ``[0.5, 1]``, unselected in ``[0, 0.5)``) rather than a calibrated
ranking, so true-but-unselected edges rank below selected false positives and every
rank-based metric (AUROC/AUPRC) for CAFE-CD is UNDERSTATED.  Two things are NOT
affected: the revision certificate, which reads the support indicator under
truncation and is a structural consequence of only ever reading rows ``<= t``; and
``F1``, which reads the support rather than the ranking.  Every rival's
``leakage_delta`` is computed entirely from that rival's own outputs on its own
metric and is unaffected either way -- which is the half of this experiment that
stands on its own merit.  Every graph behind the CAFE-CD row is stored verbatim in
the cache under ``row["raw"]`` so that row can be re-scored without refitting.

--------------------------------------------------------------------------------
PROTOCOL (identical for every method; failures recorded, never dropped)
--------------------------------------------------------------------------------
* one lag budget ``max_lag`` per dataset, one timeout, one preprocessing, for all.
* scored object: the (P,P) SUMMARY graph (max over lags of |score|, contemporaneous
  folded in) --- the object CausalRivers and CausalDynamics both score, and the only
  object on which lagged-only and contemporaneous-only methods can be compared.
  The certificate is ALSO computed on the full window graph ``(K,P,P) + (P,P)``
  (stricter: more places to change); both are recorded.
* the diagonal follows each benchmark's own protocol (CausalRivers removes it,
  CausalDynamics keeps it; synthetic follows CausalRivers).
* ``ok=False`` results (raise/timeout) are kept as failures with their error string.

Outputs
    bench/pit_board_cache.json     every raw measurement, written incrementally
    paper/tables/pit_board.tex     the board
    paper/figures/pit_board.pdf    churn curves + delta bars + prefix sweep

Run:  python3 bench/exp_pit_board.py
Env:  PIT_DATASETS   (syn_var,syn_contemp,rivers,dynamics)
      PIT_METHODS    (default: all 12 in causal_baselines.BASELINES)
      PIT_SEEDS      (2)      synthetic seeds / real-dataset instances
      PIT_TMAX       (1200)   rows used per series (prefix sweeps need length)
      PIT_NCP        (24)     max checkpoints on the expanding grid
      PIT_NMATCH     (8)      checkpoints for the sample-size-matched control
      PIT_TIMEOUT    (120)    seconds per fit
      PIT_RESUME     (1)      reuse an existing cache and only run missing cells
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_SRC = os.path.join(ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import causal_baselines as CB          # noqa: E402
import causal_metrics as CM            # noqa: E402
import cafe.audit as AUDIT             # noqa: E402  (the SAME standard as imputation)

CACHE = os.path.join(HERE, "pit_board_cache.json")
TEX = os.path.join(ROOT, "paper", "tables", "pit_board.tex")
FIG = os.path.join(ROOT, "paper", "figures", "pit_board.pdf")

DATASETS = os.environ.get("PIT_DATASETS",
                          "syn_var,syn_contemp,rivers,dynamics").split(",")
METHODS = [m for m in os.environ.get(
    "PIT_METHODS", ",".join(CB.BASELINES)).split(",") if m]
SEEDS = int(os.environ.get("PIT_SEEDS", 2))
TMAX = int(os.environ.get("PIT_TMAX", 1200))
NCP = int(os.environ.get("PIT_NCP", 24))
NMATCH = int(os.environ.get("PIT_NMATCH", 8))
TIMEOUT = float(os.environ.get("PIT_TIMEOUT", 120))
RESUME = os.environ.get("PIT_RESUME", "1") not in ("0", "false", "False")
#: regenerate the board/table/figure from the cache without re-running anything.
REPORT_ONLY = os.environ.get("PIT_REPORT_ONLY", "0") not in ("0", "false", "False")

#: prefix-count sweep -- the same audit run with more/fewer checkpoints.
SWEEP_NP = (3, 6, 12, 24)
#: checkpoints held FIXED across the sweep, so only the prefix count varies.
CERT_CKPTS = 6
TOL = 1e-6                      # mirrors cafe.audit's certificate tolerance


# =========================================================================== #
# Metrics.  AUROC / F1 come from bench/causal_metrics.py; AUPRC (average
# precision) is 8 lines of numpy so this module needs no sklearn.
# =========================================================================== #
def _auprc(y, s):
    """Average precision == area under the precision-recall curve (sklearn's
    ``average_precision_score``: sum over thresholds of (R_k - R_{k-1}) * P_k,
    with tied scores collapsed to one threshold).  NaN if a class is absent."""
    y = np.asarray(y).reshape(-1) != 0
    s = np.asarray(s, float).reshape(-1)
    s = np.where(np.isfinite(s), s, -np.inf)
    n1 = int(y.sum())
    if n1 == 0 or n1 == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    # collapse ties: keep only the last index of each equal-score run
    keep = np.append(ss[1:] != ss[:-1], True)
    tp, fp = tp[keep], fp[keep]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n1
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def _mask_for(P, include_diag):
    return np.ones((P, P), bool) if include_diag else ~np.eye(P, dtype=bool)


def score_graph(A_true, S_score, S_supp, include_diag):
    """Quality of one estimated summary graph, on the benchmark's own terms.

    ``A_true`` (P,P) 0/1 [effect, cause]; ``S_score`` the method's continuous summary
    score; ``S_supp`` the method's OWN binary support.  Returns auroc / auprc / f1 /
    precision / recall / n_edges."""
    m = _mask_for(A_true.shape[0], include_diag)
    y = (np.asarray(A_true, float)[m] != 0)
    s = np.abs(np.nan_to_num(np.asarray(S_score, float)))[m]
    h = np.asarray(S_supp, bool)[m]
    tp = int(np.sum(y & h)); fp = int(np.sum(~y & h)); fn = int(np.sum(y & ~h))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"auroc": CM.edge_auc(y, s), "auprc": _auprc(y, s), "f1": float(f1),
            "precision": float(prec), "recall": float(rec),
            "n_edges": int(h.sum()), "n_true": int(y.sum()), "n_cells": int(y.size)}


def churn(S_a, S_b):
    """How far the graph asserted earlier (``S_a``) is from the later one (``S_b``).

    Jaccard distance over the asserted edge sets, plus the decomposition the claim
    actually needs: how many edges appeared, vanished, or flipped orientation.
    Both empty -> distance 0 (nothing was ever asserted, hence nothing revised);
    the caller must read this together with ``n_edges``."""
    a = np.asarray(S_a, bool)
    b = np.asarray(S_b, bool)
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    jac = 0.0 if union == 0 else 1.0 - inter / union
    appear = int(np.sum(~a & b))
    vanish = int(np.sum(a & ~b))
    flip = 0
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        flip = int(np.sum(a & ~b & b.T & ~a.T))
    elif a.ndim == 3 and a.shape[1] == a.shape[2]:
        flip = int(np.sum(a & ~b & np.swapaxes(b, 1, 2) & ~np.swapaxes(a, 1, 2)))
    return {"jaccard": float(jac), "appear": appear, "vanish": vanish,
            "flip": flip, "n_a": int(a.sum()), "n_b": int(b.sum())}


# =========================================================================== #
# Datasets.  Each returns a list of tasks:
#   {"key", "X" (T,P), "A_true" (P,P) [effect,cause], "max_lag", "include_diag",
#    "standardize", "note"}
# =========================================================================== #
def _cap(X, tmax):
    X = np.asarray(X, float)
    return np.ascontiguousarray(X[:tmax]) if X.shape[0] > tmax else \
        np.ascontiguousarray(X)


def _syn(seed, contemp):
    """Structural VAR with a KNOWN graph and NO hidden common factor (chi=0).

    chi=0 is deliberate: a pervasive factor is the configuration CAFE's
    residualisation is built for, and including it here would confound a leakage
    audit with a CAFE advantage.  Every rival sees a clean VAR of the kind its own
    paper assumes.  ``contemp=True`` + Laplace shocks is the configuration in which
    LiNGAM/PC/FCI/DAGMA (contemporaneous-only) are identifiable at all."""
    import gen_causal_panel as G
    X3d, _, _, truth = G.gen_causal_panel(
        E=1, T=TMAX, P=6, R=1, n_lags=2, deg=1, chi=0.0,
        contemp=bool(contemp), contemp_deg=1,
        noise="laplace" if contemp else "gauss",
        entity_fe=False, time_fe=False, season=None,
        spec_radius=0.60, seed=seed, ar_self=True)
    X = np.ascontiguousarray(X3d[0])
    A = (np.abs(truth.A_lag).sum(0) != 0)
    if contemp:
        A = A | (np.abs(truth.A_contemp) != 0)
    A = A.astype(float)
    np.fill_diagonal(A, 0.0)           # scored off-diagonal (CausalRivers protocol)
    return X, A


def build_tasks():
    tasks = []
    for ds in DATASETS:
        ds = ds.strip()
        try:
            if ds in ("syn_var", "syn_contemp"):
                for s in range(SEEDS):
                    X, A = _syn(s, contemp=(ds == "syn_contemp"))
                    tasks.append(dict(
                        key=f"{ds}/seed{s}", dataset=ds, X=_cap(X, TMAX), A_true=A,
                        max_lag=3, include_diag=False, standardize=True,
                        note="structural VAR, chi=0 (no hidden factor); "
                             + ("contemporaneous DAG + Laplace shocks"
                                if ds == "syn_contemp" else "lagged only, Gaussian")))
            elif ds == "rivers":
                import datasets_causal as DC
                if not DC.causal_rivers_available():
                    print("[skip] CausalRivers not downloaded"); continue
                # CR_REPRO_CELL: the settings that reproduce the published VAR 0.85
                # (normalize=False, standardize=False, max_lag=3, no contemp channel).
                it = DC.iter_causal_rivers(region="flood_full", n_vars=5,
                                           strategy="random", resolution="6h",
                                           normalize=False, limit=SEEDS, seed=0)
                for i, (X, A, m) in enumerate(it):
                    tasks.append(dict(
                        key=f"rivers/sample{i}", dataset="rivers",
                        X=_cap(X, TMAX), A_true=np.asarray(A, float),
                        max_lag=3, include_diag=False, standardize=False,
                        note="CausalRivers flood_full random_5, 6h, "
                             "normalize=False (the published VAR cell); "
                             "diagonal removed per their protocol"))
            elif ds == "dynamics":
                import datasets_causal as DC
                if not DC.causal_dynamics_available("simple"):
                    print("[skip] CausalDynamics not downloaded"); continue
                names = DC.causal_dynamics_systems("simple")[:SEEDS]
                for nm in names:
                    X, A, m = DC.load_causal_dynamics(nm, subset="simple")
                    tasks.append(dict(
                        key=f"dynamics/{nm}", dataset="dynamics",
                        X=_cap(X, TMAX), A_true=np.asarray(A, float),
                        max_lag=3, include_diag=True, standardize=True,
                        note="CausalDynamics simple subset, Jacobian ground truth; "
                             "diagonal KEPT per their protocol"))
            else:
                print(f"[skip] unknown dataset {ds!r}")
        except Exception as e:                                   # noqa: BLE001
            print(f"[skip] dataset {ds}: {type(e).__name__}: {e}")
    return tasks

# =========================================================================== #
# THE REPORTING INTERFACE -- one contract, every method.
#
#   report(Xp, t)  = the graph the method asserts FOR TIME t when the analysis is
#                    run on the block of rows Xp.
#
# A BATCH method's answer does not depend on t: it fits on everything it was
# given and that one graph is what you would act on at every t in the block.
# That degeneracy IS the pathology under audit.  A POINT-IN-TIME method's answer
# depends on t only through rows <= t, so re-running the analysis later cannot
# change it.  Everything below -- certificate, churn, honest quality -- is
# defined on this one function, so rivals and CAFE-CD travel identical code.
# =========================================================================== #
class Fitter:
    """Rival baselines: ``report`` ignores ``t`` (batch semantics).

    Fits are memoised by row-block, so the audit costs one fit per distinct block
    however many times it is queried.  Every rival on a given task gets the same
    lag budget, the same preprocessing and the same timeout."""

    is_pit = False

    def __init__(self, task, method):
        self.X = np.asarray(task["X"], float)
        self.T, self.P = self.X.shape
        self.method = method
        self.max_lag = int(task["max_lag"])
        self.standardize = bool(task["standardize"])
        self._c = {}
        self.n_fits = 0
        self.fit_time = 0.0
        self.failures = []

    def _empty(self, err):
        return dict(ok=False, err=err,
                    supp=np.zeros((self.P, self.P), bool),
                    score=np.zeros((self.P, self.P)),
                    wsupp=np.zeros((self.max_lag + 1) * self.P * self.P, bool))

    def block(self, lo, hi):
        key = (int(lo), int(hi))
        if key in self._c:
            return self._c[key]
        Z = self.X[int(lo):int(hi)]
        t0 = time.time()
        if Z.shape[0] < max(3 * self.max_lag + 4, 20):
            out = self._empty(f"block too short ({Z.shape[0]} rows)")
        else:
            res = CB.run_baseline(self.method, Z, max_lag=self.max_lag,
                                  timeout_s=TIMEOUT, standardize=self.standardize)
            sl = np.asarray(res.support_lag, bool)
            sc = np.asarray(res.support_contemp, bool)
            out = dict(ok=bool(res.ok), err=str(res.err),
                       supp=(sl.any(axis=0) | sc),
                       score=CB.summary_graph(res, how="max", include_contemp=True),
                       wsupp=np.concatenate([sc.reshape(-1), sl.reshape(-1)]))
            if not res.ok:
                self.failures.append(f"[{lo}:{hi}] {res.err[:120]}")
        self.n_fits += 1
        self.fit_time += time.time() - t0
        self._c[key] = out
        return out

    def report(self, Xp, t):
        """Batch semantics: whatever was fitted on ``Xp``, asserted for every t."""
        return self.block(0, int(np.asarray(Xp).shape[0]))


class CafeFitter:
    """CAFE-CD: ``report(Xp, t)`` is ``cafe.discover(Xp, mode='pit')`` read at ``t``.

    ``mode='pit'`` computes ``G_t`` from rows ``<= t`` only, so the answer for time
    ``t`` cannot depend on which later rows happen to be in ``Xp``.  The memo is
    keyed on (block, t) -- NOT on ``t`` alone -- so the certificate genuinely
    re-runs the method on every prefix instead of assuming the property it tests."""

    is_pit = True

    def __init__(self, task, method="cafe_cd"):
        import cafe                                     # noqa: F401
        self._cafe = cafe
        self.X = np.asarray(task["X"], float)
        self.T, self.P = self.X.shape
        self.method = method
        self.max_lag = int(task["max_lag"])
        self._c = {}
        self.n_fits = 0
        self.fit_time = 0.0
        self.failures = []

    def _pack(self, g):
        supp = np.asarray(g.adjacency(lag=None), bool).reshape(self.P, self.P)
        score = np.asarray(g.scores(lag=None), float).reshape(self.P, self.P)
        try:
            w = np.asarray(g.adjacency(lag="all"), bool).reshape(-1)
        except Exception:                                # pragma: no cover
            w = supp.reshape(-1)
        return dict(ok=True, err="", supp=supp, score=score, wsupp=w)

    def set_checkpoint_hint(self, cps):
        """Every time the audit will ever ask about, so one prefix costs ONE call.

        Purely a speed hint: ``mode="pit"`` computes each ``G_t`` from rows ``<= t``
        regardless of which other checkpoints are requested in the same call, so
        batching cannot leak anything between them.  The memo stays keyed on
        (prefix length, t), so the certificate still re-runs the method per prefix."""
        self._hint = sorted({int(c) for c in cps})

    def report(self, Xp, t):
        Xp = np.asarray(Xp, float)
        n = int(Xp.shape[0])
        t = int(min(int(t), n - 1))
        key = (0, n, t)
        if key in self._c:
            return self._c[key]
        want = sorted({c for c in getattr(self, "_hint", ()) if c <= n - 1} | {t})
        t0 = time.time()
        try:
            g = self._cafe.discover(Xp, mode="pit", max_lag=self.max_lag,
                                    checkpoints=want)
            for c in want:
                self._c[(0, n, int(c))] = self._pack(g.at(int(c)))
            out = self._c[key]
        except Exception as e:                           # noqa: BLE001
            self.failures.append(f"[0:{n}@{t}] {type(e).__name__}: {e}"[:150])
            out = dict(ok=False, err=f"{type(e).__name__}: {e}"[:150],
                       supp=np.zeros((self.P, self.P), bool),
                       score=np.zeros((self.P, self.P)),
                       wsupp=np.zeros((self.max_lag + 1) * self.P * self.P, bool))
            self._c[key] = out
        self.n_fits += 1
        self.fit_time += time.time() - t0
        return out

    def block(self, lo, hi):
        """A batch fit on one row-block -- used ONLY by the sample-size-matched
        control, where both arms must be a fixed-length window."""
        key = (int(lo), int(hi), -1)
        if key in self._c:
            return self._c[key]
        t0 = time.time()
        try:
            g = self._cafe.discover(self.X[int(lo):int(hi)], mode="batch",
                                    max_lag=self.max_lag)
            out = self._pack(g)
        except Exception as e:                           # noqa: BLE001
            self.failures.append(f"[{lo}:{hi}] {type(e).__name__}: {e}"[:150])
            out = dict(ok=False, err=f"{type(e).__name__}: {e}"[:150],
                       supp=np.zeros((self.P, self.P), bool),
                       score=np.zeros((self.P, self.P)),
                       wsupp=np.zeros((self.max_lag + 1) * self.P * self.P, bool))
        self.n_fits += 1
        self.fit_time += time.time() - t0
        self._c[key] = out
        return out


def make_fitter(task, method):
    return CafeFitter(task) if method == "cafe_cd" else Fitter(task, method)


# =========================================================================== #
# The audit of ONE (task, method) cell.  Identical for every method.
# =========================================================================== #
def _cuts(T, n):
    """The prefix grid cafe.audit uses: ``linspace(T//3, T-1, n)``."""
    return np.unique(np.linspace(T // 3, T - 1, int(n)).astype(int))


def audit_cell(task, method):
    X = np.asarray(task["X"], float)
    A = task["A_true"]
    T = X.shape[0]
    inc = bool(task["include_diag"])
    F = make_fitter(task, method)
    t0 = time.time()
    row = {"dataset": task["dataset"], "task": task["key"], "method": method,
           "T": int(T), "P": int(X.shape[1]), "max_lag": int(task["max_lag"]),
           "is_pit": bool(F.is_pit)}

    # ------------------------------------------------------------------ #
    # (1) THE CERTIFICATE -- cafe.audit.verify_causal_series, the library
    #     standard, with the graph support passed as the 0/1 indicator it
    #     documents.  Asks: is what the method reported for time t ever
    #     rewritten once rows after t are available?
    # ------------------------------------------------------------------ #
    def ind_summary(Xp, t):
        return F.report(Xp, t)["supp"].astype(float)

    def ind_window(Xp, t):
        return F.report(Xp, t)["wsupp"].astype(float)

    # (2) CHURN -- the SIZE of that revision at prefix tau: the graph asserted for
    #     time tau by an analysis run at tau, versus by an analysis run at T.
    def churn_at(tau):
        return churn(F.report(X[:int(tau) + 1], int(tau))["supp"],
                     F.report(X, int(tau))["supp"])

    # The certificate's CHECKPOINT set is held fixed while the number of PREFIXES
    # varies, so the sweep isolates one thing: how many re-runs you have to try
    # before you catch a revision.
    cert_cps = _cuts(T, CERT_CKPTS)
    if isinstance(F, CafeFitter):
        F.set_checkpoint_hint(np.concatenate([cert_cps, _cuts(T, NCP)]
                                             + [_cuts(T, n) for n in SWEEP_NP]))

    sweep = {}
    for n in SWEEP_NP:
        cs = _cuts(T, n)
        cert = AUDIT.verify_causal_series(ind_summary, X, n_prefixes=n,
                                          tol=TOL, checkpoints=cert_cps)
        certw = AUDIT.verify_causal_series(ind_window, X, n_prefixes=n,
                                           tol=TOL, checkpoints=cert_cps)
        ch = [churn_at(k) for k in cs]
        sweep[str(n)] = {
            "n_prefixes": int(n),
            "max_edge_revision": float(cert["max_revision"]),
            "max_edge_revision_window": float(certw["max_revision"]),
            "causal": bool(cert["causal"]),
            "n_comparisons": int(cert["n_comparisons"]),
            "churn_mean": float(np.mean([c["jaccard"] for c in ch])),
            "churn_max": float(np.max([c["jaccard"] for c in ch])),
            "appear": int(np.sum([c["appear"] for c in ch])),
            "vanish": int(np.sum([c["vanish"] for c in ch])),
            "flip": int(np.sum([c["flip"] for c in ch])),
        }
    row["sweep"] = sweep
    hdr = sweep[str(max(SWEEP_NP))]
    row["max_edge_revision"] = hdr["max_edge_revision"]
    row["max_edge_revision_window"] = hdr["max_edge_revision_window"]
    row["causal"] = hdr["causal"]
    row["churn_mean"] = hdr["churn_mean"]
    row["churn_max"] = hdr["churn_max"]
    row["n_appear"] = hdr["appear"]
    row["n_vanish"] = hdr["vanish"]
    row["n_flip"] = hdr["flip"]

    # ------------------------------------------------------------------ #
    # Churn curve + the honest quality trajectory: at each tau, the graph you
    # would ACTUALLY hold at that moment, scored against ground truth.
    # ------------------------------------------------------------------ #
    grid = _cuts(T, NCP)
    curve = []
    raw_curve = []
    for k in grid:
        p = F.report(X[:int(k) + 1], int(k))
        c = churn(p["supp"], F.report(X, int(k))["supp"])
        q = score_graph(A, p["score"], p["supp"], inc)
        raw_curve.append({"tau": int(k) + 1,
                          "supp": np.asarray(p["supp"], int).tolist(),
                          "score": np.asarray(p["score"], float).tolist()})
        curve.append({"tau": int(k) + 1, "frac": float((k + 1) / T),
                      "jaccard": c["jaccard"], "appear": c["appear"],
                      "vanish": c["vanish"], "flip": c["flip"],
                      "n_edges": q["n_edges"], "auroc": q["auroc"],
                      "auprc": q["auprc"], "f1": q["f1"], "ok": bool(p["ok"])})
    row["curve"] = curve

    # Does the asserted graph settle down?  Churn at 50% and 90% of the series
    # answers "how much of what I already acted on is still being rewritten
    # when I am nine-tenths of the way through".
    for tag, frac in (("50", 0.5), ("90", 0.9)):
        cand = [c for c in curve if c["frac"] <= frac + 1e-9]
        row[f"churn_at_{tag}"] = float(cand[-1]["jaccard"]) if cand else float("nan")

    # ------------------------------------------------------------------ #
    # (3) LEAKAGE DELTA -- published protocol vs right-edge honest.
    # ------------------------------------------------------------------ #
    full = F.report(X, T - 1)
    qb = score_graph(A, full["score"], full["supp"], inc)
    row["batch"] = qb
    row["ok_batch"] = bool(full["ok"])
    row["err_batch"] = str(full["err"])[:200]

    def _m(key):
        v = [c[key] for c in curve if c[key] == c[key]]
        return float(np.mean(v)) if v else float("nan")

    row["honest"] = {"auroc": _m("auroc"), "auprc": _m("auprc"), "f1": _m("f1"),
                     "n_edges": float(np.mean([c["n_edges"] for c in curve]))}
    for k in ("auroc", "auprc", "f1"):
        row[f"delta_{k}"] = float(qb[k] - row["honest"][k])

    # ------------------------------------------------------------------ #
    # Sample-size-MATCHED look-ahead: a length-L window ending at t (all rows
    # <= t) versus a length-L window straddling t.  Identical n; the only
    # difference is permission to reach past the decision time.  Run for EVERY
    # method, CAFE-CD included -- a point-in-time method is not exempt from
    # having to show what it would gain by cheating.
    # ------------------------------------------------------------------ #
    L = max(int(T // 3), 4 * int(task["max_lag"]) + 20)
    lo_t, hi_t = L - 1, T - (L - L // 2) - 1
    matched = []
    raw_matched = []
    if hi_t > lo_t:
        for t in np.unique(np.linspace(lo_t, hi_t, NMATCH).astype(int)):
            tr = F.block(int(t) - L + 1, int(t) + 1)
            c0 = int(t) - L // 2
            ce = F.block(c0, c0 + L)
            qt = score_graph(A, tr["score"], tr["supp"], inc)
            qc = score_graph(A, ce["score"], ce["supp"], inc)
            matched.append({"t": int(t), "L": int(L),
                            "trail": {k: qt[k] for k in ("auroc", "auprc", "f1")},
                            "centr": {k: qc[k] for k in ("auroc", "auprc", "f1")},
                            "churn_tc": churn(tr["supp"], ce["supp"])["jaccard"]})
            raw_matched.append({
                "t": int(t),
                "trail_supp": np.asarray(tr["supp"], int).tolist(),
                "trail_score": np.asarray(tr["score"], float).tolist(),
                "centr_supp": np.asarray(ce["supp"], int).tolist(),
                "centr_score": np.asarray(ce["score"], float).tolist()})
    row["matched"] = matched
    row["L_match"] = int(L)
    for k in ("auroc", "auprc", "f1"):
        tv = [m["trail"][k] for m in matched if m["trail"][k] == m["trail"][k]]
        cv = [m["centr"][k] for m in matched if m["centr"][k] == m["centr"][k]]
        row[f"trail_{k}"] = float(np.mean(tv)) if tv else float("nan")
        row[f"centr_{k}"] = float(np.mean(cv)) if cv else float("nan")
        row[f"dmatch_{k}"] = row[f"centr_{k}"] - row[f"trail_{k}"]

    # ------------------------------------------------------------------ #
    # Bookkeeping.  A method that asserts NO edges never revises; that zero is
    # vacuous and must never be read as a point-in-time certificate.
    # ------------------------------------------------------------------ #
    row["n_edges_full"] = int(qb["n_edges"])
    row["degenerate"] = bool(max([c["n_edges"] for c in curve]
                                 + [qb["n_edges"]]) == 0)
    row["n_fits"] = int(F.n_fits)
    row["fit_time_s"] = float(F.fit_time)
    row["wall_s"] = float(time.time() - t0)
    row["n_failed_fits"] = int(len(F.failures))
    row["failures"] = list(F.failures[:5])
    row["ok_any"] = bool(full["ok"] or any(c["ok"] for c in curve))

    # ------------------------------------------------------------------ #
    # CAFE-CD's quality half is PROVISIONAL: cafe.discovery.CausalGraph.scores()
    # currently returns a hard two-tier step (selected edges in [0.5,1],
    # unselected in [0,0.5)), so unselected-but-true edges are ranked below
    # selected false positives and every rank-based metric (AUROC/AUPRC) is
    # understated.  The REVISION CERTIFICATE is untouched by this: it reads the
    # support indicator under truncation and is a structural consequence of only
    # ever reading rows <= t.  F1 also reads the support, not scores().
    # So that our row can be re-scored the moment the ranking is fixed, WITHOUT
    # refitting anything, every graph behind it is stored verbatim here.
    if F.is_pit:
        row["quality_provisional"] = True
        row["quality_note"] = (
            "AUROC/AUPRC understated: CausalGraph.scores() is a two-tier step, "
            "not a calibrated ranking. max_edge_revision and F1 read the support "
            "and are unaffected. Re-score from row['raw'], or delete the cafe_cd "
            "rows from the cache and re-run with PIT_RESUME=1.")
        row["raw"] = {
            "A_true": np.asarray(A, float).tolist(),
            "include_diag": bool(inc),
            "batch": {"supp": np.asarray(full["supp"], int).tolist(),
                      "score": np.asarray(full["score"], float).tolist()},
            "curve": raw_curve,
            "matched": raw_matched,
        }

    # Independent cross-check with the discovery module's own helper, when it is
    # importable: two implementations of the same certificate must agree.
    row["helper_max_edge_revision"] = _helper_cert(F, X, task)
    return row


def _helper_cert(F, X, task):
    """``cafe.discovery.max_edge_revision`` run on the same reporting function.

    Recorded as an independent confirmation of the certificate above -- if the
    library helper and ``cafe.audit.verify_causal_series`` ever disagree, the
    number in this cache says so instead of hiding it."""
    try:
        from cafe.discovery import max_edge_revision as _mer
    except Exception:
        return None
    try:
        def fn(Xp, t):
            return F.report(Xp, t)["supp"].astype(float)
        out = _mer(fn, X, n_prefixes=6, as_of=True)
        return float(out["max_revision"] if isinstance(out, dict) else out)
    except Exception as e:                               # noqa: BLE001
        return f"helper-error: {type(e).__name__}: {str(e)[:80]}"


# =========================================================================== #
# CAFE-CD availability (src/cafe/discovery.py, built by a sibling agent).  The
# rival audit above is complete and entirely independent of it.
# =========================================================================== #
def cafe_cd_available():
    try:
        import cafe
        import cafe.discovery                            # noqa: F401
        return callable(getattr(cafe, "discover", None))
    except Exception:
        return False


# =========================================================================== #
# Aggregation / reporting
# =========================================================================== #
_LABEL = {
    "var_ols": "VAR-OLS", "granger_pairwise": "Granger (pairwise)",
    "varlingam": "VARLiNGAM", "directlingam": "DirectLiNGAM",
    "dynotears": "DYNOTEARS", "dagma_linear": "DAGMA (linear)",
    "pc": "PC", "fci": "FCI", "pc_ts": "PC (time-lagged)",
    "fci_ts": "FCI (time-lagged)", "pcmci": "PCMCI", "pcmciplus": "PCMCI+",
    "cafe_cd": "CAFE-CD (pit)",
}
_DSLABEL = {"syn_var": "Synth-VAR", "syn_contemp": "Synth-contemp",
            "rivers": "CausalRivers", "dynamics": "CausalDynamics"}


def aggregate(rows):
    """Per (method, dataset) and per method, over tasks."""
    by_md, by_m = {}, {}
    for r in rows:
        by_md.setdefault((r["method"], r["dataset"]), []).append(r)
        by_m.setdefault(r["method"], []).append(r)

    def _agg(rs):
        def mean(k):
            v = [x.get(k) for x in rs]
            v = [float(x) for x in v if x is not None and x == x]
            return float(np.mean(v)) if v else float("nan")
        out = {"n_tasks": len(rs),
               "max_edge_revision": float(np.max([x["max_edge_revision"]
                                                  for x in rs])),
               "max_edge_revision_window": float(np.max(
                   [x["max_edge_revision_window"] for x in rs])),
               "causal_all": bool(all(x["causal"] for x in rs)),
               "churn_mean": mean("churn_mean"), "churn_max":
                   float(np.max([x["churn_max"] for x in rs])),
               "n_appear": int(np.sum([x["n_appear"] for x in rs])),
               "n_vanish": int(np.sum([x["n_vanish"] for x in rs])),
               "n_flip": int(np.sum([x["n_flip"] for x in rs])),
               "n_edges_full": mean("n_edges_full"),
               "churn_at_50": mean("churn_at_50"),
               "churn_at_90": mean("churn_at_90"),
               "degenerate": bool(all(x["degenerate"] for x in rs)),
               "n_failed_fits": int(np.sum([x["n_failed_fits"] for x in rs])),
               "ok_frac": float(np.mean([1.0 if x["ok_batch"] else 0.0
                                         for x in rs])),
               "fit_time_s": float(np.sum([x["fit_time_s"] for x in rs])),
               "n_fits": int(np.sum([x["n_fits"] for x in rs]))}
        for k in ("auroc", "auprc", "f1"):
            out[f"batch_{k}"] = float(np.nanmean([x["batch"][k] for x in rs])) \
                if rs else float("nan")
            out[f"honest_{k}"] = float(np.nanmean([x["honest"][k] for x in rs]))
            out[f"delta_{k}"] = mean(f"delta_{k}")
            out[f"trail_{k}"] = mean(f"trail_{k}")
            out[f"centr_{k}"] = mean(f"centr_{k}")
            out[f"dmatch_{k}"] = mean(f"dmatch_{k}")
        # prefix-count sweep, averaged over tasks
        out["sweep"] = {}
        for n in SWEEP_NP:
            k = str(n)
            sub = [x["sweep"][k] for x in rs if k in x.get("sweep", {})]
            if sub:
                out["sweep"][k] = {
                    "max_edge_revision": float(np.max(
                        [s["max_edge_revision"] for s in sub])),
                    "churn_mean": float(np.mean([s["churn_mean"] for s in sub])),
                    "churn_max": float(np.max([s["churn_max"] for s in sub])),
                    "flip": int(np.sum([s["flip"] for s in sub])),
                    "appear": int(np.sum([s["appear"] for s in sub])),
                    "vanish": int(np.sum([s["vanish"] for s in sub]))}
        return out

    return ({f"{m}|{d}": _agg(rs) for (m, d), rs in by_md.items()},
            {m: _agg(rs) for m, rs in by_m.items()})


def print_board(agg_m, agg_md, datasets):
    print("=" * 108)
    print("THE POINT-IN-TIME BOARD -- causal discovery under truncation invariance")
    print("=" * 108)
    hdr = (f"{'method':<20}{'rev':>5}{'churn':>8}{'chn.90':>8}{'flip':>6}"
           f"{'|E|':>6}{'F1 pub':>8}{'F1 hon':>8}{'dF1':>8}"
           f"{'AP pub':>8}{'AP hon':>8}{'dAP':>8}{'dAPmat':>8}")
    print(hdr); print("-" * 108)
    for m, a in sorted(agg_m.items(), key=lambda kv: -_nan0(kv[1]["delta_auprc"])):
        flag = "*" if a["degenerate"] else " "
        print(f"{_LABEL.get(m, m)[:19]:<19}{flag}"
              f"{a['max_edge_revision']:>5.0f}{a['churn_mean']:>8.3f}"
              f"{a['churn_at_90']:>8.3f}{a['n_flip']:>6d}"
              f"{a['n_edges_full']:>6.1f}"
              f"{a['batch_f1']:>8.3f}{a['honest_f1']:>8.3f}{a['delta_f1']:>+8.3f}"
              f"{a['batch_auprc']:>8.3f}{a['honest_auprc']:>8.3f}"
              f"{a['delta_auprc']:>+8.3f}{a['dmatch_auprc']:>+8.3f}")
    print("-" * 108)
    print("rev = max_edge_revision (0 = no asserted edge ever revised; 1 = revised)")
    print("churn = size of that revision: Jaccard distance between the graph you hold "
          "for time tau if you\n        run the analysis AT tau, and the graph you hold "
          "for that same tau having seen everything")
    print("dF1/dAP = leakage_delta (published minus right-edge honest); "
          "dAPmat = sample-size-matched look-ahead")
    print("* = degenerate: the method asserted NO edges anywhere, so its zero "
          "revision is vacuous")
    print()
    # per-dataset detail
    for d in datasets:
        sub = {k.split("|")[0]: v for k, v in agg_md.items()
               if k.endswith("|" + d)}
        if not sub:
            continue
        print(f"--- {_DSLABEL.get(d, d)} " + "-" * (100 - len(d)))
        print(f"{'method':<20}{'rev':>5}{'churn':>8}{'AP pub':>8}{'AP hon':>8}"
              f"{'dAP':>8}{'F1 pub':>8}{'F1 hon':>8}{'dF1':>8}{'|E|':>6}"
              f"{'fails':>7}")
        for m, a in sorted(sub.items(), key=lambda kv: -_nan0(kv[1]["delta_auprc"])):
            print(f"{_LABEL.get(m, m)[:19]:<20}{a['max_edge_revision']:>5.0f}"
                  f"{a['churn_mean']:>8.3f}{a['batch_auprc']:>8.3f}"
                  f"{a['honest_auprc']:>8.3f}{a['delta_auprc']:>+8.3f}"
                  f"{a['batch_f1']:>8.3f}{a['honest_f1']:>8.3f}"
                  f"{a['delta_f1']:>+8.3f}{a['n_edges_full']:>6.1f}"
                  f"{a['n_failed_fits']:>7d}")
        print()
    # prefix sweep
    print("--- how revision accumulates with the number of prefixes " + "-" * 45)
    print(f"{'method':<20}" + "".join(f"{'np=' + str(n):>12}" for n in SWEEP_NP))
    for m, a in sorted(agg_m.items(), key=lambda kv: -_nan0(kv[1]["churn_mean"])):
        cells = []
        for n in SWEEP_NP:
            s = a["sweep"].get(str(n))
            cells.append(f"{s['churn_mean']:.3f}/{s['max_edge_revision']:.0f}"
                         if s else "  --  ")
        print(f"{_LABEL.get(m, m)[:19]:<20}" + "".join(f"{c:>12}" for c in cells))
    print("(cell = mean churn / max_edge_revision at that number of prefixes)")
    print()

    # ------------------------------------------------------------------ #
    # The verdict, stated plainly -- including anything that went the other way.
    # ------------------------------------------------------------------ #
    print("--- verdict " + "-" * 92)
    revised = [m for m, a in agg_m.items() if a["max_edge_revision"] > TOL]
    clean = [m for m, a in agg_m.items() if a["max_edge_revision"] <= TOL]
    vac = [m for m in clean if agg_m[m]["degenerate"]]
    real = [m for m in clean if not agg_m[m]["degenerate"]]
    print(f"revise an already-asserted edge ({len(revised)}/{len(agg_m)}): "
          f"{[_LABEL.get(m, m) for m in revised]}")
    print(f"max_edge_revision == 0 and NOT vacuous ({len(real)}): "
          f"{[_LABEL.get(m, m) for m in real]}")
    if vac:
        print(f"max_edge_revision == 0 but VACUOUS -- asserted no edges "
              f"({len(vac)}): {[_LABEL.get(m, m) for m in vac]}")
    d = {m: a["delta_auprc"] for m, a in agg_m.items()
         if a["delta_auprc"] == a["delta_auprc"] and not a["degenerate"]}
    if d:
        pos = {k: v for k, v in d.items() if v > 0}
        neg = {k: v for k, v in d.items() if v < 0}
        worst = max(d, key=d.get)
        print(f"leakage_delta (AUPRC) > 0 for {len(pos)}/{len(d)} non-degenerate "
              f"methods; < 0 for {len(neg)} "
              f"({[_LABEL.get(k, k) for k in neg]})")
        print(f"  largest: {_LABEL.get(worst, worst)} {d[worst]:+.3f} AUPRC "
              f"({agg_m[worst]['batch_auprc']:.3f} published -> "
              f"{agg_m[worst]['honest_auprc']:.3f} honest)")
        print(f"  mean over non-degenerate methods: {np.mean(list(d.values())):+.3f}")
    dm = {m: a["dmatch_auprc"] for m, a in agg_m.items()
          if a["dmatch_auprc"] == a["dmatch_auprc"] and not a["degenerate"]}
    if dm:
        print(f"sample-size-matched look-ahead: mean "
              f"{np.mean(list(dm.values())):+.3f} AUPRC, "
              f"{sum(1 for v in dm.values() if v > 0)}/{len(dm)} positive")
    still = {m: a["churn_at_90"] for m, a in agg_m.items()
             if a["churn_at_90"] == a["churn_at_90"] and not a["degenerate"]}
    if still:
        w = max(still, key=still.get)
        print(f"still being rewritten at 90% of the series: mean churn "
              f"{np.mean(list(still.values())):.3f}, worst "
              f"{_LABEL.get(w, w)} {still[w]:.3f}")
    fails = {m: a["n_failed_fits"] for m, a in agg_m.items() if a["n_failed_fits"]}
    print(f"failed/timed-out fits (kept, not dropped): {fails if fails else 'none'}")
    if "cafe_cd" in agg_m:
        print()
        print("CAVEAT -- CAFE-CD's QUALITY half is provisional. "
              "cafe.discovery.CausalGraph.scores()")
        print("  currently returns a hard two-tier step, so true-but-unselected "
              "edges rank below")
        print("  selected false positives and its AUROC/AUPRC are UNDERSTATED. "
              "Its revision")
        print("  certificate is unaffected -- that reads the support under "
              "truncation and is a")
        print("  structural consequence of only ever reading rows <= t -- and so "
              "is its F1.")
        print("  Every rival's leakage_delta is computed entirely from that "
              "rival's own outputs")
        print("  on its own metric and is unaffected either way.")
    print("=" * 108)


def _nan0(x):
    return -1e9 if x != x else float(x)


# --------------------------------------------------------------------------- #
def write_table(agg_m, agg_md, meta):
    n_tasks = meta["n_tasks"]
    rows = sorted(agg_m.items(), key=lambda kv: -_nan0(kv[1]["delta_auprc"]))
    L = [
        r"\begin{table*}[tbp]\centering\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{\textbf{The point-in-time board --- a look-ahead audit for causal "
        r"\emph{discovery}.} Published time-series discovery methods are \emph{batch}: "
        r"they consume the whole series before emitting a graph, so a graph used to "
        r"decide anything at time $t$ carries information from after $t$. Every method "
        r"here is run through one neutral standard --- "
        r"\texttt{cafe.audit.verify\_causal\_series}, the same truncation-invariance "
        r"certificate CAFE publishes for its imputations, with the graph support passed "
        r"as the 0/1 indicator that function documents. Write $\mathrm{rep}(Z,t)$ for "
        r"the graph a method asserts \emph{for time} $t$ when the analysis is run on "
        r"rows $Z$. \textsc{rev} $=\max$ edge revision: $0$ iff "
        r"$\mathrm{rep}(X_{:\tau},t)=\mathrm{rep}(X,t)$ for every checkpoint "
        r"$t\le\tau$, i.e.\ nothing already asserted is ever rewritten. "
        r"\textsc{churn} is the size of that revision (Jaccard distance between the two), "
        r"averaged over prefixes and again at $\tau=0.9T$; \textsc{flip} counts edges "
        r"whose orientation reverses. $\Delta$ is the \emph{leakage delta} on each "
        r"method's own metric, own threshold and own data: quality fitted on the whole "
        r"series (as published) minus right-edge honest quality (refitted on each prefix "
        r"and read out at $t$). $\Delta_{\mathrm{m}}$ repeats it at \emph{identical "
        r"sample size} --- a length-$L$ window straddling $t$ against a length-$L$ "
        r"trailing window --- so the sample-size effect is removed and only look-ahead "
        r"remains. Mean over " + str(n_tasks) + r" tasks (synthetic with known graphs, "
        r"CausalRivers, CausalDynamics); $^{\dagger}$ marks a method that asserted no "
        r"edges at all, whose zero revision is vacuous rather than point-in-time. "
        r"$^{\ddagger}$ CAFE-CD's \emph{quality} columns are provisional --- its edge "
        r"ranking is a two-tier step rather than a calibrated score, which understates "
        r"any rank-based metric; the revision certificate reads the support under "
        r"truncation and is unaffected, as is $F_1$. Every rival's $\Delta$ is computed "
        r"entirely from that rival's own outputs on its own metric.}",
        r"\label{tab:pitboard}",
        r"\begin{tabular}{@{}lccccrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{revision certificate} & "
        r"\multicolumn{4}{c}{AUPRC} & \multicolumn{2}{c}{$F_1$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}",
        r"Method & rev & churn & churn$_{.9}$ & flips & pub. & honest & $\Delta$ & "
        r"$\Delta_{\mathrm{m}}$ & pub. & $\Delta$ \\",
        r"\midrule",
    ]
    for m, a in rows:
        nm = _LABEL.get(m, m).replace("_", r"\_")
        if m == "cafe_cd":
            nm = r"\textbf{%s}" % nm
        if a["degenerate"]:
            nm += r"$^{\dagger}$"
        if m == "cafe_cd":
            nm += r"$^{\ddagger}$"
        rev = r"$\mathbf{0}$" if a["max_edge_revision"] <= TOL else r"$1$"
        L.append(
            f"{nm} & {rev} & {a['churn_mean']:.3f} & {_f(a['churn_at_90'])} & "
            f"{a['n_flip']:d} & {_f(a['batch_auprc'])} & {_f(a['honest_auprc'])} & "
            f"{_fs(a['delta_auprc'])} & {_fs(a['dmatch_auprc'])} & "
            f"{_f(a['batch_f1'])} & {_fs(a['delta_f1'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    txt = "\n".join(L) + "\n"
    with open(TEX, "w") as f:
        f.write(txt)
    print(f"[table] wrote {TEX}")
    return txt


def _f(x):
    return "--" if x != x else f"{x:.3f}"


def _fs(x):
    return "--" if x != x else f"{x:+.3f}"


def write_figure(rows, agg_m, meta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # pragma: no cover
        print("[fig] matplotlib unavailable, skipping:", e)
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) churn curves -- how the asserted graph is revised as the series grows
    ax = axes[0, 0]
    by_m = {}
    for r in rows:
        by_m.setdefault(r["method"], []).append(r)
    cmap = plt.get_cmap("tab20")
    for i, (m, rs) in enumerate(sorted(by_m.items())):
        fr = np.array([c["frac"] for c in rs[0]["curve"]])
        vals = []
        for r in rs:
            v = np.array([c["jaccard"] for c in r["curve"]])
            if v.shape == fr.shape:
                vals.append(v)
        if not vals:
            continue
        y = np.mean(vals, axis=0)
        style = dict(lw=2.4, color="#111111", zorder=5) if m == "cafe_cd" else \
            dict(lw=1.2, color=cmap(i % 20), alpha=0.85)
        ax.plot(fr, y, label=_LABEL.get(m, m), **style)
    ax.set_xlabel(r"prefix fraction $\tau/T$ (data available at decision time)")
    ax.set_ylabel("edge churn vs. the finally-published graph")
    ax.set_title("(a) The asserted graph keeps being rewritten")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)

    # (b) leakage delta bars
    ax = axes[0, 1]
    order = sorted(agg_m.items(), key=lambda kv: _nan0(kv[1]["delta_auprc"]))
    names = [_LABEL.get(m, m) for m, _ in order]
    d = [a["delta_auprc"] for _, a in order]
    dm = [a["dmatch_auprc"] for _, a in order]
    y = np.arange(len(names))
    ax.barh(y - 0.2, d, height=0.38, color=["#111111" if m == "cafe_cd" else "#e76f51"
                                            for m, _ in order],
            edgecolor="black", lw=0.4, label=r"$\Delta$ (published $-$ honest)")
    ax.barh(y + 0.2, dm, height=0.38, color="#457b9d", edgecolor="black", lw=0.4,
            label=r"$\Delta_{\rm match}$ (same $n$, window straddles $t$)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("AUPRC borrowed from the future")
    ax.set_title("(b) What look-ahead is worth, per method")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.25, lw=0.5, axis="x")

    # (c) prefix-count sweep
    ax = axes[1, 0]
    for i, (m, a) in enumerate(sorted(agg_m.items())):
        xs = [n for n in SWEEP_NP if str(n) in a["sweep"]]
        ys = [a["sweep"][str(n)]["churn_mean"] for n in xs]
        if not xs:
            continue
        style = dict(lw=2.4, color="#111111", marker="o", zorder=5) \
            if m == "cafe_cd" else dict(lw=1.2, color=cmap(i % 20), marker=".",
                                        alpha=0.85)
        ax.plot(xs, ys, **style)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(SWEEP_NP))
    ax.set_xticklabels([str(n) for n in SWEEP_NP])
    ax.set_xlabel("number of prefixes audited")
    ax.set_ylabel("mean edge churn")
    ax.set_title("(c) More checkpoints, more revisions found")
    ax.grid(alpha=0.25, lw=0.5)

    # (d) published vs honest scatter
    ax = axes[1, 1]
    for m, a in agg_m.items():
        c = "#111111" if m == "cafe_cd" else "#e76f51"
        ax.scatter(a["honest_auprc"], a["batch_auprc"], s=34, color=c,
                   edgecolor="black", lw=0.4, zorder=4)
        ax.annotate(_LABEL.get(m, m), (a["honest_auprc"], a["batch_auprc"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    lim = [0, 1]
    ax.plot(lim, lim, ls="--", color="black", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("AUPRC you would actually have had at time $t$ (honest)")
    ax.set_ylabel("published AUPRC (fitted on the whole series)")
    ax.set_title("(d) Above the diagonal = the number owes to look-ahead")
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("The point-in-time board: every published causal-discovery method "
                 "is batch, and nobody has measured what that costs", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG)
    plt.close(fig)
    print(f"[fig] wrote {FIG}")


# =========================================================================== #
def _save(rows, meta):
    """Checkpoint after EVERY cell -- this run is long and gets interrupted."""
    slim = []
    for r in rows:
        slim.append(r)
    with open(CACHE, "w") as f:
        json.dump({"rows": slim, "meta": meta}, f, indent=1, default=float)


def _load_cache():
    if not (RESUME and os.path.exists(CACHE)):
        return []
    try:
        with open(CACHE) as f:
            d = json.load(f)
        return list(d.get("rows", []))
    except Exception as e:                                   # noqa: BLE001
        print("[cache] unreadable, starting fresh:", repr(e)[:120])
        return []


def main():
    t_start = time.time()
    print(f"PIT BOARD | datasets={DATASETS} methods={len(METHODS)} seeds={SEEDS} "
          f"T<={TMAX} ncp={NCP} nmatch={NMATCH} timeout={TIMEOUT}s")
    tasks = build_tasks()
    print(f"tasks ({len(tasks)}): {[t['key'] for t in tasks]}\n")
    if not tasks:
        print("no tasks; nothing to do"); return

    rows = _load_cache()
    done = {(r["method"], r["task"]) for r in rows}
    if done:
        print(f"[cache] resuming with {len(done)} finished cells\n")

    meta = {"datasets": DATASETS, "methods": METHODS, "seeds": SEEDS,
            "T_max": TMAX, "n_checkpoints": NCP, "n_matched": NMATCH,
            "timeout_s": TIMEOUT, "sweep_n_prefixes": list(SWEEP_NP),
            "tol": TOL, "n_tasks": len(tasks),
            "task_notes": {t["key"]: t["note"] for t in tasks},
            "task_shapes": {t["key"]: list(t["X"].shape) for t in tasks},
            "task_n_true": {t["key"]: int((t["A_true"] != 0).sum()) for t in tasks},
            "available": CB.available(),
            "cafe_cd_quality_provisional": True,
            "cafe_cd_quality_note":
                "CausalGraph.scores() is a two-tier step, not a calibrated ranking: "
                "CAFE-CD's AUROC/AUPRC are understated. max_edge_revision and F1 read "
                "the support and are unaffected. Rivals' numbers come entirely from "
                "their own outputs. Re-score from rows[...]['raw'], or delete the "
                "cafe_cd rows and re-run with PIT_RESUME=1."}

    if REPORT_ONLY:
        if not rows:
            print("nothing cached; run without PIT_REPORT_ONLY first"); return
        with open(CACHE) as f:
            meta = json.load(f).get("meta", meta)
        agg_md, agg_m = aggregate(rows)
        print_board(agg_m, agg_md,
                    [d for d in DATASETS if any(r["dataset"] == d for r in rows)])
        write_table(agg_m, agg_md, meta)
        write_figure(rows, agg_m, meta)
        return

    plan = [(t, m) for t in tasks for m in METHODS]
    if cafe_cd_available():
        plan += [(t, "cafe_cd") for t in tasks]
        print("[cafe-cd] src/cafe/discovery.py IS importable -- auditing it too\n")
    else:
        print("[cafe-cd] src/cafe/discovery.py not importable yet -- running the "
              "rival audit, which is independent of it\n")

    for i, (task, m) in enumerate(plan, 1):
        if (m, task["key"]) in done:
            continue
        t0 = time.time()
        try:
            row = audit_cell(task, m)
        except Exception as e:                               # noqa: BLE001
            print(f"  [{i}/{len(plan)}] {m:>18s} @ {task['key']:<22s} "
                  f"FAILED {type(e).__name__}: {str(e)[:100]}")
            row = {"dataset": task["dataset"], "task": task["key"], "method": m,
                   "T": int(task["X"].shape[0]), "P": int(task["X"].shape[1]),
                   "max_lag": int(task["max_lag"]),
                   "max_edge_revision": float("nan"),
                   "max_edge_revision_window": float("nan"),
                   "causal": False, "churn_mean": float("nan"),
                   "churn_max": float("nan"), "n_appear": 0, "n_vanish": 0,
                   "n_flip": 0, "sweep": {}, "curve": [],
                   "batch": {"auroc": float("nan"), "auprc": float("nan"),
                             "f1": float("nan"), "n_edges": 0},
                   "honest": {"auroc": float("nan"), "auprc": float("nan"),
                              "f1": float("nan"), "n_edges": 0},
                   "delta_auroc": float("nan"), "delta_auprc": float("nan"),
                   "delta_f1": float("nan"), "n_edges_full": 0,
                   "degenerate": False, "n_fits": 0, "fit_time_s": 0.0,
                   "wall_s": time.time() - t0, "n_failed_fits": 1,
                   "failures": [f"{type(e).__name__}: {str(e)[:160]}"],
                   "ok_batch": False, "ok_any": False,
                   "err_batch": f"{type(e).__name__}: {str(e)[:160]}"}
            for k in ("auroc", "auprc", "f1"):
                row[f"trail_{k}"] = float("nan")
                row[f"centr_{k}"] = float("nan")
                row[f"dmatch_{k}"] = float("nan")
            row["matched"] = []; row["L_match"] = 0
        else:
            print(f"  [{i}/{len(plan)}] {m:>18s} @ {task['key']:<22s} "
                  f"rev={row['max_edge_revision']:.0f} "
                  f"churn={row['churn_mean']:.3f} "
                  f"dAP={row['delta_auprc']:+.3f} "
                  f"|E|={row['n_edges_full']:d} "
                  f"fits={row['n_fits']} {row['wall_s']:.1f}s"
                  + ("  [DEGENERATE: no edges]" if row["degenerate"] else "")
                  + ("" if row["ok_batch"] else
                     f"  [FAIL: {row['err_batch'][:60]}]"))
        rows.append(row)
        done.add((m, task["key"]))
        _save(rows, meta)               # checkpoint every cell

    agg_md, agg_m = aggregate(rows)
    meta["wall_s"] = time.time() - t_start
    with open(CACHE, "w") as f:
        json.dump({"rows": rows, "agg_method": agg_m,
                   "agg_method_dataset": agg_md, "meta": meta},
                  f, indent=1, default=float)
    print(f"\n[cache] wrote {CACHE}\n")

    print_board(agg_m, agg_md, [d for d in DATASETS
                                if any(r["dataset"] == d for r in rows)])
    write_table(agg_m, agg_md, meta)
    write_figure(rows, agg_m, meta)
    print(f"\ntotal {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
