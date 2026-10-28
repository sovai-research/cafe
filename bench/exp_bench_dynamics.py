r"""CausalDynamics head-to-head: 12 external baselines + CAFE-CD, three tiers.

Run
---
    PYTHONPATH=bench .venv-bench/bin/python bench/exp_bench_dynamics.py --all
    ... --cells published,orientation,calibration,chance,h2h,runtime,pit
    ... --render                      # tables + figure from the cache only

Everything is checkpointed to ``bench/bench_dynamics_cache.json`` at
*sub-cell* granularity (one entry per tier/setting/system), so a killed run
resumes exactly where it stopped.  ``--force CELL`` re-runs one cell.

WHAT THIS BENCHMARK IS, AND WHAT IT IS NOT
------------------------------------------
CausalDynamics (NeurIPS 2025 D&B) ships its OWN evaluation outputs alongside
the data, so the published leaderboard can be recovered exactly rather than
quoted.  Doing that first (cell ``published``) is what makes the rest
interpretable, and it produces the central diagnostic of this experiment:

  * on the ``simple`` tier the ground-truth graph is dense (~69% including
    the diagonal), the null (constant-prediction) AUPRC is 0.627 averaged over
    the ten noise/confounder settings, and all ten published methods land
    within +0.025 of it.  AUROC spans [0.485, 0.516] against a 0.500 null.
    **As published, nothing on that tier beats chance.**  Cell ``calibration``
    shows WHY, and it is not that the tier is unlearnable: the shipped
    wrappers binarise (``abs(A) > 0``), which turns a dense estimate into a
    constant prediction whose AUROC is 0.500 *by construction*, and two of
    them are transposed against that tier's ground truth (see below).  Scored
    with continuous evidence and the correct orientation, an ordinary VAR is
    clearly above chance there.  Both facts belong in the report.
  * on ``coupled`` the graph is sparse (null AUPRC 0.304 over all 720
    settings) and the published spread is real but small (AUROC 0.498-0.582,
    AUPRC 0.304-0.378).  This is the tier where AUPRC carries information.
  * on ``climate`` only two systems exist and their shipped ground truth is
    (near-)symmetric -- ``coupled_enso_modes`` is exactly symmetric, and
    ``coupled_atmos_ocean`` has a single asymmetric pair out of 16.  Direction
    is therefore all but unscoreable: it is a skeleton/association task
    wearing a causal-discovery label, and no directional claim should be made
    from it either way.

We therefore report per-tier and refuse to aggregate across tiers.

THE ORIENTATION TRAP -- READ BEFORE CHANGING ANY SCORING LINE
-------------------------------------------------------------
This repo's convention is ``A[effect, cause]`` (row = the variable being
caused); see ``docs/causal_baselines.md`` Section 3.  CausalDynamics uses BOTH
conventions, one per tier, while scoring every method with the same
untransposed wrapper:

  ``simple``   ``get_adjacency_matrix_from_jac`` (systems.py:372): "A value of
               1 at position (i,j) indicates that variable j affects the
               evolution of variable i"  ->  [effect, cause]  ->  OURS.
               Verified numerically here: the shipped Rossler matrix is
               [[0,1,1],[1,1,0],[1,0,1]], exactly the binary Jacobian of
               xdot=-y-z, ydot=x+ay, zdot=b+z(x-c).
  ``coupled``  ``scm.py`` (create_graph_from_adjacency_matrix, l.216):
               "A[i,j] indicates an edge from i to j"  ->  [cause, effect]
               ->  the TRANSPOSE of ours.
  ``climate``  ``generate_climate.py`` builds ``(node_in, node_out)``, i.e.
               [cause, effect] as well -- but every shipped climate truth is
               symmetric, so it does not matter.

Both tiers are written to disk with the same ``["node_in","node_out"]`` axis
labels (``data_io.py``:78-97), so the file metadata does NOT disambiguate
them.  Cell ``orientation`` re-derives the answer from the data itself, using
a check that needs no estimator: the ``coupled`` files ship ``root_nodes``
(the nodes the simulator drove with a raw chaotic trajectory because they have
no parents), and a parentless node has an all-zero COLUMN under [cause,effect]
and an all-zero ROW under [effect,cause].  Column wins 347/400.  For
``simple`` the shipped Rossler matrix is compared against its analytic binary
Jacobian and matches exactly.

The consequence for the published leaderboard is recorded in
``PUBLISHED_ORIENTATIONS``: DYNOTEARS and PCMCI+ emit [cause,effect], VARLiNGAM
and RCD emit [effect,cause], F-PCMCI emits an undirected skeleton -- and
upstream transposes none of them.  So each directed wrapper is correctly
aligned on at most ONE of the two tiers, and part of the tier-to-tier ranking
churn on the published board is a convention artefact rather than a difference
in method quality.  It is exactly why DYNOTEARS sits BELOW chance (0.426) on
``simple`` and above it (0.546) on ``coupled``.

SELF-LOOP CONVENTION
--------------------
CausalDynamics scores the FULL P*P matrix, diagonal included: its ground truth
is a Jacobian sparsity pattern whose diagonal is a real edge (self-dynamics).
We score under that convention as the primary number -- it is the only one
comparable to the published board -- and additionally report an off-diagonal
("_off") number for every metric, because on ``simple`` the diagonal is 3 of 9
cells and is trivially predicted, so the primary number flatters everyone.
The off-diagonal number is the one that says whether a method found any
cross-variable STRUCTURE.  Both are always printed; neither is hidden.

TWO BOARDS
----------
CAFE-CD is point-in-time: its graph at time t uses only data <= t.  Every one
of the 12 rivals is batch.  ``src/cafe/benchmark.py`` already refuses to rank
causal against bidirectional imputers in one table; the same discipline
applies here.  Cell ``h2h`` is the BATCH board (everyone sees the whole
series, like-for-like).  Cell ``pit`` is the POINT-IN-TIME board: graphs are
scored after seeing only the first f in {25%, 50%, 100%} of the series, with
batch rivals refit on each prefix.  The two boards are never merged, and no
cross-board ranking is printed.

PROTOCOL (identical for every method, no exceptions)
----------------------------------------------------
* data       upstream ``eval.py`` verbatim: ``time_series`` (T, R, P) or
             (T, R, P, D) -> take ``[..., 0]`` if 4-D, transpose to (R, T, P),
             ``np.nan_to_num``.  Ground truth is ``adjacency_matrix_summary``
             if present else ``adjacency_matrix``; systems whose truth is
             all-ones or all-zeros are skipped (AUROC undefined) -- exactly
             what upstream skips.
* standardise ``bench/eval_utils.standardize_on_observed`` per realisation
             (nothing is held out here, so the mask is all-False and it is a
             plain leak-free per-column z-score).  Upstream instead z-scores
             jointly over (realisation, time); cell ``calibration`` measures
             the difference and reports it rather than assuming it away.
* lag budget K = 3 for every method in every cell except ``calibration``,
             which uses upstream's tau_max = 1 to reproduce their numbers.
* summary    ``causal_baselines.summary_graph(res, how='max',
             include_contemp=True)``: max over |lagged| and |contemporaneous|.
             The ground truth is a Jacobian, which shows up as both a lag-1
             and a contemporaneous dependence under discrete sampling, so a
             method is scored on every channel it can fill.  Methods that
             cannot fill a channel (contemp-only: directlingam, dagma_linear,
             pc, fci; lag-only: granger_pairwise) are NOT penalised for the
             missing one -- max ignores it.
* scores     CONTINUOUS |score|, not upstream's ``abs(A) > 0`` binarisation.
             Binarising throws away the ranking AUROC needs and is why four
             shipped baselines score exactly 0.500 (a constant all-ones
             prediction).  ``calibration`` reproduces the binarised numbers;
             every other cell uses continuous scores, for everyone equally.
* binary     F1 and SHD use each method's OWN support (``support_lag`` OR
             ``support_contemp``), not an oracle threshold.  SHD is the plain
             Hamming distance over all P*P cells INCLUDING the diagonal,
             matching upstream's ``Joint SHD`` divided by the batch size.
* timeout    600 s per fit, identical for all.  Failures and timeouts are
             recorded as ``n_fail`` / ``n_timeout`` and scored as an all-zero
             graph (upstream does the same); nothing is silently dropped.
* runtime    measured in a SEPARATE sequential pass (cell ``runtime``) so the
             process pool used by ``h2h`` cannot contaminate wall-clock.

CAFE-CD SLOT
------------
``cafe.discover(X)`` is expected to return an object exposing
``.adjacency(lag=None)`` and ``.scores(lag=None)``.  Until
``src/cafe/discovery.py`` lands, ``cafe_cd_available()`` is False, the method
is reported as ABSENT in every table, and every other cell still runs.  No
part of this file re-implements it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Pin BLAS to one thread BEFORE numpy is imported.  Two reasons, both load-
# bearing: (a) the runtime table claims "single-threaded CPU wall-clock", which
# is only true if the linear algebra underneath is actually single-threaded;
# (b) the head-to-head runs a process pool, and an unpinned BLAS gives each of
# 12 workers up to a core's worth of threads apiece, so the machine
# oversubscribes ~10x and every method is slowed by an amount that depends on
# how much BLAS it happens to use -- i.e. unequally.  Set before import; numpy
# reads these once.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import causal_baselines as cb                     # noqa: E402
import causal_metrics as cm                       # noqa: E402
from eval_utils import standardize_on_observed    # noqa: E402

CACHE = os.path.join(_HERE, "bench_dynamics_cache.json")
CD_ROOT = os.path.join(_ROOT, "data", "causal", "causaldynamics", "extracted")
TEX_OUT = os.path.join(_ROOT, "paper", "tables", "bench_dynamics.tex")
FIG_OUT = os.path.join(_ROOT, "paper", "figures", "bench_dynamics.pdf")

# --------------------------------------------------------------------------- #
# Configuration -- one place, so "same budget for everyone" is checkable
# --------------------------------------------------------------------------- #
MAX_LAG = 3                # uniform lag budget for the head-to-head
CALIB_LAG = 1              # upstream tau_max, calibration only
TIMEOUT_S = 600.0          # uniform per-fit timeout
N_WORKERS = max(1, min(12, (os.cpu_count() or 4) - 2))
CHANCE_SEEDS = 20          # random-score control replicates
PIT_FRACS = (0.25, 0.50, 1.00)
PIT_METHODS = ("var_ols", "varlingam", "dynotears", "pcmciplus", "pc_ts")

METHODS: Tuple[str, ...] = tuple(cb.BASELINES)          # the 12
CAFE_CD = "cafe_cd"

# CAFE-CD's ``scores()`` is known to return a hard two-tier step (selected
# edges in [0.5,1], everything else in [0,0.5)), which buries the continuous
# evidence underneath and makes every threshold-free metric a LOWER BOUND by an
# unknown margin.  While this flag is True the CAFE-CD row is computed, cached
# and printed, but is labelled PROVISIONAL and no win/loss verdict is drawn
# from it.  Clear the flag only once ``scores()`` returns continuous evidence.
CAFE_CD_PROVISIONAL = True
CAFE_CD_PROVISIONAL_NOTE = (
    "CAFE-CD numbers are PROVISIONAL: scores() currently emits a two-tier step "
    "rather than continuous evidence, so its AUROC/AUPRC are lower bounds and "
    "no ranking claim is made from them. F1/SHD come from adjacency() and are "
    "less affected but still reflect a low-recall selection.")

# Ground-truth orientation per tier.  True  => the file stores [cause, effect]
# and must be transposed into this repo's [effect, cause] before scoring.
# Derived from the generators, then re-checked empirically in cell 'orientation'.
TRUTH_IS_CAUSE_ROW = {"simple": False, "coupled": True, "climate": True}

# Tiers actually evaluated.  'simple' settings are the three that isolate the
# benchmark's own axes: clean / observation noise / hidden confounder.
SIMPLE_SETTINGS = ("noise=0.00_confounder=False",
                   "noise=1.00_confounder=False",
                   "noise=0.00_confounder=True")
CLIMATE_SETTINGS = ("coupled_atmos_ocean", "coupled_enso_modes")

# The 'coupled' inputs are a single 6.5 GB tar.gz with no per-file HTTP access,
# streamed sequentially and extracted here (720 settings, 14 096 systems).
# Running all of them would drown the other two tiers 20:1, so the coupled rows
# use a DETERMINISTIC one-factor-at-a-time design around a fixed reference
# cell: every setting below differs from the reference in exactly one axis.
# This is stated rather than sampled, so it is reproducible and cannot be
# accused of cherry-picking.  The shipped eval for ALL 720 settings is still
# used verbatim in cell 'published'.
_COUPLED_DIR = os.path.join(CD_ROOT, "coupled")

_COUPLED_REF = dict(coupling="linear", noise="1.00", systems="5",
                    confounder="False", standardize="False", timelag="1",
                    activation="mixed")
_COUPLED_SWEEP = {                       # axis -> values swept (reference incl.)
    "coupling": ("linear", "nonlinear", "periodic"),
    "noise": ("0.00", "0.50", "1.00", "1.50", "2.00"),
    "systems": ("3", "5", "10"),
    "confounder": ("False", "True"),
    "timelag": ("0", "1"),
}
COUPLED_SYSTEMS_CAP = 12                 # systems per setting (of ~20)


def _coupled_name(d: Dict[str, str]) -> str:
    return ("coupling={coupling}_noise={noise}_systems={systems}"
            "_confounder={confounder}_standardize={standardize}"
            "_timelag={timelag}_activation={activation}").format(**d)


def coupled_settings() -> List[str]:
    """The one-factor-at-a-time design, restricted to what is on disk."""
    if not os.path.isdir(_COUPLED_DIR):
        return []
    want: List[str] = []
    for axis, vals in _COUPLED_SWEEP.items():
        for v in vals:
            d = dict(_COUPLED_REF)
            d[axis] = v
            nm = _coupled_name(d)
            if nm not in want:
                want.append(nm)
    out = []
    for nm in want:
        d = os.path.join(_COUPLED_DIR, nm, "data")
        if os.path.isdir(d) and any(f.endswith(".nc") for f in os.listdir(d)):
            out.append(nm)
    return out


def tiers() -> List[Tuple[str, str, str]]:
    """[(tier_label, subset, setting)] for every tier with local inputs."""
    out: List[Tuple[str, str, str]] = []
    for st in SIMPLE_SETTINGS:
        if os.path.isdir(os.path.join(CD_ROOT, "simple", st, "data")):
            out.append((f"simple/{st}", "simple", st))
    for st in CLIMATE_SETTINGS:
        if os.path.isdir(os.path.join(CD_ROOT, "climate", st, "data")):
            out.append((f"climate/{st}", "climate", st))
    for st in coupled_settings():
        out.append((f"coupled/{st}", "coupled", st))
    return out


# Short tier-group labels used for aggregation in the tables.
def tier_group(label: str) -> str:
    return label.split("/", 1)[0]


# --------------------------------------------------------------------------- #
# Cache -- atomic, sub-cell granularity
# --------------------------------------------------------------------------- #
def cache_load() -> Dict[str, Any]:
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception as e:                          # corrupt -> keep a copy
            os.replace(CACHE, CACHE + ".corrupt")
            print(f"[cache] unreadable ({e}); moved to {CACHE}.corrupt")
    return {}


def cache_save(obj: Dict[str, Any]) -> None:
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_jsonable)
    os.replace(tmp, CACHE)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


class Cache:
    """Dict-of-dicts with a flush-on-write policy."""

    def __init__(self):
        self.d = cache_load()

    def cell(self, name: str) -> Dict[str, Any]:
        return self.d.setdefault(name, {})

    def put(self, cell: str, key: str, value: Any, flush: bool = True) -> None:
        self.cell(cell)[key] = value
        if flush:
            cache_save(self.d)

    def has(self, cell: str, key: str) -> bool:
        return key in self.d.get(cell, {})

    def drop(self, cell: str) -> None:
        self.d.pop(cell, None)
        cache_save(self.d)

    def flush(self) -> None:
        cache_save(self.d)


# --------------------------------------------------------------------------- #
# Data -- upstream eval.py, verbatim
# --------------------------------------------------------------------------- #
def system_paths(subset: str, setting: str) -> List[str]:
    d = os.path.join(CD_ROOT, subset, setting, "data")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".nc"))


def load_system(path: str, subset: str, *, upstream_std: Optional[bool] = False
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(cube (R,T,P) standardised, A_true (P,P) in [effect, cause]) or None.

    None when the system is one upstream skips (all-ones / all-zeros truth).
    ``upstream_std=True`` reproduces eval.py's joint (realisation, time)
    z-scoring; the default is the repo primitive
    ``standardize_on_observed`` applied per realisation.
    """
    import h5py
    with h5py.File(path, "r") as f:
        ts = np.asarray(f["time_series"][...], dtype=float)
        key = ("adjacency_matrix_summary" if "adjacency_matrix_summary" in f
               else "adjacency_matrix")
        A = np.asarray(f[key][...], dtype=float)
    if ts.ndim == 4:                       # partially observed -> first channel
        ts = ts[..., 0]
    ts = np.nan_to_num(ts.transpose(1, 0, 2))                      # (R, T, P)
    if np.all(A == 1) or np.all(A == 0):
        return None
    if TRUTH_IS_CAUSE_ROW[subset]:
        A = A.T                            # -> this repo's [effect, cause]
    if upstream_std is None:                       # sensitivity: raw scale
        pass
    elif upstream_std:
        ts = (ts - ts.mean(axis=(0, 1), keepdims=True)) / (
            ts.std(axis=(0, 1), keepdims=True) + 1e-8)
    else:
        z = np.zeros(ts.shape[1:], dtype=bool)
        ts = np.stack([standardize_on_observed(x, z) for x in ts])
    return np.nan_to_num(ts), A


# --------------------------------------------------------------------------- #
# CAFE-CD adapter -- a slot, not an implementation
# --------------------------------------------------------------------------- #
_CAFE_ERR = ""


def cafe_cd_available() -> bool:
    global _CAFE_ERR
    try:
        import cafe                                            # noqa: F401
        from cafe.discovery import discover                    # noqa: F401
        return True
    except Exception as e:                                     # not landed yet
        _CAFE_ERR = f"{type(e).__name__}: {e}"
        return False


def run_cafe_cd(X: np.ndarray, max_lag: int = MAX_LAG) -> "cb.BaselineResult":
    """Wrap ``cafe.discover`` in a BaselineResult so it scores identically.

    Contract, verified against ``src/cafe/discovery.py`` as landed:
    ``discover(data, mode='batch', **kw)`` returns a ``CausalGraph`` whose
    ``.adjacency(lag=None)`` and ``.scores(lag=None)`` are both ``(P, P)`` in
    the convention ``A[target, source]`` -- i.e. ``[effect, cause]``, already
    this repo's convention, so NO transpose is applied here.  ``max_lag`` is
    accepted as a keyword; the fallback call without it exists so an older
    signature still runs rather than being scored as a failure.

    KNOWN DEFECT (upstream, fix in flight): ``scores()`` is documented as a
    "two-tier monotone blend" -- every SELECTED edge in [0.5, 1], every
    UNSELECTED pair in [0, 0.5).  Every selected false positive therefore
    outranks every unselected true edge, so AUROC/AUPRC computed from it are
    LOWER BOUNDS by an unknown margin.  ``CAFE_CD_PROVISIONAL`` keeps the row
    labelled and out of every verdict until that is fixed; ``adjacency()``
    (hence F1/SHD) is less affected but still reflects a low-recall selection.

    ``discover(..., mode='pit')`` returns a ``PitGraphs`` sequence and is the
    library's native point-in-time path.  The point-in-time board here instead
    truncates the input to a prefix and calls ``mode='batch'``, because that is
    the ONLY protocol the batch rivals can also be run under; switching to
    ``mode='pit'`` would make CAFE-CD's column non-comparable to its own row.
    """
    from cafe.discovery import discover
    t0 = time.time()
    P = int(np.asarray(X).shape[1])
    K = int(max_lag)
    try:
        try:
            g = discover(X, max_lag=K)
        except TypeError:
            g = discover(X)
        S = np.abs(np.asarray(g.scores(lag=None), dtype=float))
        H = np.asarray(g.adjacency(lag=None))
        if S.ndim == 3:                                   # (K,P,P) -> summary
            S = np.abs(S).max(axis=0)
        if H.ndim == 3:
            H = H.any(axis=0)
        S = np.nan_to_num(S)[:P, :P]
        H = np.asarray(H, dtype=bool)[:P, :P]
        return cb.BaselineResult(
            A_lag=S[None].repeat(K, 0), A_contemp=np.zeros((P, P)),
            support_lag=H[None].repeat(K, 0),
            support_contemp=np.zeros((P, P), bool),
            runtime_s=time.time() - t0, ok=True, err="",
            meta={"name": CAFE_CD, "max_lag": K, "P": P,
                  "scores_lagged": True, "scores_contemp": False,
                  "contemp_symmetric": False, "point_in_time": True})
    except Exception as e:
        return cb.BaselineResult(
            A_lag=np.zeros((K, P, P)), A_contemp=np.zeros((P, P)),
            support_lag=np.zeros((K, P, P), bool),
            support_contemp=np.zeros((P, P), bool),
            runtime_s=time.time() - t0, ok=False, err=f"{type(e).__name__}: {e}",
            meta={"name": CAFE_CD, "max_lag": K, "P": P,
                  "scores_lagged": False, "scores_contemp": False,
                  "contemp_symmetric": False, "point_in_time": True})


def run_method(name: str, X: np.ndarray, max_lag: int, timeout_s: float,
               **kw) -> "cb.BaselineResult":
    """Uniform entry point for the 12 baselines AND CAFE-CD."""
    if name == CAFE_CD:
        return run_cafe_cd(X, max_lag)
    return cb.run_baseline(name, X, max_lag=max_lag, timeout_s=timeout_s, **kw)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _auc_ap(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    """(ROC-AUC, average precision) with sklearn, for verbatim comparability
    with upstream ``score.py``.  NaN when a class is absent."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, dtype=int).ravel()
    s = np.nan_to_num(np.asarray(s, dtype=float)).ravel()
    if y.min() == y.max():
        return float("nan"), float("nan")
    return (float(roc_auc_score(y, s)), float(average_precision_score(y, s)))


def score_graph(A_true: np.ndarray, S: np.ndarray,
                H: Optional[np.ndarray] = None) -> Dict[str, float]:
    """One estimated graph against one truth, both (P,P) [effect, cause].

    Primary metrics use the FULL P*P (self-loops scored -- CausalDynamics'
    convention).  ``*_off`` metrics exclude the diagonal.  SHD is the Hamming
    distance over all P*P cells, i.e. upstream's Joint SHD per graph.
    """
    A = (np.asarray(A_true, float) != 0).astype(int)
    S = np.nan_to_num(np.abs(np.asarray(S, float)))
    P = A.shape[0]
    off = ~np.eye(P, dtype=bool)
    auroc, auprc = _auc_ap(A.ravel(), S.ravel())
    auroc_o, auprc_o = _auc_ap(A[off], S[off])
    out = {"auroc": auroc, "auprc": auprc,
           "auroc_off": auroc_o, "auprc_off": auprc_o,
           "base_rate": float(A.mean()),
           "base_rate_off": float(A[off].mean())}
    if H is not None:
        Hb = (np.asarray(H) != 0).astype(int)
        tp = int(((A == 1) & (Hb == 1)).sum())
        fp = int(((A == 0) & (Hb == 1)).sum())
        fn = int(((A == 1) & (Hb == 0)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out["f1"] = float(2 * pr * rc / (pr + rc)) if pr + rc else 0.0
        tpo = int(((A[off] == 1) & (Hb[off] == 1)).sum())
        fpo = int(((A[off] == 0) & (Hb[off] == 1)).sum())
        fno = int(((A[off] == 1) & (Hb[off] == 0)).sum())
        pro = tpo / (tpo + fpo) if tpo + fpo else 0.0
        rco = tpo / (tpo + fno) if tpo + fno else 0.0
        out["f1_off"] = float(2 * pro * rco / (pro + rco)) if pro + rco else 0.0
        out["shd"] = float(np.abs(Hb - A).sum())
        out["density_hat"] = float(Hb.mean())
    return out


def support_of(res: "cb.BaselineResult") -> np.ndarray:
    """A method's OWN binary decision, collapsed to a (P,P) summary."""
    parts = []
    if getattr(res, "support_lag", None) is not None and res.support_lag.size:
        parts.append(np.asarray(res.support_lag).any(axis=0))
    if getattr(res, "support_contemp", None) is not None and res.support_contemp.size:
        parts.append(np.asarray(res.support_contemp))
    if not parts:
        return np.zeros((res.P, res.P), bool)
    out = np.zeros_like(np.asarray(parts[0], bool))
    for p in parts:
        out |= np.asarray(p, bool)
    return out


def _mean(rows: Sequence[Dict[str, float]], k: str) -> float:
    v = [r[k] for r in rows if k in r and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------------------------------- #
# CELL: published -- recover the shipped leaderboard exactly
# --------------------------------------------------------------------------- #
def _eval_dir_stats(evdir: str) -> Optional[Dict[str, float]]:
    import h5py
    if not os.path.isdir(evdir):
        return None
    acc: Dict[str, List[float]] = {}
    for fn in sorted(os.listdir(evdir)):
        if not fn.endswith(".nc"):
            continue
        try:
            with h5py.File(os.path.join(evdir, fn), "r") as f:
                for k in f.keys():
                    acc.setdefault(k, []).append(float(f[k][()]))
        except Exception:
            continue
    if not acc:
        return None
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out["n_systems"] = float(len(next(iter(acc.values()))))
    return out


def cell_published(c: Cache) -> None:
    """Average the benchmark's own per-system eval outputs, per subset."""
    out: Dict[str, Any] = {}
    for subset in ("simple", "climate", "coupled"):
        root = os.path.join(CD_ROOT, subset)
        if not os.path.isdir(root):
            continue
        per_method: Dict[str, List[Dict[str, float]]] = {}
        settings = sorted(os.listdir(root))
        for st in settings:
            ev = os.path.join(root, st, "eval")
            if not os.path.isdir(ev):
                continue
            for m in sorted(os.listdir(ev)):
                s = _eval_dir_stats(os.path.join(ev, m))
                if s:
                    per_method.setdefault(m, []).append(s)
        agg = {}
        for m, rows in per_method.items():
            agg[m] = {k: float(np.nanmean([r[k] for r in rows if k in r]))
                      for k in ("Individual_AUROC", "Individual_AUPRC",
                                "Joint_AUROC", "Joint_AUPRC",
                                "Null_AUROC", "Null_AUPRC", "Joint_SHD")}
            agg[m]["n_settings"] = len(rows)
        out[subset] = {"n_settings_total": len(settings), "methods": agg}
        # per-setting detail for the three simple cells we evaluate
        det = {}
        for st in (SIMPLE_SETTINGS if subset == "simple"
                   else CLIMATE_SETTINGS if subset == "climate" else ()):
            ev = os.path.join(root, st, "eval")
            if not os.path.isdir(ev):
                continue
            det[st] = {m: _eval_dir_stats(os.path.join(ev, m))
                       for m in sorted(os.listdir(ev))}
        if det:
            out[subset]["per_setting"] = det
    c.put("published", "data", out)
    print_published(out)


def print_published(out: Dict[str, Any]) -> None:
    print("\n=== PUBLISHED (recovered from the benchmark's own shipped eval) ===")
    for subset, blk in out.items():
        ms = blk["methods"]
        if not ms:
            continue
        null_p = np.nanmean([v["Null_AUPRC"] for v in ms.values()])
        print(f"\n{subset}  ({blk['n_settings_total']} settings; "
              f"null AUPRC = {null_p:.3f}, null AUROC = 0.500)")
        print(f"  {'method':11s} {'AUROC':>7s} {'AUPRC':>7s} "
              f"{'dAUPRC':>7s} {'SHD':>7s}  n_set")
        for m, v in sorted(ms.items(), key=lambda kv: -kv[1]["Individual_AUPRC"]):
            print(f"  {m:11s} {v['Individual_AUROC']:7.4f} "
                  f"{v['Individual_AUPRC']:7.4f} "
                  f"{v['Individual_AUPRC'] - v['Null_AUPRC']:+7.4f} "
                  f"{v['Joint_SHD']:7.1f}  {v['n_settings']:.0f}")


# --------------------------------------------------------------------------- #
# CELL: orientation -- prove the per-tier convention from data, not docstrings
# --------------------------------------------------------------------------- #
# What each SHIPPED baseline wrapper actually emits, read off
# ``src/causaldynamics/baselines/*.py``.  Upstream scores all of them against a
# ground truth whose orientation FLIPS between tiers (see the module docstring),
# without transposing anything -- so a wrapper is correctly aligned on at most
# one tier, and the tier-to-tier ranking changes on the published leaderboard
# are partly a convention artefact rather than a difference in method quality.
PUBLISHED_ORIENTATIONS = {
    "dynotears": {
        "emits": "[cause,effect]",
        "source": "baselines/dynotears.py: adj_matrix[i,j] for i=source at "
                  "lag tau_max, j=target at lag 0",
        "aligned_on": ["coupled", "climate"], "misaligned_on": ["simple"]},
    "pcmciplus": {
        "emits": "[cause,effect]",
        "source": "baselines/pcmci.py: results['p_matrix'][...,-1] < alpha; "
                  "tigramite p_matrix[i,j,tau] is i at t-tau -> j at t",
        "aligned_on": ["coupled", "climate"], "misaligned_on": ["simple"]},
    "varlingam": {
        "emits": "[effect,cause]",
        "source": "baselines/varlingam.py: lingam adjacency_matrices_[tau], "
                  "[i,j] = j at lag tau -> i",
        "aligned_on": ["simple"], "misaligned_on": ["coupled", "climate"]},
    "fpcmci": {
        "emits": "undirected skeleton",
        "source": "baselines/pcmci.py: results.get_skeleton()[-1]",
        "aligned_on": [], "misaligned_on": []},
    "rcd": {
        "emits": "[effect,cause]",
        "source": "baselines/varlingam.py: causal-learn RCD adjacency_matrix_",
        "aligned_on": ["simple"], "misaligned_on": ["coupled", "climate"]},
}


def _root_nodes_test(subset: str) -> Optional[Dict[str, Any]]:
    """Estimator-free proof of the ground-truth orientation, where possible.

    The ``coupled`` files ship ``root_nodes`` -- the nodes the simulator drove
    with a raw chaotic trajectory because they have NO parents.  A node with no
    parents has an all-zero *column* of ``adjacency_matrix`` iff the file
    stores ``[cause, effect]``, and an all-zero *row* iff it stores
    ``[effect, cause]``.  Comparing both against the shipped ``root_nodes``
    settles the convention without running any estimator, so it cannot be
    confused with an estimator that simply gets direction wrong.
    """
    import h5py
    paths: List[str] = []
    root = os.path.join(CD_ROOT, subset)
    if not os.path.isdir(root):
        return None
    for st in sorted(os.listdir(root)):
        paths += system_paths(subset, st)
    col = row = n = 0
    for p in paths[:400]:
        try:
            with h5py.File(p, "r") as f:
                if "root_nodes" not in f:
                    return None
                A = np.asarray(f["adjacency_matrix"][...], dtype=float)
                rn = np.asarray(f["root_nodes"][...]).astype(bool)
        except Exception:
            continue
        n += 1
        col += int(np.array_equal(A.sum(0) == 0, rn))
        row += int(np.array_equal(A.sum(1) == 0, rn))
    if not n:
        return None
    return {"n": n, "agree_cause_row_[cause,effect]": col,
            "agree_effect_row_[effect,cause]": row,
            "verdict": "[cause,effect] -> TRANSPOSE" if col > row
                       else "[effect,cause] -> as-is"}


def cell_orientation(c: Cache) -> None:
    """Establish each tier's ground-truth convention, then measure the
    estimator's directional accuracy under the adopted one.

    Three independent lines of evidence, in decreasing order of authority:

    1. GENERATOR SOURCE.  ``systems.py::get_adjacency_matrix_from_jac`` for
       ``simple`` ("a 1 at (i,j) indicates that variable j affects the
       evolution of variable i" = [effect, cause]); ``mlp.py::propagate_mlp``
       for ``coupled`` ("A[i,j]=1 indicates an edge from node i to node j",
       and the code does ``A = A.T`` then treats row ``idx`` as the PARENTS of
       node ``idx``, i.e. [cause, effect]).
    2. ESTIMATOR-FREE DATA CHECK.  ``_root_nodes_test`` above -- decisive on
       ``coupled``; on ``simple`` the shipped Rossler matrix is checked
       against its analytic binary Jacobian.
    3. Estimator columns below.  These are reported LAST and are NOT used to
       choose the convention, because a method that simply gets direction
       backwards would flip the answer.  Their real job is the opposite: given
       a convention fixed by (1) and (2), they say whether a linear VAR
       recovers direction or reverses it on that tier.
    """
    res: Dict[str, Any] = {"_evidence": {}}
    rossler = os.path.join(CD_ROOT, "simple", SIMPLE_SETTINGS[0], "data",
                           "Rossler_N10_T1000.nc")
    if os.path.exists(rossler):
        import h5py
        with h5py.File(rossler, "r") as f:
            A = np.asarray(f["adjacency_matrix"][...], dtype=int)
        # xdot = -y - z ; ydot = x + a y ; zdot = b + z (x - c)
        jac = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]])
        res["_evidence"]["simple/rossler_vs_analytic_jacobian"] = {
            "file": A.tolist(), "analytic_[effect,cause]": jac.tolist(),
            "match": bool(np.array_equal(A, jac))}
    for subset in ("coupled", "simple", "climate"):
        t = _root_nodes_test(subset)
        if t:
            res["_evidence"][f"{subset}/root_nodes_test"] = t
    res["_evidence"]["published_wrapper_orientations"] = PUBLISHED_ORIENTATIONS
    for label, subset, setting in tiers():
        paths = system_paths(subset, setting)
        if not paths:
            continue
        rows_id, rows_tr, n_asym = [], [], 0
        for p in paths[:12]:
            got = load_system(p, subset)
            if got is None:
                continue
            cube, A = got                              # A already un-transposed
            A_file = A.T if TRUTH_IS_CAUSE_ROW[subset] else A   # as on disk
            if not np.allclose(A_file, A_file.T):
                n_asym += 1
            for x in cube[:3]:
                r = cb.run_baseline("var_ols", x, max_lag=MAX_LAG,
                                    timeout_s=TIMEOUT_S)
                S = cb.summary_graph(r, how="max", include_contemp=True)
                rows_id.append(score_graph(A_file, S))     # file as-is
                rows_tr.append(score_graph(A_file.T, S))   # transposed
        if not rows_id:
            continue
        tr = bool(TRUTH_IS_CAUSE_ROW[subset])
        adopted = rows_tr if tr else rows_id     # scored under the ADOPTED truth
        other = rows_id if tr else rows_tr
        res[label] = {
            "n_asymmetric_truths": n_asym, "n_scored": len(rows_id),
            "auroc_file_as_is": _mean(rows_id, "auroc"),
            "auroc_transposed": _mean(rows_tr, "auroc"),
            "auroc_off_file_as_is": _mean(rows_id, "auroc_off"),
            "auroc_off_transposed": _mean(rows_tr, "auroc_off"),
            "adopted_transpose": tr,
            "var_ols_auroc_off_adopted": _mean(adopted, "auroc_off"),
            "var_ols_auroc_off_reversed": _mean(other, "auroc_off"),
        }
    c.put("orientation", "data", res)
    print_orientation(res)


def print_orientation(res: Dict[str, Any]) -> None:
    ev = res.get("_evidence", {})
    print("\n=== ORIENTATION: convention fixed by generator source + an "
          "estimator-free data check ===")
    r = ev.get("simple/rossler_vs_analytic_jacobian")
    if r:
        print(f"  simple : shipped Rossler matrix == analytic binary Jacobian "
              f"[effect,cause]?  {r['match']}   -> score AS-IS")
    for subset in ("coupled", "climate", "simple"):
        t = ev.get(f"{subset}/root_nodes_test")
        if t:
            print(f"  {subset:7s}: root_nodes agrees with all-zero COLUMN "
                  f"({t['agree_cause_row_[cause,effect]']}/{t['n']}) vs all-zero "
                  f"ROW ({t['agree_effect_row_[effect,cause]']}/{t['n']})"
                  f"  -> {t['verdict']}")
    po = ev.get("published_wrapper_orientations")
    if po:
        print("\n  What the SHIPPED baselines emit, and where that lines up "
              "with the truth (upstream never transposes):")
        print(f"    {'wrapper':11s} {'emits':20s} aligned on")
        for m, v in po.items():
            print(f"    {m:11s} {v['emits']:20s} "
                  f"{', '.join(v['aligned_on']) or 'n/a (undirected)'}"
                  + (f"   | misaligned on {', '.join(v['misaligned_on'])}"
                     if v['misaligned_on'] else ""))
    print("\n  Directional accuracy of a linear VAR UNDER THE ADOPTED "
          "convention (off-diagonal AUROC).")
    print("  'reversed' is the same estimator scored against the flipped "
          "truth; it is a property of the ESTIMATOR, not of the file.")
    print(f"  {'tier':52s} {'adopted':>8s} {'reversed':>9s} {'n_asym':>6s} "
          f"{'truth stored as':>16s}")
    for k, v in res.items():
        if k == "_evidence":
            continue
        print(f"  {k[:52]:52s} {v['var_ols_auroc_off_adopted']:8.4f} "
              f"{v['var_ols_auroc_off_reversed']:9.4f} "
              f"{v['n_asymmetric_truths']:6d} "
              f"{'[cause,effect]' if v['adopted_transpose'] else '[effect,cause]':>16s}")


# --------------------------------------------------------------------------- #
# CELL: calibration -- reproduce published cells, then measure every switch
# --------------------------------------------------------------------------- #
def _calib_run(subset: str, setting: str, method: str, *, max_lag: int,
               binarise: bool, transpose: bool, upstream_std: bool,
               lag_slice: bool, limit: Optional[int] = None,
               **kw) -> Dict[str, float]:
    """One calibration variant over a whole setting, upstream's loop shape."""
    rows = []
    for p in system_paths(subset, setting)[:limit]:
        got = load_system(p, subset, upstream_std=upstream_std)
        if got is None:
            continue
        cube, A = got
        A_file = A.T if TRUTH_IS_CAUSE_ROW[subset] else A
        per = []
        for x in cube:
            r = run_method(method, x, max_lag, TIMEOUT_S, **kw)
            if not r.ok:
                S = np.zeros_like(A_file)
            elif lag_slice:
                S = np.abs(r.A_lag[max_lag - 1])           # upstream: one slice
            else:
                S = cb.summary_graph(r, how="max", include_contemp=True)
            if binarise:
                S = (np.abs(S) > 0).astype(float)
            if transpose:
                S = S.T
            per.append(score_graph(A_file, S))
        if per:
            rows.append({k: _mean(per, k) for k in per[0]})
    return {"auroc": _mean(rows, "auroc"), "auprc": _mean(rows, "auprc"),
            "auroc_off": _mean(rows, "auroc_off"),
            "auprc_off": _mean(rows, "auprc_off"),
            "n_systems": len(rows)}


def cell_calibration(c: Cache) -> None:
    subset, setting = "simple", "noise=0.00_confounder=False"
    pub = c.d.get("published", {}).get("data", {}).get(subset, {}).get(
        "per_setting", {}).get(setting, {})

    variants = [
        # label, method, kwargs                              -> reproduces?
        ("dynotears / upstream wrapper (binarised, transposed)",
         "dynotears", dict(binarise=True, transpose=True, lag_slice=True,
                           lambda_w=0.2, lambda_a=0.1), "dynotears"),
        ("dynotears / orientation corrected",
         "dynotears", dict(binarise=True, transpose=False, lag_slice=True,
                           lambda_w=0.2, lambda_a=0.1), None),
        ("varlingam / upstream wrapper (binarised, prune off)",
         "varlingam", dict(binarise=True, transpose=False, lag_slice=True,
                           prune=False, criterion=None), "varlingam"),
        ("varlingam / lingam library defaults (prune on, BIC lag order)",
         "varlingam", dict(binarise=True, transpose=False, lag_slice=True),
         "varlingam"),
        ("varlingam / continuous scores, prune on",
         "varlingam", dict(binarise=False, transpose=False, lag_slice=True,
                           prune=True, criterion=None), None),
        ("var_ols / |coef|, lag-1 slice",
         "var_ols", dict(binarise=False, transpose=False, lag_slice=True,
                         score="abscoef"), None),
    ]
    out: Dict[str, Any] = {"setting": f"{subset}/{setting}",
                           "max_lag": CALIB_LAG, "variants": []}
    for label, method, kw, pubkey in variants:
        key = f"{label}"
        if c.has("calibration", key):
            r = c.cell("calibration")[key]
        else:
            bk = {k: v for k, v in kw.items()
                  if k not in ("binarise", "transpose", "lag_slice")}
            t0 = time.time()
            r = _calib_run(subset, setting, method, max_lag=CALIB_LAG,
                           binarise=kw["binarise"], transpose=kw["transpose"],
                           upstream_std=True, lag_slice=kw["lag_slice"], **bk)
            r["seconds"] = time.time() - t0
            c.put("calibration", key, r)
        p = pub.get(pubkey) if pubkey else None
        row = dict(label=label, method=method, **r)
        if p:
            row["published_auroc"] = p["Individual_AUROC"]
            row["published_auprc"] = p["Individual_AUPRC"]
            row["d_auroc"] = r["auroc"] - p["Individual_AUROC"]
            row["d_auprc"] = r["auprc"] - p["Individual_AUPRC"]
        out["variants"].append(row)

    # --- protocol sensitivities, all on the same cell ---------------------- #
    # NOTE: most wrappers z-score internally by default (_bl_var_ols has
    # standardize=True), which would make any OUTER standardiser a no-op and
    # the sensitivity vacuous.  The wrapper's own scaler is therefore switched
    # OFF here so the outer choice is the only one acting.  (dagma_linear and
    # dynotears already default to standardize=False -- see
    # docs/causal_baselines.md Section 3 on varsortability.)
    sens = {}
    for skey, kw in (
        ("upstream joint z-score (reference)", dict(upstream_std=True)),
        ("standardize_on_observed, per realisation", dict(upstream_std=False)),
        ("no standardisation at all", dict(upstream_std=None)),
    ):
        k = f"sens/std/{skey}"
        if not c.has("calibration", k):
            c.put("calibration", k, _calib_run(
                subset, setting, "var_ols", max_lag=CALIB_LAG, binarise=False,
                transpose=False, lag_slice=True, score="abscoef",
                standardize=False, **kw))
        sens[skey] = c.cell("calibration")[k]
    for lk, K in (("lag budget K=1", 1), ("lag budget K=3 (h2h budget)", 3)):
        k = f"sens/lag/{lk}"
        if not c.has("calibration", k):
            c.put("calibration", k, _calib_run(
                subset, setting, "var_ols", max_lag=K, binarise=False,
                transpose=False, upstream_std=False, lag_slice=False,
                score="abscoef"))
        sens[lk] = c.cell("calibration")[k]
    out["sensitivity"] = sens

    # --- cross-check our dependency-free AUC against sklearn's ------------- #
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    s = rng.random(400) + 0.4 * y
    a1 = cm.edge_auc(y, s)
    a2 = _auc_ap(y, s)[0]
    out["auc_crosscheck"] = {"causal_metrics.edge_auc": a1,
                             "sklearn.roc_auc_score": a2,
                             "abs_diff": abs(a1 - a2)}
    c.put("calibration", "summary", out)
    print_calibration(out)


def print_calibration(out: Dict[str, Any]) -> None:
    print(f"\n=== CALIBRATION on {out['setting']}  (tau_max={out['max_lag']}, "
          f"upstream z-score, self-loops scored) ===")
    print(f"  {'variant':54s} {'AUROC':>7s} {'AUPRC':>7s} "
          f"{'pub AUROC':>9s} {'pub AUPRC':>9s} {'dAUROC':>7s}")
    for r in out["variants"]:
        pa = f"{r['published_auroc']:9.4f}" if "published_auroc" in r else "        -"
        pp = f"{r['published_auprc']:9.4f}" if "published_auprc" in r else "        -"
        dd = f"{r['d_auroc']:+7.4f}" if "d_auroc" in r else "      -"
        print(f"  {r['label'][:54]:54s} {r['auroc']:7.4f} {r['auprc']:7.4f} "
              f"{pa} {pp} {dd}")
    print("\n  protocol sensitivity (var_ols |coef|):")
    for k, v in out["sensitivity"].items():
        print(f"    {k:44s} AUROC={v['auroc']:.4f}  AUPRC={v['auprc']:.4f}  "
              f"off-diag AUROC={v['auroc_off']:.4f}")
    cc = out["auc_crosscheck"]
    print(f"  AUC cross-check |causal_metrics - sklearn| = {cc['abs_diff']:.2e}")


# --------------------------------------------------------------------------- #
# CELL: chance -- measure the null instead of assuming it
# --------------------------------------------------------------------------- #
def cell_chance(c: Cache) -> None:
    res: Dict[str, Any] = {}
    for label, subset, setting in tiers():
        if c.has("chance", label):
            res[label] = c.cell("chance")[label]
            continue
        rng = np.random.default_rng(abs(hash(label)) % (2 ** 31))
        rand, const, base, base_off, Ps = [], [], [], [], []
        per_system: Dict[str, Dict[str, float]] = {}
        for p in system_paths(subset, setting):
            got = load_system(p, subset)
            if got is None:
                continue
            _, A = got
            Ps.append(A.shape[0])
            base.append(float((A != 0).mean()))
            off = ~np.eye(A.shape[0], dtype=bool)
            base_off.append(float((A != 0)[off].mean()))
            per_r, per_c = [], []
            for _ in range(CHANCE_SEEDS):
                per_r.append(score_graph(A, rng.random(A.shape)))
            per_c.append(score_graph(A, np.ones(A.shape)))
            rr = {k: _mean(per_r, k) for k in per_r[0]}
            rand.append(rr)
            const.append({k: _mean(per_c, k) for k in per_c[0]})
            per_system[os.path.basename(p)[:-3]] = {
                k: rr[k] for k in ("auroc", "auprc", "auroc_off", "auprc_off")}
        if not rand:
            continue
        entry = {
            "n_systems": len(rand), "P": float(np.mean(Ps)),
            "base_rate": float(np.mean(base)),
            "base_rate_off": float(np.mean(base_off)),
            "random_auroc": _mean(rand, "auroc"),
            "random_auprc": _mean(rand, "auprc"),
            "random_auroc_off": _mean(rand, "auroc_off"),
            "random_auprc_off": _mean(rand, "auprc_off"),
            "random_auroc_sd": float(np.std([r["auroc"] for r in rand])),
            "random_auroc_off_sd": float(np.std([r["auroc_off"] for r in rand])),
            "constant_auroc": _mean(const, "auroc"),
            "constant_auprc": _mean(const, "auprc"),
            "seeds": CHANCE_SEEDS,
            "per_system": per_system,
        }
        c.put("chance", label, entry)
        res[label] = entry
    print_chance(res)


def print_chance(res: Dict[str, Any]) -> None:
    print("\n=== CHANCE CONTROL (uniform random scores, "
          f"{CHANCE_SEEDS} seeds/system) ===")
    print(f"  {'tier':52s} {'P':>4s} {'base':>6s} {'randAUROC':>10s} "
          f"{'randAUPRC':>10s} {'offAUPRC':>9s}")
    for k, v in res.items():
        print(f"  {k[:52]:52s} {v['P']:4.1f} {v['base_rate']:6.3f} "
              f"{v['random_auroc']:10.4f} {v['random_auprc']:10.4f} "
              f"{v['random_auprc_off']:9.4f}")


# --------------------------------------------------------------------------- #
# CELL: h2h -- the batch board.  One worker per system, checkpoint per system.
# --------------------------------------------------------------------------- #
def _h2h_one_system(job) -> Tuple[str, Dict[str, Any]]:
    key, path, subset, methods, max_lag, timeout_s, r_cap, prefix = job
    got = load_system(path, subset)
    if got is None:
        return key, {"skipped": "degenerate truth (all-ones / all-zeros)"}
    cube, A = got
    if r_cap:
        cube = cube[:r_cap]
    out: Dict[str, Any] = {"P": int(A.shape[0]), "R": int(cube.shape[0]),
                           "T": int(cube.shape[1]),
                           "base_rate": float((A != 0).mean())}
    for m in methods:
        per, per_lo, n_ok, n_to, errs = [], [], 0, 0, []
        lag_capable = m not in cb.CONTEMP_ONLY
        for x in cube:
            xx = x[:max(int(round(prefix * x.shape[0])), 4 * max_lag + 8)] \
                if prefix < 1.0 else x
            r = run_method(m, xx, max_lag, timeout_s)
            if r.ok:
                n_ok += 1
                S = cb.summary_graph(r, how="max", include_contemp=True)
                Slo = cb.summary_graph(r, how="max", include_contemp=False)
                H = support_of(r)
            else:
                errs.append(r.err[:120])
                if "TIMEOUT" in r.err.upper():
                    n_to += 1
                S = Slo = np.zeros_like(A)
                H = np.zeros_like(A, dtype=bool)
            per.append(score_graph(A, S, H))
            if lag_capable:
                per_lo.append(score_graph(A, Slo))
        out[m] = {k: _mean(per, k) for k in per[0]}
        if per_lo:
            out[m].update({f"lo_{k}": _mean(per_lo, k)
                           for k in ("auroc", "auprc", "auroc_off", "auprc_off")})
        out[m].update(n_ok=n_ok, n_fail=len(per) - n_ok, n_timeout=n_to,
                      err=errs[0] if errs else "")
    return key, out


def _h2h_jobs(methods, prefix, r_cap, tag):
    jobs = []
    for label, subset, setting in tiers():
        paths = system_paths(subset, setting)
        if subset == "coupled":
            paths = paths[:COUPLED_SYSTEMS_CAP]      # deterministic, sorted
        for p in paths:
            key = f"{tag}|{label}|{os.path.basename(p)[:-3]}"
            jobs.append((key, p, subset, methods, MAX_LAG, TIMEOUT_S,
                         r_cap, prefix))
    return jobs


def _run_jobs(c: Cache, cell: str, jobs, workers: int) -> None:
    todo = [j for j in jobs if not c.has(cell, j[0])]
    print(f"[{cell}] {len(jobs)} systems, {len(todo)} to run, "
          f"{workers} workers")
    if not todo:
        return
    t0, done = time.time(), 0
    if workers <= 1:
        for j in todo:
            k, v = _h2h_one_system(j)
            c.put(cell, k, v)
            done += 1
            _progress(cell, done, len(todo), t0)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_h2h_one_system, j): j[0] for j in todo}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                kk, v = fut.result()
            except Exception as e:
                kk, v = k, {"error": f"{type(e).__name__}: {e}"}
            c.put(cell, kk, v, flush=(done % 10 == 0))
            done += 1
            _progress(cell, done, len(todo), t0)
    c.flush()


def _progress(cell, done, total, t0):
    if done % 10 == 0 or done == total:
        el = time.time() - t0
        eta = el / max(done, 1) * (total - done)
        print(f"  [{cell}] {done}/{total}  {el / 60:.1f} min elapsed, "
              f"{eta / 60:.1f} min left", flush=True)


def cell_h2h(c: Cache) -> None:
    """The 12 external baselines ONLY.  CAFE-CD is cell 'cafecd'."""
    _run_jobs(c, "h2h", _h2h_jobs(list(METHODS), 1.0, None, "batch"), N_WORKERS)
    print_h2h(aggregate_h2h(c, "h2h", ("batch", "cafecd")))


def cell_cafecd(c: Cache) -> None:
    """CAFE-CD ONLY, under cache tag 'cafecd'.

    Deliberately a separate cell with a separate cache tag and its own tier
    keys, so ``--force cafecd`` re-runs our row alone against an unchanged
    baseline board.  ``bench/bench_dynamics_cache.json`` never mixes the two.
    """
    if not cafe_cd_available():
        print(f"[cafecd] CAFE-CD NOT IMPORTABLE: {_CAFE_ERR}")
        print("[cafecd] slot left empty; every table renders it as ABSENT. "
              "Rerun `--cells cafecd` once src/cafe/discovery.py lands.")
        c.put("cafecd", "_status", {"available": False, "err": _CAFE_ERR,
                                    "ts": time.time()})
        return
    c.put("cafecd", "_status", {"available": True, "err": "", "ts": time.time(),
                                "provisional": CAFE_CD_PROVISIONAL})
    _run_jobs(c, "h2h", _h2h_jobs([CAFE_CD], 1.0, None, "cafecd"), N_WORKERS)
    if CAFE_CD_PROVISIONAL:
        print(f"\n[cafecd] {CAFE_CD_PROVISIONAL_NOTE}")
    print_h2h(aggregate_h2h(c, "h2h", ("batch", "cafecd")))


def aggregate_h2h(c: Cache, cell: str, tags) -> Dict[str, Any]:
    """Per-tier mean over systems of the per-system (mean over realisations).

    ``tags`` may name several cache tags (e.g. the baseline board and the
    separately-cached CAFE-CD row); entries for the same (tier, system) are
    merged method-wise.
    """
    if isinstance(tags, str):
        tags = (tags,)
    buckets: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    meta: Dict[str, Dict[str, float]] = {}
    live = {t[0] for t in tiers()}
    for key, v in c.cell(cell).items():
        parts = key.split("|")
        if len(parts) != 3 or parts[0] not in tags or not isinstance(v, dict):
            continue
        if "skipped" in v or "error" in v:
            continue
        label = parts[1]
        if label not in live:            # stale entry from an earlier design
            continue
        b = buckets.setdefault(label, {})
        mm = meta.setdefault(label, {"n_systems": 0, "P": 0.0, "base": 0.0})
        if parts[0] == tags[0]:          # count each system once
            mm["n_systems"] += 1
            mm["P"] += v.get("P", 0)
            mm["base"] += v.get("base_rate", 0.0)
        for m, r in v.items():
            if isinstance(r, dict):
                b.setdefault(m, []).append(r)
    out: Dict[str, Any] = {}
    for label, b in buckets.items():
        mm = meta[label]
        n = max(mm["n_systems"], 1)
        rows = {}
        for m, rs in b.items():
            rows[m] = {k: _mean(rs, k) for k in
                       ("auroc", "auprc", "auroc_off", "auprc_off",
                        "f1", "f1_off", "shd", "density_hat",
                        "lo_auroc", "lo_auprc", "lo_auroc_off", "lo_auprc_off")}
            rows[m]["n_ok"] = float(np.sum([r.get("n_ok", 0) for r in rs]))
            rows[m]["n_fail"] = float(np.sum([r.get("n_fail", 0) for r in rs]))
            rows[m]["n_timeout"] = float(np.sum([r.get("n_timeout", 0) for r in rs]))
            errs = [r.get("err", "") for r in rs if r.get("err")]
            rows[m]["err"] = errs[0] if errs else ""
        out[label] = {"n_systems": int(n), "P": mm["P"] / n,
                      "base_rate": mm["base"] / n, "methods": rows}
    return out


def group_aggregate(agg: Dict[str, Any], chance: Dict[str, Any]
                    ) -> Dict[str, Any]:
    """Collapse per-setting tiers into simple / climate / coupled groups."""
    grp: Dict[str, Any] = {}
    for label, blk in agg.items():
        g = tier_group(label)
        e = grp.setdefault(g, {"labels": [], "n_systems": 0, "P": [],
                               "base": [], "methods": {}, "chance": []})
        e["labels"].append(label)
        e["n_systems"] += blk["n_systems"]
        e["P"].append(blk["P"])
        e["base"].append(blk["base_rate"])
        if label in chance:
            e["chance"].append(chance[label])
        for m, r in blk["methods"].items():
            e["methods"].setdefault(m, []).append(r)
    out = {}
    for g, e in grp.items():
        rows = {}
        for m, rs in e["methods"].items():
            rows[m] = {k: _mean(rs, k) for k in
                       ("auroc", "auprc", "auroc_off", "auprc_off",
                        "f1", "f1_off", "shd",
                        "lo_auroc", "lo_auprc", "lo_auroc_off", "lo_auprc_off")}
            rows[m]["n_fail"] = float(np.sum([r["n_fail"] for r in rs]))
            rows[m]["n_timeout"] = float(np.sum([r["n_timeout"] for r in rs]))
        ch = e["chance"]
        out[g] = {"n_settings": len(e["labels"]), "n_systems": e["n_systems"],
                  "P": float(np.mean(e["P"])),
                  "base_rate": float(np.mean(e["base"])),
                  "chance_auprc": float(np.mean([x["random_auprc"] for x in ch]))
                  if ch else float("nan"),
                  "chance_auprc_off": float(np.mean(
                      [x["random_auprc_off"] for x in ch])) if ch else float("nan"),
                  "chance_auroc": float(np.mean([x["random_auroc"] for x in ch]))
                  if ch else float("nan"),
                  "chance_auroc_sd": float(np.mean(
                      [x["random_auroc_sd"] for x in ch])) if ch else float("nan"),
                  "methods": rows}
    return out


def print_h2h(agg: Dict[str, Any], title: str = "BATCH BOARD") -> None:
    print(f"\n=== {title} (per-tier; self-loops SCORED = benchmark "
          f"convention; '_off' excludes them) ===")
    for label in sorted(agg):
        blk = agg[label]
        print(f"\n{label}   n_systems={blk['n_systems']}  P={blk['P']:.1f}  "
              f"base_rate={blk['base_rate']:.3f}")
        print(f"  {'method':17s} {'AUPRC':>7s} {'AUROC':>7s} "
              f"{'offAUPRC':>9s} {'offAUROC':>9s} | {'lagAUPRC':>9s} "
              f"{'lagAUROC':>9s} | {'F1':>6s} {'SHD':>6s} {'fail':>5s}")
        for m, r in sorted(blk["methods"].items(),
                           key=lambda kv: -(kv[1]["auprc_off"]
                                            if kv[1]["auprc_off"] == kv[1]["auprc_off"]
                                            else -9)):
            print(f"  {m:17s} {r['auprc']:7.4f} {r['auroc']:7.4f} "
                  f"{r['auprc_off']:9.4f} {r['auroc_off']:9.4f} | "
                  f"{_fmt(r.get('lo_auprc_off'),4,'      n/a'):>9s} "
                  f"{_fmt(r.get('lo_auroc_off'),4,'      n/a'):>9s} | "
                  f"{r['f1']:6.3f} {r['shd']:6.2f} {r['n_fail']:5.0f}")


# --------------------------------------------------------------------------- #
# CELL: runtime -- sequential, uncontaminated wall-clock
# --------------------------------------------------------------------------- #
def cell_runtime(c: Cache) -> None:
    methods = list(METHODS) + ([CAFE_CD] if cafe_cd_available() else [])
    # One representative tier per (group, P) so the table spans the P range
    # that actually occurs (P=3 simple, P=4/10 climate, P=5/10 coupled) without
    # paying for every setting: runtime is a function of (P, T, K), not of the
    # noise level or the coupling family.
    want = []
    for grp_ in ("simple", "climate", "coupled"):
        cand = [t for t in tiers() if tier_group(t[0]) == grp_]
        want += cand[:1] if grp_ == "simple" else cand[:2]
    want_labels = {t[0] for t in want}
    res: Dict[str, Any] = {}
    for label, subset, setting in tiers():
        if label not in want_labels:
            continue
        key = f"runtime|{label}"
        if c.has("runtime", key):
            res[label] = c.cell("runtime")[key]
            continue
        paths = [p for p in system_paths(subset, setting)][:5]
        times: Dict[str, List[float]] = {m: [] for m in methods}
        P = 0
        for p in paths:
            got = load_system(p, subset)
            if got is None:
                continue
            cube, A = got
            P = A.shape[0]
            for m in methods:
                t0 = time.time()
                run_method(m, cube[0], MAX_LAG, TIMEOUT_S)
                times[m].append(time.time() - t0)
        if not P:
            continue
        entry = {"P": P, "T": 1000, "n_fits": len(paths),
                 "sec": {m: float(np.mean(v)) for m, v in times.items() if v}}
        c.put("runtime", key, entry)
        res[label] = entry
    print_runtime(res)


def print_runtime(res: Dict[str, Any]) -> None:
    print("\n=== CPU WALL-CLOCK, single-threaded, seconds per fit "
          f"(T=1000, K={MAX_LAG}) ===")
    labels = list(res)
    if not labels:
        return
    ms = sorted({m for v in res.values() for m in v["sec"]})
    hdr = "  ".join(f"{l.split('/')[0][:8]}/P={res[l]['P']}" for l in labels)
    print(f"  {'method':17s} {hdr}")
    for m in ms:
        cells = "  ".join(f"{res[l]['sec'].get(m, float('nan')):11.3f}"
                          for l in labels)
        print(f"  {m:17s} {cells}")


# --------------------------------------------------------------------------- #
# CELL: pit -- the point-in-time board (never merged with the batch board)
# --------------------------------------------------------------------------- #
def cell_pit(c: Cache) -> None:
    """Prefix board for the batch rivals.  CAFE-CD's prefix rows live under
    tag ``pitc<f>`` (cell ``cafecd``) so they too rerun independently."""
    methods = list(PIT_METHODS)
    if not cafe_cd_available():
        print(f"[pit] CAFE-CD slot EMPTY: {_CAFE_ERR}  "
              "(prefix board still built for the batch rivals)")
    for frac in PIT_FRACS:
        tag = f"pit{int(frac * 100)}"
        jobs = [j for j in _h2h_jobs(methods, frac, 4, tag)
                if tier_group(j[0].split("|")[1]) in ("simple", "coupled")]
        # keep the point-in-time board small: 8 coupled settings, all simple
        keep, seen = [], {}
        for j in jobs:
            lab = j[0].split("|")[1]
            g = tier_group(lab)
            if g == "coupled":
                seen.setdefault(lab, len(seen))
                if seen[lab] >= 8:
                    continue
            keep.append(j)
        _run_jobs(c, "pit", keep, N_WORKERS)
    if cafe_cd_available():
        for frac in PIT_FRACS:
            tag = f"pit{int(frac * 100)}"
            keep = [j for j in _h2h_jobs([CAFE_CD], frac, 4, "c" + tag)
                    if tier_group(j[0].split("|")[1]) in ("simple", "coupled")]
            _run_jobs(c, "pit", keep, N_WORKERS)
    out = {f"{int(f * 100)}%": group_aggregate(
        aggregate_h2h(c, "pit", (f"pit{int(f * 100)}", f"cpit{int(f * 100)}")),
        c.cell("chance"))
        for f in PIT_FRACS}
    c.put("pit", "summary", out)
    print_pit(out)


def print_pit(out: Dict[str, Any]) -> None:
    print("\n=== POINT-IN-TIME BOARD -- graphs scored after seeing only the "
          "first f of the series ===")
    print("    Batch rivals are REFIT on each prefix; CAFE-CD reads its own "
          "graph at that t from one causal pass.")
    print("    This board is NOT comparable to the batch board and is never "
          "ranked against it.")
    for frac, agg in out.items():
        for g, blk in sorted(agg.items()):
            print(f"\n  f={frac}  tier={g}  n_systems={blk['n_systems']}  "
                  f"chance AUPRC={blk['chance_auprc']:.3f}")
            print(f"    {'method':17s} {'AUPRC':>7s} {'offAUPRC':>9s} "
                  f"{'AUROC':>7s} {'offAUROC':>9s}")
            for m, r in sorted(blk["methods"].items(),
                               key=lambda kv: -(kv[1]['auprc_off']
                                                if kv[1]['auprc_off'] == kv[1]['auprc_off']
                                                else -9)):
                print(f"    {m:17s} {r['auprc']:7.4f} {r['auprc_off']:9.4f} "
                      f"{r['auroc']:7.4f} {r['auroc_off']:9.4f}")


# --------------------------------------------------------------------------- #
# Learnability verdict
# --------------------------------------------------------------------------- #
def paired_vs_chance(c: Cache, metric: str = "auprc_off") -> Dict[str, Any]:
    """Per-tier-group PAIRED comparison of every method against the random
    control measured ON THE SAME SYSTEM.

    Pairing matters here: base rate (and therefore attainable AUPRC) varies by
    an order of magnitude across systems -- 0.15 on a 10-node coupled graph,
    0.69 on a 3-node simple one -- so an unpaired mean lift mixes that variation
    into the signal.  Reported per method: the mean paired lift, its standard
    error over systems, a paired t-statistic, and the fraction of systems on
    which the method beats its own control (a distribution-free companion).
    """
    ctrl: Dict[str, Dict[str, float]] = {}
    for label, blk in c.cell("chance").items():
        for sysname, v in (blk.get("per_system") or {}).items():
            ctrl[f"{label}|{sysname}"] = v
    live = {t[0] for t in tiers()}
    pairs: Dict[str, Dict[str, List[float]]] = {}
    for key, v in c.cell("h2h").items():
        parts = key.split("|")
        if len(parts) != 3 or parts[0] not in ("batch", "cafecd"):
            continue
        label, sysname = parts[1], parts[2]
        if label not in live or not isinstance(v, dict) or "skipped" in v:
            continue
        cv = ctrl.get(f"{label}|{sysname}")
        if not cv or metric not in cv or cv[metric] != cv[metric]:
            continue
        g = tier_group(label)
        for m, r in v.items():
            if not isinstance(r, dict) or metric not in r:
                continue
            if r[metric] != r[metric]:
                continue
            pairs.setdefault(g, {}).setdefault(m, []).append(r[metric] - cv[metric])
    out: Dict[str, Any] = {}
    for g, ms in pairs.items():
        rows = {}
        for m, d in ms.items():
            a = np.asarray(d, float)
            n = a.size
            sd = float(a.std(ddof=1)) if n > 1 else float("nan")
            se = sd / np.sqrt(n) if n > 1 and sd > 0 else float("nan")
            rows[m] = {"n_systems": int(n), "mean_lift": float(a.mean()),
                       "se": float(se) if se == se else float("nan"),
                       "t": float(a.mean() / se) if se == se and se > 0
                            else float("nan"),
                       "win_rate": float((a > 0).mean())}
        out[g] = rows
    return out


def print_paired(pv: Dict[str, Any], metric: str = "auprc_off") -> None:
    print(f"\n=== PAIRED vs PER-SYSTEM RANDOM CONTROL ({metric}) ===")
    print("    lift = method minus a uniform-random score on the SAME system; "
          "|t| > 2 is the bar.")
    for g in sorted(pv):
        print(f"\n  {g}")
        print(f"    {'method':17s} {'lift':>8s} {'se':>7s} {'t':>7s} "
              f"{'win%':>6s} {'n':>4s}")
        for m, r in sorted(pv[g].items(), key=lambda kv: -kv[1]["mean_lift"]):
            star = " *" if (r["t"] == r["t"] and abs(r["t"]) > 2) else "  "
            tag = "  [PROVISIONAL]" if (m == CAFE_CD and CAFE_CD_PROVISIONAL) else ""
            print(f"    {m:17s} {r['mean_lift']:+8.4f} {r['se']:7.4f} "
                  f"{r['t']:7.2f}{star} {100 * r['win_rate']:5.1f}% "
                  f"{r['n_systems']:4d}{tag}")


def coupled_sweep(c: Cache, agg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The coupled one-factor-at-a-time design, read back as a sweep table.

    Each row is one setting, labelled by the single axis on which it differs
    from the reference cell, so the effect of noise / coupling family / number
    of subsystems (the 'across N' axis) / hidden confounder / time lag is
    readable without re-running anything.
    """
    ref = _coupled_name(_COUPLED_REF)
    rows = []
    for label, blk in agg.items():
        if tier_group(label) != "coupled":
            continue
        setting = label.split("/", 1)[1]
        axis = "reference"
        if setting != ref:
            cur = dict(kv.split("=", 1) for kv in setting.split("_")
                       if "=" in kv)
            diff = [k for k, v in cur.items() if _COUPLED_REF.get(k) != v]
            axis = ", ".join(f"{k}={cur[k]}" for k in diff) or "reference"
        ms = {m: r for m, r in blk["methods"].items()
              if r["auroc_off"] == r["auroc_off"]
              and not (m == CAFE_CD and CAFE_CD_PROVISIONAL)}
        if not ms:
            continue
        best = max(ms, key=lambda m: ms[m]["auroc_off"])
        rows.append({"axis": axis, "setting": setting, "P": blk["P"],
                     "n_systems": blk["n_systems"],
                     "base_rate": blk["base_rate"], "best_method": best,
                     "best_auroc_off": ms[best]["auroc_off"],
                     "best_auprc_off": ms[best]["auprc_off"],
                     "var_ols_auroc_off": ms.get("var_ols", {}).get(
                         "auroc_off", float("nan")),
                     "cafe_cd_auroc_off": blk["methods"].get(CAFE_CD, {}).get(
                         "auroc_off", float("nan"))})
    order = {"reference": 0}
    return sorted(rows, key=lambda r: (order.get(r["axis"], 1), r["axis"]))


def print_coupled_sweep(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\n=== COUPLED SWEEP (one factor at a time off the reference cell "
          "coupling=linear, noise=1.00, systems=5, confounder=False, "
          "timelag=1) ===")
    print(f"  {'axis changed':26s} {'P':>4s} {'sys':>4s} {'base':>6s} "
          f"{'best method':17s} {'AUROCoff':>8s} {'AUPRCoff':>8s} "
          f"{'var_ols':>8s}"
          + (f" {'cafe_cd*':>8s}" if any(r["cafe_cd_auroc_off"] ==
                                        r["cafe_cd_auroc_off"] for r in rows)
             else ""))
    for r in rows:
        line = (f"  {r['axis'][:26]:26s} {r['P']:4.1f} {r['n_systems']:4d} "
                f"{r['base_rate']:6.3f} {r['best_method']:17s} "
                f"{r['best_auroc_off']:8.4f} {r['best_auprc_off']:8.4f} "
                f"{r['var_ols_auroc_off']:8.4f}")
        if r["cafe_cd_auroc_off"] == r["cafe_cd_auroc_off"]:
            line += f" {r['cafe_cd_auroc_off']:8.4f}"
        print(line)


def learnability(group: Dict[str, Any], c: Optional["Cache"] = None
                 ) -> Dict[str, Any]:
    """Is any method separated from the MEASURED chance control on this tier?

    Metric choice is not free here, and the benchmark's own advice needs a
    caveat.  CausalDynamics recommends AUPRC over AUROC "for sparse graphs" --
    correct for ``coupled`` (off-diagonal base rate ~0.2-0.3), but the
    ``simple`` tier's ground truth is a 3x3 Jacobian pattern that is ~0.80
    dense off the diagonal.  There AUPRC starts at 0.80 by construction and has
    almost no headroom, so a method can be plainly informative while its AUPRC
    lift rounds to nothing.  We therefore:

      * judge on off-diagonal **AUROC**, which is base-rate free and so
        comparable across tiers with base rates from 0.15 to 0.80;
      * require the separation to survive PAIRING against a random score drawn
        on the same system (paired t over systems), not merely to exceed 0.5;
      * report the AUPRC lift alongside, and print the headroom
        (1 - off-diagonal base rate) so a small AUPRC lift on a dense tier is
        readable as a ceiling and not as a failure.

    Criterion, stated so it can be argued with: LEARNABLE iff the best method's
    paired off-diagonal AUROC lift exceeds 0.05 with a paired t above 3.
    Off-diagonal throughout, because the diagonal is trivially predicted and
    would let a tier look learnable on self-loops alone.
    """
    pv_roc = paired_vs_chance(c, "auroc_off") if c is not None else {}
    pv_prc = paired_vs_chance(c, "auprc_off") if c is not None else {}
    out = {}
    for g, blk in group.items():
        ms = {m: r for m, r in blk["methods"].items()
              if r["auroc_off"] == r["auroc_off"]
              and not (m == CAFE_CD and CAFE_CD_PROVISIONAL)}
        if not ms:
            continue
        proc = {m: v for m, v in pv_roc.get(g, {}).items()
                if not (m == CAFE_CD and CAFE_CD_PROVISIONAL)}
        if proc:
            best_m = max(proc, key=lambda m: proc[m]["mean_lift"])
            lift = proc[best_m]["mean_lift"]
            tstat = proc[best_m]["t"]
            win = proc[best_m]["win_rate"]
            n = proc[best_m]["n_systems"]
        else:                                   # no paired control available
            best_m = max(ms, key=lambda m: ms[m]["auroc_off"])
            lift = ms[best_m]["auroc_off"] - blk["chance_auroc"]
            tstat = win = float("nan")
            n = blk["n_systems"]
        out[g] = {
            "best_method": best_m,
            "best_auroc_off": ms.get(best_m, {}).get("auroc_off", float("nan")),
            "chance_auroc": blk["chance_auroc"],
            "auroc_off_lift": lift, "auroc_off_t": tstat,
            "auroc_off_win_rate": win, "n_systems": n,
            "best_auprc_off": ms.get(best_m, {}).get("auprc_off", float("nan")),
            "chance_auprc_off": blk["chance_auprc_off"],
            "auprc_off_lift": pv_prc.get(g, {}).get(best_m, {}).get(
                "mean_lift", float("nan")),
            "auprc_headroom": 1.0 - blk["chance_auprc_off"],
            "learnable": bool(lift > 0.05 and tstat == tstat and tstat > 3),
        }
    return out


def print_learnability(lz: Dict[str, Any]) -> None:
    print("\n=== LEARNABILITY VERDICT ===")
    print("  Criterion: best method's PAIRED off-diagonal AUROC lift over a "
          "random score on the")
    print("  same system > 0.05 with paired t > 3. AUROC (not AUPRC) because "
          "off-diagonal base")
    print("  rates span 0.15-0.80 here; AUPRC headroom is printed so a "
          "saturated tier is visible.")
    if CAFE_CD_PROVISIONAL:
        print("  [CAFE-CD is excluded from 'best method' while its scores() "
              "defect stands -- see CAFE_CD_PROVISIONAL_NOTE.]")
    print(f"\n  {'tier':9s} {'verdict':10s} {'best':17s} {'AUROCoff':>8s} "
          f"{'lift':>7s} {'t':>6s} {'win%':>5s} {'dAUPRC':>7s} {'PRChead':>7s}")
    for g, v in sorted(lz.items()):
        print(f"  {g:9s} {'LEARNABLE' if v['learnable'] else 'AT CHANCE':10s} "
              f"{v['best_method']:17s} {v['best_auroc_off']:8.4f} "
              f"{v['auroc_off_lift']:+7.4f} {v['auroc_off_t']:6.2f} "
              f"{100 * v['auroc_off_win_rate']:4.0f}% "
              f"{v['auprc_off_lift']:+7.4f} {v['auprc_headroom']:7.3f}")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(x, n=3, dash="--"):
    return dash if x is None or x != x else f"{x:.{n}f}"


def _tx(s: str) -> str:
    """Escape a Python identifier / free-text label for LaTeX text mode."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("|", r"\textbar{}")):
        s = s.replace(a, b)
    return s


def render_tex(c: Cache) -> None:
    agg = aggregate_h2h(c, "h2h", ("batch", "cafecd"))
    grp = group_aggregate(agg, c.cell("chance"))
    lz = learnability(grp, c)
    pub = c.d.get("published", {}).get("data", {})
    cal = c.cell("calibration").get("summary", {})
    rt = {k.split("|", 1)[1]: v for k, v in c.cell("runtime").items()}
    pit = c.cell("pit").get("summary", {})
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)

    L: List[str] = []
    A = L.append
    A("% bench/exp_bench_dynamics.py -- CausalDynamics head-to-head.")
    A("% Self-loop convention: the FULL P x P matrix is scored (the")
    A("% CausalDynamics convention: the ground truth is a Jacobian sparsity")
    A("% pattern whose diagonal is a real self-dynamics edge).  Columns marked")
    A("% 'off' exclude the diagonal and are the structure-only numbers.")
    A("% Ground-truth orientation: 'simple' stores [effect,cause]; 'coupled'")
    A("% and 'climate' store [cause,effect] and are transposed before scoring.")
    if CAFE_CD_PROVISIONAL:
        A("% CAFE-CD: PROVISIONAL. " + CAFE_CD_PROVISIONAL_NOTE)
    A("")

    # --- Table 1: calibration ------------------------------------------- #
    if cal:
        A(r"\begin{table}[t]\centering\small")
        A(r"\caption{Reproduction of published CausalDynamics cells "
          r"(\texttt{simple}, \texttt{noise=0.00\_confounder=False}, 54 scored "
          r"systems, $\tau_{\max}=1$, self-loops scored). Rows without a "
          r"published counterpart show what the same estimator does once the "
          r"upstream wrapper's defects are removed.}")
        A(r"\label{tab:bench-dynamics-calibration}")
        A(r"\begin{tabular}{lrrrrr}\toprule")
        A(r"variant & AUROC & AUPRC & pub.\ AUROC & pub.\ AUPRC & $\Delta$AUROC\\\midrule")
        for r in cal.get("variants", []):
            lab = _tx(r["label"])
            A(f"{lab} & {_fmt(r['auroc'],4)} & {_fmt(r['auprc'],4)} & "
              f"{_fmt(r.get('published_auroc'),4)} & "
              f"{_fmt(r.get('published_auprc'),4)} & "
              f"{_fmt(r.get('d_auroc'),4)} \\\\")
        A(r"\bottomrule\end{tabular}\end{table}")
        A("")

    # --- Table 2: per-tier head-to-head --------------------------------- #
    order = [g for g in ("simple", "coupled", "climate") if g in grp]
    if order:
        A(r"\begin{table*}[t]\centering\small")
        A(r"\caption{CausalDynamics batch board: 12 external baselines plus "
          r"the measured chance control, per tier. AUPRC is the lead metric on "
          r"the sparse (\texttt{coupled}) tier as the benchmark authors "
          r"recommend; on \texttt{simple} the ground truth is "
          r"\emph{dense}, so AUPRC is bounded below by a high base rate and "
          r"neither metric separates anything. `off' columns exclude "
          r"self-loops. Every method gets the same lag budget $K=3$, the same "
          r"preprocessing and the same 600\,s timeout.}")
        A(r"\label{tab:bench-dynamics-h2h}")
        A(r"\begin{tabular}{l" + "rrrr" * len(order) + r"}\toprule")
        A(" & " + " & ".join(
            r"\multicolumn{4}{c}{\texttt{" + g + "}}" for g in order) + r"\\")
        A("method & " + " & ".join(
            ["AUPRC & off & AUROC & off"] * len(order)) + r"\\\midrule")
        meths = sorted({m for g in order for m in grp[g]["methods"]})
        for m in meths:
            cells = []
            for g in order:
                r = grp[g]["methods"].get(m)
                cells += ([_fmt(r["auprc"]), _fmt(r["auprc_off"]),
                           _fmt(r["auroc"]), _fmt(r["auroc_off"])]
                          if r else ["--"] * 4)
            nm = _tx(m)
            if m == CAFE_CD:
                nm = (r"\textbf{" + nm + r"} (provisional)"
                      if CAFE_CD_PROVISIONAL else r"\textbf{" + nm + "}")
            A(nm + " & " + " & ".join(cells) + r"\\")
        if CAFE_CD not in meths:
            A(r"\textbf{cafe\_cd} & \multicolumn{" + str(4 * len(order))
              + r"}{c}{\emph{absent -- \texttt{src/cafe/discovery.py} not "
                r"importable at run time}}\\")
        A(r"\midrule")
        cells = []
        for g in order:
            b = grp[g]
            cells += [_fmt(b["chance_auprc"]), _fmt(b["chance_auprc_off"]),
                      _fmt(b["chance_auroc"]), _fmt(b["chance_auroc"])]
        A(r"\emph{random control} & " + " & ".join(cells) + r"\\")
        cells = []
        for g in order:
            b = grp[g]
            cells += [f"{b['n_settings']}", f"{b['n_systems']}",
                      _fmt(b["P"], 1), _fmt(b["base_rate"])]
        A(r"\midrule settings / systems / $P$ / base rate & "
          + " & ".join(cells) + r"\\")
        A(r"\bottomrule\end{tabular}\end{table*}")
        A("")

    # --- Table 3: learnability verdict ---------------------------------- #
    if lz:
        A(r"\begin{table}[t]\centering\small")
        A(r"\caption{Learnability verdict. A tier counts as learnable only if "
          r"the best method's \emph{paired} off-diagonal AUROC lift over a "
          r"uniform-random score drawn on the same system exceeds 0.05 with a "
          r"paired $t$ above 3. AUROC rather than AUPRC because off-diagonal "
          r"base rates span 0.15--0.80 across these tiers; "
          r"$\Delta$AUPRC is shown alongside, and on \texttt{simple} it is "
          r"small because the graph is dense, not because nothing is learned.}")
        A(r"\label{tab:bench-dynamics-learnable}")
        A(r"\begin{tabular}{llrrrrrl}\toprule")
        A(r"tier & best method & AUROC$_\text{off}$ & lift & $t$ & win\% & "
          r"$\Delta$AUPRC & verdict\\\midrule")
        for g in order:
            v = lz.get(g)
            if not v:
                continue
            A(f"\\texttt{{{g}}} & {_tx(v['best_method'])} & "
              f"{_fmt(v['best_auroc_off'])} & {_fmt(v['auroc_off_lift'])} & "
              f"{_fmt(v['auroc_off_t'], 1)} & "
              f"{_fmt(100 * v['auroc_off_win_rate'], 0)} & "
              f"{_fmt(v['auprc_off_lift'])} & "
              + ("learnable" if v["learnable"] else r"\textbf{at chance}")
              + r"\\")
        A(r"\bottomrule\end{tabular}\end{table}")
        A("")

    # --- Table 3b: paired control ---------------------------------------- #
    pv = paired_vs_chance(c)
    if pv:
        gs = [g for g in ("simple", "coupled", "climate") if g in pv]
        A(r"\begin{table}[t]\centering\small")
        A(r"\caption{Paired comparison against a uniform-random score "
          r"evaluated on the \emph{same} system (off-diagonal AUPRC). Pairing "
          r"is necessary because the attainable AUPRC is set by each system's "
          r"base rate, which ranges from 0.15 to 0.69 across these tiers. "
          r"$t$ is a paired $t$-statistic over systems; win\% is the fraction "
          r"of systems on which the method beats its own control.}")
        A(r"\label{tab:bench-dynamics-paired}")
        A(r"\begin{tabular}{l" + "rrr" * len(gs) + r"}\toprule")
        A(" & " + " & ".join(r"\multicolumn{3}{c}{\texttt{" + g + "}}"
                             for g in gs) + r"\\")
        A("method & " + " & ".join(["lift & $t$ & win\\%"] * len(gs))
          + r"\\\midrule")
        for m in sorted({m for g in gs for m in pv[g]}):
            cells = []
            for g in gs:
                r = pv[g].get(m)
                cells += ([_fmt(r["mean_lift"], 4), _fmt(r["t"], 2),
                           _fmt(100 * r["win_rate"], 0)] if r else ["--"] * 3)
            nm = _tx(m)
            if m == CAFE_CD and CAFE_CD_PROVISIONAL:
                nm = r"\textbf{" + nm + r"} (provisional)"
            A(nm + " & " + " & ".join(cells) + r"\\")
        A(r"\bottomrule\end{tabular}\end{table}")
        A("")

    # --- Table 4: runtime ------------------------------------------------ #
    if rt:
        cols = list(rt)[:4]
        A(r"\begin{table}[t]\centering\small")
        A(r"\caption{Single-threaded CPU wall-clock, seconds per fit "
          r"($T=1000$, $K=3$), measured in a dedicated sequential pass so the "
          r"head-to-head's process pool cannot contaminate it.}")
        A(r"\label{tab:bench-dynamics-runtime}")
        A(r"\begin{tabular}{l" + "r" * len(cols) + r"}\toprule")
        A("method & " + " & ".join(
            r"\texttt{" + k.split("/")[0] + r"} $P{=}$" + str(rt[k]["P"])
            for k in cols) + r"\\\midrule")
        for m in sorted({m for k in cols for m in rt[k]["sec"]}):
            A(_tx(m) + " & " + " & ".join(
                _fmt(rt[k]["sec"].get(m), 3) for k in cols) + r"\\")
        A(r"\bottomrule\end{tabular}\end{table}")
        A("")

    # --- Table 5: point-in-time board ------------------------------------ #
    if pit:
        A(r"\begin{table}[t]\centering\small")
        A(r"\caption{Point-in-time board. Graphs are scored after seeing only "
          r"the first $f$ of the series; batch rivals are refit on each "
          r"prefix. \textbf{These numbers are not comparable to "
          r"Table~\ref{tab:bench-dynamics-h2h} and are never ranked against "
          r"it}, exactly as \texttt{src/cafe/benchmark.py} separates causal "
          r"from bidirectional imputation.}")
        A(r"\label{tab:bench-dynamics-pit}")
        fr = list(pit)
        A(r"\begin{tabular}{ll" + "rr" * len(fr) + r"}\toprule")
        A(r"tier & method & " + " & ".join(
            r"\multicolumn{2}{c}{$f=" + f + "$}" for f in fr) + r"\\")
        A(" & & " + " & ".join(["AUPRC & off"] * len(fr)) + r"\\\midrule")
        gs = sorted({g for f in fr for g in pit[f]})
        for g in gs:
            ms = sorted({m for f in fr for m in pit[f].get(g, {}).get("methods", {})})
            for i, m in enumerate(ms):
                cells = []
                for f in fr:
                    r = pit[f].get(g, {}).get("methods", {}).get(m)
                    cells += ([_fmt(r["auprc"]), _fmt(r["auprc_off"])]
                              if r else ["--", "--"])
                A((f"\\texttt{{{g}}}" if i == 0 else "") + " & "
                  + _tx(m) + " & " + " & ".join(cells) + r"\\")
            A(r"\midrule")
        L[-1] = r"\bottomrule\end{tabular}\end{table}"
        A("")

    # --- Published board, for provenance --------------------------------- #
    if pub:
        A(r"\begin{table}[t]\centering\small")
        A(r"\caption{The CausalDynamics leaderboard, recovered by averaging "
          r"the benchmark's own shipped per-system evaluation outputs rather "
          r"than transcribed from the paper. The \emph{Null} column is a "
          r"constant prediction: on \texttt{simple} every published method "
          r"lies within $\pm0.02$ AUPRC of it.}")
        A(r"\label{tab:bench-dynamics-published}")
        A(r"\begin{tabular}{llrrrr}\toprule")
        A(r"tier & method & AUROC & AUPRC & null AUPRC & $\Delta$AUPRC\\\midrule")
        for subset in ("simple", "coupled", "climate"):
            blk = pub.get(subset)
            if not blk:
                continue
            first = True
            for m, v in sorted(blk["methods"].items(),
                               key=lambda kv: -kv[1]["Individual_AUPRC"]):
                A((f"\\texttt{{{subset}}}" if first else "") + " & "
                  + _tx(m) + " & "
                  + f"{_fmt(v['Individual_AUROC'],4)} & "
                    f"{_fmt(v['Individual_AUPRC'],4)} & "
                    f"{_fmt(v['Null_AUPRC'],4)} & "
                    f"{_fmt(v['Individual_AUPRC'] - v['Null_AUPRC'],4)}" + r"\\")
                first = False
            A(r"\midrule")
        L[-1] = r"\bottomrule\end{tabular}\end{table}"

    with open(TEX_OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[render] wrote {TEX_OUT} ({len(L)} lines)")


def render_fig(c: Cache) -> None:
    """Two rows: the level (AUPRC vs the measured control) and the TEST
    (paired off-diagonal AUROC lift with 2 s.e. bars).  The second row is what
    the learnability verdict is actually computed from, so the figure and the
    verdict cannot drift apart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate_h2h(c, "h2h", ("batch", "cafecd"))
    grp = group_aggregate(agg, c.cell("chance"))
    pv = paired_vs_chance(c, "auroc_off")
    order = [g for g in ("simple", "coupled", "climate") if g in grp]
    if not order:
        print("[render] no head-to-head data yet; figure skipped")
        return
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)

    def _ok(x):
        return x == x and np.isfinite(x)

    n = len(order)
    fig, axes = plt.subplots(2, max(n, 2), figsize=(4.4 * max(n, 2), 7.6))
    axes = np.atleast_2d(axes)

    for j, g in enumerate(order):
        blk = grp[g]
        ms = sorted([m for m in blk["methods"]
                     if _ok(blk["methods"][m]["auprc_off"])],
                    key=lambda m: blk["methods"][m]["auprc_off"])
        y = np.arange(len(ms))

        def _colors(names):
            return ["#c44e52" if m == CAFE_CD else "#4c72b0" for m in names]

        # -- row 1: level, against the measured random control -------------- #
        ax = axes[0, j]
        vals = [blk["methods"][m]["auprc_off"] for m in ms]
        bars = ax.barh(y, vals, color=_colors(ms), height=.7)
        if CAFE_CD in ms and CAFE_CD_PROVISIONAL:
            bars[ms.index(CAFE_CD)].set_hatch("///")
        ax.axvline(blk["chance_auprc_off"], color="k", ls="--", lw=1.2,
                   label=f"random control {blk['chance_auprc_off']:.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels([m + (" *" if m == CAFE_CD and CAFE_CD_PROVISIONAL
                                 else "") for m in ms], fontsize=7)
        ax.set_xlabel("off-diagonal AUPRC")
        ax.set_title(f"{g}   P={blk['P']:.1f}, {blk['n_systems']} systems,\n"
                     f"off-diag base rate {blk['chance_auprc_off']:.2f}"
                     f"  (headroom {1 - blk['chance_auprc_off']:.2f})",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=.3)

        # -- row 2: the paired test ----------------------------------------- #
        ax = axes[1, j]
        pg = pv.get(g, {})
        ms2 = sorted([m for m in ms if m in pg and _ok(pg[m]["mean_lift"])],
                     key=lambda m: pg[m]["mean_lift"])
        y2 = np.arange(len(ms2))
        lifts = [pg[m]["mean_lift"] for m in ms2]
        errs = [2 * pg[m]["se"] if _ok(pg[m]["se"]) else 0.0 for m in ms2]
        bars = ax.barh(y2, lifts, xerr=errs, color=_colors(ms2), height=.7,
                       error_kw=dict(lw=.9, ecolor="0.25"))
        if CAFE_CD in ms2 and CAFE_CD_PROVISIONAL:
            bars[ms2.index(CAFE_CD)].set_hatch("///")
        ax.axvline(0.0, color="k", lw=1.2)
        ax.set_yticks(y2)
        ax.set_yticklabels([m + (" *" if m == CAFE_CD and CAFE_CD_PROVISIONAL
                                 else "") for m in ms2], fontsize=7)
        ax.set_xlabel("paired off-diagonal AUROC lift over a random score\n"
                      "on the SAME system  (bars: $\\pm 2$ s.e.)", fontsize=8)
        ax.grid(axis="x", alpha=.3)

    for j in range(n, axes.shape[1]):
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    note = ("  * CAFE-CD provisional: scores() emits a two-tier step, so its "
            "threshold-free metrics are lower bounds."
            if CAFE_CD_PROVISIONAL and any(
                CAFE_CD in grp[g]["methods"] for g in order) else "")
    fig.suptitle("CausalDynamics: what is actually learnable "
                 "(batch board, self-loops excluded)" + ("\n" + note if note
                                                         else ""),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94 if note else .96))
    fig.savefig(FIG_OUT)
    plt.close(fig)
    print(f"[render] wrote {FIG_OUT}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
CELLS = {"published": cell_published, "orientation": cell_orientation,
         "calibration": cell_calibration, "chance": cell_chance,
         "h2h": cell_h2h, "cafecd": cell_cafecd, "runtime": cell_runtime,
         "pit": cell_pit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", default="",
                    help="comma-separated subset of " + ",".join(CELLS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--force", default="", help="drop these cells first")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="print every cached cell without recomputing")
    args = ap.parse_args(argv)

    global N_WORKERS
    if args.workers:
        N_WORKERS = args.workers

    c = Cache()
    for f in [x for x in args.force.split(",") if x]:
        c.drop(f)
        print(f"[cache] dropped cell {f!r}")

    names = list(CELLS) if args.all else [x for x in args.cells.split(",") if x]
    for nm in names:
        if nm not in CELLS:
            print(f"unknown cell {nm!r}")
            return 2
        print(f"\n########## CELL {nm} ##########")
        try:
            CELLS[nm](c)
        except Exception:
            traceback.print_exc()
            print(f"[{nm}] FAILED -- cache preserved, rerun to resume")
    c.flush()

    if args.report or names or args.render:
        agg = aggregate_h2h(c, "h2h", ("batch", "cafecd"))
        if agg:
            grp = group_aggregate(agg, c.cell("chance"))
            if args.report:
                for fn, blob in (("published", c.d.get("published", {}).get("data")),
                                 ("orientation", c.d.get("orientation", {}).get("data")),
                                 ("calibration", c.cell("calibration").get("summary"))):
                    if blob:
                        {"published": print_published,
                         "orientation": print_orientation,
                         "calibration": print_calibration}[fn](blob)
                live = {t[0] for t in tiers()}
                print_chance({k: v for k, v in c.cell("chance").items()
                              if k in live})
                print_h2h(agg)
                rt = {k.split("|", 1)[1]: v for k, v in c.cell("runtime").items()}
                if rt:
                    print_runtime(rt)
                if c.cell("pit").get("summary"):
                    print_pit(c.cell("pit")["summary"])
            print_h2h(grp, "BATCH BOARD -- TIER GROUPS")
            print_coupled_sweep(coupled_sweep(c, agg))
            print_paired(paired_vs_chance(c))
            print_learnability(learnability(grp, c))
    if args.render or args.all or names:
        render_tex(c)
        render_fig(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
