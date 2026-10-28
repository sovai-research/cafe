r"""
CAFE-CD gate 2: does a genuinely SPARSE downstream estimator move the chi crossover?

THE OPEN QUESTION
-----------------
``exp_causal_residualization.py`` measured a clean crossover: CAFE's factor-adjusted
residual beats the raw panel for causal discovery only once the common-variance share
``chi`` is high (crossover at chi ~ 0.85 for E=10,T=60,P=24,R=3,K=2,beta_hetero=0.6).
Every one of those numbers came from an UNPENALISED pooled OLS-VAR downstream.

That may have removed the exact mechanism the thesis depends on.  The FNETS
(Barigozzi/Cho/Owens, JBES 2024) and Krampe (OBES 2025) argument for factor adjustment
is about SPARSE recovery: a pervasive factor makes the observed VAR *dense*, and
density is what breaks a sparse estimator's support recovery.  Unpenalised OLS has no
sparsity to protect, and with abundant rows it simply absorbs the factor -- the factor
is spanned by its own lagged regressors.  So the honest test is to rerun the chi sweep
with a downstream estimator that actually pays for density.

WHAT IS HELD FIXED
------------------
The controlled variable is the pair (REPRESENTATION, DOWNSTREAM ESTIMATOR).  Every
estimator is fed the byte-identical representation; every representation goes through
the SAME two-way (entity, time) FWL within-transform, ``raw`` and ``oracle`` included
(CAFE removes mu and time_fe internally, raw data does not -- skipping the transform on
raw would rig the comparison in CAFE's favour).

  arms         raw | cafe (routed engine) | cafe_pe (per-entity engine, pinned)
               | pca (batch, NON-CAUSAL) | oracle
  estimators   ols          -- exp_causal_residualization.pooled_var_scores, verbatim
               grouplasso   -- cafe._skeleton.panel_group_lasso, lag-grouped panel
                               group lasso; scored threshold-free by LAMBDA ENTRY ORDER
               lasso        -- plain per-target L1 covariance-form coordinate descent
                               VAR (numpy only), same entry-order score

``cafe_pe`` exists because the engine turned out to be a live confound: the routing
diagnostic below flips to ``pooled`` at LOW chi on this generator, and the per-entity
engine is the one the OLS reference table was measured with.  Carrying both makes the
engine a controlled variable, so a shift in the crossover is attributable to the
downstream estimator rather than to a routing decision that itself moves with chi.

ENTRY-ORDER SCORING
-------------------
A sparse estimator returns a support, not a t-statistic, so it has no natural continuous
score for an AUC.  The natural sparse analogue of ``|t|`` is the LAMBDA AT WHICH AN EDGE
ENTERS: run the whole regularisation path and rank each cell by ``L - first_entry_index``
(an edge surviving the largest lambda ranks highest, one that never enters ranks 0),
tie-broken by the coefficient magnitude at the densest fit.  This is threshold-free,
needs no model selection, and is exactly the ranking a practitioner reads off a path
plot.  Binary metrics are then taken at TRUE SPARSITY (top-n_true cells), so no
estimator is rewarded or punished for its own lambda-selection rule.

The group lasso groups over LAGS, so both lag cells of a pair (i,j) enter together and
its per-lag entry score is constant in k.  That is an honest property of the estimator
(it answers "j Granger-causes i at some lag"), it is common to all four arms so it
cannot bias the cafe-minus-raw contrast, and ``auc_group`` -- the pair-level AUC, the
granularity ``SkeletonResult.skeleton()`` actually delivers -- is reported alongside.

ENGINE ROUTING FOR THE ``cafe`` ARM
-----------------------------------
Neither loading engine wins everywhere, so the arm is routed by a measured,
threshold-free diagnostic whose null is 1 BY CONSTRUCTION.  Split every entity's own
rows in half and take each half's rank-r principal subspace; alignment is the mean
cosine of the principal angles.

    within  = mean_e  align(half_e1, half_e2)      -- the sampling-noise ceiling that
                                                      would hold if loadings were equal
    between = mean_{e != f} align(half_e*, half_f*)
    ratio   = between / within                     -- >= 0.95 -> pooled, else per_entity

Both sides are measured from T/2 rows, so they carry the same estimation noise and the
ratio is 1 under identical loadings.  A fixed cutoff on the RAW alignment does not work
(0.73 vs 0.98 across configs); the within-entity normalisation is what makes the
diagnostic portable.

TRAPS THIS FILE DELIBERATELY AVOIDS
-----------------------------------
* ``beta_hetero > 0`` always.  With ``shared_factors=True`` and ``beta_hetero=0`` the
  common component is bit-identically the same series for every entity, a two-way FWL
  demean deletes it EXACTLY, and ``raw == oracle`` to four decimals -- the experiment
  cannot discriminate at all.
* ``residuals(mask_imputed=True)`` returns NaN, not 0, at imputed cells; NaNs are
  carried into the estimators as an explicit ``obs`` mask, never zero-filled naively.
* Self-loops i->i at lag k are legitimate and usually the strongest true edges;
  ``causal_metrics.flat_lagged`` keeps them.  The PRIMARY metric is ``auc_cross``
  (off-diagonal only) because the lag-1 diagonal is dense by construction and would
  flatter any detector that only fires there.

WHAT IT WRITES
--------------
    bench/causal_sparse_gate_cache.json      (checkpointed after EVERY cell; reruns resume)
    paper/tables/causal_sparse_gate.tex      \label{tab:causalsparsegate},
                                             \label{tab:causalsparsegatecross}
    paper/figures/causal_sparse_gate.pdf     \label{fig:causalsparsegate}

COST, AND WHAT WAS CUT BECAUSE OF IT
------------------------------------
Measured per fit (one arm, one cell), 2 BLAS threads: the OLS-VAR is < 10 ms at every
size; the L1 VAR is 0.1 s at P=24, 1.3 s at P=60, 2.6 s at P=80; the panel group lasso
is 1.9 s at P=24, 42 s at P=60 and ~80 s at P=80.  The full 7-chi x 8-seed grid at
P=80 would therefore be about twelve hours of group lasso alone.  The response is to
cut the GRID and never the estimator: the wide configs run 5 chi x 5 seeds, and the
path length is 25 instead of 50 (measured to change the scores by < 0.004 AUC).  The
cell loop is SEED-MAJOR, so a run that is cut short still holds complete chi curves.

Run:
    python3 bench/exp_causal_sparse_gate.py
    python3 bench/exp_causal_sparse_gate.py --tables-only
    python3 bench/exp_causal_sparse_gate.py --force        # ignore the cache
    python3 bench/exp_causal_sparse_gate.py --merge        # fold shards -> canonical cache
Env knobs:
    SG_CHIS=0.4,0.5,0.6,0.7,0.8,0.9,0.95   SG_SEEDS=8
    SG_CONFIGS=base,wideA,wideB            SG_EST=ols,grouplasso,lasso
    SG_NLAM=25                             SG_LAMRATIO=0.005
    SG_CACHE=<path>                        SG_ROUTE_RMIN=3   SG_ROUTE_CUT=0.95
Configs may be run as parallel processes by giving each its own SG_CACHE, then
`--merge`.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import cafe                                                        # noqa: E402
import causal_metrics as cm                                        # noqa: E402
from gen_causal_panel import gen_causal_panel                      # noqa: E402
# REUSE, do not reimplement: the OLS baseline, the FWL transform and the PCA arm are
# imported verbatim from the experiment that produced the number under test.
from exp_causal_residualization import (fwl_demean, pooled_var_scores,   # noqa: E402
                                        pca_residual, ahn_horenstein_rank)
from cafe._skeleton import panel_group_lasso                       # noqa: E402
from viz_common import PALETTE, style_ax                           # noqa: E402

#: Shardable so several configs can run as separate processes without racing on one
#: file; ``--merge`` folds the shards back into the canonical cache.
CACHE = os.environ.get("SG_CACHE") or os.path.join(_HERE, "causal_sparse_gate_cache.json")
CANON = os.path.join(_HERE, "causal_sparse_gate_cache.json")
TABLE = os.path.join(_ROOT, "paper", "tables", "causal_sparse_gate.tex")
FIGURE = os.path.join(_ROOT, "paper", "figures", "causal_sparse_gate.pdf")

# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
CHIS = [float(x) for x in os.environ.get("SG_CHIS", "0.4,0.5,0.6,0.7,0.8,0.9,0.95").split(",")]
SEEDS = int(os.environ.get("SG_SEEDS", "8"))
#: The wide configs carry their OWN, deliberately reduced grid.  The panel group lasso
#: costs ~76 s per arm at P=60 and ~147 s at P=80 (measured, see the module report), so
#: the full 7 x 8 grid would be ~7 h and ~12 h respectively.  The grid is cut -- never the
#: estimator -- so every cell that IS run is the identical protocol.
CONFIGS = {
    # name      E   T   P    -- n/col = E(T-K)/(K*P), the density regime that matters
    "base":  dict(E=10, T=60, P=24, chis=CHIS, seeds=SEEDS),          # n/col = 12.1
    "wideA": dict(E=8,  T=40, P=60,                                    # n/col =  2.5
                  chis=[0.4, 0.6, 0.8, 0.9, 0.95], seeds=5),
    "wideB": dict(E=10, T=40, P=80,                                    # n/col =  2.4
                  chis=[0.4, 0.6, 0.8, 0.9, 0.95], seeds=5),
}
CFG_NAMES = [c for c in os.environ.get("SG_CONFIGS", "base,wideA,wideB").split(",") if c]
ESTIMATORS = [e for e in os.environ.get("SG_EST", "ols,grouplasso,lasso").split(",") if e]
#: Path length for BOTH sparse estimators.  Measured: L=25 and L=50 over the same
#: lambda range give the same scores to within 0.004 AUC (group lasso 0.6885 vs 0.6882 at
#: P=60; L1 VAR 0.9211 vs 0.9238 at P=24) for 1.8-2.0x the speed, so the shorter path is
#: taken -- a cost decision that provably does not touch the ranking under test.
N_LAM = int(os.environ.get("SG_NLAM", "25"))
LAM_RATIO = float(os.environ.get("SG_LAMRATIO", "0.005"))
#: Override the per-config grid (used when one config is split across several shards --
#: e.g. the cheap estimators on the full grid, the group lasso on a reduced one).
FORCE_CHIS = ([float(x) for x in os.environ["SG_FORCE_CHIS"].split(",")]
              if os.environ.get("SG_FORCE_CHIS") else None)
FORCE_SEEDS = int(os.environ["SG_FORCE_SEEDS"]) if os.environ.get("SG_FORCE_SEEDS") else None
ROUTE_RMIN = int(os.environ.get("SG_ROUTE_RMIN", "3"))
ROUTE_CUT = float(os.environ.get("SG_ROUTE_CUT", "0.95"))

R_TRUE, K_TRUE, DEG, BETA_HET = 3, 2, 2, 0.6
#: ``cafe`` is the ROUTED engine (the arm under specification); ``cafe_pe`` pins the
#: per-entity engine, which is what the OLS reference table this experiment is compared
#: against actually used.  Carrying both makes the engine a CONTROLLED variable, so a
#: shift in the crossover can be attributed to the downstream estimator rather than to a
#: routing decision that happens to change with chi.
ARMS = ["raw", "cafe", "cafe_pe", "pca", "oracle"]
NONCAUSAL = {"pca"}          # batch SVD over every row, future included -- must be labelled
EST_LABEL = {"ols": "OLS-VAR (unpenalised)", "grouplasso": "panel group lasso",
             "lasso": "L1 VAR (per target)"}
ARM_LABEL = {"raw": "raw", "cafe": "cafe (routed)", "cafe_pe": "cafe (per-entity)",
             "pca": "pca", "oracle": "oracle"}


# --------------------------------------------------------------------------- #
# engine router -- measured, threshold-free, null ratio == 1 by construction
# --------------------------------------------------------------------------- #
def _top_subspace(M2d, r):
    """Orthonormal (P, r) basis of the top-r right-singular subspace of column-demeaned M."""
    Z = np.asarray(M2d, float)
    Z = Z - Z.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Vt[:r].T


def _align(U, V):
    """Mean cosine of the principal angles between two subspaces -- 1.0 == identical."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0)))


def route_engine(X3d, obs, r_max=6, r_min=None, cut=None):
    """{'engine','within','between','ratio','r','r_ah'} -- pooled iff between/within >= cut.

    Both sides are half-sample subspaces, so they carry identical sampling noise and
    the statistic is 1 in expectation when every entity shares one loading matrix.
    A fixed cutoff on the raw `between` alignment is NOT portable (0.73 vs 0.98 across
    configs); dividing by each panel's own within-entity ceiling is what makes it so.

    THE RANK IS FLOORED AT ``r_min`` (default 3), and that floor is load-bearing.  The
    ratio's discriminating power collapses when ``r`` under-shoots the true factor count,
    because a rank-1 principal direction is shared across entities even when the loadings
    are strongly heterogeneous; over-shooting merely adds equally-noisy directions to BOTH
    sides of the ratio, where the within-normalisation cancels them.  Measured on this
    generator at chi=0.8 (mean of 3 seeds, homogeneous / heterogeneous loadings):

        r = 1  ->  1.30 / 1.01     r = 3  ->  1.03 / 0.78     r = 6  ->  1.09 / 0.91
        r = 2  ->  1.17 / 0.92     r = 4  ->  1.05 / 0.84     r = 10 -> 1.07 / 1.00

    so anything below 3 inverts the decision and anything much above 6 washes it out.
    Ahn-Horenstein picks r = 1 here (one eigenvalue dominates), which is exactly the
    failure mode above -- hence the floor.  ``r = 3`` reproduces the calibration the
    diagnostic was designed against (1.03/1.05/1.04 homogeneous vs 0.78/0.80/0.79
    heterogeneous).  Nothing in the experiment RESTS on the router being right: the
    ``cafe_pe`` arm pins the per-entity engine unconditionally and is reported alongside.
    """
    r_min = ROUTE_RMIN if r_min is None else r_min
    cut = ROUTE_CUT if cut is None else cut
    n_ent, T, P = X3d.shape
    Z = np.where(obs, X3d, np.nan).reshape(-1, P)
    col = np.nanmean(Z, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    F = np.where(np.isfinite(Z), Z, col).reshape(n_ent, T, P)
    r_ah = ahn_horenstein_rank(F.reshape(-1, P) - F.reshape(-1, P).mean(0, keepdims=True),
                               kmax=min(8, P - 1))
    h = T // 2
    r = int(max(1, min(max(r_ah, r_min), r_max, h - 1, P - 1)))
    halves = [(_top_subspace(F[e, :h], r), _top_subspace(F[e, h:], r)) for e in range(n_ent)]
    within = float(np.mean([_align(a, b) for a, b in halves]))
    btw = [_align(halves[e][a], halves[f][b])
           for e in range(n_ent) for f in range(n_ent) if f != e
           for a in (0, 1) for b in (0, 1)]
    between = float(np.mean(btw)) if btw else float("nan")
    ratio = between / max(within, 1e-12)
    return {"engine": "pooled" if ratio >= cut else "per_entity",
            "within": within, "between": between, "ratio": float(ratio),
            "r": r, "r_ah": int(r_ah)}


# --------------------------------------------------------------------------- #
# pooled moments -- the SAME pairwise-complete construction pooled_var_scores uses
# --------------------------------------------------------------------------- #
def pooled_moments(E, obs, K):
    """(G, C, yy, n_eff) for the pooled lag-1..K design, available-case normalised.

    Complete-case deletion is not an option (at P=80,K=2 the design has 160 columns);
    zero-filling without renormalising by each pair's OWN valid count would shrink every
    moment by the co-observation rate.  ``_selfcheck`` asserts that the OLS |t| rebuilt
    from these moments reproduces ``pooled_var_scores`` exactly, so this helper can
    never silently drift from the reference estimator it shares a design with.
    """
    n_ent, T, P = E.shape
    ok = np.asarray(obs, bool) & np.isfinite(E)
    Z = np.where(ok, E, 0.0)
    V = ok.astype(float)
    ys, vys, xs, vxs = [], [], [], []
    for e in range(n_ent):
        ys.append(Z[e, K:T, :]);  vys.append(V[e, K:T, :])
        xs.append(np.concatenate([Z[e, K - k:T - k, :] for k in range(1, K + 1)], axis=1))
        vxs.append(np.concatenate([V[e, K - k:T - k, :] for k in range(1, K + 1)], axis=1))
    Y = np.concatenate(ys, 0);  VY = np.concatenate(vys, 0)
    X = np.concatenate(xs, 0);  VX = np.concatenate(vxs, 0)
    NG = np.maximum(VX.T @ VX, 1.0)
    G = (X.T @ X) / NG
    NC = np.maximum(VX.T @ VY, 1.0)
    C = (X.T @ Y) / NC
    yy = (Y * Y).sum(0) / np.maximum(VY.sum(0), 1.0)
    n_eff = float(np.median(NG)) if np.isfinite(np.median(NG)) else float(len(Y))
    return G, C, yy, n_eff


# --------------------------------------------------------------------------- #
# sparse downstream estimator 1: plain per-target L1 VAR (covariance-form CD)
# --------------------------------------------------------------------------- #
def lasso_path_scores(E, obs, K, n_lambda=N_LAM, lam_ratio=LAM_RATIO,
                      tol=1e-6, max_sweeps=60):
    """Entry-order score (K,P,P)[k-1, i(effect), j(cause)] from an L1 VAR path.

    Covariance-form coordinate descent (glmnet's ``G``/``c`` updates) on the pooled
    moments, solved for ALL P targets simultaneously -- every target shares one design,
    so one rank-1 update of the (A, P) gradient serves the whole panel and a full sweep
    costs A * A * P flops with no design matrix ever materialised.

    One GLOBAL lambda grid is used for every target (targets are standardised to unit
    response variance first), so entry order is comparable across equations -- which is
    what an AUC pooled over all cells requires.
    """
    G, C, yy, _ = pooled_moments(E, obs, K)
    A = G.shape[0]
    P = C.shape[1]
    d = np.sqrt(np.maximum(np.diag(G), 1e-12))
    Gs = G / np.outer(d, d)
    np.fill_diagonal(Gs, 1.0)
    sy = np.sqrt(np.maximum(yy, 1e-12))
    Cs = C / d[:, None] / sy[None, :]                       # (A,P) correlations
    Cs = np.where(np.isfinite(Cs), Cs, 0.0)

    lam_max = float(np.max(np.abs(Cs))) if Cs.size else 0.0
    if not np.isfinite(lam_max) or lam_max <= 0:
        return np.zeros((K, P, P)), np.zeros((K, P, P))
    lams = lam_max * lam_ratio ** (np.arange(n_lambda) / max(n_lambda - 1, 1))

    B = np.zeros((A, P))
    Rg = Cs.copy()                                          # Rg = Cs - Gs @ B
    first = np.full((A, P), n_lambda, dtype=np.int32)
    for l, lam in enumerate(lams):
        for _ in range(max_sweeps):
            delta_max = 0.0
            for a in range(A):
                r = Rg[a] + B[a]                            # Gs[a,a] == 1
                new = np.sign(r) * np.maximum(np.abs(r) - lam, 0.0)
                delta = new - B[a]
                nz = delta != 0.0
                if nz.any():
                    B[a] = new
                    Rg[:, nz] -= np.outer(Gs[:, a], delta[nz])
                    delta_max = max(delta_max, float(np.max(np.abs(delta))))
            if delta_max < tol:
                break
        newly = (B != 0.0) & (first == n_lambda)
        first[newly] = l

    base = (n_lambda - first).astype(float)
    mag = np.abs(B)                                          # densest fit on the path
    score = base + 0.999 * mag / (float(mag.max()) + 1e-300)
    S = score.reshape(K, P, P).transpose(0, 2, 1)            # -> [k-1, i(effect), j(cause)]
    Bk = mag.reshape(K, P, P).transpose(0, 2, 1)
    return S, Bk


# --------------------------------------------------------------------------- #
# sparse downstream estimator 2: the library's lag-grouped panel group lasso
# --------------------------------------------------------------------------- #
def grouplasso_path_scores(E, obs, K, n_lambda=N_LAM, lam_ratio=LAM_RATIO):
    """Entry-order score from ``cafe._skeleton.panel_group_lasso``'s lambda path.

    ``demean='none'``: the two-way FWL transform is applied identically to every arm
    OUTSIDE this call, so the group lasso and the OLS baseline see byte-identical input.
    ``fit_deviations=False``: the generator's graph is homogeneous across entities
    (``hetero=0``), so the deviation level can only add variance and cost -- this is the
    sparse analogue of the POOLED OLS-VAR, which is the estimator under test.
    ``screening='gapsafe'``: the duality-gap sphere test is exact while the deviation
    level is inactive, so it returns the identical solution (verified: 0.68850 under both
    screenings at P=60) for ~12% less time.
    """
    res = panel_group_lasso(np.asarray(E, float), candidates=None,
                            obs=np.asarray(obs, bool), max_lag=K,
                            n_lambda=n_lambda, lambda_ratio=lam_ratio,
                            select="ebic", screening="gapsafe", demean="none",
                            fit_deviations=False, available_case=True, verbose=False)
    sup = np.stack([np.asarray(s, bool) for s in res.path_support])     # (L,K,P,P)
    L = sup.shape[0]
    ever = sup.any(axis=0)
    first = np.where(ever, np.argmax(sup, axis=0), L)
    base = (L - first).astype(float)
    mag = np.abs(np.asarray(res.B0, float))
    score = base + 0.999 * mag / (float(mag.max()) + 1e-300)
    return score, mag, res


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def topk_hat(S, n_true):
    """Thresholded graph at TRUE sparsity: the n_true highest-|score| cells."""
    s = np.abs(np.asarray(S, float)).reshape(-1)
    s = np.where(np.isfinite(s), s, -np.inf)
    idx = np.argsort(-s, kind="mergesort")[:int(n_true)]
    H = np.zeros(s.size)
    H[idx] = 1.0
    return H.reshape(np.asarray(S).shape)


def cross_f1(A_true, S):
    """F1 on OFF-DIAGONAL lagged cells at true cross-sparsity.

    With |predicted| == |true| the confusion table is square, so precision == recall
    == F1 and the single number is unambiguous.
    """
    A_true = np.asarray(A_true, float)
    P = A_true.shape[1]
    off = ~np.eye(P, dtype=bool)
    t = (np.abs(A_true) > 0)[:, off].reshape(-1)
    s = np.abs(np.asarray(S, float))[:, off].reshape(-1)
    n = int(t.sum())
    if n == 0:
        return float("nan")
    s = np.where(np.isfinite(s), s, -np.inf)
    idx = np.argsort(-s, kind="mergesort")[:n]
    h = np.zeros(t.size, dtype=bool)
    h[idx] = True
    return float(int((t & h).sum()) / n)


def auc_group(A_true, S):
    """Pair-level (any-lag) off-diagonal AUC -- the granularity a lag-grouped penalty
    actually estimates, so the group lasso is not charged for a distinction it never
    claims to make."""
    A_true = np.asarray(A_true, float)
    P = A_true.shape[1]
    off = ~np.eye(P, dtype=bool)
    t = (np.abs(A_true).sum(axis=0) > 0)[off]
    s = np.abs(np.asarray(S, float)).max(axis=0)[off]
    return float(cm.edge_auc(t, s))


def score_arm(A_true, S):
    n_true = int((np.abs(np.asarray(A_true, float)) > 0).sum())
    H = topk_hat(S, n_true)
    m = cm.summarize_graph(A_true, S, H)
    return {"auc": float(m["auc"]), "auc_cross": float(m["auc_cross"]),
            "auc_group": auc_group(A_true, S),
            "f1": float(m["f1"]), "f1_cross": cross_f1(A_true, S),
            "shd": int(m["shd"]),
            "lag_acc": float(m.get("lag_accuracy", np.nan))}


# --------------------------------------------------------------------------- #
# paired statistics
# --------------------------------------------------------------------------- #
def perm_p_onesided(d):
    """Exact paired sign-flip permutation p-value for H1: mean(d) > 0 (n <= 20)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return float("nan")
    obs = float(d.mean())
    if n <= 20:
        signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        stats = (signs * d[None, :]).mean(axis=1)
        return float(np.mean(stats >= obs - 1e-12))
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=(20000, n))
    return float(np.mean((signs * d[None, :]).mean(axis=1) >= obs - 1e-12))


def crossover_chi(chis, deltas):
    """Smallest chi where the paired mean cafe-minus-raw turns positive (linear interp
    between the bracketing grid points).  NaN if it never crosses inside the grid."""
    chis = np.asarray(chis, float)
    d = np.asarray(deltas, float)
    ok = np.isfinite(d)
    chis, d = chis[ok], d[ok]
    if d.size == 0:
        return float("nan"), float("nan")
    if d[0] > 0:
        # Already positive at the left edge: the crossover is BELOW the grid, not AT it.
        return float("-inf"), float(chis[0])
    for i in range(1, len(d)):
        if d[i] > 0 >= d[i - 1]:
            x0, x1, y0, y1 = chis[i - 1], chis[i], d[i - 1], d[i]
            xi = x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0) if y1 != y0 else x1
            return float(xi), float(chis[i])
    return float("nan"), float("nan")


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #
_SELFCHECKED = [False]


def _selfcheck(D, O, K):
    """Assert the moment helper reproduces the reference OLS |t| bit-for-bit."""
    G, C, yy, n_eff = pooled_moments(D, O, K)
    ridge = 1e-8 * float(np.mean(np.diag(G)))
    Gr = G + ridge * np.eye(G.shape[0])
    B = np.linalg.solve(Gr, C)
    Ginv = np.linalg.solve(Gr, np.eye(Gr.shape[0]))
    sig2 = np.maximum(yy - np.einsum("ai,ai->i", C, B), 1e-12)
    se = np.sqrt(np.outer(np.maximum(np.diag(Ginv), 1e-12), sig2) / max(n_eff - 1.0, 1.0))
    mine = (np.abs(B) / np.maximum(se, 1e-12)).reshape(K, -1, C.shape[1]).transpose(0, 2, 1)
    ref = pooled_var_scores(D, O, K)
    err = float(np.max(np.abs(mine - ref)))
    assert err < 1e-9, f"pooled_moments drifted from pooled_var_scores (max |dt| = {err:g})"
    print(f"[selfcheck] pooled_moments == pooled_var_scores  (max |dt| = {err:.2e})")


def _eff_rank(res):
    """Mean ARD-active factor count as a FLOAT.  ``PanelResult.effective_rank`` casts the
    cross-entity mean to int, which floors the informative 0 < rank < 1 regime to 0."""
    pe = getattr(res, "_pe", None)
    if pe:
        return float(np.mean([r.effective_rank() for _rows, r in pe.values()]))
    return float(res.effective_rank())


def run_cell(cfg_name, chi, seed):
    cfg = CONFIGS[cfg_name]
    E_n, T, P = cfg["E"], cfg["T"], cfg["P"]
    X3d, _X2d, _meta, truth = gen_causal_panel(
        E=E_n, T=T, P=P, R=R_TRUE, n_lags=K_TRUE, deg=DEG, chi=chi,
        noise="gauss", equal_variance=True, shared_factors=True,
        beta_hetero=BETA_HET, seed=seed)
    obs_full = np.ones(X3d.shape, dtype=bool)

    router = route_engine(X3d, obs_full)
    t0 = time.time()
    res = cafe.CAFE().run(X3d, engine=router["engine"])
    t_cafe = time.time() - t0
    Ec, Oc = res.residuals(stack=True, factor_adjusted=True, mask_imputed=True)
    if router["engine"] == "per_entity":
        res_pe, Epe, Ope = res, Ec, Oc                   # same fit; do not pay for it twice
    else:
        res_pe = cafe.CAFE().run(X3d, engine="per_entity")
        Epe, Ope = res_pe.residuals(stack=True, factor_adjusted=True, mask_imputed=True)

    reps = {"raw": (X3d, obs_full),
            "cafe": (Ec.reshape(X3d.shape), Oc.reshape(X3d.shape)),
            "cafe_pe": (Epe.reshape(X3d.shape), Ope.reshape(X3d.shape)),
            "oracle": (truth.E_idio, obs_full)}
    Ep, rk = pca_residual(X3d, obs_full, rank=None)
    reps["pca"] = (Ep, obs_full)

    row = {"cfg": cfg_name, "E": E_n, "T": T, "P": P, "chi": chi, "seed": seed,
           "router": router, "pca_rank": int(rk), "t_cafe": t_cafe,
           "eff_rank": _eff_rank(res), "eff_rank_pe": _eff_rank(res_pe),
           "n_over_col": float(E_n * (T - K_TRUE) / (K_TRUE * P)),
           "res": {}, "t_est": {}}

    demeaned = {}
    for arm in ARMS:
        Ea, Oa = reps[arm]
        D = fwl_demean(np.asarray(Ea, float), np.asarray(Oa, bool))
        Ofin = np.asarray(Oa, bool) & np.isfinite(D)
        demeaned[arm] = (D, Ofin)

    if not _SELFCHECKED[0]:
        _selfcheck(*demeaned["raw"], K_TRUE)
        _SELFCHECKED[0] = True

    for est in ESTIMATORS:
        row["res"][est] = {}
        t1 = time.time()
        for arm in ARMS:
            D, Ofin = demeaned[arm]
            if est == "ols":
                S = pooled_var_scores(D, Ofin, K_TRUE)
            elif est == "lasso":
                S, _ = lasso_path_scores(D, Ofin, K_TRUE)
            elif est == "grouplasso":
                S, _, _ = grouplasso_path_scores(D, Ofin, K_TRUE)
            else:
                raise ValueError(f"unknown estimator {est!r}")
            row["res"][est][arm] = score_arm(truth.A_lag, S)
        row["t_est"][est] = time.time() - t1
    return row


# --------------------------------------------------------------------------- #
# driver with per-cell checkpointing
# --------------------------------------------------------------------------- #
def _key(r):
    return f"{r['cfg']}|{r['chi']:.4f}|{r['seed']}"


def _grid(cfg_name):
    cf = CONFIGS[cfg_name]
    chis = list(FORCE_CHIS) if FORCE_CHIS else list(cf.get("chis", CHIS))
    seeds = FORCE_SEEDS if FORCE_SEEDS else int(cf.get("seeds", SEEDS))
    return chis, seeds


def _meta_block():
    return {"configs": {k: CONFIGS[k] for k in CFG_NAMES}, "chis": CHIS, "seeds": SEEDS,
            "estimators": ESTIMATORS, "arms": ARMS, "R_true": R_TRUE, "K_true": K_TRUE,
            "deg": DEG, "beta_hetero": BETA_HET, "n_lambda": N_LAM,
            "lambda_ratio": LAM_RATIO, "primary_metric": "auc_cross",
            "noncausal_arms": sorted(NONCAUSAL)}


def _save(rows):
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"rows": rows, "meta": _meta_block()}, f, indent=1)
    os.replace(tmp, CACHE)


def main(force=False):
    rows, done = [], set()
    if os.path.exists(CACHE) and not force:
        try:
            with open(CACHE) as f:
                rows = json.load(f).get("rows", [])
            done = {_key(r) for r in rows}
            print(f"[resume] {len(rows)} cells already in {os.path.basename(CACHE)}")
        except Exception as exc:                                   # pragma: no cover
            print(f"[resume] cache unreadable ({exc}); starting fresh")
            rows, done = [], set()

    # SEED-MAJOR on purpose: a run that is cut short still holds COMPLETE chi curves at
    # fewer seeds, which is a usable (if noisier) answer.  Chi-major would leave the high-
    # chi end -- the whole point of the sweep -- empty.
    todo = [(c, chi, s) for c in CFG_NAMES for s in range(_grid(c)[1])
            for chi in _grid(c)[0] if f"{c}|{chi:.4f}|{s}" not in done]
    plan = ", ".join(f"{c}:{len(_grid(c)[0])}x{_grid(c)[1]}" for c in CFG_NAMES)
    print(f"[plan] {len(todo)} cells to run  ({plan})")
    t_start = time.time()
    for i, (c, chi, s) in enumerate(todo, 1):
        t0 = time.time()
        r = run_cell(c, chi, s)
        rows.append(r)
        _save(rows)                                                # checkpoint EVERY cell
        est_txt = "  ".join(
            f"{e}:" + "/".join(f"{r['res'][e][a]['auc_cross']:.3f}" for a in ARMS)
            for e in ESTIMATORS)
        print(f"[{i:>3}/{len(todo)}] {c:<5} chi={chi:.2f} s={s} "
              f"eng={r['router']['engine'][:3]}({r['router']['ratio']:.2f}) "
              f"{est_txt}  [{time.time()-t0:.1f}s]", flush=True)
    print(f"[done] {len(rows)} cells, {time.time()-t_start:.1f}s -> {CACHE}")
    return rows


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
#: Metrics where SMALLER is better.  Their paired difference is sign-flipped so that a
#: positive ``delta`` always means "CAFE beat raw", and every crossover, win count and
#: p-value below therefore reads the same way for every metric.
LOWER_BETTER = {"shd"}


def aggregate(rows, metric="auc_cross"):
    """{(cfg, est, chi): {arm: mean, 'delta','wins','n','p','se'}} -- paired over seeds."""
    sign = -1.0 if metric in LOWER_BETTER else 1.0
    out = {}
    cfgs = sorted({r["cfg"] for r in rows}, key=lambda c: list(CONFIGS).index(c)
                  if c in CONFIGS else 99)
    for c in cfgs:
        ests = sorted({e for r in rows if r["cfg"] == c for e in r["res"]})
        for e in ests:
            for chi in sorted({r["chi"] for r in rows if r["cfg"] == c}):
                sel = sorted([r for r in rows if r["cfg"] == c and r["chi"] == chi
                              and e in r["res"]], key=lambda r: r["seed"])
                if not sel:
                    continue
                d = {}
                for a in ARMS:
                    v = [r["res"][e][a][metric] for r in sel if a in r["res"][e]]
                    d[a] = float(np.mean(v)) if v else float("nan")
                for tag, arm in (("", "cafe"), ("_pe", "cafe_pe")):
                    if arm not in ARMS or any(arm not in r["res"][e] for r in sel):
                        d["delta" + tag] = float("nan")
                        d["se" + tag] = float("nan")
                        d["wins" + tag] = 0
                        d["p" + tag] = float("nan")
                        continue
                    dif = sign * np.array(
                        [r["res"][e][arm][metric] - r["res"][e]["raw"][metric]
                         for r in sel], float)
                    d["delta" + tag] = float(np.mean(dif))
                    d["se" + tag] = (float(np.std(dif, ddof=1) / np.sqrt(len(dif)))
                                     if len(dif) > 1 else 0.0)
                    d["wins" + tag] = int(np.sum(dif > 0))
                    d["p" + tag] = perm_p_onesided(dif)
                d["n"] = int(len(sel))
                d["eff_rank"] = float(np.mean([r["eff_rank"] for r in sel]))
                d["engine"] = max({r["router"]["engine"] for r in sel},
                                  key=lambda g: sum(r["router"]["engine"] == g for r in sel))
                d["ratio"] = float(np.mean([r["router"]["ratio"] for r in sel]))
                out[(c, e, chi)] = d
    return out


def _xstr(xi, chis):
    """Human-readable crossover: -inf means the delta is already positive at the grid's
    left edge, so the true crossover is BELOW the grid, not at its first point."""
    if xi == -np.inf:
        return f"<= {min(chis):.2f}"
    if not np.isfinite(xi):
        return f"> {max(chis):.2f}"
    return f"{xi:.3f}"


def print_report(rows, metric="auc_cross"):
    agg = aggregate(rows, metric)
    cfgs = [c for c in CONFIGS if any(r["cfg"] == c for r in rows)]
    ests = [e for e in ["ols", "grouplasso", "lasso"]
            if any(e in r["res"] for r in rows)]
    cross = {}
    for c in cfgs:
        nc = next(r["n_over_col"] for r in rows if r["cfg"] == c)
        cf = CONFIGS[c]
        arrow = " (LOWER is better; delta sign-flipped so +ve = CAFE wins)" \
            if metric in LOWER_BETTER else ""
        print(f"\n=== config {c}: E={cf['E']} T={cf['T']} P={cf['P']}  "
              f"n/col={nc:.2f}   metric={metric}{arrow} ===")
        print(f"{'est':<11}{'chi':>6}{'raw':>8}{'cafe':>8}{'cafePE':>8}{'pca':>8}"
              f"{'oracle':>8}{'delta':>9}{'+-se':>7}{'wins':>7}{'p':>7}"
              f"{'dltPE':>8}{'winPE':>7}{'pPE':>7}{'effR':>6}{'eng':>5}")
        for e in ests:
            chis = sorted({k[2] for k in agg if k[0] == c and k[1] == e})
            for chi in chis:
                d = agg[(c, e, chi)]
                print(f"{e:<11}{chi:>6.2f}{d['raw']:>8.3f}{d['cafe']:>8.3f}"
                      f"{d['cafe_pe']:>8.3f}{d['pca']:>8.3f}{d['oracle']:>8.3f}"
                      f"{d['delta']:>+9.3f}{d['se']:>7.3f}"
                      f"{d['wins']:>5d}/{d['n']:<1d}{d['p']:>7.3f}"
                      f"{d['delta_pe']:>+8.3f}{d['wins_pe']:>5d}/{d['n']:<1d}"
                      f"{d['p_pe']:>7.3f}{d['eff_rank']:>6.2f}{d['engine'][:3]:>5}")
            for tag, key in (("", "delta"), ("_pe", "delta_pe")):
                xs = [agg[(c, e, chi)][key] for chi in chis]
                xi, xg = crossover_chi(chis, xs)
                cross[(c, e, tag)] = (xi, xg)
                arm = "cafe (routed)  " if tag == "" else "cafe (per-entity)"
                print(f"{'':<11}-> chi* for {arm}: {_xstr(xi, chis):<12}"
                      f"(first grid chi with delta>0 = "
                      f"{('%.2f' % xg) if np.isfinite(xg) else 'none'})")
    return agg, cross, cfgs, ests


# --------------------------------------------------------------------------- #
# figure + table
# --------------------------------------------------------------------------- #
def make_figure(rows, agg, cfgs, ests):
    ncol = max(len(ests), len(cfgs))
    fig, axes = plt.subplots(2, ncol, figsize=(4.0 * ncol, 6.6))
    axes = np.atleast_2d(axes)
    base = cfgs[0]
    arm_col = {"raw": PALETTE["grey"], "cafe": PALETTE["blue"],
               "cafe_pe": PALETTE["teal"], "pca": PALETTE["purple"],
               "oracle": PALETTE["green"]}

    for j in range(ncol):
        ax = axes[0, j]
        if j >= len(ests):
            ax.axis("off")
            continue
        e = ests[j]
        chis = sorted({k[2] for k in agg if k[0] == base and k[1] == e})
        for a in ARMS:
            ax.plot(chis, [agg[(base, e, c)][a] for c in chis], marker="o", ms=3.5,
                    lw=1.4, ls="--" if a == "cafe_pe" else "-", color=arm_col[a],
                    label=ARM_LABEL[a] + (" *" if a in NONCAUSAL else ""))
        ax.set_xlabel(r"common-variance share $\chi$")
        ax.set_ylabel("AUC (off-diagonal lagged edges)")
        ax.set_title(f"({chr(97+j)}) {EST_LABEL[e]}\nconfig {base}", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        style_ax(ax)

    est_col = {"ols": PALETTE["slate"], "grouplasso": PALETTE["red"],
               "lasso": PALETTE["amber"]}
    for j in range(ncol):
        ax = axes[1, j]
        if j >= len(cfgs):
            ax.axis("off")
            continue
        c = cfgs[j]
        nc = next(r["n_over_col"] for r in rows if r["cfg"] == c)
        ax.axhline(0.0, color=PALETTE["ink"], lw=0.9, ls="--", zorder=1)
        for e in ests:
            chis = sorted({k[2] for k in agg if k[0] == c and k[1] == e})
            dl = np.array([agg[(c, e, x)]["delta"] for x in chis])
            se = np.array([agg[(c, e, x)]["se"] for x in chis])
            ax.plot(chis, dl, "-o", ms=3.5, lw=1.5, color=est_col[e], label=EST_LABEL[e],
                    zorder=3)
            ax.fill_between(chis, dl - se, dl + se, color=est_col[e], alpha=0.15, zorder=2)
            dp = np.array([agg[(c, e, x)]["delta_pe"] for x in chis])
            if np.all(np.isfinite(dp)):
                ax.plot(chis, dp, "--s", ms=2.8, lw=1.0, color=est_col[e], alpha=0.75,
                        zorder=3)
            xi, _ = crossover_chi(chis, dl)
            if np.isfinite(xi):
                ax.axvline(xi, color=est_col[e], lw=0.8, alpha=0.55, zorder=1)
                ax.annotate(f"{xi:.2f}", (xi, ax.get_ylim()[0]), fontsize=6.5,
                            color=est_col[e], ha="center", va="bottom", rotation=90)
        cf = CONFIGS[c]
        ax.set_xlabel(r"common-variance share $\chi$")
        ax.set_ylabel(r"paired $\Delta$ AUC (cafe $-$ raw)")
        ax.set_title(f"({chr(97+ncol+j)}) {c}: E={cf['E']}, T={cf['T']}, P={cf['P']}"
                     f"  ($n/\\mathrm{{col}}={nc:.1f}$)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        style_ax(ax)

    fig.suptitle("Does a sparse downstream estimator move the factor-adjustment "
                 "crossover?\n(* = non-causal arm;  solid = routed engine, "
                 "dashed = per-entity engine)", fontsize=10, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIGURE), exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {FIGURE}")


def make_table(agg, cross, cfgs, ests):
    L = []
    L.append(r"\begin{table}[t]\centering\small")
    L.append(r"\setlength{\tabcolsep}{3.4pt}")
    L.append(r"\caption{\textbf{Where the factor-adjustment crossover sits, and whether a "
             r"genuinely sparse downstream estimator moves it.} Off-diagonal "
             r"lagged-edge AUC (\texttt{auc\_cross}) against the common-variance share "
             r"$\chi$, for the identical four representations fed to three downstream "
             r"estimators: an unpenalised pooled OLS-VAR, the lag-grouped panel group "
             r"lasso and a per-target $L_1$ VAR. The sparse estimators are scored "
             r"threshold-free by \emph{lambda entry order}, so no model-selection rule "
             r"enters. $\Delta$ is the paired mean (\textsc{cafe}$-$\textsc{raw}) over "
             r"seeds, `w' the paired win count and $p$ an exact one-sided sign-flip "
             r"permutation test. $\chi^\star$ is the interpolated crossover. The "
             r"\textsc{pca} arm (omitted for width; always the weakest) is non-causal. "
             r"$n/\mathrm{col}=E(T-K)/(KP)$.}")
    L.append(r"\label{tab:causalsparsegate}")
    ncols = 1 + 6 * len(ests)
    L.append(r"\begin{tabular}{r" + "|rrrr@{\\ }r@{\\ }r" * len(ests) + "}")
    L.append(r"\toprule")
    L.append(r" & " + " & ".join(
        r"\multicolumn{6}{c}{%s}" % EST_LABEL[e].replace("_", r"\_") for e in ests) + r" \\")
    L.append(r"$\chi$ & " + " & ".join(
        [r"raw & cafe & orc & $\Delta$ & w & $p$"] * len(ests)) + r" \\")
    for c in cfgs:
        cf = CONFIGS[c]
        chis = sorted({k[2] for k in agg if k[0] == c})
        nc = cf["E"] * (cf["T"] - K_TRUE) / (K_TRUE * cf["P"])
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\textbf{%s}: $E{=}%d$, $T{=}%d$, $P{=}%d$, "
                 r"$n/\mathrm{col}{=}%.1f$} \\" % (ncols, c, cf["E"], cf["T"],
                                                   cf["P"], nc))
        for chi in chis:
            cells = []
            for e in ests:
                d = agg.get((c, e, chi))
                if d is None:
                    cells.append(r"-- & -- & -- & -- & -- & --")
                    continue
                bold = r"\textbf{%.3f}" % d["cafe"] if d["delta"] > 0 else "%.3f" % d["cafe"]
                cells.append("%.3f & %s & %.3f & %+.3f & %d & %.3f"
                             % (d["raw"], bold, d["oracle"], d["delta"], d["wins"], d["p"]))
            L.append(("%.2f & " % chi) + " & ".join(cells) + r" \\")
        L.append(r"\cmidrule{2-%d}" % ncols)
        for tag, lbl in (("", r"routed"), ("_pe", r"per-ent.")):
            xr = []
            for e in ests:
                xi, _ = cross.get((c, e, tag), (float("nan"), float("nan")))
                if xi == -np.inf:
                    lab = r"$\leq %.2f$" % min(chis)
                elif np.isfinite(xi):
                    lab = "%.2f" % xi
                else:
                    lab = r"$>%.2f$" % max(chis)
                xr.append(r"\multicolumn{6}{c}{$\chi^\star_{\mathrm{%s}}=%s$}"
                          % (lbl.replace(".", "").replace("-", ""), lab))
            L.append(r"\multicolumn{1}{r|}{%s} & " % lbl + " & ".join(xr) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    # ---- companion table: the crossover summary, both CAFE engines -------------- #
    L.append("")
    L.append(r"\begin{table}[t]\centering\small")
    L.append(r"\caption{\textbf{Crossover $\chi^\star$ is set by the regime, not by the "
             r"downstream estimator.} Interpolated $\chi$ at which the paired "
             r"\textsc{cafe}$-$\textsc{raw} advantage in \texttt{auc\_cross} turns "
             r"positive, for each downstream estimator and for both CAFE loading engines "
             r"(\emph{routed} = the half-sample subspace-alignment diagnostic; "
             r"\emph{per-entity} = the engine pinned, as in the OLS reference).}")
    L.append(r"\label{tab:causalsparsegatecross}")
    L.append(r"\begin{tabular}{ll" + "r" * len(ests) + "}")
    L.append(r"\toprule")
    L.append(r"config & CAFE engine & " + " & ".join(
        EST_LABEL[e].replace("_", r"\_") for e in ests) + r" \\")
    L.append(r"\midrule")
    for c in cfgs:
        chis = sorted({k[2] for k in agg if k[0] == c})
        for tag, lbl in (("", "routed"), ("_pe", "per-entity")):
            cells = []
            for e in ests:
                xi, _ = cross.get((c, e, tag), (float("nan"), float("nan")))
                if xi == -np.inf:
                    cells.append(r"$\leq %.2f$" % min(chis))
                elif np.isfinite(xi):
                    cells.append("%.2f" % xi)
                else:
                    cells.append(r"$>%.2f$" % max(chis))
            L.append(("%s & %s & " % (c if tag == "" else "", lbl)) + " & ".join(cells)
                     + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[wrote] {TABLE}")


# --------------------------------------------------------------------------- #
def main_entry():
    force = "--force" in sys.argv
    if "--merge" in sys.argv:
        rows = merge_shards()
    elif "--tables-only" in sys.argv:
        with open(CANON) as f:
            rows = json.load(f)["rows"]
    else:
        rows = main(force=force)
    if not rows:
        print("no rows; nothing to report")
        return 1
    agg, cross, cfgs, ests = print_report(rows, "auc_cross")
    for m in ("auc_group", "f1_cross", "shd"):
        print(f"\n########## secondary metric: {m} ##########")
        print_report(rows, m)
    make_figure(rows, agg, cfgs, ests)
    make_table(agg, cross, cfgs, ests)
    print("\nVERDICT (auc_cross crossover chi*, interpolated):")
    print(f"  {'config':<7}{'engine':<12}" + "".join(f"{e:>13}" for e in ests))
    for c in cfgs:
        chis = sorted({k[2] for k in agg if k[0] == c})
        for tag, lbl in (("", "routed"), ("_pe", "per-entity")):
            cells = "".join(f"{_xstr(cross.get((c, e, tag), (np.nan, np.nan))[0], chis):>13}"
                            for e in ests)
            print(f"  {c:<7}{lbl:<12}{cells}")
    return 0


def merge_shards():
    """Fold every ``*_cache*.json`` shard next to the canonical cache into one file."""
    import glob
    rows, seen = [], set()
    pats = sorted(glob.glob(os.path.join(_HERE, "causal_sparse_gate_cache*.json")))
    for path in pats:
        try:
            with open(path) as f:
                got = json.load(f).get("rows", [])
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        n_new = n_merged = 0
        for r in got:
            k = _key(r)
            if k not in seen:
                seen.add(k)
                rows.append(r)
                n_new += 1
                continue
            # Same cell from a shard that ran DIFFERENT estimators: union the estimator
            # dicts rather than dropping one.  Everything outside ``res``/``t_est`` is a
            # deterministic function of (config, chi, seed), so the first copy wins.
            base_row = rows[[_key(x) for x in rows].index(k)]
            add = {e: v for e, v in r.get("res", {}).items() if e not in base_row["res"]}
            if add:
                base_row["res"].update(add)
                base_row.setdefault("t_est", {}).update(
                    {e: r.get("t_est", {}).get(e) for e in add})
                n_merged += 1
        print(f"  {os.path.basename(path):<44} {len(got):>4} rows, {n_new:>4} new, "
              f"{n_merged:>4} estimator-merged")
    tmp = CANON + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"rows": rows, "meta": _meta_block()}, f, indent=1)
    os.replace(tmp, CANON)
    print(f"[merged] {len(rows)} cells -> {CANON}")
    return rows


if __name__ == "__main__":
    sys.exit(main_entry())
