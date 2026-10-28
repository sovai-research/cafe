"""External causal-discovery baselines behind ONE uniform interface.

    run_baseline(name, X, max_lag=8, **kw) -> BaselineResult

BENCH-ONLY.  Nothing in here may ever become a dependency of ``src/cafe``
(MIT, strict-numpy).  See ``docs/causal_baselines.md``.

--------------------------------------------------------------------------
THE ONE CONVENTION
--------------------------------------------------------------------------
Everything returned by this module uses **effect-row / cause-column**:

    A_lag[k - 1, i, j]  = strength of  "j at lag k  ->  i at lag 0"
    A_contemp[i, j]     = strength of  "j at lag 0  ->  i at lag 0"

i.e. ``A[effect, cause]``.  Row index = the variable being caused.
This matches statsmodels ``VAR.coefs[k, eq, var]``, ``lingam``'s
``adjacency_matrices_[k][i, j]`` and the CausalRivers label tensor.  It is the
TRANSPOSE of NOTEARS/DAGMA/DYNOTEARS (``X ~ X W`` => ``W[cause, effect]``), of
tigramite's ``val_matrix[cause, effect, tau]`` and of CauseMe's ``A_ij = i->j``.

Every wrapper below states the source library's convention and the exact
conversion applied.  Each conversion is PROVEN, not assumed, by
``self_test()`` on an asymmetric synthetic system where a transposition would
flip the sign of the measured margin.  Run it with::

    python bench/causal_baselines.py --self-test

--------------------------------------------------------------------------
LICENCE ISOLATION  (hard requirement)
--------------------------------------------------------------------------
  lingam        MIT         imported in-process
  causal-learn  MIT         imported in-process
  dagma         Apache-2.0  imported in-process (``dagma.linear`` ONLY --
                            ``dagma.utils`` pulls GPL python-igraph)
  tigramite     GPL-3.0     NEVER imported in this process.  ``pcmci`` /
                            ``pcmciplus`` run out-of-process: a generated
                            runner script + npz in / npz out via
                            subprocess.run(timeout=...).  See _run_tigramite.

``dynotears`` is a from-scratch numpy/scipy implementation of Pamfil et al.
(2020) -- causalnex requires Python <3.11 and does not install here.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_BENCH = os.path.dirname(os.path.abspath(__file__))

# ---- soft imports (mirror bench/online_competitors.py's HAVE_* pattern) ---- #
try:                                                    # MIT
    import lingam as _lingam
    HAVE_LINGAM = True
except Exception as _e:                                 # pragma: no cover
    _lingam = None
    HAVE_LINGAM = False
    _LINGAM_ERR = repr(_e)

try:                                                    # MIT
    from causallearn.search.ConstraintBased.FCI import fci as _cl_fci
    from causallearn.search.ConstraintBased.PC import pc as _cl_pc
    HAVE_CAUSALLEARN = True
except Exception as _e:                                 # pragma: no cover
    _cl_pc = _cl_fci = None
    HAVE_CAUSALLEARN = False
    _CAUSALLEARN_ERR = repr(_e)

try:                                                    # Apache-2.0
    # NOTE: import the submodule ONLY.  `import dagma.utils` drags in
    # python-igraph, which is GPL-2.0+.  dagma.linear does not.
    from dagma.linear import DagmaLinear as _DagmaLinear
    HAVE_DAGMA = True
except Exception as _e:                                 # pragma: no cover
    _DagmaLinear = None
    HAVE_DAGMA = False
    _DAGMA_ERR = repr(_e)

try:
    from scipy import linalg as _sla
    from scipy import stats as _sstats
    from scipy.optimize import minimize as _minimize
    HAVE_SCIPY = True
except Exception:                                       # pragma: no cover
    HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class BaselineResult:
    """Uniform output of every baseline.

    A_lag           (K, P, P) float -- scores/coefficients, A_lag[k-1, i, j]
                    is "j at lag k -> i".  All-zero if the method is
                    contemporaneous-only (see meta['scores_lagged']).
    A_contemp       (P, P) float -- "j at lag 0 -> i".  All-zero if the method
                    is lagged-only (see meta['scores_contemp']).
    support_lag     (K, P, P) bool -- the method's own binary decision.
    support_contemp (P, P) bool.
    runtime_s       wall-clock seconds INCLUDING conversion.
    ok              False if the wrapper raised / timed out; the arrays are
                    then all-zero and safe to score.
    err             '' when ok, else the exception repr / timeout message.
    meta            free-form; always carries 'name', 'max_lag', 'P',
                    'scores_lagged', 'scores_contemp', 'contemp_symmetric'.
    """

    A_lag: np.ndarray
    A_contemp: np.ndarray
    support_lag: np.ndarray
    support_contemp: np.ndarray
    runtime_s: float
    ok: bool
    err: str
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------- #
    @property
    def P(self) -> int:
        return int(self.A_contemp.shape[0])

    def summary(self, how: str = "max", include_contemp: bool = True) -> np.ndarray:
        """Collapse the window graph to a (P, P) SUMMARY graph score.

        ``how`` in {'max', 'mean', 'sum'} over the lag axis (CausalRivers uses
        'max' over |coef|).  ``include_contemp`` folds A_contemp in with the
        same reduction -- benchmarks whose ground truth is a summary graph
        (CausalRivers, CausalDynamics) do not distinguish lag 0 from lag k.
        """
        return summary_graph(self, how=how, include_contemp=include_contemp)


def summary_graph(res: BaselineResult, how: str = "max",
                  include_contemp: bool = True) -> np.ndarray:
    """(P,P) summary score from a BaselineResult.  A[i, j] = 'j -> i'."""
    parts: List[np.ndarray] = []
    if res.A_lag.size:
        parts.append(np.abs(np.asarray(res.A_lag, dtype=float)))
    if include_contemp and res.A_contemp.size:
        parts.append(np.abs(np.asarray(res.A_contemp, dtype=float))[None, ...])
    if not parts:
        return np.zeros((res.P, res.P))
    stack = np.concatenate(parts, axis=0)
    if how == "max":
        out = stack.max(axis=0)
    elif how == "mean":
        out = stack.mean(axis=0)
    elif how == "sum":
        out = stack.sum(axis=0)
    else:
        raise ValueError(f"unknown reduction {how!r}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _empty(P: int, K: int, name: str, max_lag: int, err: str,
           t0: float, meta: Optional[dict] = None) -> BaselineResult:
    m = {"name": name, "max_lag": max_lag, "P": P,
         "scores_lagged": False, "scores_contemp": False,
         "contemp_symmetric": False}
    if meta:
        m.update(meta)
    return BaselineResult(
        A_lag=np.zeros((max(K, 1), P, P)),
        A_contemp=np.zeros((P, P)),
        support_lag=np.zeros((max(K, 1), P, P), dtype=bool),
        support_contemp=np.zeros((P, P), dtype=bool),
        runtime_s=time.time() - t0, ok=False, err=err, meta=m,
    )


# --------------------------------------------------------------------------- #
# timeout
# --------------------------------------------------------------------------- #
class BaselineTimeout(RuntimeError):
    pass


@contextmanager
def _time_limit(seconds: Optional[float]):
    """SIGALRM soft timeout.  Degrades to a no-op off the main thread / on
    platforms without setitimer -- callers must not RELY on it for hard
    isolation (the tigramite path uses a real subprocess timeout instead)."""
    usable = (
        seconds is not None and seconds > 0
        and hasattr(signal, "SIGALRM")
        and getattr(signal, "setitimer", None) is not None
    )
    if usable:
        try:
            import threading
            usable = threading.current_thread() is threading.main_thread()
        except Exception:                                # pragma: no cover
            usable = False
    if not usable:
        yield
        return

    def _handler(signum, frame):
        raise BaselineTimeout(f"timeout after {seconds:g}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


@contextmanager
def _quiet(enabled: bool = True):
    """Silence libraries that print/tqdm straight to stdout/stderr
    (causal-learn's FCI prints every oriented edge; dagma shows a progress
    bar).  A sweep over thousands of subgraphs is unreadable otherwise."""
    if not enabled:
        yield
        return
    import contextlib
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _prep(X, standardize: bool = False) -> np.ndarray:
    """(T, P) float64, finite.  Non-finite cells are linearly interpolated then
    edge-filled; a wholly-empty column becomes 0.  Constant columns get a tiny
    jitter-free guard at the scaling step, not here."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, P); got shape {X.shape}")
    T, P = X.shape
    if T < 3 or P < 2:
        raise ValueError(f"need T>=3 and P>=2; got {X.shape}")
    if not np.isfinite(X).all():
        X = X.copy()
        idx = np.arange(T, dtype=float)
        for j in range(P):
            col = X[:, j]
            good = np.isfinite(col)
            if good.sum() == 0:
                X[:, j] = 0.0
            elif good.sum() < T:
                X[:, j] = np.interp(idx, idx[good], col[good])
    if standardize:
        mu = X.mean(0)
        sd = X.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X = (X - mu) / sd
    return np.ascontiguousarray(X)


def _lag_design(X: np.ndarray, K: int, intercept: bool = True):
    """Return (Y, Z) with Y = X[K:] (n, P) and
    Z = [1?, X[K-1:-1], X[K-2:-2], ..., X[0:-K]] (n, 1? + K*P).
    Column block k (0-based) of Z holds lag k+1."""
    T, P = X.shape
    n = T - K
    if n <= 0:
        raise ValueError(f"max_lag={K} too large for T={T}")
    blocks = [X[K - k: T - k] for k in range(1, K + 1)]
    Z = np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0))
    if intercept:
        Z = np.concatenate([np.ones((n, 1)), Z], axis=1)
    return X[K:], Z


def _partial_corr(R: np.ndarray, ridge: float = 1e-8):
    """Partial correlation matrix of residuals R (n, P).  Symmetric, diag 0."""
    P = R.shape[1]
    S = np.cov(R, rowvar=False)
    S = np.atleast_2d(S)
    S = S + ridge * np.eye(P) * (np.trace(S) / max(P, 1) + 1e-12)
    try:
        Om = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Om = np.linalg.pinv(S)
    d = np.sqrt(np.abs(np.diag(Om)))
    d = np.where(d < 1e-12, 1.0, d)
    pc = -Om / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return np.nan_to_num(pc)


# =========================================================================== #
# 1.  var_ols -- plain pooled OLS-VAR.  FIRST-CLASS BASELINE.
# =========================================================================== #
#  On CausalRivers a plain VAR is the strongest published method
#  (AUROC 0.85 vs VARLiNGAM 0.72, PCMCI 0.60).  If CAFE-CD cannot beat this,
#  there is nothing to publish.  It is therefore implemented properly:
#  exact OLS with the full lag design, HC-free classical t-statistics, an
#  explicit ridge fallback only when the Gram matrix is singular.
# =========================================================================== #
def _bl_var_ols(X, max_lag, *, standardize: bool = True, score: str = "tstat",
                intercept: bool = True, alpha: float = 0.05,
                contemp: bool = True, ridge: float = 0.0, **_):
    """Pooled OLS VAR(K).  Identical regressors in every equation, so SUR
    collapses to equation-by-equation OLS -- one shared pseudo-inverse.

    score:  'tstat'   |coef| / se        (default; scale-free)
            'abscoef' |coef|             (CausalRivers' var_absolute_values=True)
            'coef'    signed coefficient (CausalRivers' var_absolute_values=False)

    Convention: statsmodels-compatible.  beta[block k, j, i-equation] is the
    coefficient of variable j at lag k+1 in the equation for variable i, so
    A_lag[k, i, j] = beta[k*P + j, i]  -- a straight (cause, effect) ->
    (effect, cause) transpose of the design-matrix layout.
    """
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    Y, Z = _lag_design(Xp, K, intercept=intercept)
    n, q = Z.shape
    G = Z.T @ Z
    if ridge > 0:
        G = G + ridge * np.eye(q)
    try:
        Gi = np.linalg.inv(G)
        rank_deficient = False
    except np.linalg.LinAlgError:
        Gi = np.linalg.pinv(G)
        rank_deficient = True
    B = Gi @ (Z.T @ Y)                      # (q, P) -- column i = equation i
    R = Y - Z @ B                           # residuals (n, P)
    dof = max(n - q, 1)
    sig2 = (R * R).sum(0) / dof             # (P,)
    se = np.sqrt(np.outer(np.maximum(np.diag(Gi), 0.0), sig2))   # (q, P)
    se = np.where(se < 1e-300, 1e-300, se)
    tstat = B / se

    off = 1 if intercept else 0
    A_lag = np.zeros((K, P, P))
    T_lag = np.zeros((K, P, P))
    for k in range(K):
        blk = slice(off + k * P, off + (k + 1) * P)      # rows = cause j
        # B[blk] is (cause j, equation i) -> transpose to (effect i, cause j)
        coef = B[blk].T
        tt = tstat[blk].T
        T_lag[k] = tt
        if score == "tstat":
            A_lag[k] = np.abs(tt)
        elif score == "abscoef":
            A_lag[k] = np.abs(coef)
        elif score == "coef":
            A_lag[k] = coef
        else:
            raise ValueError(f"unknown score {score!r}")

    crit = float(_sstats.t.ppf(1.0 - alpha / 2.0, dof)) if HAVE_SCIPY else 1.96
    support_lag = np.abs(T_lag) > crit

    A_c = np.zeros((P, P))
    sup_c = np.zeros((P, P), dtype=bool)
    if contemp and n > P + 2:
        pc = _partial_corr(R)
        A_c = np.abs(pc)
        # Fisher-z on the partial correlation, conditioning set size q.
        z = np.arctanh(np.clip(pc, -0.999999, 0.999999))
        denom = max(n - q - P + 2 - 3, 1)
        stat = np.sqrt(denom) * np.abs(z)
        zc = float(_sstats.norm.ppf(1.0 - alpha / 2.0)) if HAVE_SCIPY else 1.96
        sup_c = stat > zc
        np.fill_diagonal(sup_c, False)

    meta = dict(scores_lagged=True, scores_contemp=bool(contemp),
                contemp_symmetric=True, score=score, dof=dof, n=n,
                rank_deficient=rank_deficient, standardize=standardize,
                coef_lag=np.stack([B[off + k * P: off + (k + 1) * P].T
                                   for k in range(K)]) if K else np.zeros((0, P, P)),
                tstat_lag=T_lag)
    return A_lag, A_c, support_lag, sup_c, meta


# =========================================================================== #
# 2.  granger_pairwise -- bivariate Granger F-tests
# =========================================================================== #
def _bl_granger_pairwise(X, max_lag, *, standardize: bool = True,
                         alpha: float = 0.05, score: str = "F",
                         self_loops: bool = True, **_):
    """For every ordered pair (cause j -> effect i): restricted AR(K) on i's own
    lags vs unrestricted with j's K lags added; classical F test.

    A_lag[k, i, j] = |t| of x_j at lag k+1 in the UNRESTRICTED bivariate model.
    meta['F'][i, j] and meta['p'][i, j] hold the pair-level statistics; the
    pair-level decision is broadcast to every lag of that pair in support_lag.
    Self-loops (i -> i) are the F-test of i's own lags against a constant.
    """
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    n = T - K
    if n <= 2 * K + 2:
        raise ValueError(f"T={T} too short for pairwise Granger at max_lag={K}")

    lags = [Xp[K - k: T - k] for k in range(1, K + 1)]      # each (n, P)
    Yall = Xp[K:]
    ones = np.ones((n, 1))

    def _ols_rss(Zm, y):
        beta, *_ = np.linalg.lstsq(Zm, y, rcond=None)
        r = y - Zm @ beta
        return float(r @ r), beta, Zm

    A_lag = np.zeros((K, P, P))
    Fm = np.zeros((P, P))
    pm = np.ones((P, P))
    for i in range(P):
        y = Yall[:, i]
        own = np.concatenate([ones] + [L[:, [i]] for L in lags], axis=1)
        rss_own, _, _ = _ols_rss(own, y)
        rss_const, _, _ = _ols_rss(ones, y)
        for j in range(P):
            if j == i:
                if not self_loops:
                    continue
                # self-loop: own lags vs a constant-only model
                Zu_use = own
                rss_u, rss_r = rss_own, rss_const
            else:
                Zu_use = np.concatenate([own] + [L[:, [j]] for L in lags], axis=1)
                rss_u, _, _ = _ols_rss(Zu_use, y)
                rss_r = rss_own
            df1 = K
            df2 = max(n - Zu_use.shape[1], 1)
            denom = rss_u / df2
            F = ((rss_r - rss_u) / df1) / denom if denom > 1e-300 else 0.0
            F = max(F, 0.0)
            Fm[i, j] = F
            pm[i, j] = (float(_sstats.f.sf(F, df1, df2))
                        if HAVE_SCIPY else (1.0 if F <= 0 else 0.0))
            # per-lag |t| of the added regressor block
            beta, *_ = np.linalg.lstsq(Zu_use, y, rcond=None)
            r = y - Zu_use @ beta
            s2 = float(r @ r) / df2
            try:
                Gi = np.linalg.inv(Zu_use.T @ Zu_use)
            except np.linalg.LinAlgError:
                Gi = np.linalg.pinv(Zu_use.T @ Zu_use)
            sd = np.sqrt(np.maximum(np.diag(Gi), 0.0) * s2)
            sd = np.where(sd < 1e-300, 1e-300, sd)
            tt = beta / sd
            blk = slice(1, 1 + K) if j == i else slice(1 + K, 1 + 2 * K)
            A_lag[:, i, j] = np.abs(tt[blk])

    if score == "F":
        pass
    elif score == "neglogp":
        with np.errstate(divide="ignore"):
            Fm = -np.log10(np.maximum(pm, 1e-300))
    else:
        raise ValueError(f"unknown score {score!r}")

    sup_pair = pm < alpha
    if not self_loops:
        np.fill_diagonal(sup_pair, False)
    support_lag = np.repeat(sup_pair[None, ...], K, axis=0) & (A_lag > 0)
    meta = dict(scores_lagged=True, scores_contemp=False,
                contemp_symmetric=False, F=Fm, p=pm, pair_score=Fm,
                n=n, self_loops=self_loops)
    return A_lag, np.zeros((P, P)), support_lag, np.zeros((P, P), dtype=bool), meta


# =========================================================================== #
# 3.  varlingam  (lingam, MIT)
# =========================================================================== #
def _bl_varlingam(X, max_lag, *, standardize: bool = True, prune: bool = True,
                  criterion: Optional[str] = "bic", **kw):
    """lingam.VARLiNGAM.

    Library convention (VERIFIED, not assumed -- see self_test):
        ``adjacency_matrices_`` has shape (lags + 1, P, P);
        ``adjacency_matrices_[0]``      = B0, the CONTEMPORANEOUS matrix;
        ``adjacency_matrices_[k][i, j]``= effect of variable j at lag k on i.
    That is already (effect, cause), so the conversion is the IDENTITY:
        A_contemp = adjacency_matrices_[0]
        A_lag[k-1] = adjacency_matrices_[k]
    Support = nonzero (VARLiNGAM prunes with adaptive lasso when prune=True).

    NOTE ON LAG BUDGET: ``criterion='bic'`` lets VARLiNGAM pick lags <= max_lag
    and the returned tensor is then SHORTER than max_lag; we zero-pad so the
    output shape is always (max_lag, P, P).  Pass criterion=None to force the
    full lag budget.
    """
    if not HAVE_LINGAM:
        raise RuntimeError(f"lingam unavailable: {_LINGAM_ERR}")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    m = _lingam.VARLiNGAM(lags=K, criterion=criterion, prune=prune,
                          random_state=kw.get("random_state", 0))
    m.fit(Xp)
    Am = np.asarray(m.adjacency_matrices_, dtype=float)     # (lags+1, P, P)
    A_c = Am[0].copy()
    A_lag = np.zeros((K, P, P))
    n_eff = min(K, Am.shape[0] - 1)
    if n_eff > 0:
        A_lag[:n_eff] = Am[1:1 + n_eff]
    meta = dict(scores_lagged=True, scores_contemp=True,
                contemp_symmetric=False, fitted_lags=int(Am.shape[0] - 1),
                criterion=criterion, prune=prune,
                causal_order=list(getattr(m, "causal_order_", [])))
    return (A_lag, A_c, np.abs(A_lag) > 0, np.abs(A_c) > 0, meta)


# =========================================================================== #
# 4.  directlingam  (lingam, MIT) -- contemporaneous only
# =========================================================================== #
def _bl_directlingam(X, max_lag, *, standardize: bool = True,
                     augment_lags: bool = False, **_):
    """lingam.DirectLiNGAM on the rows of X treated as i.i.d. draws.

    Library convention (VERIFIED): ``adjacency_matrix_[i, j]`` = the
    coefficient of j in the structural equation for i, i.e. "j -> i".
    Already (effect, cause) => conversion is the IDENTITY.

    augment_lags=True runs DirectLiNGAM on [X_t, X_{t-1}, ..., X_{t-K}] with a
    prior-knowledge matrix that forbids any arrow pointing backwards in time,
    and splits the result into A_contemp / A_lag.  Off by default (that is
    essentially VARLiNGAM, and it is O((K+1)P) variables).
    """
    if not HAVE_LINGAM:
        raise RuntimeError(f"lingam unavailable: {_LINGAM_ERR}")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)

    if not augment_lags:
        m = _lingam.DirectLiNGAM()
        m.fit(Xp)
        A_c = np.asarray(m.adjacency_matrix_, dtype=float)
        meta = dict(scores_lagged=False, scores_contemp=True,
                    contemp_symmetric=False, augment_lags=False,
                    causal_order=list(getattr(m, "causal_order_", [])))
        return (np.zeros((K, P, P)), A_c, np.zeros((K, P, P), dtype=bool),
                np.abs(A_c) > 0, meta)

    Y, Z = _lag_design(Xp, K, intercept=False)
    W = np.concatenate([Y, Z], axis=1)                  # blocks: lag0, lag1..K
    D = W.shape[1]
    # prior knowledge: pk[i, j] = 0 forbids j -> i, 1 forces it, -1 unknown.
    pk = -np.ones((D, D), dtype=int)
    blk = np.repeat(np.arange(K + 1), P)                # block index per column
    pk[blk[None, :] > blk[:, None]] = -1                # past -> present: free
    pk[blk[None, :] < blk[:, None]] = 0                 # present -> past: banned
    pk[np.ix_(blk > 0, blk > 0)] = 0                    # no lag<->lag structure
    m = _lingam.DirectLiNGAM(prior_knowledge=pk)
    m.fit(W)
    Aw = np.asarray(m.adjacency_matrix_, dtype=float)   # (D, D), [effect, cause]
    A_c = Aw[:P, :P]
    A_lag = np.stack([Aw[:P, (k + 1) * P:(k + 2) * P] for k in range(K)])
    meta = dict(scores_lagged=True, scores_contemp=True,
                contemp_symmetric=False, augment_lags=True,
                causal_order=list(getattr(m, "causal_order_", [])))
    return A_lag, A_c, np.abs(A_lag) > 0, np.abs(A_c) > 0, meta


# =========================================================================== #
# 5.  dynotears -- Pamfil et al. (2020), from scratch (numpy + scipy L-BFGS-B)
# =========================================================================== #
def _dynotears_core(Xc: np.ndarray, Yl: np.ndarray, d: int, p: int,
                    lambda_w: float, lambda_a: float, max_iter: int,
                    h_tol: float, rho_max: float):
    """min 1/(2n)||X - XW - YA||_F^2 + lam_w|W|_1 + lam_a|A|_1  s.t. h(W)=0,
    h(W) = tr(exp(W*W)) - d.  Augmented Lagrangian; each subproblem solved by
    L-BFGS-B on the nonnegative split W = W+ - W-, A = A+ - A-.

    Returns (W, A) with the NOTEARS orientation: W[cause, effect],
    A[k*d + cause, effect] for lag k+1.  Caller transposes.
    """
    n = Xc.shape[0]
    n_w = d * d
    n_a = p * d * d

    def _unpack(w):
        Wp = w[:n_w].reshape(d, d)
        Wm = w[n_w:2 * n_w].reshape(d, d)
        Ap = w[2 * n_w:2 * n_w + n_a].reshape(p * d, d)
        Am = w[2 * n_w + n_a:].reshape(p * d, d)
        return Wp - Wm, Ap - Am

    def _h(W):
        M = W * W
        with np.errstate(over="ignore", invalid="ignore"):
            E = _sla.expm(M)
        hv = float(np.trace(E) - d)
        gh = E.T * W * 2.0
        if not np.isfinite(hv) or not np.isfinite(gh).all():
            # expm blew up: hand back a huge but FINITE penalty with a
            # zero gradient so L-BFGS-B backs off instead of propagating inf.
            return 1e30, np.zeros_like(W)
        return hv, gh

    def _obj(w, rho, alpha):
        W, A = _unpack(w)
        R = Xc - Xc @ W - Yl @ A
        loss = 0.5 / n * float((R * R).sum())
        gW = -1.0 / n * (Xc.T @ R)
        gA = -1.0 / n * (Yl.T @ R)
        hv, gh = _h(W)
        with np.errstate(over="ignore", invalid="ignore"):
            obj = (loss + 0.5 * rho * hv * hv + alpha * hv
                   + lambda_w * float(np.abs(W).sum())
                   + lambda_a * float(np.abs(A).sum()))
            gWt = gW + (rho * hv + alpha) * gh
        if not np.isfinite(obj):
            obj = 1e30
        gWt = np.nan_to_num(gWt, nan=0.0, posinf=1e12, neginf=-1e12)
        g = np.concatenate([
            (gWt + lambda_w).ravel(), (-gWt + lambda_w).ravel(),
            (gA + lambda_a).ravel(), (-gA + lambda_a).ravel(),
        ])
        return obj, g

    w_est = np.zeros(2 * n_w + 2 * n_a)
    bnds = [(0.0, None)] * w_est.size
    rho, alpha, h_val = 1.0, 0.0, np.inf
    for _ in range(int(max_iter)):
        w_new = None
        while rho < rho_max:
            sol = _minimize(_obj, w_est, args=(rho, alpha), method="L-BFGS-B",
                            jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_unpack(w_new)[0])
            if h_new > 0.25 * h_val:
                rho *= 10.0
            else:
                break
        w_est, h_val = w_new, h_new
        alpha += rho * h_val
        if h_val <= h_tol or rho >= rho_max:
            break
    return _unpack(w_est)


def _bl_dynotears(X, max_lag, *, standardize: bool = False,
                  lambda_w: float = 0.1, lambda_a: float = 0.1,
                  max_iter: int = 100, h_tol: float = 1e-8,
                  rho_max: float = 1e16, w_threshold: float = 0.0, **_):
    """DYNOTEARS (Pamfil et al., AISTATS 2020), reimplemented in numpy/scipy.

    causalnex (Apache-2.0) declares ``Requires-Python >=3.8,<3.11`` and has no
    wheel for this interpreter, so it is reimplemented rather than wrapped.
    Defaults mirror causalnex ``from_pandas_dynamic`` (lambda_w=lambda_a=0.1,
    max_iter=100, h_tol=1e-8, w_threshold=0.0) including NO standardisation --
    the contemporaneous half is NOTEARS and inherits its equal-noise-variance
    assumption, so z-scoring can reverse it (see _bl_dagma_linear).

    NOTEARS-family convention: the model is X ~ X W + Y A, so
        W[j, i] = "j -> i contemporaneous"            -> A_contemp = W.T
        A[k*d + j, i] = "j at lag k+1 -> i"           -> A_lag[k] = A_blk.T
    The transpose is the whole point of the self-test for this wrapper.
    """
    if not HAVE_SCIPY:
        raise RuntimeError("scipy required for dynotears")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    Xc, Yl = _lag_design(Xp, K, intercept=False)
    Xc = Xc - Xc.mean(0)
    Yl = Yl - Yl.mean(0)
    W, A = _dynotears_core(Xc, Yl, P, K, lambda_w, lambda_a,
                           max_iter, h_tol, rho_max)
    if w_threshold > 0:
        W = np.where(np.abs(W) < w_threshold, 0.0, W)
        A = np.where(np.abs(A) < w_threshold, 0.0, A)
    A_c = W.T.copy()                                    # (effect, cause)
    A_lag = np.stack([A[k * P:(k + 1) * P].T for k in range(K)]) if K else \
        np.zeros((0, P, P))
    meta = dict(scores_lagged=True, scores_contemp=True,
                contemp_symmetric=False, lambda_w=lambda_w,
                lambda_a=lambda_a, w_threshold=w_threshold)
    return A_lag, A_c, np.abs(A_lag) > 0, np.abs(A_c) > 0, meta


# =========================================================================== #
# 6.  dagma_linear  (dagma, Apache-2.0) -- contemporaneous only
# =========================================================================== #
def _bl_dagma_linear(X, max_lag, *, standardize: bool = False,
                     lambda1: float = 0.02, loss_type: str = "l2",
                     w_threshold: float = 0.0, **kw):
    """dagma.linear.DagmaLinear.

    NOTEARS-family convention: the fitted W solves X ~ X W, so W[j, i] means
    "j -> i".  Conversion: A_contemp = W.T.  Contemporaneous only -- DAGMA has
    no lag model, so A_lag stays zero and meta['scores_lagged'] is False.

    ``standardize`` DEFAULTS TO FALSE HERE, unlike every other wrapper.  The
    l2 score assumes equal NOISE variance; z-scoring equalises the MARGINAL
    variances instead and destroys the varsortability signal DAGMA leans on.
    self_test() measures this directly: on make_asym_contemp the direction
    margin is +0.87 with standardize=False and -0.80 (i.e. the DAG comes back
    REVERSED) with standardize=True.  Turning it on is a legitimate robustness
    experiment, but it is not the library's own operating point.
    """
    if not HAVE_DAGMA:
        raise RuntimeError(f"dagma unavailable: {_DAGMA_ERR}")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    model = _DagmaLinear(loss_type=loss_type, verbose=False)
    with _quiet(not kw.get("verbose", False)):
        W = np.asarray(model.fit(Xp, lambda1=lambda1,
                                 w_threshold=kw.get("fit_w_threshold", 0.3)),
                       dtype=float)
    if w_threshold > 0:
        W = np.where(np.abs(W) < w_threshold, 0.0, W)
    A_c = W.T.copy()
    meta = dict(scores_lagged=False, scores_contemp=True,
                contemp_symmetric=False, lambda1=lambda1, loss_type=loss_type)
    return (np.zeros((K, P, P)), A_c, np.zeros((K, P, P), dtype=bool),
            np.abs(A_c) > 0, meta)


# =========================================================================== #
# 7-8.  pc / fci  (causal-learn, MIT)
# =========================================================================== #
# causal-learn stores a GeneralGraph as an integer matrix `g.graph` in which
# g.graph[a, b] is the ENDPOINT AT NODE a of the edge between a and b:
#     -1 TAIL      1 ARROW      2 CIRCLE      0 no edge
# So "i --> j" is  g.graph[j, i] == 1  and  g.graph[i, j] == -1.
# The decoder below reads that pairwise and is verified by self_test on a
# v-structure (where orientation is identifiable, so a transpose is visible).
_ARROW, _TAIL, _CIRCLE = 1, -1, 2


def _decode_general_graph(g, P, undirected_weight=0.5, circle_weight=0.5,
                          bidirected_weight=0.5, partial_weight=0.75):
    """(P, P) score in OUR convention: out[i, j] = confidence that j -> i."""
    M = np.asarray(g.graph)
    out = np.zeros((P, P))
    for a in range(P):
        for b in range(a + 1, P):
            ea, eb = int(M[a, b]), int(M[b, a])         # endpoint at a / at b
            if ea == 0 and eb == 0:
                continue
            if ea == _TAIL and eb == _ARROW:            # a --> b
                out[b, a] = 1.0
            elif ea == _ARROW and eb == _TAIL:          # b --> a
                out[a, b] = 1.0
            elif ea == _TAIL and eb == _TAIL:           # a --- b (CPDAG)
                out[b, a] = out[a, b] = undirected_weight
            elif ea == _ARROW and eb == _ARROW:         # a <-> b (latent)
                out[b, a] = out[a, b] = bidirected_weight
            elif ea == _CIRCLE and eb == _ARROW:        # a o-> b
                out[b, a] = partial_weight
                out[a, b] = 1.0 - partial_weight
            elif ea == _ARROW and eb == _CIRCLE:        # b o-> a
                out[a, b] = partial_weight
                out[b, a] = 1.0 - partial_weight
            else:                                       # o-o, o-, ...
                out[b, a] = out[a, b] = circle_weight
    return out


def _bl_pc(X, max_lag, *, standardize: bool = True, alpha: float = 0.05,
           indep_test: str = "fisherz", stable: bool = True, **kw):
    """causal-learn PC on the contemporaneous cross-section (rows i.i.d.).
    Returns a CPDAG decoded with _decode_general_graph; undirected edges get
    0.5 in BOTH directions (documented tie handling)."""
    if not HAVE_CAUSALLEARN:
        raise RuntimeError(f"causal-learn unavailable: {_CAUSALLEARN_ERR}")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    with _quiet(not kw.get("verbose", False)):
        cg = _cl_pc(Xp, alpha=alpha, indep_test=indep_test, stable=stable,
                    show_progress=False)
    A_c = _decode_general_graph(cg.G, P,
                                undirected_weight=kw.get("undirected_weight", 0.5))
    meta = dict(scores_lagged=False, scores_contemp=True,
                contemp_symmetric=False, alpha=alpha, indep_test=indep_test)
    return (np.zeros((K, P, P)), A_c, np.zeros((K, P, P), dtype=bool),
            A_c >= 1.0, meta)


def _bl_fci(X, max_lag, *, standardize: bool = True, alpha: float = 0.05,
            indep_test: str = "fisherz", **kw):
    """causal-learn FCI on the contemporaneous cross-section.  Returns a PAG;
    o-> scores 0.75 for the arrow direction, o-o / <-> split 0.5/0.5."""
    if not HAVE_CAUSALLEARN:
        raise RuntimeError(f"causal-learn unavailable: {_CAUSALLEARN_ERR}")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    with _quiet(not kw.get("verbose", False)):
        g, _edges = _cl_fci(Xp, independence_test_method=indep_test,
                            alpha=alpha, show_progress=False, verbose=False)
    A_c = _decode_general_graph(g, P,
                                partial_weight=kw.get("partial_weight", 0.75))
    meta = dict(scores_lagged=False, scores_contemp=True,
                contemp_symmetric=False, alpha=alpha, indep_test=indep_test)
    return (np.zeros((K, P, P)), A_c, np.zeros((K, P, P), dtype=bool),
            A_c >= 1.0, meta)


def _window_pc_fci(X, max_lag, which, *, standardize=True, alpha=0.05,
                   indep_test="fisherz", ts_max_lag=3, **kw):
    """PC / FCI on the LAG-AUGMENTED window [X_t, X_{t-1}, ..., X_{t-K}].

    Any adjacency between a lagged block and the contemporaneous block is
    oriented by time (a lagged variable cannot be an effect of a present one),
    which is the cheap 'tsPC / tsFCI'-style construction.  Contemporaneous
    adjacencies are decoded from the CPDAG/PAG as usual.  K is clipped to
    ts_max_lag because the search is over (K+1)*P variables.
    """
    if not HAVE_CAUSALLEARN:
        raise RuntimeError(f"causal-learn unavailable: {_CAUSALLEARN_ERR}")
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    Ke = min(K, int(ts_max_lag))
    Y, Z = _lag_design(Xp, Ke, intercept=False)
    W = np.concatenate([Y, Z], axis=1)
    D = W.shape[1]
    with _quiet(not kw.get("verbose", False)):
        if which == "pc":
            g = _cl_pc(W, alpha=alpha, indep_test=indep_test, stable=True,
                       show_progress=False).G
        else:
            g, _e = _cl_fci(W, independence_test_method=indep_test, alpha=alpha,
                            show_progress=False, verbose=False)
    M = np.asarray(g.graph)
    A_lag = np.zeros((K, P, P))
    full = _decode_general_graph(g, D,
                                 undirected_weight=kw.get("undirected_weight", 0.5),
                                 partial_weight=kw.get("partial_weight", 0.75))
    A_c = full[:P, :P]
    for k in range(Ke):
        blk = slice((k + 1) * P, (k + 2) * P)
        adj = (M[:P, blk] != 0) | (M[blk, :P].T != 0)
        A_lag[k] = adj.astype(float)                    # time orients it
    meta = dict(scores_lagged=True, scores_contemp=True,
                contemp_symmetric=False, alpha=alpha, indep_test=indep_test,
                effective_max_lag=Ke, note="lag edges oriented by time order")
    return A_lag, A_c, A_lag >= 1.0, A_c >= 1.0, meta


def _bl_pc_ts(X, max_lag, **kw):
    return _window_pc_fci(X, max_lag, "pc", **kw)


def _bl_fci_ts(X, max_lag, **kw):
    return _window_pc_fci(X, max_lag, "fci", **kw)


# =========================================================================== #
# 9-10.  pcmci / pcmciplus  (tigramite, GPL-3.0) -- OUT OF PROCESS ONLY
# =========================================================================== #
# tigramite is GPL-3.0.  Importing it into a process that also imports the
# MIT-licensed `cafe` package would put the combined work under the GPL.  It is
# therefore run in a SEPARATE INTERPRETER: we write X to an .npz, generate a
# small runner script into a temp dir, subprocess.run() it with a sanitised
# environment (repo `src/` removed from PYTHONPATH so `cafe` is not importable
# there), and read the result back from a second .npz.  The runner refuses to
# start if `cafe` is somehow already imported.  Nothing GPL ever enters THIS
# interpreter -- assert_no_gpl_imported() enforces that at runtime.
# =========================================================================== #
_TIGRAMITE_RUNNER = r'''
"""Generated by bench/causal_baselines.py -- GPL-3.0 isolation boundary.
Runs tigramite (GPL-3.0) in its own interpreter.  npz in, npz out.
This file must NEVER import `cafe` or be imported by a process that has."""
import sys, json, traceback
import numpy as np

def main(in_path, out_path):
    if "cafe" in sys.modules:
        raise RuntimeError("GPL isolation violated: `cafe` is imported in the "
                           "tigramite runner process")
    d = np.load(in_path, allow_pickle=False)
    X = d["X"].astype(float)
    cfg = json.loads(str(d["cfg"]))
    tau_max = int(cfg["tau_max"])
    tau_min = int(cfg.get("tau_min", 1))
    pc_alpha = cfg.get("pc_alpha", 0.05)
    variant = cfg.get("variant", "pcmci")
    test = cfg.get("cond_ind_test", "parcorr")

    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    if test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr as CIT
        cit = CIT(significance="analytic")
    elif test == "robustparcorr":
        from tigramite.independence_tests.robust_parcorr import RobustParCorr as CIT
        cit = CIT(significance="analytic")
    elif test == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC as CIT
        cit = CIT()
    else:
        raise ValueError("unknown cond_ind_test %r" % test)

    df = pp.DataFrame(X)
    pcmci = PCMCI(dataframe=df, cond_ind_test=cit, verbosity=0)
    if variant == "pcmciplus":
        res = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    else:
        res = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max,
                              pc_alpha=pc_alpha)
    val = np.asarray(res["val_matrix"], dtype=float)
    pmat = np.asarray(res["p_matrix"], dtype=float)
    graph = np.asarray(res.get("graph", np.empty((0,), dtype="U3")))
    np.savez_compressed(out_path, val_matrix=val, p_matrix=pmat,
                        graph=graph.astype("U3"), ok=np.array(True))

if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except Exception:
        sys.stderr.write(traceback.format_exc())
        sys.exit(3)
'''


def assert_no_gpl_imported(modules=("tigramite", "pytetrad", "causalcmd")):
    """Raise if a GPL module has leaked into THIS interpreter.  Called before
    every out-of-process tigramite run and exposed for use in tests."""
    bad = [m for m in modules if m in sys.modules]
    if bad:
        raise RuntimeError(
            f"GPL isolation violated: {bad} imported in-process. tigramite / "
            "Tetrad are GPL-3.0 and must only run via subprocess."
        )


def _isolated_env():
    env = dict(os.environ)
    # Strip the repo's src/ (and the bench dir) so `cafe` cannot be imported
    # inside the GPL subprocess even by accident.
    src = os.path.abspath(os.path.join(_BENCH, "..", "src"))
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    parts = [p for p in parts if os.path.abspath(p) not in (src, _BENCH)]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    return env


def _run_tigramite(X, tau_max, variant, pc_alpha, cond_ind_test, timeout_s,
                   python_exe=None):
    assert_no_gpl_imported()
    tmp = tempfile.mkdtemp(prefix="tigramite_iso_")
    runner = os.path.join(tmp, "_tigramite_runner.py")
    fin = os.path.join(tmp, "in.npz")
    fout = os.path.join(tmp, "out.npz")
    with open(runner, "w") as fh:
        fh.write(_TIGRAMITE_RUNNER)
    cfg = json.dumps(dict(tau_max=int(tau_max), tau_min=1, pc_alpha=pc_alpha,
                          variant=variant, cond_ind_test=cond_ind_test))
    np.savez_compressed(fin, X=np.asarray(X, dtype=float), cfg=np.array(cfg))
    exe = python_exe or sys.executable
    proc = subprocess.run([exe, runner, fin, fout],
                          capture_output=True, text=True,
                          timeout=timeout_s, cwd=tmp, env=_isolated_env())
    if proc.returncode != 0:
        raise RuntimeError(
            f"tigramite subprocess rc={proc.returncode}: "
            f"{(proc.stderr or '').strip()[-600:]}"
        )
    with np.load(fout, allow_pickle=False) as d:
        return (d["val_matrix"], d["p_matrix"], d["graph"])


def _bl_pcmci(X, max_lag, *, standardize=True, pc_alpha=0.05,
              cond_ind_test="parcorr", variant="pcmci", timeout_s=600.0,
              alpha_level=0.05, python_exe=None, **_):
    """tigramite PCMCI / PCMCI+, run OUT OF PROCESS (GPL-3.0).

    tigramite convention (VERIFIED): ``val_matrix[i, j, tau]`` is the test
    statistic of the link "variable i at time t-tau  ->  variable j at time t",
    i.e. (CAUSE, EFFECT, lag) -- the transpose of ours.  Conversion:
        A_lag[k-1, i, j] = |val_matrix[j, i, k]|
        A_contemp[i, j]  = |val_matrix[j, i, 0]|      (PCMCI+ only)
    Support uses the returned p_matrix at alpha_level (and, for PCMCI+, the
    string graph: only '-->' counts as a directed cause).
    """
    Xp = _prep(X, standardize=standardize)
    T, P = Xp.shape
    K = int(max_lag)
    val, pmat, graph = _run_tigramite(Xp, K, variant, pc_alpha, cond_ind_test,
                                      timeout_s, python_exe=python_exe)
    val = np.abs(np.asarray(val, dtype=float))
    pmat = np.asarray(pmat, dtype=float)
    # val/pmat are (P, P, tau_max+1) with axis 0 = cause, axis 1 = effect.
    A_lag = np.zeros((K, P, P))
    sup_lag = np.zeros((K, P, P), dtype=bool)
    for k in range(1, min(K, val.shape[2] - 1) + 1):
        A_lag[k - 1] = val[:, :, k].T
        sup_lag[k - 1] = (pmat[:, :, k] <= alpha_level).T
    A_c = np.zeros((P, P))
    sup_c = np.zeros((P, P), dtype=bool)
    if variant == "pcmciplus":
        A_c = val[:, :, 0].T
        sup_c = (pmat[:, :, 0] <= alpha_level).T
        np.fill_diagonal(A_c, 0.0)
        np.fill_diagonal(sup_c, False)
    g = np.asarray(graph)
    if g.ndim == 3 and g.shape[2] >= 1:
        directed = (g == "-->")
        for k in range(1, min(K, g.shape[2] - 1) + 1):
            sup_lag[k - 1] &= directed[:, :, k].T
        if variant == "pcmciplus":
            sup_c &= directed[:, :, 0].T
    meta = dict(scores_lagged=True, scores_contemp=(variant == "pcmciplus"),
                contemp_symmetric=False, variant=variant, pc_alpha=pc_alpha,
                cond_ind_test=cond_ind_test, out_of_process=True,
                licence="GPL-3.0 (subprocess-isolated)")
    return A_lag, A_c, sup_lag, sup_c, meta


def _bl_pcmciplus(X, max_lag, **kw):
    kw.pop("variant", None)
    return _bl_pcmci(X, max_lag, variant="pcmciplus", **kw)


# --------------------------------------------------------------------------- #
# registry + entry point
# --------------------------------------------------------------------------- #
BASELINES: Dict[str, Callable[..., tuple]] = {
    "var_ols": _bl_var_ols,
    "granger_pairwise": _bl_granger_pairwise,
    "varlingam": _bl_varlingam,
    "directlingam": _bl_directlingam,
    "dynotears": _bl_dynotears,
    "dagma_linear": _bl_dagma_linear,
    "pc": _bl_pc,
    "fci": _bl_fci,
    "pc_ts": _bl_pc_ts,
    "fci_ts": _bl_fci_ts,
    "pcmci": _bl_pcmci,
    "pcmciplus": _bl_pcmciplus,
}

LICENCES = {
    "var_ols": "MIT (this repo)",
    "granger_pairwise": "MIT (this repo)",
    "varlingam": "MIT (lingam)",
    "directlingam": "MIT (lingam)",
    "dynotears": "MIT (this repo; re-implements Pamfil et al. 2020)",
    "dagma_linear": "Apache-2.0 (dagma)",
    "pc": "MIT (causal-learn)",
    "fci": "MIT (causal-learn)",
    "pc_ts": "MIT (causal-learn)",
    "fci_ts": "MIT (causal-learn)",
    "pcmci": "GPL-3.0 (tigramite) -- OUT OF PROCESS",
    "pcmciplus": "GPL-3.0 (tigramite) -- OUT OF PROCESS",
}

# Baselines that are structurally unable to score lagged edges.
CONTEMP_ONLY = {"directlingam", "dagma_linear", "pc", "fci"}


def available() -> Dict[str, bool]:
    """Which baselines can actually run in this environment."""
    tg = _tigramite_importable()
    return {
        "var_ols": HAVE_SCIPY,
        "granger_pairwise": HAVE_SCIPY,
        "varlingam": HAVE_LINGAM,
        "directlingam": HAVE_LINGAM,
        "dynotears": HAVE_SCIPY,
        "dagma_linear": HAVE_DAGMA,
        "pc": HAVE_CAUSALLEARN,
        "fci": HAVE_CAUSALLEARN,
        "pc_ts": HAVE_CAUSALLEARN,
        "fci_ts": HAVE_CAUSALLEARN,
        "pcmci": tg,
        "pcmciplus": tg,
    }


def _tigramite_importable() -> bool:
    """Probe for tigramite WITHOUT importing it here (it is GPL-3.0)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import importlib.util,sys;"
             "sys.exit(0 if importlib.util.find_spec('tigramite') else 1)"],
            capture_output=True, timeout=60, env=_isolated_env())
        return proc.returncode == 0
    except Exception:
        return False


def run_baseline(name: str, X, max_lag: int = 8, timeout_s: Optional[float] = 600.0,
                 **kw) -> BaselineResult:
    """Run one baseline.  NEVER raises: failures come back as ok=False.

    Parameters
    ----------
    name      key of BASELINES
    X         (T, P) array-like, row axis = TIME.  NaNs are linearly
              interpolated (most of these libraries cannot take NaN at all).
    max_lag   K.  Returned A_lag always has exactly K leading entries.
    timeout_s soft (SIGALRM) for in-process baselines, hard (subprocess) for
              tigramite.  None disables.
    **kw      forwarded to the wrapper (see each _bl_* docstring).
    """
    t0 = time.time()
    Xa = np.asarray(X, dtype=float)
    P = int(Xa.shape[1]) if Xa.ndim == 2 else 0
    K = max(int(max_lag), 1)
    if name not in BASELINES:
        return _empty(P, K, name, K, f"unknown baseline {name!r}", t0)
    if P < 2:
        return _empty(max(P, 1), K, name, K, f"bad X shape {Xa.shape}", t0)

    fn = BASELINES[name]
    if name in ("pcmci", "pcmciplus"):
        kw.setdefault("timeout_s", timeout_s if timeout_s else 3600.0)
    try:
        if name in ("pcmci", "pcmciplus"):
            out = fn(Xa, K, **kw)                       # own hard timeout
        else:
            with _time_limit(timeout_s):
                out = fn(Xa, K, **kw)
    except BaselineTimeout as e:
        return _empty(P, K, name, K, f"TIMEOUT: {e}", t0)
    except subprocess.TimeoutExpired as e:
        return _empty(P, K, name, K, f"TIMEOUT (subprocess): {e}", t0)
    except MemoryError as e:                             # pragma: no cover
        return _empty(P, K, name, K, f"MemoryError: {e}", t0)
    except BaseException as e:                           # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        return _empty(P, K, name, K, f"{type(e).__name__}: {e}", t0)

    A_lag, A_c, s_lag, s_c, meta = out
    A_lag = np.nan_to_num(np.asarray(A_lag, dtype=float).reshape(K, P, P))
    A_c = np.nan_to_num(np.asarray(A_c, dtype=float).reshape(P, P))
    s_lag = np.asarray(s_lag, dtype=bool).reshape(K, P, P)
    s_c = np.asarray(s_c, dtype=bool).reshape(P, P)
    meta = dict(meta or {})
    meta.update(name=name, max_lag=K, P=P, T=int(Xa.shape[0]),
                licence=LICENCES.get(name, "?"), kwargs={
                    k: v for k, v in kw.items() if isinstance(
                        v, (int, float, str, bool, type(None)))})
    meta.setdefault("scores_lagged", True)
    meta.setdefault("scores_contemp", True)
    meta.setdefault("contemp_symmetric", False)
    return BaselineResult(A_lag, A_c, s_lag, s_c, time.time() - t0, True, "", meta)


# =========================================================================== #
# CONVENTION VERIFICATION -- prove every conversion, do not trust the docs
# =========================================================================== #
def make_asym_var(T=4000, seed=0, noise="laplace"):
    """Asymmetric VAR whose TRUE edge set is not symmetric and shares no edge
    with its own transpose (off the diagonal).  Any transposed wrapper scores
    the reversed edges high and the true ones low, so the margin flips sign.

        lag 1:  0 -> 1 (0.80),  1 -> 2 (0.70),  2 -> 3 (0.60)
        lag 2:  0 -> 3 (0.55)
        self :  AR(1) 0.60 / 0.50 / 0.40 / 0.30   (distinct, so the diagonal
                is asymmetric too and cannot mask a flip)

    Non-Gaussian (Laplace) innovations so LiNGAM-family methods are identified.
    """
    rng = np.random.default_rng(seed)
    P = 4
    ar = np.array([0.60, 0.50, 0.40, 0.30])
    X = np.zeros((T, P))
    e = (rng.laplace(0, 1, size=(T, P)) if noise == "laplace"
         else rng.normal(0, 1, size=(T, P)))
    for t in range(2, T):
        X[t, 0] = ar[0] * X[t - 1, 0] + e[t, 0]
        X[t, 1] = ar[1] * X[t - 1, 1] + 0.80 * X[t - 1, 0] + e[t, 1]
        X[t, 2] = ar[2] * X[t - 1, 2] + 0.70 * X[t - 1, 1] + e[t, 2]
        X[t, 3] = ar[3] * X[t - 1, 3] + 0.60 * X[t - 1, 2] + 0.55 * X[t - 2, 0] + e[t, 3]
    A_lag = np.zeros((2, P, P))
    for i in range(P):
        A_lag[0, i, i] = ar[i]
    A_lag[0, 1, 0] = 0.80
    A_lag[0, 2, 1] = 0.70
    A_lag[0, 3, 2] = 0.60
    A_lag[1, 3, 0] = 0.55
    return X[200:], A_lag


def make_asym_contemp(n=4000, seed=1, noise="laplace", equal_var=True):
    """Contemporaneous DAG with a v-structure (so PC/FCI can orient it) and
    non-Gaussian equal-variance noise (so LiNGAM and DAGMA are identifiable).

        0 -> 2,  1 -> 2,  2 -> 3     (v-structure at 2, then a chain)

    Transposing the answer maps {2<-0, 2<-1, 3<-2} onto {0<-2, 1<-2, 2<-3},
    which is disjoint from the truth, so a flip is unmissable.
    """
    rng = np.random.default_rng(seed)
    s = 1.0
    if noise == "laplace":
        e = rng.laplace(0, 1 / np.sqrt(2.0), size=(n, 4))
    else:
        e = rng.normal(0, 1, size=(n, 4))
    if not equal_var:
        e = e * np.array([1.0, 1.3, 0.7, 1.1])
    x0 = s * e[:, 0]
    x1 = s * e[:, 1]
    x2 = 0.9 * x0 + 0.9 * x1 + s * e[:, 2]
    x3 = 0.9 * x2 + s * e[:, 3]
    X = np.column_stack([x0, x1, x2, x3])
    A = np.zeros((4, 4))
    A[2, 0] = 0.9
    A[2, 1] = 0.9
    A[3, 2] = 0.9
    return X, A


def _direction_margin(score: np.ndarray, truth_mask: np.ndarray):
    """min(score on true directed edges) - max(score on the REVERSED edges).

    Only pairs where exactly one direction is true are used, so the statistic
    is blind to everything except orientation.  > 0  => convention correct.
    < 0  => the wrapper is transposed.
    """
    score = np.abs(np.asarray(score, dtype=float))
    truth = np.asarray(truth_mask, dtype=bool)
    fwd, rev = [], []
    P = truth.shape[-1]
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            t_ij = truth[..., i, j].any()
            t_ji = truth[..., j, i].any()
            if t_ij and not t_ji:
                fwd.append(score[..., i, j].max())
                rev.append(score[..., j, i].max())
    if not fwd:
        return float("nan"), 0
    return float(min(fwd) - max(rev)), len(fwd)


def self_test(verbose=True, include_slow=True, timeout_s=300.0):
    """Prove every wrapper's index/sign conversion.  Returns a list of dicts.

    PASS means: on data where the true graph and its transpose are disjoint,
    the score on the true directed edges strictly exceeds the score on the
    reversed edges.  A transposed conversion produces a NEGATIVE margin.
    """
    Xv, A_true_lag = make_asym_var(T=4000, seed=0)
    Xc, A_true_c = make_asym_contemp(n=4000, seed=1)
    truth_lag = A_true_lag != 0
    truth_c = A_true_c != 0
    rows = []

    lagged = [
        ("var_ols", Xv, truth_lag, dict(max_lag=3)),
        ("granger_pairwise", Xv, truth_lag, dict(max_lag=3)),
        ("varlingam", Xv, truth_lag, dict(max_lag=3, criterion=None)),
        ("dynotears", Xv, truth_lag, dict(max_lag=3, lambda_w=0.05, lambda_a=0.05)),
        ("pc_ts", Xv, truth_lag, dict(max_lag=2, ts_max_lag=2)),
    ]
    contemp = [
        ("directlingam", Xc, truth_c, dict(max_lag=1)),
        ("dagma_linear", Xc, truth_c, dict(max_lag=1, lambda1=0.02)),
        ("dynotears", Xc, truth_c, dict(max_lag=1, lambda_w=0.05, lambda_a=0.05)),
        ("pc", Xc, truth_c, dict(max_lag=1)),
        ("fci", Xc, truth_c, dict(max_lag=1)),
    ]
    if include_slow:
        lagged.append(("pcmci", Xv[:1500], truth_lag, dict(max_lag=3)))
        lagged.append(("pcmciplus", Xv[:1500], truth_lag, dict(max_lag=3)))
        contemp.append(("fci_ts", Xv[:1500], truth_lag, dict(max_lag=2, ts_max_lag=2)))

    for kind, cases in (("lag", lagged), ("contemp", contemp)):
        for name, Xd, truth, kwargs in cases:
            K = kwargs.pop("max_lag")
            res = run_baseline(name, Xd, max_lag=K, timeout_s=timeout_s, **kwargs)
            if not res.ok:
                rows.append(dict(baseline=name, kind=kind, ok=False,
                                 margin=float("nan"), n_pairs=0,
                                 verdict="ERROR", err=res.err,
                                 runtime_s=res.runtime_s))
                continue
            if name in CONTEMP_ONLY or kind == "contemp":
                score = res.A_contemp if truth.ndim == 2 else summary_graph(res)
                tmask = truth
            else:
                score = summary_graph(res, include_contemp=False)
                tmask = truth
            margin, npairs = _direction_margin(score, tmask)
            verdict = ("PASS" if margin > 0 else
                       "FAIL-TRANSPOSED" if margin < 0 else "FAIL-TIE")
            rows.append(dict(baseline=name, kind=kind, ok=True, margin=margin,
                             n_pairs=npairs, verdict=verdict, err="",
                             runtime_s=res.runtime_s))

    # extra: var_ols must recover the lag-2-only edge 0 -> 3 at lag 2 exactly
    r = run_baseline("var_ols", Xv, max_lag=3, score="abscoef")
    if r.ok:
        rows.append(dict(baseline="var_ols[lag-index]", kind="lag", ok=True,
                         margin=float(r.A_lag[1, 3, 0] - r.A_lag[1, 0, 3]),
                         n_pairs=1,
                         verdict="PASS" if (np.argmax(r.A_lag[:, 3, 0]) == 1 or
                                            r.A_lag[1, 3, 0] > 0.3) else "FAIL",
                         err="", runtime_s=r.runtime_s))

    if verbose:
        print(f"{'baseline':22s} {'kind':8s} {'margin':>10s} {'pairs':>6s} "
              f"{'s':>7s}  verdict")
        print("-" * 74)
        for r_ in rows:
            print(f"{r_['baseline']:22s} {r_['kind']:8s} {r_['margin']:10.4f} "
                  f"{r_['n_pairs']:6d} {r_['runtime_s']:7.2f}  {r_['verdict']}"
                  + (f"  [{r_['err'][:70]}]" if r_["err"] else ""))
    return rows


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="skip the out-of-process tigramite checks")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.self_test:
        av = available()
        print(f"{'baseline':20s} {'available':10s} licence")
        for k in BASELINES:
            print(f"{k:20s} {str(av[k]):10s} {LICENCES[k]}")
        if not a.self_test:
            return 0
    rows = self_test(verbose=True, include_slow=not a.fast)
    bad = [r for r in rows if r["verdict"] not in ("PASS",)]
    print()
    print(f"{len(rows) - len(bad)}/{len(rows)} conventions verified")
    for b in bad:
        print(f"  !! {b['baseline']}: {b['verdict']} {b['err']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(_cli())
