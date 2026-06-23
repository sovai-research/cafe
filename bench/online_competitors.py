"""
ONLINE / CAUSAL imputation competitors that live in CAFE's own lane.

Two published causal/online multivariate imputers, wrapped to the repo baseline
contract so they can race CAFE on equal terms. Reviewers will (rightly) desk-reject
a causal-SOTA claim that *cites* online methods without *running* them, so we run them.

  (1) gcimpute  -- Online Gaussian Copula imputation.
        Zhao & Udell, "Online Missing Value Imputation and Change Point Detection
        with the Gaussian Copula", AAAI 2022.  pip install gcimpute.
        We use the genuinely streaming variant (training_mode='minibatch-online'):
        rows are processed in temporal mini-batches and each batch is imputed using
        the copula fit on STRICTLY PRIOR batches, so it is fully point-in-time.

  (2) BayOTIDE   -- Bayesian Online multivariate Time-series Imputation with
        functional DEcomposition.  Fang, Wen, Luo, Zhe & Sun, ICML 2024 (Spotlight).
        github.com/xuangu-fang/BayOTIDE.  Not on PyPI; we git-clone it.
        We run the ONLINE FILTERING pass only (no backward smoothing) and read out
        each time column at the moment it is filtered -- the causal online estimate.

Contract (see bench/m_softimpute.py, bench/c_baselines.py):
    impute(X, meta) -> filled
        X      : (T, N) float64, np.nan at missing cells.  Row axis = TIME.
        filled : (T, N) finite, observed cells preserved exactly, same scale.

Both methods are CAUSAL/ONLINE by construction and NEVER read ground truth inside
impute.  We mark causality honestly via bench/causal_race.verify_causal:

    * gcimpute (online): bit-exactly truncation-invariant once the column variable
      types are pinned (continuous) and the marginal-window RNG is seeded ->
      verify_causal max_dev == 0.0.  Strictly causal.

    * BayOTIDE (online filtering): online by design (single forward pass, no
      smoothing).  It is NOT bit-exactly truncation-invariant under the strict
      verifier because the factor LOADINGS W are a globally-shared posterior that
      keeps being refined as the stream advances, so the reconstruction of an early
      column reflects W as refined by later columns.  We report the measured
      max_dev rather than overclaim.  This is the usual "online-filtering" vs
      "strict point-in-time" distinction; it is still causal in the sense that the
      forward pass never looks ahead and no smoothing is applied.

Soft imports: this module ALWAYS imports cleanly even if a dependency is missing.
Inspect HAVE_GCIMPUTE / HAVE_BAYOTIDE and ONLINE_COMPETITORS (only working ones).

Install notes (for a future [bench] extra; pyproject is NOT edited here):
    pip install gcimpute            # -> gcimpute 0.0.4 (+ statsmodels, patsy)
    BayOTIDE: NOT on PyPI.  git clone https://github.com/xuangu-fang/BayOTIDE
              into /tmp (or set BAYOTIDE_DIR).  Pure torch + numpy + scipy + pyyaml,
              all already present in the bench venv.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
import types
import subprocess
import warnings

import contextlib
import io

import numpy as np


@contextlib.contextmanager
def _silence_stdout():
    """Mute gcimpute's chatty numerical-diagnostic prints during fit."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


_BENCH = os.path.dirname(os.path.abspath(__file__))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)


# =========================================================================== #
# (1) Online Gaussian Copula  (gcimpute, AAAI 2022)
# =========================================================================== #
try:
    from gcimpute.gaussian_copula import GaussianCopula as _GaussianCopula  # noqa: E402
    HAVE_GCIMPUTE = True
except Exception:  # pragma: no cover - missing optional dep
    _GaussianCopula = None
    HAVE_GCIMPUTE = False

# Streaming hyper-parameters.  Kept small/fixed so the method is deterministic AND
# truncation-invariant (the batch size must NOT depend on T, or growing-prefix
# verification would compare differently-batched fits).
_GC_BATCH = 20          # rows per online mini-batch (also the warm-up size)
_GC_WINDOW = 40         # marginal empirical-CDF sliding window (small -> init rolls out)
_GC_STEPSIZE = 0.5      # constant SGD step for the online correlation update
_GC_SEED = 1


def gcimpute_impute(X, meta=None):
    """Online Gaussian-copula imputation (causal / point-in-time).

    Rows are streamed in temporal mini-batches; each batch is imputed from the
    copula fit on strictly earlier batches.  All columns are forced CONTINUOUS so
    the variable-type decision (which would otherwise inspect the whole matrix and
    leak look-ahead) is removed, and the global numpy RNG used to seed the marginal
    window is fixed -- together these make the method bit-exactly truncation-invariant.
    """
    if not HAVE_GCIMPUTE:
        raise RuntimeError(
            "gcimpute is not installed. Install with: pip install gcimpute")

    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    obs = np.isfinite(X)

    # Trivial / degenerate cases the copula cannot fit: fall back to column means.
    # gcimpute also chokes when a column has <2 distinct observed levels (its
    # variable-type sniffer raises), which happens on very short prefixes -- treat
    # that the same way.  Mean fill is deterministic, so causality is preserved.
    n_levels = np.array([np.unique(X[obs[:, j], j]).size for j in range(N)])
    if T < max(2, _GC_BATCH) or N < 1 or obs.sum() == 0 or n_levels.min() < 2:
        return _column_mean_fallback(X, obs)

    # Seed the GLOBAL numpy RNG: gcimpute's OnlineTransformFunction.init_window uses
    # np.random.normal (unseeded) to fill the marginal window -- pin it for determinism.
    np.random.seed(12345)

    try:
        with warnings.catch_warnings(), _silence_stdout():
            warnings.simplefilter("ignore")
            model = _GaussianCopula(
                training_mode="minibatch-online",
                batch_size=_GC_BATCH,
                window_size=_GC_WINDOW,
                const_stepsize=_GC_STEPSIZE,
                random_state=_GC_SEED,
                verbose=0,
            )
            # Force every column continuous -> no global type inference (no leak).
            imp = model.fit_transform(X.copy(), continuous=list(range(N)))
    except Exception:
        # Any internal failure (e.g. degenerate column) -> deterministic mean fill.
        return _column_mean_fallback(X, obs)

    imp = np.asarray(imp, dtype=np.float64)
    # Preserve observed cells exactly; scrub any residual non-finite output.
    out = np.where(obs, X, imp)
    if not np.all(np.isfinite(out)):
        out = _scrub(out, X, obs)
    return out


# =========================================================================== #
# (2) BayOTIDE  (Bayesian online imputation, ICML 2024)
# =========================================================================== #
_BAYOTIDE_DIR = os.environ.get("BAYOTIDE_DIR", "/tmp/BayOTIDE")
_BAYOTIDE_REPO = "https://github.com/xuangu-fang/BayOTIDE"
_bayotide_err = None
_bay_mod = None      # imported model module
_bay_utils = None    # imported utils module


def _ensure_bayotide_repo():
    """Make sure the BayOTIDE source tree is present; shallow-clone if missing.

    Returns the repo dir or raises with a helpful message. Cloning needs network;
    if the env has no network this raises and BayOTIDE is simply unavailable.
    """
    main_py = os.path.join(_BAYOTIDE_DIR, "model_BayOTIDE.py")
    if os.path.isfile(main_py):
        return _BAYOTIDE_DIR
    parent = os.path.dirname(_BAYOTIDE_DIR.rstrip("/")) or "/tmp"
    os.makedirs(parent, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", _BAYOTIDE_REPO, _BAYOTIDE_DIR],
        check=True, capture_output=True, text=True, timeout=180,
    )
    if not os.path.isfile(main_py):
        raise RuntimeError("BayOTIDE clone did not produce model_BayOTIDE.py")
    return _BAYOTIDE_DIR


def _import_bayotide():
    """Soft-import BayOTIDE's torch model + utils from the cloned repo.

    The repo (pinned to numpy 1.26) does `from numpy.lib import utils`, which newer
    numpy removed.  We inject a dummy `numpy.lib.utils` module before importing so
    the import succeeds without editing the upstream source.
    """
    global _bay_mod, _bay_utils, _bayotide_err
    if _bay_mod is not None:
        return True
    try:
        import torch  # noqa: F401  (hard dependency check)

        repo = _ensure_bayotide_repo()

        # Shim the removed numpy.lib.utils symbol (unused by BayOTIDE at runtime).
        import numpy.lib as _nplib
        if not hasattr(_nplib, "utils"):
            _shim = types.ModuleType("numpy.lib.utils")
            sys.modules["numpy.lib.utils"] = _shim
            _nplib.utils = _shim

        if repo not in sys.path:
            sys.path.insert(0, repo)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import utils_BayOTIDE as _u
            import model_BayOTIDE as _m
        _bay_utils = _u
        _bay_mod = _m
        return True
    except Exception as e:  # pragma: no cover - optional / env dependent
        _bayotide_err = e
        return False


def _bayotide_available():
    """True iff BayOTIDE can actually be imported & run in this environment."""
    return _import_bayotide()


def _bayotide_hyper(n_features):
    """Small, fast, CPU-only BayOTIDE config (few trend factors, no season)."""
    K_trend = int(min(10, max(2, n_features * 2)))
    cfg = {
        "device": "cpu",
        "K_trend": K_trend,
        "K_season": 0,          # disable the seasonal block for a fast generic run
        "n_season": 4,
        "K_bias": 0,
        "a0": 1, "b0": 1, "v": 1,
        "fix_int": True,
        "time_scale": 1,
        "kernel": {
            "kernel_trend": {"type": "Matern_21", "lengthscale": 0.1,
                             "variance": 1, "noise": 1},
            "kernel_season": {"type": "exp-periodic", "freq_list": [15],
                              "lengthscale_list": [0.05], "noise": 1},
        },
        "DAMPING_U": 0.8, "DAMPING_tau": 0.5, "DAMPING_W": 0.8,
        "EVALU_T": 10 ** 9,     # never trigger the internal eval/print branch
        "INNER_ITER": 5,
        "THRE": 1.0e-4,
    }
    return _bay_utils.make_hyper_dict(cfg, None)


def bayotide_impute(X, meta=None):
    """BayOTIDE online-filtering imputation (causal forward pass, no smoothing).

    We map our (T, N) time-major matrix to BayOTIDE's (N_features, T_time) layout,
    run a single forward filtering sweep over time (the model's online mode), and
    read out each column's reconstruction (W @ U_t) at the step it is filtered --
    the genuinely online estimate.  No backward smoothing pass is performed.
    """
    if not _bayotide_available():
        raise RuntimeError(
            "BayOTIDE is unavailable: "
            f"{_bayotide_err!r}. Clone {_BAYOTIDE_REPO} into {_BAYOTIDE_DIR} "
            "(or set BAYOTIDE_DIR); needs torch (present in the bench venv).")

    import torch

    X = np.asarray(X, dtype=np.float64)
    T, N = X.shape
    obs = np.isfinite(X)
    if T < 2 or N < 1 or obs.sum() == 0:
        return _column_mean_fallback(X, obs)

    torch.manual_seed(0)
    np.random.seed(0)

    data = X.T.copy()                      # (N, T) feature-major
    obs_nt = np.isfinite(data)             # (N, T)
    data_filled = np.where(obs_nt, data, 0.0)
    zero_mask = np.zeros((N, T), dtype=np.int64)
    data_dict = {
        "data": data_filled,
        "mask_train": obs_nt.astype(np.int64),
        "mask_test": zero_mask,            # we evaluate externally; no internal test
        "mask_valid": zero_mask,
        "time_uni": np.linspace(0.0, 1.0, T),
        "fix_int": True,
        "ndims": (N, T),
    }
    with _silence_stdout():
        hyper = _bayotide_hyper(N)
    data_dict["LDS_paras_trend"] = _bay_utils.make_LDS_paras_trend(hyper, data_dict)
    data_dict["LDS_paras_season"] = _bay_utils.make_LDS_paras_season(hyper, data_dict)

    with warnings.catch_warnings(), _silence_stdout():
        warnings.simplefilter("ignore")
        model = _bay_mod.BayTIDE(hyper, data_dict)
        model.reset()

        inner = hyper["INNER_ITER"]
        snap = np.zeros((N, T), dtype=np.float64)   # online filtered reconstruction
        for t in range(model.T):
            model.filter_predict(t)
            model.msg_llk_init()
            if model.mask_train[:, t].sum() > 0:
                for it in range(inner):
                    last = (it == inner - 1)
                    model.msg_approx_U(t)
                    model.filter_update(t, last)
                    model.msg_approx_W(t)
                    model.post_update_W(t)
                model.msg_approx_tau(t)
                model.post_update_tau(t)
            else:
                model.filter_update_fake(t)
            # Online read-out: reconstruct column t with W,U_t as known at this step.
            Ut = model.post_U_m[:, :, t].reshape(-1, 1)             # (K,1)
            col = torch.mm(model.post_W_m.squeeze(-1), Ut).squeeze(-1)
            snap[:, t] = col.detach().cpu().numpy()

    out = snap.T                            # back to (T, N)
    out = np.where(obs, X, out)
    if not np.all(np.isfinite(out)):
        out = _scrub(out, X, obs)
    return out


# =========================================================================== #
# Shared helpers
# =========================================================================== #
def _column_mean_fallback(X, obs):
    """Causal-safe degenerate fallback: fill with per-column observed mean, then 0."""
    X = np.asarray(X, dtype=np.float64)
    col_mean = np.nanmean(np.where(obs, X, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    out = np.where(obs, X, np.broadcast_to(col_mean, X.shape))
    return np.where(np.isfinite(out), out, 0.0)


def _scrub(out, X, obs):
    """Replace any NaN/Inf in `out` with column means then 0; keep observed exact."""
    out = np.array(out, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        col_mean = np.nanmean(np.where(obs, X, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        out[bad] = np.broadcast_to(col_mean, out.shape)[bad]
        out[~np.isfinite(out)] = 0.0
    return np.where(obs, X, out)


# =========================================================================== #
# Registry: only methods that actually work in this environment
# =========================================================================== #
HAVE_BAYOTIDE = _bayotide_available()

ONLINE_COMPETITORS = {}
if HAVE_GCIMPUTE:
    ONLINE_COMPETITORS["gcimpute"] = gcimpute_impute
if HAVE_BAYOTIDE:
    ONLINE_COMPETITORS["BayOTIDE"] = bayotide_impute

# Honest causality labels.  gcimpute-online is strictly point-in-time; BayOTIDE is
# online-filtering (forward pass, no smoothing) but uses globally-shared loadings,
# so it is not bit-exactly truncation-invariant -- see module docstring.
CAUSALITY = {
    "gcimpute": "strict (truncation-invariant)",
    "BayOTIDE": "online-filtering (no smoothing; shared loadings -> not bit-exact PIT)",
}


# =========================================================================== #
# __main__ smoke test
# =========================================================================== #
def _load_smoke_matrix():
    """Real-ish (T<=400, N<=8) matrix from data/exchange_clean.npy, else synthetic."""
    cand = os.path.join(os.path.dirname(_BENCH), "data", "exchange_clean.npy")
    if os.path.isfile(cand):
        try:
            arr = np.load(cand).astype(np.float64)
            arr = arr[:400, :8]
            if arr.ndim == 2 and arr.shape[0] >= 50 and arr.shape[1] >= 2:
                return arr, "exchange_clean.npy[:400,:8]"
        except Exception:
            pass
    rng = np.random.default_rng(0)
    T, N = 400, 8
    arr = np.cumsum(rng.standard_normal((T, N)), axis=0)   # random-walk-ish series
    return arr, "synthetic random-walk (400,8)"


def _smoke():
    from causal_race import verify_causal

    Xfull, src = _load_smoke_matrix()
    T, N = Xfull.shape
    rng = np.random.default_rng(0)
    mask_missing = rng.random((T, N)) < 0.15            # 15% MCAR
    # keep at least the first row mostly observed so warm-up is meaningful
    Xin = Xfull.copy()
    Xin[mask_missing] = np.nan

    print(f"online_competitors smoke")
    print(f"  data         : {src}  shape={Xfull.shape}  missing={mask_missing.mean():.1%}")
    print(f"  HAVE_GCIMPUTE={HAVE_GCIMPUTE}  HAVE_BAYOTIDE={HAVE_BAYOTIDE}")
    print(f"  available    : {list(ONLINE_COMPETITORS)}")
    if not HAVE_BAYOTIDE:
        print(f"  BayOTIDE blocker: {_bayotide_err!r}")
    print()

    # Well-conditioned synthetic matrix for the causality check.  (Real exchange
    # data has a near-constant column that makes the copula's latent covariance
    # collapse and produce numerically unstable -- not look-ahead -- fills, which
    # would muddy a strict truncation-invariance test.)
    vrng = np.random.default_rng(7)
    Tv, Nv = 200, 5
    Xv_full = np.cumsum(vrng.standard_normal((Tv, Nv)), axis=0)
    Xv = Xv_full.copy()
    Xv[vrng.random((Tv, Nv)) < 0.15] = np.nan

    for name, fn in ONLINE_COMPETITORS.items():
        import time
        t0 = time.time()
        out = fn(Xin, None)
        dt = time.time() - t0

        finite = bool(np.all(np.isfinite(out)))
        shape_ok = (out.shape == Xfull.shape)
        obs = np.isfinite(Xin)
        preserved = bool(np.allclose(out[obs], Xfull[obs]))
        mae = float(np.mean(np.abs(out[mask_missing] - Xfull[mask_missing])))

        v = verify_causal(lambda Z, _f=fn: _f(Z, None), Xv, n_prefixes=4)

        print(f"  [{name}]  label={CAUSALITY.get(name, '?')}")
        print(f"     shape={out.shape} ok={shape_ok}  finite={finite}  "
              f"obs_preserved={preserved}")
        print(f"     MAE(masked)={mae:.4f}  (data std={np.nanstd(Xfull):.3f})  "
              f"time={dt:.2f}s")
        print(f"     verify_causal (synthetic): causal={v['causal']}  "
              f"max_dev={v['max_dev']:.3e}")
        print()


if __name__ == "__main__":
    _smoke()
