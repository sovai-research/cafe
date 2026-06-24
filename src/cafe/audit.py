"""Model-agnostic look-ahead LEAKAGE AUDIT -- "the epsilon of imputation".

The whole time-series imputation field benchmarks in the *bidirectional* (smoothing)
setting: every method may use the future to fill the past. That is invalid for any
sequential decision (a backtest, an online controller, an early-warning monitor), yet
the field has no shared, mechanical way to *measure* how much future a method borrows.

This module generalises CAFE's internal causality verifier into a reusable, neutral
STANDARD that wraps ANY imputer ``impute_fn(X) -> filled`` (the repo baseline contract:
``X`` is ``(T, N)`` float64 with ``np.nan`` at missing cells, ROW axis = TIME, observed
cells preserved exactly) and emits two numbers:

  (a) a binary CAUSALITY CERTIFICATE -- ``max_revision``, the largest change to an
      already-imputed early cell as the series is grown one prefix at a time
      (truncation invariance). ``max_revision <= tol`` => strictly point-in-time.

  (b) a continuous LEAKAGE SCORE -- ``leakage_delta = bidir_mae - causal_mae``, the
      accuracy a method silently borrows from the future. For a batch/bidirectional
      method we *derive* its honest causal variant via a trailing-window + right-edge
      readout (the same treatment used for deep models), so every method gets a Delta,
      not just the natively causal ones.

A natively causal method (CAFE, LOCF, a Kalman filter) has ``max_revision == 0`` and
``leakage_delta == 0``. A leaky method (linear interpolation, SoftImpute) has
``max_revision > 0`` and/or ``leakage_delta > 0``.

Pure numpy core. Anything heavy (matplotlib, the bench method panel) is soft-imported
only inside the functions that need it. The audit itself depends on numpy alone.

Quick start
-----------
    import cafe.audit as A
    rep = A.leakage_report(my_impute_fn, X)        # X: (T,N) with NaNs, or clean+mask
    print(rep["causal"], rep["max_revision"], rep["leakage_delta"])
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "trailing_windows", "right_edge",
    "verify_causal", "make_causal_variant",
    "leakage_delta", "leakage_report", "audit_panel",
]


# --------------------------------------------------------------------------- #
# Windowing (self-contained copy of bench/causal_race.py so the library has no
# dependency on the bench tree).
# --------------------------------------------------------------------------- #
def trailing_windows(X: np.ndarray, L: int) -> np.ndarray:
    """For every ``t`` the causal window ``X[t-L+1 .. t]`` (left NaN-pad early on).

    Returns ``(T, L, N)``; position ``[t, -1, :]`` is row ``t`` itself, and the whole
    window contains only times ``<= t``. Vectorised."""
    X = np.asarray(X, float)
    T, N = X.shape
    pad = np.full((L - 1, N), np.nan)
    Xp = np.vstack([pad, X])                              # (T+L-1, N)
    idx = np.arange(T)[:, None] + np.arange(L)[None, :]   # (T, L)
    return Xp[idx]


def right_edge(pred_trailing: np.ndarray) -> np.ndarray:
    """Causal readout: the right-most position of each trailing window. ``(T,L,N)->(T,N)``."""
    return np.asarray(pred_trailing, float)[:, -1, :]


def _keep_observed(X: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Observed cells preserved exactly; never emit NaN/Inf."""
    X = np.asarray(X, float)
    out = np.where(np.isfinite(X), X, np.asarray(filled, float))
    if not np.all(np.isfinite(out)):
        col = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        bad = ~np.isfinite(out)
        out[bad] = np.take(col, np.where(bad)[1])
    return out


# --------------------------------------------------------------------------- #
# (a) Causality certificate: truncation invariance.
# --------------------------------------------------------------------------- #
def verify_causal(impute_fn, X: np.ndarray, n_prefixes: int = 6,
                  tol: float = 1e-6) -> dict:
    """Re-impute on growing time-prefixes; a causal method leaves earlier fills
    unchanged.

    Returns ``{"causal": bool, "max_revision": float, "n_comparisons": int}``.
    Self-contained (works on a raw NaN-holed matrix). ``max_revision`` is the
    largest ``|fill_on_prefix - fill_on_full|`` over every early cell and prefix --
    the certificate number. ``<= tol`` => strictly point-in-time.
    """
    X = np.asarray(X, float)
    T, N = X.shape
    full = np.asarray(impute_fn(X.copy()), float)
    max_rev = 0.0
    n_cmp = 0
    cuts = np.linspace(T // 3, T - 1, n_prefixes, dtype=int)
    for k in np.unique(cuts):
        pref = np.asarray(impute_fn(X[: k + 1].copy()), float)
        d = np.abs(pref - full[: k + 1])
        d = d[np.isfinite(d)]
        n_cmp += int(d.size)
        if d.size:
            max_rev = max(max_rev, float(d.max()))
    return {"causal": max_rev <= tol, "max_revision": max_rev,
            "n_comparisons": n_cmp}


# --------------------------------------------------------------------------- #
# (b) Leakage score: a causal variant of ANY imputer + Delta.
# --------------------------------------------------------------------------- #
def make_causal_variant(impute_fn, L: int = 24, chunk: int = 256):
    """Wrap a (possibly bidirectional/batch) ``impute_fn(X)->filled`` into a strict
    point-in-time filter via trailing windows + right-edge readout.

    The returned ``causal_fn(X)`` builds, for each time ``t``, the trailing window
    ``X[t-L+1 .. t]``, imputes it with the *original* method, and keeps ONLY the
    right-most (time ``t``) row -- a value that depends on data ``<= t`` only. This is
    exactly how the field's bidirectional/deep imputers are made honest. A natively
    causal method is (near-)unchanged by this wrapping; a leaky method loses the
    future it was borrowing, which is what ``leakage_delta`` then measures.
    """
    def causal_fn(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        T, N = X.shape
        TW = trailing_windows(X, L)                       # (T, L, N)
        edges = np.empty((T, N), float)
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            blk = TW[s:e].reshape(-1, N)                  # ((e-s)*L, N)
            filled = np.asarray(impute_fn(blk.copy()), float).reshape(e - s, L, N)
            edges[s:e] = right_edge(filled)
        return _keep_observed(X, edges)
    return causal_fn


def _score_masked(truth, pred, mask):
    mask = np.asarray(mask, bool)
    t = np.asarray(truth, float)[mask]
    p = np.asarray(pred, float)[mask]
    bad = ~np.isfinite(p)
    if bad.any():
        p = p.copy(); p[bad] = t[bad] + 1e3
    return float(np.mean(np.abs(p - t)))


def leakage_delta(impute_fn, clean: np.ndarray, mask: np.ndarray,
                  L: int = 24, native_causal: bool = False) -> dict:
    """The accuracy a method borrows from the future on one (clean, mask) task.

    ``clean`` is the complete ground truth (already on its scoring scale); ``mask`` is
    True where cells are held out. We score the SAME method two ways on the SAME held-out
    cells:

      * bidir  -- the method applied to the whole series at once (its standard,
                  future-using setting).
      * causal -- the method applied as a strict point-in-time filter
                  (:func:`make_causal_variant`); for a natively causal method this is
                  the method itself.

    Returns ``{"bidir_mae", "causal_mae", "leakage_delta"}`` with
    ``leakage_delta = causal_mae - bidir_mae`` -- the future-borrowed accuracy
    (>= 0 for a method that genuinely uses the future; ~0 for a causal method).
    """
    clean = np.asarray(clean, float)
    mask = np.asarray(mask, bool)
    Xobs = clean.copy()
    Xobs[mask] = np.nan

    bidir = np.asarray(impute_fn(Xobs.copy()), float)
    bidir_mae = _score_masked(clean, bidir, mask)

    if native_causal:
        causal = bidir
    else:
        causal = make_causal_variant(impute_fn, L=L)(Xobs.copy())
    causal_mae = _score_masked(clean, causal, mask)

    return {"bidir_mae": bidir_mae, "causal_mae": causal_mae,
            "leakage_delta": causal_mae - bidir_mae}


# --------------------------------------------------------------------------- #
# Internal: leak-free standardisation + mask generation so the audit is
# self-contained (mirrors bench/eval_utils.py without importing it).
# --------------------------------------------------------------------------- #
def _standardize_on_observed(clean, mask, eps=1e-9):
    clean = np.asarray(clean, float)
    mask = np.asarray(mask, bool)
    visible = (~mask) & np.isfinite(clean)
    Xstd = np.empty_like(clean)
    for j in range(clean.shape[1]):
        col, vis = clean[:, j], visible[:, j]
        if vis.sum() >= 2:
            mu, sd = col[vis].mean(), col[vis].std()
        elif vis.sum() == 1:
            mu, sd = col[vis][0], 1.0
        else:
            mu, sd = 0.0, 1.0
        Xstd[:, j] = (col - mu) / (sd + eps)
    return Xstd


def _mcar_mask(shape, rate, seed):
    return np.random.default_rng(seed).random(shape) < rate


def _block_mask(shape, rate, seed, min_gap=8, max_gap=40):
    T, N = shape
    rng = np.random.default_rng(seed)
    M = np.zeros((T, N), bool)
    target = int(round(rate * T))
    for j in range(N):
        filled, guard = 0, 0
        while filled < target and guard < 200:
            g = min(int(rng.integers(min_gap, max_gap + 1)), T)
            s = int(rng.integers(0, max(1, T - g)))
            if not M[s:s + g, j].any():
                M[s:s + g, j] = True
                filled += g
            guard += 1
    return M


def _make_mask(pattern, shape, rate, seed):
    if pattern in ("block", "subseq", "subsequence"):
        return _block_mask(shape, rate, seed)
    return _mcar_mask(shape, rate, seed)


# --------------------------------------------------------------------------- #
# The headline one-call audit.
# --------------------------------------------------------------------------- #
def leakage_report(impute_fn, X: np.ndarray, mask: np.ndarray | None = None,
                   rates=(0.10,), patterns=("block",), L: int = 24,
                   native_causal: bool = False, seed: int = 0,
                   tol: float = 1e-6, n_prefixes: int = 6) -> dict:
    """Full leakage audit of ONE imputer. Model-agnostic: ``impute_fn`` is any
    callable following the repo baseline contract.

    Parameters
    ----------
    impute_fn : callable ``(T,N) NaN-holed -> (T,N) filled`` (observed cells exact).
    X : either a NaN-holed matrix (cells already missing) OR a complete matrix.
        If ``mask`` is given, ``X`` is treated as the complete ground truth and the
        mask defines the held-out cells; leak-free standardisation is applied. If
        ``mask`` is None, missing cells are inferred from NaNs in ``X`` and the
        certificate is still computed, but the accuracy Delta is computed against a
        synthetic mask drawn over the OBSERVED cells (so we have ground truth).
    rates, patterns : the missingness grid for the accuracy Delta (averaged).
    L : trailing-window length for the derived causal variant.
    native_causal : if True, the causal variant == the method itself (skip the
        trailing-window readout). Set for methods already strictly point-in-time.
    tol : certificate tolerance on ``max_revision``.

    Returns a dict with the certificate (``causal``, ``max_revision``), the averaged
    accuracy (``bidir_mae``, ``causal_mae``, ``leakage_delta``), and bookkeeping.
    """
    X = np.asarray(X, float)
    T, N = X.shape

    # ---- (a) certificate on the AS-GIVEN missingness (raw NaN-holed matrix) ----
    if mask is not None:
        cert_input = X.copy()
        cert_input[np.asarray(mask, bool)] = np.nan
    elif np.isnan(X).any():
        cert_input = X
    else:
        # no NaNs and no mask -> punch a small certificate mask so the test has cells
        cert_input = X.copy()
        cert_input[_make_mask(patterns[0], (T, N), rates[0], seed)] = np.nan
    cert = verify_causal(impute_fn, cert_input, n_prefixes=n_prefixes, tol=tol)

    # ---- (b) accuracy Delta: need complete ground truth + a scoring mask ----
    # Build the complete matrix to score against.
    if mask is not None:
        clean_full = X
    else:
        # impute the as-given NaNs ONCE to get a dense "truth surrogate"; then we
        # hold out a fresh synthetic mask over the originally-OBSERVED cells (whose
        # values we know), so Delta is measured against real observed values.
        clean_full = np.asarray(impute_fn(X.copy()), float)

    observed = np.isfinite(X) if mask is None else np.ones_like(X, bool)
    deltas, bmaes, cmaes = [], [], []
    for pat in patterns:
        for r in rates:
            if mask is not None:
                m = np.asarray(mask, bool)
                clean = _standardize_on_observed(clean_full, m)
            else:
                m_raw = _make_mask(pat, (T, N), r, seed)
                m = m_raw & observed          # only hold out cells we truly know
                clean = clean_full
            if not m.any():
                continue
            d = leakage_delta(impute_fn, clean, m, L=L, native_causal=native_causal)
            bmaes.append(d["bidir_mae"]); cmaes.append(d["causal_mae"])
            deltas.append(d["leakage_delta"])

    bidir_mae = float(np.mean(bmaes)) if bmaes else float("nan")
    causal_mae = float(np.mean(cmaes)) if cmaes else float("nan")
    delta = float(np.mean(deltas)) if deltas else float("nan")

    return {
        "causal": bool(cert["causal"]),
        "max_revision": float(cert["max_revision"]),
        "n_comparisons": int(cert["n_comparisons"]),
        "bidir_mae": bidir_mae,
        "causal_mae": causal_mae,
        "leakage_delta": delta,
        "native_causal": bool(native_causal),
        "L": int(L),
        "tol": float(tol),
    }


def audit_panel(methods: dict, X: np.ndarray, mask: np.ndarray | None = None,
                **kw) -> list:
    """Audit a panel of methods and return rows sorted by ``leakage_delta`` (desc).

    ``methods`` maps ``name -> impute_fn`` or ``name -> (impute_fn, native_causal)``.
    Each row is the :func:`leakage_report` dict with ``"method"`` added.
    """
    rows = []
    for name, spec in methods.items():
        if isinstance(spec, (tuple, list)):
            fn, native = spec[0], bool(spec[1])
        else:
            fn, native = spec, False
        try:
            rep = leakage_report(fn, X, mask=mask, native_causal=native, **kw)
        except Exception as e:                              # pragma: no cover
            rep = {"causal": None, "max_revision": float("nan"),
                   "leakage_delta": float("nan"), "bidir_mae": float("nan"),
                   "causal_mae": float("nan"), "error": repr(e)[:160]}
        rep["method"] = name
        rows.append(rep)
    rows.sort(key=lambda r: (-(r["leakage_delta"] if r["leakage_delta"] ==
                              r["leakage_delta"] else -1)))
    return rows


if __name__ == "__main__":
    # Self-test: a causal filler (LOCF-ish) vs a leaky one (mean-of-window that
    # peeks both sides). Pure numpy, no bench deps.
    rng = np.random.default_rng(0)
    T, N = 200, 6
    X = np.sin(np.linspace(0, 20, T))[:, None] + 0.3 * rng.standard_normal((T, N))

    def locf(Z):                       # causal
        Z = np.asarray(Z, float).copy()
        for j in range(Z.shape[1]):
            last = 0.0
            for t in range(Z.shape[0]):
                if np.isfinite(Z[t, j]):
                    last = Z[t, j]
                else:
                    Z[t, j] = last
        return Z

    def lin_interp(Z):                 # leaky (reads the future)
        Z = np.asarray(Z, float).copy()
        tt = np.arange(Z.shape[0], dtype=float)
        for j in range(Z.shape[1]):
            obs = np.isfinite(Z[:, j])
            if obs.sum() >= 2:
                Z[:, j] = np.interp(tt, tt[obs], Z[obs, j])
        return Z

    rep_c = leakage_report(locf, X, native_causal=True)
    rep_l = leakage_report(lin_interp, X, native_causal=False)
    print("LOCF        :", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in rep_c.items() if k in
                            ("causal", "max_revision", "leakage_delta")})
    print("LinearInterp:", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in rep_l.items() if k in
                            ("causal", "max_revision", "leakage_delta")})
    assert rep_c["causal"] and rep_c["max_revision"] <= 1e-6
    assert (not rep_l["causal"]) or rep_l["leakage_delta"] > 1e-3
    print("audit self-test PASSED")
