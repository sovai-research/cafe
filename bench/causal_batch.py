"""Causal variants of the batch (non-causal) imputers TRMF, SoftImpute, LinearInterp.

The causal horse race (``exp_horserace.py``) registers these three batch imputers as
bidirectional-only (``("classical", bidir_fn, None)``): they take a full ``(T, N)``
matrix and fill it using the WHOLE series at once, so every fill at time ``t`` may
read observations from ``s > t`` (look-ahead). They therefore have no causal MAE and
their look-ahead gap shows ``--``.

This module gives each one a *real* causal variant via the SAME right-edge / trailing
window readout the deep models already use (``causal_race.causal_fill``):

    for every time t -> build the trailing window X[t-L+1 .. t] (only data <= t),
    re-impute that small (L, N) block with the batch imputer, and trust ONLY the
    right-most row (position t).

So the causal fill at ``t`` depends on data ``<= t`` only -- truncation-invariant,
the honest causal application of a batch model. The gap

    delta = causal_mae - bidir_mae >= 0

is exactly the accuracy the batch method silently borrows from the future.

API (for wiring into exp_horserace._classical_methods):

    from causal_batch import CAUSAL_BATCH
    CAUSAL_BATCH == {
        "SoftImpute":   ("classical", bidir_fn, causal_fn),
        "TRMF":         ("classical", bidir_fn, causal_fn),
        "LinearInterp": ("classical", bidir_fn, causal_fn),
    }
    # each fn: (T, N) NaN-holed -> (T, N) filled, finite out.

WINDOW is read from env ``HR_WINDOW`` (default 24) to match the race.

NOTES on each method's causal degeneration (this is the honest point of the exercise):
  * LinearInterp: within a trailing window the last row (position t) has no future
    point, so right-edge linear interpolation degenerates to last-observation-carried-
    forward (LOCF) -- which is the CORRECT, honest behaviour and reveals that linear
    interpolation's bidirectional accuracy was borrowed from the future.
  * TRMF / SoftImpute: re-fit a low-rank(+AR) model on each short trailing window;
    the right-edge cell is reconstructed from past data only. TRMF's per-window cost
    is bounded by reducing iterations for the windowed calls (the window is only L
    rows long, so few sweeps converge).

WARMUP rows: a trailing window for small ``t`` is left-NaN-padded (rows before t=0
do not exist). If the right-edge row has too little real history for the batch model
to be meaningful, we fall back to a simple *causal* fill (LOCF -> column running mean)
rather than crashing or emitting NaN. ``causal_race.causal_fill`` runs the batch model
on the padded window directly; we make each ``predict_fn`` robust to all-NaN/short
windows so the fallback is reached cleanly.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import causal_race as CR              # noqa: E402

WINDOW = int(os.environ.get("HR_WINDOW", 24))

# Min number of observed rows in a window before we trust the heavy batch model on its
# right edge; below this we use the simple causal fallback (LOCF -> running mean).
_MIN_HIST = 2


# --------------------------------------------------------------------------- #
# Simple causal fallback for warmup / degenerate windows.
# --------------------------------------------------------------------------- #
def _causal_fallback_edge(win: np.ndarray) -> np.ndarray:
    """Causal fill of a single (L, N) trailing window's RIGHT-EDGE row.

    Uses only data <= the edge: LOCF (most recent observed value in the column,
    within the window) -> column running-mean over the window -> 0.0. Returns the
    edge row (N,). Never reads the future (there is none past the edge anyway)."""
    win = np.asarray(win, float)
    L, N = win.shape
    out = np.full(N, np.nan)
    for j in range(N):
        col = win[:, j]
        obs = np.isfinite(col)
        if obs.any():
            # LOCF: last observed value at or before the edge
            out[j] = col[obs][-1]
        # else stays NaN -> handled by the 0.0 backstop below
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = 0.0
    return out


def _edge_from_block(block_filled: np.ndarray, block_in: np.ndarray) -> np.ndarray:
    """Right-edge readout of an imputed (L, N) block, with a per-column causal
    fallback for any column the batch model left non-finite (e.g. all-NaN in the
    window). Returns the (N,) edge row."""
    edge = np.asarray(block_filled, float)[-1].copy()
    bad = ~np.isfinite(edge)
    if bad.any():
        fb = _causal_fallback_edge(block_in)
        edge[bad] = fb[bad]
    return edge


# --------------------------------------------------------------------------- #
# Per-window predict_fn builders. causal_race.causal_fill calls predict_fn on a
# (k, L, N) chunk of trailing windows and reads right_edge() = [:, -1, :]. So each
# predict_fn must return a (k, L, N) tensor whose LAST row per window is the causal
# fill of that window's edge. We fill the whole window with the batch model (cheap on
# L rows) and let right_edge pick the last row; we additionally guarantee the last row
# is finite via the causal fallback.
# --------------------------------------------------------------------------- #
def _make_predict_fn(window_impute):
    """window_impute: (L, N) NaN-holed -> (L, N) filled. Returns a predict_fn over a
    (k, L, N) batch whose last-row-per-window is a finite causal fill."""

    def predict_fn(W3: np.ndarray) -> np.ndarray:
        W3 = np.asarray(W3, float)
        k, L, N = W3.shape
        out = W3.copy()
        for i in range(k):
            win = W3[i]                                   # (L, N)
            obs = np.isfinite(win)
            n_hist = int(obs.any(axis=1).sum())          # rows with any observation
            if n_hist < _MIN_HIST:
                # warmup / too little history: simple causal fallback on the edge.
                edge = _causal_fallback_edge(win)
                filled = win.copy()
                filled[-1] = np.where(np.isfinite(win[-1]), win[-1], edge)
                out[i] = filled
                continue
            try:
                filled = np.asarray(window_impute(win), float)
                if filled.shape != win.shape:
                    raise ValueError("shape change")
            except Exception:
                filled = win.copy()
            # keep observed cells exact (the batch imputers already do, but be safe)
            filled = np.where(obs, win, filled)
            # guarantee a finite, causal right-edge row
            edge = _edge_from_block(filled, win)
            filled[-1] = np.where(np.isfinite(win[-1]), win[-1], edge)
            out[i] = filled
        return out

    return predict_fn


# --------------------------------------------------------------------------- #
# The three batch imputers as (L, N) window-imputers.
# --------------------------------------------------------------------------- #
def _softimpute_window(win):
    import m_softimpute as S
    L, N = win.shape
    # small windows: cap rank and iterations for speed (L rows only).
    return S._impute_core(win, rank_cap=min(10, L, N), max_iter=30, tol=1e-3)


def _trmf_window(win):
    import m_trmf as Tm
    L, N = win.shape
    # Reduced settings for the short window: TRMF's _trmf_2d on L (=24) rows with a
    # handful of iterations converges fast and keeps per-window cost tiny. Lags must
    # be < L; small lags are appropriate for a short trailing window.
    k = max(1, min(6, L - 1, N))
    lags = tuple(l for l in (1, 2, 3) if l < L) or (1,)
    return Tm._trmf_2d(win, k=k, lags=lags, n_iter=8,
                       lam_w=1.0, lam_f=2.0, lam_th=1.0, seed=0)


def _linterp_window(win):
    import m_naive as Nai
    # m_naive.linear_interp uses np.interp, which CLAMPS trailing gaps to the nearest
    # (= last) observed value -- i.e. on the right edge there is no future point to
    # interpolate toward, so the edge degenerates to LOCF. That is exactly the honest
    # causal behaviour we want: linear interp gets no future credit at the edge.
    return Nai.linear_interp(win, None)


# --------------------------------------------------------------------------- #
# bidir_fn: the existing batch impute on the FULL matrix (unchanged).
# --------------------------------------------------------------------------- #
def _softimpute_bidir(X):
    import m_softimpute as S
    return S.impute(X, {})


def _trmf_bidir(X):
    import m_trmf as Tm
    return Tm.impute(X, {})


def _linterp_bidir(X):
    import m_naive as Nai
    return Nai.linear_interp(X, {})


# --------------------------------------------------------------------------- #
# causal_fn: causal_race.causal_fill with the per-window predict_fn, L = WINDOW.
# --------------------------------------------------------------------------- #
def _make_causal_fn(window_impute, chunk=256):
    pf = _make_predict_fn(window_impute)

    def causal_fn(X):
        return CR.causal_fill(pf, X, L=WINDOW, chunk=chunk)

    return causal_fn


# --------------------------------------------------------------------------- #
# The registry the author wires into exp_horserace._classical_methods().
# Family stays "classical" (figures/tables keep their classical color) but now
# carries a real causal_fn instead of None.
# --------------------------------------------------------------------------- #
CAUSAL_BATCH = {
    "SoftImpute":   ("classical", _softimpute_bidir, _make_causal_fn(_softimpute_window)),
    "TRMF":         ("classical", _trmf_bidir,       _make_causal_fn(_trmf_window)),
    "LinearInterp": ("classical", _linterp_bidir,    _make_causal_fn(_linterp_window)),
}


# --------------------------------------------------------------------------- #
# Verify + measure.
# --------------------------------------------------------------------------- #
def _make_synth(T=400, N=8, rank=3, seed=0):
    """Low-rank + AR(1) latent factors -> structured (T, N) matrix."""
    rng = np.random.default_rng(seed)
    F = np.zeros((T, rank))
    phi = 0.85
    for r in range(rank):
        e = rng.standard_normal(T)
        f = np.zeros(T)
        for t in range(1, T):
            f[t] = phi * f[t - 1] + e[t]
        F[:, r] = f
    Wl = rng.standard_normal((N, rank))
    X = F @ Wl.T + 0.1 * rng.standard_normal((T, N))
    return X


def _block_mask(shape, rate=0.1, seed=0):
    """Contiguous-block missingness (the harder, look-ahead-revealing pattern)."""
    T, N = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros((T, N), bool)
    target = int(rate * T * N)
    placed = 0
    while placed < target:
        j = rng.integers(N)
        Lb = int(rng.integers(3, 12))
        s = int(rng.integers(0, max(1, T - Lb)))
        seg = slice(s, s + Lb)
        new = (~mask[seg, j]).sum()
        mask[seg, j] = True
        placed += new
    return mask


def _standardize(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return (X - mu) / sd


def _mae(truth, pred, mask):
    d = np.abs(truth[mask] - pred[mask])
    return float(np.nanmean(d))


def _eval_dataset(name, clean, rate=0.10, seed=0):
    import time
    clean = _standardize(np.asarray(clean, float))
    T, N = clean.shape
    mask = _block_mask((T, N), rate=rate, seed=seed)
    Xobs = clean.copy()
    Xobs[mask] = np.nan
    truth = clean

    print(f"\n=== {name}  shape={clean.shape}  missing={mask.mean()*100:.1f}%  "
          f"WINDOW={WINDOW} ===")
    hdr = f"{'method':<14}{'bidir MAE':>11}{'causal MAE':>12}{'delta':>9}" \
          f"{'max_dev':>10}{'bsec':>7}{'csec':>7}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for mname, (_, bfn, cfn) in CAUSAL_BATCH.items():
        t0 = time.time()
        bpred = bfn(Xobs.copy())
        bsec = time.time() - t0
        bmae = _mae(truth, bpred, mask)

        t0 = time.time()
        cpred = cfn(Xobs.copy())
        csec = time.time() - t0
        cmae = _mae(truth, cpred, mask)

        delta = cmae - bmae
        # verify causal on a small prefix-subset (cap rows for the heavy methods)
        Xv = Xobs[: min(T, 200)]
        v = CR.verify_causal(cfn, Xv, n_prefixes=4)
        results[mname] = dict(bidir=bmae, causal=cmae, delta=delta,
                              max_dev=v["max_dev"], bsec=bsec, csec=csec)
        print(f"{mname:<14}{bmae:>11.4f}{cmae:>12.4f}{delta:>9.4f}"
              f"{v['max_dev']:>10.2e}{bsec:>7.2f}{csec:>7.2f}")
    return results


if __name__ == "__main__":
    print(f"CAUSAL_BATCH causal variants of batch imputers | WINDOW={WINDOW}")
    print("API: CAUSAL_BATCH = {name: (family, bidir_fn, causal_fn)} for",
          list(CAUSAL_BATCH))

    all_res = {}

    # 1) synthetic low-rank + AR with block missingness
    Xs = _make_synth(T=400, N=8, rank=3, seed=0)
    all_res["synthetic"] = _eval_dataset("synthetic (low-rank+AR, block)", Xs)

    # 2) real datasets, capped to ~800 rows
    ROOT = os.path.dirname(_HERE)
    for ds, fn in [("fredmd", "fredmd_clean.npy"), ("beijing", "beijing_clean.npy")]:
        try:
            X = np.load(os.path.join(ROOT, "data", fn)).astype(float)
            if X.shape[0] > 800:
                X = X[:800]
            if X.shape[1] > 40:
                X = X[:, :40]
            X = np.ascontiguousarray(X)
            all_res[ds] = _eval_dataset(ds, X)
        except Exception as e:
            print(f"  [skip] {ds}: {e!r}")

    # ---- TRMF look-ahead highlighted ----
    print("\n" + "=" * 60)
    print("TRMF LOOK-AHEAD (delta = causal - bidir), by dataset:")
    for ds, res in all_res.items():
        if "TRMF" in res:
            r = res["TRMF"]
            print(f"  {ds:<24} bidir={r['bidir']:.4f}  causal={r['causal']:.4f}"
                  f"  delta={r['delta']:+.4f}")
    print("=" * 60)
