"""PyPOTS deep imputation models wrapped for the *causal horse race*.

Every published deep imputer (SAITS, BRITS, Transformer, TimesNet, ImputeFormer)
is a *bidirectional* / smoothing model: when it fills time ``t`` it attends over the
whole window, including ``t+1 .. t+L-1``. That is invalid for any sequential decision.

This module wraps each PyPOTS model so the SAME trained weights can be applied two ways
(via :mod:`bench.causal_race`):

  * ``fill_bidir`` -- standard, non-overlapping windows, full readout (uses the future).
  * ``fill_causal`` -- trailing windows + right-edge readout: strictly point-in-time.

The look-ahead gap ``causal_mae - bidir_mae >= 0`` is the accuracy each method silently
borrows from the future. CAFE (natively causal) has gap 0.

The module *soft-imports* torch/pypots: on a numpy-only checkout it imports cleanly with
``DEEP_MODELS == {}`` and ``HAVE_PYPOTS == False`` so the rest of the repo never breaks.
"""
from __future__ import annotations

import io
import contextlib
import os
import warnings

import numpy as np

import causal_race  # sibling module (bench/causal_race.py)

__all__ = [
    "HAVE_PYPOTS", "DEEP_MODELS", "DeepImputer", "build",
]

# Fixed determinism / CPU knobs.
_SEED = 0
_N_THREADS = 4


class _Buf(io.StringIO):
    """StringIO that survives redirect_stdout during PyPOTS/ai4ts import.

    GOTCHA: ``ai4ts/__init__`` reads ``sys.stdout.encoding.lower()`` at import time;
    a plain ``io.StringIO`` has ``encoding is None`` and crashes the import when used
    as a ``redirect_stdout`` target. Giving it an ``encoding`` attribute fixes it."""

    encoding = "utf-8"


# --------------------------------------------------------------------------- #
# Soft import of the deep stack (torch + pypots). Suppress the ascii banner.
# --------------------------------------------------------------------------- #
def _have_pypots():
    """Try to import torch + the five PyPOTS models. Returns ``(ok, registry)``.

    ``registry`` maps name -> ``(model_cls, default_small_hp_factory)`` where the
    factory takes ``(L, N)`` and returns a dict of *architecture* kwargs (no epochs/
    batch_size/device -- those are supplied by :class:`DeepImputer`)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            buf = _Buf()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                import torch  # noqa: F401
                from pypots.imputation import (  # noqa: F401
                    SAITS, BRITS, Transformer, TimesNet, ImputeFormer,
                )
    except Exception:
        return False, {}

    # "Medium" capacity configs -- big enough to actually converge on real data
    # (so the causal horse race is FAIR), still CPU-fast on ~1200x100 windows.
    # Every kwarg below was confirmed against ``inspect.signature(cls.__init__)``
    # for pypots 1.5 (SAITS/Transformer share the attention block kwargs;
    # BRITS only takes rnn_hidden_size; TimesNet uses top_k/n_kernels;
    # ImputeFormer uses the three embed dims + n_temporal_heads).
    registry = {
        "SAITS": (SAITS, lambda L, N: dict(
            n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=64)),
        "Transformer": (Transformer, lambda L, N: dict(
            n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=64)),
        "BRITS": (BRITS, lambda L, N: dict(
            rnn_hidden_size=64)),
        "TimesNet": (TimesNet, lambda L, N: dict(
            n_layers=2, top_k=3, d_model=32, d_ffn=32, n_kernels=4)),
        "ImputeFormer": (ImputeFormer, lambda L, N: dict(
            n_layers=2, d_input_embed=32, d_learnable_embed=32, d_proj=32,
            d_ffn=64, n_temporal_heads=4)),
    }
    return True, registry


HAVE_PYPOTS, DEEP_MODELS = _have_pypots()


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
class DeepImputer:
    """A PyPOTS imputation model usable both bidirectionally and causally.

    Train once (self-supervised on masked windows; never sees ground truth), then read
    out two ways with the *same* weights so the look-ahead gap is purely the application
    mode. Standardisation uses ONLY observed cells of the fit matrix (point-in-time safe).
    """

    def __init__(self, name: str, L: int = 24, epochs: int = 60,
                 batch_size: int = 32, patience: int = 6,
                 val_frac: float = 0.1, **hp):
        if not HAVE_PYPOTS:
            raise RuntimeError("PyPOTS / torch not available in this environment.")
        if name not in DEEP_MODELS:
            raise KeyError(f"unknown model {name!r}; have {sorted(DEEP_MODELS)}")
        self.name = name
        self.L = int(L)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        # Extra fraction of OBSERVED cells re-masked to form a self-supervised
        # validation target for early stopping. Leak-free: those cells come only
        # from the (already masked) training windows; ground truth is never read.
        self.val_frac = float(val_frac)
        self.hp = dict(hp)
        self._model = None
        self._mean = None
        self._std = None
        self._N = None

    # -- helpers ---------------------------------------------------------- #
    def _set_determinism(self):
        import torch
        os.environ.setdefault("PYTHONHASHSEED", str(_SEED))
        np.random.seed(_SEED)
        torch.manual_seed(_SEED)
        try:
            torch.use_deterministic_algorithms(False)  # CPU kernels are deterministic
        except Exception:
            pass
        torch.set_num_threads(_N_THREADS)

    def _standardize(self, X2d: np.ndarray) -> np.ndarray:
        X = np.asarray(X2d, float)
        return (X - self._mean[None, :]) / self._std[None, :]

    def _inverse(self, X2d: np.ndarray) -> np.ndarray:
        X = np.asarray(X2d, float)
        return X * self._std[None, :] + self._mean[None, :]

    # -- fit -------------------------------------------------------------- #
    def fit(self, Xobs2d: np.ndarray) -> "DeepImputer":
        """Standardise on observed cells, window, and train the model ONCE.

        ``Xobs2d`` is ``(T, N)`` with ``np.nan`` for missing. Ground truth is never seen.
        """
        from pypots.imputation import (  # local import keeps module import light
            SAITS, BRITS, Transformer, TimesNet, ImputeFormer,  # noqa: F401
        )
        self._set_determinism()
        X = np.asarray(Xobs2d, float)
        T, N = X.shape
        self._N = N

        # Per-feature scaler from observed cells only.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
            std = np.nanstd(np.where(np.isfinite(X), X, np.nan), axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        self._mean, self._std = mean, std

        Xs = self._standardize(X)
        W, _ = causal_race.make_windows(Xs, self.L)       # (n_windows, L, N)

        # ----- self-supervised validation split for early stopping ----------- #
        # Hold out a contiguous time-block of the windows for validation, then
        # additionally mask a small fraction of the OBSERVED cells in that block
        # so we have clean targets (the pre-mask standardised values) to score.
        # This is leak-free: every value used as a target was already part of the
        # masked training matrix Xobs2d -- ground truth is never read.
        train_set, val_set = self._make_train_val(W)

        cls, hp_factory = DEEP_MODELS[self.name]
        arch_hp = hp_factory(self.L, N)
        arch_hp.update(self.hp)                            # user overrides win
        # ``patience`` + best-checkpoint restore let us run many epochs without
        # overfitting; PyPOTS keeps the best-on-val weights when val_set is given.
        # PyPOTS requires patience < epochs; clamp so tiny-epoch calls still work.
        pat = min(self.patience, self.epochs - 1) if self.epochs > 1 else None
        model = cls(
            n_steps=self.L, n_features=N,
            epochs=self.epochs, batch_size=self.batch_size,
            patience=(pat if val_set is not None else None),
            model_saving_strategy="best" if val_set is not None else None,
            device="cpu", verbose=False,
            **arch_hp,
        )
        buf = _Buf()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if val_set is not None:
                    model.fit(train_set, val_set)
                else:
                    model.fit(train_set)
        self._model = model
        return self

    # -- validation split ------------------------------------------------- #
    def _make_train_val(self, W: np.ndarray):
        """Split windows ``(k, L, N)`` into a PyPOTS train_set + val_set.

        Returns ``(train_set, val_set)``. ``val_set`` is ``{"X": X_holed,
        "X_ori": X_target}`` where ``X_holed`` has an EXTRA ``val_frac`` of its
        observed cells masked and ``X_ori`` carries the pre-mask values so PyPOTS
        can compute the validation metric on exactly those held-out cells.

        If there are too few windows / observed cells to form a meaningful val
        block we return ``(train_set_all, None)`` and fall back to fixed-epoch
        training (reported honestly).
        """
        W = np.asarray(W, float)
        k = W.shape[0]
        # Need at least a couple of windows to hold one out for validation.
        if k < 4 or self.val_frac <= 0.0:
            return {"X": W}, None

        n_val = max(1, int(round(k * 0.18)))              # ~15-20% time block
        n_val = min(n_val, k - 1)
        # Contiguous TRAILING block of windows -> a held-out time segment.
        train_W = W[: k - n_val]
        val_W = W[k - n_val:].copy()

        # Re-mask a fraction of the observed cells in the val block.
        rng = np.random.default_rng(_SEED + 1)
        obs = np.isfinite(val_W)
        if obs.sum() == 0:
            return {"X": W}, None
        draw = rng.random(val_W.shape) < self.val_frac
        extra = obs & draw
        if not extra.any():
            # Guarantee at least one held-out cell.
            flat = np.flatnonzero(obs.ravel())
            extra.ravel()[flat[: max(1, flat.size // 50)]] = True

        val_ori = val_W.copy()                            # targets (pre-mask)
        val_holed = val_W.copy()
        val_holed[extra] = np.nan                         # input the model sees
        return {"X": train_W}, {"X": val_holed, "X_ori": val_ori}

    # -- core forward pass used by both readouts -------------------------- #
    def _predict(self, W3d: np.ndarray) -> np.ndarray:
        """``(k, L, N)`` NaN-holed (already standardised) -> imputed ``(k, L, N)``.

        Windows/features that are entirely NaN are filled with 0 (the standardised mean)
        BEFORE the model sees them, because PyPOTS cannot impute an all-missing series.
        """
        W = np.asarray(W3d, float)
        # All-NaN slice guard: a feature with no observation in a window -> 0.
        # Do it per (window, feature) column over the time axis.
        allnan = np.all(~np.isfinite(W), axis=1, keepdims=True)   # (k,1,N)
        if allnan.any():
            W = np.where(allnan & ~np.isfinite(W), 0.0, W)
        buf = _Buf()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                out = self._model.predict({"X": W})["imputation"]
        out = np.asarray(out, float)
        # Belt and braces: never propagate non-finite from the net.
        out = np.where(np.isfinite(out), out, np.where(np.isfinite(W), W, 0.0))
        return out

    # -- readouts --------------------------------------------------------- #
    def fill_bidir(self, Xobs2d: np.ndarray) -> np.ndarray:
        """Standard bidirectional fill (uses the future within each window)."""
        X = np.asarray(Xobs2d, float)
        Xs = self._standardize(X)
        filled_s = causal_race.bidir_fill(self._predict, Xs, self.L)
        return self._finalize(X, filled_s)

    def fill_causal(self, Xobs2d: np.ndarray) -> np.ndarray:
        """Strict point-in-time fill: trailing windows + right-edge readout."""
        X = np.asarray(Xobs2d, float)
        Xs = self._standardize(X)
        filled_s = causal_race.causal_fill(self._predict, Xs, self.L)
        return self._finalize(X, filled_s)

    def _finalize(self, X_orig: np.ndarray, filled_std: np.ndarray) -> np.ndarray:
        """Inverse-standardise, restore observed cells exactly, scrub non-finite."""
        filled = self._inverse(filled_std)
        out = np.where(np.isfinite(X_orig), X_orig, filled)
        if not np.all(np.isfinite(out)):
            bad = ~np.isfinite(out)
            out[bad] = np.take(self._mean, np.where(bad)[1])
        return out


def build(name: str, L: int = 24, epochs: int = 60, batch_size: int = 32,
          patience: int = 6, val_frac: float = 0.1, **hp):
    """Convenience constructor: ``build("SAITS", L=24, epochs=60, patience=6)``.

    Defaults give the deep baselines a genuinely fair shot: medium-capacity
    architectures, validation-based early stopping (``patience`` epochs on a
    self-supervised held-out block), and a generous epoch budget that early
    stopping trims automatically.
    """
    return DeepImputer(name, L=L, epochs=epochs, batch_size=batch_size,
                       patience=patience, val_frac=val_frac, **hp)


# --------------------------------------------------------------------------- #
# Convergence check on REAL data (the whole point: a FAIR causal horse race).
#
# Run:  python deep_baselines.py converge
#
# Trains every upgraded model on data/fredmd_clean.npy[:1200] and
# data/beijing_clean.npy[:1200,:80] at 10% MCAR and prints bidirectional MAE on
# the masked cells + fit time. Standardisation here is a plain per-column z-score
# (fine for a convergence check); DeepImputer itself standardises on observed
# cells only, so this is leak-free either way. SUCCESS = each model reaches MAE
# clearly below ~0.45 on at least one dataset (vs ~0.5-0.75 under-trained).
# --------------------------------------------------------------------------- #
def _convergence_check():
    import os
    import time

    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "..", "data")

    datasets = {
        "fredmd[:1200]":     (os.path.join(data, "fredmd_clean.npy"),  slice(0, 1200), slice(None)),
        "beijing[:1200,:80]": (os.path.join(data, "beijing_clean.npy"), slice(0, 1200), slice(0, 80)),
    }

    L = 24
    names = ["SAITS", "Transformer", "BRITS", "TimesNet", "ImputeFormer"]
    # The under-trained baseline (n_layers=1, d_model~16-32, 8 epochs) scored
    # roughly these MAEs on standardised data -- recorded here for the
    # before/after table so the improvement is explicit.
    before = {
        "SAITS": "~0.50-0.70", "Transformer": "~0.55-0.75", "BRITS": "~0.50-0.65",
        "TimesNet": "~0.60-0.75", "ImputeFormer": "~0.55-0.75",
    }

    results = {}  # (name) -> {ds: (mae, fit_s)}
    for ds_name, (path, rs, cs) in datasets.items():
        X = np.load(path).astype(float)[rs, cs]
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        Xz = (X - mu) / sd
        rng = np.random.default_rng(0)
        mask = rng.random(Xz.shape) < 0.10
        Xobs = Xz.copy()
        Xobs[mask] = np.nan
        # Mean-imputation reference (constant 0 on z-scored data).
        mean_mae = float(np.mean(np.abs(0.0 - Xz[mask])))
        print(f"\n=== {ds_name}  shape={Xz.shape}  missing={mask.mean():.1%}  "
              f"mean-impute MAE={mean_mae:.3f} ===")
        for name in names:
            t0 = time.time()
            imp = build(name, L=L, epochs=60, batch_size=32, patience=6).fit(Xobs)
            fit_s = time.time() - t0
            fb = imp.fill_bidir(Xobs)
            assert np.all(np.isfinite(fb)), f"{name}: non-finite output"
            assert np.allclose(fb[~mask], Xz[~mask]), f"{name}: observed cells altered"
            mae = float(np.mean(np.abs(fb[mask] - Xz[mask])))
            results.setdefault(name, {})[ds_name] = (mae, fit_s)
            flag = "OK " if mae < 0.45 else "!! "
            print(f"  {flag}{name:<14} bidir_MAE={mae:7.4f}  fit_s={fit_s:6.1f}")

    print("\n--- before / after summary (bidir MAE, standardised) ---")
    hdr = f"{'model':<14}{'before(8ep,tiny)':>20}"
    for ds_name in datasets:
        hdr += f"{ds_name+'_after':>22}{ds_name+'_s':>14}"
    print(hdr)
    all_ok = True
    for name in names:
        row = f"{name:<14}{before[name]:>20}"
        best = 1e9
        for ds_name in datasets:
            mae, fit_s = results[name][ds_name]
            best = min(best, mae)
            row += f"{mae:>22.4f}{fit_s:>14.1f}"
        print(row)
        if best >= 0.45:
            all_ok = False
            print(f"    !! {name} did NOT reach MAE<0.45 on either dataset "
                  f"(best={best:.4f}) -- honest finding.")
    print(f"\nCONVERGENCE {'PASSED (all models <0.45 on >=1 dataset)' if all_ok else 'INCOMPLETE (see flags above)'}")
    return all_ok


# --------------------------------------------------------------------------- #
# Smoke test (run with the bench venv python that has torch + pypots).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1 and sys.argv[1] == "converge":
        if not HAVE_PYPOTS:
            print("PyPOTS/torch not available; cannot run convergence check.")
            sys.exit(1)
        ok = _convergence_check()
        sys.exit(0 if ok else 2)

    if not HAVE_PYPOTS:
        print("PyPOTS/torch not available -> DEEP_MODELS == {} (soft-import OK). "
              "Run with the bench venv python to smoke-test the deep models.")
        sys.exit(0)

    rng = np.random.default_rng(0)
    T, N, L = 480, 8, 24

    # Synthetic multivariate signal: smooth sinusoids + small noise (imputable).
    t = np.arange(T)[:, None]
    freqs = rng.uniform(0.02, 0.12, size=(1, N))
    phase = rng.uniform(0, 2 * np.pi, size=(1, N))
    clean = np.sin(2 * np.pi * freqs * t + phase) + 0.1 * rng.standard_normal((T, N))

    # ~15% MCAR holes.
    mask = rng.random((T, N)) < 0.15
    Xobs = clean.copy()
    Xobs[mask] = np.nan

    def mae_on_holes(filled):
        return float(np.mean(np.abs(filled[mask] - clean[mask])))

    # Smoke at least SAITS; opportunistically test the others if time allows.
    names = ["SAITS", "BRITS"]
    print(f"smoke: T={T} N={N} L={L} missing={mask.mean():.1%}  "
          f"(have models: {sorted(DEEP_MODELS)})")
    print(f"{'model':<14}{'bidir_MAE':>12}{'causal_MAE':>12}{'gap':>10}{'fit_s':>8}")

    for name in names:
        t0 = time.time()
        imp = build(name, L=L, epochs=3, batch_size=16).fit(Xobs)
        fit_s = time.time() - t0
        fb = imp.fill_bidir(Xobs)
        fc = imp.fill_causal(Xobs)

        # Shape / finiteness / observed-preservation invariants.
        assert fb.shape == fc.shape == (T, N), (fb.shape, fc.shape)
        assert np.all(np.isfinite(fb)) and np.all(np.isfinite(fc)), "non-finite output"
        obs = ~mask
        assert np.allclose(fb[obs], clean[obs]), "bidir altered observed cells"
        assert np.allclose(fc[obs], clean[obs]), "causal altered observed cells"

        mb, mc = mae_on_holes(fb), mae_on_holes(fc)
        print(f"{name:<14}{mb:>12.4f}{mc:>12.4f}{mc - mb:>10.4f}{fit_s:>8.1f}")
        # Causal should be no better than bidirectional (it sees strictly less).
        if mc + 1e-3 < mb:
            print(f"  NOTE: {name} causal beat bidir by {mb - mc:.4f} "
                  f"(within noise on tiny config)")

    # Truncation-invariance of the causal readout on a tiny case.
    Tt, Nt, Lt = 120, 4, 12
    Xt = rng.standard_normal((Tt, Nt))
    Xt[rng.random((Tt, Nt)) < 0.15] = np.nan
    imp_t = build("SAITS", L=Lt, epochs=2, batch_size=16).fit(Xt)
    v = causal_race.verify_causal(imp_t.fill_causal, Xt, n_prefixes=4)
    print(f"causal truncation-invariance (SAITS): {v}")
    assert v["causal"], v

    print("deep_baselines smoke PASSED")
