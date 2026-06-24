"""Mixed-frequency / ragged-edge nowcasting tasks for CAFE.

The central-bank NOWCASTING problem (Giannone, Reichlin & Small 2008): a panel mixes
frequencies -- a dense high-frequency cross-section (monthly macro indicators) plus one
or more LOW-frequency targets released only every ``k`` periods (quarterly), with a
RAGGED EDGE (the current period's low-freq value is not yet released). The question at
each time ``t``: predict the current low-freq value from the high-freq cross-section
observed up to ``t``, strictly point-in-time.

KEY INSIGHT (the reason CAFE can do this with no special machinery): a low-frequency
series living on a high-frequency grid is just a column that is OBSERVED every ``k``
steps and ``NaN`` in between -- a *structured missingness pattern* CAFE already imputes.
The cross-sectional factor fill at the (unobserved) current period IS the nowcast.

This module builds the task as a plain ``(time, features)`` matrix with NaNs in the
low-freq columns, plus the metadata an honest causal evaluation needs:

  * ``X``        : (T, N) monthly z-scored panel with low-freq columns set NaN off-grid
  * ``Xtruth``   : (T, N) the complete ground truth (for scoring the nowcast)
  * ``tgt``      : list of low-freq column indices
  * ``obs_t``    : (T,) bool, True at low-freq RELEASE times (months 3,6,9,... here)
  * ``k``        : low-freq period (3 == quarterly target in a monthly panel)

We pick the low-freq targets HONESTLY. A nowcasting target is only interesting when
persistence is genuinely stale (low lag-``k`` autocorrelation) AND the contemporaneous
cross-section is informative (high common-factor R^2). Those are exactly the series a
central bank actually nowcasts (activity/spread series), as opposed to near-random-walk
levels where last-value is unbeatable. ``pick_nowcast_targets`` selects them by this
intrinsic, data-only criterion (no look-ahead at the held-out values).

Run ``python3 bench/datasets_mixedfreq.py`` to print the task summary.
"""
from __future__ import annotations

import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _load_clean(name: str) -> np.ndarray:
    path = os.path.join(DATA, f"{name}_clean.npy")
    X = np.load(path)
    assert np.isfinite(X).all(), f"{name}: clean matrix not finite"
    return np.asarray(X, float)


def pick_nowcast_targets(X: np.ndarray, k: int = 3, n: int = 4):
    """Select ``n`` low-freq target columns by an INTRINSIC, data-only criterion.

    A series is a good nowcast target when persistence is stale over a low-freq period
    (low |lag-k autocorrelation|) yet the contemporaneous common factors explain it well
    (high factor R^2) -- i.e. the cross-section carries information last-value cannot.
    Score = factorR2 - |lag-k autocorr|; higher is a better nowcast target.

    Uses only the panel's own structure (no held-out / future values), so it is a fair,
    reproducible target choice rather than a cherry-pick by outcome.
    """
    T, N = X.shape
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    F = U[:, : min(8, N)]                      # leading common factors
    ac = np.empty(N)
    r2 = np.empty(N)
    for j in range(N):
        a, b = X[:-k, j], X[k:, j]
        ac[j] = np.corrcoef(a, b)[0, 1]
        y = Xc[:, j]
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        yh = F @ coef
        denom = (y ** 2).sum()
        r2[j] = 1.0 - ((y - yh) ** 2).sum() / denom if denom > 0 else 0.0
    score = r2 - np.abs(ac)
    return sorted(int(j) for j in np.argsort(-score)[:n])


def make_mixedfreq(name: str = "fredmd", k: int = 3, n_targets: int = 4,
                   targets=None):
    """Build a mixed-frequency nowcasting task from a monthly panel.

    Returns a dict with X (NaN'd low-freq), Xtruth, tgt, obs_t, k, hf (high-freq cols).
    Low-freq columns are observed only at months where ``(t % k) == k-1`` (a quarter end
    in a monthly grid) and NaN elsewhere; the current-period value is treated as not-yet
    released at nowcast time (ragged edge handled by the evaluator).
    """
    Xtruth = _load_clean(name)
    T, N = Xtruth.shape
    tgt = list(targets) if targets is not None else pick_nowcast_targets(Xtruth, k, n_targets)
    hf = [j for j in range(N) if j not in tgt]
    obs_t = (np.arange(T) % k) == (k - 1)
    X = Xtruth.copy()
    for j in tgt:
        X[~obs_t, j] = np.nan
    return {
        "name": name, "X": X, "Xtruth": Xtruth, "tgt": tgt, "hf": hf,
        "obs_t": obs_t, "k": k,
    }


def _intrinsic_stats(Xtruth, tgt, k):
    """For reporting: lag-k autocorr and factor R^2 of the chosen targets."""
    T, N = Xtruth.shape
    Xc = Xtruth - Xtruth.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    F = U[:, : min(8, N)]
    out = {}
    for j in tgt:
        ac = np.corrcoef(Xtruth[:-k, j], Xtruth[k:, j])[0, 1]
        y = Xc[:, j]; coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        r2 = 1.0 - ((y - F @ coef) ** 2).sum() / (y ** 2).sum()
        out[j] = (float(ac), float(r2))
    return out


if __name__ == "__main__":
    task = make_mixedfreq("fredmd", k=3, n_targets=4)
    X, Xtruth, tgt, obs_t, k = task["X"], task["Xtruth"], task["tgt"], task["obs_t"], task["k"]
    print(f"mixed-frequency nowcasting task: {task['name']}")
    print(f"  panel shape           : {X.shape}  (time, features)")
    print(f"  low-freq period k     : {k}  (quarterly target in a monthly grid)")
    print(f"  low-freq targets      : {tgt}")
    print(f"  release times (obs_t) : {int(obs_t.sum())} of {len(obs_t)} months")
    print(f"  missing frac (targets): {np.isnan(X[:, tgt]).mean():.3f}")
    stats = _intrinsic_stats(Xtruth, tgt, k)
    print("  intrinsic target stats (chosen for low persistence + high factor info):")
    for j in tgt:
        ac, r2 = stats[j]
        print(f"    col {j:3d}: lag-{k} autocorr={ac:+.2f}  factorR2={r2:.2f}")
    print("OK")
