"""Per-cell RECOVERABILITY CERTIFICATE for CAFE -- risk-controlled / selective imputation.

CAFE's extreme-missingness study (``bench/exp_extreme_missing.py``) mapped a *recoverability
frontier*: a near-empty feature whose loading lies in the span of the surviving cross-section
is reconstructed (and the margin grows with anchors), while an idiosyncratic column with few
anchors is NOT -- the best any mechanism-free method can do there is fall back to the level.
The honest behaviour is to fall back; the *useful* behaviour is to also say so.

This module turns CAFE's own internal state into a per-cell **recoverability score** in
``[0, 1]`` (1 = trustworthy fill, 0 = "CAFE cannot recover this cell"), built ONLY from
quantities the model already computes in its single causal forward pass:

  * the per-cell posterior predictive sigma (``res.uncertainty``);
  * the causal split-conformal scale ``q[t]`` (``conformal.py``) -- so the certificate is
    expressed on a coverage-calibrated, not a raw-Gaussian, error scale;
  * the cross-sectional ANCHOR support behind the cell (how many features were observed in
    that contemporaneous row, and how recently THIS feature itself was last seen);
  * the factor / effective-rank SUPPORT for the feature (its loading energy ``||W_j||`` and
    the active-factor prior variances ``s_fac`` -- a feature with negligible loading is not
    factor-spanned, so the cross-section carries no evidence for it);
  * the Student-t robustness weight ``row_w`` (a row the model itself flags as an outlier is
    a row whose fills should be trusted less).

The score is DESIGNED so that a thin ``CafeResult`` method can expose
``res.recoverability_score()`` and ``res.selective_imputed(min_confidence=tau)`` (the fill with
NaN wherever the score ``< tau`` -- "abstain" on cells CAFE cannot recover). numpy-only.

The central claim the certificate must EARN (validated in ``bench/exp_recoverability.py``):
*a lower certificate genuinely means higher realized error*, so gating fills by the
certificate (abstaining on the least-confident cells) drops error on the retained cells far
faster than abstaining at random. Whether that holds is an empirical question; the bench
reports it honestly, including where the certificate is only weakly calibrated.
"""
from __future__ import annotations

import numpy as np

__all__ = ["recoverability_score", "selective_impute", "certificate_components",
           "score_from_result"]

# A small floor so a perfectly-observed (sigma->0) cross-section does not divide by zero, and
# so the score is finite everywhere.
_EPS = 1e-9


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def certificate_components(
    *,
    sigma: np.ndarray,
    miss: np.ndarray,
    obs_mask: np.ndarray | None = None,
    q_grid: np.ndarray | None = None,
    row_w: np.ndarray | None = None,
    loading_energy: np.ndarray | None = None,
    active_rank: int | None = None,
) -> dict:
    """Compute the named per-cell certificate sub-signals (all in ``[0, 1]``, 1 = good).

    Every input is read from CAFE's own forward pass; nothing here looks at held-out truth.

    Parameters
    ----------
    sigma : (T, N)
        Per-cell posterior predictive std (``sqrt(cvar)``). NaN where observed.
    miss : (T, N) bool
        True where the cell was imputed (the cells the certificate scores).
    obs_mask : (T, N) bool, optional
        True where a cell was OBSERVED in the input (the contemporaneous cross-section). Used
        for the anchor-count and time-since-observed (recency) signals. If omitted these
        signals are neutral (1.0).
    q_grid : (T, N), optional
        Causal split-conformal multiplier ``q[t]`` (broadcast over features). The certificate
        is built on the calibrated scale ``q * sigma`` when available, else on raw ``sigma``.
    row_w : (T,), optional
        Student-t IRLS row weight; rows the model flags as outliers (``row_w`` well below 1)
        get a confidence penalty.
    loading_energy : (N,), optional
        Per-feature loading energy ``||W_j||`` (factor-spanned support). Low energy => the
        feature is idiosyncratic and the cross-section carries no evidence for it.
    active_rank : int, optional
        The model's effective rank (number of factors ARD keeps). When 0 the factor channel
        carries no information, so the cross-section / anchor terms get all the weight.

    Returns
    -------
    dict of (T, N) float arrays, each in ``[0, 1]`` (NaN at observed cells):
    ``c_scale`` (calibrated-sigma confidence), ``c_anchor`` (cross-sectional anchor support),
    ``c_recency`` (how recently this feature was seen), ``c_span`` (factor-spanned support),
    ``c_robust`` (row robustness).
    """
    sigma = np.asarray(sigma, float)
    miss = np.asarray(miss, bool)
    T, N = sigma.shape
    nanmask = ~miss  # observed cells: certificate undefined -> NaN

    # --- (1) calibrated-sigma confidence -----------------------------------------------
    # The dominant predictor of realized error is the model's own predictive width. Put it
    # on the conformal-calibrated scale q*sigma when we have q (so the certificate inherits
    # the coverage calibration). Map "small width => high confidence" with a soft, scale-free
    # transform anchored on the cross-sectional median width of THIS run's imputed cells, so
    # the certificate is comparable across datasets of different units.
    eff = sigma.copy()
    if q_grid is not None:
        eff = eff * np.asarray(q_grid, float)
    med = np.nanmedian(eff[miss]) if miss.any() else 1.0
    med = float(med) if np.isfinite(med) and med > _EPS else 1.0
    # ratio>1 => wider than typical => lower confidence; log keeps it symmetric & bounded.
    c_scale = _sigmoid(-np.log((eff + _EPS) / med))

    # --- (2) cross-sectional anchor support --------------------------------------------
    # More contemporaneously observed features in a row = more equations pinning the latent
    # state = a better-supported fill. Normalise by N; saturate at a modest fraction so a
    # half-observed row is already "well anchored".
    if obs_mask is not None:
        obs_mask = np.asarray(obs_mask, bool)
        n_anchor = obs_mask.sum(axis=1).astype(float)        # (T,) anchors that row
        frac = n_anchor / max(N, 1)
        c_anchor = np.clip(frac / 0.25, 0.0, 1.0)[:, None] * np.ones((1, N))
    else:
        c_anchor = np.ones((T, N))

    # --- (3) recency: how long since THIS feature was last observed ---------------------
    # A long contiguous gap is extrapolated and decays toward the prior; the longer the gap
    # the less recoverable. time-since-last-observed is strictly causal (past-only).
    if obs_mask is not None:
        tsl = np.empty((T, N), float)
        last = np.full(N, -1, int)
        for t in range(T):
            gap = np.where(last >= 0, t - last, t + 1).astype(float)
            tsl[t] = gap
            last[obs_mask[t]] = t
        # decay confidence with gap length; half-confidence at ~8 steps (matches the AR/idio
        # forecast-variance horizon the core uses), floored so a long gap never reads as 0.
        c_recency = np.clip(0.2 + 0.8 * np.exp(-tsl / 8.0), 0.0, 1.0)
    else:
        c_recency = np.ones((T, N))

    # --- (4) factor-spanned support -----------------------------------------------------
    # A feature with negligible loading energy is idiosyncratic: the surviving cross-section
    # carries no evidence for it (the recoverability frontier). Rank features by relative
    # loading energy within the run; if the model kept no active factors (rank 0) this channel
    # is uninformative and we leave it neutral.
    if loading_energy is not None and (active_rank is None or active_rank > 0):
        le = np.asarray(loading_energy, float)
        le = np.where(np.isfinite(le), le, 0.0)
        med_le = np.median(le[le > 0]) if np.any(le > 0) else 1.0
        med_le = float(med_le) if med_le > _EPS else 1.0
        c_span_feat = _sigmoid(np.log((le + _EPS) / med_le))   # (N,)
        c_span = c_span_feat[None, :] * np.ones((T, 1))
    else:
        c_span = np.ones((T, N))

    # --- (5) row robustness -------------------------------------------------------------
    # row_w = (nu+1)/(nu+u); ~1 for a clean row, ->0 for a flagged outlier row. A fill made on
    # a row the model itself distrusts should be trusted less.
    if row_w is not None:
        rw = np.clip(np.asarray(row_w, float), 0.0, 1.0)
        c_robust = rw[:, None] * np.ones((1, N))
    else:
        c_robust = np.ones((T, N))

    out = dict(c_scale=c_scale, c_anchor=c_anchor, c_recency=c_recency,
               c_span=c_span, c_robust=c_robust)
    for k in out:
        out[k] = np.where(nanmask, np.nan, np.clip(out[k], 0.0, 1.0))
    return out


# Default channel weights (geometric mean exponents). The calibrated-sigma channel is the
# primary signal; the structural channels (anchor / span / recency / robust) modulate it.
_DEFAULT_WEIGHTS = dict(c_scale=1.0, c_anchor=0.5, c_recency=0.5, c_span=0.7, c_robust=0.4)


def recoverability_score(
    *,
    sigma: np.ndarray,
    miss: np.ndarray,
    obs_mask: np.ndarray | None = None,
    q_grid: np.ndarray | None = None,
    row_w: np.ndarray | None = None,
    loading_energy: np.ndarray | None = None,
    active_rank: int | None = None,
    weights: dict | None = None,
    return_components: bool = False,
):
    """Per-cell recoverability score in ``[0, 1]`` (1 = trustworthy fill, 0 = unrecoverable).

    Combines the sub-signals from :func:`certificate_components` as a WEIGHTED GEOMETRIC MEAN
    (a logical AND: a cell is only trustworthy if its predictive width is small AND it has
    cross-sectional support AND its feature is factor-spanned AND the row is not an outlier --
    any single channel collapsing pulls the score down). numpy-only; NaN at observed cells.

    Returns the ``(T, N)`` score, or ``(score, components)`` if ``return_components``.
    """
    comp = certificate_components(
        sigma=sigma, miss=miss, obs_mask=obs_mask, q_grid=q_grid, row_w=row_w,
        loading_energy=loading_energy, active_rank=active_rank)
    w = dict(_DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    miss = np.asarray(miss, bool)
    logsum = np.zeros(sigma.shape, float)
    wsum = 0.0
    for k, wk in w.items():
        if wk == 0.0:
            continue
        ck = np.where(miss, np.clip(comp[k], _EPS, 1.0), 1.0)
        logsum = logsum + wk * np.log(ck)
        wsum += wk
    score = np.exp(logsum / max(wsum, _EPS))
    score = np.where(miss, np.clip(score, 0.0, 1.0), np.nan)
    if return_components:
        return score, comp
    return score


def score_from_result(res, *, conformal: bool = True, weights: dict | None = None,
                      return_components: bool = False, **conformal_kwargs):
    """Extract every certificate signal from a traced ``CafeResult`` and return the per-cell
    recoverability score (NaN at observed cells). This is the single place that maps CAFE's
    internal state onto :func:`recoverability_score`, so a thin ``CafeResult`` method can wrap
    it (``res.recoverability_score()``). Needs the traced run (``CAFE().run`` on 1D/2D).

    ``conformal=True`` puts the certificate on the causal split-conformal scale (``q*sigma``);
    extra kwargs (``cal_rate``, ``cal_seed``, ``window``, ``min_scores``, ``level``) are
    forwarded to the calibrator. Set ``conformal=False`` for the raw-sigma certificate (no
    second calibration pass)."""
    if getattr(res, "_trace", None) is None or len(res._trace) == 0:
        raise NotImplementedError(
            "recoverability_score needs the traced run; use CAFE().run(...) on 1D/2D data.")
    C = res._comp()
    sigma = np.sqrt(C["cvar"])
    X = np.asarray(res._X, float)
    miss = ~np.isfinite(X)
    obs_mask = np.isfinite(X)
    row_w = C["row_w"]
    core = res._core
    loading_energy = (np.sqrt(np.sum(np.asarray(core.W, float) ** 2, axis=1))
                      if core is not None and getattr(core, "W", None) is not None else None)
    active_rank = res.effective_rank()
    q_grid = None
    if conformal:
        level = conformal_kwargs.pop("level", 0.90)
        try:
            q_grid = res._conformal_multiplier_grid(
                level,
                conformal_kwargs.pop("cal_rate", 0.15),
                conformal_kwargs.pop("cal_seed", 0),
                conformal_kwargs.pop("window", 4000),
                conformal_kwargs.pop("min_scores", 30))
        except Exception:                       # calibration unavailable -> raw-sigma scale
            q_grid = None
    return recoverability_score(
        sigma=sigma, miss=miss, obs_mask=obs_mask, q_grid=q_grid, row_w=row_w,
        loading_energy=loading_energy, active_rank=active_rank, weights=weights,
        return_components=return_components)


def selective_impute(filled: np.ndarray, score: np.ndarray, min_confidence: float):
    """Return ``filled`` with NaN wherever the recoverability ``score < min_confidence``
    (abstain on cells CAFE cannot recover). Observed cells (score NaN) are always kept.

    A downstream user calls this with a risk tolerance ``tau``; lower ``tau`` keeps more (and
    riskier) fills, higher ``tau`` abstains on more cells. Pure numpy; ``filled`` is unchanged
    where retained."""
    filled = np.asarray(filled, float)
    score = np.asarray(score, float)
    abstain = np.isfinite(score) & (score < float(min_confidence))
    out = filled.copy()
    out[abstain] = np.nan
    return out
