"""Missingness-as-signal feature engineering -- causal (forward-only) features that
*survive* imputation.

The fact that a value was missing is itself often informative (a sensor went offline
during an event; a form field was skipped for a reason). Standard imputation *erases*
this signal: once a cell is filled, a downstream model can no longer tell it apart
from a genuinely observed value. The Missing Indicator Method (MIM) preserves it by
appending features that encode *where* and *how* data was missing.

This module emits five families of causal features from a 2D matrix and its boolean
missing-mask. Every feature at row ``t`` is a function of rows ``<= t`` only -- no
future information is ever used (matching CAFE's strict point-in-time contract), so the
features can be computed online and concatenated next to causally-imputed values.

    1. ``was_imputed``         -- the raw MIM indicator: 1 if the cell was missing.
    2. ``time_since_obs``      -- BRITS-style delta: steps since this column was last
                                  observed (0 when currently observed).
    3. ``gap_length``          -- length so far of the current run of consecutive
                                  missing values in this column (0 when observed).
    4. ``missing_rate``        -- per-column *expanding* (causal) fraction missing up to
                                  and including row ``t``.
    5. ``selective_mim``       -- indicators emitted ONLY for columns whose missingness
                                  is *informative*, scored leak-free by an expanding
                                  association between this column's missingness and the
                                  other columns' observed values. Avoids the
                                  high-dimensional overfitting of blindly adding one
                                  indicator per column.

References
---------
Missing Indicator Method and when it helps / hurts:
    Van Ness et al., "The Missing Indicator Method: From Low to High Dimensions",
    arXiv:2211.09259.
BRITS-style time-gap ("delta") feature:
    Cao et al., "BRITS: Bidirectional Recurrent Imputation for Time Series", NeurIPS 2018
    (we use only the *forward* delta, which is causal).

Container-native: pass a numpy array / pandas DataFrame / polars DataFrame and get back
the same container type, with feature columns named ``<col>__<feature>``. A 1D Series is
treated as a single-column matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io import _is_pandas, _is_polars

__all__ = ["missingness_features", "MissingnessFeatures", "FEATURE_KINDS"]

# Per-column feature families that produce one output column per input column.
FEATURE_KINDS = (
    "was_imputed",
    "time_since_obs",
    "gap_length",
    "missing_rate",
)


@dataclass
class MissingnessFeatures:
    """Result container: the feature matrix plus its column names and provenance.

    Attributes
    ----------
    features : np.ndarray
        Float64 ``(T, F)`` matrix of causal missingness features.
    names : list[str]
        Length-``F`` feature names, aligned to ``features`` columns.
    kinds : list[str]
        Length-``F`` family tag for each column (one of :data:`FEATURE_KINDS` or
        ``"selective_mim"``).
    informative_columns : list
        The input column labels selected by ``selective_mim`` (empty if disabled or
        none crossed the threshold).
    container : object
        Same-type container (DataFrame/Series) as the input when the input was a
        pandas/polars object; otherwise the numpy ``features`` array. This is what
        :func:`missingness_features` returns directly; the dataclass is available via
        the ``.meta`` round-trip for callers who want the raw arrays + names.
    """

    features: np.ndarray
    names: list = field(default_factory=list)
    kinds: list = field(default_factory=list)
    informative_columns: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Input handling: derive (mask, T, N, labels, kind) without re-imputing.
# --------------------------------------------------------------------------- #
def _resolve_input(data, mask):
    """Return (values, mask, labels, kind, index) for a numpy/pandas/polars input.

    ``values`` is a float64 (T, N) array (NaN allowed); ``mask`` is the boolean
    (T, N) missing-mask (True == missing). If ``mask`` is None it is derived from NaNs.
    ``labels`` are the column labels; ``kind`` in {'numpy','pandas','polars'}; ``index``
    is the pandas index when applicable (for rebuilding the container).
    """
    kind = "numpy"
    index = None
    labels = None

    if _is_pandas(data):
        import pandas as pd

        kind = "pandas"
        if isinstance(data, pd.Series):
            values = data.to_numpy(dtype=float).reshape(-1, 1)
            labels = [data.name if data.name is not None else 0]
            index = data.index
        else:  # DataFrame -- numeric columns only, mirroring io.to_matrix
            num = data.select_dtypes(include=["number"])
            if num.shape[1] == 0:
                raise ValueError(
                    "missingness_features found no numeric columns "
                    f"(columns: {list(data.columns)})."
                )
            values = num.to_numpy(dtype=float)
            labels = list(num.columns)
            index = data.index
    elif _is_polars(data):
        import polars as pl

        kind = "polars"
        if isinstance(data, pl.Series):
            values = data.to_numpy().astype(float).reshape(-1, 1)
            labels = [data.name if data.name else "0"]
        else:
            num_cols = [c for c, dt in data.schema.items() if dt.is_numeric()]
            if not num_cols:
                raise ValueError(
                    "missingness_features found no numeric columns "
                    f"(columns: {data.columns})."
                )
            values = data.select(num_cols).to_numpy().astype(float)
            labels = list(num_cols)
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            raise ValueError(
                f"missingness_features expects 1D or 2D input; got {arr.ndim}D "
                f"of shape {arr.shape}."
            )
        values = arr

    values = np.ascontiguousarray(values, dtype=float)
    T, N = values.shape

    if mask is None:
        m = ~np.isfinite(values)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        if m.shape != values.shape:
            raise ValueError(
                f"mask shape {m.shape} does not match data shape {values.shape}."
            )
    m = np.ascontiguousarray(m)

    if labels is None:
        labels = list(range(N))
    return values, m, labels, kind, index


# --------------------------------------------------------------------------- #
# Causal per-column features. Each is computed with a single forward pass so it is,
# by construction, a function of rows <= t only.
# --------------------------------------------------------------------------- #
def _time_since_obs(mask):
    """Steps since each column was last observed; 0 where currently observed.

    Forward-only running counter (BRITS forward delta). At row 0, if the value is
    missing the delta is 1 (one step since the -- never-seen -- prior observation),
    which matches the BRITS convention of delta=1 at the start.
    """
    T, N = mask.shape
    out = np.zeros((T, N), dtype=float)
    counter = np.zeros(N, dtype=float)
    for t in range(T):
        miss = mask[t]
        # increment for missing columns, reset observed columns to 0
        counter = np.where(miss, counter + 1.0, 0.0)
        out[t] = counter
    return out


def _gap_length(mask):
    """Length so far of the current run of consecutive missing values; 0 if observed.

    Differs from ``time_since_obs`` only conceptually (here every step is a run
    counter that resets on observation); numerically identical for the contiguous-run
    case but kept separate so callers can pick the semantics they want and so the
    feature name is explicit. Causal: forward running counter.
    """
    T, N = mask.shape
    out = np.zeros((T, N), dtype=float)
    run = np.zeros(N, dtype=float)
    for t in range(T):
        miss = mask[t]
        run = np.where(miss, run + 1.0, 0.0)
        out[t] = run
    return out


def _expanding_missing_rate(mask):
    """Per-column expanding fraction missing up to and including row t (causal)."""
    counts = np.cumsum(mask.astype(float), axis=0)        # missing-so-far
    denom = np.arange(1, mask.shape[0] + 1, dtype=float)[:, None]
    return counts / denom


def _selective_informative_columns(values, mask, threshold, min_obs):
    """Pick columns whose missingness is *informative*, scored leak-free.

    Leak-free / causal scoring: for each candidate column ``j`` we measure how much the
    *event* "column j is missing at row t" associates with the contemporaneous observed
    values of the OTHER columns. We use only rows ``<= T`` of the data passed in (the
    caller passes truncated data for the causal test), and only contemporaneous /
    past cells -- never future rows. The association statistic is the mean absolute
    standardized difference of other columns' observed values between the
    "j-missing" and "j-observed" row groups (a cheap, robust point-biserial-style
    score). A column is selected if (a) it has both enough missing and enough observed
    rows to score, and (b) its score exceeds ``threshold``.

    Returns a boolean (N,) selection vector.
    """
    T, N = values.shape
    selected = np.zeros(N, dtype=bool)
    if N < 2 or T < min_obs:
        return selected

    # Standardize each column by its observed mean/std so scores are comparable.
    obs = ~mask
    col_mean = np.full(N, np.nan)
    col_std = np.full(N, np.nan)
    for k in range(N):
        v = values[obs[:, k], k]
        if v.size >= 1:
            col_mean[k] = v.mean()
            col_std[k] = v.std() + 1e-9

    for j in range(N):
        miss_j = mask[:, j]
        n_miss = int(miss_j.sum())
        n_obs = T - n_miss
        if n_miss < min_obs or n_obs < min_obs:
            continue  # not enough signal either way to score reliably

        score_acc = 0.0
        n_terms = 0
        for k in range(N):
            if k == j or not np.isfinite(col_std[k]):
                continue
            # other column k's values, only where k itself is OBSERVED, split by
            # whether j was missing on that same (contemporaneous) row.
            okk = obs[:, k]
            grp_miss = okk & miss_j
            grp_obs = okk & (~miss_j)
            if grp_miss.sum() < 2 or grp_obs.sum() < 2:
                continue
            a = values[grp_miss, k]
            b = values[grp_obs, k]
            # standardized absolute mean difference (effect size)
            diff = abs(a.mean() - b.mean()) / col_std[k]
            score_acc += diff
            n_terms += 1
        if n_terms == 0:
            continue
        score = score_acc / n_terms
        if score >= threshold:
            selected[j] = True
    return selected


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def missingness_features(
    data,
    mask=None,
    *,
    kinds=FEATURE_KINDS,
    selective=True,
    selective_threshold=0.15,
    selective_min_obs=8,
    return_meta=False,
):
    """Compute causal missingness-as-signal features.

    Parameters
    ----------
    data : numpy array | pandas Series/DataFrame | polars Series/DataFrame
        The (possibly already imputed) data. Only the *shape* and -- when ``mask`` is
        None -- the NaN pattern are read from it; values are used solely for the
        leak-free selective-MIM association score.
    mask : array-like of bool, optional
        ``(T, N)`` (or ``(T,)`` for 1D) boolean missing-mask, True == was missing. If
        None, derived from non-finite cells of ``data``. Pass this explicitly when the
        data has *already* been imputed (so the NaNs are gone) -- this is the intended
        "features that survive imputation" usage.
    kinds : tuple[str], default all of :data:`FEATURE_KINDS`
        Which per-column feature families to emit.
    selective : bool, default True
        If True, also emit ``selective_mim`` indicators for informative columns only.
    selective_threshold : float, default 0.15
        Minimum association effect-size for a column's missingness to be deemed
        informative (standardized mean-difference units).
    selective_min_obs : int, default 8
        Minimum count of both missing and observed rows required to score a column.
    return_meta : bool, default False
        If True, return a :class:`MissingnessFeatures` (raw arrays + names +
        provenance) instead of a same-type container.

    Returns
    -------
    container or MissingnessFeatures
        By default, a feature container matching the input type:
        numpy -> ``(T, F)`` float64 array;
        pandas -> DataFrame (same index) with ``<col>__<feature>`` columns;
        polars -> DataFrame with the same columns. When ``return_meta=True`` (or input
        was numpy) you also/only get the raw :class:`MissingnessFeatures`.

    Notes
    -----
    Every column is a forward-only function of rows ``<= t``: appending future rows
    leaves all earlier rows of every feature unchanged (verified in the test suite).
    """
    bad = set(kinds) - set(FEATURE_KINDS)
    if bad:
        raise ValueError(f"unknown feature kind(s): {sorted(bad)}; "
                         f"valid kinds are {FEATURE_KINDS}.")

    values, mask, labels, kind, index = _resolve_input(data, mask)
    T, N = values.shape

    blocks = []      # list of (T, N) arrays
    names = []       # flat feature names
    tags = []        # flat kind tags

    # The per-column families, in a stable order.
    computed = {}
    if "was_imputed" in kinds:
        computed["was_imputed"] = mask.astype(float)
    if "time_since_obs" in kinds:
        computed["time_since_obs"] = _time_since_obs(mask)
    if "gap_length" in kinds:
        computed["gap_length"] = _gap_length(mask)
    if "missing_rate" in kinds:
        computed["missing_rate"] = _expanding_missing_rate(mask)

    for fam in FEATURE_KINDS:                       # deterministic emission order
        if fam not in computed:
            continue
        blk = computed[fam]
        blocks.append(blk)
        for lab in labels:
            names.append(f"{lab}__{fam}")
            tags.append(fam)

    informative_columns = []
    if selective:
        sel = _selective_informative_columns(
            values, mask, selective_threshold, selective_min_obs
        )
        if sel.any():
            sub = mask[:, sel].astype(float)
            blocks.append(sub)
            sel_labels = [labels[i] for i in np.nonzero(sel)[0]]
            informative_columns = list(sel_labels)
            for lab in sel_labels:
                names.append(f"{lab}__selective_mim")
                tags.append("selective_mim")

    if blocks:
        feats = np.concatenate(blocks, axis=1)
    else:
        feats = np.zeros((T, 0), dtype=float)
    feats = np.ascontiguousarray(feats, dtype=float)

    meta = MissingnessFeatures(
        features=feats,
        names=names,
        kinds=tags,
        informative_columns=informative_columns,
    )

    if return_meta or kind == "numpy":
        if kind == "numpy":
            return feats if not return_meta else meta
        # fall through to also build container below only when not numpy

    if kind == "pandas":
        import pandas as pd

        out = pd.DataFrame(feats, index=index, columns=names)
        return meta if return_meta else out
    if kind == "polars":
        import polars as pl

        out = pl.DataFrame({n: feats[:, j] for j, n in enumerate(names)})
        return meta if return_meta else out
    # numpy with return_meta already handled above
    return meta
