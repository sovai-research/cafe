"""Container adapters: accept numpy / pandas / polars (1D or 2D), run the model on a
plain float matrix, and rebuild the *same* container type, dtype and labels on the way
out. Optional dependencies (pandas, polars) are detected lazily and never imported
unless the user actually passes one of their objects.

Point-and-shoot rule: a DataFrame may freely mix numeric and non-numeric columns
(a ``date`` column, string ids, categories). Only the numeric columns are imputed;
every non-numeric column is passed through untouched and the original column order is
restored on the way out. This is what makes ``cafe.impute(df)`` work on a real, raw
DataFrame with no manual column selection.
"""
from __future__ import annotations

import numpy as np

__all__ = ["to_matrix", "from_matrix", "Ctx"]


class Ctx:
    """Round-trip context describing the original container."""
    __slots__ = ("kind", "was_1d", "index", "columns", "name", "dtypes", "schema",
                 "order", "passthrough", "panel_meta", "panel_shape")

    def __init__(self, kind, was_1d=False, index=None, columns=None,
                 name=None, dtypes=None, schema=None, order=None, passthrough=None,
                 panel_meta=None, panel_shape=None):
        self.kind = kind            # 'numpy' | 'pandas' | 'polars'
        self.was_1d = was_1d
        self.index = index
        self.columns = columns      # the numeric columns that were imputed
        self.name = name
        self.dtypes = dtypes
        self.schema = schema
        self.order = order          # original full column order (for reassembly)
        self.passthrough = passthrough or []   # [(name, original_series), ...]
        # Panel on-ramp: when the input carries an (entity, time) structure -- a 3D
        # (entity x time x feature) array, or a long-format DataFrame with two index
        # columns -- ``panel_meta`` holds the derived {entity_ids, time_ids} so the
        # caller can route to the panel core without the user hand-building integer ids.
        # ``panel_shape`` records the original 3D shape so :func:`from_matrix` can rebuild it.
        self.panel_meta = panel_meta
        self.panel_shape = panel_shape


def _codes(labels, ordered):
    """Map a sequence of labels to dense integer codes. ``ordered=True`` (time axis)
    sorts the unique labels so codes respect chronological order; ``ordered=False``
    (entity axis) preserves first-appearance order. Pure-numpy, no pandas dependency."""
    labels = list(labels)
    uniq = sorted(set(labels)) if ordered else list(dict.fromkeys(labels))
    idx = {v: i for i, v in enumerate(uniq)}
    return np.fromiter((idx[v] for v in labels), dtype=np.intp, count=len(labels))


def _panel_meta(time_labels, entity_labels):
    """Build the core's {entity_ids, time_ids} from raw (time, entity) label columns."""
    return {"entity_ids": _codes(entity_labels, ordered=False),
            "time_ids": _codes(time_labels, ordered=True)}


def _is_pandas(x):
    # pandas < 3 reports submodules like "pandas.core.frame"; pandas 3.0 reports the
    # top-level "pandas" (no trailing dot) for DataFrame/Series, so match both forms.
    m = type(x).__module__
    return m == "pandas" or m.startswith("pandas.")


def _is_polars(x):
    m = type(x).__module__
    return m == "polars" or m.startswith("polars.")


def to_matrix(data, panel=None):
    """Return (X, ctx): X is a contiguous float64 (T, N) array with NaN for missing.

    For DataFrames, only numeric columns become X; non-numeric columns are remembered
    on the ctx and passed through unchanged by :func:`from_matrix`.

    Panel on-ramp
    -------------
    ``panel=(time_col, entity_col)`` marks a long-format DataFrame as panel data: those
    two columns become the entity/time index (excluded from the imputed features) and
    ``ctx.panel_meta`` carries the derived integer ids. A 3D ``(entity, time, feature)``
    numpy array is detected automatically and flattened to a stacked matrix; its original
    shape is recorded so :func:`from_matrix` rebuilds the 3D array.
    """
    # ---- pandas ----
    if _is_pandas(data):
        import pandas as pd
        if panel is not None and isinstance(data, pd.DataFrame):
            return _to_matrix_panel_df(data, panel, kind="pandas")
        if isinstance(data, pd.Series):
            if not pd.api.types.is_numeric_dtype(data):
                raise TypeError(
                    f"CAFE can only impute numeric data; got a pandas Series of dtype "
                    f"{data.dtype!r}. Convert it to a numeric dtype first."
                )
            X = data.to_numpy(dtype=float).reshape(-1, 1)
            return np.ascontiguousarray(X), Ctx("pandas", True, data.index, None,
                                                data.name, [data.dtype])
        if isinstance(data, pd.DataFrame):
            num = data.select_dtypes(include=["number"])
            if num.shape[1] == 0:
                raise ValueError(
                    "CAFE found no numeric columns to impute in this DataFrame "
                    f"(columns: {list(data.columns)}). At least one numeric column "
                    "is required."
                )
            num_set = set(num.columns)
            passthrough = [(c, data[c]) for c in data.columns if c not in num_set]
            X = num.to_numpy(dtype=float)
            return np.ascontiguousarray(X), Ctx(
                "pandas", False, data.index, list(num.columns), None,
                list(num.dtypes), order=list(data.columns), passthrough=passthrough,
            )
    # ---- polars ----
    if _is_polars(data):
        import polars as pl
        if panel is not None and isinstance(data, pl.DataFrame):
            return _to_matrix_panel_df(data, panel, kind="polars")
        if isinstance(data, pl.Series):
            if not data.dtype.is_numeric():
                raise TypeError(
                    f"CAFE can only impute numeric data; got a polars Series of dtype "
                    f"{data.dtype}. Cast it to a numeric dtype first."
                )
            X = data.to_numpy().astype(float).reshape(-1, 1)
            return np.ascontiguousarray(X), Ctx("polars", True, None, None,
                                                data.name, None)
        if isinstance(data, pl.DataFrame):
            num_cols = [c for c, dt in data.schema.items() if dt.is_numeric()]
            if not num_cols:
                raise ValueError(
                    "CAFE found no numeric columns to impute in this DataFrame "
                    f"(columns: {data.columns}). At least one numeric column "
                    "is required."
                )
            num_set = set(num_cols)
            passthrough = [(c, data[c]) for c in data.columns if c not in num_set]
            X = data.select(num_cols).to_numpy().astype(float)
            return np.ascontiguousarray(X), Ctx(
                "polars", False, None, num_cols, None, None, data.schema,
                order=list(data.columns), passthrough=passthrough,
            )
    # ---- numpy / array-like ----
    try:
        arr = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"CAFE could not interpret input of type {type(data).__name__!r} as a "
            "numeric array. Pass a numpy array, pandas/polars Series or DataFrame, "
            "or any array-like of floats (use NaN for missing)."
        ) from e
    if arr.ndim == 1:
        return np.ascontiguousarray(arr.reshape(-1, 1)), Ctx("numpy", True)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr), Ctx("numpy", False)
    if arr.ndim == 3:
        # Panel (entity x time x feature): flatten to a stacked (E*T, F) matrix in C order
        # -- row e*T + t is entity e at time t -- and derive the integer ids. The original
        # 3D shape is recorded so from_matrix() rebuilds the cube. Reshape is the exact
        # inverse, so observed cells round-trip bit-identically.
        E, T, F = arr.shape
        ent = np.repeat(np.arange(E), T)
        tim = np.tile(np.arange(T), E)
        X = np.ascontiguousarray(arr.reshape(E * T, F))
        return X, Ctx("numpy", False,
                      panel_meta={"entity_ids": ent, "time_ids": tim},
                      panel_shape=(E, T, F))
    raise ValueError(
        f"CAFE expects 1D (series), 2D (matrix) or 3D (entity x time x feature) input; "
        f"got a {arr.ndim}D array of shape {arr.shape}."
    )


def _to_matrix_panel_df(data, panel, kind):
    """Long-format DataFrame -> (X, ctx) for the panel path. ``panel=(time_col,
    entity_col)``: those columns index the panel and are passed through untouched; every
    other numeric column is imputed. The frame is already in stacked long form, so X is
    just its numeric feature block in original row order -- no pivot, trivial round-trip."""
    time_col, entity_col = panel
    if kind == "pandas":
        cols = list(data.columns)
        for c in (time_col, entity_col):
            if c not in cols:
                raise KeyError(f"panel column {c!r} not found in DataFrame columns {cols}")
        num = data.select_dtypes(include=["number"]).drop(
            columns=[c for c in (time_col, entity_col) if c in data.select_dtypes(include=["number"]).columns],
            errors="ignore")
        if num.shape[1] == 0:
            raise ValueError(
                "CAFE found no numeric feature columns to impute in this panel "
                f"(index columns {time_col!r}/{entity_col!r} are excluded).")
        num_set = set(num.columns)
        passthrough = [(c, data[c]) for c in cols if c not in num_set]
        X = np.ascontiguousarray(num.to_numpy(dtype=float))
        meta = _panel_meta(data[time_col].tolist(), data[entity_col].tolist())
        ctx = Ctx("pandas", False, data.index, list(num.columns), None,
                  list(num.dtypes), order=cols, passthrough=passthrough, panel_meta=meta)
        return X, ctx
    # polars
    cols = list(data.columns)
    for c in (time_col, entity_col):
        if c not in cols:
            raise KeyError(f"panel column {c!r} not found in DataFrame columns {cols}")
    num_cols = [c for c, dt in data.schema.items()
                if dt.is_numeric() and c not in (time_col, entity_col)]
    if not num_cols:
        raise ValueError(
            "CAFE found no numeric feature columns to impute in this panel "
            f"(index columns {time_col!r}/{entity_col!r} are excluded).")
    num_set = set(num_cols)
    passthrough = [(c, data[c]) for c in cols if c not in num_set]
    X = np.ascontiguousarray(data.select(num_cols).to_numpy().astype(float))
    meta = _panel_meta(data[time_col].to_list(), data[entity_col].to_list())
    ctx = Ctx("polars", False, None, num_cols, None, None, data.schema,
              order=cols, passthrough=passthrough, panel_meta=meta)
    return X, ctx


def from_matrix(X, ctx: Ctx):
    """Rebuild the original container type from a result matrix X (T, N)."""
    X = np.asarray(X, float)
    if ctx.panel_shape is not None:          # 3D (entity x time x feature) round-trip
        return X.reshape(ctx.panel_shape)
    if ctx.kind == "pandas":
        import pandas as pd
        if ctx.was_1d:
            s = pd.Series(X[:, 0], index=ctx.index, name=ctx.name)
            return s.astype(ctx.dtypes[0]) if ctx.dtypes else s
        df = pd.DataFrame(X, index=ctx.index, columns=ctx.columns)
        if ctx.dtypes:                                   # best-effort dtype restore
            for c, dt in zip(ctx.columns, ctx.dtypes):
                try:
                    df[c] = df[c].astype(dt)
                except (ValueError, TypeError):
                    pass
        for name, vals in ctx.passthrough:               # re-attach non-numeric cols
            df[name] = vals
        return df[ctx.order] if ctx.order else df        # restore original order
    if ctx.kind == "polars":
        import polars as pl
        if ctx.was_1d:
            return pl.Series(ctx.name if ctx.name is not None else "", X[:, 0])
        out = pl.DataFrame({c: X[:, j] for j, c in enumerate(ctx.columns)})
        for name, vals in ctx.passthrough:               # re-attach non-numeric cols
            out = out.with_columns(vals.alias(name))
        return out.select(ctx.order) if ctx.order else out
    # numpy
    return X[:, 0] if ctx.was_1d else X
