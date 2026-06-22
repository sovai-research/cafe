"""Container adapters: accept numpy / pandas / polars (1D or 2D), run the model on a
plain float matrix, and rebuild the *same* container type, dtype and labels on the way
out. Optional dependencies (pandas, polars) are detected lazily and never imported
unless the user actually passes one of their objects.
"""
from __future__ import annotations

import numpy as np

__all__ = ["to_matrix", "from_matrix", "Ctx"]


class Ctx:
    """Round-trip context describing the original container."""
    __slots__ = ("kind", "was_1d", "index", "columns", "name", "dtypes", "schema")

    def __init__(self, kind, was_1d=False, index=None, columns=None,
                 name=None, dtypes=None, schema=None):
        self.kind = kind            # 'numpy' | 'pandas' | 'polars'
        self.was_1d = was_1d
        self.index = index
        self.columns = columns
        self.name = name
        self.dtypes = dtypes
        self.schema = schema


def _is_pandas(x):
    m = type(x).__module__
    return m.startswith("pandas.")


def _is_polars(x):
    m = type(x).__module__
    return m.startswith("polars.")


def to_matrix(data):
    """Return (X, ctx): X is a contiguous float64 (T, N) array with NaN for missing."""
    # ---- pandas ----
    if _is_pandas(data):
        import pandas as pd
        if isinstance(data, pd.Series):
            X = data.to_numpy(dtype=float).reshape(-1, 1)
            return np.ascontiguousarray(X), Ctx("pandas", True, data.index, None, data.name,
                                                [data.dtype])
        if isinstance(data, pd.DataFrame):
            X = data.to_numpy(dtype=float)
            return np.ascontiguousarray(X), Ctx("pandas", False, data.index,
                                                list(data.columns), None, list(data.dtypes))
    # ---- polars ----
    if _is_polars(data):
        import polars as pl
        if isinstance(data, pl.Series):
            X = data.to_numpy().astype(float).reshape(-1, 1)
            return np.ascontiguousarray(X), Ctx("polars", True, None, None, data.name, None)
        if isinstance(data, pl.DataFrame):
            X = data.to_numpy().astype(float)
            return np.ascontiguousarray(X), Ctx("polars", False, None, list(data.columns),
                                                None, None, data.schema)
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
        raise ValueError(
            f"CAFE got a 3D array of shape {arr.shape}. For panel (entity x time x "
            "feature) data, stack it to a 2D (rows, features) matrix and pass "
            "meta={'entity_ids': ..., 'time_ids': ...} to CAFE().run / .impute."
        )
    raise ValueError(
        f"CAFE expects 1D (series) or 2D (matrix) input; got a {arr.ndim}D array "
        f"of shape {arr.shape}."
    )


def from_matrix(X, ctx: Ctx):
    """Rebuild the original container type from a result matrix X (T, N)."""
    X = np.asarray(X, float)
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
        return df
    if ctx.kind == "polars":
        import polars as pl
        if ctx.was_1d:
            return pl.Series(ctx.name if ctx.name is not None else "", X[:, 0])
        return pl.DataFrame({c: X[:, j] for j, c in enumerate(ctx.columns)})
    # numpy
    return X[:, 0] if ctx.was_1d else X
