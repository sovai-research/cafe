"""CAFE -- Causal Adaptive Factor Estimation.

A single causal (strictly point-in-time) imputation model that also exposes, from the
*same* forward pass, the quantities a dynamic factor model naturally produces:
per-cell uncertainty, latent common factors, an anomaly score, an additive
decomposition, a cross-sectional dependency network, and forecasts.

Quick start
-----------
>>> import cafe
>>> filled = cafe.impute(df)             # zero-config; same container type back
>>> res = cafe.CAFE().run(df)            # rich result with all capabilities
>>> res.uncertainty, res.factors(), res.anomaly_scores(), res.dependency_network()
"""
from __future__ import annotations

import numpy as np

from . import _core
from .io import Ctx, from_matrix, to_matrix

__all__ = ["CAFE", "CafeResult", "impute"]


def _is_panel(meta):
    return bool(meta) and "entity_ids" in meta and "time_ids" in meta \
        and len(np.unique(meta["entity_ids"])) > 1


def _forward(X, record):
    """Forward-run the core over a 2D matrix; return (filled, core). With ``record=False``
    (the lean path used by ``impute``/``forecast``) the per-row introspection trace and the
    predictive-band stats are skipped entirely -- the imputed values are bit-identical, but
    we avoid building ~T dicts of arrays, so it is a touch faster and far lighter in memory.
    ``run`` uses ``record=True`` to expose uncertainty/factors/decomposition/etc."""
    X = np.ascontiguousarray(np.asarray(X, float))
    T, N = X.shape
    periods = _core._fourier_periods(T)
    core = _core._UnifiedCore(N, periods, E=1)
    core.record = record
    out = X.copy()
    z_prev = None
    for t in range(T):
        ft = _core._fourier_row(t, periods)
        filled, z_t = core.process_row(X[t], t, z_prev, ft, eid=0)
        out[t] = filled
        z_prev = z_t
    if np.isnan(out).any():                       # final safety net
        gm = core._mu(0)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(gm, idx[1])
    return out, core


def _run_traced(X):
    """Forward-run with tracing (rich path); return (filled, trace, core)."""
    out, core = _forward(X, record=True)
    return out, core.rows_trace, core


def _run_fast(X):
    """Forward-run without tracing (lean path); return just the filled matrix."""
    out, _ = _forward(X, record=False)
    return out


class CafeResult:
    """Everything CAFE produces in one causal pass over ``data``."""

    def __init__(self, ctx: Ctx, X_in, filled, trace, core):
        self._ctx = ctx
        self._X = X_in
        self._filled = filled
        self._trace = trace
        self._core = core
        self._N = filled.shape[1]
        self._C = None

    # -- lazy component matrices read from the trace --
    def _comp(self):
        if self._C is None:
            tr, N = self._trace, self._N
            T = len(tr)
            cvar = np.full((T, N), np.nan)
            for i, r in enumerate(tr):
                if r["cvar"] is not None:
                    cvar[i] = r["cvar"]
            self._C = dict(
                level=np.array([r["mu"] for r in tr]),
                season=np.array([r["season"] for r in tr]),
                factor=np.array([r["lr"] for r in tr]),
                Z=np.array([r["z"] for r in tr]),
                cvar=cvar,
                row_w=np.array([r["row_w"] for r in tr]),
                nu=np.array([r["nu"] for r in tr]),
                a=np.array([r["a"] for r in tr]),
                alpha=np.array([r["alpha"] for r in tr]),
            )
        return self._C

    # ---- the imputation itself ----
    @property
    def imputed(self):
        """The filled data, in the original container type/labels."""
        return from_matrix(self._filled, self._ctx)

    # ---- per-cell uncertainty ----
    @property
    def uncertainty(self):
        """Posterior predictive standard deviation per cell (same container).
        NaN where the value was observed (no imputation needed)."""
        return from_matrix(np.sqrt(self._comp()["cvar"]), self._ctx)

    def confidence_interval(self, z=1.96):
        """(lower, upper) containers around the imputation at ``z`` sigma."""
        sd = np.sqrt(self._comp()["cvar"])
        return (from_matrix(self._filled - z * sd, self._ctx),
                from_matrix(self._filled + z * sd, self._ctx))

    # ---- latent common factors (streaming robust DFM / PCA) ----
    def factors(self):
        """The learned latent factor paths z_t as a (T, R) array."""
        return self._comp()["Z"]

    def effective_rank(self):
        """Number of factors ARD keeps active (rank emerges from the data)."""
        a = self._comp()["alpha"][-1]
        return int(np.sum(a < 0.5 * np.median(a) + 1e-9)) if a.size else 0

    # ---- anomaly / outlier score (free byproduct of the Student-t weights) ----
    def anomaly_scores(self):
        """Per-time outlier score in [0, 1] (0 = perfect fit, 1 = strong outlier).

        The Student-t IRLS weight is ``w = (nu+1)/(nu+u)`` with ``u`` the standardised
        squared residual, so ``w`` exceeds 1 for well-fitting rows. We report the
        bounded, monotone-in-``u`` quantity ``u/(nu+u) = 1 - w*nu/(nu+1)`` instead --
        exactly 0 at a perfect fit and approaching 1 as the residual blows up. Causal:
        row t uses only data <= t."""
        C = self._comp()
        nu, w = C["nu"], C["row_w"]
        s = np.clip(1.0 - w * (nu / (nu + 1.0)), 0.0, 1.0)
        if self._ctx.kind == "pandas":
            import pandas as pd
            return pd.Series(s, index=self._ctx.index, name="anomaly")
        return s

    # ---- additive decomposition ----
    def decompose(self):
        """Additive parts as containers: ``level + season + factor + residual`` equals
        the (filled) data exactly, so the decomposition is complete -- ``residual`` is
        the heavy-tailed noise the structural parts don't explain (near zero at imputed
        cells, the observation noise at observed cells)."""
        C = self._comp()
        struct = C["level"] + C["season"] + C["factor"]
        parts = {"level": C["level"], "season": C["season"], "factor": C["factor"],
                 "residual": self._filled - struct}
        return {k: from_matrix(v, self._ctx) for k, v in parts.items()}

    # ---- cross-sectional dependency network ----
    def dependency_network(self):
        """(N, N) correlation matrix of the residuals = learned dependency network."""
        cov = (self._core.cM2 / self._core.cw
               - np.outer(self._core.csum / self._core.cw, self._core.csum / self._core.cw)) \
            if self._core.cw > 1e-6 else np.eye(self._N)
        d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        return cov / np.outer(d, d)

    # ---- learned dials ----
    @property
    def params(self):
        """The online-learned parameters at the final step."""
        C = self._comp()
        return dict(nu=float(C["nu"][-1]), ar=float(C["a"][-1]),
                    effective_rank=self.effective_rank())

    # ---- one-line plotting (sovai-style: pick the view with a string) ----
    def plot(self, kind="uncertainty", col=0, ax=None):
        """Plot any capability in one line. ``kind`` is one of:
        ``'uncertainty'`` (fill ± 95% band for one column), ``'factors'`` (latent factor
        paths), ``'anomaly'`` (per-time outlier score), ``'decomposition'``
        (level/season/factor/residual for one column), ``'dependency'`` (network heatmap).
        ``col`` selects the column (int index or name)."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 3.2) if kind != "dependency" else (5.2, 4.4))
        names = self._ctx.columns
        j = (names.index(col) if (isinstance(col, str) and names) else
             (col if isinstance(col, int) else 0))
        label = col if isinstance(col, str) else (names[j] if names else f"series {j}")

        if kind == "factors":
            Z = self.factors()
            for r in range(Z.shape[1]):
                ax.plot(Z[:, r], lw=1.0, label=f"factor {r + 1}")
            ax.legend(ncol=4, fontsize=8); ax.set_title("latent factor paths $z_t$")
        elif kind == "anomaly":
            ax.plot(np.asarray(self.anomaly_scores()), lw=0.8, color="C3")
            ax.set_ylim(0, 1.02); ax.set_title("anomaly score (0 = fit, 1 = outlier)")
        elif kind in ("decomposition", "decompose"):
            C = self._comp()
            resid = self._filled - (C["level"] + C["season"] + C["factor"])
            parts = {"level": C["level"], "season": C["season"],
                     "factor": C["factor"], "residual": resid}
            for nm, M in parts.items():
                ax.plot(M[:, j], lw=1.0, label=nm)
            ax.legend(ncol=4, fontsize=8); ax.set_title(f"{label}: additive decomposition")
        elif kind in ("dependency", "network"):
            net = self.dependency_network()
            im = ax.imshow(net, cmap="RdBu_r", vmin=-1, vmax=1)
            if names:
                ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
                ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
            ax.figure.colorbar(im, ax=ax, fraction=0.046); ax.set_title("dependency network")
            ax.grid(False)
        else:  # uncertainty
            sd = np.sqrt(self._comp()["cvar"][:, j])
            f = self._filled[:, j]
            t = np.arange(len(f))
            ax.fill_between(t, f - 1.96 * sd, f + 1.96 * sd, alpha=0.25, color="C0", label="95% band")
            ax.plot(t, f, color="C0", lw=1.0, label="filled")
            obs = ~np.isnan(self._X[:, j])
            ax.scatter(t[obs], self._X[obs, j], s=6, color="0.35", label="observed", zorder=3)
            ax.legend(fontsize=8); ax.set_title(f"{label}: imputation ± uncertainty")
        return ax


class CAFE:
    """Causal Adaptive Factor Estimation imputer.

    Parameters
    ----------
    causal : bool, default True
        Strictly point-in-time (no look-ahead). The only supported mode in v1.
    """

    def __init__(self, causal: bool = True):
        if not causal:
            raise NotImplementedError("CAFE is causal by design; non-causal smoothing TBD.")
        self.causal = causal

    def run(self, data, meta=None) -> CafeResult:
        """Full traced run -> a CafeResult exposing every capability. 1D/2D inputs."""
        X, ctx = to_matrix(data)
        if _is_panel(meta):
            # panel: impute via the core's panel path (rich trace is 2D-only in v1)
            filled = _core.online_impute(X, meta)
            res = CafeResult(ctx, X, np.asarray(filled, float), [], None)
            res._C = {}                                  # capabilities limited for panels
            return res
        filled, trace, core = _run_traced(X)
        return CafeResult(ctx, X, filled, trace, core)

    def impute(self, data, meta=None):
        """Return just the filled data in the original container type. Uses the lean
        forward path (no introspection trace) -- identical values, lighter + faster."""
        X, ctx = to_matrix(data)
        if _is_panel(meta):
            return from_matrix(np.asarray(_core.online_impute(X, meta), float), ctx)
        return from_matrix(_run_fast(X), ctx)

    def forecast(self, data, horizon: int):
        """Forecast ``horizon`` steps ahead by imputing appended all-missing rows
        (forecasting = imputing future cells via the AR/Kalman state). Returns the
        forecast block in the original container type."""
        X, ctx = to_matrix(data)
        T, N = X.shape
        Xf = np.vstack([X, np.full((horizon, N), np.nan)])
        fc = _run_fast(Xf)[T:]
        # rebuild a container for just the forecast rows
        fctx = Ctx(ctx.kind, ctx.was_1d, None, ctx.columns, ctx.name, ctx.dtypes, ctx.schema)
        return from_matrix(fc, fctx)


def impute(data, meta=None):
    """Zero-config causal imputation. Accepts numpy / pandas / polars (1D or 2D),
    returns the same container type with missing values filled, using only past +
    contemporaneous information (no look-ahead)."""
    return CAFE().impute(data, meta)
