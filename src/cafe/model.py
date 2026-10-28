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

# Optional missingness-as-signal features (the fact a cell WAS missing is often
# informative and is erased by imputation). Guarded so the library still imports and
# works if the module is absent; the feature path is opt-in and off by default.
try:
    from .missingness import missingness_features as _missingness_features
except Exception:  # pragma: no cover - defensive; absence must never break impute()
    _missingness_features = None

__all__ = ["CAFE", "CafeResult", "PanelResult", "PooledPanelResult", "impute"]


def _is_panel(meta):
    return bool(meta) and "entity_ids" in meta and "time_ids" in meta \
        and len(np.unique(meta["entity_ids"])) > 1


def _forward(X, record, forecast_from=None):
    """Forward-run the core over a 2D matrix; return (filled, core). With ``record=False``
    (the lean path used by ``impute``/``forecast``) the per-row introspection trace and the
    predictive-band stats are skipped entirely -- the imputed values are bit-identical, but
    we avoid building ~T dicts of arrays, so it is a touch faster and far lighter in memory.
    ``run`` uses ``record=True`` to expose uncertainty/factors/decomposition/etc.
    ``forecast_from`` (set only by :meth:`CAFE.forecast`) marks the first row index that is a
    forecast (extrapolation off the end), enabling the RW-anchor on those rows only."""
    X = np.ascontiguousarray(np.asarray(X, float))
    # Robustness parity with _core.online_impute: +/-Inf are not valid observations, so
    # treat any non-finite input cell as missing -- otherwise an Inf would propagate into
    # the solver and out into the result. (The lean run/impute path bypasses
    # online_impute, so the guard must live here too.)
    if not np.all(np.isfinite(X)):
        X = np.where(np.isfinite(X), X, np.nan)
    T, N = X.shape
    periods = _core._fourier_periods(T)
    core = _core._UnifiedCore(N, periods, E=1)
    core.record = record
    core.forecast_from = forecast_from
    out = X.copy()
    z_prev = None
    for t in range(T):
        ft = _core._fourier_row(t, periods)
        filled, z_t = core.process_row(X[t], t, z_prev, ft, eid=0)
        out[t] = filled
        z_prev = z_t
    if not np.all(np.isfinite(out)):              # final safety net: never emit NaN/Inf
        gm = core._mu(0)
        gm = np.where(np.isfinite(gm), gm, 0.0)
        idx = np.where(~np.isfinite(out))
        out[idx] = np.take(gm, idx[1])
    return out, core


def _run_traced(X):
    """Forward-run with tracing (rich path); return (filled, trace, core)."""
    out, core = _forward(X, record=True)
    return out, core.rows_trace, core


def _run_fast(X, forecast_from=None):
    """Forward-run without tracing (lean path); return just the filled matrix."""
    out, _ = _forward(X, record=False, forecast_from=forecast_from)
    return out


def _impute_per_entity(X, meta):
    """Impute each entity's series independently through the 2D core.  Rows are gathered
    per entity and processed in time order (point-in-time preserved), then scattered back
    to their original positions, so the stacked layout round-trips exactly."""
    X = np.ascontiguousarray(np.asarray(X, float))
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    out = X.copy()
    for e in np.unique(eids):
        idx = np.where(eids == e)[0]
        rows = idx[np.argsort(tids[idx], kind="stable")]   # time-ordered for causality
        out[rows] = _run_fast(X[rows])
    return out


def _run_per_entity_traced(X, meta):
    """Impute each entity through the 2D *traced* core, returning the reassembled filled
    matrix and a dict ``{entity_id -> (row_index, CafeResult)}``. Each entity's series is
    run point-in-time, so every by-product (uncertainty, factors, anomaly, decomposition)
    is strictly causal per entity; nothing crosses the entity boundary."""
    X = np.ascontiguousarray(np.asarray(X, float))
    eids = np.asarray(meta["entity_ids"])
    tids = np.asarray(meta["time_ids"])
    out = X.copy()
    per_entity = {}
    for e in np.unique(eids):
        idx = np.where(eids == e)[0]
        rows = idx[np.argsort(tids[idx], kind="stable")]   # time-ordered for causality
        filled, trace, core = _run_traced(X[rows])
        out[rows] = filled
        # a lightweight 2D context so the per-entity CafeResult returns numpy arrays
        sub_ctx = Ctx("numpy", False, None, None, None, None, None)
        per_entity[int(e)] = (rows, CafeResult(sub_ctx, X[rows], filled, trace, core))
    return out, per_entity


def _panel_stream_order(eids, tids):
    """Row indices of the panel in (time, entity) order -- the POOLED stream.

    TIME is the primary key, so truncating the panel at time ``tau`` yields exactly a
    PREFIX of this stream: every row at or before ``tau`` sees an identical history no
    matter what comes after. That is what keeps the pooled pass strictly point-in-time
    (certified by ``bench/causal.py::assert_causal`` and :func:`cafe.audit.verify_causal`).
    Entity is only the tie-break WITHIN a time, so the cross-entity information a row can
    see is contemporaneous -- explicitly permitted -- never future.
    """
    return np.lexsort((np.asarray(eids), np.asarray(tids)))


def _run_panel_traced_pooled(X, meta):
    """Drive ONE traced ``_UnifiedCore`` over the WHOLE panel in (time, entity) order.

    The per-entity engine (:func:`_run_per_entity_traced`) gives every entity its own
    core, so each estimates its loadings ``W`` from its own ``T`` rows alone. Here a
    single core sees all ``E*T`` rows, so everything the core POOLS -- the loadings ``W``,
    the ARD precisions that set the rank, the AR coefficient ``a``, the Student-t dof
    ``nu``, the robust scales, the idiosyncratic variances ``psi`` and the EW residual
    covariance -- is estimated from ``E`` times as much data, while everything that is
    genuinely per-entity stays per entity: the feature fixed effect ``mu`` (``_core._mu(e)``
    is already indexed by entity) and the latent state ``z`` (threaded per entity below,
    so the AR recursion runs along each entity's OWN time axis).

    Point-in-time. Rows are consumed in time order; a row at time ``t`` sees only rows at
    times ``<= t`` plus the contemporaneous cross-section, and every learned parameter is
    updated online after the row is imputed. Truncating the panel at ``tau`` cannot change
    any fill at or before ``tau``.

    Caveat (documented, not incidental): the core's idiosyncratic-carry state
    (``idio_last``/``idio_age``/``rho_idio``) and the last-observed anchor are per FEATURE,
    not per (entity, feature). Under an interleaved stream they therefore carry across the
    entity boundary. That channel only ever contributes to MISSING-cell fills, so it never
    touches the structural residual at an observed cell (what :meth:`residuals` returns
    and what causal discovery consumes), but it does make this engine a poorer *imputer*
    than the per-entity one on sparse panels. Hence ``engine='per_entity'`` remains the
    default.

    Returns ``(filled, order, stream_result)``: ``filled`` in the ORIGINAL stacked row
    layout, ``order`` the stream permutation, and ``stream_result`` a
    :class:`CafeResult` over the stream-ordered rows.
    """
    X = np.ascontiguousarray(np.asarray(X, float))
    if not np.all(np.isfinite(X)):
        X = np.where(np.isfinite(X), X, np.nan)
    eids = np.asarray(meta["entity_ids"]).astype(int)
    tids = np.asarray(meta["time_ids"]).astype(int)
    T_rows, N = X.shape
    E = int(eids.max()) + 1 if T_rows else 1
    order = _panel_stream_order(eids, tids)
    periods = _core._fourier_periods(T_rows)
    core = _core._UnifiedCore(N, periods, E=E)
    core.record = True
    out = X.copy()
    z_prev = [None] * E                      # latent state per ENTITY (own time axis)
    for r in order:
        e = int(eids[r])
        t = int(tids[r])
        ft = _core._fourier_row(t, periods)
        filled, z_t = core.process_row(X[r], t, z_prev[e], ft, eid=e)
        out[r] = filled
        z_prev[e] = z_t
    if not np.all(np.isfinite(out)):          # final safety net: never emit NaN/Inf
        gm = np.where(np.isfinite(core.g_cnt) & (core.g_cnt > 0),
                      core.g_sum / np.maximum(core.g_cnt, 1), 0.0)
        gm = np.where(np.isfinite(gm), gm, 0.0)
        idx = np.where(~np.isfinite(out))
        out[idx] = np.take(gm, idx[1])
    sub_ctx = Ctx("numpy", False, None, None, None, None, None)
    stream_result = CafeResult(sub_ctx, X[order], out[order],
                               core.rows_trace, core)
    return out, order, stream_result


class PanelResult:
    """Panel by-products, surfaced by running each entity through the same 2D *traced*
    causal pass used for 2-D data (the per-entity engine). Every per-entity quantity is
    strictly point-in-time. By-products are keyed by entity id (panels are inherently
    keyed by entity, and entities may have different time coverage), except the pooled
    dependency network which is a single ``(N, N)`` matrix averaged across entities."""

    def __init__(self, ctx, X_in, filled, per_entity):
        self._ctx = ctx
        self._X = X_in
        self._filled = filled
        self._pe = per_entity            # {eid -> (rows, CafeResult)}

    @property
    def imputed(self):
        """The filled panel, in the original container type/labels."""
        return from_matrix(self._filled, self._ctx)

    @property
    def uncertainty(self):
        """``{entity_id -> per-cell posterior std array}`` (NaN where observed)."""
        return {e: np.asarray(r.uncertainty) for e, (_, r) in self._pe.items()}

    def factors(self):
        """``{entity_id -> (T_e, R) latent factor paths}``."""
        return {e: r.factors() for e, (_, r) in self._pe.items()}

    def anomaly_scores(self):
        """``{entity_id -> per-time outlier score in [0,1]}`` (causal per entity)."""
        return {e: np.asarray(r.anomaly_scores()) for e, (_, r) in self._pe.items()}

    def decompose(self, full=False):
        """``{entity_id -> {level, season, factor, residual}}`` (each sums to the fill)."""
        return {e: r.decompose(full=full) for e, (_, r) in self._pe.items()}

    def residuals(self, stack=True, **kw):
        """Per-entity structural residuals (see :meth:`CafeResult.residuals`).

        ``stack=True`` (default) scatters every entity's residual block back into the
        ORIGINAL stacked row layout of the input, returning ``(E, obs)`` where ``E`` is
        ``(E*T, F)`` with ``np.nan`` at imputed cells and ``obs`` the observed-cell mask --
        so a row of ``E`` lines up with the same row of the data that was passed in.
        ``stack=False`` returns ``{entity_id: (E_e, obs_e)}`` per entity instead.

        Each entity is residualised from its OWN point-in-time pass; nothing crosses the
        entity boundary.
        """
        per = {}
        for e, (_rows, r) in self._pe.items():
            Ee, obs_e = r.residuals(return_mask=True, **kw)
            per[e] = (np.asarray(Ee, float), np.asarray(obs_e, bool))
        if not stack:
            return per
        E = np.full(self._filled.shape, np.nan)
        O = np.zeros(self._filled.shape, dtype=bool)
        for e, (rows, _r) in self._pe.items():
            Ee, obs_e = per[e]
            E[rows] = Ee
            O[rows] = obs_e
        return E, O

    def dependency_network(self, factor_adjusted=False, partial=False, robust=False,
                           at=None, halflife=None, min_pairs=8, pool=None):
        """A single pooled ``(N, N)`` residual dependency network across entities.

        ``pool='mean'`` averages the per-entity correlation matrices -- the legacy
        behaviour, and the default whenever ``factor_adjusted=False``, so calling this
        with no arguments reproduces the historical number exactly.

        ``pool='stats'`` (the default once ``factor_adjusted=True``) pools the SUFFICIENT
        STATISTICS instead: it sums the per-entity un-normalised cross-moments and
        co-observation masses and normalises ONCE. That is the right pooling -- averaging
        correlations gives a short, noisy entity the same vote as a long one, and averaged
        correlation matrices are not even guaranteed to be a valid correlation matrix of
        anything. Remaining arguments are forwarded to :meth:`CafeResult.dependency_network`.
        """
        if pool is None:
            pool = "stats" if factor_adjusted else "mean"
        if pool not in ("mean", "stats"):
            raise ValueError(f"pool must be 'mean' or 'stats'; got {pool!r}")
        legacy = (not factor_adjusted and not partial and not robust
                  and at is None and halflife is None)
        if pool == "mean":
            if legacy:
                nets = [r.dependency_network() for _, (_, r) in self._pe.items()]
            else:
                nets = [r.dependency_network(
                            factor_adjusted=factor_adjusted, partial=partial,
                            robust=robust, at=at, halflife=halflife, min_pairs=min_pairs)
                        for _, (_, r) in self._pe.items()]
            nets = [n for n in nets if np.all(np.isfinite(n))]
            return np.mean(nets, axis=0) if nets else None
        # ---- pool the sufficient statistics, normalise once ----
        S_raw = Wp = None
        sw = sw2 = 0.0
        for _, (_, r) in self._pe.items():
            S_e, W_e, sw_e, sw2_e = r._resid_moments(
                factor_adjusted=factor_adjusted, robust=robust, at=at, halflife=halflife)
            S_raw = S_e if S_raw is None else S_raw + S_e
            Wp = W_e if Wp is None else Wp + W_e
            sw += sw_e
            sw2 += sw2_e
        if S_raw is None:
            return None
        if partial:
            n_eff = (sw * sw) / max(sw2, 1e-12)
            return CafeResult._partial_from_moments(S_raw, Wp, n_eff, min_pairs=min_pairs)
        return CafeResult._corr_from_moments(S_raw, Wp, min_pairs=min_pairs)

    def effective_rank(self):
        """Mean number of ARD-active factors across entities (each entity has its OWN
        loadings here, hence its own rank; the mean is the panel-level summary that
        matches :meth:`PooledPanelResult.effective_rank`'s single pooled number)."""
        rs = [r.effective_rank() for _, (_, r) in self._pe.items()]
        return int(np.mean(rs)) if rs else 0

    def loadings(self):
        """``{entity_id -> (W_snapshots, s_fac_snapshots, w_ver_per_row)}``.

        One loading history PER ENTITY -- that is what this engine estimates. The pooled
        engine returns a single shared history instead; see
        :meth:`PooledPanelResult.loadings`.
        """
        return {e: r.loadings() for e, (_, r) in self._pe.items()}

    @property
    def params(self):
        """``{entity_id -> {nu, ar, effective_rank}}``."""
        return {e: r.params for e, (_, r) in self._pe.items()}


class PooledPanelResult:
    """Panel by-products from the POOLED engine (``CAFE().run(..., engine='pooled')``).

    One :class:`CafeResult` over a single causal pass that consumes the WHOLE panel in
    (time, entity) order, so the loadings ``W`` -- and with them the ARD rank, the AR
    coefficient, the Student-t dof and every running scale -- are estimated from all
    ``E*T`` rows, while ``mu`` and the latent state ``z`` remain per entity. Contrast
    :class:`PanelResult`, where each entity gets an independent core and therefore
    estimates its own ``W`` from its own ``T`` rows.

    The API mirrors :class:`PanelResult` so an experiment can swap engines with one
    keyword. Quantities that are genuinely per entity (``factors``, ``uncertainty``,
    ``anomaly_scores``, ``decompose``) come back keyed by entity id exactly as they do
    there; quantities that this engine deliberately POOLS (``effective_rank``,
    ``loadings``, ``params``) come back as a single object rather than a dict, because
    there is only one of each.
    """

    def __init__(self, ctx, X_in, filled, order, eids, tids, stream_result):
        self._ctx = ctx
        self._X = X_in
        self._filled = filled
        self._order = np.asarray(order)          # stream position -> original row
        self._eids = np.asarray(eids)
        self._tids = np.asarray(tids)
        self._res = stream_result                # CafeResult over the stream-ordered rows
        self._er = None

    # -- stream positions of each entity's rows, already in ascending time order --
    def _entity_rows(self):
        if self._er is None:
            ent_stream = self._eids[self._order]
            self._er = {int(e): np.where(ent_stream == e)[0]
                        for e in np.unique(ent_stream)}
        return self._er

    def _split(self, A_stream):
        """``(n_stream, ...) -> {entity_id -> that entity's rows, time-ordered}``."""
        A = np.asarray(A_stream)
        return {e: A[rows] for e, rows in self._entity_rows().items()}

    @property
    def stream_result(self):
        """The underlying :class:`CafeResult` over the stream-ordered ``(E*T, F)`` rows.

        Its row ``i`` is original row ``order[i]``; use it for anything the panel-level
        API does not wrap.
        """
        return self._res

    @property
    def stream_order(self):
        """``order`` such that ``stream row i`` is ``original row order[i]``."""
        return self._order

    @property
    def imputed(self):
        """The filled panel, in the original container type/labels and row layout."""
        return from_matrix(self._filled, self._ctx)

    @property
    def uncertainty(self):
        """``{entity_id -> per-cell posterior std array}`` (NaN where observed)."""
        return self._split(np.asarray(self._res.uncertainty, float))

    def factors(self):
        """``{entity_id -> (T_e, R) latent factor paths}``.

        The paths are per entity (each entity's ``z`` follows its own AR recursion) but
        they live in ONE shared loading space, so unlike the per-entity engine they are
        directly comparable -- and averageable -- across entities.
        """
        return self._split(self._res.factors())

    def anomaly_scores(self):
        """``{entity_id -> per-time outlier score in [0,1]}``."""
        return self._split(np.asarray(self._res.anomaly_scores(), float))

    def decompose(self, full=False):
        """``{entity_id -> {level, season, factor, residual}}`` (each sums to the fill)."""
        parts = self._res._decompose_parts()
        if not full:
            factor = (parts["factor"] + parts["time_fe"]
                      + parts["carry"] + parts["cross_section"])
            parts = {"level": parts["level"], "season": parts["season"],
                     "factor": factor, "residual": parts["residual"]}
        return {e: {k: v[rows] for k, v in parts.items()}
                for e, rows in self._entity_rows().items()}

    def residuals(self, stack=True, **kw):
        """Structural residuals from the pooled pass (see :meth:`CafeResult.residuals`).

        ``stack=True`` (default) returns ``(E, obs)`` in the ORIGINAL stacked row layout
        of the input -- row ``r`` of ``E`` lines up with row ``r`` of the data that was
        passed in -- with ``np.nan`` at imputed cells. ``stack=False`` returns
        ``{entity_id: (E_e, obs_e)}``. Identical contract to
        :meth:`PanelResult.residuals`, so the two engines are drop-in swappable.

        The factor term subtracted here is ``W z_{e,t}`` with ``W`` POOLED over every
        entity -- the whole point of this engine.
        """
        Es, obs_s = self._res.residuals(return_mask=True, **kw)
        Es = np.asarray(Es, float)
        obs_s = np.asarray(obs_s, bool)
        if not stack:
            return {e: (Es[rows], obs_s[rows]) for e, rows in self._entity_rows().items()}
        E = np.full(self._filled.shape, np.nan)
        O = np.zeros(self._filled.shape, dtype=bool)
        E[self._order] = Es
        O[self._order] = obs_s
        return E, O

    def dependency_network(self, factor_adjusted=False, partial=False, robust=False,
                           at=None, halflife=None, min_pairs=8, pool=None):
        """A single pooled ``(N, N)`` residual dependency network.

        There is only one core here, so the network is estimated from its moments
        directly -- no cross-entity pooling step is needed and ``pool`` is accepted only
        so the call is signature-compatible with :meth:`PanelResult.dependency_network`.
        ``at`` indexes the STREAM (time-major), which is what makes the reported object
        truncation-invariant; pass ``at=None`` (the default) for the network as it stands
        at the end of the panel.
        """
        if pool not in (None, "mean", "stats"):
            raise ValueError(f"pool must be None, 'mean' or 'stats'; got {pool!r}")
        return self._res.dependency_network(
            factor_adjusted=factor_adjusted, partial=partial, robust=robust,
            at=at, halflife=halflife, min_pairs=min_pairs)

    def effective_rank(self):
        """Number of factors the POOLED ARD keeps active -- a single number, because
        there is a single shared loading matrix."""
        return self._res.effective_rank()

    def loadings(self):
        """``(W_snapshots, s_fac_snapshots, w_ver_per_row)`` for the ONE pooled loading
        history. ``w_ver_per_row`` is given in the ORIGINAL stacked row layout (row ``r``
        of the input used ``W_snapshots[w_ver[r] - 1]``), so it aligns with
        :meth:`residuals` and :attr:`imputed`."""
        Ws, Sf, wv = self._res.loadings()
        wv_orig = np.zeros(self._filled.shape[0], dtype=int)
        wv_orig[self._order] = np.asarray(wv, int)
        return Ws, Sf, wv_orig

    @property
    def params(self):
        """``{nu, ar, effective_rank}`` -- one pooled set, not one per entity."""
        return self._res.params


def _impute_panel(X, meta, engine="joint"):
    """Run panel imputation with the chosen ``engine``: 'joint' (the validated cross-entity
    bilinear core -- pools the contemporaneous cross-section, the right default whenever
    entities share structure) or 'per_entity' (independent 2D per series -- better only
    when each series is rich enough that the cross-section adds nothing).

    There is deliberately no auto-router: which engine wins is a property of the data-
    generating process (is the cross-section informative?), not of data shape, and it
    cannot be read off shape or availability without either guessing wrong on a whole class
    of panels or peeking at held-out error (which would leak look-ahead). See
    bench/tensor_probe/e6,e7 for the evidence. So the user picks; the default is the
    always-safe 'joint'."""
    if engine == "joint":
        return np.asarray(_core.online_impute(X, meta), float)
    if engine == "per_entity":
        return _impute_per_entity(X, meta)
    raise ValueError(f"engine must be 'joint' or 'per_entity'; got {engine!r}")


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
        self._Cx = None                 # lazily stacked OPTIONAL trace fields

    # -- lazily stacked OPTIONAL trace field (present only on newer traces) --
    def _trace_field(self, key):
        """Stack an optional per-row trace field into a ``(T, ...)`` array (cached).

        Guarded by key PRESENCE, so a trace produced by an older core -- one that never
        recorded ``resid_c`` / ``w_ver`` -- still works and simply yields ``None`` instead
        of raising. Kept out of the eager :meth:`_comp` build on purpose: these fields are
        needed only by the residual/discovery API and each costs a full ``(T, N)`` copy,
        which every ordinary traced user (uncertainty, factors, decompose) would otherwise
        pay for nothing.
        """
        tr = self._trace
        if not tr or key not in tr[0]:
            return None
        if self._Cx is None:
            self._Cx = {}
        if key not in self._Cx:
            self._Cx[key] = np.array([r[key] for r in tr])
        return self._Cx[key]

    # -- lazy component matrices read from the trace --
    def _comp(self):
        if self._C is None:
            tr, N = self._trace, self._N
            T = len(tr)
            cvar = np.full((T, N), np.nan)
            for i, r in enumerate(tr):
                if r["cvar"] is not None:
                    cvar[i] = r["cvar"]
            # time_fe is a per-row scalar (shared cross-sectional level) broadcast over
            # features; the attr_* arrays are the faithful per-cell contributions the core
            # actually summed into each fill (see decompose()).
            time_fe = np.array([r["time_fe"] for r in tr])[:, None] * np.ones((1, N))
            self._C = dict(
                level=np.array([r["mu"] for r in tr]),
                season=np.array([r["season"] for r in tr]),
                factor=np.array([r["lr"] for r in tr]),       # raw factor estimate
                time_fe=time_fe,
                # contributions AS USED in the fill (post cross-section fusion):
                attr_factor=np.array([r["attr_factor"] for r in tr]),
                attr_carry=np.array([r["attr_carry"] for r in tr]),
                attr_xsec=np.array([r["attr_xsec"] for r in tr]),
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

    # ---- CAUSAL split-conformal recalibration of the predictive band ----
    def _calibrator(self, cal_rate, cal_seed, window, min_scores):
        """Lazily fit (and cache) the causal split-conformal calibrator for THIS run's
        data. A second internal causal pass hides a fraction ``cal_rate`` of observed
        cells and scores them as held-out; the original imputation is untouched."""
        from .conformal import ConformalCalibrator
        if self._trace is None or len(self._trace) == 0:
            raise NotImplementedError(
                "calibrated intervals need the traced run; use CAFE().run(...) "
                "(not impute) on 1D/2D data. Panel calibration is future work.")
        key = (round(float(cal_rate), 6), int(cal_seed), int(window), int(min_scores))
        cache = getattr(self, "_calib_cache", None)
        if cache is None:
            cache = self._calib_cache = {}
        if key not in cache:
            cache[key] = ConformalCalibrator.from_data(
                self._X, cal_rate=cal_rate, cal_seed=cal_seed,
                window=window, min_scores=min_scores)
        return cache[key]

    def _conformal_multiplier_grid(self, level, cal_rate, cal_seed, window, min_scores):
        """(T, N) point-in-time conformal multiplier q[t] for the central ``level``
        interval. q[t] is the (1-alpha) empirical quantile of the calibration scores from
        rows < t (trailing window) -- so appending future rows never changes an earlier
        row's band, and the imputation itself is untouched. Falls back to the raw Gaussian
        z while too few scores have accrued."""
        cal = self._calibrator(cal_rate, cal_seed, window, min_scores)
        q = cal.grid(level)
        return q[:, None] * np.ones((1, self._N))

    def calibrated_uncertainty(self, level=0.90, cal_rate=0.15, cal_seed=0,
                               window=4000, min_scores=30):
        """Calibrated HALF-WIDTH per cell for the central ``level`` interval (same
        container; NaN where observed). This is ``q(level)*sigma`` -- the causal
        split-conformal recalibration of CAFE's raw per-cell sigma, so the band hits
        nominal ``level`` coverage instead of over-covering. Point-in-time: the
        multiplier at row ``t`` uses only held-out calibration residuals from rows
        ``< t`` (a second internal causal pass; ``res.imputed`` is unchanged)."""
        sd = np.sqrt(self._comp()["cvar"])
        q = self._conformal_multiplier_grid(level, cal_rate, cal_seed, window, min_scores)
        return from_matrix(q * sd, self._ctx)

    def calibrated_interval(self, level=0.90, cal_rate=0.15, cal_seed=0,
                            window=4000, min_scores=30):
        """(lower, upper) containers for the CALIBRATED central ``level`` predictive
        interval ``mu +/- q(level)*sigma``. ``q(level)`` is the causal trailing-window
        split-conformal multiplier (the (1-alpha) empirical quantile of held-out
        calibration residuals). Strictly point-in-time and imputation-preserving: the
        centre ``mu`` (``res.imputed``) is identical with or without calibration; only the
        band width changes so that observed coverage approaches ``level``."""
        sd = np.sqrt(self._comp()["cvar"])
        q = self._conformal_multiplier_grid(level, cal_rate, cal_seed, window, min_scores)
        hw = q * sd
        return (from_matrix(self._filled - hw, self._ctx),
                from_matrix(self._filled + hw, self._ctx))

    # ---- per-cell recoverability certificate (selective / risk-controlled imputation) ----
    def recoverability_score(self, conformal=True, weights=None, return_components=False,
                             **conformal_kwargs):
        """Per-cell recoverability certificate in [0,1] (1 = trustworthy fill, 0 = CAFE
        cannot recover this cell), NaN where observed. Built from this run's own state
        (posterior sigma, the causal conformal scale, cross-sectional anchor support,
        factor/loading support, Student-t robustness) -- no held-out truth is used.
        Needs the traced run (CAFE().run on 1D/2D). ``conformal=True`` expresses the
        certificate on the calibrated q*sigma scale; extra kwargs go to the calibrator."""
        from .recoverability import score_from_result
        s = score_from_result(self, conformal=conformal, weights=weights,
                              return_components=return_components, **conformal_kwargs)
        if return_components:
            score, comp = s
            return from_matrix(score, self._ctx), comp
        return from_matrix(s, self._ctx)

    def selective_imputed(self, min_confidence=0.5, **kwargs):
        """The imputation with NaN wherever the recoverability certificate
        < ``min_confidence`` (abstain on cells CAFE cannot recover rather than return a
        confident-but-wrong value). Observed cells are always kept. ``kwargs`` are
        forwarded to :meth:`recoverability_score`."""
        from .recoverability import score_from_result, selective_impute
        sc = score_from_result(self, **kwargs)
        out = selective_impute(self._filled, sc, min_confidence)
        return from_matrix(out, self._ctx)

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
    def _decompose_parts(self):
        """Raw (numpy) faithful additive components that sum EXACTLY to the filled data.

        The model fills a missing cell with the sum of *real* additive terms:

            filled = level(mu) + season + time_fe + factor + carry + cross_section + resid

        where, at a MISSING cell, ``factor``/``carry``/``cross_section`` are the post-fusion
        contributions actually summed in (the low-rank/AR factor share, the idiosyncratic
        AR carry share, and the cross-sectional conditional-mean share -- see _core), and
        ``resid`` is EXACTLY zero (the model added nothing else). At an OBSERVED cell the
        structural terms are the model's fit and ``resid`` is the genuine observation noise
        the structure does not explain. Previously time_fe + carry + cross_section were
        silently dumped into ``residual``, falsely implying residual ~ 0 only by absorbing
        real structural signal; these are now their own named, attributable terms.
        """
        C = self._comp()
        parts = {
            "level":         C["level"],          # per-feature fixed effect (mu)
            "season":        C["season"],         # Fourier seasonal mean
            "time_fe":       C["time_fe"],         # contemporaneous shared level (time FE)
            "factor":        C["attr_factor"],     # low-rank/AR factor contribution (as used)
            "carry":         C["attr_carry"],      # idiosyncratic AR carry (gap extrapolation)
            "cross_section": C["attr_xsec"],       # cross-sectional conditional-mean fill
        }
        struct = sum(parts.values())
        parts["residual"] = self._filled - struct   # ~0 at imputed cells; noise at observed
        return parts

    def decompose(self, full=False):
        """Faithful additive structural attribution of the filled data.

        Every part sums EXACTLY to the filled value. By default (``full=False``) the
        granular dynamic terms are grouped for a compact, backward-compatible view with
        the classical four parts ``level + season + factor + residual``, but -- unlike
        before -- ``factor`` now honestly carries ALL the dynamic structure used to fill a
        cell (low-rank/AR factor + contemporaneous time fixed-effect + idiosyncratic carry
        + cross-sectional conditional mean), so ``residual`` is GENUINELY ~0 at imputed
        cells (the model added no unexplained term there) and is the real observation noise
        at observed cells. It no longer hides real structural signal inside "residual".

        Pass ``full=True`` to get the fully itemised attribution with the dynamic term
        split into its named channels: ``level, season, time_fe, factor, carry,
        cross_section, residual`` (each a same-type container). This is the honest
        structural breakdown of exactly what the model summed into each fill.
        """
        parts = self._decompose_parts()
        if full:
            return {k: from_matrix(v, self._ctx) for k, v in parts.items()}
        # compact 4-part view: fold the dynamic channels into one honest "factor" term.
        factor = (parts["factor"] + parts["time_fe"]
                  + parts["carry"] + parts["cross_section"])
        compact = {"level": parts["level"], "season": parts["season"],
                   "factor": factor, "residual": parts["residual"]}
        return {k: from_matrix(v, self._ctx) for k, v in compact.items()}

    # ---- missingness-as-signal features (causal MIM + decay/gap features) ----
    def missingness_features(self, **kwargs):
        """Causal missingness-as-signal features for THIS run's original missing pattern.

        Imputation erases *where* data was missing, yet that pattern is often itself
        informative (a sensor offline during an event; a deliberately skipped field).
        This returns forward-only (point-in-time) features -- was-imputed indicators,
        time-since-last-observation, gap lengths, expanding missing-rate, and a
        leak-free selective MIM -- computed from the ORIGINAL missing mask of the data
        passed to :meth:`CAFE.run`, in the same container type/labels as ``imputed``.

        Keyword args are forwarded to :func:`cafe.missingness.missingness_features`
        (e.g. ``kinds=``, ``selective=``, ``return_meta=True``). Returns ``None`` if the
        optional missingness module is unavailable.
        """
        if _missingness_features is None:
            return None
        # Score against the (causally) imputed values but key features off the original
        # NaN mask, so the features "survive" imputation as intended.
        mask = np.isnan(self._X)
        return _missingness_features(self.imputed, mask=mask, **kwargs)

    # ---- structural residuals (the input to any causal-discovery routine) ----
    def residuals(self, factor_adjusted=True, mask_imputed=True, robust=False,
                  return_mask=False, as_container=False):
        """The model's structural residuals ``E`` -- what is left of the data after the
        causal one-pass fit removes level, season and the shared time fixed effect (and,
        by default, the common factor). This is the quantity a causal-discovery routine
        should consume: the *idiosyncratic* shocks, with the confounding common structure
        already projected out by the same point-in-time pass that did the imputation.

        Parameters
        ----------
        factor_adjusted : bool, default True
            Subtract the common-factor contribution ``W z_t``, leaving the idiosyncratic
            residual ``x - mu - season - time_fe - W z_t``. A latent common factor is a
            textbook confounder: leave it in and every pair of series that merely loads on
            it looks connected. ``False`` leaves it in -- that is precisely the quantity
            the legacy :meth:`dependency_network` accumulator is built from, so it is kept
            for comparability, not because it is the better input.
        mask_imputed : bool, default True
            Return ``np.nan`` -- **never 0** -- at cells that were imputed rather than
            observed. A zero is indistinguishable from a genuine datum and would silently
            shrink every covariance by the co-observation rate (and bias every correlation
            toward 0), while a NaN forces the consumer to do pairwise-complete moments.
            An imputed cell carries NO information here in any case: with ``robust=True``
            it is identically 0 by construction, and otherwise it is a deterministic
            function of the model's own fill (the leftover of the cross-section/carry
            fusion against the raw factor term), i.e. the model talking to itself.
            Feeding those cells to a discovery routine manufactures dependencies out of
            the imputer's own smoothing. Turn the mask off only for plotting.
        robust : bool, default False
            **The default matters.** ``False`` returns the RAW residual. ``True`` returns
            the winsorized ``resid_c`` -- the residual the core feeds to its own factor
            solve -- which is clipped precisely to kill the tail. The tail is exactly what
            identifies edge DIRECTION in every LiNGAM-family method (non-Gaussianity IS
            the identifying signal), so winsorized residuals must never reach an
            orientation routine. Use ``robust=True`` only for scale/ covariance estimates
            that you want outlier-insensitive, never for orientation.
        return_mask : bool, default False
            Also return the observed-cell boolean mask as ``(E, obs)``.
        as_container : bool, default False
            Return ``E`` in the original container type/labels instead of a raw array.

        Returns
        -------
        ndarray (T, N), or ``(E, obs)`` when ``return_mask=True``.
        """
        C = self._comp()
        if robust:
            E = self._trace_field("resid_c")
            if E is None:
                raise NotImplementedError(
                    "robust residuals need a trace recorded by a core that stores "
                    "'resid_c'; re-run with cafe.CAFE().run(...).")
            # resid_c already has mu, season AND the time fixed effect removed, and is
            # exactly 0 at unobserved cells (masked below).
            E = np.array(E, float, copy=True)
        else:
            E = self._filled - C["level"] - C["season"] - C["time_fe"]
        if factor_adjusted:
            E = E - C["factor"]
        obs = np.isfinite(self._X)
        if mask_imputed:
            E = np.where(obs, E, np.nan)
        out = from_matrix(E, self._ctx) if as_container else E
        return (out, obs) if return_mask else out

    # ---- the loadings that produced each row's factor term ----
    def loadings(self):
        """``(W_snapshots, s_fac_snapshots, w_ver_per_row)``.

        ``W`` is piecewise-constant between refits, so the core snapshots it once per
        refit instead of once per row (same information, ~20x less memory). ``W_snapshots``
        is ``(K, N, R)``, ``s_fac_snapshots`` is ``(K, R)`` (the empirical-Bayes factor
        prior variances at the same instants), and ``w_ver_per_row`` is ``(T,)``: row ``t``
        used ``W_snapshots[w_ver[t] - 1]``, or -- when ``w_ver[t] == 0`` -- the untouched
        random initialisation from before the first refit (no snapshot exists for it, and
        ``W_ready`` was False, so the factor term was inert there anyway).
        """
        core = self._core
        Ws = np.asarray(getattr(core, "W_snapshots", []) or [], float)
        Sf = np.asarray(getattr(core, "s_fac_snapshots", []) or [], float)
        wv = self._trace_field("w_ver")
        wv = (np.zeros(len(self._trace or []), dtype=int) if wv is None
              else np.asarray(wv, int))
        return Ws, Sf, wv

    # ---- pairwise-complete EW residual moments (shared by the network estimators) ----
    def _resid_moments(self, factor_adjusted=True, robust=False, at=None, halflife=None):
        """UN-normalised weighted cross-moments of the residuals over OBSERVED cells.

        Returns ``(S_raw, Wp, sw, sw2)`` where ``S_raw = (w*E)^T E`` and ``Wp = (w*O)^T O``
        are the pairwise-complete sufficient statistics (so they can be POOLED across
        entities before a single normalisation), and ``sw``/``sw2`` are the weight mass and
        its square-sum, from which ``n_eff = sw**2 / sw2``.

        The weight is the model's own exponential forgetting ``lam**(t0-t)`` times the
        Student-t row weight ``row_w[t]`` the core already learned, and is exactly 0 for
        ``t > t0`` -- so ``at=t0`` yields the network AS IT STOOD at time ``t0``, using
        rows ``<= t0`` only. That is what makes the network truncation-invariant, i.e.
        auditable by :func:`cafe.audit.verify_causal_series`.

        The mean correction is deliberately dropped: ``E`` is zero-mean by construction
        (the level, season and time-FE terms are exactly the fitted mean), and a
        pairwise-complete mean would cost three more ``(N, N)`` arrays for nothing.
        """
        E = np.asarray(self.residuals(factor_adjusted=factor_adjusted, robust=robust,
                                      mask_imputed=True), float)
        T = E.shape[0]
        O = np.isfinite(E)
        Ez = np.where(O, E, 0.0)
        Of = O.astype(float)
        t0 = int(T - 1 if at is None else at)
        hl = float(halflife) if halflife else float(_core.HALFLIFE)
        lam = 0.5 ** (1.0 / max(hl, 1e-9))
        t = np.arange(T)
        age = np.clip(t0 - t, 0, None)                 # clip first: t > t0 must not blow up
        w = np.where(t <= t0, lam ** age, 0.0) * np.asarray(self._comp()["row_w"], float)
        S_raw = (w[:, None] * Ez).T @ Ez
        Wp = (w[:, None] * Of).T @ Of
        return S_raw, Wp, float(w.sum()), float(np.sum(w * w))

    @staticmethod
    def _corr_from_moments(S_raw, Wp, min_pairs=8.0):
        """Pairwise-complete correlation from pooled sufficient statistics."""
        S = S_raw / np.maximum(Wp, 1e-12)
        S = np.where(Wp < float(min_pairs), 0.0, S)     # too little co-observation -> 0
        d = np.sqrt(np.maximum(np.diag(S), 0.0))
        dd = np.outer(d, d)
        C = np.where(dd > 1e-12, S / np.maximum(dd, 1e-12), 0.0)
        C = np.clip(C, -1.0, 1.0)
        np.fill_diagonal(C, 1.0)
        return C

    @staticmethod
    def _partial_from_moments(S_raw, Wp, n_eff, min_pairs=8.0):
        """Partial (conditional) correlation: linear shrinkage, then precision.

        A plain correlation lights up an ``i -> j -> k`` chain at every pair, so it cannot
        separate a direct dependency from a transitive one; the precision matrix can. The
        shrinkage intensity ``g = N / (N + n_eff)`` carries no tuned constant -- it is the
        same "dimension vs effective sample size" ratio CAFE already uses for its seasonal
        gain -- and keeps the matrix invertible when ``n_eff`` is small relative to ``N``.
        """
        S = S_raw / np.maximum(Wp, 1e-12)
        S = np.where(Wp < float(min_pairs), 0.0, S)
        S = 0.5 * (S + S.T)
        N = S.shape[0]
        g = N / (N + max(float(n_eff), 1e-9))
        tr = float(np.mean(np.maximum(np.diag(S), 0.0)))
        S_s = (1.0 - g) * S + g * max(tr, 1e-12) * np.eye(N)
        try:
            P = np.linalg.solve(S_s, np.eye(N))
        except np.linalg.LinAlgError:                    # pragma: no cover - defensive
            P = np.linalg.pinv(S_s)
        dp = np.sqrt(np.maximum(np.diag(P), 1e-12))
        C = -P / np.outer(dp, dp)
        C = np.clip(0.5 * (C + C.T), -1.0, 1.0)
        np.fill_diagonal(C, 1.0)
        return C

    # ---- cross-sectional dependency network ----
    def dependency_network(self, factor_adjusted=False, partial=False, robust=False,
                           at=None, halflife=None, min_pairs=8):
        """(N, N) residual dependency network.

        With **all defaults** this is bit-for-bit the historical estimator: the correlation
        of the core's own pooled exponentially-weighted residual covariance accumulator
        (which is built from ``full_resid``, i.e. the *factor-contaminated* residual, and
        uses model fills at missing cells). It is kept exactly as it was so published
        numbers reproduce.

        Any non-default argument routes to the trace-based estimator instead:

        factor_adjusted : bool, default False
            ``True`` subtracts the common factor ``W z_t`` first. The latent factor is a
            confounder -- leaving it in makes every pair that merely loads on it look
            connected -- so ``True`` is the right input for dependency *structure*, while
            ``False`` reproduces the legacy quantity.
        partial : bool, default False
            ``True`` returns partial (conditional) correlations via a shrunk precision
            matrix, which does not light up transitive ``i -> j -> k`` chains.
        robust : bool, default False
            Use the winsorized residual. Off by default: winsorizing destroys the tail,
            and the tail is what carries direction information (see :meth:`residuals`).
        at : int, optional
            Report the network AS IT STOOD at row ``at`` (rows ``> at`` get weight 0), so
            the reported object is truncation-invariant and can be certified with
            :func:`cafe.audit.verify_causal_series`. Default: the last row.
        halflife : float, optional
            Exponential forgetting half-life in rows (default: the core's own 200).
        min_pairs : float, default 8
            Minimum weighted co-observation mass for a pair to be estimated at all; below
            it the entry is set to 0 rather than reported from near-nothing.
        """
        if (not factor_adjusted and not partial and not robust
                and at is None and halflife is None):
            # ---- legacy path, byte-identical (do not touch) ----
            cov = (self._core.cM2 / self._core.cw
                   - np.outer(self._core.csum / self._core.cw, self._core.csum / self._core.cw)) \
                if self._core.cw > 1e-6 else np.eye(self._N)
            d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
            return cov / np.outer(d, d)
        S_raw, Wp, sw, sw2 = self._resid_moments(
            factor_adjusted=factor_adjusted, robust=robust, at=at, halflife=halflife)
        if partial:
            n_eff = (sw * sw) / max(sw2, 1e-12)
            return self._partial_from_moments(S_raw, Wp, n_eff, min_pairs=min_pairs)
        return self._corr_from_moments(S_raw, Wp, min_pairs=min_pairs)

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
            # faithful itemised attribution (drops all-zero channels for legibility)
            parts = {nm: np.asarray(M) for nm, M in self._decompose_parts().items()}
            for nm, M in parts.items():
                if not np.any(M[:, j]):           # skip channels inactive for this series
                    continue
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

    def run(self, data, meta=None, panel=None, engine="per_entity"):
        """Full traced run exposing every capability. For 1D/2D inputs returns a
        :class:`CafeResult`; for panel inputs (3D array or ``panel=(time_col, entity_col)``)
        returns a :class:`PanelResult` or :class:`PooledPanelResult` depending on
        ``engine`` -- both strictly point-in-time.

        ``panel=(time_col, entity_col)`` routes a long-format DataFrame through the panel
        path; a 3D ``(entity, time, feature)`` array is detected automatically. Either way
        the derived ``{entity_ids, time_ids}`` come from :func:`to_matrix`; an explicit
        ``meta`` still wins if supplied.

        Parameters
        ----------
        engine : {'per_entity', 'pooled'}, default 'per_entity'
            Panel inputs only (a 1D/2D input has no entity axis to pool over, so any
            non-default value raises).

            ``'per_entity'`` -- the default and the historical behaviour: every entity is
            run through its own 2D traced core, so it estimates its OWN loadings ``W``
            from its own ``T`` rows and nothing crosses the entity boundary. Returns a
            :class:`PanelResult`.

            ``'pooled'`` -- one core consumes the whole panel in (time, entity) order, so
            ``W`` (and the ARD rank, AR coefficient, dof and scales) is estimated from all
            ``E*T`` rows while ``mu`` and the latent state ``z`` stay per entity. Returns a
            :class:`PooledPanelResult` with the same by-product API. Pool when the
            entities plausibly SHARE a loading structure and each ``T`` is too short to
            identify it alone; keep the default when loadings are heterogeneous across
            entities, where a single shared ``W`` cannot represent the common component.

            (Note the unrelated ``engine=`` on :meth:`impute`, which selects the *fill*
            strategy -- 'joint' or 'per_entity' -- not the traced by-product engine.)
        """
        X, ctx = to_matrix(data, panel=panel)
        meta = meta or ctx.panel_meta
        if engine not in ("per_entity", "pooled"):
            raise ValueError(
                f"engine must be 'per_entity' or 'pooled'; got {engine!r}")
        if _is_panel(meta):
            if engine == "pooled":
                filled, order, sres = _run_panel_traced_pooled(X, meta)
                return PooledPanelResult(ctx, X, np.asarray(filled, float), order,
                                         np.asarray(meta["entity_ids"]),
                                         np.asarray(meta["time_ids"]), sres)
            # Panel by-products are surfaced via the per-entity traced engine: each entity's
            # series goes through the full 2D traced core, so uncertainty/factors/anomaly/
            # decomposition are real and strictly causal per entity. (The cross-entity joint
            # factor view is future work -- the joint engine's ARD currently collapses the
            # shared time-factor to rank ~0; we expose the residual-derived network instead.)
            filled, per_entity = _run_per_entity_traced(X, meta)
            return PanelResult(ctx, X, np.asarray(filled, float), per_entity)
        if engine != "per_entity":
            raise ValueError(
                "engine='pooled' needs panel data (a 3D array, panel=(time_col, "
                "entity_col), or meta with entity_ids/time_ids); this input has no "
                "entity axis to pool over.")
        filled, trace, core = _run_traced(X)
        return CafeResult(ctx, X, filled, trace, core)

    def impute(self, data, meta=None, return_mask=False, missingness_kwargs=None,
               panel=None, engine="joint"):
        """Return just the filled data in the original container type. Uses the lean
        forward path (no introspection trace) -- identical values, lighter + faster.

        Parameters
        ----------
        return_mask : bool, default False
            If True (and the optional :mod:`cafe.missingness` module is available),
            also compute causal missingness-as-signal features from the ORIGINAL missing
            pattern and return ``(filled, features)`` instead of just ``filled``. These
            forward-only features (was-imputed indicator, time-since-observed, gap length,
            expanding missing-rate, selective MIM) preserve the signal that imputation
            would otherwise erase, for use as extra columns in a downstream model.
            Default behaviour (``return_mask=False``) is unchanged and fully
            backward-compatible. If the module is unavailable, ``features`` is ``None``.
        missingness_kwargs : dict, optional
            Extra keyword args forwarded to
            :func:`cafe.missingness.missingness_features`.
        """
        X, ctx = to_matrix(data, panel=panel)
        meta = meta or ctx.panel_meta
        if _is_panel(meta):
            filled = from_matrix(_impute_panel(X, meta, engine=engine), ctx)
        else:
            filled = from_matrix(_run_fast(X), ctx)
        if not return_mask:
            return filled
        feats = None
        if _missingness_features is not None:
            mask = np.isnan(np.asarray(X, float))
            feats = _missingness_features(filled, mask=mask,
                                          **(missingness_kwargs or {}))
        return filled, feats

    def forecast(self, data, horizon: int):
        """Forecast ``horizon`` steps ahead by imputing appended all-missing rows
        (forecasting = imputing future cells via the AR/Kalman state). Returns the
        forecast block in the original container type."""
        X, ctx = to_matrix(data)
        T, N = X.shape
        Xf = np.vstack([X, np.full((horizon, N), np.nan)])
        # forecast_from=T marks the appended rows as forecasts, enabling the RW-anchor
        # there only (interior-gap imputation is untouched).
        fc = _run_fast(Xf, forecast_from=T)[T:]
        # rebuild a container for just the forecast rows
        fctx = Ctx(ctx.kind, ctx.was_1d, None, ctx.columns, ctx.name, ctx.dtypes, ctx.schema)
        return from_matrix(fc, fctx)


def impute(data, meta=None, return_mask=False, missingness_kwargs=None, panel=None,
           engine="joint"):
    """Zero-config causal imputation. Accepts numpy / pandas / polars (1D or 2D), a 3D
    ``(entity, time, feature)`` array, or a long-format DataFrame with ``panel=(time_col,
    entity_col)``. Returns the same container type with missing values filled, using only
    past + contemporaneous information (no look-ahead).

    For panel data, ``engine`` selects the imputation strategy: ``'joint'`` (default, the
    validated cross-entity bilinear core -- pools the contemporaneous cross-section) or
    ``'per_entity'`` (independent 2D per series -- opt in when each series is rich enough
    that the cross-section adds nothing, e.g. long history with many features).

    With ``return_mask=True`` also returns causal missingness-as-signal features for the
    original missing pattern as ``(filled, features)`` (see :meth:`CAFE.impute`); the
    default single-value return is unchanged.
    """
    return CAFE().impute(data, meta, return_mask=return_mask,
                         missingness_kwargs=missingness_kwargs, panel=panel, engine=engine)
