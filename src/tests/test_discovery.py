"""Contract tests for the public CAFE-CD assembler (``cafe.discovery``).

The components underneath (``_screen``, ``_skeleton``, ``_orient``, ``_diagnostics``)
have their own contract suites.  What is asserted HERE is only what the assembler is
responsible for:

1.  **It recovers structure end to end.**  On a synthetic panel with a known sparse
    VAR generator, ``cafe.discover`` returns a majority of the true lagged edges and
    ranks the true edges above the false ones (AUC), from the raw NaN-holed panel with
    no tuning whatsoever.
2.  **The point-in-time sequence is certified.**  ``mode="pit"`` passes
    ``cafe.audit.verify_causal_series`` on the 0/1 support indicator with
    ``max_revision == 0`` EXACTLY -- not to a tolerance.  A control test shows the same
    certificate has teeth: a reporter that centres its window on ``t`` (and so peeks
    one row past it) is caught with ``max_revision == 1``.
3.  **The engine router works, and the normalisation is what makes it work.**  A
    homogeneous-loading panel routes to ``pooled`` and a heterogeneous one to
    ``per_entity`` -- and the two fixtures are chosen so their RAW subspace alignments
    interleave, i.e. no fixed threshold on raw alignment can separate them.  That is
    the whole reason the statistic is normalised by the within-entity ceiling.
4.  **Containers round-trip.**  numpy / pandas / polars, 2-D and 3-D and long-format,
    all give the same graph with labels attached.
5.  **Degenerate input never raises.**  P=2, T=10, an all-NaN column, a constant
    column, an all-NaN panel: an empty-but-valid graph plus a note saying why.
6.  **numpy only.**  ``import cafe.discovery`` and a full ``discover`` run must not
    pull in scipy / sklearn / pandas / polars / matplotlib / numba / torch.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import cafe  # noqa: E402
import cafe.audit  # noqa: E402
from cafe.discovery import (  # noqa: E402
    CausalGraph,
    PitGraphs,
    choose_engine,
    default_max_lag,
    discover,
    loading_homogeneity_ratio,
    max_edge_revision,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
#: ``(source, target)`` pairs of the lag-1 generator used by the recovery tests.
TRUE_EDGES = [(1, 0), (2, 1), (3, 2), (4, 0), (5, 3)]


def var_panel(seed=0, n_ent=4, T=300, P=6, nan=0.05, coef=0.55, ar=0.35):
    """Sparse VAR(1) idiosyncratic part + a strong common AR(1) factor + MCAR holes.

    The factor is the confounder CAFE is supposed to project out before discovery: it
    loads on every series, so a method that skips the residualisation step sees a dense
    graph instead of ``TRUE_EDGES``.  Innovations are Laplace so the contemporaneous
    branch has a non-Gaussian signal to work with.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((P, P))
    for s, t in TRUE_EDGES:
        A[t, s] = coef
    np.fill_diagonal(A, ar)
    cubes = []
    for _ in range(n_ent):
        eps = rng.laplace(0.0, 1.0, size=(T, P))
        x = np.zeros((T, P))
        for t in range(1, T):
            x[t] = A @ x[t - 1] + eps[t]
        f = np.zeros(T)
        for t in range(1, T):
            f[t] = 0.9 * f[t - 1] + rng.normal()
        cubes.append(x + 1.5 * np.outer(f, np.linspace(1.0, 2.0, P)))
    X = np.stack(cubes)
    X[rng.random(X.shape) < nan] = np.nan
    return X


def var_series(seed=0, T=240, P=6, nan=0.05):
    """The same generator with a single entity -- a plain ``(T, P)`` matrix."""
    return var_panel(seed=seed, n_ent=1, T=T, P=P, nan=nan)[0]


def loading_panel(n_ent, T, P, R, noise, rot, seed):
    """Factor panel whose entities share (``rot=0``) or do not share their loadings."""
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(P, R))
    out = []
    for _ in range(n_ent):
        W = W0 + rot * rng.normal(size=(P, R)) if rot > 0 else W0
        F = rng.normal(size=(T, R))
        out.append(F @ W.T + noise * rng.normal(size=(T, P)))
    return np.stack(out)


#: Homogeneous loadings, but short and noisy -- so the RAW alignment is only ~0.84-0.95.
HOMO_KW = dict(n_ent=6, T=100, P=12, R=3, noise=1.5, rot=0.0)
#: Heterogeneous loadings, but long and clean -- so the RAW alignment is ~0.86-0.94 too.
HETERO_KW = dict(n_ent=5, T=800, P=12, R=3, noise=0.1, rot=0.25)


def _true_mask(P=6):
    y = np.zeros((P, P), dtype=bool)
    for s, t in TRUE_EDGES:
        y[t, s] = True
    return y


def _auc(scores, truth, off_diag=True):
    P = scores.shape[0]
    keep = ~np.eye(P, dtype=bool) if off_diag else np.ones((P, P), bool)
    pos = scores[keep & truth]
    neg = scores[keep & ~truth]
    if pos.size == 0 or neg.size == 0:                          # pragma: no cover
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


# --------------------------------------------------------------------------- #
# 1. end to end recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_a_majority_of_the_true_lagged_edges(seed):
    """Zero-config on a raw NaN-holed panel: a majority of the truth comes back."""
    X = var_panel(seed=seed)
    g = discover(X)
    assert isinstance(g, CausalGraph)
    A = g.adjacency(lag="lagged", exclude_self=True).any(axis=0)
    found = {(j, i) for i in range(6) for j in range(6) if A[i, j]}
    hits = len(found & set(TRUE_EDGES))
    assert hits >= 3, f"recovered only {hits}/{len(TRUE_EDGES)}: {sorted(found)}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_rank_true_edges_above_false_ones(seed):
    """``scores`` is dense and monotone in evidence, so an AUC scorer can use it."""
    g = discover(var_panel(seed=seed))
    S = g.scores(lag="lagged", exclude_self=True).max(axis=0)
    assert S.shape == (6, 6)
    assert np.all((S >= 0.0) & (S <= 1.0))
    assert _auc(S, _true_mask()) >= 0.75


def test_scores_rank_selected_edges_above_unselected_ones_on_average():
    """`scores` is EVIDENCE, `adjacency` is the DECISION -- deliberately not a partition.

    An earlier design hard-tiered selected edges into [0.5, 1] and everything else into
    [0, 0.5).  That buried every true-but-unselected edge beneath every false-but-selected
    one and cost ~0.34 of AUC on ground truth (0.595 vs 0.936 for the continuous score),
    because a threshold-free metric needs a global ranking, not a step.  So we assert
    *stochastic* separation, not a partition: selection must shift the score distribution
    strongly upward while still letting a well-evidenced unselected pair outrank a weak
    selected one.
    """
    g = discover(var_panel(seed=0))
    A = np.asarray(g.adjacency(lag="lagged"), bool)
    S = np.asarray(g.scores(lag="lagged"), float)
    assert A.any(), "fixture should select something"
    assert S[A].mean() - S[~A].mean() > 0.15
    assert np.median(S[A]) > np.quantile(S[~A], 0.9)


def test_two_dimensional_input_also_recovers_structure():
    g = discover(var_series(seed=0, T=400))
    assert g.meta["shape"][0] == 1 and g.meta["engine"] == "per_entity"
    S = g.scores(lag="lagged", exclude_self=True).max(axis=0)
    assert _auc(S, _true_mask()) >= 0.7


def test_lagged_edges_are_oriented_by_time_for_free():
    g = discover(var_panel(seed=0))
    lagged = [e for e in g.edges if e.lag > 0]
    assert lagged
    assert all(e.orientation == "-->" for e in lagged)
    assert all(e.identification_method == "time" for e in lagged)
    assert all(e.confidence == 1.0 for e in lagged)


def test_contemporaneous_layer_is_routed_not_assumed():
    """Contemporaneous edges either carry an identification branch or abstain with a
    reason -- the assembler never silently orients them."""
    g = discover(var_panel(seed=0))
    for e in g.edges_at(0):
        assert e.identification_method in ("lingam", "cumulant", "dagma", "pag",
                                           "unidentified")
        if e.orientation != "-->":
            assert e.reason, "an abstention must be auditable"


# --------------------------------------------------------------------------- #
# 2. the point-in-time certificate
# --------------------------------------------------------------------------- #
def test_pit_mode_returns_one_graph_per_checkpoint():
    X = var_series(seed=0, T=240)
    G = discover(X, mode="pit", checkpoints=[100, 160, 220])
    assert isinstance(G, PitGraphs) and len(G) == 3
    assert G.checkpoints == [100, 160, 220]
    assert all(isinstance(g, CausalGraph) for g in G)
    assert G.at(160) is G[1]
    assert G.adjacency().shape == (3, 6, 6)
    assert G.scores(lag="lagged").shape[0] == 3
    assert "checkpoints" in G.summary()


def test_pit_truncation_certificate_is_exactly_zero():
    """``max_revision == 0`` EXACTLY on the 0/1 support indicator -- no tolerance."""
    X = var_series(seed=0, T=240)
    G = discover(X, mode="pit", checkpoints=[110, 170])
    rep = G.max_edge_revision(X, n_prefixes=4, checkpoints=[110, 170])
    assert rep["max_revision"] == 0.0
    assert rep["causal"] is True
    assert rep["n_comparisons"] > 0
    assert rep["protocol"] == "as_of"


def test_pit_certificate_holds_for_a_panel_and_for_every_lag_slice():
    X = var_panel(seed=0, n_ent=3, T=200)
    g = discover(X)
    for lag in (None, 0, 1, "lagged", "all"):
        rep = g.max_edge_revision(X, n_prefixes=3, checkpoints=[120], lag=lag)
        assert rep["max_revision"] == 0.0, lag


def test_pit_graphs_are_bit_identical_to_discovering_on_the_prefix():
    """``G_t`` is literally ``discover(X[:t+1])`` -- truncation-invariance by
    construction, not by convention."""
    X = var_series(seed=1, T=220)
    G = discover(X, mode="pit", checkpoints=[120, 200])
    for t, g in zip(G.checkpoints, G):
        ref = discover(X[: t + 1])
        assert np.array_equal(g.adjacency(lag="all"), ref.adjacency(lag="all"))
        assert np.array_equal(g.scores(lag="all"), ref.scores(lag="all"))


def test_the_certificate_has_teeth():
    """A reporter whose window is CENTRED on ``t`` borrows rows after ``t``; the same
    certificate must catch it.  Without this the zero above would be vacuous."""
    X = var_series(seed=0, T=240)

    def leaky(Z, t):
        Z = np.nan_to_num(np.asarray(Z, float), nan=0.0)
        lo, hi = max(0, int(t) - 60), min(Z.shape[0], int(t) + 60)
        W = np.diff(Z[lo:hi], axis=0)                  # window straddles t
        C = np.abs(np.nan_to_num(np.corrcoef(W.T), nan=0.0))
        np.fill_diagonal(C, 0.0)
        P = C.shape[0]
        iu = np.triu_indices(P, 1)
        keep = np.argsort(C[iu])[::-1][:P]             # top-P pairs
        A = np.zeros((P, P), dtype=bool)
        A[iu[0][keep], iu[1][keep]] = True
        return A | A.T

    rep = max_edge_revision(leaky, X, n_prefixes=4, checkpoints=[100, 150])
    assert rep["protocol"] == "as_of"
    assert rep["max_revision"] == 1.0
    assert rep["causal"] is False


def test_max_edge_revision_prefix_protocol_is_detected_from_arity():
    X = var_series(seed=0, T=200)
    rep = max_edge_revision(lambda Z: discover(Z), X, n_prefixes=3, checkpoints=[100])
    assert rep["protocol"] == "prefix"
    assert rep["max_revision"] == 0.0


def test_max_edge_revision_accepts_a_bare_array_reporter():
    """A rival that returns a plain adjacency matrix needs no adapter."""
    X = var_series(seed=0, T=200)
    P = X.shape[1]

    def frozen(Z, t):
        return np.eye(P, dtype=bool)

    rep = max_edge_revision(frozen, X, n_prefixes=3, checkpoints=[100])
    assert rep["max_revision"] == 0.0


# --------------------------------------------------------------------------- #
# 3. the engine router
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_router_pools_homogeneous_loadings(seed):
    X = loading_panel(seed=seed, **HOMO_KW)
    parts = loading_homogeneity_ratio(X, return_parts=True)
    assert parts["ratio"] >= 0.95, parts
    assert choose_engine(X)[0] == "pooled"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_router_separates_heterogeneous_loadings(seed):
    X = loading_panel(seed=seed, **HETERO_KW)
    parts = loading_homogeneity_ratio(X, return_parts=True)
    assert parts["ratio"] < 0.95, parts
    assert choose_engine(X)[0] == "per_entity"


def test_a_fixed_threshold_on_raw_alignment_cannot_do_this():
    """The point of normalising by the within-entity ceiling.

    The two fixtures are built so their RAW between-entity alignments interleave: the
    heterogeneous panel is cleaner and longer, so its raw alignment is as high as the
    homogeneous-but-noisy one's.  No absolute cutoff on raw alignment orders them
    correctly; the ratio -- whose null is 1 by construction -- does.
    """
    homo = [loading_homogeneity_ratio(loading_panel(seed=s, **HOMO_KW),
                                      return_parts=True) for s in range(4)]
    het = [loading_homogeneity_ratio(loading_panel(seed=s, **HETERO_KW),
                                     return_parts=True) for s in range(4)]
    raw_h = np.array([p["between"] for p in homo])
    raw_e = np.array([p["between"] for p in het])
    # the raw alignments overlap: some heterogeneous panels look MORE aligned than
    # some homogeneous ones, so no single cutoff can separate the two families
    assert raw_e.max() > raw_h.min() and raw_h.max() > raw_e.min()
    # the normalised ratio separates them cleanly
    assert min(p["ratio"] for p in homo) >= 0.95 > max(p["ratio"] for p in het)


def test_the_homogeneity_null_is_one():
    """Identical loadings => between and within are draws from the same distribution."""
    for seed in range(3):
        p = loading_homogeneity_ratio(
            loading_panel(n_ent=5, T=200, P=10, R=3, noise=0.3, rot=0.0, seed=seed),
            return_parts=True)
        assert abs(p["ratio"] - 1.0) < 0.05, p


def test_router_is_undefined_and_harmless_without_an_entity_axis():
    X = var_series(seed=0, T=200)
    assert not np.isfinite(loading_homogeneity_ratio(X[None][:, :, :]))
    eng, parts = choose_engine(X[None])
    assert eng == "per_entity" and parts["reason"]


def test_router_survives_missing_data():
    rng = np.random.default_rng(0)
    X = loading_panel(seed=0, **HOMO_KW)
    X[rng.random(X.shape) < 0.15] = np.nan
    assert choose_engine(X)[0] == "pooled"


def test_engine_choice_is_reported_and_overridable():
    X = var_panel(seed=0, n_ent=3, T=200)
    auto = discover(X)
    assert auto.meta["engine"] in ("pooled", "per_entity")
    assert np.isfinite(auto.meta["homogeneity_ratio"])
    forced = discover(X, engine="per_entity")
    assert forced.meta["engine"] == "per_entity"
    assert not np.isfinite(forced.meta["homogeneity_ratio"])   # router not consulted


# --------------------------------------------------------------------------- #
# 4. containers
# --------------------------------------------------------------------------- #
def test_numpy_2d_and_3d_shapes_and_lag_slices():
    g = discover(var_panel(seed=0, n_ent=2, T=200))
    P, K = g.n_vars, g.max_lag
    assert g.adjacency().shape == (P, P)
    assert g.adjacency(lag=0).shape == (P, P)
    assert g.adjacency(lag="lagged").shape == (K, P, P)
    assert g.adjacency(lag="all").shape == (K + 1, P, P)
    assert g.scores(lag="all").shape == (K + 1, P, P)
    # the summary graph is the OR over every lag
    assert np.array_equal(g.adjacency(), g.adjacency(lag="all").any(axis=0))
    with pytest.raises(ValueError):
        g.adjacency(lag=K + 5)
    with pytest.raises(ValueError):
        g.adjacency(lag="nonsense")


def test_pandas_round_trip_2d():
    pd = pytest.importorskip("pandas")
    X = var_series(seed=0, T=250)
    names = [f"v{i}" for i in range(X.shape[1])]
    g_np = discover(X)
    g_pd = discover(pd.DataFrame(X, columns=names))
    assert g_pd.names == names
    assert np.array_equal(g_np.adjacency(lag="all"), g_pd.adjacency(lag="all"))
    df = g_pd.to_pandas()
    assert len(df) == len(g_pd.edges)
    if len(df):
        assert set(df["source_name"]) <= set(names)
        assert list(df.columns) == list(CausalGraph._COLUMNS)


def test_polars_round_trip_2d():
    pl = pytest.importorskip("polars")
    X = var_series(seed=0, T=250)
    names = [f"v{i}" for i in range(X.shape[1])]
    g_np = discover(X)
    g_pl = discover(pl.DataFrame({c: X[:, i] for i, c in enumerate(names)}))
    assert g_pl.names == names
    assert np.array_equal(g_np.adjacency(lag="all"), g_pl.adjacency(lag="all"))
    assert g_pl.to_polars().shape == (len(g_pl.edges), len(CausalGraph._COLUMNS))


def test_long_format_panel_matches_the_equivalent_cube():
    pd = pytest.importorskip("pandas")
    X = var_panel(seed=0, n_ent=2, T=200)
    E, T, P = X.shape
    names = [f"v{i}" for i in range(P)]
    long = pd.DataFrame({
        "t": np.tile(np.arange(T), E),
        "e": np.repeat(np.arange(E), T),
        **{c: np.concatenate([X[k][:, i] for k in range(E)])
           for i, c in enumerate(names)},
    })
    g_cube = discover(X)
    g_long = discover(long, panel=("t", "e"))
    assert g_long.meta["shape"] == (E, T, P)
    assert g_long.names == names
    assert np.array_equal(g_cube.adjacency(lag="all"), g_long.adjacency(lag="all"))


def test_reusing_an_existing_pass_matches_a_fresh_discover():
    X = var_series(seed=0, T=250)
    res = cafe.CAFE().run(X)
    g_reuse = res.discover()
    g_fresh = discover(X)
    assert np.array_equal(g_reuse.adjacency(lag="all"), g_fresh.adjacency(lag="all"))
    assert g_reuse.meta["reused_result"] == "CafeResult"
    with pytest.raises(TypeError):
        res.discover(engine="pooled")


@pytest.mark.parametrize("engine", ["per_entity", "pooled"])
def test_reusing_a_panel_pass_matches_a_fresh_discover(engine):
    X = var_panel(seed=0, n_ent=3, T=200)
    res = cafe.CAFE().run(X, engine=engine)
    g_reuse = res.discover()
    g_fresh = discover(X, engine=engine)
    assert g_reuse.meta["engine"] == engine
    assert np.array_equal(g_reuse.adjacency(lag="all"), g_fresh.adjacency(lag="all"))


# --------------------------------------------------------------------------- #
# 5. degenerate input never raises
# --------------------------------------------------------------------------- #
def _degenerate_cases():
    rng = np.random.default_rng(0)
    yield "P=2", rng.normal(size=(60, 2))
    yield "P=1", rng.normal(size=(60, 1))
    yield "T=10", rng.normal(size=(10, 4))
    yield "T=3", rng.normal(size=(3, 4))
    yield "all-nan-column", np.column_stack(
        [rng.normal(size=200), np.full(200, np.nan), rng.normal(size=200)])
    yield "constant-column", np.column_stack(
        [rng.normal(size=200), np.ones(200), rng.normal(size=200)])
    yield "all-nan", np.full((80, 4), np.nan)
    yield "all-zero", np.zeros((60, 3))
    yield "one-entity-cube", rng.normal(size=(1, 60, 3))
    yield "tiny-panel", rng.normal(size=(2, 12, 3))


@pytest.mark.parametrize("name,X", list(_degenerate_cases()),
                         ids=[n for n, _ in _degenerate_cases()])
def test_degenerate_input_returns_a_valid_graph_and_never_raises(name, X):
    g = discover(X)
    P = X.shape[-1]
    assert isinstance(g, CausalGraph)
    assert g.n_vars == P and g.max_lag >= 1
    assert g.adjacency().shape == (P, P)
    assert g.adjacency(lag="all").shape == (g.max_lag + 1, P, P)
    assert g.scores(lag="all").shape == (g.max_lag + 1, P, P)
    assert np.all(np.isfinite(g.scores(lag="all")))
    assert isinstance(g.summary(), str) and g.summary()
    # anything the pipeline declined to do must be recorded, not silently dropped
    if not g.edges:
        assert g.notes or g.meta.get("n_candidates") is not None


def test_pit_mode_on_a_degenerate_input_still_returns_a_sequence():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    G = discover(X, mode="pit", checkpoints=[10, 20, 29])
    assert len(G) == 3 and all(g.n_vars == 3 for g in G)


def test_bad_arguments_raise_clearly():
    X = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ValueError):
        discover(X, mode="smoothing")
    with pytest.raises(TypeError):
        discover(X, not_a_real_option=3)


def test_default_max_lag_rule_is_documented_and_bounded():
    assert default_max_lag(3) == 1
    assert default_max_lag(100) == 4
    assert default_max_lag(1000) == 6
    assert default_max_lag(10 ** 7) <= 12
    for T in (6, 20, 60, 200, 5000):
        K = default_max_lag(T)
        assert 1 <= K <= 12 and 5 * K <= T


def test_summary_reports_the_decisions_that_were_made():
    g = discover(var_panel(seed=0, n_ent=3, T=200))
    s = g.summary()
    for token in ("CAFE-CD graph", "engine", "screening", "skeleton", "diagnostics"):
        assert token in s
    assert g.diagnostics is not None
    assert g.diagnostics.n_vars == g.n_vars
    assert g.n_edges == int(g.adjacency().sum())
    assert g.label(0) == "x0"


# --------------------------------------------------------------------------- #
# 6. numpy only
# --------------------------------------------------------------------------- #
def test_discovery_is_numpy_only():
    """``import cafe.discovery`` plus a full batch/pit run pulls in nothing heavy."""
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {os.path.join(HERE, '..')!r})\n"
        "import cafe, cafe.discovery\n"
        "rng = np.random.default_rng(0)\n"
        "X = rng.standard_normal((3, 120, 5))\n"
        "X[rng.random(X.shape) < 0.1] = np.nan\n"
        "g = cafe.discover(X)\n"
        "g.adjacency(); g.scores(); g.summary()\n"
        "G = cafe.discover(X, mode='pit', checkpoints=[80, 119])\n"
        "G.max_edge_revision(X, n_prefixes=2, checkpoints=[80])\n"
        "cafe.loading_homogeneity_ratio(X)\n"
        "heavy = [m for m in ('scipy', 'sklearn', 'pandas', 'polars', 'matplotlib',\n"
        "                     'numba', 'torch') if m in sys.modules]\n"
        "print('HEAVY=' + ','.join(heavy))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("HEAVY="), out.stdout


def test_the_public_surface_is_exported():
    for name in ("discover", "CausalGraph", "PitGraphs", "Edge",
                 "loading_homogeneity_ratio", "max_edge_revision"):
        assert hasattr(cafe, name), name
        assert name in cafe.__all__, name


# --------------------------------------------------------------------------- #
# 7. the scoring router -- structural, anchored, and it abstains by default
# --------------------------------------------------------------------------- #
# Two of the three lagged evidence components are FULLY CONDITIONAL (the refit |t| and
# the group lasso's path occupancy).  They are only readable as statements about one
# source while the design can tell the sources apart, so the assembler measures that
# and re-weights.  The measurement is a generalised-VIF reciprocal with a null of 1 by
# construction, the only threshold on it is the majority point 1/2, and it never sees
# anything but the design's own second moments.
def _lag_shifted(T=1200, P=4, rho=0.9, noise=0.15, seed=0):
    """``P`` series that are DELAYED copies of one driver.

    Their lag blocks overlap almost completely, so no conditional coefficient is
    identified -- and, unlike a contemporaneous duplicate, the common part is not
    something the factor adjustment upstream can project out, which is exactly the
    situation the router exists for.
    """
    rng = np.random.default_rng(seed)
    f = np.zeros(T + P)
    for t in range(1, T + P):
        f[t] = rho * f[t - 1] + rng.standard_normal()
    X = np.column_stack([f[P - j:P - j + T] for j in range(P)])
    return X + noise * rng.standard_normal((T, P))


def test_scoring_router_abstains_on_a_separable_design():
    """At the null the blend is exactly what it was before the router existed:
    ``w_bivariate`` is 0.0 EXACTLY, so the conditional refit keeps its whole share."""
    g = discover(var_panel(seed=0))
    c = g.meta["conditioning"]
    assert c["ok"] and not c["reason"]
    assert c["branch"] == "conditional"
    assert c["w_bivariate"] == 0.0
    assert c["source_tolerance"] > 0.5 and c["n_sources_below_half"] == 0
    assert not [n for n in g.notes if "scoring router" in n]


def test_scoring_router_moves_weight_to_bivariate_when_sources_are_not_separable():
    g = discover(_lag_shifted(noise=0.05), max_lag=3)
    c = g.meta["conditioning"]
    assert c["ok"]
    assert c["source_tolerance"] < 0.5 and c["n_sources_below_half"] == 4
    assert 0.0 < c["w_bivariate"] <= 1.0 and c["branch"] == "blended"
    assert [n for n in g.notes if "scoring router" in n], g.notes
    S = g.scores(lag="all")
    assert np.all(np.isfinite(S)) and np.all((S >= 0.0) & (S <= 1.0))


@pytest.mark.parametrize("noise,expect_fire", [(0.05, True), (0.3, True), (1.0, False)])
def test_scoring_router_weight_is_monotone_in_the_diagnostic(noise, expect_fire):
    """A blend, not a switch: the bivariate share falls continuously to exactly zero
    as the sources separate, and crosses zero at the majority point and nowhere else."""
    from cafe.discovery import SOURCE_SEPARABILITY
    c = discover(_lag_shifted(noise=noise), max_lag=3).meta["conditioning"]
    fired = c["w_bivariate"] > 0.0
    assert fired is expect_fire, c
    assert fired == (c["min_source_tolerance"] < SOURCE_SEPARABILITY)


def test_scoring_router_keeps_adjacency_and_scores_separate():
    """The router touches the EVIDENCE only.  The decision is the lasso support and
    must be untouched by how the evidence is weighted."""
    X = _lag_shifted(noise=0.05)
    g = discover(X, max_lag=3)
    assert g.meta["conditioning"]["w_bivariate"] > 0.0
    A = np.asarray(g.adjacency(lag="lagged"), bool)
    assert int(A.sum()) == int(g.meta["n_support"])


def test_scoring_router_decision_is_truncation_invariant():
    """It reads the design's moments on the prefix it is handed, so the graph at a
    checkpoint is still bit-identical to discovering on that prefix."""
    X = _lag_shifted(T=600, noise=0.05)
    G = discover(X, mode="pit", checkpoints=[300, 500], max_lag=3)
    for t, g in zip(G.checkpoints, G.graphs):
        ref = discover(X[: t + 1], max_lag=3)
        assert (g.meta["conditioning"]["w_bivariate"]
                == ref.meta["conditioning"]["w_bivariate"])
        assert np.array_equal(g.scores(lag="all"), ref.scores(lag="all"))
    rep = G.max_edge_revision(X, n_prefixes=3, checkpoints=[300, 500])
    assert rep["max_revision"] == 0.0


def test_conditioning_diagnostic_is_exported_from_discovery():
    from cafe.discovery import conditioning_diagnostic
    assert callable(conditioning_diagnostic)
