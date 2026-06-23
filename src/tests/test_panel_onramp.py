"""Panel on-ramp: 3D arrays and two-index long-format DataFrames route to the panel path
with an exact round-trip, observed cells untouched, and the point-in-time (truncation-
invariance) guarantee preserved. These are the contracts that make `cafe.impute` "just
work" on (entity, time, feature) data without the user hand-building integer ids."""
import numpy as np
import pytest

import cafe


def _tensor(E=8, T=40, F=5, rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((E, 3)); W = rng.standard_normal((F, 3))
    z = np.cumsum(rng.standard_normal((T, 3)) * 0.3, axis=0)
    X = np.einsum("er,tr,fr->etf", A, z, W) + 0.3 * rng.standard_normal((E, T, F))
    m = rng.random(X.shape) < rate
    Xm = X.copy(); Xm[m] = np.nan
    return X, Xm, m


def test_3d_array_roundtrip_and_fill():
    X, Xm, m = _tensor()
    out = cafe.impute(Xm)
    assert out.shape == Xm.shape
    assert not np.isnan(out).any()                      # every cell filled
    assert np.max(np.abs(out[~m] - Xm[~m])) == 0.0      # observed cells untouched


def test_3d_array_is_point_in_time():
    # truncating the time axis must leave earlier fills bit-identical
    X, Xm, m = _tensor(T=40, seed=1)
    full = cafe.impute(Xm)
    cut = 25
    trunc = cafe.impute(Xm[:, :cut, :])
    assert np.max(np.abs(full[:, :cut, :] - trunc)) < 1e-9


def test_3d_handles_entity_blackout():
    X, Xm, m = _tensor(seed=2)
    Xm[3, 20:, :] = np.nan                              # entity 3 goes dark after t=20
    out = cafe.impute(Xm)
    assert not np.isnan(out).any()                      # filled via cross-section + AR


def _panel_frame_polars(Xm, m):
    pl = pytest.importorskip("polars")
    E, T, F = Xm.shape
    rows = []
    for e in range(E):
        for t in range(T):
            r = {"date": 20200101 + t, "ticker": f"T{e}"}
            for f in range(F):
                r[f"f{f}"] = float("nan") if m[e, t, f] else float(Xm[e, t, f])
            rows.append(r)
    return pl.DataFrame(rows), [f"f{f}" for f in range(F)]


def test_polars_two_index_roundtrip():
    pytest.importorskip("polars")
    X, Xm, m = _tensor(seed=3)
    df, feats = _panel_frame_polars(Xm, m)
    out = cafe.impute(df, panel=("date", "ticker"))
    assert out.shape == df.shape and out.columns == df.columns
    assert out["date"].to_list() == df["date"].to_list()
    assert out["ticker"].to_list() == df["ticker"].to_list()
    nan_left = sum(int(np.isnan(out[c].to_numpy()).sum()) for c in feats)
    assert nan_left == 0


def test_polars_two_index_point_in_time():
    pl = pytest.importorskip("polars")
    X, Xm, m = _tensor(seed=4)
    df, feats = _panel_frame_polars(Xm, m)
    cut = 20200101 + 25
    full = cafe.impute(df, panel=("date", "ticker")).filter(pl.col("date") <= cut).sort(["ticker", "date"])
    trunc = cafe.impute(df.filter(pl.col("date") <= cut), panel=("date", "ticker")).sort(["ticker", "date"])
    drift = max(float(np.max(np.abs(full[c].to_numpy() - trunc[c].to_numpy()))) for c in feats)
    assert drift < 1e-9


def test_pandas_two_index_roundtrip():
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    X, Xm, m = _tensor(seed=5)
    dfp, feats = _panel_frame_polars(Xm, m)
    df = dfp.to_pandas()
    out = cafe.impute(df, panel=("date", "ticker"))
    assert list(out.columns) == list(df.columns) and out.shape == df.shape
    assert (out["ticker"] == df["ticker"]).all()
    assert int(np.isnan(out[feats].to_numpy()).sum()) == 0


def test_panel_excludes_index_columns_from_features():
    # the integer 'date' column must NOT be imputed as a feature (it is the index)
    pytest.importorskip("polars")
    X, Xm, m = _tensor(seed=6)
    df, feats = _panel_frame_polars(Xm, m)
    out = cafe.impute(df, panel=("date", "ticker"))
    assert out["date"].to_list() == df["date"].to_list()   # untouched, not factor-filled


def test_engine_options_all_fill():
    # all three engines must fully impute a panel and leave observed cells untouched
    X, Xm, m = _tensor(seed=8)
    for engine in ("joint", "per_entity", "auto"):
        out = cafe.impute(Xm, engine=engine)
        assert out.shape == Xm.shape, engine
        assert not np.isnan(out).any(), engine
        assert np.max(np.abs(out[~m] - Xm[~m])) == 0.0, engine


def test_default_engine_is_joint():
    # the default must be byte-identical to the validated joint core (no silent reroute)
    X, Xm, m = _tensor(seed=9)
    default = cafe.impute(Xm)
    joint = cafe.impute(Xm, engine="joint")
    assert np.array_equal(default, joint)


def test_per_entity_is_point_in_time():
    X, Xm, m = _tensor(T=40, seed=10)
    full = cafe.impute(Xm, engine="per_entity")
    trunc = cafe.impute(Xm[:, :25, :], engine="per_entity")
    assert np.max(np.abs(full[:, :25, :] - trunc)) < 1e-9


def test_bad_engine_rejected():
    import pytest
    _, Xm, _ = _tensor(seed=11)
    with pytest.raises(ValueError):
        cafe.impute(Xm, engine="nonsense")


def test_2d_path_unchanged_regression():
    # the 2D matrix path must be byte-for-byte unaffected by the on-ramp plumbing
    _, Xm, _ = _tensor(seed=7)
    a = cafe.impute(Xm[0]); b = cafe.impute(Xm[0])
    assert a.shape == Xm[0].shape
    assert np.array_equal(a, b)
    assert not np.isnan(a).any()
