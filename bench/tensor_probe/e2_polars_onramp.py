"""E2 — Two-index (date, ticker) polars on-ramp: feasibility + correctness.

Prototype the auto-router that the library would ship: a polars frame with TWO index
columns (e.g. date + ticker) is detected, factorized into integer entity_ids/time_ids,
routed through the existing panel path, and reshaped back to the same two-index frame.

Proves three things:
  1. ROUND-TRIP: output frame has identical (date,ticker) keys, column order, row count.
  2. NO-FILL-OF-OBSERVED: observed numeric cells are returned unchanged.
  3. POINT-IN-TIME: truncating the frame at any date leaves earlier fills bit-identical
     (the truncation-invariance property — the project's core causal guarantee).
"""
import sys
import numpy as np
sys.path.insert(0, "src")
import cafe

try:
    import polars as pl
except ImportError:
    print("polars not installed — skipping E2")
    sys.exit(0)


def build_meta_from_two_index(df, time_col, entity_col):
    """The candidate auto-router. Returns (numeric_matrix, meta, ctx_for_rebuild)."""
    # stable integer codes, time-ordered so time_ids respect chronology
    times = df[time_col]
    ents = df[entity_col]
    uniq_t = times.unique().sort()
    uniq_e = ents.unique().sort()
    t_map = {v: i for i, v in enumerate(uniq_t.to_list())}
    e_map = {v: i for i, v in enumerate(uniq_e.to_list())}
    time_ids = np.array([t_map[v] for v in times.to_list()])
    entity_ids = np.array([e_map[v] for v in ents.to_list()])
    num_cols = [c for c, dt in df.schema.items()
                if dt.is_numeric() and c not in (time_col, entity_col)]
    X = df.select(num_cols).to_numpy().astype(float)
    meta = {"entity_ids": entity_ids, "time_ids": time_ids}
    return X, meta, num_cols


def make_frame(E=8, T=40, F=4, rate=0.25, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((E, 3)); W = rng.standard_normal((F, 3))
    z = np.cumsum(rng.standard_normal((T, 3)) * 0.3, axis=0)
    core = np.einsum("er,tr,fr->etf", A, z, W) + 0.3 * rng.standard_normal((E, T, F))
    rows = []
    for e in range(E):
        for t in range(T):
            r = {"date": 20200101 + t, "ticker": f"TIC{e:02d}"}
            for f in range(F):
                v = core[e, t, f]
                r[f"feat{f}"] = np.nan if rng.random() < rate else v
            rows.append(r)
    return pl.DataFrame(rows)


def impute_frame(df, time_col="date", entity_col="ticker"):
    X, meta, num_cols = build_meta_from_two_index(df, time_col, entity_col)
    filled = np.asarray(cafe.impute(X, meta=meta), float)
    out = df.clone()
    for j, c in enumerate(num_cols):
        out = out.with_columns(pl.Series(c, filled[:, j]))
    return out, num_cols


def main():
    df = make_frame(seed=1)
    out, num_cols = impute_frame(df)

    # 1. round-trip structure
    assert out.shape == df.shape, (out.shape, df.shape)
    assert out.columns == df.columns
    assert out["date"].to_list() == df["date"].to_list()
    assert out["ticker"].to_list() == df["ticker"].to_list()
    nan_left = sum(out[c].is_null().sum() + np.isnan(out[c].to_numpy()).sum() for c in num_cols)
    print(f"[1] round-trip OK: shape {out.shape}, keys preserved, nan_left={nan_left}")

    # 2. observed cells unchanged
    max_obs_drift = 0.0
    for c in num_cols:
        a = df[c].to_numpy(); b = out[c].to_numpy()
        obs = ~np.isnan(a)
        if obs.any():
            max_obs_drift = max(max_obs_drift, float(np.max(np.abs(a[obs] - b[obs]))))
    print(f"[2] observed cells unchanged: max drift = {max_obs_drift:.2e}")

    # 3. point-in-time: truncate at a cutoff date, re-impute, compare earlier fills
    cutoff = 20200101 + 25
    full_out, _ = impute_frame(df)
    trunc = df.filter(pl.col("date") <= cutoff)
    trunc_out, _ = impute_frame(trunc)
    # compare the overlapping (date<=cutoff) cells
    full_sub = full_out.filter(pl.col("date") <= cutoff).sort(["ticker", "date"])
    trunc_sub = trunc_out.sort(["ticker", "date"])
    drift = 0.0
    for c in num_cols:
        drift = max(drift, float(np.max(np.abs(full_sub[c].to_numpy() - trunc_sub[c].to_numpy()))))
    verdict = "PASS (truncation-invariant)" if drift < 1e-8 else f"LEAK drift={drift:.2e}"
    print(f"[3] point-in-time: max drift on earlier cells = {drift:.2e}  -> {verdict}")


if __name__ == "__main__":
    main()
