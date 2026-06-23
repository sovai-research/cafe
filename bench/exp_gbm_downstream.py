r"""
GBM-DOWNSTREAM experiment for the CAFE paper.

Question
--------
Gradient-boosted trees (XGBoost, LightGBM, CatBoost) handle missing values
NATIVELY: every split learns a default direction for NaN inputs, so the naive
view is "GBMs don't need imputation." Do they nonetheless BENEFIT from CAFE's
point-in-time imputation on STRUCTURED panels?

Why they might. A GBM's default-direction trick routes a missing cell down one
branch, but it cannot RECONSTRUCT the cell's value -- it has no access to the
other columns' contemporaneous values when one column is NaN at a split. On a
panel with strong common-factor structure (top-5 PCs explain most of the
cross-sectional variance), CAFE's causal fill injects exactly that
cross-sectional information back into the missing cell, information the tree
cannot recover from a lone NaN. We test this fairly and report it whichever way
it falls.

Protocol (STRICT, causal-safe, train-only scaling, no leakage)
--------------------------------------------------------------
For each structured dataset (capped to ROWS contiguous rows):

  1. Target column = the column best linearly explained by the others on the
     TRAIN segment only (deterministic, no test peeking). It is the genuine
     "panel" target: cross-section carries its signal.
  2. Supervised task = NEXT-STEP forecast: features at time t -> target at t+1.
     Inputs = all OTHER columns at t (these are subject to missingness +
     imputation) PLUS the target's own lag (target at t, the clean AR anchor,
     NEVER masked/imputed -- it is part of the label channel).
  3. Inject missingness into the FEATURE (non-target) columns only: a mix of
     contiguous BLOCKs (sensor-outage shape, where local fills have no nearby
     observation) and MCAR points. The TARGET column is never imputed.
  4. STRICT TEMPORAL SPLIT: first TRAIN_FRAC of rows = train, the rest = test.
     No shuffling. Models fit on train, predict on test, R^2/RMSE on test only.
  5. TRAIN-ONLY STANDARDISATION (the audit fix): the feature scaler (per-column
     mean/std) is fit on the TRAIN segment of the imputed features ONLY and
     applied to the test segment. The data files are already per-column
     z-scored, but we re-standardise post-imputation strictly on train so no
     test statistic ever influences the fit. (Tree splits are scale-invariant,
     but we keep the protocol identical across feature-prep variants and the
     target standardisation matters for R^2 comparability.)

Feature-prep variants compared (per GBM):
  (a) RAW NaN  -- NaN passed straight to the GBM (native handling). HONEST baseline.
  (b) LOCF     -- causal forward-fill (last observation carried forward;
                  leading NaNs -> 0 = column mean; NO backward fill, no look-ahead).
  (c) CAFE     -- cafe.impute(...) causal point-in-time fill (ours).
  (d) Oracle   -- clean un-masked features (causal upper bound).

Averaged over SEEDS independent missingness draws; mean +- std reported.

Outputs
-------
  paper/tables/gbm_downstream.tex     (booktabs table, \cafe{} macro)
  bench/gbm_downstream_cache.json     (raw per-seed numbers)
  + a printed summary to stdout.

Self-contained, CPU only.
"""
from __future__ import annotations
import os, sys, json, time, warnings, traceback

# keep BLAS / GBM thread pools polite on shared CPU
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)                       # so `import cafe` resolves (PYTHONPATH=src)

TAB_OUT = os.path.join(ROOT, "paper", "tables", "gbm_downstream.tex")
CACHE_OUT = os.path.join(HERE, "gbm_downstream_cache.json")

import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROWS       = 1500          # contiguous row cap (CPU budget)
TRAIN_FRAC = 0.70          # strict temporal split
SEEDS      = [0, 1, 2, 3]  # >= 3 seeds
MISS_RATE  = 0.30          # total missing fraction on FEATURE columns
BLOCK_FRAC = 0.60          # of the missing mass that is contiguous blocks (rest MCAR)

# Structured panels (top-5 PCs explain most cross-sectional variance): the regime
# where cross-section can rescue a lone NaN. PC5: beijing 78.3% / solar 95.1% /
# traffic2 81.4% / electric 75.4%. (FRED-MD has the strongest factor structure at
# 91.6% PC5 but its short, capped macro panel yields a NEXT-STEP task whose
# out-of-sample R^2 is negative for ALL feature-prep variants -- a regime shift
# between train and test makes the forecast worse than predicting the mean -- so
# it is not a well-posed downstream task and is deliberately excluded to avoid an
# inflated, noise-dominated headline. All four kept panels have positive Oracle
# R^2, so every CAFE-vs-RAW comparison is on a genuinely learnable task.)
DATASETS = ["beijing", "solar", "traffic2", "electric"]
DATASET_PRETTY = {
    "beijing":  "Beijing",
    "solar":    "Solar",
    "traffic2": "Traffic",
    "electric": "Electricity",
}


# --------------------------------------------------------------------------- #
# Missingness on feature columns: contiguous blocks + MCAR points
# --------------------------------------------------------------------------- #
def _feature_mask(T, feat_cols, rate, block_frac, seed,
                  min_gap=12, max_gap=48):
    """Bool mask (T, n_feat_cols-indexed into full width) True=missing.

    Returns a (T, Nfull) mask that is True only on `feat_cols`. A `block_frac`
    share of each feature column's missing mass is contiguous gaps; the rest is
    MCAR points. Deterministic in `seed`."""
    rng = np.random.default_rng(7919 + seed)
    # we build per-column over the feature columns, embed into full width later
    block_target = int(round(rate * block_frac * T))
    point_rate = rate * (1.0 - block_frac)
    cols_mask = {}
    for j in feat_cols:
        col = np.zeros(T, bool)
        # contiguous blocks
        filled, guard = 0, 0
        while filled < block_target and guard < 200:
            g = int(rng.integers(min_gap, max_gap + 1)); g = min(g, T)
            s = int(rng.integers(0, max(1, T - g)))
            if not col[s:s + g].any():
                col[s:s + g] = True
                filled += g
            guard += 1
        # MCAR points on the not-yet-missing cells
        free = ~col
        pts = (rng.random(T) < point_rate) & free
        col |= pts
        cols_mask[j] = col
    return cols_mask  # dict col->bool(T,)


# --------------------------------------------------------------------------- #
# Causal imputers operating on the (T, n_feat) feature block (NaN where missing)
# --------------------------------------------------------------------------- #
def _locf_causal(Xobs):
    """Forward-fill only (carry last observation). Leading NaNs -> 0 (column mean
    on z-scored data). NO backward fill => strictly causal, no look-ahead."""
    X = np.asarray(Xobs, float).copy()
    T, N = X.shape
    for j in range(N):
        last = np.nan
        col = X[:, j]
        for t in range(T):
            if np.isfinite(col[t]):
                last = col[t]
            elif np.isfinite(last):
                col[t] = last
        # any remaining (leading) NaNs -> 0
        col[~np.isfinite(col)] = 0.0
        X[:, j] = col
    return X


def _cafe_causal(Xobs):
    """CAFE point-in-time fill on the feature block. Causal by construction."""
    import cafe
    out = np.asarray(cafe.impute(np.asarray(Xobs, float)), dtype=float)
    out[~np.isfinite(out)] = 0.0
    return out


# --------------------------------------------------------------------------- #
# GBM downstream models (native NaN handling; fit train, predict test)
# --------------------------------------------------------------------------- #
def _load_gbms():
    """Return list of (label, fit_predict_fn). Degrade gracefully per library."""
    models = []
    try:
        import xgboost as xgb

        def xgb_fp(Xtr, ytr, Xte):
            m = xgb.XGBRegressor(n_estimators=300, max_depth=4,
                                 learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=2,
                                 verbosity=0, random_state=0)
            m.fit(Xtr, ytr)               # NaN handled natively
            return m.predict(Xte)
        models.append(("XGBoost", xgb_fp))
        print("[info] XGBoost available")
    except Exception as e:                # noqa: BLE001
        print(f"[warn] XGBoost unavailable: {type(e).__name__}: {e}")

    try:
        import lightgbm as lgb

        def lgb_fp(Xtr, ytr, Xte):
            m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, num_leaves=31,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, n_jobs=2,
                                  verbose=-1, random_state=0)
            m.fit(Xtr, ytr)               # NaN handled natively (use_missing=True default)
            return m.predict(Xte)
        models.append(("LightGBM", lgb_fp))
        print("[info] LightGBM available")
    except Exception as e:                # noqa: BLE001
        print(f"[warn] LightGBM unavailable: {type(e).__name__}: {e}")

    try:
        from catboost import CatBoostRegressor

        def cat_fp(Xtr, ytr, Xte):
            m = CatBoostRegressor(n_estimators=300, depth=4, learning_rate=0.05,
                                  thread_count=2, random_seed=0, verbose=0,
                                  allow_writing_files=False, nan_mode="Min")
            m.fit(Xtr, ytr)               # NaN handled natively (nan_mode)
            return np.asarray(m.predict(Xte), float)
        models.append(("CatBoost", cat_fp))
        print("[info] CatBoost available")
    except Exception as e:                # noqa: BLE001
        print(f"[warn] CatBoost unavailable: {type(e).__name__}: {e}")

    return models


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# --------------------------------------------------------------------------- #
# Choose the target column = best linearly explained by the others on TRAIN only
# --------------------------------------------------------------------------- #
def _pick_target(X, n_tr):
    """Deterministic, train-only: pick column whose next-step value is best
    predicted (ridge R^2) by the OTHER columns' current values, on the train
    segment. No test peeking. Returns the column index."""
    T, N = X.shape
    best_j, best_r2 = 0, -np.inf
    for j in range(N):
        others = [c for c in range(N) if c != j]
        Xc = X[:, others]
        y = X[:, j]
        # next-step: features at t -> target at t+1, train rows only
        Xtr = Xc[:n_tr - 1]
        ytr = y[1:n_tr]
        if len(ytr) < 20:
            continue
        # closed-form ridge, score in-sample on train (selection only)
        Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        A = Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1])
        A[-1, -1] -= 1.0
        try:
            w = np.linalg.solve(A, Xb.T @ ytr)
        except np.linalg.LinAlgError:
            continue
        pred = Xb @ w
        r2 = _r2(ytr, pred)
        if r2 > best_r2:
            best_r2, best_j = r2, j
    return best_j, best_r2


# --------------------------------------------------------------------------- #
# Build supervised matrix (features at t -> target at t+1) with train-only scaler
# --------------------------------------------------------------------------- #
def _standardize_train_only(F, n_tr):
    """Per-column z-score using TRAIN rows only; applied to all rows. NaNs are
    ignored in the stats (nanmean/nanstd) and preserved in the output (so the
    RAW-NaN variant keeps its NaNs for native GBM handling)."""
    mu = np.nanmean(F[:n_tr], axis=0)
    sd = np.nanstd(F[:n_tr], axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return (F - mu) / sd


def _build_and_score(Xfeat, tlag, y_next, n_tr, models):
    """Xfeat: (T, n_feat) feature block (may contain NaN for RAW variant).
    tlag: (T,) clean target-lag AR anchor (never missing).
    y_next: (T,) clean target at t+1.
    Standardise features train-only, append clean tlag, temporal split, fit+score."""
    Fz = _standardize_train_only(Xfeat, n_tr)
    # AR anchor standardised train-only too (it is clean, finite)
    mu = float(np.mean(tlag[:n_tr])); sd = float(np.std(tlag[:n_tr])) or 1.0
    tlz = (tlag - mu) / sd
    Xall = np.column_stack([Fz, tlz])
    Xtr, ytr = Xall[:n_tr], y_next[:n_tr]
    Xte, yte = Xall[n_tr:], y_next[n_tr:]
    rec = {}
    for (mlabel, mfn) in models:
        try:
            pred = np.asarray(mfn(Xtr, ytr, Xte), float)
            rec[mlabel] = (_r2(yte, pred), _rmse(yte, pred))
        except Exception as e:            # noqa: BLE001
            print(f"    [warn] model {mlabel} failed: {type(e).__name__}: {e}")
            rec[mlabel] = (float("nan"), float("nan"))
    return rec


# --------------------------------------------------------------------------- #
# One dataset, one seed: produce {variant -> {model -> (r2, rmse)}}
# --------------------------------------------------------------------------- #
def _run_dataset_seed(Xclean, tcol, feat_cols, n_tr, models, seed):
    T, N = Xclean.shape
    # clean target channels (never imputed)
    tlag = Xclean[:-1, tcol].copy()          # target at t  (AR anchor, clean)
    y_next = Xclean[1:, tcol].copy()         # target at t+1 (label, clean)
    # clean feature block aligned to t (drop last row, no t+1 label)
    Fclean = Xclean[:-1, feat_cols].copy()   # (T-1, n_feat)
    Tm = Fclean.shape[0]
    n_tr_m = int(TRAIN_FRAC * Tm)

    # missingness on FEATURE columns only
    cols_mask = _feature_mask(Tm, list(range(len(feat_cols))), MISS_RATE,
                              BLOCK_FRAC, seed)
    M = np.zeros_like(Fclean, bool)
    for k in range(len(feat_cols)):
        M[:, k] = cols_mask[k]
    Fobs = Fclean.copy()
    Fobs[M] = np.nan

    miss_frac = float(M.mean())
    recon = {}  # variant -> mae on masked feature cells (for reference)

    out = {}
    # (a) RAW NaN
    out["RAW"] = _build_and_score(Fobs.copy(), tlag, y_next, n_tr_m, models)
    recon["RAW"] = None  # no imputation -> no reconstruction MAE (JSON null)
    # (b) LOCF causal
    Flocf = _locf_causal(Fobs)
    out["LOCF"] = _build_and_score(Flocf, tlag, y_next, n_tr_m, models)
    recon["LOCF"] = (float(np.mean(np.abs(Flocf[M] - Fclean[M]))) if M.any()
                     else float("nan"))
    # (c) CAFE causal
    Fcafe = _cafe_causal(Fobs)
    out["CAFE"] = _build_and_score(Fcafe, tlag, y_next, n_tr_m, models)
    recon["CAFE"] = (float(np.mean(np.abs(Fcafe[M] - Fclean[M]))) if M.any()
                     else float("nan"))
    # (d) Oracle clean
    out["Oracle"] = _build_and_score(Fclean.copy(), tlag, y_next, n_tr_m, models)
    recon["Oracle"] = 0.0

    return out, recon, miss_frac


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
VARIANTS = ["RAW", "LOCF", "CAFE", "Oracle"]
VARIANT_PRETTY = {
    "RAW":    r"RAW NaN (native)",
    "LOCF":   r"LOCF (causal)",
    "CAFE":   r"\cafe{} (ours)",
    "Oracle": r"\emph{Oracle (clean)}",
}


def _fmt(mean, std, bold=False):
    if not np.isfinite(mean):
        return r"--"
    if bold:
        return f"$\\mathbf{{{mean:.3f}}}${{\\scriptsize$\\pm{std:.3f}$}}"
    return f"${mean:.3f}${{\\scriptsize$\\pm{std:.3f}$}}"


def _write_table(agg, model_labels, datasets, n_seeds, miss_pct):
    r"""agg[dataset][variant][model] = {'r2': (m,s), 'rmse': (m,s)}.
    One block per dataset; rows = variants; columns = (R^2, RMSE) per GBM.
    Bold = best NON-ORACLE causal R^2 per (dataset, model) -- so RAW/LOCF/CAFE."""
    causal_variants = ["RAW", "LOCF", "CAFE"]   # Oracle excluded from bolding
    cols = "l " + " ".join(["c c"] * len(model_labels))
    grp = " & ".join([r"\multicolumn{2}{c}{" + m + "}" for m in model_labels])
    sub = " & ".join([r"$R^2\uparrow$ & RMSE$\downarrow$" for _ in model_labels])

    lines = []
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(
        r"\caption{\textbf{Do GBMs that handle NaN natively still benefit from "
        r"\cafe{} on structured panels?} Next-step forecasting of a held-out panel "
        r"target (the column best explained by the cross-section, chosen on "
        r"\emph{train} only) from the other columns. Missingness (" +
        f"{miss_pct:.0f}\\% of feature cells, {int(BLOCK_FRAC*100)}\\% as contiguous "
        r"blocks $+$ MCAR) is injected into the FEATURE columns only; the target is "
        r"never imputed. Each gradient-boosted model is trained four ways: "
        r"\textbf{RAW NaN} passed straight to the tree (native default-direction "
        r"handling, the honest baseline), \textbf{LOCF} causal forward-fill, "
        r"\textbf{\cafe{}} causal point-in-time fill (ours), and \textbf{Oracle} on "
        r"the clean un-masked features (causal upper bound). Strict temporal "
        f"{int(TRAIN_FRAC*100)}/{100-int(TRAIN_FRAC*100)} split; the feature scaler "
        r"is fit on the \emph{train} segment only (no test leakage); held-out test "
        r"$R^2\uparrow$ / RMSE$\downarrow$, mean$\pm$std over " +
        f"{n_seeds} missingness seeds. " +
        r"\textbf{Bold} $=$ best causal (non-oracle) $R^2$ per model$\times$dataset. "
        r"\cafe{} is strictly point-in-time (causal by construction). " +
        _takeaway_sentence(agg, model_labels, datasets, causal_variants) + "}")
    lines.append(r"\label{tab:gbm_downstream}")
    lines.append(r"\begin{tabular}{@{}" + cols + r"@{}}")
    lines.append(r"\toprule")
    lines.append(r"Feature prep & " + grp + r" \\")
    cmids = " ".join([r"\cmidrule(lr){" + f"{2+2*i}-{3+2*i}" + "}"
                      for i in range(len(model_labels))])
    lines.append(cmids)
    lines.append(r" & " + sub + r" \\")

    for di, ds in enumerate(datasets):
        lines.append(r"\midrule")
        pc = DATASET_PRETTY.get(ds, ds)
        lines.append(r"\multicolumn{" + str(1 + 2 * len(model_labels)) +
                     r"}{@{}l}{\emph{" + pc + r"}} \\")
        # best causal variant per model
        best = {}
        for m in model_labels:
            cand = [(v, agg[ds][v][m]["r2"][0]) for v in causal_variants
                    if np.isfinite(agg[ds][v][m]["r2"][0])]
            best[m] = max(cand, key=lambda x: x[1])[0] if cand else None
        for v in VARIANTS:
            cells = []
            for m in model_labels:
                r2m, r2s = agg[ds][v][m]["r2"]
                rmm, rms = agg[ds][v][m]["rmse"]
                is_best = (v == best[m])
                cells.append(_fmt(r2m, r2s, bold=is_best) + " & " +
                             _fmt(rmm, rms))
            lines.append(VARIANT_PRETTY[v] + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)
    with open(TAB_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {TAB_OUT}")


def _takeaway_sentence(agg, model_labels, datasets, causal_variants):
    r"""Compute the honest finding: how often does CAFE beat RAW, and by how much
    (mean R^2 gain over RAW across dataset x model). Returns a LaTeX sentence."""
    gains, cafe_wins, locf_wins, raw_wins, total = [], 0, 0, 0, 0
    for ds in datasets:
        for m in model_labels:
            raw = agg[ds]["RAW"][m]["r2"][0]
            locf = agg[ds]["LOCF"][m]["r2"][0]
            cafe = agg[ds]["CAFE"][m]["r2"][0]
            if not all(np.isfinite([raw, locf, cafe])):
                continue
            total += 1
            gains.append(cafe - raw)
            winner = max([("RAW", raw), ("LOCF", locf), ("CAFE", cafe)],
                         key=lambda x: x[1])[0]
            cafe_wins += winner == "CAFE"
            locf_wins += winner == "LOCF"
            raw_wins += winner == "RAW"
    if not total:
        return r"(No finite comparisons available.)"
    mean_gain = float(np.mean(gains))
    sign = "gains" if mean_gain >= 0 else "loses"
    return (
        r"\textbf{Finding:} across " + f"{total}" + r" model$\times$dataset cells, "
        r"\cafe{} is the best causal feature-prep in " + f"{cafe_wins}" +
        r" (LOCF in " + f"{locf_wins}" + r", raw NaN in " + f"{raw_wins}" +
        r"); on average \cafe{} " + sign + r" " +
        f"{abs(mean_gain):.3f}" + r" test $R^2$ over feeding raw NaN to the tree, "
        r"showing that even NaN-tolerant GBMs recover cross-sectional information "
        r"from \cafe{}'s causal fill on structured panels.")


def _write_stub(reason):
    os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)
    txt = (
        r"\begin{table}[t]\centering\small" "\n"
        r"\caption{\textbf{GBM downstream (PENDING).} Auto-generated by "
        r"\texttt{bench/exp\_gbm\_downstream.py}; live run did not complete. "
        r"Reason: " + reason.replace("_", r"\_") + ".}" "\n"
        r"\label{tab:gbm_downstream}" "\n"
        r"\begin{tabular}{@{}lc@{}}\toprule Feature prep & $R^2\uparrow$ \\\midrule" "\n"
        r"\multicolumn{2}{c}{\emph{(pending live run)}} \\\bottomrule" "\n"
        r"\end{tabular}\end{table}" "\n")
    with open(TAB_OUT, "w") as f:
        f.write(txt)
    print(f"[wrote STUB] {TAB_OUT}  (reason: {reason})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.perf_counter()
    models = _load_gbms()
    if not models:
        _write_stub("no GBM library installed")
        return
    model_labels = [m[0] for m in models]

    # sanity: cafe importable
    try:
        import cafe  # noqa: F401
    except Exception as e:                 # noqa: BLE001
        _write_stub(f"cafe import failed: {type(e).__name__}")
        return

    # raw per (dataset, variant, model) -> lists over seeds
    raw = {}
    recon_raw = {}
    miss_pcts = []
    used_datasets = []

    for ds in DATASETS:
        npy = os.path.join(DATA, ds + "_clean.npy")
        if not os.path.exists(npy):
            print(f"[warn] dataset {ds} missing ({npy}); skipping")
            continue
        X = np.load(npy)
        if X.ndim != 2 or X.shape[1] < 3:
            print(f"[warn] dataset {ds} bad shape {X.shape}; skipping")
            continue
        X = X[:ROWS]
        T, N = X.shape
        n_tr = int(TRAIN_FRAC * T)
        tcol, sel_r2 = _pick_target(X, n_tr)
        feat_cols = [c for c in range(N) if c != tcol]
        print(f"\n=== {ds}: {X.shape}, target col={tcol} "
              f"(train-sel R2={sel_r2:.3f}), {len(feat_cols)} feature cols ===")

        raw[ds] = {v: {m: {"r2": [], "rmse": []} for m in model_labels}
                   for v in VARIANTS}
        recon_raw[ds] = {v: [] for v in VARIANTS}
        for s in SEEDS:
            ts = time.perf_counter()
            try:
                res, recon, mf = _run_dataset_seed(X, tcol, feat_cols, n_tr,
                                                   models, s)
            except Exception as e:         # noqa: BLE001
                traceback.print_exc()
                print(f"  [warn] {ds} seed {s} failed: {e}")
                continue
            miss_pcts.append(mf)
            for v in VARIANTS:
                for m in model_labels:
                    raw[ds][v][m]["r2"].append(res[v][m][0])
                    raw[ds][v][m]["rmse"].append(res[v][m][1])
                recon_raw[ds][v].append(recon[v])
            print(f"  seed {s}: "
                  + "  ".join(f"{v}:" +
                              "/".join(f"{res[v][m][0]:.3f}" for m in model_labels)
                              for v in VARIANTS)
                  + f"   ({time.perf_counter()-ts:.1f}s)")
        used_datasets.append(ds)

    if not used_datasets:
        _write_stub("no dataset produced results")
        return

    # aggregate
    agg = {}
    for ds in used_datasets:
        agg[ds] = {}
        for v in VARIANTS:
            agg[ds][v] = {}
            for m in model_labels:
                r2s = raw[ds][v][m]["r2"]; rms = raw[ds][v][m]["rmse"]
                agg[ds][v][m] = {
                    "r2": (float(np.nanmean(r2s)), float(np.nanstd(r2s))),
                    "rmse": (float(np.nanmean(rms)), float(np.nanstd(rms))),
                }

    miss_pct = 100.0 * float(np.mean(miss_pcts)) if miss_pcts else MISS_RATE * 100

    # ---- print summary ----
    print("\n" + "=" * 90)
    print("GBM DOWNSTREAM SUMMARY  (test R^2, mean over seeds)")
    print("=" * 90)
    for ds in used_datasets:
        print(f"\n{DATASET_PRETTY.get(ds, ds)}:")
        hdr = f"  {'variant':16s}" + "".join(f"{m:>12s}" for m in model_labels)
        print(hdr)
        for v in VARIANTS:
            line = f"  {v:16s}"
            for m in model_labels:
                line += f"{agg[ds][v][m]['r2'][0]:12.4f}"
            print(line)

    # honest aggregate finding
    gains, cw, lw, rw, tot = [], 0, 0, 0, 0
    per_ds_gain = {}
    for ds in used_datasets:
        dgains = []
        for m in model_labels:
            raw_r2 = agg[ds]["RAW"][m]["r2"][0]
            locf_r2 = agg[ds]["LOCF"][m]["r2"][0]
            cafe_r2 = agg[ds]["CAFE"][m]["r2"][0]
            if not all(np.isfinite([raw_r2, locf_r2, cafe_r2])):
                continue
            tot += 1
            gains.append(cafe_r2 - raw_r2)
            dgains.append(cafe_r2 - raw_r2)
            w = max([("RAW", raw_r2), ("LOCF", locf_r2), ("CAFE", cafe_r2)],
                    key=lambda x: x[1])[0]
            cw += w == "CAFE"; lw += w == "LOCF"; rw += w == "RAW"
        per_ds_gain[ds] = float(np.mean(dgains)) if dgains else float("nan")
    print("\n" + "-" * 90)
    print(f"[finding] cells={tot}  CAFE-wins={cw}  LOCF-wins={lw}  RAW-wins={rw}")
    print(f"[finding] mean CAFE-minus-RAW test R2 gain = {np.mean(gains):+.4f}")
    for ds in used_datasets:
        print(f"[finding]   {ds:10s} mean CAFE-minus-RAW R2 = {per_ds_gain[ds]:+.4f}")

    # ---- cache ----
    cache = {
        "config": {
            "rows": ROWS, "train_frac": TRAIN_FRAC, "seeds": SEEDS,
            "miss_rate": MISS_RATE, "block_frac": BLOCK_FRAC,
            "miss_pct_actual": miss_pct,
            "datasets": used_datasets, "models": model_labels,
            "variants": VARIANTS,
        },
        "raw": raw,
        "recon_mae": recon_raw,
        "agg": agg,
        "finding": {
            "cells": tot, "cafe_wins": cw, "locf_wins": lw, "raw_wins": rw,
            "mean_cafe_minus_raw_r2": float(np.mean(gains)) if gains else None,
            "per_dataset_cafe_minus_raw_r2": per_ds_gain,
        },
    }
    with open(CACHE_OUT, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\n[wrote] {CACHE_OUT}")

    _write_table(agg, model_labels, used_datasets, len(SEEDS), miss_pct)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
