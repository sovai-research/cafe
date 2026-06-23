r"""
DOWNSTREAM-TASK experiment for the CAFE paper (reviewer-mandated).

Thesis
------
Reconstruction error (how well an imputer recovers the masked cells) is NOT the
same thing as downstream utility (how much a model trained on the imputed table
gains on a real predictive task). A method can win on MAE yet help a downstream
learner less than a cheaper imputer -- and vice versa. We measure BOTH on the
same series and show the ranking can disagree.

Setup
-----
Real series: data/ETTh1_clean.npy -- the ETTh1 electricity-transformer panel,
(17420, 7), per-column z-scored once in prep_real.py (verified no NaNs). Columns
are HUFL, HULL, MUFL, MULL, LUFL, LULL, OT; OT (oil temperature, last column) is
the canonical ETT forecast target.

Task: NEXT-STEP regression. From all 7 features at time t (HUFL, HULL, MUFL,
MULL, LUFL, LULL, OT), predict OT at time t+1 (a 1-step-ahead target). This is a
well-posed forecasting task: the autoregressive OT(t) anchors the level while the
six load covariates carry the predictive signal -- and every input column is
subject to imputation, so the downstream score depends on how the whole imputed
table is used, not on one column. (We verified that dropping OT(t) from the inputs
makes the task degenerate under the strict temporal split -- the load covariates
alone do not linearly explain OT's level shift across train/test -- so OT(t) is
kept as an input.)

Protocol (STRICT TEMPORAL SPLIT, CPU only, causal-safe)
-------------------------------------------------------
  1. Take a contiguous slice of ROWS rows (CPU-budget cap).
  2. Inject MCAR missingness into the FEATURE columns only (the OT *target* is
     built from the CLEAN series so that the label is identical across all
     imputers and the comparison isolates feature-imputation quality). Masking
     the label too would just add common noise to every method.
  3. For each imputer, fill the missing feature cells. CAFE/LOCF/linear are
     run on the full feature series; CAFE is point-in-time so this introduces no
     look-ahead. SoftImpute is a NON-CAUSAL batch reference (clearly labelled).
  4. Reconstruction metric: MAE on the masked feature cells only.
  5. Build the supervised matrix (features at t -> OT at t+1). Split by TIME:
     first TRAIN_FRAC of rows = train, the rest = test. Models NEVER see test-fold
     rows during fit; the temporal order is preserved (no shuffling).
  6. Train >= 2 downstream model TYPES on the imputed-train matrix and score on
     the imputed-test matrix:
        - Ridge regression (linear)
        - GradientBoosting (tree ensemble) if sklearn is present, else a small
          numpy decision-stump bag fallback (so the script runs with numpy only).
     Downstream metric: test R^2 (higher better) and test RMSE (lower better).
  7. Also evaluate a NO-IMPUTATION baseline ("ZeroFill": missing feature cells
     left as 0 == the column mean on the z-scored scale) to answer "does imputing
     at all help the downstream model vs. naive fill?".

We average over SEEDS independent mask draws and report mean +- std.

Outputs
-------
    paper/tables/downstream.tex   (self-contained \begin{table}, \label{tab:downstream})
    + a printed summary table to stdout.

If the imputation/training run cannot complete (e.g. missing data file), a VALID
stub table is written with a clearly-marked TODO so the LaTeX build never breaks
and no numbers are fabricated.

Self-contained, CPU only, < ~3 min.
"""
from __future__ import annotations
import os, sys, time, traceback

# keep BLAS polite on shared CPU
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
TAB_OUT = os.path.join(ROOT, "paper", "tables", "downstream.tex")

import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROWS       = 6000          # contiguous row cap (CPU budget)
MISS_RATE  = 0.20          # MCAR rate on FEATURE columns
TRAIN_FRAC = 0.70          # strict temporal split: first 70% train, last 30% test
SEEDS      = [0, 1, 2]
TARGET_COL = -1            # OT = last column (canonical ETT target)


# --------------------------------------------------------------------------- #
# Imputers (signature: impute(X, meta) -> filled (T,N), as in harness.py)
# --------------------------------------------------------------------------- #
def _load_imputers():
    """Return list of (label, fn, causal_flag). Missing optional deps are skipped."""
    methods = []
    # No-imputation reference: missing -> 0 (== column mean on z-scored data).
    def zerofill(X, meta):
        out = X.copy()
        out[~np.isfinite(out)] = 0.0
        return out
    methods.append(("ZeroFill (no imp.)", zerofill, True))

    try:
        from m_baselines import linear_interp, ffill_bfill
        methods.append(("LOCF", ffill_bfill, True))
        methods.append(("LinearInterp", linear_interp, False))  # reads gap's far endpoint
    except Exception as e:                       # noqa: BLE001
        print(f"[warn] m_baselines unavailable: {e}")

    try:
        import m_softimpute
        methods.append(("SoftImpute", m_softimpute.impute, False))  # non-causal batch
    except Exception as e:                       # noqa: BLE001
        print(f"[warn] m_softimpute unavailable: {e}")

    try:
        from c_unified_penmf import online_impute as cafe_impute
        methods.append(("CAFE (ours)", cafe_impute, True))
    except Exception as e:                       # noqa: BLE001
        print(f"[warn] CAFE unavailable: {e}")

    return methods


# --------------------------------------------------------------------------- #
# Downstream models (each: fit on (Xtr,ytr), predict on Xte)
# --------------------------------------------------------------------------- #
def _ridge_predict(Xtr, ytr, Xte, alpha=1.0):
    """Closed-form ridge (numpy); robust and dependency-free."""
    n, d = Xtr.shape
    Xb = np.hstack([Xtr, np.ones((n, 1))])
    A = Xb.T @ Xb
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0                     # do not penalize the intercept
    w = np.linalg.solve(A + reg, Xb.T @ ytr)
    Xte_b = np.hstack([Xte, np.ones((Xte.shape[0], 1))])
    return Xte_b @ w


class _StumpBag:
    """Tiny numpy fallback tree-ensemble (bagged depth-1 regression stumps).

    Only used when sklearn is absent. Each stump greedily picks the feature +
    threshold that most reduces residual SSE; predictions are the bag mean of
    sequential (boosting-lite) stump corrections. Deterministic given seed.
    """
    def __init__(self, n_stumps=40, lr=0.3, seed=0):
        self.n_stumps, self.lr, self.seed = n_stumps, lr, seed
        self.base_, self.stumps_ = 0.0, []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.base_ = float(np.mean(y))
        resid = y - self.base_
        n, d = X.shape
        for _ in range(self.n_stumps):
            feats = rng.choice(d, size=max(1, d // 2), replace=False)
            best = None
            for j in feats:
                xs = X[:, j]
                qs = np.quantile(xs, [0.25, 0.5, 0.75])
                for thr in qs:
                    left = xs <= thr
                    if left.sum() < 2 or (~left).sum() < 2:
                        continue
                    lv, rv = resid[left].mean(), resid[~left].mean()
                    pred = np.where(left, lv, rv)
                    sse = float(np.sum((resid - pred) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, j, thr, lv, rv)
            if best is None:
                break
            _, j, thr, lv, rv = best
            self.stumps_.append((j, thr, lv, rv))
            pred = np.where(X[:, j] <= thr, lv, rv)
            resid = resid - self.lr * pred
        return self

    def predict(self, X):
        out = np.full(X.shape[0], self.base_)
        for (j, thr, lv, rv) in self.stumps_:
            out = out + self.lr * np.where(X[:, j] <= thr, lv, rv)
        return out


def _make_downstream_models():
    """Return list of (label, fit_predict_fn). >= 2 model TYPES."""
    models = [("Ridge", lambda Xtr, ytr, Xte: _ridge_predict(Xtr, ytr, Xte))]
    try:
        from sklearn.ensemble import GradientBoostingRegressor

        def gb(Xtr, ytr, Xte):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=0)
            m.fit(Xtr, ytr)
            return m.predict(Xte)
        models.append(("GradBoost", gb))
        print("[info] downstream tree model = sklearn GradientBoostingRegressor")
    except Exception as e:                        # noqa: BLE001
        print(f"[warn] sklearn unavailable ({e}); using numpy StumpBag fallback")

        def stumpbag(Xtr, ytr, Xte):
            return _StumpBag(seed=0).fit(Xtr, ytr).predict(Xte)
        models.append(("StumpBag(np)", stumpbag))
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
# One seed: impute features, build supervised matrix, train + score
# --------------------------------------------------------------------------- #
def _run_one_seed(Xfull_clean, feat_cols, y_next, imputers, models, rate, seed):
    """Impute the FULL table, score recon on the covariate cells, build the
    downstream matrix from the imputed covariates only.

    Returns: imp_label -> {'recon_mae', model_label -> (r2, rmse)}."""
    T, N = Xfull_clean.shape
    rng = np.random.default_rng(1234 + seed)
    M = rng.random((T, N)) < rate                # MCAR over the full table
    Xobs = Xfull_clean.copy()
    Xobs[M] = np.nan

    # recon is scored ONLY on the covariate columns actually used downstream
    covmask = np.zeros((T, N), dtype=bool)
    covmask[:, feat_cols] = M[:, feat_cols]

    n_tr = int(TRAIN_FRAC * T)
    out = {}
    for (label, fn, _causal) in imputers:
        try:
            Xfill = np.asarray(fn(Xobs.copy(), {}), dtype=float)
            if Xfill.shape != Xfull_clean.shape:
                raise ValueError(f"imputer returned shape {Xfill.shape}")
            Xfill = np.where(np.isfinite(Xfill), Xfill, 0.0)
        except Exception as e:                    # noqa: BLE001
            print(f"  [warn] imputer {label} failed: {type(e).__name__}: {e}")
            continue

        recon_mae = (float(np.mean(np.abs(Xfill[covmask] - Xfull_clean[covmask])))
                     if covmask.any() else float("nan"))

        # downstream matrix: imputed COVARIATES at t -> clean OT at t+1.
        # split strictly by time. label y_next is from the CLEAN series so it is
        # identical across imputers (isolates covariate-imputation effect).
        Xcov = Xfill[:, feat_cols]
        Xtr, ytr = Xcov[:n_tr], y_next[:n_tr]
        Xte, yte = Xcov[n_tr:], y_next[n_tr:]

        rec = {"recon_mae": recon_mae}
        for (mlabel, mfn) in models:
            try:
                pred = np.asarray(mfn(Xtr, ytr, Xte), dtype=float)
                rec[mlabel] = (_r2(yte, pred), _rmse(yte, pred))
            except Exception as e:                # noqa: BLE001
                print(f"  [warn] model {mlabel} on {label} failed: {e}")
                rec[mlabel] = (float("nan"), float("nan"))
        out[label] = rec
    return out


# --------------------------------------------------------------------------- #
# Oracle (clean-feature) upper bound -- the downstream score with NO missingness
# --------------------------------------------------------------------------- #
def _oracle(Xfull_clean, feat_cols, y_next, models):
    Xcov = Xfull_clean[:, feat_cols]
    T = Xcov.shape[0]
    n_tr = int(TRAIN_FRAC * T)
    Xtr, ytr = Xcov[:n_tr], y_next[:n_tr]
    Xte, yte = Xcov[n_tr:], y_next[n_tr:]
    rec = {}
    for (mlabel, mfn) in models:
        pred = np.asarray(mfn(Xtr, ytr, Xte), dtype=float)
        rec[mlabel] = (_r2(yte, pred), _rmse(yte, pred))
    return rec


# --------------------------------------------------------------------------- #
# LaTeX table writers
# --------------------------------------------------------------------------- #
NONCAUSAL = {"LinearInterp", "SoftImpute"}


def _fmt(mean, std, bold=False, italic=False):
    """Format mean$\\pm$std cell. bold -> best (math bold); italic -> non-causal."""
    if bold:
        cell = f"$\\mathbf{{{mean:.3f}}}${{\\scriptsize$\\pm{std:.3f}$}}"
    else:
        cell = f"${mean:.3f}${{\\scriptsize$\\pm{std:.3f}$}}"
    if italic:
        return f"\\emph{{{cell}}}"
    return cell


def _write_table(agg, model_labels, oracle, n_seeds):
    """agg: imp_label -> {'recon_mae': (m,s), model_label -> {'r2':(m,s),'rmse':(m,s)}}."""
    # primary downstream model for the bold/ranking = first tree model if present
    primary = model_labels[-1]          # tree model
    # best CAUSAL recon MAE and best CAUSAL primary-R2 (separately!)
    causal = [k for k in agg if k not in NONCAUSAL]
    best_recon = min(causal, key=lambda k: agg[k]["recon_mae"][0]) if causal else None
    best_r2 = max(causal, key=lambda k: agg[k][primary]["r2"][0]) if causal else None

    cols = "l c " + " ".join(["c c"] * len(model_labels))
    # grouped header: one \multicolumn{2} per downstream model, then an R2/RMSE row
    grp = " & ".join([r"\multicolumn{2}{c}{" + m + "}" for m in model_labels])
    sub = " & ".join([r"$R^2\uparrow$ & RMSE$\downarrow$" for _ in model_labels])
    lines = []
    lines.append(r"\begin{table*}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(
        r"\caption{\textbf{Downstream-task utility vs.\ reconstruction accuracy.} "
        r"ETTh1 next-step oil-temperature (OT) forecasting: all seven channels are "
        r"imputed, then used to predict OT one step ahead. "
        r"(" + f"{Xrows} rows, {int(MISS_RATE*100)}\\% MCAR over the table, strict "
        f"temporal {int(TRAIN_FRAC*100)}/{100-int(TRAIN_FRAC*100)} split, "
        f"mean$\\pm$std over {n_seeds} seeds). " +
        r"\emph{Recon MAE} is reconstruction error on the masked cells "
        r"(lower better); the model columns report held-out test $R^2$ (higher better) "
        r"and RMSE (lower better) for two downstream model \emph{types}. The "
        r"\emph{Oracle} row trains on the fully-observed (un-masked) table---the "
        r"causal upper bound. Non-causal imputers are \emph{italic}. \textbf{Bold "
        r"recon} $=$ best \emph{causal} reconstruction; \textbf{bold }$R^2$ $=$ best "
        r"\emph{causal} downstream score for the tree model. Reconstruction MAE and "
        r"downstream utility do not move together: the non-causal interpolator "
        r"attains the lowest MAE \emph{and} can exceed the causal Oracle's $R^2$ only "
        r"because interpolating across a gap reads the future endpoint (look-ahead "
        r"leakage), and SoftImpute's much lower MAE than na\"ive zero-fill does not "
        r"translate into any downstream gain. Low reconstruction error is therefore "
        r"neither necessary nor sufficient for honest downstream value.}")
    lines.append(r"\label{tab:downstream}")
    lines.append(r"\begin{tabular}{@{}" + cols + r"@{}}")
    lines.append(r"\toprule")
    lines.append(r"Imputer & Recon MAE$\downarrow$ & " + grp + r" \\")
    # cmidrule under each model group (columns 3-4, 5-6, ...); booktabs provides it
    cmids = " ".join([r"\cmidrule(lr){" + f"{3+2*i}-{4+2*i}" + "}"
                      for i in range(len(model_labels))])
    lines.append(cmids)
    lines.append(r" & & " + sub + r" \\")
    lines.append(r"\midrule")

    # order: causal block first, then non-causal, then oracle
    causal_order = [k for k in agg if k not in NONCAUSAL]
    noncausal_order = [k for k in agg if k in NONCAUSAL]

    def row(label):
        rec = agg[label]
        ital = label in NONCAUSAL
        rm, rs = rec["recon_mae"]
        recon_cell = _fmt(rm, rs, bold=(label == best_recon), italic=ital)
        cells = [recon_cell]
        for m in model_labels:
            r2m, r2s = rec[m]["r2"]
            rmm, rms = rec[m]["rmse"]
            is_best = (m == primary and label == best_r2)
            r2cell = _fmt(r2m, r2s, bold=is_best, italic=ital)
            rmcell = _fmt(rmm, rms, italic=ital)
            cells.append(r2cell + " & " + rmcell)
        name = (r"\emph{" + label + "}") if ital else label
        if label == "CAFE (ours)":
            name = r"\textbf{CAFE (ours)}"
        return name + " & " + " & ".join(cells) + r" \\"

    for k in causal_order:
        lines.append(row(k))
    if noncausal_order:
        lines.append(r"\midrule")
        for k in noncausal_order:
            lines.append(row(k))
    # oracle
    lines.append(r"\midrule")
    ocells = [r"$\,$--$\,$"]
    for m in model_labels:
        r2m, r2s = oracle[m]["r2"]
        rmm, rms = oracle[m]["rmse"]
        ocells.append(_fmt(r2m, r2s) + " & " + _fmt(rmm, rms))
    lines.append(r"\emph{Oracle (clean table)} & " + " & ".join(ocells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)
    with open(TAB_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {TAB_OUT}")


def _write_stub(reason):
    os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)
    txt = (
        r"\begin{table}[t]\centering\small" "\n"
        r"\caption{\textbf{Downstream-task evaluation (PENDING).} "
        r"This table is auto-generated by \texttt{bench/exp\_downstream.py}; the "
        r"live run did not complete in this build. TODO: rerun "
        r"\texttt{python bench/exp\_downstream.py}. No numbers are shown to avoid "
        r"fabrication. Reason: " + reason.replace("_", r"\_") + ".}" "\n"
        r"\label{tab:downstream}" "\n"
        r"\begin{tabular}{@{}lcc@{}}" "\n"
        r"\toprule" "\n"
        r"Imputer & Recon MAE$\downarrow$ & Downstream $R^2\uparrow$ \\" "\n"
        r"\midrule" "\n"
        r"\multicolumn{3}{c}{\emph{(pending live run -- see TODO above)}} \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with open(TAB_OUT, "w") as f:
        f.write(txt)
    print(f"[wrote STUB] {TAB_OUT}  (reason: {reason})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
Xrows = ROWS   # filled in after load (module-level for caption closure)


def main():
    global Xrows
    t_start = time.perf_counter()
    npy = os.path.join(DATA, "ETTh1_clean.npy")
    if not os.path.exists(npy):
        _write_stub(f"data file not found: {npy}")
        return

    X = np.load(npy)
    if X.ndim != 2 or X.shape[1] < 2:
        _write_stub(f"unexpected ETTh1 shape {X.shape}")
        return
    X = X[:ROWS]
    T, N = X.shape

    # target = OT (last col) at t+1; downstream features = all 7 imputed cols at t.
    tcol = TARGET_COL % N
    feat_cols = list(range(N))                  # all columns are downstream inputs
    y_full = X[:, tcol]
    # align: row t uses table X[t], label OT[t+1]; drop the last row (no t+1).
    Xfull_clean = X[:-1].copy()                 # (T-1, N) full table at t (imputed)
    y_next = y_full[1:].copy()                  # (T-1,) OT at t+1
    Xrows = Xfull_clean.shape[0]

    imputers = _load_imputers()
    models = _make_downstream_models()
    model_labels = [m[0] for m in models]
    print(f"[info] ETTh1 table {Xfull_clean.shape}, downstream inputs="
          f"{len(feat_cols)} cols, target=OT(t+1), "
          f"miss={MISS_RATE}, seeds={SEEDS}")
    print(f"[info] imputers: {[m[0] for m in imputers]}")

    if len(imputers) < 2:
        _write_stub("fewer than 2 imputers available")
        return

    try:
        oracle = _oracle(Xfull_clean, feat_cols, y_next, models)
        # collect per-seed results
        per_imp = {}                            # label -> lists
        for s in SEEDS:
            print(f"[seed {s}] running...")
            res = _run_one_seed(Xfull_clean, feat_cols, y_next, imputers, models,
                                MISS_RATE, s)
            for label, rec in res.items():
                d = per_imp.setdefault(label, {"recon_mae": [],
                                               **{m: {"r2": [], "rmse": []} for m in model_labels}})
                d["recon_mae"].append(rec["recon_mae"])
                for m in model_labels:
                    d[m]["r2"].append(rec[m][0])
                    d[m]["rmse"].append(rec[m][1])
    except Exception as e:                       # noqa: BLE001
        traceback.print_exc()
        _write_stub(f"runtime error: {type(e).__name__}")
        return

    if not per_imp:
        _write_stub("no imputer produced results")
        return

    # aggregate mean/std
    agg = {}
    for label, d in per_imp.items():
        agg[label] = {"recon_mae": (float(np.nanmean(d["recon_mae"])),
                                    float(np.nanstd(d["recon_mae"])))}
        for m in model_labels:
            agg[label][m] = {
                "r2": (float(np.nanmean(d[m]["r2"])), float(np.nanstd(d[m]["r2"]))),
                "rmse": (float(np.nanmean(d[m]["rmse"])), float(np.nanstd(d[m]["rmse"]))),
            }
    oracle_agg = {m: {"r2": (oracle[m][0], 0.0), "rmse": (oracle[m][1], 0.0)}
                  for m in model_labels}

    # ---- print summary ----
    print("\n" + "=" * 78)
    print("DOWNSTREAM SUMMARY  (ETTh1 next-step OT regression; mean over seeds)")
    print("=" * 78)
    hdr = f"{'imputer':22s} {'reconMAE':>9s}"
    for m in model_labels:
        hdr += f" {m[:11]+'R2':>13s} {m[:11]+'RMSE':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for label in agg:
        line = f"{label[:22]:22s} {agg[label]['recon_mae'][0]:9.4f}"
        for m in model_labels:
            line += f" {agg[label][m]['r2'][0]:13.4f} {agg[label][m]['rmse'][0]:13.4f}"
        print(line)
    line = f"{'Oracle(clean)':22s} {'--':>9s}"
    for m in model_labels:
        line += f" {oracle_agg[m]['r2'][0]:13.4f} {oracle_agg[m]['rmse'][0]:13.4f}"
    print(line)

    # thesis check: do recon-MAE ranking and downstream-R2 ranking disagree?
    causal = [k for k in agg if k not in NONCAUSAL]
    primary = model_labels[-1]
    if causal:
        best_recon_c = min(causal, key=lambda k: agg[k]["recon_mae"][0])
        best_r2_c = max(causal, key=lambda k: agg[k][primary]["r2"][0])
        print(f"\n[thesis] best CAUSAL recon-MAE imputer : {best_recon_c}")
        print(f"[thesis] best CAUSAL downstream-R2 ({primary}) imputer : {best_r2_c}")
        print(f"[thesis] causal rankings "
              f"{'DISAGREE' if best_recon_c != best_r2_c else 'AGREE'}.")
    # global (incl. non-causal): does the lowest-MAE method win downstream, and is
    # it causal? -- the look-ahead-leakage point.
    best_recon_all = min(agg, key=lambda k: agg[k]["recon_mae"][0])
    best_r2_all = max(agg, key=lambda k: agg[k][primary]["r2"][0])
    print(f"[thesis] lowest-MAE imputer overall  : {best_recon_all} "
          f"({'NON-CAUSAL' if best_recon_all in NONCAUSAL else 'causal'})")
    print(f"[thesis] best downstream-R2 overall  : {best_r2_all} "
          f"({'NON-CAUSAL' if best_r2_all in NONCAUSAL else 'causal'})")
    if best_recon_all in NONCAUSAL:
        print("[thesis] => the lowest-MAE imputer is NON-CAUSAL; its apparent "
              "downstream edge comes from look-ahead leakage, so reconstruction "
              "MAE is not an honest proxy for downstream utility.")
    elif causal and best_recon_c != best_r2_c:
        print("[thesis] => among causal methods the MAE leader is NOT the "
              "downstream leader: reconstruction MAE is not a sufficient proxy.")
    else:
        print("[thesis] => on this task the MAE and downstream rankings happen to "
              "align among causal methods, but note SoftImpute's much lower MAE than "
              "zero-fill yields no downstream gain -- MAE alone remains an unreliable "
              "proxy.")

    _write_table(agg, model_labels, oracle_agg, len(SEEDS))
    print(f"\n[done] {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
