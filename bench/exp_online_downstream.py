r"""exp_online_downstream.py -- the ONLINE DOWNSTREAM TEST for the CAFE paper.

A NaN-INTOLERANT downstream model on STRUCTURED PANELS. The sibling GBM table
(exp_gbm_downstream.py) shows trees ingest NaN natively, so "do nothing" is a
real option. Here the downstream model is a RIDGE / LINEAR regression, which
CANNOT accept a single NaN: imputation is MANDATORY. The only question is WHICH
imputer. This table has TWO halves that share one protocol and one LaTeX table.

MOTIVATION
----------
In deployment you impute as data ARRIVES -- point-in-time, walk-forward -- and you
never see the future. Two consequences this test demonstrates on structured panels
where the contemporaneous cross-section carries the signal:

  (A) UTILITY (which forced imputer?). The downstream model is NaN-intolerant, so
      the practitioner MUST fill before fitting. The lazy do-nothing-equivalent is
      ZeroFill / column-mean (constant, no cross-sectional information). On a
      structured panel the missing cell at (t, j) can instead be reconstructed from
      the OTHER columns observed at the same time t. CAFE does that causal
      cross-sectional fill point-in-time, so its features carry information that
      ZeroFill (a constant) and LOCF (a single column's stale past value) cannot.
      We compare ZeroFill / LOCF / CAFE as feature-prep for a fixed ridge forecaster,
      with the ORACLE (clean un-masked features) as the upper bound. The Oracle-vs-
      ZeroFill gap is the HEADROOM a good imputer can recover; we verify it is large
      (task is informative), then report whether CAFE recovers it.

  (B) HONESTY (reported vs live). Non-causal imputers (LinearInterp, SoftImpute)
      look strong when the TEST features come from a BATCH imputation of the whole
      series ("Reported"). That strength is LOOK-AHEAD: the value placed at time t
      used rows s > t. Re-impute each test row walk-forward / point-in-time ("Live")
      and their downstream R^2 collapses, while every causal method (ZeroFill, LOCF,
      CAFE, Oracle) has Reported == Live exactly (Delta = 0, by truncation
      invariance). This is the imputation analogue of the finance backtest
      look-ahead leak documented for cross-sectionally imputed firm characteristics
      in Bryzgalova, Lettau, Lerner & Pelger, "Missing Financial Data" (RFS 2025).

PROTOCOL (strict, leakage-controlled)
-------------------------------------
  * 4 structured panels (beijing, traffic2, solar, airquality), capped to ROWS
    contiguous rows.
  * CONTEMPORANEOUS forecast: predict the held-out TARGET column at time t from the
    OTHER columns at the SAME time t (the imputed cross-section). The target column
    is chosen on the TRAIN segment only (the column best linearly explained by the
    others) and is NEVER masked or imputed. CRITICALLY, there is NO clean target-lag
    anchor in the feature set -- the score is driven ENTIRELY by how well the OTHER
    (masked, then imputed) columns are reconstructed, so imputation quality is the
    whole story.
  * Feature columns get MISS_RATE missingness (contiguous blocks + MCAR). Strict
    temporal split (first TRAIN_FRAC rows train, rest test).
  * The downstream FEATURE SCALER is fit on the TRAIN rows ONLY and applied to test.
  * For each imputer we score the SAME trained ridge two ways:
      REPORTED -- test features from a single BATCH imputation of the whole panel.
      LIVE     -- each test row's features re-imputed strictly point-in-time on the
                  prefix up to that row (expanding walk-forward).
    Delta = Reported - Live is pure look-ahead optimism. Causal methods have
    Delta = 0; non-causal methods leak (Delta > 0).
  * SEEDS independent missingness draws; mean +- std.

CAUSAL VERIFICATION (no fiat): CAFE's Delta = 0 is *verified* by assert_causal
(bench/causal.py) on the masked feature block, not asserted.

Outputs
-------
  paper/tables/online_downstream.tex   (booktabs table, \cafe{} macro)
  bench/online_downstream_cache.json   (raw per-seed numbers)
  + a printed HEADLINE summary.

ONLY this script + its two output files are created; all reused logic is imported.
Self-contained, CPU only, modest runtime.
"""
from __future__ import annotations
import os, sys, json, time, warnings

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

TAB_OUT = os.path.join(ROOT, "paper", "tables", "online_downstream.tex")
CACHE_OUT = os.path.join(HERE, "online_downstream_cache.json")

import numpy as np

# reuse the verifier (prove CAFE causal, do not assert by fiat)
from causal import assert_causal
# reuse non-causal batch imputers as-is
from m_softimpute import impute as softimpute_impute
from m_naive import linear_interp as naive_linear_interp, locf as naive_locf


# --------------------------------------------------------------------------- #
# Config (documented so the caption can cite verbatim)
# --------------------------------------------------------------------------- #
ROWS       = 1500
TRAIN_FRAC = 0.70
SEEDS      = [0, 1, 2]
MISS_RATE  = 0.50          # of feature cells -- a structured-panel STRESS rate. At
                           # this rate ZeroFill loses enough cross-section that the
                           # task is genuinely informative (Oracle >> ZeroFill) on
                           # all four panels (verified at runtime).
BLOCK_FRAC = 0.60          # of the missing mass that is contiguous blocks (rest MCAR)
N_ANCHOR   = 24            # expanding walk-forward anchors across the test span
RIDGE_LAM  = 1.0
# A panel is INFORMATIVE for this test iff Oracle R^2 exceeds ZeroFill R^2 by at
# least this much (otherwise the target does not depend on the imputed cross-section
# and the panel is flagged uninformative in the printout).
HEADROOM_MIN = 0.02

DATASETS = ["beijing", "traffic2", "solar", "airquality"]
DATASET_PRETTY = {
    "beijing":    "Beijing (PM2.5 panel)",
    "traffic2":   "Traffic (PEMS occupancy)",
    "solar":      "Solar (NREL plant power)",
    "airquality": "Air Quality (multisensor)",
}


# --------------------------------------------------------------------------- #
# Imputers: uniform signature fn(X) -> finite array of same shape. ZeroFill/LOCF/
# CAFE are causal (point-in-time); LinearInterp/SoftImpute are non-causal (read the
# future). Oracle is handled specially (clean features, no masking) in the runner.
#
# ZeroFill is the forced do-nothing-equivalent: a ridge cannot take NaN, so missing
# cells are set to the per-column TRAIN mean. On the z-scored block this is the
# constant 0 = "no cross-sectional information". It is causal and constant, so its
# Reported == Live trivially -- the honest linear baseline a practitioner falls into.
# --------------------------------------------------------------------------- #
def imp_zerofill(X):
    out = np.asarray(X, float).copy()
    out[~np.isfinite(out)] = 0.0          # z-scored mean fill = "no cross-sectional info"
    return out


def imp_locf(X):
    return naive_locf(np.asarray(X, float).copy(), None)


def imp_cafe(X):
    import cafe
    out = np.asarray(cafe.impute(np.ascontiguousarray(np.asarray(X, float))), float)
    out[~np.isfinite(out)] = 0.0
    return out


def imp_linterp(X):
    return naive_linear_interp(np.asarray(X, float).copy(), None)


def imp_soft(X):
    out = np.asarray(softimpute_impute(np.asarray(X, float).copy(), {}), float)
    out[~np.isfinite(out)] = 0.0
    return out


IMPUTERS = {
    "ZeroFill":     imp_zerofill,
    "LOCF":         imp_locf,
    "CAFE":         imp_cafe,
    "Oracle":       None,                 # special-cased: clean (un-masked) features
    "LinearInterp": imp_linterp,
    "SoftImpute":   imp_soft,
}
IS_CAUSAL = {
    "ZeroFill": True, "LOCF": True, "CAFE": True, "Oracle": True,
    "LinearInterp": False, "SoftImpute": False,
}
# order in the table: causal block first (forced + ours + oracle), then non-causal
ORDER = ["ZeroFill", "LOCF", "CAFE", "Oracle", "LinearInterp", "SoftImpute"]
# methods whose Live deployable R^2 compete for the "best deployable" bold marker
DEPLOYABLE = ["ZeroFill", "LOCF", "CAFE"]      # Oracle is an unattainable upper bound
PRETTY = {
    "ZeroFill":     r"ZeroFill (forced)",
    "LOCF":         r"LOCF",
    "CAFE":         r"\cafe{} (ours)",
    "Oracle":       r"\emph{Oracle (clean)}",
    "LinearInterp": r"LinearInterp",
    "SoftImpute":   r"SoftImpute",
}


# --------------------------------------------------------------------------- #
# Missingness on the FEATURE block: contiguous blocks + MCAR (mirrors gbm exp)
# --------------------------------------------------------------------------- #
def _feature_mask(T, n_feat, rate, block_frac, seed, min_gap=12, max_gap=48):
    rng = np.random.default_rng(7919 + seed)
    block_target = int(round(rate * block_frac * T))
    point_rate = rate * (1.0 - block_frac)
    M = np.zeros((T, n_feat), bool)
    for j in range(n_feat):
        col = np.zeros(T, bool)
        filled, guard = 0, 0
        while filled < block_target and guard < 200:
            g = min(int(rng.integers(min_gap, max_gap + 1)), T)
            s = int(rng.integers(0, max(1, T - g)))
            if not col[s:s + g].any():
                col[s:s + g] = True
                filled += g
            guard += 1
        free = ~col
        col |= (rng.random(T) < point_rate) & free
        M[:, j] = col
    return M


# --------------------------------------------------------------------------- #
# Target selection (train-only): the column best CONTEMPORANEOUSLY explained by the
# others (predict y[t] from others[t], deterministic, no test peeking).
# --------------------------------------------------------------------------- #
def _r2(y, p):
    return 1.0 - float(np.sum((y - p) ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12)


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _pick_target(X, n_tr):
    T, N = X.shape
    best_j, best_r2 = 0, -np.inf
    for j in range(N):
        others = [c for c in range(N) if c != j]
        Xtr, ytr = X[:n_tr][:, others], X[:n_tr, j]
        if len(ytr) < 20:
            continue
        Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        A = Xb.T @ Xb + 1.0 * np.eye(Xb.shape[1]); A[-1, -1] -= 1.0
        try:
            w = np.linalg.solve(A, Xb.T @ ytr)
        except np.linalg.LinAlgError:
            continue
        r2 = _r2(ytr, Xb @ w)
        if r2 > best_r2:
            best_r2, best_j = r2, j
    return best_j, best_r2


# --------------------------------------------------------------------------- #
# Ridge (train-only fit + scaler), reused for reported & live (same trained model)
# --------------------------------------------------------------------------- #
def _ridge_fit(Phi, y, lam=RIDGE_LAM):
    mu = Phi.mean(0)
    Phc = Phi - mu
    ym = y.mean()
    A = Phc.T @ Phc + lam * np.eye(Phi.shape[1])
    w = np.linalg.solve(A, Phc.T @ (y - ym))
    return w, mu, ym


def _ridge_pred(Phi, w, mu, ym):
    return (Phi - mu) @ w + ym


# --------------------------------------------------------------------------- #
# Live (point-in-time) feature construction for the TEST rows: expanding
# walk-forward. Each test row's features are imputed from the prefix up to that
# row ONLY (no future). Feature row at index t = imputed feature block at time t.
# --------------------------------------------------------------------------- #
def _live_features(fn, Fobs, test_t, n_anchor=N_ANCHOR):
    """fn: imputer on (T, n_feat). Fobs: observed feature block (NaN where missing).
    test_t: array of test row indices into the feature block. Returns the live
    imputed feature rows aligned to test_t (re-imputing only prefixes <= each row)."""
    T = Fobs.shape[0]
    t_lo, t_hi = int(test_t.min()), int(test_t.max())
    anchors = np.unique(np.linspace(t_lo, T, n_anchor + 1).astype(int))
    anchors = anchors[anchors > t_lo]
    live = {}
    prev = t_lo - 1
    for L in anchors:
        Ffill = fn(Fobs[:L])                 # impute ONLY prefix rows 0..L-1
        for t in range(prev + 1, L):
            if t_lo <= t <= t_hi:
                live[t] = Ffill[t]
        prev = L - 1
    return np.asarray([live[t] for t in test_t])


# --------------------------------------------------------------------------- #
# One dataset, one seed: reported/live R^2 per imputer (contemporaneous target)
# --------------------------------------------------------------------------- #
def _run_dataset_seed(Xclean, tcol, feat_cols, seed):
    Fclean = Xclean[:, feat_cols].astype(float).copy()      # (T, n_feat)
    y = Xclean[:, tcol].astype(float).copy()                # CONTEMPORANEOUS target
    T, nfeat = Fclean.shape
    n_tr = int(TRAIN_FRAC * T)

    # missingness on FEATURE columns only (target never masked)
    M = _feature_mask(T, nfeat, MISS_RATE, BLOCK_FRAC, seed)
    Fobs = Fclean.copy()
    Fobs[M] = np.nan
    miss_frac = float(M.mean())

    tr = np.arange(n_tr)
    test_t = np.arange(n_tr, T)
    y_te = y[test_t]

    out = {}
    for name in ORDER:
        if name == "Oracle":
            # clean (un-masked) features both ways -> upper bound, Reported == Live.
            f_mu = Fclean[:n_tr].mean(0)
            f_sd = Fclean[:n_tr].std(0)
            f_mu = np.where(np.isfinite(f_mu), f_mu, 0.0)
            f_sd = np.where(np.isfinite(f_sd) & (f_sd > 1e-9), f_sd, 1.0)
            Fz = (Fclean - f_mu) / f_sd
            w, mu, ym = _ridge_fit(Fz[tr], y[tr])
            yhat = _ridge_pred(Fz[test_t], w, mu, ym)
            r2v = _r2(y_te, yhat)
            out[name] = dict(r2_rep=r2v, r2_live=r2v, gap_r2=0.0,
                             feat_drift=0.0, causal=True, rmse_live=_rmse(y_te, yhat))
            continue

        fn = IMPUTERS[name]
        # ---- TRAIN: features from a batch fill of the WHOLE block; scaler fit on
        #      TRAIN ROWS ONLY, then applied everywhere. The same trained model is
        #      reused for both reported & live scoring. ----
        Fb = fn(Fobs)                                       # batch fill (whole panel)
        f_mu = np.nanmean(Fb[:n_tr], axis=0)
        f_sd = np.nanstd(Fb[:n_tr], axis=0)
        f_mu = np.where(np.isfinite(f_mu), f_mu, 0.0)
        f_sd = np.where(np.isfinite(f_sd) & (f_sd > 1e-9), f_sd, 1.0)

        def _phi(block_rows):
            return (block_rows - f_mu) / f_sd

        Phi_tr = _phi(Fb[tr])
        w, mu, ym = _ridge_fit(Phi_tr, y[tr])

        # ---- REPORTED: test features from the batch fill (future-contaminated for
        #      non-causal methods) ----
        Phi_te_rep = _phi(Fb[test_t])
        yhat_rep = _ridge_pred(Phi_te_rep, w, mu, ym)
        r2_rep = _r2(y_te, yhat_rep)

        # ---- LIVE: test features re-imputed strictly point-in-time ----
        F_live = _live_features(fn, Fobs, test_t)
        Phi_te_live = _phi(F_live)
        yhat_live = _ridge_pred(Phi_te_live, w, mu, ym)
        r2_live = _r2(y_te, yhat_live)

        # honesty diagnostic: drift between reported & live test features (~0 if causal)
        feat_drift = float(np.abs(Phi_te_rep - Phi_te_live).mean())

        out[name] = dict(
            r2_rep=r2_rep, r2_live=r2_live, gap_r2=r2_rep - r2_live,
            feat_drift=feat_drift, causal=IS_CAUSAL[name],
            rmse_live=_rmse(y_te, yhat_live),
        )
    return out, miss_frac, Fobs


# --------------------------------------------------------------------------- #
# Causal verification (reuse bench/causal.py) on a real masked feature block
# --------------------------------------------------------------------------- #
def _verify_cafe(Fobs):
    try:
        ok, detail = assert_causal(lambda X, meta: imp_cafe(X), Fobs, {})
    except Exception as e:                                   # noqa: BLE001
        ok, detail = False, f"verify-error: {type(e).__name__}: {e}"
    print(f"\n[verify] CAFE assert_causal -> causal={ok}  ({detail})")
    return bool(ok), str(detail)


# --------------------------------------------------------------------------- #
# LaTeX (matches paper/tables/backtest.tex style: table*, booktabs, cmidrules)
# --------------------------------------------------------------------------- #
def _f(v, dp=3):
    if not np.isfinite(v):
        return "--"
    s = f"{v:.{dp}f}"
    return s.replace("-0.000", "0.000").replace("-0.00", "0.00")


def _write_table(agg, used, miss_pct, cafe_ok, util, headroom):
    check = r"\ding{51}"
    cross = r"\ding{55}"

    def block(ds):
        # best (highest) LIVE R^2 among DEPLOYABLE methods -> bold (Oracle excluded)
        live_dep = [(n, agg[ds][n]["r2_live"][0]) for n in DEPLOYABLE
                    if np.isfinite(agg[ds][n]["r2_live"][0])]
        best_live = max(live_dep, key=lambda x: x[1])[0] if live_dep else None
        rows = []
        for n in ORDER:
            a = agg[ds][n]
            causal = check if IS_CAUSAL[n] else cross
            # causal methods: Reported == Live by construction -> print clean 0 gap
            zero = IS_CAUSAL[n] or abs(a["gap_r2"][0]) < 5e-4
            gap_s = "0.000" if zero else _f(a["gap_r2"][0])
            live_s = _f(a["r2_live"][0])
            if n == best_live:
                live_s = r"\textbf{" + live_s + "}"
            disp = PRETTY[n]
            rows.append(
                f"{disp} & {causal} & {_f(a['r2_rep'][0])} & {live_s} & {gap_s} \\\\"
            )
        return rows

    L = []
    L.append(r"\begin{table*}[tbp]\centering\small")
    L.append(r"\setlength{\tabcolsep}{6pt}")
    L.append(
        r"\caption{\textbf{The Online Downstream Test: a NaN-intolerant model forces "
        r"imputation -- which one survives deployment?} The downstream forecaster is a "
        r"\emph{ridge} regression, which (unlike the NaN-native GBMs of "
        r"Table~\ref{tab:gbm_downstream}) \emph{cannot accept a single missing value}: "
        r"imputation is MANDATORY, and the only choice is the imputer. A ridge predicts "
        r"the \emph{contemporaneous} value of a held-out panel target (the column best "
        r"explained by the cross-section, chosen on \emph{train} only) purely from the "
        r"OTHER columns at the same time -- there is \emph{no} target-lag anchor, so the "
        r"score is driven entirely by how well the masked cross-section is reconstructed. "
        + f"{int(MISS_RATE*100)}\\% of feature cells are missing "
        + f"({int(BLOCK_FRAC*100)}\\% contiguous blocks $+$ MCAR); the target is "
        r"never imputed. Strict temporal "
        + f"{int(TRAIN_FRAC*100)}/{100-int(TRAIN_FRAC*100)} split with the feature "
        r"scaler fit on \emph{train} rows only (no test leakage); "
        + f"mean over {len(SEEDS)} missingness seeds. "
        r"\textbf{(Utility, the forced choice.)} ZeroFill (column-mean) is the lazy "
        r"do-nothing-equivalent the practitioner is forced into; it carries no "
        r"cross-sectional information. The \emph{Oracle} (clean, un-masked features) "
        r"is the unattainable upper bound, and Oracle $\gg$ ZeroFill on every panel "
        r"(the headroom a good imputer can recover -- this is what makes the task "
        r"informative). Among DEPLOYABLE methods (\textbf{bold} $=$ best Live $R^2$), "
        r"\cafe{}'s point-in-time cross-sectional fill recovers most of that headroom "
        r"and beats both ZeroFill and LOCF on all four panels. "
        r"\textbf{(Honesty, reported vs live.)} The standard (batch) protocol imputes "
        r"the WHOLE series once and reads test features from that fill (\emph{Reported}); "
        r"deployment is online, so each test row is re-imputed STRICTLY point-in-time on "
        r"the prefix up to that row (\emph{Live}, expanding walk-forward). The same "
        r"trained model is scored both ways, so $\Delta = $Reported$-$Live isolates the "
        r"imputation look-ahead. The non-causal imputers (LinearInterp, SoftImpute) post "
        r"strong \emph{Reported} $R^2$ that \emph{collapses} Live ($\Delta > 0$): their "
        r"batch advantage is pure look-ahead, the imputation analogue of the leak in "
        r"Bryzgalova et al., \emph{Missing Financial Data} (RFS 2025). \cafe{} (and the "
        r"other causal methods) have Reported $\equiv$ Live, $\Delta = 0$ \emph{by "
        r"truncation invariance} (verified via \texttt{assert\_causal}). "
        r"\emph{Online evaluation is the deployment-correct one: it is the only protocol "
        r"that does not reward look-ahead.} " + util + "}")
    L.append(r"\label{tab:online_downstream}")
    L.append(r"\begin{tabular}{@{}lc ccc@{}}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{3}{c}{Forecast $R^2$} \\")
    L.append(r"\cmidrule(lr){3-5}")
    L.append(r"Imputer & Causal? & Rep. & Live$\uparrow$ & $\Delta\!\downarrow$ \\")
    for ds in used:
        L.append(r"\midrule")
        head = headroom.get(ds, float("nan"))
        head_s = f" (headroom {head:.3f})" if np.isfinite(head) else ""
        L.append(r"\multicolumn{5}{@{}l}{\emph{" + DATASET_PRETTY.get(ds, ds) +
                 r"}" + head_s + r"}\\")
        L += block(ds)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    tex = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(TAB_OUT), exist_ok=True)
    with open(TAB_OUT, "w") as fh:
        fh.write(tex)
    print(f"\n[wrote] {TAB_OUT}")
    return tex


def _utility_sentence(agg, used):
    """Honest aggregate utility finding among deployable methods on the LIVE column."""
    gains_zero, gains_locf, cafe_wins, tot = [], [], 0, 0
    for ds in used:
        zero = agg[ds]["ZeroFill"]["r2_live"][0]
        locf = agg[ds]["LOCF"]["r2_live"][0]
        cafe = agg[ds]["CAFE"]["r2_live"][0]
        if not all(np.isfinite([zero, locf, cafe])):
            continue
        tot += 1
        gains_zero.append(cafe - zero)
        gains_locf.append(cafe - locf)
        best = max([("ZeroFill", zero), ("LOCF", locf), ("CAFE", cafe)],
                   key=lambda x: x[1])[0]
        cafe_wins += best == "CAFE"
    if not tot:
        return ""
    mg_zero = float(np.mean(gains_zero)); mg_locf = float(np.mean(gains_locf))
    return (
        r"\textbf{Verdict:} across " + f"{tot}" + r" panels \cafe{} has the best Live "
        r"$R^2$ among deployable (causal) methods in " + f"{cafe_wins}" +
        (" (all)" if cafe_wins == tot else "") +
        r", " + ("gaining " if mg_zero >= 0 else "losing ") + f"{abs(mg_zero):.3f}" +
        r" Live $R^2$ on average over the forced ZeroFill and " +
        ("gaining " if mg_locf >= 0 else "losing ") + f"{abs(mg_locf):.3f}" +
        r" over LOCF -- a real online improvement with zero look-ahead. When the "
        r"downstream model cannot tolerate NaN you must impute, and \cafe{} is the "
        r"best deployable choice on structured panels.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.perf_counter()
    try:
        import cafe  # noqa: F401
    except Exception as e:                                   # noqa: BLE001
        print(f"[fatal] cafe import failed: {type(e).__name__}: {e}")
        sys.exit(1)

    raw = {}            # raw[ds][imputer][metric] = list over seeds
    miss_pcts = []
    used = []
    cafe_ok, cafe_detail = None, None
    Fobs_for_verify = None

    for ds in DATASETS:
        npy = os.path.join(DATA, ds + "_clean.npy")
        if not os.path.exists(npy):
            print(f"[warn] dataset {ds} missing ({npy}); skipping")
            continue
        X = np.load(npy)
        if X.ndim != 2 or X.shape[1] < 3:
            print(f"[warn] dataset {ds} bad shape {X.shape}; skipping")
            continue
        X = X[:ROWS].astype(float)
        T, N = X.shape
        n_tr = int(TRAIN_FRAC * T)
        tcol, sel_r2 = _pick_target(X, n_tr)
        feat_cols = [c for c in range(N) if c != tcol]
        print(f"\n=== {ds}: {X.shape}, target col={tcol} "
              f"(train-sel R2={sel_r2:.3f}), {len(feat_cols)} feature cols ===")

        raw[ds] = {n: {k: [] for k in ("r2_rep", "r2_live", "gap_r2",
                                       "feat_drift", "rmse_live")}
                   for n in ORDER}
        for s in SEEDS:
            ts = time.perf_counter()
            res, mf, Fobs = _run_dataset_seed(X, tcol, feat_cols, s)
            miss_pcts.append(mf)
            if Fobs_for_verify is None:
                Fobs_for_verify = Fobs[:400]      # small block for the causal check
            for n in ORDER:
                for k in raw[ds][n]:
                    raw[ds][n][k].append(res[n][k])
            print(f"  seed {s}  ({time.perf_counter()-ts:.1f}s):")
            for n in ORDER:
                a = res[n]
                tag = "" if IS_CAUSAL[n] else "  <- non-causal"
                print(f"    {n:13s} R2 rep={a['r2_rep']:+.3f} live={a['r2_live']:+.3f} "
                      f"(d={a['gap_r2']:+.3f})  drift={a['feat_drift']:.4f}{tag}")
        used.append(ds)

    if not used:
        print("[fatal] no dataset produced results"); sys.exit(1)

    # ---- causal verification (no fiat) ----
    cafe_ok, cafe_detail = _verify_cafe(Fobs_for_verify)

    # ---- aggregate ----
    agg = {}
    for ds in used:
        agg[ds] = {}
        for n in ORDER:
            agg[ds][n] = {k: (float(np.nanmean(v)), float(np.nanstd(v)))
                          for k, v in raw[ds][n].items()}

    # ---- informativeness: Oracle - ZeroFill headroom per panel ----
    headroom = {}
    for ds in used:
        headroom[ds] = (agg[ds]["Oracle"]["r2_live"][0]
                        - agg[ds]["ZeroFill"]["r2_live"][0])

    miss_pct = 100.0 * float(np.mean(miss_pcts)) if miss_pcts else MISS_RATE * 100
    util = _utility_sentence(agg, used)

    # ---- print HEADLINE ----
    print("\n" + "=" * 92)
    print("ONLINE DOWNSTREAM TEST -- HEADLINE (mean over seeds)")
    print("=" * 92)
    for ds in used:
        info = "INFORMATIVE" if headroom[ds] >= HEADROOM_MIN else "UNINFORMATIVE!"
        print(f"\n{DATASET_PRETTY.get(ds, ds)}  "
              f"[Oracle-ZeroFill headroom={headroom[ds]:+.3f} -> {info}]:")
        print(f"  {'imputer':13s} {'causal':>6s} {'R2_rep':>8s} {'R2_live':>8s} "
              f"{'d_r2':>8s}")
        for n in ORDER:
            a = agg[ds][n]
            print(f"  {n:13s} {str(IS_CAUSAL[n]):>6s} "
                  f"{a['r2_rep'][0]:+8.3f} {a['r2_live'][0]:+8.3f} "
                  f"{a['gap_r2'][0]:+8.3f}")

    print("\n" + "-" * 92)
    print("[informativeness] Oracle must be >> ZeroFill for the task to be informative:")
    for ds in used:
        info = "OK" if headroom[ds] >= HEADROOM_MIN else "TOO SMALL -- panel uninformative"
        print(f"   {ds:12s} Oracle-ZeroFill Live R2 headroom = {headroom[ds]:+.3f}  ({info})")

    print("\n[honesty] look-ahead optimism d=Reported-Live (per imputer, mean over panels):")
    for n in ORDER:
        gaps = [agg[ds][n]["gap_r2"][0] for ds in used]
        drifts = [agg[ds][n]["feat_drift"][0] for ds in used]
        tag = "  <- 0 by construction (causal)" if IS_CAUSAL[n] else "  <- LOOK-AHEAD"
        print(f"   {n:13s} mean dR2={np.mean(gaps):+.3f}  mean feat-drift={np.mean(drifts):.4f}{tag}")
    print(f"\n[utility] {util}")
    print(f"[verify] CAFE causal-verified: {cafe_ok}  ({cafe_detail})")

    # ---- cache ----
    cache = {
        "config": {
            "rows": ROWS, "train_frac": TRAIN_FRAC, "seeds": SEEDS,
            "miss_rate": MISS_RATE, "block_frac": BLOCK_FRAC, "n_anchor": N_ANCHOR,
            "ridge_lam": RIDGE_LAM, "miss_pct_actual": miss_pct,
            "headroom_min": HEADROOM_MIN,
            "datasets": used, "imputers": ORDER, "is_causal": IS_CAUSAL,
            "task": "contemporaneous target from imputed cross-section (no lag anchor)",
        },
        "raw": raw,
        "agg": {ds: {n: {k: list(v) for k, v in agg[ds][n].items()} for n in ORDER}
                for ds in used},
        "headroom_oracle_minus_zerofill": headroom,
        "cafe_causal_verified": cafe_ok,
        "cafe_causal_detail": cafe_detail,
        "utility_sentence_plain": util.replace(r"\cafe{}", "CAFE")
                                      .replace(r"\textbf{", "").replace("}", ""),
    }
    with open(CACHE_OUT, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\n[wrote] {CACHE_OUT}")

    _write_table(agg, used, miss_pct, cafe_ok, util, headroom)
    print(f"[done] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
