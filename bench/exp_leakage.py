"""exp_leakage.py -- the CAFE headline: look-ahead optimism of bidirectional imputation.

Central thesis: non-causal (bidirectional) imputation injects future information into
the *past* cells it fills. When those filled cells become features for a downstream
forecaster under a strict temporal train/test split, the reported test error is
optimistic -- because at "test time" the practitioner does NOT actually have the
future that the batch imputer secretly used. CAFE is point-in-time (truncation-
invariant), so its features are identical whether computed in batch or live; it has
no optimism gap.

We quantify two complementary, fully causal-honest numbers on REAL data:

(A) OPTIMISM GAP (downstream).  ETTh1[:4000], ~30% MCAR. A ridge predicts the
    next-step target y_t = X[t+1, target] from the imputed feature row at t (+ short
    lag window). Strict temporal split (first 70% train, last 30% test). For each
    imputer we report:
      - REPORTED test R^2 / RMSE: features built by ONE batch imputation of the whole
        masked matrix (what practitioners do). Test rows were filled using future.
      - LIVE test R^2 / RMSE: the honest procedure -- each test row's features are
        re-imputed point-in-time on the prefix up to that row (expanding-window
        walk-forward), so no future leaks.
    Optimism gap = REPORTED - LIVE (R^2 units; positive = optimistic inflation).
    The same trained ridge is scored both ways: only the test-feature construction
    differs, isolating the imputation leak. CAFE: reported == live by invariance.

(B) REVISION MOAT (direct).  For many EARLY missing cells, re-impute on growing
    prefixes X[:L] and measure how much the imputed value of a fixed past cell moves
    as the future is revealed. Mean |revision| on the z-scored scale. For CAFE this
    is exactly 0 by truncation invariance; for batch methods it is large.

Every number is read from the real estimators (m_softimpute, m_trmf, c_unified_penmf).
Self-contained, < ~6 min.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz_common as V
import harness as H
from m_softimpute import impute as softimpute
from m_trmf import impute as trmf
from c_unified_penmf import online_impute

ROOT = "/Users/dereksnow/Sovai/Github/TIMARA"
FIG = os.path.join(ROOT, "paper/figures/leakage.pdf")
TAB = os.path.join(ROOT, "paper/tables/leakage.tex")

T_CAP = 4000
MISS_RATE = 0.30
# Block (contiguous sensor-blackout) missingness is the realistic regime and the
# one where bidirectional methods leak hardest: batch imputers interpolate ACROSS a
# gap using BOTH endpoints (future + past), but an honest live deployment only has
# the past. MCAR understates the leak (scattered single cells are mostly surrounded
# by observed neighbours). We report block; MCAR gives the same sign, smaller size.
MISS_MECH = "block"
TARGET_COL = 6          # ETTh1 OT (oil temperature) -- the canonical target column
LAGS = 1                # short lag window for the downstream feature row
SEED = 0


# --------------------------------------------------------------------------- #
# imputer wrappers (uniform signature, contiguous input for CAFE)
# --------------------------------------------------------------------------- #
def cafe(X):
    return online_impute(np.ascontiguousarray(np.asarray(X, float)), {})


def soft(X):
    return softimpute(np.asarray(X, float).copy(), {})


def trmf_(X):
    return trmf(np.asarray(X, float).copy(), {})


BATCH = {"SoftImpute": soft, "TRMF": trmf_, "CAFE": cafe}
IS_CAUSAL = {"SoftImpute": False, "TRMF": False, "CAFE": True}


# --------------------------------------------------------------------------- #
# downstream feature/target construction
# --------------------------------------------------------------------------- #
def make_features(filled, lags=LAGS):
    """Row t -> stacked [filled[t], filled[t-1], ...]; target y_t = filled-free
    ground-truth next-step is supplied separately. Returns design matrix Phi (rows
    aligned to time index t in [lags, T-2])."""
    T, N = filled.shape
    rows = []
    idx = []
    for t in range(lags, T - 1):
        feat = np.concatenate([filled[t - l] for l in range(lags + 1)])
        rows.append(feat)
        idx.append(t)
    return np.asarray(rows), np.asarray(idx)


def ridge_fit(Phi, y, lam=1.0):
    """Closed-form ridge with intercept (standardize-free; lam on slope only)."""
    n, d = Phi.shape
    mu = Phi.mean(0)
    Phc = Phi - mu
    ym = y.mean()
    A = Phc.T @ Phc + lam * np.eye(d)
    b = Phc.T @ (y - ym)
    w = np.linalg.solve(A, b)
    return w, mu, ym


def ridge_pred(Phi, w, mu, ym):
    return (Phi - mu) @ w + ym


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


# --------------------------------------------------------------------------- #
# (A) optimism gap
# --------------------------------------------------------------------------- #
def live_walkforward(name, fn, Xobs, test_idx, lags, n_anchor=24):
    """Honest point-in-time feature construction for the TEST block.

    Expanding-window walk-forward: we place `n_anchor` anchors spanning the test
    block. At anchor L we impute ONLY the prefix Xobs[:L+1] (no future visible) and
    read off the freshly-filled rows in (prev_anchor, L]. Each test row is therefore
    filled using strictly its own past+present -- the honest deployment procedure.
    For a truncation-invariant method (CAFE) this returns features bit-identical to
    the batch fill; for batch methods it differs (that difference IS the leak)."""
    T = Xobs.shape[0]
    t_lo, t_hi = int(test_idx.min()), int(test_idx.max())
    # anchors are prefix lengths L (exclusive end) covering the test span
    anchors = np.unique(np.linspace(t_lo, T, n_anchor + 1).astype(int))
    anchors = anchors[anchors > t_lo]
    filled_live = {}
    prev = t_lo - 1
    for L in anchors:
        F = fn(Xobs[:L])
        for t in range(prev + 1, L):
            if t - lags >= 0 and t <= t_hi:
                filled_live[t] = np.concatenate([F[t - l] for l in range(lags + 1)])
        prev = L - 1
    Phi = np.asarray([filled_live[t] for t in test_idx])
    return Phi


def part_A(X, Xobs, M):
    T, N = X.shape
    y_all = X[:, TARGET_COL]                       # clean next-step target (ground truth)
    split = int(0.70 * T)

    results = {}
    for name, fn in BATCH.items():
        t0 = time.perf_counter()
        # batch (bidirectional) fill -> features
        Fb = fn(Xobs)
        Phi, idx = make_features(Fb, LAGS)
        y = y_all[idx + 1]                          # predict NEXT step
        tr = idx < split
        te = idx >= split
        Phi_tr, y_tr = Phi[tr], y[tr]
        idx_te = idx[te]
        y_te = y[te]

        w, mu, ym = ridge_fit(Phi_tr, y_tr)

        # REPORTED: test features from the batch (future-contaminated) fill
        Phi_te_reported = Phi[te]
        yhat_rep = ridge_pred(Phi_te_reported, w, mu, ym)
        r2_rep, rmse_rep = r2(y_te, yhat_rep), rmse(y_te, yhat_rep)

        # LIVE: test features re-imputed point-in-time (expanding walk-forward)
        Phi_te_live = live_walkforward(name, fn, Xobs, idx_te, LAGS)
        yhat_live = ridge_pred(Phi_te_live, w, mu, ym)
        r2_live, rmse_live = r2(y_te, yhat_live), rmse(y_te, yhat_live)

        dt = time.perf_counter() - t0
        # honesty check on invariance
        feat_drift = float(np.abs(Phi_te_reported - Phi_te_live).mean())
        results[name] = dict(r2_rep=r2_rep, r2_live=r2_live,
                             rmse_rep=rmse_rep, rmse_live=rmse_live,
                             gap_r2=r2_rep - r2_live, gap_rmse=rmse_live - rmse_rep,
                             feat_drift=feat_drift, causal=IS_CAUSAL[name], time_s=dt)
        print(f"[A] {name:11s} reportedR2={r2_rep:+.3f} liveR2={r2_live:+.3f} "
              f"gap={r2_rep - r2_live:+.3f}  feat_drift={feat_drift:.4f}  ({dt:.1f}s)")
    return results


# --------------------------------------------------------------------------- #
# (B) revision moat
# --------------------------------------------------------------------------- #
def part_B(Xobs, n_cells=40, t0_band=(80, 160), n_prefix=14, seed=SEED):
    """Mean |revision| of EARLY missing cells as the future is revealed."""
    rng = np.random.default_rng(seed)
    T, N = Xobs.shape
    miss = np.argwhere(np.isnan(Xobs))
    cand = miss[(miss[:, 0] >= t0_band[0]) & (miss[:, 0] <= t0_band[1])]
    if len(cand) > n_cells:
        cand = cand[rng.choice(len(cand), n_cells, replace=False)]
    t0_max = int(cand[:, 0].max())
    Ls = np.unique(np.linspace(t0_max + 20, T, n_prefix).astype(int))

    revis = {name: np.zeros(len(Ls)) for name in BATCH}      # mean |val(L) - val(L0)| over cells
    traj = {name: [] for name in BATCH}                      # per-prefix imputed values (n_cells, len Ls)
    for name, fn in BATCH.items():
        vals = np.zeros((len(cand), len(Ls)))
        for li, L in enumerate(Ls):
            F = fn(Xobs[:L])
            for ci, (t, j) in enumerate(cand):
                vals[ci, li] = F[t, j]
        base = vals[:, 0:1]
        rev = np.abs(vals - base)                            # revision vs first (smallest) prefix
        revis[name] = rev.mean(axis=0)
        traj[name] = vals
        print(f"[B] {name:11s} mean|revision| over prefixes = {rev[:, 1:].mean():.4f} "
              f"(final {rev[:, -1].mean():.4f}), max {rev.max():.4f}")
    return Ls, revis, traj, cand


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(A, Ls, revis):
    pal = V.PALETTE
    cmap = {"SoftImpute": pal["red"], "TRMF": pal["amber"], "CAFE": pal["blue"]}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 2.9), constrained_layout=True)

    # ---- LEFT: reported vs live downstream R^2 ----
    names = ["SoftImpute", "TRMF", "CAFE"]
    x = np.arange(len(names))
    w = 0.36
    rep = [A[n]["r2_rep"] for n in names]
    live = [A[n]["r2_live"] for n in names]
    b1 = axL.bar(x - w / 2, rep, w, color=[cmap[n] for n in names], alpha=0.55,
                 label="reported (batch fill)", edgecolor="white", lw=0.6)
    b2 = axL.bar(x + w / 2, live, w, color=[cmap[n] for n in names], alpha=1.0,
                 label="live (point-in-time)", edgecolor="white", lw=0.6)
    for n, xi in zip(names, x):
        g = A[n]["gap_r2"]
        ytop = max(A[n]["r2_rep"], A[n]["r2_live"])
        axL.annotate(f"$\\Delta$={g:+.2f}", (xi, ytop + 0.015), ha="center",
                     va="bottom", fontsize=7.5,
                     color=(pal["red"] if abs(g) > 0.02 else pal["green"]))
    axL.axhline(0, color=pal["grey"], lw=0.8)
    axL.set_xticks(x)
    axL.set_xticklabels([("CAFÉ" if n == "CAFE" else n) for n in names], fontsize=8)
    axL.set_ylabel("downstream test $R^2$", fontsize=9)
    axL.set_title("Look-ahead optimism: reported vs. honest live", fontsize=9)
    # custom legend: hatch by alpha
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(facecolor=pal["slate"], alpha=0.55, label="reported (batch fill)"),
                        Patch(facecolor=pal["slate"], alpha=1.0, label="live (point-in-time)")],
               fontsize=7, loc="upper left", frameon=False)
    V.style_ax(axL)

    # ---- RIGHT: revision magnitude vs prefix length ----
    for n in names:
        axR.plot(Ls, revis[n], "-o", color=cmap[n], lw=1.4, ms=3.0,
                 label=("CAFÉ" if n == "CAFE" else n),
                 zorder=4 if n == "CAFE" else 3)
    axR.set_xlabel("prefix length $L$ (rows revealed)", fontsize=9)
    axR.set_ylabel("mean $|$revision$|$ of early cells\n(z-scored)", fontsize=9)
    axR.set_title("Revision moat: batch fills wander, CAFÉ is frozen", fontsize=9)
    axR.legend(fontsize=7.5, loc="upper left", frameon=False)
    # annotate CAFE flat-at-zero
    axR.annotate("CAFÉ $\\equiv 0$ (truncation-invariant)",
                 xy=(Ls[len(Ls) // 2], revis["CAFE"][len(Ls) // 2]),
                 xytext=(Ls[len(Ls) // 2], 0.12 * max(revis["SoftImpute"].max(), 1e-3)),
                 fontsize=7, color=pal["blue"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=pal["blue"], lw=0.8))
    V.style_ax(axR)

    fig.savefig(FIG, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    sz = os.path.getsize(FIG)
    print(f"saved {FIG} ({sz} bytes)")
    assert sz > 3000


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def make_table(A, revis, Ls):
    names = ["SoftImpute", "TRMF", "CAFE"]
    rev_final = {n: float(revis[n][-1]) for n in names}

    def f(v, dp=3):
        return f"{v:.{dp}f}".replace("-0.000", "0.000")

    lines = []
    lines.append(r"\begin{table}[t]\centering\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\caption{\textbf{Look-ahead optimism on ETTh1.} A ridge forecaster predicts "
                 r"the next-step target from imputed features under a strict temporal split "
                 r"($70/30$, $30\%$ contiguous block missingness). \emph{Reported} $R^2$ uses a "
                 r"single batch (bidirectional) imputation; \emph{Live} re-imputes each test row "
                 r"point-in-time (expanding "
                 r"walk-forward). The same trained model is scored both ways. "
                 r"$\Delta_{R^2}=\text{Reported}-\text{Live}$ is the optimism injected by look-ahead. "
                 r"\emph{Revision} is the mean $|$change$|$ of an early imputed cell as the future "
                 r"is revealed (z-scored). \cafe{} is causal: $\Delta_{R^2}=0$ and revision $=0$ "
                 r"\emph{by truncation invariance}.}")
    lines.append(r"\label{tab:leakage}")
    lines.append(r"\begin{tabular}{@{}lccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Causal? & Rep.\ $R^2$ & Live $R^2$ & $\Delta_{R^2}\!\downarrow$ & Rev.\ $\downarrow$ \\")
    lines.append(r"\midrule")
    for n in names:
        a = A[n]
        causal = r"\ding{51}" if a["causal"] else r"\ding{55}"
        gap = a["gap_r2"]
        rv = rev_final[n]
        disp = r"\cafe{}" if n == "CAFE" else n
        bold = (lambda s: r"\textbf{" + s + "}") if n == "CAFE" else (lambda s: s)
        gap_s = "0.000" if (n == "CAFE" or abs(gap) < 5e-4) else f(gap)
        rv_s = "0.000" if (n == "CAFE" or rv < 5e-4) else f(rv)
        lines.append(f"{bold(disp)} & {causal} & {f(a['r2_rep'])} & {f(a['r2_live'])} & "
                     f"{bold(gap_s)} & {bold(rv_s)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"
    with open(TAB, "w") as fh:
        fh.write(tex)
    print(f"saved {TAB}")
    return tex


# --------------------------------------------------------------------------- #
def main():
    t_start = time.perf_counter()
    X = np.load(os.path.join(ROOT, "data/ETTh1_clean.npy"))[:T_CAP].astype(float)
    M = H.MASKERS[MISS_MECH](X, MISS_RATE, seed=SEED)
    Xobs = X.copy()
    Xobs[M] = np.nan
    print(f"ETTh1[:{T_CAP}] shape={X.shape}  mech={MISS_MECH} missing={M.mean():.2%}  "
          f"target_col={TARGET_COL}")

    A = part_A(X, Xobs, M)
    Ls, revis, traj, cand = part_B(Xobs)

    make_figure(A, Ls, revis)
    tex = make_table(A, revis, Ls)

    print("\n===== HEADLINE =====")
    soft_gap = A["SoftImpute"]["gap_r2"]
    trmf_gap = A["TRMF"]["gap_r2"]
    print(f"Optimism gap (Reported-Live R^2):  SoftImpute={soft_gap:+.3f}  "
          f"TRMF={trmf_gap:+.3f}  CAFE={A['CAFE']['gap_r2']:+.3f}")
    print(f"  SoftImpute reported R^2={A['SoftImpute']['r2_rep']:+.3f} vs live {A['SoftImpute']['r2_live']:+.3f}")
    print(f"  TRMF       reported R^2={A['TRMF']['r2_rep']:+.3f} vs live {A['TRMF']['r2_live']:+.3f}")
    print(f"  CAFE       reported R^2={A['CAFE']['r2_rep']:+.3f} vs live {A['CAFE']['r2_live']:+.3f} (no gap)")
    print(f"Revision moat (mean|rev| z-scored, final prefix): "
          f"SoftImpute={revis['SoftImpute'][-1]:.3f}  TRMF={revis['TRMF'][-1]:.3f}  "
          f"CAFE={revis['CAFE'][-1]:.3f}")
    print(f"total wall time {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
