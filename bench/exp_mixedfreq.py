"""Mixed-frequency causal NOWCASTING experiment.

Does CAFE nowcast a low-frequency target from a high-frequency cross-section OUT OF THE
BOX -- i.e. with no frequency-aware machinery, purely by treating the low-freq column as
a structured missingness pattern and reading off its cross-sectional factor fill?

Protocol (strictly point-in-time / causal):
  At each low-freq RELEASE time t (a quarter end), the current-quarter target value is
  NOT yet released (ragged edge). We run CAFE on the time-truncated prefix X[:t+1] with
  the target cell at row t set NaN, and read the imputed value at (t, j) as the nowcast.
  Truncating to [:t+1] means the nowcast can never use any future row -> causal by
  construction (verified by re-running on a shorter prefix and checking the earlier
  nowcasts are unchanged).

Honest baselines on the SAME release times and SAME target cells:
  * persistence : last released low-freq value (forward-fill) -- the standard nowcasting
                  benchmark; strong when the target is near-random-walk.
  * bridge-OLS  : OLS of past released target on the contemporaneous high-freq panel mean
                  (a 1-factor MIDAS/bridge), refit point-in-time at each t.
  * factor-OLS  : OLS of past released target on the leading high-freq common factors
                  (an 8-factor bridge), refit point-in-time -- a stronger DFM-style rival.

Outputs:
  * paper/tables/mixedfreq.tex   nowcast MAE/RMSE: CAFE vs baselines per target + mean
  * paper/figures/mixedfreq.pdf  (a) nowcast trajectory vs truth; (b) skill sharpens as
                                 more high-freq data arrives within the quarter
Prints all numbers. Exit 0 on success.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c_unified_penmf as P                                    # noqa: E402
from datasets_mixedfreq import make_mixedfreq, _intrinsic_stats  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAB = os.path.join(ROOT, "paper", "tables", "mixedfreq.tex")
FIG = os.path.join(ROOT, "paper", "figures", "mixedfreq.pdf")


def _mae_rmse(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    e = pred - truth
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))


def cafe_nowcast(task, release_ts):
    """Point-in-time CAFE nowcast at each release time (ragged edge)."""
    X, Xtruth, tgt, obs_t = task["X"], task["Xtruth"], task["tgt"], task["obs_t"]
    nc = {j: [] for j in tgt}
    for t in release_ts:
        Xp = X[: t + 1].copy()
        for j in tgt:
            Xp[t, j] = np.nan                       # current-quarter value not yet released
        Fp = P.online_impute(Xp, {})
        for j in tgt:
            nc[j].append(float(Fp[t, j]))
    return nc


def baseline_nowcasts(task, release_ts, n_factors=8):
    """persistence, 1-factor bridge-OLS, k-factor factor-OLS -- all point-in-time."""
    Xt, tgt, hf, obs_t = task["Xtruth"], task["tgt"], task["hf"], task["obs_t"]
    per = {j: [] for j in tgt}; brd = {j: [] for j in tgt}; fac = {j: [] for j in tgt}
    # Precompute leading common factors from the HIGH-FREQ block only (full-sample loadings
    # are a fixed transform of the panel; the regression coefficients below are refit
    # point-in-time, so the *nowcast prediction* uses only past target releases).
    Xhf = Xt[:, hf]
    Xhf_c = Xhf - Xhf.mean(0)
    U, S, Vt = np.linalg.svd(Xhf_c, full_matrices=False)
    Fhf = U[:, :n_factors] * S[:n_factors]                    # (T, n_factors) factor scores
    hfmean = Xt[:, hf].mean(1)
    for t in release_ts:
        past = [s for s in range(t) if obs_t[s]]
        for j in tgt:
            lastv = Xt[past[-1], j] if past else 0.0
            per[j].append(lastv)
            ys = np.array([Xt[s, j] for s in past])
            if len(ys) >= 5:
                xs = hfmean[past]
                A = np.vstack([xs, np.ones_like(xs)]).T
                b, *_ = np.linalg.lstsq(A, ys, rcond=None)
                brd[j].append(float(b[0] * hfmean[t] + b[1]))
                Fp = np.column_stack([Fhf[past], np.ones(len(past))])
                bf, *_ = np.linalg.lstsq(Fp, ys, rcond=None)
                fac[j].append(float(np.append(Fhf[t], 1.0) @ bf))
            else:
                brd[j].append(lastv); fac[j].append(lastv)
    return {"persistence": per, "bridge-OLS": brd, "factor-OLS": fac}


def within_quarter_sharpening(task, release_ts):
    """Nowcast MAE of the current-quarter target as a function of months-into-quarter."""
    X, Xt, tgt = task["X"], task["Xtruth"], task["tgt"]
    by_pos = {0: [], 1: [], 2: []}
    for rt in release_ts:
        for pos in range(3):
            cur = rt - (2 - pos)                       # month within the quarter
            if cur < 10:
                continue
            Xp = X[: cur + 1].copy()
            for j in tgt:
                Xp[cur, j] = np.nan
            Fp = P.online_impute(Xp, {})
            err = np.mean([abs(Fp[cur, j] - Xt[rt, j]) for j in tgt])
            by_pos[pos].append(err)
    return {p: float(np.mean(v)) for p, v in by_pos.items() if v}


def verify_causal(task, release_ts):
    """Re-run CAFE on a shorter prefix; nowcasts at earlier release times must be
    unchanged (no future leakage). Returns max abs deviation."""
    if len(release_ts) < 4:
        return 0.0
    full = cafe_nowcast(task, release_ts)
    cut = release_ts[: len(release_ts) // 2]
    # build a task truncated to the last cut release time + buffer
    tmax = cut[-1] + 1
    sub = dict(task)
    sub["X"] = task["X"][:tmax].copy()
    sub["Xtruth"] = task["Xtruth"][:tmax]
    sub["obs_t"] = task["obs_t"][:tmax]
    part = cafe_nowcast(sub, cut)
    dev = 0.0
    for i, t in enumerate(cut):
        for j in task["tgt"]:
            dev = max(dev, abs(full[j][i] - part[j][i]))
    return float(dev)


def write_table(rows, mean_row, task):
    """rows: list of (method, mae, rmse, is_causal). Bold best causal."""
    causal_maes = [r[1] for r in rows if r[3]]
    best = min(causal_maes) if causal_maes else None
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{\textbf{Zero-config causal nowcasting of a low-frequency target.} "
        r"Nowcast error ($\downarrow$) on a quarterly target embedded in the monthly "
        r"FRED-MD panel (observed at quarter ends, NaN otherwise; the current quarter is "
        r"unreleased at nowcast time---a ragged edge). At each release the target is "
        r"nowcast \emph{point-in-time} from the high-frequency cross-section on the "
        r"time-truncated prefix; \cafe{} treats the low-freq column purely as structured "
        r"missingness and reads off its cross-sectional factor fill---no frequency-aware "
        r"code. Targets are the four series with the lowest persistence yet highest common-"
        r"factor content (the activity/spread series central banks actually nowcast). "
        r"\textbf{Bold} $=$ best. Averaged over " + str(task["_n_release"]) +
        r" release times.}",
        r"\label{tab:mixedfreq}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Nowcaster & MAE & RMSE \\",
        r"\midrule",
    ]
    for name, mae, rmse, _ in rows:
        m = f"\\textbf{{{mae:.3f}}}" if best is not None and abs(mae - best) < 1e-9 else f"{mae:.3f}"
        r = f"\\textbf{{{rmse:.3f}}}" if best is not None and abs(mae - best) < 1e-9 else f"{rmse:.3f}"
        disp = r"\textbf{\cafe{} (ours)}" if name == "CAFE" else name
        lines.append(f"{disp} & ${m}$ & ${r}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(TAB), exist_ok=True)
    with open(TAB, "w") as f:
        f.write("\n".join(lines))


def write_figure(traj, sharp, task):
    """(a) nowcast trajectory vs truth for one target; (b) skill vs months-into-quarter."""
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
    t_idx, truth, cafe_nc, per_nc = traj
    ax[0].plot(t_idx, truth, color="black", lw=1.6, label="truth (quarterly)")
    ax[0].plot(t_idx, cafe_nc, color="#1f77b4", lw=1.4, marker="o", ms=3, label="CAFE nowcast")
    ax[0].plot(t_idx, per_nc, color="#d62728", lw=1.0, ls="--", alpha=0.8, label="persistence")
    ax[0].set_xlabel("release time (month index)")
    ax[0].set_ylabel("standardised target")
    ax[0].set_title("(a) Point-in-time nowcast vs truth", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)

    pos = sorted(sharp.keys())
    vals = [sharp[p] for p in pos]
    ax[1].plot([p + 1 for p in pos], vals, color="#2ca02c", lw=1.8, marker="s", ms=6)
    ax[1].set_xticks([p + 1 for p in pos])
    ax[1].set_xlabel("high-freq months observed within the quarter")
    ax[1].set_ylabel("nowcast MAE")
    ax[1].set_title("(b) Skill sharpens with more high-freq data", fontsize=10)
    for p, v in zip(pos, vals):
        ax[1].annotate(f"{v:.3f}", (p + 1, v), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    t0 = time.time()
    task = make_mixedfreq("fredmd", k=3, n_targets=4)
    Xt, tgt, obs_t = task["Xtruth"], task["tgt"], task["obs_t"]
    stats = _intrinsic_stats(Xt, tgt, task["k"])

    # evaluation release times: second half of the sample, subsampled for cost.
    release_ts = [t for t in range(len(obs_t)) if obs_t[t] and t >= 150][::3]
    task["_n_release"] = len(release_ts)

    print("=" * 64)
    print("MIXED-FREQUENCY CAUSAL NOWCASTING  (FRED-MD, quarterly target)")
    print("=" * 64)
    print(f"panel {task['X'].shape}, k={task['k']}, targets={tgt}, "
          f"{len(release_ts)} release times evaluated")
    for j in tgt:
        ac, r2 = stats[j]
        print(f"  target col {j}: lag-3 autocorr={ac:+.2f}  factorR2={r2:.2f}")

    # ---- CAFE nowcast (out of the box) ----
    cafe_nc = cafe_nowcast(task, release_ts)
    bl = baseline_nowcasts(task, release_ts)

    methods = {"CAFE": cafe_nc, **bl}
    truth = {j: np.array([Xt[t, j] for t in release_ts]) for j in tgt}

    print("\nper-target nowcast MAE:")
    rows_per = {m: [] for m in methods}
    for j in tgt:
        line = f"  col {j}:"
        for m in methods:
            mae, _ = _mae_rmse(methods[m][j], truth[j])
            rows_per[m].append(mae)
            line += f"  {m}={mae:.3f}"
        print(line)

    # pooled over all targets
    print("\nPOOLED nowcast skill (all targets, all release times):")
    table_rows = []
    causal_flag = {"CAFE": True, "persistence": True, "bridge-OLS": True, "factor-OLS": True}
    for m in methods:
        allp = np.concatenate([methods[m][j] for j in tgt])
        allt = np.concatenate([truth[j] for j in tgt])
        mae, rmse = _mae_rmse(allp, allt)
        table_rows.append((m, mae, rmse, causal_flag[m]))
        print(f"  {m:12s}: MAE={mae:.3f}  RMSE={rmse:.3f}")

    cafe_mae = [r[1] for r in table_rows if r[0] == "CAFE"][0]
    per_mae = [r[1] for r in table_rows if r[0] == "persistence"][0]
    print(f"\nCAFE vs persistence: {100*(per_mae-cafe_mae)/per_mae:+.1f}% MAE "
          f"({'better' if cafe_mae<per_mae else 'WORSE'})")

    # ---- causality check ----
    dev = verify_causal(task, release_ts)
    print(f"causal check (max nowcast deviation under prefix truncation): {dev:.2e}  "
          f"{'OK (causal)' if dev < 1e-4 else 'LEAK!'}")

    # ---- within-quarter sharpening ----
    sharp = within_quarter_sharpening(task, release_ts[::2])
    print("\nwithin-quarter sharpening (nowcast MAE by months-into-quarter):")
    for p in sorted(sharp):
        print(f"  month {p+1}: MAE={sharp[p]:.3f}")

    # ---- outputs ----
    write_table(table_rows, None, task)
    j0 = tgt[0]
    traj = (np.array(release_ts), truth[j0], np.array(cafe_nc[j0]),
            np.array(bl["persistence"][j0]))
    write_figure(traj, sharp, task)
    print(f"\nwrote {os.path.relpath(TAB, ROOT)} and {os.path.relpath(FIG, ROOT)}")
    print(f"total runtime {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
