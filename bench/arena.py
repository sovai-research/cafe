"""
ARENA — the active-research-loop scorer + champion/challenger RATCHET.

A model "config" is just an impute fn `f(X, meta) -> filled`. The arena scores it
across the FULL benchmark suite (diverse synthetic + real published datasets),
evaluating EACH case over MULTIPLE mask seeds (reusing the seed-rigor machinery
from exp_seeds.py) so the numbers are mean +/- 95% CI rather than single-seed
point estimates. It produces two aggregate axes:

    accuracy  = mean MAE over cases (LOWER better)   <- PRIMARY selection axis
    speed     = total wall-clock seconds over cases  (lower better)

We lead with MAE (not correlation): correlation is affine-invariant and hides
bias (the MNAR failure mode the project admits). Correlation is still recorded
as a SECONDARY diagnostic (mean_corr) and never used for the gate.

RATCHET: a challenger becomes champion ONLY if it improves the (MAE, time) pair
with no individual case regressing in MAE by more than --maxdrop-mae, plus a
value-or-speed win expressed in MAE terms (see gate()). Anything else "fails"
and triggers a literature-research step (done by the loop driver, not here).

Multi-seed: with N seeds per case, the per-case MAE is the mean over seeds and
each case carries a 95% Student-t CI half-width. The gate compares per-case mean
MAE; the aggregate `mae_mean` is the mean of per-case mean-MAEs and `mae_ci` is
the Student-t CI over the per-case means.

Usage:
    python3 arena.py --eval c_router:online_impute --name CausalRouter
    python3 arena.py --eval c_router:online_impute --name CausalRouter --set-champion
    python3 arena.py --eval c_chal_seasonal:online_impute --name Chal_seasonal --challenge
    python3 arena.py --ledger          # show champion + history
"""
import os, sys, json, time, argparse
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
try:
    from scipy import stats as _stats          # used for the Student-t CI (as in exp_seeds.py)
except Exception:                              # noqa: BLE001 - scipy optional; fall back below
    _stats = None
from harness import DATASETS, MASKERS, metrics

DATA = os.path.join(HERE, "..", "data")
_FAST_ENV = os.environ.get("TIMARA_FAST", "") not in ("", "0", "false")
LEDGER = os.path.join(HERE, "arena_ledger_fast.json" if _FAST_ENV else "arena_ledger.json")

# Multi-seed config: number of mask seeds folded into each case's MAE.
# FAST keeps it cheap; full run is more rigorous. (exp_seeds.py uses 6/12.)
N_SEEDS = int(os.environ.get("TIMARA_ARENA_SEEDS", "3" if _FAST_ENV else "5"))


# --------------------------------------------------------------------------- #
# Build the case suite: synthetic (harness) + real published datasets.
# Each case: name -> (clean_2d, meta, mech, rate, group)
# --------------------------------------------------------------------------- #
def _real_block_mask(X, rate, seed=1):
    rng = np.random.default_rng(seed); T, N = X.shape; M = np.zeros((T, N), bool)
    blen = max(1, int(rate * T))
    for j in range(N):
        s = int(rng.integers(0, max(1, T - blen))); M[s:s + blen, j] = True
    return M


def _load_real():
    """Real published datasets, loaded lazily; skip silently if absent."""
    cases = {}
    vldb = {"airq": "airq_normal.txt", "chlorine": "chlorine_normal.txt",
            "drift": "drift10_normal.txt", "temp": "temp_normal.txt"}
    for k, fn in vldb.items():
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            X = np.loadtxt(p)
            cases[f"real_{k}_block10"] = (X, {}, "block", 0.10, "real-vldb")
    # point-protocol reals (TSI-Bench style, 10% MCAR point) — prepared by prep_real.py
    for k in ("ETTh1", "beijing", "electricity"):
        p = os.path.join(DATA, f"{k}_clean.npy")
        if os.path.exists(p):
            X = np.load(p)
            cases[f"real_{k}_mcar10"] = (X, {}, "mcar", 0.10, "real-point")
    return cases


def _cap_case(clean, meta, cap, e_fast):
    """Shrink a case for FAST iteration while preserving its structure/type."""
    if meta and "entity_ids" in meta:                      # panel: keep first e_fast entities
        eids = np.asarray(meta["entity_ids"])
        keep = eids < e_fast
        m2 = dict(meta)
        for k in ("entity_ids", "time_ids"):
            if k in meta:
                m2[k] = np.asarray(meta[k])[keep]
        m2["E"] = int(e_fast)
        return clean[keep], m2
    return clean[:cap], meta                                # 1d/2d: truncate time


def build_suite(fast=False, cap=600, e_fast=8):
    suite = {}
    for n, (c, m, mech, rate) in DATASETS.items():
        grp = "panel" if m else ("1d" if c.shape[1] == 1 else "2d")
        if fast:
            c, m = _cap_case(c, m, cap, e_fast)
        suite[n] = (c, m, mech, rate, grp)
    for n, (c, m, mech, rate, grp) in _load_real().items():
        if fast:
            c, m = _cap_case(c, m, cap, e_fast)             # also caps temp/beijing rows
        suite[n] = (c, m, mech, rate, grp)
    return suite


FAST = os.environ.get("TIMARA_FAST", "") not in ("", "0", "false")
SUITE = build_suite(fast=FAST)


# --------------------------------------------------------------------------- #
# Statistics — Student-t 95% CI half-width for the mean (mirrors exp_seeds._t_ci)
# --------------------------------------------------------------------------- #
def _t_ci(x, conf=0.95):
    """Two-sided Student-t CI half-width for the MEAN of x. Falls back to a
    normal-approx z-critical if scipy is unavailable."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 0.0
    sem = float(np.std(x, ddof=1) / np.sqrt(n))
    if _stats is not None:
        tcrit = float(_stats.t.ppf(0.5 + conf / 2.0, df=n - 1))
    else:
        tcrit = 1.959963984540054                  # z for 95%
    return tcrit * sem


# --------------------------------------------------------------------------- #
# Evaluate one config across the suite, folding over N mask seeds per case.
# --------------------------------------------------------------------------- #
def eval_config(fn, suite=None, n_seeds=None):
    suite = suite or SUITE
    n_seeds = N_SEEDS if n_seeds is None else int(n_seeds)
    seeds = list(range(n_seeds))
    rows = {}
    for name, (clean, meta, mech, rate, grp) in suite.items():
        is_vldb_block = (mech == "block" and grp == "real-vldb")
        masker = _real_block_mask if is_vldb_block else MASKERS[mech]
        maes, corrs, rmses = [], [], []
        dt_total = 0.0
        ok = True
        err = None
        for seed in seeds:
            M = masker(clean, rate, seed)
            Xo = clean.copy(); Xo[M] = np.nan
            t0 = time.perf_counter()
            try:
                pred = np.asarray(fn(Xo.copy(), dict(meta)), float)
                dt_total += time.perf_counter() - t0
                mt = metrics(clean, pred, M)
                maes.append(mt["mae"]); corrs.append(mt["corr"]); rmses.append(mt["rmse"])
            except Exception as e:                          # noqa: BLE001
                dt_total += time.perf_counter() - t0
                ok = False
                err = f"{type(e).__name__}: {e}"
                break
        if ok and maes:
            fmae = [v for v in maes if np.isfinite(v)]
            fcorr = [v for v in corrs if np.isfinite(v)]
            frmse = [v for v in rmses if np.isfinite(v)]
            rows[name] = dict(
                mae=float(np.mean(fmae)) if fmae else float("nan"),
                mae_ci=_t_ci(fmae) if fmae else 0.0,
                corr=float(np.mean(fcorr)) if fcorr else float("nan"),
                rmse=float(np.mean(frmse)) if frmse else float("nan"),
                time_s=dt_total, n_seeds=len(maes), group=grp, ok=True)
        else:
            rows[name] = dict(mae=float("nan"), mae_ci=0.0, corr=float("nan"),
                              rmse=float("nan"), time_s=dt_total,
                              n_seeds=len(maes), group=grp, ok=False, err=err)
    # PRIMARY axis = MAE (lower better). corr kept as a secondary diagnostic.
    maes = [r["mae"] for r in rows.values() if r["ok"] and np.isfinite(r["mae"])]
    corrs = [r["corr"] for r in rows.values() if r["ok"] and np.isfinite(r["corr"])]
    agg = dict(
        mae_mean=float(np.mean(maes)) if maes else float("nan"),
        mae_ci=_t_ci(maes) if maes else 0.0,                # CI over per-case mean MAEs
        max_mae=float(np.max(maes)) if maes else float("nan"),
        # --- secondary diagnostic: correlation (NOT used for the gate) ---
        corr_mean=float(np.mean(corrs)) if corrs else float("nan"),
        min_corr=float(np.min(corrs)) if corrs else float("nan"),
        total_time=float(sum(r["time_s"] for r in rows.values())),
        n_fail=sum(1 for r in rows.values() if not r["ok"]),
        n_cases=len(rows),
        n_seeds=n_seeds)
    return agg, rows


# --------------------------------------------------------------------------- #
# Backward-compatible ledger accessors (old entries are corr-only, higher-better)
# --------------------------------------------------------------------------- #
def _agg_mae(agg):
    """Primary accuracy (MAE, lower better). Legacy corr-only entries lack MAE."""
    return agg.get("mae_mean", float("nan")) if agg else float("nan")


def _agg_corr(agg):
    """Secondary diagnostic (corr). Reads new `corr_mean` or legacy `mean_corr`."""
    if not agg:
        return float("nan")
    return agg.get("corr_mean", agg.get("mean_corr", float("nan")))


def _row_mae(r):
    return r.get("mae", float("nan")) if r else float("nan")


def _load_ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"champion": None, "history": []}


def _save_ledger(L):
    json.dump(L, open(LEDGER, "w"), indent=1)


def gate(champ, chal, maxdrop_mae=0.02, cost_budget=0.10):
    """VALUE ratchet on MAE (LOWER is better; mirrors the prior corr-based gate
    structure with the metric flipped). Accept iff:
      - no single case's MAE regresses (grows) by more than `maxdrop_mae`
        (default 0.02 on the standardized scale the suite uses), AND
      - EITHER  MAE improves (drops) AND time grows <= cost_budget (default +10%)
                AND the relative MAE gain >= the relative time cost,
        OR      faster-or-equal at no MAE loss (pure speed/Pareto win).
    Note the sign flip vs the old corr gate: an improvement means MAE goes DOWN,
    so d_mae = champ_mae - chal_mae is POSITIVE when the challenger is better.
    Returns (accept, reason)."""
    if champ is None:
        return True, "no champion yet"
    c_mae, c_t = _agg_mae(champ["agg"]), champ["agg"]["total_time"]
    n_mae, n_t = _agg_mae(chal["agg"]), chal["agg"]["total_time"]
    if not np.isfinite(c_mae):
        # Legacy corr-only champion has no MAE to compare against. The only way
        # to ratchet onto an MAE basis is to set a fresh champion explicitly.
        return False, "champion is legacy corr-only (no MAE baseline); re-run --set-champion to migrate"
    d_mae = c_mae - n_mae                              # >0 means challenger lower error (better)
    rel_cost = (n_t - c_t) / max(c_t, 1e-9)           # >0 means slower
    rel_gain = d_mae / max(abs(c_mae), 1e-9)
    # per-case regression guard (MAE must not GROW anywhere by > maxdrop_mae)
    worst = 0.0
    for k, r in chal["rows"].items():
        cr = champ["rows"].get(k)
        rm, cm = _row_mae(r), _row_mae(cr) if cr else float("nan")
        if cr and r.get("ok") and cr.get("ok") and np.isfinite(rm) and np.isfinite(cm):
            worst = max(worst, rm - cm)               # positive => challenger worse on this case
    if worst > maxdrop_mae:
        return False, f"case MAE regressed by {worst:.4f} (> {maxdrop_mae})"
    # pure speed/Pareto win: faster-or-equal with no MAE loss
    if d_mae >= -1e-9 and rel_cost < -1e-6:
        return True, f"speed win (MAE {-d_mae:+.4f}, time {rel_cost*100:+.1f}%)"
    # value win: any genuine MAE reduction for <= +10% time (user rule)
    if d_mae > 1e-6 and rel_cost <= cost_budget and rel_gain >= rel_cost:
        return True, f"value win (MAE {-d_mae:+.4f}, cost {rel_cost*100:+.1f}%)"
    if rel_cost > cost_budget:
        return False, f"too slow ({rel_cost*100:+.1f}% > {cost_budget*100:.0f}% budget)"
    if d_mae > 1e-6:
        return False, f"gain too small for cost (MAE {-d_mae:+.4f}, time {rel_cost*100:+.1f}%)"
    return False, f"no MAE reduction (MAE {-d_mae:+.4f}, time {rel_cost*100:+.1f}%)"


def _load_fn(spec):
    mod, _, attr = spec.partition(":")
    m = __import__(mod)
    return getattr(m, attr[:-2])() if attr.endswith("()") else getattr(m, attr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval"); ap.add_argument("--name", default="cfg")
    ap.add_argument("--set-champion", action="store_true")
    ap.add_argument("--challenge", action="store_true")
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--maxdrop-mae", "--maxdrop", dest="maxdrop_mae",
                    type=float, default=0.02,
                    help="max per-case MAE increase tolerated (lower better; default 0.02)")
    ap.add_argument("--seeds", type=int, default=None,
                    help=f"mask seeds per case (default {N_SEEDS}; TIMARA_FAST shrinks)")
    a = ap.parse_args()

    if a.ledger:
        L = _load_ledger(); ch = L["champion"]
        if ch:
            agg = ch["agg"]
            mae = _agg_mae(agg); corr = _agg_corr(agg)
            mae_ci = agg.get("mae_ci", 0.0); nseed = agg.get("n_seeds", 1)
            if np.isfinite(mae):
                print(f"CHAMPION: {ch['name']}  mae={mae:.4f}+/-{mae_ci:.4f} "
                      f"(corr={corr:.4f}) total_time={agg['total_time']:.2f}s "
                      f"[n_seeds={nseed}]")
            else:                                   # legacy corr-only entry
                print(f"CHAMPION: {ch['name']}  mae=n/a(legacy) corr={corr:.4f} "
                      f"total_time={agg['total_time']:.2f}s [legacy corr-only]")
        print(f"history: {len(L['history'])} entries")
        for h in L["history"][-12:]:
            agg = h.get("agg", {})
            mae = _agg_mae(agg); corr = _agg_corr(agg)
            mae_s = f"mae={mae:.4f}" if np.isfinite(mae) else "mae=n/a"
            print(f"  {h['name']:22s} {mae_s} (corr={corr:.4f}) "
                  f"time={agg.get('total_time', float('nan')):.2f}s -> {h.get('verdict','')}")
        return

    if not a.eval:
        print("need --eval module:fn"); return
    fn = _load_fn(a.eval)
    t0 = time.perf_counter()
    agg, rows = eval_config(fn, n_seeds=a.seeds)
    print(f"[{a.name}] mae={agg['mae_mean']:.4f}+/-{agg['mae_ci']:.4f} "
          f"max_mae={agg['max_mae']:.4f} (corr={agg['corr_mean']:.4f}) "
          f"total_time={agg['total_time']:.2f}s fails={agg['n_fail']}/{agg['n_cases']} "
          f"n_seeds={agg['n_seeds']} (eval wall {time.perf_counter()-t0:.1f}s)")
    # per-group breakdown (PRIMARY = MAE; corr shown as secondary)
    for grp in ("1d", "2d", "panel", "real-vldb", "real-point"):
        gm = [r["mae"] for r in rows.values() if r["group"] == grp and r["ok"] and np.isfinite(r["mae"])]
        gc = [r["corr"] for r in rows.values() if r["group"] == grp and r["ok"] and np.isfinite(r["corr"])]
        if gm:
            print(f"    {grp:11s} mae={np.mean(gm):.4f}  (corr={np.mean(gc):.4f}, {len(gm)} cases)")
    entry = {"name": a.name, "agg": agg, "rows": rows}

    if a.set_champion:
        L = _load_ledger(); L["champion"] = entry
        L["history"].append({**entry, "verdict": "set-champion"}); _save_ledger(L)
        print("  -> set as champion")
    elif a.challenge:
        L = _load_ledger()
        ok, reason = gate(L["champion"], entry, a.maxdrop_mae)
        print(f"  GATE: {'ACCEPT' if ok else 'REJECT'} — {reason}")
        entry_log = {"name": a.name, "agg": agg, "verdict": ("ACCEPT" if ok else "REJECT") + ": " + reason}
        L["history"].append(entry_log)
        if ok:
            L["champion"] = entry
        _save_ledger(L)


if __name__ == "__main__":
    main()
