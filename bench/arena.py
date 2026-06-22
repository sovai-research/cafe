"""
ARENA — the active-research-loop scorer + champion/challenger RATCHET.

A model "config" is just an impute fn `f(X, meta) -> filled`. The arena scores it
across the FULL benchmark suite (diverse synthetic + real published datasets),
producing two aggregate axes:
    accuracy  = mean corr over cases (higher better)
    speed     = total wall-clock seconds over cases (lower better)

RATCHET: a challenger becomes champion ONLY if it Pareto-improves the pair --
mean_corr >= champion AND total_time <= champion, with at least one strict
(and no individual case regressing by more than --maxdrop). Anything else "fails"
and triggers a literature-research step (done by the loop driver, not here).

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
from harness import DATASETS, MASKERS, metrics

DATA = os.path.join(HERE, "..", "data")
LEDGER = os.path.join(HERE, "arena_ledger.json")


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


def build_suite():
    suite = {}
    for n, (c, m, mech, rate) in DATASETS.items():
        grp = "panel" if m else ("1d" if c.shape[1] == 1 else "2d")
        suite[n] = (c, m, mech, rate, grp)
    suite.update(_load_real())
    return suite


SUITE = build_suite()


# --------------------------------------------------------------------------- #
# Evaluate one config across the suite
# --------------------------------------------------------------------------- #
def eval_config(fn, suite=None, seed=0):
    suite = suite or SUITE
    rows = {}
    for name, (clean, meta, mech, rate, grp) in suite.items():
        masker = _real_block_mask if mech == "block" and grp == "real-vldb" else MASKERS[mech]
        M = masker(clean, rate, seed) if mech != "block" or grp == "real-vldb" else MASKERS["block"](clean, rate, seed)
        Xo = clean.copy(); Xo[M] = np.nan
        t0 = time.perf_counter()
        try:
            pred = np.asarray(fn(Xo.copy(), dict(meta)), float)
            dt = time.perf_counter() - t0
            mt = metrics(clean, pred, M)
            rows[name] = dict(corr=mt["corr"], rmse=mt["rmse"], time_s=dt, group=grp, ok=True)
        except Exception as e:                          # noqa: BLE001
            rows[name] = dict(corr=float("nan"), rmse=float("nan"),
                              time_s=time.perf_counter() - t0, group=grp, ok=False,
                              err=f"{type(e).__name__}: {e}")
    corrs = [r["corr"] for r in rows.values() if r["ok"] and np.isfinite(r["corr"])]
    agg = dict(mean_corr=float(np.mean(corrs)) if corrs else float("nan"),
               min_corr=float(np.min(corrs)) if corrs else float("nan"),
               total_time=float(sum(r["time_s"] for r in rows.values())),
               n_fail=sum(1 for r in rows.values() if not r["ok"]),
               n_cases=len(rows))
    return agg, rows


def _load_ledger():
    if os.path.exists(LEDGER):
        return json.load(open(LEDGER))
    return {"champion": None, "history": []}


def _save_ledger(L):
    json.dump(L, open(LEDGER, "w"), indent=1)


def gate(champ, chal, maxdrop=0.03):
    """Both-axes ratchet. Returns (accept: bool, reason: str)."""
    if champ is None:
        return True, "no champion yet"
    acc_ok = chal["agg"]["mean_corr"] >= champ["agg"]["mean_corr"] - 1e-9
    spd_ok = chal["agg"]["total_time"] <= champ["agg"]["total_time"] + 1e-9
    strict = (chal["agg"]["mean_corr"] > champ["agg"]["mean_corr"] + 1e-6) or \
             (chal["agg"]["total_time"] < champ["agg"]["total_time"] - 1e-6)
    # per-case regression guard (stay leader everywhere)
    worst = 0.0
    for k, r in chal["rows"].items():
        cr = champ["rows"].get(k)
        if cr and r["ok"] and cr["ok"] and np.isfinite(r["corr"]) and np.isfinite(cr["corr"]):
            worst = max(worst, cr["corr"] - r["corr"])
    if not acc_ok:
        return False, f"accuracy regressed ({chal['agg']['mean_corr']:.4f} < {champ['agg']['mean_corr']:.4f})"
    if not spd_ok:
        return False, f"slower ({chal['agg']['total_time']:.2f}s > {champ['agg']['total_time']:.2f}s)"
    if worst > maxdrop:
        return False, f"a case regressed by {worst:.3f} (> {maxdrop})"
    if not strict:
        return False, "no strict improvement on either axis"
    return True, "PARETO-IMPROVES both axes"


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
    ap.add_argument("--maxdrop", type=float, default=0.03)
    a = ap.parse_args()

    if a.ledger:
        L = _load_ledger(); ch = L["champion"]
        if ch:
            print(f"CHAMPION: {ch['name']}  mean_corr={ch['agg']['mean_corr']:.4f} "
                  f"min_corr={ch['agg']['min_corr']:.4f} total_time={ch['agg']['total_time']:.2f}s")
        print(f"history: {len(L['history'])} entries")
        for h in L["history"][-12:]:
            print(f"  {h['name']:22s} corr={h['agg']['mean_corr']:.4f} "
                  f"time={h['agg']['total_time']:.2f}s -> {h.get('verdict','')}")
        return

    if not a.eval:
        print("need --eval module:fn"); return
    fn = _load_fn(a.eval)
    t0 = time.perf_counter()
    agg, rows = eval_config(fn)
    print(f"[{a.name}] mean_corr={agg['mean_corr']:.4f} min_corr={agg['min_corr']:.4f} "
          f"total_time={agg['total_time']:.2f}s fails={agg['n_fail']}/{agg['n_cases']} "
          f"(eval wall {time.perf_counter()-t0:.1f}s)")
    # per-group breakdown
    for grp in ("1d", "2d", "panel", "real-vldb", "real-point"):
        gc = [r["corr"] for r in rows.values() if r["group"] == grp and r["ok"] and np.isfinite(r["corr"])]
        if gc:
            print(f"    {grp:11s} mean_corr={np.mean(gc):.4f}  ({len(gc)} cases)")
    entry = {"name": a.name, "agg": agg, "rows": rows}

    if a.set_champion:
        L = _load_ledger(); L["champion"] = entry
        L["history"].append({**entry, "verdict": "set-champion"}); _save_ledger(L)
        print("  -> set as champion")
    elif a.challenge:
        L = _load_ledger()
        ok, reason = gate(L["champion"], entry, a.maxdrop)
        print(f"  GATE: {'ACCEPT' if ok else 'REJECT'} — {reason}")
        entry_log = {"name": a.name, "agg": agg, "verdict": ("ACCEPT" if ok else "REJECT") + ": " + reason}
        L["history"].append(entry_log)
        if ok:
            L["champion"] = entry
        _save_ledger(L)


if __name__ == "__main__":
    main()
