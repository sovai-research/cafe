"""
Fast active-loop driver. Evaluates the champion + every challenger file in FAST
mode and applies the both-axes ratchet — the whole iteration in a few seconds,
no agents. Competitors are NEVER run (we compare to published numbers offline);
this only times OUR configs.

  TIMARA_FAST=1 python3 loop.py                 # score all c_chal_*.py vs champion
  TIMARA_FAST=1 python3 loop.py --adopt         # also promote the best valid winner
  TIMARA_FAST=1 python3 loop.py --only c_chal_robust,c_chal_speed
"""
import os, sys, glob, time, argparse
os.environ.setdefault("TIMARA_FAST", "1")
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import arena


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adopt", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--maxdrop", type=float, default=0.03)
    a = ap.parse_args()

    L = arena._load_ledger()
    champ = L["champion"]
    cbar = (f"champion={champ['name']} corr={champ['agg']['mean_corr']:.4f} "
            f"time={champ['agg']['total_time']:.2f}s" if champ else "no champion")
    print(f"[FAST suite, {len(arena.SUITE)} cases] {cbar}\n" + "-" * 78)

    only = set(s.strip() for s in a.only.split(",") if s.strip())
    files = sorted(glob.glob(os.path.join(HERE, "c_chal_*.py")))
    mods = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if only:
        mods = [m for m in mods if m in only]

    results = []
    for mod in mods:
        try:
            fn = arena._load_fn(f"{mod}:online_impute")
        except Exception as e:                              # noqa: BLE001
            print(f"  {mod:24s} IMPORT-FAIL {type(e).__name__}: {e}"); continue
        t0 = time.perf_counter()
        try:
            agg, rows = arena.eval_config(fn)
        except Exception as e:                              # noqa: BLE001
            print(f"  {mod:24s} EVAL-FAIL {type(e).__name__}: {e}"); continue
        entry = {"name": mod, "agg": agg, "rows": rows}
        ok, reason = arena.gate(champ, entry, a.maxdrop)
        results.append((entry, ok, reason))
        dc = agg["mean_corr"] - (champ["agg"]["mean_corr"] if champ else 0)
        dt = agg["total_time"] - (champ["agg"]["total_time"] if champ else 0)
        print(f"  {mod:24s} corr={agg['mean_corr']:.4f}({dc:+.4f}) "
              f"time={agg['total_time']:.2f}s({dt:+.2f}) min={agg['min_corr']:.3f} "
              f"-> {'WIN' if ok else 'no'}: {reason}")

    winners = sorted([r for r in results if r[1]], key=lambda r: -r[0]["agg"]["mean_corr"])
    if a.adopt and winners and champ is not None:
        best = winners[0][0]
        L["champion"] = best
        L["history"].append({"name": best["name"], "agg": best["agg"], "verdict": "ADOPTED"})
        arena._save_ledger(L)
        print(f"\nADOPTED new champion: {best['name']} "
              f"(corr {best['agg']['mean_corr']:.4f}, time {best['agg']['total_time']:.2f}s)")
    elif not winners:
        print("\nNo challenger cleared the ratchet -> iteration FAILS -> trigger literature research.")


if __name__ == "__main__":
    main()
