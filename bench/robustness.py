"""
ROBUSTNESS / INVARIANTS contract — "nothing catches us by surprise."

Separate from the arena (which measures accuracy on realistic data). This throws a
large battery of EDGE-CASE inputs at an imputer and asserts hard invariants:
  I1 no exception                         I4 output finite everywhere (no NaN/Inf)
  I2 output shape == input shape          I5 observed cells not wildly corrupted
  I3 returns an ndarray                   I6 finishes under a time bound

Covers: scales (1x1 .. large, wide, tall), missingness (0%, 100%, all-NaN col/row,
single obs/miss), values (constant, zeros, huge/tiny, negative, integer, with Inf),
and 1D/2D/3D shapes. Every champion MUST pass this, not just score well.

  python3 robustness.py --eval c_auto:online_impute
  python3 robustness.py            # tests a default set of current models
"""
import os, sys, time, argparse, signal
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np


def _panel_meta(E, T):
    eids = np.repeat(np.arange(E), T); tids = np.tile(np.arange(T), E)
    return {"entity_ids": eids, "time_ids": tids, "E": E, "T": T}


def cases():
    rng = np.random.default_rng(0)
    C = {}
    def add(name, X, meta=None):
        C[name] = (np.asarray(X, float), meta or {})

    # --- scales / shapes ---
    add("1x1_obs", [[1.0]])
    add("1x1_nan", [[np.nan]])
    add("2x2", rng.standard_normal((2, 2)))
    add("single_row_1xN", rng.standard_normal((1, 8)))
    add("single_col_Tx1", rng.standard_normal((50, 1)))
    add("tiny_3x3_somenan", [[1, np.nan, 3], [np.nan, 5, 6], [7, 8, np.nan]])
    add("wide_NggT", rng.standard_normal((6, 200)))
    add("tall_TggN", rng.standard_normal((2000, 3)))

    # --- missingness extremes ---
    base = rng.standard_normal((40, 6))
    add("miss_0pct", base.copy())
    alln = base.copy(); alln[:] = np.nan
    add("miss_100pct_allnan", alln)
    b2 = base.copy(); b2[:, 2] = np.nan
    add("whole_col_nan", b2)
    b3 = base.copy(); b3[10, :] = np.nan
    add("whole_row_nan", b3)
    b4 = base.copy(); m = rng.random(b4.shape) < 0.97; b4[m] = np.nan
    add("miss_97pct", b4)
    b5 = base.copy(); b5[:] = np.nan; b5[0, 0] = 1.0
    add("single_obs_only", b5)
    b6 = base.copy(); b6[3, 4] = np.nan
    add("single_missing_cell", b6)

    # --- pathological values ---
    add("constant_col", np.where(np.arange(30)[:, None] % 1 == 0,
                                  np.tile([5.0, np.nan, 5.0, 5.0, 5.0], (30, 1)), 0))
    z = np.zeros((30, 5)); z[rng.random(z.shape) < 0.3] = np.nan
    add("all_zeros_somenan", z)
    huge = rng.standard_normal((30, 5)) * 1e10; huge[rng.random(huge.shape) < 0.3] = np.nan
    add("huge_magnitude", huge)
    tiny = rng.standard_normal((30, 5)) * 1e-10; tiny[rng.random(tiny.shape) < 0.3] = np.nan
    add("tiny_magnitude", tiny)
    mixed = rng.standard_normal((40, 4)) * np.array([1e6, 1.0, 1e-3, 1e3])
    mixed[rng.random(mixed.shape) < 0.3] = np.nan
    add("mixed_scales", mixed)
    intg = rng.integers(0, 10, (40, 5)).astype(float); intg[rng.random(intg.shape) < 0.3] = np.nan
    add("integer_valued", intg)
    inf = rng.standard_normal((30, 5)); inf[0, 0] = np.inf; inf[1, 1] = -np.inf
    inf[rng.random(inf.shape) < 0.2] = np.nan
    add("contains_inf", inf)

    # --- 1D ---
    s = rng.standard_normal((100, 1)); s[rng.random(s.shape) < 0.3] = np.nan
    add("1d_series", s)

    # --- 3D / panel ---
    E, T, F = 4, 12, 3
    p = rng.standard_normal((E * T, F)); p[rng.random(p.shape) < 0.3] = np.nan
    add("panel_small", p, _panel_meta(E, T))
    add("panel_single_entity", rng.standard_normal((1 * 10, 3)), _panel_meta(1, 10))
    add("panel_single_time", rng.standard_normal((5 * 1, 3)), _panel_meta(5, 1))
    pb = rng.standard_normal((E * T, F)); pb[5:9, :] = np.nan      # blackout span
    add("panel_blackout", pb, _panel_meta(E, T))
    return C


def check(fn, X, meta, time_budget=15.0):
    Xin = X.copy()
    obs = np.isfinite(Xin)
    class TO(Exception): pass
    def _h(s, f): raise TO()
    old = signal.signal(signal.SIGALRM, _h); signal.setitimer(signal.ITIMER_REAL, time_budget)
    t0 = time.perf_counter()
    try:
        out = fn(X.copy(), dict(meta))
    except TO:
        return False, f"TIMEOUT >{time_budget}s"
    except Exception as e:
        return False, f"CRASH {type(e).__name__}: {str(e)[:60]}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    dt = time.perf_counter() - t0
    try:
        out = np.asarray(out, float)
    except Exception:
        return False, "output not array-like"
    if out.shape != Xin.shape:
        return False, f"shape {out.shape} != {Xin.shape}"
    if not np.all(np.isfinite(out)):
        n = int((~np.isfinite(out)).sum())
        return False, f"{n} non-finite output cells"
    # observed cells (finite & not inf in input) should be ~preserved or at least finite
    finite_obs = obs & np.isfinite(Xin)
    if finite_obs.any():
        dev = np.abs(out[finite_obs] - Xin[finite_obs])
        scale = np.abs(Xin[finite_obs]).mean() + 1e-6
        if np.nanmax(dev) > 1e3 * scale + 1e3:
            return False, "observed cells grossly corrupted"
    return True, f"ok ({dt*1000:.0f}ms)"


def run(fn, name):
    C = cases(); npass = 0; fails = []
    for cn, (X, meta) in C.items():
        ok, msg = check(fn, X, meta)
        if ok:
            npass += 1
        else:
            fails.append((cn, msg))
        print(f"  {'PASS' if ok else 'FAIL'}  {cn:24s} {msg}")
    print(f"\n[{name}] {npass}/{len(C)} invariants passed; {len(fails)} FAILS")
    return npass, len(C), fails


def _load(spec):
    mod, _, attr = spec.partition(":"); m = __import__(mod)
    return getattr(m, attr[:-2])() if attr.endswith("()") else getattr(m, attr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="")
    a = ap.parse_args()
    targets = [a.eval] if a.eval else ["c_auto:online_impute", "c_router:online_impute",
                                       "c_online_trmf:online_impute", "c_fe_lowrank:online_impute"]
    for t in targets:
        print(f"\n===== {t} =====")
        try:
            run(_load(t), t)
        except Exception as e:
            print(f"  could not load {t}: {e}")
