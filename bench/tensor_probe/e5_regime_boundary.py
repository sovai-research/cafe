"""E5 — Where does joint-panel stop helping and start hurting?

E1 showed panel WINS under heavy missing / weak shared signal, but LOSES badly in the
'wide_long' regime (T=120, F=20, R=5). Pin down which axis flips it, so a *structural*
router (no value-peeking -> causal-clean) can decide joint-vs-per-entity from data shape
alone. We sweep one axis at a time from the base config.
"""
import sys
import numpy as np
sys.path.insert(0, "src")
import cafe
from e1_does_tensor_help import make_tensor, mask_mcar, mae, run_panel, run_perentity


def sweep(name, base, axis, values, seeds=range(6)):
    print(f"\n-- vary {axis} (others at base) --")
    print(f"{axis:>8} | {'panel':>8} | {'perent':>8} | panel adv | winner")
    for v in values:
        cfg = dict(base); cfg[axis] = v
        pm, qm = [], []
        for s in seeds:
            X = make_tensor(cfg["E"], cfg["T"], cfg["F"], cfg["R"], s, cfg["shared"])
            Xm, m = mask_mcar(X, cfg["rate"], s)
            pm.append(mae(X, run_panel(Xm), m))
            qm.append(mae(X, run_perentity(Xm), m))
        pm, qm = np.array(pm), np.array(qm)
        adv = 100 * (qm - pm) / qm
        win = "PANEL" if pm.mean() < qm.mean() else "perent"
        print(f"{v:>8} | {pm.mean():.4f} | {qm.mean():.4f} | {adv.mean():+6.1f}% | {win}")


def main():
    base = dict(E=12, T=60, F=8, R=3, rate=0.2, shared=1.0)
    sweep("T", base, "T", [40, 60, 90, 120, 180])
    sweep("F", base, "F", [4, 8, 16, 24, 40])
    sweep("R", base, "R", [1, 2, 3, 5, 8])
    sweep("rate", base, "rate", [0.1, 0.2, 0.35, 0.5, 0.65])
    sweep("E", base, "E", [4, 8, 16, 32, 64])


if __name__ == "__main__":
    main()
