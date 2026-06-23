"""E6 — Calibrate the structural router threshold.

Sample many random panel shapes; for each, measure joint-panel vs per-entity MAE (avg
over seeds) and label the true winner. Then test how well a SHAPE-ONLY score separates
the two, and pick the threshold that maximizes agreement. Shape-only => causal-clean
(no peeking at held-out error to choose the engine).

Candidate score (high => per-entity favored, low => joint favored):
    s = log( (1-rate)*T * F / E )
  -- long observed history per series ((1-rate)*T) and many features (F) reward a rich
     per-entity model; many entities (E) reward pooling/borrowing (joint).
"""
import sys
import numpy as np
sys.path.insert(0, "src")
import cafe
from e1_does_tensor_help import make_tensor, mask_mcar, mae, run_panel, run_perentity


def score(E, T, F, rate):
    return np.log((1.0 - rate) * T * F / E)


def main():
    rng = np.random.default_rng(123)
    samples = []
    for _ in range(45):
        E = int(rng.integers(4, 60))
        T = int(rng.integers(30, 200))
        F = int(rng.integers(3, 40))
        R = int(rng.integers(1, 6))
        rate = float(rng.uniform(0.1, 0.6))
        pm, qm = [], []
        for s in range(4):
            X = make_tensor(E, T, F, R, s, 1.0)
            Xm, m = mask_mcar(X, rate, s)
            pm.append(mae(X, run_panel(Xm), m))
            qm.append(mae(X, run_perentity(Xm), m))
        pj, pe = np.mean(pm), np.mean(qm)
        winner = "joint" if pj < pe else "per_entity"
        samples.append((E, T, F, rate, score(E, T, F, rate), winner, pj, pe))

    scores = np.array([s[4] for s in samples])
    is_pe = np.array([s[5] == "per_entity" for s in samples])
    # pick threshold maximizing accuracy of rule "s >= thr -> per_entity"
    cands = np.unique(scores)
    best_thr, best_acc = None, -1
    for thr in cands:
        pred_pe = scores >= thr
        acc = np.mean(pred_pe == is_pe)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"n={len(samples)}  per_entity_true={is_pe.sum()}  joint_true={(~is_pe).sum()}")
    print(f"best threshold s* = {best_thr:.3f}  -> rule 's>=s*: per_entity' acc = {best_acc:.0%}")
    # how costly are the disagreements? (regret = MAE lost by following the rule)
    regret = []
    for (E, T, F, rate, s, w, pj, pe) in samples:
        pick_pe = s >= best_thr
        chosen = pe if pick_pe else pj
        best = min(pj, pe)
        regret.append((chosen - best) / best)
    print(f"mean routing regret vs oracle = {100*np.mean(regret):.2f}%  "
          f"(max {100*np.max(regret):.1f}%)")
    print(f"\n{'E':>4}{'T':>5}{'F':>4}{'rate':>6}{'score':>8}  {'true':>10} {'pick':>10}  ok")
    for (E, T, F, rate, s, w, pj, pe) in sorted(samples, key=lambda x: x[4]):
        pick = "per_entity" if s >= best_thr else "joint"
        print(f"{E:>4}{T:>5}{F:>4}{rate:>6.2f}{s:>8.2f}  {w:>10} {pick:>10}  {'.' if pick==w else 'X'}")


if __name__ == "__main__":
    main()
