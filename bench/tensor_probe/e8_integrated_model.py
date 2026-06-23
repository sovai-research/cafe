"""E8 — One integrated model: common (shared) + idiosyncratic (per-entity), no router.

    integrated = joint_recon  +  per_entity( X - joint_recon )

Stage 1 is the validated joint core's structural fit (entity_FE + time_FE + shared low-
rank) -- exposed at every cell via the new recon_out hook. Stage 2 runs the per-entity 2D
core on the RESIDUAL, capturing entity-specific structure the shared stage cannot. The sum
is exact at observed cells by construction. This is stagewise additive (backfitting), one
model, not a choice.

Hypothesis:
  * FE-panel / benchmark : residual after the shared fit is ~noise -> stage 2 adds ~0 ->
    integrated ~ joint  (NO REGRESSION).
  * CP-tensor / rich     : shared fit underfits -> structured residual -> stage 2 activates
    -> integrated approaches per_entity (MOVES UP TENSOR).
"""
import sys, time
import numpy as np
sys.path.insert(0, "src")
sys.path.insert(0, "bench")
import cafe
from cafe._core import _impute_panel
from cafe.model import _impute_per_entity
import harness
from e1_does_tensor_help import make_tensor, mask_mcar, mae
from e7_adaptive_blend import cp_case, fe_case


def joint(X, meta):
    return np.asarray(_impute_panel(X, meta), float)


def per_entity(X, meta):
    return _impute_per_entity(X, meta)


def integrated(X, meta):
    ro = {}
    _impute_panel(X, meta, recon_out=ro)
    Jr = np.asarray(ro["recon"], float)
    Jr = np.where(np.isfinite(Jr), Jr, 0.0)
    resid = np.where(np.isnan(X), np.nan, X - Jr)      # observed residual; nan at missing
    Pr = _impute_per_entity(resid, meta)               # idiosyncratic stage fills the gaps
    out = np.asarray(X, float).copy()
    miss = np.isnan(out)
    out[miss] = Jr[miss] + Pr[miss]
    return out


def evaluate(name, cases):
    rows, tj, ti = [], 0.0, 0.0
    for (Xs, meta, truth, ms) in cases:
        t0 = time.time(); J = joint(Xs, meta); tj += time.time() - t0
        P = per_entity(Xs, meta)
        t0 = time.time(); I = integrated(Xs, meta); ti += time.time() - t0
        rows.append((mae(truth, J, ms), mae(truth, P, ms), mae(truth, I, ms)))
    a = np.array(rows)
    mj, mp, mi = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
    oracle = np.minimum(a[:, 0], a[:, 1]).mean()
    reg_i = 100 * (a[:, 2] - np.minimum(a[:, 0], a[:, 1])) / np.minimum(a[:, 0], a[:, 1])
    vs_joint = 100 * (a[:, 2] - a[:, 0]) / a[:, 0]
    print(f"{name:>14} | joint {mj:.4f} | perent {mp:.4f} | oracle {oracle:.4f} | "
          f"INTEGRATED {mi:.4f} | vs-joint {vs_joint.mean():+.2f}% | vs-oracle {reg_i.mean():+.2f}%")
    print(f"{'':>14} | time: joint {tj:.2f}s  integrated {ti:.2f}s ({ti/max(tj,1e-9):.1f}x)")
    return vs_joint.mean()


def main():
    cp_cases, fe_cases = [], []
    for s in range(6):
        cp_cases.append(cp_case(20, 70, 12, 3, 0.25, s))
        cp_cases.append(cp_case(12, 120, 20, 5, 0.20, s))     # rich per-entity
        fe_cases.append(fe_case(20, 80, 10, 0.20, s))         # benchmark panel_mcar20
        fe_cases.append(fe_case(20, 80, 10, 0.40, s))         # benchmark panel_mar40
    print("Integrated common+idiosyncratic vs the two single engines:\n")
    vj_cp = evaluate("CP-tensor", cp_cases)
    vj_fe = evaluate("FE-panel", fe_cases)
    print(f"\nVERDICT: integrated vs joint -> CP {vj_cp:+.1f}% (want big negative = recovers "
          f"tensor win), FE {vj_fe:+.1f}% (want ~0 = no regression)")


if __name__ == "__main__":
    main()
