"""E7 — Can a causal-clean adaptive blend beat picking one engine, across BOTH DGPs?

Shape-only routing fails because the CP-tensor DGP (per-entity wins) and the FE-panel DGP
(joint wins) overlap in (E,T,F,rate). Instead of choosing globally (which also leaks look-
ahead), blend the two engines PER CELL with a weight set by each cell's own data
availability up to t -- truncation-invariant, so the causal guarantee survives:

    effective_obs[e,t,f] = (#observed (e,f) before t) * 1/(1+gap_since_last_obs)
    w_per = effective_obs / (effective_obs + TAU)          # high -> trust per-entity
    blend = w_per * per_entity + (1 - w_per) * joint

A data-rich cell (long warm history, short gap) leans per-entity; a cold-start / long-
blackout cell leans joint cross-section. We measure MAE for joint, per_entity, the oracle
pick (min of the two), and the blend -- across CP-tensors, FE-panels, and the exact
benchmark panel shapes -- and report regret vs the oracle. A good blend has ~0 regret on
both DGPs AND never loses to joint on the FE/benchmark panels (no champion regression).
"""
import sys
import numpy as np
sys.path.insert(0, "src")
sys.path.insert(0, "bench")
import cafe
from cafe.model import _impute_per_entity, _impute_panel
import harness
from e1_does_tensor_help import make_tensor, mask_mcar, mae


def joint(X, meta):
    return _impute_panel(X, meta, engine="joint")


def per_entity(X, meta):
    return _impute_per_entity(X, meta)


def blend_weights(X, meta, tau):
    """Per-cell w_per in [0,1] from availability up to t. Truncation-invariant."""
    X = np.asarray(X, float)
    eids = np.asarray(meta["entity_ids"]); tids = np.asarray(meta["time_ids"])
    R, F = X.shape
    w = np.zeros((R, F))
    for e in np.unique(eids):
        idx = np.where(eids == e)[0]
        order = idx[np.argsort(tids[idx], kind="stable")]
        cnt = np.zeros(F)                 # observed count of (e,f) strictly before t
        last_t = np.full(F, -1.0)         # time of last observation
        for r in order:
            t = float(tids[r])
            gap = np.where(last_t >= 0, t - last_t, 1e6)
            recency = 1.0 / (1.0 + gap)
            eff = cnt * recency
            w[r] = eff / (eff + tau)
            obs = ~np.isnan(X[r])
            cnt[obs] += 1.0
            last_t[obs] = t
    return w


def blend(X, meta, J, P, tau):
    w = blend_weights(X, meta, tau)
    out = w * P + (1.0 - w) * J
    # keep observed cells exact (both engines already do; guard anyway)
    obs = ~np.isnan(np.asarray(X, float))
    out[obs] = np.asarray(X, float)[obs]
    return out


def stack3d(Xm):
    E, T, F = Xm.shape
    ent = np.repeat(np.arange(E), T); tim = np.tile(np.arange(T), E)
    return Xm.reshape(E * T, F), {"entity_ids": ent, "time_ids": tim}


def cp_case(E, T, F, R, rate, seed):
    X = make_tensor(E, T, F, R, seed, 1.0)
    Xm, m = mask_mcar(X, rate, seed)
    Xs, meta = stack3d(Xm)
    truth = X.reshape(E * T, F); ms = m.reshape(E * T, F)
    return Xs, meta, truth, ms


def fe_case(E, T, F, rate, seed):
    X, meta = harness.gen_panel(E=E, T=T, F=F, seed=seed)
    rng = np.random.default_rng(seed + 7)
    ms = rng.random(X.shape) < rate
    Xm = X.copy(); Xm[ms] = np.nan
    return Xm, meta, X, ms


def evaluate(name, cases, tau):
    rows = []
    for (Xs, meta, truth, ms) in cases:
        J = joint(Xs, meta); P = per_entity(Xs, meta)
        B = blend(Xs, meta, J, P, tau)
        mj = mae(truth, J, ms); mp = mae(truth, P, ms); mb = mae(truth, B, ms)
        rows.append((mj, mp, mb))
    rows = np.array(rows)
    mj, mp, mb = rows[:, 0].mean(), rows[:, 1].mean(), rows[:, 2].mean()
    oracle = np.minimum(rows[:, 0], rows[:, 1]).mean()
    reg_blend = 100 * (rows[:, 2] - np.minimum(rows[:, 0], rows[:, 1])) / np.minimum(rows[:, 0], rows[:, 1])
    reg_joint = 100 * (rows[:, 0] - np.minimum(rows[:, 0], rows[:, 1])) / np.minimum(rows[:, 0], rows[:, 1])
    print(f"{name:>16} | joint {mj:.4f} | perent {mp:.4f} | oracle {oracle:.4f} | "
          f"BLEND {mb:.4f} | blend-regret {reg_blend.mean():+.2f}% | joint-regret {reg_joint.mean():+.2f}%")
    return reg_blend.mean(), reg_joint.mean()


def main():
    rng = np.random.default_rng(7)
    cp_cases, fe_cases = [], []
    for s in range(6):
        cp_cases.append(cp_case(20, 70, 12, 3, 0.25, s))      # per-entity-favored region
        cp_cases.append(cp_case(12, 120, 20, 5, 0.20, s))     # wide_long (per-entity strong)
        fe_cases.append(fe_case(20, 80, 10, 0.20, s))         # benchmark panel_mcar20 shape
        fe_cases.append(fe_case(20, 80, 10, 0.40, s))         # benchmark panel_mar40 shape

    print("TAU sweep (blend regret vs oracle; want ~0 on both, and blend<=joint on FE):")
    for tau in [1.0, 2.0, 4.0, 8.0, 16.0]:
        print(f"\n-- tau={tau} --")
        rb_cp, rj_cp = evaluate("CP-tensor", cp_cases, tau)
        rb_fe, rj_fe = evaluate("FE-panel", fe_cases, tau)
        print(f"   summary tau={tau}: blend mean regret CP {rb_cp:+.2f}% / FE {rb_fe:+.2f}%  "
              f"(joint would regret CP {rj_cp:+.2f}% / FE {rj_fe:+.2f}%)")


if __name__ == "__main__":
    main()
