"""E8b — Integrated model with a SHRUNK idiosyncratic stage.

    integrated = joint_recon  +  gamma * per_entity( X - joint_recon )

gamma in [0,1] gates the idiosyncratic stage by how PREDICTABLE the observed residual is.
Real entity-specific structure (CP idiosyncratic factors are AR) has temporal
autocorrelation; a pure-noise residual (FE panel, fully explained by the shared stage)
does not. So gamma = clip(lag-1 autocorrelation of the observed residual, 0, 1), per
entity (or per entity-feature). White residual -> gamma~0 -> integrated~joint (no
regression). Structured residual -> gamma~1 -> recovers the per-entity tensor win.

gamma uses only observed residual values, and (in this prototype) a global autocorr; the
shippable version computes it online/expanding so it stays truncation-invariant.
"""
import sys
import numpy as np
sys.path.insert(0, "src")
sys.path.insert(0, "bench")
from cafe._core import _impute_panel
from cafe.model import _impute_per_entity
from e1_does_tensor_help import mae
from e7_adaptive_blend import cp_case, fe_case


def _lag1_autocorr(vals):
    if vals.size < 4:
        return 0.0
    a, b = vals[:-1], vals[1:]
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def _gamma(X, resid, meta, per_feature):
    """Per-entity (or per entity-feature) shrinkage from observed-residual autocorrelation."""
    eids = np.asarray(meta["entity_ids"]); tids = np.asarray(meta["time_ids"])
    R, F = X.shape
    g = np.zeros((R, F))
    for e in np.unique(eids):
        idx = np.where(eids == e)[0]
        order = idx[np.argsort(tids[idx], kind="stable")]
        if per_feature:
            for f in range(F):
                col = resid[order, f]
                obs = col[~np.isnan(col)]
                gv = max(0.0, _lag1_autocorr(obs))
                g[order, f] = gv
        else:
            acs = []
            for f in range(F):
                col = resid[order, f]; obs = col[~np.isnan(col)]
                acs.append(max(0.0, _lag1_autocorr(obs)))
            g[np.ix_(order, np.arange(F))] = float(np.mean(acs))
    return g


def integrated(X, meta, mode="entity"):
    ro = {}
    _impute_panel(X, meta, recon_out=ro)
    Jr = np.where(np.isfinite(ro["recon"]), ro["recon"], 0.0)
    resid = np.where(np.isnan(X), np.nan, X - Jr)
    Pr = _impute_per_entity(resid, meta)
    out = np.asarray(X, float).copy(); miss = np.isnan(out)
    if mode == "none":
        out[miss] = Jr[miss] + Pr[miss]
        return out
    per_feature = mode in ("feature", "feature_sq", "feature_thr")
    gamma = _gamma(X, resid, meta, per_feature=per_feature)
    if mode in ("feature_sq", "entity_sq"):
        gamma = gamma ** 2                      # sharpen: kills weak (noise) autocorr
    elif mode in ("feature_thr", "entity_thr"):
        gamma = np.where(gamma >= 0.3, gamma, 0.0)   # hard gate on real structure
    out[miss] = Jr[miss] + gamma[miss] * Pr[miss]
    return out


def joint(X, meta):
    return np.asarray(_impute_panel(X, meta), float)


def evaluate(name, cases, mode):
    a = []
    for (Xs, meta, truth, ms) in cases:
        J = joint(Xs, meta); I = integrated(Xs, meta, mode)
        a.append((mae(truth, J, ms), mae(truth, I, ms)))
    a = np.array(a)
    vs = 100 * (a[:, 1] - a[:, 0]) / a[:, 0]
    print(f"  {name:>12} | joint {a[:,0].mean():.4f} | integrated {a[:,1].mean():.4f} "
          f"| vs-joint {vs.mean():+.2f}%")
    return vs.mean()


def main():
    cp_cases, fe_cases = [], []
    for s in range(6):
        cp_cases.append(cp_case(20, 70, 12, 3, 0.25, s))
        cp_cases.append(cp_case(12, 120, 20, 5, 0.20, s))
        fe_cases.append(fe_case(20, 80, 10, 0.20, s))
        fe_cases.append(fe_case(20, 80, 10, 0.40, s))
    for mode in ("none", "feature", "feature_sq", "feature_thr", "entity_thr"):
        print(f"\n== gamma mode: {mode} ==")
        vc = evaluate("CP-tensor", cp_cases, mode)
        vf = evaluate("FE-panel", fe_cases, mode)
        print(f"  -> CP {vc:+.1f}% (want negative), FE {vf:+.1f}% (want ~0)")


if __name__ == "__main__":
    main()
