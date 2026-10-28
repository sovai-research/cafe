"""The foundational experiment for CAFE-CD: does factor residualisation help causal discovery?

QUESTION
    A causal-discovery routine run on a high-dimensional panel sees pervasive common
    co-movement that is NOT causal structure -- it is a factor.  Does subtracting CAFE's
    free, point-in-time factor estimate recover the true lagged causal graph better than
    (a) running on the raw data, and (b) running on a plain batch-PCA residual?  And does
    any advantage GROW with the number of variables P?

    This is the gate the whole CAFE-CD programme rests on.  "Factor-adjust then sparse-VAR"
    is prior art for single time series (FNETS, Barigozzi/Cho/Owens JBES 2024; Krampe OBES
    2025) -- so if CAFE's residual merely matches a batch PCA residual, the contribution
    collapses to that literature and we must reposition onto the point-in-time and
    missingness axes instead.  We would rather learn that in week 2 than month 6.

DESIGN
    Ground truth from bench/gen_causal_panel.py: latent AR factors + a KNOWN sparse lagged
    SVAR on the idiosyncratic component, variance-equalised so the generator does not leak
    the causal order through scale (var-/R2-sortability audited in-band).

    Six arms, ALL fed to the SAME deliberately dumb downstream estimator (pooled OLS-VAR
    with pairwise-complete moments and |t| scores), so the controlled variable is the
    REPRESENTATION and nothing else:

      raw          observed X                      -- the null hypothesis
      cafe         res.residuals(factor_adjusted=True)
      cafe_contam  res.residuals(factor_adjusted=False)  -- isolates the lr contamination
      pca          batch PCA residual, rank by Ahn-Horenstein  [NON-CAUSAL: uses all rows]
      oracle       truth.E_idio                     -- the ceiling
      pca_oracle_r batch PCA at the TRUE rank R     -- removes rank-selection as a factor

    Every arm receives the identical two-way (entity, time) FWL within-transform.  Skipping
    it on `raw` would rig the comparison in CAFE's favour, because CAFE removes mu and
    time_fe internally and raw data does not.

    PRIMARY METRIC is auc_cross -- the OFF-DIAGONAL lagged edges.  The lag-1 diagonal is
    dense (every node has an AR self-loop by construction) and trivially detectable, so the
    all-cells `auc` flatters any detector that only fires there.

WHAT IT WRITES
    bench/causal_residualization_cache.json      the cache (this file writes
                                                 ONLY this; the chi-crossover
                                                 table it used to feed was
                                                 superseded by
                                                 exp_causal_sparse_gate.py)

Run:
    python3 bench/exp_causal_residualization.py
    python3 bench/exp_causal_residualization.py --tables-only
Env knobs:
    CR_PS=6,10,16,24,40   CR_SEEDS=5   CR_E=20   CR_T=200   CR_CHI=0.6   CR_MISS=0,0.2
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import cafe                                    # noqa: E402
import causal_metrics as cm                    # noqa: E402
from gen_causal_panel import gen_causal_panel, mask_panel   # noqa: E402
from sortability import var_sortability, r2_sortability     # noqa: E402

CACHE = os.path.join(_HERE, "causal_residualization_cache.json")
FIGURE = os.path.join(_ROOT, "paper", "figures", "causal_residualization.pdf")

PS = [int(x) for x in os.environ.get("CR_PS", "6,10,16,24,40").split(",")]
SEEDS = int(os.environ.get("CR_SEEDS", "5"))
N_ENT = int(os.environ.get("CR_E", "20"))
T_LEN = int(os.environ.get("CR_T", "200"))
CHI = float(os.environ.get("CR_CHI", "0.6"))
MISS = [float(x) for x in os.environ.get("CR_MISS", "0,0.2").split(",")]
R_TRUE = 3
K_TRUE = 2
DEG = 2

ARMS = ["raw", "cafe", "cafe_contam", "pca", "pca_oracle_r", "oracle"]
NONCAUSAL = {"pca", "pca_oracle_r"}          # use the whole sample; must be labelled


# --------------------------------------------------------------------------- #
# representation helpers
# --------------------------------------------------------------------------- #
def fwl_demean(E, obs):
    """Exact two-way (entity, time) within-transform on (n_ent, T, P), NaN-aware.

    Applied to EVERY arm.  CAFE removes mu and time_fe internally; raw data does not, so
    omitting this on `raw` would hand CAFE a free win that is an artefact of preprocessing.
    """
    Z = np.where(obs, E, np.nan)
    with np.errstate(invalid="ignore"):
        m_ent = np.nanmean(Z, axis=1, keepdims=True)      # per (entity, var)
        m_tim = np.nanmean(Z, axis=0, keepdims=True)      # per (time, var)
        m_all = np.nanmean(Z, axis=(0, 1), keepdims=True)
    m_ent = np.nan_to_num(m_ent)
    m_tim = np.nan_to_num(m_tim)
    m_all = np.nan_to_num(m_all)
    D = Z - m_ent - m_tim + m_all
    sd = np.nanstd(D.reshape(-1, D.shape[2]), axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return D / sd


def ahn_horenstein_rank(X2d, kmax=8):
    """Eigenvalue-ratio rank estimator (Ahn & Horenstein 2013).  Not oracle-fed."""
    n, p = X2d.shape
    S = X2d.T @ X2d / max(n, 1)
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    ev = np.maximum(ev, 1e-12)
    kmax = int(min(kmax, len(ev) - 1))
    if kmax < 1:
        return 1
    ratios = ev[:kmax] / ev[1:kmax + 1]
    return int(np.argmax(ratios) + 1)


def pca_residual(X3d, obs, rank=None):
    """Batch PCA residual.  NON-CAUSAL: the SVD sees every row, including the future."""
    n_ent, T, P = X3d.shape
    Z = np.where(obs, X3d, np.nan).reshape(-1, P)
    col = np.nanmean(Z, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    Zf = np.where(np.isfinite(Z), Z, col)                 # column-mean pre-imputation
    Zc = Zf - Zf.mean(axis=0, keepdims=True)
    if rank is None:
        rank = ahn_horenstein_rank(Zc)
    rank = int(max(1, min(rank, P - 1)))
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    fit = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    return (Zc - fit).reshape(n_ent, T, P), rank


# --------------------------------------------------------------------------- #
# the fixed downstream estimator -- identical for every arm
# --------------------------------------------------------------------------- #
def pooled_var_scores(E, obs, K):
    """Pooled lag-1..K OLS-VAR on (n_ent, T, P); returns |t| as (K, P, P)[k-1, i, j].

    Moments are PAIRWISE-COMPLETE over observed cells.  Complete-case deletion is not an
    option: at P=40, K=2 and 20% MCAR the design has 80 columns, so the retained fraction
    would be ~0.8**81 ~ 0.  Zero-filling without renormalising by each pair's own valid
    count would instead shrink every coefficient by the co-observation rate.
    """
    n_ent, T, P = E.shape
    ok = obs & np.isfinite(E)
    Z = np.where(ok, E, 0.0)
    V = ok.astype(float)

    ys, vys, xs, vxs = [], [], [], []
    for e in range(n_ent):
        ys.append(Z[e, K:T, :]);   vys.append(V[e, K:T, :])
        xs.append(np.concatenate([Z[e, K - k:T - k, :] for k in range(1, K + 1)], axis=1))
        vxs.append(np.concatenate([V[e, K - k:T - k, :] for k in range(1, K + 1)], axis=1))
    Y = np.concatenate(ys, 0); VY = np.concatenate(vys, 0)
    X = np.concatenate(xs, 0); VX = np.concatenate(vxs, 0)

    G = X.T @ X
    NG = np.maximum(VX.T @ VX, 1.0)
    G = G / NG
    C = X.T @ Y
    NC = np.maximum(VX.T @ VY, 1.0)
    C = C / NC
    yy = (Y * Y).sum(0) / np.maximum(VY.sum(0), 1.0)

    d = np.sqrt(np.maximum(np.diag(G), 1e-12))
    ridge = 1e-8 * float(np.mean(np.diag(G)))
    Gr = G + ridge * np.eye(G.shape[0])
    try:
        B = np.linalg.solve(Gr, C)
        Ginv = np.linalg.solve(Gr, np.eye(Gr.shape[0]))
    except np.linalg.LinAlgError:
        B = np.linalg.lstsq(Gr, C, rcond=None)[0]
        Ginv = np.linalg.pinv(Gr)

    n_eff = float(np.median(NG)) if np.isfinite(np.median(NG)) else float(len(Y))
    sig2 = np.maximum(yy - np.einsum("ai,ai->i", C, B), 1e-12)
    se = np.sqrt(np.outer(np.maximum(np.diag(Ginv), 1e-12), sig2) / max(n_eff - 1.0, 1.0))
    Tst = np.abs(B) / np.maximum(se, 1e-12)
    del d
    return Tst.reshape(K, P, P).transpose(0, 2, 1)        # -> [k-1, i(effect), j(cause)]


# --------------------------------------------------------------------------- #
def run_cell(P, seed, miss, pattern):
    X3d, X2d, meta, truth = gen_causal_panel(
        E=N_ENT, T=T_LEN, P=P, R=R_TRUE, n_lags=K_TRUE, deg=DEG,
        chi=CHI, noise="gauss", equal_variance=True, seed=seed)
    obs_full = np.ones(X3d.shape, dtype=bool)
    Xobs = X3d.copy()
    if miss > 0:
        Xm = mask_panel(pattern, X3d, miss, seed=seed)
        Xobs = Xm
        obs_full = np.isfinite(Xm)

    reps, ranks = {}, {}
    reps["raw"] = (Xobs, obs_full)
    reps["oracle"] = (truth.E_idio, np.ones(X3d.shape, bool))

    t0 = time.time()
    res = cafe.CAFE().run(Xobs)
    t_cafe = time.time() - t0
    Ec, Oc = res.residuals(stack=True, factor_adjusted=True, mask_imputed=True)
    reps["cafe"] = (Ec.reshape(X3d.shape), Oc.reshape(X3d.shape))
    Ek, Ok = res.residuals(stack=True, factor_adjusted=False, mask_imputed=True)
    reps["cafe_contam"] = (Ek.reshape(X3d.shape), Ok.reshape(X3d.shape))
    eff_rank = int(np.mean([r.effective_rank() for _rows, r in res._pe.values()]))

    Ep, rk = pca_residual(Xobs, obs_full, rank=None)
    reps["pca"] = (Ep, obs_full); ranks["pca"] = rk
    Ep2, rk2 = pca_residual(Xobs, obs_full, rank=R_TRUE)
    reps["pca_oracle_r"] = (Ep2, obs_full); ranks["pca_oracle_r"] = rk2

    out = {"P": P, "seed": seed, "miss": miss, "pattern": pattern,
           "eff_rank": eff_rank, "pca_rank": rk, "t_cafe": t_cafe,
           "var_sort": float(var_sortability(truth.E_idio, truth.skeleton())),
           "r2_sort": float(r2_sortability(truth.E_idio, truth.skeleton()))}
    for arm in ARMS:
        E, O = reps[arm]
        D = fwl_demean(np.asarray(E, float), np.asarray(O, bool))
        Ofin = np.asarray(O, bool) & np.isfinite(D)
        t1 = time.time()
        S = pooled_var_scores(D, Ofin, K_TRUE)
        m = cm.summarize_graph(truth.A_lag, S, None)
        out[arm] = {"auc": float(m["auc"]), "auc_cross": float(m["auc_cross"]),
                    "lag_acc": float(m.get("lag_accuracy", np.nan)),
                    "t_est": time.time() - t1}
    return out


def main():
    rows = []
    total = len(PS) * SEEDS * sum(1 for m in MISS for _ in (["mcar", "block"] if m > 0 else ["none"]))
    i = 0
    for P in PS:
        for miss in MISS:
            pats = ["none"] if miss == 0 else ["mcar", "block"]
            for pat in pats:
                for seed in range(SEEDS):
                    i += 1
                    r = run_cell(P, seed, miss, pat)
                    rows.append(r)
                    print(f"[{i:>3}/{total}] P={P:>2} miss={miss} {pat:<5} s={seed}  "
                          + "  ".join(f"{a}={r[a]['auc_cross']:.3f}" for a in ARMS), flush=True)
    meta = {"PS": PS, "SEEDS": SEEDS, "E": N_ENT, "T": T_LEN, "chi": CHI,
            "R_true": R_TRUE, "K_true": K_TRUE, "deg": DEG, "arms": ARMS,
            "primary_metric": "auc_cross", "noncausal_arms": sorted(NONCAUSAL)}
    with open(CACHE, "w") as f:
        json.dump({"rows": rows, "meta": meta}, f, indent=1)
    print(f"\nwrote {CACHE}  ({len(rows)} cells)")
    return rows, meta


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        with open(CACHE) as f:
            d = json.load(f)
        rows, meta = d["rows"], d["meta"]
    else:
        rows, meta = main()
