"""
Real-data head-to-head on bench-vldb20 (Khayati et al., VLDB 2020) — our strongest turf.
Datasets are already z-scored. Protocol family B: TRANSDUCTIVE (full-matrix, non-causal),
contiguous block removed per column; RMSE on removed cells (z-scored scale).

We DO NOT re-run competitors — we compare to the paper's published cluster numbers.
Published RMSE @10% block (Khayati 2020, vol13 p768):
  airq ~0.33-0.34 (CDRec/DynaMMo/SoftImpute/TRMF cluster)
  chlorine ~0.15-0.20 (TRMF/SoftImpute)   temp ~0.195 (STMVL)   drift ~0.18-0.20 (CDRec)
"""
import os, sys, time
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DSETS = {"airq": "airq_normal.txt", "chlorine": "chlorine_normal.txt",
         "drift": "drift10_normal.txt", "temp": "temp_normal.txt"}
# published leading-cluster RMSE @10% block (ballpark, from the paper)
PUBLISHED = {"airq": "0.33-0.34", "chlorine": "0.15-0.20", "drift": "0.18-0.20", "temp": "~0.195"}


def block_mask(X, rate, seed=1):
    """One contiguous block of length rate*T per column at a seeded position."""
    rng = np.random.default_rng(seed); T, N = X.shape; M = np.zeros((T, N), bool)
    blen = max(1, int(rate * T))
    for j in range(N):
        s = int(rng.integers(0, max(1, T - blen)))
        M[s:s + blen, j] = True
    return M


def rmse(true, pred, m):
    e = (np.asarray(pred, float) - true)[m]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


def load(name):
    return np.loadtxt(os.path.join(DATA, DSETS[name]))


def get_methods():
    from m_softimpute import impute as softimpute
    from m_trmf import impute as trmf
    from c_oracle_ref import trmf_oracle, mcnnm_oracle
    from c_router import online_impute as router
    return [
        ("SoftImpute", softimpute, "non-causal"),
        ("TRMF", trmf, "non-causal"),
        ("TRMF-oracle", trmf_oracle, "non-causal"),
        ("MCNNM-oracle", mcnnm_oracle, "non-causal"),
        ("CausalRouter", router, "CAUSAL"),
    ]


def main(rates=(0.1, 0.2, 0.4)):
    meths = get_methods()
    for name in DSETS:
        X = load(name); T, N = X.shape
        print(f"\n=== {name} ({T}x{N})  published@10%: {PUBLISHED[name]} (transductive) ===")
        print(f"{'method':14s} {'type':11s} " + " ".join(f"{int(r*100)}%".rjust(14) for r in rates))
        for mname, fn, typ in meths:
            cells = []
            for r in rates:
                M = block_mask(X, r); Xo = X.copy(); Xo[M] = np.nan
                t0 = time.perf_counter()
                try:
                    P = fn(Xo.copy(), {}); dt = time.perf_counter() - t0
                    cells.append(f"{rmse(X, P, M):.3f}({dt*1000:.0f}ms)")
                except Exception as e:                       # noqa: BLE001
                    cells.append(f"ERR:{type(e).__name__}")
            print(f"{mname:14s} {typ:11s} " + " ".join(c.rjust(14) for c in cells))


if __name__ == "__main__":
    main()
