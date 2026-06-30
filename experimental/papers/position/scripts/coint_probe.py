"""
Does CAFE's FX loss come from 'lack of cointegrating pairs'?

Common-trends model for N nonstationary series:
    X_t = B F_t + U_t
  F_t : r-dim random walk      (r common stochastic trends)
  B   : N x r loadings
  U_t : stationary AR(1) idiosyncratic, |phi|<1   (mean-reverting)

If r < N there are N-r cointegrating vectors: stationary linear combos.
The missing cell at time t is then pinned by the contemporaneous cross-section
(other series -> B F_t) PLUS the fact that U_t mean-reverts -> a factor imputer
beats last-value. If r = N (each series its own trend), no cointegration ->
last-value is optimal and the factor model adds nothing.

We sweep r = N..1 and report MAE(CAFE)/MAE(LOCF) on held-out scattered cells.
Prediction: ratio < 1 (CAFE wins) as r shrinks; ratio ~1 as r -> N.
Then we locate the real FX panel on the same axis.
"""
import numpy as np
import cafe

rng_master = np.random.default_rng(0)

def make_panel(N, T, r, phi, sig_f=1.0, sig_u=1.0, seed=0):
    rng = np.random.default_rng(seed)
    # r common random-walk trends
    F = np.cumsum(sig_f * rng.standard_normal((T, r)), axis=0)
    B = rng.standard_normal((N, r)) / np.sqrt(r)
    # stationary idiosyncratic AR(1)
    U = np.zeros((T, N))
    e = sig_u * np.sqrt(1 - phi**2) * rng.standard_normal((T, N))
    for t in range(1, T):
        U[t] = phi * U[t-1] + e[t]
    X = F @ B.T + U
    return X

def locf(Xobs):
    Z = Xobs.copy()
    for j in range(Z.shape[1]):
        last = np.nan
        for t in range(Z.shape[0]):
            if np.isfinite(Z[t, j]): last = Z[t, j]
            elif np.isfinite(last): Z[t, j] = last
        # backfill any leading nan with column mean
        m = np.nanmean(Z[:, j]) if np.isfinite(np.nanmean(Z[:, j])) else 0.0
        Z[np.isnan(Z[:, j]), j] = m
    return Z

def zscore(X):  # per-col, leak-free enough for this controlled study
    return (X - X.mean(0)) / (X.std(0) + 1e-9)

def eval_panel(X, rate=0.10, seed=1):
    X = zscore(X)
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < rate
    Xobs = X.copy(); Xobs[mask] = np.nan
    cafe_fill = np.asarray(cafe.impute(Xobs))
    locf_fill = locf(Xobs)
    mae = lambda P: float(np.mean(np.abs(P[mask] - X[mask])))
    return mae(cafe_fill), mae(locf_fill)

print(f"{'r (trends)':>10} {'coint_rank':>11} {'CAFE':>7} {'LOCF':>7} {'ratio':>7}  winner")
N, T, phi = 12, 1500, 0.85
for r in [12, 10, 8, 6, 4, 2, 1]:
    cs, ls = [], []
    for s in range(3):
        X = make_panel(N, T, r, phi, seed=s)
        c, l = eval_panel(X, seed=10+s)
        cs.append(c); ls.append(l)
    c, l = np.mean(cs), np.mean(ls)
    coint = N - r
    win = "CAFE" if c < l else "LOCF"
    print(f"{r:>10} {coint:>11} {c:>7.3f} {l:>7.3f} {c/l:>7.2f}  {win}")

print()
print("phi=0 (idiosyncratic also random-walk -> NO cointegration regardless of r):")
for r in [4, 1]:
    cs, ls = [], []
    for s in range(3):
        X = make_panel(N, T, r, phi=0.0, seed=s)  # U becomes iid -> X = BF + noise, F rw
        c, l = eval_panel(X, seed=10+s)
        cs.append(c); ls.append(l)
    c, l = np.mean(cs), np.mean(ls)
    print(f"  r={r}: CAFE={c:.3f} LOCF={l:.3f} ratio={c/l:.2f}")

# Real FX panel, same protocol
print()
X = np.load('/Users/dereksnow/Sovai/Github/TIMARA/data/exchange_clean.npy')
cs, ls = [], []
for s in range(3):
    c, l = eval_panel(X[:3000], seed=10+s)  # cap for speed
    cs.append(c); ls.append(l)
print(f"REAL FX (exchange_clean, 8 ccy): CAFE={np.mean(cs):.3f} LOCF={np.mean(ls):.3f} ratio={np.mean(cs)/np.mean(ls):.2f}")
