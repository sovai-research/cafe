import sys; sys.path.insert(0,".")
import numpy as np, time, statistics as st
import c_unified_penmf as BASE
import c_up_s6 as MINE
rng=np.random.default_rng(0)
def mk2d(T,N,miss,rank):
    Z=rng.standard_normal((T,rank));W=rng.standard_normal((rank,N))
    X=Z@W+0.3*rng.standard_normal((T,N));X[rng.random((T,N))<miss]=np.nan;return X
def mkpanel(E,T,N,miss,rank):
    rowsX=[];eids=[];tids=[];G=rng.standard_normal((T,rank))
    for e in range(E):
        L=rng.standard_normal((rank,N));Xe=G@L+0.3*rng.standard_normal((T,N))
        rowsX.append(Xe);eids+=[e]*T;tids+=list(range(T))
    X=np.vstack(rowsX);X[rng.random(X.shape)<miss]=np.nan
    return X,{"entity_ids":np.array(eids),"time_ids":np.array(tids)}
cases={"tall":(mk2d(2000,20,0.2,3),{}),"wide":(mk2d(300,200,0.2,8),{})}
Xp,mp=mkpanel(8,250,12,0.3,3);cases["panel"]=(Xp,mp)
def bench(fn,X,meta,reps=5):
    ts=[]
    for _ in range(reps):
        s=time.perf_counter();fn(X.copy(),meta);ts.append(time.perf_counter()-s)
    return st.median(ts)*1e3
for name,(X,meta) in cases.items():
    b=bench(BASE.online_impute,X,meta);m=bench(MINE.online_impute,X,meta)
    print(f"{name}: base={b:.0f}ms mine={m:.0f}ms speedup={b/m:.2f}x")
