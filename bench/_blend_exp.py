"""Test: blend EW-cov and TRMF for 2D, weighting EW-cov by its OWN conditional R2.

alpha[t,j] = reliability of EW-cov for cell (t,j)
           = (1 - condvar_j/margvar_j)  -- causal, from past covariance only.
When cross-section is uninformative (high-rank/heavy-tail) R2->0 -> trust TRMF.
"""
import os, sys
for v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v,"2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.linalg import cho_factor, cho_solve

HALFLIFE=200.0; RIDGE=1e-2; WARM=5

def ewcov_with_conf(X, meta=None):
    """Return (pred, alpha) where alpha[t,j] in [0,1] is EW-cov conditional R2."""
    X=np.ascontiguousarray(np.asarray(X,float)); T,N=X.shape
    out=X.copy(); alpha=np.zeros((T,N))
    lam=0.5**(1.0/HALFLIFE)
    w=0.0; Sx=np.zeros(N); M2=np.zeros((N,N)); col_sum=np.zeros(N); col_cnt=np.zeros(N)
    have=False; I=RIDGE*np.eye(N)
    for t in range(T):
        row=X[t]; obs=~np.isnan(row); miss=~obs
        if miss.any():
            gmean=np.where(col_cnt>0,col_sum/np.maximum(col_cnt,1),0.0)
            filled=gmean.copy()
            if have and obs.any() and w>1e-6:
                mean=Sx/w; cov=M2/w-np.outer(mean,mean)
                o=np.where(obs)[0]; m=np.where(miss)[0]
                Soo=cov[np.ix_(o,o)]+I[np.ix_(o,o)]; Smo=cov[np.ix_(m,o)]
                xo=row[o]-mean[o]
                try:
                    c=cho_factor(Soo,lower=True,check_finite=False)
                    sol=cho_solve(c,xo,check_finite=False)
                    filled[m]=mean[m]+Smo@sol
                    # conditional variance per missing cell: diag(Smm - Smo Soo^-1 Som)
                    margvar=np.maximum(np.diag(cov)[m],1e-9)
                    SooinvSom=cho_solve(c,Smo.T,check_finite=False)      # (|o|,|m|)
                    condvar=margvar-np.einsum('ij,ji->i',Smo,SooinvSom)
                    alpha[t,m]=np.clip(1.0-condvar/margvar,0.0,1.0)
                except Exception:
                    filled[m]=mean[m]
            out[t,miss]=filled[miss]
        full=out[t]; w=lam*w+1.0; Sx=lam*Sx+full; M2=lam*M2+np.outer(full,full)
        col_sum[obs]+=row[obs]; col_cnt[obs]+=1
        if w>=WARM: have=True
    if np.isnan(out).any():
        gm=np.where(col_cnt>0,col_sum/np.maximum(col_cnt,1),0.0); idx=np.where(np.isnan(out)); out[idx]=np.take(gm,idx[1])
    return out, alpha

if __name__=="__main__":
    from arena import SUITE, MASKERS, metrics
    from c_online_trmf import online_impute as trmf
    print(f'{"case":24s} {"TRMF":>6} {"EW":>6} {"R2bl":>6}  meanR2')
    tot_t=tot_b=0; n=0
    for name,(clean,meta,mech,rate,grp) in SUITE.items():
        if meta or clean.shape[1]==1: continue
        M=MASKERS[mech](clean,rate,0); Xo=clean.copy(); Xo[M]=np.nan
        Pt=np.asarray(trmf(Xo.copy(),dict(meta)),float)
        Pe,al=ewcov_with_conf(Xo.copy(),meta)
        Pe=np.asarray(Pe,float)
        Pb=al*Pe+(1-al)*Pt
        rt=metrics(clean,Pt,M)['corr']; re=metrics(clean,Pe,M)['corr']; rb=metrics(clean,Pb,M)['corr']
        mr=al[M].mean()
        flag='*' if rb>max(rt,re)-0.003 and rb>rt+0.005 else ('!' if rb<rt-0.03 else '')
        print(f'{name:24s} {rt:6.3f} {re:6.3f} {rb:6.3f}  {mr:.2f} {flag}')
        tot_t+=rt; tot_b+=rb; n+=1
    print(f'2D mean: TRMF={tot_t/n:.4f}  R2blend={tot_b/n:.4f}')
