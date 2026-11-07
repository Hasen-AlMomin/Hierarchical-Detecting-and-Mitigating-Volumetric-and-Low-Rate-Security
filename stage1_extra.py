import json, numpy as np, pandas as pd
from math import factorial
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
F=['FER','SIGS','FSR','PLV','BLV']
df=pd.read_csv('new/gathered.csv',header=None,names=F+['type']); X=df[F].values; y=df['type'].values.astype(int); N=len(df)
OUT={}
itr,ite=train_test_split(np.arange(N),random_state=0,test_size=.25)
Xtr,ytr,Xte,yte=X[itr],y[itr],X[ite],y[ite]
nb=GaussianNB().fit(Xtr,ytr); kn=KNeighborsClassifier(5,n_jobs=2).fit(Xtr,ytr)
pn,pk=nb.predict_proba(Xte)[:,1],kn.predict_proba(Xte)[:,1]
vote=((pn+pk)/2>.5)
gate=((pn>=.5)&(pk>0))
OUT['vote_vs_gate']=dict(agree_pct=float((vote==gate).mean()*100), vote_attacks=int(vote.sum()), gate_attacks=int(gate.sum()))
# exact: vote iff pn+pk>1
for lo in (0,.2,.4,.6,.8):
    pass
OUT['nb_hard_pct']=float(((pn<1e-3)|(pn>1-1e-3)).mean()*100)
OUT['pk_distribution_when_nb_attack']={str(v):float(((pk==v)[pn>.5]).mean()*100) for v in (0,.2,.4,.6,.8,1.0)}
OUT['spearman']=pd.DataFrame(X,columns=F).corr('spearman').round(3).to_dict()
OUT['pearson_PLV_BLV']=float(np.corrcoef(X[:,3],X[:,4])[0,1])
# exact Shapley on MinMax-scaled features
def shap(Xa,ya,Xb,seed=0):
    nbm=GaussianNB().fit(Xa,ya); knm=KNeighborsClassifier(5,n_jobs=2).fit(Xa,ya)
    f=lambda Z:(nbm.predict_proba(Z)[:,1]+knm.predict_proba(Z)[:,1])/2
    rng=np.random.default_rng(seed); Bg=Xa[rng.choice(len(Xa),40,replace=False)]; S=rng.choice(len(Xb),120,replace=False)
    M=5; masks=np.array([[(s>>j)&1 for j in range(M)] for s in range(1<<M)],bool); wt={n:factorial(n)*factorial(M-n-1)/factorial(M) for n in range(M)}
    phi=np.zeros((len(S),M))
    for r,i in enumerate(S):
        Z=np.vstack([np.where(m[None,:],Xb[i][None,:],Bg) for m in masks]); v=f(Z).reshape(1<<M,len(Bg)).mean(1)
        for j in range(M):
            for s in range(1<<M):
                if not masks[s,j]: phi[r,j]+=wt[masks[s].sum()]*(v[s|(1<<j)]-v[s])
    ma=np.abs(phi).mean(0); return {F[j]:float(ma[j]/ma.sum()*100) for j in range(M)}
sc=MinMaxScaler().fit(Xtr)
OUT['shap_minmax']=shap(sc.transform(Xtr),ytr,sc.transform(Xte))
# NB-alone and KNN-alone shapley, raw (who creates the BLV share)
def shap_model(model_fn):
    pass
# fold-0 anatomy of the blocked CV
B=[(0,59271),(59271,145894),(145894,N)]
fold=np.zeros(N,int)
for s,e in B:
    n=e-s; fold[s:e]=np.minimum((np.arange(n)*5)//n,4)
rows=[]
for bi,(s,e) in enumerate(B):
    for fo in range(5):
        m=np.flatnonzero(fold[s:e]==fo)+s; d=df.iloc[m]; idl=(np.abs(X[m][:,[0,1,3,4]]).sum(1)==0)
        rows.append(dict(block=bi,fold=fo,rows=len(m),idle_pct=float(idl.mean()*100),FER_med=float(d.FER.median()),FSR_med=float(d.FSR.median()),PLV_med=float(d.PLV.median()),BLV_med=float(d.BLV.median()),FER_p90=float(d.FER.quantile(.9)),FSR_lt05_pct=float((d.FSR<.5).mean()*100)))
OUT['fold_anatomy']=rows
# fold-0 predictions by block
PURGE=100; res={}
f=0; te_i=np.flatnonzero(fold==f); near=np.zeros(N,bool)
for s,e in B:
    m=np.flatnonzero(fold[s:e]==f)+s; near[max(s,m.min()-PURGE):min(e,m.max()+PURGE+1)]=True
tr_i=np.flatnonzero(~near)
nb0=GaussianNB().fit(X[tr_i],y[tr_i]); kn0=KNeighborsClassifier(5,n_jobs=2).fit(X[tr_i],y[tr_i])
p0=((nb0.predict_proba(X[te_i])[:,1]+kn0.predict_proba(X[te_i])[:,1])/2>.5).astype(int)
for bi,(s,e) in enumerate(B):
    m=(te_i>=s)&(te_i<e); yy=y[te_i][m]; pp=p0[m]
    res[bi]=dict(n=int(m.sum()),acc=float((yy==pp).mean()*100),flag_rate=float(pp.mean()*100),label=int(yy[0]))
OUT['fold0_by_block']=res
json.dump(OUT,open('stage1_extra.json','w'),indent=1)
print(json.dumps({k:v for k,v in OUT.items() if k!='fold_anatomy'},indent=1))
print(pd.DataFrame(rows).round(2).to_string())
