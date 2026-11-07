import json, numpy as np, pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
F=['FER','SIGS','FSR','PLV','BLV']
a=pd.read_csv('new/gathered.csv',header=None,names=F+['type']); b=pd.read_csv('result.csv',header=None,names=F+['type'])
Xa,ya,Xb,yb=a[F].values,a['type'].values.astype(int),b[F].values,b['type'].values.astype(int)
nb=GaussianNB().fit(Xa,ya); kn=KNeighborsClassifier(5,n_jobs=2).fit(Xa,ya)
p=((nb.predict_proba(Xb)[:,1]+kn.predict_proba(Xb)[:,1])/2>.5).astype(int)
idle=(np.abs(Xb[:,[0,1,3,4]]).sum(1)==0)
out={}
out['n']=len(b); out['attack_n']=int(yb.sum())
out['stage1_flags_attack_pct']=float(p[yb==1].mean()*100); out['stage1_flags_normal_pct']=float(p[yb==0].mean()*100)
out['stage1_flags_attack_active_pct']=float(p[(yb==1)&~idle].mean()*100); out['stage1_flags_attack_idle_pct']=float(p[(yb==1)&idle].mean()*100)
out['stage1_flags_normal_active_pct']=float(p[(yb==0)&~idle].mean()*100)
out['attack_idle_pct']=float(idle[yb==1].mean()*100)
out['overall_acc']=float((p==yb).mean()*100)
print(json.dumps(out,indent=1))
print('result.csv attack active-window medians:'); print(b[(yb==1)&~idle][F].median().round(3).to_dict()); print('result.csv normal active medians:'); print(b[(yb==0)&~idle][F].median().round(3).to_dict())
chg=np.flatnonzero(np.diff(yb))+1; print('label blocks result.csv:',[(int(yb[s]),int(s),int(e)) for s,e in zip(np.r_[0,chg],np.r_[chg,len(yb)])][:8])
json.dump(out,open('cross_stage.json','w'))
