"""Training-fitted harmonization, Fisher features and EROS-SMOTE for Res-DAE."""
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

def fit_combat(x,sites,age,sex):
    x=np.asarray(x,dtype=float);sites=np.asarray(sites).astype(str);sex=np.asarray(sex).astype(int)
    cov=pd.DataFrame(dict(site=sites,age=age,sex=sex))
    result=neuroCombat(dat=x.T,covars=cov,batch_col='site',categorical_cols=['sex'],continuous_cols=['age'])
    levels=np.asarray(result['estimates']['batches']).astype(str);sexlevels=np.unique(sex)
    design=np.column_stack([(sites==s).astype(float) for s in levels]+[(sex==s).astype(float) for s in sexlevels[1:]]+[np.asarray(age,dtype=float)])
    beta=np.linalg.lstsq(design,x,rcond=None)[0]
    e=result['estimates']
    model=dict(levels=levels,sexlevels=sexlevels,beta_cov=beta[len(levels):],grand=e['stand.mean'][:,0],
               var=np.asarray(e['var.pooled']).reshape(-1),gamma=e['gamma.star'],delta=e['delta.star'])
    transformed=apply_combat(x,sites,age,sex,model)
    np.testing.assert_allclose(transformed,result['data'].T,atol=1e-8,rtol=1e-8)
    return transformed,model

def apply_combat(x,sites,age,sex,model):
    sites=np.asarray(sites).astype(str);sex=np.asarray(sex).astype(int)
    assert set(sites)<=set(model['levels']),'Previously unseen site'
    mod=np.column_stack([(sex==s).astype(float) for s in model['sexlevels'][1:]]+[np.asarray(age,dtype=float)])
    mean=model['grand'][None,:]+mod@model['beta_cov']
    rows=np.asarray([np.where(model['levels']==s)[0][0] for s in sites])
    z=(x-mean)/np.sqrt(model['var'])
    return ((z-model['gamma'][rows])/np.sqrt(model['delta'][rows]))*np.sqrt(model['var'])+mean

def fisher(x,y):
    overall=x.mean(0);numerator=np.zeros(x.shape[1]);denominator=np.zeros(x.shape[1])
    for cls in [0,1]:
        a=x[y==cls];numerator+=len(a)*(a.mean(0)-overall)**2;denominator+=len(a)*a.var(0)
    return numerator/np.maximum(denominator,1e-12)

def fit_res_preprocessor(fc,y,sites,age,sex,indices,seed=1):
    tri=np.triu_indices(90,1);raw=np.asarray(fc[:,tri[0],tri[1]],dtype=float)
    harmonized,combat=fit_combat(raw,sites,age,sex)
    score=fisher(harmonized,y);selected=np.argsort(-score,kind='stable')[:raw.shape[1]//2]
    real=harmonized[:,selected];mean=real.mean(0);std=np.maximum(real.std(0),1e-8);real=(real-mean)/std
    # EROS similarities from harmonized FC matrices, before Fisher truncation.
    mats=np.broadcast_to(np.eye(90),(len(fc),90,90)).copy();mats[:,tri[0],tri[1]]=harmonized;mats[:,tri[1],tri[0]]=harmonized
    cov=np.asarray([np.cov(a,rowvar=False) for a in mats]);eigenvalues,eigenvectors=np.linalg.eigh(cov)
    eigenvalues=np.maximum(eigenvalues[:,::-1],0);eigenvectors=eigenvectors[:,:,::-1]
    weights=(eigenvalues/np.maximum(eigenvalues.sum(1,keepdims=True),1e-12)).mean(0);weights/=weights.sum()
    counts=np.bincount(y,minlength=2);minor=int(np.argmin(counts));minor_ids=np.flatnonzero(y==minor)
    v=eigenvectors[minor_ids]
    similarities=np.zeros((len(v),len(v)))
    for e,w in enumerate(weights):similarities+=w*np.abs(v[:,:,e]@v[:,:,e].T)
    np.fill_diagonal(similarities,-np.inf);neighbors=np.argsort(-similarities,axis=1,kind='stable')[:,:min(5,len(v)-1)]
    rng=np.random.default_rng(seed);synthetic=[];pairs=[]
    for _ in range(int(counts.max()-counts.min())):
        i=int(rng.integers(len(minor_ids)));j=int(rng.choice(neighbors[i]));a=int(minor_ids[i]);b=int(minor_ids[j]);weight=float(rng.random())
        synthetic.append(real[a]+weight*(real[b]-real[a]));pairs.append(dict(a=int(indices[a]),b=int(indices[b]),weight=weight,label=minor))
    training=np.concatenate([real,np.asarray(synthetic).reshape(-1,len(selected))],axis=0)
    labels=np.concatenate([y,np.full(len(synthetic),minor,dtype=np.int64)])
    state=dict(combat=combat,selected=selected,fisher_scores=score,mean=mean,std=std,fit_indices=list(map(int,indices)),
               augmentation_pairs=pairs,eros_weights=weights,tri=tri)
    return training.astype(np.float32),labels,state

def apply_res_preprocessor(fc,sites,age,sex,state):
    a,b=state['tri'];harmonized=apply_combat(np.asarray(fc[:,a,b],float),sites,age,sex,state['combat'])
    return ((harmonized[:,state['selected']]-state['mean'])/state['std']).astype(np.float32)

def fb_scaler(ts,lengths,indices):
    total=0.;squares=0.;count=0
    for i in indices:
        a=ts[i,:,:int(lengths[i])].astype(float);total+=a.sum();squares+=(a*a).sum();count+=a.size
    mean=total/count;std=np.sqrt(squares/count-mean*mean);assert std>0
    return dict(mean=float(mean),std=float(std),fit_indices=list(map(int,indices)),observed_values=count)

def temporal_inputs(ts,lengths,kind,scaler=None):
    out=np.zeros_like(ts)
    for i,n in enumerate(lengths):
        a=ts[i,:,:int(n)].astype(float)
        if kind=='fbnetgen':a=(a-scaler['mean'])/scaler['std']
        else:a=(a-a.mean(1,keepdims=True))/np.maximum(a.std(1,keepdims=True),1e-8)
        out[i,:,:int(n)]=a.astype(np.float32)
    return out
