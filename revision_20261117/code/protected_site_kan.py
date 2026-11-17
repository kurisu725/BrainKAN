"""Training-only diagnosis-protected site nuisance adjustment for a KAN.

The nuisance fit never predicts diagnosis or selects FCs/ROIs. Training
diagnosis is an unpenalized protected column; only the site columns are
removed from FCs. Application requires site IDs, never target labels.
"""
import numpy as np
import torch


def nuisance_design(sites,levels,proportions):
    # An unseen site receives the average (zero centered) training offset.
    sites=np.asarray(sites);levels=np.asarray(levels)
    known=np.isin(sites,levels)
    design=(sites[:,None]==levels[None]).astype(np.float64)-np.asarray(proportions)[None]
    design[~known]=0
    return design


def fit_nuisance(raw,labels,sites,train,site_ridge=10.):
    train=np.asarray(train);yy=np.asarray(labels)[train].astype(np.float64)
    levels=np.unique(np.asarray(sites)[train]);proportions=np.array([(np.asarray(sites)[train]==s).mean() for s in levels])
    ss=nuisance_design(np.asarray(sites)[train],levels,proportions)
    design=np.column_stack([np.ones(len(train)),yy-yy.mean(),ss])
    reg=np.diag(np.r_[0.,0.,np.full(len(levels),site_ridge)])
    coef=np.linalg.solve(design.T@design+reg,design.T@np.asarray(raw,dtype=np.float64)[train])
    return dict(levels=levels.tolist(),proportions=proportions.tolist(),site_coefficients=coef[2:].astype(np.float32),
        site_ridge=site_ridge,fit_indices=train.tolist(),protected_columns=['intercept','training diagnosis'],
        target_used_only_to_protect_training_effect=True)


def remove_site(raw,sites,state,feature_indices=None):
    beta=np.asarray(state['site_coefficients'])
    if feature_indices is not None:beta=beta[:,feature_indices]
    ss=nuisance_design(sites,state['levels'],state['proportions'])
    return np.asarray(raw,dtype=np.float32)-(ss@beta).astype(np.float32)


def prepare(raw,labels,sites,ntrain,mode,keep=None,site_ridge=10.,nuisance=None,device='cuda'):
    """Fit every pooled quantity on training; subject moments use kept FCs."""
    state=dict(mode=mode,fitted_indices=list(range(ntrain)))
    if mode.startswith('protected_site_'):
        if nuisance is None:nuisance=fit_nuisance(raw,labels,sites,np.arange(ntrain),site_ridge)
        adjusted=remove_site(raw,sites,nuisance);state['nuisance']=nuisance
    else:adjusted=np.asarray(raw,dtype=np.float32).copy()
    if keep is None:keep=np.ones(raw.shape[1],dtype=bool)
    index=np.flatnonzero(keep);z=np.ascontiguousarray(adjusted[:,keep])
    if mode.endswith('subject_z'):
        z=(z-z.mean(1,keepdims=True))/z.std(1,keepdims=True).clip(min=1e-4)
        state['subject_normalization_features']=index.tolist()
    mean=z[:ntrain].mean(0);std=z[:ntrain].std(0).clip(min=1e-3)
    z=np.clip((z-mean)/std,-3,3);center=z[:ntrain].mean(0);z=z-center
    state.update(mean=mean,std=std,center=center,feature_indices=index.tolist())
    return torch.tensor(z,dtype=torch.float32,device=device),state


def replay(raw_kept,sites,state):
    index=np.asarray(state['feature_indices'])
    z=np.asarray(raw_kept,dtype=np.float32)
    if 'nuisance' in state:z=remove_site(z,sites,state['nuisance'],index)
    z=np.ascontiguousarray(z)
    if state['mode'].endswith('subject_z'):
        assert state['subject_normalization_features']==index.tolist()
        z=(z-z.mean(1,keepdims=True))/z.std(1,keepdims=True).clip(min=1e-4)
    return np.clip((z-np.asarray(state['mean']))/np.asarray(state['std']),-3,3)-np.asarray(state['center'])
