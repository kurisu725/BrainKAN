"""A KAN's learned ROI gates persist in its continuation objective.

The same pre-gate univariate functions own the amplitude/shape penalties.
Gates are native parameters fitted by the previous training-only cross-fit.
Holding their factors in the design prevents an unconstrained re-fit from
erasing all amplitude information in those gates. No external predictor is
fitted. Inference compiles each factor into its own function coefficients.
"""
import copy
import numpy as np
import torch
from functional_kan import gather_basis
from joint_staged_kan import fixed_first_q
from protected_site_kan import replay
from regularized_functional import function_penalty
from quadratic_prox_kan import solve

PAIRS=np.stack(np.triu_indices(90,1))


def compact_source(source,gates,zero_threshold=1e-6):
    gates=torch.as_tensor(gates,dtype=torch.float64).cpu()
    rois=np.flatnonzero(gates.numpy()>zero_threshold)
    raw_mask=np.isin(PAIRS[0],rois)&np.isin(PAIRS[1],rois);raw_ids=np.flatnonzero(raw_mask)
    keep=np.isin(np.asarray(source['feature_indices']),raw_ids)
    first={k:copy.deepcopy(source[k]) for k in ['basis','scaler']}
    first['basis']=gather_basis(first['basis'],keep)
    scaler=first['scaler'];old_ids=np.asarray(scaler['feature_indices']);raw_keep=np.isin(old_ids,raw_ids)
    assert scaler['mode']=='edge_z' and 'nuisance' not in scaler
    for key in ['mean','std','center']:scaler[key]=np.asarray(scaler[key])[raw_keep].copy()
    scaler['feature_indices']=raw_ids.tolist()
    first.update(feature_indices=np.asarray(source['feature_indices'])[keep],rois=rois,
        normalization_raw_indices=raw_ids,training_indices=np.asarray(source['training_indices']))
    ids=first['feature_indices'];factor=gates[torch.as_tensor(PAIRS[0,ids])]*gates[torch.as_tensor(PAIRS[1,ids])]
    theta=source['model']['weight'][keep].double().clone()
    return first,factor,theta


@torch.no_grad()
def inputs(raw,source,device='cuda'):
    ids=np.asarray(source['scaler']['feature_indices'])
    z=replay(raw[:,ids],None,source['scaler'])
    keep=np.isin(ids,source['feature_indices']);z=torch.tensor(z[:,keep],device=device,dtype=torch.float64)
    q=fixed_first_q(z,source['basis']);return q,z


@torch.no_grad()
def continue_native(q,z,y,n,factor,theta,bias,lam,settings):
    """Optimize pre-gate functions while keeping native gate amplitudes fixed."""
    factor=factor.to(q.device).double();theta=theta.to(q.device).double();bias=torch.as_tensor(bias,device=q.device,dtype=torch.float64)
    penalty=function_penalty(q[:n],z[:n],settings['first_rms'],settings['first_nonlinearity'])
    gated=q[:n]*factor[None,:,None]
    result=solve(gated,y[:n],lam,penalty,settings['max_iterations'],settings['tolerance'],initial=dict(weight=theta,bias=bias))
    effective=result['weight']*factor.cpu()[:,None]
    logits=q.flatten(1)@effective.to(q.device).flatten()+result['bias'].to(q.device)
    return result,effective,logits.cpu().numpy(),penalty.cpu()


@torch.no_grad()
def margin(raw,first,device='cuda'):
    q,_=inputs(raw,first,device)
    return (q.flatten(1)@first['model']['weight'].to(q.device).flatten()+first['model']['bias'].to(q.device)).cpu().numpy()
