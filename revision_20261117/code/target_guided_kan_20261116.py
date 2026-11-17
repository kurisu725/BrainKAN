"""The existing native BrainKAN with optional train-only supervised initialization.

Three regularized least-squares directions initialize three KAN input channels;
they do not rank/select ROIs or supply a separate final classifier. All complete
KAN functions learn jointly and are pruned/scored by the existing mechanism.
New experiments are isolated from all frozen manuscript experiments.
"""
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from threadpoolctl import threadpool_limits

from conditional_site_kan import prepare, restore, cpu_export, transform
from bootstrap_native_kan import bootstrap_draw, deletion_contrasts, whole_output_scores, compile_branch
from activity_entropy_auc_search_kan import fit
from activity_entropy_auc_search import inference_weights, plain
from roi_contribution_thresholds import threshold_rois

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit'))
from run_current_parameters_20261116 import record_fit


@torch.no_grad()
def initialize(raw, labels, indices, cfg):
    """Inputs here contain only the current branch's training bootstrap draw."""
    labels=np.asarray(labels);assert len(raw)==len(labels)==len(indices)
    net,state,_=prepare(raw,indices,cfg['width'],float(labels.mean()))
    info=dict(kind='PCA',training_people=len(labels),training_indices=list(map(int,indices)))
    if cfg['initialization']=='supervised3':
        scaler=state['scaler']
        z=torch.as_tensor(np.clip((raw-scaler['mean'])/scaler['std'],-3,3)-scaler['clip_center'],
                          dtype=torch.float64,device='cuda')
        target=torch.as_tensor(labels-labels.mean(),dtype=z.dtype,device=z.device)
        gram=z@z.T/z.shape[1]
        identity=torch.eye(len(z),dtype=z.dtype,device=z.device)
        strengths=cfg['initialization_ridge_strengths']
        assert len(strengths)<cfg['width']
        scales=[]
        for channel,strength in enumerate(strengths):
            direction=z.T@torch.linalg.solve(gram+strength*identity,target)/z.shape[1]
            scale=(z@direction).square().mean().sqrt().clamp_min(.05)
            net.first[:,0,channel].copy_(direction/scale)
            scales.append(float(scale))
        net.anchor.copy_(net.first.detach())
        info.update(kind='Training-only supervised KAN coefficient initialization',
                    ridge_strengths=strengths,projection_scales=scales,
                    supervised_channels=len(strengths),separate_final_classifier=False,
                    roi_ranking_or_selection=False)
    else:
        assert cfg['initialization']=='pca'
    dtype=torch.float32 if cfg['precision']=='float32' else torch.float64
    net=net.to(dtype=dtype)
    state['function_covariance']=state['function_covariance'].to(dtype=dtype)
    state['coefficient_initialization']=info
    return net,state,info


def restored(weights,cfg):
    return restore(weights).to(dtype=torch.float32 if cfg['precision']=='float32' else torch.float64)


def basis(raw,state,cfg):
    return transform(raw,state).to(dtype=torch.float32 if cfg['precision']=='float32' else torch.float64)


def metrics(y,logits,threshold=0.):
    prediction=np.asarray(logits)>=threshold
    p,r,f,_=precision_recall_fscore_support(y,prediction,average='macro',zero_division=0)
    return dict(accuracy=float(accuracy_score(y,prediction)),auc=float(roc_auc_score(y,logits)),
                precision=float(p),recall=float(r),f1=float(f),
                accuracy_at_zero=float(accuracy_score(y,np.asarray(logits)>=0)))


def calibration(labels,values,probabilities):
    probabilities=np.asarray(probabilities,dtype=float)
    thresholds=np.log(probabilities/(1-probabilities))
    accuracies=np.array([np.mean((values>=t)==labels) for t in thresholds])
    chosen=min(range(len(thresholds)),key=lambda i:(-accuracies[i],abs(thresholds[i]),i))
    return float(thresholds[chosen]),dict(source='development only',probabilities=probabilities.tolist(),
        logit_thresholds=thresholds.tolist(),accuracy=accuracies.tolist(),chosen_index=chosen,
        test_used=False)


def forward(raw,branches,cfg,verify=True):
    logits=[];maximum_error=0.;boundaries=[]
    tolerance=1e-3 if cfg['precision']=='float32' else 2e-8
    for branch in branches:
        net=restored(dict(branch['weights'],anchor=branch['weights']['first']),cfg)
        with torch.no_grad():
            value=net(basis(raw,branch['state'],cfg)).cpu().numpy()
            assert np.isfinite(value).all()
            keep=np.asarray(branch['state']['feature_indices'])
            removed=np.ones(4005,dtype=bool);removed[keep]=False
            changed=raw[:4].copy();changed[:,removed]=np.nan
            deleted=net(basis(changed,branch['state'],cfg)).cpu().numpy()
            error=float(np.max(np.abs(deleted-value[:4])))
            assert error<tolerance
        boundaries.append(dict(passed=True,deleted_fc_nan_error=error,physical_fc_inputs=len(keep)))
        logits.append(value);del net
    values=np.stack(logits);mean=values.mean(0)
    if verify:
        from conditional_site_math import transform as numpy_transform, components
        take=np.arange(min(16,len(raw)))
        with threadpool_limits(limits=4):
            reference=np.mean([components(numpy_transform(raw[take],b['state']),b['weights'])[0] for b in branches],axis=0)
        maximum_error=float(np.max(np.abs(reference-mean[take])))
        assert maximum_error<tolerance,(maximum_error,tolerance)
    return mean,values,dict(passed=True,independent_numpy_people=min(16,len(raw)),
                            maximum_logit_error=maximum_error,tolerance=tolerance,
                            deleted_input_boundary_checks=boundaries)


def train_native(raw,labels,indices,cfg,folder,write):
    """Train on train only; predict valid. Test rows are not accepted here."""
    n=len(indices['train']);assert len(raw)==n+len(indices['valid'])==len(labels)
    draws=[bootstrap_draw(labels[:n],s) for s in cfg['bootstrap_seeds']]
    assert len(cfg['bootstrap_seeds'])==cfg['branches']
    sources=[];states=[];qs=[];effects=[];fit_refs=[];initializations=[];pruning=[];movement=[]
    start=time.perf_counter()
    for branch,draw in enumerate(draws):
        net,state,info=initialize(raw[draw],labels[draw],indices['train'][draw],cfg)
        initializations.append(info);q=basis(raw,state,cfg)
        initial_first=net.first.detach().cpu().clone()
        result,_=fit(net,q,labels,draw,state,cfg['warm_updates'],cfg['warm_candidates'],cfg)
        changed=float((result['weights']['first']-initial_first).abs().max())
        assert changed>0 and bool(result['weights']['head'].abs().sum()>0)
        movement.append(dict(branch=branch,maximum_input_coefficient_change=changed,
                             head_coefficient_norm=float(result['weights']['head'].norm())))
        del initial_first
        fit_refs.append(record_fit(folder/'source'/f'branch_{branch}.json',result,'source',branch))
        net=restored(result['weights'],cfg)
        pruned=net.prune_functions(state['function_covariance'].to('cuda'),cfg['weak_fraction'])
        assert pruned['removed_individual_curves']==int(4005*cfg['width']*cfg['weak_fraction'])
        pruning.append(dict(branch=branch,removed_input_functions=pruned['removed_individual_curves'],
                            total_input_functions=4005*cfg['width']))
        effects.append(deletion_contrasts(net,q[:n],state['feature_indices']))
        sources.append(cpu_export(net));states.append(state);qs.append(q)
        del net,result
    scores=whole_output_scores(effects,cfg['branches'])
    rois,cutoff=threshold_rois(scores,dict(family='mass',value=cfg['roi_mass']))
    np.savez_compressed(folder/'training_contributions.npz',training_indices=indices['train'],
        roi_scores=scores,signed_branch_effects=np.stack(effects),rois=rois)
    branches=[];valid_values=[]
    for branch in range(cfg['branches']):
        child,state,keep=compile_branch(restored(sources[branch],cfg),states[branch],rois)
        child=child.to(dtype=qs[branch].dtype)
        result,values=fit(child,qs[branch][:,keep].contiguous(),labels,draws[branch],state,
                           cfg['continuation_updates'],cfg['continuation_candidates'],cfg)
        fit_refs.append(record_fit(folder/'native'/f'branch_{branch}.json',result,'native',branch))
        branches.append(dict(weights=inference_weights(result['weights']),state=state))
        valid_values.append(values[n:]);del child,result
    mean=np.mean(valid_values,axis=0)
    threshold,threshold_record=calibration(labels[n:],mean,cfg['threshold_probability_grid'])
    np.savez_compressed(folder/'development_predictions.npz',pool_indices=indices['valid'],labels=labels[n:],
        logits=mean,branch_logits=np.stack(valid_values),threshold=threshold,predictions=mean>=threshold)
    result=dict(development=metrics(labels[n:],mean,threshold),threshold=threshold,
        threshold_calibration=threshold_record,rois=rois.tolist(),retained_rois=len(rois),
        roi_threshold=cutoff,initializations=initializations,fit_records=fit_refs,
        physical_parameters=sum((10*len(b['state']['feature_indices'])+10)*cfg['width']+1 for b in branches),
        physical_inputs_per_branch=[len(b['state']['feature_indices']) for b in branches],
        weak_function_pruning=pruning,training_coefficient_movement=movement,
        elapsed_seconds=time.perf_counter()-start,weight_files_written=0)
    write(folder/'development_complete.json',result)
    return result,dict(branches=branches,sources=sources,states=states,qs=qs,draws=draws)


def train_dense(raw,labels,indices,cfg,folder,cache,write):
    n=len(indices['train']);branches=[];values=[];fits=[]
    for branch in range(cfg['branches']):
        child,state,keep=compile_branch(restored(cache['sources'][branch],cfg),cache['states'][branch],np.arange(90))
        child=child.to(dtype=cache['qs'][branch].dtype)
        result,v=fit(child,cache['qs'][branch][:,keep].contiguous(),labels,cache['draws'][branch],state,
                       cfg['continuation_updates'],cfg['continuation_candidates'],cfg)
        fits.append(record_fit(folder/'matched_dense'/f'branch_{branch}.json',result,'matched_dense',branch))
        branches.append(dict(weights=inference_weights(result['weights']),state=state))
        values.append(v[n:]);del child,result
    mean=np.mean(values,axis=0)
    threshold,cal=calibration(labels[n:],mean,cfg['threshold_probability_grid'])
    np.savez_compressed(folder/'dense_development_predictions.npz',pool_indices=indices['valid'],labels=labels[n:],
        logits=mean,branch_logits=np.stack(values),threshold=threshold,predictions=mean>=threshold)
    record=dict(development=metrics(labels[n:],mean,threshold),threshold=threshold,
                threshold_calibration=cal,fit_records=fits,retained_rois=90,weight_files_written=0)
    return record,branches
