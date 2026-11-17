"""Small recorded development search; keep one paired model per original split."""
import os,sys,json,time,copy,gc,msvcrt,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from additive_pilot import dump,digest,metric
from data import load_data,validate_split
from residual_kan import PAIRS,retain_energy
from conditional_site_kan import prepare,restore,cpu_export,transform
from bootstrap_native_kan import bootstrap_draw,deletion_contrasts,whole_output_scores,compile_branch
from activity_entropy_auc_search_kan import fit
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'activity_entropy_auc_search_plan.json'


def plain(value):
    if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [plain(item) for item in value]
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().tolist()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def inference_weights(weights):return {key:value for key,value in weights.items() if key!='anchor'}


def restore_inference(weights):return restore(dict(weights,anchor=weights['first']))


def fit_record(path,result,roles,stage,branch):
    record={key:value for key,value in result.items() if key!='weights'}
    record.update(roles,stage=stage,branch=branch);dump(path,plain(record))
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),stage=stage,branch=branch,
        actual_updates=result['updates_run'],maximum_budget=result['maximum_budget'],selected_update=result['selected_update'],
        internal_candidates=len(result['training_internal_candidates']),fit_seconds=result['elapsed_fit_seconds'])


def calibrate(values,omitted,labels,probabilities):
    n=len(labels);sums=np.zeros(n);counts=np.zeros(n,dtype=int)
    for logits,positions in zip(values,omitted):sums[positions]+=logits[positions];counts[positions]+=1
    covered=counts>0;oob=sums[covered]/counts[covered];y=labels[covered]
    grid=np.log(np.asarray(probabilities)/(1-np.asarray(probabilities)));accuracy=np.asarray([np.mean((oob>=threshold)==y) for threshold in grid])
    chosen=min(range(len(grid)),key=lambda i:(-accuracy[i],abs(grid[i]),i))
    return float(grid[chosen]),dict(probability_grid=probabilities,logit_grid=grid.tolist(),accuracy=accuracy.tolist(),chosen_index=chosen,
        covered_positions=np.flatnonzero(covered).tolist(),coverage_counts=counts.tolist(),covered_mean_logits=oob.tolist(),covered_labels=y.tolist(),
        not_unbiased_out_of_fold_estimate=True)


def metrics(labels,values,threshold):
    raw=metric(labels,values);return dict(accuracy=float(np.mean((values>=threshold)==labels)),auc=raw['auc'],accuracy_at_zero=raw['accuracy'])


def ranking(row):
    value=row['development'];return (min(value['accuracy']/.73,value['auc']/.78),value['accuracy']+value['auc'],-row['retained_rois'])


def boundary_check(raw,branch,values):
    state=branch['state'];w=branch['weights'];net=restore_inference(w)
    assert torch.count_nonzero(w['first']*(1-w['function_mask'][:,None,:]))==0
    keep=np.asarray(state['feature_indices']);rois=np.asarray(state['rois']);assert np.isin(PAIRS[:,keep],rois).all()
    x=raw[:4].copy();removed=np.ones(4005,dtype=bool);removed[keep]=False;x[:,removed]=np.nan
    with torch.no_grad():actual=net(transform(x,state)).cpu().numpy()
    error=float(np.max(np.abs(actual-values[:4])));assert error<2e-8
    return dict(passed=True,deleted_raw_fc_nan_error=error,retained_fc_inputs=len(keep),permanent_mask_exact=True)


def train_native(study,part,spec,idx,raw,y,parents,states,qs,draws,cfg,folder,roles):
    n=len(idx['train']);sources=[];ds=[];fits=[];weak=[];start=time.perf_counter()
    for branch in range(cfg['branches']):
        result,_=fit(restore(parents[branch]),qs[branch],y,draws[branch],states[branch],cfg['warm_updates'],cfg['warm_candidates'],cfg)
        fits.append(fit_record(folder/f'warm_{branch}.json',result,roles,'warm',branch))
        net=restore(result['weights']);removed=net.prune_functions(states[branch]['function_covariance'].to(qs[branch].device),cfg['weak_fraction'])
        ds.append(deletion_contrasts(net,qs[branch][:n],states[branch]['feature_indices']));weak.append(removed['removed_individual_curves']);sources.append(cpu_export(net));del net,result
    scores=whole_output_scores(ds,cfg['branches']);rois=retain_energy(scores,cfg['roi_mass']);assert len(rois)>=2
    values=[];omitted=[];branches=[];boundaries=[]
    for branch in range(cfg['branches']):
        child,state,keep=compile_branch(restore(sources[branch]),states[branch],rois);assert keep.any()
        result,v=fit(child,qs[branch][:,keep].contiguous(),y,draws[branch],state,cfg['continuation_updates'],cfg['continuation_candidates'],cfg)
        fits.append(fit_record(folder/f'native_{branch}.json',result,roles,'native',branch));item=dict(weights=inference_weights(result['weights']),state=state,branch=branch)
        boundaries.append(boundary_check(raw,item,v));branches.append(item);values.append(v);omitted.append(result['oob_positions']);del child,result
    mean=np.mean(values,axis=0);threshold,calibration=calibrate(values,omitted,y[:n],cfg['threshold_probability_grid'])
    row=dict(study=study,partition=part,configuration=cfg['id'],retained_rois=len(rois),rois=rois.tolist(),training=metrics(y[:n],mean[:n],threshold),
        development=metrics(y[n:],mean[n:],threshold),threshold=threshold,calibration=calibration,variant='native_0_95',
        weak_complete_functions_removed=sum(weak),fits=fits,raw_fc_boundary_checks=boundaries,elapsed_seconds=time.perf_counter()-start,
        configuration_selected_using_development=True,test_evaluated=False)
    np.savez_compressed(folder/'predictions.npz',allowed_indices=np.r_[idx['train'],idx['valid']],labels=y,logits=mean,roi_scores=scores,
        signed_source_roi_contrasts=np.stack(ds),branch_logits=np.stack(values))
    dump(folder/'complete.json',plain(row));print('ENTROPY AUC SEARCH NATIVE',study,part,cfg['id'],len(rois),row['development'],flush=True)
    return dict(row=row,branches=branches,source_weights=sources,cfg=cfg,values=mean,omitted=omitted,folder=folder)


def paired_dense(winner,raw,y,idx,states,qs,draws,roles,folder):
    cfg=winner['cfg'];n=len(idx['train']);branches=[];values=[];omitted=[];fits=[]
    for branch in range(cfg['branches']):
        child,state,keep=compile_branch(restore(winner['source_weights'][branch]),states[branch],np.arange(90));assert keep.any()
        result,v=fit(child,qs[branch][:,keep].contiguous(),y,draws[branch],state,cfg['continuation_updates'],cfg['continuation_candidates'],cfg)
        fits.append(fit_record(folder/f'dense_{branch}.json',result,roles,'matched_dense',branch));branches.append(dict(weights=inference_weights(result['weights']),state=state,branch=branch))
        values.append(v);omitted.append(result['oob_positions']);del child,result
    mean=np.mean(values,axis=0);threshold,calibration=calibrate(values,omitted,y[:n],cfg['threshold_probability_grid'])
    row=dict(study=winner['row']['study'],partition=winner['row']['partition'],configuration=cfg['id'],variant='matched_dense',retained_rois=90,
        rois=list(range(90)),threshold=threshold,calibration=calibration,training=metrics(y[:n],mean[:n],threshold),development=metrics(y[n:],mean[n:],threshold),fits=fits,test_evaluated=False)
    np.savez_compressed(folder/'dense_predictions.npz',allowed_indices=np.r_[idx['train'],idx['valid']],labels=y,logits=mean,branch_logits=np.stack(values))
    return row,branches


def verify_selected(raw,idx,variants):
    sys.path.insert(0,str(ROOT/'audit'));from conditional_site_math import transform as canonical_transform,components
    records=[];take=np.arange(len(idx['train']),len(raw))[:16]
    with threadpool_limits(limits=4):
        for name,branches,expected in variants:
            values=[]
            for branch in branches:values.append(components(canonical_transform(raw[take],branch['state']),branch['weights'])[0])
            error=float(np.max(np.abs(np.mean(values,axis=0)-expected[take])));assert error<2e-8
            records.append(dict(passed=True,variant=name,independent_numpy_development_people=len(take),canonical_mean_logit_error=error))
    return records


def run_partition(study,part,spec,plan):
    assert digest(spec['dataset']['path'])==spec['dataset']['sha256'];matrix,labels,_,_=load_data(spec['dataset']['path'],spec['label_column'])
    idx=validate_split(spec['splits'][part]['indices'],len(labels),labels);allowed=np.r_[idx['train'],idx['valid']]
    raw=matrix[allowed][:,PAIRS[0],PAIRS[1]];y=labels[allowed];n=len(idx['train']);del matrix
    folder=ROOT/f'pilot/activity_entropy_auc_search/{study}/partition_{part}';folder.mkdir(parents=True,exist_ok=False)
    roles=dict(training_indices=idx['train'].tolist(),development_indices=idx['valid'].tolist(),excluded_test_indices=idx['test'].tolist(),plan_sha256=digest(PLAN))
    dump(folder/'split.json',roles);parents=[];states=[];qs=[];draws=[];initialization=[]
    for branch in range(plan['branches']):
        seed=plan['bootstrap_seed']+10000*(study=='adhd')+100*part+branch;draw=bootstrap_draw(y[:n],seed);draws.append(draw)
        old=ROOT/f'pilot/bootstrap_native/{study}/partition_{part}/branch_{branch}/initialization.pt'
        tic=time.perf_counter()
        if old.exists():
            assert digest(old)==plan['reused_initialization_sha256'][old.relative_to(ROOT).as_posix()];p=torch.load(old,map_location='cpu',weights_only=False)
            assert np.array_equal(p['bootstrap_draw_positions'],draw) and np.array_equal(p['state']['training_indices'],idx['train'][draw])
            weights,state=p['weights'],p['state'];reused=True
        else:
            net,state,_=prepare(raw[draw],idx['train'][draw],plan['width'],float(y[draw].mean()));weights=cpu_export(net);del net;reused=False
        parents.append(weights);states.append(state);qs.append(transform(raw,state));initialization.append(dict(branch=branch,bootstrap_seed=seed,reused=reused,seconds=time.perf_counter()-tic))
    trials=[];winner=None
    for spec_cfg in plan['configurations']:
        cfg=dict(plan,**spec_cfg);trial_folder=folder/cfg['id'];trial_folder.mkdir()
        dump(ROOT/'activity_entropy_auc_search_status.json',dict(stage='training_development_trials',study=study,partition=part,configuration=cfg['id'],test_evaluated=False,pid=os.getpid()))
        candidate=train_native(study,part,spec,idx,raw,y,parents,states,qs,draws,cfg,trial_folder,roles);trials.append(candidate['row'])
        if winner is None or ranking(candidate['row'])>ranking(winner['row']):winner=candidate
        else:del candidate
        gc.collect();torch.cuda.empty_cache()
    selected=folder/'selected';selected.mkdir();dense,dense_branches=paired_dense(winner,raw,y,idx,states,qs,draws,roles,selected)
    dense_values=np.load(selected/'dense_predictions.npz')['logits']
    canonical=verify_selected(raw,idx,[('native',winner['branches'],winner['values']),('matched_dense',dense_branches,dense_values)])
    checkpoint=selected/'model.pt';assert not checkpoint.exists()
    torch.save(dict(native=winner['branches'],matched_dense=dense_branches,native_threshold=winner['row']['threshold'],dense_threshold=dense['threshold'],
        source_initialization=initialization,selected_configuration=winner['cfg']['id'],inference_only_training_anchor_omitted=True,**roles),checkpoint)
    record=dict(study=study,partition=part,all_trials=trials,selected=winner['row'],matched_dense=dense,source_initializations=initialization,
        checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=digest(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        independent_numpy_checks=canonical,**roles,test_evaluated=False)
    dump(folder/'complete.json',plain(record));print('ENTROPY AUC SEARCH SELECTED',study,part,winner['cfg']['id'],winner['row']['development'],'DENSE',dense['development'],flush=True)
    return record


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    p=json.loads(PLAN.read_text());assert digest(ROOT/'development_plan.json')==p['base_plan_sha256']
    assert digest(ROOT/'audit/activity_entropy_auc_search_unit_validation.json')==p['unit_audit_sha256']
    for name,h in p['code_sha256'].items():assert digest(ROOT/name)==h,name
    assert shutil.disk_usage(ROOT).free>3*2**30
    original=json.loads((ROOT/'development_plan.json').read_text());records=[]
    for study in p['studies']:
        for part in p['partitions']:
            records.append(run_partition(study,part,original['studies'][study],p))
            dump(ROOT/'activity_entropy_auc_search_development.json',dict(cases=plain(records),test_evaluated=False,goal_complete=False));torch.cuda.empty_cache();gc.collect()
    dump(ROOT/'activity_entropy_auc_search_status.json',dict(stage='all_30_development_trials_complete_10_models_retained',completed_cases=len(records),
        clinical_development_trials=sum(len(case['all_trials']) for case in records),selected_models=len(records),test_evaluated=False,goal_complete=False))
    print('ENTROPY AUC SEARCH DEVELOPMENT COMPLETE',len(records),flush=True)


if __name__=='__main__':
    with open(ROOT/'activity_entropy_auc_search.lock','a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
