"""Recorded exploratory search for five qualifying BrainKAN test combinations.

Pilot recipes and decision cutoffs use development data. The later selection
of five divisions explicitly uses their observed test metrics, as requested.
Selected results are exploratory test-selected outcomes, not unbiased testing.
No model weights or copies of the original FC dataset are written.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'target_test_search_20261116'
PLAN=OUT/'plan.json'
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'audit'))
import target_guided_kan_20261116 as engine
from activity_entropy_auc_search import plain
from data import load_data,validate_split
from residual_kan import PAIRS
from load_selected_activity_entropy_split import load_selected_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def freeze():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    sanity=ROOT/'audit/target_initialization_sanity_20261116/complete.json'
    assert json.loads(sanity.read_text(encoding='utf-8'))['passed']
    original=json.loads((ROOT/'final_test_20261116/plan.json').read_text(encoding='utf-8'))
    datasets=json.loads((ROOT/'development_plan.json').read_text(encoding='utf-8'))['studies']
    source_files=list(original['source_code_sha256'])+[
        'code/target_guided_kan_20261116.py','audit/run_target_test_search_20261116.py']
    configuration=dict(original['configuration'],initialization_ridge_strengths=[.1,1.,10.],
        threshold_probability_grid=[round(.30+.025*i,3) for i in range(17)])
    configuration.pop('width',None)
    plan=dict(created_at=now(),objective='Five exploratory qualifying test combinations per dataset.',
        user_targets={'abide':dict(accuracy=.73,auc=.78),'adhd':dict(accuracy=.71,auc=.72)},
        strict_greater_than=True,require_automatic_roi_reduction=True,selected_results_per_dataset=5,
        configuration=configuration,
        pilot_recipes=[dict(id='pca32_f32',initialization='pca',width=32,precision='float32'),
                       dict(id='guided32_f32',initialization='supervised3',width=32,precision='float32'),
                       dict(id='guided64_f32',initialization='supervised3',width=64,precision='float32')],
        pilot_splits=[s for s in original['splits'] if s['selected_round'] in [1,2]],
        pilot_selection='One recipe per dataset by its two development split means: minimum of target-normalized ACC/AUC, then sum ACC+AUC, then recipe order. No pilot test forwards.',
        dataset_specifications={s:dict(dataset=datasets[s]['dataset'],label_column=datasets[s]['label_column']) for s in ['abide','adhd']},
        dataset_roi_mass={'abide':.90,'adhd':.95},
        random_seeds={s:[base+i for i in range(200)] for s,base in [('abide',1761601),('adhd',2761601)]},
        search_order='ABIDE then ADHD, increasing recorded split seed; stop each dataset after five strict qualifying results.',
        split_method='Diagnosis-stratified 80/10/10; sorted original indices within each role; one split seed for both stratified partitions.',
        bootstrap_seed_rule='8161601 + 100000*(study==adhd) + 100*trial + branch',
        checkpoint_rule='Existing training-internal AUC rule, source120/continuation80, patience4 after update20.',
        decision_rule='Choose probability cutoff on that split development ACC; ties closest to .5, then grid order. Never use test to change the cutoff.',
        test_selection_used=True,
        interpretation='Five combinations are selected using test ACC/AUC. They are exploratory test-selected results and cannot be presented as unbiased independent test estimates. Keep every attempted result and disclose the search count.',
        matched_control='For every qualifying candidate, run the full-90-ROI continuation from exactly the same pruned source. A test-selected comparison is descriptive.',
        storage='RAM models only; no .pt/.pth/.ckpt files or copied FC datasets. Save small assignments, predictions, contribution effects and scalar logs.',
        manuscript_policy='Experiments never write to the delivered manuscript files. User edits remain permitted; hashes below only identify the delivered starting snapshot.',
        manuscript_starting_snapshot_sha256=sha(ROOT/'documents/manuscript_from_black_20261116/template.tex'),
        red_manuscript_starting_snapshot_sha256=sha(ROOT/'documents/manuscript_from_black_red_20261116/template.tex'),
        phenotype_columns={'abide':dict(subject_id=0,site=1,diagnosis=2),'adhd':dict(subject_id=1,site=0,diagnosis=3)},
        numerical_preflight=dict(path=sanity.relative_to(ROOT).as_posix(),sha256=sha(sanity),test_evaluated=False),
        source_sha256={p:sha(ROOT/p) for p in sorted(set(source_files))})
    write(PLAN,plan)
    print('FROZEN EXPLORATORY TARGET SEARCH',sha(PLAN),flush=True)


def cfg_for(plan,recipe,study,seeds):
    return dict(plan['configuration'],**recipe,roi_mass=plan['dataset_roi_mass'][study],bootstrap_seeds=seeds)


def qualifies(record,plan):
    target=plan['user_targets'][record['study']]
    return (record['test']['accuracy']>target['accuracy'] and record['test']['auc']>target['auc']
            and record['retained_rois']<90)


def status(**values):
    write(OUT/'status.json',dict(updated_at=now(),pid=os.getpid(),**values))


def run_pilot(plan):
    rows=[]
    for split in plan['pilot_splits']:
        study,n=split['study'],split['selected_round']
        roles,_=load_selected_split(study,n)
        idx={r:roles[r]['original_indices'] for r in ['train','valid','test']}
        raw=np.concatenate([roles['train']['x'],roles['valid']['x']])[:,PAIRS[0],PAIRS[1]]
        labels=np.r_[roles['train']['y'],roles['valid']['y']]
        del roles
        for recipe in plan['pilot_recipes']:
            folder=OUT/'pilot'/study/f'round_{n}'/recipe['id']
            marker=folder/'record.json'
            if marker.exists():
                record=json.loads(marker.read_text(encoding='utf-8'))
            else:
                folder.mkdir(parents=True,exist_ok=True)
                status(stage='development_pilot',study=study,round=n,recipe=recipe['id'],test_evaluated=False)
                cfg=cfg_for(plan,recipe,study,split['bootstrap_seeds'])
                result,cache=engine.train_native(raw,labels,idx,cfg,folder,write)
                # This independent replay is on development only, before recipe adoption.
                v,_,check=engine.forward(raw[len(idx['train']):],cache['branches'],cfg)
                with np.load(folder/'development_predictions.npz') as old:
                    np.testing.assert_allclose(v,old['logits'],atol=1e-4,rtol=0)
                record=dict(result,study=study,round=n,recipe=recipe['id'],configuration=cfg,
                    test_evaluated=False,independent_development_forward=check,plan_sha256=sha(PLAN))
                if recipe['id']=='pca32_f32':
                    reference=ROOT/f'final_test_20261116/{study}/round_{n}/native/development_replay.npz'
                    with np.load(reference) as previous:
                        record['old_float64_development_maximum_logit_difference']=float(np.max(np.abs(v-previous['logits'])))
                write(marker,record)
                del cache
                gc.collect();torch.cuda.empty_cache()
            rows.append(record)
            print('PILOT',study,n,recipe['id'],record['development'],'ROI',record['retained_rois'],flush=True)
    means=[];adopted={}
    for study in ['abide','adhd']:
        target=plan['user_targets'][study];candidates=[]
        for order,recipe in enumerate(plan['pilot_recipes']):
            group=[r for r in rows if r['study']==study and r['recipe']==recipe['id']]
            assert len(group)==2
            accuracy=float(np.mean([r['development']['accuracy'] for r in group]))
            auc=float(np.mean([r['development']['auc'] for r in group]))
            rank=(min(accuracy/target['accuracy'],auc/target['auc']),accuracy+auc,-order)
            candidates.append((rank,recipe));means.append(dict(study=study,recipe=recipe['id'],accuracy=accuracy,auc=auc))
        adopted[study]=max(candidates,key=lambda x:x[0])[1]
    selection=dict(passed=True,created_before_new_test_forwards=True,test_used=False,means=means,
                   recipes=adopted,plan_sha256=sha(PLAN))
    path=OUT/'adopted_search_settings.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==selection
    else:
        write(path,selection)
    return selection


def save_assignments(folder,idx,labels,phenotype,study):
    id_col,site_col=(0,1) if study=='abide' else (1,0)
    assert len(np.unique(phenotype[:,id_col]))==len(labels)
    assert len(np.unique(phenotype[:,site_col]))==(17 if study=='abide' else 7)
    with (folder/'assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['zero_based_original_index','subject_id','label','site','role'])
        writer.writeheader()
        for role in ['train','valid','test']:
            writer.writerows(dict(zero_based_original_index=int(i),subject_id=str(phenotype[i,id_col]),
                label=int(labels[i]),site=str(phenotype[i,site_col]),role=role) for i in idx[role])


def run_search(study,plan,selection):
    spec=plan['dataset_specifications'][study]
    assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
    matrices,labels,_,phenotype=load_data(spec['dataset']['path'],spec['label_column'])
    full_raw=np.ascontiguousarray(matrices[:,PAIRS[0],PAIRS[1]])
    del matrices
    rows=[];selected=[]
    for trial,seed in enumerate(plan['random_seeds'][study]):
        folder=OUT/'search'/study/f'trial_{trial:03d}_seed_{seed}'
        marker=folder/'record.json'
        if marker.exists():
            record=json.loads(marker.read_text(encoding='utf-8'))
        else:
            folder.mkdir(parents=True,exist_ok=True)
            train,rest=train_test_split(np.arange(len(labels)),test_size=.2,random_state=seed,stratify=labels)
            valid,test=train_test_split(rest,test_size=.5,random_state=seed,stratify=labels[rest])
            idx=validate_split({k:np.sort(v) for k,v in dict(train=train,valid=valid,test=test).items()},len(labels),labels)
            save_assignments(folder,idx,labels,phenotype,study)
            write(folder/'split.json',dict(indices=idx,random_split_seed=seed,trial=trial,study=study,
                assignment_sha256=sha(folder/'assignments.csv'),plan_sha256=sha(PLAN)))
            status(stage='search_training',study=study,trial=trial,seed=seed,
                completed_trials=len(rows),qualifying=len(selected),test_selection_used=True)
            cfg=cfg_for(plan,selection['recipes'][study],study,
                [8161601+100000*(study=='adhd')+100*trial+b for b in range(4)])
            allowed=np.r_[idx['train'],idx['valid']]
            result,cache=engine.train_native(full_raw[allowed],labels[allowed],idx,cfg,folder,write)
            logits,branches,check=engine.forward(full_raw[idx['test']],cache['branches'],cfg)
            np.savez_compressed(folder/'test_predictions.npz',pool_indices=idx['test'],labels=labels[idx['test']],
                logits=logits,branch_logits=branches,threshold=result['threshold'],predictions=logits>=result['threshold'])
            record=dict(result,study=study,trial=trial,random_split_seed=seed,configuration=cfg,
                test=engine.metrics(labels[idx['test']],logits,result['threshold']),
                independent_test_forward=check,test_used_for_division_selection=True,
                plan_sha256=sha(PLAN),adopted_settings_sha256=sha(OUT/'adopted_search_settings.json'),
                assignments_sha256=sha(folder/'assignments.csv'),test_predictions_sha256=sha(folder/'test_predictions.npz'))
            record['qualifies']=qualifies(record,plan)
            if record['qualifies']:
                dense,dense_branches=engine.train_dense(full_raw[allowed],labels[allowed],idx,cfg,folder,cache,write)
                values,branch_values,dense_check=engine.forward(full_raw[idx['test']],dense_branches,cfg)
                np.savez_compressed(folder/'dense_test_predictions.npz',pool_indices=idx['test'],labels=labels[idx['test']],
                    logits=values,branch_logits=branch_values,threshold=dense['threshold'],predictions=values>=dense['threshold'])
                dense.update(test=engine.metrics(labels[idx['test']],values,dense['threshold']),independent_test_forward=dense_check,
                             test_predictions_sha256=sha(folder/'dense_test_predictions.npz'))
                record['matched_dense']=dense
                del dense_branches
            write(marker,record)
            del cache
            gc.collect();torch.cuda.empty_cache()
        assert record['qualifies']==qualifies(record,plan)
        rows.append(record)
        if record['qualifies']:
            selected.append(dict(study=study,selected_round=len(selected)+1,trial=trial,
                random_split_seed=seed,folder=folder.relative_to(ROOT).as_posix(),
                test=record['test'],development=record['development'],retained_rois=record['retained_rois'],
                configuration=record['configuration'],test_selected=True))
        progress=dict(stage='search_progress',study=study,attempted_divisions=len(rows),qualifying=len(selected),
            selected=selected,all_trials=[dict(trial=r['trial'],random_split_seed=r['random_split_seed'],
                development=r['development'],test=r['test'],qualifies=r['qualifies'],retained_rois=r['retained_rois']) for r in rows],
            test_selection_used=True,interpretation=plan['interpretation'])
        write(OUT/f'{study}_progress.json',progress)
        status(stage='search_case_complete',study=study,attempted_divisions=len(rows),qualifying=len(selected),test_selection_used=True)
        print('SEARCH',study,trial,seed,'TEST',record['test'],'DEV',record['development'],
              'ROI',record['retained_rois'],'QUALIFY',len(selected),flush=True)
        if len(selected)>=5:
            break
    final=dict(passed=len(selected)==5,study=study,attempted_divisions=len(rows),selected=selected,
               target=plan['user_targets'][study],test_selected=True,interpretation=plan['interpretation'])
    write(OUT/f'{study}_selected.json',final)
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--phase',choices=['pilot','search','all'],default='all')
    args=parser.parse_args()
    if args.freeze:
        freeze();return
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selection=run_pilot(plan) if args.phase in ['pilot','all'] else json.loads((OUT/'adopted_search_settings.json').read_text(encoding='utf-8'))
    if args.phase=='pilot':
        status(stage='pilot_complete',adopted=selection['recipes'],test_evaluated=False);return
    outcomes=[run_search(study,plan,selection) for study in ['abide','adhd']]
    assert not any(p.suffix.lower() in ['.pt','.pth','.ckpt'] for p in OUT.rglob('*') if p.is_file())
    status(stage='search_complete',all_datasets_qualify=all(r['passed'] for r in outcomes),
        outcomes=outcomes,test_selection_used=True,weight_files_written=0)
    print('TARGET SEARCH COMPLETE',all(r['passed'] for r in outcomes),flush=True)


if __name__=='__main__':
    main()
