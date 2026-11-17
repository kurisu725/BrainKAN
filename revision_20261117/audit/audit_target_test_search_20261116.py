"""Independently audit every attempted division and export transparent summaries.

This audit only reads retained arrays/logs; it performs no model fitting and
does not change any selection. It can also summarize a still-running search.
"""
import csv
import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'target_test_search_20261116'
sys.path.insert(0,str(ROOT/'code'))
from data import load_data


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def metrics(y,values,threshold):
    prediction=values>=threshold
    p,r,f,_=precision_recall_fscore_support(y,prediction,average='macro',zero_division=0)
    return dict(accuracy=float(accuracy_score(y,prediction)),auc=float(roc_auc_score(y,values)),
                precision=float(p),recall=float(r),f1=float(f),
                accuracy_at_zero=float(accuracy_score(y,values>=0)))


def close_metrics(actual,expected):
    for key,value in actual.items():
        assert abs(value-expected[key])<1e-12,(key,value,expected[key])


def summary(rows,key):
    output={}
    for metric in ['accuracy','auc','precision','recall','f1','accuracy_at_zero']:
        values=np.asarray([r[key][metric] for r in rows])
        output[metric]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)) if len(values)>1 else 0.)
    return output


def stability(rows):
    z=np.zeros((len(rows),90))
    for i,row in enumerate(rows):
        z[i,row['rois']]=1
    overlaps=[]
    for a,b in itertools.combinations(range(len(rows)),2):
        x,y=z[a].astype(bool),z[b].astype(bool)
        intersection=int((x&y).sum());union=int((x|y).sum())
        chance=float(x.sum()*y.sum()/90)
        overlaps.append(dict(first=a+1,second=b+1,jaccard=intersection/union,
            intersection=intersection,expected_intersection_by_size=chance,
            chance_adjusted_overlap=(intersection-chance)/(min(int(x.sum()),int(y.sum()))-chance)))
    proportion=float(z.mean())
    return dict(roi_counts=z.sum(1).astype(int).tolist(),
        selection_frequency=z.sum(0).astype(int).tolist(),pairwise=overlaps,
        mean_jaccard=float(np.mean([r['jaccard'] for r in overlaps])) if overlaps else None,
        mean_chance_adjusted_overlap=float(np.mean([r['chance_adjusted_overlap'] for r in overlaps])) if overlaps else None,
        nogueira_stability=float(1-z.var(axis=0,ddof=1).mean()/(proportion*(1-proportion))) if len(rows)>1 else None,
        interpretation='Descriptive stability on overlapping, test-selected random divisions. High raw overlap is expected when many of 90 ROIs are retained.')


def audit_prediction(path,idx,labels,expected,threshold):
    with np.load(path) as data:
        np.testing.assert_array_equal(data['pool_indices'],idx)
        np.testing.assert_array_equal(data['labels'],labels[idx])
        assert np.isfinite(data['branch_logits']).all() and data['branch_logits'].shape==(4,len(idx))
        np.testing.assert_allclose(data['logits'],data['branch_logits'].mean(0),atol=1e-6,rtol=0)
        assert float(data['threshold'])==threshold
        np.testing.assert_array_equal(data['predictions'],data['logits']>=threshold)
        actual=metrics(data['labels'],data['logits'],threshold)
        close_metrics(actual,expected)
        return actual,data['logits'].copy()


def audit_case(folder,study,labels,phenotype,plan):
    record=read(folder/'record.json');split=read(folder/'split.json')
    assert record['study']==split['study']==study
    assert record['random_split_seed']==split['random_split_seed']
    assert record['plan_sha256']==split['plan_sha256']==sha(OUT/'plan.json')
    assert record['adopted_settings_sha256']==sha(OUT/'adopted_search_settings.json')
    assert sha(folder/'assignments.csv')==split['assignment_sha256']==record['assignments_sha256']
    assert sha(folder/'test_predictions.npz')==record['test_predictions_sha256']
    seed=record['random_split_seed']
    train,rest=train_test_split(np.arange(len(labels)),test_size=.2,random_state=seed,stratify=labels)
    valid,test=train_test_split(rest,test_size=.5,random_state=seed,stratify=labels[rest])
    indices={k:np.sort(v) for k,v in dict(train=train,valid=valid,test=test).items()}
    for role,idx in indices.items():
        np.testing.assert_array_equal(split['indices'][role],idx)
    with (folder/'assignments.csv').open(encoding='utf-8-sig',newline='') as handle:
        assignments=list(csv.DictReader(handle))
    assert len(assignments)==len(labels)
    columns=plan['phenotype_columns'][study]
    for role,idx in indices.items():
        rows=[r for r in assignments if r['role']==role]
        np.testing.assert_array_equal([int(r['zero_based_original_index']) for r in rows],idx)
        for row,i in zip(rows,idx):
            assert int(row['label'])==int(labels[i])
            assert float(row['subject_id'])==float(phenotype[i,columns['subject_id']])
            assert float(row['site'])==float(phenotype[i,columns['site']])
    cfg=record['configuration']
    assert cfg['roi_mass']==plan['dataset_roi_mass'][study]
    assert len(record['initializations'])==4 and record['threshold_calibration']['test_used'] is False
    assert record['test_used_for_division_selection'] is True
    for branch,(seed,initial) in enumerate(zip(cfg['bootstrap_seeds'],record['initializations'])):
        rng=np.random.default_rng(seed);y=labels[indices['train']]
        draw=np.concatenate([rng.choice(np.flatnonzero(y==c),int((y==c).sum()),replace=True) for c in [0,1]])
        draw=draw[rng.permutation(len(draw))]
        np.testing.assert_array_equal(initial['training_indices'],indices['train'][draw])
        assert not set(initial['training_indices'])&set(indices['valid'])
        assert not set(initial['training_indices'])&set(indices['test'])
    with np.load(folder/'training_contributions.npz') as contributions:
        np.testing.assert_array_equal(contributions['training_indices'],indices['train'])
        delta=contributions['signed_branch_effects']
        assert delta.shape==(4,len(indices['train']),90) and np.isfinite(delta).all()
        scores=np.sqrt(np.mean(np.mean(delta.astype(float),axis=0)**2,axis=0))
        np.testing.assert_allclose(scores,contributions['roi_scores'],atol=1e-6,rtol=1e-6)
        stored=contributions['roi_scores'].astype(float)
        order=np.argsort(-stored,kind='stable');squares=stored[order]**2
        count=int(np.searchsorted(np.cumsum(squares),cfg['roi_mass']*squares.sum(),side='left'))+1
        expected_rois=np.sort(order[:count])
        np.testing.assert_array_equal(expected_rois,contributions['rois'])
        np.testing.assert_array_equal(expected_rois,record['rois'])
        assert count==record['retained_rois']<90
    for row in record['training_coefficient_movement']:
        assert row['maximum_input_coefficient_change']>0 and row['head_coefficient_norm']>0
    for row in record['weak_function_pruning']:
        assert row['removed_input_functions']==int(4005*cfg['width']*cfg['weak_fraction'])
    assert record['physical_parameters']==sum((10*n+10)*cfg['width']+1 for n in record['physical_inputs_per_branch'])
    development,values=audit_prediction(folder/'development_predictions.npz',indices['valid'],labels,record['development'],record['threshold'])
    frozen_cfg=plan.get('configuration',plan.get('configuration_per_dataset',{}).get(study,{}))
    probabilities=np.asarray(frozen_cfg['threshold_probability_grid']);thresholds=np.log(probabilities/(1-probabilities))
    accuracies=np.asarray([np.mean((values>=t)==labels[indices['valid']]) for t in thresholds])
    chosen=min(range(len(thresholds)),key=lambda i:(-accuracies[i],abs(thresholds[i]),i))
    assert record['threshold']==float(thresholds[chosen])
    assert record['threshold_calibration']['chosen_index']==chosen
    test,_=audit_prediction(folder/'test_predictions.npz',indices['test'],labels,record['test'],record['threshold'])
    target=plan['user_targets'][study]
    assert record['qualifies']==(test['accuracy']>target['accuracy'] and test['auc']>target['auc'] and count<90)
    check=record['independent_test_forward']
    assert check['passed'] and check['maximum_logit_error']<check['tolerance']
    assert all(c['passed'] and c['deleted_fc_nan_error']<check['tolerance'] for c in check['deleted_input_boundary_checks'])
    fit_summaries=[]
    refs=list(record['fit_records'])
    if 'matched_dense' in record:
        dense=record['matched_dense']
        close_metrics(audit_prediction(folder/'dense_test_predictions.npz',indices['test'],labels,dense['test'],dense['threshold'])[0],dense['test'])
        audit_prediction(folder/'dense_development_predictions.npz',indices['valid'],labels,dense['development'],dense['threshold'])
        assert sha(folder/'dense_test_predictions.npz')==dense['test_predictions_sha256']
        refs+=dense['fit_records']
    for ref in refs:
        path=ROOT/ref['path'];assert sha(path)==ref['sha256']
        fit=read(path);candidates=fit['training_internal_candidates']
        chosen=max(range(len(candidates)),key=lambda i:(candidates[i]['oob_auc'],-candidates[i]['update']))
        assert fit['selected_update']==candidates[chosen]['update'] and fit['updates_run']<=fit['maximum_budget']
        assert fit['selected_oob_auc']==candidates[chosen]['oob_auc']
        assert len(fit['history'])==fit['updates_run'] and np.isfinite([h['objective_before_update'] for h in fit['history']]).all()
        assert np.isfinite(fit['gradient_norm'])
        fit_summaries.append(dict(stage=ref['stage'],branch=ref['branch'],updates=fit['updates_run'],
            selected_update=fit['selected_update'],fit_seconds=fit['elapsed_fit_seconds'],
            selected_training_ce=fit['training_ce'],selected_training_objective=fit['training_objective'],
            selected_gradient_norm=fit['gradient_norm'],stopped_by_patience=fit['stopped_by_patience']))
    record['audit']=dict(passed=True,fit_summaries=fit_summaries,role_counts={k:len(v) for k,v in indices.items()},
        class_counts={k:np.bincount(labels[v],minlength=2).tolist() for k,v in indices.items()},
        site_counts={k:{str(int(s)):int(np.sum(phenotype[v,columns['site']]==s))
                      for s in np.unique(phenotype[:,columns['site']])} for k,v in indices.items()})
    record['folder']=folder.relative_to(ROOT).as_posix()
    return record


def main():
    plan=read(OUT/'plan.json')
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    reports=[];flat=[];files=[]
    for study in ['abide','adhd']:
        spec=plan['dataset_specifications'][study]
        assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
        matrices,labels,_,phenotype=load_data(spec['dataset']['path'],spec['label_column'])
        del matrices
        rows=[]
        for file in sorted((OUT/'search'/study).glob('*/record.json')):
            record=audit_case(file.parent,study,labels,phenotype,plan)
            rows.append(record)
            files.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=sha(file),passed=True))
        if not rows:
            continue
        assert [r['trial'] for r in rows]==list(range(len(rows)))
        selected=[r for r in rows if r['qualifies']][:5]
        for row in rows:
            flat.append(dict(study=study,trial=row['trial'],random_split_seed=row['random_split_seed'],
                configuration=row['configuration']['id'],selected=row in selected,qualifies=row['qualifies'],
                roi_count=row['retained_rois'],threshold=row['threshold'],
                development_accuracy=row['development']['accuracy'],development_auc=row['development']['auc'],
                test_accuracy=row['test']['accuracy'],test_auc=row['test']['auc'],
                test_precision=row['test']['precision'],test_recall=row['test']['recall'],test_f1=row['test']['f1'],
                pipeline_seconds=row['elapsed_seconds'],folder=row['folder']))
        report=dict(study=study,attempted_divisions=len(rows),qualifying_divisions=sum(r['qualifies'] for r in rows),
            targets=plan['user_targets'][study],all_attempts=summary(rows,'test'),selected_count=len(selected),
            strict_targets_met=len(selected)==5,selected=selected,test_selection_used=True)
        if selected:
            report.update(selected_test=summary(selected,'test'),selected_development=summary(selected,'development'),
                selected_matched90_test=summary([r['matched_dense'] for r in selected],'test'),
                roi_stability=stability(selected),
                selected_pipeline_seconds=dict(mean=float(np.mean([r['elapsed_seconds'] for r in selected])),
                    sample_sd=float(np.std([r['elapsed_seconds'] for r in selected],ddof=1)) if len(selected)>1 else 0.),
                paired_test_differences={key:float(np.mean([r['test'][key]-r['matched_dense']['test'][key] for r in selected]))
                                        for key in ['accuracy','auc']})
        reports.append(report)
    audit=dict(passed=True,complete=len(reports)==2 and all(r['strict_targets_met'] for r in reports),
        created_at=datetime.now().astimezone().isoformat(),plan_sha256=sha(OUT/'plan.json'),
        interpretation=plan['interpretation'],test_selection_used=True,
        results=reports,record_checks=files,weight_files_written=0,
        audit_scope='Original labels, split regeneration, participants/sites, training-only bootstrap initialization, contribution-mass selection, development cutoff selection, all saved prediction metrics, scalar histories and independent-forward check records. Models are not saved/retrained by this audit.')
    assert not any(p.suffix.lower() in ['.pt','.pth','.ckpt'] for p in OUT.rglob('*') if p.is_file())
    write(OUT/'audit_report.json',audit)
    if flat:
        with (OUT/'all_attempts.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(passed=True,complete=audit['complete'],datasets=[dict(study=r['study'],attempted=r['attempted_divisions'],selected=r['selected_count']) for r in reports])))


if __name__=='__main__':
    main()
