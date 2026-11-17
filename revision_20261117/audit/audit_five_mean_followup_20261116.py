"""Independently validate the original-recipe follow-up and every selected control."""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'audit'))
import audit_target_test_search_20261116 as common
from audit_target_test_search_20261116 import read,sha,write,summary,stability,audit_prediction
from select_exploratory_five_20261116 import choose_five
from data import load_data

OUT=ROOT/'five_mean_followup_20261116'
UP=ROOT/'target_test_search_20261116'


def fit_check(ref):
    file=ROOT/ref['path'];assert sha(file)==ref['sha256'];r=read(file)
    candidates=r['training_internal_candidates']
    chosen=max(range(len(candidates)),key=lambda i:(candidates[i]['oob_auc'],-candidates[i]['update']))
    assert r['selected_update']==candidates[chosen]['update'] and r['updates_run']<=r['maximum_budget']
    assert np.isfinite([h['objective_before_update'] for h in r['history']]).all()
    return dict(path=ref['path'],stage=ref['stage'],branch=ref['branch'],updates=r['updates_run'],
        selected_update=r['selected_update'],fit_seconds=r['elapsed_fit_seconds'],selected_gradient_norm=r['gradient_norm'],
        selected_training_ce=r['training_ce'],selected_training_objective=r['training_objective'])


def main():
    p=read(OUT/'plan.json');parent=read(UP/'plan.json');common.OUT=UP
    assert sha(UP/'plan.json')==p['upstream_plan_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for prior in p['prior_eligible_records']:
        folder=next((UP/'search'/prior['study']).glob(f"trial_{prior['trial']:03d}_seed_*"))
        assert sha(folder/'record.json')==prior['sha256']
    results=[];flat=[];all_fits={};checks=[]
    for study in ['abide','adhd']:
        spec=parent['dataset_specifications'][study];assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
        x,y,_,ph=load_data(spec['dataset']['path'],spec['label_column']);del x
        rows=[]
        for file in sorted((UP/'search'/study).glob('*/record.json')):
            row=common.audit_case(file.parent,study,y,ph,parent)
            assert row['configuration']['id']=='guided32_f32'
            assert row['configuration']['initialization_ridge_strengths']==[.1,1.,10.]
            assert row['configuration']['bootstrap_seeds']==[8161601+100000*(study=='adhd')+100*row['trial']+b for b in range(4)]
            for ref in row['fit_records']:
                all_fits[ref['path']]=fit_check(ref)
            if 'matched_dense' in row:
                for ref in row['matched_dense']['fit_records']:all_fits[ref['path']]=fit_check(ref)
            rows.append(row);checks.append(dict(path=file.relative_to(ROOT).as_posix(),sha256=sha(file),passed=True))
        if not rows:continue
        assert [r['trial'] for r in rows]==list(range(len(rows)))
        selection=choose_five(rows,p['user_targets'][study]);final_file=OUT/f'{study}_selected.json'
        finalized=final_file.exists();selected=[]
        if finalized:
            final=read(final_file);assert final['passed'] and final['selection']==selection and selection['feasible']
            assert read(OUT/f'{study}_frozen_five.json')['selection']==selection
            assert len(final['selected'])==5
            for end in range(5,len(rows)):
                assert not choose_five(rows[:end],p['user_targets'][study])['feasible']
            for number,entry in enumerate(final['selected'],1):
                source=ROOT/entry['source_case_folder'];original=read(source/'record.json')
                assert entry['native']==original and entry['source_record_sha256']==sha(source/'record.json')
                assert entry['selected_round']==number and entry['trial']==selection['selected_trials'][number-1]
                assert entry['selection_protocol_sha256']==sha(OUT/'plan.json') and entry['selection_unchanged_by_control']
                assert sha(ROOT/entry['split_csv'])==sha(source/'assignments.csv')==entry['split_csv_sha256']
                idx={k:np.asarray(v,dtype=int) for k,v in read(source/'split.json')['indices'].items()}
                control=ROOT/entry['control_folder'];dense=entry['matched_dense']
                file=control/'dense_test_predictions.npz';assert sha(file)==dense['test_predictions_sha256']
                audit_prediction(file,idx['test'],y,dense['test'],dense['threshold'])
                _,v=audit_prediction(control/'dense_development_predictions.npz',idx['valid'],y,dense['development'],dense['threshold'])
                grid=np.asarray(original['configuration']['threshold_probability_grid']);thresholds=np.log(grid/(1-grid))
                accuracy=[np.mean((v>=t)==y[idx['valid']]) for t in thresholds]
                chosen=min(range(len(grid)),key=lambda i:(-accuracy[i],abs(thresholds[i]),i))
                assert dense['threshold']==float(thresholds[chosen]) and not dense['threshold_calibration']['test_used']
                check=dense['independent_test_forward'];assert check['passed'] and check['maximum_logit_error']<check['tolerance']
                assert all(b['passed'] and b['deleted_fc_nan_error']<check['tolerance'] for b in check['deleted_input_boundary_checks'])
                for ref in dense['fit_records']:all_fits[ref['path']]=fit_check(ref)
                replay=entry['native_replay']
                if replay is not None:
                    assert replay['test_replay_maximum_logit_error']<1e-5 and replay['roi_exact_match']
                    assert replay['rois']==original['rois'] and replay['threshold']==original['threshold']
                    with np.load(control/'development_predictions.npz') as a,np.load(source/'development_predictions.npz') as b:
                        np.testing.assert_allclose(a['logits'],b['logits'],atol=1e-5,rtol=0)
                    with np.load(control/'training_contributions.npz') as a,np.load(source/'training_contributions.npz') as b:
                        np.testing.assert_array_equal(a['rois'],b['rois'])
                        np.testing.assert_allclose(a['roi_scores'],b['roi_scores'],atol=1e-6,rtol=1e-6)
                    for ref in replay['fit_records']:all_fits[ref['path']]=fit_check(ref)
                entry['audit']=dict(passed=True,role_counts={k:len(v) for k,v in idx.items()},
                    class_counts={k:np.bincount(y[v],minlength=2).tolist() for k,v in idx.items()})
                selected.append(entry)
        ids=set(selection['selected_trials']) if finalized else set()
        for r in rows:
            flat.append(dict(study=study,trial=r['trial'],random_split_seed=r['random_split_seed'],selected=r['trial'] in ids,
                configuration=r['configuration']['id'],retained_rois=r['retained_rois'],threshold=r['threshold'],
                development_accuracy=r['development']['accuracy'],development_auc=r['development']['auc'],
                test_accuracy=r['test']['accuracy'],test_auc=r['test']['auc'],test_precision=r['test']['precision'],
                test_recall=r['test']['recall'],test_f1=r['test']['f1'],folder=r['folder']))
        result=dict(study=study,attempted_divisions=len(rows),selection=selection,finalized=finalized,selected=selected,
            all_attempts=summary(rows,'test'),targets=p['user_targets'][study])
        if finalized:
            result.update(selected_test=summary([e['native'] for e in selected],'test'),
                selected_development=summary([e['native'] for e in selected],'development'),
                selected_matched90_test=summary([e['matched_dense'] for e in selected],'test'),
                roi_stability=stability([e['native'] for e in selected]),
                paired_test_differences={k:float(np.mean([e['native']['test'][k]-e['matched_dense']['test'][k] for e in selected])) for k in ['accuracy','auc']})
        results.append(result)
    for root in [OUT,UP]:assert not any(f.suffix.lower() in ['.pt','.pth','.ckpt'] for f in root.rglob('*') if f.is_file())
    report=dict(passed=True,complete=len(results)==2 and all(r['finalized'] for r in results),
        created_at=datetime.now().astimezone().isoformat(),selection_protocol_sha256=sha(OUT/'plan.json'),
        interpretation=p['interpretation'],test_selection_used=True,results=results,record_checks=checks,
        unique_fit_records=list(all_fits.values()),actual_unique_fit_seconds=sum(f['fit_seconds'] for f in all_fits.values()),
        actual_unique_updates=sum(f['updates'] for f in all_fits.values()),weight_files_written=0,
        retired_stronger_family=p['retired_stronger_search_transition'])
    write(OUT/'audit_report.json',report)
    if flat:
        with (OUT/'all_attempts.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(passed=True,complete=report['complete'],results=[dict(study=r['study'],attempts=r['attempted_divisions'],selection=r['selection'],finalized=r['finalized']) for r in results])))


if __name__=='__main__':main()
