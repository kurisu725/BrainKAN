"""Independent labels, participant roles, metrics and ROI checks for all NYU repeats."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'audit'))
from data import load_data
from audit_target_test_search_20261116 import read,write,sha,audit_prediction,summary,stability

OUT=ROOT/'target_test_search_20261116/cross_site'


def main():
    p=read(OUT/'plan.json');spec=p['dataset_specification']
    assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
    assert sha(OUT/'assignments.csv')==p['assignments_sha256']
    for path,digest in p['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    x,y,_,phenotype=load_data(spec['dataset']['path'],spec['label_column']);del x
    test=np.flatnonzero(phenotype[:,1]==11);other=np.flatnonzero(phenotype[:,1]!=11)
    train,valid=train_test_split(other,test_size=.1,stratify=y[other],random_state=p['participant_partition_seed'])
    idx={k:np.sort(v) for k,v in dict(train=train,valid=valid,test=test).items()}
    for role,indices in idx.items():
        np.testing.assert_array_equal(indices,p['indices'][role])
    with (OUT/'assignments.csv').open(encoding='utf-8-sig',newline='') as handle:
        assignments=list(csv.DictReader(handle))
    assert len(assignments)==len(y)
    for role,indices in idx.items():
        rows=[r for r in assignments if r['role']==role]
        np.testing.assert_array_equal([int(r['zero_based_original_index']) for r in rows],indices)
        for row,i in zip(rows,indices):
            assert int(row['label'])==int(y[i]) and float(row['site'])==phenotype[i,1]
            assert float(row['subject_id'])==phenotype[i,0]
    rows=[];fits=[]
    for repeat in range(p['repeats']):
        folder=OUT/f'repeat_{repeat}';r=read(folder/'record.json')
        assert r['repeat']==repeat and r['plan_sha256']==sha(OUT/'plan.json')
        assert r['all_repeats_reported'] and not r['site_used_for_selection']
        assert r['both_models_fit_before_site_forward']
        assert r['configuration']['bootstrap_seeds']==p['bootstrap_seeds'][repeat]
        with np.load(folder/'training_contributions.npz') as d:
            np.testing.assert_array_equal(d['training_indices'],idx['train'])
            effects=d['signed_branch_effects'];assert effects.shape==(4,len(idx['train']),90)
            scores=np.sqrt((effects.astype(float).mean(0)**2).mean(0))
            np.testing.assert_allclose(scores,d['roi_scores'],rtol=1e-6,atol=1e-6)
            scores=d['roi_scores'].astype(float);order=np.argsort(-scores,kind='stable')
            energy=scores[order]**2;k=int(np.searchsorted(np.cumsum(energy),.9*energy.sum()))+1
            np.testing.assert_array_equal(np.sort(order[:k]),r['native']['rois'])
        for branch,(seed,initial) in enumerate(zip(r['configuration']['bootstrap_seeds'],r['native']['initializations'])):
            labels=y[idx['train']];rng=np.random.default_rng(seed)
            draw=np.concatenate([rng.choice(np.flatnonzero(labels==c),int((labels==c).sum()),replace=True) for c in [0,1]])
            draw=draw[rng.permutation(len(draw))]
            np.testing.assert_array_equal(initial['training_indices'],idx['train'][draw])
        for variant,dev_name in [('native','development_predictions.npz'),('matched_dense','dense_development_predictions.npz')]:
            case=r[variant];file=folder/f'{variant}_site_predictions.npz'
            assert sha(file)==case['site_predictions_sha256']
            audit_prediction(file,idx['test'],y,case['test'],case['threshold'])
            _,values=audit_prediction(folder/dev_name,idx['valid'],y,case['development'],case['threshold'])
            grid=np.asarray(r['configuration']['threshold_probability_grid']);cutoffs=np.log(grid/(1-grid))
            accuracy=[np.mean((values>=c)==y[idx['valid']]) for c in cutoffs]
            chosen=min(range(len(grid)),key=lambda i:(-accuracy[i],abs(cutoffs[i]),i))
            assert case['threshold']==float(cutoffs[chosen]) and not case['threshold_calibration']['test_used']
            check=case['independent_site_forward']
            assert check['passed'] and check['maximum_logit_error']<check['tolerance']
            assert all(c['passed'] and c['deleted_fc_nan_error']<check['tolerance'] for c in check['deleted_input_boundary_checks'])
            for ref in case['fit_records']:
                file=ROOT/ref['path'];assert sha(file)==ref['sha256']
                fit=read(file);candidates=fit['training_internal_candidates']
                chosen=max(range(len(candidates)),key=lambda i:(candidates[i]['oob_auc'],-candidates[i]['update']))
                assert fit['selected_update']==candidates[chosen]['update']
                assert np.isfinite([h['objective_before_update'] for h in fit['history']]).all()
                fits.append(dict(repeat=repeat,stage=ref['stage'],branch=ref['branch'],fit_seconds=fit['elapsed_fit_seconds'],
                    updates=fit['updates_run'],selected_update=fit['selected_update']))
        rows.append(r)
    assert len(rows)==5 and sum(f['stage']=='source' for f in fits)==20
    result=dict(passed=True,plan_sha256=sha(OUT/'plan.json'),rows=rows,fit_records=fits,
        role_counts={k:len(v) for k,v in idx.items()},
        role_class_counts={k:np.bincount(y[v],minlength=2).tolist() for k,v in idx.items()},
        native=summary([r['native'] for r in rows],'test'),
        matched_dense=summary([r['matched_dense'] for r in rows],'test'),
        roi_stability=stability([r['native'] for r in rows]),
        one_site=True,historically_exposed_site=True,all_repeats_reported=True,site_used_for_selection=False,
        interpretation='Five bootstrap repeats of one historically exposed held-out NYU site with the adopted ABIDE recipe; not five independent sites and not a pristine external cohort.',weight_files_written=0)
    write(OUT/'audit_report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ['passed','role_counts','native','matched_dense']}))


if __name__=='__main__':
    main()
