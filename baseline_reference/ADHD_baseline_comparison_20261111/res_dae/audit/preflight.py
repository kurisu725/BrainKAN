"""Fit and verify the five training-only preprocessors before model training."""
from pathlib import Path
import sys,py_compile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from storage import read_json,write_json,digest
from experiment import get_inputs
from preprocess import apply_res_preprocessor

def main():
    assert not (ROOT/'frozen_manifest.json').exists()
    plan=read_json(ROOT/'plan.json');data=dict(np.load(ROOT/'data/adhd_fc_metadata.npz'))
    assert digest(ROOT/'data/adhd_fc_metadata.npz')==plan['data_sha256']
    checks=[]
    with threadpool_limits(limits=2):
        for sp in plan['splits']:
            tr=np.asarray(sp['indices']['train']);dv=np.asarray(sp['indices']['development']);te=np.asarray(sp['indices']['test'])
            assert (len(tr),len(dv),len(te))==(679,86,82) and len(np.unique(np.concatenate([tr,dv,te])))==847
            inputs=get_inputs(data,sp,'res_dae');s=inputs['prep']
            assert s['fit_indices']==tr.tolist() and np.isfinite(inputs['train_x']).all() and np.isfinite(inputs['dev_x']).all()
            assert inputs['train_x'].shape[1]==2002 and len(set(s['selected']))==2002
            assert np.bincount(inputs['train_y'])[0]==np.bincount(inputs['train_y'])[1]
            raw=data['phenotypes'][tr,4];valid=np.isfinite(raw)
            assert s['sex_imputation']['fit_indices']==tr[valid].tolist()
            assert s['sex_imputation']['value']==int(np.bincount(raw[valid].astype(int),minlength=2).argmax())
            transformed=apply_res_preprocessor(data['fc'][tr],data['sites'][tr],data['phenotypes'][tr,3],raw,s)
            np.testing.assert_allclose(transformed,inputs['train_x'][:len(tr)],atol=1e-6)
            alone=apply_res_preprocessor(data['fc'][dv[:1]],data['sites'][dv[:1]],data['phenotypes'][dv[:1],3],data['phenotypes'][dv[:1],4],s)
            np.testing.assert_allclose(alone,inputs['dev_x'][:1],atol=1e-6)
            for pair in s['augmentation_pairs']:
                assert pair['a'] in tr and pair['b'] in tr and data['labels'][pair['a']]==data['labels'][pair['b']]==pair['label']
            checks.append(dict(partition=sp['id'],training_only=True,balanced_training_samples=len(inputs['train_y']),
                selected_edges=len(s['selected']),sex_imputation=s['sex_imputation'],preprocessor_sha256=digest(ROOT/'runs'/f'partition_{sp["id"]}_res_dae_preprocessor.pt')))
            print('Validated preprocessor',sp['id'],flush=True)
    write_json(ROOT/'audit/preprocessing_tests.json',dict(passed=True,partitions=checks,all_training_only=True,heldout_transform_batch_independent=True))
    for name,h in read_json(ROOT/'previous_artifacts.json').items():assert digest(name)==h,name
    for p in list((ROOT/'code').rglob('*.py'))+list((ROOT/'audit').glob('*.py')):py_compile.compile(str(p),doraise=True)
    print('Preprocessing and source checks passed. Run resume test, then seal the plan.',flush=True)

if __name__=='__main__':main()
