"""All five prescribed repeats of the adopted ABIDE recipe on held-out NYU.

The entire NYU site stays out of coefficient initialization, fitting, stopping,
ROI scoring and cutoff calibration. Repeats share participant roles and are
not five independent sites. No repeat is selected by its site performance.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'audit'))
import target_guided_kan_20261116 as engine
from run_target_test_search_20261116 import sha,write,now,save_assignments,cfg_for
from data import load_data,validate_split
from residual_kan import PAIRS

OUT=ROOT/'target_test_search_20261116/cross_site'
PLAN=OUT/'plan.json'


def freeze():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    parent=ROOT/'target_test_search_20261116/plan.json'
    adopted=ROOT/'target_test_search_20261116/adopted_search_settings.json'
    plan=json.loads(parent.read_text(encoding='utf-8'))
    selection=json.loads(adopted.read_text(encoding='utf-8'))
    spec=plan['dataset_specifications']['abide']
    assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
    matrix,labels,_,phenotype=load_data(spec['dataset']['path'],spec['label_column'])
    del matrix
    test=np.flatnonzero(phenotype[:,1]==11)
    other=np.flatnonzero(phenotype[:,1]!=11)
    train,valid=train_test_split(other,test_size=.1,stratify=labels[other],random_state=9761601)
    indices=validate_split({k:np.sort(v) for k,v in dict(train=train,valid=valid,test=test).items()},len(labels),labels)
    assert len(test)==171
    cfg=cfg_for(plan,selection['recipes']['abide'],'abide',[])
    record=dict(created_at=now(),dataset_specification=spec,indices={k:v.tolist() for k,v in indices.items()},
        parent_plan_sha256=sha(parent),adopted_settings_sha256=sha(adopted),configuration=cfg,
        heldout_site_code=11,heldout_site_name='NYU',historically_exposed_site=True,
        participant_partition_seed=9761601,non_nyu_train_fraction=.9,
        bootstrap_seeds=[[9161601+100*r+b for b in range(4)] for r in range(5)],
        repeats=5,all_repeats_reported=True,site_used_for_selection=False,
        description='One site, one fixed non-NYU training/development partition, five prescribed four-branch bootstrap repeats. All preprocessing, label-assisted initialization, functions, ROI scores, stopping and cutoff use non-NYU data. Both native and matched90 models finish fitting before each NYU forward. Every repeat is reported; no disk weights.',
        source_sha256=dict(plan['source_sha256'],**{'audit/run_target_site_20261116.py':sha(__file__)}))
    OUT.mkdir(parents=True,exist_ok=True)
    save_assignments(OUT,indices,labels,phenotype,'abide')
    record['assignments_sha256']=sha(OUT/'assignments.csv')
    write(PLAN,record)
    print('FROZEN TARGET NYU',sha(PLAN),flush=True)


def run():
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    for name,digest in p['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    spec=p['dataset_specification']
    assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
    assert sha(OUT/'assignments.csv')==p['assignments_sha256']
    matrix,labels,_,phenotype=load_data(spec['dataset']['path'],spec['label_column'])
    raw=np.ascontiguousarray(matrix[:,PAIRS[0],PAIRS[1]]);del matrix
    idx={k:np.asarray(v,dtype=int) for k,v in p['indices'].items()}
    assert np.all(phenotype[idx['test'],1]==11) and np.all(phenotype[np.r_[idx['train'],idx['valid']],1]!=11)
    allowed=np.r_[idx['train'],idx['valid']]
    rows=[]
    for repeat,seeds in enumerate(p['bootstrap_seeds']):
        folder=OUT/f'repeat_{repeat}'
        if (folder/'record.json').exists():
            record=json.loads((folder/'record.json').read_text(encoding='utf-8'))
        else:
            write(OUT/'status.json',dict(stage='training',repeat=repeat,pid=os.getpid(),updated_at=now()))
            folder.mkdir(parents=True,exist_ok=True)
            cfg=dict(p['configuration'],bootstrap_seeds=seeds)
            native,cache=engine.train_native(raw[allowed],labels[allowed],idx,cfg,folder,write)
            dense,branches=engine.train_dense(raw[allowed],labels[allowed],idx,cfg,folder,cache,write)
            record=dict(repeat=repeat,native=native,matched_dense=dense,configuration=cfg,
                all_repeats_reported=True,site_used_for_selection=False,plan_sha256=sha(PLAN),
                both_models_fit_before_site_forward=True)
            for variant,models in [('native',cache['branches']),('matched_dense',branches)]:
                values,branch_values,check=engine.forward(raw[idx['test']],models,cfg)
                filename=folder/f'{variant}_site_predictions.npz'
                cutoff=record[variant]['threshold']
                np.savez_compressed(filename,pool_indices=idx['test'],labels=labels[idx['test']],logits=values,
                    branch_logits=branch_values,threshold=cutoff,predictions=values>=cutoff)
                record[variant].update(test=engine.metrics(labels[idx['test']],values,cutoff),
                    independent_site_forward=check,site_predictions_sha256=sha(filename))
            write(folder/'record.json',record)
            del cache,branches
            gc.collect();torch.cuda.empty_cache()
        rows.append(record)
        print('NYU',repeat,'NATIVE',record['native']['test'],'90ROI',record['matched_dense']['test'],flush=True)
    assert len(rows)==5
    assert not any(p.suffix.lower() in ['.pt','.pth','.ckpt'] for p in OUT.rglob('*') if p.is_file())
    write(OUT/'status.json',dict(stage='complete',records=rows,repeats=5,all_repeats_reported=True,
        test_selection_used=False,weight_files_written=0,updated_at=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.freeze:
        freeze()
    else:
        run()


if __name__=='__main__':
    main()
