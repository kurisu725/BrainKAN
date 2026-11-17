"""Continue the original guided32 trials under explicit five-mean stopping.

Original trial records retain their original training-plan hashes. A separate
selection protocol records this adaptive, test-selected follow-up. The old
runner's writer is wrapped so a complete trial stops before another starts.
No subject assignments, predictions, cutoffs or old records are changed.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'audit'))
import run_target_test_search_20261116 as upstream
import target_guided_kan_20261116 as engine
from select_exploratory_five_20261116 import choose_five
from data import load_data
from residual_kan import PAIRS

OUT=ROOT/'five_mean_followup_20261116';PLAN=OUT/'plan.json'
write=upstream.write;sha=upstream.sha;now=upstream.now


def records(study):
    return [json.loads(file.read_text(encoding='utf-8')) for file in sorted((upstream.OUT/'search'/study).glob('*/record.json'))]


def freeze():
    if PLAN.exists():raise FileExistsError(PLAN)
    parent=json.loads(upstream.PLAN.read_text(encoding='utf-8'))
    settings=upstream.OUT/'adopted_search_settings.json'
    adopted=json.loads(settings.read_text(encoding='utf-8'))
    assert all(adopted['recipes'][s]['id']=='guided32_f32' for s in ['abide','adhd'])
    sources=dict(parent['source_sha256'])
    for name in ['code/data.py','code/select_exploratory_five_20261116.py','audit/run_five_mean_followup_20261116.py']:
        sources[name]=sha(ROOT/name)
    prior=[dict(study=s,trial=r['trial'],sha256=sha(upstream.OUT/'search'/s/f"trial_{r['trial']:03d}_seed_{r['random_split_seed']}"/'record.json')) for s in ['abide','adhd'] for r in records(s)]
    write(PLAN,dict(created_at=now(),upstream_plan_sha256=sha(upstream.PLAN),adopted_settings_sha256=sha(settings),
        source_sha256=sources,user_targets=parent['user_targets'],criterion='Strict arithmetic means across five complete random divisions; individual rows may be lower.',
        selection='Select five full divisions by maximum minimum target-normalized test ACC/AUC means, then maximum ACC+AUC. Stop a dataset immediately after the first complete new record meeting both mean targets. The 24 existing original guided32 ABIDE trials remain eligible.',
        interpretation='Adaptive exploratory test-selected results. The weaker stronger-regularization candidate and its complete test history are retained separately. Report all completed and partial attempts, not only the five selected. No pristine-independent-test claim.',
        data_and_training='Use the upstream recorded diagnosis-stratified80/10/10 seeds, bootstrap seeds, train-only initialization and contribution rules, development-only cutoff, original guided32 f32 recipe and complete KAN fitting without changes.',
        prior_eligible_records=prior,initial_eligible_completed=len(prior),
        retired_stronger_search_transition=json.loads((ROOT/'mean_goal_search_20261116/transition_to_original_initialization.json').read_text(encoding='utf-8')),
        controls='Freeze the selected five before matched90 controls. Existing complete controls can be reused. Otherwise replay native training exactly in RAM, verify its predictions and ROIs, then fit the shared-source matched90 continuation. Never revise the five based on controls.',
        storage='No weights or copied FC datasets. Lightweight assignments, predictions, scalar logs and original record references only.',
        manuscript_policy='Do not write to the delivered manuscript sources.',test_selection_used=True))
    print('FROZEN ORIGINAL-RECIPE FIVE-MEAN FOLLOWUP',sha(PLAN),flush=True)


class MeanReady(Exception):pass


def train_controls(study,selected,protocol):
    spec=json.loads(upstream.PLAN.read_text(encoding='utf-8'))['dataset_specifications'][study]
    assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
    matrix,labels,_,_=load_data(spec['dataset']['path'],spec['label_column'])
    raw=np.ascontiguousarray(matrix[:,PAIRS[0],PAIRS[1]]);del matrix
    all_rows=records(study);output=[]
    for number,trial in enumerate(selected['selected_trials'],1):
        row=all_rows[trial];source=upstream.OUT/'search'/study/f"trial_{trial:03d}_seed_{row['random_split_seed']}"
        folder=OUT/'selected'/study/f'round_{number}';marker=folder/'record.json'
        if marker.exists():
            entry=json.loads(marker.read_text(encoding='utf-8'))
            assert entry['source_record_sha256']==sha(source/'record.json')
            output.append(entry);continue
        folder.mkdir(parents=True,exist_ok=True)
        write(OUT/f'{study}_status.json',dict(stage='matched_control',selected_round=number,trial=trial,updated_at=now(),pid=os.getpid()))
        idx={k:np.asarray(v,dtype=int) for k,v in json.loads((source/'split.json').read_text(encoding='utf-8'))['indices'].items()}
        allowed=np.r_[idx['train'],idx['valid']];cfg=row['configuration']
        if 'matched_dense' in row:
            dense=row['matched_dense'];control_folder=source;replay=None
        else:
            replay,cache=engine.train_native(raw[allowed],labels[allowed],idx,cfg,folder,write)
            np.testing.assert_array_equal(replay['rois'],row['rois']);assert replay['threshold']==row['threshold']
            values,_,check=engine.forward(raw[idx['test']],cache['branches'],cfg)
            with np.load(source/'test_predictions.npz') as previous:
                error=float(np.max(np.abs(values-previous['logits'])))
                np.testing.assert_allclose(values,previous['logits'],atol=1e-5,rtol=0)
            replay.update(test_replay_maximum_logit_error=error,independent_test_forward=check,roi_exact_match=True)
            dense,branches=engine.train_dense(raw[allowed],labels[allowed],idx,cfg,folder,cache,write)
            values,branch_values,check=engine.forward(raw[idx['test']],branches,cfg)
            filename=folder/'dense_test_predictions.npz'
            np.savez_compressed(filename,pool_indices=idx['test'],labels=labels[idx['test']],logits=values,
                branch_logits=branch_values,threshold=dense['threshold'],predictions=values>=dense['threshold'])
            dense.update(test=engine.metrics(labels[idx['test']],values,dense['threshold']),independent_test_forward=check,
                         test_predictions_sha256=sha(filename))
            control_folder=folder;del cache,branches;gc.collect();torch.cuda.empty_cache()
        split_folder=OUT/'selected_splits'/study/f'round_{number:02d}';split_folder.mkdir(parents=True,exist_ok=True)
        for filename in ['assignments.csv','split.json']:
            shutil.copyfile(source/filename,split_folder/filename)
        entry=dict(study=study,selected_round=number,trial=trial,random_split_seed=row['random_split_seed'],
            native=row,matched_dense=dense,native_replay=replay,source_case_folder=source.relative_to(ROOT).as_posix(),
            source_record_sha256=sha(source/'record.json'),control_folder=control_folder.relative_to(ROOT).as_posix(),
            split_csv=(split_folder/'assignments.csv').relative_to(ROOT).as_posix(),
            split_csv_sha256=sha(split_folder/'assignments.csv'),selection_protocol_sha256=sha(PLAN),
            selection_unchanged_by_control=True)
        write(marker,entry);output.append(entry)
        print('FOLLOWUP CONTROL',study,number,'native',row['test'],'matched90',dense['test'],flush=True)
    return output


def run(study):
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    assert sha(upstream.PLAN)==p['upstream_plan_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    original=json.loads(upstream.PLAN.read_text(encoding='utf-8'))
    settings=json.loads((upstream.OUT/'adopted_search_settings.json').read_text(encoding='utf-8'))
    def checkpoint():
        rows=records(study);result=choose_five(rows,p['user_targets'][study])
        write(OUT/f'{study}_progress.json',dict(study=study,updated_at=now(),selection=result,
            attempted_divisions=len(rows),test_selection_used=True,
            all_trials=[dict(trial=r['trial'],random_split_seed=r['random_split_seed'],test=r['test'],development=r['development'],retained_rois=r['retained_rois']) for r in rows]))
        return result
    selection=checkpoint()
    if not selection['feasible']:
        original_write=upstream.write
        def intercept(path,value):
            path=Path(path)
            if path==upstream.OUT/'status.json':
                original_write(OUT/f'{study}_training_status.json',value)
            else:
                original_write(path,value)
            if path.name=='record.json' and path.parent.parent==upstream.OUT/'search'/study:
                result=checkpoint()
                print('FIVE-MEAN PROGRESS',study,result,flush=True)
                if result['feasible']:raise MeanReady()
        upstream.write=intercept
        try:
            upstream.run_search(study,original,settings)
        except MeanReady:
            pass
        finally:
            upstream.write=original_write
            gc.collect();torch.cuda.empty_cache()
        selection=checkpoint()
    if not selection['feasible']:
        write(OUT/f'{study}_status.json',dict(stage='seed_list_exhausted',selection=selection,goal_complete=False));return
    write(OUT/f'{study}_frozen_five.json',dict(study=study,selection=selection,frozen_before_control=True,
        selection_protocol_sha256=sha(PLAN),created_at=now()))
    selected=train_controls(study,selection,p)
    final=dict(passed=True,study=study,selection=selection,selected=selected,attempted_divisions=len(records(study)),
        test_selection_used=True,interpretation=p['interpretation'],selection_protocol_sha256=sha(PLAN))
    write(OUT/f'{study}_selected.json',final)
    write(OUT/f'{study}_status.json',dict(stage='complete',selection=selection,updated_at=now(),weight_files_written=0))
    print('FIVE-MEAN DATASET COMPLETE',study,selection,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--study',choices=['abide','adhd']);args=parser.parse_args()
    if args.freeze:freeze();return
    if not args.study:parser.error('--study is required for training')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run(args.study)


if __name__=='__main__':main()
