"""Bounded paired development comparison; never evaluates held-out test rows."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code')); sys.path.insert(0, str(ROOT/'audit'))
import target_guided_kan_20261116 as original
import site_centered_guided_kan_20261116 as centered
from data import load_data, make_split, validate_split
from residual_kan import PAIRS
from run_target_test_search_20261116 import sha, write, now, save_assignments

OUT = ROOT/'site_centered_development_20261116'
PLAN = OUT/'plan.json'


def freeze():
    if PLAN.exists(): raise FileExistsError(PLAN)
    snapshot = ROOT/'five_mean_followup_20261116/stage_freeze_20261116.json'
    assert json.loads(snapshot.read_text(encoding='utf-8'))['final_independent_audit_passed']
    parent = json.loads((ROOT/'target_test_search_20261116/plan.json').read_text(encoding='utf-8'))
    names = set(parent['source_sha256']) | {'code/site_centered_guided_kan_20261116.py',
        'audit/run_site_centered_development_20261116.py', 'audit/audit_site_centered_development_20261116.py'}
    config = dict(parent['configuration'], initialization='supervised3', width=32, precision='float32')
    plan = dict(created_at=now(), completed_stage_sha256=sha(snapshot),
        objective='Paired development comparison of global versus within-site ridge assistance to KAN coefficients.',
        configuration=config, dataset_specifications=parent['dataset_specifications'],
        phenotype_columns=parent['phenotype_columns'], dataset_roi_mass=parent['dataset_roi_mass'],
        random_seeds={'abide':[3761601,3761602,3761603], 'adhd':[4761601,4761602,4761603]},
        recipes=['global_guided32', 'within_site_guided32'], expected_cases=12,
        bootstrap_seed_rule='9161601 + 100000*(study==adhd) + 100*zero_based_round + branch',
        split_method='Diagnosis-stratified 80/10/10; all three predefined divisions reported without performance filtering.',
        method='Only initial three linear coefficient directions change: ridge on training-bootstrap features and labels centered within training site. Strengths [0.1,1,10]. Projection RMS scaling uses the original centered training design. Other PCA channels and all subsequent KAN fitting/pruning/ROI thresholding remain identical.',
        advance_rule='Finish all 12 development cases. A candidate advances only if mean development ACC and AUC are both strictly better than the paired global control for that dataset. No automatic test phase.',
        interpretation='Exploratory development on historically exposed cohorts with overlapping subjects; development cutoff-calibrated ACC is optimistic. No independent-validation claim.',
        storage='No model weights or copied FC MATs; small logs, predictions, assignments and training contribution arrays only.',
        test_evaluated=False, source_sha256={name:sha(ROOT/name) for name in sorted(names)})
    write(PLAN, plan)
    print('DEVELOPMENT PLAN FROZEN', sha(PLAN), flush=True)


def run():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    for name, digest in plan['source_sha256'].items(): assert sha(ROOT/name) == digest, name
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    completed = 0
    for study in ['abide','adhd']:
        spec = plan['dataset_specifications'][study]
        assert sha(spec['dataset']['path']) == spec['dataset']['sha256']
        matrices, labels, _, phenotype = load_data(spec['dataset']['path'], spec['label_column'])
        for round_index, seed in enumerate(plan['random_seeds'][study]):
            indices = validate_split({k:np.sort(v) for k,v in make_split(labels,seed).items()},len(labels),labels)
            split_folder = OUT/study/f'round_{round_index+1:02d}'
            split_folder.mkdir(parents=True,exist_ok=True)
            save_assignments(split_folder,indices,labels,phenotype,study)
            write(split_folder/'split.json',dict(study=study,round=round_index+1,random_split_seed=seed,
                indices=indices,assignment_sha256=sha(split_folder/'assignments.csv'),plan_sha256=sha(PLAN)))
            allowed = np.r_[indices['train'],indices['valid']]
            raw = np.ascontiguousarray(matrices[allowed][:,PAIRS[0],PAIRS[1]])
            y = labels[allowed]
            training_sites = phenotype[indices['train'], plan['phenotype_columns'][study]['site']]
            for recipe in plan['recipes']:
                folder=split_folder/recipe; folder.mkdir(exist_ok=True)
                marker=folder/'record.json'
                if marker.exists():
                    record=json.loads(marker.read_text(encoding='utf-8'))
                    assert record['plan_sha256']==sha(PLAN)
                else:
                    write(OUT/'status.json',dict(stage='development_training',pid=os.getpid(),updated_at=now(),
                        study=study,round=round_index+1,recipe=recipe,completed_cases=completed,test_evaluated=False))
                    cfg=dict(plan['configuration'],id=recipe,roi_mass=plan['dataset_roi_mass'][study],
                             bootstrap_seeds=[9161601+100000*(study=='adhd')+100*round_index+b for b in range(4)])
                    if recipe=='global_guided32':
                        result,cache=original.train_native(raw,y,indices,cfg,folder,write)
                    else:
                        result,cache=centered.train_native(raw,y,indices,cfg,folder,write,training_sites)
                    values,_,check=original.forward(raw[len(indices['train']):],cache['branches'],cfg)
                    with np.load(folder/'development_predictions.npz') as d:
                        np.testing.assert_allclose(values,d['logits'],atol=1e-4,rtol=0)
                    record=dict(result,study=study,round=round_index+1,random_split_seed=seed,recipe=recipe,
                                configuration=cfg,test_evaluated=False,independent_development_forward=check,
                                plan_sha256=sha(PLAN),development_predictions_sha256=sha(folder/'development_predictions.npz'))
                    write(marker,record); del cache
                    gc.collect(); torch.cuda.empty_cache()
                completed+=1
                print('DEVELOPMENT',study,round_index+1,recipe,json.dumps(record['development']),
                      'ROI',record['retained_rois'],flush=True)
        del matrices
    assert completed==plan['expected_cases']
    from audit_site_centered_development_20261116 import main as audit
    audit()
    write(OUT/'status.json',dict(stage='complete',updated_at=now(),completed_cases=completed,test_evaluated=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--freeze',action='store_true')
    if parser.parse_args().freeze: freeze()
    else: run()
