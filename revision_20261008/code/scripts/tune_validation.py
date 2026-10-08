"""Search on development data, freeze a winner, then explicitly evaluate it once."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainkan.cli import write_json, evaluate_artifact
from brainkan.data import load_data, validate_split, file_hash
import numpy as np
import torch

ALLOWED = {'epochs', 'batch_size', 'hidden', 'grid_size', 'spline_order', 'lr',
           'weight_decay', 'lr_gamma', 'modes', 'direct_threshold', 'minmax_threshold',
           'zscore_threshold'}


def choose_winner(candidates):
    """Only development AUC/loss enter selection; original order breaks exact ties."""
    eligible = [c for c in candidates if c['validation']['auc'] is not None
                and np.isfinite(c['validation']['auc']) and np.isfinite(c['validation']['loss'])]
    if not eligible:
        raise ValueError('No candidate has a defined development AUC/loss')
    return max(eligible, key=lambda c: (c['validation']['auc'], -c['validation']['loss']))


def search(args):
    grid = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    if not isinstance(grid, list) or not grid:
        raise ValueError('Candidates must be a nonempty JSON list')
    for c in grid:
        if not isinstance(c, dict) or set(c)-ALLOWED:
            raise ValueError('Unsupported candidate keys; data splits and ROI source must stay fixed')
        if not c.get('modes') or any(m not in ('direct','minmax','zscore') for m in c['modes']):
            raise ValueError('Each candidate must declare nonempty direct/minmax/zscore modes')
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'candidate_plan.json', grid)
    selector = {'epochs':50,'batch_size':96,'hidden':[256,128,64,32,16],
                'grid_size':5,'spline_order':3,'lr':.001,'weight_decay':.0001,'lr_gamma':.8}
    # Freeze stage one before observing any development scores, even while stage two is tuned.
    selector.update({k:v for k,v in grid[0].items() if k in selector})
    if args.selector_config:
        selector = json.loads(Path(args.selector_config).read_text(encoding='utf-8'))
    write_json(out/'fixed_selector.json',selector)
    if args.roi_source == 'train':
        def roi_rule(c):
            return (c['modes'],c.get('direct_threshold',0.0),c.get('minmax_threshold',.5),c.get('zscore_threshold',0.0))
        if len(grid[0]['modes'])!=1 or any(roi_rule(c)!=roi_rule(grid[0]) for c in grid):
            raise ValueError('Strict training-only ROI protocol requires ONE prespecified mode and threshold, unchanged across downstream candidates')
    base = [sys.executable, str(ROOT/'run.py'), 'run', '--data', str(Path(args.data).resolve()),
            '--dataset', args.dataset, '--roi-source', args.roi_source, '--explainer', 'permutation',
            '--selector-checkpoint', 'final', '--selection-only', '--seed', str(args.seed),
            '--device', args.device, '--threads', str(args.threads),
            '--permutation-repeats', str(args.permutation_repeats),
            '--selector-config',str(out/'fixed_selector.json')]
    if args.split_file:
        base += ['--split-file', str(Path(args.split_file).resolve())]
    records=[]
    for i, settings in enumerate(grid):
        run_dir=out/f'candidate_{i:03d}'
        command=base+['--output', str(run_dir)]
        for key,value in settings.items():
            command += ['--'+key.replace('_','-')]
            command += list(map(str, value)) if isinstance(value,list) else [str(value)]
        subprocess.run(command, cwd=ROOT, check=True)
        metrics=json.loads((run_dir/'validation_metrics.json').read_text(encoding='utf-8'))
        for mode in settings['modes']:
            name=f'{mode}_kan'
            if name not in metrics:
                continue  # Empty ROI selection is recorded in the run, never repaired using test data.
            record=metrics[name].copy()
            record.update(run_dir=run_dir.name, model=name, candidate_index=i)
            records.append(record)
    write_json(out/'validation_candidates.json', records)
    winner=choose_winner(records)
    run_dir=out/winner['run_dir']
    winner.update(selection_rule='maximum development AUC; minimum loss; first declared candidate',
                  roi_source=args.roi_source, final_test_used=False,
                  checkpoint_sha256=file_hash(run_dir/winner['checkpoint']),
                  config_sha256=file_hash(run_dir/'config.json'),
                  split_sha256=file_hash(run_dir/'split_indices.npz'))
    write_json(out/'selected_model.json', winner)
    print('Frozen winner:', out/'selected_model.json')
    print('No final-test predictions were computed. Use the evaluate subcommand after freezing this search.')


def evaluate(args):
    selection=Path(args.selection).resolve()
    winner=json.loads(selection.read_text(encoding='utf-8'))
    run_dir=selection.parent/winner['run_dir']
    for name,key in [(winner['checkpoint'],'checkpoint_sha256'),('config.json','config_sha256'),
                     ('split_indices.npz','split_sha256')]:
        if file_hash(run_dir/name)!=winner[key]:
            raise ValueError(f'Frozen artifact changed: {name}')
    config=json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    if not config['selection_only']:
        raise ValueError('Candidate was not produced in selection-only mode')
    data=args.data or config['data']
    if file_hash(data)!=config['data_sha256']:
        raise ValueError('Input MAT differs from the frozen development search')
    x,y,_,_=load_data(data,config['label_column'])
    with np.load(run_dir/'split_indices.npz',allow_pickle=False) as stored:
        idx=validate_split({k:stored[k] for k in ('train','valid','test')},len(y),y)
    config['device']=args.device
    torch.set_num_threads(args.threads)
    destination=selection.parent/'final_test'
    destination.mkdir()  # Refuse a second evaluation into the same experiment record.
    write_json(destination/'evaluation_started.json',{'selected_model_sha256':file_hash(selection)})
    result=evaluate_artifact(run_dir,winner['checkpoint'],x,y,idx['test'],config,
                             destination/'predictions.csv')
    write_json(destination/'metrics.json', {'model':winner['model'], 'test':result,
                 'n_roi':winner['n_roi'], 'selection_sha256':file_hash(selection),
                 'data_sha256':config['data_sha256'], 'roi_source':winner['roi_source']})
    print('Final test:',destination/'metrics.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('search')
    s.add_argument('--data',required=True);s.add_argument('--dataset',choices=['abide','adhd'],default='abide')
    s.add_argument('--candidates',required=True);s.add_argument('--output',required=True)
    s.add_argument('--split-file');s.add_argument('--seed',type=int,default=0)
    s.add_argument('--selector-config',help='Prespecified full-input training configuration JSON')
    s.add_argument('--roi-source',choices=['train','valid'],default='valid')
    s.add_argument('--permutation-repeats',type=int,default=10)
    s.add_argument('--device',default='auto');s.add_argument('--threads',type=int,default=4)
    s.set_defaults(func=search)
    s=sub.add_parser('evaluate');s.add_argument('--selection',required=True);s.add_argument('--data')
    s.add_argument('--device',default='cpu');s.add_argument('--threads',type=int,default=4)
    s.set_defaults(func=evaluate)
    args=p.parse_args();args.func(args)


if __name__=='__main__':main()
