"""Plan, develop, then evaluate prespecified fixed-split reviewer experiments.

No formal experiment runs on import or plan creation. The user must separately
start `run`, and `evaluate` is a separate step after all five runs are frozen.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainkan.cli import write_json, save_predictions, evaluate_artifact
from brainkan.data import load_data, validate_split, file_hash
from brainkan.training import fit, predict, classification_metrics, load_checkpoint
from brainkan.roi import select_rois, transform_scores
from brainkan.reviewer import explain_fc, compare_scores, set_stability


def sources():
    files = list((ROOT/'brainkan').glob('*.py'))+[Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def create_plan(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    data = Path(args.data).resolve()
    column = 2 if args.dataset == 'abide' else 3
    x, y, idx, _ = load_data(data, column)
    if args.split_file:
        with np.load(args.split_file, allow_pickle=False) as z:
            if str(z['data_sha256']) != file_hash(data):
                raise ValueError('Split data checksum mismatch')
            idx = {k:z[k] for k in ('train','valid','test')}
    idx = validate_split(idx, len(y), y)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('Seeds must be distinct nonnegative integers')
    if (not args.alphas or len(set(args.alphas)) != len(args.alphas)
            or any(not 0 <= a < 1 for a in args.alphas)):
        raise ValueError('Prespecify distinct attenuation factors in [0,1)')
    if args.epochs < 1 or args.repeats < 1 or not args.hidden or min(args.hidden) < 1:
        raise ValueError('Invalid training or repetition count')
    plan = {'protocol': 'new fixed-split stochastic replications; not recovery of five historical splits',
            'dataset': args.dataset, 'data': str(data), 'data_sha256': file_hash(data),
            'label_column': column, 'indices': {k:v.tolist() for k,v in idx.items()},
            'full_n_roi': x.shape[1], 'seeds': args.seeds, 'alphas': args.alphas,
            'include_symmetric_permutation': args.symmetric,
            'thresholds': {'direct':0., 'minmax':.5, 'zscore':0.},
            'config': {'epochs': args.epochs, 'batch_size':96, 'hidden':args.hidden,
                       'grid_size':5, 'spline_order':3, 'lr':.001, 'weight_decay':.0001,
                       'lr_gamma':.8, 'permutation_repeats':args.repeats},
            'roi_source':'valid', 'checkpoint_selection':'final', 'sources':sources(),
            'report_all_prespecified_arms':True, 'choose_best_seed_or_test_arm':False,
            'structural_audit':{'subjects':8, 'units':8, 'scope':'sampled development matrices, not exhaustive'},
            'estimated_explanation_dataset_passes_per_seed':1 + x.shape[1]**2*args.repeats
                + len(args.alphas)*(1+x.shape[1])
                + (1+x.shape[1]*(x.shape[1]-1)//2*args.repeats if args.symmetric else 0)}
    write_json(args.output, plan)
    print('Plan saved; no fitting or final-test prediction performed:', args.output)


def validate_plan(plan):
    if sources() != plan['sources']:
        raise ValueError('Code changed since plan creation; review and generate a new plan')
    if file_hash(plan['data']) != plan['data_sha256']:
        raise ValueError('Data changed since plan creation')
    x, y, _, _ = load_data(plan['data'], plan['label_column'])
    idx = validate_split({k:np.asarray(v, dtype=np.int64) for k,v in plan['indices'].items()},len(y),y)
    if x.shape[1] != plan['full_n_roi'] or plan['checkpoint_selection'] != 'final':
        raise ValueError('Invalid plan dimension or checkpoint rule')
    if plan['roi_source'] != 'valid' or plan['choose_best_seed_or_test_arm'] or not plan['report_all_prespecified_arms']:
        raise ValueError('This protocol reports all declared development-derived arms')
    return x, y, idx


def runtime(args):
    torch.set_num_threads(args.threads)
    device = args.device
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    return device


def develop(args):
    plan = read_json(args.plan)
    x, y, idx = validate_plan(plan)
    device = runtime(args)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'plan.json',plan)
    np.savez_compressed(out/'split_indices.npz',**idx,data_sha256=plan['data_sha256'])
    tr, va = idx['train'], idx['valid']
    # No prediction path below receives idx['test'].
    records, by_arm = [], {}
    for seed in plan['seeds']:
        folder = out/f'seed_{seed}'; folder.mkdir()
        config = {**plan['config'],'seed':seed,'device':device,'full_n_roi':x.shape[1]}
        model, history = fit(x[tr],y[tr],None,None,np.arange(x.shape[1]),config,folder/'full_kan.pt',selection='final')
        write_json(folder/'full_history.json',history)
        write_json(folder/'config.json',config)
        arms = [('entry_permutation','entry_permutation',None)]
        if plan['include_symmetric_permutation']:
            arms.append(('symmetric_permutation','symmetric_permutation',None))
        arms += [(f'attenuation_{a:g}','roi_attenuation',a) for a in plan['alphas']]
        scores_by_arm = {}
        for name, method, alpha in arms:
            target = folder/name; target.mkdir()
            scores, repeated, edges, metadata = explain_fc(model,x[va],y[va],config,method,alpha,
                audit_subjects=plan['structural_audit']['subjects'],audit_units=plan['structural_audit']['units'])
            metadata['pool_indices'] = va.tolist()
            write_json(target/'explanation.json',metadata)
            np.save(target/'repeat_roi_scores.npy',repeated)
            np.save(target/'roi_scores.npy',scores)
            if edges is not None: np.save(target/'edge_scores.npy',edges)
            scores_by_arm[name] = scores
            write_json(target/'score_uncertainty.json', {'repeat_mean':scores.tolist(),
                'repeat_sample_sd':repeated.std(0,ddof=1).tolist() if len(repeated)>1 else None,
                'scope':'permutation Monte Carlo variability conditional on this fitted model; attenuation is deterministic'})
        del model
        write_json(folder/'ranking_comparisons.json',[
            {'arm_a':a,'arm_b':b,**compare_scores(scores_by_arm[a],scores_by_arm[b])}
            for a,b in itertools.combinations(scores_by_arm,2)])
        full, _ = load_checkpoint(folder/'full_kan.pt',device)
        prob = predict(full,x[va],np.arange(x.shape[1]),config['batch_size'],device)
        records.append({'seed':seed,'arm':'full_kan','checkpoint':str((folder/'full_kan.pt').relative_to(out)),
                        'validation':classification_metrics(y[va],prob),'n_roi':x.shape[1]})
        save_predictions(folder/'full_validation.csv',va,y[va],prob)
        del full
        for name, scores in scores_by_arm.items():
            for mode, threshold in plan['thresholds'].items():
                target = folder/name/mode; target.mkdir()
                arm = f'{name}/{mode}'
                try: rois = select_rois(scores,mode,threshold)
                except ValueError as exc:
                    records.append({'seed':seed,'arm':arm,'status':'undefined_or_empty_selection','reason':str(exc)})
                    continue
                write_json(target/'rois.json',{'indices_zero_based':rois.tolist(),'indices_one_based':(rois+1).tolist(),
                    'threshold':threshold,'mode':mode,'comparison':'<','transformed_scores':transform_scores(scores,mode).tolist()})
                classifier, history = fit(x[tr],y[tr],None,None,rois,config,target/'classifier.pt',selection='final')
                write_json(target/'history.json',history)
                prob = predict(classifier,x[va],rois,config['batch_size'],device)
                save_predictions(target/'validation_predictions.csv',va,y[va],prob)
                records.append({'seed':seed,'arm':arm,'checkpoint':str((target/'classifier.pt').relative_to(out)),
                    'validation':classification_metrics(y[va],prob),'n_roi':len(rois),
                    'checkpoint_epoch':config['epochs'],'checkpoint_selection':'final'})
                by_arm.setdefault(arm,[]).append((seed,rois.tolist()))
                del classifier
        write_json(out/'development_records.json',records)
    write_json(out/'roi_seed_stability.json',{a:{'seeds':[v[0] for v in rows],
        **set_stability([v[1] for v in rows],x.shape[1])} for a,rows in by_arm.items()})
    structural_rows, ranking_rows = [], []
    for seed in plan['seeds']:
        folder=out/f'seed_{seed}'
        for path in folder.glob('*/explanation.json'):
            meta=read_json(path)
            for a in meta['structural_audits']:
                structural_rows.append({'seed':seed,'arm':path.parent.name,'unit':str(a['unit']),
                    'n_sampled_subjects':a['n'],'max_asymmetry':a['max_asymmetry'],
                    'max_diagonal_error':a['max_diagonal_error'],'out_of_range_fraction':a['out_of_range_fraction'],
                    'min_eigenvalue_symmetric_part':a['minimum_eigenvalue_of_symmetric_part'],
                    'fraction_below_tolerance':a['fraction_symmetric_part_below_tolerance']})
        ranking_rows += [{'seed':seed,**r} for r in read_json(folder/'ranking_comparisons.json')]
    write_csv(out/'structural_audit_sample.csv',structural_rows)
    write_csv(out/'ranking_comparisons.csv',ranking_rows)
    # Freeze every output before opening any final-test prediction path.
    sealed = {str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'frozen.json',{'files':sealed,'final_test_used':False,'n_runs':len(plan['seeds'])})
    print('Development complete and frozen. No test predictions computed:',out)


def evaluate(args):
    out = Path(args.experiment).resolve()
    frozen = read_json(out/'frozen.json')
    for name, digest in frozen['files'].items():
        if file_hash(out/name) != digest: raise ValueError('Frozen artifact changed: '+name)
    plan = read_json(out/'plan.json')
    x,y,idx = validate_plan(plan)
    device = runtime(args)
    destination = out/'final_test'; destination.mkdir(exist_ok=False)
    write_json(destination/'started.json',{'frozen_sha256':file_hash(out/'frozen.json')})
    results = []
    for record in read_json(out/'development_records.json'):
        row = dict(record)
        if 'checkpoint' in row:
            name = f'seed_{row["seed"]}__'+row['arm'].replace('/','__')+'.csv'
            row['test'] = evaluate_artifact(out,row['checkpoint'],x,y,idx['test'],
                {**plan['config'],'device':device},destination/name)
        results.append(row)
    write_json(destination/'all_runs.json',results)
    summary = {}
    for arm in sorted({r['arm'] for r in results}):
        rows = [r for r in results if r['arm']==arm and 'test' in r]
        metrics = {}
        for metric in ['accuracy','precision','recall','f1','auc']:
            values = [r['test'][metric] for r in rows if r['test'][metric] is not None]
            metrics[metric] = {'values':values,'n':len(values),'mean':float(np.mean(values)) if values else None,
                'std_sample':float(np.std(values,ddof=1)) if len(values)>1 else None}
        summary[arm] = {'n_completed':len(rows),'n_requested':len(plan['seeds']),
                        'complete':len(rows)==len(plan['seeds']),'metrics':metrics}
    write_json(destination/'summary.json',{'protocol':plan['protocol'], 'arms':summary,
        'inference_scope':'stochastic variability on one fixed test cohort; not five independent test cohorts',
        'p_values_computed':False,'best_test_arm_selected':False})
    write_csv(destination/'per_run_metrics.csv',[
        {'seed':r['seed'],'arm':r['arm'],'n_roi':r['n_roi'],**r['test']} for r in results if 'test' in r])
    write_csv(destination/'summary.csv',[
        {'arm':arm,'n_completed':s['n_completed'],'n_requested':s['n_requested'],
         'metric':metric,'mean':m['mean'],'std_sample':m['std_sample'],'n':m['n']}
        for arm,s in summary.items() for metric,m in s['metrics'].items()])
    print('All prespecified arms evaluated; no seed or method chosen from test:',destination)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--data',required=True);q.add_argument('--dataset',choices=['abide','adhd'],required=True)
    q.add_argument('--output',required=True);q.add_argument('--split-file');q.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    q.add_argument('--alphas',nargs='+',type=float,default=[.75,.5,0.]);q.add_argument('--symmetric',action='store_true')
    q.add_argument('--epochs',type=int,default=50);q.add_argument('--repeats',type=int,default=10)
    q.add_argument('--hidden',nargs='+',type=int,default=[256,128,64,32,16]);q.set_defaults(func=create_plan)
    q=sub.add_parser('run');q.add_argument('--plan',required=True);q.add_argument('--output',required=True);q.set_defaults(func=develop)
    q=sub.add_parser('evaluate');q.add_argument('--experiment',required=True);q.set_defaults(func=evaluate)
    for name in ['run','evaluate']:
        q=sub.choices[name];q.add_argument('--device',default='auto');q.add_argument('--threads',type=int,default=4)
    args=p.parse_args();args.func(args)


if __name__=='__main__': main()
