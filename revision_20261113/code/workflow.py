"""Portable orchestration of the recorded training-only EN + fixed-grid KAN."""
from pathlib import Path
import argparse
import csv
import gc
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data import load_data, validate_split
from selection import fit_selector, rank_consensus, top_k
from storage import read_json, write_json, digest, save_npz, save_torch
from trainer import train_candidate, SafePause
from models import build_model
from run_review import input_features, predict

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['ACC', 'P', 'R', 'F1', 'AUC']


def classification_metrics(y, probabilities, predicted=None):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(y), 2) or not np.isfinite(probabilities).all():
        raise ValueError('Expected two finite class probabilities per subject')
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=2e-6, rtol=0)
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Invalid class probability range')
    predicted = probabilities.argmax(1) if predicted is None else np.asarray(predicted, dtype=int)
    return dict(zip(METRICS, map(float, [accuracy_score(y, predicted),
        precision_score(y, predicted, average='macro', zero_division=0),
        recall_score(y, predicted, average='macro', zero_division=0),
        f1_score(y, predicted, average='macro', zero_division=0), roc_auc_score(y, probabilities[:, 1])])))


def read_predictions(path):
    path = Path(path)
    if path.suffix == '.csv':
        a = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
        return (a['pool_index'].astype(int), a['label'].astype(int),
                np.column_stack([a['prob_0'], a['prob_1']]).astype('float32'), a['pred'].astype(int))
    with np.load(path) as a:
        ids = a['pool_indices'] if 'pool_indices' in a else a['indices']
        y = a['labels'] if 'labels' in a else a['y']
        if 'probabilities' in a:
            probabilities = a['probabilities']
        else:
            score = a['probability']
            probabilities = np.column_stack([1-score, score])
        pred = a['prediction'] if 'prediction' in a else probabilities.argmax(1)
        return ids.astype(int), y.astype(int), probabilities, pred.astype(int)


def write_predictions(path, indices, labels, probabilities):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pool_index', 'label', 'prob_0', 'prob_1', 'pred'])
        w.writerows((int(i), int(y), float(a), float(b), int(np.argmax([a,b])))
                    for i, y, (a,b) in zip(indices, labels, probabilities))


def summarize(rows):
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    sample_sd=float(np.std([r[k] for r in rows], ddof=1)) if len(rows)>1 else None,
                    values=[r[k] for r in rows]) for k in METRICS}


def verify_records(output=None):
    manifest = ROOT / 'MANIFEST.sha256.json'
    if manifest.exists():
        for name, expected in read_json(manifest).items():
            if digest(ROOT / name) != expected:
                raise ValueError('Changed release file: ' + name)
    studies = {s: read_json(ROOT / 'records/studies' / f'{s}.json') for s in ['abide','adhd','nyu']}
    selectors = pairs = 0
    for study in studies.values():
        if digest(ROOT / study['selector_score_file']) != study['selector_score_sha256']:
            raise ValueError('Changed selector scores')
        with np.load(ROOT / study['selector_score_file']) as z:
            all_scores = z['scores']
            assert z['partitions'].tolist() == [s['id'] for s in study['splits']]
        for s, scores in zip(study['splits'], all_scores):
            idx = validate_split(s['indices'], study['dataset']['n_subjects'])
            train = set(idx['train'])
            halves = study['half_samples'][str(s['id'])]
            assert len(halves) == 50 and scores.shape == (100,90)
            for pair in halves:
                a,b = set(pair['a']),set(pair['b'])
                assert len(a)==len(b)==len(train)//2 and not a&b
                assert a|b <= train and set(pair['omitted'])==train-a-b
                pairs += 1
            ranks = rank_consensus(scores)
            np.testing.assert_allclose(ranks, s['selection']['scores'], atol=1e-12, rtol=0)
            assert top_k(ranks,75).tolist() == s['selection']['rois']
            selectors += len(scores)
    records = read_json(ROOT / 'records/final_records.json')
    computed, cohorts = [], {}
    for r in records:
        path = ROOT / r['file']
        assert digest(path)==r['sha256'], r['file']
        ids,y,prob,pred=read_predictions(path)
        split=next(s for s in studies[r['study']]['splits'] if s['id']==r['partition'])
        assert ids.tolist()==split['indices']['test']
        key=(r['study'],r['partition'])
        if key in cohorts:
            np.testing.assert_array_equal(cohorts[key],y)
        cohorts[key]=y
        expected_pred=prob.argmax(1) if r['threshold'] is None else (prob[:,1]>=r['threshold']).astype(int)
        np.testing.assert_array_equal(pred,expected_pred)
        m=classification_metrics(y,prob,pred)
        np.testing.assert_allclose([m[k] for k in METRICS],[r['expected'][k] for k in METRICS],atol=1e-8,rtol=1e-8)
        computed.append(dict(study=r['study'],method=r['method'],partition=r['partition'],**m))
    assert len(computed)==110 and selectors==2000 and pairs==1000
    scopes=read_json(ROOT/'records/reporting_scopes.json');tables={}
    for scope, spec in scopes.items():
        study='adhd' if scope.startswith('adhd') else scope
        table={}
        for method in sorted({r['method'] for r in computed if r['study']==study}):
            rr=sorted([r for r in computed if r['study']==study and r['method']==method and r['partition'] in spec['partitions']],key=lambda r:r['partition'])
            assert len(rr)==len(spec['partitions'])
            table[method]=summarize(rr)
        tables[scope]=table
    result=dict(passed=True,final_predictions=110,half_selectors=selectors,complementary_pairs=pairs,
                metrics='fractions; P/R/F1 macro; sample SD ddof=1',tables=tables)
    if output:
        write_json(output,result)
    print(f'Verified {len(computed)} final prediction records, {selectors} half-selector rankings and {pairs} complementary pairs.')
    for scope, table in tables.items():
        m=table['brainkan']
        print(f'{scope}: BrainKAN ACC {100*m["ACC"]["mean"]:.2f} +/- {100*m["ACC"]["sample_sd"]:.2f}%; '
              f'AUC {100*m["AUC"]["mean"]:.2f} +/- {100*m["AUC"]["sample_sd"]:.2f}%')
    return result


def parameter_count(n, cfg):
    sizes=[n*n]+cfg['hidden']+[2]
    return sum(a*b*(cfg['grid_size']+cfg['spline_order']+2) for a,b in zip(sizes[:-1],sizes[1:]))


def development_choice_key(row):
    m=row['development']
    return (-m['accuracy'],-m['auc'],m['loss'],row['parameter_count'],row['candidate'])


def train(args):
    study=read_json(ROOT/'records/studies'/f'{args.study}.json')
    if digest(args.data)!=study['dataset']['sha256']:
        raise ValueError('MAT SHA256 differs from the recorded dataset; provide the retained file')
    x,y,_,_=load_data(args.data,study['label_column'])
    if len(y)!=study['dataset']['n_subjects'] or x.shape[1:]!=(90,90):
        raise ValueError('Unexpected subject count or ROI dimensions')
    device=args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly pass --device cpu for a CPU run')
    torch.set_num_threads(args.threads)
    chosen=set(map(int,args.partitions.split(','))) if args.partitions else {s['id'] for s in study['splits']}
    if not chosen or chosen-{s['id'] for s in study['splits']}:
        raise ValueError('Unknown partition identifier')
    if args.development_search and args.study!='adhd':
        raise ValueError('The archived per-partition Full KAN search is an ADHD protocol')
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/'state').mkdir(exist_ok=True)
    for d in ['selectors','runs','final_test']:
        (out/d).mkdir(exist_ok=True)
    script_hash={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    signature=dict(study=args.study,partitions=sorted(chosen),dataset_sha256=digest(args.data),
                   study_sha256=digest(ROOT/'records/studies'/f'{args.study}.json'),
                   development_search=args.development_search,device=device,source_hashes=script_hash)
    if (out/'run_plan.json').exists():
        if read_json(out/'run_plan.json')!=signature:
            raise ValueError('Output belongs to another plan/code version; use a new directory')
    else:
        if any(any((out/d).iterdir()) for d in ['selectors','runs','final_test']):
            raise ValueError('Output contains unrecognized prior files')
        write_json(out/'run_plan.json',signature)
    phash=digest(out/'run_plan.json')
    paused=lambda:(out/'PAUSE').exists()
    def pause_check():
        if paused():
            raise SafePause()
    def retire(job):
        path=out/'state/current.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['job']==job:
                path.unlink()
    def fit(s,rois,cfg,job):
        folder=out/'runs'/job
        folder.mkdir(exist_ok=True)
        tr,dv=s['indices']['train'],s['indices']['valid']
        def progress(epoch,entry,saved):
            write_json(out/'status.json',dict(stage='training',job=job,epoch=epoch,saved_epoch=saved))
        payload,prob,history=train_candidate(x[tr][:,rois][:,:,rois],y[tr],x[dv][:,rois][:,:,rois],y[dv],
            cfg,job,rois,phash,out,device=device,on_epoch=progress,pause_check=paused,interval=5)
        write_json(folder/'history.json',history)
        write_predictions(folder/'development_predictions.csv',dv,y[dv],prob)
        return payload,history[-1]['development']
    try:
        for s in study['splits']:
            p=s['id']
            if p not in chosen:
                continue
            pause_check()
            validate_split(s['indices'],len(y),y)
            folder=out/'selectors'/f'p{p}'
            folder.mkdir(exist_ok=True)
            score_rows=[]
            for pair in study['half_samples'][str(p)]:
                for side in ['a','b']:
                    pause_check()
                    path=folder/f'half_{pair["pair"]:02d}_{side}.npz'
                    ids=pair[side]
                    if not path.exists():
                        result=fit_selector(x[ids],y[ids],study['selector'])
                        save_npz(path,indices=ids,**{k:v for k,v in result.items() if k!='metadata'})
                        write_json(path.with_suffix('.json'),result['metadata'])
                    with np.load(path) as z:
                        np.testing.assert_array_equal(z['indices'],ids)
                        score_rows.append(z['scores'].copy())
                    write_json(out/'status.json',dict(stage='selecting',partition=p,completed_half_samples=len(score_rows)))
            ranks=rank_consensus(score_rows)
            rois=top_k(ranks,study['roi_budget']).tolist()
            write_json(folder/'selection.json',dict(rois=rois,scores=ranks.tolist(),fit_indices=s['indices']['train'],final_test_used=False))
            cfg=s['config']
            if args.development_search:
                best=None
                for candidate in study['full90_candidates']:
                    pause_check()
                    job=f'full90_p{p}_{candidate["id"]}'
                    marker=out/'runs'/job/'complete.json'
                    if marker.exists():
                        row=read_json(marker)
                    else:
                        payload,dev=fit(s,list(range(90)),candidate['config'],job)
                        row=dict(candidate=candidate['id'],config=candidate['config'],development=dev,
                                 parameter_count=parameter_count(90,candidate['config']))
                        if best is None or development_choice_key(row)<development_choice_key(best):
                            save_torch(out/f'full90_leader_p{p}.pt',payload)
                        write_json(marker,row)
                        retire(job)
                        del payload
                    if best is None or development_choice_key(row)<development_choice_key(best):
                        best=row
                cfg=best['config']
                write_json(folder/'full90_choice.json',best)
            job=f'en75_p{p}'
            marker=out/'runs'/job/'complete.json'
            if not marker.exists():
                payload,dev=fit(s,rois,cfg,job)
                path=out/'runs'/job/'model.pt'
                save_torch(path,{**payload,'rois':rois,'partition':p,'study':args.study,'dataset_sha256':digest(args.data)})
                write_json(marker,dict(config=cfg,rois=rois,development=dev,model_sha256=digest(path),final_test_used=False))
                retire(job)
                del payload
            else:
                r=read_json(marker)
                assert r['rois']==rois and r['config']==cfg
                assert digest(marker.parent/'model.pt')==r['model_sha256']
                retire(job)
            gc.collect()
            torch.cuda.empty_cache()
        pause_check()
        files={p.relative_to(out).as_posix():digest(p) for d in ['runs','selectors'] for p in (out/d).rglob('*') if p.is_file()}
        frozen=dict(plan_sha256=phash,files=files,final_test_used=False)
        if (out/'development_frozen.json').exists():
            assert read_json(out/'development_frozen.json')==frozen
        else:
            write_json(out/'development_frozen.json',frozen)
        # Final features/labels are first evaluated only after every requested development fit is sealed.
        for s in study['splits']:
            p=s['id']
            if p not in chosen:
                continue
            pause_check()
            ck=torch.load(out/'runs'/f'en75_p{p}'/'model.pt',map_location='cpu',weights_only=False)
            model=build_model('kan_legacy',len(ck['rois']),ck['config']).to(device)
            model.load_state_dict(ck['state'])
            ids=s['indices']['test'];rois=ck['rois']
            prob=predict(model,input_features(x[ids][:,rois][:,:,rois],'kan_legacy'),device)
            write_predictions(out/'final_test'/f'en75_p{p}.csv',ids,y[ids],prob)
            write_json(out/'final_test'/f'en75_p{p}.json',dict(metrics=classification_metrics(y[ids],prob),
                development_frozen_sha256=digest(out/'development_frozen.json')))
            del model,ck
        write_json(out/'status.json',dict(stage='complete',partitions=sorted(chosen)))
    except SafePause:
        write_json(out/'status.json',dict(stage='paused',safe_to_restart=True))
        print('Paused safely. Remove the output PAUSE file and repeat the same command to resume.')
        return 75
    return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify',help='Recompute all published metrics and EN ranks without the MAT files')
    v.add_argument('--output',type=Path)
    t=sub.add_parser('run',help='Fit training-half EN, select 75 ROIs, train KAN, then evaluate')
    t.add_argument('--study',choices=['abide','adhd','nyu'],required=True)
    t.add_argument('--data',type=Path,required=True)
    t.add_argument('--output',type=Path,required=True)
    t.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    t.add_argument('--partitions',help='Comma-separated recorded partition/repeat IDs; default all')
    t.add_argument('--development-search',action='store_true',help='ADHD: repeat the archived 12-candidate Full KAN development search')
    t.add_argument('--threads',type=int,default=4)
    a=ap.parse_args()
    if a.command=='verify':
        verify_records(a.output)
    else:
        sys.exit(train(a))
