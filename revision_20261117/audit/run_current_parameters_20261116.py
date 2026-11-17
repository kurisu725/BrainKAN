"""Paired one-factor sensitivity of the current activity-entropy BrainKAN.

Nine pre-recorded configurations, ten frozen participant assignments, and no
weight files. Identical source fits are shared only when the changed factor
acts after source fitting (pruning fraction or retained contribution mass).
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'audit'))
from activity_entropy_auc_search import plain, boundary_check, verify_selected, inference_weights
from activity_entropy_auc_search_kan import fit
from bootstrap_native_kan import bootstrap_draw, deletion_contrasts, whole_output_scores, compile_branch
from conditional_site_kan import prepare, restore, cpu_export, transform
from load_selected_activity_entropy_split import load_selected_split
from residual_kan import PAIRS, retain_energy

OUT = ROOT / 'parameters_current_20261116'
PLAN = OUT / 'plan.json'
SELECTED = ROOT / 'analysis/activity_entropy_selected_supplement_report.json'
CODE = ['code/activity_entropy_auc_search.py', 'code/activity_entropy_auc_search_kan.py',
        'code/activity_entropy_kan.py', 'code/bootstrap_native_kan.py',
        'code/conditional_site_kan.py', 'code/residual_kan.py',
        'code/load_selected_activity_entropy_split.py', 'audit/conditional_site_math.py',
        'audit/full_function_math.py', 'audit/run_current_parameters_20261116.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(plain(value), indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def freeze():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    selected = json.loads(SELECTED.read_text(encoding='utf-8'))
    default = copy.deepcopy(selected['selected_five_per_dataset'][0]['source_configuration'])
    assert all(c['source_configuration'] == default for c in selected['selected_five_per_dataset'])
    settings = [('default', 'Reference', {}),
                ('width16', 'Hidden width', {'width': 16}),
                ('width64', 'Hidden width', {'width': 64}),
                ('entropy0', 'Activity entropy', {'input_activity_entropy': 0., 'head_activity_entropy': 0.}),
                ('entropy50', 'Activity entropy', {'input_activity_entropy': .5, 'head_activity_entropy': .5}),
                ('prune0', 'Weak-function fraction', {'weak_fraction': 0.}),
                ('prune20', 'Weak-function fraction', {'weak_fraction': .2}),
                ('mass90', 'Squared contribution mass', {'roi_mass': .90}),
                ('mass99', 'Squared contribution mass', {'roi_mass': .99})]
    configurations = []
    for name, factor, change in settings:
        cfg = dict(default, **change)
        cfg.update(id=name, factor=factor)
        configurations.append(cfg)
    splits = []
    for c in selected['selected_five_per_dataset']:
        entry = next(e for e in selected['selected_split_files']
                     if (e['study'], e['selected_round']) == (c['study'], c['selected_round']))
        record = json.loads((ROOT / c['checkpoint']).with_name('complete.json').read_text(encoding='utf-8'))
        prediction = (ROOT / record['native']['fits'][0]['path']).parent / 'predictions.npz'
        splits.append(dict(study=c['study'], selected_round=c['selected_round'],
                           random_split_seed=c['random_split_seed'], bootstrap_seeds=c['bootstrap_seeds'],
                           split_csv=entry['path'], split_csv_sha256=entry['sha256'],
                           original_prediction=prediction.relative_to(ROOT).as_posix(),
                           original_prediction_sha256=sha(prediction), expected_default_rois=c['rois'],
                           expected_default_metrics=c['native']))
    plan = dict(created_at=datetime.now().astimezone().isoformat(),
                objective='Replace Table III with sensitivity experiments of the current activity-entropy BrainKAN.',
                selected_report_sha256=sha(SELECTED), configurations=configurations, splits=splits,
                source_code_sha256={p: sha(ROOT / p) for p in CODE},
                cases=90, source_groups_per_split=5, actual_source_branch_fits=200,
                continuation_branch_fits=360, total_branch_fits=560,
                execution_order=['default', 'prune0', 'prune20', 'mass90', 'mass99',
                                 'entropy0', 'entropy50', 'width16', 'width64'],
                design='One factor at a time; report all nine configurations on both datasets and all five fixed divisions.',
                interpretation='Descriptive sensitivity on the already selected exploratory development divisions; not independent final tests.',
                pairing='Same train/development rows, bootstrap seeds, threshold and update/monitor budgets in every configuration. Width changes only the number of initialized/trained hidden channels.',
                source_reuse='The reference source fits are reused in RAM for five pruning/mass settings. Every continuation starts from its own copy of the appropriate source. No warm-fit reuse for changed entropy or width.',
                selection='All configurations are reported. No configuration winner or new participant split is selected; the original reference recipe remains the main result.',
                checkpoint_rule='Earliest strict maximum AUC on bootstrap-omitted training people; patience four candidates after at least 20 updates; caps source120/continuation80.',
                metric_rule='Zero logit threshold; ACC, ROC AUC, macro precision/recall/F1; mean and sample SD over the five divisions.',
                scope='Training rows alone fit normalization, initial coefficients and contribution scores. Development rows are prediction only in each new fit. Excluded-test rows are never forwarded.',
                storage='No intermediate or final weights; retain predictions, ROI scores, scalar histories, configurations and checks. Reused source fits count once in actual cost.',
                device='CUDA float64; four CPU threads; deterministic algorithms; RTX4060 Laptop GPU')
    write(PLAN, plan)
    print('FROZEN', PLAN, sha(PLAN), flush=True)


def metrics(labels, logits):
    pred = np.asarray(logits) >= 0
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='macro', zero_division=0)
    return dict(accuracy=float(accuracy_score(labels, pred)), auc=float(roc_auc_score(labels, logits)),
                precision=float(precision), recall=float(recall), f1=float(f1))


def record_fit(path, result, stage, branch):
    scalar = {k: v for k, v in result.items() if k not in ['weights', 'training_internal_candidates']}
    scalar['training_internal_candidates'] = [{k: v for k, v in row.items() if k != 'oob_logits'}
                                              for row in result['training_internal_candidates']]
    scalar.update(stage=stage, branch=branch, checkpoint_selection_uses_development=False)
    write(path, scalar)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), stage=stage, branch=branch,
                actual_updates=result['updates_run'], selected_update=result['selected_update'],
                fit_seconds=result['elapsed_fit_seconds'])


def run_split(entry, plan):
    study, number = entry['study'], entry['selected_round']
    folder = OUT / study / f'round_{number}'
    if (folder / 'complete.json').exists():
        print('ALREADY COMPLETE', study, number, flush=True)
        return json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
    folder.mkdir(parents=True, exist_ok=True)
    assert sha(ROOT / entry['split_csv']) == entry['split_csv_sha256']
    roles, metadata = load_selected_split(study, number)
    idx = {k: v['original_indices'] for k, v in roles.items()}
    n = len(idx['train'])
    raw = np.concatenate([roles['train']['x'], roles['valid']['x']])[:, PAIRS[0], PAIRS[1]]
    y = np.r_[roles['train']['y'], roles['valid']['y']]
    del roles
    original = np.load(ROOT / entry['original_prediction'])
    np.testing.assert_array_equal(original['allowed_indices'], np.r_[idx['train'], idx['valid']])
    np.testing.assert_array_equal(original['labels'], y)
    draws = [bootstrap_draw(y[:n], seed) for seed in entry['bootstrap_seeds']]
    np.savez_compressed(folder / 'bootstrap_positions.npz', **{f'branch_{b}': d for b, d in enumerate(draws)})
    write(folder / 'split.json', dict(entry, training_indices=idx['train'], development_indices=idx['valid'],
                                   excluded_test_indices=idx['test'], plan_sha256=sha(PLAN)))
    configs = {c['id']: c for c in plan['configurations']}
    source_groups = [('default', ['default', 'prune0', 'prune20', 'mass90', 'mass99']),
                     ('entropy0', ['entropy0']), ('entropy50', ['entropy50']),
                     ('width16', ['width16']), ('width64', ['width64'])]
    prepared = None
    cases = []
    sources_done = []
    for source_name, case_names in source_groups:
        cfg = configs[source_name]
        # Entropy variants use exactly the reference initialization and inputs.
        if prepared is None or prepared['width'] != cfg['width']:
            prepared = dict(width=cfg['width'], parents=[], states=[], qs=[], seconds=0.)
            tic = time.perf_counter()
            for draw in draws:
                net, state, _ = prepare(raw[draw], idx['train'][draw], cfg['width'], float(y[draw].mean()))
                prepared['parents'].append(cpu_export(net))
                prepared['states'].append(state)
                prepared['qs'].append(transform(raw, state))
                del net
            prepared['seconds'] = time.perf_counter() - tic
        source_folder = folder / 'sources' / source_name
        source_folder.mkdir(parents=True, exist_ok=True)
        unmasked, source_fits = [], []
        for branch in range(cfg['branches']):
            write(OUT / 'status.json', dict(stage='source', study=study, selected_round=number,
                                           source_group=source_name, branch=branch,
                                           completed_cases=len(list(OUT.glob('*/round_*/cases/*/complete.json'))),
                                           total_cases=90, pid=os.getpid()))
            result, _ = fit(restore(prepared['parents'][branch]), prepared['qs'][branch], y, draws[branch],
                            prepared['states'][branch], cfg['warm_updates'], cfg['warm_candidates'], cfg)
            source_fits.append(record_fit(source_folder / f'branch_{branch}.json', result, 'source', branch))
            unmasked.append(result['weights'])
            del result
        sources_done.append(dict(source_group=source_name, fits=source_fits,
                                 initialization_seconds=prepared['seconds'],
                                 initialization_shared_with_reference=(source_name in ['entropy0', 'entropy50'])))
        prune_cache = {}
        for name in case_names:
            case_cfg = configs[name]
            output = folder / 'cases' / name
            output.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            fraction = case_cfg['weak_fraction']
            if fraction not in prune_cache:
                masked, contrasts, weak = [], [], []
                for branch in range(cfg['branches']):
                    net = restore(unmasked[branch])
                    removed = net.prune_functions(prepared['states'][branch]['function_covariance'].to('cuda'), fraction)
                    contrasts.append(deletion_contrasts(net, prepared['qs'][branch][:n],
                                                        prepared['states'][branch]['feature_indices']))
                    weak.append(removed['removed_individual_curves'])
                    masked.append(cpu_export(net))
                    del net
                prune_cache[fraction] = dict(weights=masked, scores=whole_output_scores(contrasts, cfg['branches']), weak=weak)
                del contrasts
            cached = prune_cache[fraction]
            rois = retain_energy(cached['scores'], case_cfg['roi_mass'])
            values, branches, continuation, boundaries = [], [], [], []
            for branch in range(cfg['branches']):
                write(OUT / 'status.json', dict(stage='continuation', study=study, selected_round=number,
                                               configuration=name, branch=branch,
                                               completed_cases=len(list(OUT.glob('*/round_*/cases/*/complete.json'))),
                                               total_cases=90, pid=os.getpid()))
                net, state, keep = compile_branch(restore(cached['weights'][branch]), prepared['states'][branch], rois)
                result, v = fit(net, prepared['qs'][branch][:, keep].contiguous(), y, draws[branch], state,
                                case_cfg['continuation_updates'], case_cfg['continuation_candidates'], case_cfg)
                continuation.append(record_fit(output / f'continuation_{branch}.json', result, 'continuation', branch))
                item = dict(weights=inference_weights(result['weights']), state=state, branch=branch)
                boundaries.append(boundary_check(raw, item, v))
                branches.append(item)
                values.append(v)
                del result, net
            logits = np.mean(values, axis=0)
            verification = verify_selected(raw, idx, [('current_parameter_' + name, branches, logits)])
            physical_inputs = [len(b['state']['feature_indices']) for b in branches]
            stored_parameters = sum((10 * e + 10) * case_cfg['width'] + 1 for e in physical_inputs)
            active_input_functions = [int(b['weights']['function_mask'].sum()) for b in branches]
            if name == 'default':
                error = float(np.max(np.abs(logits - original['logits'])))
                assert error < 2e-8, ('reference_reproduction', study, number, error)
                np.testing.assert_array_equal(rois, entry['expected_default_rois'])
            else:
                error = None
            np.savez_compressed(output / 'predictions.npz', allowed_indices=np.r_[idx['train'], idx['valid']],
                                labels=y, logits=logits, branch_logits=np.stack(values), rois=rois,
                                roi_scores=cached['scores'])
            row = dict(study=study, selected_round=number, random_split_seed=entry['random_split_seed'],
                       configuration=name, config=case_cfg, source_group=source_name,
                       source_fits=source_fits, continuation_fits=continuation,
                       rois=rois, retained_rois=len(rois), physical_inputs_per_branch=physical_inputs,
                       physical_parameters=stored_parameters, active_input_functions_per_branch=active_input_functions,
                       weak_functions_removed_per_branch=cached['weak'], training=metrics(y[:n], logits[:n]),
                       development=metrics(y[n:], logits[n:]), threshold=0.,
                       reference_reproduction_max_logit_error=error,
                       independent_numpy_checks=verification, deleted_input_boundary_checks=boundaries,
                       post_source_pipeline_seconds=time.perf_counter()-start,
                       source_fit_seconds=sum(f['fit_seconds'] for f in source_fits),
                       continuation_fit_seconds=sum(f['fit_seconds'] for f in continuation),
                       all_configurations_reported=True, configuration_selected_using_development=False,
                       excluded_test_evaluated=False, weight_files_saved=0, plan_sha256=sha(PLAN))
            write(output / 'complete.json', row)
            cases.append(plain(row))
            print('PARAMETER CASE', study, number, name, 'ROI', len(rois), row['development'],
                  'reference_error', error, flush=True)
            del branches, values
            gc.collect()
        del prune_cache, unmasked
        gc.collect()
        torch.cuda.empty_cache()
    record = dict(study=study, selected_round=number, cases=cases, sources=sources_done,
                  complete=True, weight_files_saved=0, test_evaluated=False)
    write(folder / 'complete.json', record)
    del prepared
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.freeze:
        freeze()
        return
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    p = json.loads(PLAN.read_text(encoding='utf-8'))
    assert sha(SELECTED) == p['selected_report_sha256']
    for name, digest in p['source_code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    started = time.perf_counter()
    records = []
    with threadpool_limits(limits=4):
        for entry in p['splits']:
            records.append(run_split(entry, p))
    assert sum(len(r['cases']) for r in records) == p['cases']
    assert not any(OUT.rglob('*.pt')) and not any(OUT.rglob('*.pth'))
    write(OUT / 'status.json', dict(stage='all_90_parameter_cases_complete', cases=90,
                                   elapsed_seconds=time.perf_counter()-started,
                                   excluded_test_evaluated=False, weight_files_saved=0))
    print('ALL CURRENT PARAMETER EXPERIMENTS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
