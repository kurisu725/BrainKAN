import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from scipy.io import savemat
from .data import load_data, make_split, validate_split, file_hash, features
from .training import fit, predict, classification_metrics, load_checkpoint
from .explain import lime_contributions, permutation_loss
from .roi import roi_scores, select_rois, transform_scores


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def resolve_data(args):
    column = args.label_column
    if column is None:
        column = {'abide': 2, 'adhd': 3}[args.dataset]
    return load_data(args.data, column), column


def inspect(args):
    (x, y, indices, ph), column = resolve_data(args)
    validate_split(indices, len(y), y)
    report = {'file': str(Path(args.data).resolve()), 'sha256': file_hash(args.data),
              'label_column_zero_based': column, 'shape': list(x.shape),
              'classes': {str(k): int((y == k).sum()) for k in (0, 1)},
              'max_asymmetry': float(np.max(np.abs(x-x.transpose(0, 2, 1)))),
              'splits': {k: {'n': len(v), 'controls': int((y[v] == 0).sum()),
                             'patients': int((y[v] == 1).sum())} for k, v in indices.items()}}
    print(json.dumps(report, indent=2))
    if args.output:
        write_json(args.output, report)


def splits(args):
    (x, y, stored, _), column = resolve_data(args)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fingerprint = file_hash(args.data)
    for seed in args.seeds:
        idx = validate_split(make_split(y, seed), len(y), y)
        target = out / f'seed_{seed}.npz'
        if target.exists():
            raise FileExistsError(target)
        np.savez_compressed(target, **idx, data_sha256=fingerprint)
        write_json(target.with_suffix('.json'), {'seed': seed, 'data_sha256': fingerprint,
                   'label_column': column, 'protocol': 'new diagnosis-stratified 80/10/10',
                   'index_space': 'concatenated net_train, net_valid, net_test; zero-based',
                   'historical_split': False})


def save_predictions(path, ids, y, prob):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pool_index', 'label', 'p_control', 'p_patient', 'prediction'])
        for i, target, row in zip(ids, y, prob):
            writer.writerow([int(i), int(target), float(row[0]), float(row[1]), int(row.argmax())])


def run(args):
    config = vars(args).copy()
    config.pop('func', None)
    (x, y, idx, _), column = resolve_data(args)
    fingerprint = file_hash(args.data)
    if args.split_file:
        with np.load(args.split_file, allow_pickle=False) as stored:
            if 'data_sha256' not in stored or str(stored['data_sha256']) != fingerprint:
                raise ValueError('Split file must record the SHA256 of this exact MAT input')
            idx = {k: stored[k] for k in ('train', 'valid', 'test')}
    idx = validate_split(idx, len(y), y)
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config['device'].startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if args.epochs < 1 or args.batch_size < 1 or args.lime_samples < 2 or args.permutation_repeats < 1:
        raise ValueError('Invalid training/explanation counts')
    if args.explain_subjects < 0 or args.grid_size < 1 or args.spline_order < 1 or any(v < 1 for v in args.hidden):
        raise ValueError('Invalid architecture or explanation settings')
    if args.roi_source == 'train' and args.selector_checkpoint != 'final':
        raise ValueError('Training-only ROI selection requires --selector-checkpoint final')
    if args.roi_source == 'valid' and args.explainer != 'permutation':
        raise ValueError('Development-set ROI selection currently supports permutation only')
    thresholds = {'direct': args.direct_threshold, 'minmax': args.minmax_threshold,
                  'zscore': args.zscore_threshold}
    if not all(np.isfinite(v) for v in thresholds.values()):
        raise ValueError('Thresholds must be finite')
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output must be empty to avoid mixing runs: {out}')
    out.mkdir(parents=True, exist_ok=True)
    config.update(label_column=column, data_sha256=fingerprint, full_n_roi=x.shape[1],
                  data=str(Path(args.data).resolve()),
                  index_space='concatenated train/valid/test',
                  classifier_checkpoint=args.classifier_checkpoint,
                  validation_selection=('final epoch; development scores used for hyperparameter selection only'
                      if args.classifier_checkpoint == 'final' else 'max AUC; ties min cross entropy'),
                  torch_version=str(torch.__version__), python_version=platform.python_version())
    selector_config = config.copy()
    if args.selector_config:
        overrides = json.loads(Path(args.selector_config).read_text(encoding='utf-8'))
        allowed = {'epochs','batch_size','hidden','grid_size','spline_order','lr','weight_decay','lr_gamma'}
        if not isinstance(overrides, dict) or set(overrides)-allowed:
            raise ValueError('Unsupported selector configuration keys')
        selector_config.update(overrides)
    if (selector_config['epochs'] < 1 or selector_config['batch_size'] < 1
            or selector_config['grid_size'] < 1 or selector_config['spline_order'] < 1
            or any(v < 1 for v in selector_config['hidden'])):
        raise ValueError('Invalid fixed selector configuration')
    config['resolved_selector_settings'] = {k:selector_config[k] for k in
        ['epochs','batch_size','hidden','grid_size','spline_order','lr','weight_decay','lr_gamma','seed']}
    write_json(out / 'config.json', config)
    np.savez_compressed(out / 'split_indices.npz', **idx, data_sha256=fingerprint)
    tr, va, te = (idx[k] for k in ('train', 'valid', 'test'))
    all_rois = np.arange(x.shape[1])
    full_model, history = fit(x[tr], y[tr], x[va], y[va], all_rois, selector_config, out / 'full_kan.pt',
                              selection=args.selector_checkpoint)
    write_json(out / 'full_kan_history.json', history)
    # Stage 1 and explanation operate exclusively on training/validation arrays.
    results, roi_sets, checkpoints = {}, {'full': all_rois}, {'full_kan': (out/'full_kan.pt', all_rois)}
    if args.modes:
        explain = lime_contributions if args.explainer == 'lime' else permutation_loss
        explanation_ids = tr if args.roi_source == 'train' else va
        edge, metadata = explain(full_model, x[explanation_ids], y[explanation_ids], selector_config)
        metadata.update(cohort=args.roi_source, pool_indices=explanation_ids.tolist(),
                        selector_checkpoint=args.selector_checkpoint,
                        selection_rule='row plus column sum; transformed score < threshold',
                        perturbation='independent raw FC entries; LayerNorm recomputed per subject',
                        final_test_used=False)
        del full_model
        scores = roi_scores(edge)
        np.save(out / 'edge_contributions.npy', edge)
        np.savetxt(out / 'edge_contributions.csv', edge, delimiter=',')
        np.savetxt(out / 'roi_scores.csv', np.column_stack([all_rois, scores]),
                   delimiter=',', header='roi_zero_based,score', comments='')
        write_json(out/'explanation.json', metadata)
        for mode in args.modes:
            try:
                rois = select_rois(scores, mode, thresholds[mode])
            except ValueError as exc:
                results[mode] = {'status': 'no_valid_roi_subset', 'reason': str(exc)}
                continue
            roi_sets[mode] = rois
            write_json(out / f'rois_{mode}.json', {'mode': mode, 'threshold': thresholds[mode],
                       'comparison': '<', 'roi_source': args.roi_source, 'n_roi': len(rois),
                       'raw_scores': scores.tolist(), 'transformed_scores': transform_scores(scores, mode).tolist(),
                       'indices_zero_based': rois.tolist(), 'indices_one_based': (rois+1).tolist()})
            model, history = fit(x[tr], y[tr], x[va], y[va], rois, config, out/f'{mode}_kan.pt',
                                  selection=args.classifier_checkpoint)
            del model
            write_json(out/f'{mode}_kan_history.json', history)
            checkpoints[f'{mode}_kan'] = (out/f'{mode}_kan.pt', rois)
    else:
        del full_model
    for mode, rois in roi_sets.items():
        if 'mlp' in args.ablations:
            model, history = fit(x[tr], y[tr], x[va], y[va], rois, config, out/f'{mode}_mlp.pt',
                                  kind='mlp', selection=args.classifier_checkpoint)
            del model
            write_json(out/f'{mode}_mlp_history.json', history)
            checkpoints[f'{mode}_mlp'] = (out/f'{mode}_mlp.pt', rois)
    # Development scores are for selection; no test predictions are made here.
    for name, (path, rois) in checkpoints.items():
        model, ckpt = load_checkpoint(path, config['device'])
        prob = predict(model, x[va], rois, args.batch_size, config['device'])
        results[name] = {'n_roi': len(rois), 'checkpoint': path.name,
                         'checkpoint_epoch': ckpt['epoch'],
                         'checkpoint_selection': ckpt['checkpoint_selection'],
                         'validation': classification_metrics(y[va], prob),
                         'parameters': sum(p.numel() for p in model.parameters())}
        save_predictions(out/f'{name}_validation_predictions.csv', va, y[va], prob)
        del model
    if 'svm' in args.ablations:
        from sklearn.svm import SVC
        import joblib
        for mode, rois in roi_sets.items():
            a, b = features(x[tr], rois).numpy(), features(x[va], rois).numpy()
            candidates = []
            for c in (0.1, 1.0, 10.0):
                model = SVC(C=c, kernel='rbf', probability=True, random_state=args.seed).fit(a, y[tr])
                val = classification_metrics(y[va], model.predict_proba(b))
                candidates.append((val['auc'], -val['loss'], model, val))
            _, _, model, val = max(candidates, key=lambda row: row[:2])
            joblib.dump({'model': model, 'rois': rois.tolist(), 'config': config}, out/f'{mode}_svm.joblib')
            results[f'{mode}_svm'] = {'checkpoint': f'{mode}_svm.joblib',
                                     'n_roi': len(rois), 'C': model.C, 'validation': val}
            save_predictions(out/f'{mode}_svm_validation_predictions.csv', va, y[va], model.predict_proba(b))
    write_json(out/'validation_metrics.json', results)
    if not args.selection_only:
        # All prespecified fits and selection decisions are complete before testing.
        for name, result in results.items():
            if 'checkpoint' in result:
                result['test'] = evaluate_artifact(out, result['checkpoint'], x, y, te, config,
                                                    out/f'{name}_test_predictions.csv')
    write_json(out/'metrics.json', results)
    print(f'Completed. Results: {out / "metrics.json"}')


def evaluate_artifact(run_dir, checkpoint, x, y, test_ids, config, prediction_path):
    path = Path(run_dir)/checkpoint
    if path.suffix == '.joblib':
        import joblib
        saved = joblib.load(path)
        prob = saved['model'].predict_proba(features(x[test_ids], saved['rois']).numpy())
    else:
        model, saved = load_checkpoint(path, config['device'])
        prob = predict(model, x[test_ids], saved['rois'], config['batch_size'], config['device'])
    save_predictions(prediction_path, test_ids, y[test_ids], prob)
    return classification_metrics(y[test_ids], prob)


def infer(args):
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    x = np.load(args.input, allow_pickle=False)
    if x.ndim != 3 or x.shape[1] != x.shape[2] or not len(x) or not np.isfinite(x).all():
        raise ValueError('Input must be finite (subjects, full ROI, full ROI) .npy')
    if 'full_n_roi' in ckpt['config'] and x.shape[1] != ckpt['config']['full_n_roi']:
        raise ValueError('Input must use the original full atlas, not an already cropped submatrix')
    prob = predict(model, x, ckpt['rois'], args.batch_size, args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, prob, delimiter=',', header='p_control,p_patient', comments='')


def analyze(args):
    path = Path(args.input)
    if path.suffix == '.npy':
        edge = np.load(path, allow_pickle=False)
    elif args.legacy_csv:
        # Original pandas CSV: index column, 8100 feature columns, one or more runs.
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        edge = np.asarray([list(map(float, row[1:])) for row in rows[1:]]).mean(0)
        n = int(np.sqrt(len(edge)))
        edge = edge.reshape(n, n)
    else:
        edge = np.loadtxt(path, delimiter=',')
    scores = roi_scores(edge)
    result = {'source': str(path), 'source_sha256': file_hash(path),
              'usage': 'analysis only; archived scores are not training-set provenance',
              'roi_scores': scores.tolist(), 'modes': {}}
    for mode in ['direct', 'minmax', 'zscore']:
        try:
            selected = select_rois(scores, mode)
            result['modes'][mode] = {'count': len(selected), 'zero_based': selected.tolist()}
        except ValueError as e:
            result['modes'][mode] = {'error': str(e)}
    if args.thresholds:
        values = transform_scores(scores, args.mode)
        sets = [set(np.flatnonzero(values < t).tolist()) for t in args.thresholds]
        pairs = [len(a&b)/len(a|b) if a|b else 1.0 for i,a in enumerate(sets) for b in sets[i+1:]]
        counts = np.array([sum(i in s for s in sets) for i in range(len(scores))])
        result['threshold_sensitivity'] = {
            'mode': args.mode, 'thresholds': args.thresholds, 'roi_counts': [len(s) for s in sets],
            'jaccard_mean': float(np.mean(pairs)) if pairs else None,
            'jaccard_std_population': float(np.std(pairs)) if pairs else None,
            'consensus': np.flatnonzero(counts == len(sets)).tolist(),
            'stable_at_least_80_percent': np.flatnonzero(counts >= np.ceil(.8*len(sets))).tolist()}
    write_json(args.output, result)
    print(json.dumps(result['modes'], indent=2))


def demo(args):
    rng = np.random.default_rng(41)
    data = {}
    for split, n in [('train', 24), ('valid', 8), ('test', 8)]:
        y = np.arange(n) % 2
        x = rng.normal(0, .2, (n, 6, 6)).astype(np.float32)
        x = (x + x.transpose(0, 2, 1)) / 2
        x[:, 0, 1] += y * .8; x[:, 1, 0] = x[:, 0, 1]
        data['net_' + split] = x
        data['phenotype_' + split] = np.column_stack([np.arange(n), np.zeros(n), y])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, data)
    print('Synthetic smoke-test data:', args.output)


def main():
    parser = argparse.ArgumentParser(description='BrainKAN curated research package')
    commands = parser.add_subparsers(dest='command', required=True)
    def data_args(p):
        p.add_argument('--data', required=True)
        p.add_argument('--dataset', choices=['abide', 'adhd'], default='abide')
        p.add_argument('--label-column', type=int, help='zero-based override (ABIDE=2, ADHD=3)')
    p = commands.add_parser('inspect'); data_args(p)
    p.add_argument('--output'); p.set_defaults(func=inspect)
    p = commands.add_parser('splits'); data_args(p)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--output', required=True); p.set_defaults(func=splits)
    p = commands.add_parser('run'); data_args(p)
    p.add_argument('--output', required=True); p.add_argument('--split-file')
    p.add_argument('--epochs', type=int, default=50); p.add_argument('--batch-size', type=int, default=96)
    p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--hidden', nargs='+', type=int, default=[256,128,64,32,16])
    p.add_argument('--grid-size', type=int, default=5); p.add_argument('--spline-order', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lr-gamma', type=float, default=0.8, help='.8=author-confirmed original schedule; 1=no decay')
    p.add_argument('--modes', nargs='*', choices=['direct','minmax','zscore'], default=['direct','minmax','zscore'])
    p.add_argument('--explainer', choices=['lime','permutation'], default='permutation')
    p.add_argument('--roi-source', choices=['train','valid'], default='valid',
                   help='valid=author-described development selection; train=strict training-only selector')
    p.add_argument('--selector-checkpoint', choices=['final','validation'], default='final')
    p.add_argument('--classifier-checkpoint', choices=['final','validation'], default='final',
                   help='final=author-confirmed historical evaluation rule; validation=optional alternative')
    p.add_argument('--selector-config', help='JSON with fixed full-input selector settings, independent of downstream tuning')
    p.add_argument('--direct-threshold', type=float, default=0.0)
    p.add_argument('--minmax-threshold', type=float, default=0.5)
    p.add_argument('--zscore-threshold', type=float, default=0.0)
    p.add_argument('--selection-only', action='store_true', help='Never evaluate the final test cohort')
    p.add_argument('--lime-samples', type=int, default=5000)
    p.add_argument('--lime-kernel-width', type=float)
    p.add_argument('--target-class', type=int, choices=[0,1], default=1)
    p.add_argument('--explain-subjects', type=int, default=0, help='0=all training subjects; >0=approximate LIME subset')
    p.add_argument('--permutation-repeats', type=int, default=10)
    p.add_argument('--ablations', nargs='*', choices=['mlp','svm'], default=[])
    p.set_defaults(func=run)
    p = commands.add_parser('predict')
    p.add_argument('--checkpoint', required=True); p.add_argument('--input', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=96); p.set_defaults(func=infer)
    p = commands.add_parser('analyze')
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--legacy-csv', action='store_true')
    p.add_argument('--mode', choices=['direct','minmax','zscore'], default='minmax')
    p.add_argument('--thresholds', nargs='+', type=float); p.set_defaults(func=analyze)
    p = commands.add_parser('demo')
    p.add_argument('--output', required=True); p.set_defaults(func=demo)
    args = parser.parse_args()
    args.func(args)
