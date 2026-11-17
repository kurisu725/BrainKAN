"""Train-only within-site assistance to the same native KAN pipeline.

Site information changes the three initial coefficient directions only. It is
not an inference input, ROI ranking method, or separate final classifier.
"""
import time
import numpy as np
import torch
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
import target_guided_kan_20261116 as base


def centered_design(z, y, sites):
    design = z.copy()
    response = y.astype(float).copy()
    for site in np.unique(sites):
        chosen = sites == site
        design[chosen] -= design[chosen].mean(axis=0)
        response[chosen] -= response[chosen].mean()
    return design, response


@torch.no_grad()
def initialize(raw, labels, indices, sites, cfg):
    assert len(raw) == len(labels) == len(indices) == len(sites)
    net, state, _ = base.prepare(raw, indices, cfg['width'], float(labels.mean()))
    scaler = state['scaler']
    z = np.clip((raw - scaler['mean']) / scaler['std'], -3, 3) - scaler['clip_center']
    design, response = centered_design(z, labels, sites)
    assert np.dot(response, response) > 0
    x = torch.as_tensor(design, dtype=torch.float64, device='cuda')
    target = torch.as_tensor(response, dtype=x.dtype, device=x.device)
    original = torch.as_tensor(z, dtype=x.dtype, device=x.device)
    gram = x @ x.T / x.shape[1]
    identity = torch.eye(len(x), dtype=x.dtype, device=x.device)
    scales = []; errors = []
    with threadpool_limits(limits=4):
        cpu_gram = design @ design.T / design.shape[1]
        for channel, strength in enumerate(cfg['initialization_ridge_strengths']):
            direction = x.T @ torch.linalg.solve(gram + strength * identity, target) / x.shape[1]
            scale = (original @ direction).square().mean().sqrt().clamp_min(.05)
            net.first[:, 0, channel].copy_(direction / scale)
            reference = design.T @ solve(cpu_gram + strength * np.eye(len(design)), response,
                                         assume_a='pos') / design.shape[1]
            expected = reference / max(float(np.sqrt(np.mean((z @ reference) ** 2))), .05)
            error = float(np.max(np.abs(expected - (direction / scale).cpu().numpy())))
            assert error < 1e-7, error
            scales.append(float(scale)); errors.append(error)
    net.anchor.copy_(net.first.detach())
    groups = []
    for site in np.unique(sites):
        chosen = sites == site
        groups.append(dict(site=float(site), bootstrap_rows=int(chosen.sum()),
                           class_counts=np.bincount(labels[chosen], minlength=2).tolist(),
                           centered_response_mean=float(response[chosen].mean()),
                           maximum_centered_feature_mean=float(np.max(np.abs(design[chosen].mean(0))))))
    info = dict(kind='Train-only within-site centered ridge KAN coefficient initialization',
                training_people=len(labels), training_indices=list(map(int, indices)),
                training_sites=sites.tolist(), site_groups=groups, ridge_strengths=cfg['initialization_ridge_strengths'],
                projection_scales=scales, independent_cpu_maximum_coefficient_errors=errors,
                independent_cpu_tolerance=1e-7, supervised_channels=3,
                separate_final_classifier=False, roi_ranking_or_selection=False, sites_used_at_inference=False)
    net = net.to(dtype=torch.float32)
    state['function_covariance'] = state['function_covariance'].to(dtype=torch.float32)
    state['coefficient_initialization'] = info
    return net, state, info


def train_native(raw, labels, indices, cfg, folder, write, training_sites):
    """The caller supplies training sites only; no held-out rows are accepted."""
    n = len(indices['train'])
    assert len(raw) == n + len(indices['valid']) == len(labels)
    assert len(training_sites) == n and cfg['precision'] == 'float32'
    draws = [base.bootstrap_draw(labels[:n], seed) for seed in cfg['bootstrap_seeds']]
    sources = []; states = []; qs = []; effects = []; refs = []; initializations = []; pruning = []; movement = []
    start = time.perf_counter()
    for branch, draw in enumerate(draws):
        net, state, info = initialize(raw[draw], labels[draw], indices['train'][draw], training_sites[draw], cfg)
        initializations.append(info); q = base.basis(raw, state, cfg)
        original = net.first.detach().cpu().clone()
        result, _ = base.fit(net, q, labels, draw, state, cfg['warm_updates'], cfg['warm_candidates'], cfg)
        changed = float((result['weights']['first'] - original).abs().max())
        assert changed > 0 and bool(result['weights']['head'].abs().sum() > 0)
        movement.append(dict(branch=branch, maximum_input_coefficient_change=changed,
                             head_coefficient_norm=float(result['weights']['head'].norm())))
        refs.append(base.record_fit(folder/'source'/f'branch_{branch}.json', result, 'source', branch))
        net = base.restored(result['weights'], cfg)
        weak = net.prune_functions(state['function_covariance'].to('cuda'), cfg['weak_fraction'])
        assert weak['removed_individual_curves'] == int(4005 * cfg['width'] * cfg['weak_fraction'])
        pruning.append(dict(branch=branch, removed_input_functions=weak['removed_individual_curves'],
                            total_input_functions=4005 * cfg['width']))
        effects.append(base.deletion_contrasts(net, q[:n], state['feature_indices']))
        sources.append(base.cpu_export(net)); states.append(state); qs.append(q)
        del net, result, original
    scores = base.whole_output_scores(effects, cfg['branches'])
    rois, cutoff = base.threshold_rois(scores, dict(family='mass', value=cfg['roi_mass']))
    np.savez_compressed(folder/'training_contributions.npz', training_indices=indices['train'],
                        roi_scores=scores, signed_branch_effects=np.stack(effects), rois=rois)
    branches = []; values = []
    for branch in range(cfg['branches']):
        child, state, keep = base.compile_branch(base.restored(sources[branch], cfg), states[branch], rois)
        child = child.to(dtype=qs[branch].dtype)
        result, logits = base.fit(child, qs[branch][:, keep].contiguous(), labels, draws[branch], state,
                                  cfg['continuation_updates'], cfg['continuation_candidates'], cfg)
        refs.append(base.record_fit(folder/'native'/f'branch_{branch}.json', result, 'native', branch))
        branches.append(dict(weights=base.inference_weights(result['weights']), state=state))
        values.append(logits[n:]); del child, result
    mean = np.mean(values, axis=0)
    threshold, calibration = base.calibration(labels[n:], mean, cfg['threshold_probability_grid'])
    np.savez_compressed(folder/'development_predictions.npz', pool_indices=indices['valid'], labels=labels[n:],
                        logits=mean, branch_logits=np.stack(values), threshold=threshold, predictions=mean>=threshold)
    record = dict(development=base.metrics(labels[n:], mean, threshold), threshold=threshold,
                  threshold_calibration=calibration, rois=rois.tolist(), retained_rois=len(rois), roi_threshold=cutoff,
                  initializations=initializations, fit_records=refs,
                  physical_parameters=sum((10*len(b['state']['feature_indices'])+10)*cfg['width']+1 for b in branches),
                  physical_inputs_per_branch=[len(b['state']['feature_indices']) for b in branches],
                  weak_function_pruning=pruning, training_coefficient_movement=movement,
                  elapsed_seconds=time.perf_counter()-start, weight_files_written=0)
    write(folder/'development_complete.json', record)
    return record, dict(branches=branches, sources=sources, states=states, qs=qs, draws=draws)
