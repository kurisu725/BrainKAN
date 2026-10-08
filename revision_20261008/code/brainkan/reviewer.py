"""FC-only reviewer experiments; no raw time-series processing or clinical claims."""
import itertools
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import log_loss
from .training import predict
from .roi import roi_scores


def structure_report(x, tolerance=1e-5):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != x.shape[2] or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty (subjects, ROI, ROI) matrices')
    symmetric = (x + x.transpose(0, 2, 1)) / 2
    eigen = np.linalg.eigvalsh(symmetric)[:, 0]
    return {'n': len(x), 'tolerance': tolerance,
            'max_asymmetry': float(np.max(np.abs(x-x.transpose(0, 2, 1)))),
            'max_diagonal_error': float(np.max(np.abs(np.diagonal(x, axis1=1, axis2=2)-1))),
            'out_of_range_fraction': float(np.mean(np.abs(x) > 1+tolerance)),
            'minimum_eigenvalue_of_symmetric_part': float(eigen.min()),
            'fraction_symmetric_part_below_tolerance': float(np.mean(eigen < -tolerance))}


def require_correlations(x):
    report = structure_report(x)
    if (report['max_asymmetry'] > report['tolerance'] or
            report['max_diagonal_error'] > report['tolerance'] or
            report['out_of_range_fraction'] or report['fraction_symmetric_part_below_tolerance']):
        raise ValueError('ROI attenuation requires raw correlation matrices; see structure_report')
    return report


def attenuate_roi(x, roi, alpha):
    """D C D + I - D^2; raw FC in, raw FC out, before subject LayerNorm."""
    if not 0 <= alpha <= 1 or not 0 <= roi < x.shape[1]:
        raise ValueError('Require 0 <= alpha <= 1 and a valid ROI')
    result = x.copy()
    result[:, roi, :] *= alpha
    result[:, :, roi] *= alpha
    result[:, roi, roi] += 1-alpha**2
    return result


def permute_entry(x, i, j, order, symmetric=False):
    result = x.copy()
    result[:, i, j] = x[order, i, j]
    if symmetric:
        result[:, j, i] = x[order, j, i]
    return result


def loss_on(model, x, y, config):
    return float(log_loss(y, predict(model, x, np.arange(x.shape[1]),
                                    config['batch_size'], config['device']), labels=[0, 1]))


def explain_fc(model, x, y, config, method, alpha=None, audit_subjects=8, audit_units=8):
    """Return ROI scores, repeat-level scores, metadata and bounded structural audits.

    Permutation units are edges; attenuation units are entire ROIs. They estimate
    different interventions. All differences use baseline loss minus perturbed loss.
    """
    if method not in ('entry_permutation', 'symmetric_permutation', 'roi_attenuation'):
        raise ValueError(method)
    n = x.shape[1]
    baseline = loss_on(model, x, y, config)
    repeats = 1 if method == 'roi_attenuation' else config['permutation_repeats']
    rng = np.random.default_rng(config['seed'])
    audit_rng = np.random.default_rng(20260928)  # auditing must not change explanation RNG
    if method == 'roi_attenuation':
        require_correlations(x)
        units = [(i, i) for i in range(n)]
    elif method == 'symmetric_permutation':
        units = list(itertools.combinations(range(n), 2))
    else:
        units = list(itertools.product(range(n), repeat=2))
    audit_ids = np.sort(audit_rng.choice(len(x), min(audit_subjects, len(x)), replace=False))
    audit_unit_ids = set(audit_rng.choice(len(units), min(audit_units, len(units)), replace=False).tolist())
    scores = np.zeros((repeats, n))
    edge = np.zeros((n, n)) if method != 'roi_attenuation' else None
    unit_records, audits = [], []
    for number, (i, j) in enumerate(units):
        differences = []
        for r in range(repeats):
            perturbed = (attenuate_roi(x, i, alpha) if method == 'roi_attenuation'
                         else permute_entry(x, i, j, rng.permutation(len(x)),
                                            symmetric=method == 'symmetric_permutation'))
            delta = baseline-loss_on(model, perturbed, y, config)
            differences.append(delta)
            scores[r, i] += delta
            if method != 'roi_attenuation':
                scores[r, j] += delta  # diagonal counted twice, matching original aggregation
            if r == 0 and number in audit_unit_ids:
                audits.append({'unit': [i, j], 'repeat': r,
                               'subject_local_indices': audit_ids.tolist(),
                               **structure_report(perturbed[audit_ids])})
        average = float(np.mean(differences))
        unit_records.append({'unit': [i, j], 'loss_differences': differences})
        if edge is not None:
            if method == 'entry_permutation':
                edge[i, j] = average
            else:
                # Half on each mirrored entry makes row+column aggregate count
                # the jointly permuted undirected edge once per endpoint.
                edge[i, j] = edge[j, i] = average/2
        if (number+1) % n == 0 or number+1 == len(units):
            print(f'{method} alpha={alpha}: {number+1}/{len(units)} units', flush=True)
    metadata = {'method': method, 'alpha': alpha, 'baseline_loss': baseline,
                'cohort': 'development', 'final_test_used': False, 'repeats': repeats,
                'sign': 'baseline cross entropy minus perturbed cross entropy',
                'unit_scores': unit_records, 'structural_audit_scope': 'sampled units, first repeat, sampled development subjects',
                'structural_audits': audits,
                'original_structure': structure_report(x),
                'interpretation': 'ROI connection attenuation sensitivity, not edge permutation or causal effect'
                    if method == 'roi_attenuation' else 'dataset-level permutation loss reliance'}
    if edge is not None:
        np.testing.assert_allclose(roi_scores(edge), scores.mean(0), atol=1e-10)
    return scores.mean(0), scores, edge, metadata


def compare_scores(a, b, top_k=20):
    a, b = np.asarray(a), np.asarray(b)
    k = min(top_k, len(a))
    ia, ib = set(np.argsort(a, kind='stable')[:k]), set(np.argsort(b, kind='stable')[:k])
    corr = None if np.ptp(a) == 0 or np.ptp(b) == 0 else float(spearmanr(a, b).statistic)
    return {'spearman': corr, 'matched_top_k': k, 'top_k_jaccard': len(ia & ib)/len(ia | ib),
            'top_k_purpose': 'size-controlled ranking comparison only; not the classifier ROI rule'}


def set_stability(sets, n_roi):
    sets = [set(map(int, x)) for x in sets]
    pairs = [{'run_a': i, 'run_b': j, 'jaccard': len(a & b)/len(a | b)}
             for (i, a), (j, b) in itertools.combinations(enumerate(sets), 2) if a | b]
    return {'n_completed': len(sets), 'pairwise_jaccard': pairs,
            'selection_frequency': [sum(i in s for s in sets)/len(sets) for i in range(n_roi)] if sets else [],
            'scope': 'fixed-split stochastic-run stability, not independent-split or unseen-site stability'}
