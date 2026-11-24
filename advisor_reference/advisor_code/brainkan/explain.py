"""Two explicitly different explainers. Neither is claimed to recover old results."""
import numpy as np
from sklearn.metrics import log_loss
from .training import predict


def permutation_loss(model, x_train, y_train, config):
    """Loss difference on the explicitly declared training or development cohort.

    Argument names are retained for compatibility. The caller records exact subject
    indices; final held-out test subjects must never be supplied for ROI selection.
    Independent FC-entry permutations do not preserve correlation-matrix structure.
    """
    n_roi = x_train.shape[1]
    rois = np.arange(n_roi)
    rng = np.random.default_rng(config['seed'])
    baseline = log_loss(y_train, predict(model, x_train, rois, config['batch_size'], config['device']), labels=[0, 1])
    work = x_train.copy().reshape(len(x_train), -1)
    result = np.zeros(work.shape[1])
    for j in range(work.shape[1]):
        original = work[:, j].copy()
        for _ in range(config['permutation_repeats']):
            work[:, j] = original[rng.permutation(len(work))]
            prob = predict(model, work.reshape(x_train.shape), rois, config['batch_size'], config['device'])
            result[j] += baseline - log_loss(y_train, prob, labels=[0, 1])
        work[:, j] = original
        if (j + 1) % n_roi == 0:
            print(f'Permutation: {j+1}/{work.shape[1]} edges', flush=True)
    return result.reshape(n_roi, n_roi) / config['permutation_repeats'], {'baseline_loss': float(baseline)}


def lime_contributions(model, x_train, y_train, config):
    """New tabular LIME implementation; all background and explained subjects are training data.

    Quartile-discretized features give local indicator coefficients. Negating the
    target-class coefficients implements the manuscript's negative-support convention.
    Historical LIME settings are missing: these choices are explicitly reconstructed.
    """
    from lime.lime_tabular import LimeTabularExplainer
    n_roi = x_train.shape[1]
    flat = x_train.reshape(len(x_train), -1)
    kwargs = dict(mode='classification', class_names=['control', 'patient'],
                  discretize_continuous=True, discretizer='quartile',
                  feature_selection='none', random_state=config['seed'])
    if config['lime_kernel_width'] is not None:
        kwargs['kernel_width'] = config['lime_kernel_width']
    explainer = LimeTabularExplainer(flat, **kwargs)
    def predict_fn(z):
        return predict(model, np.asarray(z, dtype=np.float32).reshape(-1, n_roi, n_roi),
                       np.arange(n_roi), config['batch_size'], config['device'])
    local_ids = np.arange(len(flat))
    if config['explain_subjects']:
        rng = np.random.default_rng(config['seed'])
        local_ids = np.sort(rng.choice(local_ids, min(len(local_ids), config['explain_subjects']), replace=False))
    values = np.zeros(flat.shape[1])
    fidelity = []
    for pos, i in enumerate(local_ids):
        result = explainer.explain_instance(flat[i], predict_fn, labels=(config['target_class'],),
                                            num_features=flat.shape[1], num_samples=config['lime_samples'])
        for edge, weight in result.local_exp[config['target_class']]:
            values[edge] -= weight
        fidelity.append(float(result.score))
        print(f'LIME: subject {pos+1}/{len(local_ids)}', flush=True)
    return (values / len(local_ids)).reshape(n_roi, n_roi), {
        'training_local_indices': local_ids.tolist(), 'surrogate_r2': fidelity,
        'sign': 'negative target-class indicator coefficient',
        'target_class': config['target_class'], 'discretizer': 'quartile',
        'kernel_width': config['lime_kernel_width'] or 0.75 * np.sqrt(flat.shape[1]),
        'feature_selection': 'none', 'num_samples': config['lime_samples'],
        'historical_settings_recovered': False,
    }
