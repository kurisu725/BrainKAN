"""Convert nonnegative KAN output-deletion magnitudes to original ROI sets.

The scores are RMS changes in the complete KAN ensemble logit. Higher means
greater output influence, without a class-support sign or a claim that deleting
the region must hurt accuracy. Selection thresholds use training scores only.
"""
import numpy as np


def threshold_rois(scores, rule):
    c = np.asarray(scores, dtype=np.float64)
    if c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError('Expected a finite, nonnegative vector of ROI influences.')
    family = rule['family']
    spread = float(np.ptp(c))
    if family == 'all':
        selected = np.arange(len(c))
        threshold = None
        details = dict(degenerate=False)
    elif spread <= np.finfo(float).eps * max(float(c.max()), 1.):
        # Indistinguishable scores supply no evidence for eliminating any ROI.
        selected = np.arange(len(c))
        threshold = float(c.min())
        details = dict(degenerate=True, reason='All ROI scores are indistinguishable; retain all.')
    elif family == 'minmax':
        t = float(rule['value'])
        if not 0 <= t <= 1:
            raise ValueError('Min-max threshold must lie in [0, 1].')
        threshold = float(c.min() + t * spread)
        selected = np.flatnonzero(c >= threshold)
        details = dict(degenerate=False, normalized_threshold=t)
    elif family == 'zscore':
        t = float(rule['value'])
        threshold = float(c.mean() + t * c.std(ddof=0))
        selected = np.flatnonzero(c >= threshold)
        details = dict(degenerate=False, z_threshold=t, mean=float(c.mean()), population_sd=float(c.std()))
    elif family == 'otsu':
        # Exact 1-D Otsu split at distinct adjacent values, without histogram bins.
        # Maximize n_low*n_high*(mean_high-mean_low)^2; ties prefer the lower cut.
        v = np.sort(c, kind='stable')
        k = np.flatnonzero(v[:-1] < v[1:]) + 1
        prefix = np.cumsum(v)
        n = len(v)
        low = prefix[k-1] / k
        high = (prefix[-1] - prefix[k-1]) / (n-k)
        objective = k * (n-k) * (high-low)**2
        cut = int(k[int(np.argmax(objective))])
        threshold = float(v[cut-1] + (v[cut]-v[cut-1]) / 2)
        selected = np.flatnonzero(c >= threshold)
        details = dict(degenerate=False, low_group_size=cut, high_group_size=n-cut,
                       between_class_objective=float(objective.max()))
    elif family == 'mass':
        fraction = float(rule['value'])
        if not 0 < fraction <= 1:
            raise ValueError('Squared-mass fraction must lie in (0, 1].')
        order = np.argsort(-c, kind='stable')
        energy = c[order]**2
        count = min(len(c), int(np.searchsorted(np.cumsum(energy), fraction*energy.sum(), side='left'))+1)
        selected = np.sort(order[:count])
        threshold = float(c[order[count-1]])
        details = dict(degenerate=False, requested_squared_mass=fraction,
                       realized_squared_mass=float(energy[:count].sum()/energy.sum()),
                       boundary_ties='Stable original atlas order, as in the previous mass rule.')
    else:
        raise ValueError('Unknown threshold family: ' + family)
    return np.asarray(selected, dtype=int), dict(rule=rule, raw_threshold=threshold,
                                                retained_rois=len(selected), **details)
