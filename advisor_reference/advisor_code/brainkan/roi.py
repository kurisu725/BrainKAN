"""Paper Eq. (2) and the three strict threshold rules."""
import numpy as np


def roi_scores(edge):
    edge = np.asarray(edge, dtype=float)
    if edge.ndim != 2 or edge.shape[0] != edge.shape[1] or not np.isfinite(edge).all():
        raise ValueError('Expected a finite square edge-contribution matrix')
    # Diagonal is counted twice, exactly as the manuscript equation states.
    return edge.sum(axis=0) + edge.sum(axis=1)


def transform_scores(scores, mode):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected finite ROI scores')
    if mode == 'direct':
        return scores.copy()
    if mode == 'minmax':
        span = np.ptp(scores)
        if span == 0:
            raise ValueError('Min-max normalization is undefined for constant ROI scores')
        return (scores - scores.min()) / span
    if mode == 'zscore':
        sd = scores.std(ddof=0)
        if sd == 0:
            raise ValueError('Z-score normalization is undefined for constant ROI scores')
        return (scores - scores.mean()) / sd
    raise ValueError('Unknown ROI mode: ' + mode)


def select_rois(scores, mode, threshold=None):
    threshold = (0.5 if mode == 'minmax' else 0.0) if threshold is None else threshold
    if not np.isfinite(threshold):
        raise ValueError('ROI threshold must be finite')
    values = transform_scores(scores, mode)
    selected = np.flatnonzero(values < threshold)
    # Retain ascending contribution order, as in the original matrixAuto lists.
    selected = selected[np.argsort(np.asarray(scores)[selected], kind='stable')]
    if len(selected) == 0:
        raise ValueError(f'{mode} selected zero ROIs; no undocumented top-k fallback is applied')
    return selected
