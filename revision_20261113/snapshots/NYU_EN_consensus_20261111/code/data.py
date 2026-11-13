from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data(path, label_column=2):
    """Read the retained MATLAB split schema; never infer a new label encoding."""
    mat = loadmat(path)
    xs, ys, phenotypes, indices = [], [], [], {}
    offset = 0
    n_roi = None
    for split in ('train', 'valid', 'test'):
        x = np.asarray(mat['net_' + split], dtype=np.float32)
        ph = np.asarray(mat['phenotype_' + split])
        if x.ndim != 3 or x.shape[1] != x.shape[2] or not len(x):
            raise ValueError(f'net_{split} must have shape (subjects, ROI, ROI)')
        if ph.ndim != 2 or ph.shape[0] != len(x) or not 0 <= label_column < ph.shape[1]:
            raise ValueError(f'phenotype_{split} does not match matrices/label column')
        y = ph[:, label_column]
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Expected finite FC matrices and explicit binary labels 0/1')
        if n_roi is not None and n_roi != x.shape[1]:
            raise ValueError('ROI dimensions differ between splits')
        n_roi = x.shape[1]
        indices[split] = np.arange(offset, offset + len(x))
        offset += len(x)
        xs.append(x); ys.append(y.astype(np.int64)); phenotypes.append(ph)
    return np.concatenate(xs), np.concatenate(ys), indices, np.concatenate(phenotypes)


def make_split(y, seed):
    """New diagnosis-stratified 80/10/10 split, NOT recovered historical seeds."""
    train, rest = train_test_split(np.arange(len(y)), test_size=0.2,
                                  random_state=seed, stratify=y)
    valid, test = train_test_split(rest, test_size=0.5,
                                  random_state=seed, stratify=y[rest])
    return {'train': train, 'valid': valid, 'test': test}


def validate_split(indices, n, y=None):
    arrays = [np.asarray(indices[k]) for k in ('train', 'valid', 'test')]
    if any(a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu' for a in arrays):
        raise ValueError('Split arrays must be nonempty one-dimensional integer indices')
    all_i = np.concatenate(arrays)
    if len(all_i) != n or not np.array_equal(np.sort(all_i), np.arange(n)):
        raise ValueError('Splits must be disjoint and cover every input subject exactly once')
    if y is not None and any(len(np.unique(y[a])) != 2 for a in arrays):
        raise ValueError('Each split must contain both classes')
    return {k: np.asarray(indices[k], dtype=np.int64) for k in ('train', 'valid', 'test')}


def features(x, rois):
    """Original per-subject LayerNorm, AFTER extracting the ROI submatrix."""
    rois = np.asarray(rois, dtype=int)
    if rois.ndim != 1 or not len(rois) or len(np.unique(rois)) != len(rois):
        raise ValueError('ROI indices must be a nonempty unique list')
    if rois.min() < 0 or rois.max() >= x.shape[1]:
        raise ValueError('ROI index out of range')
    sub = np.ascontiguousarray(x[:, rois][:, :, rois], dtype=np.float32)
    return F.layer_norm(torch.from_numpy(sub), sub.shape[1:]).flatten(1)
