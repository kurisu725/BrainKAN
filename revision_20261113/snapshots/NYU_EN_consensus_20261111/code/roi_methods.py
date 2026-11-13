"""ROI selector adaptations; inputs contain training or development data only."""
from pathlib import Path
import warnings
import numpy as np
from scipy.stats import rankdata
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F
from data import features
from cpss import roi_l2_scores
import run_review as r


def rank_consensus(score_rows):
    a = np.asarray(score_rows, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite repeated ROI scores')
    # Equal scores receive equal average ranks; high value = important.
    return np.mean([rankdata(row, method='average') / len(row) for row in a], axis=0)


def top_k(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 < k <= len(scores):
        raise ValueError('Invalid scores or ROI budget')
    ids = np.arange(len(scores))
    return np.sort(ids[np.lexsort((ids, -scores))[:k]])


def threshold(scores, mode):
    """Scores use loss(permuted)-loss(original); keep larger values."""
    a = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite importance')
    if mode == 'winsor_minmax':
        lo, hi = np.quantile(a, [.05, .95])
        cut = (lo + hi) / 2
    elif mode == 'minmax':
        lo, hi = a.min(), a.max()
        cut = (lo + hi) / 2
    elif mode == 'zscore':
        lo, hi = float(a.mean()), float(a.std(ddof=0))
        cut = lo  # z > 0 is equivalent to importance > mean.
    else:
        raise ValueError(mode)
    ids = np.flatnonzero(a > cut)
    return ids, dict(mode=mode,cut=float(cut),lower=float(lo),upper=float(hi),comparison='>')


def permute_roi_block(x, roi, donors):
    """Move both incident edge directions together; preserve recipient diagonal.

    This is an importance perturbation, not a generated physiological connectome.
    Symmetry is preserved. Positive semidefiniteness is not imposed.
    """
    x = np.asarray(x)
    ids = np.asarray(donors, dtype=int)
    if not np.array_equal(np.sort(ids), np.arange(len(x))):
        raise ValueError('Donors must permute this cohort only')
    other = np.delete(np.arange(x.shape[1]), roi)
    z = x.copy()
    z[:, roi, other] = x[ids][:, roi, other]
    z[:, other, roi] = x[ids][:, other, roi]
    return z


@torch.no_grad()
def grouped_importance(model, x_dev, y_dev, *, repeats, seed, folder, callback=None):
    """Raw FC block permutation -> per-subject LayerNorm -> fixed last model."""
    model.eval()
    n_roi = x_dev.shape[1]
    yy = torch.as_tensor(y_dev, dtype=torch.long, device='cuda')
    def loss(x):
        tensor = features(x, np.arange(n_roi)).to('cuda')
        return float(F.cross_entropy(model(tensor), yy))
    base = loss(x_dev)
    dest = Path(folder) / 'group_partial.npz'
    values = np.zeros((repeats,n_roi),dtype=np.float64)
    done = 0
    if dest.exists():
        with np.load(dest) as saved:
            values = saved['values'].copy()
            done = int(saved['done'])
            if not np.isclose(base,float(saved['baseline']),rtol=0,atol=1e-7):
                raise ValueError('Resume importance baseline changed')
    for roi in range(done,n_roi):
        for rep in range(repeats):
            # Shared donor permutations across the three model seeds.
            rng = np.random.default_rng(seed + roi * 1009 + rep)
            altered = permute_roi_block(x_dev,roi,rng.permutation(len(x_dev)))
            values[rep,roi] = loss(altered) - base
        tmp = dest.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,values=values,done=roi+1,baseline=base)
        tmp.replace(dest)
        if callback is not None:
            callback(roi+1)
    return values, base


def recursive_nodes(x_train,y_train,train_indices,folder,cfg):
    """Node-group adaptation of SVM-RFE, refitting the induced graph each step.

    Standardization and SVM see only the specified training cohort. Importance
    is sum of squared standardized coefficients on the currently retained edges.
    """
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    rois = np.arange(x_train.shape[1]); snapshots = {}; history = []
    while len(rois) >= min(cfg['budgets']):
        if len(rois) in cfg['budgets']:
            snapshots[str(len(rois))] = rois.tolist()
        if len(rois) == min(cfg['budgets']):
            break
        target = folder / f'step_{len(rois):02d}.npz'
        if target.exists():
            with np.load(target) as a:
                if not np.array_equal(a['rois'],rois):
                    raise ValueError('RFE resume path differs')
                scores = a['scores']; iterations = int(a['iterations'])
        else:
            i,j = np.triu_indices(len(rois),1)
            raw = x_train[:,rois[i],rois[j]].astype(np.float64)
            scaler = StandardScaler().fit(raw)
            model = LinearSVC(C=cfg['C'],dual=True,tol=cfg['tol'],max_iter=cfg['max_iter'],random_state=cfg['seed'])
            with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                model.fit(scaler.transform(raw),y_train)
            if any(issubclass(w.category,ConvergenceWarning) for w in ws):
                raise RuntimeError('SVM node-RFE solver did not converge: '+str([str(w.message) for w in ws]))
            scores = roi_l2_scores(model.coef_,len(rois)) ** 2
            iterations = int(model.n_iter_)
            tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream,rois=rois,scores=scores,coef=model.coef_,
                    intercept=model.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                    train_indices=train_indices,iterations=iterations)
            tmp.replace(target)
        remove = np.lexsort((rois,scores))[:cfg['step']]
        history.append(dict(n_roi=len(rois),rois=rois.tolist(),scores=scores.tolist(),removed=rois[remove].tolist(),iterations=iterations))
        rois = np.delete(rois,remove)
    r.write(folder/'history.json',history)
    r.write(folder/'snapshots.json',snapshots)
    return snapshots
