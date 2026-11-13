"""CPSS with an explicit Elastic Net edge-to-node base selector.

No final-test features, labels, metrics or baseline results enter selection.
CPSS aggregates selection events over disjoint complementary half-samples.
The edge-to-node L2 mapping is a project adaptation; no nominal FDR claim.
"""
import math,time,warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def complementary_pairs(train_indices, pairs, seed):
    train_indices=np.asarray(train_indices,dtype=np.int64)
    if len(np.unique(train_indices))!=len(train_indices):
        raise ValueError('Duplicate training subjects')
    rng=np.random.default_rng(seed);half=len(train_indices)//2;out=[]
    for number in range(pairs):
        shuffled=rng.permutation(train_indices)
        out.append(dict(pair=number,a=np.sort(shuffled[:half]).tolist(),
                        b=np.sort(shuffled[half:2*half]).tolist(),
                        omitted=shuffled[2*half:].tolist()))
    return out


def roi_l2_scores(coefficients,n_roi):
    i,j=np.triu_indices(n_roi,1)
    beta=np.asarray(coefficients,dtype=np.float64).reshape(-1)
    if len(beta)!=len(i) or not np.isfinite(beta).all():
        raise ValueError('Invalid undirected-edge coefficient vector')
    squares=np.zeros(n_roi,dtype=np.float64)
    np.add.at(squares,i,beta*beta);np.add.at(squares,j,beta*beta)
    return np.sqrt(squares)


def top_rois(scores,budget):
    scores=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(scores).all() or budget<0:raise ValueError('Invalid ROI scores/budget')
    # Never promote a node with no selected incident connection to satisfy a budget.
    eligible=np.flatnonzero(scores>0)
    order=np.lexsort((eligible,-scores[eligible]))
    return np.sort(eligible[order[:budget]])


def fit_selector(x_train,y_train,cfg):
    """Only this sample fits scaling and coefficients; returns no test metrics."""
    if len(np.unique(y_train))!=2:raise ValueError('Half-sample does not contain both classes')
    n_roi=x_train.shape[1];i,j=np.triu_indices(n_roi,1)
    raw=x_train[:,i,j].astype(np.float64)
    scaler=StandardScaler().fit(raw);features=scaler.transform(raw)
    model=LogisticRegression(C=cfg['C'],l1_ratio=cfg['l1_ratio'],penalty='elasticnet',
        solver='saga',max_iter=cfg['max_iter'],tol=cfg['tol'],random_state=cfg['seed'],n_jobs=1)
    tic=time.monotonic()
    with threadpool_limits(limits=cfg['threads']),warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');model.fit(features,y_train)
    messages=[str(w.message) for w in ws]
    if any(issubclass(w.category,ConvergenceWarning) for w in ws):
        raise RuntimeError('Elastic Net did not converge with frozen solver budget: '+'; '.join(messages))
    scores=roi_l2_scores(model.coef_,n_roi)
    return dict(scores=scores,coef=model.coef_.reshape(-1),intercept=model.intercept_,
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                metadata=dict(seconds=time.monotonic()-tic,iterations=int(model.n_iter_[0]),
                    n_train=len(y_train),class_counts=np.bincount(y_train,minlength=2).tolist(),
                    nonzero_edges=int(np.count_nonzero(model.coef_)),warnings=messages))


def aggregate(scores,budget,threshold):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=2 or len(scores)%2:raise ValueError('Expected paired selection scores')
    if not .5<threshold<=1:raise ValueError('CPSS frequency threshold must exceed one half')
    selected=np.zeros_like(scores,dtype=np.uint8)
    for row,values in enumerate(scores):selected[row,top_rois(values,budget)]=1
    counts=selected.sum(0);required=math.ceil(threshold*len(scores)-1e-12)
    rois=np.flatnonzero(counts>=required)
    return dict(rois=rois.tolist(),selection_counts=counts.tolist(),
                selection_frequency=(counts/len(scores)).tolist(),
                base_selected_counts=selected.sum(1).tolist(),base_budget=budget,
                threshold=threshold,required_count=required,n_half_samples=len(scores))
