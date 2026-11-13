"""Training-half Elastic Net selector; this module does not apply a CPSS frequency threshold."""
import time, warnings
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


from scipy.stats import rankdata


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

