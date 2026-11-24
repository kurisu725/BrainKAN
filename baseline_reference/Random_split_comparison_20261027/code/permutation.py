"""Retained edge-major permutation estimator with durable edge checkpoints."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import log_loss
from roi import roi_scores
from data import features
from run_review import predict as predict_tensor

def predict(model,x,rois):
    return predict_tensor(model,features(x,rois),next(model.parameters()).device,96)

def fast_permutation(model,x_dev,y_dev,seed,repeats=10,chunk=4,edge_limit=None,checkpoint=None):
    """Same edge-major RNG sequence and log-loss as the sequential implementation.

    Full-input inference skips redundant full-ROI array reindexing. The same
    contiguous values enter CPU LayerNorm, with the original batch size and RNG.
    Save the complete score array and generator state every 90 edges.
    """
    n=x_dev.shape[1];rng=np.random.default_rng(seed);rois=np.arange(n)
    baseline=log_loss(y_dev,predict(model,x_dev,rois),labels=[0,1]);flat=x_dev.reshape(len(x_dev),-1)
    limit=n*n if edge_limit is None else edge_limit
    diffs=np.zeros((repeats,n*n),dtype=float);pending=[];keys=[];start=0
    identity=dict(seed=seed,repeats=repeats,n=n,subjects=len(x_dev),limit=limit,chunk=chunk,
                  input_sha256=hashlib.sha256(x_dev.tobytes()+y_dev.tobytes()).hexdigest())
    if checkpoint is not None and Path(checkpoint).exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            assert json.loads(str(saved['identity']))==identity
            assert abs(float(saved['baseline'])-baseline)<1e-12
            diffs=saved['diffs'].copy();start=int(saved['next_edge']);rng.bit_generator.state=json.loads(str(saved['rng']))
        print('Resumed importance at edge',start,'/',limit,flush=True)
    @torch.no_grad()
    def full_input_probabilities(batch):
        # Equivalent to features(batch, arange(n)); the matrix is already the
        # complete ordered submatrix, so the two fancy-index copies are redundant.
        tensor=F.layer_norm(torch.from_numpy(np.ascontiguousarray(batch,dtype=np.float32)),(n,n)).flatten(1)
        model.eval()
        return model(tensor.to(next(model.parameters()).device)).softmax(1).cpu().numpy()
    def flush():
        if not pending:return
        batch=np.concatenate(pending).reshape(-1,n,n)
        probs=full_input_probabilities(batch).reshape(len(pending),len(x_dev),2)
        for (r,e),prob in zip(keys,probs):diffs[r,e]=baseline-log_loss(y_dev,prob,labels=[0,1])
        pending.clear();keys.clear()
    for e in range(start,limit):
        for r in range(repeats):
            xx=flat.copy();xx[:,e]=flat[rng.permutation(len(flat)),e]
            pending.append(xx);keys.append((r,e))
            if len(pending)==chunk:flush()
        if (e+1)%90==0 or e+1==limit:
            flush()
            if checkpoint is not None:
                checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
                temp=checkpoint.with_suffix('.tmp')
                with temp.open('wb') as stream:
                    np.savez(stream,diffs=diffs,next_edge=e+1,baseline=baseline,
                             rng=json.dumps(rng.bit_generator.state),identity=json.dumps(identity))
                temp.replace(checkpoint)
            print('site importance',e+1,'/',limit,flush=True)
    flush();edges=diffs.mean(0).reshape(n,n)
    repeated=diffs.reshape(repeats,n,n).sum(1)+diffs.reshape(repeats,n,n).sum(2)
    return roi_scores(edges),repeated,edges,dict(baseline_loss=float(baseline),repeats=repeats,seed=seed,
           order='edge-major then repeat',inference_chunk=chunk,normalization='per-subject CPU LayerNorm',
           full_input_reindex='redundant copies omitted; bitwise normalized-tensor and score equivalence checked',checkpoint_every_edges=90)
