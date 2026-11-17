"""Bootstrap-trained complete KAN branches with one output-based ROI boundary.

This is explicitly a KAN ensemble. Its arithmetic mean logit can also be
represented by a block-diagonal multibranch KAN. ROI deletion is evaluated
on that whole output, never by voting or averaging branch ROI rankings.
"""
import copy,time
import numpy as np
import torch
from conditional_site_kan import prepare,transform,compact,restore,cpu_export
from residual_kan import PAIRS


def bootstrap_draw(labels,seed):
    y=np.asarray(labels);assert set(np.unique(y))=={0,1};rng=np.random.default_rng(seed)
    draw=np.concatenate([rng.choice(np.flatnonzero(y==c),int((y==c).sum()),replace=True) for c in [0,1]])
    return draw[rng.permutation(len(draw))]


def base_regularizer(net,q,cov,cfg):
    # Exact zero-alignment objective of the audited complete-curve parent.
    d=torch.zeros(1,len(q),device=q.device,dtype=q.dtype)
    return net.regularizer(q,cov,d,dict(cfg,alignment_lambda=0.))


@torch.no_grad()
def ensemble_logits(nets,qs):
    assert len(nets)==len(qs)>0
    return torch.stack([net(q) for net,q in zip(nets,qs)]).mean(0)


@torch.no_grad()
def deletion_contrasts(net,q,features,rois=np.arange(90)):
    """Exact full-output minus incident-function-deleted output, per person."""
    h=net.hidden(q);full=net(q);c=net.effective_first();pairs=PAIRS[:,features];values=[]
    for roi in rois:
        incident=(pairs==roi).any(0);delta=q[:,incident].flatten(1)@c[incident].flatten(0,1)
        without=net.head_terms(h-delta).sum(1)+net.bias;values.append(full-without)
    return torch.stack(values,1).cpu().numpy()


def whole_output_scores(branch_contrasts,count):
    # Signed effects are averaged BEFORE squaring; this is not mean branch RMS.
    d=np.mean(np.stack(branch_contrasts[:count]),axis=0)
    return np.sqrt(np.mean(d*d,axis=0))


@torch.no_grad()
def compile_branch(net,state,rois,device='cuda'):
    return compact(net,state,rois,device)
