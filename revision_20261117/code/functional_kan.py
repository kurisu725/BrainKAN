"""A native one-layer spline KAN with proximal sparsification of functions.

Each FC input owns a learned centered SiLU-plus-cubic-spline function. A
training-only orthonormal change of basis makes its coefficient group norm
equal to its empirical RMS function output. Group soft-thresholding then
zeros complete functions inside this KAN; no external selector is fitted.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from legacy_kan import KANLinear


@torch.no_grad()
def make_basis(z,ntrain,eigen_floor=1e-5):
    layer=KANLinear(z.shape[1],1,grid_size=5,spline_order=3,grid_range=[-3,3]).to(z.device)
    raw=torch.cat([torch.cat([F.silu(x)[...,None],layer.b_splines(x)],-1) for x in z.split(96)])
    mean=raw[:ntrain].mean(0);center=raw-mean
    covariance=torch.einsum('ned,nek->edk',center[:ntrain],center[:ntrain])/ntrain
    values,vectors=torch.linalg.eigh(covariance)
    active=values>eigen_floor;inverse=torch.where(active,values.clamp_min(eigen_floor).rsqrt(),0)
    whitening=vectors*inverse[:,None,:]
    q=torch.einsum('ned,edk->nek',center,whitening)
    state=dict(mean=mean.cpu(),whitening=whitening.cpu(),grid=layer.grid.cpu(),
               eigenvalues=values.cpu(),active_directions=active.cpu(),eigen_floor=eigen_floor)
    return q.contiguous(),state


@torch.no_grad()
def replay_basis(z,state):
    layer=KANLinear(z.shape[1],1,grid_size=5,spline_order=3,grid_range=[-3,3]).to(z.device)
    layer.grid.copy_(state['grid'].to(z.device))
    raw=torch.cat([torch.cat([F.silu(x)[...,None],layer.b_splines(x)],-1) for x in z.split(96)])
    return torch.einsum('ned,edk->nek',raw-state['mean'].to(z.device),state['whitening'].to(z.device)).contiguous()


@torch.no_grad()
def lipschitz(matrix,iterations=35):
    # The unpenalized intercept is an extra constant input column.
    u=torch.sin(torch.arange(1,len(matrix)+1,device=matrix.device,dtype=matrix.dtype));u/=u.norm()
    for _ in range(iterations):
        v=matrix@(matrix.T@u)+u.sum();u=v/v.norm().clamp_min(1e-12)
    spectral=u@(matrix@(matrix.T@u)+u.sum())
    return max(float(spectral)*.25/len(matrix)*1.1,.25)


@torch.no_grad()
def lambda_maximum(q,y):
    yy=torch.as_tensor(y,device=q.device,dtype=q.dtype);residual=yy-yy.mean()
    gradient=torch.einsum('ned,n->ed',q,residual)/len(yy)
    return float(gradient.norm(dim=1).max())


@torch.no_grad()
def solve(q,y,lam,max_iterations=1200,tolerance=1e-5,initial=None,initial_lipschitz=None):
    """Monotone FISTA for this KAN's logistic + sum(RMS function) objective."""
    n,edges,dimensions=q.shape;matrix=q.flatten(1)
    yy=torch.as_tensor(y,device=q.device,dtype=q.dtype)
    if initial is None:
        weight=torch.zeros(edges,dimensions,device=q.device);bias=torch.logit(yy.mean().clamp(.001,.999))
    else:weight=initial['weight'].to(q.device).clone();bias=initial['bias'].to(q.device).clone()
    step_lipschitz=initial_lipschitz or lipschitz(matrix)
    smooth=lambda w,b:(F.softplus(matrix@w.flatten()+b)-yy*(matrix@w.flatten()+b)).mean()
    objective=lambda w,b:smooth(w,b)+lam*w.norm(dim=1).sum()
    z=weight.clone();zb=bias.clone();momentum=1.;previous=float(objective(weight,bias));history=[];converged=False
    for iteration in range(1,max_iterations+1):
        logits=matrix@z.flatten()+zb;residual=logits.sigmoid()-yy
        gradient=(matrix.T@residual/n).view(edges,dimensions);gb=residual.mean()
        smooth_z=(F.softplus(logits)-yy*logits).mean()
        for _ in range(20):
            trial=z-gradient/step_lipschitz
            norm=trial.norm(dim=1,keepdim=True)
            candidate=trial*(1-lam/(step_lipschitz*norm.clamp_min(1e-20))).clamp_min(0)
            cb=zb-gb/step_lipschitz;delta=candidate-z;db=cb-zb
            value=smooth(candidate,cb)
            upper=smooth_z+(gradient*delta).sum()+gb*db+step_lipschitz*.5*(delta.square().sum()+db.square())
            if float(value-upper)<=2e-7:break
            step_lipschitz*=2
        else:raise FloatingPointError('Native functional KAN line search failed')
        current=float(value+lam*candidate.norm(dim=1).sum())
        if not math.isfinite(current):raise FloatingPointError('Nonfinite functional KAN objective')
        if current>previous+2e-7 and momentum>1:
            z=weight.clone();zb=bias.clone();momentum=1.;continue
        residual_norm=float((step_lipschitz*delta).norm()+abs(step_lipschitz*db))
        new_momentum=(1+math.sqrt(1+4*momentum**2))/2
        factor=(momentum-1)/new_momentum
        new_z=candidate+factor*(candidate-weight);new_zb=cb+factor*(cb-bias)
        # A gradient-direction restart improves robustness of the accelerated path.
        if float(((z-candidate)*(candidate-weight)).sum()+(zb-cb)*(cb-bias))>0:
            new_momentum=1.;new_z=candidate.clone();new_zb=cb.clone()
        change=abs(previous-current);weight=candidate;bias=cb;z=new_z;zb=new_zb;momentum=new_momentum
        previous=current
        if iteration%10==0 or iteration==1:
            history.append(dict(iteration=iteration,objective=current,proximal_gradient_norm=residual_norm,
                nonzero_functions=int((weight.norm(dim=1)>1e-8).sum()),lipschitz=step_lipschitz))
        if iteration>=50 and residual_norm<=tolerance*(1+float(weight.norm())):
            converged=True;break
    return dict(weight=weight.cpu(),bias=bias.cpu(),history=history,iterations=iteration,
                converged=converged,objective=previous,lambda_value=lam,lipschitz=step_lipschitz,
                proximal_gradient_norm=residual_norm,nonzero_functions=int((weight.norm(dim=1)>1e-8).sum()))


@torch.no_grad()
def margin(q,model):return (q.flatten(1)@model['weight'].to(q.device).flatten()+model['bias'].to(q.device)).cpu().numpy()


@torch.no_grad()
def native_scores(q,model,pairs,nroi=90):
    values=(q*model['weight'].to(q.device)[None]).sum(-1)
    activity=(values-values.mean(0,keepdim=True)).abs().mean(0)
    score=torch.zeros(nroi,device=q.device);pair=torch.as_tensor(pairs,device=q.device)
    score.index_add_(0,pair[0],activity);score.index_add_(0,pair[1],activity)
    return score.cpu().numpy(),activity.cpu().numpy()


def gather_basis(state,keep):
    keep=torch.as_tensor(keep,dtype=torch.bool)
    return {key:value[keep] if torch.is_tensor(value) and value.ndim and value.shape[0]==len(keep) else value
            for key,value in state.items()}
