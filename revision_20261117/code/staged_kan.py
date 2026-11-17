"""Two-layer KAN initialized by cross-fitted versions of its own first layer.

The first layer owns complete centered SiLU/cubic-spline FC functions. Each
regional intermediate sums half of its incident functions. The second layer
owns linear/SiLU/cubic-spline functions of those regional intermediates. Its
identity initialization reproduces the first layer's additive classifier.
No external classifier or ROI-ranking model is involved.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from legacy_kan import KANLinear
from functional_kan import make_basis, replay_basis, gather_basis, lambda_maximum
from regularized_functional import function_penalty, solve, transfer_functions
from protected_site_kan import prepare, replay

PAIRS=np.stack(np.triu_indices(90,1))


@torch.no_grad()
def first_fit(raw,y,sites,ntrain,mode,settings,training_indices,rois=None,source=None):
    """Fit/continue first-layer functions using only the supplied fit roles."""
    rois=np.arange(90) if rois is None else np.asarray(rois,dtype=int)
    raw_keep=np.isin(PAIRS[0],rois)&np.isin(PAIRS[1],rois)
    z,scaler=prepare(raw,y,sites,ntrain,mode,keep=raw_keep)
    scaler['fitted_indices']=np.asarray(training_indices).tolist()
    raw_ids=np.flatnonzero(raw_keep)
    if source is None:
        q,basis=make_basis(z,ntrain,settings['eigen_floor'])
        penalty=function_penalty(q[:ntrain],z[:ntrain],settings['first_rms'],settings['first_nonlinearity'])
        lam=lambda_maximum(q[:ntrain],y[:ntrain])*settings['first_lambda_fraction']
        pre=solve(q[:ntrain],y[:ntrain],lam,penalty,settings['max_iterations'],settings['tolerance'])
        activity=pre['weight'].norm(dim=1).numpy()
        weak=np.argsort(activity,kind='stable')[:int(len(activity)*settings['weak_fraction'])]
        active=np.ones(len(activity),dtype=bool);active[weak]=False
        pre_record=dict(model=pre,basis=basis,raw_feature_indices=raw_ids,
                        function_rms=activity,weak_positions=weak,active_mask=active)
        q=q[:,active].contiguous();basis=gather_basis(basis,active)
        initial=dict(weight=pre['weight'][active],bias=pre['bias'])
        penalty=penalty[active]
        z=z[:,active].contiguous(); feature_ids=raw_ids[active]
        projection=0.
    else:
        feature_ids=np.intersect1d(raw_ids,np.asarray(source['feature_indices']),assume_unique=True)
        active=np.isin(raw_ids,feature_ids)
        source_keep=np.isin(np.asarray(source['feature_indices']),feature_ids)
        z=z[:,active].contiguous();q,basis=make_basis(z,ntrain,settings['eigen_floor'])
        initial=transfer_functions(dict(weight=source['model']['weight'][source_keep],bias=source['model']['bias']),
                                   gather_basis(source['basis'],source_keep),basis)
        projection=initial['canonical_coefficient_projection_error']
        lam=source['model']['lambda_value']
        penalty=function_penalty(q[:ntrain],z[:ntrain],settings['first_rms'],settings['first_nonlinearity'])
        pre_record=None
    model=solve(q[:ntrain],y[:ntrain],lam,penalty,settings['max_iterations'],settings['tolerance'],initial=initial)
    result=dict(model=model,basis=basis,scaler=scaler,feature_indices=feature_ids,rois=rois,
                training_indices=np.asarray(training_indices),pre_mask=pre_record,
                canonical_projection_error=projection,normalization_raw_indices=raw_ids)
    h=regional_values(q,model,feature_ids,rois)
    return h,result


@torch.no_grad()
def regional_values(q,model,feature_ids,rois):
    values=(q*model['weight'].to(q.device)[None]).sum(-1).double()
    incidence=torch.zeros(len(feature_ids),len(rois),device=q.device,dtype=torch.float64)
    pair=PAIRS[:,np.asarray(feature_ids)]
    lookup={int(r):i for i,r in enumerate(rois)}
    local=np.array([[lookup[int(r)] for r in pair[a]] for a in range(2)])
    rows=torch.arange(len(feature_ids),device=q.device)
    incidence[rows,torch.as_tensor(local[0],device=q.device)]=.5
    incidence[rows,torch.as_tensor(local[1],device=q.device)]=.5
    h=values@incidence+model['bias'].to(q.device).double()/len(rois)
    return h,values,local


@torch.no_grad()
def replay_first(raw_upper_kept,sites,source,device='cuda'):
    z=replay(raw_upper_kept,sites,source['scaler'])
    active=np.isin(np.asarray(source['scaler']['feature_indices']),source['feature_indices'])
    zz=torch.tensor(z[:,active],dtype=torch.float32,device=device)
    q=replay_basis(zz,source['basis'])
    return regional_values(q,source['model'],source['feature_indices'],source['rois'])


@torch.no_grad()
def head_raw(h,normalization,grid=None):
    mu=normalization['mean'].to(h.device);scale=normalization['scale'].to(h.device)
    z=(h-mu)/scale
    layer=KANLinear(h.shape[1],1,grid_size=5,spline_order=3,grid_range=[-3,3]).to(h.device).double()
    if grid is not None:layer.grid.copy_(grid.to(h.device))
    raw=torch.cat([z[...,None],F.silu(z)[...,None],layer.b_splines(z)],-1)
    return raw,layer.grid.cpu()


@torch.no_grad()
def head_basis(h,normalization=None,eigen_floor=1e-8):
    h=h.double()
    if normalization is None:
        normalization=dict(mean=h.mean(0).cpu(),scale=h.std(0,unbiased=False).clamp_min(1e-4).cpu())
    raw,grid=head_raw(h,normalization);mean=raw.mean(0);center=raw-mean
    covariance=torch.einsum('ned,nek->edk',center,center)/len(h)
    # Preserve the explicit linear direction exactly. Whiten nonlinear
    # residuals conditional on it, instead of discarding part of the
    # identity function through an eigenvalue threshold on the joint basis.
    linear_variance=covariance[:,0,0].clamp_min(1e-16)
    regression=covariance[:,0,1:]/linear_variance[:,None]
    residual_cov=covariance[:,1:,1:]-covariance[:,1:,0,None]*regression[:,None,:]
    values,vectors=torch.linalg.eigh((residual_cov+residual_cov.transpose(-1,-2))*.5)
    inverse=torch.where(values>eigen_floor,values.clamp_min(eigen_floor).rsqrt(),0)
    residual_white=vectors*inverse[:,None,:]
    whitening=torch.zeros_like(covariance)
    whitening[:,0,0]=linear_variance.rsqrt()
    whitening[:,1:,1:]=residual_white
    whitening[:,0,1:]=-torch.einsum('ej,ejk->ek',regression,residual_white)
    state=dict(normalization=normalization,mean=mean.cpu(),whitening=whitening.cpu(),
               grid=grid,eigenvalues=values.cpu(),eigen_floor=eigen_floor)
    q=torch.einsum('ned,edk->nek',center,whitening).contiguous()
    return q,state


@torch.no_grad()
def replay_head_basis(h,state):
    raw,_=head_raw(h.double(),state['normalization'],state['grid'])
    return torch.einsum('ned,edk->nek',raw-state['mean'].to(h.device),state['whitening'].to(h.device)).contiguous()


@torch.no_grad()
def identity_anchor(state,device):
    white=state['whitening'].to(device)
    canonical=torch.zeros_like(state['mean'],device=device)
    canonical[:,0]=state['normalization']['scale'].to(device)
    weight=torch.einsum('ekd,ed->ek',torch.linalg.pinv(white,rtol=1e-10),canonical)
    bias=state['normalization']['mean'].to(device).sum()+(state['mean'].to(device)*canonical).sum()
    error=float((torch.einsum('edk,ek->ed',white,weight)-canonical).norm()/canonical.norm().clamp_min(1e-12))
    return dict(weight=weight,bias=bias,projection_error=error)


@torch.no_grad()
def continue_anchor(source,state,keep,device):
    old=source['basis'];old_model=source['model']
    canonical=torch.einsum('edk,ek->ed',old['whitening'][keep].to(device),old_model['weight'][keep].to(device))
    white=state['whitening'].to(device)
    weight=torch.einsum('ekd,ed->ek',torch.linalg.pinv(white,rtol=1e-10),canonical)
    shift=((state['mean'].to(device)-old['mean'][keep].to(device))*canonical).sum()
    error=float((torch.einsum('edk,ek->ed',white,weight)-canonical).norm()/canonical.norm().clamp_min(1e-12))
    return dict(weight=weight,bias=old_model['bias'].to(device)+shift,projection_error=error)


@torch.no_grad()
def solve_head(q,y,anchor,prior,lambda_fraction=.03,max_iterations=2000,tolerance=1e-7):
    """Native function-group proximal fit around retained KAN functions.

    Objective: CE + lambda sum_j RMS(psi_j) + prior/2 sum_j
    RMS(psi_j - anchor_j)^2. The training basis is orthonormal, so these
    are whole-function norms. The unpenalized intercept is fitted jointly.
    """
    matrix=q.flatten(1);n,regions,dim=q.shape
    yy=torch.as_tensor(y,device=q.device,dtype=torch.float64)
    w0=anchor['weight'].to(q.device);weight=w0.clone();bias=anchor['bias'].to(q.device).clone()
    lam=lambda_maximum(q,yy)*lambda_fraction
    # A deterministic spectral norm bound, followed by a checked line search.
    u=torch.cos(torch.arange(n,device=q.device,dtype=q.dtype));u/=u.norm()
    for _ in range(40):
        v=matrix@(matrix.T@u)+u.sum();u=v/v.norm().clamp_min(1e-20)
    L=max(float(u@(matrix@(matrix.T@u)+u.sum()))*.25/n*1.05,.25)
    ce=lambda w,b:(F.softplus(matrix@w.flatten()+b)-yy*(matrix@w.flatten()+b)).mean()
    objective=lambda w,b:ce(w,b)+.5*prior*(w-w0).square().sum()+lam*w.norm(dim=1).sum()
    z=weight.clone();zb=bias.clone();t=1.;previous=float(objective(weight,bias));history=[];converged=False
    for iteration in range(1,max_iterations+1):
        logits=matrix@z.flatten()+zb;residual=logits.sigmoid()-yy
        gradient=(matrix.T@residual/n).view(regions,dim);gb=residual.mean()
        ce_z=(F.softplus(logits)-yy*logits).mean()
        for _ in range(35):
            proposal=z-gradient/L+prior*w0/L
            candidate=proposal*(1-lam/(L*proposal.norm(dim=1,keepdim=True).clamp_min(1e-30))).clamp_min(0)/(1+prior/L)
            cb=zb-gb/L;delta=candidate-z;db=cb-zb;value=ce(candidate,cb)
            upper=ce_z+(gradient*delta).sum()+gb*db+.5*L*(delta.square().sum()+db.square())
            if float(value-upper)<=1e-12:break
            L*=2
        else:raise FloatingPointError('Staged KAN head line search failed')
        current=float(value+.5*prior*(candidate-w0).square().sum()+lam*candidate.norm(dim=1).sum())
        if not math.isfinite(current):raise FloatingPointError('Nonfinite staged KAN objective')
        if current>previous+1e-12 and t>1:
            z=weight.clone();zb=bias.clone();t=1.;continue
        rr=(matrix@candidate.flatten()+cb).sigmoid()-yy
        grad=(matrix.T@rr/n).view(regions,dim)+prior*(candidate-w0)
        norms=candidate.norm(dim=1);nonzero=norms>1e-12
        kkt=torch.where(nonzero,(grad+lam*candidate/norms.clamp_min(1e-30)[:,None]).norm(dim=1),
                        (grad.norm(dim=1)-lam).clamp_min(0))
        residual_norm=float(torch.sqrt(kkt.square().sum()+rr.mean().square()))
        new_t=(1+math.sqrt(1+4*t*t))/2;factor=(t-1)/new_t
        new_z=candidate+factor*(candidate-weight);new_b=cb+factor*(cb-bias)
        if float(((z-candidate)*(candidate-weight)).sum()+(zb-cb)*(cb-bias))>0:
            new_t=1.;new_z=candidate.clone();new_b=cb.clone()
        weight=candidate;bias=cb;z=new_z;zb=new_b;t=new_t;previous=current
        if iteration%10==0 or iteration==1:
            history.append(dict(iteration=iteration,objective=current,kkt_residual=residual_norm,
                                nonzero_functions=int(nonzero.sum())))
        if iteration>=50 and residual_norm<=tolerance*(1+float(weight.norm())):
            converged=True;break
    return dict(weight=weight.cpu(),bias=bias.cpu(),anchor_weight=w0.cpu(),prior=prior,lambda_value=lam,
                iterations=iteration,converged=converged,kkt_residual=residual_norm,objective=previous,
                history=history,nonzero_functions=int((weight.norm(dim=1)>1e-12).sum()))


@torch.no_grad()
def head_margin(h,head):
    q=replay_head_basis(h,head['basis'])
    return q.flatten(1)@head['model']['weight'].to(h.device).flatten()+head['model']['bias'].to(h.device)


@torch.no_grad()
def native_roi_scores(h,values,local,head):
    """Same KAN's RMS margin change under recipient + incident-function gates."""
    base=head_margin(h,head);scores=[]
    for roi in range(h.shape[1]):
        affected=(local==roi).any(0)
        change=torch.zeros_like(h)
        selected=np.flatnonzero(affected)
        for endpoint in range(2):
            index=torch.as_tensor(local[endpoint,selected],device=h.device)
            change.index_add_(1,index,.5*values[:,selected])
        trial=h-change
        q=replay_head_basis(trial,head['basis'])
        terms=(q*head['model']['weight'].to(h.device)[None]).sum(-1)
        terms[:,roi]=0
        margin=terms.sum(1)+head['model']['bias'].to(h.device)
        scores.append(torch.sqrt((base-margin).square().mean()))
    return torch.stack(scores).cpu().numpy()
