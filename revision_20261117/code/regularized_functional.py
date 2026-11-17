"""Function-sparse KAN with internal amplitude and nonlinearity penalties."""
import math
import torch
from torch.nn import functional as F
from functional_kan import lipschitz


@torch.no_grad()
def function_penalty(q,z,rms_weight,nonlinearity_weight):
    n=len(q);cov=torch.einsum('ned,nek->edk',q,q)/n
    corr=torch.einsum('ned,ne->ed',q,z)/n/z.square().mean(0).clamp_min(1e-8).sqrt()[:,None]
    residual=cov-corr[:,:,None]*corr[:,None,:]
    penalty=rms_weight*cov+nonlinearity_weight*residual
    eigen,vectors=torch.linalg.eigh((penalty+penalty.transpose(-1,-2))*.5)
    # Clamp only floating-point negative eigenvalues of this PSD covariance.
    if float(eigen.min())<-.01:raise ValueError('Invalid function covariance penalty')
    return (vectors*eigen.clamp_min(0)[:,None,:])@vectors.transpose(-1,-2)


@torch.no_grad()
def transfer_functions(old_model,old_basis,new_basis):
    """Preserve retained univariate SiLU/spline coefficients when changing basis.

If the new training basis drops an unidentifiable direction, this is its
projection onto the new identified function space, with an intercept shift.
The input normalization itself is allowed to change after ROI deletion.
"""
    device=new_basis['whitening'].device
    old_white=old_basis['whitening'].to(device);weight=old_model['weight'].to(device)
    canonical=torch.einsum('edk,ek->ed',old_white,weight)
    white=new_basis['whitening'].to(device)
    # Solve in float64: multiplying approximate float32 eigenvectors by their
    # eigenvalues needlessly perturbs an otherwise identical function basis.
    inverse=torch.linalg.pinv(white.double(),rtol=1e-9)
    new_weight=torch.einsum('ekd,ed->ek',inverse,canonical.double()).to(weight.dtype)
    reconstructed=torch.einsum('edk,ek->ed',white,new_weight)
    shift=((new_basis['mean'].to(device)-old_basis['mean'].to(device))*canonical).sum()
    return dict(weight=new_weight.cpu(),bias=(old_model['bias'].to(device)+shift).cpu(),
                canonical_coefficient_projection_error=float((reconstructed-canonical).norm()/canonical.norm().clamp_min(1e-12)))


@torch.no_grad()
def solve(q,y,lam,penalty,max_iterations=1600,tolerance=1e-5,initial=None):
    n,edges,dimensions=q.shape;matrix=q.flatten(1);yy=torch.as_tensor(y,device=q.device,dtype=q.dtype)
    penalty=penalty.to(q.device)
    weight=torch.zeros(edges,dimensions,device=q.device) if initial is None else initial['weight'].to(q.device).clone()
    bias=torch.logit(yy.mean().clamp(.001,.999)) if initial is None else initial['bias'].to(q.device).clone()
    quad_grad=lambda w:torch.einsum('edk,ek->ed',penalty,w)
    quad=lambda w:.5*(w*quad_grad(w)).sum()
    def smooth(w,b):
        v=matrix@w.flatten()+b
        return (F.softplus(v)-yy*v).mean()+quad(w)
    L=lipschitz(matrix)+float(torch.linalg.eigvalsh(penalty).max())
    z=weight.clone();zb=bias.clone();t=1.;previous=float(smooth(weight,bias)+lam*weight.norm(dim=1).sum())
    history=[];converged=False
    for iteration in range(1,max_iterations+1):
        logits=matrix@z.flatten()+zb;residual=logits.sigmoid()-yy
        gradient=(matrix.T@residual/n).view(edges,dimensions)+quad_grad(z);gb=residual.mean()
        smooth_z=(F.softplus(logits)-yy*logits).mean()+quad(z)
        for _ in range(20):
            proposal=z-gradient/L
            candidate=proposal*(1-lam/(L*proposal.norm(dim=1,keepdim=True).clamp_min(1e-20))).clamp_min(0)
            cb=zb-gb/L;delta=candidate-z;db=cb-zb;value=smooth(candidate,cb)
            upper=smooth_z+(gradient*delta).sum()+gb*db+.5*L*(delta.square().sum()+db.square())
            if float(value-upper)<=2e-7:break
            L*=2
        else:raise FloatingPointError('Function-regularized KAN line search failed')
        current=float(value+lam*candidate.norm(dim=1).sum())
        if not math.isfinite(current):raise FloatingPointError('Nonfinite native objective')
        if current>previous+2e-7 and t>1:
            z=weight.clone();zb=bias.clone();t=1.;continue
        mapping=float((L*delta).norm()+abs(L*db));new_t=(1+math.sqrt(1+4*t*t))/2
        factor=(t-1)/new_t;new_z=candidate+factor*(candidate-weight);new_b=cb+factor*(cb-bias)
        if float(((z-candidate)*(candidate-weight)).sum()+(zb-cb)*(cb-bias))>0:
            new_t=1.;new_z=candidate.clone();new_b=cb.clone()
        weight=candidate;bias=cb;z=new_z;zb=new_b;t=new_t;previous=current
        if iteration%10==0 or iteration==1:
            history.append(dict(iteration=iteration,objective=current,proximal_gradient_norm=mapping,
                                nonzero_functions=int((weight.norm(dim=1)>1e-8).sum())))
        if iteration>=50 and mapping<=tolerance*(1+float(weight.norm())):converged=True;break
    return dict(weight=weight.cpu(),bias=bias.cpu(),history=history,iterations=iteration,converged=converged,
        objective=previous,lambda_value=lam,proximal_gradient_norm=mapping,
        nonzero_functions=int((weight.norm(dim=1)>1e-8).sum()))
