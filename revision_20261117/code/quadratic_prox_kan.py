"""Native function-sparse KAN solver with the quadratic penalty in the prox.

For one function group, the exact metric prox minimizes
 .5 ||w-u||^2 + lambda/L ||w|| + .5/L w' P w.
Diagonalize I+P/L. If ||u|| exceeds lambda/L, its positive norm r solves
 sum_j v_j^2 / (d_j*r + lambda/L)^2 = 1.
This removes the large quadratic eigenvalue from the smooth-loss step size.
"""
import math
import torch
from torch.nn import functional as F
from functional_kan import lipschitz


class FunctionMetricProx:
    def __init__(self,penalty,bisections=36):
        self.penalty=penalty
        eig,vec=torch.linalg.eigh((penalty.double()+penalty.double().transpose(-1,-2))*.5)
        if float(eig.min())<-.01:raise ValueError('The native quadratic penalty must be PSD')
        self.eigen=eig.clamp_min(0);self.vectors=vec;self.bisections=bisections

    @torch.no_grad()
    def __call__(self,proposal,lam,L):
        v=torch.einsum('edk,ed->ek',self.vectors,proposal.double())
        diagonal=1+self.eigen/L;threshold=lam/L
        if lam==0:solution=v/diagonal
        else:
            norm=v.norm(dim=1);active=norm>threshold
            low=torch.zeros_like(norm);high=norm/diagonal.min(1).values
            for _ in range(self.bisections):
                middle=(low+high)*.5
                excess=(v/(diagonal*middle[:,None]+threshold)).square().sum(1)>1
                low=torch.where(excess,middle,low);high=torch.where(excess,high,middle)
            radius=(low+high)*.5
            solution=v*radius[:,None]/(diagonal*radius[:,None]+threshold)
            solution*=active[:,None]
        return torch.einsum('edk,ek->ed',self.vectors,solution).to(proposal.dtype)


@torch.no_grad()
def kkt_residual(matrix,yy,weight,bias,penalty,lam):
    margin=matrix@weight.flatten()+bias;residual=margin.sigmoid()-yy
    gradient=(matrix.T@residual/len(yy)).reshape_as(weight)+torch.einsum('edk,ek->ed',penalty,weight)
    norm=weight.norm(dim=1,keepdim=True);active=norm[:,0]>0
    error=gradient.clone()
    error[active]+=lam*weight[active]/norm[active]
    g=gradient[~active];gn=g.norm(dim=1,keepdim=True)
    error[~active]=g*(1-lam/gn.clamp_min(1e-30)).clamp_min(0)
    return float((error.square().sum()+residual.mean().square()).sqrt())


@torch.no_grad()
def solve(q,y,lam,penalty,max_iterations=1600,tolerance=1e-5,initial=None):
    n,edges,dimensions=q.shape;matrix=q.flatten(1);yy=torch.as_tensor(y,device=q.device,dtype=q.dtype)
    penalty=penalty.to(q.device);prox=FunctionMetricProx(penalty)
    weight=torch.zeros(edges,dimensions,device=q.device) if initial is None else initial['weight'].to(q.device).clone()
    bias=torch.logit(yy.mean().clamp(.001,.999)) if initial is None else initial['bias'].to(q.device).clone()
    def ce(w,b):
        margin=matrix@w.flatten()+b
        return (F.softplus(margin)-yy*margin).mean()
    def nonsmooth(w):
        return lam*w.norm(dim=1).sum()+.5*(w*torch.einsum('edk,ek->ed',penalty,w)).sum()
    L=lipschitz(matrix);z=weight.clone();zb=bias.clone();t=1.
    previous=float(ce(weight,bias)+nonsmooth(weight));history=[];converged=False;kkt=float('inf')
    for iteration in range(1,max_iterations+1):
        margin=matrix@z.flatten()+zb;residual=margin.sigmoid()-yy
        gradient=(matrix.T@residual/n).view(edges,dimensions);gb=residual.mean()
        old_ce=(F.softplus(margin)-yy*margin).mean()
        for _ in range(20):
            candidate=prox(z-gradient/L,lam,L);cb=zb-gb/L
            delta=candidate-z;db=cb-zb;new_ce=ce(candidate,cb)
            upper=old_ce+(gradient*delta).sum()+gb*db+.5*L*(delta.square().sum()+db.square())
            if float(new_ce-upper)<=2e-7:break
            L*=2
        else:raise FloatingPointError('Quadratic-prox KAN CE line search failed')
        current=float(new_ce+nonsmooth(candidate))
        if not math.isfinite(current):raise FloatingPointError('Nonfinite native quadratic-prox objective')
        if current>previous+2e-7 and t>1:
            z=weight.clone();zb=bias.clone();t=1.;continue
        mapping=float((L*delta).norm()+abs(L*db))
        new_t=(1+math.sqrt(1+4*t*t))/2;factor=(t-1)/new_t
        new_z=candidate+factor*(candidate-weight);new_b=cb+factor*(cb-bias)
        if float(((z-candidate)*(candidate-weight)).sum()+(zb-cb)*(cb-bias))>0:
            new_t=1.;new_z=candidate.clone();new_b=cb.clone()
        weight=candidate;bias=cb;z=new_z;zb=new_b;t=new_t;previous=current
        if iteration%10==0 or iteration==1 or iteration==max_iterations:
            kkt=kkt_residual(matrix,yy,weight,bias,penalty,lam)
            history.append(dict(iteration=iteration,objective=current,proximal_gradient_norm=mapping,
                kkt_residual=kkt,nonzero_functions=int((weight.norm(dim=1)>1e-8).sum()),lipschitz=L))
            if iteration>=30 and kkt<=tolerance*(1+float(weight.norm())):
                converged=True;break
    return dict(weight=weight.cpu(),bias=bias.cpu(),history=history,iterations=iteration,converged=converged,
        objective=previous,lambda_value=lam,proximal_gradient_norm=mapping,kkt_residual=kkt,
        nonzero_functions=int((weight.norm(dim=1)>1e-8).sum()),solver='native quadratic-metric group prox with CE acceleration')
