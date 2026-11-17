"""A KAN skip path plus a small tied-function nonlinear KAN branch.

The existing KAN's learned functions initialize the same classifier. Its
linear function projections initialize shared ROI factors, not an external
ranker. Both paths obey complete-function masks and raw ROI deletion.
"""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from persistent_gate_kan import PAIRS,inputs,compact_source


def primitive(z,grid):
    x=z[...,None];grid=grid.to(z.device).to(z.dtype)
    b=((x>=grid[:,:-1])&(x<grid[:,1:])).to(z.dtype)
    for degree in range(1,4):
        b=(x-grid[:,:-(degree+1)])/(grid[:,degree:-1]-grid[:,:-(degree+1)])*b[:,:,:-1]+(grid[:,degree+1:]-x)/(grid[:,degree+1:]-grid[:,1:-degree])*b[:,:,1:]
    return torch.cat([z[...,None],F.silu(z)[...,None],b],-1)


class ResidualKAN(nn.Module):
    def __init__(self,u,v,hidden_scale,pairs,bias,curve=None,device='cuda',dtype=torch.float32):
        super().__init__();u=torch.as_tensor(u,device=device,dtype=dtype);v=torch.as_tensor(v,device=device,dtype=dtype)
        width=u.shape[1];edges=len(pairs[0]);self.width=width
        if curve is None:
            curve=torch.zeros(width,10,device=device,dtype=dtype);curve[:,0]=1
        else:curve=torch.as_tensor(curve,device=device,dtype=dtype)
        self.u=nn.Parameter(u.clone());self.v=nn.Parameter(v.clone());self.curve=nn.Parameter(curve.clone())
        self.head=nn.Parameter(torch.zeros(width,10,device=device,dtype=dtype))
        self.base_amplitude=nn.Parameter(torch.ones((),device=device,dtype=dtype))
        self.bias=nn.Parameter(torch.as_tensor(bias,device=device,dtype=dtype).clone())
        self.register_buffer('initial_u',u.clone());self.register_buffer('initial_v',v.clone());self.register_buffer('initial_curve',curve.clone())
        self.register_buffer('initial_bias',self.bias.detach().clone())
        self.register_buffer('hidden_scale',torch.as_tensor(hidden_scale,device=device,dtype=dtype).clone())
        self.register_buffer('pairs',torch.as_tensor(pairs,device=device,dtype=torch.long))
        self.register_buffer('function_mask',torch.ones(edges,width,device=device,dtype=dtype))
        grid=(torch.arange(-3,9,device=device,dtype=dtype)*.4-1).expand(width,-1).contiguous()
        self.register_buffer('head_grid',grid)

    def coefficients(self):
        a=(self.u[self.pairs[0]]*self.v[self.pairs[1]]+self.u[self.pairs[1]]*self.v[self.pairs[0]])/math.sqrt(2)
        return a*self.function_mask/self.hidden_scale[None]

    def hidden(self,basis):
        a=self.coefficients();weight=a[:,None,:]*self.curve.T[None]
        return basis.flatten(1)@weight.flatten(0,1)

    def head_terms(self,h):
        z=h.tanh();q=primitive(z,self.head_grid)
        zero=primitive(torch.zeros(1,self.width,device=h.device,dtype=h.dtype),self.head_grid)
        return ((q-zero)*self.head[None]).sum(-1)

    def forward(self,basis,base_values):
        return self.base_amplitude*base_values.sum(1)+self.head_terms(self.hidden(basis)).sum(1)+self.bias

    def function_activity(self,covariance):
        energy=torch.einsum('kd,edj,kj->ek',self.curve,covariance,self.curve).clamp_min(0)
        return self.coefficients().abs()*(energy+1e-20).sqrt()

    def regularizer(self,basis,covariance,cfg):
        activity=self.function_activity(covariance)
        first=cfg['first_activity']*activity.sum()/self.function_mask.sum().clamp_min(1).sqrt()
        terms=self.head_terms(self.hidden(basis));centered=terms-terms.mean(0)
        head=cfg['head_activity']*(centered.square().mean(0)+1e-20).sqrt().sum()/math.sqrt(self.width)
        anchor=cfg['factor_anchor']*((self.u-self.initial_u).square().sum()+(self.v-self.initial_v).square().sum())
        anchor+=cfg['curve_anchor']*(self.curve-self.initial_curve).square().sum()
        anchor+=cfg['head_l2']*self.head.square().sum()
        anchor+=cfg['base_anchor']*((self.base_amplitude-1).square()+(self.bias-self.initial_bias).square())
        return first+head+anchor

    @torch.no_grad()
    def prune_functions(self,covariance,fraction):
        activity=self.function_activity(covariance)
        ids=torch.argsort(activity,dim=0,stable=True)[:int(len(activity)*fraction)]
        self.function_mask.scatter_(0,ids,0)
        return dict(activity=activity.cpu().numpy(),removed_per_channel=len(ids),mask=self.function_mask.cpu().numpy())

    @torch.no_grad()
    def roi_scores(self,basis,base_values,nroi):
        h=self.hidden(basis);full=self(basis,base_values);a=self.coefficients()
        curve_values=torch.einsum('ned,kd->nek',basis,self.curve)*a[None]
        score=[]
        for roi in range(nroi):
            incident=(self.pairs==roi).any(0)
            delta=curve_values[:,incident].sum(1)
            without=self.base_amplitude*(base_values[:,~incident].sum(1))+self.head_terms(h-delta).sum(1)+self.bias
            score.append((full-without).square().mean().sqrt())
        return torch.stack(score).cpu().numpy()


@torch.no_grad()
def prepare(raw,parent,ntrain,width,device='cuda'):
    assert len(parent['training_indices'])==ntrain
    q,z=inputs(raw,parent,device);base=(q*parent['model']['weight'].to(device).double()[None]).sum(-1)
    raw_basis=primitive(z,parent['basis']['grid'])
    mean=raw_basis[:ntrain].mean(0);basis=(raw_basis-mean).contiguous()
    covariance=torch.einsum('ned,nej->edj',basis[:ntrain],basis[:ntrain])/ntrain
    slopes=(base[:ntrain]*z[:ntrain]).mean(0)/z[:ntrain].square().mean(0).clamp_min(1e-8)
    pair=PAIRS[:,parent['feature_indices']];b=torch.zeros(90,90,dtype=torch.float64,device=device)
    b[pair[0],pair[1]]=slopes/2;b[pair[1],pair[0]]=slopes/2
    eigen,vectors=torch.linalg.eigh(b)
    order=torch.argsort(eigen.abs(),descending=True,stable=True)[:width]
    scale=eigen[order].abs().clamp_min(.001).sqrt()*2**.25
    u=vectors[:,order]*scale[None];v=u*torch.where(eigen[order]>=0,1.,-1.)[None]
    a=(u[pair[0]]*v[pair[1]]+u[pair[1]]*v[pair[0]])/math.sqrt(2)
    initial_hidden=basis[:ntrain,:,0]@a
    hidden_scale=initial_hidden.square().mean(0).sqrt().clamp_min(.05)
    net=ResidualKAN(u,v,hidden_scale,pair,parent['model']['bias'],device=device)
    # Retain only the parent quantities that inference actually uses.
    parent_small={k:copy.deepcopy(parent[k]) for k in ['basis','scaler','model','feature_indices','rois','training_indices']}
    state=dict(parent=parent_small,curve_mean=mean.cpu(),curve_covariance=covariance.cpu(),
        feature_indices=np.asarray(parent['feature_indices']),rois=np.arange(90),training_subjects=ntrain,
        initial_eigenvalues=eigen[order].cpu(),initial_slope_matrix=b.cpu())
    del q,z,raw_basis
    return net,basis,base,state


@torch.no_grad()
def replay_inputs(raw,state,device='cuda'):
    parent=state['parent'];q,z=inputs(raw,parent,device)
    base=(q*parent['model']['weight'].to(device).double()[None]).sum(-1)
    basis=primitive(z,parent['basis']['grid'])-state['curve_mean'].to(device).double()
    return basis.contiguous(),base


@torch.no_grad()
def compact(net,state,rois,device='cuda',dtype=torch.float32):
    rois=np.asarray(rois,dtype=int);old_rois=np.asarray(state['rois'])
    assert np.isin(rois,old_rois).all() and len(rois)>=2
    feature_keep=np.isin(PAIRS[0,state['feature_indices']],rois)&np.isin(PAIRS[1,state['feature_indices']],rois)
    gates=np.isin(np.arange(90),rois).astype(float)
    parent,factor,theta=compact_source(state['parent'],gates)
    assert (factor==1).all();parent['model']=dict(weight=theta,bias=state['parent']['model']['bias'])
    lookup={int(r):i for i,r in enumerate(rois)};pair=PAIRS[:,parent['feature_indices']]
    local=np.array([[lookup[int(r)] for r in pair[a]] for a in range(2)])
    keep_rois=np.isin(old_rois,rois)
    child=ResidualKAN(net.u.detach()[keep_rois],net.v.detach()[keep_rois],net.hidden_scale,local,net.bias.detach(),
        net.curve.detach(),device=device,dtype=dtype)
    with torch.no_grad():
        for name in ['head','base_amplitude','initial_curve','initial_bias','head_grid']:
            getattr(child,name).copy_(getattr(net,name).detach().to(device).to(dtype))
        child.initial_u.copy_(net.initial_u[keep_rois].to(device).to(dtype))
        child.initial_v.copy_(net.initial_v[keep_rois].to(device).to(dtype))
        child.function_mask.copy_(net.function_mask[feature_keep].to(device).to(dtype))
    child_state=dict(parent=parent,curve_mean=state['curve_mean'][feature_keep].clone(),
        curve_covariance=state['curve_covariance'][feature_keep].clone(),feature_indices=parent['feature_indices'],
        rois=rois,training_subjects=state['training_subjects'])
    return child,child_state,feature_keep


def retain_energy(scores,fraction,rois=None):
    scores=np.asarray(scores,dtype=np.float64);assert np.isfinite(scores).all() and (scores>=0).all()
    rois=np.arange(len(scores)) if rois is None else np.asarray(rois)
    order=np.argsort(-scores,kind='stable');energy=scores[order]**2;total=energy.sum()
    if total<=1e-24:return np.array([],dtype=int)
    count=int(np.searchsorted(np.cumsum(energy),fraction*total,side='left'))+1
    return np.sort(rois[order[:count]])


def cpu_export(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def restore(weights,device='cuda',dtype=torch.float64):
    net=ResidualKAN(weights['u'],weights['v'],weights['hidden_scale'],weights['pairs'],weights['bias'],
        weights['curve'],device=device,dtype=dtype)
    net.load_state_dict({k:v.to(device).to(dtype) if v.is_floating_point() else v.to(device) for k,v in weights.items()})
    return net
