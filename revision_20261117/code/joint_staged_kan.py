"""End-to-end continuation of a pretrained native two-layer KAN.

Regional offsets preserve retained hidden functions when centered FC bases
change. All inference uses only the physically retained raw ROI submatrix.
Unlike the staged diagnostic, both learned layers optimize the same final
classification objective after pruning. All arithmetic here is float64.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from legacy_kan import KANLinear
from protected_site_kan import prepare
from staged_kan import PAIRS,head_basis,continue_anchor


def raw_first_basis(z,grid=None):
    layer=KANLinear(z.shape[1],1,grid_size=5,spline_order=3,grid_range=[-3,3]).to(z.device).double()
    if grid is not None:layer.grid.copy_(grid.to(z.device))
    return torch.cat([F.silu(z)[...,None],layer.b_splines(z)],-1),layer.grid


@torch.no_grad()
def fixed_first_q(z,state):
    raw,_=raw_first_basis(z.double(),state['grid'])
    return torch.einsum('ned,edk->nek',raw-state['mean'].to(z.device).double(),state['whitening'].to(z.device).double()).contiguous()


@torch.no_grad()
def precise_first_basis(z,n,eigen_floor=1e-5):
    raw,grid=raw_first_basis(z);mu=raw[:n].mean(0);center=raw-mu
    covariance=torch.einsum('ned,nek->edk',center[:n],center[:n])/n
    vals,vecs=torch.linalg.eigh((covariance+covariance.transpose(-1,-2))*.5)
    inv=torch.where(vals>eigen_floor,vals.clamp_min(eigen_floor).rsqrt(),0)
    white=vecs*inv[:,None,:]
    q=torch.einsum('ned,edk->nek',center,white).contiguous()
    state=dict(mean=mu.cpu(),whitening=white.cpu(),grid=grid.cpu(),eigenvalues=vals.cpu(),eigen_floor=eigen_floor)
    return q,state


@torch.no_grad()
def transfer_first(weight,old_basis,new_basis):
    canonical=torch.einsum('edk,ek->ed',old_basis['whitening'].double(),weight.cpu().double())
    white=new_basis['whitening'].double()
    new=torch.einsum('ekd,ed->ek',torch.linalg.pinv(white,rtol=1e-10),canonical)
    shift=((new_basis['mean'].double()-old_basis['mean'].double())*canonical).sum(1)
    error=float((torch.einsum('edk,ek->ed',white,new)-canonical).norm()/canonical.norm().clamp_min(1e-12))
    return new,shift,error


def incidence(feature_ids,rois,device):
    lookup={int(r):i for i,r in enumerate(rois)};pair=PAIRS[:,feature_ids]
    local=np.array([[lookup[int(r)] for r in pair[a]] for a in range(2)])
    mat=torch.zeros(len(feature_ids),len(rois),device=device,dtype=torch.float64)
    rows=torch.arange(len(feature_ids),device=device)
    mat[rows,torch.tensor(local[0],device=device)]=.5;mat[rows,torch.tensor(local[1],device=device)]=.5
    return mat,local


class JointStagedKAN(nn.Module):
    def __init__(self,first_weight,regional_bias,head,feature_ids,rois,device='cuda'):
        super().__init__();self.rois=np.asarray(rois);self.feature_ids=np.asarray(feature_ids)
        inc,self.local=incidence(self.feature_ids,self.rois,device);self.register_buffer('incidence',inc)
        self.first_weight=nn.Parameter(first_weight.to(device).double().clone())
        self.regional_bias=nn.Parameter(regional_bias.to(device).double().clone())
        self.head_weight=nn.Parameter(head['model']['weight'].to(device).double().clone())
        self.head_bias=nn.Parameter(head['model']['bias'].to(device).double().clone())
        self.head_state=copy.deepcopy(head['basis'])
        for name in ['mean','whitening','grid']:self.register_buffer('head_'+name,head['basis'][name].to(device).double())
        self.register_buffer('head_norm_mean',head['basis']['normalization']['mean'].to(device).double())
        self.register_buffer('head_norm_scale',head['basis']['normalization']['scale'].to(device).double())

    def head_terms(self,h):
        z=(h-self.head_norm_mean)/self.head_norm_scale;x=z[...,None];grid=self.head_grid
        b=((x>=grid[:,:-1])&(x<grid[:,1:])).to(z.dtype)
        for degree in range(1,4):
            b=(x-grid[:,:-(degree+1)])/(grid[:,degree:-1]-grid[:,:-(degree+1)])*b[:,:,:-1]+(grid[:,degree+1:]-x)/(grid[:,degree+1:]-grid[:,1:-degree])*b[:,:,1:]
        raw=torch.cat([z[...,None],F.silu(z)[...,None],b],-1)
        q=torch.einsum('ned,edk->nek',raw-self.head_mean,self.head_whitening)
        return (q*self.head_weight[None]).sum(-1)

    def components(self,q):
        values=(q*self.first_weight[None]).sum(-1)
        h=values@self.incidence+self.regional_bias
        terms=self.head_terms(h)
        return terms.sum(1)+self.head_bias,values,h,terms

    def forward(self,q):return self.components(q)[0]

    @torch.no_grad()
    def native_scores(self,q):
        full,values,h,_=self.components(q);scores=[]
        for roi in range(len(self.rois)):
            affected=(self.local==roi).any(0)
            delta=values[:,affected]@self.incidence[affected]
            terms=self.head_terms(h-delta);terms[:,roi]=0
            scores.append(torch.sqrt((full-(terms.sum(1)+self.head_bias)).square().mean()))
        return torch.stack(scores).cpu().numpy()

    @torch.no_grad()
    def masked_margin(self,q,keep_rois):
        _,values,h,_=self.components(q);keep=np.isin(self.feature_ids,self.feature_ids[np.isin(PAIRS[0,self.feature_ids],keep_rois)&np.isin(PAIRS[1,self.feature_ids],keep_rois)])
        deleted=~keep;delta=values[:,deleted]@self.incidence[deleted]
        terms=self.head_terms(h-delta);terms[:,~np.isin(self.rois,keep_rois)]=0
        return terms.sum(1)+self.head_bias

    def export_head(self):
        return dict(basis=copy.deepcopy(self.head_state),model=dict(weight=self.head_weight.detach().cpu().clone(),bias=self.head_bias.detach().cpu().clone()))


def subset_basis(state,mask):
    return {key:(val[mask] if torch.is_tensor(val) and val.ndim and val.shape[0]==len(mask) else copy.deepcopy(val)) for key,val in state.items()}


@torch.no_grad()
def initialize(raw,y,sites,n,mode,training_indices,source,head_source):
    feature_ids=np.asarray(source['feature_indices'])
    z,scaler=prepare(raw,y,sites,n,mode);scaler['fitted_indices']=np.asarray(training_indices).tolist()
    zz=z[:,feature_ids].double();basis=copy.deepcopy(source['basis']);q=fixed_first_q(zz,basis)
    weights=source['model']['weight'].double();error=0.
    rb=torch.full((90,),float(source['model']['bias'])/90,dtype=torch.float64)
    head=copy.deepcopy(head_source)
    net=JointStagedKAN(weights,rb,head,feature_ids,np.arange(90))
    state=dict(scaler=scaler,basis=basis,feature_indices=feature_ids,rois=np.arange(90),
        training_indices=np.asarray(training_indices),first_projection_error=error,head_projection_error=0.)
    return net,q,zz,state


@torch.no_grad()
def compact(raw,y,sites,n,mode,training_indices,source_net,source_state,rois):
    rois=np.asarray(rois);raw_mask=np.isin(PAIRS[0],rois)&np.isin(PAIRS[1],rois);raw_ids=np.flatnonzero(raw_mask)
    features=np.intersect1d(raw_ids,source_state['feature_indices'],assume_unique=True)
    oldmask=np.isin(source_state['feature_indices'],features);active=np.isin(raw_ids,features)
    z,scaler=prepare(raw,y,sites,n,mode,keep=raw_mask);scaler['fitted_indices']=np.asarray(training_indices).tolist()
    zz=z[:,active].double();basis=subset_basis(source_state['basis'],oldmask);q=fixed_first_q(zz,basis)
    weights=source_net.first_weight.detach().cpu()[oldmask];error=0.
    roi_keep=np.isin(source_net.rois,rois)
    # Keep learned regional constants; do not redistribute one global bias
    # across a different number of regions after deleting a recipient.
    rb=source_net.regional_bias.detach().cpu()[roi_keep]
    fullhead=source_net.export_head();headstate=subset_basis(fullhead['basis'],roi_keep)
    headstate['normalization']={key:val[roi_keep] for key,val in fullhead['basis']['normalization'].items()}
    head=dict(basis=headstate,model=dict(weight=fullhead['model']['weight'][roi_keep],bias=fullhead['model']['bias']))
    net=JointStagedKAN(weights,rb,head,features,rois)
    state=dict(scaler=scaler,basis=basis,feature_indices=features,rois=rois,
        training_indices=np.asarray(training_indices),first_projection_error=error,head_projection_error=0.)
    return net,q,zz,state


def objective(net,q,y,penalty,lam,anchor,cfg):
    logits,values,h,terms=net.components(q)
    ce=F.binary_cross_entropy_with_logits(logits,y)
    centered=values-values.mean(0);rms=(centered.square().mean(0)+1e-20).sqrt()
    native=lam*rms.sum()+.5*(net.first_weight*torch.einsum('edk,ek->ed',penalty,net.first_weight)).sum()
    hc=h-h.mean(0);tc=terms-terms.mean(0)
    slope=(hc*tc).mean(0)/hc.square().mean(0).clamp_min(1e-12)
    nonlinear=tc-hc*slope
    shape=.5*cfg['head_shape']*nonlinear.square().mean(0).sum()
    head_rms=cfg['head_function_sparsity']*(tc.square().mean(0)+1e-20).sqrt().sum()
    prior=.5*cfg['head_anchor']*(net.head_weight-anchor['head_weight']).square().sum()+.5*cfg['regional_anchor']*(net.regional_bias-anchor['regional_bias']).square().sum()
    return ce+native+shape+head_rms+prior,dict(ce=float(ce.detach()),first_native=float(native.detach()),head_shape=float(shape.detach()),head_rms=float(head_rms.detach()),anchor=float(prior.detach()))
