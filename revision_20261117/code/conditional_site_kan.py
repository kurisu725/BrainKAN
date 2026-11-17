"""Learn complete KAN curves with training-only conditional site alignment.

The extra loss aligns current hidden means across sites within each diagnosis.
It neither ranks ROIs nor enters inference. All input curves include an
explicit linear term, SiLU, and eight cubic spline coefficients.
"""
import copy,math,time
import numpy as np
import torch
from torch import nn
from residual_kan import primitive,PAIRS


class ConditionalSiteKAN(nn.Module):
    def __init__(self,first,head,bias,anchor,device='cuda'):
        super().__init__();self.first=nn.Parameter(torch.as_tensor(first,device=device,dtype=torch.float64).clone());self.head=nn.Parameter(torch.as_tensor(head,device=device,dtype=torch.float64).clone());self.bias=nn.Parameter(torch.as_tensor(bias,device=device,dtype=torch.float64).clone())
        self.width=self.first.shape[-1];assert self.first.shape[1]==10 and self.head.shape==(self.width,10)
        self.register_buffer('anchor',torch.as_tensor(anchor,device=device,dtype=torch.float64).clone());self.register_buffer('function_mask',torch.ones((len(self.first),self.width),device=device,dtype=torch.float64))
        self.register_buffer('head_grid',(torch.arange(-3,9,device=device,dtype=torch.float64)*.4-1).expand(self.width,-1).contiguous())

    def effective_first(self):return self.first*self.function_mask[:,None,:]
    def hidden(self,q):return q.flatten(1)@self.effective_first().flatten(0,1)
    def head_terms(self,h):
        z=h.tanh();p=primitive(z,self.head_grid);zero=primitive(torch.zeros(1,self.width,device=h.device,dtype=h.dtype),self.head_grid)
        return ((p-zero)*self.head[None]).sum(-1)
    def forward(self,q):return self.head_terms(self.hidden(q)).sum(1)+self.bias
    def curve_energy(self,cov):return torch.einsum('edh,edj,ejh->eh',self.effective_first(),cov,self.effective_first()).clamp_min(0)
    def alignment(self,q,contrasts):return .5*(contrasts@self.hidden(q).tanh()).square().sum()/self.width
    def regularizer(self,q,cov,contrasts,cfg):
        first=self.effective_first();delta=(self.first-self.anchor)*self.function_mask[:,None,:]
        anchor=.5*cfg['anchor_l2']*delta.square().sum();nonlinear=.5*cfg['nonlinear_l2']*first[:,1:].square().sum()
        group=cfg['function_group']*((self.curve_energy(cov)+1e-20).sqrt()*self.function_mask).sum()/math.sqrt(self.width)
        head=.5*cfg['head_l2']*self.head.square().sum();terms=self.head_terms(self.hidden(q));centered=terms-terms.mean(0)
        activity=cfg['head_activity']*(centered.square().mean(0)+1e-20).sqrt().sum()/math.sqrt(self.width)
        return anchor+nonlinear+group+head+activity+cfg['alignment_lambda']*self.alignment(q,contrasts)

    @torch.no_grad()
    def prune_functions(self,cov,fraction):
        values=self.curve_energy(cov).sqrt().cpu().numpy();weak=np.argsort(values.ravel(),kind='stable')[:int(values.size*fraction)];mask=np.ones(values.size,dtype=bool);mask[weak]=False;mask=mask.reshape(values.shape)
        self.function_mask.copy_(torch.as_tensor(mask,device=self.first.device,dtype=torch.float64));self.first.mul_(self.function_mask[:,None,:]);return dict(activity=values,weak_flat_positions=weak,active_mask=mask,removed_individual_curves=len(weak))

    @torch.no_grad()
    def roi_scores(self,q,features,rois):
        h=self.hidden(q);full=self(q);c=self.effective_first();pairs=PAIRS[:,features];scores=[]
        for roi in rois:
            incident=(pairs==roi).any(0);delta=q[:,incident].flatten(1)@c[incident].flatten(0,1);without=self.head_terms(h-delta).sum(1)+self.bias;scores.append((full-without).square().mean().sqrt())
        return torch.stack(scores).cpu().numpy()


def contrast_matrix(labels,sites):
    y=np.asarray(labels);s=np.asarray(sites);assert len(y)==len(s) and set(np.unique(y))=={0,1} and np.isfinite(s).all();n=len(y);rows=[];groups=[]
    for label in [0,1]:
        members=y==label;ny=int(members.sum())
        for site in np.unique(s[members]):
            selected=members&(s==site);ng=int(selected.sum());v=selected.astype(float)/ng-members.astype(float)/ny;rows.append(math.sqrt(ng/n)*v);groups.append(dict(label=label,site=float(site),count=ng,class_count=ny))
    return np.stack(rows),groups


def transform(raw,state,device='cuda'):
    x=np.asarray(raw[:,state['feature_indices']],dtype=np.float64);s=state['scaler'];z=np.clip((x-s['mean'])/s['std'],-3,3)-s['clip_center'];zz=torch.as_tensor(z,device=device,dtype=torch.float64);b=state['basis']
    return (primitive(zz,b['grid'].to(device))-b['mean'].to(device)).contiguous()


@torch.no_grad()
def prepare(fit_raw,training_indices,width,positive_fraction,device='cuda'):
    tic=time.perf_counter();x=np.asarray(fit_raw,dtype=np.float64);n,e=x.shape;assert e==4005 and n==len(training_indices) and n>width
    mean=x.mean(0);std=x.std(0).clip(min=1e-6);clipped=np.clip((x-mean)/std,-3,3);center=clipped.mean(0);z=torch.as_tensor(clipped-center,device=device,dtype=torch.float64)
    grid=(torch.arange(-3,9,device=device,dtype=torch.float64)*1.2-3).expand(e,-1).contiguous();p=primitive(z,grid);mu=p.mean(0);q=p-mu;cov=torch.einsum('ned,nej->edj',q,q)/n
    gram=z@z.T/n;eigen,vectors=torch.linalg.eigh((gram+gram.T)*.5);order=torch.arange(n-1,n-width-1,-1,device=device);values=eigen[order];assert float(values.min())>1e-8
    a=z.T@vectors[:,order]/(math.sqrt(n)*values[None]);sign=torch.sign(a[a.abs().argmax(0),torch.arange(width,device=device)]);a*=sign[None]
    first=torch.zeros(e,10,width,device=device,dtype=torch.float64);first[:,0]=a;head=torch.zeros(width,10,device=device,dtype=torch.float64);bias=float(np.log(positive_fraction/(1-positive_fraction)));net=ConditionalSiteKAN(first,head,bias,first,device)
    state=dict(feature_indices=np.arange(e),rois=np.arange(90),scaler=dict(mean=mean,std=std,clip_center=center),basis=dict(grid=grid.cpu(),mean=mu.cpu()),function_covariance=cov.cpu(),training_indices=np.asarray(training_indices),training_subjects=n,initialization_eigenvalues=values.cpu())
    if q.is_cuda:torch.cuda.synchronize(q.device)
    return net,state,time.perf_counter()-tic


@torch.no_grad()
def compact(net,state,rois,device='cuda'):
    rois=np.asarray(rois,dtype=int);ids=np.asarray(state['feature_indices']);pairs=PAIRS[:,ids];keep=np.isin(pairs[0],rois)&np.isin(pairs[1],rois)&net.function_mask.cpu().numpy().astype(bool).any(1)
    child=ConditionalSiteKAN(net.first.detach()[keep],net.head.detach(),net.bias.detach(),net.anchor[keep],device);child.function_mask.copy_(net.function_mask[keep]);s=copy.deepcopy(state);s.update(feature_indices=ids[keep],rois=rois,function_covariance=state['function_covariance'][keep].clone())
    for key in s['scaler']:s['scaler'][key]=np.asarray(state['scaler'][key])[keep].copy()
    for key in ['grid','mean']:s['basis'][key]=state['basis'][key][keep].clone()
    return child,s,keep


def cpu_export(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
def restore(w,device='cuda'):
    net=ConditionalSiteKAN(w['first'],w['head'],w['bias'],w['anchor'],device);net.load_state_dict({k:v.to(device) for k,v in w.items()});return net
