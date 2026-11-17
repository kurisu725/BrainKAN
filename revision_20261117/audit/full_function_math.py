"""Independent NumPy full-function KAN predictions, objective and gradients."""
import numpy as np
from scipy.special import expit


def arr(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def primitives(z,grid):
    grid=arr(grid);x=z[...,None];b=((x>=grid[:,:-1])&(x<grid[:,1:])).astype(float)
    for degree in range(1,4):
        b=(x-grid[:,:-(degree+1)])/(grid[:,degree:-1]-grid[:,:-(degree+1)])*b[:,:,:-1]+(grid[:,degree+1:]-x)/(grid[:,degree+1:]-grid[:,1:-degree])*b[:,:,1:]
        if degree==2:b2=b.copy()
    derivative=3*b2[:,:,:-1]/(grid[:,3:-1]-grid[:,:-4])-3*b2[:,:,1:]/(grid[:,4:]-grid[:,1:-3])
    sigmoid=expit(z)
    values=np.concatenate([z[...,None],(z*sigmoid)[...,None],b],axis=-1)
    deriv=np.concatenate([np.ones_like(z)[...,None],(sigmoid+z*sigmoid*(1-sigmoid))[...,None],derivative],axis=-1)
    return values,deriv


def components(q,w):
    c=arr(w['first']).astype(float)*arr(w['function_mask'])[:,None,None]
    h=q.reshape(len(q),-1)@c.reshape(-1,c.shape[-1]);z=np.tanh(h)
    basis,deriv=primitives(z,w['head_grid']);zero,_=primitives(np.zeros((1,z.shape[1])),w['head_grid']);basis-=zero
    terms=np.einsum('nkd,kd->nk',basis,arr(w['head']));values=terms.sum(1)+float(w['bias'])
    return values,h,terms,basis,deriv,c


def objective_gradient(q,labels,penalty,lam,w,cfg):
    values,h,terms,basis,deriv,c=components(q,w);n=len(labels);head=arr(w['head']).astype(float);mask=arr(w['function_mask'])
    ce=float(np.mean(np.logaddexp(0,values)-labels*values));norm=np.linalg.norm(c.reshape(len(c),-1),axis=1)
    quadratic=.5*np.einsum('edk,edj,ejk->',c,penalty,c);group=lam*norm.sum()
    centered=terms-terms.mean(0);rms=np.sqrt((centered*centered).mean(0)+1e-20)
    head_l2=.5*cfg['head_l2']*(head*head).sum();activity=cfg['head_activity']*rms.sum()/np.sqrt(c.shape[-1])
    regularizer=float(quadratic+group+head_l2+activity)
    residual=(expit(values)-labels)/n
    derivative_terms=residual[:,None]+cfg['head_activity']/np.sqrt(c.shape[-1])*centered/(n*rms[None])
    grad_head=np.einsum('nk,nkd->kd',derivative_terms,basis)+cfg['head_l2']*head
    derivative_h=derivative_terms*np.einsum('nkd,kd->nk',deriv,head)*(1-np.tanh(h)**2)
    grad_first=(q.reshape(n,-1).T@derivative_h).reshape(c.shape)
    grad_first+=np.einsum('edj,ejk->edk',(penalty+penalty.swapaxes(-1,-2))*.5,c)
    grad_first+=lam*np.divide(c,norm[:,None,None],out=np.zeros_like(c),where=norm[:,None,None]>0)
    grad_first*=mask[:,None,None]
    grad_bias=float(residual.sum());gradient_norm=float(np.sqrt((grad_first*grad_first).sum()+(grad_head*grad_head).sum()+grad_bias**2))
    return dict(values=values,ce=ce,regularizer=regularizer,objective=ce+regularizer,
        gradient_norm=gradient_norm,grad_first=grad_first,grad_head=grad_head,grad_bias=grad_bias)


def function_activity(q,w):
    c=arr(w['first']).astype(float)*arr(w['function_mask'])[:,None,None]
    cov=np.einsum('ned,nej->edj',q,q,optimize=True)/len(q)
    return np.sqrt(np.einsum('edk,edj,ejk->e',c,cov,c,optimize=True).clip(min=0))


def roi_scores(q,w,feature_ids,pairs,rois):
    full,h,_,_,_,c=components(q,w);pair=pairs[:,feature_ids];scores=[]
    for roi in rois:
        incident=(pair==roi).any(0);delta=q[:,incident].reshape(len(q),-1)@c[incident].reshape(-1,c.shape[-1])
        z=np.tanh(h-delta);b,_=primitives(z,w['head_grid']);zero,_=primitives(np.zeros((1,z.shape[1])),w['head_grid'])
        without=np.einsum('nkd,kd->n',b-zero,arr(w['head']))+float(w['bias'])
        scores.append(float(np.sqrt(np.mean((full-without)**2))))
    return np.asarray(scores)
