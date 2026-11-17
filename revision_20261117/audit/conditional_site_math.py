"""Independent NumPy graph and analytic conditional-mean-loss gradients."""
import numpy as np
from scipy.special import expit
from full_function_math import arr,primitives


def transform(raw,state):
    x=np.asarray(raw[:,state['feature_indices']],dtype=float);s=state['scaler'];z=np.clip((x-s['mean'])/s['std'],-3,3)-s['clip_center']
    return primitives(z,state['basis']['grid'])[0]-arr(state['basis']['mean'])


def components(q,w):
    c=arr(w['first'])*arr(w['function_mask'])[:,None,:];h=q.reshape(len(q),-1)@c.reshape(-1,c.shape[-1]);z=np.tanh(h)
    p,dp=primitives(z,w['head_grid']);zero=primitives(np.zeros((1,z.shape[1])),w['head_grid'])[0];p=p-zero;terms=np.einsum('nhd,hd->nh',p,arr(w['head']));v=terms.sum(1)+float(w['bias'])
    return v,h,z,terms,p,dp,c


def alignment(q,w,d):
    z=components(q,w)[2];return float(.5*np.sum((d@z)**2)/z.shape[1])


def objective_gradient(q,y,cov,d,w,cfg):
    v,h,z,terms,p,dp,c=components(q,w);n=len(q);head=arr(w['head']);mask=arr(w['function_mask']);width=c.shape[-1];delta=(arr(w['first'])-arr(w['anchor']))*mask[:,None,:]
    energy=np.einsum('edh,edj,ejh->eh',c,cov,c,optimize=True);norm=np.sqrt(np.maximum(energy,0)+1e-20);centered=terms-terms.mean(0);rms=np.sqrt((centered*centered).mean(0)+1e-20)
    ae=.5*np.sum((d@z)**2)/width;ce=float(np.mean(np.logaddexp(0,v)-y*v));reg=.5*cfg['anchor_l2']*(delta*delta).sum()+.5*cfg['nonlinear_l2']*(c[:,1:]**2).sum()+cfg['function_group']*(norm*mask).sum()/np.sqrt(width)
    reg+=.5*cfg['head_l2']*(head*head).sum()+cfg['head_activity']*rms.sum()/np.sqrt(width)+cfg['alignment_lambda']*ae;reg=float(reg)
    r=(expit(v)-y)/n;dt=r[:,None]+cfg['head_activity']/np.sqrt(width)*centered/(n*rms[None]);gh=np.einsum('nh,nhd->hd',dt,p)+cfg['head_l2']*head
    dh=dt*np.einsum('nhd,hd->nh',dp,head);dh+=cfg['alignment_lambda']/width*(d.T@(d@z));dh*=1-z*z
    gf=(q.reshape(n,-1).T@dh).reshape(c.shape)+cfg['anchor_l2']*delta
    gf[:,1:]+=cfg['nonlinear_l2']*c[:,1:];group=np.einsum('edj,ejh->edh',(cov+cov.transpose(0,2,1))/2,c,optimize=True)/norm[:,None,:];group*=((energy>=0)*mask)[:,None,:]
    gf+=cfg['function_group']/np.sqrt(width)*group;gf*=mask[:,None,:];gb=float(r.sum());gn=float(np.sqrt((gf*gf).sum()+(gh*gh).sum()+gb*gb))
    return dict(values=v,ce=ce,regularizer=reg,objective=ce+reg,alignment_energy=float(ae),gradient_norm=gn,grad_first=gf,grad_head=gh,grad_bias=gb)


def activity(cov,w):
    c=arr(w['first'])*arr(w['function_mask'])[:,None,:];return np.sqrt(np.maximum(np.einsum('edh,edj,ejh->eh',c,cov,c,optimize=True),0))


def roi_scores(q,w,feature_ids,pairs,rois):
    full,h,_,_,_,_,c=components(q,w);pair=pairs[:,feature_ids];scores=[]
    for roi in rois:
        incident=(pair==roi).any(0);delta=q[:,incident].reshape(len(q),-1)@c[incident].reshape(-1,c.shape[-1]);z=np.tanh(h-delta);p=primitives(z,w['head_grid'])[0];zero=primitives(np.zeros((1,z.shape[1])),w['head_grid'])[0]
        without=np.einsum('nhd,hd->n',p-zero,arr(w['head']))+float(w['bias']);scores.append(np.sqrt(np.mean((full-without)**2)))
    return np.asarray(scores)


def contrasts(y,sites):
    rows=[];groups=[];n=len(y)
    for label in sorted(set(y)):
        c=np.flatnonzero(y==label)
        for site in sorted(set(sites[c])):
            g=c[sites[c]==site];v=np.zeros(n);v[g]=1/len(g);v[c]-=1/len(c);rows.append(v*np.sqrt(len(g)/n));groups.append(dict(label=int(label),site=float(site),count=len(g),class_count=len(c)))
    return np.asarray(rows),groups
