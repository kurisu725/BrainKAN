"""Official architectures with AAL90 input and current-library adapters."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from vendor.li_braingnn.braingnn import Network
from vendor.braingb import GCN
from vendor.BNT.bnt import BrainNetworkTransformer

def build(name):
    if name=='braingnn':return Network(90,.5,2,k=8,R=90)
    if name=='braingb':
        args=SimpleNamespace(pooling='concat',hidden_dim=360,n_GNN_layers=2,edge_emb_dim=256,gcn_mp_type='node_concate',bucket_sz=.05)
        return GCN(90,args,90,2)
    if name=='brainnettf':
        cfg=OmegaConf.create({'dataset':{'node_sz':90},'model':{'pos_encoding':'none','sizes':[90,45],'pooling':[False,True],'orthogonal':True,'freeze_center':True,'project_assignment':True}})
        return BrainNetworkTransformer(cfg)
    raise ValueError(name)

def graph(fc):
    # The archived correlation profile is the node feature. Only edges are absolute.
    b,n,_=fc.shape
    mask=~torch.eye(n,dtype=torch.bool,device=fc.device)
    row,col=mask.nonzero(as_tuple=True)
    edge=torch.stack([row,col])[None]+torch.arange(b,device=fc.device)[:,None,None]*n
    edge=edge.permute(1,0,2).reshape(2,-1)
    attr=fc[:,row,col].abs().reshape(-1)
    batch=torch.arange(b,device=fc.device).repeat_interleave(n)
    return fc.reshape(b*n,n),edge,attr,batch

def forward(name,model,fc):
    if name=='brainnettf':return model(None,fc),None
    x,e,w,b=graph(fc)
    if name=='braingb':return model(x,e,w,b),None
    pos=torch.eye(90,device=fc.device).repeat(len(fc),1)
    result=model(x,e,b,w,pos)
    return result[0],result[1:]

def li_regularizer(aux,y):
    w1,w2,s1,s2=aux
    def topk(s):
        s=s.sort(dim=1).values;k=int(s.shape[1]*.5)
        return -(s[:,-k:]+1e-10).log().mean()-(1-s[:,:k]+1e-10).log().mean()
    consistency=s1.new_zeros(())
    for c in (0,1):
        s=s1[y==c].sigmoid()
        if len(s):
            # Equivalent to trace(S.T (nI-11.T) S) / n^2.
            consistency+=((s-s.mean(0))**2).sum()/len(s)
    return .1*(topk(s1)+topk(s2)+consistency)

def train_loss(name,model,x,y):
    if name in ('brainnettf','braingb'):
        import numpy as np
        lam=float(np.random.beta(1.,1.));per=torch.randperm(len(y),device=y.device)
        out,_=forward(name,model,lam*x+(1-lam)*x[per])
        reduction='sum' if name=='brainnettf' else 'mean'
        return lam*F.cross_entropy(out,y,reduction=reduction)+(1-lam)*F.cross_entropy(out,y[per],reduction=reduction)
    out,aux=forward(name,model,x)
    loss=F.cross_entropy(out,y)
    if name=='braingnn':loss+=li_regularizer(aux,y)
    return loss
