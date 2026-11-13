"""Independent reviewed variants. Old experimental checkpoints are never rewritten."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from legacy_kan import KANLinear, KAN


class AuditedKANLinear(KANLinear):
    @torch.no_grad()
    def update_grid(self, x, margin=0.01, chunk_size=32):
        """Fit on training activations only. Reproject UNSCALED coefficients.

        Chunk over input features to avoid allocating batch*8100*256 tensors.
        CPU float64 SVD solves also handle repeated/constant feature values.
        Grid relocation approximates the old curve; it is not exact in general.
        """
        if not self.training:
            raise RuntimeError('Grid adaptation is forbidden in eval mode')
        if x.ndim != 2 or x.shape[1] != self.in_features or len(x) < self.grid_size+1:
            raise ValueError('Insufficient or invalid calibration activations')
        if not torch.isfinite(x).all() or margin < 0 or not 0 < self.grid_eps <= 1:
            raise ValueError('Finite inputs, margin>=0 and grid_eps in (0,1] required')
        old_grid=self.grid.detach().double().cpu().clone()
        old_weight=self.spline_weight.detach().double().cpu()
        work=x.detach().double().cpu()
        sorted_x=work.sort(dim=0).values
        idx=torch.linspace(0,len(work)-1,self.grid_size+1).long()
        adaptive=sorted_x[idx]
        span=(sorted_x[-1]-sorted_x[0]+2*margin).clamp_min(1e-4)
        step=span/self.grid_size
        uniform=torch.arange(self.grid_size+1,dtype=work.dtype)[:,None]*step+sorted_x[0]-margin
        core=self.grid_eps*uniform+(1-self.grid_eps)*adaptive
        grid=torch.cat([core[:1]-step*torch.arange(self.spline_order,0,-1)[:,None],core,
                        core[-1:]+step*torch.arange(1,self.spline_order+1)[:,None]]).T
        def basis(xx,gg):
            xx=xx.unsqueeze(-1)
            b=((xx>=gg[:,:-1])&(xx<gg[:,1:])).to(xx.dtype)
            for k in range(1,self.spline_order+1):
                b=(xx-gg[:,:-(k+1)])/(gg[:,k:-1]-gg[:,:-(k+1)])*b[:,:,:-1]+(gg[:,k+1:]-xx)/(gg[:,k+1:]-gg[:,1:-k])*b[:,:,1:]
            return b
        new_weight=torch.empty_like(old_weight)
        err=den=0.0
        for start in range(0,self.in_features,chunk_size):
            sl=slice(start,min(start+chunk_size,self.in_features))
            xx=work[:,sl]
            target=torch.bmm(basis(xx,old_grid[sl]).transpose(0,1),old_weight[:,sl].permute(1,2,0))
            design=basis(xx,grid[sl]).transpose(0,1)
            coeff=torch.linalg.lstsq(design,target,driver='gelsd').solution
            new_weight[:,sl]=coeff.permute(2,0,1)
            # These are errors for the per-edge unscaled spline curves.
            err+=float((torch.bmm(design,coeff)-target).square().sum())
            den+=float(target.square().sum())
        if not torch.isfinite(new_weight).all() or not (grid.diff(dim=1)>0).all():
            raise FloatingPointError('Invalid adaptive grid or projected coefficients')
        self.grid.copy_(grid.to(self.grid))
        self.spline_weight.copy_(new_weight.to(self.spline_weight))
        return {'unscaled_projection_relative_l2':math.sqrt(err/max(den,1e-30))}

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Effective coefficient penalty; finite value/gradient at all-zero weights.

        This is a coefficient proxy, not pykan's sample-based activation penalty.
        Normalized L1 avoids changing lambda merely because width changed.
        """
        l1=self.scaled_spline_weight.abs().mean(-1)
        tiny=max(1e-12,torch.finfo(l1.dtype).eps)
        p=l1/l1.sum().clamp_min(tiny)
        entropy=-(p*p.clamp_min(tiny).log()).sum()/math.log(max(2,p.numel()))
        return regularize_activation*l1.mean()+regularize_entropy*entropy


class AuditedKAN(nn.Module):
    def __init__(self,widths,grid_size=5,spline_order=3):
        super().__init__()
        self.layers=nn.ModuleList([AuditedKANLinear(a,b,grid_size=grid_size,spline_order=spline_order)
                                  for a,b in zip(widths,widths[1:])])
    def forward(self,x):
        for layer in self.layers:x=layer(x)
        return x
    @torch.no_grad()
    def adapt_grid(self,x,optimizer=None):
        if not self.training:raise RuntimeError('Training-only grid adaptation')
        report=[]
        for layer in self.layers:
            before=float((layer.b_splines(x).abs().sum(-1)==0).float().mean())
            row=layer.update_grid(x)
            after=float((layer.b_splines(x).abs().sum(-1)==0).float().mean())
            row.update(in_features=layer.in_features,zero_basis_before=before,zero_basis_after=after)
            report.append(row)
            # Adam moments/SGD momentum are expressed in the old spline basis.
            if optimizer is not None:optimizer.state.pop(layer.spline_weight,None)
            x=layer(x)
        return report
    def regularization_loss(self):
        return sum(layer.regularization_loss() for layer in self.layers)/len(self.layers)


class E2E(nn.Module):
    def __init__(self,a,b,n):
        super().__init__();self.n=n
        self.row=nn.Conv2d(a,b,(1,n));self.col=nn.Conv2d(a,b,(n,1))
    def forward(self,x):
        return self.row(x).expand(-1,-1,-1,self.n)+self.col(x).expand(-1,-1,self.n,-1)


class BrainNetCNN(nn.Module):
    """Retained PyTorch topology, optional legacy output leaky-ReLU.

    Raw two-class logits are the default classification adaptation. The original
    ann4brains Caffe examples use regression; no exact Caffe training claim.
    """
    def __init__(self,n,p=.5,legacy_output=False):
        super().__init__();self.n=n;self.p=p;self.legacy_output=legacy_output
        self.e1=E2E(1,32,n);self.e2=E2E(32,64,n)
        self.e2n=nn.Conv2d(64,1,(1,n));self.n2g=nn.Conv2d(1,256,(n,1))
        self.f1=nn.Linear(256,128);self.f2=nn.Linear(128,30);self.f3=nn.Linear(30,2)
    def forward(self,x):
        x=x.reshape(-1,1,self.n,self.n)
        for layer in (self.e1,self.e2,self.e2n):x=F.leaky_relu(layer(x),.33)
        x=F.dropout(F.leaky_relu(self.n2g(x),.33),self.p,self.training).flatten(1)
        for layer in (self.f1,self.f2):x=F.dropout(F.leaky_relu(layer(x),.33),self.p,self.training)
        logits=self.f3(x)
        return F.leaky_relu(logits,.33) if self.legacy_output else logits


def normalized_adjacency(fc):
    """Nonnegative |FC| topology, self loops; signed FC remains node features.

    Magnitudes define edges to avoid negative degree square roots. This explicit
    graph-classification adaptation is not BrainGB or a published ABIDE score.
    """
    a=fc.abs().clone();n=a.shape[-1]
    i=torch.arange(n,device=a.device);a[:,i,i]=1.
    inv=a.sum(-1).clamp_min(1e-12).rsqrt()
    return inv[:,:,None]*a*inv[:,None,:]


class DenseGraphConv(nn.Module):
    """Kipf-Welling A_hat X W + bias, batched dense evaluation."""
    def __init__(self,a,b):
        super().__init__();self.weight=nn.Parameter(torch.empty(a,b));self.bias=nn.Parameter(torch.zeros(b))
        nn.init.uniform_(self.weight,-1/math.sqrt(b),1/math.sqrt(b))
    def forward(self,x,adj):return adj@(x@self.weight)+self.bias


class ConnectivityGCN(nn.Module):
    def __init__(self,n,hidden=32,dropout=.5):
        super().__init__();self.n=n;self.p=dropout
        self.g1=DenseGraphConv(n,hidden);self.g2=DenseGraphConv(hidden,hidden)
        self.out=nn.Linear(hidden,2)
    def forward(self,fc):
        fc=fc.reshape(-1,self.n,self.n);adj=normalized_adjacency(fc)
        x=F.dropout(F.relu(self.g1(fc,adj)),self.p,self.training)
        x=F.relu(self.g2(x,adj))
        return self.out(x.mean(1))


def build_model(kind,n,cfg):
    if kind=='kan_legacy':return KAN([n*n]+cfg['hidden']+[2],grid_size=cfg.get('grid_size',5),spline_order=cfg.get('spline_order',3))
    if kind=='kan_adaptive':return AuditedKAN([n*n]+cfg['hidden']+[2])
    if kind=='brainnetcnn':return BrainNetCNN(n,cfg['dropout'],cfg.get('legacy_output',False))
    if kind=='gcn':return ConnectivityGCN(n,cfg.get('gcn_hidden',32),cfg['dropout'])
    if kind=='mlp':
        dims=[n*n]+cfg['hidden']+[2];layers=[]
        for i,(a,b) in enumerate(zip(dims,dims[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(dims)-2:layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    raise ValueError(kind)
