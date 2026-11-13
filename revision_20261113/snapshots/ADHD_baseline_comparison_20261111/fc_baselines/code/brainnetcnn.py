import torch
from torch import nn
from torch.nn import functional as F

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


