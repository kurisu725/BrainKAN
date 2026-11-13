"""Documented AAL90/variable-length ports and a paper-based Res-DAE implementation."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch_geometric.data import Data,Batch
from fbnetgen_official import FBNETGEN
from fbnetgen_loss import mixup_cluster_loss
from brainnetgnn_vendor.encoders_fMRI import NatureOneCNN
from brainnetgnn_vendor.All_Architecture import combinedModel
from brainnetgnn_vendor.pyg_class import Net

class VariableFBNetGen(FBNETGEN):
    def __init__(self,embedding_size=16):
        cfg=dict(extractor_type='gru',embedding_size=embedding_size,window_size=4,num_gru_layers=4,dropout=.5,graph_generation='product')
        super().__init__(cfg,roi_num=90,node_feature_dim=90,time_series=100)
    def forward(self,t,nodes,lengths):
        b,r,_=t.shape;windows=lengths.cpu()//self.extract.kernel_size
        maximum=int(windows.max());raw=t[:,:,:maximum*4].contiguous().view(b*r,maximum,4)
        lens=windows.repeat_interleave(r)
        packed=pack_padded_sequence(raw,lens,batch_first=True,enforce_sorted=False)
        output,_=self.extract.gru(packed);out,_=pad_packed_sequence(output,batch_first=True)
        last=out[torch.arange(b*r,device=t.device),lens.to(t.device)-1].reshape(b,r,-1)
        emb=self.extract.linear(last).softmax(-1)
        matrix=self.emb2graph(emb)[:,:,:,0]
        return self.predictor(matrix,nodes),matrix

class VariableBrainNetGNN(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        args=SimpleNamespace(feature_size=64,feature_size_pre_training=256,no_downsample=True,end_with_relu=False)
        self.core=combinedModel(NatureOneCNN(1,args),Net(n_regions=90),1,PT='',exp='NPT',device=device)
        self.core.init_weight(PT='NPT')
    def forward(self,t,lengths):
        b,r,_=t.shape
        encoded=torch.stack([self.core.encoder(t[i,:,:int(lengths[i])].unsqueeze(1)) for i in range(b)])
        auxiliary=self.core.encoder_decoder(encoded.reshape(b,-1))
        nodes,attn=self.core.multi_head_attention(encoded)
        grid=torch.arange(r,device=t.device)
        edges=torch.stack([grid.repeat_interleave(r),grid.repeat(r)])
        batch=Batch.from_data_list([Data(x=nodes[i],edge_index=edges,y=torch.zeros(1,device=t.device),edge_attr=attn[i]) for i in range(b)])
        graph,_=self.core.graph(batch)
        return auxiliary+self.core.decoder(graph)

class ResidualBlock(nn.Module):
    def __init__(self,n_in,n_hidden,n_out):
        super().__init__()
        self.main=nn.Sequential(nn.Linear(n_in,n_hidden),nn.Tanh(),nn.BatchNorm1d(n_hidden),nn.Linear(n_hidden,n_out))
        self.skip=nn.Linear(n_in,n_out)
    def forward(self,x):return self.main(x)+self.skip(x)

class ResDAE(nn.Module):
    def __init__(self,n_input,hidden=512,latent=128,dropout=.5):
        super().__init__()
        self.encoder=ResidualBlock(n_input,hidden,latent)
        self.decoder=ResidualBlock(latent,hidden,n_input)
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(latent,1))
    def forward(self,x,add_noise=False,reconstruct=True):
        noisy=x+torch.randn_like(x) if add_noise else x
        code=self.encoder(noisy)
        logits=self.classifier(code).squeeze(-1)
        return logits,self.decoder(code) if reconstruct else None
    def freeze_encoder(self):
        for module in [self.encoder,self.decoder]:
            module.eval()
            for p in module.parameters():p.requires_grad_(False)

def build(kind,cfg,device='cpu'):
    if kind=='fbnetgen':return VariableFBNetGen(cfg['embedding_size']).to(device)
    if kind=='brainnetgnn':return VariableBrainNetGNN(device).to(device)
    if kind=='res_dae':return ResDAE(cfg['n_input'],cfg['hidden'],cfg['latent'],cfg['dropout']).to(device)
    raise ValueError(kind)

def fb_objective(model,t,fc,y,lengths):
    import numpy as np
    lam=float(np.random.beta(1,1));perm=torch.randperm(len(y),device=t.device)
    # Mix only overlapping observed windows; padded tails never enter the GRU.
    mixed_lengths=torch.minimum(lengths,lengths[perm.cpu()])
    logits,matrix=model(lam*t+(1-lam)*t[perm],lam*fc+(1-lam)*fc[perm],mixed_lengths)
    ce=lam*nn.functional.cross_entropy(logits,y)+(1-lam)*nn.functional.cross_entropy(logits,y[perm])
    return 2*ce+mixup_cluster_loss(matrix,y,y[perm],lam)+1e-4*torch.norm(matrix,p=1)

def gnn_regularization(model):
    core=model.core
    return 1e-5*sum(p.abs().sum() for module in [core.encoder_decoder,core.graph,core.key_layer,core.value_layer,core.query_layer,core.multihead_attn]
                    for name,p in module.named_parameters() if 'bias' not in name)
