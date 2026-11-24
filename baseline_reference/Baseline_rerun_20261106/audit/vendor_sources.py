from pathlib import Path
import shutil, json, hashlib
R=Path(__file__).resolve().parents[1]
V=R/'code/vendor'; V.mkdir(exist_ok=True)
(V/'__init__.py').write_text('')
def cp(src,dest):
    src=R/'sources'/src; dest=V/dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
cp('BrainGB/src/models/gcn.py','braingb.py')
for p in (R/'sources/BrainNetworkTransformer/source/models/BNT').rglob('*.py'):
    cp(str(p.relative_to(R/'sources')),Path('BNT')/p.relative_to(R/'sources/BrainNetworkTransformer/source/models/BNT'))
cp('BrainNetworkTransformer/source/models/base.py','base.py')
for name in ('braingnn.py','braingraphconv.py','inits.py'):
    cp('BrainGNN/net/'+name,'li_braingnn/'+name)
(V/'li_braingnn/__init__.py').write_text('')
p=V/'li_braingnn/braingnn.py'
s=p.read_text().replace('from net.braingraphconv','from .braingraphconv').replace('self.pool1.weight','self.pool1.select.weight').replace('self.pool2.weight','self.pool2.select.weight')
p.write_text(s,encoding='utf-8')
p=V/'li_braingnn/braingraphconv.py'
s=p.read_text().replace('from net.brainmsgpassing import MyMessagePassing','from torch_geometric.nn import MessagePassing as MyMessagePassing').replace('from net.inits','from .inits')
p.write_text(s,encoding='utf-8')
p=V/'BNT/bnt.py';s=p.read_text().replace('nhead=4','nhead=2');p.write_text(s,encoding='utf-8')
p=V/'BNT/components/transformer_encoder.py';s=p.read_text().replace('key_padding_mask: Optional[Tensor])','key_padding_mask: Optional[Tensor], is_causal: bool = False)').replace('need_weights=True)','need_weights=True, is_causal=is_causal)');p.write_text(s,encoding='utf-8')
s=p.read_text().replace('super().__init__(d_model, nhead, dim_feedforward, dropout, activation,\n                         layer_norm_eps, batch_first, norm_first, device, dtype)', 'super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)')
p.write_text(s,encoding='utf-8')
patches={
 'BrainGNN':['PyG 2.5 MessagePassing in place of legacy custom propagation; mathematical mean aggregation retained','TopKPooling.select.weight API replacement; author double sigmoid retained','Input 90 FC features and 90 positional identities; graph abs archived FC without loops, explicitly an FC-input adaptation'],
 'BrainGB':['Unchanged official GCN module; node_concate + concat selected in historical manuscript; AAL90 input'],
 'BrainNetworkTransformer':['Attention heads 4 -> 2 because AAL90 feature width is not divisible by 4','Pooling 100/200 -> 45/90, preserving ABIDE one-half ROI ratio','PyTorch 2.2 is_causal support and keyword constructor to preserve default bias=True across library versions; model/loss otherwise retained']}
(R/'audit/adapter_changes.json').write_text(json.dumps(patches,indent=2),encoding='utf-8')
