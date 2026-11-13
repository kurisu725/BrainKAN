from pathlib import Path
import csv,io
import numpy as np
from sklearn.metrics import precision_score,recall_score
from storage import read_json,write_json,digest,save_torch,atomic
from run_review import metric

ROOT=Path(__file__).resolve().parents[1]
BASE=dict(epochs=50,batch_size=96,hidden=[256,128,64,32,16],lr=.001,weight_decay=.0001,
          gamma=.8,optimizer='adamw',dropout=.3,grid_epochs=[],grid_sample=128,
          regularization=0.,shuffle=True,kind='kan_legacy',seed=1,grid_size=5,spline_order=3)
ARCHITECTURES={'original':[256,128,64,32,16],'medium':[128,64,32],'compact':[64,16]}
GRIDS=[(5,3),(10,3),(10,5)]
TRAINING_PRESETS=[dict(id='lr3e-4_g095',lr=.0003,gamma=.95),dict(id='lr3e-4_g100',lr=.0003,gamma=1.),
                  dict(id='lr1e-3_g095',lr=.001,gamma=.95),dict(id='lr1e-3_g100',lr=.001,gamma=1.)]
DEFAULT_ID='A_original_g5_s3'

def stage_a():
    return [dict(id=f'A_{name}_g{g}_s{s}',config={**BASE,'hidden':h,'grid_size':g,'spline_order':s})
            for name,h in ARCHITECTURES.items() for g,s in GRIDS]
def stage_b(winner,presets):
    return [dict(id='B_'+p['id'],config={**winner['config'],'lr':p['lr'],'gamma':p['gamma']},
                 architecture_source=winner['job']) for p in presets]
def parameter_count(n,cfg):
    sizes=[n*n]+cfg['hidden']+[2]
    return sum(a*b*(cfg['grid_size']+cfg['spline_order']+2) for a,b in zip(sizes[:-1],sizes[1:]))
def geometry(model,n,cfg):
    sizes=[n*n]+cfg['hidden']+[2];details=[]
    assert len(model.layers)==len(sizes)-1
    for layer,a,b in zip(model.layers,sizes[:-1],sizes[1:]):
        assert layer.grid_size==cfg['grid_size'] and layer.spline_order==cfg['spline_order']
        assert list(layer.spline_weight.shape)==[b,a,cfg['grid_size']+cfg['spline_order']]
        details.append(dict(input=a,output=b,spline_weight_shape=list(layer.spline_weight.shape)))
    count=sum(p.numel() for p in model.parameters());assert count==parameter_count(n,cfg)
    return dict(parameters=count,layers=details)
def metrics(y,p):
    m=metric(y,p);pred=p.argmax(1)
    return {**m,'precision_macro':float(precision_score(y,pred,average='macro',zero_division=0)),
            'recall_macro':float(recall_score(y,pred,average='macro',zero_division=0))}
def choice_key(row):
    m=row['development'];return (-m['auc'],m['loss'],row['parameter_count'],row['candidate'])
def write_predictions(path,indices,y,p):
    out=io.StringIO(newline='');w=csv.writer(out);w.writerow(['pool_index','label','prob_0','prob_1','pred'])
    w.writerows((int(i),int(t),float(a),float(b),int(np.argmax([a,b]))) for i,t,(a,b) in zip(indices,y,p))
    atomic(path,lambda f:f.write(out.getvalue().encode('utf8')))
def read_predictions(path):
    a=np.genfromtxt(path,delimiter=',',names=True)
    return a['pool_index'].astype(int),a['label'].astype(int),np.column_stack([a['prob_0'],a['prob_1']]).astype('float32')
def verify_files(files,base=None):
    for p,h in files.items():
        path=Path(p) if base is None else Path(base)/p
        if digest(path)!=h:raise ValueError('Changed protected file: '+str(path))
def leader_path(root,branch,partition):return Path(root)/'selected'/f'{branch}_p{partition}.json'
def commit_leader(root,row,payload=None,external=None):
    """Commit new pointer before retiring previous local weights; never alter source models."""
    root=Path(root);path=leader_path(root,row['branch'],row['partition'])
    old=read_json(path) if path.exists() else None
    if old is not None and choice_key(old['record'])<=choice_key(row):return old
    stem=f"{row['branch']}_p{row['partition']}"
    if external is not None:
        model_path=Path(external).resolve();sha=digest(model_path)
    else:
        assert payload is not None
        slot='b' if old and Path(old['model_path']).name==stem+'_a.pt' else 'a'
        model_path=(root/'selected'/f'{stem}_{slot}.pt').resolve();save_torch(model_path,payload);sha=digest(model_path)
    new=dict(record=row,model_path=str(model_path),model_sha256=sha)
    write_json(path,new)
    if old and old['model_path']!=str(model_path):
        retired=Path(old['model_path']).resolve()
        if retired.parent==(root/'selected').resolve() and retired.name in [stem+'_a.pt',stem+'_b.pt']:
            retired.unlink(missing_ok=True)
    return new
def retire_checkpoint(root,job):
    import torch
    p=Path(root)/'state/current.pt'
    if p.exists():
        s=torch.load(p,map_location='cpu',weights_only=False)
        if s['job']!=job:raise ValueError('Unexpected rolling checkpoint during retirement')
        del s;p.unlink()
