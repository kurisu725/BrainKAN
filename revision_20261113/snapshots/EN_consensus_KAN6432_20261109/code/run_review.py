"""Train/develop -> freeze -> final evaluation. No test metrics in development."""
from pathlib import Path
import argparse,csv,gc,hashlib,json,os,random,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_auc_score,log_loss,balanced_accuracy_score,f1_score
import joblib
from data import load_data,features,validate_split,file_hash
from models import build_model,AuditedKAN

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'protocol_template.json'
BASE=dict(epochs=50,batch_size=96,hidden=[256,128,64,32,16],lr=.001,weight_decay=.0001,
          gamma=.8,optimizer='adamw',dropout=.3,grid_epochs=[],grid_sample=128,regularization=0.,shuffle=True)
PRESETS={
 'kan_fixed_replay':dict(kind='kan_legacy'),
 'kan_grid_only':dict(kind='kan_adaptive',grid_epochs=[1,5,10]),
 'kan_compact_regularized':dict(kind='kan_adaptive',hidden=[64,16],grid_epochs=[1,5,10],gamma=.95,regularization=1e-4),
 'bnc_legacy_replay':dict(kind='brainnetcnn',legacy_output=True),
 'bnc_raw_logits':dict(kind='brainnetcnn',legacy_output=False),
 'bnc_port_sgd':dict(kind='brainnetcnn',legacy_output=False,optimizer='sgd',lr=.01,weight_decay=.0005,
                     gamma=1.,batch_size=20,dropout=.5,momentum=.9,nesterov=True),
 'mlp_replay':dict(kind='mlp'),
 'gcn_fc':dict(kind='gcn',gcn_hidden=32,dropout=.5,lr=.001,gamma=1.,weight_decay=.0005),
 'logistic_fc':dict(kind='logistic',C_values=[.001,.01,.1,1.,10.]),
 'linear_svm_fc':dict(kind='linear_svm',C_values=[.001,.01,.1,1.,10.])}

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),'utf-8');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text('utf-8'))
def sources():return {p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def metric(y,prob):
    if prob.shape!=(len(y),2) or not np.isfinite(prob).all():raise ValueError('Invalid probabilities')
    pred=prob.argmax(1)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),auc=float(roc_auc_score(y,prob[:,1])),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),f1_macro=float(f1_score(y,pred,average='macro',zero_division=0)),
                loss=float(log_loss(y,prob,labels=[0,1])))

def input_features(x,kind):
    if kind in ('logistic','linear_svm'):
        # Upper triangle excludes the diagonal and duplicate undirected edges.
        i,j=np.triu_indices(x.shape[1],k=1);return x[:,i,j].copy()
    if kind=='gcn':return torch.from_numpy(np.ascontiguousarray(x))
    return features(x,np.arange(x.shape[1]))

@torch.no_grad()
def predict(model,tensor,device,batch_size=96):
    model.eval();prob=[]
    for start in range(0,len(tensor),batch_size):prob.append(model(tensor[start:start+batch_size].to(device)).softmax(1).cpu().numpy())
    return np.concatenate(prob)

def save_predictions(path,indices,y,p):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['pool_index','label','prob_0','prob_1','pred'])
        w.writerows((int(i),int(t),float(a),float(b),int(b>a)) for i,t,(a,b) in zip(indices,y,p))

def fit_classical(x_train,y_train,x_dev,y_dev,cfg):
    # StandardScaler sees the training cohort only. C uses development AUC.
    candidates=[];best=None;bestkey=None
    for c in cfg['C_values']:
        estimator=(LogisticRegression(C=c,max_iter=4000,solver='lbfgs',random_state=cfg['seed']) if cfg['kind']=='logistic'
                   else SVC(C=c,kernel='linear',probability=True,random_state=cfg['seed']))
        model=make_pipeline(StandardScaler(),estimator);model.fit(x_train,y_train)
        prob=model.predict_proba(x_dev);m=metric(y_dev,prob);key=(m['auc'],-m['loss'],-c)
        candidates.append({'C':c,'development':m})
        if bestkey is None or key>bestkey:bestkey=key;best=model
    return best,candidates

def make_plan(dest,protocol,models,seeds,epochs=None,data_path=None):
    if Path(dest).exists():raise FileExistsError(dest)
    prior=read(SOURCE);ds=prior['datasets']['abide'];idx=(prior['site']['indices'] if protocol=='site' else ds['indices'])
    if data_path is not None:
        ds={**ds,'path':str(Path(data_path).resolve())}
    if file_hash(ds['path'])!=ds['sha256']:raise ValueError('Expected the retained ABIDE matrix file; SHA256 mismatch')
    jobs=[]
    for name in models:
        for seed in seeds:
            cfg={**BASE,**PRESETS[name],'seed':seed}
            if epochs is not None:
                cfg['epochs']=epochs;cfg['grid_epochs']=[i for i in cfg['grid_epochs'] if i<=epochs]
            jobs.append(dict(id=f'{name}_s{seed}',variant=name,config=cfg))
    plan=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),purpose='implementation audit / separate new baseline protocol',
              protocol=protocol,data=ds['path'],data_sha256=ds['sha256'],indices=idx,jobs=jobs,source_sha256=sources(),
              checkpoint_selection='final prescribed epoch; no best-test or best-development epoch',
              hyperparameter_selection='only classical C uses development AUC, then log loss, then smaller C',
              final_test_used=False)
    write(dest,plan)

def verify_plan(path):
    p=read(path)
    if p['source_sha256']!=sources():raise ValueError('Code differs from the frozen plan; create a separately named new plan')
    if p['data_sha256']!=file_hash(p['data']):raise ValueError('Data checksum changed')
    return p

def fit_neural(train_x,train_y,dev_x,dev_y,cfg,folder,device):
    """No final-test arguments. Grid calibration takes training tensors only."""
    seed_all(cfg['seed']);model=build_model(cfg['kind'],train_x.shape[1],cfg).to(device)
    xx=input_features(train_x,cfg['kind']);vv=input_features(dev_x,cfg['kind'])
    params=model.parameters()
    opt=(torch.optim.SGD(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'],momentum=cfg['momentum'],nesterov=cfg['nesterov'])
         if cfg['optimizer']=='sgd' else torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay']))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=cfg['gamma'])
    # Independent generator prevents calibration sampling from changing minibatch order.
    cal_ids=torch.randperm(len(xx),generator=torch.Generator().manual_seed(cfg['seed']+1701))[:cfg['grid_sample']]
    loader=DataLoader(TensorDataset(xx,torch.from_numpy(train_y)),batch_size=cfg['batch_size'],shuffle=cfg['shuffle'],num_workers=0)
    history=[];events=[];resume=folder/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['config']!=cfg:raise ValueError('Resume configuration mismatch')
        model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];events=saved['grid_events']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    for epoch in range(len(history)+1,cfg['epochs']+1):
        model.train();tic=time.monotonic()
        if isinstance(model,AuditedKAN) and epoch in cfg['grid_epochs']:
            events.append({'epoch':epoch,'training_pool_positions':cal_ids.tolist(),
                           'layers':model.adapt_grid(xx[cal_ids].to(device),opt)})
        total=regtotal=0.
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
            ce=nn.functional.cross_entropy(model(xb),yb)
            reg=model.regularization_loss()*cfg['regularization'] if cfg['regularization'] else ce.new_zeros(())
            loss=ce+reg
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            opt.step();total+=float(ce)*len(yb);regtotal+=float(reg)*len(yb)
        dev=metric(dev_y,predict(model,vv,device))
        history.append(dict(epoch=epoch,training_ce=total/len(xx),regularization=regtotal/len(xx),development=dev,
                            lr=opt.param_groups[0]['lr'],seconds=time.monotonic()-tic))
        scheduler.step();write(folder/'history.json',history);write(folder/'grid_events.json',events)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save(dict(config=cfg,state=state,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,
                        grid_events=events,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        numpy_rng=np.random.get_state(),python_rng=random.getstate()),resume.with_suffix('.tmp'))
        resume.with_suffix('.tmp').replace(resume)
        print(folder.name,'epoch',epoch,'train_ce',round(total/len(xx),5),'dev_auc',round(dev['auc'],5),flush=True)
    torch.save(dict(config=cfg,state={k:v.detach().cpu() for k,v in model.state_dict().items()},n_roi=train_x.shape[1],epoch=cfg['epochs']),folder/'model.pt')
    return model,metric(train_y,predict(model,xx,device)),metric(dev_y,predict(model,vv,device))

def develop(plan_path,device):
    plan=verify_plan(plan_path);out=Path(plan_path).with_suffix('');out.mkdir(exist_ok=True)
    x,y,_,_=load_data(plan['data'],2);idx=validate_split(plan['indices'],len(x),y)
    tr,dv=idx['train'],idx['valid'];rows=[]
    for job in plan['jobs']:
        folder=out/job['id'];folder.mkdir(exist_ok=True);cfg=job['config'];marker=folder/'complete.json'
        if marker.exists():
            row=read(marker)
            if row['config']!=cfg:raise ValueError('Completed configuration mismatch')
            for name,digest in row['files'].items():
                if file_hash(folder/name)!=digest:raise ValueError('Completed file changed: '+str(folder/name))
            rows.append(row);continue
        failure=folder/'failure.json'
        if failure.exists():
            rows.append(read(failure));continue
        write(out/'status.json',dict(state='running',job=job['id'],pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S')))
        print('JOB',job['id'],flush=True)
        try:
            if cfg['kind'] in ('logistic','linear_svm'):
                tx=input_features(x[tr],cfg['kind']);vx=input_features(x[dv],cfg['kind'])
                model,candidates=fit_classical(tx,y[tr],vx,y[dv],cfg);joblib.dump(model,folder/'model.joblib')
                write(folder/'C_selection.json',candidates)
                tp=model.predict_proba(tx);vp=model.predict_proba(vx);train_m=metric(y[tr],tp);dev_m=metric(y[dv],vp)
            else:
                model,train_m,dev_m=fit_neural(x[tr],y[tr],x[dv],y[dv],cfg,folder,device)
                tp=predict(model,input_features(x[tr],cfg['kind']),device);vp=predict(model,input_features(x[dv],cfg['kind']),device)
            save_predictions(folder/'train_predictions.csv',tr,y[tr],tp);save_predictions(folder/'development_predictions.csv',dv,y[dv],vp)
            write(folder/'config.json',cfg)
            row=dict(job=job['id'],variant=job['variant'],config=cfg,prediction_dtype=str(vp.dtype),training=train_m,development=dev_m,final_test_used=False,
                     files={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.name not in ['resume.pt','complete.json']})
            write(marker,row);rows.append(row);write(out/'development_summary.json',rows)
            del model;gc.collect();torch.cuda.empty_cache()
        except FloatingPointError:
            failed=dict(job=job['id'],variant=job['variant'],config=cfg,status='numerical_failure',reason=traceback.format_exc(),final_test_used=False)
            write(failure,failed);rows.append(failed);write(out/'development_summary.json',rows)
            print('FAILED CONFIGURATION',job['id'],failed['reason'],flush=True)
            if 'model' in locals():del model
            gc.collect();torch.cuda.empty_cache()
    write(out/'development_summary.json',rows)
    write(out/'status.json',dict(state=('development_complete_with_failures' if any(r.get('status')=='numerical_failure' for r in rows) else 'development_complete'),jobs=len(rows),final_test_used=False))

def freeze(plan_path):
    p=verify_plan(plan_path);out=Path(plan_path).with_suffix('');target=out/'frozen.json'
    if target.exists():raise FileExistsError('Already frozen; existing manifest preserved')
    files={}
    for j in p['jobs']:
        folder=out/j['id']
        if (folder/'failure.json').exists():raise RuntimeError('Numerical failure must be reported; create a separate corrected plan before final evaluation')
        row=read(folder/'complete.json')
        if row['config']!=j['config']:raise ValueError('Configuration mismatch')
        for name,digest in row['files'].items():
            if file_hash(folder/name)!=digest:raise ValueError('Changed artifact')
            files[str((folder/name).relative_to(out))]=digest
    write(target,dict(plan_sha256=file_hash(plan_path),files=files,final_test_used=False))

def evaluate(plan_path,device):
    p=verify_plan(plan_path);out=Path(plan_path).with_suffix('');frozen=read(out/'frozen.json')
    if file_hash(plan_path)!=frozen['plan_sha256']:raise ValueError('Plan changed after freezing')
    for name,digest in frozen['files'].items():
        if file_hash(out/name)!=digest:raise ValueError('Frozen artifact changed')
    target=out/'final_test'
    if target.exists():raise FileExistsError('Final evaluation already exists; never overwrite')
    x,y,_,_=load_data(p['data'],2);idx=validate_split(p['indices'],len(x),y)['test'];target.mkdir();rows=[]
    for j in p['jobs']:
        folder=out/j['id'];cfg=j['config'];xx=input_features(x[idx],cfg['kind'])
        if cfg['kind'] in ('logistic','linear_svm'):prob=joblib.load(folder/'model.joblib').predict_proba(xx)
        else:
            ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            model=build_model(cfg['kind'],x.shape[1],cfg).to(device);model.load_state_dict(ck['state']);prob=predict(model,xx,device)
            del model;gc.collect();torch.cuda.empty_cache()
        save_predictions(target/(j['id']+'.csv'),idx,y[idx],prob);rows.append(dict(job=j['id'],metrics=metric(y[idx],prob)))
    write(target/'summary.json',rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['plan','develop','freeze','evaluate'])
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--protocol',choices=['site','fixed'],default='site')
    ap.add_argument('--models',nargs='+',choices=list(PRESETS),default=['kan_fixed_replay','kan_grid_only','bnc_legacy_replay','bnc_raw_logits','mlp_replay','gcn_fc','logistic_fc','linear_svm_fc']);ap.add_argument('--seeds',type=int,nargs='+',default=list(range(5)))
    ap.add_argument('--data',type=Path);ap.add_argument('--epochs',type=int);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args()
    torch.set_num_threads(4)
    if a.command=='plan':make_plan(a.plan,a.protocol,a.models,a.seeds,a.epochs,a.data)
    elif a.command=='develop':develop(a.plan,a.device)
    elif a.command=='freeze':freeze(a.plan)
    else:evaluate(a.plan,a.device)
