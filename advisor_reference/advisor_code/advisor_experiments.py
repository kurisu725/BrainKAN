"""Frozen, resumable experiments for the advisor's 26 October review.

plan -> run (all development artifacts) -> evaluate (frozen models only).
Legacy ROI replay is a diagnostic, not a newly independent ROI selection.
"""
from pathlib import Path
import argparse, ast, csv, gc, hashlib, itertools, json, os, random, sys, time, traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.metrics import log_loss
import joblib
from brainkan.data import load_data, features, validate_split, file_hash
from brainkan.training import build_model, seed_everything, classification_metrics
from brainkan.roi import select_rois, roi_scores, transform_scores

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=Path(r'E:\Brain\BrainKAN\BrainKAN-complete')
BASE=dict(epochs=50,batch_size=96,hidden=[256,128,64,32,16],grid_size=5,spline_order=3,
          lr=.001,weight_decay=.0001,lr_gamma=.8,shuffle=True,dropout=.3)

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),'utf-8');tmp.replace(path)

def read_json(path):return json.loads(Path(path).read_text('utf-8'))

def sources():
    return {str(p.relative_to(ROOT/'code')):file_hash(p) for p in (ROOT/'code').rglob('*.py') if 'tests' not in p.parts}

def status(**kw):
    write_json(ROOT/'experiments'/'status.json',dict(pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S'),**kw))

class E2E(nn.Module):
    def __init__(self,a,b,n):
        super().__init__();self.n=n;self.row=nn.Conv2d(a,b,(1,n));self.col=nn.Conv2d(a,b,(n,1))
    def forward(self,x):return self.row(x).expand(-1,-1,-1,self.n)+self.col(x).expand(-1,-1,self.n,-1)

class BrainNetCNN(nn.Module):
    """E2E(32,64), E2N(1), N2G(256), FC(128,30,2); no AGCA extension.

    Mirrors the retained Convolutional_Architecture BrainNetCNN topology.
    Dropout is .3 (the manuscript's development choice) and respects eval().
    The retained leaky-ReLU slope .33 and output activation are explicit here.
    """
    def __init__(self,n,p=.3):
        super().__init__();self.n=n;self.p=p
        self.e1=E2E(1,32,n);self.e2=E2E(32,64,n)
        self.e2n=nn.Conv2d(64,1,(1,n));self.n2g=nn.Conv2d(1,256,(n,1))
        self.f1=nn.Linear(256,128);self.f2=nn.Linear(128,30);self.f3=nn.Linear(30,2)
    def forward(self,x):
        x=x.reshape(-1,1,self.n,self.n)
        for layer in (self.e1,self.e2,self.e2n):x=F.leaky_relu(layer(x),.33)
        x=F.dropout(F.leaky_relu(self.n2g(x),.33),self.p,self.training).flatten(1)
        for layer in (self.f1,self.f2):x=F.dropout(F.leaky_relu(layer(x),.33),self.p,self.training)
        return F.leaky_relu(self.f3(x),.33)

def model_for(kind,n,cfg):
    return BrainNetCNN(n,cfg['dropout']) if kind=='brainnetcnn' else build_model(n,cfg,kind)

@torch.no_grad()
def predict(model,x,rois,batch=96):
    model.eval();out=[];device=next(model.parameters()).device
    for i in range(0,len(x),batch):
        out.append(model(features(x[i:i+batch],rois).to(device)).softmax(1).cpu().numpy())
    return np.concatenate(out)

def fit_neural(x_train,y_train,x_dev,y_dev,rois,cfg,kind,folder):
    """Only training and development data are accepted. Always return epoch 50."""
    seed_everything(cfg['seed']);model=model_for(kind,len(rois),cfg).to(cfg['device'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=cfg['lr_gamma'])
    loader=DataLoader(TensorDataset(features(x_train,rois),torch.from_numpy(y_train)),
                      batch_size=cfg['batch_size'],shuffle=cfg['shuffle'],num_workers=0)
    history=[];start=0;partial=folder/'resume.pt'
    if partial.exists():
        saved=torch.load(partial,map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['rois']==list(map(int,rois))
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        history=saved['history'];start=len(history)
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    criterion=nn.CrossEntropyLoss()
    for epoch in range(start+1,cfg['epochs']+1):
        model.train();total=0.;tic=time.monotonic()
        for xx,yy in loader:
            xx,yy=xx.to(cfg['device']),yy.to(cfg['device']);opt.zero_grad(set_to_none=True)
            loss=criterion(model(xx),yy)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();opt.step();total+=loss.item()*len(yy)
        # Development curves diagnose convergence; no checkpoint selection.
        dev=classification_metrics(y_dev,predict(model,x_dev,rois))
        history.append(dict(epoch=epoch,train_loss=total/len(y_train),lr=opt.param_groups[0]['lr'],
                            development=dev,seconds=time.monotonic()-tic))
        sched.step()
        write_json(folder/'history.json',history)
        if epoch%10==0 or epoch==cfg['epochs']:
            state=dict(config=cfg,rois=list(map(int,rois)),model=model.state_dict(),optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),history=history,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
            torch.save(state,partial.with_suffix('.tmp'));partial.with_suffix('.tmp').replace(partial)
        if epoch%10==0:print(folder.name,epoch,round(total/len(y_train),5),flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,
                    rois=list(map(int,rois)),kind=kind,epoch=cfg['epochs'],checkpoint_selection='final'),folder/'model.pt')
    return model,history

def save_predictions(path,indices,y,prob):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['pool_index','label','prob_0','prob_1','predicted'])
        writer.writerows(zip(map(int,indices),map(int,y),prob[:,0],prob[:,1],prob.argmax(1)))

def load_neural(folder):
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    m=model_for(ck['kind'],len(ck['rois']),ck['config']);m.load_state_dict(ck['state_dict']);m.to(ck['config']['device']).eval()
    return m,ck

def fast_permutation(model,x_dev,y_dev,seed,repeats=10,chunk=4,edge_limit=None):
    """Same edge-major RNG sequence and log-loss as the sequential implementation.

    Only the inference batching changes. Normalization remains on CPU per subject.
    """
    n=x_dev.shape[1];rng=np.random.default_rng(seed);rois=np.arange(n)
    baseline=log_loss(y_dev,predict(model,x_dev,rois),labels=[0,1]);flat=x_dev.reshape(len(x_dev),-1)
    limit=n*n if edge_limit is None else edge_limit
    diffs=np.zeros((repeats,n*n),dtype=float);pending=[];keys=[]
    def flush():
        if not pending:return
        batch=np.concatenate(pending).reshape(-1,n,n)
        probs=predict(model,batch,rois,batch=len(batch)).reshape(len(pending),len(x_dev),2)
        for (r,e),prob in zip(keys,probs):diffs[r,e]=baseline-log_loss(y_dev,prob,labels=[0,1])
        pending.clear();keys.clear()
    for e in range(limit):
        for r in range(repeats):
            xx=flat.copy();xx[:,e]=flat[rng.permutation(len(flat)),e]
            pending.append(xx);keys.append((r,e))
            if len(pending)==chunk:flush()
        if (e+1)%90==0:print('site importance',e+1,'/',limit,flush=True)
    flush();edges=diffs.mean(0).reshape(n,n)
    repeated=diffs.reshape(repeats,n,n).sum(1)+diffs.reshape(repeats,n,n).sum(2)
    return roi_scores(edges),repeated,edges,dict(baseline_loss=float(baseline),repeats=repeats,seed=seed,
           order='edge-major then repeat',inference_chunk=chunk,normalization='per-subject CPU LayerNorm')

def make_plan():
    out=ROOT/'experiments';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Plan already exists')
    plan=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),seeds=list(range(5)),base=BASE,datasets={},jobs=[],sources=sources(),
              checkpoint_selection='final epoch 50',selection_uses='development only',final_test_after='all development jobs frozen',
              purpose='performance-gap diagnostics, unified baselines, ABIDE site-independent test')
    for ds,col,fn in [('abide',2,'ALLASD1_aal.mat'),('adhd',3,'ADHD1_aal.mat')]:
        p=PREVIOUS/'data'/fn;x,y,idx,ph=load_data(p,col)
        uidcol=0 if ds=='abide' else 1
        for a,b in itertools.combinations(idx,2):
            assert not set(ph[idx[a],uidcol])&set(ph[idx[b],uidcol])
        plan['datasets'][ds]=dict(path=str(p),sha256=file_hash(p),label_column=col,indices={k:v.tolist() for k,v in idx.items()},
                                  subject_id_column=uidcol)
        if ds=='abide':
            # Choose site groups by cohort size only, before running a model.
            sites,counts=np.unique(ph[:,1].astype(int),return_counts=True)
            ordered=sorted(zip(sites,counts),key=lambda z:(-z[1],z[0]));testsite,devsite=int(ordered[0][0]),int(ordered[1][0])
            siteidx=dict(train=np.flatnonzero(~np.isin(ph[:,1],[testsite,devsite])),
                         valid=np.flatnonzero(ph[:,1]==devsite),test=np.flatnonzero(ph[:,1]==testsite))
            validate_split(siteidx,len(y),y)
            plan['site']=dict(dataset=ds,site_column=1,test_site=testsite,development_site=devsite,
                              rule='largest site group for test; second largest for development; all remaining sites for training',
                              indices={k:v.tolist() for k,v in siteidx.items()})
    archived=read_json(PREVIOUS/'docs'/'archived_abide_roi_analysis.json')
    scores=np.array(archived['roi_scores'])
    plan['archived_rois']={mode:select_rois(scores,mode).tolist() for mode in ('direct','minmax','zscore')}
    plan['archived_roi_source']=dict(path=str(PREVIOUS/'docs'/'archived_abide_roi_analysis.json'),sha256=file_hash(PREVIOUS/'docs'/'archived_abide_roi_analysis.json'),scope='historical diagnostic; unknown generating cohort provenance')
    # Unified baseline jobs: these are new fits; existing automatic KAN arms are verified separately.
    for ds in plan['datasets']:
        for seed in plan['seeds']:
            for kind in ('kan','mlp','brainnetcnn','svm'):
                plan['jobs'].append(dict(id=f'unified_{ds}_{kind}_s{seed}',phase='unified',dataset=ds,seed=seed,kind=kind,rois='full',changes={}))
    for seed in plan['seeds']:
        # Fixed legacy ROI sets isolate training-order effects from new ROI estimation.
        for mode in ('direct','minmax','zscore'):
            for shuffle in (False,True):
                plan['jobs'].append(dict(id=f'legacy_{mode}_shuffle{int(shuffle)}_s{seed}',phase='legacy',dataset='abide',seed=seed,
                                         kind='kan',rois='archived_'+mode,changes={'shuffle':shuffle}))
        plan['jobs'].append(dict(id=f'legacy_full_shuffle0_s{seed}',phase='legacy',dataset='abide',seed=seed,kind='kan',rois='full',changes={'shuffle':False}))
        for gamma in (.95,1.):
            plan['jobs'].append(dict(id=f'convergence_gamma{gamma:g}_s{seed}',phase='convergence',dataset='abide',seed=seed,kind='kan',rois='full',changes={'lr_gamma':gamma}))
        plan['jobs'].append(dict(id=f'legacy_full_width512_s{seed}',phase='legacy',dataset='abide',seed=seed,kind='kan',rois='full',changes={'hidden':[512,256,128,64,32,16],'shuffle':False}))
        for kind in ('kan','mlp','brainnetcnn','svm'):
            plan['jobs'].append(dict(id=f'site_{kind}_s{seed}',phase='site',dataset='abide',seed=seed,kind=kind,rois='full',changes={}))
        for mode in ('direct','minmax','zscore'):
            plan['jobs'].append(dict(id=f'site_entry_{mode}_s{seed}',phase='site',dataset='abide',seed=seed,kind='kan',rois='site_'+mode,changes={}))
    # Finish cheap high-priority diagnostics before the site explanation stage.
    order={'legacy':0,'unified':1,'convergence':2,'site':3};plan['jobs'].sort(key=lambda j:(order[j['phase']],j['seed'],j['id']))
    # Site full KAN must precede its ROI-selection jobs.
    site=[j for j in plan['jobs'] if j['phase']=='site']
    plan['jobs']=[j for j in plan['jobs'] if j['phase']!='site']+sorted(site,key=lambda j:(j['seed'],j['rois']!='full',j['id']))
    write_json(out/'plan.json',plan);print('Frozen plan:',len(plan['jobs']),'model fits',flush=True)
    return plan

def verify_plan():
    p=ROOT/'experiments'/'plan.json';plan=read_json(p)
    if plan['sources']!=sources():raise ValueError('Execution code changed after plan creation')
    for ds,v in plan['datasets'].items():
        if file_hash(v['path'])!=v['sha256']:raise ValueError('Dataset checksum changed: '+ds)
    return plan

def run_job(job,plan,cache):
    out=ROOT/'experiments';folder=out/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'complete.json'
    if marker.exists():
        saved=read_json(marker)
        for name,digest in saved['files'].items():
            if file_hash(folder/name)!=digest:raise ValueError('Completed model changed: '+job['id'])
        return
    x,y=cache[job['dataset']]
    idx={k:np.array(v,dtype=int) for k,v in (plan['site']['indices'] if job['phase']=='site' else plan['datasets'][job['dataset']]['indices']).items()}
    tr,va=idx['train'],idx['valid'];rois=np.arange(x.shape[1]);cfg={**plan['base'],**job['changes'],'seed':job['seed'],'device':'cuda'}
    if job['rois'].startswith('archived_'):rois=np.array(plan['archived_rois'][job['rois'][9:]])
    if job['rois'].startswith('site_'):
        explain=out/'explanations'/f'seed_{job["seed"]}';explain.mkdir(parents=True,exist_ok=True)
        if not (explain/'complete.json').exists():
            selector,ck=load_neural(out/'runs'/f'site_kan_s{job["seed"]}')
            scores,repeated,edges,meta=fast_permutation(selector,x[va],y[va],job['seed'])
            meta.update(pool_indices=va.tolist(),selector_sha256=file_hash(out/'runs'/f'site_kan_s{job["seed"]}'/'model.pt'))
            np.save(explain/'roi_scores.npy',scores);np.save(explain/'repeat_roi_scores.npy',repeated);np.save(explain/'edge_scores.npy',edges)
            write_json(explain/'explanation.json',meta)
            write_json(explain/'complete.json',{'files':{p.name:file_hash(p) for p in explain.iterdir() if p.is_file()}})
            del selector;gc.collect();torch.cuda.empty_cache()
        integrity=read_json(explain/'complete.json')
        for name,digest in integrity['files'].items():
            if file_hash(explain/name)!=digest:raise ValueError('Site explanation changed')
        scores=np.load(explain/'roi_scores.npy');mode=job['rois'][5:]
        try:rois=select_rois(scores,mode)
        except ValueError as exc:
            write_json(folder/'failure.json',{'reason':str(exc),'job':job});return
        write_json(folder/'selection.json',dict(mode=mode,threshold=.5 if mode=='minmax' else 0.,comparison='<',
                    raw_scores=scores.tolist(),transformed_scores=transform_scores(scores,mode).tolist(),indices=rois.tolist(),pool_indices=va.tolist()))
    write_json(folder/'config.json',dict(job=job,config=cfg,rois=rois.tolist(),indices={k:v.tolist() for k,v in idx.items()}))
    if job['kind']=='svm':
        seed_everything(job['seed']);ft=features(x[tr],rois).numpy();fv=features(x[va],rois).numpy()
        candidates=[];best=None;bestkey=None
        for c in (.1,1.,10.):
            model=SVC(C=c,kernel='rbf',gamma='scale',probability=True,random_state=job['seed'])
            model.fit(ft,y[tr]);prob=model.predict_proba(fv);metrics=classification_metrics(y[va],prob)
            candidates.append(dict(C=c,development=metrics));key=(metrics['auc'],-metrics['loss'],-c)
            if bestkey is None or key>bestkey:best,bestkey=model,key
        model=best;joblib.dump(model,folder/'model.joblib');prob=model.predict_proba(fv)
        write_json(folder/'svm_selection.json',dict(candidates=candidates,selected_C=model.C,selection='development AUC, then loss, then smaller C',calibration='training-only libsvm internal probability calibration'))
    else:
        model,history=fit_neural(x[tr],y[tr],x[va],y[va],rois,cfg,job['kind'],folder)
        prob=predict(model,x[va],rois)
    save_predictions(folder/'development_predictions.csv',va,y[va],prob)
    write_json(folder/'development_metrics.json',classification_metrics(y[va],prob))
    # Resume state is for interrupted fits; completed model/data provenance is sealed.
    files={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.name!='resume.pt'}
    write_json(marker,dict(job=job['id'],epoch=50 if job['kind']!='svm' else None,n_roi=len(rois),files=files))
    del model;gc.collect();torch.cuda.empty_cache()

def run(phase='all'):
    plan=verify_plan();torch.set_num_threads(4);cache={}
    for ds,v in plan['datasets'].items():
        x,y,_,_=load_data(v['path'],v['label_column']);cache[ds]=(x,y)
    selected=[j for j in plan['jobs'] if phase=='all' or j['phase']==phase]
    for i,j in enumerate(selected):
        status(state='running',phase=j['phase'],job=j['id'],position=i+1,total=len(selected));print('JOB',i+1,'/',len(selected),j['id'],flush=True)
        run_job(j,plan,cache)
    roots=[ROOT/'experiments'/'runs'/j['id'] for j in selected]
    if phase in ('all','site'):roots.append(ROOT/'experiments'/'explanations')
    frozen={str(p.relative_to(ROOT/'experiments')):file_hash(p) for base in roots for p in base.rglob('*') if p.is_file() and p.name!='resume.pt'}
    write_json(ROOT/'experiments'/f'frozen_{phase}.json',dict(plan_sha256=file_hash(ROOT/'experiments'/'plan.json'),files=frozen,job_ids=[j['id'] for j in selected],final_test_used=False))
    status(state='development_complete',phase=phase,total=len(selected))

def evaluate(phase='all'):
    plan=verify_plan();out=ROOT/'experiments';frozen=read_json(out/f'frozen_{phase}.json')
    assert frozen['plan_sha256']==file_hash(out/'plan.json')
    for name,digest in frozen['files'].items():
        if file_hash(out/name)!=digest:raise ValueError('Frozen artifact changed: '+name)
    target=out/'final_test';target.mkdir(exist_ok=True);cache={};rows=[]
    for ds,v in plan['datasets'].items():
        x,y,_,_=load_data(v['path'],v['label_column']);cache[ds]=(x,y)
    for j in plan['jobs']:
        if j['id'] not in frozen['job_ids']:continue
        folder=out/'runs'/j['id']
        if (folder/'failure.json').exists():rows.append(dict(job=j,status='failed_selection'));continue
        x,y=cache[j['dataset']];saved=read_json(folder/'config.json');rois=np.array(saved['rois'])
        cohorts=['test','valid'] if j['phase']=='legacy' else ['test']
        for cohort in cohorts:
            a=np.array(saved['indices'][cohort]);dest=target/f'{j["id"]}_{cohort}.csv'
            if dest.exists():
                arr=np.genfromtxt(dest,delimiter=',',names=True);assert np.array_equal(arr['pool_index'].astype(int),a) and np.array_equal(arr['label'].astype(int),y[a])
                prob=np.column_stack([arr['prob_0'],arr['prob_1']])
            else:
                if j['kind']=='svm':model=joblib.load(folder/'model.joblib');prob=model.predict_proba(features(x[a],rois).numpy())
                else:model,ck=load_neural(folder);prob=predict(model,x[a],rois)
                save_predictions(dest,a,y[a],prob);del model;gc.collect();torch.cuda.empty_cache()
            rows.append(dict(job=j,cohort=cohort,n_roi=len(rois),metrics=classification_metrics(y[a],prob),prediction_sha256=file_hash(dest)))
    write_json(target/f'{phase}_runs.json',rows);status(state='complete',phase=phase,evaluated_records=len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','run','evaluate','all']);p.add_argument('--phase',default='all');a=p.parse_args()
    try:
        if a.command=='plan':make_plan()
        if a.command=='all' and a.phase=='all':
            for phase in ('legacy','unified','convergence','site'):
                run(phase);evaluate(phase)
            status(state='complete',phase='all',completed_jobs=len(verify_plan()['jobs']))
        elif a.command=='all':run(a.phase);evaluate(a.phase)
        elif a.command=='run':run(a.phase)
        elif a.command=='evaluate':evaluate(a.phase)
    except Exception:
        status(state='failed',error=traceback.format_exc());raise
