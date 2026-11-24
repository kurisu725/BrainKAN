from pathlib import Path
from datetime import datetime
import os,random,time,math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import accuracy_score,roc_auc_score,precision_recall_fscore_support,log_loss
from models import build,forward,train_loss
from storage import read_json,write_json,digest,save_torch,save_npz
ROOT=Path(__file__).resolve().parents[1]
class SafePause(Exception):pass
def now():return datetime.now().astimezone().isoformat()
def status(state,**extra):write_json(ROOT/'status.json',dict(state=state,updated=now(),pid=os.getpid(),**extra))
def pause_check():
    if (ROOT/'PAUSE').exists():raise SafePause()
def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def rng():return dict(py=random.getstate(),np=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r['py']);np.random.set_state(r['np']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def metrics(y,p):
    pred=p.argmax(1);pr,re,f1,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    return dict(acc=float(accuracy_score(y,pred)),auc=float(roc_auc_score(y,p[:,1])),precision=float(pr),recall=float(re),f1=float(f1),loss=float(log_loss(y,p,labels=[0,1])))
def verify_frozen():
    frozen=read_json(ROOT/'frozen_manifest.json')
    for path,sha in frozen.items():
        if digest(ROOT/path)!=sha:raise RuntimeError('Frozen input changed: '+path)
    p=read_json(ROOT/'plan.json')
    if digest(p['dataset']['path'])!=p['dataset']['sha256']:raise RuntimeError('Dataset checksum mismatch')
    return p
def data(plan):
    m=loadmat(plan['dataset']['path']);x=np.concatenate([m['net_'+s] for s in ('train','valid','test')]).astype('float32')
    ph=np.concatenate([m['phenotype_'+s] for s in ('train','valid','test')]);y=ph[:,2].astype('int64');ids=ph[:,0].astype('int64')
    assert np.isfinite(x).all() and len(set(ids))==882 and set(y)=={0,1}
    return torch.from_numpy(x),torch.from_numpy(y),ids
@torch.no_grad()
def predict(name,model,x,idx,batch=16):
    model.eval();ps=[]
    for start in range(0,len(idx),batch):
        out,_=forward(name,model,x[idx[start:start+batch]].cuda());ps.append(out.softmax(1).cpu().numpy())
    return np.concatenate(ps)
def scheduler_for(name,opt,cfg,n_steps):
    if name=='braingnn':return torch.optim.lr_scheduler.StepLR(opt,20,gamma=.5)
    if name=='brainnettf':return torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs']*n_steps,eta_min=1e-5)
    return None
def train_job(name,part,cfg,plan,x,y,ids):
    job=f'{name}_partition_{part["id"]}';d=ROOT/'runs'/job;d.mkdir(exist_ok=True)
    if (d/'record.json').exists():
        record=read_json(d/'record.json')
        assert digest(d/'model.pt')==record['model_sha256']
        current=ROOT/'state/current.pt'
        if current.exists() and torch.load(current,map_location='cpu')['job']==job:current.unlink()
        return
    pause_check();seed(plan['training_seed']);model=build(name).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    tr=np.array(part['indices']['train']);dv=np.array(part['indices']['valid']);bs=cfg['batch_size']
    n_steps=len(tr)//bs if cfg['drop_last'] else math.ceil(len(tr)/bs)
    sched=scheduler_for(name,opt,cfg,n_steps);checkpoint=ROOT/'state/current.pt'
    epoch=0;history=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu')
        assert saved['job']==job and saved['plan_sha256']==digest(ROOT/'plan.json')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        if sched:sched.load_state_dict(saved['scheduler'])
        epoch=saved['epoch'];history=saved['history'];restore_rng(saved['rng'])
    start_epoch=epoch
    while epoch<cfg['epochs']:
        pause_check();t=time.monotonic();model.train();order=torch.randperm(len(tr)).numpy();total=0.;seen=0
        status('training',job=job,epoch_completed=epoch,epoch_target=cfg['epochs'])
        for offset in range(0,len(tr),bs):
            ix=tr[order[offset:offset+bs]]
            if cfg['drop_last'] and len(ix)<bs:continue
            xx=x[ix].cuda();yy=y[ix].cuda();opt.zero_grad(set_to_none=True)
            loss=train_loss(name,model,xx,yy)
            if not torch.isfinite(loss):raise FloatingPointError(f'{job}: nonfinite loss at epoch {epoch+1}')
            loss.backward();opt.step()
            if name=='brainnettf':sched.step()
            total+=float(loss.detach())*(1 if name=='brainnettf' else len(ix));seen+=len(ix)
        if name=='braingnn':sched.step()
        epoch+=1;pr=predict(name,model,x,dv);mt=metrics(y[dv].numpy(),pr)
        history.append(dict(epoch=epoch,train_loss=total/seen,development=mt,lr=opt.param_groups[0]['lr'],seconds=time.monotonic()-t))
        saved=dict(job=job,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict() if sched else None,epoch=epoch,history=history,rng=rng(),plan_sha256=digest(ROOT/'plan.json'))
        save_torch(checkpoint,saved);write_json(d/'history.json',history)
        status('training',job=job,epoch_completed=epoch,epoch_target=cfg['epochs'],seconds_last_epoch=history[-1]['seconds'])
        print(f'{now()} {job} epoch {epoch}/{cfg["epochs"]} train_loss={total/seen:.5f} dev_acc={mt["acc"]:.4f} seconds={history[-1]["seconds"]:.1f}',flush=True)
        pause_check()
    pr=predict(name,model,x,dv);mt=metrics(y[dv].numpy(),pr)
    save_torch(d/'model.pt',dict(model=model.state_dict(),job=job,epoch=epoch,plan_sha256=digest(ROOT/'plan.json')))
    save_npz(d/'development_predictions.npz',pool_indices=dv,subject_ids=ids[dv],labels=y[dv].numpy(),probabilities=pr)
    write_json(d/'record.json',dict(job=job,method=name,partition=part['id'],n_roi=90,epochs=epoch,development=mt,checkpoint_selection='last epoch',model_sha256=digest(d/'model.pt'),development_predictions_sha256=digest(d/'development_predictions.npz'),plan_sha256=digest(ROOT/'plan.json'),finished=now()))
    checkpoint.unlink();del model,opt;torch.cuda.empty_cache()
def run():
    torch.set_num_threads(2);plan=verify_frozen();x,y,ids=data(plan)
    # Development phase for all jobs is completed before any final test is read.
    for part in plan['splits']:
        for name,cfg in plan['models'].items():train_job(name,part,cfg,plan,x,y,ids)
    seal=ROOT/'development_frozen.json'
    if not seal.exists():
        write_json(seal,dict(created=now(),plan_sha256=digest(ROOT/'plan.json'),artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted((ROOT/'runs').glob('*/*')) if p.name in ('model.pt','record.json','development_predictions.npz')}))
    frozen=read_json(seal)
    for path,sha in frozen['artifacts'].items():assert digest(ROOT/path)==sha,path
    for part in plan['splits']:
        for name in plan['models']:
            pause_check();job=f'{name}_partition_{part["id"]}';d=ROOT/'runs'/job
            if (d/'final.json').exists():continue
            status('final_evaluation',job=job);model=build(name).cuda();model.load_state_dict(torch.load(d/'model.pt',map_location='cpu')['model'])
            ix=np.array(part['indices']['test']);pr=predict(name,model,x,ix)
            save_npz(d/'predictions.npz',pool_indices=ix,subject_ids=ids[ix],labels=y[ix].numpy(),probabilities=pr)
            write_json(d/'final.json',dict(job=job,method=name,partition=part['id'],n_roi=90,metrics=metrics(y[ix].numpy(),pr),predictions_sha256=digest(d/'predictions.npz'),development_frozen_sha256=digest(seal),created=now()))
            del model;torch.cuda.empty_cache()
