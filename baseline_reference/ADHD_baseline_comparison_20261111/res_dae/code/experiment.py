"""Five-partition reproduction. Development completes globally before test access."""
from pathlib import Path
import argparse,copy,json,os,random,sys,time,traceback
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,log_loss
from storage import read_json,write_json,save_torch,save_npz,digest
from runtime_lock import acquire
from models import build,fb_objective,gnn_regularization
from preprocess import fit_res_preprocessor,apply_res_preprocessor,fb_scaler,temporal_inputs
ROOT=Path(__file__).resolve().parents[1]

def now():return datetime.now().astimezone().isoformat()
def status(stage,**kw):write_json(ROOT/'status.json',dict(stage=stage,time=now(),pid=os.getpid(),**kw))
class Paused(Exception):pass
def pause_check():
    if (ROOT/'PAUSE').exists():raise Paused()
def verify_frozen():
    manifest=read_json(ROOT/'frozen_manifest.json')
    for p,h in manifest.items():assert digest(ROOT/p)==h,f'Frozen file changed: {p}'
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state():return dict(py=random.getstate(),np=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
def restore_rng(s):
    random.setstate(s['py']);np.random.set_state(s['np']);torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:torch.cuda.set_rng_state_all(s['cuda'])
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def metrics(y,p,threshold=.5):
    pred=(p>=threshold).astype(int)
    return dict(ACC=float(accuracy_score(y,pred)),P=float(precision_score(y,pred,average='macro',zero_division=0)),
                R=float(recall_score(y,pred,average='macro',zero_division=0)),F1=float(f1_score(y,pred,average='macro',zero_division=0)),
                AUC=float(roc_auc_score(y,p)),loss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])))
def choose_threshold(y,p):
    def key(t):
        pred=p>=t;j=pred[y==1].mean()-pred[y==0].mean()
        return (-float(j),abs(t-.5),t)
    return float(min(np.linspace(0,1,11),key=key))

def get_inputs(data,split,kind):
    tr=np.asarray(split['indices']['train']);dev=np.asarray(split['indices']['development']);part=split['id']
    prepfile=ROOT/'runs'/f'partition_{part}_{kind}_preprocessor.pt'
    if prepfile.exists():s=torch.load(prepfile,map_location='cpu')
    elif kind=='res_dae':
        train_x,train_y,prep=fit_res_preprocessor(data['fc'][tr],data['labels'][tr],data['sites'][tr],data['phenotypes'][tr,3],data['phenotypes'][tr,4],tr)
        s=dict(train_x=train_x,train_y=train_y,prep=prep);save_torch(prepfile,s)
    elif kind=='fbnetgen':
        s=dict(prep=fb_scaler(data['time_series'],data['lengths'],tr));save_torch(prepfile,s)
    else:
        s=dict(prep=dict(mode='within_subject_per_roi_zscore',fit_indices=list(map(int,tr))));save_torch(prepfile,s)
    assert s['prep']['fit_indices']==list(map(int,tr))
    if kind=='res_dae':
        def transform(idx):return apply_res_preprocessor(data['fc'][idx],data['sites'][idx],data['phenotypes'][idx,3],data['phenotypes'][idx,4],s['prep'])
        return dict(train_x=s['train_x'],train_y=s['train_y'],dev_x=transform(dev),dev_y=data['labels'][dev],prep=s['prep'],transform=transform)
    # Per-subject normalization or frozen training scalar; padded frames stay zero.
    normalized=temporal_inputs(data['time_series'],data['lengths'],kind,s['prep'])
    return dict(train_x=normalized[tr],train_y=data['labels'][tr],dev_x=normalized[dev],dev_y=data['labels'][dev],
                train_fc=data['fc'][tr],dev_fc=data['fc'][dev],train_len=data['lengths'][tr],dev_len=data['lengths'][dev],prep=s['prep'],normalized=normalized)

@torch.no_grad()
def predict(model,kind,x,device,batch_size,fc=None,lengths=None):
    model.eval();values=[]
    for start in range(0,len(x),batch_size):
        a=torch.as_tensor(x[start:start+batch_size],device=device)
        if kind=='fbnetgen':logit=model(a,torch.as_tensor(fc[start:start+batch_size],device=device),torch.as_tensor(lengths[start:start+batch_size]))[0]
        elif kind=='brainnetgnn':logit=model(a,torch.as_tensor(lengths[start:start+batch_size]))
        else:logit=model(a,False,False)[0]
        prob=logit.sigmoid() if kind=='res_dae' else logit.softmax(-1)[:,1]
        values.append(prob.cpu().numpy())
    result=np.concatenate(values).astype(float);assert np.isfinite(result).all();return result

def training_epoch(model,optimizer,kind,inputs,cfg,epoch,device):
    model.train()
    if kind=='res_dae' and epoch>cfg['joint_epochs']:model.freeze_encoder()
    order=np.random.permutation(len(inputs['train_y']));total=0.
    for start in range(0,len(order),cfg['batch_size']):
        ids=order[start:start+cfg['batch_size']];x=torch.as_tensor(inputs['train_x'][ids],device=device);y=torch.as_tensor(inputs['train_y'][ids],device=device)
        optimizer.zero_grad(set_to_none=True)
        if kind=='fbnetgen':loss=fb_objective(model,x,torch.as_tensor(inputs['train_fc'][ids],device=device),y,torch.as_tensor(inputs['train_len'][ids]))
        elif kind=='brainnetgnn':loss=torch.nn.functional.cross_entropy(model(x,torch.as_tensor(inputs['train_len'][ids])),y)+gnn_regularization(model)
        else:
            joint=epoch<=cfg['joint_epochs'];logit,recon=model(x,add_noise=joint,reconstruct=joint)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,y.float())
            if joint:loss=loss+torch.nn.functional.mse_loss(recon,x)
        assert torch.isfinite(loss),f'Nonfinite training loss {kind} epoch {epoch}'
        loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
    return total/len(order)

def train_job(job,inputs,device):
    name=job['id'];cfg=job['config'];kind=job['kind'];recordfile=ROOT/'runs'/f'{name}.json'
    if recordfile.exists():return read_json(recordfile)
    seed_all(1);model=build(kind,cfg,device);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.1,patience=10) if kind=='brainnetgnn' else None
    checkpoint=ROOT/'state/current.pt';epoch=0;history=[];best_loss=float('inf');best_epoch=0;best_state=None;bad_epochs=0;started=now()
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu');assert ck['job_id']==name,(ck['job_id'],name)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        if scheduler is not None:scheduler.load_state_dict(ck['scheduler'])
        epoch=ck['epoch'];history=ck['history'];best_loss=ck['best_loss'];best_epoch=ck['best_epoch'];best_state=ck['best_state'];bad_epochs=ck['bad_epochs'];started=ck['started'];restore_rng(ck['rng'])
    while epoch<cfg['epochs'] and not(kind=='brainnetgnn' and bad_epochs>=15):
        pause_check();epoch+=1;t=time.monotonic();loss=training_epoch(model,optimizer,kind,inputs,cfg,epoch,device)
        row=dict(epoch=epoch,training_loss=loss,lr=optimizer.param_groups[0]['lr'])
        if kind=='brainnetgnn':
            p=predict(model,kind,inputs['dev_x'],device,cfg['batch_size'],inputs.get('dev_fc'),inputs.get('dev_len'))
            measure=metrics(inputs['dev_y'],p);objective=measure['loss']+float(gnn_regularization(model).detach())
            row.update(development=measure,development_objective=objective)
            if objective<best_loss:best_loss=objective;best_epoch=epoch;best_state=cpu_state(model);bad_epochs=0
            else:bad_epochs+=1
            scheduler.step(objective)
        row['seconds']=time.monotonic()-t;history.append(row)
        save_torch(checkpoint,dict(job_id=name,epoch=epoch,model=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,
                   best_loss=best_loss,best_epoch=best_epoch,best_state=best_state,bad_epochs=bad_epochs,history=history,rng=rng_state(),started=started))
        status('training',job=name,epoch=epoch,max_epochs=cfg['epochs'],last_epoch_seconds=row['seconds'],completed_jobs=len(list((ROOT/'runs').glob('*_candidate_*.json'))))
        print(f'{now()} {name} epoch={epoch} loss={loss:.6f} seconds={row["seconds"]:.2f}',flush=True)
        pause_check()
    if kind=='brainnetgnn':model.load_state_dict(best_state)
    else:best_epoch=epoch
    p=predict(model,kind,inputs['dev_x'],device,cfg['batch_size'],inputs.get('dev_fc'),inputs.get('dev_len'))
    threshold=choose_threshold(inputs['dev_y'],p) if kind=='res_dae' else .5
    measure=metrics(inputs['dev_y'],p,threshold);devpath=ROOT/'runs'/f'{name}_development.npz'
    save_npz(devpath,y=inputs['dev_y'],probability=p,prediction=(p>=threshold).astype(int))
    record=dict(job_id=name,kind=kind,partition=job['partition'],config=cfg,started=started,completed=now(),epochs_trained=epoch,checkpoint_epoch=best_epoch,
                threshold=threshold,development_metrics=measure,development_predictions_sha256=digest(devpath),history=history)
    selected=ROOT/'selected'/f'partition_{job["partition"]}_{kind}.pt'
    leaderpath=selected.with_suffix('.json');old=read_json(leaderpath) if leaderpath.exists() else None
    def key(r):return (-r['development_metrics']['AUC'],r['development_metrics']['loss'],r['job_id'])
    if old is None or key(record)<key(old):
        save_torch(selected,dict(model=cpu_state(model),kind=kind,config=cfg,threshold=threshold,checkpoint_epoch=best_epoch,job_id=name))
        record['selected_weight_sha256_at_commit']=digest(selected);write_json(leaderpath,record)
    write_json(recordfile,record)
    checkpoint.unlink(missing_ok=True)
    del model,optimizer,best_state
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return record

def run(device):
    verify_frozen();plan=read_json(ROOT/'plan.json')
    from dependency import dependency_state
    ready,reason=dependency_state(plan['dependency']);assert ready,reason
    data=dict(np.load(ROOT/'data/adhd_fc_metadata.npz',allow_pickle=False))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    checkpoint=ROOT/'state/current.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu')
        if (ROOT/'runs'/f'{saved["job_id"]}.json').exists():checkpoint.unlink()
        del saved
    for split in plan['splits']:
        for kind in plan['methods']:
            pause_check();jobs=[j for j in plan['jobs'] if j['partition']==split['id'] and j['kind']==kind]
            if all((ROOT/'runs'/f'{j["id"]}.json').exists() for j in jobs):continue
            status('preparing_training_inputs',partition=split['id'],kind=kind)
            inputs=get_inputs(data,split,kind)
            for job in jobs:train_job(job,inputs,device)
    verify_frozen();pause_check();sealpath=ROOT/'development_frozen.json'
    if not sealpath.exists():
        files={str(p.relative_to(ROOT)):digest(p) for folder in ['runs','selected'] for p in sorted((ROOT/folder).glob('*')) if p.is_file()}
        assert len(list((ROOT/'selected').glob('*.pt')))==5
        assert len(list((ROOT/'runs').glob('*_candidate_*.json')))==len(plan['jobs'])
        write_json(sealpath,dict(time=now(),files=files,rule='All candidates and five partitions completed; Res-DAE candidate uses development AUC, loss, identifier; threshold uses development Youden.'))
    seal=read_json(sealpath)
    for p,h in seal['files'].items():assert digest(ROOT/p)==h,p
    for split in plan['splits']:
        idx=np.asarray(split['indices']['test']);part=split['id']
        for kind in plan['methods']:
            pause_check();name=f'partition_{part}_{kind}';recordpath=ROOT/'final_test'/f'{name}.json'
            if recordpath.exists():continue
            status('final_evaluation',partition=part,kind=kind)
            state=torch.load(ROOT/'selected'/f'{name}.pt',map_location='cpu');model=build(kind,state['config'],device);model.load_state_dict(state['model'])
            prep=torch.load(ROOT/'runs'/f'{name}_preprocessor.pt',map_location='cpu')['prep']
            if kind=='res_dae':
                x=apply_res_preprocessor(data['fc'][idx],data['sites'][idx],data['phenotypes'][idx,3],data['phenotypes'][idx,4],prep)
                p=predict(model,kind,x,device,state['config']['batch_size'])
            else:
                x=temporal_inputs(data['time_series'][idx],data['lengths'][idx],kind,prep)
                p=predict(model,kind,x,device,state['config']['batch_size'],data['fc'][idx],data['lengths'][idx])
            y=data['labels'][idx];pred=(p>=state['threshold']).astype(int);npz=recordpath.with_suffix('.npz')
            save_npz(npz,indices=idx,subject_ids=data['subject_ids'][idx],y=y,probability=p,prediction=pred)
            write_json(recordpath,dict(partition=part,kind=kind,job_id=state['job_id'],metrics=metrics(y,p,state['threshold']),threshold=state['threshold'],
                      checkpoint_epoch=state['checkpoint_epoch'],n_test=len(idx),n_input_roi=90,n_selected_edges=2002 if kind=='res_dae' else None,
                      weight_sha256=digest(ROOT/'selected'/f'{name}.pt'),prediction_sha256=digest(npz),development_frozen_sha256=digest(sealpath),completed=now()))
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    status('complete',development_jobs=len(plan['jobs']),final_evaluations=5)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if not args.run:raise SystemExit('Use --run via the dependency controller.')
    handle=acquire(ROOT/'state/worker.lock')
    if handle is None:raise SystemExit('Worker is already running.')
    try:run(args.device)
    except Paused:status('paused',checkpoint_exists=(ROOT/'state/current.pt').exists());sys.exit(75)
    except Exception as e:status('failed',error=repr(e));traceback.print_exc();sys.exit(1)
