"""One fixed [64,32] KAN on each frozen training-only EN-consensus Top-75 selection."""
from pathlib import Path
import os,sys,time,gc,traceback
import numpy as np
import torch
from storage import read_json,write_json,digest,save_torch
from runtime_lock import acquire
from data import load_data,validate_split
from models import build_model
from run_review import input_features,predict
from trainer import train_candidate,SafePause
from study import ROOT,parameter_count,geometry,metrics,write_predictions,verify_files,retire_checkpoint

def status(stage,**kw):write_json(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S'),**kw))
def pause():return (ROOT/'PAUSE').exists()
def check_pause():
    if pause():raise SafePause()
def verify_plan():
    p=read_json(ROOT/'plan.json');h=digest(ROOT/'plan.json');assert h==read_json(ROOT/'plan_seal.json')['sha256']
    verify_files(p['source_sha256'],ROOT);verify_files(p['previous_artifacts'])
    assert digest(p['dataset']['path'])==p['dataset']['sha256'];return p,h
def validate_selection(selection,idx):
    assert selection['source']=='100 train-half Elastic Net rankings' and not selection['final_test_used']
    assert selection['fit_indices']==idx['train'] and selection['score_indices']==idx['train']
    scores=np.asarray(selection['scores']);ids=np.arange(90);expected=np.sort(ids[np.lexsort((ids,-scores))[:75]]).tolist()
    assert selection['rois']==expected and len(expected)==75
    assert not set(selection['fit_indices']) & (set(idx['valid'])|set(idx['test']))

def main():
    lock=acquire(ROOT/'state/worker.lock')
    if lock is None:return
    try:
        torch.set_num_threads(4);plan,h=verify_plan();check_pause();cfg=plan['config'];rows=[]
        if (ROOT/'state/current.pt').exists():
            ck=torch.load(ROOT/'state/current.pt',map_location='cpu',weights_only=False);j=ck['job'];del ck
            if (ROOT/'runs'/j/'complete.json').exists():retire_checkpoint(ROOT,j)
        x,y,_,ph=load_data(plan['dataset']['path']);assert len(np.unique(ph[:,0]))==len(y)
        for split in plan['splits']:
            part=split['id'];idx=validate_split(split['indices'],len(y),y);selection=plan['selections'][str(part)]
            validate_selection(selection,split['indices']);rois=selection['rois'];job=f'en75_6432_p{part}'
            folder=ROOT/'runs'/job;folder.mkdir(exist_ok=True);marker=folder/'complete.json'
            if marker.exists():
                row=read_json(marker);assert row['config']==cfg and row['rois']==rois;verify_files(row['files'],folder);rows.append(row);continue
            check_pause();status('developing',job=job,completed=len(rows),total=5,epoch=0);print('JOB',job,flush=True)
            def on_epoch(epoch,entry,saved):
                status('developing',job=job,completed=len(rows),total=5,epoch=epoch,epochs=50,saved_epoch=saved,development=entry['development'])
                print(job,'epoch',epoch,'train_ce',round(entry['training_ce'],5),'dev_acc',round(entry['development']['accuracy'],5),flush=True)
            tr,dv=idx['train'],idx['valid']
            payload,prob,history=train_candidate(x[tr][:,rois][:,:,rois],y[tr],x[dv][:,rois][:,:,rois],y[dv],cfg,job,rois,h,ROOT,
                on_epoch=on_epoch,pause_check=pause,interval=5)
            model=build_model('kan_legacy',75,cfg);geo=geometry(model,75,cfg);model.load_state_dict(payload['state']);del model
            save_torch(folder/'model.pt',payload);write_json(folder/'history.json',history);write_json(folder/'selection.json',selection)
            write_predictions(folder/'development_predictions.csv',dv,y[dv],prob)
            row=dict(job=job,branch='en75_6432',partition=part,config=cfg,rois=rois,n_roi=75,parameter_count=parameter_count(75,cfg),
                geometry=geo,development=metrics(y[dv],prob),plan_sha256=h,final_test_used=False)
            write_json(folder/'config.json',row);row['files']={p.name:digest(p) for p in folder.iterdir() if p.name!='complete.json'}
            write_json(marker,row);rows.append(row);retire_checkpoint(ROOT,job);del payload;gc.collect();torch.cuda.empty_cache()
        check_pause();assert len(rows)==5
        choices=[dict(record=r,model_path=str((ROOT/'runs'/r['job']/'model.pt').resolve()),
                      model_sha256=digest(ROOT/'runs'/r['job']/'model.pt')) for r in rows]
        files={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'runs').glob('*/*') if p.is_file()}
        frozen=dict(plan_sha256=h,development_records=5,choices=choices,files=files,final_test_used=False)
        if (ROOT/'development_frozen.json').exists():assert read_json(ROOT/'development_frozen.json')==frozen
        else:write_json(ROOT/'development_frozen.json',frozen)
        verify_files(files,ROOT)
        for c in choices:
            check_pause();r=c['record'];job=r['job'];jp=ROOT/'final_test'/f'{job}.json';cp=jp.with_suffix('.csv')
            if jp.exists():
                saved=read_json(jp);assert saved['job']==job and digest(cp)==saved['prediction_sha256'];continue
            status('final_evaluation',job=job,completed_development=5)
            ck=torch.load(c['model_path'],map_location='cpu',weights_only=False);assert ck['epoch']==50 and ck['config']==cfg
            model=build_model('kan_legacy',75,cfg).to('cuda');model.load_state_dict(ck['state'])
            ids=plan['splits'][r['partition']]['indices']['test'];rois=r['rois']
            prob=predict(model,input_features(x[ids][:,rois][:,:,rois],'kan_legacy'),'cuda')
            write_predictions(cp,ids,y[ids],prob);write_json(jp,dict(job=job,partition=r['partition'],n_roi=75,
                metrics=metrics(y[ids],prob),config=cfg,model_sha256=c['model_sha256'],prediction_sha256=digest(cp),
                development_frozen_sha256=digest(ROOT/'development_frozen.json')))
            del model,ck;gc.collect();torch.cuda.empty_cache()
        verify_files(plan['previous_artifacts']);status('evaluated',new_trainings=5,final_predictions=5)
    except SafePause:
        prior=read_json(ROOT/'status.json') if (ROOT/'status.json').exists() else {};saved=None
        if (ROOT/'state/current.pt').exists():
            ck=torch.load(ROOT/'state/current.pt',map_location='cpu',weights_only=False);saved=dict(job=ck['job'],epoch=ck['epoch']);del ck
        status('paused',previous=prior,saved_checkpoint=saved);sys.exit(75)
    except Exception:status('failed',error=traceback.format_exc());raise
    finally:lock.close()
if __name__=='__main__':main()
