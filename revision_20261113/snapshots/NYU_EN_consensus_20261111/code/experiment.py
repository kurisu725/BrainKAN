"""Frozen NYU EN coefficient-rank consensus; five stochastic repeats of one held-out-site split."""
from pathlib import Path
import sys,os,time,gc,traceback,json,subprocess
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import torch
from storage import read_json,write_json,digest,save_torch,save_npz
from runtime_lock import acquire
from data import load_data,validate_split
from cpss import fit_selector,roi_l2_scores
from roi_methods import rank_consensus,top_k
from trainer import train_candidate,SafePause
from models import build_model
from run_review import predict,input_features
from study import metrics,write_predictions,verify_files,retire_checkpoint

ROOT=Path(__file__).resolve().parents[1]
_data=None
def status(stage,**kw):
    write_json(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S'),**kw))
def paused():return (ROOT/'PAUSE').exists()
def check_pause():
    if paused():raise SafePause()
def init_worker(dataset):
    global _data
    torch.set_num_threads(1)
    x,y,_,_=load_data(dataset['path'],dataset['label_column']);_data=(x,y)
def fit_half(job,cfg,plan_hash):
    x,y=_data;ids=np.asarray(job['indices'],dtype=int)
    fit=fit_selector(x[ids],y[ids],cfg)
    folder=ROOT/'selectors'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    save_npz(folder/'fit.npz',**{k:v for k,v in fit.items() if k!='metadata'})
    meta={**job,**fit['metadata'],'plan_sha256':plan_hash,'fit_sha256':digest(folder/'fit.npz'),
          'final_test_used':False,'scaler_fit_indices':job['indices'],'coefficient_fit_indices':job['indices']}
    write_json(folder/'complete.json',meta)
    return dict(id=job['id'],seconds=meta['seconds'],iterations=meta['iterations'])
def jobs_for(plan):
    jobs=[]
    for p in range(5):
        for pair in plan['half_samples'][str(p)]:
            for side in ['a','b']:
                jobs.append(dict(id=f'p{p}/pair_{pair["pair"]:02d}_{side}',repeat=p,
                    pair=pair['pair'],side=side,indices=pair[side]))
    return jobs
def selectors(plan,phash):
    jobs=jobs_for(plan);pending=[];done=0
    for job in jobs:
        folder=ROOT/'selectors'/job['id'];marker=folder/'complete.json'
        if marker.exists():
            m=read_json(marker)
            assert m['plan_sha256']==phash and m['indices']==job['indices']
            assert digest(folder/'fit.npz')==m['fit_sha256'];done+=1
        else:pending.append(job)
    status('selecting',completed_selectors=done,total_selectors=500,workers=plan['selector_workers'])
    if pending:
        iterator=iter(pending);active={}
        with ProcessPoolExecutor(max_workers=plan['selector_workers'],initializer=init_worker,initargs=(plan['dataset'],)) as executor:
            for _ in range(plan['selector_workers']):
                job=next(iterator,None)
                if job:active[executor.submit(fit_half,job,plan['selector'],phash)]=job
            while active:
                finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    job=active.pop(future);result=future.result();done+=1
                    print('SELECTOR',result,'completed',done,'of 500',flush=True)
                    status('pausing_selectors' if paused() else 'selecting',completed_selectors=done,total_selectors=500,
                           running_selectors=len(active),last=result)
                    if not paused():
                        nxt=next(iterator,None)
                        if nxt:active[executor.submit(fit_half,nxt,plan['selector'],phash)]=nxt
                if paused() and not active:raise SafePause()
    check_pause();assert done==500
    for p in range(5):
        score_rows=[];hashes={}
        for job in [j for j in jobs if j['repeat']==p]:
            path=ROOT/'selectors'/job['id']/'fit.npz';hashes[str(path.relative_to(ROOT))]=digest(path)
            with np.load(path) as a:
                np.testing.assert_allclose(a['scores'],roi_l2_scores(a['coef'],90),rtol=1e-12,atol=1e-12)
                score_rows.append(a['scores'])
        q=rank_consensus(score_rows);rois=top_k(q,75).tolist()
        selection=dict(repeat=p,rois=rois,scores=q.tolist(),fit_indices=plan['splits'][p]['indices']['train'],
            source='100 train-half Elastic Net rankings',selector_files=hashes,final_test_used=False,plan_sha256=phash)
        path=ROOT/'selections'/f'p{p}.json'
        if path.exists():assert read_json(path)==selection
        else:write_json(path,selection)


def dependency_workers(plan):
    dependency=Path(plan['gpu_queue_after'])
    if not dependency.exists():return []
    # Inspect actual command lines, including both Windows venv parent/child PIDs.
    quoted=str(dependency).replace("'","''")
    command=("@((Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\") | "
             "Where-Object { $_.CommandLine -and $_.CommandLine.Contains('"+quoted+"') "
             "-and $_.CommandLine.Contains('experiment.py') } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True,timeout=30)
    found=json.loads(result.stdout.strip() or '[]')
    return found if isinstance(found,list) else [found]
def wait_for_gpu(plan):
    while True:
        check_pause();workers=dependency_workers(plan)
        if not workers:
            write_json(ROOT/'audit/gpu_queue_release.json',dict(dependency=plan['gpu_queue_after'],
                worker_processes=[],time=time.strftime('%Y-%m-%dT%H:%M:%S')))
            return
        status('waiting_gpu',completed_selectors=500,dependency=plan['gpu_queue_after'],
               worker_pids=[p['ProcessId'] for p in workers])
        time.sleep(15)

def main():
    lock=acquire(ROOT/'state/worker.lock')
    if lock is None:return
    try:
        plan=read_json(ROOT/'plan.json');phash=digest(ROOT/'plan.json')
        assert phash==read_json(ROOT/'plan_seal.json')['sha256']
        verify_files(plan['source_sha256'],ROOT);verify_files(plan['previous_artifacts'])
        assert digest(plan['dataset']['path'])==plan['dataset']['sha256']
        check_pause();selectors(plan,phash)
        wait_for_gpu(plan)
        torch.set_num_threads(4)
        x,y,_,ph=load_data(plan['dataset']['path'],2)
        assert len(np.unique(ph[:,0]))==882
        rows=[]
        checkpoint=ROOT/'state/current.pt'
        if checkpoint.exists():
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False);job=ck['job'];del ck
            if (ROOT/'runs'/job/'complete.json').exists():retire_checkpoint(ROOT,job)
        for split in plan['splits']:
            p=split['id'];idx=validate_split(split['indices'],len(y),y)
            cfg={**plan['config'],'seed':split['training_seed']}
            selection=read_json(ROOT/'selections'/f'p{p}.json')
            assert selection['fit_indices']==split['indices']['train']
            for variant in plan['variants']:
                job=f'{variant}_p{p}';folder=ROOT/'runs'/job;folder.mkdir(parents=True,exist_ok=True)
                rois=selection['rois'] if variant=='en75' else list(range(90))
                marker=folder/'complete.json'
                if marker.exists():
                    row=read_json(marker);assert row['config']==cfg and row['rois']==rois
                    verify_files(row['files'],folder);rows.append(row);continue
                check_pause();status('training',job=job,completed_models=len(rows),total_models=10,epoch=0,completed_selectors=500)
                def on_epoch(epoch,entry,saved):
                    status('training',job=job,completed_models=len(rows),total_models=10,epoch=epoch,epochs=50,saved_epoch=saved,completed_selectors=500)
                    print(job,'epoch',epoch,'train_ce',entry['training_ce'],'dev_acc',entry['development']['accuracy'],flush=True)
                tr,dv=idx['train'],idx['valid']
                payload,prob,history=train_candidate(x[tr][:,rois][:,:,rois],y[tr],x[dv][:,rois][:,:,rois],y[dv],cfg,job,rois,phash,ROOT,
                    on_epoch=on_epoch,pause_check=paused,interval=5)
                save_torch(folder/'model.pt',payload);write_json(folder/'history.json',history)
                write_predictions(folder/'development_predictions.csv',dv,y[dv],prob)
                row=dict(job=job,variant=variant,repeat=p,config=cfg,rois=rois,n_roi=len(rois),
                    development=metrics(y[dv],prob),plan_sha256=phash,final_test_used=False)
                row['files']={f.name:digest(f) for f in folder.iterdir() if f.name!='complete.json'}
                write_json(marker,row);rows.append(row);retire_checkpoint(ROOT,job)
                del payload;gc.collect();torch.cuda.empty_cache()
        check_pause();assert len(rows)==10
        paths=[p for d in ['runs','selectors','selections'] for p in (ROOT/d).rglob('*') if p.is_file()]
        frozen=dict(plan_sha256=phash,development_records=10,selectors=500,
                    files={str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)},final_test_used=False)
        if (ROOT/'development_frozen.json').exists():assert read_json(ROOT/'development_frozen.json')==frozen
        else:write_json(ROOT/'development_frozen.json',frozen)
        for row in rows:
            cfg=row['config']
            check_pause();job=row['job'];jp=ROOT/'final_test'/f'{job}.json';cp=jp.with_suffix('.csv')
            if jp.exists():assert digest(cp)==read_json(jp)['prediction_sha256'];continue
            status('final_evaluation',job=job,completed_models=10,completed_selectors=500)
            folder=ROOT/'runs'/job;payload=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
            assert payload['epoch']==50 and payload['config']==cfg
            rois=row['rois'];ids=plan['splits'][row['repeat']]['indices']['test']
            model=build_model('kan_legacy',len(rois),cfg).to('cuda');model.load_state_dict(payload['state'])
            prob=predict(model,input_features(x[ids][:,rois][:,:,rois],'kan_legacy'),'cuda')
            write_predictions(cp,ids,y[ids],prob)
            write_json(jp,dict(job=job,variant=row['variant'],repeat=row['repeat'],n_roi=len(rois),
                metrics=metrics(y[ids],prob),prediction_sha256=digest(cp),model_sha256=digest(folder/'model.pt'),
                development_frozen_sha256=digest(ROOT/'development_frozen.json')))
            del payload,model;gc.collect();torch.cuda.empty_cache()
        verify_files(plan['previous_artifacts']);status('evaluated',selectors=500,models=10,predictions=10)
    except SafePause:
        previous=read_json(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        status('paused',previous=previous);sys.exit(75)
    except Exception:
        status('failed',error=traceback.format_exc());raise
    finally:lock.close()
if __name__=='__main__':main()
