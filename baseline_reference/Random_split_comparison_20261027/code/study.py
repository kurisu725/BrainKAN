"""Five predeclared diagnosis-stratified holdouts. All development ends before test."""
from pathlib import Path
import argparse,csv,gc,hashlib,itertools,json,os,sys,time,traceback
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score
import joblib
import run_review as r
from data import load_data,file_hash,validate_split,features
from models import build_model
from permutation import fast_permutation
from roi import roi_scores,select_rois,transform_scores

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'E:\Brain\BrainKAN\BrainKAN-complete\data\ALLASD1_aal.mat')
PRIOR=ROOT.parent/'Advisor_revision_20261026'
EXPECTED_DATA_SHA='0e69963407181c6df5eeab9dab4c97bfa39f9df98365918c658dd2683f054cac'
NAMES={
 'kan_fixed':'Full KAN / fixed grid','kan_adaptive':'Full KAN / adaptive grid',
 'brainkan_fixed_direct':'BrainKAN / fixed / Direct','brainkan_fixed_minmax':'BrainKAN / fixed / Min-max',
 'brainkan_fixed_zscore':'BrainKAN / fixed / Z-score',
 'brainkan_adaptive_direct':'BrainKAN / adaptive / Direct','brainkan_adaptive_minmax':'BrainKAN / adaptive / Min-max',
 'brainkan_adaptive_zscore':'BrainKAN / adaptive / Z-score',
 'mlp':'MLP','brainnetcnn':'BrainNetCNN / raw logits','gcn':'FC-GCN',
 'logistic':'L2 logistic regression','linear_svm':'Linear SVM','rbf_svm':'RBF-SVM',
 'brainkan_fixed_development_selected':'BrainKAN / fixed / development-selected threshold',
 'brainkan_adaptive_development_selected':'BrainKAN / adaptive / development-selected threshold'}
FULL_MAP={'kan_fixed':'kan_fixed_replay','kan_adaptive':'kan_grid_only','mlp':'mlp_replay',
          'brainnetcnn':'bnc_raw_logits','gcn':'gcn_fc','logistic':'logistic_fc','linear_svm':'linear_svm_fc'}
CLASSICAL=('logistic','linear_svm','rbf_svm')
def write(p,v):r.write(p,v)
def read(p):return r.read(p)
def source_hashes():return {p.name:file_hash(p) for p in (ROOT/'code').glob('*.py')}
def status(state,**kw):
    write(ROOT/'status.json',dict(state=state,pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S'),**kw))
def preserve_old_manifest():
    items={}
    for base in (PRIOR,PRIOR/'site_resumed'):
        p=read(base/'experiments/plan.json')
        for name,digest in p['sources'].items():items[str(base/'code'/name)]=digest
    p=read(PRIOR/'site_resumed/experiments/frozen_site.json')
    for name,digest in p['files'].items():items[str(PRIOR/'site_resumed/experiments'/name)]=digest
    old=ROOT.parent/'Implementation_review_20261026/release'
    for name,digest in read(old/'release_manifest.json').items():items[str(old/name)]=digest
    for name,digest in items.items():
        if file_hash(name)!=digest:raise ValueError('Old artifact differs from its original manifest: '+name)
    return items

def make_plan():
    dest=ROOT/'plan.json'
    if dest.exists():raise FileExistsError('Existing plan is immutable')
    if file_hash(DATA)!=EXPECTED_DATA_SHA:raise ValueError('Unexpected data')
    x,y,original,ph=load_data(DATA,2)
    if len(np.unique(ph[:,0]))!=len(x):raise ValueError('Subject identifiers are not unique; group splitting required')
    counts={k:len(v) for k,v in original.items()};splits=[]
    for number,seed in enumerate(range(5)):
        remain,test=train_test_split(np.arange(len(x)),test_size=counts['test'],random_state=seed,stratify=y)
        train,valid=train_test_split(remain,test_size=counts['valid'],random_state=1000+seed,stratify=y[remain])
        idx=validate_split({k:np.sort(v) for k,v in dict(train=train,valid=valid,test=test).items()},len(x),y)
        cohorts={}
        for k,v in idx.items():
            cohorts[k]=dict(n=len(v),class_counts=np.bincount(y[v],minlength=2).tolist(),
                            site_counts={str(int(s)):int((ph[v,1]==s).sum()) for s in np.unique(ph[v,1])},
                            subject_ids=ph[v,0].astype(int).tolist())
        for a,b in itertools.combinations(idx,2):
            assert not set(ph[idx[a],0])&set(ph[idx[b],0])
        splits.append(dict(id=number,split_seed=seed,development_split_seed=1000+seed,
                           indices={k:v.tolist() for k,v in idx.items()},cohorts=cohorts))
    # Sort indices to preserve original pool ordering; shuffling remains enabled for training.
    jobs=[]
    for sp in splits:
        for variant in list(FULL_MAP)+['rbf_svm']:
            cfg={**r.BASE,**(r.PRESETS[FULL_MAP[variant]] if variant in FULL_MAP else
                            dict(kind='rbf_svm',C_values=[.1,1.,10.])), 'seed':1}
            jobs.append(dict(id=f'partition_{sp["id"]}_{variant}',partition=sp['id'],variant=variant,
                             config=cfg,selection=None))
    for sp in splits:
        for family in ('fixed','adaptive'):
            cfg={**r.BASE,**r.PRESETS['kan_fixed_replay' if family=='fixed' else 'kan_grid_only'],'seed':1}
            for mode in ('direct','minmax','zscore'):
                variant=f'brainkan_{family}_{mode}'
                jobs.append(dict(id=f'partition_{sp["id"]}_{variant}',partition=sp['id'],variant=variant,
                                 config=cfg,selection=dict(family=family,mode=mode,threshold=.5 if mode=='minmax' else 0.)))
    plan=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'),dataset=dict(path=str(DATA),sha256=EXPECTED_DATA_SHA,
             n_subjects=len(x),n_rois=x.shape[1],original_cohort_sizes=counts),splits=splits,jobs=jobs,
             protocol='five repeated stratified random holdouts, not five-fold CV or recovered historical partitions',
             training_seed=1,checkpoint='final epoch 50',source_sha256=source_hashes(),
             explanation=dict(method='entry permutation',repeats=10,seed=1,chunk=4,sign='baseline loss minus perturbed loss',
                              cohort='development',thresholds={'direct':0.,'minmax':.5,'zscore':0.},comparison='<',
                              each_partition_and_family_recomputed=True),
             selection=dict(neural_hyperparameters='fixed presets; no epoch selection',
                            classical_C='development AUC, then lower log loss, then smaller C',
                            primary_brainkan='development AUC, then lower log loss, then fewer ROIs, then minmax/zscore/direct'),
             final_evaluation='after all partitions/models/explanations and threshold choices have been frozen',
             prediction_metrics='ACC, macro precision/recall/F1, AUC, balanced ACC and log loss; sample SD across five partitions')
    write(ROOT/'previous_artifacts.json',preserve_old_manifest());write(dest,plan)
    write(ROOT/'split_manifest.json',dict(protocol=plan['protocol'],splits=splits))
    status('planned',total_models=len(jobs),total_partitions=len(splits))
    return plan

def verify_plan():
    p=read(ROOT/'plan.json')
    if p['source_sha256']!=source_hashes():raise ValueError('Frozen execution sources changed')
    if file_hash(p['dataset']['path'])!=p['dataset']['sha256']:raise ValueError('Data changed')
    return p

def verify_marker(folder,name='complete.json'):
    v=read(folder/name)
    for key,digest in v['files'].items():
        if file_hash(folder/key)!=digest:raise ValueError('Completed artifact changed: '+str(folder/key))
    return v
def load_model(folder,device='cuda'):
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=build_model(ck['config']['kind'],ck['n_roi'],ck['config']).to(device)
    model.load_state_dict(ck['state']);return model.eval(),ck
def selected_input(x,rois):
    return np.ascontiguousarray(x[:,rois][:,:,rois],dtype=np.float32)
def ensure_explanation(plan,partition,family,x_dev,y_dev,dev_indices):
    folder=ROOT/'explanations'/f'partition_{partition}_{family}';folder.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'runs'/f'partition_{partition}_kan_{family}'
    verify_marker(parent);parent_sha=file_hash(parent/'model.pt')
    if (folder/'complete.json').exists():
        done=verify_marker(folder)
        if done['parent_sha256']!=parent_sha:raise ValueError('Selector changed')
        return np.load(folder/'roi_scores.npy')
    status('explaining',partition=partition,family=family,total_models=len(plan['jobs']),
           completed_models=len(list((ROOT/'runs').glob('*/complete.json'))))
    model,_=load_model(parent)
    cfg=plan['explanation']
    scores,repeated,edge,meta=fast_permutation(model,x_dev,y_dev,cfg['seed'],repeats=cfg['repeats'],
                                              chunk=cfg['chunk'],checkpoint=folder/'partial.npz')
    meta.update(cohort='development',pool_indices=dev_indices.tolist(),parent_sha256=parent_sha,
                final_test_used=False,partition=partition,family=family)
    np.save(folder/'roi_scores.npy',scores);np.save(folder/'repeated_roi_scores.npy',repeated);np.save(folder/'edge_scores.npy',edge)
    write(folder/'explanation.json',meta)
    write(folder/'complete.json',dict(parent_sha256=parent_sha,files={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}))
    del model;gc.collect();torch.cuda.empty_cache();return scores

def run_job(job,plan,x,y):
    folder=ROOT/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return verify_marker(folder)
    if (folder/'failure.json').exists():return read(folder/'failure.json')
    cfg=job['config'];sp=plan['splits'][job['partition']];idx={k:np.array(v) for k,v in sp['indices'].items()}
    tr,dv=idx['train'],idx['valid'];rois=np.arange(x.shape[1])
    try:
        if job['selection']:
            spec=job['selection'];scores=ensure_explanation(plan,job['partition'],spec['family'],x[dv],y[dv],dv)
            rois=select_rois(scores,spec['mode'],spec['threshold'])
            write(folder/'selection.json',dict(**spec,raw_scores=scores.tolist(),transformed_scores=transform_scores(scores,spec['mode']).tolist(),
                                              rois=rois.tolist(),development_indices=dv.tolist(),comparison='<',final_test_used=False))
        write(folder/'config.json',dict(job=job,rois=rois.tolist(),indices=sp['indices']))
        tx,vx=selected_input(x[tr],rois),selected_input(x[dv],rois)
        status('training',job=job['id'],partition=job['partition'],total_models=len(plan['jobs']),
               completed_models=len(list((ROOT/'runs').glob('*/complete.json'))))
        if cfg['kind'] in CLASSICAL:
            if cfg['kind']=='rbf_svm':
                ft=features(tx,np.arange(len(rois))).numpy();fv=features(vx,np.arange(len(rois))).numpy()
                bestkey=None;candidates=[]
                for c in cfg['C_values']:
                    candidate=SVC(C=c,kernel='rbf',gamma='scale',probability=True,random_state=cfg['seed']).fit(ft,y[tr])
                    m=r.metric(y[dv],candidate.predict_proba(fv));key=(m['auc'],-m['loss'],-c)
                    candidates.append(dict(C=c,development=m))
                    if bestkey is None or key>bestkey:bestkey=key;model=candidate
            else:
                ft=r.input_features(tx,cfg['kind']);fv=r.input_features(vx,cfg['kind'])
                model,candidates=r.fit_classical(ft,y[tr],fv,y[dv],cfg)
            joblib.dump(model,folder/'model.joblib');write(folder/'C_selection.json',candidates)
            tp=model.predict_proba(ft);vp=model.predict_proba(fv)
        else:
            model,_,_=r.fit_neural(tx,y[tr],vx,y[dv],cfg,folder,'cuda')
            tp=r.predict(model,r.input_features(tx,cfg['kind']),'cuda')
            vp=r.predict(model,r.input_features(vx,cfg['kind']),'cuda')
        r.save_predictions(folder/'train_predictions.csv',tr,y[tr],tp)
        r.save_predictions(folder/'development_predictions.csv',dv,y[dv],vp)
        row=dict(job=job['id'],partition=job['partition'],variant=job['variant'],n_roi=len(rois),
                 prediction_dtype=str(vp.dtype),training=r.metric(y[tr],tp),development=r.metric(y[dv],vp),
                 final_test_used=False,files={p.name:file_hash(p) for p in folder.iterdir()
                 if p.is_file() and p.name not in ('complete.json','resume.pt','failure.json')})
        write(folder/'complete.json',row);del model;gc.collect();torch.cuda.empty_cache();return row
    except (FloatingPointError,ValueError) as exc:
        # Selection with zero ROIs and nonfinite training are recorded, never replaced by top-k or best epoch.
        if isinstance(exc,ValueError) and 'selected zero ROIs' not in str(exc):raise
        failure=dict(job=job['id'],partition=job['partition'],variant=job['variant'],status='failed',error=traceback.format_exc(),final_test_used=False)
        write(folder/'failure.json',failure);gc.collect();torch.cuda.empty_cache();return failure

def develop():
    plan=verify_plan();x,y,_,_=load_data(plan['dataset']['path'],2);rows=[]
    for job in plan['jobs']:
        print('JOB',job['id'],flush=True);rows.append(run_job(job,plan,x,y))
        write(ROOT/'development_summary.json',rows)
    choices={}
    preference={'minmax':0,'zscore':1,'direct':2}
    for sp in plan['splits']:
        for family in ('fixed','adaptive'):
            group=[row for row in rows if row['partition']==sp['id'] and row['variant'].startswith('brainkan_'+family+'_') and row.get('status')!='failed']
            if group:
                best=min(group,key=lambda row:(-row['development']['auc'],row['development']['loss'],row['n_roi'],preference[row['variant'].split('_')[-1]]))
                choices[f'partition_{sp["id"]}_{family}']=dict(job=best['job'],variant=best['variant'],development=best['development'],n_roi=best['n_roi'])
    write(ROOT/'development_threshold_choices.json',choices)
    frozen={}
    for base in [ROOT/'runs',ROOT/'explanations']:
        for p in base.rglob('*'):
            if p.is_file() and p.name!='resume.pt':frozen[str(p.relative_to(ROOT))]=file_hash(p)
    for name in ['development_summary.json','development_threshold_choices.json']:
        frozen[name]=file_hash(ROOT/name)
    manifest=dict(plan_sha256=file_hash(ROOT/'plan.json'),files=frozen,completed_models=sum(row.get('status')!='failed' for row in rows),
                  failed_models=sum(row.get('status')=='failed' for row in rows),all_development_complete=True,final_test_used=False)
    if (ROOT/'frozen.json').exists():
        if read(ROOT/'frozen.json')!=manifest:raise ValueError('Existing development freeze differs')
    else:write(ROOT/'frozen.json',manifest)
    status('development_complete',completed_models=manifest['completed_models'],failed_models=manifest['failed_models'])

def prediction_metrics(y,p):
    out=r.metric(y,p);pred=p.argmax(1)
    out['precision_macro']=float(precision_score(y,pred,average='macro',zero_division=0))
    out['recall_macro']=float(recall_score(y,pred,average='macro',zero_division=0))
    return out

def evaluate():
    plan=verify_plan();freeze=read(ROOT/'frozen.json')
    if freeze['plan_sha256']!=file_hash(ROOT/'plan.json') or not freeze['all_development_complete']:raise ValueError('Invalid freeze')
    for name,digest in freeze['files'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Frozen development artifact changed: '+name)
    x,y,_,_=load_data(plan['dataset']['path'],2);target=ROOT/'final_test';target.mkdir(exist_ok=True);rows=[]
    status('evaluating',completed_models=freeze['completed_models'],total_models=len(plan['jobs']))
    for job in plan['jobs']:
        folder=ROOT/'runs'/job['id']
        if (folder/'failure.json').exists():
            rows.append(read(folder/'failure.json'));continue
        ck=read(folder/'config.json');rois=np.array(ck['rois']);ids=np.array(ck['indices']['test']);kind=job['config']['kind']
        path=target/(job['id']+'.csv');marker=target/(job['id']+'.json')
        if marker.exists():
            row=read(marker)
            if row['prediction_sha256']!=file_hash(path):raise ValueError('Final predictions changed')
            rows.append(row);continue
        if path.exists():raise ValueError('Unsealed final prediction file: '+str(path))
        xx=selected_input(x[ids],rois)
        if kind in CLASSICAL:
            model=joblib.load(folder/'model.joblib')
            tensor=features(xx,np.arange(len(rois))).numpy() if kind=='rbf_svm' else r.input_features(xx,kind)
            prob=model.predict_proba(tensor)
        else:
            model,_=load_model(folder);prob=r.predict(model,r.input_features(xx,kind),'cuda')
        r.save_predictions(path,ids,y[ids],prob)
        row=dict(job=job['id'],partition=job['partition'],variant=job['variant'],n_roi=len(rois),prediction_dtype=str(prob.dtype),
                 metrics=prediction_metrics(y[ids],prob),prediction_sha256=file_hash(path),
                 frozen_sha256=file_hash(ROOT/'frozen.json'))
        write(marker,row);rows.append(row);del model;gc.collect();torch.cuda.empty_cache()
    write(target/'records.json',rows);return rows

def summarize():
    plan=verify_plan();rows=read(ROOT/'final_test/records.json');good=[row for row in rows if row.get('status')!='failed']
    choices=read(ROOT/'development_threshold_choices.json');byjob={row['job']:row for row in good}
    selected=[]
    for key,value in choices.items():
        row=byjob.get(value['job'])
        if row:
            family=key.split('_')[-1]
            selected.append({**row,'variant':f'brainkan_{family}_development_selected','selected_from':value['job']})
    allrows=good+selected;groups={}
    for variant in NAMES:
        values=[v for v in allrows if v['variant']==variant]
        if not values:continue
        metricnames=list(values[0]['metrics']);stats={}
        for key in metricnames+['n_roi']:
            vals=[v['n_roi'] if key=='n_roi' else v['metrics'][key] for v in values]
            stats[key]=dict(mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None,values=vals)
        groups[variant]=dict(name=NAMES[variant],n_completed=len(values),stats=stats)
    # Descriptive ROI stability across these overlapping repeated holdouts.
    stability={}
    for family in ('fixed','adaptive'):
        for mode in ('direct','minmax','zscore'):
            sets=[]
            for sp in plan['splits']:
                p=ROOT/'runs'/f'partition_{sp["id"]}_brainkan_{family}_{mode}'/'selection.json'
                if p.exists():sets.append((sp['id'],set(read(p)['rois'])))
            pairs=[dict(partition_a=a,partition_b=b,jaccard=len(sa&sb)/len(sa|sb))
                   for (a,sa),(b,sb) in itertools.combinations(sets,2)]
            stability[family+'_'+mode]=dict(pairwise=pairs,mean_jaccard=float(np.mean([p['jaccard'] for p in pairs])) if pairs else None,
                                          interpretation='descriptive overlapping partition pairs, not independent repetitions')
    write(ROOT/'comparison.json',dict(groups=groups,records=allrows,failed=[r for r in rows if r.get('status')=='failed'],roi_stability=stability))
    metrics=['accuracy','auc','precision_macro','recall_macro','f1_macro','balanced_accuracy','loss']
    with (ROOT/'per_partition.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['variant','partition','selected_job','n_roi']+metrics)
        for row in allrows:w.writerow([row['variant'],row['partition'],row.get('selected_from',row['job']),row['n_roi']]+[row['metrics'][k] for k in metrics])
    def fmt(g,k,percent=True):
        v=g['stats'][k];scale=100 if percent else 1;return f"{v['mean']*scale:.2f} ± {v['sd']*scale:.2f}" if v['sd'] is not None else f"{v['mean']*scale:.2f}"
    lines=['# ABIDE 五次随机划分结果','',
           '数据：同一份 ABIDE 882 人、90 ROI。每次训练709、开发89、最终测试84人，按诊断类别分层；五份划分预先保存，训练随机种子统一为1。各划分之间允许受试者重复出现，各自内部三队列互斥。这是五次重复随机留出，不是五折交叉验证，也不是恢复历史划分。','',
           '神经网络取第50轮。每份划分、每种KAN网格设置均重新计算10次开发集置换贡献，重新选ROI、重新训练子模型。全部开发任务与阈值选择完成并封存后，才执行最终测试。','',
           '下表是最终测试结果，分类指标为百分比，±为五次划分的样本标准差。Development-selected 使用各自开发集选定的阈值规则；Direct/Min-max/Z-score 全部预设结果同时列出。','',
           '| 模型 | 完成划分数 | ROI | ACC | AUC | 宏F1 |','|---|---:|---:|---:|---:|---:|']
    for name,g in groups.items():lines.append(f"| {g['name']} | {g['n_completed']} | {fmt(g,'n_roi',False)} | {fmt(g,'accuracy')} | {fmt(g,'auc')} | {fmt(g,'f1_macro')} |")
    lines+=['','## 每次划分','','| 模型 | 划分 | ROI | ACC | AUC |','|---|---:|---:|---:|---:|']
    for row in allrows:
        lines.append(f"| {NAMES[row['variant']]} | {row['partition']} | {row['n_roi']} | {row['metrics']['accuracy']*100:.2f} | {row['metrics']['auc']*100:.2f} |")
    lines+=['','## 记录','','- plan.json / split_manifest.json：冻结配置、受试者索引、类别与站点分布。',
            '- runs/：每个模型的最后一轮权重、开发结果、ROI选择记录。',
            '- explanations/：每份划分重新生成的贡献与重复级分数。',
            '- final_test/：逐受试者预测，可重新计算全部指标。',
            '- validation.json：结果复算与旧文件校验。',
            '- per_partition.csv / comparison.json：完整逐次指标和汇总。',
            '',
            '原固定划分和NYU站点实验继续独立保存；本批随机划分通常在训练和测试中共享站点，站点泛化结果仍以原NYU实验表述。']
    if len(rows)>len(good):lines+=['','失败配置：']+[f"- {row['job']}: {row.get('error','')}" for row in rows if row.get('status')=='failed']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n','utf-8')
    return groups

def validate():
    plan=verify_plan();x,y,_,ph=load_data(plan['dataset']['path'],2);records=read(ROOT/'final_test/records.json')
    verified=0;largest=0.
    for sp in plan['splits']:
        idx=validate_split({k:np.array(v) for k,v in sp['indices'].items()},len(x),y)
        for a,b in itertools.combinations(idx,2):assert not set(ph[idx[a],0])&set(ph[idx[b],0])
    for row in records:
        if row.get('status')=='failed':continue
        p=ROOT/'final_test'/(row['job']+'.csv');a=np.genfromtxt(p,delimiter=',',names=True)
        ids=np.array(plan['splits'][row['partition']]['indices']['test'])
        assert np.array_equal(ids,a['pool_index'].astype(int)) and np.array_equal(y[ids],a['label'].astype(int))
        prob=np.column_stack([a['prob_0'],a['prob_1']]).astype(row['prediction_dtype'])
        assert np.array_equal(a['pred'].astype(int),prob.argmax(1))
        metrics=prediction_metrics(y[ids],prob)
        largest=max(largest,max(abs(v-row['metrics'][k]) for k,v in metrics.items()));verified+=1
    if largest>1e-12:raise ValueError('Metric recomputation mismatch')
    old=read(ROOT/'previous_artifacts.json')
    for name,digest in old.items():
        if file_hash(name)!=digest:raise ValueError('Old artifact changed: '+name)
    frozen=read(ROOT/'frozen.json')
    for name,digest in frozen['files'].items():assert file_hash(ROOT/name)==digest
    result=dict(verified=True,partitions=len(plan['splits']),planned_models=len(plan['jobs']),completed_test_records=verified,
                failed_configs=len(records)-verified,max_metric_difference=largest,old_artifacts_unchanged=len(old),
                development_frozen_before_test=True,training_seed_fixed=1)
    write(ROOT/'validation.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','develop','evaluate','summarize','validate','all']);args=p.parse_args()
    torch.set_num_threads(4)
    if args.command=='plan':make_plan()
    elif args.command=='all':
        develop();evaluate();summarize();v=validate();status('complete',**v)
    elif args.command=='develop':develop()
    elif args.command=='evaluate':evaluate()
    elif args.command=='summarize':summarize()
    else:validate()

if __name__=='__main__':
    lock=ROOT/'run.lock'
    active=False
    try:
        if len(sys.argv)>1 and sys.argv[1] in ('all','develop','evaluate'):
            # Windows advisory lock is automatically released after a crash.
            import msvcrt
            stream=lock.open('a+b');stream.seek(0)
            if lock.stat().st_size==0:stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);active=True
        main()
    except Exception:
        if active or not lock.exists():status('failed',error=traceback.format_exc())
        raise
