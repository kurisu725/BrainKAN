"""Development-only exact fitting of a one-layer spline KAN.

The linear algebra solves the coefficients of this KAN itself. It is not an
external classifier, a feature preselector, or an Elastic Net fit. Each input
connection has an independently learned identity-plus-cubic-spline function.
Pruning scores evaluate those fitted functions on training subjects only.
"""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,roc_auc_score
from data import load_data,validate_split
from legacy_kan import KANLinear

ROOT=Path(__file__).resolve().parents[1]


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8');temp.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(x,train,sites,mode,subject_mask=None):
    """Every pooled statistic is trained on train, never validation or test."""
    z=x.copy();state={'mode':mode,'fitted_indices':list(map(int,train))}
    if mode=='subject_z':
        selected=z if subject_mask is None else z[:,subject_mask]
        z=(z-selected.mean(1,keepdims=True))/selected.std(1,keepdims=True).clip(min=1e-4)
        state['subject_normalization_features']=list(range(x.shape[1])) if subject_mask is None else np.where(subject_mask)[0].tolist()
    if mode=='site_residual_z':
        global_mean=z[train].mean(0);site_means={}
        for site in np.unique(sites[train]):
            rows=train[sites[train]==site]
            # Shrink very small training-site groups toward the pooled mean.
            site_means[float(site)]=(z[rows].sum(0)+10*global_mean)/(len(rows)+10)
        z=np.stack([v-site_means.get(float(s),global_mean) for v,s in zip(z,sites)])
        state['site_means']={str(s):v.tolist() for s,v in site_means.items()}
        state['unknown_site_mean']=global_mean.tolist()
    mean=z[train].mean(0);std=z[train].std(0).clip(min=1e-3)
    z=np.clip((z-mean)/std,-3,3)
    # The unpenalized intercept absorbs this training centering exactly.
    center=z[train].mean(0);z=z-center
    state.update(mean=mean.tolist(),std=std.tolist(),center=center.tolist())
    return torch.as_tensor(z,dtype=torch.float32,device='cuda'),state


@torch.no_grad()
def spline_features(x,ntrain):
    # A fixed KAN grid; centering and linear orthogonalization use train only.
    layer=KANLinear(x.shape[1],1,grid_size=5,spline_order=3,grid_range=[-3,3]).cuda()
    b=torch.cat([layer.b_splines(v) for v in x.split(128)])
    mean=b[:ntrain].mean(0);b=b-mean
    slope=(b[:ntrain]*x[:ntrain,:,None]).mean(0)/x[:ntrain].square().mean(0).clamp_min(1e-5)[:,None]
    b=b-x[:,:,None]*slope
    # One common training scale per edge, preserving a coherent function shape.
    scale=b[:ntrain].square().mean((0,2)).sqrt().clamp_min(.03)
    b=b/scale[None,:,None]
    return b,dict(basis_mean=mean.cpu(),basis_linear=slope.cpu(),basis_scale=scale.cpu())


def metric(y,s):return {'accuracy':float(accuracy_score(y,s>0)), 'auc':float(roc_auc_score(y,s))}


@torch.no_grad()
def fit(x,b,y,ntrain,keep,gamma,rf,pairs=None):
    tr=x[:ntrain,keep];va=x[ntrain:,keep]
    tb=b[:ntrain,keep];vb=b[ntrain:,keep]
    p=int(keep.sum());kernel=(tr@tr.T+gamma*tb.flatten(1)@tb.flatten(1).T)/p
    alpha=float(kernel.diag().mean())*rf
    target=torch.as_tensor(2*y[:ntrain]-1,dtype=torch.float32,device='cuda')
    intercept=target.mean();target=target-intercept
    dual=torch.linalg.solve(kernel+alpha*torch.eye(ntrain,device='cuda'),target)
    linear=tr.T@dual/p;nonlinear=gamma*torch.einsum('nek,n->ek',tb,dual)/p
    scores=torch.cat([tr@linear+torch.einsum('nek,ek->n',tb,nonlinear),
                      va@linear+torch.einsum('nek,ek->n',vb,nonlinear)])+intercept
    values=tr*linear[None]+torch.einsum('nek,ek->ne',tb,nonlinear)
    edge=values.abs().mean(0)
    full_edge=torch.zeros(x.shape[1],device='cuda');full_edge[keep]=edge
    if pairs is None:pairs=torch.triu_indices(90,90,1,device='cuda')
    roi=torch.zeros(90,device='cuda')
    roi.index_add_(0,pairs[0],full_edge)
    off_diagonal=pairs[0]!=pairs[1]
    roi.index_add_(0,pairs[1,off_diagonal],full_edge[off_diagonal])
    result={'linear':linear.cpu(),'spline':nonlinear.cpu(),'intercept':intercept.cpu(),
            'feature_indices':torch.where(keep)[0].cpu(),'gamma':gamma,'ridge_factor':rf,
            'normal_equation_residual':float((kernel@dual+alpha*dual-target).abs().max())}
    return scores.cpu().numpy(),roi.cpu().numpy(),result


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(4);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    torch.use_deterministic_algorithms(True)
    plan=json.loads((ROOT/'development_plan.json').read_text(encoding='utf8'))
    design=plan['analytic_kan_pilot'];records=[]
    for study in ['abide','adhd']:
        spec=plan['studies'][study]
        assert digest(spec['dataset']['path'])==spec['dataset']['sha256']
        matrices,y,_,ph=load_data(spec['dataset']['path'],spec['label_column'])
        ij=np.triu_indices(90,1);raw=matrices[:,ij[0],ij[1]]
        sites=ph[:,0 if study=='adhd' else 1]
        for split in spec['splits'][:2]:
            idx=validate_split(split['indices'],len(y),y);ntrain=len(idx['train'])
            # Test inputs are not even materialized in this experiment.
            allowed=np.r_[idx['train'],idx['valid']]
            yy=y[allowed];ss=sites[allowed];xx=raw[allowed]
            output=ROOT/'pilot'/study/f'partition_{split["id"]}';output.mkdir(parents=True,exist_ok=True)
            best=None
            for mode in design['preprocess']:
                x,state=features(xx,np.arange(ntrain),ss,mode)
                state['fitted_indices']=list(map(int,idx['train']))
                b,bs=spline_features(x,ntrain)
                for gamma in design['nonlinear_strength']:
                    for rf in design['ridge_factors']:
                        key=f'{mode}_g{gamma}_r{rf}'
                        path=output/(key+'.json')
                        if path.exists():
                            old=json.loads(path.read_text());records.extend(old)
                            continue
                        start=time.perf_counter();all_keep=torch.ones(x.shape[1],dtype=torch.bool,device='cuda')
                        _,roi,_=fit(x,b,yy,ntrain,all_keep,gamma,rf)
                        order=np.argsort(-roi,kind='stable');rows=[]
                        for k in design['retained_rois']:
                            rois=order[:k];keep=np.isin(ij[0],rois)&np.isin(ij[1],rois)
                            scores,_,model=fit(x,b,yy,ntrain,torch.as_tensor(keep,device='cuda'),gamma,rf)
                            row=dict(study=study,partition=split['id'],preprocess=mode,gamma=gamma,ridge_factor=rf,
                                     rois=sorted(rois.tolist()),n_rois=k,training=metric(yy[:ntrain],scores[:ntrain]),
                                     development=metric(yy[ntrain:],scores[ntrain:]),
                                     normal_equation_residual=model['normal_equation_residual'])
                            rows.append(row)
                            rank=(sum(row['development'].values())/2,-k)
                            if best is None or rank>best[0]:
                                best=(rank,row)
                                torch.save(dict(model=model,transform=state,basis=bs,training_indices=idx['train'],
                                           development_indices=idx['valid'],roi_scores=roi,config=row),output/'best_model.pt')
                        torch.cuda.synchronize()
                        for row in rows:row['configuration_seconds']=time.perf_counter()-start
                        dump(path,rows);records.extend(rows)
                        dump(ROOT/'status.json',dict(stage='additive_development_pilot',pid=os.getpid(),study=study,
                             partition=split['id'],configuration=key,best_current_process=None if best is None else best[1]))
                        print(study,split['id'],key,'best',best[1]['development'] if best else None,flush=True)
                del b,x;torch.cuda.empty_cache()
    dump(ROOT/'pilot/additive_records.json',records)
    summaries=[]
    keys=sorted({(r['preprocess'],r['gamma'],r['ridge_factor'],r['n_rois']) for r in records})
    for study in ['abide','adhd']:
        for key in keys:
            rows=[r for r in records if r['study']==study and (r['preprocess'],r['gamma'],r['ridge_factor'],r['n_rois'])==key]
            summaries.append(dict(study=study,preprocess=key[0],gamma=key[1],ridge_factor=key[2],n_rois=key[3],
                                  mean_development={m:float(np.mean([r['development'][m] for r in rows])) for m in ['accuracy','auc']}))
    summaries.sort(key=lambda r:sum(r['mean_development'].values()),reverse=True)
    dump(ROOT/'pilot/additive_summary.json',summaries)
    dump(ROOT/'status.json',dict(stage='additive_pilot_complete',pid=os.getpid(),records=len(records)))
    print('Pilot completed',len(records),'records',flush=True)


if __name__=='__main__':main()
