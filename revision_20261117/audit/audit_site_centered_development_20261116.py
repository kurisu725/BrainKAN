"""Independent role, contribution, cutoff, metric and stopping audit."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code')); sys.path.insert(0,str(ROOT/'audit'))
from data import load_data
from audit_target_test_search_20261116 import read,sha,write,audit_prediction,summary,stability

OUT=ROOT/'site_centered_development_20261116'


def main():
    plan=read(OUT/'plan.json')
    for name,digest in plan['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    assert not list(OUT.rglob('*test*predictions*'))
    assert not any(f.suffix.lower() in ['.pt','.pth','.ckpt','.mat'] for f in OUT.rglob('*'))
    results=[]; fit_cost={}
    for study in ['abide','adhd']:
        spec=plan['dataset_specifications'][study]
        assert sha(spec['dataset']['path'])==spec['dataset']['sha256']
        matrices,labels,_,ph=load_data(spec['dataset']['path'],spec['label_column']); del matrices
        rows=[]
        for round_index,seed in enumerate(plan['random_seeds'][study]):
            folder=OUT/study/f'round_{round_index+1:02d}'
            split=read(folder/'split.json'); assert split['plan_sha256']==sha(OUT/'plan.json')
            train,rest=train_test_split(np.arange(len(labels)),test_size=.2,random_state=seed,stratify=labels)
            valid,test=train_test_split(rest,test_size=.5,random_state=seed,stratify=labels[rest])
            idx={k:np.sort(v) for k,v in dict(train=train,valid=valid,test=test).items()}
            for role,values in idx.items(): np.testing.assert_array_equal(values,split['indices'][role])
            assert sha(folder/'assignments.csv')==split['assignment_sha256']
            for recipe in plan['recipes']:
                case=folder/recipe; row=read(case/'record.json'); cfg=row['configuration']
                assert row['plan_sha256']==sha(OUT/'plan.json') and row['test_evaluated'] is False
                assert cfg['initialization_ridge_strengths']==[.1,1.,10.] and cfg['roi_mass']==plan['dataset_roi_mass'][study]
                assert cfg['bootstrap_seeds']==[9161601+100000*(study=='adhd')+100*round_index+b for b in range(4)]
                for b,info in enumerate(row['initializations']):
                    rng=np.random.default_rng(cfg['bootstrap_seeds'][b]); y=labels[idx['train']]
                    draw=np.concatenate([rng.choice(np.flatnonzero(y==c),int((y==c).sum()),replace=True) for c in [0,1]])
                    draw=draw[rng.permutation(len(draw))]
                    np.testing.assert_array_equal(info['training_indices'],idx['train'][draw])
                    if recipe=='within_site_guided32':
                        np.testing.assert_array_equal(info['training_sites'],ph[idx['train'][draw],plan['phenotype_columns'][study]['site']])
                        assert max(info['independent_cpu_maximum_coefficient_errors'])<1e-7
                        assert not info['sites_used_at_inference'] and not info['roi_ranking_or_selection']
                        for group in info['site_groups']:
                            chosen=np.asarray(info['training_sites'])==group['site']
                            assert group['class_counts']==np.bincount(y[draw][chosen],minlength=2).tolist()
                            assert abs(group['centered_response_mean'])<1e-12 and group['maximum_centered_feature_mean']<1e-10
                with np.load(case/'training_contributions.npz') as d:
                    np.testing.assert_array_equal(d['training_indices'],idx['train'])
                    scores=np.sqrt(np.mean(d['signed_branch_effects'].astype(float).mean(0)**2,axis=0))
                    np.testing.assert_allclose(scores,d['roi_scores'],atol=1e-6,rtol=1e-6)
                    scores=d['roi_scores'].astype(float); order=np.argsort(-scores,kind='stable'); square=scores[order]**2
                    count=int(np.searchsorted(np.cumsum(square),cfg['roi_mass']*square.sum(),side='left'))+1
                    np.testing.assert_array_equal(np.sort(order[:count]),row['rois'])
                assert sha(case/'development_predictions.npz')==row['development_predictions_sha256']
                _,logits=audit_prediction(case/'development_predictions.npz',idx['valid'],labels,row['development'],row['threshold'])
                grid=np.asarray(cfg['threshold_probability_grid']); thresholds=np.log(grid/(1-grid))
                acc=[np.mean((logits>=t)==labels[idx['valid']]) for t in thresholds]
                chosen=min(range(len(grid)),key=lambda i:(-acc[i],abs(thresholds[i]),i))
                assert row['threshold']==thresholds[chosen] and not row['threshold_calibration']['test_used']
                check=row['independent_development_forward']; assert check['passed'] and check['maximum_logit_error']<check['tolerance']
                assert all(b['deleted_fc_nan_error']<check['tolerance'] for b in check['deleted_input_boundary_checks'])
                assert all(m['maximum_input_coefficient_change']>0 for m in row['training_coefficient_movement'])
                for ref in row['fit_records']:
                    assert sha(ROOT/ref['path'])==ref['sha256']; fit=read(ROOT/ref['path'])
                    candidates=fit['training_internal_candidates']
                    best=max(range(len(candidates)),key=lambda i:(candidates[i]['oob_auc'],-candidates[i]['update']))
                    assert fit['selected_update']==candidates[best]['update'] and not fit['development_checkpoint_selection']
                    assert all(np.isfinite(h['objective_before_update']) and np.isfinite(h['gradient_norm_before_update']) for h in fit['history'])
                    fit_cost[ref['path']]=dict(updates=fit['updates_run'],seconds=fit['elapsed_fit_seconds'])
                rows.append(row)
        recipes=[]
        for recipe in plan['recipes']:
            group=[r for r in rows if r['recipe']==recipe]; assert len(group)==3
            roi=stability(group); roi['interpretation']='Descriptive overlap on three predefined development divisions, not independent biological replication.'
            recipes.append(dict(recipe=recipe,development=summary(group,'development'),roi_stability=roi))
        baseline,candidate=[r['development'] for r in recipes]
        differences={key:candidate[key]['mean']-baseline[key]['mean'] for key in ['accuracy','auc']}
        results.append(dict(study=study,recipes=recipes,paired_mean_differences=differences,
                            meets_advance_rule=all(v>0 for v in differences.values()),rows=rows))
    report=dict(passed=True,complete=True,test_evaluated=False,cases=12,results=results,
                actual_fit_seconds=sum(f['seconds'] for f in fit_cost.values()),actual_updates=sum(f['updates'] for f in fit_cost.values()),
                weight_files_written=0,plan_sha256=sha(OUT/'plan.json'),interpretation=plan['interpretation'])
    write(OUT/'audit_report.json',report)
    lines=['# 站点中心化初始化：固定划分开发比较','','各数据集预先固定三组随机划分，共十二个配对开发案例；全部报告，没有新测试集预测。开发 ACC 使用本组开发标签校准分类阈值，不能作为主表测试结果。','','| 数据集 | 初始化 | 开发 ACC % | 开发 AUC % | ROI 数 |','|---|---|---:|---:|---|']
    for result in results:
        for r in result['recipes']:
            a=r['development']['accuracy']; u=r['development']['auc']
            lines.append(f"| {result['study'].upper()} | {r['recipe']} | {a['mean']*100:.2f} ± {a['sample_sd']*100:.2f} | {u['mean']*100:.2f} ± {u['sample_sd']*100:.2f} | {r['roi_stability']['roi_counts']} |")
        lines.extend(['',f"{result['study'].upper()} 同时提高平均 ACC 与 AUC：{result['meets_advance_rule']}。"])
    lines.extend(['','仅改变三个 KAN 隐通道的训练内辅助系数初始化，其余完整函数训练、活动熵、剪弱函数、KAN 输出贡献与 mass90/mass95 自动选 ROI 流程相同。站点不进入推理。没有外部 ROI 排名器。', '',
                  '历史上两队列已被反复探索，人员跨划分重叠；本轮属于开发证据。全部划分名单、逐组预测、贡献、停止候选和标量轨迹在同目录。零模型权重、零 FC MAT 副本。旧版五组测试结果与正文未覆盖。'])
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=12,test_evaluated=False,results=[dict(study=r['study'],differences=r['paired_mean_differences'],advance=r['meets_advance_rule']) for r in results])),flush=True)


if __name__=='__main__': main()
