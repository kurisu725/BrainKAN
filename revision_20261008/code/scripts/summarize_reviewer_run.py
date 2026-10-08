"""Summarize actual frozen reviewer outputs without inventing missing results."""
from pathlib import Path
import json,sys
import numpy as np

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def display(values,percent=False):
    v=[x for x in values if x is not None]
    if not v:return 'NA'
    a=np.asarray(v)*(100 if percent else 1)
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}' if len(a)>1 else f'{a[0]:.2f} (n=1)'

def main(root):
    lines=['# Reviewer experiments: actual results','',
        'Five stochastic runs on one fixed subject split per dataset. All models use their final training epoch.',
        'All prespecified arms are reported. Standard deviations summarize stochastic fitting on the fixed test cohort.','']
    for dataset in ['abide','adhd']:
        folder=root/dataset
        plan=read(folder/'plan.json');summary=read(folder/'final_test/summary.json')
        rows=read(folder/'final_test/all_runs.json')
        lines += ['## '+dataset.upper(),'',
            '| Arm | Completed / requested | ROI count | ACC (%) | Macro F1 (%) | AUC (%) |',
            '|---|---:|---:|---:|---:|---:|']
        for arm,s in summary['arms'].items():
            counts=[r['n_roi'] for r in rows if r['arm']==arm and 'test' in r]
            columns=[arm,f"{s['n_completed']} / {s['n_requested']}",display(counts)]
            columns += [display(s['metrics'][m]['values'],True) for m in ['accuracy','f1','auc']]
            lines.append('| '+' | '.join(columns)+' |')
        failures=[r for r in rows if 'test' not in r]
        if failures:
            lines+=['','Incomplete/undefined arms (not silently omitted):']
            lines += [f"- Seed {r['seed']}, {r['arm']}: {r.get('reason',r.get('status','not evaluated'))}" for r in failures]
        stability=read(folder/'roi_seed_stability.json')
        lines+=['','### ROI stability on the fixed split','',
            '| Arm | Runs with a selected subset | Pairwise Jaccard mean ± SD |','|---|---:|---:|']
        for arm,s in stability.items():
            lines.append(f"| {arm} | {s['n_completed']} | {display([p['jaccard'] for p in s['pairwise_jaccard']])} |")
        comparisons={};structures={}
        for seed in plan['seeds']:
            for row in read(folder/f'seed_{seed}/ranking_comparisons.json'):
                comparisons.setdefault((row['arm_a'],row['arm_b']),[]).append(row)
            for p in (folder/f'seed_{seed}').glob('*/explanation.json'):
                structures.setdefault(p.parent.name,[]).extend(read(p)['structural_audits'])
        lines+=['','### Perturbation comparison','',
            '| Arm A | Arm B | Spearman mean ± SD | Matched top-20 Jaccard mean ± SD |','|---|---|---:|---:|']
        for (a,b),rs in comparisons.items():
            lines.append(f"| {a} | {b} | {display([r['spearman'] for r in rs])} | {display([r['top_k_jaccard'] for r in rs])} |")
        lines+=['','### Sampled structural diagnostics','',
            'Diagnostics cover the declared development-subject/unit sample and first perturbation repeat. For asymmetric inputs, the eigenvalue is calculated from the symmetric part.','',
            '| Arm | Audited perturbation batches | Maximum asymmetry | Maximum diagonal error | Minimum eigenvalue of symmetric part |',
            '|---|---:|---:|---:|---:|']
        for arm,rs in structures.items():
            lines.append(f"| {arm} | {len(rs)} | {max(r['max_asymmetry'] for r in rs):.4g} | {max(r['max_diagonal_error'] for r in rs):.4g} | {min(r['minimum_eigenvalue_of_symmetric_part'] for r in rs):.4g} |")
        lines+=['','Raw probabilities, full metrics, per-unit loss differences and selection frequencies remain in the corresponding experiment folders.','']
    lines += ['## Evaluation design', '', 'Results describe five stochastic seeds on fixed subject assignments. ROI stability summarizes seed variation; perturbation comparisons summarize method sensitivity. Negative importance is baseline loss minus perturbed loss. The selected regions are model-derived candidates.', '']
    (root/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main(Path(sys.argv[1]))
