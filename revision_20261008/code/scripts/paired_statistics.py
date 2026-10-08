"""Paired t-tests with Holm correction per metric; requires actual per-split results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', required=True, help='CSV columns: method,split,metric,value')
p.add_argument('--reference', default='BrainKAN')
p.add_argument('--output', required=True)
args = p.parse_args()
groups = {}
with open(args.input, encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        group = groups.setdefault(row['metric'], {}).setdefault(row['method'], {})
        if row['split'] in group: raise ValueError('Duplicate method/split/metric')
        value = float(row['value'])
        if not np.isfinite(value): raise ValueError('Metrics must be finite')
        group[row['split']] = value
results = {}
for metric, methods in groups.items():
    ref = methods[args.reference]
    comparisons = []
    for method, scores in methods.items():
        if method == args.reference: continue
        if set(scores) != set(ref) or len(ref)<2: raise ValueError('Require identical split IDs and >=2 pairs')
        keys = sorted(ref)
        a, b = np.array([ref[k] for k in keys]), np.array([scores[k] for k in keys])
        delta = a-b
        if np.all(delta==0): pvalue = 1.0
        elif np.std(delta)==0: pvalue = 0.0
        else: pvalue = float(ttest_rel(a,b).pvalue)
        comparisons.append({'baseline':method,'n_pairs':len(keys),'mean_difference':float(delta.mean()),'p':pvalue})
    order = sorted(range(len(comparisons)),key=lambda i:comparisons[i]['p'])
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1., comparisons[i]['p']*(len(order)-rank)))
        comparisons[i]['p_holm'] = maximum
        comparisons[i]['reference_better_significant'] = maximum<.05 and comparisons[i]['mean_difference']>0
    results[metric] = comparisons
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
Path(args.output).write_text(json.dumps(results,indent=2),encoding='utf-8')
