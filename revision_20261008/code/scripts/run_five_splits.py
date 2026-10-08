"""New repeatable splits; the historical five seeds were not present in the sources."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', choices=['abide', 'adhd'], required=True)
p.add_argument('--data', required=True)
p.add_argument('--output', required=True)
p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
p.add_argument('--explainer', choices=['lime','permutation'], default='permutation')
p.add_argument('--roi-source', choices=['train','valid'], default='valid')
p.add_argument('--direct-threshold', type=float, default=0.0)
p.add_argument('--minmax-threshold', type=float, default=0.5)
p.add_argument('--zscore-threshold', type=float, default=0.0)
p.add_argument('--epochs', type=int, default=50)
p.add_argument('--device', default='auto')
args = p.parse_args()
out = Path(args.output).resolve()
if out.exists() and any(out.iterdir()): raise FileExistsError(out)
out.mkdir(parents=True, exist_ok=True)
base = [sys.executable, str(ROOT/'run.py')]
common = ['--data', str(Path(args.data).resolve()), '--dataset', args.dataset]
subprocess.run(base+['splits']+common+['--output', str(out/'splits'), '--seeds']+list(map(str,args.seeds)), check=True)
runs = []
for seed in args.seeds:
    target = out/f'seed_{seed}'
    subprocess.run(base+['run']+common+['--split-file', str(out/'splits'/f'seed_{seed}.npz'),
        '--output', str(target), '--seed', str(seed), '--epochs', str(args.epochs),
        '--device', args.device, '--explainer', args.explainer, '--roi-source', args.roi_source,
        '--direct-threshold', str(args.direct_threshold), '--minmax-threshold', str(args.minmax_threshold),
        '--zscore-threshold', str(args.zscore_threshold)], check=True)
    runs.append(json.loads((target/'metrics.json').read_text(encoding='utf-8')))
summary = {'historical_reproduction': False, 'seeds':args.seeds, 'methods':{}}
names = sorted(set().union(*(set(r) for r in runs)))
for name in names:
    valid = [r[name]['test'] for r in runs if name in r and 'test' in r[name]]
    if not valid: continue
    summary['methods'][name] = {'n_completed':len(valid), 'metrics':{}}
    for key in ['accuracy','precision','recall','f1','auc']:
        values = [r[key] for r in valid]
        summary['methods'][name]['metrics'][key] = {'values':values, 'mean':float(np.mean(values)),
            'std_sample':float(np.std(values,ddof=1)) if len(values)>1 else None}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
