from pathlib import Path
import json, hashlib
import numpy as np
from scipy.io import loadmat
R = Path(__file__).resolve().parents[1]
old = json.loads((R.parent/'Random_split_comparison_20261027/plan.json').read_text())
m = loadmat(old['dataset']['path'])
fc = np.concatenate([m['net_'+s] for s in ('train','valid','test')])
ph = np.concatenate([m['phenotype_'+s] for s in ('train','valid','test')])
paths = json.loads((R/'audit/time_series_availability.json').read_text())['raw_files']
report = {'fc_shape':list(fc.shape), 'fc_range':[float(fc.min()),float(fc.max())], 'examples':[], 'lengths':{}}
for i,sid in enumerate(ph[:,0].astype(int)):
    p = Path(paths[str(sid)])
    ts = np.loadtxt(p)
    report['lengths'][str(sid)] = len(ts)
    if i < 3:
        cor = np.corrcoef(ts[:,:90].T)
        ix = np.triu_indices(90,1)
        a,b=fc[i][ix],cor[ix]
        report['examples'].append({'id':int(sid),'fc_diag':np.diag(fc[i])[:5].tolist(),'direct_corr':float(np.corrcoef(a,b)[0,1]),'tanh_max_abs':float(np.abs(np.tanh(a)-b).max()),'corr_first100':float(np.corrcoef(a,np.corrcoef(ts[:100,:90].T)[ix])[0,1])})
report['min_length'] = min(report['lengths'].values())
(R/'audit/input_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='lengths'},indent=2),flush=True)
