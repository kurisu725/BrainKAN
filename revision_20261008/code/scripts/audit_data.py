"""Export the one recovered fixed split and raw phenotype rows with source mapping."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brainkan.data import load_data, validate_split, file_hash

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', required=True)
p.add_argument('--dataset', required=True, choices=['abide','adhd'])
p.add_argument('--output', required=True)
args = p.parse_args()
out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
column = 2 if args.dataset == 'abide' else 3
x, y, indices, ph = load_data(args.data, column)
validate_split(indices,len(y),y)
source = loadmat(args.data)
fingerprint = file_hash(args.data)
mapping = {}
if 'phenotype' in source and 'net' in source:
    for i, row in enumerate(source['phenotype']):
        key = tuple(row.tolist())
        if key in mapping: raise ValueError('Duplicate full phenotype rows in original net pool')
        mapping[key] = i
source_ids = []
for i, row in enumerate(ph):
    source_id = mapping.get(tuple(row.tolist()), -1)
    if source_id >= 0:
        np.testing.assert_allclose(x[i], source['net'][source_id], atol=1e-6, rtol=1e-6)
    source_ids.append(source_id)
if mapping and (len(set(source_ids)) != len(y) or -1 in source_ids):
    raise ValueError('Stored splits do not uniquely map to original net pool')
np.savez_compressed(out/'stored_split.npz', **indices, data_sha256=fingerprint)
with (out/'subjects.csv').open('w',newline='',encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['split','split_row_zero_based','pool_index_zero_based','source_net_row_zero_based']+
                    [f'phenotype_col_{i}' for i in range(ph.shape[1])])
    for split, ids in indices.items():
        for local_i, pool_i in enumerate(ids):
            writer.writerow([split,local_i,int(pool_i),source_ids[pool_i]]+ph[pool_i].tolist())
report = {'dataset':args.dataset,'data_sha256':fingerprint,'label_column_zero_based':column,
          'index_space':'concatenated net_train, net_valid, net_test',
          'source_net_mapping_verified':bool(mapping),'subjects':len(y),
          'historical_five_seeds_recovered':False,
          'split_counts':{k:len(v) for k,v in indices.items()},
          'phenotype_column_unique_counts':[len(np.unique(ph[:,i])) for i in range(ph.shape[1])]}
(out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
