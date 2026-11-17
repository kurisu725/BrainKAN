"""Reconstruct a reported split from the original MAT without copying datasets.

Example (the default command only prints counts):
    python code/load_selected_activity_entropy_split.py --study abide --round 1
Add --export-npz PATH only when a separate array file is actually wanted.
No model is trained or evaluated by this utility.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from data import load_data

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_selected_split(study, selected_round, report=None, source_mat=None):
    """Return train/valid/test FC arrays, labels, indices and original phenotypes.

    CSV row order is preserved. The source file must match its recorded hash,
    including when source_mat supplies a relocated copy of the original MAT.
    """
    report_path = Path(report) if report else ROOT / 'analysis/activity_entropy_selected_supplement_report.json'
    result = json.loads(report_path.read_text(encoding='utf-8'))
    if not result['passed']:
        raise ValueError('The selected-result report has not passed its checks.')
    entry = next(s for s in result['selected_split_files']
                 if s['study'] == study and s['selected_round'] == selected_round)
    split_path = ROOT / entry['path']
    if sha256(split_path) != entry['sha256']:
        raise ValueError('Split CSV hash differs from the reported assignment.')
    spec = json.loads((ROOT / 'development_plan.json').read_text(encoding='utf-8'))['studies'][study]
    mat_path = Path(source_mat) if source_mat else Path(spec['dataset']['path'])
    if sha256(mat_path) != spec['dataset']['sha256']:
        raise ValueError('MAT hash differs from the original source data.')
    matrix, labels, _, phenotype = load_data(mat_path, spec['label_column'])
    with split_path.open(encoding='utf-8-sig', newline='') as handle:
        assignments = list(csv.DictReader(handle))
    indices = [int(row['zero_based_original_index']) for row in assignments]
    if sorted(indices) != list(range(len(labels))):
        raise ValueError('Assignments must partition every original participant exactly once.')
    roles = {}
    for role in ['train', 'valid', 'test']:
        rows = [row for row in assignments if row['role'] == role]
        idx = np.asarray([int(row['zero_based_original_index']) for row in rows], dtype=int)
        if len(idx) != entry['counts'][role]['total']:
            raise ValueError('Role size differs from the reported split.')
        np.testing.assert_array_equal(labels[idx], [int(row['label']) for row in rows])
        roles[role] = dict(x=matrix[idx], y=labels[idx], original_indices=idx,
                           phenotype=phenotype[idx])
    return roles, dict(study=study, selected_round=selected_round,
                      random_split_seed=entry['random_split_seed'],
                      split_csv=entry['path'], split_csv_sha256=entry['sha256'],
                      source_mat_sha256=spec['dataset']['sha256'],
                      report_sha256=sha256(report_path),
                      no_model_forwards=True, no_parameter_updates=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, choices=['abide', 'adhd'])
    parser.add_argument('--round', required=True, type=int, choices=range(1, 6), dest='selected_round')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--source-mat', type=Path)
    parser.add_argument('--export-npz', type=Path)
    args = parser.parse_args()
    roles, metadata = load_selected_split(args.study, args.selected_round, args.report, args.source_mat)
    if args.export_npz:
        if args.export_npz.exists():
            raise FileExistsError(f'Refusing to overwrite {args.export_npz}')
        arrays = {f'{role}_{key}': value for role, values in roles.items()
                  for key, value in values.items()}
        with args.export_npz.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
    counts = {role: dict(total=len(values['y']),
                        classes=np.bincount(values['y'].astype(int), minlength=2).tolist(),
                        fc_shape=list(values['x'].shape)) for role, values in roles.items()}
    print(json.dumps(dict(**metadata, counts=counts,
                         exported_npz=str(args.export_npz) if args.export_npz else None)))


if __name__ == '__main__':
    main()
