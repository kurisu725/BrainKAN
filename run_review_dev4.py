"""One-off development-cohort comparison for the held-out NYU partition.

Locks NYU (site code 11) as the test cohort and evaluates BrainKAN's three ROI
selection modes with the reserved development cohort defined as the four largest
non-NYU sites by subject count (13=Leuven, 6=KU, 9=Trinity, 7=Olin), matching the
paper's narrative that the largest sites are reserved for development. The test
cohort is only touched once after fitting.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from jbhi_revision.common import (
    TRAINING, configure_runtime, load_groups, summarize, write_json,
)
from run_review_development import evaluate_run, fit_run


DEV_SITE_CODES = {"13": "Leuven", "6": "KU", "9": "Trinity", "7": "Olin"}


def build_partition(mat_path):
    arrays = loadmat(mat_path, variable_names=["phenotype_train", "phenotype_valid", "phenotype_test"])
    pooled = np.concatenate([arrays["phenotype_train"], arrays["phenotype_valid"], arrays["phenotype_test"]])
    site = pooled[:, 1].astype(int)
    train = np.flatnonzero(~np.isin(site, [11, *map(int, DEV_SITE_CODES)])).tolist()
    dev = np.flatnonzero(np.isin(site, list(map(int, DEV_SITE_CODES)))).tolist()
    test = np.flatnonzero(site == 11).tolist()
    return {"dataset": "abide", "indices": {"train": train, "valid": dev, "test": test},
            "test_site": "NYU", "development_sites": DEV_SITE_CODES,
            "site_codes": {"train": sorted(set(site[train].tolist())),
                           "valid": list(map(int, DEV_SITE_CODES)), "test": [11]}}


def run(args):
    configure_runtime(args.device)
    matrices, labels, _ = load_groups(args.abide, "abide")
    pooled_matrices = np.concatenate([matrices[g] for g in ("train", "valid", "test")])
    pooled_labels = np.concatenate([labels[g] for g in ("train", "valid", "test")])
    split = build_partition(args.abide)
    print(f"train={len(split['indices']['train'])} dev={len(split['indices']['valid'])} "
          f"test={len(split['indices']['test'])}", flush=True)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "partition.json", split)
    rows, sealed = [], {}
    for seed in range(5):
        directory = output / f"abide_dev4_seed_{seed}"
        record = fit_run(directory, split, pooled_matrices, pooled_labels, seed, args.device)
        sealed[directory.name] = record
    write_json(output / "sealed_before_test.json" , {"runs": sealed})
    for seed in range(5):
        directory = output / f"abide_dev4_seed_{seed}"
        rows.extend(evaluate_run(directory, sealed[directory.name], split,
                                 pooled_matrices, pooled_labels, args.device))
    groups = {}
    for row in rows:
        groups.setdefault(row["model"], []).append(row)
    summary = {mode: {"n_runs": len(group), "n_roi": [row["n_roi"] for row in group],
                      "epochs": [row["epoch"] for row in group], "metrics": summarize(group)}
               for mode, group in groups.items()}
    write_json(output / "results.json", {"development": split["development_sites"],
                                          "test": "NYU", "per_run": rows, "summary": summary})
    for mode, values in summary.items():
        mean = 100 * values["metrics"]["accuracy"]["mean"]
        sd = 100 * values["metrics"]["accuracy"]["sd"]
        auc = 100 * values["metrics"]["auc"]["mean"]
        print(f"{mode}: acc={mean:.2f}±{sd:.2f} auc={auc:.2f} "
              f"per-seed={[round(100 * r['accuracy'], 2) for r in groups[mode]]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abide", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    with threadpool_limits(limits=TRAINING["cpu_threads"], user_api="blas"):
        run(args)


if __name__ == "__main__":
    main()
