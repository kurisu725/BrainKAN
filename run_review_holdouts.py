"""Evaluate the original Min-max mode on the five archived holdouts per dataset."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from jbhi_revision.common import (
    TRAINING, configure_runtime, fit_kan, load_groups, predict, sha256, write_json, write_predictions,
)
from run_review_experiments import GROUPS, make_plan, new_model, read_json, save_model, train_job


def holdout_plan(inputs, environment):
    plan = make_plan(inputs, environment)
    partitions = {name: split for name, split in plan["partitions"].items() if name.startswith("abide_holdout_")}
    labels = inputs["adhd"][1]
    pooled = np.concatenate([labels[group] for group in GROUPS])
    for split_seed in range(5):
        remaining, test = train_test_split(np.arange(len(pooled)), test_size=len(labels["test"]),
                                           stratify=pooled, random_state=split_seed)
        train, development = train_test_split(remaining, test_size=len(labels["valid"]),
                                               stratify=pooled[remaining], random_state=1000 + split_seed)
        partitions[f"adhd_holdout_{split_seed}"] = {
            "dataset": "adhd", "test_split_seed": split_seed, "development_split_seed": 1000 + split_seed,
            "indices": {group: np.sort(indices).tolist()
                        for group, indices in zip(GROUPS, (train, development, test))},
        }
    plan["partitions"] = partitions
    plan["jobs"] = [{"partition": name, "seed": 1, "evaluate": ["minmax"]} for name in partitions]
    plan["code_sha256"][Path(__file__).name] = sha256(__file__)
    return plan


def reuse_selection(directory, job, split, matrices, labels, device, review, reference):
    run_id = directory.name
    selection_path = review / run_id / "selection.json"
    selection_digest = reference["runs"][run_id]["files"]["selection.json"]
    if sha256(selection_path) != selection_digest:
        raise ValueError(f"Changed frozen ABIDE selection: {selection_path}")
    selection = read_json(selection_path)
    if (selection["partition"] != job["partition"] or selection["seed"] != job["seed"] or
            selection["score_cohort"] != "train" or selection["score_pool_indices"] != split["indices"]["train"]):
        raise ValueError(f"Incorrect ABIDE selection provenance: {selection_path}")
    complete = directory / "complete.json"
    if complete.exists():
        record = read_json(complete)
        if record["reference_selection_sha256"] != selection_digest or any(
                sha256(directory / name) != digest for name, digest in record["files"].items()):
            raise ValueError(f"Changed holdout artifacts: {directory}")
        return record
    directory.mkdir(exist_ok=True)
    rois = np.asarray(selection["selections"]["minmax"]["indices"], dtype=int)
    model, history, _ = fit_kan(matrices, labels, rois, job["seed"], device, shuffle=True)
    save_model(directory, "minmax", model, rois)
    write_json(directory / "selection.json", selection)
    write_json(directory / "minmax_history.json", history)
    record = {"fitted_models": ["minmax"], "reference_selection_sha256": selection_digest,
              "files": {name: sha256(directory / name) for name in (
                  "minmax.pt", "selection.json", "minmax_history.json")}}
    write_json(complete, record)
    return record


def run(args):
    environment = configure_runtime(args.device)
    inputs = {name: load_groups(path, name) for name, path in (("abide", args.abide), ("adhd", args.adhd))}
    plan = holdout_plan(inputs, environment)
    reference = None
    if args.review is not None:
        reference = read_json(args.review / "frozen_before_test.json")
        source_plan = read_json(args.review / "plan.json")
        if (reference["plan_sha256"] != sha256(args.review / "plan.json") or
                any(source_plan[key] != plan[key] for key in ("training", "selection", "datasets")) or
                any(source_plan["partitions"][name] != split for name, split in plan["partitions"].items()
                    if split["dataset"] == "abide")):
            raise ValueError("ABIDE reference plan is incompatible with the requested holdouts")
        plan["abide_selection_plan_sha256"] = reference["plan_sha256"]
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Frozen holdout plan mismatch. Use a new output directory.")
    write_json(plan_path, plan)
    pools = {dataset: (np.concatenate([matrices[group] for group in GROUPS]),
                       np.concatenate([labels[group] for group in GROUPS]))
             for dataset, (matrices, labels, _) in inputs.items()}
    del inputs
    completed = {}
    for job in plan["jobs"]:
        split = plan["partitions"][job["partition"]]
        matrices, labels = pools[split["dataset"]]
        train = np.asarray(split["indices"]["train"])
        directory = output / f"{job['partition']}_seed_{job['seed']}"
        print(f"Fitting Min-max for {job['partition']}", flush=True)
        if reference is not None and split["dataset"] == "abide":
            record = reuse_selection(directory, job, split, matrices[train], labels[train], args.device,
                                     args.review, reference)
        else:
            record = train_job(directory, job, split, matrices[train], labels[train], args.device, None)
        completed[directory.name] = record
    write_json(output / "frozen_before_test.json", {"plan_sha256": sha256(plan_path), "runs": completed})
    for job in plan["jobs"]:
        split = plan["partitions"][job["partition"]]
        directory = output / f"{job['partition']}_seed_{job['seed']}"
        checkpoint = torch.load(directory / "minmax.pt", map_location="cpu", weights_only=True)
        rois = np.asarray(checkpoint["rois"], dtype=int)
        model = new_model(rois)
        model.load_state_dict(checkpoint["state"])
        matrices, labels = pools[split["dataset"]]
        test = np.asarray(split["indices"]["test"])
        probabilities = predict(model.to(args.device), matrices[test], rois)
        write_predictions(directory / "minmax_test.csv", test, labels[test], probabilities)
        del model, checkpoint
    print("Finished five-holdout test evaluation for both datasets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abide", type=Path, required=True)
    parser.add_argument("--adhd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="Reuse sealed training-only ABIDE selections")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    with threadpool_limits(limits=TRAINING["cpu_threads"], user_api="blas"):
        run(args)


if __name__ == "__main__":
    main()
