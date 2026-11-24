"""Run the three original thresholds with training-only, correlation-preserving ROI scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from efficient_kan import KAN
from jbhi_revision.common import (
    MAT_HASHES, TRAINING, configure_runtime, fit_kan, load_groups, predict, sha256, write_json, write_predictions,
)


ALPHA = 0.5
THRESHOLDS = {"direct": 0.0, "minmax": 0.5, "zscore": 0.0}
GROUPS = ("train", "valid", "test")
FULL = np.arange(90)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_plan(inputs, environment):
    datasets, partitions, jobs = {}, {}, []
    for dataset, (_, labels, phenotypes) in inputs.items():
        offset = 0
        indices = {}
        for group in GROUPS:
            indices[group] = list(range(offset, offset + len(labels[group])))
            offset += len(labels[group])
        all_phenotypes = np.concatenate([phenotypes[group] for group in GROUPS])
        subjects = all_phenotypes[:, 0 if dataset == "abide" else 1]
        if len(np.unique(subjects)) != offset:
            raise ValueError(f"Duplicate subject identifiers in {dataset}")
        datasets[dataset] = {"sha256": MAT_HASHES[dataset], "n": offset}
        fixed = f"{dataset}_fixed"
        partitions[fixed] = {"dataset": dataset, "indices": indices}
        for seed in range(5):
            jobs.append({"partition": fixed, "seed": seed, "evaluate": ["full", *THRESHOLDS]})
        if dataset != "abide":
            continue
        sites = all_phenotypes[:, 1].astype(int)
        unique, counts = np.unique(sites, return_counts=True)
        held_out, development = [site for site, _ in sorted(zip(unique.tolist(), counts.tolist()),
                                                           key=lambda row: (-row[1], row[0]))][:2]
        if (held_out, development) != (11, 5):
            raise ValueError("Expected the archived NYU and UM site codes")
        site_indices = {"train": np.flatnonzero(~np.isin(sites, [held_out, development])).tolist(),
                        "valid": np.flatnonzero(sites == development).tolist(),
                        "test": np.flatnonzero(sites == held_out).tolist()}
        partitions["abide_nyu"] = {"dataset": dataset, "indices": site_indices,
                                    "test_site": "NYU", "reserved_development_site": "UM",
                                    "site_codes": {group: np.unique(sites[ids]).tolist() for group, ids in site_indices.items()}}
        for seed in range(5):
            jobs.append({"partition": "abide_nyu", "seed": seed, "evaluate": ["full", "minmax"]})
        pooled_labels = np.concatenate([labels[group] for group in GROUPS])
        for split_seed in range(5):
            remaining, test = train_test_split(np.arange(offset), test_size=len(indices["test"]),
                                               stratify=pooled_labels, random_state=split_seed)
            train, valid = train_test_split(remaining, test_size=len(indices["valid"]),
                                            stratify=pooled_labels[remaining], random_state=1000 + split_seed)
            partition = f"abide_holdout_{split_seed}"
            partitions[partition] = {"dataset": dataset, "test_split_seed": split_seed,
                                     "development_split_seed": 1000 + split_seed,
                                     "indices": {group: np.sort(ids).tolist()
                                                 for group, ids in zip(GROUPS, (train, valid, test))}}
            jobs.append({"partition": partition, "seed": 1, "evaluate": []})

    return {"pool_order": "stored MAT train, valid, test concatenated; zero-based indices",
            "datasets": datasets, "partitions": partitions, "jobs": jobs,
            "training": {**TRAINING, "shuffle": True, "checkpoint": "last epoch"},
            "selection": {"data": "train", "method": "ROI correlation attenuation", "alpha": ALPHA,
                          "sign": "original cross entropy minus perturbed cross entropy", "strict_thresholds": THRESHOLDS},
            "environment": environment,
            "code_sha256": {path: sha256(Path(__file__).parent / path) for path in (
                "run_review_experiments.py", "jbhi_revision/common.py", "efficient_kan/kan.py")}}


def new_model(rois):
    return KAN([len(rois)**2, *TRAINING["hidden"], 2], grid_size=TRAINING["grid_size"],
               spline_order=TRAINING["spline_order"])


def archived_selector(archive_root, job, split, device):
    partition, seed = job["partition"], job["seed"]
    advisor = archive_root / "Advisor_revision_20261026"
    if partition.endswith("_fixed"):
        directory = advisor / "experiments/runs" / f"unified_{split['dataset']}_kan_s{seed}"
    elif partition == "abide_nyu":
        directory = advisor / "site_resumed/experiments/runs" / f"site_kan_s{seed}"
    else:
        directory = archive_root / "Random_split_comparison_20261027/runs" / f"partition_{split['test_split_seed']}_kan_fixed"
    record = read_json(directory / "complete.json")
    for name in ("model.pt", "config.json"):
        if sha256(directory / name) != record["files"][name]:
            raise ValueError(f"Archive checksum mismatch in {directory / name}")
    config = read_json(directory / "config.json")
    if config["indices"] != split["indices"] or config["rois"] != FULL.tolist():
        raise ValueError(f"Archive partition or ROI mismatch in {directory}")
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    settings = checkpoint["config"]
    if (checkpoint["epoch"] != TRAINING["epochs"] or settings["seed"] != seed or
            settings["hidden"] != TRAINING["hidden"] or settings["batch_size"] != TRAINING["batch_size"] or
            settings["lr"] != TRAINING["learning_rate"] or settings["weight_decay"] != TRAINING["weight_decay"] or
            settings.get("lr_gamma", settings.get("gamma")) != TRAINING["lr_gamma"] or not settings["shuffle"] or
            settings.get("grid_epochs") or settings.get("regularization", 0) != 0):
        raise ValueError(f"Archive training configuration mismatch in {directory}")
    model = new_model(FULL)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint.get("state")))
    provenance = {"run": directory.relative_to(archive_root).as_posix(), "model_sha256": record["files"]["model.pt"],
                  "config_sha256": record["files"]["config.json"]}
    return model.to(device).eval(), provenance


def correlation_audit(matrices):
    matrices = matrices.astype(np.float64)
    symmetric = (matrices + matrices.transpose(0, 2, 1)) / 2
    minimum = float(np.linalg.eigvalsh(symmetric)[:, 0].min())
    asymmetry = float(np.abs(matrices - matrices.transpose(0, 2, 1)).max())
    diagonal = float(np.abs(np.diagonal(matrices, axis1=1, axis2=2) - 1).max())
    maximum = float(np.abs(matrices).max())
    if minimum < -1e-5 or asymmetry > 1e-5 or diagonal > 1e-5 or maximum > 1 + 1e-5:
        raise ValueError("Expected symmetric, unit-diagonal, positive semidefinite correlation matrices")
    return {"minimum_eigenvalue": minimum, "maximum_asymmetry": asymmetry,
            "maximum_diagonal_error": diagonal, "maximum_absolute_entry": maximum}


def score_rois(model, train, targets):
    """Perturb raw correlations by D X D + I - D^2 before per-subject LayerNorm."""
    audit = correlation_audit(train)
    baseline = log_loss(targets, predict(model, train, FULL), labels=[0, 1])
    losses, audit_sample = [], []
    for roi in FULL:
        perturbed = train.copy()
        perturbed[:, roi, :] *= ALPHA
        perturbed[:, :, roi] *= ALPHA
        perturbed[:, roi, roi] += 1 - ALPHA**2
        audit_sample.append(correlation_audit(perturbed[:1]))
        losses.append(log_loss(targets, predict(model, perturbed, FULL), labels=[0, 1]))
    scores = baseline - np.array(losses)
    if not np.isfinite(scores).all() or np.ptp(scores) == 0:
        raise ValueError("ROI score transformation requires varying, finite scores")
    values = {"direct": scores, "minmax": (scores - scores.min()) / np.ptp(scores),
              "zscore": (scores - scores.mean()) / scores.std(ddof=0)}
    selections = {}
    for mode, threshold in THRESHOLDS.items():
        chosen = np.flatnonzero(values[mode] < threshold)
        chosen = chosen[np.argsort(scores[chosen], kind="stable")]
        selections[mode] = {"indices": chosen.tolist(), "transformed_scores": values[mode].tolist(),
                            "retained_nonnegative_raw_scores": int(np.count_nonzero(scores[chosen] >= 0))}
    return {"baseline_cross_entropy": float(baseline), "perturbed_cross_entropies": losses,
            "scores": scores.tolist(), "selections": selections, "original_structure": audit,
            "perturbed_structure_sample": {"subjects": 1, "roi_interventions": len(FULL),
                                            "minimum_eigenvalue": min(row["minimum_eigenvalue"] for row in audit_sample),
                                            **{key: max(row[key] for row in audit_sample) for key in (
                                                "maximum_asymmetry", "maximum_diagonal_error", "maximum_absolute_entry")}}}


def save_model(directory, name, model, rois):
    path = directory / f"{name}.pt"
    torch.save({"state": {key: value.cpu() for key, value in model.state_dict().items()},
                "rois": list(map(int, rois)), "epoch": TRAINING["epochs"]}, path)


def train_job(directory, job, split, train, targets, device, archive_root):
    if (directory / "complete.json").exists():
        finished = read_json(directory / "complete.json")
        for name, digest in finished["files"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Changed training artifact: {directory / name}")
        return finished
    directory.mkdir(exist_ok=True)
    seed = job["seed"]
    if archive_root is None:
        full, history, _ = fit_kan(train, targets, FULL, seed, device, shuffle=True)
        write_json(directory / "full_history.json", history)
        selector = {"run": "fresh training-only fit"}
    else:
        full, selector = archived_selector(archive_root, job, split, device)
    if "full" in job["evaluate"]:
        save_model(directory, "full", full, FULL)
    selection = score_rois(full, train, targets)
    selection.update({"partition": job["partition"], "seed": seed, "score_cohort": "train",
                      "score_pool_indices": split["indices"]["train"], "selector": selector})
    write_json(directory / "selection.json", selection)
    del full
    fitted = []
    for mode in job["evaluate"]:
        if mode != "full":
            rois = np.asarray(selection["selections"][mode]["indices"], dtype=int)
            if not len(rois):
                continue
            print(f"  {mode}: {len(rois)} ROIs", flush=True)
            reduced, history, _ = fit_kan(train, targets, rois, seed, device, shuffle=True)
            save_model(directory, mode, reduced, rois)
            write_json(directory / f"{mode}_history.json", history)
            del reduced
        fitted.append(mode)
    result = {"fitted_models": fitted, "files": {path.name: sha256(path) for path in directory.iterdir()
                                               if path.suffix in (".pt", ".json")}}
    write_json(directory / "complete.json", result)
    return result


def run(data_paths, output, device, archive_root):
    environment = configure_runtime(device)
    inputs = {dataset: load_groups(path, dataset) for dataset, path in data_paths.items()}
    plan = make_plan(inputs, environment)
    plan["selector_source"] = "verified frozen selectors" if archive_root else "fresh fits"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        if read_json(output / "plan.json") != plan:
            raise ValueError("Frozen plan mismatch. Use a new output directory.")
    else:
        write_json(output / "plan.json", plan)
    pools = {dataset: (np.concatenate([matrices[group] for group in GROUPS]),
                       np.concatenate([labels[group] for group in GROUPS]))
             for dataset, (matrices, labels, _) in inputs.items()}
    del inputs
    completed = {}
    for job in plan["jobs"]:
        split = plan["partitions"][job["partition"]]
        matrices, labels = pools[split["dataset"]]
        train = np.asarray(split["indices"]["train"])
        run_id = f"{job['partition']}_seed_{job['seed']}"
        print(f"Training and scoring {run_id}", flush=True)
        completed[run_id] = train_job(output / run_id, job, split, matrices[train], labels[train], device, archive_root)
    write_json(output / "frozen_before_test.json", {"plan_sha256": sha256(output / "plan.json"), "runs": completed})
    for job in plan["jobs"]:
        split = plan["partitions"][job["partition"]]
        run_id = f"{job['partition']}_seed_{job['seed']}"
        directory = output / run_id
        matrices, labels = pools[split["dataset"]]
        for mode in completed[run_id]["fitted_models"]:
            checkpoint = torch.load(directory / f"{mode}.pt", map_location="cpu", weights_only=True)
            rois = np.asarray(checkpoint["rois"], dtype=int)
            model = new_model(rois)
            model.load_state_dict(checkpoint["state"])
            test = np.asarray(split["indices"]["test"])
            probabilities = predict(model.to(device), matrices[test], rois)
            write_predictions(directory / f"{mode}_test.csv", test, labels[test], probabilities)
            del model, checkpoint
    print("Finished frozen test evaluation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abide", type=Path, required=True)
    parser.add_argument("--adhd", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--archive-root", type=Path, help="Reuse checked full-KAN weights from local frozen runs")
    options = parser.parse_args()
    with threadpool_limits(limits=TRAINING["cpu_threads"], user_api="blas"):
        run({"abide": options.abide, "adhd": options.adhd}, options.out, options.device, options.archive_root)


if __name__ == "__main__":
    main()
