"""Development-selected checkpoints for the three ROI modes of the reviewer experiments.

Each partition keeps its frozen subject assignments. ROI sets come from the paper's
training-only scorer with its three score transforms and fixed thresholds. The
training epoch is selected by development AUC (then log loss) from checkpoints saved
every five epochs. The test cohort is evaluated once per fitted model, after the
selections are sealed on disk.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from jbhi_revision.common import (
    TRAINING, configure_runtime, features, fit_kan, load_groups, metrics, predict_features, read_predictions,
    sha256, summarize, write_json, write_predictions,
)
from run_review_experiments import ALPHA, GROUPS, THRESHOLDS, new_model, read_json, score_rois


EPOCHS = TRAINING["epochs"]
CHECKPOINT_EVERY = 5
MODES = ("direct", "minmax", "zscore")
FIXED_SEED_PARTITIONS = ("abide_fixed", "adhd_fixed", "abide_nyu")


PARTITION_SEEDS = {name: list(range(5)) for name in FIXED_SEED_PARTITIONS}


def fit_mode(partition, seed):
    return f"{partition}_seed_{seed}"


def fit_with_checkpoints(train, targets, rois, seed, device):
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    model = new_model(rois).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(features(train, rois), torch.from_numpy(targets)),
                        batch_size=TRAINING["batch_size"], shuffle=True, num_workers=0, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"],
                                  weight_decay=TRAINING["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=TRAINING["lr_gamma"])
    criterion = nn.CrossEntropyLoss()
    checkpoints, history = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        history.append({"epoch": epoch, "training_loss": total / len(targets),
                        "lr": optimizer.param_groups[0]["lr"]})
        if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
            checkpoints.append((epoch, {name: value.detach().to("cpu", copy=True)
                                        for name, value in model.state_dict().items()}))
        scheduler.step()
    return model, checkpoints, history


def choose_checkpoint(model, checkpoints, development, rois, device):
    dev_x, dev_y = features(development[0], rois), development[1]
    candidates = []
    for epoch, state in checkpoints:
        model.load_state_dict(state)
        probabilities = predict_features(model.eval(), dev_x)
        candidates.append({"epoch": epoch, **metrics(dev_y, probabilities)})
    best = max(candidates, key=lambda row: (row["auc"], -row["log_loss"]))
    model.load_state_dict(checkpoints[[epoch for epoch, _ in checkpoints].index(best["epoch"])][1])
    return best, candidates, model.eval()


def planned_runs(args):
    requested = set(args.only.split(",")) if args.only else set()
    runs = [(partition, seed) for partition, seeds in PARTITION_SEEDS.items()
            for seed in seeds if not requested or partition in requested]
    return runs


def make_run_plan(args, inputs, environment):
    review_plan = read_json(args.review / "plan.json")
    partitions = {name: review_plan["partitions"][name] for name in PARTITION_SEEDS}
    plan = {"pool_order": review_plan["pool_order"],
            "datasets": review_plan["datasets"],
            "partitions": partitions,
            "jobs": [{"partition": partition, "seed": seed, "evaluate": list(MODES)}
                     for partition, seed in planned_runs(args)],
            "training": {**TRAINING, "shuffle": True,
                         "checkpoint": f"development AUC then log loss, checkpoints every {CHECKPOINT_EVERY} epochs"},
            "selection": {"data": "train", "method": "ROI correlation attenuation", "alpha": ALPHA,
                          "strict_thresholds": THRESHOLDS,
                          "development_cohort": "the reserved valid indices of each partition"},
            "environment": environment,
            "code_sha256": {path: sha256(Path(__file__).parent / path)
                            for path in ("run_review_development.py", "jbhi_revision/common.py",
                                         "efficient_kan/kan.py")}}
    return plan


def fit_run(directory, split, matrices, labels, seed, device):
    train, valid = np.asarray(split["indices"]["train"]), np.asarray(split["indices"]["valid"])
    directory.mkdir(exist_ok=True)
    if (directory / "sealed.json").exists():
        return read_json(directory / "sealed.json")
    print(f"Fitting {directory.name}", flush=True)
    if (directory / "selection.json").exists():
        selection = read_json(directory / "selection.json")
    else:
        full, _, _ = fit_kan(matrices[train], labels[train], np.arange(90), seed, device, shuffle=True)
        selection = score_rois(full, matrices[train], labels[train])
        selection.update({"partition": directory.name.rsplit("_seed_", 1)[0], "seed": seed,
                          "score_cohort": "train", "score_pool_indices": split["indices"]["train"],
                          "selector": {"run": "fresh training-only fit"}})
        write_json(directory / "selection.json", selection)
        del full
    fitted, development = {}, (matrices[valid], labels[valid])
    for mode in MODES:
        rois = np.asarray(selection["selections"][mode]["indices"], dtype=int)
        if not len(rois):
            fitted[mode] = {"n_roi": 0}
            continue
        model, checkpoints, history = fit_with_checkpoints(matrices[train], labels[train], rois, seed, device)
        best, candidates, model = choose_checkpoint(model, checkpoints, development, rois, device)
        torch.save({"state": model.state_dict(), "rois": rois.tolist(), "epoch": best["epoch"]},
                   directory / f"{mode}.pt")
        write_json(directory / f"{mode}_history.json", history)
        fitted[mode] = {"n_roi": len(rois), "epoch": best["epoch"], "development": best,
                        "candidates": candidates}
        del model, checkpoints
        print(f"  {mode}: {len(rois)} ROIs, epoch {best['epoch']}, development AUC {best['auc']:.3f}",
              flush=True)
    record = {"partition": directory.name.rsplit("_seed_", 1)[0], "seed": seed, "fitted": fitted,
              "files": {path.name: sha256(path) for path in directory.iterdir()
                        if path.suffix in (".pt", ".json") and path.name != "sealed.json"}}
    write_json(directory / "sealed.json", record)
    return record


def evaluate_run(directory, record, split, matrices, labels, device):
    for file_name, digest in record["files"].items():
        if sha256(directory / file_name) != digest:
            raise ValueError(f"Sealed artifact changed before test evaluation: {directory / file_name}")
    done = directory / "test_done.json"
    if done.exists():
        return read_json(directory / "per_run_rows.json")
    test = np.asarray(split["indices"]["test"])
    rows = []
    for mode in MODES:
        if not record["fitted"][mode]["n_roi"]:
            continue
        checkpoint = torch.load(directory / f"{mode}.pt", map_location="cpu", weights_only=True)
        rois = np.asarray(checkpoint["rois"], dtype=int)
        model = new_model(rois)
        model.load_state_dict(checkpoint["state"])
        probabilities = predict_features(model.to(device), features(matrices[test], rois))
        path = directory / f"{mode}_test.csv"
        write_predictions(path, test, labels[test], probabilities)
        _, known, _ = read_predictions(path)
        rows.append({"partition": record["partition"], "seed": record["seed"], "model": mode,
                     "n_test": len(test), "n_roi": len(rois), "epoch": checkpoint["epoch"],
                     **metrics(known, probabilities), "prediction_sha256": sha256(path)})
        del model, checkpoint
    write_json(directory / "per_run_rows.json", rows)
    write_json(done, {"n_models": len(record["fitted"])})
    return rows


def run(args):
    environment = configure_runtime(args.device)
    inputs = {name: load_groups(path, name) for name, path in (("abide", args.abide), ("adhd", args.adhd))}
    plan = make_run_plan(args, inputs, environment)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        if read_json(output / "plan.json") != plan:
            raise ValueError("Frozen plan mismatch. Use a new output directory.")
    else:
        write_json(output / "plan.json", plan)
    pools = {name: (np.concatenate([matrices[group] for group in GROUPS]),
                    np.concatenate([labels[group] for group in GROUPS]))
             for name, (matrices, labels, _) in inputs.items()}
    del inputs

    sealed = {}
    for partition, seed in planned_runs(args):
        split = plan["partitions"][partition]
        matrices, labels = pools[split["dataset"]]
        record = fit_run(output / fit_mode(partition, seed), split, matrices, labels, seed, args.device)
        sealed[fit_mode(partition, seed)] = record
    write_json(output / "sealed_before_test.json",
               {"plan_sha256": sha256(output / "plan.json"), "runs": sealed})

    rows = []
    for run_id, record in sealed.items():
        split = plan["partitions"][record["partition"]]
        matrices, labels = pools[split["dataset"]]
        rows.extend(evaluate_run(output / run_id, record, split, matrices, labels, args.device))
    results_path = output / "results.json"
    if results_path.exists():
        previous = read_json(results_path)
        for row in previous.get("per_run", []):
            key = (row["partition"], row["seed"], row["model"])
            if key not in {(entry["partition"], entry["seed"], entry["model"]) for entry in rows}:
                rows.append(row)
    groups = {}
    for row in rows:
        groups.setdefault((row["partition"], row["model"]), []).append(row)
    summary = {f"{partition}/{mode}": {"n_runs": len(group), "n_roi": [row["n_roi"] for row in group],
                                       "epochs": [row["epoch"] for row in group], "metrics": summarize(group)}
               for (partition, mode), group in groups.items()}
    write_json(results_path, {"selection_protocol": plan["selection"], "per_run": rows, "summary": summary})
    for key, values in summary.items():
        cells = " ".join(f"{metric}={100 * values['metrics'][metric]['mean']:.2f}±{100 * values['metrics'][metric]['sd']:.2f}"
                         for metric in ("accuracy", "auc"))
        print(f"{key}: ROIs={values['n_roi']} epochs={values['epochs']} {cells}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abide", type=Path, required=True)
    parser.add_argument("--adhd", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True, help="Existing review directory holding the fixed and NYU plan")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None, help="Comma-separated partition names to run; others are skipped")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    with threadpool_limits(limits=TRAINING["cpu_threads"], user_api="blas"):
        run(args)


if __name__ == "__main__":
    main()
