"""Reproduce the historical ABIDE 64-ROI diagnostic from Advisor_revision_20261026."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jbhi_revision.common import (
    MAT_HASHES,
    METRICS,
    TRAINING,
    configure_runtime,
    fit_kan,
    load_groups,
    metrics,
    predict,
    read_predictions,
    sha256,
    summarize,
    write_json,
    write_predictions,
)


BASE_COMMIT = "66df2c7bab9cc490920687f07e6b7e4dd8081c37"
SEEDS = (0, 1, 2, 3, 4)
ROI_ORDER = np.array([
    85, 78, 86, 79, 82, 19, 62, 15, 65, 52, 45, 80, 3, 28, 54, 56, 89, 32,
    59, 42, 8, 55, 44, 31, 72, 60, 40, 9, 87, 81, 6, 53, 43, 46, 69, 66, 30,
    5, 34, 21, 68, 57, 88, 1, 83, 75, 14, 84, 39, 11, 70, 76, 16, 12, 50, 77,
    24, 18, 27, 63, 29, 20, 64, 35,
])
SPLIT = {"train": range(709), "valid": range(709, 798), "test": range(798, 882)}
EXPECTED_CORRECT_COUNTS = (66, 65, 66, 68, 65)
EXPECTED_ROUNDED = ((78.6, 1.5), (78.5, 1.5), (78.1, 1.5), (78.3, 1.5), (85.6, 0.4))


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["base_commit"] != BASE_COMMIT or manifest["mat_sha256"] != MAT_HASHES["abide"] or
            manifest["roi_indices_zero_based"] != ROI_ORDER.tolist() or manifest["seeds"] != list(SEEDS) or
            manifest["training"] != {**TRAINING, "shuffle": False, "checkpoint": "last epoch"}):
        raise ValueError("Frozen history configuration does not match the reference")
    rows = []
    reference_labels = None
    for seed in SEEDS:
        path = output / f"seed_{seed}_test.csv"
        indices, labels, probabilities = read_predictions(path)
        if not np.array_equal(indices, SPLIT["test"]) or (
                reference_labels is not None and not np.array_equal(labels, reference_labels)):
            raise ValueError(f"Inconsistent test rows for seed {seed}")
        reference_labels = labels
        rows.append({"seed": seed, "n_correct": int((probabilities.argmax(1) == labels).sum()),
                     "prediction_sha256": sha256(path), **metrics(labels, probabilities)})
    summary = summarize(rows)
    rounded = tuple((round(summary[key]["mean"] * 100, 1), round(summary[key]["sd"] * 100, 1))
                    for key in METRICS)
    matched = rounded == EXPECTED_ROUNDED and tuple(row["n_correct"] for row in rows) == EXPECTED_CORRECT_COUNTS
    payload = {"scope": "historical diagnostic; ROI provenance is not independent of test data",
               "matched": matched, "metrics": summary, "runs": rows}
    write_json(output / "results.json", payload)
    for key in METRICS:
        print(f"{key}: {summary[key]['mean']:.4%} +/- {summary[key]['sd']:.4%}", flush=True)
    if not matched:
        raise SystemExit("Results were saved, but the repeat did not match the historical reference.")


def run(data, output, device):
    environment = configure_runtime(device)
    matrices, labels, _ = load_groups(data, "abide", ("train",))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {
        "base_commit": BASE_COMMIT,
        "reference": "Advisor_revision_20261026/experiments/final_test/legacy_runs.json, legacy_minmax_shuffle0",
        "mat_sha256": MAT_HASHES["abide"],
        "seeds": list(SEEDS),
        "split_indices_in_stored_train_valid_test_order": {key: list(value) for key, value in SPLIT.items()},
        "roi_indices_zero_based": ROI_ORDER.tolist(),
        "training": {**TRAINING, "shuffle": False, "checkpoint": "last epoch"},
        "environment": environment,
        "code_sha256": {path: sha256(Path(__file__).parent / path) for path in (
            "run_legacy_78.py", "jbhi_revision/common.py", "efficient_kan/kan.py")},
    })
    for seed in SEEDS:
        print(f"Training historical 64-ROI KAN, seed {seed}", flush=True)
        model, history, epoch = fit_kan(matrices["train"], labels["train"], ROI_ORDER, seed, device, shuffle=False)
        torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()}, "epoch": epoch},
                   output / f"seed_{seed}.pt")
        write_json(output / f"seed_{seed}_history.json", history)
        del model
    test, targets, _ = load_groups(data, "abide", ("test",))
    from efficient_kan import KAN

    for seed in SEEDS:
        checkpoint = torch.load(output / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = KAN([4096, *TRAINING["hidden"], 2], grid_size=5, spline_order=3)
        model.load_state_dict(checkpoint["state_dict"])
        probabilities = predict(model.to(device), test["test"], ROI_ORDER)
        write_predictions(output / f"seed_{seed}_test.csv", SPLIT["test"], targets["test"], probabilities)
        del model, checkpoint
    verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run", help="Retrain five models and evaluate the fixed historical test cohort")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    check = commands.add_parser("verify", help="Recompute the manuscript metrics using exported probabilities")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.out)
    else:
        print("Historical diagnostic. The archived ROI scores are consistent with reuse of the legacy test cohort.", flush=True)
        run(args.data, args.out, args.device)


if __name__ == "__main__":
    main()
