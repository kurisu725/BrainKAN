"""Recompute matched five-holdout metrics and paired comparisons from saved predictions."""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from jbhi_revision.common import METRICS, read_predictions, sha256, summarize, write_json
from run_review_experiments import read_json
from summarize_review import checked_selections, summarize_stability


BASELINES = {
    "abide": (
        ("Baseline_rerun_20261106", "fc", ("braingnn", "braingb", "brainnettf")),
        ("Random_split_comparison_20261027", "csv", ("brainnetcnn",)),
        ("Remaining_baselines_20261108", "final", ("fbnetgen", "brainnetgnn", "res_dae")),
    ),
    "adhd": (
        ("ADHD_baseline_comparison_20261111/fc_baselines", "fc", ("braingnn", "braingb", "brainnetcnn", "brainnettf")),
        ("ADHD_baseline_comparison_20261111/res_dae", "final", ("res_dae",)),
    ),
}

MODEL_NAMES = {
    "minmax": "BrainKAN Min-max", "braingnn": "BrainGNN (FC adaptation)", "braingb": "BrainGB",
    "brainnetcnn": "BrainNetCNN", "fbnetgen": "FBNetGen", "brainnetgnn": "BrainNetGNN",
    "brainnettf": "BrainNetTF", "res_dae": "Res-DAE (paper implementation)",
}


def classification_metrics(labels, predicted, positive_probability):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], average="macro", zero_division=0
    )
    return {"accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "auc": float(roc_auc_score(labels, positive_probability))}


def check_baseline_plan(plan, partition, expected, dataset_hash):
    dataset = plan["dataset"]
    if dataset.get("mat_sha256", dataset.get("sha256")) != dataset_hash or plan["training_seed"] != 1:
        raise ValueError("Mismatched baseline data or training seed")
    archived = plan["splits"]
    split = archived[str(partition)] if isinstance(archived, dict) else archived[partition]
    indices = {"valid" if role == "development" else role: values for role, values in split["indices"].items()}
    if indices != expected:
        raise ValueError(f"Baseline and BrainKAN assignments differ for holdout {partition}")


def baseline_prediction(root, partition, model, layout):
    if layout == "fc":
        run_dir = root / "runs" / f"{model}_partition_{partition}"
        path = run_dir / "predictions.npz"
        record = read_json(run_dir / "final.json")
        seal = "development_frozen.json"
    else:
        stem = root / "final_test" / f"partition_{partition}_{model}"
        path = stem.with_suffix(".csv" if layout == "csv" else ".npz")
        record = read_json(stem.with_suffix(".json"))
        seal = "frozen.json" if layout == "csv" else "development_frozen.json"
    digest = sha256(path)
    if digest != record.get("prediction_sha256", record.get("predictions_sha256")):
        raise ValueError(f"Changed baseline prediction file: {path}")
    expected_seal = record.get("development_frozen_sha256", record.get("frozen_sha256"))
    if sha256(root / seal) != expected_seal:
        raise ValueError(f"Changed baseline training seal: {path}")
    if layout == "csv":
        data = np.genfromtxt(path, delimiter=",", names=True)
        indices, labels, probability = data["pool_index"], data["label"], data["prob_1"]
        predicted = np.column_stack([data["prob_0"], probability]).argmax(axis=1)
        np.testing.assert_array_equal(data["pred"], predicted)
        fields = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "auc")
    else:
        with np.load(path, allow_pickle=False) as data:
            if layout == "fc":
                indices, labels = data["pool_indices"], data["labels"]
                probabilities = data["probabilities"]
                probability, predicted = probabilities[:, 1], probabilities.argmax(axis=1)
                fields = ("acc", "precision", "recall", "f1", "auc")
            else:
                indices, labels, probability = data["indices"], data["y"], data["probability"]
                predicted = (probability >= record["threshold"]).astype(int)
                np.testing.assert_array_equal(data["prediction"], predicted)
                fields = ("ACC", "P", "R", "F1", "AUC")
    if not np.all(np.isfinite(probability) & (probability >= 0) & (probability <= 1)):
        raise ValueError(f"Invalid baseline probabilities: {path}")
    computed = classification_metrics(labels, predicted, probability)
    np.testing.assert_allclose(list(computed.values()), [record["metrics"][field] for field in fields],
                               atol=1e-12, rtol=0)
    return indices, labels, computed, path, digest


def paired_statistics(models, n_train, n_test):
    results = []
    for metric in METRICS:
        family = []
        reference = models["minmax"]
        for name, rows in models.items():
            if name == "minmax":
                continue
            np.testing.assert_array_equal([row["holdout"] for row in rows], [row["holdout"] for row in reference])
            differences = np.array([left[metric] - right[metric] for left, right in zip(reference, rows)])
            mean, sd = differences.mean(), differences.std(ddof=1)
            corrected_se = sd * np.sqrt(1 / len(differences) + n_test / n_train)
            p_value = float(2 * t.sf(abs(mean) / corrected_se, df=len(differences) - 1)) if corrected_se else 1.0
            family.append({"metric": metric, "baseline": name, "paired_differences": differences.tolist(),
                           "mean_difference": float(mean), "sample_sd": float(sd),
                           "corrected_standard_error": float(corrected_se), "p": p_value})
        adjusted = 0.0
        for rank, entry in enumerate(sorted(family, key=lambda row: row["p"])):
            adjusted = max(adjusted, min(1.0, entry["p"] * (len(family) - rank)))
            entry.update(p_holm=adjusted, family_size=len(family))
        results.extend(family)
    return results


def export(out, baseline_root):
    plan = read_json(out / "plan.json")
    frozen = read_json(out / "frozen_before_test.json")
    if sha256(out / "plan.json") != frozen["plan_sha256"]:
        raise ValueError("Holdout plan changed after training")
    per_run, selections, stats, summary = [], {}, {}, {}
    table = ["# Matched five-holdout comparison", "",
             "Five diagnosis-stratified random holdouts per dataset with model seed 1. The cohorts overlap across holdouts.",
             "Mean ± sample SD over the five holdouts. P, R, and F1 are macro averages.",
             "The archived baseline adaptations share subject assignments and use their documented training schedules.",
             "Paired tests use a corrected resampled t statistic and Holm adjustment within each dataset and metric.", ""]
    for dataset, archives in BASELINES.items():
        models = {}
        for partition in range(5):
            name = f"{dataset}_holdout_{partition}"
            directory = out / f"{name}_seed_1"
            reference = frozen["runs"][directory.name]
            split = plan["partitions"][name]["indices"]
            if (sum(map(len, split.values())) != plan["datasets"][dataset]["n"] or
                    any(set(left) & set(right) for left, right in combinations(split.values(), 2))):
                raise ValueError(f"Invalid holdout assignments: {name}")
            selection_path = directory / "selection.json"
            if sha256(selection_path) != reference["files"]["selection.json"]:
                raise ValueError(f"Changed held-out ROI selections: {name}")
            selection = read_json(selection_path)
            if (selection["score_cohort"] != "train" or selection["score_pool_indices"] != split["train"] or
                    selection["seed"] != 1 or selection["partition"] != name):
                raise ValueError(f"Incorrect ROI scoring cohort: {name}")
            sets = checked_selections(selection)
            selections.setdefault(dataset, {})[name] = sets
            path = directory / "minmax_test.csv"
            indices, labels, probabilities = read_predictions(path)
            np.testing.assert_array_equal(indices, split["test"])
            matches = [{"model": "minmax", "n_roi": len(sets["minmax"]),
                        **classification_metrics(labels, probabilities.argmax(axis=1), probabilities[:, 1]),
                        "prediction": path.relative_to(out).as_posix(), "prediction_source": "holdouts",
                        "prediction_sha256": sha256(path)}]
            for source, layout, model_names in archives:
                root = baseline_root / source
                check_baseline_plan(read_json(root / "plan.json"), partition, split, plan["datasets"][dataset]["sha256"])
                for model in model_names:
                    archive_ids, archive_labels, scores, prediction_file, digest = baseline_prediction(
                        root, partition, model, layout
                    )
                    np.testing.assert_array_equal(indices, archive_ids)
                    np.testing.assert_array_equal(labels, archive_labels)
                    matches.append({"model": model, "n_roi": 90, **scores,
                                    "prediction": prediction_file.relative_to(baseline_root).as_posix(),
                                    "prediction_source": "baseline_reference", "prediction_sha256": digest})
            for record in matches:
                row = {"dataset": dataset, "holdout": partition, "seed": 1, "n_test": len(indices), **record}
                models.setdefault(row["model"], []).append(row)
                per_run.append(row)
        stats[dataset] = paired_statistics(models, len(split["train"]), len(split["test"]))
        summary[dataset] = {model: {"n_runs": len(rows), "metrics": summarize(rows),
                                   "roi_counts": [row["n_roi"] for row in rows]} for model, rows in models.items()}
        table.extend([f"## {'ABIDE' if dataset == 'abide' else 'ADHD-200'}", "",
                      "| Model | Mean ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for model, result in summary[dataset].items():
            cells = [f"{100 * result['metrics'][metric]['mean']:.2f} ± {100 * result['metrics'][metric]['sd']:.2f}"
                     for metric in METRICS]
            table.append(f"| {MODEL_NAMES[model]} | {np.mean(result['roi_counts']):.2f} | " + " | ".join(cells) + " |")
        table.append("")
    stability = summarize_stability(selections)
    table.extend(["## Min-max selection stability across holdouts", "",
                  "| Dataset | Mean ROI count | Mean pairwise Jaccard ± sample SD |", "|---|---:|---:|"])
    for row in stability:
        if row["mode"] == "minmax":
            table.append(f"| {row['group']} | {row['mean_roi_count']:.2f} | "
                         f"{row['mean_jaccard']:.3f} ± {row['sd_jaccard']:.3f} |")
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    write_json(out / "metrics_and_stability.json", {"test_prediction_files": len(per_run),
                                                   "metrics": summary, "stability": stability})
    write_json(out / "paired_statistics.json", {"method": "two-sided corrected resampled t test",
        "standard_error": "sample SD times sqrt(1/n_holdouts + n_test/n_train)",
        "adjustment": "Holm over baselines for each dataset and metric", "comparisons": stats})
    (out / "tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(f"Verified {len(per_run)} matched holdout prediction files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    args = parser.parse_args()
    export(args.out, args.baseline_root)


if __name__ == "__main__":
    main()
