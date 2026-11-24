"""Verify the kept predictions and ROI scores, then build the review tables."""
import argparse
import csv
from itertools import combinations, product
from pathlib import Path

import numpy as np

from jbhi_revision.common import METRICS, metrics, read_predictions, sha256, summarize, write_json
from run_review_experiments import THRESHOLDS, read_json


def checked_prediction(path, expected_indices, known_labels):
    indices, labels, probabilities = read_predictions(path)
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"Incorrect evaluation split: {path}")
    if known_labels is not None and not np.array_equal(labels, known_labels):
        raise ValueError(f"Incorrect evaluation labels: {path}")
    return labels, metrics(labels, probabilities.astype(np.float32))


def checked_selections(selection):
    raw = np.asarray(selection["scores"])
    reconstructed = selection["baseline_cross_entropy"] - np.asarray(selection["perturbed_cross_entropies"])
    np.testing.assert_array_equal(raw, reconstructed)
    scaled = {"direct": raw, "minmax": (raw - raw.min()) / np.ptp(raw),
              "zscore": (raw - raw.mean()) / raw.std(ddof=0)}
    for mode, threshold in THRESHOLDS.items():
        order = np.argsort(raw, kind="stable")
        expected = order[scaled[mode][order] < threshold]
        actual = selection["selections"][mode]
        np.testing.assert_array_equal(expected, actual["indices"])
        np.testing.assert_array_equal(scaled[mode], actual["transformed_scores"])
    return {mode: set(selection["selections"][mode]["indices"]) for mode in THRESHOLDS}


def paired_tests(rows):
    output = []
    for partition in ("abide_fixed", "adhd_fixed", "abide_nyu"):
        models = {}
        for row in rows:
            if row["partition"] == partition:
                models.setdefault(row["model"], {})[row["seed"]] = row
        for metric in METRICS:
            family = []
            for mode in THRESHOLDS:
                for baseline in ("full", "mlp", "brainnetcnn", "svm"):
                    if mode not in models or baseline not in models or set(models[mode]) != set(models[baseline]):
                        continue
                    seeds = sorted(models[mode])
                    diffs = np.array([models[mode][seed][metric] - models[baseline][seed][metric] for seed in seeds])
                    if len(diffs) != 5:
                        continue
                    observed = abs(diffs.sum())
                    sign_sums = [abs(np.dot(signs, diffs)) for signs in product((-1, 1), repeat=len(diffs))]
                    p_value = float(np.mean(np.array(sign_sums) >= observed - 1e-12))
                    family.append({"partition": partition, "metric": metric, "mode": mode, "baseline": baseline,
                                   "seeds": seeds, "paired_differences": diffs.tolist(), "mean_difference": float(diffs.mean()),
                                   "sd_difference": float(diffs.std(ddof=1)), "p": p_value})
            previous = 0.0
            for rank, entry in enumerate(sorted(family, key=lambda row: row["p"])):
                previous = max(previous, min(1.0, entry["p"] * (len(family) - rank)))
                entry.update(p_holm=previous, family_size=len(family))
            output.extend(family)
    return output


def summarize_stability(selections):
    records = []
    for group, group_rows in selections.items():
        for mode in THRESHOLDS:
            sets = {key: values[mode] for key, values in group_rows.items()}
            pairs = [{"first": first, "second": second, "jaccard": len(left & right) / len(left | right)}
                     for (first, left), (second, right) in combinations(sets.items(), 2) if left | right]
            overlaps = [row["jaccard"] for row in pairs]
            sizes = list(map(len, sets.values()))
            records.append({"group": group, "mode": mode, "n_runs": len(sets), "roi_counts": sizes,
                            "mean_roi_count": float(np.mean(sizes)), "pairs": pairs,
                            "mean_jaccard": float(np.mean(overlaps)) if overlaps else None,
                            "sd_jaccard": float(np.std(overlaps, ddof=1)) if len(overlaps) > 1 else None,
                            "selection_frequency": [sum(roi in ids for ids in sets.values()) / len(sets)
                                                    for roi in range(90)]})
    return records


def export(out, unified, legacy, legacy_reference):
    plan = read_json(out / "plan.json")
    frozen = read_json(out / "frozen_before_test.json")
    if sha256(out / "plan.json") != frozen["plan_sha256"]:
        raise ValueError("Plan changed after training")
    rows, selections, labels_by_split, zero_roi, audits = [], {}, {}, [], []
    for job in plan["jobs"]:
        partition, seed = job["partition"], job["seed"]
        split = plan["partitions"][partition]
        run_id = f"{partition}_seed_{seed}"
        run_dir = out / run_id
        record = frozen["runs"][run_id]
        selection_path = run_dir / "selection.json"
        if sha256(selection_path) != record["files"]["selection.json"]:
            raise ValueError(f"Selection changed after training: {run_id}")
        selection = read_json(selection_path)
        if selection["score_pool_indices"] != split["indices"]["train"] or selection["score_cohort"] != "train":
            raise ValueError(f"ROI scoring cohort mismatch: {run_id}")
        if any(set(left) & set(right) for left, right in combinations(split["indices"].values(), 2)):
            raise ValueError(f"Overlapping roles in {partition}")
        group = "abide_holdouts" if partition.startswith("abide_holdout_") else partition
        selections.setdefault(group, {})[run_id] = checked_selections(selection)
        audits.append({"run": run_id, "source": selection["original_structure"],
                       "perturbed": selection["perturbed_structure_sample"],
                       "negative_raw_scores": int(np.count_nonzero(np.array(selection["scores"]) < 0)),
                       "retained_nonnegative_scores": {mode: selection["selections"][mode]["retained_nonnegative_raw_scores"]
                                                      for mode in THRESHOLDS}})
        for mode in job["evaluate"]:
            count = 90 if mode == "full" else len(selection["selections"][mode]["indices"])
            if not count:
                zero_roi.append({"partition": partition, "seed": seed, "mode": mode})
                continue
            path = run_dir / f"{mode}_test.csv"
            if mode not in record["fitted_models"]:
                raise ValueError(f"Missing planned classifier: {run_id}/{mode}")
            labels, computed = checked_prediction(path, split["indices"]["test"], labels_by_split.get(partition))
            labels_by_split[partition] = labels
            rows.append({"partition": partition, "model": mode, "seed": seed, "n_test": len(labels), "n_roi": count,
                         **computed, "origin": "current train-only study", "prediction_sha256": sha256(path),
                         "prediction": path.relative_to(out).as_posix()})
    if unified is not None:
        for archived in read_json(unified / "unified_runs.json"):
            job = archived["job"]
            partition = f"{job['dataset']}_fixed"
            path = unified / f"{job['id']}_test.csv"
            if sha256(path) != archived["prediction_sha256"]:
                raise ValueError(f"Archived prediction checksum mismatch: {path}")
            labels, computed = checked_prediction(path, plan["partitions"][partition]["indices"]["test"], labels_by_split[partition])
            for key in METRICS:
                np.testing.assert_allclose(computed[key], archived["metrics"][key], rtol=0, atol=1e-12)
            if job["kind"] == "kan":
                corresponding = next(row for row in rows if row["partition"] == partition and
                                     row["seed"] == job["seed"] and row["model"] == "full")
                for key in METRICS:
                    np.testing.assert_allclose(computed[key], corresponding[key], rtol=0, atol=1e-12)
            else:
                rows.append({"partition": partition, "model": job["kind"], "seed": job["seed"], "n_test": len(labels),
                             "n_roi": 90, **computed, "origin": "Advisor unified baseline",
                             "prediction_sha256": sha256(path), "prediction": f"unified/{path.name}"})
    legacy_audit = None
    if legacy is not None:
        legacy_rows = read_json(legacy / "results.json")
        archived_rows = [row for row in read_json(legacy_reference / "legacy_runs.json")
                         if row["job"]["id"].startswith("legacy_minmax_shuffle0_") and row["cohort"] == "test"]
        if not legacy_rows["matched"] or len(archived_rows) != 5:
            raise ValueError("Incomplete historical verification")
        max_probability_difference = 0.0
        for seed, archived in enumerate(sorted(archived_rows, key=lambda row: row["job"]["seed"])):
            previous = legacy_reference / f"{archived['job']['id']}_test.csv"
            current = legacy / f"seed_{seed}_test.csv"
            if sha256(previous) != archived["prediction_sha256"]:
                raise ValueError(f"Archived historical prediction checksum mismatch: {previous}")
            new_ids, new_labels, new_probs = read_predictions(current)
            old_ids, old_labels, old_probs = read_predictions(previous)
            np.testing.assert_array_equal(new_ids, old_ids)
            np.testing.assert_array_equal(new_labels, old_labels)
            np.testing.assert_array_equal(new_probs.argmax(1), old_probs.argmax(1))
            for key in METRICS:
                np.testing.assert_allclose(legacy_rows["runs"][seed][key], archived["metrics"][key], rtol=0, atol=1e-12)
            max_probability_difference = max(max_probability_difference, float(np.abs(new_probs - old_probs).max()))
        legacy_audit = {"n_runs": len(archived_rows), "prediction_classes_identical": True,
                        "all_five_metrics_identical": True, "maximum_probability_difference": max_probability_difference}
    models = {}
    for row in rows:
        models.setdefault(row["partition"], {}).setdefault(row["model"], []).append(row)
    aggregate = {partition: {model: {"n_runs": len(group), "roi_mean": float(np.mean([row["n_roi"] for row in group])),
                                     "roi_sd": float(np.std([row["n_roi"] for row in group], ddof=1)) if len(group) > 1 else None,
                                     "metrics": summarize(group)} for model, group in model_rows.items()}
                 for partition, model_rows in models.items()}
    stability = summarize_stability(selections)
    with (out / "per_run.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "metrics_and_stability.json", {
        "performance": aggregate, "stability": stability, "empty_selections": zero_roi,
        "score_and_structure_audit": audits, "historical_reproduction": legacy_audit,
        "verified_test_predictions": len(rows), "units": "metric values 0 to 1; sample SD",
    })
    write_json(out / "paired_statistics.json", {
        "test": "two-sided exact paired sign-flip test, seed-level differences on the same cohort",
        "adjustment": "Holm across model comparisons for each partition and metric",
        "scope": "seed variability conditional on the held-out cohort",
        "comparisons": paired_tests(rows),
    })
    write_tables(out / "tables.md", aggregate, stability, zero_roi)
    print(f"Verified {len(rows)} full-cohort prediction files and {len(audits)} train-only ROI score sets.")


def write_tables(path, aggregate, stability, empty):
    lines = ["# BrainKAN reviewer experiments", "",
             "Performance is mean ± sample SD across training seeds 0–4. Precision, recall and F1 are macro averages.",
             "The fixed-cohort baselines use the same test subjects as the corresponding KAN results.",
             "The held-out site is NYU (171 subjects). UM (113 subjects) is reserved and 598 subjects enter training.", ""]
    names = {"abide_fixed": "ABIDE fixed split", "adhd_fixed": "ADHD-200 fixed split", "abide_nyu": "ABIDE held-out NYU",
             "abide_holdouts": "ABIDE five random holdouts"}
    for partition, models in sorted(aggregate.items()):
        lines += [f"## {names[partition]}", "", "| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model, values in models.items():
            scores = [f"{100 * values['metrics'][key]['mean']:.2f} ± {100 * values['metrics'][key]['sd']:.2f}" for key in METRICS]
            lines.append(f"| {model} | {values['n_runs']} | {values['roi_mean']:.2f} ± {values['roi_sd']:.2f} | " + " | ".join(scores) + " |")
        lines.append("")
    if empty:
        lines += [f"The strict threshold returned an empty ROI set in {len(empty)} planned fits. These runs have no classifier metrics.", ""]
    lines += ["## ROI selection stability", "", "| Comparison | Mode | Runs | Mean ROI count | Mean pairwise Jaccard ± SD |",
              "|---|---|---:|---:|---:|"]
    for row in stability:
        overlap = f"{row['mean_jaccard']:.3f} ± {row['sd_jaccard']:.3f}" if row['sd_jaccard'] is not None else "—"
        lines.append(f"| {names[row['group']]} | {row['mode']} | {row['n_runs']} | {row['mean_roi_count']:.2f} | {overlap} |")
    lines += ["", "The fixed cohort and held-out site rows vary training seeds. The random holdout rows vary the data partition at seed 1.",
              "Pairwise Jaccard comparisons share runs and are descriptive."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unified", type=Path, help="Advisor unified final_test directory")
    parser.add_argument("--legacy", type=Path, help="The current historical replay directory")
    parser.add_argument("--legacy-reference", type=Path, help="Advisor legacy final_test directory")
    args = parser.parse_args()
    if bool(args.legacy) != bool(args.legacy_reference):
        parser.error("--legacy and --legacy-reference are required together")
    export(args.out, args.unified, args.legacy, args.legacy_reference)


if __name__ == "__main__":
    main()
