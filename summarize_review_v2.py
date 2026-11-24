"""Summarise the training-only development experiments (review_v2 output)."""
import argparse
from pathlib import Path

from run_review_development import MODES
from run_review_experiments import read_json


HEADERS = {"abide_fixed": "ABIDE fixed split",
           "adhd_fixed": "ADHD-200 fixed split",
           "abide_nyu": "ABIDE held-out NYU"}


def summarise(directory):
    results = read_json(directory / "results.json")
    summary = results["summary"]
    plan = read_json(directory / "plan.json")
    lines = ["# BrainKAN training-only development experiments (fixed cohorts + held-out NYU)", ""]
    table_line = plan["training"]["checkpoint"]
    lines.append(f"Checkpoint selection: {table_line}. Means are across training seeds 0-4.")
    lines.append("")
    for partition in ("abide_fixed", "adhd_fixed", "abide_nyu"):
        lines += [f"## {HEADERS[partition]}", "",
                  "| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, value in sorted(summary.items()):
            name, mode = key.split("/")
            if name != partition or mode not in MODES:
                continue
            cells = " | ".join(
                f"{100 * value['metrics'][metric]['mean']:.2f} ± {100 * value['metrics'][metric]['sd']:.2f}"
                for metric in ("accuracy", "precision", "recall", "f1", "auc"))
            roi_mean = sum(value["n_roi"]) / len(value["n_roi"])
            lines.append(f"| {mode} | {value['n_runs']} | {roi_mean:.2f} | {cells} |")
        lines.append("")
    output = directory / "tables_v2.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = summarise(args.out)
    for line in lines[:40]:
        print(line)


if __name__ == "__main__":
    main()
