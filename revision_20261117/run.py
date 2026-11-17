"""Verify the published snapshot or run its unchanged native KAN training code."""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    manifest = read(ROOT / "MANIFEST.json")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Published file differs: " + entry["path"])
    provenance = read(ROOT / "SOURCE_PROVENANCE.json")
    for entry in provenance["byte_identical_python_files"]:
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Original training source differs: " + entry["path"])
    results = read(ROOT / "records/selected_results.json")
    assignments = read(ROOT / "records/selected_splits.json")["cases"]
    with (ROOT / "records/all_attempts.csv").open(encoding="utf-8-sig", newline="") as handle:
        attempts = list(csv.DictReader(handle))
    means = {}
    for study, n in [("abide", 882), ("adhd", 847)]:
        cases = [row for row in results["cases"] if row["study"] == study]
        assert len(cases) == 5
        assert len([row for row in attempts if row["study"] == study]) == results["attempted_original_recipe_divisions"][study]
        means[study] = {}
        for key in ["accuracy", "auc"]:
            values = [row["test"][key] for row in cases]
            means[study][key] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))
        for row in cases:
            split = next(item for item in assignments if (item["study"], item["selected_round"]) == (study, row["selected_round"]))
            indices = [i for role in ["train", "valid", "test"] for i in split["indices"][role]]
            assert sorted(indices) == list(range(n))
            assert len(row["rois"]) == row["retained_rois"] < 90
            assert row["configuration"]["roi_mass"] == (.90 if study == "abide" else .95)
            original = next(item for item in attempts if item["study"] == study and int(item["trial"]) == row["trial"])
            for key in ["accuracy", "auc"]:
                assert float(original["test_" + key]) == row["test"][key]
    return dict(passed=True, published_files=len(manifest["files"]),
                unchanged_training_source_files=len(provenance["byte_identical_python_files"]),
                selected_test_means=means, model_training_performed=False,
                interpretation=results["interpretation"])


def numerical_imports():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if sys.platform != "win32":
        raise RuntimeError("The unchanged study scripts import Windows msvcrt. Use Windows with CUDA for training; verify works on other platforms.")
    sys.path.insert(0, str(ROOT / "code"))
    sys.path.insert(0, str(ROOT / "audit"))
    import numpy as np
    import torch
    import target_guided_kan_20261116 as engine
    from data import load_data, make_split, validate_split
    return np, torch, engine, load_data, make_split, validate_split


def load_inputs(args):
    np, torch, engine, load_data, make_split, validate_split = numerical_imports()
    records = read(ROOT / "records/selected_results.json")
    spec = records["datasets"][args.study]
    if sha256(args.data) != spec["metadata"]["sha256"]:
        raise ValueError("The MAT does not match the recorded original data hash and row order.")
    matrices, labels, _, phenotype = load_data(args.data, spec["label_column"])
    if args.family == "manuscript":
        case = next(row for row in records["cases"] if (row["study"], row["selected_round"]) == (args.study, args.round))
        assignment = next(row for row in read(ROOT / "records/selected_splits.json")["cases"]
                          if (row["study"], row["selected_round"]) == (args.study, args.round))
        indices = validate_split({key: np.asarray(value, dtype=np.int64)
                                  for key, value in assignment["indices"].items()}, len(labels), labels)
        cfg = case["configuration"].copy()
        seed = case["random_split_seed"]
    else:
        plan = read(ROOT / "configs/site_centered_development.json")
        if args.round > 3:
            raise ValueError("The development comparison has exactly three predefined rounds.")
        seed = plan["random_seeds"][args.study][args.round - 1]
        indices = validate_split({key: np.sort(value) for key, value in make_split(labels, seed).items()}, len(labels), labels)
        cfg = dict(plan["configuration"], id=args.initializer,
                   roi_mass=plan["dataset_roi_mass"][args.study],
                   bootstrap_seeds=[9161601 + 100000 * (args.study == "adhd") + 100 * (args.round - 1) + b for b in range(4)])
        case = None
    regenerated = {key: np.sort(value) for key, value in make_split(labels, seed).items()}
    for key in indices:
        np.testing.assert_array_equal(indices[key], regenerated[key])
    return np, torch, engine, matrices, labels, phenotype, indices, cfg, spec, case


def data_check(args):
    np, _, _, matrices, labels, _, indices, cfg, _, _ = load_inputs(args)
    return dict(passed=True, study=args.study, family=args.family,
                original_mat_sha256=sha256(args.data), matrix_shape=list(matrices.shape),
                roles={key: dict(people=len(value), class_counts=np.bincount(labels[value], minlength=2).tolist())
                       for key, value in indices.items()}, configuration=cfg,
                saved_assignments_match_seed=True, model_training_performed=False)


def train(args):
    if args.family == "development" and args.evaluate_test:
        raise ValueError("The site-centered comparison is development-only.")
    if args.family == "manuscript" and args.initializer != "global_guided32":
        raise ValueError("The manuscript recipe uses global_guided32. Select the development family to compare initializers.")
    verify()
    np, torch, engine, matrices, labels, phenotype, indices, cfg, spec, reference = load_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError("This unchanged training implementation requires an NVIDIA CUDA device.")
    folder = ROOT / "outputs" / args.family / args.study / f"round_{args.round:02d}" / args.initializer
    if folder.exists():
        raise FileExistsError(f"Output already exists: {folder}. Use a fresh checkout or move the prior output.")
    folder.mkdir(parents=True)
    from run_current_parameters_20261116 import write
    from residual_kan import PAIRS
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    allowed = np.r_[indices["train"], indices["valid"]]
    raw = np.ascontiguousarray(matrices[allowed][:, PAIRS[0], PAIRS[1]])
    y = labels[allowed]
    write(folder / "run_settings.json", dict(
        study=args.study, family=args.family, selected_round=args.round, configuration=cfg,
        indices=indices, original_mat_sha256=spec["metadata"]["sha256"],
        evaluate_test=args.evaluate_test, with_dense=args.with_dense,
        torch_version=torch.__version__, numpy_version=np.__version__,
    ))
    if args.initializer == "within_site_guided32":
        import site_centered_guided_kan_20261116 as centered
        sites = phenotype[indices["train"], spec["site_column"]]
        result, cache = centered.train_native(raw, y, indices, cfg, folder, write, sites)
    else:
        result, cache = engine.train_native(raw, y, indices, cfg, folder, write)
    replay, _, checks = engine.forward(raw[len(indices["train"]):], cache["branches"], cfg)
    with np.load(folder / "development_predictions.npz") as stored:
        np.testing.assert_allclose(replay, stored["logits"], atol=1e-4, rtol=0)
    result["independent_development_forward"] = checks
    if args.evaluate_test:
        test_raw = np.ascontiguousarray(matrices[indices["test"]][:, PAIRS[0], PAIRS[1]])
        logits, branches, check = engine.forward(test_raw, cache["branches"], cfg)
        result["test"] = engine.metrics(labels[indices["test"]], logits, result["threshold"])
        result["independent_test_forward"] = check
        np.savez_compressed(folder / "test_predictions.npz", pool_indices=indices["test"],
                            labels=labels[indices["test"]], logits=logits, branch_logits=branches,
                            threshold=result["threshold"], predictions=logits >= result["threshold"])
    if args.with_dense:
        dense, branches = engine.train_dense(raw, y, indices, cfg, folder, cache, write)
        if args.evaluate_test:
            logits, branch_logits, check = engine.forward(test_raw, branches, cfg)
            dense["test"] = engine.metrics(labels[indices["test"]], logits, dense["threshold"])
            dense["independent_test_forward"] = check
            np.savez_compressed(folder / "dense_test_predictions.npz", pool_indices=indices["test"],
                                labels=labels[indices["test"]], logits=logits, branch_logits=branch_logits,
                                threshold=dense["threshold"], predictions=logits >= dense["threshold"])
        result["matched90"] = dense
    result.update(study=args.study, family=args.family, configuration=cfg, test_evaluated=args.evaluate_test)
    if reference is not None:
        result["historical_reference"] = {key: reference[key] for key in ["development", "test", "retained_rois"]}
    write(folder / "result.json", result)
    return dict(output=folder.relative_to(ROOT).as_posix(), development=result["development"],
                test=result.get("test"), retained_rois=result["retained_rois"], weight_files_written=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check published bytes, assignments and recorded metrics; no data or training.")
    for name in ["check-data", "train"]:
        command = commands.add_parser(name, help="Validate original MAT and splits." if name == "check-data" else "Fit the unchanged native KAN.")
        command.add_argument("--study", required=True, choices=["abide", "adhd"])
        command.add_argument("--data", required=True, type=Path)
        command.add_argument("--round", required=True, type=int, choices=range(1, 6))
        command.add_argument("--family", choices=["manuscript", "development"], default="manuscript")
        command.add_argument("--initializer", choices=["global_guided32", "within_site_guided32"], default="global_guided32")
        if name == "train":
            command.add_argument("--evaluate-test", action="store_true", help="Also evaluate this fixed division's test people after development calibration.")
            command.add_argument("--with-dense", action="store_true", help="Continue a matched 90-ROI control from the identical sources.")
    args = parser.parse_args()
    result = verify() if args.command == "verify" else data_check(args) if args.command == "check-data" else train(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
