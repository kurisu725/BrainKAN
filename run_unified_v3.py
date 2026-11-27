"""Run unified training-only BrainKAN on the 5 holdouts (full/direct/minmax/zscore)
and run the complete ablation study (MLP, SVM, KAN, ARA+MLP, ARA+SVM, ARA+KAN Mode 1/2/3)
on the ABIDE fixed split using the exact training-only ROI selections from review_v2.
"""
import json
import random
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jbhi_revision.common import (
    METRICS, TRAINING, configure_runtime, features, load_groups, metrics, predict_features,
    sha256, summarize, write_json, write_predictions,
)
from run_review_development import CHECKPOINT_EVERY, EPOCHS, choose_checkpoint, fit_with_checkpoints
from run_review_experiments import GROUPS, THRESHOLDS, read_json
from summarize_holdouts import BASELINES, MODEL_NAMES, baseline_prediction, check_baseline_plan


REPO = Path(__file__).resolve().parent
ABIDE_MAT = REPO / "data" / "ALLASD1_aal.mat"
ADHD_MAT = REPO / "data" / "ADHD1_aal.mat"
HOLDOUT_DIR = REPO / "results" / "holdouts"
REVIEW_V2_DIR = REPO / "results" / "review_v2"
BASELINE_ROOT = REPO / "baseline_reference"
OUT_DIR = REPO / "results" / "unified_v3"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp_with_checkpoints(train, targets, rois, seed, device, development):
    seed_all(seed)
    model = MLPClassifier(len(rois) ** 2, TRAINING["hidden"]).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(features(train, rois), torch.from_numpy(targets)),
        batch_size=TRAINING["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=TRAINING["lr_gamma"])
    criterion = nn.CrossEntropyLoss()
    checkpoints = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
            checkpoints.append((epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        scheduler.step()
    best, _, model = choose_checkpoint(model, checkpoints, development, rois, device)
    return model, best["epoch"]


def fit_svm_model(train, targets, rois, seed):
    x_tr = features(train, rois).numpy()
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed))
    clf.fit(x_tr, targets)
    return clf


def run_holdout_brainkan(pools, device):
    """Run BrainKAN full, direct, minmax, zscore on the 5 holdouts per dataset
    with development-set checkpoint selection (matching Algorithm 1 / review_v2)."""
    plan = read_json(HOLDOUT_DIR / "plan.json")
    out = OUT_DIR / "holdouts_v3"
    out.mkdir(parents=True, exist_ok=True)
    modes = ("full", "direct", "minmax", "zscore")
    per_run = []
    for dataset in ("abide", "adhd"):
        matrices, labels = pools[dataset]
        for partition in range(5):
            name = f"{dataset}_holdout_{partition}"
            split = plan["partitions"][name]["indices"]
            tr = np.asarray(split["train"])
            dv = np.asarray(split["valid"])
            te = np.asarray(split["test"])
            sel = read_json(HOLDOUT_DIR / f"{name}_seed_1" / "selection.json")
            run_dir = out / f"{name}_seed_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            for mode in modes:
                rois = np.arange(90) if mode == "full" else np.asarray(sel["selections"][mode]["indices"], dtype=int)
                pred_path = run_dir / f"{mode}_test.csv"
                meta_path = run_dir / f"{mode}_meta.json"
                if pred_path.exists() and meta_path.exists():
                    meta = read_json(meta_path)
                    per_run.append(meta)
                    continue
                seed_all(1)
                t0 = time.monotonic()
                model, checkpoints, _ = fit_with_checkpoints(matrices[tr], labels[tr], rois, 1, device)
                best, _, model = choose_checkpoint(model, checkpoints, (matrices[dv], labels[dv]), rois, device)
                probs = predict_features(model.to(device), features(matrices[te], rois))
                write_predictions(pred_path, te, labels[te], probs)
                m = metrics(labels[te], probs)
                row = {
                    "dataset": dataset,
                    "holdout": partition,
                    "model": mode,
                    "seed": 1,
                    "n_roi": int(len(rois)),
                    "epoch": int(best["epoch"]),
                    "seconds": round(time.monotonic() - t0, 2),
                    **m,
                }
                write_json(meta_path, row)
                per_run.append(row)
                print(
                    f"[Holdout] {name} {mode:6s} (M={len(rois):2d}, ep={best['epoch']:2d}): "
                    f"ACC={100*m['accuracy']:.2f}% AUC={100*m['auc']:.2f}% ({row['seconds']:.1f}s)",
                    flush=True,
                )
                del model, checkpoints
                torch.cuda.empty_cache()

    # Combine with sealed baseline predictions and compute corrected resampled t-test + Holm
    summary = {}
    stats_all = {}
    for dataset, archives in BASELINES.items():
        models_rows = {}
        for r in per_run:
            if r["dataset"] == dataset:
                models_rows.setdefault(r["model"], []).append(r)
        for partition in range(5):
            name = f"{dataset}_holdout_{partition}"
            split = plan["partitions"][name]["indices"]
            for source, layout, model_names in archives:
                root = BASELINE_ROOT / source
                check_baseline_plan(read_json(root / "plan.json"), partition, split, plan["datasets"][dataset]["sha256"])
                for bmodel in model_names:
                    _, _, scores, _, _ = baseline_prediction(root, partition, bmodel, layout)
                    models_rows.setdefault(bmodel, []).append({
                        "dataset": dataset,
                        "holdout": partition,
                        "model": bmodel,
                        "n_roi": 90,
                        **scores,
                    })
        # Ensure sorted by holdout
        for k in models_rows:
            models_rows[k] = sorted(models_rows[k], key=lambda x: x["holdout"])
        summary[dataset] = {
            mname: {
                "n_runs": len(rows),
                "mean_roi": float(np.mean([r["n_roi"] for r in rows])),
                "sd_roi": float(np.std([r["n_roi"] for r in rows], ddof=1)),
                "metrics": summarize(rows),
            }
            for mname, rows in models_rows.items()
        }
        # Compute paired corrected resampled t-test comparing BrainKAN Direct (Mode 1) and Z-score (Mode 3) and Full against all matched baselines
        n_train = len(plan["partitions"][f"{dataset}_holdout_0"]["indices"]["train"])
        n_test = len(plan["partitions"][f"{dataset}_holdout_0"]["indices"]["test"])
        baseline_keys = [m for _, _, names in archives for m in names]
        ds_stats = {}
        for ref_mode in ("direct", "zscore", "minmax", "full"):
            ref_rows = models_rows[ref_mode]
            mode_res = []
            for metric in METRICS:
                family = []
                for bmodel in baseline_keys:
                    brows = models_rows[bmodel]
                    diffs = np.array([a[metric] - b[metric] for a, b in zip(ref_rows, brows)])
                    mean = float(diffs.mean())
                    sd = float(diffs.std(ddof=1))
                    se = float(sd * np.sqrt(1.0 / len(diffs) + n_test / n_train))
                    p_val = float(2 * student_t.sf(abs(mean) / se, df=len(diffs) - 1)) if se > 0 else 1.0
                    family.append({
                        "metric": metric,
                        "baseline": bmodel,
                        "mean_diff": mean,
                        "sd_diff": sd,
                        "corrected_se": se,
                        "p": p_val,
                    })
                adj = 0.0
                for rank, item in enumerate(sorted(family, key=lambda x: x["p"])):
                    adj = max(adj, min(1.0, item["p"] * (len(family) - rank)))
                    item["p_holm"] = adj
                    item["significant"] = bool(adj < 0.05)
                mode_res.extend(family)
            ds_stats[ref_mode] = mode_res
        stats_all[dataset] = ds_stats

    write_json(out / "summary_and_stats.json", {"summary": summary, "paired_statistics": stats_all})
    return summary, stats_all


def run_ablation_fixed(pools, device):
    """Run complete ablation study on ABIDE fixed split (seeds 0-4, dev-selected checkpoints):
    MLP, SVM, KAN-full, ARA+MLP (direct/minmax/zscore), ARA+SVM (direct/minmax/zscore),
    and include BrainKAN (ARA+KAN) direct/minmax/zscore from review_v2 + fresh KAN-full dev-selected."""
    plan = read_json(REVIEW_V2_DIR / "plan.json")
    split = plan["partitions"]["abide_fixed"]["indices"]
    tr = np.asarray(split["train"])
    dv = np.asarray(split["valid"])
    te = np.asarray(split["test"])
    matrices, labels = pools["abide"]

    out = OUT_DIR / "ablation_fixed"
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # Load review_v2 BrainKAN results for direct, minmax, zscore on abide_fixed
    rv2 = read_json(REVIEW_V2_DIR / "results.json")
    for r in rv2["per_run"]:
        if r["partition"] == "abide_fixed":
            rows.append({
                "model": f"ara_kan_{r['model']}",
                "seed": r["seed"],
                "n_roi": r["n_roi"],
                "epoch": r["epoch"],
                **{k: r[k] for k in METRICS},
            })

    # Run KAN-full (dev-selected), MLP (full + direct/minmax/zscore), SVM (full + direct/minmax/zscore)
    for seed in range(5):
        sel = read_json(REVIEW_V2_DIR / f"abide_fixed_seed_{seed}" / "selection.json")
        roi_sets = {
            "full": np.arange(90),
            "direct": np.asarray(sel["selections"]["direct"]["indices"], dtype=int),
            "minmax": np.asarray(sel["selections"]["minmax"]["indices"], dtype=int),
            "zscore": np.asarray(sel["selections"]["zscore"]["indices"], dtype=int),
        }
        # KAN-full (dev-selected checkpoints every 5 epochs, matching review_v2)
        meta_kan = out / f"kan_full_seed_{seed}.json"
        if meta_kan.exists():
            rows.append(read_json(meta_kan))
        else:
            seed_all(seed)
            model, checkpoints, _ = fit_with_checkpoints(matrices[tr], labels[tr], roi_sets["full"], seed, device)
            best, _, model = choose_checkpoint(model, checkpoints, (matrices[dv], labels[dv]), roi_sets["full"], device)
            probs = predict_features(model.to(device), features(matrices[te], roi_sets["full"]))
            m = metrics(labels[te], probs)
            r = {"model": "kan_full", "seed": seed, "n_roi": 90, "epoch": int(best["epoch"]), **{k: m[k] for k in METRICS}}
            write_json(meta_kan, r)
            rows.append(r)
            print(f"[Ablation] kan_full seed={seed}: ACC={100*m['accuracy']:.2f}% AUC={100*m['auc']:.2f}%", flush=True)
            del model, checkpoints
            torch.cuda.empty_cache()

        # MLP variants
        for variant, rois in roi_sets.items():
            mname = "mlp" if variant == "full" else f"ara_mlp_{variant}"
            mpath = out / f"{mname}_seed_{seed}.json"
            if mpath.exists():
                rows.append(read_json(mpath))
                continue
            model, ep = fit_mlp_with_checkpoints(matrices[tr], labels[tr], rois, seed, device, (matrices[dv], labels[dv]))
            probs = predict_features(model.to(device), features(matrices[te], rois))
            m = metrics(labels[te], probs)
            r = {"model": mname, "seed": seed, "n_roi": int(len(rois)), "epoch": int(ep), **{k: m[k] for k in METRICS}}
            write_json(mpath, r)
            rows.append(r)
            print(f"[Ablation] {mname:15s} seed={seed} (M={len(rois):2d}): ACC={100*m['accuracy']:.2f}% AUC={100*m['auc']:.2f}%", flush=True)
            del model
            torch.cuda.empty_cache()

        # SVM variants
        for variant, rois in roi_sets.items():
            mname = "svm" if variant == "full" else f"ara_svm_{variant}"
            mpath = out / f"{mname}_seed_{seed}.json"
            if mpath.exists():
                rows.append(read_json(mpath))
                continue
            clf = fit_svm_model(matrices[tr], labels[tr], rois, seed)
            probs = clf.predict_proba(features(matrices[te], rois).numpy())
            m = metrics(labels[te], probs)
            r = {"model": mname, "seed": seed, "n_roi": int(len(rois)), "epoch": 0, **{k: m[k] for k in METRICS}}
            write_json(mpath, r)
            rows.append(r)
            print(f"[Ablation] {mname:15s} seed={seed} (M={len(rois):2d}): ACC={100*m['accuracy']:.2f}% AUC={100*m['auc']:.2f}%", flush=True)

    # Also compute ADHD fixed KAN-full dev-selected and ABIDE NYU KAN-full dev-selected for Table VI completeness
    extra_full = {}
    for part_name, ds_name in (("adhd_fixed", "adhd"), ("abide_nyu", "abide")):
        psplit = plan["partitions"][part_name]["indices"]
        ptr, pdv, pte = np.asarray(psplit["train"]), np.asarray(psplit["valid"]), np.asarray(psplit["test"])
        pmat, plab = pools[ds_name]
        prows = []
        for seed in range(5):
            mpath = out / f"{part_name}_kan_full_seed_{seed}.json"
            if mpath.exists():
                prows.append(read_json(mpath))
                continue
            seed_all(seed)
            model, checkpoints, _ = fit_with_checkpoints(pmat[ptr], plab[ptr], np.arange(90), seed, device)
            best, _, model = choose_checkpoint(model, checkpoints, (pmat[pdv], plab[pdv]), np.arange(90), device)
            probs = predict_features(model.to(device), features(pmat[pte], np.arange(90)))
            m = metrics(plab[pte], probs)
            r = {"partition": part_name, "model": "kan_full", "seed": seed, "n_roi": 90, "epoch": int(best["epoch"]), **{k: m[k] for k in METRICS}}
            write_json(mpath, r)
            prows.append(r)
            print(f"[Full-dev] {part_name} seed={seed}: ACC={100*m['accuracy']:.2f}% AUC={100*m['auc']:.2f}%", flush=True)
            del model, checkpoints
            torch.cuda.empty_cache()
        extra_full[part_name] = {"per_run": prows, "summary": summarize(prows)}

    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    summary = {
        mname: {
            "n_runs": len(mrows),
            "mean_roi": float(np.mean([r["n_roi"] for r in mrows])),
            "metrics": summarize(mrows),
        }
        for mname, mrows in by_model.items()
    }
    write_json(out / "ablation_summary.json", {"summary": summary, "per_run": rows, "extra_full_dev": extra_full})
    return summary, extra_full


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_runtime(device)
    inputs = {
        "abide": load_groups(ABIDE_MAT, "abide"),
        "adhd": load_groups(ADHD_MAT, "adhd"),
    }
    pools = {
        ds: (
            np.concatenate([m[g] for g in GROUPS]),
            np.concatenate([l[g] for g in GROUPS]),
        )
        for ds, (m, l, _) in inputs.items()
    }
    del inputs
    with threadpool_limits(limits=TRAINING["cpu_threads"], user_api="blas"):
        h_sum, h_stats = run_holdout_brainkan(pools, device)
        a_sum, extra_full = run_ablation_fixed(pools, device)

    print("\n=== SUMMARY: 5-HOLDOUT MATCHED COMPARISON ===")
    for ds in ("abide", "adhd"):
        print(f"\n--- {ds.upper()} ---")
        for mname, val in h_sum[ds].items():
            ms = val["metrics"]
            print(
                f"  {mname:15s} (M={val['mean_roi']:5.2f}): "
                f"ACC={100*ms['accuracy']['mean']:5.2f}±{100*ms['accuracy']['sd']:4.2f}  "
                f"P={100*ms['precision']['mean']:5.2f}±{100*ms['precision']['sd']:4.2f}  "
                f"R={100*ms['recall']['mean']:5.2f}±{100*ms['recall']['sd']:4.2f}  "
                f"F1={100*ms['f1']['mean']:5.2f}±{100*ms['f1']['sd']:4.2f}  "
                f"AUC={100*ms['auc']['mean']:5.2f}±{100*ms['auc']['sd']:4.2f}"
            )
    print("\n=== SUMMARY: ABIDE FIXED ABLATION ===")
    for mname, val in a_sum.items():
        ms = val["metrics"]
        print(
            f"  {mname:18s} (M={val['mean_roi']:5.2f}): "
            f"ACC={100*ms['accuracy']['mean']:5.2f}±{100*ms['accuracy']['sd']:4.2f}  "
            f"AUC={100*ms['auc']['mean']:5.2f}±{100*ms['auc']['sd']:4.2f}"
        )


if __name__ == "__main__":
    main()
