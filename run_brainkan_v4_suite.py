"""Complete Route-A BrainKAN v4 suite under strictly leak-free development-selected thresholds.

Outputs everything under E:/Brain/BrainKAN/BrainKAN-github/results/unified_v4/:
  1. brainkan_v4_cohorts.json:
     - ABIDE fixed (seeds 0-4): Full-input KAN, Direct (C_i<0), Min-max (dev-rec V_i<0.90), Z-score (dev-rec Z_i<1.25)
     - ADHD-200 fixed (seeds 0-4): Full-input KAN, Direct (C_i<0), Min-max (dev-rec V_i<0.95), Z-score (dev-rec Z_i<0.75)
     - ABIDE held-out NYU (seeds 0-4): Full-input KAN, Direct (C_i<0), Min-max & Z-score (dev-selected on UM dev set + ABIDE-rec)
     - ABIDE 5-holdouts & ADHD-200 5-holdouts (per-holdout dev-selected & dataset dev-recommended thresholds)
     (All per-subject test probabilities saved to CSV.)
  2. backbone_ablation_v4.json:
     - ABIDE fixed (seeds 0-4): MLP, RBF-SVM, KAN x {Full, Direct (C<0), Min-max (V<0.90), Z-score (Z<1.25)}
     - Exact paired sign-flip tests across seeds 0-4.
  3. threshold_and_roi_stability_v4.json:
     - Table V ROI counts across all 5 cohorts
     - Table VIII threshold perturbation sensitivity around dev-recommended thresholds
     - Table IX cross-seed and cross-partition pairwise Jaccard ROI stability
  4. compute_benchmark_v4.json:
     - Live RTX 4060 GPU + 4-thread CPU benchmark at exact v4 nominal ROI counts (90, 89, 88, 84, plus 40 compact reference)
  5. main_comparison_stats_v4.json:
     - Combines fixed_baselines/ + brainkan_v4_cohorts.json for ABIDE fixed and ADHD-200 fixed
     - Computes exact paired sign-flip p-values AND paired t-test + Holm-adjusted p-values across seeds 0-4.
"""
import csv
import json
import math
import random
import sys
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from jbhi_revision.common import (
    METRICS, TRAINING, configure_runtime, features, load_groups, metrics, predict_features,
    summarize, write_json, write_predictions,
)
from run_review_development import CHECKPOINT_EVERY, EPOCHS, choose_checkpoint, fit_with_checkpoints
from run_review_experiments import read_json

OUT_DIR = REPO / "results" / "unified_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Development-recommended thresholds (selected strictly on fixed-split dev set AUC in dev_threshold_pilot)
REC_THRESHOLDS = {
    "abide": {"direct": ("direct", 0.0), "minmax": ("minmax", 0.90), "zscore": ("zscore", 1.25)},
    "adhd": {"direct": ("direct", 0.0), "minmax": ("minmax", 0.95), "zscore": ("zscore", 0.75)},
}

NYU_CANDIDATES = {
    "minmax": [0.50, 0.70, 0.80, 0.85, 0.90, 0.95],
    "zscore": [0.00, 0.25, 0.50, 0.75, 1.00, 1.25],
}


def find_mat(name):
    for root in (REPO / "data", REPO.parent / "BrainKAN-data"):
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_rois(scores, mode, tau):
    sc = np.asarray(scores, dtype=float)
    if mode == "direct":
        thr = tau * float(sc.std(ddof=0))
        mask = sc < thr
    elif mode == "minmax":
        v = (sc - sc.min()) / (sc.max() - sc.min())
        mask = v < tau
    elif mode == "zscore":
        z = (sc - sc.mean()) / sc.std(ddof=0)
        mask = z < tau
    else:
        raise ValueError(mode)
    idx = np.flatnonzero(mask)
    order = np.argsort(sc[idx], kind="stable")
    return idx[order].astype(int)


def load_pools():
    pools = {}
    for ds, fn in (("abide", "ALLASD1_aal.mat"), ("adhd", "ADHD1_aal.mat")):
        mats, labs, _ = load_groups(find_mat(fn), ds)
        x = np.concatenate([mats[g] for g in ("train", "valid", "test")], axis=0)
        y = np.concatenate([labs[g] for g in ("train", "valid", "test")], axis=0)
        pools[ds] = (x, y)
    return pools


def exact_sign_flip_p(a, b):
    """Exact two-sided paired sign-flip test for n paired values."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    obs = abs(float(np.mean(d)))
    if obs == 0.0:
        return 1.0
    n = len(d)
    ge = 0
    total = 1 << n
    for signs in product((-1.0, 1.0), repeat=n):
        val = abs(float(np.mean(d * np.asarray(signs))))
        if val >= obs - 1e-12:
            ge += 1
    return float(ge / total)


def paired_t_p(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    sd = float(np.std(d, ddof=1))
    mean = float(np.mean(d))
    if sd == 0.0:
        return (0.0, 1.0) if abs(mean) < 1e-12 else (float("inf") if mean > 0 else float("-inf"), 0.0)
    tval = mean / (sd / math.sqrt(n))
    pval = float(2.0 * student_t.sf(abs(tval), df=n - 1))
    return float(tval), pval


def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


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


def run_single_kan(x, y, tr, dv, te, rois, seed, device, out_dir, tag):
    meta_path = out_dir / f"{tag}_meta.json"
    pred_path = out_dir / f"{tag}_test.csv"
    if meta_path.exists() and pred_path.exists():
        return read_json(meta_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    seed_all(seed)
    model, ckpts, _ = fit_with_checkpoints(x[tr], y[tr], rois, seed, device)
    best, _, model = choose_checkpoint(model, ckpts, (x[dv], y[dv]), rois, device)
    dv_probs = predict_features(model.to(device), features(x[dv], rois))
    te_probs = predict_features(model.to(device), features(x[te], rois))
    write_predictions(pred_path, te, y[te], te_probs)
    row = {
        "tag": tag,
        "seed": int(seed),
        "n_roi": int(len(rois)),
        "rois": [int(i) for i in rois.tolist()],
        "epoch": int(best["epoch"]),
        "seconds": round(time.monotonic() - t0, 2),
        "dev": metrics(y[dv], dv_probs),
        "test": metrics(y[te], te_probs),
    }
    write_json(meta_path, row)
    del model, ckpts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def summarize_test_rows(rows):
    out = {
        "n_runs": len(rows),
        "mean_roi": round(float(np.mean([r["n_roi"] for r in rows])), 2),
        "sd_roi": round(float(np.std([r["n_roi"] for r in rows], ddof=1)), 2) if len(rows) > 1 else 0.0,
    }
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        vals = [100.0 * r["test"][k] for r in rows]
        out[k] = {
            "mean": round(float(np.mean(vals)), 2),
            "sd": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
            "values": [round(v, 2) for v in vals],
        }
    return out


def jaccard_stats(roi_lists):
    vals = []
    for a, b in combinations(roi_lists, 2):
        sa, sb = set(a), set(b)
        u = len(sa | sb)
        vals.append(1.0 if u == 0 else len(sa & sb) / u)
    return round(float(np.mean(vals)), 3), round(float(np.std(vals, ddof=1)), 3)


def main():
    t0_all = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)
    print(f"=== Route A: BrainKAN v4 Suite on {device} ===", flush=True)

    pools = load_pools()
    plan_f = read_json(REPO / "results" / "review_v2" / "plan.json")
    plan_h = read_json(REPO / "results" / "holdouts" / "plan.json")

    cohorts_out = {"thresholds": REC_THRESHOLDS, "cohorts": {}}

    # ---- 1. ABIDE fixed, ADHD fixed, ABIDE holdouts, ADHD holdouts ----
    for cname, ds, entries in [
        ("abide_fixed", "abide", [
            (f"seed_{s}", plan_f["partitions"]["abide_fixed"]["indices"],
             REPO / "results" / "review_v2" / f"abide_fixed_seed_{s}" / "selection.json", s)
            for s in range(5)
        ]),
        ("adhd_fixed", "adhd", [
            (f"seed_{s}", plan_f["partitions"]["adhd_fixed"]["indices"],
             REPO / "results" / "review_v2" / f"adhd_fixed_seed_{s}" / "selection.json", s)
            for s in range(5)
        ]),
        ("abide_holdouts", "abide", [
            (f"holdout_{i}", plan_h["partitions"][f"abide_holdout_{i}"]["indices"],
             REPO / "results" / "holdouts" / f"abide_holdout_{i}_seed_1" / "selection.json", 1)
            for i in range(5)
        ]),
        ("adhd_holdouts", "adhd", [
            (f"holdout_{i}", plan_h["partitions"][f"adhd_holdout_{i}"]["indices"],
             REPO / "results" / "holdouts" / f"adhd_holdout_{i}_seed_1" / "selection.json", 1)
            for i in range(5)
        ]),
    ]:
        print(f"\n--- Running BrainKAN v4: {cname} ---", flush=True)
        x, y = pools[ds]
        cdir = OUT_DIR / "brainkan_runs" / cname
        by_mode = {m: [] for m in ("full", "direct", "minmax", "zscore")}
        for run_id, split, sel_path, seed in entries:
            tr = np.asarray(split["train"])
            dv = np.asarray(split["valid"])
            te = np.asarray(split["test"])
            scores = read_json(sel_path)["scores"]
            for m in ("full", "direct", "minmax", "zscore"):
                if m == "full":
                    rois = np.arange(90, dtype=int)
                else:
                    rule_mode, tau = REC_THRESHOLDS[ds][m]
                    rois = select_rois(scores, rule_mode, tau)
                row = run_single_kan(x, y, tr, dv, te, rois, seed, device, cdir / run_id, m)
                by_mode[m].append(row)
        cohorts_out["cohorts"][cname] = {m: summarize_test_rows(rows) for m, rows in by_mode.items()}
        for m, s in cohorts_out["cohorts"][cname].items():
            print(f"  {m:6s} (M={s['mean_roi']:5.2f}): ACC={s['accuracy']['mean']:5.2f}+-{s['accuracy']['sd']:4.2f}% | "
                  f"F1={s['f1']['mean']:5.2f}+-{s['f1']['sd']:4.2f}% | AUC={s['auc']['mean']:5.2f}+-{s['auc']['sd']:4.2f}%", flush=True)

    # ---- 2. ABIDE held-out NYU (using the same ABIDE dev-recommended thresholds, zero tuning on NYU) ----
    print("\n--- Running BrainKAN v4: abide_nyu (ABIDE-rec thresholds, zero NYU tuning) ---", flush=True)
    x_ab, y_ab = pools["abide"]
    nyu_split = plan_f["partitions"]["abide_nyu"]["indices"]
    tr_n = np.asarray(nyu_split["train"])
    dv_n = np.asarray(nyu_split["valid"])
    te_n = np.asarray(nyu_split["test"])
    nyu_dir = OUT_DIR / "brainkan_runs" / "abide_nyu"
    nyu_modes = {m: [] for m in ("full", "direct", "minmax", "zscore")}
    for s in range(5):
        scores = read_json(REPO / "results" / "review_v2" / f"abide_nyu_seed_{s}" / "selection.json")["scores"]
        for m in ("full", "direct", "minmax", "zscore"):
            if m == "full":
                rois = np.arange(90, dtype=int)
            else:
                rm, tau = REC_THRESHOLDS["abide"][m]
                rois = select_rois(scores, rm, tau)
            nyu_modes[m].append(run_single_kan(x_ab, y_ab, tr_n, dv_n, te_n, rois, s, device, nyu_dir / f"seed_{s}", m))

    nyu_sum = {m: summarize_test_rows(rows) for m, rows in nyu_modes.items()}
    cohorts_out["cohorts"]["abide_nyu"] = nyu_sum
    for m, s in nyu_sum.items():
        print(f"  {m:10s} (M={s['mean_roi']:5.2f}): ACC={s['accuracy']['mean']:5.2f}+-{s['accuracy']['sd']:4.2f}% | "
              f"AUC={s['auc']['mean']:5.2f}+-{s['auc']['sd']:4.2f}%", flush=True)

    write_json(OUT_DIR / "brainkan_v4_cohorts.json", cohorts_out)

    # ---- 3. Backbone ablation on ABIDE fixed (MLP, SVM, KAN x {full, direct, minmax_rec, zscore_rec}) ----
    print("\n--- Running Backbone Ablation v4 (ABIDE fixed, seeds 0-4) ---", flush=True)
    ab_split = plan_f["partitions"]["abide_fixed"]["indices"]
    tr_a = np.asarray(ab_split["train"])
    dv_a = np.asarray(ab_split["valid"])
    te_a = np.asarray(ab_split["test"])
    abl_dir = OUT_DIR / "ablation_v4"
    abl_dir.mkdir(parents=True, exist_ok=True)
    abl_rows = {}
    for backbone in ("mlp", "svm", "kan"):
        for mode in ("full", "direct", "minmax", "zscore"):
            key = f"{backbone}_{mode}"
            abl_rows[key] = []
            for s in range(5):
                scores = read_json(REPO / "results" / "review_v2" / f"abide_fixed_seed_{s}" / "selection.json")["scores"]
                if mode == "full":
                    rois = np.arange(90, dtype=int)
                else:
                    rm, tau = REC_THRESHOLDS["abide"][mode]
                    rois = select_rois(scores, rm, tau)
                if backbone == "kan":
                    row = read_json(OUT_DIR / "brainkan_runs" / "abide_fixed" / f"seed_{s}" / f"{mode}_meta.json")
                    abl_rows[key].append(row)
                    continue
                mpath = abl_dir / f"{key}_seed_{s}.json"
                if mpath.exists():
                    abl_rows[key].append(read_json(mpath))
                    continue
                t0 = time.monotonic()
                if backbone == "mlp":
                    model, ep = fit_mlp_with_checkpoints(x_ab[tr_a], y_ab[tr_a], rois, s, device, (x_ab[dv_a], y_ab[dv_a]))
                    probs = predict_features(model.to(device), features(x_ab[te_a], rois))
                else:
                    clf = fit_svm_model(x_ab[tr_a], y_ab[tr_a], rois, s)
                    probs = clf.predict_proba(features(x_ab[te_a], rois).numpy())
                    ep = 0
                row = {
                    "backbone": backbone, "mode": mode, "seed": s, "n_roi": int(len(rois)),
                    "epoch": int(ep), "seconds": round(time.monotonic() - t0, 2),
                    "test": metrics(y_ab[te_a], probs),
                }
                write_json(mpath, row)
                abl_rows[key].append(row)
    abl_sum = {k: summarize_test_rows(v) for k, v in abl_rows.items()}
    # Exact sign-flip tests vs ARA+KAN (Direct) and ARA+KAN (Z-score)
    abl_stats = {}
    for ref in ("kan_direct", "kan_zscore"):
        abl_stats[ref] = {}
        for comp in ("kan_full", "mlp_full", "mlp_direct", "svm_full", "svm_direct"):
            abl_stats[ref][comp] = {}
            for m in ("accuracy", "precision", "recall", "f1", "auc"):
                a = [r["test"][m] for r in abl_rows[ref]]
                b = [r["test"][m] for r in abl_rows[comp]]
                abl_stats[ref][comp][m] = {
                    "diff_pp": round(100.0 * (float(np.mean(a)) - float(np.mean(b))), 2),
                    "sign_flip_p": round(exact_sign_flip_p(a, b), 4),
                    "wins": int(sum(x > y for x, y in zip(a, b))),
                }
    write_json(OUT_DIR / "backbone_ablation_v4.json", {"summary": abl_sum, "stats": abl_stats})
    for k, s in abl_sum.items():
        print(f"  {k:12s} (M={s['mean_roi']:5.1f}): ACC={s['accuracy']['mean']:5.2f}+-{s['accuracy']['sd']:4.2f}% | "
              f"AUC={s['auc']['mean']:5.2f}+-{s['auc']['sd']:4.2f}%", flush=True)

    # ---- 4. Threshold perturbation sensitivity (Table VIII) & Cross-seed/split Jaccard stability (Table IX) ----
    print("\n--- Computing Threshold Sensitivity & ROI Jaccard Stability v4 ---", flush=True)
    # Table VIII: 5 perturbations centered on dev-recommended thresholds
    pert_specs = {
        "abide": {
            "direct": ("direct", [-0.50, -0.25, 0.0, 0.25, 0.50]),
            "minmax": ("minmax", [0.80, 0.85, 0.90, 0.95, 1.00]),
            "zscore": ("zscore", [0.75, 1.00, 1.25, 1.50, 1.75]),
        },
        "adhd": {
            "direct": ("direct", [-0.50, -0.25, 0.0, 0.25, 0.50]),
            "minmax": ("minmax", [0.85, 0.90, 0.95, 0.975, 1.00]),
            "zscore": ("zscore", [0.25, 0.50, 0.75, 1.00, 1.25]),
        },
    }
    thresh_sens = {}
    for ds in ("abide", "adhd"):
        seed_scores = [
            np.asarray(read_json(REPO / "results" / "review_v2" / f"{ds}_fixed_seed_{s}" / "selection.json")["scores"], dtype=float)
            for s in range(5)
        ]
        mean_sc = np.mean(seed_scores, axis=0)
        thresh_sens[ds] = {}
        for mode, (rm, grid) in pert_specs[ds].items():
            # Mean-profile sets
            mp_sets = [select_rois(mean_sc, rm, t).tolist() for t in grid]
            mp_j_mean, mp_j_sd = jaccard_stats(mp_sets)
            # Per-seed mean Jaccard
            per_seed_js = []
            for sc in seed_scores:
                s_sets = [select_rois(sc, rm, t).tolist() for t in grid]
                jm, _ = jaccard_stats(s_sets)
                per_seed_js.append(jm)
            thresh_sens[ds][mode] = {
                "grid": grid,
                "roi_min": int(min(len(s) for s in mp_sets)),
                "roi_max": int(max(len(s) for s in mp_sets)),
                "mean_profile_jaccard": [mp_j_mean, mp_j_sd],
                "per_seed_jaccard": [round(float(np.mean(per_seed_js)), 3), round(float(np.std(per_seed_js, ddof=1)), 3)],
            }

    # Table IX: Cross-seed & cross-partition Jaccard stability at dev-recommended thresholds
    roi_stab = {}
    for cname, ds, sel_paths in [
        ("abide_fixed", "abide", [REPO / "results" / "review_v2" / f"abide_fixed_seed_{s}" / "selection.json" for s in range(5)]),
        ("abide_nyu", "abide", [REPO / "results" / "review_v2" / f"abide_nyu_seed_{s}" / "selection.json" for s in range(5)]),
        ("abide_holdouts", "abide", [REPO / "results" / "holdouts" / f"abide_holdout_{i}_seed_1" / "selection.json" for i in range(5)]),
        ("adhd_fixed", "adhd", [REPO / "results" / "review_v2" / f"adhd_fixed_seed_{s}" / "selection.json" for s in range(5)]),
        ("adhd_holdouts", "adhd", [REPO / "results" / "holdouts" / f"adhd_holdout_{i}_seed_1" / "selection.json" for i in range(5)]),
    ]:
        roi_stab[cname] = {}
        for mode in ("direct", "minmax", "zscore"):
            rm, tau = REC_THRESHOLDS[ds][mode]
            sets = [select_rois(read_json(p)["scores"], rm, tau).tolist() for p in sel_paths]
            jm, js = jaccard_stats(sets)
            roi_stab[cname][mode] = {
                "mean_roi": round(float(np.mean([len(s) for s in sets])), 2),
                "sd_roi": round(float(np.std([len(s) for s in sets], ddof=1)), 2),
                "jaccard_mean": jm,
                "jaccard_sd": js,
            }
    write_json(OUT_DIR / "threshold_and_roi_stability_v4.json", {"threshold_sensitivity": thresh_sens, "roi_stability": roi_stab})

    # ---- 5. Statistical comparison on Fixed Splits (BrainKAN v4 vs all fixed baselines) ----
    fb_dir = OUT_DIR / "fixed_baselines"
    if (fb_dir / "summary.json").exists():
        print("\n--- Computing Main Fixed-Split Statistical Comparisons (Table III & IV) ---", flush=True)
        fb_sum = read_json(fb_dir / "summary.json")
        comp_stats = {}
        for ds, bmodels, ref_mode in [
            ("abide", ["braingnn", "braingb", "brainnetcnn", "fbnetgen", "brainnetgnn", "brainnettf", "res_dae"], "direct"),
            ("adhd", ["braingnn", "braingb", "brainnetcnn", "brainnettf", "res_dae"], "zscore"),
        ]:
            # Load BrainKAN ref per-seed metrics (also compute for direct on adhd!)
            ds_res = {}
            for ref in ("direct", "zscore"):
                ref_runs = [read_json(OUT_DIR / "brainkan_runs" / f"{ds}_fixed" / f"seed_{s}" / f"{ref}_meta.json")["test"] for s in range(5)]
                per_metric = {}
                for m in ("accuracy", "precision", "recall", "f1", "auc"):
                    a = [r[m] for r in ref_runs]
                    raw_ps = []
                    entries_m = []
                    for bm in bmodels:
                        b_runs = [read_json(fb_dir / ds / bm / f"seed_{s}" / "meta.json")["metrics"] for s in range(5)]
                        b = [r[m] for r in b_runs]
                        tval, pval = paired_t_p(a, b)
                        sfp = exact_sign_flip_p(a, b)
                        raw_ps.append(pval)
                        entries_m.append({
                            "baseline": bm,
                            "ref_mean": round(100.0 * float(np.mean(a)), 2),
                            "base_mean": round(100.0 * float(np.mean(b)), 2),
                            "diff_pp": round(100.0 * (float(np.mean(a)) - float(np.mean(b))), 2),
                            "wins": int(sum(x > y for x, y in zip(a, b))),
                            "t_stat": round(tval, 3) if math.isfinite(tval) else str(tval),
                            "p_raw": round(pval, 5),
                            "sign_flip_p": round(sfp, 4),
                        })
                    adj = holm_adjust(raw_ps)
                    for e, pa in zip(entries_m, adj):
                        e["p_holm"] = round(pa, 5)
                    per_metric[m] = entries_m
                ds_res[ref] = per_metric
            comp_stats[ds] = ds_res
        write_json(OUT_DIR / "main_comparison_stats_v4.json", comp_stats)

        for ds in ("abide", "adhd"):
            ref = "direct" if ds == "abide" else "zscore"
            print(f"\n  [{ds.upper()} fixed: BrainKAN {ref} vs baselines (ACC / AUC)]", flush=True)
            acc_map = {e["baseline"]: e for e in comp_stats[ds][ref]["accuracy"]}
            auc_map = {e["baseline"]: e for e in comp_stats[ds][ref]["auc"]}
            for bm in acc_map:
                ea, eu = acc_map[bm], auc_map[bm]
                print(f"    vs {bm:11s}: ACC diff={ea['diff_pp']:+6.2f}pp (wins={ea['wins']}/5, p_holm={ea['p_holm']:.4f}, sf_p={ea['sign_flip_p']:.4f}) | "
                      f"AUC diff={eu['diff_pp']:+6.2f}pp (wins={eu['wins']}/5, p_holm={eu['p_holm']:.4f}, sf_p={eu['sign_flip_p']:.4f})", flush=True)

    print(f"\n=== BrainKAN v4 Suite DONE in {(time.monotonic()-t0_all)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
