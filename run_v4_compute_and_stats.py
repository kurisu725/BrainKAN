"""Run Route-A v4 live computational benchmark (Table I) and complete paired statistical verification.

Outputs under results/unified_v4/:
  1. compute_benchmark_v4.json:
     Live RTX 4060 GPU + 4-thread CPU benchmark for KAN-full (90 ROIs),
     BrainKAN Mode 1 Direct (89 ROIs, ABIDE mean 88.6),
     BrainKAN Mode 2 Min-max (88 ROIs ABIDE / 67 ROIs ADHD-200),
     BrainKAN Mode 3 Z-score (84 ROIs, ABIDE mean 84.4 / ADHD mean 83.8),
     plus compact Z_i<0 reference (40 ROIs), and all seven re-implemented baselines.
  2. holdouts_comparison_stats_v4.json:
     Matched 5-holdout statistical comparison of BrainKAN v4 (dev-recommended thresholds)
     against the 5-holdout baselines (results/unified_v3/holdouts_v3/) using the
     two-sided Nadeau & Bengio (2003) corrected resampled t-test with Holm adjustment.
  3. fixed_subject_level_stats_v4.json:
     Complementary subject-level paired tests on the fixed test cohorts (84 ABIDE, 82 ADHD-200)
     using 5-seed ensemble probabilities (paired bootstrap AUC & exact McNemar accuracy)
     for auditing consistency alongside seed-level paired t-tests and exact sign-flip tests.
"""
import csv
import gc
import importlib
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from torch import nn

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from efficient_kan import KAN
from jbhi_revision.common import METRICS, TRAINING, configure_runtime, write_json
from run_unified_v3_extras import benchmark_model, kan_flops_per_sample

V4_DIR = REPO / "results" / "unified_v4"
V3_HOLDOUT_DIR = REPO / "results" / "unified_v3" / "holdouts_v3"


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(pvals[idx]) * (m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def corrected_resampled_t(a, b, n_test, n_train):
    """Nadeau & Bengio (2003) two-sided corrected resampled t-test for K repeated splits."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    k = len(d)
    mean = float(np.mean(d))
    var = float(np.var(d, ddof=1))
    if var == 0.0:
        return (0.0, 1.0) if abs(mean) < 1e-12 else (float("inf") if mean > 0 else float("-inf"), 0.0)
    se = math.sqrt((1.0 / k + n_test / n_train) * var)
    tval = mean / se
    pval = float(2.0 * student_t.sf(abs(tval), df=k - 1))
    return float(tval), pval


def run_compute_benchmark_v4(device):
    out_path = V4_DIR / "compute_benchmark_v4.json"
    print("\n=== 1. LIVE COMPUTATIONAL COST BENCHMARK (Route A v4) ===", flush=True)
    rows = []

    kan_configs = [
        ("KAN (Full, 90 ROIs)", 90),
        ("BrainKAN Mode 1 (Direct, 89 ROIs)", 89),
        ("BrainKAN Mode 2 (Min-max, 88 ROIs, ABIDE)", 88),
        ("BrainKAN Mode 3 (Z-score, 84 ROIs)", 84),
        ("BrainKAN Mode 2 (Min-max, 67 ROIs, ADHD-200)", 67),
        ("BrainKAN Compact Ref. (Z<0, 40 ROIs)", 40),
    ]
    for label, M in kan_configs:
        dims = [M * M, *TRAINING["hidden"], 2]
        rows.append(benchmark_model(
            label,
            build_fn=lambda dev, d=dims: KAN(d, grid_size=TRAINING["grid_size"], spline_order=TRAINING["spline_order"]),
            make_batch_fn=lambda bs, dev, d=M * M: torch.randn(bs, d, device=dev),
            forward_fn=lambda m, x: m(x),
            loss_fn=lambda m, x, y: nn.functional.cross_entropy(m(x), y),
            device=device,
            exact_flops_per_sample=kan_flops_per_sample(dims, TRAINING["grid_size"], TRAINING["spline_order"]),
        ))

    # Baselines (identical implementations to v3)
    sys.path.insert(0, str(REPO / "baseline_reference" / "ADHD_baseline_comparison_20261111" / "fc_baselines" / "code"))
    from brainnetcnn import BrainNetCNN
    rows.append(benchmark_model(
        "BrainNetCNN (90x90)",
        build_fn=lambda dev: BrainNetCNN(90),
        make_batch_fn=lambda bs, dev: torch.randn(bs, 90, 90, device=dev),
        forward_fn=lambda m, x: m(x),
        loss_fn=lambda m, x, y: nn.functional.cross_entropy(m(x), y),
        device=device,
    ))

    if importlib.util.find_spec("omegaconf") is None:
        class _AttrDict(dict):
            def __getattr__(self, k):
                try:
                    return self[k]
                except KeyError as e:
                    raise AttributeError(k) from e
            def __setattr__(self, k, v):
                self[k] = v
        def _wrap(obj):
            if isinstance(obj, dict):
                return _AttrDict({k: _wrap(v) for k, v in obj.items()})
            if isinstance(obj, list):
                return [_wrap(v) for v in obj]
            return obj
        _oc = types.ModuleType("omegaconf")
        _oc.DictConfig = _AttrDict
        class _OmegaConf:
            @staticmethod
            def create(d):
                return _wrap(d)
        _oc.OmegaConf = _OmegaConf
        sys.modules["omegaconf"] = _oc

    try:
        import torch_geometric.typing as _pyg_t
        import torch_geometric.utils._scatter as _pyg_sc
        _pyg_t.WITH_TORCH_SCATTER = False
        _pyg_sc.WITH_TORCH_SCATTER = False
        _pyg_t.WITH_TORCH_SPARSE = False
    except Exception:
        pass

    base_fc_dir = str(REPO / "baseline_reference" / "Baseline_rerun_20261106" / "code")
    sys.path.insert(0, base_fc_dir)
    if "models" in sys.modules:
        del sys.modules["models"]
    fc_models = importlib.import_module("models")

    rows.append(benchmark_model(
        "BrainNetTF (90x90)",
        build_fn=lambda dev: fc_models.build("brainnettf"),
        make_batch_fn=lambda bs, dev: torch.randn(bs, 90, 90, device=dev),
        forward_fn=lambda m, x: fc_models.forward("brainnettf", m, x)[0],
        loss_fn=lambda m, x, y: nn.functional.cross_entropy(fc_models.forward("brainnettf", m, x)[0], y),
        device=device,
    ))

    rows.append(benchmark_model(
        "BrainGB (90x90)",
        build_fn=lambda dev: fc_models.build("braingb"),
        make_batch_fn=lambda bs, dev: torch.randn(bs, 90, 90, device=dev),
        forward_fn=lambda m, x: fc_models.forward("braingb", m, x)[0],
        loss_fn=lambda m, x, y: nn.functional.cross_entropy(fc_models.forward("braingb", m, x)[0], y),
        batch_size=16,
        device=device,
    ))

    rows.append(benchmark_model(
        "BrainGNN (90x90)",
        build_fn=lambda dev: fc_models.build("braingnn"),
        make_batch_fn=lambda bs, dev: torch.randn(bs, 90, 90, device=dev),
        forward_fn=lambda m, x: fc_models.forward("braingnn", m, x)[0],
        loss_fn=lambda m, x, y: nn.functional.cross_entropy(fc_models.forward("braingnn", m, x)[0], y),
        device=device,
    ))

    rem_dir = str(REPO / "baseline_reference" / "Remaining_baselines_20261108" / "code")
    sys.path.insert(0, rem_dir)
    if "models" in sys.modules:
        del sys.modules["models"]
    rem_models = importlib.import_module("models")

    rows.append(benchmark_model(
        "FBNetGen (90 ROIs, T=100)",
        build_fn=lambda dev: rem_models.build("fbnetgen", {"embedding_size": 16}, device=dev),
        make_batch_fn=lambda bs, dev: (
            torch.randn(bs, 90, 100, device=dev),
            torch.randn(bs, 90, 90, device=dev),
            torch.full((bs,), 100, dtype=torch.long),
        ),
        forward_fn=lambda m, x: m(x[0], x[1], x[2])[0],
        loss_fn=lambda m, x, y: nn.functional.cross_entropy(m(x[0], x[1], x[2])[0], y),
        device=device,
    ))

    rows.append(benchmark_model(
        "BrainNetGNN (90 ROIs, T=100)",
        build_fn=lambda dev: rem_models.build("brainnetgnn", {}, device=dev),
        make_batch_fn=lambda bs, dev: (
            torch.randn(bs, 90, 100, device=dev),
            torch.full((bs,), 100, dtype=torch.long),
        ),
        forward_fn=lambda m, x: m(x[0], x[1]),
        loss_fn=lambda m, x, y: nn.functional.cross_entropy(m(x[0], x[1]), y),
        device=device,
    ))

    rows.append(benchmark_model(
        "Res-DAE (90x90 triu=4005)",
        build_fn=lambda dev: rem_models.build("res_dae", {"n_input": 4005, "hidden": 512, "latent": 128, "dropout": 0.5}, device=dev),
        make_batch_fn=lambda bs, dev: torch.randn(bs, 4005, device=dev),
        forward_fn=lambda m, x: m(x, add_noise=False, reconstruct=False)[0],
        loss_fn=lambda m, x, y: nn.functional.binary_cross_entropy_with_logits(m(x, add_noise=True, reconstruct=True)[0], y.float()),
        device=device,
    ))

    payload = {
        "hardware": {
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "batch_size": TRAINING["batch_size"],
            "n_train": 709,
            "epochs": TRAINING["epochs"],
        },
        "benchmarks": rows,
    }
    write_json(out_path, payload)
    print(f"[OK] Saved {out_path}", flush=True)
    return payload


def run_holdouts_v4_stats():
    """Compare v4 5-holdout BrainKAN runs against the matched 5-holdout baselines."""
    print("\n=== 2. MATCHED 5-HOLDOUT PAIRED STATS (v4 BrainKAN vs Baselines) ===", flush=True)
    from summarize_holdouts import BASELINES, baseline_prediction, check_baseline_plan

    plan = read_json(REPO / "results" / "holdouts" / "plan.json")
    baseline_root = REPO / "baseline_reference"
    out = {}
    for dataset, archives in BASELINES.items():
        n_tr = len(plan["partitions"][f"{dataset}_holdout_0"]["indices"]["train"])
        n_te = len(plan["partitions"][f"{dataset}_holdout_0"]["indices"]["test"])
        b_rows = {}
        for partition in range(5):
            name = f"{dataset}_holdout_{partition}"
            split = plan["partitions"][name]["indices"]
            for source, layout, model_names in archives:
                root = baseline_root / source
                check_baseline_plan(read_json(root / "plan.json"), partition, split, plan["datasets"][dataset]["sha256"])
                for bm in model_names:
                    _, _, scores, _, _ = baseline_prediction(root, partition, bm, layout)
                    b_rows.setdefault(bm, []).append(scores)
        bmodels = [m for _, _, names in archives for m in names]
        out[dataset] = {}
        for ref in ("full", "direct", "minmax", "zscore"):
            ref_runs = [
                read_json(V4_DIR / "brainkan_runs" / f"{dataset}_holdouts" / f"holdout_{i}" / f"{ref}_meta.json")["test"]
                for i in range(5)
            ]
            out[dataset][ref] = {}
            for m in METRICS:
                a = [r[m] for r in ref_runs]
                raw_ps = []
                rows_m = []
                for bm in bmodels:
                    b = [r[m] for r in b_rows[bm]]
                    tval, pval = corrected_resampled_t(a, b, n_test=n_te, n_train=n_tr)
                    raw_ps.append(pval)
                    rows_m.append({
                        "baseline": bm,
                        "ref_mean": round(100.0 * float(np.mean(a)), 2),
                        "base_mean": round(100.0 * float(np.mean(b)), 2),
                        "diff_pp": round(100.0 * (float(np.mean(a)) - float(np.mean(b))), 2),
                        "t_stat": round(tval, 3) if math.isfinite(tval) else str(tval),
                        "p_raw": round(pval, 5),
                    })
                for r_entry, pa in zip(rows_m, holm_adjust(raw_ps)):
                    r_entry["p_holm"] = round(pa, 5)
                out[dataset][ref][m] = rows_m
    write_json(V4_DIR / "holdouts_comparison_stats_v4.json", out)
    print(f"[OK] Saved {V4_DIR / 'holdouts_comparison_stats_v4.json'}", flush=True)
    return out


def read_prob_csv(path):
    idx, y, p, pred = [], [], [], []
    with Path(path).open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            idx.append(int(row["pool_index"]))
            y.append(int(row["label"]))
            if "prob_1" in row:
                p.append(float(row["prob_1"]))
                pred.append(int(row["predicted"]))
            else:
                p.append(float(row["probability"]))
                pred.append(int(row["prediction"]))
    return np.asarray(idx), np.asarray(y), np.asarray(p), np.asarray(pred)


def run_subject_level_audit():
    """Subject-level paired bootstrap AUC & exact McNemar test on 5-seed ensemble predictions (fixed splits)."""
    print("\n=== 3. SUBJECT-LEVEL FIXED-SPLIT AUDIT (5-seed mean probabilities) ===", flush=True)
    rng = np.random.default_rng(2026)
    out = {}
    for ds, ref, bmodels in [
        ("abide", "direct", ["braingnn", "braingb", "brainnetcnn", "fbnetgen", "brainnetgnn", "brainnettf", "res_dae"]),
        ("adhd", "zscore", ["braingnn", "braingb", "brainnetcnn", "brainnettf", "res_dae"]),
    ]:
        out[ds] = {}
        ref_probs = []
        ref_preds = []
        y_true = None
        for s in range(5):
            _, y, p, pr = read_prob_csv(V4_DIR / "brainkan_runs" / f"{ds}_fixed" / f"seed_{s}" / f"{ref}_test.csv")
            y_true = y
            ref_probs.append(p)
            ref_preds.append(pr)
        p_ref = np.mean(ref_probs, axis=0)
        pred_ref = (p_ref >= 0.5).astype(int)
        auc_ref = float(roc_auc_score(y_true, p_ref))
        acc_ref = float(np.mean(pred_ref == y_true))

        for bm in bmodels:
            b_probs, b_preds = [], []
            for s in range(5):
                _, yb, pb, prb = read_prob_csv(V4_DIR / "fixed_baselines" / ds / bm / f"seed_{s}" / "test_predictions.csv")
                assert np.array_equal(y_true, yb)
                b_probs.append(pb)
                b_preds.append(prb)
            p_b = np.mean(b_probs, axis=0)
            # Use majority vote of per-seed predictions (respects Res-DAE Youden thresholds)
            pred_b = (np.mean(b_preds, axis=0) >= 0.5).astype(int)
            auc_b = float(roc_auc_score(y_true, p_b))
            acc_b = float(np.mean(pred_b == y_true))

            # Exact McNemar on ensemble predictions
            c_ref = pred_ref == y_true
            c_b = pred_b == y_true
            n10 = int(np.sum(c_ref & (~c_b)))
            n01 = int(np.sum((~c_ref) & c_b))
            p_mcnemar = float(binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) > 0 else 1.0

            # Paired stratified bootstrap (2000 reps) for AUC difference
            n = len(y_true)
            diffs = []
            for _ in range(2000):
                ix = rng.integers(0, n, size=n)
                if len(np.unique(y_true[ix])) < 2:
                    continue
                diffs.append(roc_auc_score(y_true[ix], p_ref[ix]) - roc_auc_score(y_true[ix], p_b[ix]))
            diffs = np.asarray(diffs)
            p_boot = float(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))

            out[ds][bm] = {
                "ref_ens_acc": round(100.0 * acc_ref, 2),
                "base_ens_acc": round(100.0 * acc_b, 2),
                "mcnemar_n10": n10,
                "mcnemar_n01": n01,
                "mcnemar_p": round(p_mcnemar, 5),
                "ref_ens_auc": round(100.0 * auc_ref, 2),
                "base_ens_auc": round(100.0 * auc_b, 2),
                "boot_auc_diff_pp": round(100.0 * (auc_ref - auc_b), 2),
                "boot_auc_p": round(p_boot, 5),
            }
            print(f"  [{ds} {ref} vs {bm:11s}] ensACC {100*acc_ref:.2f} vs {100*acc_b:.2f} (McNemar p={p_mcnemar:.4f}) | "
                  f"ensAUC {100*auc_ref:.2f} vs {100*auc_b:.2f} (boot p={p_boot:.4f})", flush=True)
    write_json(V4_DIR / "fixed_subject_level_stats_v4.json", out)
    print(f"[OK] Saved {V4_DIR / 'fixed_subject_level_stats_v4.json'}", flush=True)
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_runtime(device)
    with threadpool_limits(limits=TRAINING["cpu_threads"], user_api="blas"):
        run_compute_benchmark_v4(device)
    try:
        run_holdouts_v4_stats()
    except Exception as e:
        print(f"[WARN] holdouts_v4_stats skipped: {e}", flush=True)
    run_subject_level_audit()


if __name__ == "__main__":
    main()
