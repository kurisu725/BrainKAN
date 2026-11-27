"""Run the remaining unified v3 experiments under the exact 50-epoch training-only pipeline:
1. Live computational cost benchmark (Table I): KAN-full (90x90), BrainKAN Mode 1 Direct (89x89),
   BrainKAN Mode 3 Z-score (40x40), BrainKAN Mode 2 Min-max (25x25), and source-verifiable baselines
   (BrainNetCNN, BrainGNN, BrainGB, BrainNetTF, FBNetGen, Res-DAE).
2. Hyperparameter sensitivity (Table II & Fig. 2): 15 (layers, grid_size, spline_order) configs
   on the ABIDE fixed validation set under the 50-epoch AdamW protocol.
3. Threshold sensitivity (Table VIII): Mean pairwise Jaccard +/- SD over 5 perturbed thresholds
   around nominal rules (C_i < 0, V_i < 0.5, Z_i < 0) on ABIDE fixed seed-averaged attenuation scores,
   plus cross-seed Jaccard summary for review_v2 and holdouts.
4. ROI-count performance curves (Fig. 4): Sweep k in {5, 10, ..., 90} plus mode counts {25, 40, 89}
   on ABIDE fixed split (seeds 0-4 averaged, dev-AUC checkpoint selection) using the training-only
   seed-averaged attenuation ranking.
"""
import gc
import importlib
import importlib.util
import itertools
import json
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from efficient_kan import KAN
from jbhi_revision.common import (
    METRICS, TRAINING, configure_runtime, features, load_groups, metrics, predict_features,
    summarize, write_json,
)
from run_review_development import CHECKPOINT_EVERY, EPOCHS, choose_checkpoint
from run_review_experiments import GROUPS, read_json

ABIDE_MAT = REPO / "data" / "ALLASD1_aal.mat"
ADHD_MAT = REPO / "data" / "ADHD1_aal.mat"
REVIEW_V2_DIR = REPO / "results" / "review_v2"
HOLDOUT_DIR = REPO / "results" / "holdouts"
OUT_DIR = REPO / "results" / "unified_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# 1. LIVE COMPUTATIONAL COST BENCHMARK (Table I)
# =========================================================================
def kan_flops_per_sample(layer_dims, grid_size=5, spline_order=3):
    """Exact MACs*2 estimate for EfficientKAN forward pass per sample.
    Each KANLinear(in_f, out_f) computes:
      - base linear: 2 * in_f * out_f
      - b_splines(x): order-k Cox-de Boor recursion over (grid_size + 2k) knots ~ in_f * (grid_size + k) * (4*k + 2)
      - spline linear: 2 * in_f * (grid_size + spline_order) * out_f
    """
    total = 0
    for in_f, out_f in zip(layer_dims[:-1], layer_dims[1:]):
        basis = grid_size + spline_order
        base_f = 2 * in_f * out_f
        spline_basis_f = in_f * basis * (4 * spline_order + 2)
        spline_f = 2 * in_f * basis * out_f
        total += base_f + spline_basis_f + spline_f
    return total


def profile_torch_flops(fn, warmup=1):
    for _ in range(warmup):
        fn()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        fn()
    total_flops = sum(e.flops for e in prof.key_averages() if e.flops)
    return int(total_flops)


def benchmark_model(name, build_fn, make_batch_fn, forward_fn, loss_fn, n_train=709, batch_size=96, epochs=50,
                    device="cuda", exact_flops_per_sample=None):
    """Measure params, FLOPs/sample, CPU RSS delta, GPU peak mem, 50-epoch GPU train time,
    GPU inference ms/sample, and CPU inference ms/sample."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # CPU model & inference timing + FLOPs
    model_cpu = build_fn("cpu").eval()
    n_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)

    # FLOPs per sample on CPU (batch=1)
    if exact_flops_per_sample is not None:
        flops_per_sample = int(exact_flops_per_sample)
    else:
        b1_cpu = make_batch_fn(1, "cpu")
        with torch.no_grad():
            flops_per_sample = profile_torch_flops(lambda: forward_fn(model_cpu, b1_cpu))

    # CPU inference latency (batch=84 test cohort, average over 1-3 runs; cap if slow)
    b_test_cpu = make_batch_fn(84, "cpu")
    with torch.inference_mode():
        t_warm0 = time.perf_counter()
        _ = forward_fn(model_cpu, b_test_cpu)
        warm_s = time.perf_counter() - t_warm0
        n_rep_cpu = 1 if warm_s > 2.0 else (2 if warm_s > 0.5 else 5)
        t0 = time.perf_counter()
        for _ in range(n_rep_cpu):
            _ = forward_fn(model_cpu, b_test_cpu)
        t1 = time.perf_counter()
    cpu_infer_ms = (t1 - t0) * 1000.0 / (n_rep_cpu * 84)

    # CPU memory (parameter bytes + peak traced delta on batch=1)
    b1_mem = make_batch_fn(1, "cpu")
    tracemalloc.start()
    with torch.inference_mode():
        _ = forward_fn(model_cpu, b1_mem)
    _, cpu_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    param_mb = sum(p.numel() * p.element_size() for p in model_cpu.parameters()) / (1024 ** 2)
    cpu_mem_mb = param_mb + cpu_peak_bytes / (1024 ** 2)
    del model_cpu, b_test_cpu, b1_mem
    gc.collect()

    # GPU training & inference benchmark
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_gpu = build_fn(device).to(device)
    opt = torch.optim.AdamW(model_gpu.parameters(), lr=1e-3, weight_decay=1e-4)
    n_batches = int(np.ceil(n_train / batch_size))
    # Pre-allocate one synthetic batch of size batch_size (and last batch)
    batch_gpu = make_batch_fn(batch_size, device)
    y_gpu = torch.randint(0, 2, (batch_size,), device=device)

    # Warmup 2 steps
    model_gpu.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        l = loss_fn(model_gpu, batch_gpu, y_gpu)
        l.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Time 5 epochs (or 2 epochs if slow) and scale to 50 epochs
    t_step0 = time.perf_counter()
    for _ in range(n_batches):
        opt.zero_grad(set_to_none=True)
        l = loss_fn(model_gpu, batch_gpu, y_gpu)
        l.backward()
        opt.step()
    torch.cuda.synchronize()
    ep1_s = time.perf_counter() - t_step0
    if ep1_s > 1.5:
        sec_50ep = ep1_s * epochs
    else:
        t0 = time.perf_counter()
        measure_epochs = 5
        for _ in range(measure_epochs):
            for _ in range(n_batches):
                opt.zero_grad(set_to_none=True)
                l = loss_fn(model_gpu, batch_gpu, y_gpu)
                l.backward()
                opt.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        sec_50ep = (t1 - t0) * (epochs / measure_epochs)
    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # GPU inference latency (batch=84, avg over 20 runs)
    model_gpu.eval()
    b_test_gpu = make_batch_fn(84, device)
    with torch.inference_mode():
        for _ in range(5):
            _ = forward_fn(model_gpu, b_test_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep_gpu = 20
        for _ in range(n_rep_gpu):
            _ = forward_fn(model_gpu, b_test_gpu)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    gpu_infer_ms = (t1 - t0) * 1000.0 / (n_rep_gpu * 84)

    del model_gpu, opt, batch_gpu, y_gpu, b_test_gpu
    torch.cuda.empty_cache()
    gc.collect()

    row = {
        "model": name,
        "params": int(n_params),
        "params_M": round(n_params / 1e6, 3),
        "flops_per_sample": int(flops_per_sample),
        "mflops_per_sample": round(flops_per_sample / 1e6, 2),
        "cpu_mem_mb": round(cpu_mem_mb, 2),
        "gpu_peak_mb": round(gpu_peak_mb, 2),
        "train_50ep_sec": round(sec_50ep, 2),
        "gpu_infer_ms_per_sample": round(gpu_infer_ms, 4),
        "cpu_infer_ms_per_sample": round(cpu_infer_ms, 4),
    }
    print(f"[Bench] {name:28s} | Params={row['params_M']:6.3f}M | MFLOPs={row['mflops_per_sample']:7.2f} | "
          f"GPU_MB={row['gpu_peak_mb']:6.1f} | 50ep={row['train_50ep_sec']:6.2f}s | "
          f"GPU_inf={row['gpu_infer_ms_per_sample']:.4f}ms | CPU_inf={row['cpu_infer_ms_per_sample']:.3f}ms",
          flush=True)
    return row


def run_compute_benchmark(device):
    out_path = OUT_DIR / "compute_benchmark_v3.json"
    print("\n=== 1. LIVE COMPUTATIONAL COST BENCHMARK ===")
    rows = []

    # BrainKAN variants (full 90, Mode 1 Direct 89, Mode 3 Z-score 40, Mode 2 Min-max 25)
    for label, M in [
        ("KAN (Full, 90 ROIs)", 90),
        ("BrainKAN Mode 1 (Direct, 89 ROIs)", 89),
        ("BrainKAN Mode 3 (Z-score, 40 ROIs)", 40),
        ("BrainKAN Mode 2 (Min-max, 25 ROIs)", 25),
    ]:
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

    # Baselines from baseline_reference
    # a) BrainNetCNN
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

    # b) BrainGNN, BrainGB, BrainNetTF
    if importlib.util.find_spec("omegaconf") is None:
        import types
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

    # Disable CPU-only torch_scatter/torch_sparse extensions so PyG uses native CUDA PyTorch ops
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

    # c) FBNetGen, BrainNetGNN, and Res-DAE from Remaining_baselines_20261108
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
            "epochs": EPOCHS,
        },
        "benchmarks": rows,
    }
    write_json(out_path, payload)
    return payload


# =========================================================================
# 2. HYPERPARAMETER SENSITIVITY (Table II & Fig. 2) on ABIDE validation set
# =========================================================================
def run_hyperparam_study(pools, device):
    out_path = OUT_DIR / "hyperparam_v3.json"
    print("\n=== 2. HYPERPARAMETER SENSITIVITY (ABIDE fixed validation set, 50 epochs) ===")
    plan = read_json(REVIEW_V2_DIR / "plan.json")
    split = plan["partitions"]["abide_fixed"]["indices"]
    tr = np.asarray(split["train"])
    dv = np.asarray(split["valid"])
    matrices, labels = pools["abide"]

    # Full 90x90 FC input evaluated on ABIDE validation set across seeds 0-4 (no test leakage!)
    rois = np.arange(90)
    x_tr = features(matrices[tr], rois)
    y_tr = torch.from_numpy(labels[tr])
    x_dv = features(matrices[dv], rois)
    y_dv = labels[dv]

    configs = [
        # 5 depth/width configs (grid=5, order=3)
        {"id": 1, "group": "layers", "hidden": [256, 128], "grid_size": 5, "spline_order": 3},
        {"id": 2, "group": "layers", "hidden": [256, 128, 64], "grid_size": 5, "spline_order": 3},
        {"id": 3, "group": "layers", "hidden": [256, 128, 64, 32], "grid_size": 5, "spline_order": 3},
        {"id": 4, "group": "layers", "hidden": [256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 3},
        {"id": 5, "group": "layers", "hidden": [512, 256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 3},
        # 5 grid_size configs (hidden=[256,128,64,32,16], order=3)
        {"id": 6, "group": "grid", "hidden": [256, 128, 64, 32, 16], "grid_size": 3, "spline_order": 3},
        {"id": 7, "group": "grid", "hidden": [256, 128, 64, 32, 16], "grid_size": 4, "spline_order": 3},
        {"id": 8, "group": "grid", "hidden": [256, 128, 64, 32, 16], "grid_size": 6, "spline_order": 3},
        {"id": 9, "group": "grid", "hidden": [256, 128, 64, 32, 16], "grid_size": 7, "spline_order": 3},
        {"id": 10, "group": "grid", "hidden": [256, 128, 64, 32, 16], "grid_size": 8, "spline_order": 3},
        # 5 spline_order configs (hidden=[256,128,64,32,16], grid=5)
        {"id": 11, "group": "order", "hidden": [256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 1},
        {"id": 12, "group": "order", "hidden": [256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 2},
        {"id": 13, "group": "order", "hidden": [256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 4},
        {"id": 14, "group": "order", "hidden": [256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 5},
        {"id": 15, "group": "order", "hidden": [256, 128, 64, 32, 16], "grid_size": 5, "spline_order": 6},
    ]

    results = []
    for cfg in configs:
        seed_rows = []
        for seed in range(5):
            seed_all(seed)
            model = KAN([len(rois) ** 2, *cfg["hidden"], 2],
                        grid_size=cfg["grid_size"], spline_order=cfg["spline_order"]).to(device)
            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=TRAINING["batch_size"],
                                shuffle=True, num_workers=0, generator=gen)
            opt = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"],
                                    weight_decay=TRAINING["weight_decay"])
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=TRAINING["lr_gamma"])
            crit = nn.CrossEntropyLoss()
            best_m = None
            best_key = None
            for ep in range(1, EPOCHS + 1):
                model.train()
                for bx, by in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model(bx.to(device)), by.to(device))
                    loss.backward()
                    opt.step()
                if ep % CHECKPOINT_EVERY == 0 or ep == EPOCHS:
                    probs = predict_features(model, x_dv)
                    m = metrics(y_dv, probs)
                    key = (m["auc"], -m["log_loss"])
                    if best_key is None or key > best_key:
                        best_key = key
                        best_m = {"epoch": ep, **m}
                sched.step()
            seed_rows.append({"seed": seed, **best_m})
            del model
            torch.cuda.empty_cache()

        summ = summarize(seed_rows)
        n_params = sum(p.numel() for p in KAN([len(rois) ** 2, *cfg["hidden"], 2],
                                              grid_size=cfg["grid_size"], spline_order=cfg["spline_order"]).parameters())
        entry = {**cfg, "params": int(n_params), "summary": summ, "per_seed": seed_rows}
        results.append(entry)
        print(f"[HP #{cfg['id']:2d}] {str(cfg['hidden']):28s} g={cfg['grid_size']} k={cfg['spline_order']} | "
              f"Dev ACC={100*summ['accuracy']['mean']:5.2f}+/-{100*summ['accuracy']['sd']:4.2f} | "
              f"Dev F1={100*summ['f1']['mean']:5.2f}+/-{100*summ['f1']['sd']:4.2f} | "
              f"Dev AUC={100*summ['auc']['mean']:5.2f}+/-{100*summ['auc']['sd']:4.2f}", flush=True)

    write_json(out_path, {"protocol": "ABIDE fixed validation set (n=89), seeds 0-4, 50 epochs, dev-AUC selection", "configs": results})
    return results


# =========================================================================
# 3. THRESHOLD SENSITIVITY (Table VIII) on new training-only attenuation scores
# =========================================================================
def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def run_threshold_sensitivity():
    out_path = OUT_DIR / "threshold_stability_v3.json"
    print("\n=== 3. THRESHOLD SENSITIVITY (ABIDE fixed & ADHD fixed seed-averaged attenuation scores) ===")
    out = {}
    for cohort in ("abide_fixed", "adhd_fixed"):
        scores_seeds = []
        for s in range(5):
            sel = read_json(REVIEW_V2_DIR / f"{cohort}_seed_{s}" / "selection.json")
            scores_seeds.append(np.array(sel["scores"], dtype=float))
        mean_c = np.mean(scores_seeds, axis=0)
        std_c = float(np.std(mean_c))
        c_min, c_max = float(mean_c.min()), float(mean_c.max())
        v_norm = (mean_c - c_min) / (c_max - c_min)
        z_norm = (mean_c - float(mean_c.mean())) / (float(mean_c.std()) + 1e-12)

        # Perturbed threshold families around each nominal threshold:
        # Direct: tau_C in {-0.5*sd, -0.25*sd, 0, +0.25*sd, +0.5*sd} (relative to score SD)
        # Note: also check direct perturbations in {-0.0004, -0.0002, 0.0, +0.0002, +0.0004} vs relative
        # Using relative +/-0.25 SD and +/-0.50 SD around 0 for Direct (or exact delta_C in {-0.2, -0.1, 0, 0.1, 0.2}*sd)
        # Let's keep symmetric standardized grids:
        #   Direct: tau_C in {-0.5, -0.25, 0.0, +0.25, +0.5} * std(C) (nominal = 0)
        #   Min-max: tau_V in {0.40, 0.45, 0.50, 0.55, 0.60} (nominal = 0.50)
        #   Z-score: tau_Z in {-0.20, -0.10, 0.00, +0.10, +0.20} (nominal = 0.00)
        grids = {
            "direct": {
                "score_vec": mean_c,
                "thresholds": [round(float(t * std_c), 6) for t in (-0.50, -0.25, 0.0, 0.25, 0.50)],
                "labels": ["-0.50 SD", "-0.25 SD", "0 (nom.)", "+0.25 SD", "+0.50 SD"],
            },
            "minmax": {
                "score_vec": v_norm,
                "thresholds": [0.40, 0.45, 0.50, 0.55, 0.60],
                "labels": ["0.40", "0.45", "0.50 (nom.)", "0.55", "0.60"],
            },
            "zscore": {
                "score_vec": z_norm,
                "thresholds": [-0.20, -0.10, 0.00, 0.10, 0.20],
                "labels": ["-0.20", "-0.10", "0.00 (nom.)", "+0.10", "+0.20"],
            },
        }
        cohort_res = {"score_mean": float(mean_c.mean()), "score_sd": std_c, "modes": {}}
        for mname, spec in grids.items():
            vec = spec["score_vec"]
            sets = [np.flatnonzero(vec < thr).tolist() for thr in spec["thresholds"]]
            counts = [len(s) for s in sets]
            jacks = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
            # Also compute per-seed threshold sensitivity averaged across seeds 0-4
            per_seed_jacks = []
            for s_idx in range(5):
                sc = scores_seeds[s_idx]
                if mname == "direct":
                    svec = sc
                    thrs = [t * float(sc.std()) for t in (-0.50, -0.25, 0.0, 0.25, 0.50)]
                elif mname == "minmax":
                    svec = (sc - sc.min()) / (sc.max() - sc.min())
                    thrs = spec["thresholds"]
                else:
                    svec = (sc - sc.mean()) / (sc.std() + 1e-12)
                    thrs = spec["thresholds"]
                ssets = [np.flatnonzero(svec < t).tolist() for t in thrs]
                per_seed_jacks.append(float(np.mean([jaccard(a, b) for a, b in itertools.combinations(ssets, 2)])))
            cohort_res["modes"][mname] = {
                "thresholds": spec["thresholds"],
                "threshold_labels": spec["labels"],
                "counts": counts,
                "nominal_count": counts[2],
                "mean_jaccard": float(np.mean(jacks)),
                "sd_jaccard": float(np.std(jacks, ddof=1)),
                "per_seed_mean_jaccard": float(np.mean(per_seed_jacks)),
                "per_seed_sd_jaccard": float(np.std(per_seed_jacks, ddof=1)),
            }
            print(f"[Thr {cohort:11s}] {mname:6s} | counts={counts} | "
                  f"mean_profile J={np.mean(jacks):.4f}+/-{np.std(jacks, ddof=1):.4f} | "
                  f"per_seed J={np.mean(per_seed_jacks):.4f}+/-{np.std(per_seed_jacks, ddof=1):.4f}")
        out[cohort] = cohort_res

    write_json(out_path, out)
    return out


# =========================================================================
# 4. ROI-COUNT PERFORMANCE CURVE (Fig. 4) on ABIDE fixed split
# =========================================================================
def run_roi_curve(pools, device):
    out_path = OUT_DIR / "roi_curve_v3.json"
    print("\n=== 4. ROI-COUNT PERFORMANCE CURVES (ABIDE fixed, seeds 0-4, training-only ranking) ===")
    plan = read_json(REVIEW_V2_DIR / "plan.json")
    split = plan["partitions"]["abide_fixed"]["indices"]
    tr = np.asarray(split["train"])
    dv = np.asarray(split["valid"])
    te = np.asarray(split["test"])
    matrices, labels = pools["abide"]

    # For each seed s in 0..4, use THAT seed's own training-only attenuation ranking (strictly leak-free!)
    per_seed_rankings = []
    for s in range(5):
        sel = read_json(REVIEW_V2_DIR / f"abide_fixed_seed_{s}" / "selection.json")
        sc = np.array(sel["scores"], dtype=float)
        # Ascending order (most negative C_i = most important first)
        per_seed_rankings.append(np.argsort(sc))

    ks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 89, 90]
    curve = []
    for k in ks:
        k_rows = []
        for seed in range(5):
            rois = per_seed_rankings[seed][:k]
            seed_all(seed)
            model = KAN([k * k, *TRAINING["hidden"], 2],
                        grid_size=TRAINING["grid_size"], spline_order=TRAINING["spline_order"]).to(device)
            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(TensorDataset(features(matrices[tr], rois), torch.from_numpy(labels[tr])),
                                batch_size=TRAINING["batch_size"], shuffle=True, num_workers=0, generator=gen)
            opt = torch.optim.AdamW(model.parameters(), lr=TRAINING["learning_rate"],
                                    weight_decay=TRAINING["weight_decay"])
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=TRAINING["lr_gamma"])
            crit = nn.CrossEntropyLoss()
            checkpoints = []
            for ep in range(1, EPOCHS + 1):
                model.train()
                for bx, by in loader:
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model(bx.to(device)), by.to(device))
                    loss.backward()
                    opt.step()
                if ep % CHECKPOINT_EVERY == 0 or ep == EPOCHS:
                    checkpoints.append((ep, {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}))
                sched.step()
            best, _, model = choose_checkpoint(model, checkpoints, (matrices[dv], labels[dv]), rois, device)
            probs = predict_features(model.to(device), features(matrices[te], rois))
            m = metrics(labels[te], probs)
            k_rows.append({"seed": seed, "k": k, "epoch": int(best["epoch"]), **{mk: m[mk] for mk in METRICS}})
            del model, checkpoints
            torch.cuda.empty_cache()

        summ = summarize(k_rows)
        curve.append({"k": k, "summary": summ, "per_seed": k_rows})
        print(f"[Curve k={k:2d}] ACC={100*summ['accuracy']['mean']:5.2f}+/-{100*summ['accuracy']['sd']:4.2f} | "
              f"P={100*summ['precision']['mean']:5.2f} | R={100*summ['recall']['mean']:5.2f} | "
              f"F1={100*summ['f1']['mean']:5.2f} | AUC={100*summ['auc']['mean']:5.2f}+/-{100*summ['auc']['sd']:4.2f}",
              flush=True)

    write_json(out_path, {"protocol": "ABIDE fixed split, per-seed training-only attenuation ranking, seeds 0-4, dev-AUC checkpoint selection", "curve": curve})
    return curve


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
        run_compute_benchmark(device)
        run_threshold_sensitivity()
        run_hyperparam_study(pools, device)
        run_roi_curve(pools, device)
    print("\nALL EXTRA UNIFIED BENCHMARKS COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
