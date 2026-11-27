"""Run all matched baselines on the original fixed splits (seeds 0-4) for ABIDE and ADHD-200.

Uses the exact official baseline adapters and hyperparameters from baseline_reference/:
  - ABIDE fixed (709/89/84, seeds 0-4): BrainNetCNN, BrainGNN, Res-DAE, BrainNetGNN, BrainNetTF, BrainGB, FBNetGen
  - ADHD-200 fixed (679/86/82, seeds 0-4): BrainNetCNN, BrainGNN, Res-DAE, BrainNetTF, BrainGB
    (FBNetGen and BrainNetGNN require regional BOLD time series unavailable for ADHD-200.)

Results and per-subject test probabilities are saved atomically after every (dataset, model, seed)
under results/unified_v4/fixed_baselines/ so execution is fully resumable.
"""
import csv
import importlib.util
import json
import math
import random
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import torch_geometric.typing as _pyg_t
import torch_geometric.utils._scatter as _pyg_sc
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score

warnings.filterwarnings("ignore", message="The usage of `scatter")

_pyg_t.WITH_TORCH_SCATTER = False
_pyg_sc.WITH_TORCH_SCATTER = False
_pyg_t.WITH_TORCH_SPARSE = False

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

REPO = Path(__file__).resolve().parent
FC_CODE = REPO / "baseline_reference" / "ADHD_baseline_comparison_20261111" / "fc_baselines" / "code"
REM_CODE = REPO / "baseline_reference" / "Remaining_baselines_20261108" / "code"
ADHD_RES_CODE = REPO / "baseline_reference" / "ADHD_baseline_comparison_20261111" / "res_dae" / "code"

ABIDE_NPZ = REPO.parent / "Remaining_baselines_20261108" / "data" / "abide_dparsf.npz"
ADHD_NPZ = REPO.parent / "ADHD_baseline_comparison_20261111" / "res_dae" / "data" / "adhd_fc_metadata.npz"
NEUROCOMBAT_SITE = str(REPO.parent / "Remaining_baselines_20261108" / "venv" / "Lib" / "site-packages")

OUT_DIR = REPO / "results" / "unified_v4" / "fixed_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calc_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], average="macro", zero_division=0)
    probs2 = np.column_stack([1.0 - prob, prob])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(roc_auc_score(y, prob)),
        "log_loss": float(log_loss(y, np.clip(probs2, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def save_csv(path, indices, labels, probs, thr=0.5):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pool_index", "label", "probability", "prediction"])
        for idx, y, p in zip(indices, labels, probs):
            w.writerow([int(idx), int(y), f"{float(p):.8f}", int(p >= thr)])
    tmp.replace(path)


def write_json_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def summarize_runs(rows):
    out = {}
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        vals = [100.0 * r[k] for r in rows]
        out[k] = {
            "mean": round(float(np.mean(vals)), 2),
            "sd": round(float(np.std(vals, ddof=1)), 2),
            "values": [round(v, 2) for v in vals],
        }
    return out


def choose_youden(y, p):
    def key(t):
        pred = p >= t
        j = pred[y == 1].mean() - pred[y == 0].mean()
        return (-float(j), abs(t - 0.5), t)
    return float(min(np.linspace(0, 1, 11), key=key))


def run_fc_baseline(dataset, name, x_fc, y_all, split, fc_models, device):
    cfgs = {
        "brainnetcnn": {"epochs": 50, "batch_size": 96, "lr": 1e-3, "weight_decay": 1e-4, "drop_last": False},
        "braingnn": {"epochs": 100, "batch_size": 100, "lr": 0.01, "weight_decay": 0.005, "drop_last": False},
        "brainnettf": {"epochs": 200, "batch_size": 16, "lr": 1e-4, "weight_decay": 1e-4, "drop_last": True},
        "braingb": {"epochs": 100, "batch_size": 16, "lr": 1e-4, "weight_decay": 1e-4, "drop_last": False},
    }
    cfg = cfgs[name]
    tr = np.asarray(split["train"], dtype=int)
    dv = np.asarray(split["valid"], dtype=int)
    te = np.asarray(split["test"], dtype=int)
    bs = cfg["batch_size"]
    n_steps = len(tr) // bs if cfg["drop_last"] else math.ceil(len(tr) / bs)

    @torch.no_grad()
    def predict_idx(model, idx, batch=32):
        model.eval()
        ps = []
        for start in range(0, len(idx), batch):
            out, _ = fc_models.forward(name, model, x_fc[idx[start:start + batch]].to(device))
            ps.append(out.softmax(1)[:, 1].cpu().numpy())
        return np.concatenate(ps)

    runs = []
    for seed in SEEDS:
        run_dir = OUT_DIR / dataset / name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(meta["metrics"])
            print(f"[SKIP] {dataset:5s} {name:11s} seed={seed}: ACC={100*meta['metrics']['accuracy']:.2f}% AUC={100*meta['metrics']['auc']:.2f}%", flush=True)
            continue

        t0 = time.monotonic()
        seed_all(seed)
        model = fc_models.build(name).to(device)
        opt_cls = torch.optim.AdamW if name == "brainnetcnn" else torch.optim.Adam
        opt = opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        if name == "brainnetcnn":
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)
        elif name == "braingnn":
            sched = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.5)
        elif name == "brainnettf":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * n_steps, eta_min=1e-5)
        else:
            sched = None

        for ep in range(1, cfg["epochs"] + 1):
            model.train()
            order = torch.randperm(len(tr)).numpy()
            for offset in range(0, len(tr), bs):
                ix = tr[order[offset:offset + bs]]
                if cfg["drop_last"] and len(ix) < bs:
                    continue
                opt.zero_grad(set_to_none=True)
                loss = fc_models.train_loss(name, model, x_fc[ix].to(device), y_all[ix].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{dataset} {name} seed={seed} ep={ep} nonfinite loss")
                loss.backward()
                opt.step()
                if name == "brainnettf":
                    sched.step()
            if name in ("braingnn", "brainnetcnn"):
                sched.step()

        p_dv = predict_idx(model, dv)
        p_te = predict_idx(model, te)
        m_dv = calc_metrics(y_all[dv].numpy(), p_dv)
        m_te = calc_metrics(y_all[te].numpy(), p_te)
        sec = round(time.monotonic() - t0, 1)

        save_csv(run_dir / "test_predictions.csv", te, y_all[te].numpy(), p_te)
        meta = {
            "dataset": dataset,
            "model": name,
            "seed": seed,
            "protocol": "official_last_epoch",
            "config": cfg,
            "seconds": sec,
            "dev_metrics": m_dv,
            "metrics": m_te,
        }
        write_json_atomic(meta_path, meta)
        runs.append(m_te)
        print(
            f"[DONE] {dataset:5s} {name:11s} seed={seed} ({sec:6.1f}s) -> "
            f"ACC={100*m_te['accuracy']:.2f}% F1={100*m_te['f1']:.2f}% AUC={100*m_te['auc']:.2f}%",
            flush=True,
        )
        del model, opt, sched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summarize_runs(runs)


def run_res_dae(dataset, data, split, rem_models, prep_mod, device):
    tr = np.asarray(split["train"], dtype=int)
    dv = np.asarray(split["valid"], dtype=int)
    te = np.asarray(split["test"], dtype=int)

    # Fit preprocessor once on training split (seed=1, matching official experiment.py)
    all_done = all((OUT_DIR / dataset / "res_dae" / f"seed_{s}" / "meta.json").exists() for s in SEEDS)
    if all_done:
        runs = []
        for s in SEEDS:
            meta = json.loads((OUT_DIR / dataset / "res_dae" / f"seed_{s}" / "meta.json").read_text(encoding="utf-8"))
            runs.append(meta["metrics"])
            print(f"[SKIP] {dataset:5s} res_dae     seed={s}: ACC={100*meta['metrics']['accuracy']:.2f}% AUC={100*meta['metrics']['auc']:.2f}%", flush=True)
        return summarize_runs(runs)

    print(f"[PREP] {dataset:5s} res_dae preprocessor...", flush=True)
    if dataset == "abide":
        age_col, sex_col = 3, 4
    else:
        # Check ADHD phenotypes columns: in ADHD_baseline_comparison_20261111/res_dae/code/experiment.py it uses phenotypes[:,3], phenotypes[:,4]
        age_col, sex_col = 3, 4

    tr_x, tr_y, prep = prep_mod.fit_res_preprocessor(
        data["fc"][tr], data["labels"][tr], data["sites"][tr],
        data["phenotypes"][tr, age_col], data["phenotypes"][tr, sex_col], tr,
    )
    dv_x = prep_mod.apply_res_preprocessor(
        data["fc"][dv], data["sites"][dv], data["phenotypes"][dv, age_col], data["phenotypes"][dv, sex_col], prep
    )
    te_x = prep_mod.apply_res_preprocessor(
        data["fc"][te], data["sites"][te], data["phenotypes"][te, age_col], data["phenotypes"][te, sex_col], prep
    )
    dv_y = data["labels"][dv]
    te_y = data["labels"][te]

    cands = [
        {"n_input": int(tr_x.shape[1]), "hidden": h, "latent": h // 2, "dropout": 0.5,
         "lr": lr, "epochs": 250, "joint_epochs": 200, "batch_size": 100, "weight_decay": 1e-4}
        for h in (256, 512)
        for lr in (1e-4, 1e-3)
    ]

    @torch.no_grad()
    def predict_res(model, x_arr):
        model.eval()
        out = []
        for s in range(0, len(x_arr), 100):
            a = torch.as_tensor(x_arr[s:s + 100], device=device)
            logit = model(a, False, False)[0]
            out.append(logit.sigmoid().cpu().numpy())
        return np.concatenate(out).astype(float)

    runs = []
    for seed in SEEDS:
        run_dir = OUT_DIR / dataset / "res_dae" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(meta["metrics"])
            print(f"[SKIP] {dataset:5s} res_dae     seed={seed}: ACC={100*meta['metrics']['accuracy']:.2f}% AUC={100*meta['metrics']['auc']:.2f}%", flush=True)
            continue

        t0 = time.monotonic()
        best_cand = None
        best_key = None
        for ci, cfg in enumerate(cands):
            seed_all(seed)
            model = rem_models.build("res_dae", cfg, device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            for ep in range(1, cfg["epochs"] + 1):
                model.train()
                if ep > cfg["joint_epochs"]:
                    model.freeze_encoder()
                order = np.random.permutation(len(tr_y))
                for start in range(0, len(order), cfg["batch_size"]):
                    ids = order[start:start + cfg["batch_size"]]
                    xb = torch.as_tensor(tr_x[ids], device=device)
                    yb = torch.as_tensor(tr_y[ids], device=device)
                    opt.zero_grad(set_to_none=True)
                    joint = ep <= cfg["joint_epochs"]
                    logit, recon = model(xb, add_noise=joint, reconstruct=joint)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb.float())
                    if joint:
                        loss = loss + torch.nn.functional.mse_loss(recon, xb)
                    loss.backward()
                    opt.step()
            p_dv = predict_res(model, dv_x)
            thr = choose_youden(dv_y, p_dv)
            m_dv = calc_metrics(dv_y, p_dv, thr)
            ckey = (m_dv["auc"], -m_dv["log_loss"], -ci)
            if best_key is None or ckey > best_key:
                best_key = ckey
                best_cand = (cfg, thr, m_dv, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            del model, opt

        cfg, thr, m_dv, st = best_cand
        model = rem_models.build("res_dae", cfg, device)
        model.load_state_dict(st)
        p_te = predict_res(model, te_x)
        m_te = calc_metrics(te_y, p_te, thr)
        sec = round(time.monotonic() - t0, 1)

        save_csv(run_dir / "test_predictions.csv", te, te_y, p_te, thr=thr)
        meta = {
            "dataset": dataset,
            "model": "res_dae",
            "seed": seed,
            "protocol": "official_4cand_dev_auc_youden",
            "config": cfg,
            "threshold": thr,
            "seconds": sec,
            "dev_metrics": m_dv,
            "metrics": m_te,
        }
        write_json_atomic(meta_path, meta)
        runs.append(m_te)
        print(
            f"[DONE] {dataset:5s} res_dae     seed={seed} ({sec:6.1f}s) thr={thr:.1f} -> "
            f"ACC={100*m_te['accuracy']:.2f}% F1={100*m_te['f1']:.2f}% AUC={100*m_te['auc']:.2f}%",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summarize_runs(runs)


def run_ts_baseline(kind, data, split, rem_models, rem_prep, device):
    """Run BrainNetGNN or FBNetGen on ABIDE fixed split (seeds 0-4)."""
    tr = np.asarray(split["train"], dtype=int)
    dv = np.asarray(split["valid"], dtype=int)
    te = np.asarray(split["test"], dtype=int)

    all_done = all((OUT_DIR / "abide" / kind / f"seed_{s}" / "meta.json").exists() for s in SEEDS)
    if all_done:
        runs = []
        for s in SEEDS:
            meta = json.loads((OUT_DIR / "abide" / kind / f"seed_{s}" / "meta.json").read_text(encoding="utf-8"))
            runs.append(meta["metrics"])
            print(f"[SKIP] abide {kind:11s} seed={s}: ACC={100*meta['metrics']['accuracy']:.2f}% AUC={100*meta['metrics']['auc']:.2f}%", flush=True)
        return summarize_runs(runs)

    if kind == "fbnetgen":
        cfg = {"embedding_size": 16, "batch_size": 16, "lr": 1e-4, "weight_decay": 1e-4, "epochs": 500}
        scaler = rem_prep.fb_scaler(data["time_series"], data["lengths"], tr)
        norm_ts = rem_prep.temporal_inputs(data["time_series"], data["lengths"], "fbnetgen", scaler)
    elif kind == "brainnetgnn":
        cfg = {"batch_size": 32, "epochs": 3000, "k": 5, "l1": 1e-5, "lr": 5e-4, "n_classes": 2, "n_roi": 90, "weight_decay": 0.0}
        norm_ts = rem_prep.temporal_inputs(data["time_series"], data["lengths"], "brainnetgnn", None)
    else:
        raise ValueError(kind)

    tr_x, tr_y, tr_fc, tr_len = norm_ts[tr], data["labels"][tr], data["fc"][tr], data["lengths"][tr]
    dv_x, dv_y, dv_fc, dv_len = norm_ts[dv], data["labels"][dv], data["fc"][dv], data["lengths"][dv]
    te_x, te_y, te_fc, te_len = norm_ts[te], data["labels"][te], data["fc"][te], data["lengths"][te]

    @torch.no_grad()
    def predict_ts(model, x_arr, fc_arr, len_arr):
        model.eval()
        vals = []
        bs = cfg["batch_size"]
        for s in range(0, len(x_arr), bs):
            a = torch.as_tensor(x_arr[s:s + bs], device=device)
            if kind == "fbnetgen":
                logit = model(a, torch.as_tensor(fc_arr[s:s + bs], device=device), torch.as_tensor(len_arr[s:s + bs]))[0]
            else:
                logit = model(a, torch.as_tensor(len_arr[s:s + bs]))
            vals.append(logit.softmax(-1)[:, 1].cpu().numpy())
        return np.concatenate(vals).astype(float)

    runs = []
    for seed in SEEDS:
        run_dir = OUT_DIR / "abide" / kind / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(meta["metrics"])
            print(f"[SKIP] abide {kind:11s} seed={seed}: ACC={100*meta['metrics']['accuracy']:.2f}% AUC={100*meta['metrics']['auc']:.2f}%", flush=True)
            continue

        t0 = time.monotonic()
        seed_all(seed)
        model = rem_models.build(kind, cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=10) if kind == "brainnetgnn" else None
        best_loss = float("inf")
        best_ep = 0
        best_state = None
        bad_epochs = 0

        for ep in range(1, cfg["epochs"] + 1):
            if kind == "brainnetgnn" and bad_epochs >= 15:
                break
            model.train()
            order = np.random.permutation(len(tr_y))
            bs = cfg["batch_size"]
            for start in range(0, len(order), bs):
                ids = order[start:start + bs]
                xb = torch.as_tensor(tr_x[ids], device=device)
                yb = torch.as_tensor(tr_y[ids], device=device)
                opt.zero_grad(set_to_none=True)
                if kind == "fbnetgen":
                    loss = rem_models.fb_objective(model, xb, torch.as_tensor(tr_fc[ids], device=device), yb, torch.as_tensor(tr_len[ids]))
                else:
                    loss = torch.nn.functional.cross_entropy(model(xb, torch.as_tensor(tr_len[ids])), yb) + rem_models.gnn_regularization(model)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"abide {kind} seed={seed} ep={ep} nonfinite loss")
                loss.backward()
                opt.step()

            if kind == "brainnetgnn":
                p_dv = predict_ts(model, dv_x, dv_fc, dv_len)
                m_dv = calc_metrics(dv_y, p_dv)
                obj = m_dv["log_loss"] + float(rem_models.gnn_regularization(model).detach())
                if obj < best_loss:
                    best_loss = obj
                    best_ep = ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                sched.step(obj)
            elif ep % 100 == 0:
                print(f"  [abide {kind} seed={seed}] epoch {ep}/{cfg['epochs']} ({time.monotonic()-t0:.0f}s)", flush=True)

        if kind == "brainnetgnn" and best_state is not None:
            model.load_state_dict(best_state)
            ep_used = best_ep
        else:
            ep_used = cfg["epochs"]

        p_dv = predict_ts(model, dv_x, dv_fc, dv_len)
        p_te = predict_ts(model, te_x, te_fc, te_len)
        m_dv = calc_metrics(dv_y, p_dv)
        m_te = calc_metrics(te_y, p_te)
        sec = round(time.monotonic() - t0, 1)

        save_csv(run_dir / "test_predictions.csv", te, te_y, p_te)
        meta = {
            "dataset": "abide",
            "model": kind,
            "seed": seed,
            "protocol": "official_early_stop" if kind == "brainnetgnn" else "official_last_epoch",
            "config": cfg,
            "epoch_used": ep_used,
            "seconds": sec,
            "dev_metrics": m_dv,
            "metrics": m_te,
        }
        write_json_atomic(meta_path, meta)
        runs.append(m_te)
        print(
            f"[DONE] abide {kind:11s} seed={seed} (ep={ep_used:3d}, {sec:6.1f}s) -> "
            f"ACC={100*m_te['accuracy']:.2f}% F1={100*m_te['f1']:.2f}% AUC={100*m_te['auc']:.2f}%",
            flush=True,
        )
        del model, opt, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summarize_runs(runs)


def main():
    t0_all = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Route A: Fixed-Split Baselines Suite (seeds 0-4) on {device} ===", flush=True)

    plan_f = json.loads((REPO / "results" / "review_v2" / "plan.json").read_text(encoding="utf-8"))
    abide_split = plan_f["partitions"]["abide_fixed"]["indices"]
    adhd_split = plan_f["partitions"]["adhd_fixed"]["indices"]

    abide_data = dict(np.load(ABIDE_NPZ, allow_pickle=False))
    adhd_data = dict(np.load(ADHD_NPZ, allow_pickle=False))

    abide_fc = torch.from_numpy(abide_data["fc"].astype(np.float32))
    abide_y = torch.from_numpy(abide_data["labels"].astype(np.int64))
    adhd_fc = torch.from_numpy(adhd_data["fc"].astype(np.float32))
    adhd_y = torch.from_numpy(adhd_data["labels"].astype(np.int64))

    assert len(abide_fc) == 882 and len(adhd_fc) == 847

    sys.path.insert(0, str(FC_CODE))
    fc_models = load_module("fc_models", FC_CODE / "models.py")

    sys.path.append(NEUROCOMBAT_SITE)
    sys.path.insert(0, str(REM_CODE))
    rem_models = load_module("rem_models", REM_CODE / "models.py")
    rem_prep = load_module("rem_prep", REM_CODE / "preprocess.py")
    adhd_res_prep = load_module("adhd_res_prep", ADHD_RES_CODE / "preprocess.py")

    summary_path = OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"abide": {}, "adhd": {}}

    # Phase 1: Fast models on ABIDE & ADHD-200 (~45 min total)
    for m in ("brainnetcnn", "braingnn"):
        summary["abide"][m] = run_fc_baseline("abide", m, abide_fc, abide_y, abide_split, fc_models, device)
        write_json_atomic(summary_path, summary)
    summary["abide"]["res_dae"] = run_res_dae("abide", abide_data, abide_split, rem_models, rem_prep, device)
    write_json_atomic(summary_path, summary)
    summary["abide"]["brainnetgnn"] = run_ts_baseline("brainnetgnn", abide_data, abide_split, rem_models, rem_prep, device)
    write_json_atomic(summary_path, summary)
    summary["abide"]["brainnettf"] = run_fc_baseline("abide", "brainnettf", abide_fc, abide_y, abide_split, fc_models, device)
    write_json_atomic(summary_path, summary)

    for m in ("brainnetcnn", "braingnn"):
        summary["adhd"][m] = run_fc_baseline("adhd", m, adhd_fc, adhd_y, adhd_split, fc_models, device)
        write_json_atomic(summary_path, summary)
    summary["adhd"]["res_dae"] = run_res_dae("adhd", adhd_data, adhd_split, rem_models, adhd_res_prep, device)
    write_json_atomic(summary_path, summary)
    summary["adhd"]["brainnettf"] = run_fc_baseline("adhd", "brainnettf", adhd_fc, adhd_y, adhd_split, fc_models, device)
    write_json_atomic(summary_path, summary)

    # Phase 2: Slow models (BrainGB on ABIDE & ADHD-200, FBNetGen on ABIDE)
    summary["abide"]["braingb"] = run_fc_baseline("abide", "braingb", abide_fc, abide_y, abide_split, fc_models, device)
    write_json_atomic(summary_path, summary)
    summary["adhd"]["braingb"] = run_fc_baseline("adhd", "braingb", adhd_fc, adhd_y, adhd_split, fc_models, device)
    write_json_atomic(summary_path, summary)
    summary["abide"]["fbnetgen"] = run_ts_baseline("fbnetgen", abide_data, abide_split, rem_models, rem_prep, device)
    write_json_atomic(summary_path, summary)

    print(f"\n=== ALL FIXED-SPLIT BASELINES COMPLETE in {(time.monotonic()-t0_all)/60:.1f} min ===", flush=True)
    for ds in ("abide", "adhd"):
        print(f"\n--- {ds.upper()} FIXED SPLIT (seeds 0-4) ---", flush=True)
        for m, s in summary[ds].items():
            print(
                f"  {m:12s}: ACC={s['accuracy']['mean']:5.2f}+-{s['accuracy']['sd']:4.2f}% | "
                f"P={s['precision']['mean']:5.2f}+-{s['precision']['sd']:4.2f}% | "
                f"R={s['recall']['mean']:5.2f}+-{s['recall']['sd']:4.2f}% | "
                f"F1={s['f1']['mean']:5.2f}+-{s['f1']['sd']:4.2f}% | "
                f"AUC={s['auc']['mean']:5.2f}+-{s['auc']['sd']:4.2f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
