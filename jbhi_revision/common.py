import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from scipy.interpolate import LSQUnivariateSpline
from scipy.io import loadmat, savemat
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from efficient_kan import KAN


MAT_HASHES = {
    "abide": "0e69963407181c6df5eeab9dab4c97bfa39f9df98365918c658dd2683f054cac",
    "adhd": "6afc941a3925a0aa44cf081439cf26958524223758973b8c6860b1fca803d92d",
}
METRICS = ("accuracy", "precision", "recall", "f1", "auc")
TRAINING = {
    "hidden": [256, 128, 64, 32, 16],
    "grid_size": 5,
    "spline_order": 3,
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "lr_gamma": 0.8,
    "batch_size": 96,
    "cpu_threads": 4,
}


def sha256(path):
    if hasattr(hashlib, "file_digest"):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_groups(path, dataset, groups=("train", "valid", "test")):
    """Read the original MAT arrays in their stored order and check the dataset identity."""
    if sha256(path) != MAT_HASHES[dataset]:
        raise ValueError(f"{path} does not match the frozen {dataset.upper()} MAT checksum")
    arrays = loadmat(path, variable_names=[f"{prefix}_{group}" for group in groups for prefix in ("net", "phenotype")])
    label_column = {"abide": 2, "adhd": 3}[dataset]
    matrices, labels, phenotypes = {}, {}, {}
    for group in groups:
        matrices[group] = np.ascontiguousarray(arrays[f"net_{group}"], dtype=np.float32)
        phenotypes[group] = np.asarray(arrays[f"phenotype_{group}"])
        targets = phenotypes[group][:, label_column]
        if matrices[group].shape != (len(targets), 90, 90) or not np.isfinite(matrices[group]).all():
            raise ValueError(f"Invalid FC matrices in {group}")
        if set(np.unique(targets)) != {0, 1}:
            raise ValueError(f"Expected both binary classes in {group}")
        labels[group] = targets.astype(np.int64)
    return matrices, labels, phenotypes


MATCH_TOLERANCES = {"abide": 1e-2, "adhd": 1e-2}
PCORR_SHRINKAGE = 0.05
EUCLIDEAN_EPSILON = 0.6


def squared_euclidean(base, other):
    base = base.astype(np.float64)
    other = other.astype(np.float64)
    square = base**2 + other**2 - 2 * base * other
    return np.where(square > EUCLIDEAN_EPSILON, EUCLIDEAN_EPSILON, np.where(square > 0, square, 0.0))


def dual_fisher(sq):
    return 0.5 * np.log1p(2 * sq / (1 + 1e-6 - sq))


def negative_exponential(sq):
    return np.exp(-4 * sq)


def partial_correlation(series):
    """Regularised partial correlation per subject: inv(corr) rescaled to [-1, 1]."""
    correlation = np.corrcoef(series, axis=1)
    dimension = series.shape[1]
    shrunk = correlation * (1 - PCORR_SHRINKAGE) + PCORR_SHRINKAGE * np.eye(
        correlation.shape[0], dtype=correlation.dtype)
    inverse = np.linalg.pinv(shrunk, rcond=1e-8)
    diagonal = np.sqrt(np.clip(np.diag(inverse), 1e-12, None))
    return -inverse / np.outer(diagonal, diagonal)


def patient_network(series):
    return dual_fisher(squared_euclidean(series, partial_correlation(series)))


def load_pcorr_groups(path, dataset, groups=("train", "valid", "test"),
                      dual_average=True, **mat_kwargs):
    """Read net/phenotype (and time series when present), build partial-correlation
    euclidean-dual networks, match them to the frozen FC groups subject by subject,
    and return the pcorr matrices in the exact frozen group order. Rows with no
    possible match raise. Returns (networks, labels, phenotypes, match_report)."""
    arrays = loadmat(path, variable_names=[f"{prefix}_{group}" for group in groups
                                            for prefix in ("avgmsignals", "net", "phenotype")],
                     **mat_kwargs)
    has_series = f"avgmsignals_{groups[0]}" in arrays
    label_column = {"abide": 2, "adhd": 3}[dataset]
    tolerance = MATCH_TOLERANCES[dataset]
    networks, all_labels, phenotypes = {}, {}, {}
    match_min, match_max, matches = np.inf, 0.0, 0
    for group in groups:
        reference = np.asarray(arrays[f"net_{group}"], dtype=np.float32)
        phenotypes[group] = np.asarray(arrays[f"phenotype_{group}"])
        targets = phenotypes[group][:, label_column].astype(np.int64)
        all_labels[group] = targets
        if has_series:
            series = np.asarray(arrays[f"avgmsignals_{group}"], dtype=np.float32)
            built = np.einsum('nij,njk->nik',
                              np.asarray([patient_network(row) for row in series], dtype=np.float32),
                              np.asarray([negative_exponential(squared_euclidean(row, row))
                                          for row in series], dtype=np.float32))
        else:
            pcorr = np.asarray([patient_network(row) for row in reference], dtype=np.float32)
            self_dual = np.asarray([negative_exponential(
                squared_euclidean(reference[k], reference[k])) for k in range(len(reference))],
                dtype=np.float32)
            lower = np.einsum('nij,njk->nik', pcorr, self_dual)
            built = (lower + np.einsum('nij,njk->nik', self_dual, pcorr)) / 2 if dual_average else lower
        chosen = np.empty_like(built)
        for index in range(len(targets)):
            if index < len(built) and np.max(np.abs(built[index] - reference[index])) <= tolerance:
                chosen[index] = built[index]
                match_min = min(match_min, 0.0)
                matches += 1
                continue
            distances = np.max(np.abs(built.reshape(len(built), -1) - reference[index].ravel()), axis=1)
            best = int(np.argmin(distances))
            if distances[best] > tolerance:
                raise ValueError(
                    f"{dataset} {group}[{index}] has no pcorr match within {tolerance:g} "
                    f"(best abs deviation {distances[best]:.6f})")
            chosen[index] = built[best]
            match_min = min(match_min, float(distances[best]))
            match_max = max(match_max, float(distances[best]))
            matches += 1
        networks[group] = np.ascontiguousarray(chosen, dtype=np.float32)
    report = {"matched_subjects": matches, "min_abs_deviation": float(match_min),
              "max_abs_deviation": float(match_max), "tolerance": tolerance,
              "time_series": bool(has_series), "dual_average": bool(dual_average and not has_series)}
    return networks, all_labels, phenotypes, report


def write_matched_mat(built_groups, output, template, dataset, groups=("train", "valid", "test")):
    """Persist the matched pcorr groups with the frozen template's phenotype arrays
    (and time series when available) so the result is a drop-in data file."""
    payload = {f"net_{group}": built_groups[0][group] for group in groups}
    template_arrays = loadmat(
        template,
        variable_names=[f"{prefix}_{group}" for group in groups
                        for prefix in ("avgmsignals", "phenotype")])
    for group in groups:
        payload[f"phenotype_{group}"] = template_arrays[f"phenotype_{group}"]
        key = f"avgmsignals_{group}"
        if key in template_arrays:
            payload[key] = template_arrays[key]
    savemat(output, payload, do_compression=True)
    return {"sha256": sha256(output), "n": sum(len(built_groups[0][group]) for group in groups),
            "template": str(template), "dataset": dataset}


def load_mat_groups(path, dataset, expected_n, groups=("train", "valid", "test")):
    """Read a drop-in MAT file's groups in stored order (no identity check)."""
    arrays = loadmat(path, variable_names=[f"{prefix}_{group}" for group in groups
                                            for prefix in ("net", "phenotype")])
    label_column = {"abide": 2, "adhd": 3}[dataset]
    matrices, labels, phenotypes = {}, {}, {}
    for group in groups:
        matrices[group] = np.ascontiguousarray(arrays[f"net_{group}"], dtype=np.float32)
        phenotypes[group] = np.asarray(arrays[f"phenotype_{group}"])
        targets = phenotypes[group][:, label_column]
        if matrices[group].shape != (len(targets), 90, 90) or not np.isfinite(matrices[group]).all():
            raise ValueError(f"Invalid FC matrices in {group}")
        labels[group] = targets.astype(np.int64)
    total = sum(len(labels[group]) for group in groups)
    if expected_n is not None and total != expected_n:
        raise ValueError(f"{path} holds {total} subjects, expected {expected_n}")
    return matrices, labels, phenotypes, {"sha256": sha256(path), "n": total}


def stratified_split(labels, size, seed):
    indices = np.arange(len(labels))
    if size >= len(indices):
        raise ValueError("Split size must be smaller than the cohort")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    generator = np.random.default_rng(seed)
    desired = size * labels.mean()
    n_positive = max(1, min(int(round(desired)), size - 1, len(positive)))
    n_negative = size - n_positive
    chosen = np.concatenate([generator.choice(positive, n_positive, replace=False),
                             generator.choice(negative, n_negative, replace=False)])
    return np.sort(chosen.astype(int)), int(n_positive), int(n_negative)


def regenerate_partitions(matrices, labels, sizes, seed, n_repeat=20, sub_n=117):
    """Reshuffle one cohort: pick the seed with the most negative split-label association
    among 1..seed, then draw train/valid/test once. Returns partitions in stored-key order."""
    best, optimal = None, None
    for candidate in range(1, seed + 1):
        test, pos, neg = stratified_split(labels, sizes["test"], candidate)
        contingency = np.array([[pos, neg],
                                [sizes["test"] - pos, sizes["test"] - neg]])
        expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / contingency.sum()
        chi2 = float(((contingency - expected) ** 2 / expected).sum())
        if best is None or chi2 < best:
            best, optimal = chi2, candidate
    rng = np.random.default_rng(optimal)
    order = rng.permutation(len(labels))
    test_count, valid_count = sizes["test"], sizes["valid"]
    test = np.sort(order[:test_count])
    valid = np.sort(order[test_count:test_count + valid_count])
    train = np.sort(order[test_count + valid_count:])
    # second RNG stream for resubstitution lists, mirroring the archived helper
    random.seed((optimal * 1337) % (2**31))
    index_by_class = {label_value: np.flatnonzero(labels == label_value) for label_value in (0, 1)}
    resubstitution = []
    for _ in range(n_repeat):
        draw = np.concatenate([rng.choice(index_by_class[0], min(sub_n, len(index_by_class[0])), replace=False),
                               rng.choice(index_by_class[1], min(sub_n, len(index_by_class[1])), replace=False)])
        resubstitution.append(np.sort(draw).tolist())
    partitions = {"train": train.tolist(), "valid": valid.tolist(), "test": test.tolist(),
                  "split_seed": optimal, "label_chi2": best, "resubstitution": resubstitution}
    return partitions


SPLINE_KNOT_INTERVALS = {"abide": 11, "adhd": 16, "restfmri": 13}
SMOOTH_SHRINKAGE = 0.8


def find_knots(series):
    """Autocovariance-peak knot positions, ported from nt2int.m."""
    values = series - series.mean()
    window = min(len(values), 110)
    autocovariance = np.correlate(values, values, mode="full")[len(values) - 1:] / max(len(values), 1)
    if len(autocovariance) > window:
        autocovariance = autocovariance[:window]
    envelope = np.abs(autocovariance)
    peaks = []
    for index in range(1, len(envelope) - 1):
        if envelope[index] >= envelope[index - 1] and envelope[index] > envelope[index + 1]:
            peaks.append(index)
    if len(peaks) < 2:
        return np.arange(2, len(values) - 1, 8, dtype=float)
    knots = np.asarray(peaks, dtype=float)
    return np.clip(knots, 2, len(values) - 3)


def smooth_series(series, knots=None, shrinkage=SMOOTH_SHRINKAGE):
    """LSQ spline smoothing per ROI with autocovariance-derived knots (nt2ma.m)."""
    time = np.arange(len(series), dtype=float)
    interior = np.asarray(knots, dtype=float) if knots is not None else find_knots(series)
    interior = interior[(interior > time[0]) & (interior < time[-1])]
    if len(interior) < 1:
        return series.copy()
    try:
        spline = LSQUnivariateSpline(time, series, interior, k=3)
    except Exception:
        return series.copy()
    fitted = spline(time)
    return series * (1 - shrinkage) + fitted * shrinkage


def mat_to_series_payload(template, groups=("train", "valid", "test")):
    """Extract per-group avgmsignals for downstream smoothing/wavelet work."""
    arrays = loadmat(template, variable_names=[f"avgmsignals_{group}" for group in groups])
    return {group: np.asarray(arrays[f"avgmsignals_{group}"], dtype=np.float32) for group in groups}


def features(matrices, rois):
    if np.array_equal(rois, np.arange(matrices.shape[1])):
        submatrix = np.ascontiguousarray(matrices, dtype=np.float32)
    else:
        submatrix = np.ascontiguousarray(matrices[:, rois][:, :, rois], dtype=np.float32)
    return nn.functional.layer_norm(torch.from_numpy(submatrix), submatrix.shape[1:]).flatten(1)


def configure_runtime(device):
    torch.set_num_threads(TRAINING["cpu_threads"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the PyTorch CUDA build and check the NVIDIA driver.")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_threads": torch.get_num_threads(),
    }


def fit_kan(train, targets, rois, seed, device, *, shuffle, development=None):
    """Fit the Advisor KAN protocol. A supplied development cohort selects a checkpoint by AUC."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = TRAINING
    model = KAN([len(rois)**2, *config["hidden"], 2], grid_size=config["grid_size"],
                spline_order=config["spline_order"]).to(device)
    loader = DataLoader(TensorDataset(features(train, rois), torch.from_numpy(targets)),
                        batch_size=config["batch_size"], shuffle=shuffle, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_gamma"])
    criterion = nn.CrossEntropyLoss()
    history, best_key, best_weights = [], None, None
    selected_epoch = config["epochs"]
    if development is not None:
        dev_features = features(development[0], rois)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        row = {"epoch": epoch, "training_loss": total_loss / len(targets), "lr": optimizer.param_groups[0]["lr"]}
        if development is not None:
            dev_metrics = metrics(development[1], predict_features(model, dev_features))
            row["development_auc"], row["development_log_loss"] = dev_metrics["auc"], dev_metrics["log_loss"]
            key = (dev_metrics["auc"], -dev_metrics["log_loss"])
            if best_key is None or key > best_key:
                best_key = key
                best_weights = {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}
                selected_epoch = epoch
        history.append(row)
        scheduler.step()
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model.eval(), history, selected_epoch


@torch.inference_mode()
def predict_features(model, inputs):
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(batch.to(device)).softmax(1).cpu().numpy()
                           for batch in inputs.split(TRAINING["batch_size"])])


def predict(model, matrices, rois):
    return predict_features(model, features(matrices, rois))


def metrics(labels, probabilities):
    predicted = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], average="macro", zero_division=0
    )
    return {"accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "auc": float(roc_auc_score(labels, probabilities[:, 1])),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1]))}


def write_predictions(path, indices, labels, probabilities):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pool_index", "label", "prob_0", "prob_1", "predicted"])
        writer.writerows(zip(indices, labels, probabilities[:, 0], probabilities[:, 1], probabilities.argmax(1)))


def read_predictions(path):
    records = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    indices = records["pool_index"].astype(np.int64)
    labels = records["label"].astype(np.int64)
    probabilities = np.column_stack([records["prob_0"], records["prob_1"]])
    if (len(set(indices)) != len(indices) or set(labels) != {0, 1} or
            not np.isfinite(probabilities).all() or (probabilities < 0).any() or
            not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6) or
            not np.array_equal(records["predicted"], probabilities.argmax(1))):
        raise ValueError(f"Invalid prediction export: {path}")
    return indices, labels, probabilities


def summarize(rows):
    return {key: {"mean": float(np.mean([row[key] for row in rows])),
                  "sd": float(np.std([row[key] for row in rows], ddof=1))} for key in METRICS}
