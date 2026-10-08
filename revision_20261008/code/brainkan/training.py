import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from .kan import KAN
from .data import features


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(n_roi, config, kind='kan'):
    dims = [n_roi * n_roi] + list(config['hidden']) + [2]
    if kind == 'kan':
        return KAN(dims, grid_size=config['grid_size'], spline_order=config['spline_order'])
    if kind == 'mlp':
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    raise ValueError(kind)


def predict(model, x, rois, batch_size, device):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            result.append(model(features(x[start:start+batch_size], rois).to(device)).softmax(1).cpu().numpy())
    return np.concatenate(result)


def classification_metrics(y, probabilities):
    pred = probabilities.argmax(1)
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(metrics.accuracy_score(y, pred)),
        'precision': float(metrics.precision_score(y, pred, average='macro', zero_division=0)),
        'recall': float(metrics.recall_score(y, pred, average='macro', zero_division=0)),
        'f1': float(metrics.f1_score(y, pred, average='macro', zero_division=0)),
        'auc': float(metrics.roc_auc_score(y, probabilities[:, 1])) if len(np.unique(y)) == 2 else None,
        'balanced_accuracy': float(metrics.balanced_accuracy_score(y, pred)),
        'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
        'specificity': float(tn / (tn + fp)) if tn + fp else None,
        'loss': float(metrics.log_loss(y, probabilities, labels=[0, 1])),
        'n': int(len(y)),
    }


def fit(x_train, y_train, x_valid, y_valid, rois, config, path, kind='kan', selection='validation'):
    """Training and model selection have no test arguments."""
    if selection not in ('validation', 'final'):
        raise ValueError('Checkpoint selection must be validation or final')
    if selection == 'validation' and (x_valid is None or y_valid is None):
        raise ValueError('Validation checkpoint selection requires development data')
    seed_everything(config['seed'])
    device = config['device']
    model = build_model(len(rois), config, kind).to(device)
    loader = DataLoader(TensorDataset(features(x_train, rois), torch.from_numpy(y_train)),
                        batch_size=config['batch_size'], shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_gamma'])
    criterion = nn.CrossEntropyLoss()
    best_key, history = None, []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward(); optimizer.step()
            total += loss.item() * len(targets)
        valid = (classification_metrics(y_valid, predict(model, x_valid, rois, config['batch_size'], device))
                 if selection == 'validation' else None)
        history.append({'epoch': epoch, 'train_loss': total/len(y_train), 'valid': valid,
                        'lr': optimizer.param_groups[0]['lr']})
        key = ((valid['auc'] if valid['auc'] is not None else -1.0, -valid['loss'])
               if valid is not None else (epoch,))
        if best_key is None or key > best_key:
            best_key = key
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': config, 'rois': list(map(int, rois)), 'kind': kind,
                        'epoch': epoch, 'validation': valid, 'checkpoint_selection': selection}, path)
        print(f'{path.stem} epoch {epoch}/{config["epochs"]}: selection={selection}', flush=True)
        scheduler.step()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, history


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(len(ckpt['rois']), ckpt['config'], ckpt['kind'])
    model.load_state_dict(ckpt['state_dict']); model.to(device).eval()
    return model, ckpt
