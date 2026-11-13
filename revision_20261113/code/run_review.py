"""Numerical routines copied verbatim from the recorded experiment."""
import random
import numpy as np
import torch
from data import features
from sklearn.metrics import (accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def metric(y,prob):
    if prob.shape!=(len(y),2) or not np.isfinite(prob).all():raise ValueError('Invalid probabilities')
    pred=prob.argmax(1)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),auc=float(roc_auc_score(y,prob[:,1])),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),f1_macro=float(f1_score(y,pred,average='macro',zero_division=0)),
                loss=float(log_loss(y,prob,labels=[0,1])))


def input_features(x,kind):
    if kind in ('logistic','linear_svm'):
        # Upper triangle excludes the diagonal and duplicate undirected edges.
        i,j=np.triu_indices(x.shape[1],k=1);return x[:,i,j].copy()
    if kind=='gcn':return torch.from_numpy(np.ascontiguousarray(x))
    return features(x,np.arange(x.shape[1]))


@torch.no_grad()
def predict(model,tensor,device,batch_size=96):
    model.eval();prob=[]
    for start in range(0,len(tensor),batch_size):prob.append(model(tensor[start:start+batch_size].to(device)).softmax(1).cpu().numpy())
    return np.concatenate(prob)

