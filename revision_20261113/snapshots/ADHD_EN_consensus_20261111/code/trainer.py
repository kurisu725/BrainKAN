"""Fixed-grid KAN with bounded storage and safe epoch-boundary pause."""
from pathlib import Path
import random,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from models import build_model
from run_review import seed_all,input_features,predict,metric
from storage import save_torch

class SafePause(Exception):pass

def train_candidate(train_x,train_y,dev_x,dev_y,cfg,job,rois,plan_sha,root,
                    device='cuda',on_epoch=None,pause_check=None,interval=5):
    checkpoint_path=Path(root)/'state/current.pt'
    if pause_check and pause_check():raise SafePause()
    seed_all(cfg['seed']);model=build_model(cfg['kind'],train_x.shape[1],cfg).to(device)
    xx=input_features(train_x,cfg['kind']);vv=input_features(dev_x,cfg['kind'])
    assert cfg['optimizer']=='adamw' and not cfg['grid_epochs'] and cfg['regularization']==0
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=cfg['gamma'])
    loader=DataLoader(TensorDataset(xx,torch.from_numpy(train_y.astype(np.int64))),
        batch_size=cfg['batch_size'],shuffle=cfg['shuffle'],num_workers=0)
    history=[];saved_epoch=-1
    if checkpoint_path.exists():
        saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if (saved['job'],saved['config'],saved['plan_sha256'],saved['rois'])!=(job,cfg,plan_sha,list(rois)):
            raise ValueError('Rolling checkpoint belongs to a different job/configuration/plan/ROI list')
        model.load_state_dict(saved['state']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history=saved['history'];saved_epoch=saved['epoch'];assert saved_epoch==len(history)
        torch.set_rng_state(saved['torch_rng'])
        if device!='cpu':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);del saved
    def checkpoint():
        nonlocal saved_epoch
        if saved_epoch==len(history):return
        save_torch(checkpoint_path,dict(job=job,config=cfg,rois=list(rois),plan_sha256=plan_sha,
            epoch=len(history),state={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device!='cpu' else [],
            numpy_rng=np.random.get_state(),python_rng=random.getstate()))
        saved_epoch=len(history)
    for epoch in range(len(history)+1,cfg['epochs']+1):
        if pause_check and pause_check():checkpoint();raise SafePause()
        model.train();tic=time.monotonic();total=0.
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device);opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(xb),yb)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            opt.step();total+=float(loss)*len(yb)
        development=metric(dev_y,predict(model,vv,device,cfg['batch_size']))
        history.append(dict(epoch=epoch,training_ce=total/len(xx),development=development,
                            lr=opt.param_groups[0]['lr'],seconds=time.monotonic()-tic))
        scheduler.step()
        if epoch%interval==0 or epoch==cfg['epochs']:checkpoint()
        if on_epoch:on_epoch(epoch,history[-1],saved_epoch)
        if pause_check and pause_check():checkpoint();raise SafePause()
    checkpoint()
    prob=predict(model,vv,device,cfg['batch_size'])
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    return dict(config=cfg,state=state,n_roi=len(rois),epoch=cfg['epochs']),prob,history
