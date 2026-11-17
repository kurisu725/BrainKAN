"""Activity-entropy KAN fitting with early internal AUC monitoring and patience."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from bootstrap_native_kan import base_regularizer,cpu_export
from activity_entropy_kan import activity_entropy_penalty
from sklearn.metrics import roc_auc_score


def fit(net,q,labels,draw,state,budget,candidates,cfg):
    draw=np.asarray(draw,dtype=int);n=len(draw);steps=list(map(int,candidates))
    assert draw.min()>=0 and draw.max()<n and len(labels)>=n and steps==sorted(set(steps)) and steps[-1]==budget
    omitted=np.flatnonzero(np.bincount(draw,minlength=n)==0);assert set(labels[omitted])=={0,1}
    device=q.device;x=q[torch.as_tensor(draw,device=device)];target=torch.as_tensor(labels[draw],device=device,dtype=q.dtype)
    xo=q[torch.as_tensor(omitted,device=device)];yo=torch.as_tensor(labels[omitted],device=device,dtype=q.dtype)
    cov=state['function_covariance'].to(device);history=[];monitors=[];best=None;best_ce=float('inf');best_auc=-1.;bad=0;monitor_seconds=0.;tic=time.perf_counter()
    optimizer=torch.optim.Adam([dict(params=[net.first],lr=cfg['first_learning_rate']),dict(params=[net.head,net.bias],lr=cfg['head_learning_rate'])])
    def terms():
        ce=F.binary_cross_entropy_with_logits(net(x),target);base=base_regularizer(net,x,cov,cfg)
        entropy,parts=activity_entropy_penalty(net,x,cov,cfg['input_activity_entropy'],cfg['head_activity_entropy'])
        return ce,base+entropy,entropy,parts
    stopped=False
    for step in range(1,budget+1):
        optimizer.zero_grad(set_to_none=True);ce,reg,entropy,parts=terms();objective=ce+reg;assert torch.isfinite(objective)
        objective.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']));assert np.isfinite(gn);optimizer.step()
        history.append(dict(update=step,training_ce_before_update=float(ce.detach()),regularizer_before_update=float(reg.detach()),
            entropy_penalty_before_update=float(entropy.detach()),objective_before_update=float(objective.detach()),gradient_norm_before_update=gn))
        if step in steps:
            probe=time.perf_counter()
            with torch.no_grad():v=net(xo);loss=float(F.binary_cross_entropy_with_logits(v,yo));logits=v.cpu().numpy().copy()
            score=float(roc_auc_score(labels[omitted],logits));assert np.isfinite(loss) and np.isfinite(score);monitors.append(dict(update=step,oob_ce=loss,oob_auc=score,oob_logits=logits))
            if score>best_auc:best_auc=score;best_ce=loss;best=cpu_export(net);bad=0
            else:bad+=1
            monitor_seconds+=time.perf_counter()-probe
            if step>=cfg['minimum_updates'] and bad>=cfg['patience_candidates']:stopped=True;break
    chosen=max(range(len(monitors)),key=lambda i:(monitors[i]['oob_auc'],-monitors[i]['update']));assert best is not None and best_ce==monitors[chosen]['oob_ce'] and best_auc==monitors[chosen]['oob_auc']
    net.load_state_dict({key:value.to(device) for key,value in best.items()});net.zero_grad(set_to_none=True)
    ce,reg,entropy,parts=terms();objective=ce+reg;objective.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in net.parameters())))
    if q.is_cuda:torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-tic
    with torch.no_grad():values=net(q).cpu().numpy()
    result=dict(weights=cpu_export(net),updates_run=len(history),maximum_budget=budget,stopped_by_patience=stopped,
        selected_update=monitors[chosen]['update'],selected_oob_ce=best_ce,selected_oob_auc=best_auc,selected_candidate_index=chosen,
        training_internal_candidates=monitors,oob_positions=omitted,history=history,training_ce=float(ce.detach()),regularizer=float(reg.detach()),
        entropy_penalty=float(entropy.detach()),input_function_activity_entropy=float(parts['input_entropy'].detach()),
        head_function_activity_entropy=float(parts['head_entropy'].detach()),training_objective=float(objective.detach()),gradient_norm=gn,
        elapsed_fit_seconds=elapsed,internal_monitor_seconds=monitor_seconds,development_checkpoint_selection=False)
    assert torch.count_nonzero(net.first.detach()*(1-net.function_mask[:,None,:]))==0
    return result,values
