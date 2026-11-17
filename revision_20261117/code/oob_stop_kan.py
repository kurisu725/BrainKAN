"""Training-internal early selection of complete-KAN optimization exposure."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from bootstrap_native_kan import base_regularizer,cpu_export
from additive_pilot import metric


def choose_candidate(candidates):
    """Strict CE minimum with earliest candidate winning exact ties."""
    assert candidates and all(np.isfinite(c['oob_ce']) for c in candidates)
    assert all(candidates[i]['update']<candidates[i+1]['update'] for i in range(len(candidates)-1))
    return min(range(len(candidates)),key=lambda i:(candidates[i]['oob_ce'],candidates[i]['update']))


def fit(net,q,labels,draw,state,budget,candidates,cfg):
    draw=np.asarray(draw,dtype=int);n=len(draw);assert draw.min()>=0 and draw.max()<n and len(labels)>=n
    steps=list(map(int,candidates));assert steps==sorted(set(steps)) and 0<steps[0]<=steps[-1]==budget
    omitted=np.flatnonzero(np.bincount(draw,minlength=n)==0);assert len(omitted)>1 and set(np.unique(labels[omitted]))=={0,1}
    x=q[torch.as_tensor(draw,device=q.device)];target=torch.as_tensor(labels[draw],device=q.device,dtype=torch.float64);x_oob=q[torch.as_tensor(omitted,device=q.device)];y_oob=torch.as_tensor(labels[omitted],device=q.device,dtype=torch.float64)
    initial=cpu_export(net);cov=state['function_covariance'].to(q.device);optimizer=torch.optim.Adam([dict(params=[net.first],lr=cfg['first_learning_rate']),dict(params=[net.head,net.bias],lr=cfg['head_learning_rate'])]);history=[];monitors=[];best=None;best_ce=float('inf');monitor_seconds=0.;tic=time.perf_counter()
    net.first.requires_grad_(True)
    for step in range(1,budget+1):
        optimizer.zero_grad(set_to_none=True);ce=F.binary_cross_entropy_with_logits(net(x),target);reg=base_regularizer(net,x,cov,cfg);objective=ce+reg;assert torch.isfinite(objective);objective.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']));assert np.isfinite(gn);optimizer.step()
        history.append(dict(update=step,head_only=False,training_ce_before_update=float(ce.detach()),regularizer_before_update=float(reg.detach()),objective_before_update=float(objective.detach()),gradient_norm_before_update=gn))
        if step in steps:
            probe_tic=time.perf_counter()
            with torch.no_grad():values=net(x_oob);oob_ce=float(F.binary_cross_entropy_with_logits(values,y_oob));logits=values.cpu().numpy().copy()
            assert np.isfinite(oob_ce);monitors.append(dict(update=step,oob_ce=oob_ce,oob_logits=logits))
            if oob_ce<best_ce:best_ce=oob_ce;best=cpu_export(net)
            monitor_seconds+=time.perf_counter()-probe_tic
    chosen=choose_candidate(monitors);assert best is not None and best_ce==monitors[chosen]['oob_ce'];net.load_state_dict({key:value.to(q.device) for key,value in best.items()});net.zero_grad(set_to_none=True)
    ce=F.binary_cross_entropy_with_logits(net(x),target);reg=base_regularizer(net,x,cov,cfg);objective=ce+reg;objective.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in net.parameters())))
    if q.is_cuda:torch.cuda.synchronize(q.device)
    elapsed=time.perf_counter()-tic
    with torch.no_grad():values=net(q).cpu().numpy()
    return dict(weights=cpu_export(net),initial_weights=initial,updates_run=budget,head_only_updates=0,joint_updates=budget,selected_update=monitors[chosen]['update'],selected_oob_ce=best_ce,selected_candidate_index=chosen,training_internal_candidates=monitors,oob_positions=omitted,history=history,training_ce=float(ce.detach()),regularizer=float(reg.detach()),training_objective=float(objective.detach()),gradient_norm=gn,bootstrap_training=metric(labels[draw],values[draw]),elapsed_fit_seconds=elapsed,internal_monitor_seconds=monitor_seconds,development_checkpoint_selection=False,numerical_global_convergence_claim=False),values
