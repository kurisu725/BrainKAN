"""Synthetic-only candidate: entropy of complete KAN function RMS activity.

This module is not a frozen real-cohort protocol or an accuracy result.
It supplies an internal differentiable penalty, not a ROI ranker/classifier.
The original KAN paper motivates activation-magnitude/entropy sparsification;
this candidate uses normalized RMS activity of whole curves and is different
from both the paper's empirical absolute activity and legacy coefficient L1.
"""
import math,time
import numpy as np
import torch
from torch.nn import functional as F
from bootstrap_native_kan import base_regularizer,cpu_export
from oob_stop_kan import choose_candidate
from additive_pilot import metric


def normalized_activity_entropy(energy,mask=None,epsilon=1e-20):
    assert epsilon>0
    values=energy.clamp_min(0)
    if mask is not None:
        assert mask.shape==energy.shape and bool(((mask==0)|(mask==1)).all())
        values=values[mask.bool()]
    else:values=values.flatten()
    if values.numel()<=1:return energy.sum()*0.
    activity=(values+epsilon).sqrt();p=activity/activity.sum()
    return -(p*p.log()).sum()/math.log(values.numel())


def activity_entropy_penalty(net,q,cov,input_weight,head_weight):
    assert input_weight>=0 and head_weight>=0
    first=normalized_activity_entropy(net.curve_energy(cov),net.function_mask)
    terms=net.head_terms(net.hidden(q));centered=terms-terms.mean(0)
    head=normalized_activity_entropy(centered.square().mean(0))
    return input_weight*first+head_weight*head,dict(input_entropy=first,head_entropy=head)


def fit(net,q,labels,draw,state,budget,candidates,cfg):
    """Synthetic-tested future fit path; real-cohort use requires a frozen plan."""
    draw=np.asarray(draw,dtype=int);n=len(draw);assert draw.min()>=0 and draw.max()<n and len(labels)>=n;steps=list(map(int,candidates));assert steps==sorted(set(steps)) and 0<steps[0]<=steps[-1]==budget;omitted=np.flatnonzero(np.bincount(draw,minlength=n)==0);assert len(omitted)>1 and set(labels[omitted])=={0,1}
    x=q[torch.as_tensor(draw,device=q.device)];target=torch.as_tensor(labels[draw],device=q.device,dtype=torch.float64);xo=q[torch.as_tensor(omitted,device=q.device)];yo=torch.as_tensor(labels[omitted],device=q.device,dtype=torch.float64);cov=state['function_covariance'].to(q.device);initial=cpu_export(net);optimizer=torch.optim.Adam([dict(params=[net.first],lr=cfg['first_learning_rate']),dict(params=[net.head,net.bias],lr=cfg['head_learning_rate'])]);history=[];monitors=[];best=None;best_ce=float('inf');monitor_seconds=0.;tic=time.perf_counter();net.first.requires_grad_(True)
    def terms():
        ce=F.binary_cross_entropy_with_logits(net(x),target);base=base_regularizer(net,x,cov,cfg);entropy,parts=activity_entropy_penalty(net,x,cov,cfg['input_activity_entropy'],cfg['head_activity_entropy']);return ce,base+entropy,entropy,parts
    for step in range(1,budget+1):
        optimizer.zero_grad(set_to_none=True);ce,reg,entropy,parts=terms();objective=ce+reg;assert torch.isfinite(objective);objective.backward();gn=float(torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']));assert np.isfinite(gn);optimizer.step();history.append(dict(update=step,head_only=False,training_ce_before_update=float(ce.detach()),regularizer_before_update=float(reg.detach()),entropy_penalty_before_update=float(entropy.detach()),objective_before_update=float(objective.detach()),gradient_norm_before_update=gn))
        if step in steps:
            probe=time.perf_counter()
            with torch.no_grad():v=net(xo);loss=float(F.binary_cross_entropy_with_logits(v,yo));logits=v.cpu().numpy().copy()
            assert np.isfinite(loss);monitors.append(dict(update=step,oob_ce=loss,oob_logits=logits))
            if loss<best_ce:best_ce=loss;best=cpu_export(net)
            monitor_seconds+=time.perf_counter()-probe
    selected=choose_candidate(monitors);assert best is not None and best_ce==monitors[selected]['oob_ce'];net.load_state_dict({key:value.to(q.device) for key,value in best.items()});net.zero_grad(set_to_none=True);ce,reg,entropy,parts=terms();objective=ce+reg;objective.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in net.parameters())))
    if q.is_cuda:torch.cuda.synchronize(q.device)
    elapsed=time.perf_counter()-tic
    with torch.no_grad():values=net(q).cpu().numpy()
    return dict(weights=cpu_export(net),initial_weights=initial,updates_run=budget,head_only_updates=0,joint_updates=budget,selected_update=monitors[selected]['update'],selected_oob_ce=best_ce,selected_candidate_index=selected,training_internal_candidates=monitors,oob_positions=omitted,history=history,training_ce=float(ce.detach()),regularizer=float(reg.detach()),entropy_penalty=float(entropy.detach()),input_function_activity_entropy=float(parts['input_entropy'].detach()),head_function_activity_entropy=float(parts['head_entropy'].detach()),training_objective=float(objective.detach()),gradient_norm=gn,bootstrap_training=metric(labels[draw],values[draw]),elapsed_fit_seconds=elapsed,internal_monitor_seconds=monitor_seconds,development_checkpoint_selection=False,numerical_global_convergence_claim=False),values
