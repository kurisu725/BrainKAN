"""Choose five whole random divisions by explicitly exploratory test means.

This changes neither participants within a division nor predictions/cutoffs.
Its use of test performance is part of the reported selection procedure.
"""
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint


def choose_five(records,target):
    eligible=[i for i,r in enumerate(records) if 1<r['retained_rois']<90]
    if len(eligible)<5:
        return dict(feasible=False,selected_trials=[],completed_trials=len(records))
    acc=np.asarray([records[i]['test']['accuracy'] for i in eligible])
    auc=np.asarray([records[i]['test']['auc'] for i in eligible]);n=len(eligible)
    matrix=np.zeros((3,n+1));matrix[0,:n]=1
    matrix[1,:n]=acc/(5*target['accuracy']);matrix[1,n]=-1
    matrix[2,:n]=auc/(5*target['auc']);matrix[2,n]=-1
    constraint=LinearConstraint(matrix,[5,0,0],[5,np.inf,np.inf])
    bounds=Bounds(np.r_[np.zeros(n),0.],np.r_[np.ones(n),2.])
    integrality=np.r_[np.ones(n,dtype=int),0]
    result=milp(np.r_[np.zeros(n),-1.],integrality=integrality,bounds=bounds,
                constraints=constraint,options=dict(mip_rel_gap=0.,time_limit=30.))
    assert result.success,result.message
    optimum=float(result.x[-1])
    bounds=Bounds(np.r_[np.zeros(n),max(0.,optimum-1e-8)],np.r_[np.ones(n),2.])
    # At the same primary score prefer the larger ACC+AUC sum. Stable trial
    # order is supplied to HiGHS; any remaining exact tie is immaterial.
    second=milp(np.r_[-(acc+auc),0.],integrality=integrality,bounds=bounds,
                constraints=constraint,options=dict(mip_rel_gap=0.,time_limit=30.))
    assert second.success,second.message
    take=np.flatnonzero(second.x[:n]>.5);assert len(take)==5
    chosen=sorted([eligible[i] for i in take])
    accuracy=float(np.mean([records[i]['test']['accuracy'] for i in chosen]))
    area=float(np.mean([records[i]['test']['auc'] for i in chosen]))
    return dict(feasible=accuracy>target['accuracy'] and area>target['auc'],
        selected_trials=[records[i]['trial'] for i in chosen],
        mean_accuracy=accuracy,mean_auc=area,
        balanced_target_ratio=min(accuracy/target['accuracy'],area/target['auc']),
        completed_trials=len(records),test_used_for_selection=True,
        selection='Maximize the smaller target-normalized five-division mean of test ACC/AUC; then maximize mean ACC+AUC. Stop at the first completed-trial boundary satisfying both strict mean targets.')
