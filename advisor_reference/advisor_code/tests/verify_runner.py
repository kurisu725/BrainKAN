"""Check computation and interruption recovery on synthetic data before freezing."""
import sys,tempfile,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import advisor_experiments as a
from brainkan.reviewer import explain_fc
from sklearn.metrics import roc_auc_score,accuracy_score
torch.set_num_threads(2)
rng=np.random.default_rng(12)
x=rng.normal(size=(20,3,3)).astype('float32');y=(np.arange(20)%2).astype(np.int64)
cfg={**a.BASE,'hidden':[6,4],'seed':3,'device':'cuda','epochs':12,'batch_size':7,'permutation_repeats':3}
with tempfile.TemporaryDirectory(prefix='brainkan_validation_') as tmp:
    f=Path(tmp);whole=f/'whole';resumed=f/'resumed';whole.mkdir();resumed.mkdir()
    model,history=a.fit_neural(x[:14],y[:14],x[14:],y[14:],np.arange(3),cfg,'kan',whole)
    original_write=a.write_json
    def interrupt_after_eleven(path,value):
        original_write(path,value)
        if Path(path).name=='history.json' and len(value)==11:raise InterruptedError('simulated interruption')
    a.write_json=interrupt_after_eleven
    try:a.fit_neural(x[:14],y[:14],x[14:],y[14:],np.arange(3),cfg,'kan',resumed)
    except InterruptedError:pass
    finally:a.write_json=original_write
    recovered,h2=a.fit_neural(x[:14],y[:14],x[14:],y[14:],np.arange(3),cfg,'kan',resumed)
    max_resume_difference=max(float((value-recovered.state_dict()[key]).abs().max()) for key,value in model.state_dict().items())
    for key,value in model.state_dict().items():torch.testing.assert_close(value,recovered.state_dict()[key],rtol=1e-5,atol=1e-7)
    p=a.predict(model,x,np.arange(3),batch=7)
    met=a.classification_metrics(y,p)
    assert met['accuracy']==accuracy_score(y,p.argmax(1))
    assert met['auc']==roc_auc_score(y,p[:,1])
    sequential=explain_fc(model,x,y,cfg,'entry_permutation')
    batched=a.fast_permutation(model,x,y,cfg['seed'],repeats=3,chunk=4)
    for q in (0,1,2):np.testing.assert_allclose(sequential[q],batched[q],atol=1e-6,rtol=1e-5)
    cnn=a.BrainNetCNN(3).cuda().eval()
    xx=a.features(x,np.arange(3)).cuda()
    torch.testing.assert_close(cnn(xx),cnn(xx),rtol=0,atol=0)
    assert cnn(xx).shape==(20,2)
    scores=np.array([-3.,-1.,.4,1.2,5.])
    assert set(a.select_rois(scores,'direct'))==set(np.flatnonzero(scores<0))
    assert set(a.select_rois(scores,'minmax'))==set(np.flatnonzero(scores<(scores.min()+scores.max())/2))
    assert set(a.select_rois(scores,'zscore'))==set(np.flatnonzero(scores<scores.mean()))
    print(json.dumps({'resume_max_weight_difference':max_resume_difference,'multi_batch_metrics':True,'permutation_max_difference':float(np.max(np.abs(sequential[0]-batched[0]))),'cnn_eval_deterministic':True,'threshold_algebra':True}))
