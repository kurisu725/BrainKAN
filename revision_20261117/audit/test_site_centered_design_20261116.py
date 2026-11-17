"""Check invariance to site offsets and independent primal/dual equivalence."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from site_centered_guided_kan_20261116 import centered_design


def main():
    rng=np.random.default_rng(9161616)
    sites=np.repeat([1,2,3],8)
    labels=np.tile([0,0,0,1,1,1,0,1],3)
    raw=rng.normal(size=(24,11))
    x,y=centered_design(raw,labels,sites)
    offsets=rng.normal(size=(3,11)); label_offsets=np.array([-.9,2.1,8.])
    shifted_x,shifted_y=centered_design(raw+offsets[sites-1],labels+label_offsets[sites-1],sites)
    np.testing.assert_allclose(x,shifted_x,atol=1e-12,rtol=0)
    np.testing.assert_allclose(y,shifted_y,atol=1e-12,rtol=0)
    errors=[]
    for strength in [.1,1.,10.]:
        dual=x.T@solve(x@x.T/x.shape[1]+strength*np.eye(len(x)),y,assume_a='pos')/x.shape[1]
        primal=solve(x.T@x+x.shape[1]*strength*np.eye(x.shape[1]),x.T@y,assume_a='pos')
        np.testing.assert_allclose(dual,primal,atol=1e-12,rtol=0)
        errors.append(float(np.max(np.abs(dual-primal))))
    _,pure_response=centered_design(raw,np.repeat([0,1,1],8),sites)
    np.testing.assert_array_equal(pure_response,np.zeros(len(raw)))
    result=dict(passed=True,site_feature_and_response_offset_invariance=True,
                primal_dual_maximum_errors=errors,pure_site_response_is_zero=True,
                real_cohort_forward_or_training=False,weights_written=0)
    (ROOT/'site_centered_development_20261116/design_numerical_validation.json').write_text(
        json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
