import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.io import savemat
from brainkan import reviewer
from brainkan.explain import permutation_loss
from brainkan.kan import KAN

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reviewer_runner',ROOT/'scripts/reviewer_experiments.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def correlations(n=8,p=4,seed=3):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(n,p,10)); c=a@a.transpose(0,2,1)
    d=np.sqrt(np.diagonal(c,axis1=1,axis2=2))
    return (c/d[:,:,None]/d[:,None,:]).astype('float32')


class ReviewerTests(unittest.TestCase):
    def test_attenuation_preserves_correlation_and_unaffected_block(self):
        x=correlations(); original=x.copy()
        for a in [0.,.5,.75,1.]:
            y=reviewer.attenuate_roi(x,1,a)
            report=reviewer.require_correlations(y)
            self.assertEqual(report['fraction_symmetric_part_below_tolerance'],0)
            keep=[0,2,3]
            np.testing.assert_array_equal(x[:,keep][:,:,keep],y[:,keep][:,:,keep])
        np.testing.assert_array_equal(reviewer.attenuate_roi(x,1,1),x)
        self.assertTrue(np.all(reviewer.attenuate_roi(x,1,0)[:,1,[0,2,3]]==0))
        np.testing.assert_array_equal(x,original)
        bad=x.copy();bad[:,0,1]=2
        with self.assertRaises(ValueError):reviewer.require_correlations(bad)

    def test_symmetric_permutation_couples_edges_and_restores_source(self):
        x=correlations();copy=x.copy(); order=np.arange(len(x))[::-1]
        y=reviewer.permute_entry(x,0,2,order,True)
        np.testing.assert_array_equal(y[:,0,2],x[order,0,2])
        np.testing.assert_array_equal(y,y.transpose(0,2,1))
        np.testing.assert_array_equal(x,copy)

    def test_new_entry_arm_matches_existing_implementation(self):
        torch.set_num_threads(2); x=correlations(p=3);y=np.arange(len(x))%2
        model=KAN([9,3,2]);config=dict(seed=9,batch_size=5,device='cpu',permutation_repeats=2)
        old,_=permutation_loss(model,x,y,config)
        scores,repeat,edge,meta=reviewer.explain_fc(model,x,y,config,'entry_permutation')
        np.testing.assert_allclose(old,edge,rtol=0,atol=1e-10)
        np.testing.assert_allclose(scores,repeat.mean(0))
        baseline=meta['baseline_loss']
        self.assertTrue(np.isfinite(baseline))

    def test_identity_attenuation_has_zero_loss_effect(self):
        torch.set_num_threads(2);x=correlations(p=3);y=np.arange(len(x))%2
        model=KAN([9,3,2]);config=dict(seed=2,batch_size=5,device='cpu',permutation_repeats=2)
        score,_,edge,_=reviewer.explain_fc(model,x,y,config,'roi_attenuation',alpha=1)
        np.testing.assert_allclose(score,0,atol=1e-10);self.assertIsNone(edge)

    def test_fixed_split_development_then_frozen_test(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mat={}
            for seed,name in enumerate(['train','valid','test']):
                mat['net_'+name]=correlations(p=3,seed=seed+30)
                mat['phenotype_'+name]=np.column_stack([np.arange(8),np.zeros(8),np.arange(8)%2])
            savemat(root/'input.mat',mat)
            args=argparse.Namespace(data=str(root/'input.mat'),dataset='abide',split_file=None,
                seeds=[0,1],alphas=[.5],symmetric=True,epochs=1,repeats=2,hidden=[3],output=str(root/'plan.json'))
            runner.create_plan(args)
            # Use a permissive declared threshold only in this synthetic test to
            # ensure every branch reaches classifier fitting. Formal plans use defaults.
            plan=runner.read_json(root/'plan.json');plan['thresholds']={'zscore':100.}
            runner.write_json(root/'plan.json',plan)
            real_predict=reviewer.predict
            test=mat['net_test']
            def guarded(model,x,*a,**kw):
                self.assertFalse(any(np.array_equal(row,t) for row in x for t in test),'Test subject accessed during development')
                return real_predict(model,x,*a,**kw)
            with patch('brainkan.reviewer.predict',side_effect=guarded),patch.object(runner,'predict',side_effect=guarded):
                runner.develop(argparse.Namespace(plan=str(root/'plan.json'),output=str(root/'run'),device='cpu',threads=2))
            self.assertFalse((root/'run/final_test').exists())
            records=runner.read_json(root/'run/development_records.json')
            self.assertEqual(len({r['seed'] for r in records}),2)
            for row in records:
                self.assertIn('checkpoint',row)
                ck=torch.load(root/'run'/row['checkpoint'],weights_only=True)
                self.assertEqual(ck['epoch'],1);self.assertEqual(ck['checkpoint_selection'],'final')
            runner.evaluate(argparse.Namespace(experiment=str(root/'run'),device='cpu',threads=2))
            summary=runner.read_json(root/'run/final_test/summary.json')
            self.assertEqual(len(summary['arms']),4)
            self.assertTrue(all(v['complete'] for v in summary['arms'].values()))
            with self.assertRaises(FileExistsError):
                runner.evaluate(argparse.Namespace(experiment=str(root/'run'),device='cpu',threads=2))
            (root/'run/seed_0/config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):
                runner.evaluate(argparse.Namespace(experiment=str(root/'run'),device='cpu',threads=2))


if __name__=='__main__':unittest.main()
