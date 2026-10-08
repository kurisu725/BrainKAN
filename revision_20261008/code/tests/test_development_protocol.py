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
from brainkan import cli
from brainkan.training import fit
from brainkan.roi import select_rois

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tune_validation',ROOT/'scripts/tune_validation.py')
tune=importlib.util.module_from_spec(spec);spec.loader.exec_module(tune)


class DevelopmentProtocolTests(unittest.TestCase):
    def test_author_confirmed_cli_training_defaults(self):
        captured=[]
        with patch('sys.argv',['run.py','run','--data','unused.mat','--output','unused']),patch('brainkan.cli.run',side_effect=lambda a:captured.append(a)):
            cli.main()
        args=captured[0]
        self.assertEqual(args.epochs,50)
        self.assertEqual(args.lr,.001)
        self.assertEqual(args.lr_gamma,.8)
        self.assertEqual(args.weight_decay,.0001)
        self.assertEqual(args.classifier_checkpoint,'final')

    def test_threshold_is_strict_and_customizable(self):
        np.testing.assert_array_equal(select_rois([-3.,-1.,1.],'direct',-1.),[0])
        np.testing.assert_array_equal(select_rois([-3.,-1.,1.],'zscore',0.),[0])
        np.testing.assert_array_equal(select_rois([-3.,-1.,1.],'zscore',0.1),[0,1])
        with self.assertRaises(ValueError):select_rois([-1,1],'direct',float('nan'))

    def test_hyperparameter_selection_ignores_test_scores(self):
        records=[{'validation':{'auc':.8,'loss':.6},'test':{'auc':1}},
                 {'validation':{'auc':.8,'loss':.4},'test':{'auc':0}}]
        self.assertIs(tune.choose_winner(records),records[1])
        records[0]['test']['auc']=0;records[1]['test']['auc']=1
        self.assertIs(tune.choose_winner(records),records[1])

    def test_fixed_epoch_selector_never_evaluates_validation(self):
        torch.set_num_threads(2)
        config=dict(seed=9,device='cpu',hidden=[3],grid_size=2,spline_order=1,
                    batch_size=4,lr=.001,weight_decay=.0001,lr_gamma=1.,epochs=2)
        x=np.random.default_rng(1).normal(size=(8,2,2)).astype('float32');y=np.arange(8,dtype=np.int64)%2
        with tempfile.TemporaryDirectory() as d, patch('brainkan.training.predict',side_effect=AssertionError('Validation accessed')):
            path=Path(d)/'selector.pt'
            model,history=fit(x,y,None,None,[0,1],config,path,selection='final')
            ckpt=torch.load(path,weights_only=True)
            self.assertEqual(ckpt['epoch'],2)
            self.assertIsNone(ckpt['validation'])
            self.assertTrue(all(h['valid'] is None for h in history))

    def test_selection_only_routes_validation_and_never_predicts_test(self):
        from brainkan.training import predict as real_predict
        rng=np.random.default_rng(7);mat={}
        arrays={}
        for i,k in enumerate(['train','valid','test']):
            x=rng.normal(size=(8,2,2)).astype('float32')
            x[:,0,0]=np.arange(8)+100*i
            mat['net_'+k]=arrays[k]=x
            mat['phenotype_'+k]=np.column_stack([np.arange(8),np.zeros(8),np.arange(8)%2])
        def explain(model,x,y,config):
            np.testing.assert_array_equal(x,arrays['valid'])
            np.testing.assert_array_equal(y,np.arange(8)%2)
            return np.array([[-1.,0.],[0.,-2.]]),{}
        def guarded_predict(model,x,*args):
            self.assertTrue(np.all(x[:,0,0]<200),'Final test accessed during selection')
            return real_predict(model,x,*args)
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/'data.mat';out=Path(d)/'run';savemat(data,mat)
            argv=['run.py','run','--data',str(data),'--output',str(out),'--epochs','2',
                  '--hidden','3','--grid-size','2','--spline-order','1','--batch-size','4',
                  '--device','cpu','--threads','2','--modes','direct','--selection-only']
            with patch('sys.argv',argv),patch('brainkan.cli.permutation_loss',side_effect=explain),patch('brainkan.cli.predict',side_effect=guarded_predict):
                cli.main()
            metadata=json.loads((out/'explanation.json').read_text())
            self.assertEqual(metadata['cohort'],'valid')
            self.assertEqual(metadata['pool_indices'],list(range(8,16)))
            self.assertFalse(list(out.glob('*test_predictions.csv')))
            results=json.loads((out/'metrics.json').read_text())
            self.assertTrue(all('test' not in r for r in results.values()))
            self.assertIn('direct_kan',results)
            self.assertEqual(results['direct_kan']['checkpoint_epoch'],2)
            self.assertEqual(results['direct_kan']['checkpoint_selection'],'final')
            self.assertEqual(results['full_kan']['checkpoint_epoch'],2)
            history=json.loads((out/'direct_kan_history.json').read_text())
            self.assertAlmostEqual(history[0]['lr'],.001)
            self.assertAlmostEqual(history[1]['lr'],.0008)


if __name__=='__main__':unittest.main()
