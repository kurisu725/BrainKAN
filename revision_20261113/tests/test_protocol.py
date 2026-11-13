"""Protocol, numerical equivalence, and safe-resume checks; no full study runs."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from scipy.io import savemat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from data import validate_split, features
from selection import complementary_pairs, roi_l2_scores, rank_consensus, top_k, fit_selector
from workflow import classification_metrics, development_choice_key
from trainer import train_candidate, SafePause
import workflow


class ProtocolTests(unittest.TestCase):
    def test_cohort_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_split(dict(train=[0,1],valid=[2,3],test=[3,4]),6)

    def test_half_samples_are_training_only_and_complementary(self):
        train=np.arange(10,19)
        pairs=complementary_pairs(train,50,7)
        self.assertEqual(pairs,complementary_pairs(train,50,7))
        for pair in pairs:
            a,b=set(pair['a']),set(pair['b'])
            self.assertFalse(a&b)
            self.assertEqual(len(a),4)
            self.assertEqual(set(pair['omitted']),set(train)-a-b)

    def test_edge_coefficients_contribute_to_both_incident_rois(self):
        np.testing.assert_allclose(roi_l2_scores([3,4,12],3),[5,np.sqrt(153),np.sqrt(160)])

    def test_rank_consensus_uses_average_ties_and_sorted_roi_ids(self):
        ranks=rank_consensus([[0,2,2,1],[0,4,4,1]])
        np.testing.assert_allclose(ranks,[.25,.875,.875,.5])
        self.assertEqual(top_k(ranks,2).tolist(),[1,2])
        self.assertEqual(top_k([2,0,2],1).tolist(),[0])

    def test_layernorm_follows_subnetwork_extraction(self):
        x=np.arange(18,dtype='float32').reshape(2,3,3)
        actual=features(x,[2,0])
        sub=torch.from_numpy(np.ascontiguousarray(x[:,[2,0]][:,:,[2,0]]))
        expected=torch.nn.functional.layer_norm(sub,(2,2)).flatten(1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_selector_standardizer_uses_only_passed_training_rows(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(30,4,4)).astype('float32')
        y=np.tile([0,1],15)
        ids=np.arange(20)
        cfg=dict(C=.1,l1_ratio=.5,max_iter=20000,tol=1e-5,seed=1,threads=1)
        result=fit_selector(x[ids],y[ids],cfg)
        i,j=np.triu_indices(4,1)
        raw=x[ids][:,i,j].astype(float)
        np.testing.assert_allclose(result['scaler_mean'],raw.mean(0),rtol=0,atol=1e-14)
        np.testing.assert_allclose(result['scaler_scale'],raw.std(0),rtol=0,atol=1e-14)

    def test_auc_uses_probabilities_and_macro_f1_uses_all_subjects(self):
        y=np.array([0,0,1,1])
        p=np.array([[.6,.4],[.51,.49],[.9,.1],[.2,.8]])
        m=classification_metrics(y,p)
        self.assertEqual(m['ACC'],.75)
        self.assertEqual(m['AUC'],.5)
        self.assertAlmostEqual(m['F1'],(0.8+2/3)/2)

    def test_development_choice_prefers_acc_then_auc_then_loss(self):
        rows=[dict(candidate='b',parameter_count=10,development=dict(accuracy=.7,auc=.8,loss=.5)),
              dict(candidate='a',parameter_count=8,development=dict(accuracy=.8,auc=.6,loss=.8))]
        self.assertEqual(min(rows,key=development_choice_key)['candidate'],'a')

    def test_complete_workflow_on_synthetic_mat_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'records/studies').mkdir(parents=True)
            rng=np.random.default_rng(17)
            x=rng.normal(size=(20,90,90)).astype('float32')
            y=np.tile([0,1],10)
            mat={}
            for name,ids in [('train',list(range(12))),('valid',list(range(12,16))),('test',list(range(16,20)))]:
                mat['net_'+name]=x[ids]
                mat['phenotype_'+name]=np.column_stack([ids,np.zeros(len(ids)),y[ids]])
            data=root/'synthetic.mat'
            savemat(data,mat)
            cfg=dict(kind='kan_legacy',hidden=[2],grid_size=3,spline_order=2,seed=1,
                     optimizer='adamw',grid_epochs=[],regularization=0,lr=.001,weight_decay=.0001,
                     gamma=.8,batch_size=4,shuffle=True,epochs=2)
            study=dict(dataset=dict(sha256=workflow.digest(data),n_subjects=20),label_column=2,
                       roi_budget=75,selector=dict(C=.1,l1_ratio=.5,max_iter=20000,tol=1e-5,seed=1,threads=1),
                       splits=[dict(id=0,config=cfg,indices=dict(train=list(range(12)),valid=list(range(12,16)),test=list(range(16,20))))],
                       half_samples={'0':[dict(pair=0,a=list(range(6)),b=list(range(6,12)),omitted=[])]})
            (root/'records/studies/abide.json').write_text(json.dumps(study),encoding='utf-8')
            args=SimpleNamespace(study='abide',data=data,output=root/'output',device='cpu',
                                 partitions=None,development_search=False,threads=1)
            with patch.object(workflow,'ROOT',root):
                self.assertEqual(workflow.train(args),0)
                before=workflow.digest(args.output/'final_test/en75_p0.csv')
                self.assertTrue((args.output/'development_frozen.json').is_file())
                ck=torch.load(args.output/'runs/en75_p0/model.pt',map_location='cpu',weights_only=False)
                self.assertEqual(ck['epoch'],2)
                self.assertEqual(len(ck['rois']),75)
                self.assertFalse((args.output/'state/current.pt').exists())
                self.assertEqual(workflow.train(args),0)
                self.assertEqual(before,workflow.digest(args.output/'final_test/en75_p0.csv'))

    def test_epoch_boundary_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(3)
        x=rng.normal(size=(16,3,3)).astype('float32')
        y=np.tile([0,1],8)
        cfg=dict(kind='kan_legacy',hidden=[4],grid_size=3,spline_order=2,seed=1,
                 optimizer='adamw',grid_epochs=[],regularization=0,lr=.001,weight_decay=.0001,
                 gamma=.8,batch_size=4,shuffle=True,epochs=3)
        with tempfile.TemporaryDirectory() as temp:
            full=Path(temp)/'full';resumed=Path(temp)/'resumed'
            full.mkdir();resumed.mkdir()
            trigger=[False]
            def on_epoch(epoch,entry,saved):
                if epoch==1:
                    trigger[0]=True
            with self.assertRaises(SafePause):
                train_candidate(x[:12],y[:12],x[12:],y[12:],cfg,'test',[0,1,2],
                                'plan',resumed,device='cpu',interval=5,
                                pause_check=lambda:trigger[0],on_epoch=on_epoch)
            # Compare from identical weights and RNG states, avoiding roundoff
            # from separate CPU least-squares initializations of the spline.
            (full/'state').mkdir()
            shutil.copyfile(resumed/'state/current.pt',full/'state/current.pt')
            a,pa,_=train_candidate(x[:12],y[:12],x[12:],y[12:],cfg,'test',[0,1,2],
                                  'plan',full,device='cpu',interval=5)
            trigger[0]=False
            def pause_after_second_epoch(epoch,entry,saved):
                if epoch==2:
                    trigger[0]=True
            with self.assertRaises(SafePause):
                train_candidate(x[:12],y[:12],x[12:],y[12:],cfg,'test',[0,1,2],
                                'plan',resumed,device='cpu',interval=5,
                                pause_check=lambda:trigger[0],on_epoch=pause_after_second_epoch)
            b,pb,_=train_candidate(x[:12],y[:12],x[12:],y[12:],cfg,'test',[0,1,2],
                                  'plan',resumed,device='cpu',interval=5)
            for name in a['state']:
                torch.testing.assert_close(a['state'][name],b['state'][name],rtol=0,atol=0)
            np.testing.assert_array_equal(pa,pb)


if __name__=='__main__':
    unittest.main()
