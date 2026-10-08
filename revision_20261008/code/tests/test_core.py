import importlib.util
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.io import savemat
from brainkan.data import features, validate_split, load_data
from brainkan.kan import KAN
from brainkan.roi import roi_scores, select_rois, transform_scores
from brainkan.training import predict, classification_metrics
from brainkan.explain import permutation_loss


ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kan_matches_original_forward_and_gradients(self):
        spec = importlib.util.spec_from_file_location('original_kan', ROOT/'archive/brainkan_original/efficient_kan/kan.py')
        original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        a, b = KAN([9, 5, 2]), original.KAN([9, 5, 2])
        b.load_state_dict(a.state_dict())
        x = torch.randn(7, 9, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        torch.testing.assert_close(a(x), b(y), rtol=0, atol=0)
        a(x).sum().backward(); b(y).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_parameter_counts_match_paper(self):
        for n, expected in [(30, 2739520), (43, 5168960), (90, 21171520)]:
            model = KAN([n*n, 256, 128, 64, 32, 16, 2])
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_aggregation_and_threshold_boundaries(self):
        edge = np.array([[-1., .5], [1.5, 2.]])
        np.testing.assert_array_equal(roi_scores(edge), [0., 6.])
        c = np.array([-2., 0., 2.])
        for mode in ['direct', 'minmax', 'zscore']:
            np.testing.assert_array_equal(select_rois(c, mode), [0])
        with self.assertRaises(ValueError): select_rois([0., 1.], 'direct')
        with self.assertRaises(ValueError): transform_scores([1., 1.], 'minmax')

    def test_normalization_after_subnetwork_and_subject_independence(self):
        x = np.random.default_rng(3).normal(size=(3, 5, 5)).astype('float32')
        a = features(x, [4, 1, 2])
        b = features(x[:1], [4, 1, 2])
        torch.testing.assert_close(a[:1], b)
        torch.testing.assert_close(a.mean(1), torch.zeros(3), atol=1e-6, rtol=0)
        with self.assertRaises(ValueError): features(x, [1, 1])

    def test_metrics_keep_probabilities_from_all_batches(self):
        x = np.random.default_rng(8).normal(size=(17, 3, 3)).astype('float32')
        y = np.arange(17) % 2
        model = KAN([9, 5, 2])
        a = predict(model, x, [0,1,2], 6, 'cpu')
        b = predict(model, x, [0,1,2], 17, 'cpu')
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        result = classification_metrics(y, a)
        self.assertEqual(result['n'], 17)
        self.assertAlmostEqual(result['auc'], roc_auc_score(y, a[:,1]))

    def test_split_rejects_overlap_and_missing_classes(self):
        with self.assertRaises(ValueError):
            validate_split({'train': np.array([0,1]), 'valid': np.array([1,2]), 'test': np.array([3])}, 5)
        with self.assertRaises(ValueError):
            validate_split({'train': np.array([0,1]), 'valid': np.array([2,3]), 'test': np.array([4,5])}, 6, np.ones(6))

    def test_adhd_uses_fourth_column(self):
        d = {}
        for k in ['train','valid','test']:
            d['net_'+k] = np.zeros((2,3,3))
            d['phenotype_'+k] = np.array([[1, 10, 1, 0], [1, 11, 1, 1]])
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'adhd.mat'; savemat(p, d)
            _, y, idx, _ = load_data(p, 3)
            validate_split(idx, 6, y)
            np.testing.assert_array_equal(y, [0,1,0,1,0,1])
            _, wrong, idx, _ = load_data(p, 2)
            with self.assertRaises(ValueError): validate_split(idx, 6, wrong)

    def test_permutation_restores_input_and_negative_support(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.stack([-x[:,0], x[:,0]], dim=1)
        y = np.arange(20) % 2
        x = np.zeros((20, 2, 2), dtype=np.float32)
        x[:,0,0] = y * 2. - 1.
        original = x.copy()
        edge, meta = permutation_loss(SimpleModel(), x, y, {
            'seed': 8, 'batch_size': 7, 'device':'cpu', 'permutation_repeats':2})
        np.testing.assert_array_equal(x, original)
        self.assertLess(edge[0,0], 0)
        self.assertTrue(np.isfinite(edge).all())


if __name__ == '__main__':
    unittest.main()
