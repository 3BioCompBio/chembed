import unittest
import torch
from chembed import metrics

class TestMetrics(unittest.TestCase):
    
    def test_tanimoto_cos_loss_warns(self):
        zs = torch.zeros((3,3))
        fps = torch.zeros_like(zs)
        with self.assertWarns(DeprecationWarning):
            metrics.tanimoto_cos_loss(zs, fps)

if __name__ == "__main__":
    unittest.main()
