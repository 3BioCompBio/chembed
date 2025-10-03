import unittest

import torch
from chembed.checkpoint_utils import load_vae_from_hub

class TestLoadModelFromHub(unittest.TestCase):

    def test_load_vae_cpu(self):
        vae = load_vae_from_hub(device='cpu')
        self.assertTrue(hasattr(vae, "d_model"))

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU available")
    def test_load_vae_gpu(self):
        vae = load_vae_from_hub(device='cuda')
        self.assertTrue(hasattr(vae, "d_model"))


if __name__=="__main__":
    unittest.main()
