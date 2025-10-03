import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import torch

from chembed.evaluate import evaluate_latent_space

class TestEvaluateLatentSpace(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        smiles_list = ["CC", "CCC", "CCCC", "CCO"]
        self.df = pd.DataFrame({"smiles": smiles_list, "MolWt": [len(s) for s in smiles_list]})
        self.vae = object()

    @patch("chembed.evaluate.enc.encode_selfies_in_df")
    def test_evaluate_latent_space_no_properties(self, mock_encode):
        mock_encode.return_value = torch.vstack([
            torch.from_numpy(np.full((2, 2), (-1.0, 0.0), dtype=np.float32)),
            torch.from_numpy(np.full((2, 2), ( 1.0, 0.0), dtype=np.float32)),
        ])

        res = evaluate_latent_space(self.vae, self.df, self.device, batch_size=2)

        mock_encode.assert_called_once()
        for k in res:
            if k.startswith('trustworthiness'):
                self.assertGreaterEqual(res[k], 0.0)
                self.assertLessEqual(res[k], 1.0)

    def test_error_when_no_cols(self):
        with self.assertRaises(ValueError):
            evaluate_latent_space(self.vae, pd.DataFrame({"foo": ["CC"]}), self.device)

    @patch("chembed.evaluate.enc.encode_selfies_in_df")
    def test_evaluate_latent_space_with_properties(self, mock_encode):
        mock_encode.return_value = torch.vstack([
            torch.from_numpy(np.full((2, 2), (-1.0, 0.0), dtype=np.float32)),
            torch.from_numpy(np.full((2, 2), ( 1.0, 0.0), dtype=np.float32)),
        ])
        res = evaluate_latent_space(self.vae, self.df, self.device, batch_size=2, properties_to_evaluate=['MolWt'])
        mock_encode.assert_called_once()
        self.assertIn('pearson_correlation_euclidean_MolWt_distance', res.keys())


if __name__ == "__main__":
    unittest.main()

