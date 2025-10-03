import unittest
import tempfile
from pathlib import Path

import torch
import torch.testing as tt

from chembed.regressor import PropertiesRegressor
from chembed.utils import set_random_seed_everywhere


class TestPropertiesRegressor(unittest.TestCase):
    def setUp(self):
        self.properties = ["MolWt", "MolLogP", "TPSA"]
        self.stats = {
            "MolWt":   {"mean": 300.0, "std": 50.0},
            "MolLogP": {"mean": 2.0,   "std": 1.0},
            "TPSA":    {"mean": 60.0,  "std": 20.0},
        }
        self.z_dim = 256
        self.B = 4
        self.device = torch.device("cpu")

    def test_shapes_and_forward_no_layernorm(self):
        reg = PropertiesRegressor(self.properties, self.stats, self.z_dim, use_layer_norm=False, layer_norm_eps=1e-5).to(self.device)

        z_flat = torch.randn(self.B, self.z_dim, device=self.device)
        y = reg(z_flat)
        self.assertEqual(y.shape, (self.B, len(self.properties)))
        tt.assert_close(y, reg.predict_normalized(z_flat))

    def test_shapes_with_layernorm(self):
        reg = PropertiesRegressor(self.properties, self.stats, self.z_dim, use_layer_norm=True, layer_norm_eps=1e-5).to(self.device)
        z = torch.randn(self.B, self.z_dim, device=self.device)
        y = reg(z)
        self.assertEqual(y.shape, (self.B, len(self.properties)))

    def test_denormalize_and_back(self):
        reg = PropertiesRegressor(self.properties, self.stats, self.z_dim, use_layer_norm=False, layer_norm_eps=1e-5).to(self.device)
        z = torch.randn(3, self.z_dim, device=self.device)
        tt.assert_close(reg.predict_unnormalized(z), reg.denormalize_properties(reg.predict_normalized(z)))

    def test_from_checkpoint(self):
        base_reg = PropertiesRegressor(self.properties, self.stats, self.z_dim, use_layer_norm=True, layer_norm_eps=1e-5).to(self.device)
        state = {
            "hyper_parameters": {
                "properties_statistics": self.stats,
                "model_config": {
                    "d_latent": self.z_dim,
                    "properties": self.properties,
                    "layer_norm_in_regressor": True,
                    "layer_norm_eps": 1e-5,
                },
            },
            "state_dict": {f"regressor.{k}": v for k, v in base_reg.state_dict().items()},
        }

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td)/"reg_new.ckpt"
            torch.save(state, ckpt)
            loaded_reg = PropertiesRegressor.from_checkpoint(ckpt, device=self.device)

        self.assertEqual(loaded_reg.properties, self.properties)
        z = torch.randn(self.B, self.z_dim, device=self.device)
        y_base_reg = base_reg(z)
        y_loaded_reg = loaded_reg(z)
        tt.assert_close(y_loaded_reg, y_base_reg)


if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main(verbosity=2)
