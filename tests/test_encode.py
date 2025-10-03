import unittest
from pathlib import Path
import tempfile
import pandas as pd
import torch
import torch.testing as tt

from chembed import encode as enc
from chembed.utils import set_random_seed_everywhere


class Dummy_VAE:
    def __init__(self, vocab, device, d_model):
        self.vocab = vocab
        self.device = torch.device(device)
        self.d_bottleneck = 1
        self.d_model = d_model
        self.latent_dimension = (self.d_bottleneck, self.d_model)

    def eval(self):
        pass

    def encode(self, tokens: torch.Tensor):
        B = tokens.shape[0]
        base = torch.arange(B, dtype=torch.float32, device=self.device).reshape(B, 1, 1)
        mu = base.expand(B, 1, self.d_model).clone() # mu[b, i, :] = b
        sigma = torch.randn_like(mu)
        return mu, sigma


class TestEncode(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.vocab = ['<START>', '<STOP>', '[C]', '[O]']
        self.smiles_list = ['C', 'CC', 'COC']
        self.selfies_list = ["[C]", "[C][C]", "[C][O][C]"]
        self.df = pd.DataFrame({"selfies": self.selfies_list})
        self.device = 'cpu'
        self.d_model = 32
        self.vae = Dummy_VAE(vocab=self.vocab, device=self.device, d_model=self.d_model) 

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp/name
        path.write_text(text, encoding="utf-8")
        return path

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_encode_selfies_in_df_lower_batch_size(self):
        batch_size = 2
        z = enc.encode_selfies_in_df(self.df, self.vae, batch_size=batch_size, use_sigma=False)
        self.assertEqual(z.shape, (len(self.df), 1, self.d_model))

        for b in range(len(self.df)):
            b_mod = b % batch_size
            tt.assert_close(z[b], torch.full((1, self.d_model), float(b_mod)))


    def test_encode_selfies_in_df_with_sigma(self):
        batch_size = len(self.df)
        z = enc.encode_selfies_in_df(self.df, self.vae, batch_size=batch_size, use_sigma=True)

        self.assertEqual(z.shape, (len(self.df), 1, self.d_model))

        for b in range(len(self.df)):
            b_mod = b % batch_size
            self.assertFalse(torch.equal(z[b], torch.full((1, self.d_model), float(b_mod))))


    def test_encode_multiple_selfies_empty_raises(self):
        with self.assertRaises(ValueError):
            enc.encode_multiple_selfies([], self.vae)

    def test_encode_multiple_selfies_list(self):
        batch_size = 2
        z = enc.encode_multiple_selfies(self.selfies_list, self.vae, batch_size=batch_size)
        self.assertEqual(z.shape, (len(self.selfies_list), 1, self.d_model))

        for b in range(len(self.df)):
            b_mod = b % batch_size
            tt.assert_close(z[b], torch.full((1, self.d_model), float(b_mod)))


    def test_encode_multiple_smiles(self):
        batch_size = 2
        z = enc.encode_multiple_smiles(self.smiles_list, self.vae, batch_size=batch_size)

        for b in range(len(self.df)):
            b_mod = b % batch_size
            tt.assert_close(z[b], torch.full((1, self.d_model), float(b_mod)))

    def test_cli(self):
        f = self._write("test.smi", "SMILES Name\nCCO ethanol\nCCN amine\n")
        output_file = self.tmp/'output.npy'
        enc.main(["--input_smiles", str(f), "--output", str(output_file)])
        self.assertTrue(output_file.is_file())



if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main()

