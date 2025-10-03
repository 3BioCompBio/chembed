from pathlib import Path
import tempfile
import unittest
import torch
import pandas as pd

from chembed.transformer_vae import SELFIESTransformerVAE
from chembed import data_handler
from chembed.utils import set_random_seed_everywhere

from build_tiny_model import build_tiny_model


class TestIntegrationTransformerVAE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")

        cls._tiny_model_tmpdir = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls._tiny_model_tmpdir.name)/"vae_tiny.ckpt"
        build_tiny_model(cls.checkpoint_path, cls.device)

        cls.vae = SELFIESTransformerVAE.from_checkpoint(cls.checkpoint_path, device=cls.device, strict=True)
        cls.vae.eval()

        cls.selfies = ["[C]", "[C][O]", "[O][C]", "[C][C]"]
        cls.df = pd.DataFrame({"selfies": cls.selfies})

        word_to_index = {t: i for i, t in enumerate(cls.vae.vocab)}
        tokens = [data_handler.selfies_string_to_tokens(s, word_to_index, False) for s in cls.selfies]
        cls.tokens = data_handler.collate_fn_tokens(tokens)


    def test_teacher_forcing_reconstruction(self):
        mu, _ = self.vae.encode(self.tokens)
        logits = self.vae.decode_training(mu, self.tokens)
        out_tokens = data_handler.logits_to_tokens(logits)
        self.assertTrue((out_tokens == self.tokens).all().item())

    def test_greedy_reconstruction(self):
        out = self.vae(self.tokens)
        out_tokens = data_handler.logits_to_tokens(out['logits'])
        out_selfies = data_handler.tokens_to_selfies_strings(out_tokens, self.vae.vocab)
        self.assertEqual(out_selfies, self.selfies)


if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main()
