import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from chembed import evaluate


class DummyVAE:
    def __init__(self, device, vocab_size=8, latent_dim=8):
        self.vocab = [f"T{i}" for i in range(vocab_size)]
        self.vocab_to_index = {"<STOP>": 1, "<PAD>": 0}
        self.latent_dimension = (latent_dim,)
        self.device = device

    def to(self, device):
        return self

    def eval(self):
        return self

    def decode_inference(self, z):
        B = z.shape[0]
        T = 3
        V = len(self.vocab)
        return torch.zeros((B, T, V), dtype=torch.float32, device=z.device)



class SELFIESBatcher:
    """Returns the next pre-defined list of SELFIES strings on each call."""
    def __init__(self, batches):
        self._batches = list(batches)
        self._idx = 0

    def __call__(self, tokens, vocab):
        if self._idx >= len(self._batches):
            raise AssertionError("SELFIES batches exhausted")
        out = self._batches[self._idx]
        self._idx += 1
        # sanity: batch size must match
        if tokens.shape[0] != len(out):
            raise AssertionError("Batch size mismatch in tokens_to_selfies_strings stub")
        return out




class TestGenerationEvaluation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.vae = DummyVAE(self.device)

    def test_uniqueness_and_validity_basic(self):
        s_batches = SELFIESBatcher([["[C][C]", "[C][C]"], ["[O][O]", "[O][O]"]])

        with patch("chembed.evaluate.data_handler.tokens_to_selfies_strings", side_effect=s_batches):
            res = evaluate.evaluate_generation(
                vae=self.vae,
                device=self.device,
                nb_samples=4,
                batch_size=2,
                test_novelty=False,
                train_file=None,
                smiles_column_name="smiles",
            )

        self.assertIn("uniqueness_4", res)
        self.assertIn("validity_4", res)
        self.assertAlmostEqual(res["uniqueness_4"], 0.5)
        self.assertAlmostEqual(res["validity_4"], 1.0)

    def test_batched_vs_nb_samples_not_multiple(self):
        s_batches = SELFIESBatcher([["[C][C]", "[O][O]"], ["[C][C]"]])
        with patch("chembed.evaluate.data_handler.tokens_to_selfies_strings", side_effect=s_batches):
            res = evaluate.evaluate_generation(
                vae=self.vae,
                device=self.device,
                nb_samples=3,
                batch_size=2,
                test_novelty=False,
                train_file=None,
                smiles_column_name="smiles",
            )
            self.assertIn("uniqueness_3", res)
            self.assertIn("validity_3", res)
            self.assertAlmostEqual(res["uniqueness_3"], 2/3)
            self.assertAlmostEqual(res["validity_3"], 1.0)

    def test_novelty_appended_when_requested(self):
        s_batches = SELFIESBatcher([["[C][C]", "[N][XX]"], ["[C][C]"]])

        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td)/"train.csv"

            novelty_spy = MagicMock(return_value={
                "selfies_novelty_score": 1.0,
                "mol_novelty_score": 0.5,
                "fingerprint_novelty_score": 0.0,
            })

            with patch.object(evaluate, "get_novelty_scores_from_train_file", side_effect=novelty_spy), \
                patch("chembed.evaluate.data_handler.tokens_to_selfies_strings", side_effect=s_batches):
                res = evaluate.evaluate_generation(
                    vae=self.vae,
                    device=self.device,
                    nb_samples=3,
                    batch_size=2,
                    test_novelty=True,
                    train_file=train_path,
                    smiles_column_name="smiles",
                )

            self.assertIn("uniqueness_3", res)
            self.assertIn("validity_3", res)
            self.assertIn("selfies_novelty_score_3", res)
            self.assertIn("mol_novelty_score_3", res)
            self.assertIn("fingerprint_novelty_score_3", res)
            self.assertEqual(res["selfies_novelty_score_3"], 1.0)
            self.assertEqual(res["mol_novelty_score_3"], 0.5)
            self.assertEqual(res["fingerprint_novelty_score_3"], 0.0)
            self.assertEqual(res["validity_3"], 2/3)

            self.assertTrue(novelty_spy.called)
            args, kwargs = novelty_spy.call_args
            gen_selfies_arg = args[0]
            gen_mols_arg = args[1]
            self.assertEqual(len(gen_selfies_arg), 3)
            self.assertEqual(len(gen_mols_arg), 2)


    def test_write_invalid_to_file(self):
        s_batches = SELFIESBatcher([["[C][C]", "[N][XX]"], ["[C][C]"]])

        with tempfile.TemporaryDirectory() as td:
            invalid_file = Path(td)/'invalid.csv'

            with patch("chembed.evaluate.data_handler.tokens_to_selfies_strings", side_effect=s_batches):
                evaluate.evaluate_generation(
                        vae=self.vae,
                        device=self.device,
                        nb_samples=3,
                        batch_size=2,
                        test_novelty=False,
                        train_file=None,
                        smiles_column_name='smiles',
                        write_invalid_to=invalid_file
                        )

                self.assertTrue(invalid_file.is_file())
                
                with open(invalid_file, 'r') as f:
                    lines = [line.rstrip() for line in f.readlines()]
                self.assertEqual(len(lines), 1)
                self.assertEqual(lines[0], "[N][XX]")




if __name__ == "__main__":
    unittest.main()

