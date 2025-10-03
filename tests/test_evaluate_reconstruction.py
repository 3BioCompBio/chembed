from typing import List

import unittest

import tempfile
from pathlib import Path
import pandas as pd

import torch
from torch import nn
from torch import Tensor

from chembed.evaluate import get_substitution_counts_for_batch, evaluate_reconstruction
from chembed import data_handler


class DummyVAE(nn.Module):
    def __init__(self, vocab: List[str]):
        super().__init__()
        self.vocab = vocab
        self.idx_O = self.vocab.index('[O]')
        self.idx_C = self.vocab.index('[C]')

    def forward(self, tokens: Tensor):
        # in tokens[0], change all [O] to [C], otherwise don't change anything
        out_tokens = tokens.clone()
        mask = (out_tokens[0]==self.idx_O)
        out_tokens[0][mask] = self.idx_C
        logits = data_handler.tokens_to_logits(out_tokens, len(self.vocab))
        return {'logits': logits}


class TestEvaluateReconstruction(unittest.TestCase):

    def setUp(self):
        self.vocab = ['<START>', '<STOP>', '[C]', '[O]', '[C@]']
        self.word_to_index = {w:i for i, w in enumerate(self.vocab)}
        self.vae = DummyVAE(self.vocab)
        self.device = torch.device('cpu')


    def test_counts_basic(self):
        input_selfies = ['[C][C][C][O]',
                         '[C][C@][C][O]']
        output_selfies = ['[C][C][C][O]',
                          '[C][C][C][O]']

        input_tokens = data_handler.collate_fn_tokens([data_handler.selfies_string_to_tokens(s, self.word_to_index) for s in input_selfies])
        output_tokens = data_handler.collate_fn_tokens([data_handler.selfies_string_to_tokens(s, self.word_to_index) for s in output_selfies])

        substitutions = get_substitution_counts_for_batch(input_tokens, output_tokens, self.vocab)
        
        self.assertEqual(set(substitutions.keys()), set(['[C]','[C@]', '[O]', '<STOP>']))

        self.assertEqual(substitutions['[C]']['[C]'], 5)
        self.assertEqual(substitutions['[C]']['[O]'], 0)
        self.assertEqual(substitutions['[C]']['[C@]'], 0)
        self.assertEqual(substitutions['[C]']['<STOP>'], 0)
        self.assertEqual(substitutions['[C@]']['[C]'], 1)
        self.assertEqual(substitutions['[C@]']['[O]'], 0)
        self.assertEqual(substitutions['[C@]']['<STOP>'], 0)
        self.assertEqual(substitutions['[O]']['[O]'], 2)
        self.assertEqual(substitutions['[O]']['[C]'], 0)


    def test_evaluate_reconstruction(self):

        test_df = pd.DataFrame({'selfies': ['[C][O]','[C][O]', '[C][C]']})

        with tempfile.TemporaryDirectory() as tmpd:
            reconstructed_path = Path(tmpd) / 'reconstructed.csv'

            res = evaluate_reconstruction(self.vae,
                                          test_df = test_df,
                                          device = self.device,
                                          write_reconstructed_to = reconstructed_path,
                                          batch_size = 2,
                                          num_workers = 0
                                          )

            self.assertEqual(res['total_nb_strings'], len(test_df))
            self.assertEqual(res['nb_strings_reconstructed'], 2)
            self.assertAlmostEqual(res['string_reconstruction_accuracy'], 2/3)

            self.assertEqual(res['total_nb_tokens'], 9) # stop token is in it
            self.assertEqual(res['nb_tokens_reconstructed'], 8)
            self.assertAlmostEqual(res['token_reconstruction_accuracy'], 8/9)

            self.assertIn('[O]', res['substitutions'])
            self.assertEqual(res['substitutions']['[O]']['[C]'], 1)

            self.assertEqual(res['reconstruction_accuracy_per_token']['[O]'], 0.5)
            self.assertEqual(res['reconstruction_accuracy_per_token']['[C]'], 1.0)

            with open(reconstructed_path, 'r') as f:
                lines = f.read().strip().splitlines()
                self.assertEqual(lines[0], "original,reconstructed")
                self.assertEqual(lines[1], "[C][O],[C][C]")
                self.assertEqual(lines[2], "[C][O],[C][O]")
                self.assertEqual(lines[3], "[C][C],[C][C]")
                

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU available")
    def test_evaluate_reconstruction_gpu(self):

        test_df = pd.DataFrame({'selfies': ['[C][O]','[C][O]', '[C][C]']})
        device = torch.device('cuda')

        with tempfile.TemporaryDirectory() as tmpd:
            reconstructed_path = Path(tmpd) / 'reconstructed.csv'

            res = evaluate_reconstruction(self.vae,
                                          test_df = test_df,
                                          device = device,
                                          write_reconstructed_to = reconstructed_path,
                                          batch_size = 2,
                                          num_workers = 0
                                          )

            self.assertEqual(res['total_nb_strings'], len(test_df))
            self.assertEqual(res['nb_strings_reconstructed'], 2)
            self.assertAlmostEqual(res['string_reconstruction_accuracy'], 2/3)

            self.assertEqual(res['total_nb_tokens'], 9) # stop token is in it
            self.assertEqual(res['nb_tokens_reconstructed'], 8)
            self.assertAlmostEqual(res['token_reconstruction_accuracy'], 8/9)

            self.assertIn('[O]', res['substitutions'])
            self.assertEqual(res['substitutions']['[O]']['[C]'], 1)

            with open(reconstructed_path, 'r') as f:
                lines = f.read().strip().splitlines()
                self.assertEqual(lines[0], "original,reconstructed")
                self.assertEqual(lines[1], "[C][O],[C][C]")
                self.assertEqual(lines[2], "[C][O],[C][O]")
                self.assertEqual(lines[3], "[C][C],[C][C]")


if __name__ == "__main__":
    unittest.main()
