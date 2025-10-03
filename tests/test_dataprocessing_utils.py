import unittest

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from chembed import dataprocessing_utils as dpu
from chembed.data_handler import load_vocab
from chembed.build_vocab import main as build_vocab_main


class TestDataProcessingUtils(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_build_vocab(self):
        selfies_list = ['[C][O][C]', '[C][C][C]', '[F]']
        vocab = dpu.build_vocab(selfies_list)
        self.assertTrue(isinstance(vocab, list))
        self.assertEqual(set(vocab), set(['[C]', '[O]', '[F]']))

    def test_build_vocab_file_from_train_file(self):
        selfies_list = ['[C][O][C]', '[C][C][C]', '[F]']
        df = pd.DataFrame({'selfies': selfies_list})
        train_file = self.tmp/'train.csv'
        df.to_csv(train_file, index=False)
        vocab_file = self.tmp/'vocab.json'
        dpu.build_vocab_file_from_train_file(train_file, vocab_file)
        self.assertTrue(vocab_file.is_file())
        vocab = load_vocab(vocab_file)
        self.assertEqual(set(vocab), set(['[C]', '[O]', '[F]', '<START>', '<STOP>']))

    def test_build_vocab_file_from_train_file_cli(self):
        selfies_list = ['[C][O][C]', '[C][C][C]', '[F]']
        df = pd.DataFrame({'selfies': selfies_list})
        train_file = self.tmp/'train.csv'
        df.to_csv(train_file, index=False)
        vocab_file = self.tmp/'vocab.json'
        args = [str(train_file), str(vocab_file)]
        build_vocab_main(args)
        self.assertTrue(vocab_file.is_file())
        vocab = load_vocab(vocab_file)
        self.assertEqual(set(vocab), set(['[C]', '[O]', '[F]', '<START>', '<STOP>']))

    def test_return_df_with_clipped_properties(self):
        N = 100
        to_clip_values = np.random.normal(loc=42.0, scale=0.001, size=N).tolist()
        to_clip_ind = 5
        to_clip_values[to_clip_ind] = 100000
        no_clip_values = np.random.normal(loc=42.0, scale=0.001, size=N).tolist()
        no_clip_values[to_clip_ind] = 100000
        df = pd.DataFrame.from_dict({'toto': to_clip_values, 'titi': no_clip_values}) 
        df = dpu.return_df_with_clipped_properties(df, properties_to_clip = ['toto'], epsilon_quantile = 0.1)
        self.assertNotIn('titi_clipped', df.columns)
        self.assertIn('toto_clipped', df.columns)
        self.assertEqual(df['titi'].tolist(), no_clip_values)
        self.assertEqual(df['toto'].tolist(), to_clip_values)
        self.assertLess(df['toto_clipped'].iloc[to_clip_ind], to_clip_values[to_clip_ind])

    def test_normalize_and_compute_statistics(self):
        old_df = pd.DataFrame.from_dict({'toto': [0.0, 1.0, 0.5], 'titi': [0.0, 1.0, 0.5]})
        df, stats = dpu.normalize_and_compute_statistics(old_df, properties = ['titi'])
        self.assertEqual(old_df['titi'].tolist(), df['titi'].tolist())
        self.assertIn('normalized_titi', df.columns)
        self.assertNotIn('normalized_toto', df.columns)
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats['titi']['mean'], 0.5)


if __name__ == "__main__":
    unittest.main()
