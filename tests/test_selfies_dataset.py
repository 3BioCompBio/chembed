import unittest
from unittest.mock import patch

from pathlib import Path
import pandas as pd
import torch
import torch.testing as tt

from chembed import data_handler
from chembed.data_handler import PAD_TOKEN
from chembed.utils import set_random_seed_everywhere

class TestSELFIESDataset(unittest.TestCase):
    def setUp(self):
        self.vocab = ['<START>', '<STOP>', '[C]', '[O]']
        self.word_to_index = {t: i for i, t in enumerate(self.vocab)}
        self.properties = ['MolWt', 'MolLogP']


    def test_dataset_with_normalized_props_and_precomputed_fp_string(self):
        df = pd.DataFrame([{
            'selfies': '[C][O]',
            'normalized_MolWt': 0.1,
            'normalized_MolLogP': -0.2,
            'fingerprint': '[1, 0, 1, 0]',
            }])

        dataset = data_handler.SELFIESDataset(
            df=df,
            vocab=self.vocab,
            properties=self.properties,
            return_fingerprints=True,
            return_properties=True,
            use_normalized_properties=True,
            replace_if_not_in_vocab=False,
        )
        tokens, props, fps = dataset[0]

        tt.assert_close(tokens, torch.tensor([2, 3, 1], dtype=torch.long))
        tt.assert_close(props, torch.tensor([0.1, -0.2], dtype=torch.float32))
        tt.assert_close(fps, torch.tensor([1, 0, 1, 0], dtype=torch.int64))



    def test_dataset_with_raw_props_and_computed_fp(self):
        df = pd.DataFrame([{
            'selfies': '[C]',
            'smiles': 'C',
            'MolWt': 12.0,
            'MolLogP': 0.5,
        }])

        with patch.object(data_handler, 'get_fingerprint_from_smiles', return_value=[0, 1, 0, 1]):
            dataset = data_handler.SELFIESDataset(
                df=df,
                vocab=self.vocab,
                properties=self.properties,
                return_fingerprints=True,
                return_properties=True,
                use_normalized_properties=False,
                replace_if_not_in_vocab=False,
            )
            tokens, props, fps = dataset[0]

        tt.assert_close(tokens, torch.tensor([2, 1], dtype=torch.long))
        tt.assert_close(props, torch.tensor([12.0, 0.5], dtype=torch.float32))
        tt.assert_close(fps, torch.tensor([0, 1, 0, 1], dtype=torch.int64))


    def test_dataset_no_props_no_fp(self):
        df = pd.DataFrame([{'selfies': '[C][O]'}])

        dataset = data_handler.SELFIESDataset(
            df=df,
            vocab=self.vocab,
            properties=self.properties,
            return_fingerprints=False,
            return_properties=False,
            use_normalized_properties=True,
        )
        tokens, props, fps = dataset[0]

        tt.assert_close(tokens, torch.tensor([2, 3, 1], dtype=torch.long))
        self.assertIsNone(props)
        self.assertIsNone(fps)


    def test_datamodule_with_precomputed_fps_and_normalized_props(self):
        df_train = pd.DataFrame([
            {'selfies': '[C]',   'normalized_MolWt': 0.1, 'normalized_MolLogP': 0.2, 'fingerprint': '[1,0,0,1]'},
            {'selfies': '[C][O]','normalized_MolWt': 0.3, 'normalized_MolLogP': 0.4, 'fingerprint': '[0,1,1,0]'},
        ])
        df_val = df_train.copy()


        with patch.object(data_handler, 'read_df', side_effect=[df_train, df_val]) as mock_read_df:
            train_config = {
                'batch_size': 2,
                'num_workers': 0,
                'train_with_tanimoto_similarity': True,     # triggers fingerprint column usage
                'use_precomputed_fingerprints': True,
                'train_with_properties': True,
                'use_normalized_properties': True,
                'replace_if_not_in_vocab': False,
            }

            datamodule = data_handler.SELFIESDataModule(
                train_data_path=Path('train.parquet'),
                validation_data_path=Path('validation.parquet'),
                properties=self.properties,
                vocab=self.vocab,
                train_config=train_config,
            )

            calls = mock_read_df.call_args_list
            self.assertEqual(len(calls), 2)

            for c in calls:
                _path, cols = c.args
                self.assertIn('selfies', cols)
                self.assertIn('normalized_MolWt', cols)
                self.assertIn('normalized_MolLogP', cols)
                self.assertIn('fingerprint', cols)

        # iterate one batch
        dataloader = datamodule.val_dataloader()
        batch = next(iter(dataloader))
        tokens, props, fps = batch

        # tokens should be padded to length 3
        self.assertEqual(tokens.shape, (len(df_train), 3))
        tt.assert_close(tokens[0], torch.tensor([2, PAD_TOKEN, PAD_TOKEN], dtype=torch.long))
        tt.assert_close(tokens[1], torch.tensor([2, 3, PAD_TOKEN], dtype=torch.long))

        self.assertEqual(props.shape, (len(df_train), len(self.properties)))

        self.assertEqual(fps.shape, (len(df_train), 4))


    def test_datamodule_computed_fps_and_raw_props(self):
        df_train = pd.DataFrame([{'selfies': '[C]', 'smiles': 'C', 'MolWt': 12.0, 'MolLogP': 0.5}])
        df_val = df_train.copy()

        with patch.object(data_handler, 'get_fingerprint_from_smiles', return_value=[1, 1, 0, 0]), \
             patch.object(data_handler, 'read_df', side_effect=[df_train, df_val]) as mock_read_df:

            train_config = {
                'batch_size': 1,
                'num_workers': 0,
                'train_with_tanimoto_similarity': True,
                'use_precomputed_fingerprints': False, 
                'train_with_properties': True,
                'use_normalized_properties': False,
                'replace_if_not_in_vocab': False,
            }

            datamodule = data_handler.SELFIESDataModule(
                train_data_path=Path('train.csv'),
                validation_data_path=Path('validation.csv'),
                properties=self.properties,
                vocab=self.vocab,
                train_config=train_config,
            )

            # verify 'smiles' requested, not 'fingerprint'
            _, cols1 = mock_read_df.call_args_list[0].args
            _, cols2 = mock_read_df.call_args_list[1].args
            for cols in (cols1, cols2):
                self.assertIn('smiles', cols)
                self.assertNotIn('fingerprint', cols)
                self.assertIn('MolWt', cols)
                self.assertIn('MolLogP', cols)

            dataloader = datamodule.train_dataloader()
            tokens, props, fps = next(iter(dataloader))

        tt.assert_close(tokens, torch.tensor([[2, 1]], dtype=torch.long))
        tt.assert_close(props, torch.tensor([[12.0, 0.5]], dtype=torch.float32))
        tt.assert_close(fps, torch.tensor([[1, 1, 0, 0]], dtype=torch.int64))


if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main(verbosity=2)

