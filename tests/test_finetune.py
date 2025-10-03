import unittest
from unittest.mock import patch, MagicMock

from pathlib import Path
import tempfile
import pandas as pd
import selfies as sf
import re

from chembed import finetune, checkpoint_utils
from chembed.dataprocessing_utils import build_vocab_file_from_train_file
from chembed import data_handler
from chembed import encode as enc


class TestFinetune(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.selfies_train = ['[X][X][X]', '[X][X]', '[X][X][X][X]']
        df_train = pd.DataFrame.from_dict({'selfies': self.selfies_train})
        df_train['len'] = df_train['selfies'].apply(lambda s: len(list(sf.split_selfies(s))))
        self.train_file = self.tmp/'train.csv'
        df_train.to_csv(self.train_file, index=False)
        self.validation_file = self.tmp/'validation.csv'
        df_train.to_csv(self.validation_file, index=False)
        self.vocab_file = self.tmp/'vocab.json'
        build_vocab_file_from_train_file(self.train_file, self.vocab_file)
        self.log_dir = self.tmp/'logs'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_cli_reconstruction(self):
        vae_before = checkpoint_utils.load_vae_from_hub('cpu')
        vocab_before = vae_before.vocab
        model_name = "Billy"
        batch_size = 3

        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.validation_file),
                "--model_name", str(model_name),
                "--log_dir", str(self.log_dir),
                "--max_epochs", str(1),
                "--dont_train_with_tanimoto_similarity",
                "--dont_train_with_properties",
                "--vocab", str(self.vocab_file),
                "--batch_size", str(batch_size)
                ]
        finetune.main(args)
        new_vae = checkpoint_utils.load_vae_from_checkpoint(self.log_dir/model_name/"version_0"/"checkpoints"/"last.ckpt", 'cpu')

        new_vocab = data_handler.load_vocab(self.vocab_file) 
        self.assertEqual(set(new_vae.vocab), set(vocab_before).union(new_vocab))

        enc.encode_multiple_selfies(self.selfies_train, new_vae)


    def test_cli_with_properties_but_no_file_raises(self):
        model_name = "Billy"
        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.validation_file),
                "--model_name", str(model_name),
                "--log_dir", str(self.log_dir),
                "--max_epochs", str(1),
                "--dont_train_with_tanimoto_similarity",
                "--train_with_properties",
                "--vocab", str(self.vocab_file),
                ]

        with self.assertRaises(ValueError) as e:
            finetune.main(args)

        self.assertIn("must be provided", str(e.exception))



    def test_cli_with_properties_but_no_file_raises_even_when_providing_properties_list(self):
        model_name = "Billy"
        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.validation_file),
                "--model_name", str(model_name),
                "--log_dir", str(self.log_dir),
                "--max_epochs", str(1),
                "--dont_train_with_tanimoto_similarity",
                "--train_with_properties",
                "--vocab", str(self.vocab_file),
                "--properties", "len"
                ]

        with self.assertRaises(ValueError) as e:
            finetune.main(args)
        self.assertIn("must be provided", str(e.exception))


    @patch("chembed.finetune.finetune_vae", return_value=None)
    def test_cli_no_new_vocab(self, mock_finetune: MagicMock):
        model_name = "Billy"
        with self.assertWarnsRegex(UserWarning, re.escape('vocab')):
            args = [
                    "--train_path", str(self.train_file),
                    "--validation_path", str(self.validation_file),
                    "--model_name", str(model_name),
                    "--log_dir", str(self.log_dir),
                    "--max_epochs", str(1),
                    "--dont_train_with_tanimoto_similarity",
                    "--dont_train_with_properties",
                    ]
            finetune.main(args)
            mock_finetune.assert_called_once()



if __name__ == "__main__":
    unittest.main()

