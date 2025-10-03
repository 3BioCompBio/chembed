import unittest
import os
from pathlib import Path
import tempfile
import pandas as pd
import selfies as sf

from chembed import dataprocessing_utils as dpu 
from chembed.utils import write_json
from chembed import train as chembed_train
from chembed import mol_utils

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.smiles = ['C', 'CO', 'OC', 'CC']
        df = pd.DataFrame.from_dict({'smiles': self.smiles})
        df['len'] = df['smiles'].apply(len)
        df['selfies'] = df['smiles'].apply(sf.encoder)
        df['fingerprint'] = df['smiles'].apply(mol_utils.get_fingerprint_from_smiles)
        self.properties = ['len']
        df, stats = dpu.normalize_and_compute_statistics(df, self.properties)
        self.train_file = self.tmp/'train.csv'
        df.to_csv(self.train_file, index=False)
        self.stats_file = self.tmp/'stats.json'
        write_json(stats, self.stats_file)
        self.vocab_file = self.tmp/'vocab.json'
        dpu.build_vocab_file_from_train_file(self.train_file, self.vocab_file)
        self.log_dir = self.tmp/'logs'

    def tearDown(self):
        self.tmpdir.cleanup()


    def test_cli_train_precomputed_fingerprints(self):
        model_name = "Billy"
        batch_size = 3
        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.train_file),
                "--properties_statistics_path", str(self.stats_file),
                "--properties", *self.properties,
                "--log_dir", str(self.log_dir),
                "--model_name", model_name,
                "--batch_size", str(batch_size),
                "--num_workers", str(0),
                "--max_epochs", str(1),
                "--vocab", str(self.vocab_file),
                "--use_precomputed_fingerprints"
                ]
        chembed_train.main(args)
        checkpath = self.log_dir/model_name/'version_0'/'checkpoints'/'last.ckpt'
        self.assertTrue(checkpath.is_file())
        

    def test_cli_train_no_precomputed_fingerprints(self):
        model_name = "Billy"
        batch_size = 3
        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.train_file),
                "--properties_statistics_path", str(self.stats_file),
                "--properties", *self.properties,
                "--log_dir", str(self.log_dir),
                "--model_name", model_name,
                "--batch_size", str(batch_size),
                "--num_workers", str(0),
                "--max_epochs", str(1),
                "--vocab", str(self.vocab_file),
                ]
        chembed_train.main(args)
        checkpath = self.log_dir/model_name/'version_0'/'checkpoints'/'last.ckpt'
        self.assertTrue(checkpath.is_file())


    def test_cli_deterministic_raises(self):
        model_name = "Billy"
        batch_size = 3
        if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
            del os.environ['CUBLAS_WORKSPACE_CONFIG']
        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.train_file),
                "--properties_statistics_path", str(self.stats_file),
                "--properties", *self.properties,
                "--log_dir", str(self.log_dir),
                "--model_name", model_name,
                "--batch_size", str(batch_size),
                "--num_workers", str(0),
                "--max_epochs", str(1),
                "--vocab", str(self.vocab_file),
                "--strictly_deterministic"
                ]
        with self.assertRaises(RuntimeError):
            chembed_train.main(args)

    def test_cli_train_batch_size_too_large(self):
        model_name = "Billy"
        batch_size = 256
        args = [
                "--train_path", str(self.train_file),
                "--validation_path", str(self.train_file),
                "--properties_statistics_path", str(self.stats_file),
                "--properties", *self.properties,
                "--log_dir", str(self.log_dir),
                "--model_name", model_name,
                "--batch_size", str(batch_size),
                "--num_workers", str(0),
                "--max_epochs", str(1),
                "--vocab", str(self.vocab_file),
                "--use_precomputed_fingerprints"
                ]
        chembed_train.main(args)
        checkpath = self.log_dir/model_name/'version_0'/'checkpoints'/'last.ckpt'
        self.assertTrue(checkpath.is_file())
 

if __name__ == "__main__":
    unittest.main()
