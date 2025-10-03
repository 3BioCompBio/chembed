import unittest
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import pandas as pd
import selfies as sf

from chembed import dataprocessing_utils as dpu 
from chembed.utils import write_json
from chembed import mol_utils

class TestTrainDeterminism(unittest.TestCase):
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

    def test_cli_deterministic_does_not_raise_if_set(self):
        model_name = 'Billy'
        env = dict(os.environ)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        cmd = [
            sys.executable, "-m", "chembed.train",
            "--train_path", str(self.train_file),
            "--validation_path", str(self.train_file),
            "--properties_statistics_path", str(self.stats_file),
            "--properties", *self.properties,
            "--log_dir", str(self.log_dir),
            "--model_name", model_name,
            "--batch_size", "3",
            "--num_workers", "0",
            "--max_epochs", "1",
            "--vocab", str(self.vocab_file),
            "--strictly_deterministic",
        ]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
        checkpath = self.log_dir/model_name/'version_0'/'checkpoints'/'last.ckpt'
        self.assertTrue(checkpath.is_file())

if __name__=="__main__":
    unittest.main()
