import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
import sys
import os

from chembed.utils import write_df, read_df

MAIN_DIR = Path(os.path.realpath(__file__)).parent.parent
SCRIPTS_DIR = MAIN_DIR/'scripts'
sys.path.append(str(SCRIPTS_DIR))
import preprocess_properties #noqa: E402

class TestPreprocessProperties(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_preprocess_cli(self):
        input_file = self.tmp/'input.csv'
        N = 1000
        titi_properties = np.random.normal(loc=42.0, scale=2, size=N).tolist()
        to_clip_index = 2
        titi_properties[to_clip_index] = 10000000
        toto_properties = np.random.normal(loc=12.0, scale=1.5, size=N).tolist()
        tata_properties = np.random.normal(loc=0.0, scale=1.0, size=N).tolist()
        df = pd.DataFrame({'selfies': ['[X]']*N,
                           'titi': titi_properties,
                           'toto': toto_properties,
                           'tata': tata_properties
                           })
        write_df(df, input_file)

        output_file = self.tmp/'output.csv'
        output_stats = self.tmp/'stats.json'

        epsilon_quantile = 0.01
        args = [str(input_file),
                str(output_file),
                "--properties_to_clip", "titi",
                "--epsilon_quantile", str(epsilon_quantile),
                "--properties_to_normalize", "titi", "toto",
                "--output_stats", str(output_stats)
                ]

        preprocess_properties.main(args)
        self.assertTrue(output_file.is_file())
        self.assertTrue(output_stats.is_file())

        out_df = read_df(output_file)
        for c in df.columns:
            self.assertIn(c, out_df.columns)
        self.assertIn('titi_clipped', out_df.columns)
        self.assertIn('normalized_toto', out_df.columns)
        self.assertIn('normalized_titi_clipped', out_df.columns)
        self.assertNotIn('normalized_titi', out_df.columns)
        self.assertNotIn('toto_clipped', out_df.columns)
        self.assertNotIn('tata_clipped', out_df.columns)
        self.assertNotIn('normalized_tata', out_df.columns)
        self.assertNotIn('normalized_tata_clipped', out_df.columns)
        np.testing.assert_allclose(df['titi'].tolist(), out_df['titi'].tolist())
        np.testing.assert_allclose(df['tata'].tolist(), out_df['tata'].tolist())
        self.assertLess(out_df['titi_clipped'].iloc[to_clip_index], out_df['titi'].iloc[to_clip_index])
        self.assertAlmostEqual(out_df['normalized_titi_clipped'].mean(), 0.0)
        self.assertAlmostEqual(out_df['normalized_toto'].mean(), 0.0)
        self.assertAlmostEqual(out_df['normalized_titi_clipped'].std(), 1.0)
        self.assertAlmostEqual(out_df['normalized_toto'].std(), 1.0)

if __name__=="__main__":
    unittest.main()
