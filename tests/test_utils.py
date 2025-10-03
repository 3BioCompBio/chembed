import unittest

from pathlib import Path
import tempfile
import pandas as pd

from chembed import utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_read_write_df(self):
        df = pd.DataFrame.from_dict({'titi': ['toto'], 'tata': [0.0]})
        for ext in ['.csv', '.parquet', '.pkl']:
            df_file = self.tmp/f'dataset{ext}'
            utils.write_df(df, df_file)
            df_read = utils.read_df(df_file)
            pd.testing.assert_frame_equal(df, df_read)

    def test_read_write_df_with_sep(self):
        df = pd.DataFrame.from_dict({'titi': ['toto'], 'tata': [0.0]})
        sep = ';'
        df_file = self.tmp/'dataset.csv'
        utils.write_df(df, df_file, sep=sep)
        df_read = utils.read_df(df_file, sep=sep)
        pd.testing.assert_frame_equal(df, df_read)


if __name__=="__main__":
    unittest.main()
