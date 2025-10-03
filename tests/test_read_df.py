import unittest
import tempfile
import shutil
from pathlib import Path

import pandas as pd

from chembed.utils import read_df

class TestReadDf(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="read_df_tests_"))
        self.df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [10, 20, 30],
                "c": ["x", "y", "z"],
            }
        )
        self.csv_path = self.tmpdir/"data.csv"
        self.df.to_csv(self.csv_path, index=False)

        self.pkl_path = self.tmpdir/"data.pkl"
        self.df.to_pickle(self.pkl_path)

        self.parquet_path = self.tmpdir/"data.parquet"
        self.df.to_parquet(self.parquet_path, engine="fastparquet")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_csv_required_and_desired(self):
        out = read_df(self.csv_path, required_columns=["a"], desired_columns=["b", "x"], sep=",")
        self.assertListEqual(list(out.columns), ["a", "b"])

    def test_csv_required_missing_raises(self):
        with self.assertRaises(KeyError):
            read_df(self.csv_path, required_columns=["missing"], desired_columns=["a"], sep=",")

    def test_csv_desired_only(self):
        out = read_df(self.csv_path, required_columns=None, desired_columns=["b", "missing"], sep=",")
        self.assertListEqual(list(out.columns), ["b"])

    def test_csv_no_hints_loads_all(self):
        out = read_df(self.csv_path, required_columns=None, desired_columns=None, sep=",")
        self.assertListEqual(list(out.columns), ["a", "b", "c"])

    def test_pkl_required_and_desired(self):
        out = read_df(self.pkl_path, required_columns=["b"], desired_columns=["c", "nope"])
        self.assertListEqual(list(out.columns), ["b", "c"])

    def test_pkl_required_missing_raises(self):
        with self.assertRaises(KeyError):
            read_df(self.pkl_path, required_columns=["zzz"], desired_columns=None)

    def test_pkl_desired_only(self):
        out = read_df(self.pkl_path, required_columns=None, desired_columns=["c"])
        self.assertListEqual(list(out.columns), ["c"])

    def test_parquet_required_and_desired(self):
        out = read_df(self.parquet_path, required_columns=["a"], desired_columns=["c", "nope"])
        self.assertListEqual(list(out.columns), ["a", "c"])

    def test_parquet_required_missing_raises(self):
        with self.assertRaises(KeyError):
            read_df(self.parquet_path, required_columns=["missing"], desired_columns=["a"])

    def test_parquet_desired_only(self):
        out = read_df(self.parquet_path, required_columns=None, desired_columns=["b", "nope"])
        self.assertListEqual(list(out.columns), ["b"])


if __name__ == "__main__":
    unittest.main()

