import unittest
import tempfile
from pathlib import Path
from chembed.utils import read_smiles_or_selfies

class TestReadSmiles(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp/name
        path.write_text(text, encoding="utf-8")
        return path

    def test_no_header(self):
        f = self._write("no_header.smi", "CCO ethanol\nc1ccccc1 benzene\n")
        self.assertEqual(read_smiles_or_selfies(f), ["CCO", "c1ccccc1"])

    def test_with_header(self):
        f = self._write("with_header.smi", "SMILES Name\nCCO ethanol\nCCN amine\n")
        self.assertEqual(read_smiles_or_selfies(f), ["CCO", "CCN"])

    def test_blank_lines(self):
        f = self._write("blank.smi", "SMILES Name\n\nCCO ethanol\nc1ccccc1 benzene\n")
        self.assertEqual(read_smiles_or_selfies(f), ["CCO", "c1ccccc1"])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_smiles_or_selfies(self.tmp/"does_not_exist.smi")

if __name__ == "__main__":
    unittest.main()
