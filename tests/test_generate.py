import unittest
from pathlib import Path
import tempfile

from chembed import generate as gen


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp/name
        path.write_text(text, encoding="utf-8")
        return path

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_cli(self):
        output_file = self.tmp/'output.csv'
        N = 10
        gen.main([str(N), str(output_file)])
        self.assertTrue(output_file.is_file())
        with open(output_file, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), N+1)

if __name__ == "__main__":
    unittest.main()

