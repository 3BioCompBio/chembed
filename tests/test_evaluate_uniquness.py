import unittest

from chembed.metrics import uniqueness


class TestUniqueness(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(uniqueness([]), 0.0)

    def test_basic(self):
        smiles = ["CCO", "CCO", "N#N"]
        self.assertAlmostEqual(uniqueness(smiles), 2 / 3)


if __name__ == "__main__":
    unittest.main()
