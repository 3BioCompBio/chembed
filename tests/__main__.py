import sys
import os
import unittest
from pathlib import Path

THIS_DIR = Path(os.path.realpath(__file__)).parent


if __name__ == "__main__":
    start_dir = THIS_DIR.resolve()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(start_dir))

    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
