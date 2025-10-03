import unittest

from chembed import token_embeddings_utils
from chembed.utils import set_random_seed_everywhere

class TestTokenEmbeddingsUtils(unittest.TestCase):

    def test_break_down_token(self):
        token = "[#B-1]"
        token_broken_down = token_embeddings_utils.break_down_token(token)
        self.assertEqual(token_broken_down, ('#', 'B', '', None, '-1'))


if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main()
