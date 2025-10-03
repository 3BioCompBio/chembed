import unittest
import warnings

import selfies as sf
import torch
import torch.testing as tt

from chembed import data_handler
from chembed.errors import UnknownTokenError, TokenReplacementWarning

class TestTokenize(unittest.TestCase):
    def setUp(self):
        self.vocab = ['<START>', '<STOP>', '[C]', '[H]']
        self.word_to_index = {t: i for i, t in enumerate(self.vocab)}

    def test_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            word = '[C]'
            idx = data_handler.get_word_to_index(word, self.word_to_index, replace_if_not_in_vocab=False)
            self.assertEqual(idx, self.word_to_index.get(word))
            self.assertEqual(len(w), 0)

    def test_missing_token_no_replace_raises(self):
        with self.assertRaises(UnknownTokenError):
            data_handler.get_word_to_index('TOTO', self.word_to_index, replace_if_not_in_vocab=False)

    def test_hydrogen_isotope_replace_warns(self):
        with self.assertRaises(TokenReplacementWarning):
            data_handler.get_word_to_index('[2H]', self.word_to_index, replace_if_not_in_vocab=True)

    def test_missing_token_replace_warns(self):
        with self.assertWarns(TokenReplacementWarning):
            data_handler.get_word_to_index('[C@]', self.word_to_index, replace_if_not_in_vocab=True)

    def test_missing_token_replace(self):
        idx = data_handler.get_word_to_index('[C@]', self.word_to_index, replace_if_not_in_vocab=True)
        self.assertEqual(self.vocab[idx], '[C]')

    def test_selfies_string_to_tokens(self):
        selfies = "[C][C][H][C]"
        tokens = data_handler.selfies_string_to_tokens(selfies, self.word_to_index, replace_if_not_in_vocab=False)
        self.assertEqual(len(tokens), len(list(sf.split_selfies(selfies)))+1)
        tt.assert_close(tokens, torch.tensor([2, 2, 3, 2, 1]), rtol=0, atol=0)

    def test_tokens_to_selfies_string(self):
        tokens = torch.tensor([2, 2, 3, 2, 1])
        selfies = data_handler.tokens_to_selfies_string(tokens, self.vocab)
        self.assertEqual(selfies, "[C][C][H][C]")

    def test_tokens_to_selfies_strings(self):
        batch_tokens = torch.tensor([[2, 2, 3, 2, 1], [2, 2, 1, 1, 1]])
        selfies_list = data_handler.tokens_to_selfies_strings(batch_tokens, self.vocab)
        self.assertEqual(selfies_list, ["[C][C][H][C]", "[C][C]"])

    def test_selfies_to_tokens_to_logits_to_tokens_to_selfies(self):
        selfies_list = ['[C][H][C][H][H][C]', '[C][C]']
        tokens = [data_handler.selfies_string_to_tokens(s, self.word_to_index) for s in selfies_list]
        tokens = data_handler.collate_fn_tokens(tokens)
        logits = data_handler.tokens_to_logits(tokens, vocab_size=len(self.vocab))
        tokens = data_handler.logits_to_tokens(logits)
        decoded = data_handler.tokens_to_selfies_strings(tokens, self.vocab)
        self.assertEqual(selfies_list, decoded)



if __name__ == "__main__":
    unittest.main()
