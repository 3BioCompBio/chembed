import unittest
from unittest.mock import patch, MagicMock

import torch
from pathlib import Path
import tempfile

from chembed import decode as dec
from chembed import data_handler
from chembed.utils import set_random_seed_everywhere


class Dummy_VAE:
    def __init__(self, vocab, device, d_model):
        self.vocab = vocab
        self.word_to_index = {w: i for i, w in enumerate(vocab)}
        self.device = torch.device(device)
        self.d_bottleneck = 1
        self.d_model = d_model
        self.latent_dimension = (self.d_bottleneck, self.d_model)
        self.selfies_list = ["[C]", "[C][O]", "[O][C]"]

    def eval(self):
        pass

    def decode_inference(self, z: torch.Tensor):
        B = z.shape[0]
        selfies_decoded = [self.selfies_list[b % 3] for b in range(B)]
        batch_tokens = data_handler.collate_fn_tokens([data_handler.selfies_string_to_tokens(s, self.word_to_index) for s in selfies_decoded])
        logits = data_handler.tokens_to_logits(batch_tokens, len(self.vocab))
        return logits


class TestDecode(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.vocab = ['<START>', '<STOP>', '[C]', '[O]']
        self.device = 'cpu'
        self.d_model = 16
        self.vae = Dummy_VAE(vocab=self.vocab, device=self.device, d_model=self.d_model)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_zs(self, n):
        return torch.zeros((n, 1, self.d_model), dtype=torch.float32)

    def test_decode_zs_to_selfies_(self):
        N = 5
        batch_size = N
        zs = self._make_zs(N)

        selfies_list = dec.decode_zs_to_selfies(zs, self.vae, batch_size=batch_size)
        self.assertEqual(len(selfies_list), N)

        expected = ["[C]", "[C][O]", "[O][C]", "[C]", "[C][O]"]
        for i in range(N):
            self.assertEqual(selfies_list[i], expected[i])


    def test_decode_zs_to_selfies_with_lower_batch_size(self):
        N = 5
        batch_size = 3
        zs = self._make_zs(N)

        selfies_list = dec.decode_zs_to_selfies(zs, self.vae, batch_size=batch_size)
        self.assertEqual(len(selfies_list), N)

        expected = ["[C]", "[C][O]", "[O][C]", "[C]", "[C][O]"]
        for i in range(N):
            self.assertEqual(selfies_list[i], expected[i])


    def test_decode_zs_to_selfies_empty(self):
        zs = self._make_zs(0)
        out = dec.decode_zs_to_selfies(zs, self.vae, batch_size=4)
        self.assertEqual(out, [])


    def test_decode_zs_from_numpy(self):
        N = 5
        batch_size = 3
        zs = self._make_zs(N)
        zs = zs.numpy()

        selfies_list = dec.decode_zs_to_selfies(zs, self.vae, batch_size=batch_size)
        self.assertEqual(len(selfies_list), N)

        expected = ["[C]", "[C][O]", "[O][C]", "[C]", "[C][O]"]
        for i in range(N):
            self.assertEqual(selfies_list[i], expected[i])


    def test_decode_zs_wrong_shape_raises(self):
        zs_bad = torch.zeros((3, 10, 9, self.d_model))
        with self.assertRaises(ValueError):
            dec.decode_zs_to_selfies(zs_bad, self.vae)


    def test_decode_zs_to_selfies_squeezed(self):
        N = 5
        batch_size = N
        zs = self._make_zs(N).squeeze(1)

        selfies_list = dec.decode_zs_to_selfies(zs, self.vae, batch_size=batch_size)
        self.assertEqual(len(selfies_list), N)

        expected = ["[C]", "[C][O]", "[O][C]", "[C]", "[C][O]"]
        for i in range(N):
            self.assertEqual(selfies_list[i], expected[i])



    @patch("chembed.decode.load_tensor", return_value=None)
    @patch("chembed.decode.decode_zs_to_selfies", return_value=['AAAAAA'])
    def test_cli(self, mock_load_tensor: MagicMock, mock_decode_to_selfies: MagicMock):
        input_file = self.tmp/'test.pt'
        output_file = self.tmp/'test.selfies'
        dec.main([str(input_file), "--decode_to_selfies", '-o', str(output_file)])
        mock_load_tensor.assert_called_once()
        mock_decode_to_selfies.assert_called_once()
        self.assertTrue(output_file.is_file())



if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main()

