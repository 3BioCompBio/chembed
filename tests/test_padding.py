import unittest
import torch
import torch.testing as tt

from chembed import data_handler


class TestPadding(unittest.TestCase):
    def test_pad_tokens_to_match_length(self):
        t0 = torch.tensor([5, 6, 7], dtype=torch.long)
        t1 = torch.tensor([8, 9], dtype=torch.long)
        out = data_handler.pad_tokens_to_match_length([t0, t1], max_length=5)

        self.assertEqual(out.shape, (2, 5))
        self.assertEqual(out.dtype, torch.long)

        tt.assert_close(out[0, :3], t0)
        tt.assert_close(out[1, :2], t1)

        tt.assert_close(out[0, 3:], torch.full((2,), data_handler.PAD_TOKEN, dtype=torch.long))
        tt.assert_close(out[1, 2:], torch.full((3,), data_handler.PAD_TOKEN, dtype=torch.long))


    def test_pad_logits_to_match_length_preserves_and_pads(self):
        B, L, V = 2, 3, 7
        max_length = 6

        logits = torch.randn(B, L, V, dtype=torch.float32)
        out = data_handler.pad_logits_to_match_length(logits, max_length=max_length)

        self.assertEqual(out.shape, (B, max_length, V))
        self.assertEqual(out.dtype, logits.dtype)
        self.assertEqual(out.device, logits.device)

        tt.assert_close(out[:, :L, :], logits)
        self.assertTrue(torch.all(out[:, L:, :].argmax(dim=-1) == data_handler.PAD_TOKEN).item())

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU available")
    def test_pad_logits_to_match_length_preserves_and_pads_gpu(self):
        B, L, V = 2, 3, 7
        max_length = 6

        device = torch.device('cuda')

        logits = torch.randn(B, L, V, dtype=torch.float32).to(device)
        out = data_handler.pad_logits_to_match_length(logits, max_length=max_length)

        self.assertEqual(out.shape, (B, max_length, V))
        self.assertEqual(out.dtype, logits.dtype)
        self.assertEqual(out.device, logits.device)

        tt.assert_close(out[:, :L, :], logits)
        self.assertTrue(torch.all(out[:, L:, :].argmax(dim=-1) == data_handler.PAD_TOKEN).item())


    def test_pad_logits_when_not_needed(self):
        B, L, V = 1, 4, 5
        logits = torch.randn(B, L, V)
        out = data_handler.pad_logits_to_match_length(logits, max_length=L)
        tt.assert_close(out, logits)


if __name__ == "__main__":
    unittest.main(verbosity=2)

