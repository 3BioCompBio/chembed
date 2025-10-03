import unittest
import torch
import torch.testing as tt

from chembed.transformer_vae import SELFIESTransformerVAE
from chembed import data_handler
from chembed.utils import set_random_seed_everywhere


class TestExpandVocab(unittest.TestCase):
    def setUp(self):
        self.vocab = ["<START>", "<STOP>", "[C]", "[O]"]
        self.word_to_index = {t: i for i, t in enumerate(self.vocab)}

        self.model_config = {
            "d_model": 16,
            "d_bottleneck": 1,
            "max_len": 16,
            "dropout": 0.0,
            "nhead": 2,
            "dim_feedforward_encoder": 32,
            "dim_feedforward_decoder": 32,
            "nb_layers_encoder": 1,
            "nb_layers_decoder": 1,
            "layer_norm_eps": 1e-5,
            "use_log_sigma": True,
            "min_vae_sigma": 1e-4,
            "initialize_embedding_weights": False,
        }

        self.vae = SELFIESTransformerVAE(self.vocab, self.model_config).to("cpu").eval()
        self.stop = self.vae.stop_token_index
        self.start = self.vae.start_token_index

        self.selfies_strings = ["[C]", "[C][O]", "[O][C]", "[C][C]"]
        tokens = [data_handler.selfies_string_to_tokens(s, self.word_to_index, replace_if_not_in_vocab=False) for s in self.selfies_strings]
        self.tokens = data_handler.collate_fn_tokens(tokens)

    def _encode_decode_training(self, tokens):
        """Helper: deterministic teacher-forcing path (no sampling)."""
        with torch.no_grad():
            mu, _ = self.vae.encode(tokens)
            logits = self.vae.decode_training(mu, tokens)
        return mu, logits


    def test_expand_vocab_shapes_and_weights_preserved(self):
        enc_emb_before = self.vae.encoder_embedding[0].weight.detach().clone()
        dec_emb_before = self.vae.decoder_embedding[0].weight.detach().clone()
        old_V = enc_emb_before.shape[0]

        # expand with new token
        new_token = "[F]"
        self.vae.expand_vocab([new_token])

        self.assertEqual(self.vae.nb_tokens, old_V + 1)
        self.assertIn(new_token, self.vae.vocab_to_index)

        tt.assert_close(self.vae.encoder_embedding[0].weight[:old_V], enc_emb_before)
        tt.assert_close(self.vae.decoder_embedding[0].weight[:old_V], dec_emb_before)


    def test_decode_training_preserved_on_old_vocab_sequences(self):
        mu_before, logits_before = self._encode_decode_training(self.tokens)
        old_V = logits_before.size(-1)

        # expand vocab 
        new_tokens = ["[F]", "[Cl]"]
        self.vae.expand_vocab(new_tokens)

        mu_after, logits_after = self._encode_decode_training(self.tokens)

        tt.assert_close(mu_after, mu_before)

        self.assertEqual(logits_after.shape[:2], logits_before.shape[:2])
        tt.assert_close(logits_after[..., :old_V], logits_before)

        out_before = data_handler.logits_to_tokens(logits_before)
        out_after = data_handler.logits_to_tokens(logits_after[..., :old_V])
        tt.assert_close(out_after, out_before)


    def test_tokenization_fails_before_and_succeeds_after_expand(self):
        seq_with_new = "[C][F][O]" # not in vocab

        # before expansion: should raise UnknownTokenError 
        with self.assertRaises(Exception):
            data_handler.selfies_string_to_tokens(seq_with_new, self.word_to_index, replace_if_not_in_vocab=False)

        # expand vocab
        self.vae.expand_vocab(["[F]"])
        new_word_to_index = self.vae.vocab_to_index
        tokens = data_handler.selfies_string_to_tokens(seq_with_new, new_word_to_index, replace_if_not_in_vocab=False)
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertGreaterEqual(tokens.max().item(), new_word_to_index["[F]"]) 


if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main()
