from pathlib import Path

import torch

from chembed.transformer_vae import SELFIESTransformerVAE
from chembed import data_handler
from chembed.metrics import reconstruction_loss

def build_tiny_model(output_path: Path, device='cpu'):
    vocab = ["<START>", "<STOP>", "[C]", "[O]"]
    word_to_index = {t: i for i, t in enumerate(vocab)}
    model_config = {
        "d_model": 16, "d_bottleneck": 1, "max_len": 8,
        "dropout": 0.0, "nhead": 2,
        "dim_feedforward_encoder": 32, "dim_feedforward_decoder": 32,
        "nb_layers_encoder": 1, "nb_layers_decoder": 1,
        "layer_norm_eps": 1e-5, "use_log_sigma": True,
        "min_vae_sigma": 1e-4, "initialize_embedding_weights": False,
    }
    vae = SELFIESTransformerVAE(vocab, model_config).to(device).train()
    selfies_list = ["[C]", "[C][O]", "[O][C]", "[C][C]"]
    tokens = [data_handler.selfies_string_to_tokens(s, word_to_index, replace_if_not_in_vocab=False) for s in selfies_list]
    tokens = data_handler.collate_fn_tokens(tokens)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for step in range(1000):
        opt.zero_grad()
        mu, _ = vae.encode(tokens)
        logits = vae.decode_training(mu, tokens)
        loss = reconstruction_loss(logits, tokens) 
        loss.backward()
        opt.step()

    vae.eval()
    state = {
        "hyper_parameters": {"vocab": vocab, "model_config": model_config},
        "state_dict": {f"vae.{k}": v for k, v in vae.state_dict().items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)
