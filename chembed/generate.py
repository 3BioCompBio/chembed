from typing import List, Optional, Sequence

import argparse
from pathlib import Path
import selfies as sf
import torch
import pandas as pd

from chembed import checkpoint_utils
from chembed import data_handler
from chembed.transformer_vae import SELFIESTransformerVAE

@torch.inference_mode()
def generate_selfies(vae: SELFIESTransformerVAE, nb_samples: int, std: float, batch_size: int = 1024) -> List[str]:

    all_generated_selfies = []

    for i in range(0, nb_samples, batch_size):
        this_batch_size = min(batch_size, nb_samples - i)
        batch = torch.randn((this_batch_size, *vae.latent_dimension), device=vae.device) * std
        decoded_logits = vae.decode_inference(batch)
        tokens = data_handler.logits_to_tokens(decoded_logits)
        decoded_selfies = data_handler.tokens_to_selfies_strings(tokens, vae.vocab)
        all_generated_selfies.extend(decoded_selfies)

    return all_generated_selfies


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("nb_samples", type=int)
    parser.add_argument("output_file", type=Path)
    vae_group = parser.add_mutually_exclusive_group()
    vae_group.add_argument("--model", type=str, help="HuggingFace model repo id", default="3BioCompBio/chembed-default")
    vae_group.add_argument("--checkpath", type=Path, help="Path to local checkpoint (if not using HuggingFace)", default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--std", type=float, default=1.0)
    args = parser.parse_args(argv)

    device = torch.device(args.device)

    if args.checkpath:
        vae = checkpoint_utils.load_vae_from_checkpoint(args.checkpath, device) 
    else:
        vae = checkpoint_utils.load_vae_from_hub(device = device, repo_id = args.model)
    vae.eval()

    output_file = args.output_file
    output_file.parent.mkdir(exist_ok=True, parents=True)

    selfiess = []
    while len(selfiess) < args.nb_samples:
        generated_selfies = generate_selfies(vae, min(args.batch_size, args.nb_samples-len(selfiess)), args.std)
        selfiess.extend(generated_selfies)
    
    df = pd.DataFrame.from_dict({'selfies':selfiess})
    df['smiles'] = df['selfies'].apply(sf.decoder)

    df.to_csv(output_file, index=False)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
