from typing import Union, List, Optional, Sequence
from numpy.typing import NDArray

import argparse
from pathlib import Path

import selfies as sf

import torch
from torch import Tensor

import math

from chembed import data_handler, checkpoint_utils
from chembed.utils import load_tensor, write_iterable_to_file
from chembed.errors import configure_warning_filters


@torch.inference_mode()
def decode_zs_to_selfies(zs: Union[Tensor, NDArray], vae, batch_size: int = 256) -> List[str]:
    vae.eval()

    if not isinstance(zs, Tensor):
        zs = torch.as_tensor(zs)

    if zs.dim() != 3 or zs.shape[1] != 1:
        if zs.dim() == 2:
            zs = zs.unsqueeze(1)
        else:
            raise ValueError(f"zs must be of shape (N, 1, d), got shape {zs.shape}")

    decoded = []
    
    nb_batches = int(math.ceil(len(zs) / batch_size))

    for batch_idx in range(nb_batches):
        batch = zs[batch_idx*batch_size:min(len(zs), (batch_idx+1)*batch_size)]
        batch = batch.to(vae.device)
        logits = vae.decode_inference(batch)
        tokens = data_handler.logits_to_tokens(logits) 
        batch_decoded = data_handler.tokens_to_selfies_strings(tokens, vae.vocab)
        decoded.extend(batch_decoded)
    
    return decoded


def decode_zs_to_smiles(zs: Tensor, vae, batch_size: int = 256) -> List[str]:
    selfies_decoded = decode_zs_to_selfies(zs, vae, batch_size=batch_size)
    return [sf.decoder(s) for s in selfies_decoded]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Path to file containing the embeddings")
    decode_to_group = parser.add_mutually_exclusive_group()
    decode_to_group.add_argument("--decode_to_selfies", action='store_true', dest='decode_to_selfies', help="Decode to SELFIES (default)")
    decode_to_group.add_argument("--decode_to_smiles", action='store_false', dest='decode_to_selfies', help="Decode to SMILES (default: decode to SELFIES)")
    parser.set_defaults(decode_to_selfies=True)
    vae_group = parser.add_mutually_exclusive_group()
    vae_group.add_argument("--model", type=str, help="HuggingFace model repo id", default="3BioCompBio/chembed-default")
    vae_group.add_argument("--checkpath", type=Path, help="Path to local checkpoint (if not using HuggingFace)", default=None)
    parser.add_argument("--output", '-o', type=Path, required=True, help="Path to where decoded molecules will be stored")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args(argv)

    configure_warning_filters()

    device = torch.device(args.device)

    if args.checkpath:
        vae = checkpoint_utils.load_vae_from_checkpoint(args.checkpath, device) 
    else:
        vae = checkpoint_utils.load_vae_from_hub(device = device, repo_id = args.model)

    z = load_tensor(args.input, map_location = device) 

    if args.decode_to_selfies:
        decoded = decode_zs_to_selfies(z, vae, args.batch_size)
    else:
        decoded = decode_zs_to_smiles(z, vae, args.batch_size)

    write_iterable_to_file(decoded, args.output)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
