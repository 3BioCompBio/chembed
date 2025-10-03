from typing import List, Sequence, Optional

import argparse
from pathlib import Path

import selfies as sf
import torch
from torch import Tensor
import pandas as pd

from chembed import data_handler, mol_utils, checkpoint_utils
from chembed.utils import read_smiles_or_selfies, save_tensor
from chembed.errors import configure_warning_filters

@torch.inference_mode()
def encode_selfies_in_df(df: pd.DataFrame, vae, batch_size: int = 256, replace_if_not_in_vocab: bool = False, use_sigma: bool = False) -> Tensor:

    vae.eval()
    
    selfies_dataset = data_handler.SELFIESDataset(df, vae.vocab, properties=[], return_fingerprints=False, return_properties=False, replace_if_not_in_vocab=replace_if_not_in_vocab)
    dataloader = torch.utils.data.DataLoader(selfies_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_handler.collate_fn, num_workers=1, drop_last=False)

    N = len(selfies_dataset)
    zs = torch.empty((N, vae.d_bottleneck, vae.d_model), dtype=torch.float32)

    offset = 0
    for batch_idx, batch in enumerate(dataloader):
        tokens, _, _ = batch
        tokens = tokens.to(vae.device)

        mu, sigma = vae.encode(tokens)

        if not use_sigma:
            z = mu
        else:
            z = mu + sigma * torch.randn_like(mu) 

        this_batch_size = z.shape[0]
        zs[offset: offset+this_batch_size] = z.detach().cpu()
        offset += this_batch_size
    
    return zs


def encode_multiple_selfies(selfies_list: List[str], vae, batch_size: int = 256, replace_if_not_in_vocab: bool = False, use_sigma: bool = False) -> Tensor:
    if not selfies_list:
        raise ValueError("SELFIES list is empty")
    df = pd.DataFrame({'selfies': selfies_list})
    return encode_selfies_in_df(df, vae, batch_size=batch_size, replace_if_not_in_vocab=replace_if_not_in_vocab, use_sigma=use_sigma)


def encode_multiple_smiles(smiles_list: List[str], vae, batch_size: int = 256, replace_if_not_in_vocab: bool = False, use_sigma: bool = False) -> Tensor:
    smiles_list = [mol_utils.standardize_smiles(s) for s in smiles_list]
    selfies_list = [sf.encoder(s) for s in smiles_list]
    return encode_multiple_selfies(selfies_list, vae, batch_size=batch_size, replace_if_not_in_vocab=replace_if_not_in_vocab, use_sigma=use_sigma)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_selfies", type=Path, help="Path to input SELFIES file")
    input_group.add_argument("--input_smiles", type=Path, help="Path to input SMILES file")
    vae_group = parser.add_mutually_exclusive_group()
    vae_group.add_argument("--model", type=str, help="HuggingFace model repo id", default="3BioCompBio/chembed-default")
    vae_group.add_argument("--checkpath", type=Path, help="Path to local checkpoint (if not using HuggingFace)", default=None)
    parser.add_argument("--output", '-o', type=Path, required=True, help="Path to where encoded molecules will be stored")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replace_if_not_in_vocab", action='store_true', default=False)
    args = parser.parse_args(argv)

    configure_warning_filters()

    device = torch.device(args.device)

    if args.checkpath:
        vae = checkpoint_utils.load_vae_from_checkpoint(args.checkpath, device) 
    else:
        vae = checkpoint_utils.load_vae_from_hub(device = device, repo_id = args.model)

    if args.input_smiles:
        smiles = read_smiles_or_selfies(args.input_smiles)
        z = encode_multiple_smiles(smiles, vae, args.batch_size, replace_if_not_in_vocab = args.replace_if_not_in_vocab)
    else:
        selfies = read_smiles_or_selfies(args.input_smiles)
        z = encode_multiple_selfies(selfies, vae, args.batch_size, replace_if_not_in_vocab = args.replace_if_not_in_vocab)

    save_tensor(z, args.output)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
