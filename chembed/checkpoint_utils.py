from typing import Union

from pathlib import Path
from huggingface_hub import hf_hub_download

import torch

from chembed.regressor import PropertiesRegressor
from chembed.transformer_vae import SELFIESTransformerVAE
from chembed.errors import configure_warning_filters

def load_vae_from_checkpoint(checkpath: Path, device: Union[torch.device, str], strict: bool = True) -> SELFIESTransformerVAE:
    configure_warning_filters()
    if not Path(checkpath).is_file():
        raise FileNotFoundError(checkpath)
    return SELFIESTransformerVAE.from_checkpoint(checkpath, device, strict=strict)
    

def load_regressor_from_checkpoint(checkpath: Path, device: Union[torch.device, str], strict: bool = True) -> PropertiesRegressor:
    if not Path(checkpath).is_file():
        raise FileNotFoundError(checkpath)
    return PropertiesRegressor.from_checkpoint(checkpath, device, weights_only=False) 


def load_vae_from_hub(device: Union[torch.device, str], repo_id: str = "3BioCompBio/chembed-default", filename: str = "model.ckpt" , strict: bool = True) -> SELFIESTransformerVAE:
    checkpath = hf_hub_download(repo_id = repo_id, filename = filename)
    return load_vae_from_checkpoint(Path(checkpath), device = device, strict = strict)


def get_last_checkpath_in_dir(logdir: Path) -> Path:
    if isinstance(logdir, str):
        logdir = Path(logdir)
    if not logdir.is_dir():
        raise FileNotFoundError(logdir)
    versions = sorted((p for p in logdir.glob("version_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not versions:
        raise FileNotFoundError(f"No version_* directories found in {logdir}")
    checkpath = versions[0]/'checkpoints'/'last.ckpt'
    if not checkpath.is_file():
        raise FileNotFoundError(checkpath)
    return checkpath

def load_last_checkpoint_in_dir(logdir: Path, device: Union[torch.device, str], strict: bool = True) -> SELFIESTransformerVAE:
    checkpath = get_last_checkpath_in_dir(logdir)
    vae = load_vae_from_checkpoint(checkpath, device, strict=strict)
    return vae
