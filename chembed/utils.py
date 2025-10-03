from typing import List, Dict, Set, Optional, Iterable, Union

from pathlib import Path
import pkg_resources
import pandas as pd
from fastparquet import ParquetFile
import numpy as np
import selfies as sf
import json

import torch
from torch import Tensor
import pytorch_lightning as pl

from chembed.mol_utils import * # noqa: F403

def _select_columns_from_available(available_columns: List[str], required_columns: Optional[List[str]] = None, desired_columns: Optional[List[str]] = None) -> Optional[List[str]]:
    if required_columns is not None:
        missing_columns = [c for c in required_columns if c not in available_columns]
        if missing_columns:
            raise KeyError(f'Missing required columns: {missing_columns}')
    wanted_columns = []
    if required_columns:
        wanted_columns.extend(required_columns)
    if desired_columns:
        wanted_columns.extend([c for c in desired_columns if c in available_columns and c not in wanted_columns])
    if not wanted_columns:
        return None
    return list(wanted_columns)

def _get_available_columns_csv(df_path: Path, sep: str) -> List[str]:
    return list(pd.read_csv(df_path, sep=sep, nrows=0).columns)

def _get_available_columns_parquet(df_path: Path) -> List[str]:
    return ParquetFile(str(df_path)).columns


def read_df(df_path: Path, required_columns: List[str]=None, desired_columns: List[str]=None, sep: str=',') -> pd.DataFrame:
    """ reads pandas dataframe from any format.
    - required_columns: load exactly these columns (error if missing)
    - desired_columns: load only the subset that exists """
    if isinstance(df_path, str):
        df_path = Path(df_path)
    file_extension = df_path.suffix
    if file_extension == '.csv':
        available_columns = _get_available_columns_csv(df_path, sep)
        columns = _select_columns_from_available(available_columns, required_columns, desired_columns)
        if not columns:
            return pd.read_csv(df_path, sep=sep)
        return pd.read_csv(df_path, usecols=columns, sep=sep)
    elif file_extension == '.parquet':
        available_columns = _get_available_columns_parquet(df_path)
        columns = _select_columns_from_available(available_columns, required_columns, desired_columns)
        if not columns:
            return pd.read_parquet(df_path)
        return pd.read_parquet(df_path, columns=columns)
    elif file_extension == '.pkl':
        df = pd.read_pickle(df_path)
        available_columns = list(df.columns)
        columns = _select_columns_from_available(available_columns, required_columns, desired_columns)
        if not columns:
            return df
        else:
            return df[columns]
    else:
        raise NotImplementedError(file_extension)

def write_df(df: pd.DataFrame, output_path: Path, sep: str=',') -> None:
    """ writes pandas dataframe to any format depending on file extension """
    if isinstance(output_path, str):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_extension = output_path.suffix
    if file_extension == '.csv':
        df.to_csv(output_path, index=False, sep=sep)
    elif file_extension == '.parquet':
        df.to_parquet(output_path, index=False)
    elif file_extension == '.pkl':
        df.to_pickle(output_path)
    else:
        raise NotImplementedError(file_extension)

def get_resource_file(filename: str) -> Path:
    """ returns path to package resources file """
    resource_path_str = pkg_resources.resource_filename('chembed', f'resources/{filename}')
    resource_path = Path(resource_path_str)
    if not resource_path.is_file():
        raise FileNotFoundError(resource_path_str)
    return resource_path

def read_json(filepath: Path) -> Dict:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(filepath)
    with open(filepath, 'r') as jf:
        data = json.load(jf)
    return data

def write_json(data: Union[Dict, List], filepath: Path) -> None:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(filepath, 'w') as jf:
        json.dump(data, jf)

def write_iterable_to_file(to_write: Iterable, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as of:
        for elt in to_write:
            of.write(f'{elt}\n')

def read_smiles_or_selfies(input_file: Path) -> List[str]:
    """ assumes that SMILES or SELFIES are in the first column """
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    s_list = []
    with open(input_file, 'r') as f:
        first_line = next(f)
        if ('smi' not in first_line.lower()) and ('selfies' not in first_line.lower()): # skip if there's a header
            s = first_line.strip().split()[0]
            s_list.append(s)
        for line in f:
            line = line.strip()
            if not line:
                continue
            s_list.append(line.split()[0])
    return s_list

def save_tensor(z: Tensor, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix=='.npy':
        np.save(output_file, z.cpu().numpy())
    else:
        torch.save(z, output_file)

def load_tensor(input_file: Path, map_location : Optional[torch.device] = None) -> Tensor:
    if input_file.suffix=='.npy':
        z = np.load(input_file)
        return torch.from_numpy(z)
    else:
        return torch.load(input_file, map_location = map_location)

def get_tokens(selfies: str) -> Set:
    """ returns SELFIES tokens as a set """
    return set(sf.split_selfies(selfies))


def compute_token_overall_counts(selfies_list: List[str]) -> Dict[str, int]:
    """ counts the number of times each token appears in a list of SELFIES """
    token_dict = {}
    for selfies in selfies_list:
        for token in get_tokens(selfies):
            if token not in token_dict:
                token_dict[token] = 0
            token_dict[token] += 1
    return token_dict


def compute_token_overall_frequencies(selfies_list: List[str]) -> Dict[str, float]:
    """ get the frequency at which each token appears in a list of SELFIES, computed over the total number of tokens """
    token_counts = compute_token_overall_counts(selfies_list)
    total_counts = sum(token_counts.values())   
    token_frequencies = {t: c/total_counts for t, c in token_counts.items()}
    return token_frequencies


def selfies_rarity_score(selfies: str, token_frequencies: Dict[str,float]) -> float:
    """ computes a 'rarity score' for a SELFIES string, taking into account the rarity of the tokens it contains """
    tokens = sf.split_selfies(selfies)
    ps = np.array([token_frequencies[token] for token in tokens])
    L = len(ps)
    return sum([1/p for p in ps])/L

def set_random_seed_everywhere(seed: Optional[int] = None) -> None:
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
