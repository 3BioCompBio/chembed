from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd
import multiprocessing as mp

from chembed.utils import get_tokens, read_df, write_json

def build_vocab(all_selfies: List[str], processes: int = mp.cpu_count()) -> List[str]:
    """Builds the vocabulary given the input SELFIES strings (i.e. the list of all SELFIES tokens that appear in it)."""
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=processes) as pool:
        res = pool.map(get_tokens, all_selfies)
    vocab_set = set().union(*res) if res else set()
    vocab_list = sorted(vocab_set)
    return vocab_list


def build_vocab_file_from_train_file(train_file: Path, output_vocab_file: Path, processes: int = mp.cpu_count()) -> None:
    df_train = read_df(train_file, required_columns=['selfies'])
    vocab = build_vocab(df_train['selfies'].tolist())
    write_json(vocab, output_vocab_file)


def clip_values(x: float, x_min: float, x_max: float) -> float:
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x

def return_df_with_clipped_properties(df: pd.DataFrame, properties_to_clip: List[str], epsilon_quantile: float) -> pd.DataFrame:
    for prop in properties_to_clip:
        prop_min = df[prop].quantile(epsilon_quantile)
        prop_max = df[prop].quantile(1-epsilon_quantile)
        df[f"{prop}_clipped"] = df[prop].apply(lambda x: clip_values(x, prop_min, prop_max))
    return df

def normalize_and_compute_statistics(df: pd.DataFrame, properties: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str,float]]]:
    """ for each property in @property, computes the mean and std in the corresponding column in @df and normalizes it.
        outputs the normalized dataframe and the dictionary {property: {mean: ..., std: ...}}"""
    stats = {}
    for prop in properties:
        stats[prop] = {'mean': float(df[prop].mean()), 'std': float(df[prop].std())}
        df['normalized_{}'.format(prop)] = (df[prop]-stats[prop]['mean'])/stats[prop]['std']
    return df, stats

