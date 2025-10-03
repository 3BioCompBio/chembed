import argparse
from pathlib import Path
import numpy as np
import selfies as sf

from chembed.utils import read_df, read_json

def score(selfies, token_frequencies):
    tokens = sf.split_selfies(selfies)
    ps = np.array([token_frequencies[token] for token in tokens])
    L = len(ps)
    return sum([1/p for p in ps])/L

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--a", type=float, default=0.005)
    parser.add_argument("--token_counts", type=Path, required=True)
    args = parser.parse_args()

    input_file = args.input_file
    output_dir = input_file.parent
    output_file = output_dir/"{}_a_{}.parquet".format(input_file.stem, args.a) 

    token_counts = read_json(args.token_counts)
    total_counts = sum(token_counts.values())
    token_frequencies = {t: c/total_counts for t, c in token_counts.items()}

    df = read_df(input_file)

    df['score'] = df['selfies'].apply(lambda x: score(x, token_frequencies))

    df['nb_samples'] = df['score'].apply(lambda s: int(np.ceil(args.a*s)))


    print("a=", args.a)
    print("nb_samples before:", len(df))
    print("nb_samples after:", sum(df['nb_samples']))

    df = df.loc[df.index.repeat(df['nb_samples'])]
    print("len(df)=", len(df))

    df = df.drop(columns=['score', 'nb_samples'])

    df.to_parquet(output_file, index=False)
