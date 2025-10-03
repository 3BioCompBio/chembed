from pathlib import Path
import argparse
from collections import Counter
from itertools import chain
import selfies as sf

from chembed.utils import write_json, read_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()

    input_file = args.input_file

    df = read_df(input_file, required_columns=['selfies'])

    selfies_tokens = df['selfies'].dropna().map(sf.split_selfies)
    token_counts = Counter(chain.from_iterable(selfies_tokens))

    write_json(token_counts, args.output_file)
