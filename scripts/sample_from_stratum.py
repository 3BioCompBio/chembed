from pathlib import Path
import polars as pl
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    input_file = args.input_file
    df = pl.read_parquet(input_file)

    nb_samples = 1_000_000
    total = df.height
    df_stratified = df.group_by("stratum").map_groups(lambda g: g.sample(n=round(g.shape[0] / total * nb_samples), seed=42))

    output_dir = input_file.parent
    output_file = output_dir/f'{input_file.stem}_sample_{nb_samples}.parquet'
    df_stratified.write_parquet(output_file)
