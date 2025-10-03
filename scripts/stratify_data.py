from pathlib import Path
import argparse
import re
import polars as pl
from chembed.utils import read_json

RARE_TOKEN_THRESHOLD = 100_000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("token_counts", type=Path)
    args = parser.parse_args()

    input_file = args.input_file

    assert input_file.suffix.endswith('.parquet')

    output_dir = input_file.parent
    output_file = output_dir/f'{input_file.stem}_with_stratum.parquet'

    token_counts = read_json(args.token_counts)
    rare_tokens = [k for k, v in token_counts.items() if v <= RARE_TOKEN_THRESHOLD]

    stats = (
        pl.scan_parquet(input_file)
        .select(pl.col("selfies").str.count_matches("[", literal=True).alias("selfies_len"))
        .select(
            pl.col("selfies_len").quantile(1/3).alias("q1"),
            pl.col("selfies_len").quantile(2/3).alias("q2"),
            pl.col("selfies_len").max().alias("max_len"),
        )
        .collect()
    )

    q1 = int(stats["q1"][0])
    q2 = int(stats["q2"][0])
    max_len = int(stats["max_len"][0])


    edges = [-1, q1, q2, max_len]
    labels = ["short", "medium", "long"]

   
    ldf = pl.scan_parquet(input_file)

    selfies_len = pl.col("selfies").str.count_matches("[", literal=True).alias("selfies_len")
    pat = "|".join(re.escape(t) for t in rare_tokens)
    has_rare_token = pl.col("selfies").str.contains(pat).alias("has_rare_token")

    bin_expr = (
        pl.when(pl.col("selfies_len") <= q1).then(pl.lit("short"))
         .when(pl.col("selfies_len") <= q2).then(pl.lit("medium"))
         .otherwise(pl.lit("long"))
    )

    plan = (
        ldf
        .with_columns([selfies_len, has_rare_token])
        .with_columns([
            bin_expr.alias("selfies_len_bin"),
            pl.format("({},{})", bin_expr, pl.col("has_rare_token")).alias("stratum"),
        ])
    )

    plan.drop(["has_rare_token", "selfies_len", "selfies_len_bin"]).sink_parquet(output_file)
