from typing import Optional, Sequence

from pathlib import Path
import argparse

from chembed.utils import read_df, write_df, write_json
from chembed import dataprocessing_utils as dpu

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--properties_to_clip", nargs='+', default=[])
    parser.add_argument("--epsilon_quantile", type=float, help="Values of properties to clip will be clipped at the epsilon_quantile^th value as a minimum and the (1.0-epsilon_quantile)^th value as a maximum")
    parser.add_argument("--properties_to_normalize", nargs='+', default=[])
    parser.add_argument("--output_stats", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.properties_to_normalize and not args.output_stats:
        raise ValueError("You asked for properties to be normalized but did not provide an output path for the properties statistics (--output_stats)")

    if not args.input_file.is_file():
        raise FileNotFoundError(args.input_file)

    df = read_df(args.input_file)
    df = dpu.return_df_with_clipped_properties(df, args.properties_to_clip, args.epsilon_quantile)

    properties_to_normalize = [f"{p}_clipped" if p in args.properties_to_clip else p for p in args.properties_to_normalize]
    df, stats = dpu.normalize_and_compute_statistics(df, properties_to_normalize)

    write_df(df, args.output_file)
    write_json(stats, args.output_stats)

    return 0

if __name__=="__main__":
    raise SystemExit(main())
