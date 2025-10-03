from typing import Optional, Sequence

from pathlib import Path
import argparse

from chembed import dataprocessing_utils as dpu


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args(argv)

    dpu.build_vocab_file_from_train_file(args.input_file, args.output_file, processes=args.num_workers)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
