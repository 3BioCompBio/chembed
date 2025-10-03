from pathlib import Path
import argparse
from pandarallel import pandarallel
import selfies as sf
import time

from chembed.utils import write_df, read_df
from chembed.mol_utils import smiles_to_selfies
from chembed.errors import configure_warning_filters

def try_encode_to_selfies(smiles: str, already_standardized: bool) -> str:
    try:
        if already_standardized:
            return sf.encoder(smiles)
        else:
            return smiles_to_selfies(smiles) 
    except Exception as e:
        print(e)
        return None


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--already_standardized", action='store_true', default=False)
    parser.add_argument("--drop_fail_encode", action='store_true', default=False)
    args = parser.parse_args()

    configure_warning_filters()

    pandarallel.initialize(nb_workers=args.num_workers)
    df = read_df(args.input_file)

    t_start = time.time()

    # try encoding to selfies
    df['selfies'] = df['smiles'].parallel_apply(lambda smiles: try_encode_to_selfies(smiles, already_standardized=args.already_standardized))
    
    if args.drop_fail_encode:
        df = df.dropna(subset=['selfies'])

    t_end = time.time()
    print("Total processing time:", t_end-t_start)

    write_df(df, args.output_file)
