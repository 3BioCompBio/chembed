import os
from pathlib import Path
import argparse
from pandarallel import pandarallel

from chembed.utils import read_df, write_df
from chembed.mol_utils import get_fingerprint_from_smiles

MAIN_DIR = Path(os.path.realpath(__file__)).parent.parent

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        output_file = args.input_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = read_df(args.input_file)

    if 'fingerprint' in df.columns:
        raise ValueError('fingerprint column already in df')

    if 'smiles' not in df.columns:
        raise ValueError('smiles column not in df')

    pandarallel.initialize(nb_workers = args.num_workers)

    df['fingerprint'] = df['smiles'].parallel_apply(get_fingerprint_from_smiles) 

    write_df(df, output_file)


