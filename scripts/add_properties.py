from typing import Dict, List

from pathlib import Path
import argparse
import pandas as pd
from pandarallel import pandarallel
import time

from rdkit import Chem

from chembed.mol_utils import mol_property_from_mol
from chembed.utils import write_df, read_df


def compute_descriptors(smiles: str, descriptors: List[str]) -> Dict[str,float]:
    mol = Chem.MolFromSmiles(smiles)
    out = {}
    for descriptor in descriptors:
        out[descriptor] = mol_property_from_mol(mol, descriptor)
    return out


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--properties", nargs='+', required=True)
    args = parser.parse_args()

    pandarallel.initialize(nb_workers=args.num_workers)
    df = read_df(args.input_file)

    t_start = time.time()

    desc_df = df['smiles'].parallel_apply(lambda x: compute_descriptors(x, args.properties)).apply(pd.Series)
    df = pd.concat([df, desc_df], axis=1)

    t_end = time.time()
    print("Total processing time:", t_end-t_start)

    write_df(df, args.output_file)
