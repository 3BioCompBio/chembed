from pathlib import Path
import argparse
import pandas as pd
from pandarallel import pandarallel
import time

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from chembl_structure_pipeline import standardizer

from chembed.mol_utils import rdkit_property_from_smiles
from chembed.utils import write_df

SALT_REMOVER = SaltRemover()
ORGANIC_ATOMS = {"B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "H", "Sn"}

def remove_isotope_values(mol: Chem.Mol) -> Chem.Mol:
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol

def prefilter_and_standardize(smiles: str) -> str:
    if "C" not in smiles: # no carbon -> discard
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(e)
        return None
    if mol is None: # rdkit can't parse -> discard
        return None
    try:
        mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True) # remove salts
    except Exception as e:
        print(e)
        return None
    if not all(atom.GetSymbol() in ORGANIC_ATOMS for atom in mol.GetAtoms()): # not all atoms are in the organic subset -> discard
        return None
    try:
        mol = remove_isotope_values(mol) # remove isotope values
    except Exception as e:
        print(e)
        return None
    try:
        mol = standardizer.standardize_mol(mol)
    except Exception as e:
        print(e)
        return None
    try:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True, canonical=True)
        return smiles
    except Exception as e:
        print(e)
        return None
                


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    pandarallel.initialize(nb_workers=args.num_workers)

    df = pd.read_csv(args.input_file, sep='\t', header=None, names=['CID', 'smiles'])

    t_start = time.time()

    # split mixtures of compounds
    df['smiles'] = df['smiles'].str.split('.')
    df = df.explode('smiles')
    df = df.drop_duplicates(subset=['smiles'])

    # prefilter and standardize
    df['smiles'] = df['smiles'].parallel_apply(lambda smiles: prefilter_and_standardize(smiles))
    df = df.dropna()

    # remove duplicates
    df = df.drop_duplicates(subset=['smiles'])

    # compute molecular weight
    df['MolWt'] = df['smiles'].parallel_apply(lambda smiles: rdkit_property_from_smiles(smiles, 'MolWt'))

    # filter small compounds
    df = df[df['MolWt']<=600]

    t_end = time.time()
    print("Total processing time:", t_end-t_start)

    write_df(df, args.output_file)
