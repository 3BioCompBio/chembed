from typing import List

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED, RDConfig
from rdkit import DataStructs
import sys
import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from chembl_structure_pipeline import standardizer

import selfies as sf

FPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
MAX_SA_SCORE = 10
MIN_QED_SCORE = 0.0


def remove_isotope_values(mol: Chem.Mol) -> Chem.Mol:
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol

def get_largest_fragment(smiles: str) -> str:
    smiless = smiles.split('.')
    smiless.sort(key=len, reverse=True)
    return smiless[0]

def standardize_smiles(smiles: str) -> str:
    smiles = get_largest_fragment(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = remove_isotope_values(mol)
    mol = standardizer.standardize_mol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, kekuleSmiles=True)

def smiles_to_selfies(smiles: str) -> str:
    return sf.encoder(standardize_smiles(smiles))

def get_fingerprint_from_mol(mol: Chem.Mol) -> DataStructs.cDataStructs.ExplicitBitVect:
	return FPGEN.GetFingerprint(mol)

def get_fingerprint_from_smiles(smiles: str) -> List:
	mol = Chem.MolFromSmiles(smiles)
	return list(get_fingerprint_from_mol(mol))

def rdkit_property_from_mol(mol: Chem.Mol, prop: str) -> float:
    return getattr(Descriptors, prop)(mol)

def rdkit_property_from_smiles(smiles: str, prop: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return rdkit_property_from_mol(mol, prop)

def mol_property_from_mol(mol: Chem.Mol, prop: str) -> float:
    if prop == 'QED':
        try:
            return QED.qed(mol)
        except OverflowError:
            return MIN_QED_SCORE
    elif prop == 'SA_Score':
        try:
            return sascorer.calculateScore(mol)
        except ZeroDivisionError:
            return MAX_SA_SCORE
    else:
        return rdkit_property_from_mol(mol, prop)

def mol_property_from_smiles(smiles: str, prop: str) -> float:
    return mol_property_from_mol(Chem.MolFromSmiles(smiles), prop)
