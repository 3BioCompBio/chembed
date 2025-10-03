from pathlib import Path
import argparse
import selfies as sf
import numpy as np
import csv
import Levenshtein

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

FPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

def fingerprint_from_smiles(smiles: str):
	mol = Chem.MolFromSmiles(smiles)
	return FPGEN.GetFingerprint(mol)

def smiles_no_stereo_from_selfies(selfies: str) -> str:
	try:
		smiles = sf.decoder(selfies)
		mol = Chem.MolFromSmiles(smiles)
		return Chem.MolToSmiles(mol, isomericSmiles=False)
	except Exception:
		return None

def tanimoto_from_smiles(s1: str, s2: str) -> float:
	try:
		fp1 = fingerprint_from_smiles(s1)
		fp2 = fingerprint_from_smiles(s2)
		return DataStructs.TanimotoSimilarity(fp1, fp2)
	except Exception:
		return np.nan


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("reconstructed", type=Path)
	parser.add_argument("--output", type=Path, default=None)
	args = parser.parse_args()

	reconstructed_file = args.reconstructed
	assert reconstructed_file.is_file()
	assert reconstructed_file.suffix == '.csv'

	output = args.output
	if output is None:
		output = reconstructed_file.parent/f'{reconstructed_file.stem}_analyzed.csv'
	output.parent.mkdir(parents=True, exist_ok=True)

	out_header = ['selfies', 'reconstructed_selfies', 'smiles_no_stereo', 'reconstructed_smiles_no_stereo', 'L_target', 'L_reconstructed', 'levenshtein_distance', 'tanimoto']

	with open(reconstructed_file, 'r') as rf, open(output, 'w') as of:
		csv_reader = csv.DictReader(rf)
		csv_writer = csv.DictWriter(of, fieldnames=out_header)
		csv_writer.writeheader()
		for in_row in csv_reader:
			row = {}
			row['selfies'] = in_row['original']
			row['reconstructed_selfies'] = in_row['reconstructed']
			if row['selfies'] != row['reconstructed_selfies']:
				target_tokens = list(sf.split_selfies(row['selfies']))
				target_reconstructed = list(sf.split_selfies(row['reconstructed_selfies']))
				row['L_target'] = len(target_tokens)
				row['L_reconstructed'] = len(target_reconstructed)
				row['levenshtein_distance'] = Levenshtein.distance(target_tokens, target_reconstructed)
				row['smiles_no_stereo'] = smiles_no_stereo_from_selfies(row['selfies'])
				row['reconstructed_smiles_no_stereo'] = smiles_no_stereo_from_selfies(row['reconstructed_selfies'])
				row['tanimoto'] = tanimoto_from_smiles(row['smiles_no_stereo'], row['reconstructed_smiles_no_stereo'])
				csv_writer.writerow(row)
				of.flush()
