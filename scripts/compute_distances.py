import os
from pathlib import Path
import pandas as pd
import argparse

import torch

from chembed.utils import read_df
from chembed import checkpoint_utils
from chembed.encode import encode_selfies_in_df

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
FPGEN = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

SCRIPT_DIR = Path(os.path.realpath(__file__)).parent
MAIN_DIR = SCRIPT_DIR.parent

def get_fingerprint_from_smiles(smi):
	mol = Chem.MolFromSmiles(smi)
	fp = FPGEN.GetFingerprint(mol)
	return fp


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_file", type=Path)
	parser.add_argument("--device", type=str, default='cuda')
	parser.add_argument("--output_file", type=Path)
	parser.add_argument("--nb_pairs", type=int, default=1_000_000)
	parser.add_argument("--random_seed", type=int, default=42)
	parser.add_argument("--batch_size", type=int, default=1024)
	args = parser.parse_args()
	
	input_file = args.input_file

	device = torch.device(args.device)

	vae = checkpoint_utils.load_vae_from_hub(device)
	vae.eval()

	output_file = args.output_file
	if not output_file:
		output_file = Path('results')/'pubchem_small_organic'/f'test_{args.nb_pairs}_pairs_seed_{args.random_seed}.csv'
	output_file.parent.mkdir(parents=True, exist_ok=True)

	df = read_df(input_file, required_columns=['smiles', 'selfies'])
	df_1 = df.sample(n=args.nb_pairs, random_state=args.random_seed)
	df_2 = df.sample(n=args.nb_pairs, random_state=args.random_seed+1)
	del df

	df_1['fingerprint'] = df_1['smiles'].apply(get_fingerprint_from_smiles)
	df_2['fingerprint'] = df_2['smiles'].apply(get_fingerprint_from_smiles)
	tanimotos = [DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(df_1['fingerprint'], df_2['fingerprint'])]

	z_1 = encode_selfies_in_df(df_1, vae, args.batch_size) 
	z_2 = encode_selfies_in_df(df_2, vae, args.batch_size)
	distances = torch.norm(z_1 - z_2, dim=2, keepdim=True).flatten().cpu().tolist()

	res_df = pd.DataFrame({
		'smiles1': df_1['smiles'].tolist(),
		'smiles2': df_2['smiles'].tolist(),
		'tanimoto_similarity': tanimotos,
		'euclidean_distance': distances
		})

	print(res_df)

	res_df.to_csv(output_file, index=False)
