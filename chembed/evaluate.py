from typing import Sequence, Dict, Optional, List

from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import csv
import selfies as sf
import argparse
from scipy import stats as st
import numpy as np

from rdkit import Chem

import torch
from torch.utils.data import DataLoader
from torch import Tensor

from chembed.mol_utils import get_fingerprint_from_mol, get_fingerprint_from_smiles, mol_property_from_mol, mol_property_from_smiles
from chembed import data_handler
from chembed.transformer_vae import SELFIESTransformerVAE
from chembed.metrics import string_reconstruction_accuracy_sum, uniqueness, trustworthiness_score, compute_euclidean_distance_matrix, compute_tanimoto_similarity_matrix, reconstruction_loss_unreduced, kl_divergence_loss_unreduced, compute_L1_distance_matrix
from chembed.utils import write_json, read_df, write_iterable_to_file
from chembed import checkpoint_utils
from chembed.errors import configure_warning_filters, warn
from chembed.generate import generate_selfies
from chembed import encode as enc
from chembed import moses_metrics


def get_novelty_scores_from_train_csv_file(generated_selfies: Sequence[str], valid_generated_mols: Sequence[Chem.Mol], train_file: Path, smiles_column_name: str = 'smiles') -> Dict[str, float]: 
    """ compares the generated molecules to the ones in @train_file to see how much the SELFIES, molecules, fingerprints are novel.
        it also inputs the list of mols because we already have them since the validity was computed before
        so valid_generated_mols ONLY contains valid mols
        @train_file must be a csv to be read one by line
    """

    novel_selfies = set(generated_selfies)
    novel_canonical_smiles = set(Chem.MolToSmiles(mol, canonical=True) for mol in valid_generated_mols)
    novel_fps = set(get_fingerprint_from_mol(mol).ToBitString() for mol in valid_generated_mols)

    if not (novel_selfies or novel_canonical_smiles or novel_fps):
        return {
            "selfies_novelty_score": 0.0,
            "mol_novelty_score": 0.0,
            "fingerprint_novelty_score": 0.0,
        }
        
    nb_selfies_init = len(novel_selfies)
    nb_canonical_smiles_init = len(novel_canonical_smiles)
    nb_fps_init = len(novel_fps)

    with open(train_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        if ('selfies' not in csv_reader.fieldnames) or (smiles_column_name not in csv_reader.fieldnames):
            raise ValueError(f"{train_file} header is {csv_reader.fieldnames}")

        for row in csv_reader:
            if not (novel_selfies or novel_canonical_smiles or novel_fps):
                break

            novel_selfies.discard(row['selfies'])

            smiles = row[smiles_column_name]
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True) 
                if novel_canonical_smiles:
                    novel_canonical_smiles.discard(canonical_smiles)
                
                if novel_fps:
                    fp = get_fingerprint_from_mol(mol).ToBitString()
                    novel_fps.discard(fp)
    return {
            'selfies_novelty_score': len(novel_selfies)/max(float(nb_selfies_init), 1.0),
            'mol_novelty_score': len(novel_canonical_smiles)/max(float(nb_canonical_smiles_init), 1.0),
            'fingerprint_novelty_score': len(novel_fps)/max(float(nb_fps_init), 1.0)
        }



def get_novelty_scores_from_train_df(generated_selfies: Sequence[str], valid_generated_mols: Sequence[Chem.Mol], train_df: pd.DataFrame, smiles_column_name: str = 'smiles') -> Dict[str, float]: 
    """ compares the generated molecules to the ones in @train_df to see how much the SELFIES, molecules, fingerprints are novel.
        it also inputs the list of mols because we already have them since the validity was computed before
        so valid_generated_mols ONLY contains valid mols
        @train_df must be a pandas DataFrame with columns 'selfies' and @smiles_column_name
    """

    novel_selfies = set(generated_selfies)
    novel_canonical_smiles = set(Chem.MolToSmiles(mol, canonical=True) for mol in valid_generated_mols)
    novel_fps = set(get_fingerprint_from_mol(mol).ToBitString() for mol in valid_generated_mols)

    if not (novel_selfies or novel_canonical_smiles or novel_fps):
        return {
            "selfies_novelty_score": 0.0,
            "mol_novelty_score": 0.0,
            "fingerprint_novelty_score": 0.0,
        }
    
    nb_selfies_init = len(novel_selfies)
    nb_canonical_smiles_init = len(novel_canonical_smiles)
    nb_fps_init = len(novel_fps)

    if ('selfies' not in train_df.columns) or (smiles_column_name not in train_df.columns):
        raise ValueError(f"DataFrame columns are {list(train_df.columns)}")

    if novel_selfies:
        for val in train_df['selfies'].dropna().unique().tolist():
            if not novel_selfies:
                break
            novel_selfies.discard(val)

    if novel_canonical_smiles or novel_fps:
        for smiles in train_df[smiles_column_name].dropna().unique().tolist():
            if not (novel_canonical_smiles or novel_fps):
                break
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if novel_canonical_smiles:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                novel_canonical_smiles.discard(canonical_smiles)
            if novel_fps:
                fp = get_fingerprint_from_mol(mol).ToBitString()
                novel_fps.discard(fp)

    return {
            'selfies_novelty_score': len(novel_selfies)/max(float(nb_selfies_init), 1.0),
            'mol_novelty_score': len(novel_canonical_smiles)/max(float(nb_canonical_smiles_init), 1.0),
            'fingerprint_novelty_score': len(novel_fps)/max(float(nb_fps_init), 1.0)
        }




def get_novelty_scores_from_train_file(generated_selfies: Sequence[str], valid_generated_mols: Sequence[Chem.Mol], train_file: Path, smiles_column_name: str = 'smiles') -> Dict[str, float]: 
    """ compares the generated molecules to the ones in @train_file to see how much the SELFIES, molecules, fingerprints are novel.
        it also inputs the list of mols because we already have them since the validity was computed before
        so valid_generated_mols ONLY contains valid mols
    """
    if isinstance(train_file, str):
        train_file = Path(train_file)
    if not train_file.is_file():
        raise FileNotFoundError(train_file)
    if train_file.suffix.endswith('csv'):
        return get_novelty_scores_from_train_csv_file(generated_selfies, valid_generated_mols, train_file, smiles_column_name=smiles_column_name)
    else:
        train_df = read_df(train_file)
        return get_novelty_scores_from_train_df(generated_selfies, valid_generated_mols, train_df, smiles_column_name=smiles_column_name)


def evaluate_property_distributions(generated_mols: List[Chem.Mol], ref_file: Path, properties: List[str]) -> Dict[str,float]:
    res = {}

    if ref_file is None:
        raise ValueError("To test property distributions, a reference file must be provided")
    df = read_df(ref_file, desired_columns=properties)

    for prop in properties:
        if prop in df.columns:
            input_property_values = df[prop]
        else:
            warn(UserWarning, f"{prop} not found in {ref_file}: will be computed (might be long)")
            input_property_values = df['smiles'].apply(lambda s: mol_property_from_smiles(s, prop))
        generated_property_values = [mol_property_from_mol(mol, prop) for mol in generated_mols] 
        res[f'{prop}_wasserstein'] = st.wasserstein_distance(input_property_values, generated_property_values).item()

    return res



def evaluate_generated(all_decoded_selfies: List[str], device: torch.device, test_novelty: bool = False, train_file: Optional[Path] = None, smiles_column_name: str = "smiles", write_invalid_to: Optional[Path] = None, test_property_distributions: bool = False, ref_file_for_prop_distrib_test: Optional[Path] = None, properties_for_prop_distrib_test: List[str] = ['MolWt', 'MolLogP', 'QED', 'SA_Score', 'TPSA', 'BertzCT', 'Kappa1', 'Kappa2', 'Kappa3']) -> Dict[str, float]:

    nb_samples = len(all_decoded_selfies)

    if write_invalid_to:
        invalid = []
    
    all_decoded_smiles = []
    for s in all_decoded_selfies:
        try:
            smiles = sf.decoder(s)
            all_decoded_smiles.append(smiles)
        except Exception:
            if write_invalid_to:
                invalid.append(s)

    valid_mols = []
    for s in all_decoded_smiles:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_mols.append(mol)
            else:
                if write_invalid_to:
                    invalid.append(s)
        except Exception:
            if write_invalid_to:
                invalid.append(s)

    if write_invalid_to and invalid:
        write_iterable_to_file(invalid, write_invalid_to)

    fingerprints = [get_fingerprint_from_mol(mol) for mol in valid_mols]

    res: Dict[str, float] = {
        f"uniqueness_{nb_samples}": uniqueness(all_decoded_smiles),
        f"validity_{nb_samples}": (len(valid_mols) / float(len(all_decoded_selfies))) if all_decoded_selfies else 0.0,
        f"IntDiv1_{nb_samples}": moses_metrics.IntDiv1(fingerprints, device),
        f"IntDiv2_{nb_samples}": moses_metrics.IntDiv2(fingerprints, device)
    }

    if test_novelty:
        if train_file is None:
            raise ValueError("To test novelty, path to train data must be provided.")
        novelty = get_novelty_scores_from_train_file(all_decoded_selfies, valid_mols, train_file, smiles_column_name)
        for k, v in novelty.items():
            res[f"{k}_{nb_samples}"] = v

    if test_property_distributions:
        res.update(evaluate_property_distributions(valid_mols, ref_file_for_prop_distrib_test, properties_for_prop_distrib_test))
    return res


@torch.no_grad()
def evaluate_generation(vae: SELFIESTransformerVAE, device: torch.device, nb_samples: int, batch_size: int = 1024, test_novelty: bool = False, train_file: Optional[Path] = None, smiles_column_name: str = "smiles", std: float = 1.0, write_invalid_to: Optional[Path] = None, test_property_distributions: bool = False, ref_file_for_prop_distrib_test: Optional[Path] = None, properties_for_prop_distrib_test: List[str] = ['MolWt', 'MolLogP', 'QED', 'SA_Score', 'TPSA', 'BertzCT', 'Kappa1', 'Kappa2', 'Kappa3']) -> Dict[str, float]:

    vae.to(device)
    vae.eval()

    all_decoded_selfies = generate_selfies(vae, nb_samples, std=std, batch_size=batch_size)

    return evaluate_generated(all_decoded_selfies, device, test_novelty=test_novelty, train_file=train_file, smiles_column_name=smiles_column_name, write_invalid_to=write_invalid_to, test_property_distributions=test_property_distributions, ref_file_for_prop_distrib_test=ref_file_for_prop_distrib_test, properties_for_prop_distrib_test=properties_for_prop_distrib_test) 



def get_substitution_counts_for_batch(padded_input_tokens: Tensor, padded_output_tokens: Tensor, vocab: List[str]) -> defaultdict[str, Counter[str]]:
    if padded_input_tokens.shape != padded_output_tokens.shape:
        raise ValueError(f"Input shapes don't match: {padded_input_tokens.shape}, {padded_output_tokens.shape}")

    substitutions = defaultdict(Counter)

    for it, ot in zip(padded_input_tokens.tolist(), padded_output_tokens.tolist()):
        L = len(it)
        for i in range(L):
            token_in = vocab[it[i]]
            token_out = vocab[ot[i]]
            substitutions[token_in][token_out] += 1

    return substitutions


@torch.no_grad()
def evaluate_loss_functions(vae: SELFIESTransformerVAE, test_df: pd.DataFrame, device: torch.device, batch_size: int = 1024, num_workers: int = 0) -> Dict[str,float]:
    vae.to(device)
    vae.eval()

    test_set = data_handler.SELFIESDataset(test_df, vae.vocab, properties=[], return_fingerprints=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=data_handler.collate_fn, num_workers=num_workers)

    res_dict = {
            'reconstruction_loss': 0,
            'kl_div_loss': 0
            }

    for batch_idx, batch in enumerate(test_dataloader):
        tokens, _, _ = batch
        tokens = tokens.to(device)

        vae_output = vae(tokens)

        padded_logits, padded_tokens = data_handler.pad_logits_or_tokens(vae_output['logits'], tokens)

        res_dict['reconstruction_loss'] += reconstruction_loss_unreduced(padded_logits, padded_tokens)
        res_dict['kl_div_loss'] += kl_divergence_loss_unreduced(vae_output['mu'], vae_output['sigma'])
    
    for k, v in res_dict.items():
        res_dict[k] = v/len(test_set)

    return res_dict


@torch.no_grad()
def evaluate_reconstruction(vae: SELFIESTransformerVAE, test_df: pd.DataFrame, device: torch.device, write_reconstructed_to: Optional[Path] = None, batch_size: int = 1024, num_workers: int = 0, output_substitutions: bool = True, output_accuracy_per_token: bool = True) -> Dict:
    """ SELFIES must be obtained with the standardization pipeline """
    vae.to(device)
    vae.eval()

    test_set = data_handler.SELFIESDataset(test_df, vae.vocab, properties=[]) 
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=data_handler.collate_fn, num_workers=num_workers) 

    compute_substitutions = output_substitutions or output_accuracy_per_token

    writer = None
    if write_reconstructed_to is not None:
        write_reconstructed_to = Path(write_reconstructed_to)
        write_reconstructed_to.parent.mkdir(parents=True, exist_ok=True)
        writer = open(write_reconstructed_to, 'w')
        writer.write('original,reconstructed\n')

    exact_match_count = 0
    token_match_count = 0
    total_nb_tokens = 0
    total_nb_strings = 0
    if compute_substitutions:
        substitutions = defaultdict(Counter)

    for batch_idx, batch in enumerate(test_dataloader):
        tokens, _, _ = batch
        tokens = tokens.to(device)
        vae_output = vae(tokens)
        padded_logits, padded_input_tokens = data_handler.pad_logits_or_tokens(vae_output['logits'], tokens)
        padded_output_tokens = data_handler.logits_to_tokens(padded_logits)

        exact_match_count += int(string_reconstruction_accuracy_sum(padded_output_tokens, padded_input_tokens).item()) 
        token_match_count += int((padded_input_tokens == padded_output_tokens).sum().item())
        total_nb_tokens += int(padded_input_tokens.numel())
        total_nb_strings += tokens.shape[0]

        if compute_substitutions:
            substitutions.update(get_substitution_counts_for_batch(padded_input_tokens, padded_output_tokens, vae.vocab))

        if writer is not None:
            input_selfies = data_handler.tokens_to_selfies_strings(tokens, vae.vocab)
            output_selfies = data_handler.tokens_to_selfies_strings(padded_output_tokens, vae.vocab)
            for in_s, out_s in zip(input_selfies, output_selfies):
                writer.write(f"{in_s},{out_s}\n")


    if writer is not None:
        writer.close()


    res = {
            'nb_strings_reconstructed': exact_match_count,
            'nb_tokens_reconstructed': token_match_count,
            'total_nb_strings': total_nb_strings,
            'total_nb_tokens': total_nb_tokens,
            'string_reconstruction_accuracy': exact_match_count / total_nb_strings,
            'token_reconstruction_accuracy': token_match_count / total_nb_tokens,
            }

    if output_substitutions:
        res['substitutions'] = {k: dict(v) for k, v in substitutions.items()}

    if output_accuracy_per_token:
        res['reconstruction_accuracy_per_token'] = {k: c[k] / sum(c.values()) if sum(c.values()) else 0.0 for k, c in substitutions.items()}
    

    return res




def evaluate_latent_space(vae: SELFIESTransformerVAE, test_df: pd.DataFrame, device: torch.device, batch_size: int = 1024, replace_if_not_in_vocab = True, properties_to_evaluate = []) -> Dict[str,float]:
    # I know it's not efficient and could be combined with evaluate_reconstruction but it's simpler like this, especially since this function will probably be called on a smaller subset anyway
    if 'fingerprint' not in test_df.columns:
        if 'smiles' not in test_df.columns:
            raise ValueError("Neither fingerprint nor smiles found in df.columns")
        test_df['fingerprint'] = test_df['smiles'].apply(get_fingerprint_from_smiles)

    fps = torch.stack([torch.tensor(fp) for fp in test_df['fingerprint']])
    M_tanimoto = compute_tanimoto_similarity_matrix(fps)

    zs = enc.encode_selfies_in_df(test_df, vae, batch_size = batch_size, replace_if_not_in_vocab = replace_if_not_in_vocab)
    D_euclidean = compute_euclidean_distance_matrix(zs)

    res = {}

    mask = torch.triu(torch.ones_like(D_euclidean, dtype=bool), diagonal=1)
    x = M_tanimoto[mask].flatten()
    y = D_euclidean[mask].flatten()

    res['pearson_correlation_euclidean_tanimoto'] = st.pearsonr(x, y).statistic.item() 
    res['spearman_correlation_euclidean_tanimoto'] = st.spearmanr(x, y).statistic.item()

    for n_neighbors in [5, 10, 20, 30]:
        if n_neighbors < len(test_df) // 2:
            res[f'trustworthiness_{n_neighbors}'] = trustworthiness_score(zs, fps, n_neighbors=n_neighbors).item() 


    idx = np.triu_indices_from(D_euclidean)
    for prop in properties_to_evaluate:
        if prop not in test_df.columns:
            raise ValueError(f"{prop} not found in df.columns")

        Dp = compute_L1_distance_matrix(test_df[prop].tolist())
        x = Dp[idx]
        y = D_euclidean[idx]
        res[f'pearson_correlation_euclidean_{prop}_distance'] = st.pearsonr(x,y).statistic.item()

    return res



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output_res_dict", '-o', type=Path, required=True)
    parser.add_argument("--evaluate_reconstruction", action='store_true', default=False)
    parser.add_argument("--test_data_path", type=Path, default=None)
    parser.add_argument("--write_reconstructed_to", type=Path, default=None)
    parser.add_argument("--evaluate_generation", action='store_true', default=False)
    parser.add_argument("--write_invalid_to", type=Path, default=None)
    parser.add_argument("--nb_samples_for_generation", type=int, default=10_000)
    parser.add_argument("--evaluate_novelty", action='store_true', default=False)
    parser.add_argument("--train_data_csv", type=Path, default=None)
    parser.add_argument("--evaluate_latent_space", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args(argv)

    configure_warning_filters()

    if not (args.evaluate_reconstruction or args.evaluate_generation or args.evaluate_latent_space):
        raise ValueError("You must specify at least one of --evaluate_reconstruction or --evaluate_generation or --evaluate_latent_space")

    if (args.evaluate_novelty) and not (args.evaluate_generation):
        raise ValueError("If you want to evaluate novelty, you must also --evaluate_generation")

    if (args.evaluate_novelty) and not (args.train_data_csv):
        raise ValueError("If you want to evaluate novelty, you must provide the train dataset in csv format --train_data_csv")

    if (args.write_reconstructed_to) and not (args.evaluate_reconstruction):
        raise ValueError("You provided a path to write reconstructed outputs but did not ask to --evaluate_reconstruction")

    if (args.evaluate_reconstruction or args.evaluate_latent_space) and not (args.test_data_path):
        raise ValueError("If you want to evaluate reconstruction, you must provide a test dataset --test_data_path")

    device = torch.device(args.device)
    vae = checkpoint_utils.load_vae_from_checkpoint(args.vae, device)

    res = {}

    if args.evaluate_reconstruction:
       test_df = read_df(args.test_data_path, required_columns=['selfies'])
       res.update(evaluate_reconstruction(vae, test_df, device=device, write_reconstructed_to=args.write_reconstructed_to, write_invalid_to=args.write_invalid_to, batch_size=args.batch_size, num_workers=args.num_workers))

    if args.evaluate_latent_space:
        if not args.test_data_path.is_file():
            raise FileNotFoundError(args.test_data_path)
        try:
            test_df = read_df(args.test_data_path, required_columns=['selfies', 'fingerprint'])
        except KeyError:
            test_df = read_df(args.test_data_path, required_columns=['selfies', 'smiles'])
        res.update(evaluate_latent_space(vae, test_df, device=device, batch_size=args.batch_size))
            

    if args.evaluate_generation:
       res.update(evaluate_generation(vae, device, nb_samples=args.nb_samples_for_generation, batch_size=args.batch_size, test_novelty=args.evaluate_novelty, train_file=args.train_data_csv))

    write_json(res, args.output_res_dict)

    return res

if __name__=="__main__":
    res = main()
    print(res)
