from typing import Dict, List, Set
from pathlib import Path
import argparse
import csv
import pandas as pd

from chembed.train import main as train_main
from chembed import evaluate
from chembed import checkpoint_utils
from chembed.utils import read_df
from chembed.transformer_vae import SELFIESTransformerVAE

def test_vae(vae: SELFIESTransformerVAE, batch_size: int, nb_samples_generation: int, train_file: Path, test_file: Path, nb_subsample_for_latent_test: int, ref_file: Path, properties_to_evaluate=['MolWt', 'MolLogP', 'TPSA', 'BertzCT', 'BalabanJ','Kappa1', 'Kappa2', 'Kappa3']) -> Dict[str,float]:
    df_test = read_df(test_file) 
    df_test_latent = df_test.head(nb_subsample_for_latent_test)
    device = vae.device
    res = {}
    res.update(evaluate.evaluate_loss_functions(vae, df_test, device, batch_size=batch_size))
    res.update(evaluate.evaluate_reconstruction(vae, df_test, device, batch_size=batch_size, compute_substitutions=False))
    res.update(evaluate.evaluate_generation(vae, device, nb_samples=nb_samples_generation, batch_size=batch_size, test_novelty=True, train_file=train_file, test_property_distributions=True, ref_file_for_prop_distrib_test=ref_file))
    res.update(evaluate.evaluate_latent_space(vae, df_test_latent, device, batch_size=batch_size, properties_to_evaluate=properties_to_evaluate))
    return res


def test_vae_from_model_dir(model_dir: Path, device: str, batch_size: int, nb_samples_generation: int, train_file: Path, test_file: Path, nb_subsample_for_latent_test: int, ref_file: Path) -> Dict[str,float]:
    vae = checkpoint_utils.load_last_checkpoint_in_dir(model_dir, device)
    return test_vae(vae, batch_size, nb_samples_generation, train_file, test_file, nb_subsample_for_latent_test, ref_file)
    

def load_existing_keys(output_file: Path) -> Set[str]:
    if not output_file.is_file():
        return set()
    df = pd.read_csv(output_file, sep=';')
    return set(df['name'].to_list())


def log_experiment(row: Dict, output_file: Path):
    exists = output_file.is_file()
    with open(output_file, 'a') as of:
        fieldnames = list(row.keys())
        csv_writer = csv.DictWriter(of, fieldnames=fieldnames, delimiter=';')
        if not exists:
            csv_writer.writeheader()
        csv_writer.writerow(row)


def run_ablations_and_log(runs: Dict, output_file: Path, log_dir: Path, common_flags: List, device_str: str, batch_size: int, nb_samples_generation: int, nb_subsample_for_latent_test: int, train_file: Path, test_file: Path, ref_file: Path):
    already_ran = load_existing_keys(output_file)

    for run in runs:
        run_name = "baseline" 

        flags = common_flags.copy()

        # token embeddings initialization
        if run['initialize_token_embeddings']:
            flags += ['--initialize_embedding_weights']
            run_name+="_init"
        else:
            flags += ['--dont_initialize_embedding_weights']
            run_name+="_noinit"

        # KL divergence
        flags += ['--kl_div_coefficient', str(run['kl_div_coefficient'])]
        run_name += f"_KL_{run['kl_div_coefficient']}"

        # properties
        if run['properties_loss_coefficient']==0.0:
            flags += ['--dont_train_with_properties']
        else:
            flags += ['--properties_loss_coefficient', str(run['properties_loss_coefficient'])]
        run_name += f"_p_{run['properties_loss_coefficient']}"

        # Tanimoto
        if run['tanimoto_loss_coefficient']==0.0:
            flags += ['--dont_train_with_tanimoto']
        else:
            flags += ['--tanimoto_loss_coefficient', str(run['tanimoto_loss_coefficient'])]
        run_name += f"_T_{run['tanimoto_loss_coefficient']}"

        # random seed
        flags += ['--random_seed', str(run['seed'])]
        run_name+=f"_s_{run['seed']}"


        # run name
        print(run_name)
        run['name'] = run_name
        flags += ['--model_name', run_name] 
        
        if run_name not in already_ran:

            model_log_dir = log_dir/run_name

            if not model_log_dir.is_dir():
                train_main(flags)

            run.update(test_vae_from_model_dir(model_log_dir, device_str, batch_size=batch_size, nb_samples_generation=nb_samples_generation, nb_subsample_for_latent_test=nb_subsample_for_latent_test, train_file=train_file, test_file=test_file, ref_file=ref_file))

            log_experiment(run, output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True, type=Path)
    parser.add_argument("--validation_path", required=True, type=Path)
    parser.add_argument("--properties_statistics_path", type=Path)
    parser.add_argument("--log_dir", required=True, type=Path)
    parser.add_argument("--gpu_devices", nargs="+", type=int, default=[0])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seeds", nargs="+", default=['42','43','44'])
    parser.add_argument("--use_precomputed_fingerprints", action='store_true', default=False)
    parser.add_argument("--nb_samples_generation", type=int, default=10000)
    parser.add_argument("--nb_subsample_for_latent_test", type=int, default=2000)
    parser.add_argument("--ref_file", type=Path, required=True)
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    device_str = f'cuda:{args.gpu_devices[0]}'

    common_flags = [
        "--train_path", str(args.train_path),
        "--validation_path", str(args.validation_path),
        "--log_dir", str(args.log_dir),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--gpu_devices", *[str(g) for g in args.gpu_devices],
        "--max_epochs", str(args.max_epochs),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--properties_statistics_path", str(args.properties_statistics_path),
        "--strictly_deterministic"
    ]

    if args.use_precomputed_fingerprints:
        common_flags += ['--use_precomputed_fingerprints']

    runs = []

    KL_DEFAULT = 0.2
    T_DEFAULT = 0.5

    for seed in args.seeds:
        # baseline: no properties, no Tanimoto, KL default, initialization
        runs.append({'properties_loss_coefficient': 0.0, 'tanimoto_loss_coefficient': 0.0, 'kl_div_coefficient': KL_DEFAULT, 'initialize_token_embeddings': True, 'seed': seed})

        # baseline but no initialization
        runs.append({'properties_loss_coefficient': 0.0, 'tanimoto_loss_coefficient': 0.0, 'kl_div_coefficient': KL_DEFAULT, 'initialize_token_embeddings': False, 'seed': seed})

        # baseline with Tanimoto, no properties
        for t in [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]:
            runs.append({'properties_loss_coefficient': 0.0, 'tanimoto_loss_coefficient': t, 'kl_div_coefficient': KL_DEFAULT, 'initialize_token_embeddings': True, 'seed': seed})

        # baseline with properties, no Tanimoto
        for p in [0.0, 1.0]:
            runs.append({'properties_loss_coefficient': p, 'tanimoto_loss_coefficient': 0.0, 'kl_div_coefficient': KL_DEFAULT, 'initialize_token_embeddings': True, 'seed': seed})

        # baseline with KL sweep, Tanimoto or not, properties or not
        for kl in [0.0, 0.4, 0.6, 0.8, 1.0, 2.0]:
            for t in [0.0, 0.5]:
                for p in [0.0, 1.0]:
                    runs.append({'properties_loss_coefficient': p, 'tanimoto_loss_coefficient': t, 'kl_div_coefficient': kl, 'initialize_token_embeddings': True, 'seed': seed})


    output_file = args.log_dir/'summary.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    run_ablations_and_log(runs, output_file, log_dir=args.log_dir, common_flags=common_flags, device_str=device_str, batch_size=args.batch_size, nb_samples_generation=args.nb_samples_generation, nb_subsample_for_latent_test=args.nb_subsample_for_latent_test, train_file=args.train_path, test_file=args.validation_path, ref_file=args.ref_file) 
