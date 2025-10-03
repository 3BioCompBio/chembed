import os
from pathlib import Path
import torch

from chembed.utils import set_random_seed_everywhere
from chembed.transformer_vae import SELFIESTransformerVAE
from chembed.data_handler import load_vocab

from run_ablation_studies import log_experiment, test_vae

MAIN_DIR = Path(os.path.realpath(__file__)).parent.parent

if __name__=="__main__":
   output_dir = MAIN_DIR/'logs'/'ablation_studies'/'pubchem_preprocessed_with_selfies_with_properties_small_train_train_with_stratum_sample_1000000_no_training'
   output_dir.mkdir(parents=True, exist_ok=True)

   output_file = output_dir/'ablation_summary_2.csv'

   nb_samples_generation = 10_000
   nb_subsamples_for_latent_test = 2_000
   batch_size = 1024 

   data_dir = Path('/home/hugo/data/pubchem/data')
   train_file = data_dir/'pubchem_preprocessed_with_selfies_with_properties_small_train_train_with_stratum_sample_1000000_train.parquet'
   test_file = data_dir/'pubchem_preprocessed_with_selfies_with_properties_small_train_train_with_stratum_sample_1000000_validation.parquet'

   model_config = {
        'vae_model_class': 'SELFIESTransformerVAE',
        'max_len': 256,
        'dim_feedforward_encoder': 1024,
        'nb_layers_encoder': 6,
        'nhead': 8,
        'dim_feedforward_decoder': 512,
        'nb_layers_decoder': 6,
        'dropout': 0.1,
        'min_vae_sigma': 1e-4,
        'properties': [],
        'layer_norm_eps': 1e-2,
        'layer_norm_in_regressor': False,
        'use_log_sigma': True,
        'd_model': 256,
        'd_bottleneck': 1,
        'd_latent': 256
        }
 

   vocab = load_vocab()

   device = torch.device('cuda:1')

   for initialize_embedding_weights in [True, False]:
       for seed in [42, 43, 44]:
           model_config['initialize_embedding_weights'] = initialize_embedding_weights
           set_random_seed_everywhere(seed)
           vae = SELFIESTransformerVAE(vocab, model_config).to(device)
           row = {'name': f'initialize_{initialize_embedding_weights}_s{seed}'}
           row.update(test_vae(vae, batch_size, nb_samples_generation, train_file, test_file, nb_subsamples_for_latent_test, train_file))
           log_experiment(row, output_file)
