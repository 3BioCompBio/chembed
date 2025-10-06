# chembed
![License](https://img.shields.io/github/license/htalibart/chembed)

chembed is a large-scale Variational AutoEncoder based on SELFIES representations with a structured, chemistry-aware latent space for molecular encoding.

## Features
- Variational Autoencoder with chemistry-aware molecular embeddings
- Continuous latent space enabling smooth interpolation and molecule generation
- High reconstruction accuracy
- High validity of generated molecules
- Supports downstream molecular optimization and drug design tasks

## Installation
```
conda env create -f environment.yml
pip install .
```

    TODO pip
    TODO conda

## Usage
    
### As a Python module
chembed is primarily designed as a Python module for integration into your own code.

#### Load the pre-trained model
A pre-trained model is available from HuggingFace (3BioCompBio/chembed-default). To load it, simply run
```python
import torch
from chembed import checkpoint_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vae = checkpoint_utils.load_vae_from_hub(device)
```


#### Encode and decode

```python
import selfies as sf
import torch

from chembed import encode as enc
from chembed import decode as dec
from chembed import checkpoint_utils
from chembed.mol_utils import standardize_smiles

# load VAE
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vae = checkpoint_utils.load_vae_from_hub(device)
vae.eval()

# encode from SMILES
smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCN(CC)CC(=O)NC1=C(C)C=CC=C1C']
zs = enc.encode_multiple_smiles(smiles_list, vae)

# or from SELFIES (molecule must be standardized first)
smiles_list = ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCN(CC)CC(=O)NC1=C(C)C=CC=C1C']
selfies_list = [sf.encoder(standardize_smiles(s)) for s in smiles_list]
zs = enc.encode_multiple_selfies(selfies_list, vae)

# decode to SELFIES
decoded_selfies = dec.decode_zs_to_selfies(zs, vae)
print(decoded_selfies) # -> ['[C][C][=Branch1][C][=O][O][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=Branch1][C][=O][O]', '[C][C][N][Branch1][Ring1][C][C][C][C][=Branch1][C][=O][N][C][=C][Branch1][C][C][C][=C][C][=C][Ring1][#Branch1][C]'] 

# or to SMILES
decoded_smiles = dec.decode_zs_to_smiles(zs, vae)
print(decoded_smiles) # -> ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CCN(CC)CC(=O)NC1=C(C)C=CC=C1C'] 
```


#### Linear interpolations in latent space

```python
import torch
from chembed import checkpoint_utils
from chembed import encode as enc
from chembed import decode as dec

# load VAE
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vae = checkpoint_utils.load_vae_from_hub(device)
vae.eval()

# molecules to interpolate
smiles1 = 'O=CN(C=O)C1=NN2C=CC=C2N1'
smiles2 = 'NC1=CC=CN1C1=CSN=C1C=O'

# perform interpolation
zs = enc.encode_multiple_smiles([smiles1, smiles2], vae)
z_crossover = (zs[0] + zs[1]) / 2

# decode
smiles_crossover = dec.decode_zs_to_smiles(z_crossover, vae)[0]
print(smiles_crossover) # -> NC1=CC=CN1C2=NN=CC=C2C=O
```

See notebook `notebooks/examples/latent_space_interpolation.ipynb`


#### Fitness optimization

```python
import torch
from chembed import checkpoint_utils
from chembed.downstream import ga_optimizer
from chembed.utils import set_random_seed_everywhere

# define function to maximize (here a dummy function that returns the number of carbons)
def my_function_to_maximize(smiles: str) -> float:
    return smiles.count('C')

# set seed for reproducibility (not necessary)
set_random_seed_everywhere(42)

# load VAE
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vae = checkpoint_utils.load_vae_from_hub(device)
vae.eval()

# initial set 
initial_smiles = ['CCC']

# define search algorithm hyperparameters
hyperparameters = {
        'nb_generations': 100,
        'batch_size': 256,
        'pop_size': 1,
        'std': 0.5,
        'nb_best': 'automatic',
        'epsilon_best_fitness': 0.0,
        'crossovers': False
        }

# run optimization
out_df = ga_optimizer.maximize(initial_smiles_list=initial_smiles, fitness_function=my_function_to_maximize, vae=vae, hyperparameters=hyperparameters)

# get list of best SMILES candidates
print(out_df['smiles'].to_list()) # -> ['CCCCCCCCCCCCCC']
```


### From the command line
We also provide command line tools for encoding, decoding, generating, training and fine-tuning without writing Python code. By default, the [default model from HuggingFace](https://huggingface.co/3BioCompBio/chembed-default) is loaded.


#### Encode
Encode from SMILES:
```
chembed-encode --input_smiles input_smiles.smi --output embeddings.npy
```

Encode from SELFIES:
```
chembed-encode --input_selfies input_selfies.selfies --output embeddings.npy
```

Embeddings will be saved in numpy format if the output file ends with `.npy`, otherwise in PyTorch's own serialization format (based on pickle).

If your SELFIES strings contain tokens unseen during training, an `UnknownTokenError` will be raised. You can [fine-tune the model on your own dataset](#fine-tune), or use `--replace_if_not_in_vocab` to automatically replace unseen tokens with semantically closest ones.


#### Decode

Decode to SELFIES:
```
chembed-decode embeddings.npy --decode_to_selfies --output output_selfies.selfies 
```

Decode to SMILES:
```
chembed-decode embeddings.npy --decode_to_smiles --output output_smiles.smi 
```


#### Generate
```
chembed-generate 1000 generated.csv
```
will generate a csv file with 1000 SELFIES and associated SMILES



#### Train

To train a model from scratch **without** the auxiliary property regression task:
```
chembed-train --train_path /path/to/train \
            --validation_path /path/to/validation \
            --log_dir my_logs/ \
            --model_name my_model \
            --dont_train_with_properties \
            --use_precomputed_fingerprints
```

Supported formats for train and validation files: `.csv`, `.parquet` and `.pkl`.
Each file must contain a column named `selfies` with standardized SELFIES strings (see [Data pre-processing](#data-pre-processing)). If `--use_precomputed_fingerprints`  is set, the file must include a column `fingerprint` with precomputed Morgan fingerprints. If not, a `smiles` column is required to compute them automatically (slower).


To train **with** the auxiliary property regression task (e.g. with the same properties as in the paper):
```
chembed-train --train_path /path/to/train \
            --validation_path /path/to/validation \
            --log_dir my_logs/ \
            --model_name my_model \
            --properties MolWt MolLogP TPSA BertzCT Kappa1 Kappa2_clipped Kappa3_clipped \
            --properties_statistics_path /path/to/property_statistics.json
```

`/path/to/property_statistics.json` should contain a dictionary with the mean and standard deviation for each property, e.g.:
```
{
"MolLogP": {"mean": 3.15176230019694, "std": 1.840075209019122},
"TPSA": {"mean": 66.8524829233, "std": 32.90379350253526},
"BertzCT": {"mean": 748.3025719454026, "std": 364.6349709092715},
"Kappa1": {"mean": 17.274857205900112, "std": 5.06249998433574},
"Kappa2_clipped": {"mean": 7.458480283718226, "std": 2.5961259559267154},
"Kappa3_clipped": {"mean": 4.192245061283509, "std": 2.023560920288246}
}
```

For each `property`, the train and validation files must include a column `normalized_property` containing standardized values (i.e. subtract the mean and divide by standard deviation). See [Data pre-processing](#data-pre-processing) for generating these files.

To use a custom SELFIES vocabulary, specify `--vocab /path/to/vocab.json`. If omitted, the default vocabulary (`chembed/resources/vocab.json` covering all SELFIES in the PubChem dataset) is used.


#### Fine-tune

Fine-tuning works like [training](#train): 
```
chembed-finetune --train_path my_training_set.csv \
                --validation_path my_validation_set.csv \
                --log_dir my_logs/ \
                --model_name my_finetuned_model \
                --vocab my_vocab.json \
                --checkpath my_logs/my_model/version_0/last.ckpt
```
If ```--checkpath``` is omitted, the default model from HuggingFace is loaded.
If provided, the vocabulary of the pre-trained model will be expanded to include new tokens from ```--vocab```





#### Data pre-processing

We provide multiple data processing scripts to suit different use cases.

Standardize SMILES and add SELFIES to an existing SMILES dataset:
```
python scripts/add_selfies.py --input_file dataset.csv --output_file dataset_with_selfies.csv
```

Build a SELFIES vocabulary:
```
python scripts/build_vocab.py dataset_with_selfies.csv vocab.json
```

Compute molecular properties (all RDKit descriptors are supported):
```
python scripts/add_properties.py --input_file dataset_with_selfies.csv \
                                --output_file dataset_with_properties.csv \
                                --properties MolWt MolLogP TPSA BertzCT Kappa1 Kappa2 Kappa3
```

Preprocess properties (clip values, standardize, generate stats):
```
python scripts/preprocess_properties.py dataset_with_properties.csv dataset_with_standardized_properties.csv \
                                        --properties_to_clip Kappa2 Kappa3 \
                                        --properties_to_normalize MolWt MolLogP TPSA BertzCT Kappa1 Kappa2 Kappa3 \
                                        --output_stats property_statistics.json
```

Precompute fingerprints:
```
python scripts/precompute_fingerprints.py dataset_without_fingerprints.csv --output_file dataset_with_fingerprints.csv
```

Split train/test:
```
python scripts/split_random_train_test.py dataset_with_selfies.csv --output_train my_train_set.csv --output_test my_test_set.csv
```


Oversample rows with rare SELFIES tokens:
```
python scripts/get_overall_token_counts train.csv token_counts.json
python scripts/duplicate_samples_given_token_frequencies.py train.csv \
                                                            --a 1e-5 \
                                                            --token_counts token_counts.json
```




## Dataset

The raw dataset used for training and evaluation is hosted on Zenodo: [10.5281/zenodo.17277040](https://doi.org/10.5281/zenodo.17277040). It contains standardized SMILES, SELFIES, and raw molecular property values. To process it for training:
```
python scripts/preprocess_properties.py train.parquet \
                                        --properties_to_clip Kappa2 Kappa3 \
                                        --properties MolWt MolLogP TPSA BertzCT Kappa1 Kappa2_clipped Kappa3_clipped \
                                        --output_stats property_statistics.json
python scripts/get_overall_token_counts train.parquet token_counts.json
python scripts/split_random_train_test.py train.parquet \
                                        --test_size 0.2 \
                                        --output_train train_train.parquet \
                                        --output_test train_validation.parquet
python scripts/duplicate_samples_given_token_frequencies.py train_train.parquet \
                                                            --a 1e-5 \
                                                            --token_counts token_counts.json
```


## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE.txt) file for details.



## Citation

If you use this method, please cite:
```bibtex
@article{talibart2025chembed,
  title   = {Learning a chemistry-aware latent space for molecular encoding and generation with a large-scale Transformer Variational Autoencoder},
  author  = {Talibart, H. and Gilis, D.},
  journal = {Journal Name},
  year    = {2025},
  volume  = {XX},
  number  = {YY},
  pages   = {ZZZ--ZZZ},
  doi     = {10.XXXX/XXXXX}
}
```
