# chembed
![License](https://img.shields.io/github/license/htalibart/chembed)

chembed is a large-scale Variational AutoEncoder based on SELFIES inputs with a structured, chemistry-aware latent space.

## Features
TODO 

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

See notebook TODO
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
We also include command line entry points so you can run common tasks straight from the terminal without writing any code.
TODO
TODO fine-tuning




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
  year    = {2023},
  volume  = {XX},
  number  = {YY},
  pages   = {ZZZ--ZZZ},
  doi     = {10.XXXX/XXXXX}
}
```
