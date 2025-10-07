# Experiments

This directory contains scripts used for the molecular optimization experiments described in the paper. These research scripts rely on external datasets and utilities, and therefore are not part of the core chembed package. They are provided as-is to support transparency and reproducibility. Because they depend on external resources and required custom setup on our side, we cannot guarantee support for issues specific to these scripts, and their documentation is limited.

Experiments on the Tartarus benchmark were performed using data and scripts from [their github repository](https://github.com/aspuru-guzik-group/Tartarus).

## Training

Datasets were downloaded from [https://github.com/aspuru-guzik-group/Tartarus/tree/main/datasets](https://github.com/aspuru-guzik-group/Tartarus/tree/main/datasets) and pre-processed:
- SMILES were standardized and SELFIES added (`scripts/add_selfies.py`)
- Vocabulary was computed (`scripts/build_vocab.py`)
- Properties were computed (`scripts/add_properties.py`), clipped and standardized (`scripts/preprocess_properties.py`)
- Morgan fingerprints were pre-computed (`scripts/precompute_fingerprints.py`)
- Split into train (80%) and test (20%), ensuring that all tokens appeared in the train set

Models were finetuned from the main chembed model with `chembed-finetune`.


## Environment

To run experiments on a computer cluster, we made Tartarus pip-installable and built a Singularity container (see `build_tartarus_singularity.sh`)


## Optimization

Experiments were ran with the script `optimize_fitness.py`, e.g. for the 1SYH docking task:
```
python optimize_fitness.py /path/to/docking/vae/checkpath.ckpt \
                        docking_train_set.csv \
                        1syh_xp_output_dir/ \
                        --column_name 1syh_score \
                        --minimize \
                        --nb_generations 10 \
                        --pop_size 500 \
                        --nb_best_during_optimization automatic \
                        --nb_best_initial_max 20 \
                        --nb_best_initial_strategy automatic \
                        --epsilon_best_fitness 0.01 \
                        --crossovers \
                        --std 0.5
```
