from typing import Callable, Optional, Dict, List, Tuple

import os
import sys
from pathlib import Path
import selfies as sf
import pandas as pd
import numpy as np
import time

import torch
from torch import Tensor

from chembed.encode import encode_multiple_selfies
from chembed.decode import decode_zs_to_selfies 

from rdkit import Chem
from rdkit.Chem import QED
from rdkit import RDLogger

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score')) # noqa: E402
import sascorer

import logging
RDLogger.DisableLog('rdApp.*') # noqa: E402
logger = logging.getLogger(__name__)

INITIAL_GENERATION = -1


def evaluate_if_missing(smiles: str, fitness_function: Callable[[str], float], previous_fitness_evaluations: Dict[str,float]) -> float:
    """ returns the fitness value for @smiles, using cache @previous_fitness_evaluations """
    if smiles in previous_fitness_evaluations:
        evaluation = previous_fitness_evaluations[smiles]
    else:
        evaluation = fitness_function(smiles)
        previous_fitness_evaluations[smiles] = evaluation
    return evaluation


def write_to_csv(df: pd.DataFrame, output_file: Path, new_file: bool):
    """ appends dataframe to csv if already exists, otherwise creates it """
    if new_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['selfies', 'smiles']:
        df = df.drop_duplicates(subset=col, keep="first") if col in df.columns else df
    df = df.dropna().reset_index(drop=True)
    return df


def get_all_crossovers(zs: Tensor) -> Tensor:
    """ returns all crossovers for zs, i.e. (zs[0]+zs[1])/2, (zs[1]+zs[2])/2, etc. """
    all_crossovers = 0.5*(zs[:, None, :] + zs[None, :, :])
    indices = torch.triu_indices(len(zs), len(zs), offset=1)
    new_zs = all_crossovers[indices[0], indices[1]]
    return new_zs


def sample_around(z: Tensor, nb_samples_per_seed: int, std: float) -> Tensor:
    """ returns Gaussian samples around each seed @z[i] with a given standard deviation @std """
    gens = [z + std*torch.randn_like(z) for _ in range(nb_samples_per_seed)]
    return torch.vstack(gens)


def check_validity(smiles: str, filter_sascore: Optional[float] = None, filter_qed: Optional[float] = None):
    """ returns True if the SMILES is valid according to RDKit, also checks SAScore and QED if thresholds are provided """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return np.nan
    else:
        if filter_sascore is not None:
            sa_score = sascorer.calculateScore(mol)
            if sa_score > filter_sascore:
                return np.nan
        if filter_qed is not None:
            qed = QED.qed(mol)
            if qed < filter_qed:
                return np.nan
        return smiles

def get_df_where_fitness_close_to_max(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """ returns subdf with all rows whose fitness values are close to the max fitness value within a chosen @epsilon """
    best_fitness = df['fitness'].max()
    return df[df['fitness'] >= best_fitness*(1-epsilon)]


def get_best_df_with_generated(df: pd.DataFrame, nb_best: int, epsilon: float) -> pd.DataFrame:
    if nb_best=='automatic':
        df_best_overall = get_df_where_fitness_close_to_max(df, epsilon=epsilon)
        df_generated = df[df['generation'] > INITIAL_GENERATION]
        df_best_generated = get_df_where_fitness_close_to_max(df_generated, epsilon=0)
        df_best_both = pd.concat([df_best_overall, df_best_generated])
        df_best_both = df_best_both.drop_duplicates(subset=['smiles'])
        return df_best_both
    else:
        try:
            nb_best = int(nb_best)
        except ValueError:
            raise NotImplementedError(nb_best)
        return df.nlargest(nb_best, 'fitness')


def init_from_collector(collector_file: Path, hyperparameters: Dict) -> Tuple[pd.DataFrame, int, Dict[str,float]]:
    """ initializes from an existing collector file.
        returns the current best datagrame, the last generation index evaluated, and a cache dictionary of already computed fitnesses """
    collector_df = pd.read_csv(collector_file)

    last_row = collector_df.tail(1).iloc[0]
    last_gen = last_row['generation']

    best_df = get_best_df_with_generated(collector_df, hyperparameters['nb_best'], hyperparameters['epsilon_best_fitness'])

    previous_fitness_evaluations = {}
    collector_df_unique = collector_df.drop_duplicates(subset='smiles')
    for i, row in collector_df_unique.iterrows():
        previous_fitness_evaluations[row['smiles']] = row['fitness']

    return best_df, last_gen, previous_fitness_evaluations


def check_resume_inputs(all_collector_file: Optional[Path], best_collector_file: Optional[Path]) -> None:
    if (all_collector_file is None or best_collector_file is None or not all_collector_file.is_file() or not best_collector_file.is_file()):
        raise FileNotFoundError("Resume requested but collector files are missing or invalid.")



def init_state(
        initial_smiles_list: List[str],
        fitness_function: Callable[[str], float],
        hyperparameters: Dict,
        resume: bool,
        all_collector_file: Optional[Path] = None,
        initial_fitness_list: Optional[List[float]] = None
        ) -> Tuple[pd.DataFrame, int, Dict[str, float]]:

    if resume:
        collector_df = pd.read_csv(all_collector_file)
        last_row = collector_df.tail(1).iloc[0]
        last_gen = int(last_row["generation"])
        best_df = get_best_df_with_generated(collector_df, hyperparameters["nb_best"], hyperparameters["epsilon_best_fitness"])
        previous_fitness_evaluations = {row["smiles"]: row["fitness"] for i, row in collector_df.drop_duplicates(subset="smiles").iterrows()}
        start_gen = last_gen + 1
        return best_df, start_gen, previous_fitness_evaluations

    best_df = pd.DataFrame({"smiles": initial_smiles_list})
    best_df["selfies"] = best_df["smiles"].apply(sf.encoder)
    if initial_fitness_list is None:
        previous_fitness_evaluations = {}
        initial_fitness_list = [evaluate_if_missing(s, fitness_function, previous_fitness_evaluations) for s in best_df["smiles"]]
    else:
        previous_fitness_evaluations = {s: v for s, v in zip(initial_smiles_list, initial_fitness_list)}
    best_df["fitness"] = initial_fitness_list
    best_df["generation"] = [INITIAL_GENERATION] * len(best_df)
    best_df["generation_time"] = [np.nan] * len(best_df)
    start_gen = 0
    return best_df, start_gen, previous_fitness_evaluations




def generate_population(best_df: pd.DataFrame, vae, batch_size: int, crossovers: bool, filter_sascore: Optional[float], filter_qed: Optional[float]) -> Tuple[pd.DataFrame, Tensor]:
    z_initial = encode_multiple_selfies(best_df['selfies'].tolist(), vae, batch_size=batch_size, replace_if_not_in_vocab=False, use_sigma=True)
    if crossovers:
        z_crossovers = get_all_crossovers(z_initial)
        z_initial = torch.vstack([z_initial, z_crossovers])
    initial_selfies = decode_zs_to_selfies(z_initial, vae, batch_size)
    current_df = pd.DataFrame.from_dict({'selfies': initial_selfies})
    current_df['smiles'] = current_df['selfies'].apply(sf.decoder)
    current_df['smiles'] = current_df['smiles'].apply(lambda s: check_validity(s, filter_sascore=filter_sascore, filter_qed=filter_qed))
    current_df = clean_df(current_df)
    return current_df, z_initial



def mutate_until_size(current_df: pd.DataFrame, z_initial: Tensor, vae, batch_size: int, pop_size: int, std: float, filter_sascore: Optional[float], filter_qed: Optional[float]) -> pd.DataFrame:
    while len(current_df) < pop_size:
        nb_mutations = pop_size - len(current_df)
        logger.info("generating %d mutations", nb_mutations)
        nb_samples_per_seed = int(np.ceil(nb_mutations / len(z_initial)))
        z_mutations = sample_around(z_initial, nb_samples_per_seed, std)
        selfies_mutations = decode_zs_to_selfies(z_mutations, vae, batch_size)
        df_mutations = pd.DataFrame.from_dict({'selfies': selfies_mutations})
        df_mutations = df_mutations.drop_duplicates(subset='selfies')
        df_mutations['smiles'] = df_mutations['selfies'].apply(sf.decoder)
        df_mutations['smiles'] = df_mutations['smiles'].apply(lambda s: check_validity(s, filter_sascore=filter_sascore, filter_qed=filter_qed))
        df_mutations = df_mutations.dropna()
        current_df = pd.concat([current_df, df_mutations])
        current_df = clean_df(current_df)
        logger.debug("len(current_df)=%d", len(current_df))
    if len(current_df) > pop_size:
        current_df = current_df.sample(n=pop_size)
    return current_df


def evaluate_population(current_df: pd.DataFrame, fitness_function: Callable[[str], float], previous_fitness_evaluations: Dict[str,float]) -> List[float]:
    logger.info("evaluating fitness...")
    fitness_evaluations = []
    for smiles_index, smiles in enumerate(current_df['smiles']):
        start_fitness_evaluation = time.time()
        logger.debug("evaluation %d %s", smiles_index, smiles)
        fitness_evaluations.append(evaluate_if_missing(smiles, fitness_function, previous_fitness_evaluations))
        logger.debug("eval_time_s=%.3f", time.time()-start_fitness_evaluation)
    logger.info("fitness evaluated")
    current_df = current_df.copy()
    return fitness_evaluations


def select_best(best_df: pd.DataFrame, current_df: pd.DataFrame, nb_best, epsilon_best_fitness: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("len(current_df)=%d", len(current_df))
    current_best_df = get_best_df_with_generated(current_df, nb_best, epsilon=epsilon_best_fitness)
    logger.debug("len(current_best_df)=%d", len(current_best_df))
    logger.debug("current_best_df_fitnesses=%s", current_best_df['fitness'].tolist())
    best_df = pd.concat([best_df, current_best_df])
    best_df = get_best_df_with_generated(best_df, nb_best, epsilon=epsilon_best_fitness)
    logger.info("len(best_df)=%d", len(best_df))
    logger.debug("best_df_fitnesses=%s", best_df['fitness'].tolist())
    return best_df, current_best_df


def maximize(
        initial_smiles_list: List[str],
        fitness_function: Callable[[str], float],
        vae,
        hyperparameters: Dict,
        initial_fitness_list: List[float] = None,
        all_collector_file: Optional[Path] = None,
        best_collector_file: Optional[Path] = None,
        resume: bool = False,
        filter_sascore: Optional[float] = None,
        filter_qed: Optional[float] = None
        ) -> pd.DataFrame:

    batch_size = int(hyperparameters["batch_size"])
    pop_size = int(hyperparameters["pop_size"])
    std = float(hyperparameters["std"])
    nb_best = hyperparameters["nb_best"]
    epsilon_best_fitness = float(hyperparameters["epsilon_best_fitness"])
    crossovers = bool(hyperparameters["crossovers"])
    nb_generations = int(hyperparameters["nb_generations"])


    if resume:
        check_resume_inputs(all_collector_file, best_collector_file)

    best_df, start_gen, previous_fitness_evaluations = init_state(initial_smiles_list, fitness_function, hyperparameters, resume=resume, all_collector_file=all_collector_file, initial_fitness_list=initial_fitness_list)

    for gen in range(start_gen, nb_generations):
        start_time = time.time()

        current_df, z_initial = generate_population(best_df=best_df, vae=vae, batch_size=batch_size, crossovers=crossovers, filter_sascore=filter_sascore, filter_qed=filter_qed)

        if len(current_df) < pop_size:
            current_df = mutate_until_size(current_df=current_df, z_initial=z_initial, vae=vae, batch_size=batch_size, pop_size=pop_size, std=std, filter_sascore=filter_sascore, filter_qed=filter_qed)

        current_df['fitness'] = evaluate_population(current_df=current_df, fitness_function=fitness_function, previous_fitness_evaluations=previous_fitness_evaluations)
        current_df['generation'] = [gen] * len(current_df)

        gen_time = time.time()-start_time
        current_df['generation_time'] = [gen_time] * len(current_df)

        if all_collector_file is not None:
            write_to_csv(current_df, all_collector_file, new_file=(gen==0))

        best_df, current_best_df = select_best(best_df=best_df, current_df=current_df, nb_best=nb_best, epsilon_best_fitness=epsilon_best_fitness)

        if best_collector_file is not None:
            write_to_csv(best_df, best_collector_file, new_file=(gen==0))

    return best_df




def minimize(
        initial_smiles_list: List[str],
        fitness_function: Callable[[str], float],
        vae,
        hyperparameters: Dict,
        initial_fitness_list: List[float] = None,
        all_collector_file: Optional[Path] = None,
        best_collector_file: Optional[Path] = None,
        resume: bool = False,
        filter_sascore: Optional[float] = None,
        filter_qed: Optional[float] = None
        ) -> pd.DataFrame:

    return maximize(initial_smiles_list, lambda x: -1*fitness_function(x), vae, hyperparameters, initial_fitness_list=initial_fitness_list, all_collector_file=all_collector_file, best_collector_file=best_collector_file, filter_sascore=filter_sascore, filter_qed=filter_qed)

