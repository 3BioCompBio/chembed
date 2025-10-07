from typing import Callable, Any

import argparse
import time
from pathlib import Path
import pandas as pd

import torch

from chembed import checkpoint_utils
from chembed.downstream import ga_optimizer as chembed_optimizer

from tartarus import pce, tadf, reactivity, docking

def pce_pcbm_sas(smiles: str, **kwargs) -> float:
    pcbm, pcdtbt = pce.get_properties(smiles)
    return pcbm

def pce_pcdtbt_sas(smiles: str, **kwargs) -> float:
    pcbm, pcdtbt = pce.get_properties(smiles)
    return pcdtbt

def singlet_triplet_value(smiles: str, **kwargs) -> float:
    st, osc, combined = tadf.get_properties(smiles)
    return -st

def oscillator_strength(smiles: str, **kwargs) -> float:
    st, osc, combined = tadf.get_properties(smiles)
    return osc

def multi_objective_value(smiles: str, **kwargs) -> float:
    st, osc, combined = tadf.get_properties(smiles)
    return combined

def Ea(smiles: str, scratch: Path, **kwargs) -> float:
    Ea, Er, sum_Ea_Er, diff_Ea_Er = reactivity.get_properties(smiles, scratch=str(scratch))
    return Ea

def Er(smiles: str, scratch: Path, **kwargs) -> float:
    Ea, Er, sum_Ea_Er, diff_Ea_Er = reactivity.get_properties(smiles, scratch=str(scratch))
    return Er

def sum_Ea_Er(smiles: str, scratch: Path, **kwargs) -> float:
    Ea, Er, sum_Ea_Er, diff_Ea_Er = reactivity.get_properties(smiles, scratch=str(scratch))
    return sum_Ea_Er

def diff_Ea_Er(smiles: str, scratch: Path, **kwargs) -> float:
    Ea, Er, sum_Ea_Er, diff_Ea_Er = reactivity.get_properties(smiles, scratch=str(scratch))
    return diff_Ea_Er

def get_docking_score_function(target: str, docking_program: str, filter_molecules: bool, **kwargs) -> Callable[[str],float]:
    return lambda smiles: docking.perform_calc_single(smiles, target, docking_program=docking_program, filter_molecules=filter_molecules)

def negate_function(f: Callable) -> Callable:
    return lambda *args, **kwargs: -f(*args, **kwargs)

def get_fitness_function(column_name: str, minimize: bool, **kwargs) -> Callable[[str],float]:
    column_name = column_name.replace('-', '_')
    if column_name in ['1syh_score', '4lde_score', '6y2f_score']:
        target = column_name.split('_')[0]
        f = get_docking_score_function(target, docking_program='qvina', **kwargs)
    elif column_name in ['1syh_score_smina', '4lde_score_smina', '6y2f_score_smina']:
        target = column_name.split('_')[0]
        f = get_docking_score_function(target, docking_program='smina', **kwargs)
    else:
        f = lambda smiles: eval(column_name)(smiles, **kwargs)
    if minimize:
        f = negate_function(f)
    return f

def fitness_check(fitness_function: Callable, smiles: str, expected_fitness: float, epsilon_fitness_check=10) -> float:
    print(f"performing fitness check on smiles={smiles}...")
    start_time = time.time()
    res = fitness_function(smiles)
    end_time = time.time()
    print("fitness run time:", end_time-start_time)
    error = abs(res - expected_fitness)
    if error > epsilon_fitness_check:
        raise RuntimeError(f"fitness check failed: output {res}, expected {expected_fitness}, > {epsilon_fitness_check}")
    else:
        print(f"fitness ok: output {res}, expected {expected_fitness}, < {epsilon_fitness_check}")
    return error

def None_or_float(argument: Any) -> None | float:
    if (argument=='None') or (argument is None):
        return None
    else:
        return float(argument)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vae", type=Path)
    parser.add_argument("input_dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--nb_generations", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=500)
    parser.add_argument("--nb_best_during_optimization", type=str, default='automatic')
    parser.add_argument("--nb_best_initial_max", type=int, default=20)
    parser.add_argument("--nb_best_initial_strategy", type=str, default='automatic', choices=['automatic', 'fixed'])
    parser.add_argument("--epsilon_best_fitness", type=float, default=0.05)
    crossovers_group = parser.add_mutually_exclusive_group(required=True)
    crossovers_group.add_argument('--crossovers', dest='crossovers', action='store_true')
    crossovers_group.add_argument('--no_crossovers', dest='crossovers', action='store_false')
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--column_name", type=str, required=True)
    direction_group = parser.add_mutually_exclusive_group(required=True)
    direction_group.add_argument("--minimize", dest='minimize', action='store_true')
    direction_group.add_argument("--maximize", dest='minimize', action='store_false')
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest='resume', action='store_true', default=False)
    resume_group.add_argument("--new", dest='resume', action='store_false')
    parser.add_argument("--scratch", type=Path, default=Path('/tmp/'))
    parser.add_argument("--dont_filter_molecules", action='store_true', default=False, help="only for docking")
    parser.add_argument("--filter_sascore", type=None_or_float, default=None, help='Threshold to filter molecules by synthetic accessibility in the validity check (recommended: 4.5)')
    parser.add_argument("--filter_qed", type=None_or_float, default=None, help='Threshold to filter molecules by drug-likeliness in the validity check (recommended: 0.3)')
    args = parser.parse_args()

    device = torch.device(args.device)

    vae = checkpoint_utils.load_vae_from_checkpoint(args.vae, device=device)    
    vae.eval()

    hyperparameters = {
        'nb_generations': args.nb_generations,
        'pop_size': args.pop_size,
        'nb_best': args.nb_best_during_optimization,
        'std': args.std,
        'crossovers': args.crossovers,
        'batch_size': args.batch_size,
        'epsilon_best_fitness': args.epsilon_best_fitness
        }   

    column_name=args.column_name
    minimize = args.minimize

    fitness_function = get_fitness_function(column_name, minimize=minimize, scratch=args.scratch, filter_molecules=(not args.dont_filter_molecules))

    if column_name.endswith('smina'):
        column_name = column_name[:-len('_smina')]

    df = pd.read_csv(args.input_dataset, usecols=['smiles', column_name])
    df = df.sort_values(by=column_name, ascending=minimize)

    acceptable_fitness_error = df[column_name].std()

    if args.nb_best_initial_strategy=="automatic":
        best_fitness = df.iloc[0][column_name]
        nb_best_initial = len(df[df[column_name]==best_fitness])
        nb_best_initial = min(nb_best_initial, args.nb_best_initial_max)
    else:
        nb_best_initial = args.nb_best_initial_max
    
    df = df.head(nb_best_initial)

    initial_smiles_list = df['smiles'].tolist()
    initial_fitness_values = df[column_name].tolist()
    if minimize:
        initial_fitness_values = [-x for x in initial_fitness_values]

    print("initial smiles:", initial_smiles_list)
    print("initial fitness values:", initial_fitness_values)

    output_dir = args.output_dir
    if output_dir.is_dir() and not args.resume:
        raise FileExistsError(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    error = fitness_check(fitness_function, initial_smiles_list[0], initial_fitness_values[0])
    # if value in dataset is too different from value outputted by the fitness function, don't use dataset values
    if error > acceptable_fitness_error:
        print(f"Fitness error: {error} > {acceptable_fitness_error}. Not trusting initial fitness values.")
        initial_fitness_values = None
    else:
        print(f"When computing fitness: {error} difference. Keeping initial fitness values")

    best_df = chembed_optimizer.maximize(initial_smiles_list, fitness_function, vae, hyperparameters, initial_fitness_list=initial_fitness_values, all_collector_file=output_dir/'collector.csv', best_collector_file=output_dir/'best_collector.csv', resume=args.resume, filter_sascore=args.filter_sascore, filter_qed=args.filter_qed)

    print("best df:", best_df)
