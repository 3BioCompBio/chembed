from pathlib import Path
import argparse
import pandas as pd
import multiprocessing as mp

from chembed.utils import write_json

def chunk_reader(path, column_name, chunksize):
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=[column_name]):
        yield chunk[column_name]

def process_chunk(args):
    chunk, samples = args
    return set(chunk).intersection(samples)


def parallel_find(csv_path, samples, column_name, num_threads, chunksize=10_000):
    with mp.Pool(num_threads) as pool:
        res = pool.map(process_chunk, ((chunk, samples) for chunk in chunk_reader(csv_path, column_name, chunksize)))
    found = set().union(*res)
    return found


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--column_name", type=str, default='selfies')
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--write_found_to", type=Path, default=None)
    args = parser.parse_args()

    input_file = args.generated
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    df = pd.read_csv(input_file)

    dataset_file = args.dataset
    if not dataset_file.is_file():
        raise FileNotFoundError(dataset_file)
    if dataset_file.suffix!='.csv':
        raise ValueError("dataset format should be csv")
    
    samples = set(df[args.column_name].tolist())

    output_file = args.output
    if output_file is None:
        output_file = input_file.parent/f"{input_file.stem}_novelty.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_found = parallel_find(dataset_file, samples, args.column_name, args.num_threads, chunksize=args.chunksize)
    
    if args.write_found_to:
        found_file = args.write_found_to
        found_file.parent.mkdir(parents=True, exist_ok=True)
        with open(found_file, 'w') as of:
            for found in all_found:
                of.write(f'{found}\n')

    novelty = 1-len(all_found)/len(samples)
    print("Found in dataset:", len(all_found), "/", len(samples), " ; novelty =", novelty)

    res = {'Novelty': novelty}

    write_json(res, output_file)
