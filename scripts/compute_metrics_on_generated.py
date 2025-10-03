from pathlib import Path
import argparse
import pandas as pd
import torch

from chembed.utils import write_json
from chembed.evaluate import evaluate_generated


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--write_invalid_to", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    input_file = args.generated
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    df = pd.read_csv(input_file)

    selfies = df['selfies'].tolist()

    res = evaluate_generated(selfies, args.device, test_novelty=False, train_file=None, write_invalid_to=args.write_invalid_to)

    output_file = args.output
    if output_file is None:
        output_file = input_file.parent/f"{input_file.stem}_metrics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(res)
    
    write_json(res, output_file)
