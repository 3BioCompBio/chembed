from pathlib import Path
import csv
import argparse
import selfies as sf
from collections import Counter
from chembed.utils import write_json

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=Path)
    parser.add_argument('output_file', type=Path)
    args = parser.parse_args()

    input_file = args.input_file

    token_counts = Counter()

    with open(input_file, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            token_counts.update(sf.split_selfies(row['selfies']))
    
    write_json(token_counts, args.output_file)
