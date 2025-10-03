from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    input_file = args.input_file 
    df = pd.read_parquet(input_file)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['stratum'], random_state=42)

    output_dir = input_file.parent
    train_df.to_parquet(output_dir/f'{input_file.stem}_train.parquet', index=False)
    val_df.to_parquet(output_dir/f'{input_file.stem}_validation.parquet', index=False)
