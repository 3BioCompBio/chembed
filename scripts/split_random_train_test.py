from pathlib import Path
import argparse

from sklearn.model_selection import train_test_split

from chembed.utils import read_df, write_df

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_train", type=Path, default=None)
    parser.add_argument("--output_test", type=Path, default=None)
    args = parser.parse_args()

    input_file = args.input_file

    output_train = args.output_train
    if output_train is None:
        output_train = input_file.parent/f'{input_file.stem}_train{input_file.suffix}'
    output_train.parent.mkdir(parents=True, exist_ok=True)

    output_test = args.output_test
    if output_test is None:
        output_test = input_file.parent/f'{input_file.stem}_test{input_file.suffix}'
    output_test.parent.mkdir(parents=True, exist_ok=True)

    df = read_df(input_file)

    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=args.random_seed, shuffle=True) 
    
    write_df(df_train, output_train)
    write_df(df_test, output_test)
