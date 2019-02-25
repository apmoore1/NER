import argparse
import json
from itertools import product
from typing import List, Tuple
from pathlib import Path

import pandas as pd
import seaborn as sns

def get_encoder_results(results: List[Tuple[float, float, str]], encoder: str, 
                        data_split: str) -> List[float]:
    encoder_results = []
    data_split = data_split.strip().lower()
    encoder = encoder.lower()
    for result in results:
        val_result, test_result, result_encoder = result
        if result_encoder.lower() != encoder:
            continue
        if data_split == 'test':
            encoder_results.append(test_result)
        elif data_split == 'val':
            encoder_results.append(val_result)
        else:
            raise ValueError(f'data_split ({data_split}) has to be either val '
                             'or test.')
    return encoder_results


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    seeds_result_help = "File path to results where the models were trained "\
                        "on different random seeds"
    data_seeds_result_help = "File path to results where the models were "\
                             "trained on different data splits and random seeds"
    parser.add_argument("seeds_result_fp", help=seeds_result_help, 
                        type=parse_path)
    parser.add_argument("data_seeds_result_fp", help=data_seeds_result_help,
                        type=parse_path)
    parser.add_argument("plot_fp", help="File path to the plot",
                        type=parse_path)
    parser.add_argument("val_or_test_split", help="Create the visulisation "\
                        "from the validation or test results", type=str)
    parser.add_argument("--remove_bottom_5", help="Removes the bottom 5 results"\
                        " for each encoder for each set of results", 
                        action="store_true")
    args = parser.parse_args()

    encoders = ["CNN", "LSTM"]
    training_methods_and_results = [('Seed and Data split', args.data_seeds_result_fp),
                                    ('Seed', args.seeds_result_fp)]
    encoder_and_folder = product(encoders, training_methods_and_results)

    f1_scores = []
    encoder_names = []
    training_method_names = []
    for encoder, train_method_result in encoder_and_folder:
        train_method_name, result_fp = train_method_result
        with result_fp.open('r') as result_file:
            results = json.load(result_file)
            f1_values = get_encoder_results(results, encoder, 
                                            args.val_or_test_split)
            # Remove the bottom 5 results for each encoder on each train method
            # as some results did not converage as well.
            if args.remove_bottom_5:
                f1_values = sorted(f1_values)[5:]
            for f1_value in f1_values:
                f1_scores.append(f1_value)
                encoder_names.append(encoder)
                training_method_names.append(train_method_name)
            print(f'Number of results: {len(f1_values)} for encoder: {encoder}'
                  f' on training method: {train_method_name}')
    all_data = pd.DataFrame({'F1': f1_scores, 'Encoder': encoder_names, 
                             'Training Method': training_method_names})
    ax = sns.violinplot(y="F1", x="Training Method", hue="Encoder",
                        data=all_data, inner="quartile", split=True)
    ax.set_xlabel("")
    plot_fp: Path = args.plot_fp
    plot_fp.touch()

    fig = ax.figure
    fig.set_figwidth(5)
    fig.savefig(str(plot_fp.resolve()))



