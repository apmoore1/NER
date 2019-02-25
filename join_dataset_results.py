import argparse
import json
from pathlib import Path
import re


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir_help = "The root directory storing all of the directories that "\
                    "contain different data distributions and result files"
    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    parser.add_argument("result_file", help="File path to store all results to",
                        type=parse_path)
    args = parser.parse_args()
    data_folder_fp: Path = args.data_dir
    all_results = []
    for sub_folder in data_folder_fp.iterdir():
        sub_folder: Path
        if not sub_folder.is_dir():
            continue
        if not re.search(r'\d+', sub_folder.name):
            continue
        sub_folder_results_fp = Path(sub_folder, 'results.json').resolve()
        with sub_folder_results_fp.open('r') as results:
            all_results.append(json.load(results))
    with args.result_file.open('w+') as result_file:
        json.dump(all_results, result_file)
    

    