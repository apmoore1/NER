import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import List, Tuple, Union, Dict
import statistics

def get_encoder_results(results: List[List[Union[float, str]]], encoder: str
                        ) -> List[Tuple[float, float]]:
    encoder_results = []
    for result in results:
        validation_f1, test_f1, encoder_used = result
        if encoder_used == encoder:
            encoder_results.append((validation_f1, test_f1))
    return encoder_results

def get_stats(results: List[Tuple[float, float]], dataset: str = 'test',
              num_results: int = 5) -> Dict[str, float]:
    assert len(results) == num_results

    dataset_index = 1 if dataset == 'test' else 0
    dataset_results = [result[dataset_index] for result in results]
    max_value = max(dataset_results)
    min_value = min(dataset_results)
    median = statistics.median(dataset_results)
    mean = statistics.mean(dataset_results)
    return {'max': max_value, 'min': min_value, 'median': median, 'mean': mean}

def compare_stats(stat_result_0: Dict[str, float], 
                  stat_result_1: Dict[str, float]
                  ) -> Tuple[Dict[str, int], Dict[str, int]]:
    compare_result_0: Dict[str, int] = {}
    compare_result_1: Dict[str, int] = {}
    for field_name, field_value_0 in stat_result_0.items():
        field_value_1 = stat_result_1[field_name] 
        if field_value_0 > field_value_1:
            compare_result_0[field_name] = 1
            compare_result_1[field_name] = 0
        elif field_value_0 == field_value_1:
            raise ValueError('Cannot have the same value, result0:'
                             f' {field_value_0}\nresults1: {field_value_1}')
        else:
            compare_result_0[field_name] = 0
            compare_result_1[field_name] = 1
    return (compare_result_0, compare_result_1)

def analysis_results(result_dir: Path, encoder: str) -> Dict[str, float]:
    result_fp = Path(result_dir, 'results.json')
    with result_fp.open('r') as result_file:
        results = json.load(result_file)

    encoder_results = get_encoder_results(results, encoder)
    return get_stats(encoder_results)

def compare_results(result_dir: Path, encoders: Tuple[str, str],
                    encoder_stats: Dict[str, Dict[str, int]]
                    ) -> Dict[str, Dict[str, int]]:
    assert len(encoders) == 2

    enocder_0_results = analysis_results(result_dir, encoders[0])
    enocder_1_results = analysis_results(result_dir, encoders[1])

    comp_results = compare_stats(enocder_0_results, enocder_1_results)
    for index, comp_result in enumerate(comp_results):
        encoder = encoders[index]
        for field_name, is_better in comp_result.items():
            encoder_stats[encoder][field_name] += is_better
    return encoder_stats


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir_help = "The root directory storing all of the copy directories"
    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    args = parser.parse_args()

    encoder_stats = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(0, 200):
        copy_dir = Path(args.data_dir, f'{i}')
        for copy_dir_files in copy_dir.iterdir():
            file_name = copy_dir_files.name
            if file_name == 'results.json':
                break
        else:
            continue
        encoders = ('lstm', 'cnn')
        compare_results(copy_dir, encoders, encoder_stats)
    print(encoder_stats)
    