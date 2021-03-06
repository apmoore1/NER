'''
Train set should be of size:
14041
Dev:
3250
Test:
3453
'''
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
from sklearn.model_selection import train_test_split

def read_data(data_fp: Path) -> List[Tuple[str, str]]:
    '''
    Parses the CoNLL formatted file at the given file 
    path and turns the file into a A list of tuples 
    where the first String is the sentence and the 
    second are the related word level labels.

    :param data_fp: Path to a CoNLL formatted file.
    :returns: A list of tuples where the first String 
              is the sentence and the second are the 
              related word level labels
    '''
    all_sentence_data = []
    with data_fp.open('r') as ner_data:
        sentence_words = []
        sentence_ner_labels = []
        for line in ner_data:
            line = line.strip()
            if not line:
                num_words = len(sentence_words)
                if num_words:
                    num_labels = len(sentence_ner_labels)
                    if num_words != num_labels:
                        raise ValueError(f'The number of words {num_words} '
                                         'should equal the number of labels '
                                         f'{num_labels}')
                    words = ' '.join(sentence_words)
                    labels = ' '.join(sentence_ner_labels)
                    all_sentence_data.append((words, labels))

                    sentence_words = []
                    sentence_ner_labels = []
                continue
            if line == '-DOCSTART- -X- -X- O':
                continue
            
            word, _, _, ner_label = line.split()
            sentence_words.append(word)
            sentence_ner_labels.append(ner_label)
    return all_sentence_data

def num_unique_labels(sentence_label_data: List[Tuple[str, str]]
                      ) -> int:
    '''
    Given the output of the :py:func:`read_data` function it returns 
    the number of unique labels within the data.

    :param sentence_label_data: A list of tuples where the first 
                                String is the sentence and the 
                                second are the related word level 
                                labels
    :returns: The number of unique labels.
    '''
    labels = set()
    for sentence, joined_labels in sentence_label_data:
        split_labels = joined_labels.split()
        for label in split_labels:
            labels.add(label)
    return len(labels)

def shuffle_data(sentence_label_data: List[Tuple[str, str]],
                 test_size: float = 0.2, dev_size: float = 0.2,
                 random_state: 'np.random.RandomState' = np.random.RandomState()
                 ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    '''
    Given the output of :py:func:`read_data` it will split the data 
    into random train, dev, test splits.

    :param sentence_label_data: A list of tuples where the first 
                                String is the sentence and the 
                                second are the related word level 
                                labels
    :param test_size: The size of the test data in relation to the 
                      whole dataset
    :param dev_size: The size of the dev data in relation to the 
                     data after it has been split into train and test
                     and this would come from the train set not the 
                     test.
    :param random_state: Whether or not the process should be repeatable
                         if so specify a number e.g. 42 else leave to 
                         default.
    :returns: A Tuple of (train, dev, test) where the data is in the 
              same format as the input.
    '''

    num_classes = num_unique_labels(sentence_label_data)
        
    train, test = train_test_split(sentence_label_data, 
                                   test_size=test_size,
                                   random_state=random_state)
    train, dev = train_test_split(train, test_size=dev_size,
                                  random_state=random_state)
    for dataset in [train, dev, test]:
        if num_unique_labels(dataset) != num_classes:
            return shuffle_data(sentence_label_data, test_size, 
                                dev_size, random_state)
    return train, dev, test

def data_to_file(sentence_label_data: List[Tuple[str, str]],
                 data_fp: Path) -> None:
    '''
    Write the data to the given file in CoNLL 2003 format.

    :param sentence_label_data: A list of tuples where the first 
                                String is the sentence and the 
                                second are the related word level 
                                labels
    :param data_fp: File to write the data to.
    :returns: Nothing
    '''
    with data_fp.open('w+') as data_file:
        for index, sentence_joined_labels in enumerate(sentence_label_data):
            sentence, joined_labels = sentence_joined_labels
            words = sentence.split()
            labels = joined_labels.split()

            num_words = len(words)
            num_labels = len(labels)
            if num_words != num_labels:
                raise ValueError(f'The number of words {num_words} should be the '
                                 f'same as the number of labels {num_labels}')
            for word_index, word in enumerate(words):
                label = labels[word_index]
                if index == 0 and word_index == 0:
                    data_file.write(f'{word} -X- -X- {label}')
                else:
                    data_file.write(f'\n{word} -X- -X- {label}')
            data_file.write('\n')

def create_n_dataset_folders(data_dir: Path, n_copies: int,
                             copy_dir: Path, **shuffle_kwargs
                             ) -> None:
    '''
    Creates *n* number of dataset folders based on the CoNLL 
    format within the given `copy_dir` within seperate folders 
    named 0-*n*.

    The setup of these dataset folders are the same as the 
    given data_dir which should contain train.txt, dev.txt 
    and a test.txt files. The difference being that these 
    copy directories will copy the data by re-shuffling 
    into different train, dev, test splits where the 
    shuffling is controlled by the :py:func:`shuffle_data`
    function and the keywords arguments to this function 
    are input to the :py:func:`shuffle_data` function.

    :param data_dir: The original data directory which 
                     the data is copied from.
    :param n_copies: Number of times the directory is 
                     copied.
    :param copy_dir: Root directory to store all the 
                     copy directories.
    :param shuffle_kwargs: Keyword arguments to pass 
                           to :py:func:`shuffle_data`
                           which shuffles the data 
                           before copying.
    :returns: Nothing.
    '''
    copy_dir.mkdir(parents=True, exist_ok=True)
    train_data = read_data(Path(data_dir, 'train.txt'))
    dev_data = read_data(Path(data_dir, 'dev.txt'))
    test_data = read_data(Path(data_dir, 'test.txt'))
    all_data = train_data + dev_data + test_data
    for index in range(n_copies):
        all_temp_data = shuffle_data(all_data, **shuffle_kwargs)
        file_names = ['train.txt', 'dev.txt', 'test.txt']

        new_data_dir = Path(copy_dir, f'{index}')
        new_data_dir.mkdir(parents=True, exist_ok=True)

        for file_name, temp_data in zip(file_names, all_temp_data):
            data_to_file(temp_data, Path(new_data_dir, file_name))

def test_n_dataset_folders(copy_dir: Path
                           ) -> List[Tuple[Path, Path]]:
    '''
    Given the directory that the data was copied *n* times 
    it will test each created copy directory with each 
    copy directory and return a List of a Tuples that 
    contain identical data directories if any.

    :param copy_dir: Root directory to that stores all of 
                      the copy directories.
    :returns: A List of Tuples of two Paths which are 
              Paths to two identical directories that 
              are different copy directories. If the
              List is empty then none of the copy 
              directories are the same.
    '''
    def compare_data(dir_iter, current_dir
                     ) -> List[Tuple[Path, Path]]:
        same_dirs = []
        for other_copy_dir in dir_iter:
            if other_copy_dir == current_dir:
                continue
            if not other_copy_dir.is_dir():
                continue
            if not re.match('\d+', other_copy_dir.name):
                continue
            print(f'Comparing to {other_copy_dir.name}')

            file_names = ['train.txt', 'dev.txt', 'test.txt']
            for file_name in file_names:
                data = read_data(Path(current_dir, file_name))
                data_1 = read_data(Path(other_copy_dir, file_name))
                if data == data_1:
                    same_dirs.append((current_dir, other_copy_dir))
                    break
        return same_dirs
                    
    identical_dirs = []
    for progress_index, data_copy_dir in enumerate(copy_dir.iterdir()):
        print(f'in {data_copy_dir.name}')
        if not data_copy_dir.is_dir():
            continue
        if not re.match('\d+', data_copy_dir.name):
            continue

        identical_dirs.extend(compare_data(copy_dir.iterdir(), 
                                           data_copy_dir))
        print(f'Progress: {identical_dirs}')
        print(f'Number done: {progress_index}')
    return identical_dirs
        

import argparse

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_dir_help = "The directory that stores the train, dev, and test "\
                    "files for the CoNLL 2003 dataset"
    copy_dir_help = "Path to store all of the copy directories within"
    num_copies_help = "The number of copy directories to create"

    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    parser.add_argument("copy_dir", help=copy_dir_help, type=parse_path)
    parser.add_argument("num_copies", help=num_copies_help, type=int)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    copy_dir = args.copy_dir
    num_copy_directories = args.num_copies

    train_data = read_data(Path(data_dir, 'train.txt'))
    assert len(train_data) == 14041
    dev_data = read_data(Path(data_dir, 'dev.txt'))
    assert len(dev_data) == 3250
    test_data = read_data(Path(data_dir, 'test.txt'))
    assert len(test_data) == 3453
    all_data = train_data + dev_data + test_data
    assert len(all_data) == (14041 + 3250 + 3453)

    create_n_dataset_folders(data_dir, num_copy_directories, copy_dir)
    num_identical_copies = len(test_n_dataset_folders(copy_dir))
    print(f'Created {num_copy_directories} and {num_identical_copies} are '
          'identical')


