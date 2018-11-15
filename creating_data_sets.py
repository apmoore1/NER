'''
I need to copy the function for number of classes and shuffling from the 
previous project:
https://github.com/apmoore1/bilstm-cnn-crf-ner/blob/master/neural_ner/neuralnets/BiLSTM.py

These need to be adapted. From the looks of things we cannot use the strattified
approach but just use random shuffling and then check if there are the same 
number of classes.

We are going to have to convert the data at first in to (sentence: NER labels) 
and then shuffle and check and then convert back into text files. We want to 
make sure that this is completely random.

Train set should be of size:
14041
Dev:
3250
Test:
3453
'''
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

def read_data(data_fp: Path) -> List[Tuple[str, str]]:
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
            
if __name__ == '__main__':
    data_dir = Path('..', 'conll_2003')
    train_data = read_data(Path(data_dir, 'train.txt'))
    assert len(train_data) == 14041
    dev_data = read_data(Path(data_dir, 'dev.txt'))
    assert len(dev_data) == 3250
    test_data = read_data(Path(data_dir, 'test.txt'))
    assert len(test_data) == 3453
    all_data = train_data + dev_data + test_data
    assert len(all_data) == (14041 + 3250 + 3453)
    train, dev, test = shuffle_data(all_data)
    print(f'{len(train)} {len(dev)} {len(test)}')
    print(train[0])
    print()
    print(dev[0])
    print()
    print(test[0])
    data_to_file(train, Path(data_dir, 'new_train.txt'))
    data_to_file(dev, Path(data_dir, 'new_dev.txt'))
    data_to_file(test, Path(data_dir, 'new_test.txt'))
    print('done')


