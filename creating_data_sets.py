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
            
if __name__ == '__main__':
    data_dir = Path('/', 'home', 'andrew', 'Documents', 'conll_2003')
    train_data = read_data(Path(data_dir, 'train.txt'))
    assert len(train_data) == 14041
    dev_data = read_data(Path(data_dir, 'dev.txt'))
    assert len(dev_data) == 3250
    test_data = read_data(Path(data_dir, 'test.txt'))
    assert len(test_data) == 3453


'''
    @classmethod
    def get_num_classes(cls, dataset: List[Dict[str, Any]]) -> int:
        '''
        Given the output of :py:func:`neural_ner.util.CoNLL.readCoNLL`
        returns the number of NER classes in the dataset.
        '''
        class_data = []
        for sentence in dataset:
            class_data.extend(sentence['NER_BIO'])
        return len(set(class_data))

    @classmethod
    def random_shuffle(cls, train: List[Dict[str, Any]], dev: List[Dict[str, Any]], 
                       test: List[Dict[str, Any]], test_size: float = 0.2, 
                       dev_size:float = 0.2, 
                       random_state: 'np.random.RandomState' = np.random.RandomState()
                       ) -> List[List[Dict[str, Any]]]:
        '''
        Combines the train, development and test data and then re-splits the 
        data into different train, development and test data.
        
        :param train: The output of :py:func:`neural_ner.util.CoNLL.readCoNLL`
        :param dev: The output of :py:func:`neural_ner.util.CoNLL.readCoNLL`
        :param test: The output of :py:func:`neural_ner.util.CoNLL.readCoNLL`
        :param test_size: Size of the test set as a fraction of all of the data
        :param dev_size: Size of the development set as a fraction of the 
                        train data (train data created after the initial data
                        has been split into train and test splits)
        :param random_state: Random state for splitting the data. Default 
                            is completely random.
        :return: A list of size 3 representing train, dev, and test data. 
                Where the data is in the same format as the original input.
        '''
        num_classes = cls.get_num_classes(train)
        all_data = train + dev + test
        train, test = train_test_split(all_data, test_size=test_size,
                                       random_state=random_state)
        train, dev = train_test_split(train, test_size=dev_size,
                                      random_state=random_state)
        for dataset in [train, dev, test]:
            if cls.get_num_classes(dataset) != num_classes:
                return cls.random_shuffle(train, dev, test, test_size, dev_size,
                                          random_state)
        return train, dev, test
'''
            
