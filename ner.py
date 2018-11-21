import argparse
import json
from pathlib import Path
import tempfile
from typing import List, Tuple
import random
import shutil

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.seq2vec_encoders import CnnEncoder, PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models import CrfTagger
from allennlp.training.trainer import Trainer, TensorboardWriter
from allennlp.commands.evaluate import evaluate
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper
from tensorboardX import SummaryWriter
import torch
from torch import optim
import numpy as np


def set_random_env(cuda: int, random_seed: int, numpy_seed: int, 
                   torch_seed: int):
    '''

    Reference:
    https://github.com/allenai/allennlp/blob/master/allennlp/common/util.py#L178-L207
    '''
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if cuda == 0:
        torch.cuda.manual_seed_all(torch_seed)

def predict(cuda_device: int, char_encoder: str, data_dir: Path,
            glove_path: Path, temp_dir: Path, random_seed: int = 13370, 
            numpy_seed: int = 1337, torch_seed: int = 133
            ) -> List[Tuple[float, float, str]]:
    '''
    This allows you to train an NER model that has either a CNN character 
    encoder or LSTM based on the `char_encoder` argument. The encoded 
    characters are then combined with 100D Glove vectors and put through 
    a Bi-Directional LSTM.

    This is based on the following two papers:
    
    1. CNN character encoder version `Ma and Hovy \
       <https://arxiv.org/abs/1603.01354>`_
    2. LSTM character encoder version `Lample et al. \
       <https://arxiv.org/abs/1603.01360>`_

    :param cuda_device: Whether to use GPU or CPU, CPU = -1, GPU = 0
    :param char_encoder: Whether to use an LSTM or CNN. Acceptable values are: 
                         1. lstm, 2. cnn
    :param data_dir: A file path to a directory that contains three files: 
                     1. train.txt, 2. dev.txt, 3. test.txt that are the 
                     train, dev, and test files respectively in CONLL 2003 
                     format where the NER labels are in BIO format.
    :param glove_path: A file path to the `Glove 6 billion word vectors 100D \
                       <https://nlp.stanford.edu/projects/glove/>`_
    :returns: The results as a list of tuples which are 
              (dev f1 score, test f1 score, char encoder) where the list 
              represents a different trained model using the same train, dev, 
              and test split but different random seed.
    '''
    #
    # The dataset we are using has already been formatted from IOB1 to BIO
    # When reading the dataset state the coding is the orignal as this will not  
    # affect the labels i.e. the labels and schema is not checked.

    label_encoding = 'BIO'
    constrain_crf_decoding = True
    dropout = 0.5

    char_embedding_dim = 30
    cnn_window_size = (3,)
    cnn_filters = 50
    cnn_output_dim = len(cnn_window_size) * cnn_filters

    lstm_char_dim = 25
    lstm_char_output_dim = lstm_char_dim * 2

    word_embedding_dim = 100
    # LSTM size is that of Ma and Hovy
    lstm_dim = 100

    # Dropout applies dropout after the encoded text and after the word embedding.

    

    #tensorboard_dir = Path('..', 'tensorboard ner')
    #tensorboard_dir.mkdir(parents=True, exist_ok=True)

    #train_log = SummaryWriter(Path(tensorboard_dir, "log", "train"))
    #validation_log = SummaryWriter(Path(tensorboard_dir, "log", "validation"))

    train_fp = Path(data_dir, 'train.txt')
    dev_fp = Path(data_dir, 'dev.txt')
    test_fp = Path(data_dir, 'test.txt')
    result_fp = Path(data_dir, 'results.json')
    result_data = []
    if result_fp.exists():
        with result_fp.open('r') as json_file:
            result_data = json.load(json_file)

    indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', 
                                            lowercase_tokens=True),
                'chars': TokenCharactersIndexer(namespace='token_characters')}

    conll_reader = Conll2003DatasetReader(token_indexers=indexers)
    train_dataset = conll_reader.read(cached_path(train_fp))
    dev_dataset = conll_reader.read(cached_path(dev_fp))
    test_dataset = conll_reader.read(cached_path(test_fp))

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset + test_dataset)

    char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_characters"), 
                            embedding_dim=char_embedding_dim)

    if char_encoder.strip().lower() == 'lstm':
        character_lstm = torch.nn.LSTM(char_embedding_dim, lstm_char_dim, 
                                    batch_first=True, bidirectional=True)
        character_lstm_wrapper = PytorchSeq2VecWrapper(character_lstm)
        token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, 
                                                         encoder=character_lstm_wrapper)
        total_char_embedding_dim = lstm_char_output_dim
    elif char_encoder.strip().lower() == 'cnn':
        character_cnn = CnnEncoder(embedding_dim=char_embedding_dim, 
                                   num_filters=cnn_filters, 
                                   ngram_filter_sizes=cnn_window_size, 
                                   output_dim=cnn_output_dim)
        token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, 
                                                         encoder=character_cnn)
        total_char_embedding_dim = cnn_output_dim
    else:
        raise ValueError('The Character encoder can only be `lstm` or `cnn` '
                         f'and not {char_encoder}')

    glove_path = cached_path(glove_path)
    glove_100_weights = _read_pretrained_embeddings_file(glove_path, 
                                                         word_embedding_dim, 
                                                         vocab, 'tokens')
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=word_embedding_dim,
                                weight=glove_100_weights)

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding,
                                            "chars": token_character_encoder})

    total_embedding_dim = word_embedding_dim + total_char_embedding_dim
    lstm = torch.nn.LSTM(total_embedding_dim, lstm_dim, batch_first=True, 
                         bidirectional=True)
    lstm_wrapper = PytorchSeq2SeqWrapper(lstm)


    model = CrfTagger(vocab, word_embeddings, lstm_wrapper, 
                    label_encoding=label_encoding, dropout=dropout, 
                    constrain_crf_decoding=constrain_crf_decoding)

    optimizer = optim.SGD(model.parameters(), lr=0.015, weight_decay=1e-8)
    schedule = LearningRateWithoutMetricsWrapper(torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9524))
    iterator = BucketIterator(batch_size=64, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    temp_dir_fp = str(temp_dir.resolve())
    temp_folder_path = tempfile.mkdtemp(dir=temp_dir_fp)
    
    set_random_env(cuda_device, random_seed, numpy_seed, torch_seed)
    trainer = Trainer(model=model, grad_clipping=5.0, 
                    learning_rate_scheduler=schedule,
                    serialization_dir=temp_folder_path,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=train_dataset,
                    validation_dataset=dev_dataset,
                    shuffle=True,
                    cuda_device=cuda_device,
                    patience=5,
                    num_epochs=1000)

    #trainer._tensorboard = TensorboardWriter(train_log=train_log, 
    #                                        validation_log=validation_log)
    interesting_metrics = trainer.train()
    best_model_weights = Path(temp_folder_path, 'best.th')
    best_model_state = torch.load(best_model_weights)
    model.load_state_dict(best_model_state)
    test_result = evaluate(model, test_dataset, iterator, cuda_device)
    dev_result = evaluate(model, dev_dataset, iterator, cuda_device)
    test_f1 = test_result['f1-measure-overall']
    dev_f1 = dev_result['f1-measure-overall']
    result_data.append((dev_f1, test_f1, char_encoder))

    with result_fp.open('w+') as json_file:
        json.dump(result_data, json_file)
    print(f'{interesting_metrics}')
    return result_data
    #train_log.close()
    #validation_log.close()

    #print('done')
    #print('finish')
    
    #print(f'{time.time() - t}')


def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    data_dir_help = "The directory that stores the train, dev, and test "\
                    "files for the CoNLL 2003 dataset"
    glove_help = "The path to the 100 dimension Glove Embedding"
    num_runs_help = "Number of times to train and predict on the dev and test"\
                    " set. Each run will be with a different random seed."

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help=data_dir_help, type=parse_path)
    parser.add_argument("glove_file", help=glove_help, type=parse_path)
    parser.add_argument("char_encoder", help="Character encoder to use", 
                        type=str, choices=['lstm', 'cnn'])
    parser.add_argument("num_runs", help=num_runs_help, type=int)
    parser.add_argument("temp_dir", help='Directory to store temp directories',
                        type=parse_path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    glove_fp = args.glove_file
    char_encoder = args.char_encoder
    num_runs = args.num_runs
    temp_dir = args.temp_dir
    cuda = -1
    if args.cuda:
        cuda = 0
    
    # Ensure different seeds for each run
    random_seeds = random.sample(range(1, 100000), num_runs)
    numpy_seeds = random.sample(range(1, 100000), num_runs)
    torch_seeds = random.sample(range(1, 100000), num_runs)

    for run in range(num_runs):
        random_seed = random_seeds[run]
        numpy_seed = numpy_seeds[run]
        torch_seed = torch_seeds[run]
        result = predict(cuda, char_encoder, data_dir, glove_fp, temp_dir,
                         random_seed, numpy_seed, torch_seed)
        if args.verbose:
            print(f'Run {run} completed. Results so far:\n{result}')
    if args.verbose:
        print('Finished')