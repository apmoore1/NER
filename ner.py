import json
from pathlib import Path
import tempfile

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


def predict(cuda_device: int, char_encoder: str, data_dir: Path,
            glove_path: Path):
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
    cnn_filters = 30
    cnn_output_dim = len(cnn_window_size) * cnn_filters

    lstm_char_dim = 25
    lstm_char_output_dim = lstm_char_dim * 2

    word_embedding_dim = 100
    # LSTM size is that of Ma and Hovy
    lstm_dim = 200

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

    with tempfile.TemporaryDirectory(dir=Path('.')) as temp_dir:
        trainer = Trainer(model=model, grad_clipping=5.0, 
                        learning_rate_scheduler=schedule,
                        serialization_dir=temp_dir,
                        optimizer=optimizer,
                        iterator=iterator,
                        train_dataset=train_dataset,
                        validation_dataset=dev_dataset,
                        shuffle=True,
                        cuda_device=cuda_device,
                        patience=3,
                        num_epochs=1000)

        #trainer._tensorboard = TensorboardWriter(train_log=train_log, 
        #                                        validation_log=validation_log)
        interesting_metrics = trainer.train()
        best_model_weights = Path(temp_dir, 'best.th')
        best_model_state = torch.load(best_model_weights)
        model.load_state_dict(best_model_state)
        test_result = evaluate(model, test_dataset, iterator, cuda_device)
        dev_result = evaluate(model, dev_dataset, iterator, cuda_device)
        test_f1 = test_result['f1-measure-overall']
        dev_f1 = dev_result['f1-measure-overall']
        result_data.append((dev_f1, test_f1))

    with result_fp.open('w+') as json_file:
        json.dump(result_data, json_file)
    #train_log.close()
    #validation_log.close()

    #print('done')
    #print('finish')
    #print(f'{interesting_metrics}')
    #print(f'{time.time() - t}')



import argparse

if __name__ == '__main__':
    glove_fp = Path('/home/andrew/glove.6B/glove.6B.100d.txt')
    data_dir = Path('/', 'home', 'andrew', 'Documents', 'conll_2003')
    parser = argparse.ArgumentParser()
    parser.parse_args()
    predict(-1, 'lstm', data_dir, glove_fp)
    print('done')