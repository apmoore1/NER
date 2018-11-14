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


def predict():
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

    word_embedding_dim = 100
    #total_embedding_dim = word_embedding_dim + cnn_output_dim
    total_embedding_dim = word_embedding_dim + (lstm_char_dim * 2)
    # LSTM size is that of Ma and Hovy
    lstm_dim = 200

    cuda_device = -1

    # Dropout applies dropout after the encoded text and after the word embedding.

    data_dir = Path('/', 'home', 'andrew', 'Documents', 'conll_2003')

    tensorboard_dir = Path('..', 'tensorboard ner')
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_log = SummaryWriter(Path(tensorboard_dir, "log", "train"))
    validation_log = SummaryWriter(Path(tensorboard_dir, "log", "validation"))

    train_fp = Path(data_dir, 'train.txt')
    dev_fp = Path(data_dir, 'dev.txt')
    test_fp = Path(data_dir, 'test.txt')

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

    character_lstm = torch.nn.LSTM(char_embedding_dim, lstm_char_dim, 
                                batch_first=True, bidirectional=True)
    character_lstm_wrapper = PytorchSeq2VecWrapper(character_lstm)

    character_cnn = CnnEncoder(embedding_dim=char_embedding_dim, 
                            num_filters=cnn_filters, 
                            ngram_filter_sizes=cnn_window_size, 
                            output_dim=cnn_output_dim)
    #token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, 
    #                                                 encoder=character_cnn)
    token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, 
                                                    encoder=character_lstm_wrapper)

    glove_fp = cached_path('/home/andrew/glove.6B/glove.6B.100d.txt')
    glove_100_weights = _read_pretrained_embeddings_file(glove_fp, 
                                                        word_embedding_dim, 
                                                        vocab, 'tokens')
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=word_embedding_dim,
                                weight=glove_100_weights)

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding,
                                            "chars": token_character_encoder})


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

        trainer._tensorboard = TensorboardWriter(train_log=train_log, 
                                                validation_log=validation_log)
        interesting_metrics = trainer.train()
        best_model_weights = Path(temp_dir, 'best.th')
        best_model_state = torch.load(best_model_weights)
        model.load_state_dict(best_model_state)
        test_result = evaluate(model, test_dataset, iterator, cuda_device)
        dev_result = evaluate(model, dev_dataset, iterator, cuda_device)
        test_f1 = test_result['f1-measure-overall']
        dev_f1 = dev_result['f1-measure-overall']

    train_log.close()
    validation_log.close()

    print('done')
    print('finish')
    print(f'{interesting_metrics}')
    print(f'{time.time() - t}')

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    predict()