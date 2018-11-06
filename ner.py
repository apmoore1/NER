from pathlib import Path

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.data.token_indexers import TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary

from allennlp.models import CrfTagger

#
# The dataset we are using has already been formatted from IOB1 to BIO
# When reading the dataset state the coding is the orignal as this will not  
# affect the labels i.e. the labels and schema is not checked.

label_encoding = 'BIO'
constrain_crf_decoding = True

# Dropout applies dropout after the encoded text and after the word embedding.

data_dir = Path('/', 'home', 'andrew', 'Documents', 'conll_2003')

train_fp = Path(data_dir, 'train.txt')
dev_fp = Path(data_dir, 'dev.txt')
test_fp = Path(data_dir, 'test.txt')

indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens', 
                                           lowercase_tokens=True),
            'chars': TokenCharactersIndexer(namespace='token_characters')}

conll_reader = Conll2003DatasetReader(token_indexers=indexers, coding_scheme='BIOUL')
train_dataset = conll_reader.read(cached_path(train_fp))

vocab = Vocabulary.from_instances(train_dataset)

char_embedding = Embedding(num_embeddings=vocab.get_vocab_size("token_characters"), 
                           embedding_dim=16)
character_cnn = CnnEncoder(embedding_dim=16, num_filters=30, 
                           output_dim=50)
token_character_encoder = TokenCharactersEncoder(embedding=char_embedding, 
                                                 encoder=character_cnn)

glove_fp = cached_path('/home/andrew/glove.6B/glove.6B.100d.txt')
glove_100_weights = _read_pretrained_embeddings_file(glove_fp, 100, vocab, 'tokens')
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=100,
                            weight=glove_100_weights)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding,
                                          "chars": token_character_encoder})
print('done')
print('finish')