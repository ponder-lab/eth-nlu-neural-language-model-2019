
# Data Paths
PATH_TRAIN = "../data/sentences/sentences.train"
PATH_VALID = "../data/sentences/sentences.eval"
PATH_CONTINUATION = "../data/sentences/sentences.continuation"
PATH_TEST = "../data/sentences/sentences_test.txt"
PATH_VOCAB = "../data/vocab.txt"
PATH_EXTERNAL_EMBEDDING = "../data/wordembeddings-dim100.word2vec"
PATH_EMBEDDING_MATRIX = "../data/embedding_matrix.npy"

OUTPUT_DIR = "../out"

SPECIAL = {
    "bos" : "<bos>",
    "eos" : "<eos>",
    "pad" : "<pad>"
}

SENTENCE_LENGTH = 30
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 512
VOCAB_SIZE = 20000

EMBEDDING_SIZE = 100
LSTM_HIDDEN_STATE_SIZE_A = 512
LSTM_HIDDEN_STATE_SIZE_B = 1024
LSTM_OUTPUT_SIZE = 512

EPOCHS = 5
GRADIENT_CLIPPING_NORM = 5

SUMMARY_FREQ = 500
