# Standard packages
import numpy as np
import pandas as pd
import os

PATH_TRAIN = "../data/sentences/sentences.train"
PATH_VALID = "../data/sentences/sentences.eval"
PATH_CONTINUATION = "../data/sentences/sentences.continuation"
PATH_TEST = "../data/sentences/sentences_test.txt"
PATH_VOCAB = "../data/vocab.txt"
PATH_SUBMISSION = "../data/submission/"
PATH_EMBEDDING = "../data/wordembeddings-dim100.word2vec"
PATH_LOG = '../logs'
PATH_CHECKPOINTS = '../tf_ckpts'

SPECIAL = {
    "bos" : "<bos>",
    "eos" : "<eos>",
    "pad" : "<pad>"
}

SENTENCE_LENGTH = 30
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 512
VOCAB_SIZE = 20000


WORD_EMBEDDINGS = np.array(pd.read_csv('../data/embedding_matrix.csv', header=None).values.tolist()) # <class 'numpy.ndarray'>


# TODO load pretrained word embeddings and check that it's working in combination with tf.function
EMBEDDING_SIZE = 100
LSTM_HIDDEN_STATE_SIZE_A = 512
LSTM_HIDDEN_STATE_SIZE_B = 1024
LSTM_OUTPUT_SIZE = 512

EPOCHS = 5
GRADIENT_CLIPPING_NORM = 5

SUMMARY_FREQ = 200
