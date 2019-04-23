'''
Code that loads the pretrained embedding
'''

# Standard packages
import tensorflow as tf
import numpy as np
import pandas as pd
from gensim import models

def load_embedding(dim_embedding, vocab_size):

    try:
        emb = np.load(PATH_EMBEDDING_MATRIX)
        assert emb.shape[0] == vocab_size
        assert emb.shape[1] == dim_embedding
        print(f"Using Cached Embedding Matrix: {PATH_EMBEDDING_MATRIX}")
    except IOError: # no embedding matrix found
        print(f"Creating New Embedding Matrix from External Embeddings: {PATH_EXTERNAL_EMBEDDING}")
        emb = load_external_embedding(path=PATH_EXTERNAL_EMBEDDING, dim_embedding=dim_embedding, vocab_size=vocab_size)

        np.save(PATH_EMBEDDING_MATRIX, emb)


    return emb


def load_external_embedding(path, dim_embedding, vocab_size):
    '''
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    vocab = open(PATH_VOCAB, 'r')
    idx = 0
    for tok in vocab:

        tok = tok[:-1]

        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        idx += 1

    external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print(f'index after loading embedding ==> {idx}')

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    emb = external_embedding

    print('embedding type: ', type(emb))
    print(f'embedding shape: {emb.shape}')

    return emb
