'''
Code that loads the pretrained embedding
'''

# Standard packages
import tensorflow as tf
import numpy as np
import pandas as pd 
from gensim import models

def load_embedding(vocab, path, dim_embedding, vocab_size):
    '''
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''

    print("Loading external embeddings from %s" % path)
    
    print(f'type of vocab: {type(vocab)}')

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)  
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0
    
    vocab = open('../data/vocab.txt', 'r')
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

    np.save('../data/embedding_matrix.npy',emb)
    #can be removed if np.save works
    pd.DataFrame(emb).to_csv('../data/embedding_matrix.csv', header=False, index=False)

    return emb
