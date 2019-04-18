'''
Natural Language Understanding 

Project 1: Neural Language Model
Task 1: RNN Language Modelling

Definition of the RNN Language Model class, the LanguageModellError class 
used for the computation of the loss and the perplexity class used as a metric. 

Authors: Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber
Date: 14.04.2019
Version: 2
'''
# Standard packages
import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import Model
import sys
from collections import Counter

# Local modules
from util import * #@UnusedWildImport pylint: disable=unused-wildcard-import
from global_variable import * #@UnusedWildImport pylint: disable=unused-wildcard-import
from embedding import load_embedding

class LanguageModel(Model):
    def __init__(self, vocab_size, sentence_length, embedding_size, hidden_state_size, output_size, batch_size, word_embeddings=None , index_to_word_table=None):
        super(LanguageModel, self).__init__()
        
  
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.embedding_size = embedding_size
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        if word_embeddings is not None:
            # TODO this does not work yet with @tf.function
            
            #word_embeddings = load_embedding(vocab=index_to_word_table, path=PATH_EMBEDDING, dim_embedding=100, vocab_size=VOCAB_SIZE)
            #emb = np.array(pd.read_csv('../data/embedding_matrix.csv', header=None).values.tolist()) # <class 'numpy.ndarray'>
            #added numpy save (still has to be tested)
            #emb = np.load('../data/embedding_matrix.npy')
            #weights = [emb] # <class 'list'>
            init = tf.constant_initializer(word_embeddings)
            #print(f'type of weights: {type(weights)}')
            trainable = True
        else:
            init = tf.initializers.GlorotUniform()
            trainable = True
        
        
        self.embedding = tf.keras.layers.Embedding(
            
            input_dim= vocab_size,
            output_dim= embedding_size,
            input_length= sentence_length,
            embeddings_initializer = init,
            trainable = trainable,
            
            #embeddings_initializer='uniform', # TODO [nku] ?
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
        )
        
        self.lstm_cell = tf.keras.layers.LSTMCell(
            # dimensionality of the output space
            units=hidden_state_size,
            kernel_initializer='glorot_uniform', # xavier initializer
            name="lstm_cell",
            
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            implementation=1
        )

        # hidden state dimension to vocab size dimension
        if output_size != hidden_state_size: 
    
            self.projection_layer  = tf.keras.layers.Dense(
                output_size,
                input_shape=(None, output_size), 
                activation=None,
                use_bias=True,
                kernel_initializer='glorot_uniform', # xavier initializer
                name="Wp",

                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )
            
            
        self.softmax_layer  = tf.keras.layers.Dense(
                vocab_size,
                input_shape=(None, output_size), 
                activation=None,
                use_bias=False,
                kernel_initializer='glorot_uniform', # xavier initializer
                name="W",

                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )

    #@tf.function 
    def call(self, sentence_id_batch):
        
        print(f"Language Model Call: Input={sentence_id_batch.shape}")
        
        # TODO: check if this is really how a static version of the forward pass is done?
        
        # initialize lstm state
        init_state = tf.zeros([self.batch_size, self.hidden_state_size])
        state = (init_state, init_state)
        
        logits  = []
        
        # embedding layer -> gets a sentence as input and performs the embedding
        sentence_embedding_batch = self.embedding(sentence_id_batch)
        print(f'sentence embedding batch shape: {sentence_embedding_batch.shape}')
        
        for pos in range(self.sentence_length-1):
            
            # extract word -> identity returns a tensor of the same dimension
            # dimensions: [batch, sentence length, embedding size]
            # selects a slice of the cube -> every embedding, every sentence in the batch, but always
            # one certain position
            word_embedding_batch =  tf.identity(sentence_embedding_batch[:, pos, :], name=f"word_{pos}")
            
            # output \in [batch_size, hidden_state_size]
            # state  \in [batch_size, hidden_state_size]
            # lstm cell has two outputs: one for the prediction of the next word (y_t) and
            # one for the state (h_t)
            output, state = self.lstm_cell(word_embedding_batch, state)
            
            # project y_t down to output size |vocab|
            if self.output_size != self.hidden_state_size:
                output = self.projection_layer(output) # \in [batch_size, output_size]
            
            # apply softmax weights to obtain logits
            output = self.softmax_layer(output) # \in [batch_size, vocab_size]
            
            logits.append(output)
        
        # \in [batch_size, sentence_length-1, vocab_size]
        logits = tf.stack(logits, axis=1)
        #print(f'logits: {logits}')
        
        # \in [batch_size, sentence_length-1, vocab_size]
        preds = tf.nn.softmax(logits, name=None)
        #print(f'pred: {preds}')
        
        #print(f"logits shape = {logits.shape}") 
        #print(f"preds shape = {preds.shape}") 
        
        return logits, preds  
    
class LanguageModelError(tf.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred must be logits        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        
        # average over batch and sentence length
        # y_pred \in [64, 29, 20'000]
        # y_true \in [64, 29]
        # sparse softmax takes these inputs with different dimensions
        # loss \in [64, 29] -> for every word in every sentence in every batch
        # we compute the loss
        # math.reduce_mean sums up the entire matrix and divides by #elements
        loss = tf.math.reduce_mean(loss)
        
        return loss

class Perplexity(tf.metrics.Metric):
    ##TODO test if it is working
    def __init__(self, name='perplexity', **kwargs):
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(dtype=tf.float32,name='perp_sum', initializer='zeros')
        self.n = self.add_weight(dtype=tf.int32,name='perp_n', initializer='zeros')
        
    def update_state_sample(self,sentence_true, sentence_pred):
        '''
        Assumes that at most one sentence get added at a time (otherwise the filtering out
        of <pad> would not word as planned) and adds up the perplexity of the predicted
        sentence. 

        Arguments: 
            - sentence_true = actual sentence -> sentence_true in \[64, 29]
            - sentence_pred = predicted sentence (output distribution of the model) -> 
              sentence_pred in \[64, 29, 20000] = \[batch_size, sentence_length, vocab_size]
        '''
        shape = sentence_pred.shape
        for i in range(shape[0]):
            true = sentence_true[i]
            pred = sentence_pred[i]
            if (true==tf.constant(2,dtype=tf.int64)): ##Hardcoded to find <pad> (2)symbol wich is not part of the perplexity calculation
                break
            self.n.assign_add(tf.constant(1))
            p = pred[true] #pick right probability from probability vector

            #because I did not find log a logarithm with base 2
            log = tf.math.divide(tf.math.log(p),tf.math.log(tf.constant(2.,dtype=tf.float32)))
            self.sum.assign_add(log)
        return

    def update_state(self,sentence_true, sentence_pred, sample_weight=None):
        '''
        Depending on whether the input is a batch or a single sentence, calls the helper
        function for the respective input. Can handle batch sizes

        Arguments: 
            - sentence_true = actual sentence -> sentence_true in \[64, 29]
            - sentence_pred = predicted sentence (output distribution of the model) -> 
              sentence_pred in \[64, 29, 20000] = \[batch_size, sentence_length, vocab_size]
        '''
        shape = sentence_pred.shape
        if(len(shape)==3):#batch mode
            for i in range(shape[0]):
                self.update_state_sample(sentence_true=sentence_true[i,:], sentence_pred=sentence_pred[i,:,:])
                #sys.stdout.write("\r %d" % i) #progress 
        else:
            self.update_state_sample(sentence_true=sentence_true, sentence_pred=sentence_pred)
        

    def result(self):
        '''
        Computes the actual perplexity value. 
        
        Returns: 
            - res: -1 if result gets called without updating the state first,
                   otherwise the summed up perplexity value
        '''
        if tf.equal(self.n,tf.constant(0)):
            return tf.constant(-1)
        neg_n=tf.cast(tf.math.negative(self.n),tf.float32)
        exponent = tf.math.divide(self.sum,neg_n)
        base = tf.constant(2,dtype=tf.float32)
        res = tf.math.pow(base,exponent)
        return res

