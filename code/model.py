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
import sys
from collections import Counter

# Local modules
from util import * #@UnusedWildImport pylint: disable=unused-wildcard-import
from global_variable import * #@UnusedWildImport pylint: disable=unused-wildcard-import
from embedding import load_embedding

class LanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, sentence_length, embedding_size, hidden_state_size, output_size, batch_size, word_embeddings=None , index_to_word_table=None):
        super(LanguageModel, self).__init__(self)

        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.embedding_size = embedding_size
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.batch_size = batch_size

        if word_embeddings is not None:
            # use pretrained word embeddings
            init = tf.constant_initializer(word_embeddings)
        else:
            # init embeddings with xavier initializer
            init = tf.initializers.GlorotUniform()

        self.embedding = tf.keras.layers.Embedding(
            input_dim= vocab_size,
            output_dim= embedding_size,
            input_length= sentence_length,
            embeddings_initializer=init,
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
        )

        self.lstm_cell = tf.keras.layers.LSTMCell(
            # dimensionality of the output space
            units=hidden_state_size,
            kernel_initializer=tf.initializers.GlorotUniform(), # xavier initializer
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
                kernel_initializer=tf.initializers.GlorotUniform(), # xavier initializer
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
                kernel_initializer=tf.initializers.GlorotUniform(), # xavier initializer
                name="W",

                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None
            )

    def call(self, sentence_id_batch):

        # initialize lstm state
        init_state = tf.zeros([self.batch_size, self.hidden_state_size])
        state = (init_state, init_state)

        logits  = []

        # embedding layer -> gets a sentence as input and performs the embedding
        sentence_embedding_batch = self.embedding(sentence_id_batch)

        for pos in range(self.sentence_length-1):

            # extract word -> identity returns a tensor of the same dimension
            # dimensions: [batch, sentence length, embedding size]
            # selects a slice of the cube -> every embedding, every sentence in the batch, but always
            # one certain position
            word_embedding_batch =  sentence_embedding_batch[:, pos, :]

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

        return logits

@tf.function
def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

class Perplexity(tf.metrics.Metric):

    def __init__(self, name='perplexity', **kwargs):
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(dtype=tf.float64,name='perp_sum_log_probs', initializer='zeros')
        self.n = self.add_weight(dtype=tf.int32,name='perp_n', initializer='zeros')

        # pre define constant tensor [0,1,2,..., ] which is used at every update_state
        # as part of required index (think more efficient than creating new every time)
        self.range = tf.range((SENTENCE_LENGTH-1)*BATCH_SIZE, dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''

        Arguments:
            - y_true = masked labels for the true index of the word for every pos  ->
                        (y_true in \[n])
            - y_pred = masked probabilities (not logits!) for each
                        non-padding position in batch
                        (y_pred \in [n, VOCAB_SIZE])
        '''

        n = tf.shape(y_pred)[0]
        self.n.assign_add(n)

        # use only required slice of constant [0,1,2,..., n-1]
        range = self.range[:n]

        # build gather index by merging range [0,1,2, ...] with y_true
        indices = tf.stack([range, y_true], axis=1)

        # select for every sample the probability of the true word (label)
        probs = tf.gather_nd(params=y_pred, indices=indices) # probs \in [n]

        log_probs = log2(probs) # log_probs \in [n]

        sum_log_probs = tf.reduce_sum(log_probs) # sum_log_probs \in scalar

        self.sum.assign_add(tf.cast(sum_log_probs, dtype=tf.float64))

    def result(self):
        '''
        Computes the actual perplexity value.

        Returns:
            - res: -1 if result gets called without updating the state first,
                   otherwise the summed up perplexity value
        '''
        # TODO maybe handle div by 0 error

        #if tf.equal(self.n,tf.constant(0)):
        #    return tf.constant(-1)

        perp = tf.math.pow(tf.constant(2, dtype=tf.float64), -1/tf.cast(self.n, dtype=tf.float64) * self.sum)

        return perp
