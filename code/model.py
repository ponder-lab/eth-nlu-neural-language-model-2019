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

    @tf.function
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

    def step(self,word_id,state=None):
        '''
        Method performs one step of model
        Arguments:
            - word_id: a batch size of wordsbased on which the model performs a step
            - state: if given the state of previous step
        Returns:
            - word_id: a batch of words computed in this step
            - state: the new state    
        '''
        #initial state 
        if state == None:
            init_state = tf.zeros([self.batch_size, self.hidden_state_size])
            state = (init_state, init_state)

        word_embedding_batch = self.embedding(word_id)

        output, state = self.lstm_cell(word_embedding_batch, state)

        # project y_t down to output size |vocab|
        if self.output_size != self.hidden_state_size:#beacuse we only generate in 1c, this will always be the case. 
            output = self.projection_layer(output) # \in [batch_size, output_size]

        output = self.softmax_layer(output)

        return output, state