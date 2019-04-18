#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from collections import Counter

from gensim import models

print(tf.__version__)


# ## Hyperparameter

# In[43]:


PATH_TRAIN = "./data/sentences/sentences.train"
PATH_VALID = "./data/sentences/sentences.eval"
PATH_VOCAB = "./data/vocab.txt"
PATH_EMBEDDING = "./data/wordembeddings-dim100.word2vec"

SPECIAL = {
    "bos" : "<bos>",
    "eos" : "<eos>",
    "pad" : "<pad>"
}

SENTENCE_LENGTH = 30
BATCH_SIZE = 64
VOCAB_SIZE = 20000


WORD_EMBEDDINGS = 1 # TODO load pretrained word embeddings and check that it's working in combination with tf.function
EMBEDDING_SIZE = 100
LSTM_HIDDEN_STATE_SIZE= 512
LSTM_OUTPUT_SIZE = 512

EPOCHS = 1
LEARNING_RATE = 0.001
GRADIENT_CLIPPING_NORM = 5


# ## Input Pipeline

# In[44]:


def build_vocab(input_file, output_file, top_k=None, special=None):  
    '''
    builds a vocubulary output_file of size top_k, taking the most frequent words 
    in the input_file and also adding the special symbols from the given dict
    '''
    with open(input_file) as f:
        wordcount = Counter(f.read().split())
        wordcount = wordcount.most_common(top_k-len(special)-1)
        
    with open(output_file, "w") as f:
        for symbol in special.values():
            f.write(f"{symbol}\n")
            
        for word, _ in wordcount:
            f.write(f"{word}\n")
            
build_vocab(input_file=PATH_TRAIN, output_file=PATH_VOCAB, top_k=VOCAB_SIZE, special=SPECIAL)


# In[45]:


def build_vocab_lookup(filename, unknown_value):
    '''
    builds lookup tables for the mapping: word (str) <--> wordId (int)
    '''

    word_to_index_table_initializer = tf.lookup.TextFileInitializer(filename, 
                                                      tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
                                                      tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, 
                                                      delimiter=" ")
    
    word_to_index_table = tf.lookup.StaticVocabularyTable(word_to_index_table_initializer, num_oov_buckets=1)
    
    
    
    index_to_word_table_initializer = tf.lookup.TextFileInitializer(filename,
                                                          tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
                                                          tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
                                                          delimiter=" ")
    index_to_word_table = tf.lookup.StaticHashTable(index_to_word_table_initializer, unknown_value)
   
    return word_to_index_table, index_to_word_table


def build_dataset(filename, vocab):
    '''
    builds a dataset from the given file and vocabulary
    '''
    
    # load dataset from text file
    dataset = tf.data.TextLineDataset(filename)

    # tokenize sentence
    dataset = dataset.map(lambda sentence: tf.strings.split([sentence], sep=' ').values)

    # add <bos> and <eos>
    dataset = dataset.map(lambda sentence: tf.concat([[SPECIAL['bos']], sentence, [SPECIAL['eos']]], axis=0))

    # filter out sentences longer than 30
    dataset = dataset.filter(lambda sentence: tf.shape(sentence)[0] <= SENTENCE_LENGTH)

    # pad all sentences to length 30
    dataset = dataset.map(lambda sentence: tf.pad(sentence, [[0,SENTENCE_LENGTH - tf.shape(sentence)[0]]], mode='CONSTANT', constant_values=SPECIAL['pad']))
    
    # map words to id
    dataset = dataset.map(lambda sentence: vocab.lookup(sentence))
    
    # map to sentence and labels
    dataset = dataset.map(lambda sentence: (sentence, sentence[1:SENTENCE_LENGTH]))
    
    return dataset


# #### Input Pipeline Test

# In[46]:


word_to_index_table, index_to_word_table = build_vocab_lookup(PATH_VOCAB, "<unk>")

ds_train = build_dataset(PATH_TRAIN, vocab=word_to_index_table)

for x in ds_train:
    print(x[0].shape)
    print(x[1].shape)
    print(x[0])
    print(index_to_word_table.lookup(x[0]))
    break

ds_train = ds_train.batch(BATCH_SIZE)

print('\n')

for x in ds_train:
    print(x[0].shape)
    print(x[1].shape)
    print(x[0][0,:])
    print(x[1].shape)
    print(x[1][0,:])
    break


# # Load embedding

# In[62]:


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
    
    print(f'type of model.vocab: {type(model.vocab)}')

    '''
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
    '''

    for idx in range(20000):
        tok = vocab.lookup(idx)
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    
    emb = external_embedding
    return emb
    
    #placeholder is not needed anymore
    #retrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    
    #Update 'ref' by assigning 'value' to it.
    #Update the value of `emb` to pretrained_embeddings -> which was only a placeholder.
    #-> this placeholder was fed by the external embedding
    #assign_op = emb.assign(pretrained_embeddings)
    
    #session.run(assign_op, {pretrained_embeddings: external_embedding}) # here, embeddings are actually set
    
    # the entire block is not needed anymore -> the only thing that it does is assigning the external 
    # embedding to the tensor 'emb'


# ## Model

# In[63]:



class LanguageModel(Model):
    def __init__(self, vocab_size, sentence_length, embedding_size, hidden_state_size, output_size, batch_size, word_embeddings=None):
        super(LanguageModel, self).__init__()
        
  
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.embedding_size = embedding_size
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        if word_embeddings is not None:
            # TODO this does not work yet with @tf.function
            # vocab correct? or is it the inverse table?
            emb = load_embedding(vocab=index_to_word_table, path=PATH_EMBEDDING, dim_embedding=100, vocab_size=VOCAB_SIZE)
            #emb = tf.Variable(word_embeddings, name=None)
            weights = [emb]
            trainable = False
        else:
            weights = None
            trainable = True
        
        
        self.embedding = tf.keras.layers.Embedding(
            input_dim= vocab_size,
            output_dim= embedding_size,
            input_length= sentence_length,
            weights = weights,
            trainable = trainable,
            
            embeddings_initializer='uniform', # TODO [nku] ?
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
                use_bias=False,
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
            # one for the state
            output, state = self.lstm_cell(word_embedding_batch, state)
            
            # project y_t down to output size |vocab|
            if self.output_size != self.hidden_state_size:
                output = self.projection_layer(output) # \in [batch_size, output_size]
            
            # apply softmax weights to obtain logits
            output = self.softmax_layer(output) # \in [batch_size, vocab_size]
            
            logits.append(output)
        
        # \in [batch_size, sentence_length-1, vocab_size]
        logits = tf.stack(logits, axis=1)
        
        # \in [batch_size, sentence_length-1, vocab_size]
        preds = tf.nn.softmax(logits, name=None)
        
        # print(f"logits shape = {logits.shape}") 
        # print(f"preds shape = {preds.shape}") 
        
        return logits, preds  
    


# In[64]:


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


# In[65]:


@tf.function # comment tf.function out for debugging 
def train_step(sentence, labels):
    with tf.GradientTape() as tape: 
        # within this context all ops are recorded =>
        # can calc gradient of any tensor computed in this context with respect to any trainable var
            
        logits, preds = model(sentence)
        # print(f"logits = {logits.shape}  preds = {preds.shape}") 

        loss = loss_object(y_true=labels, y_pred=logits)

        # print(f"loss  {loss}")
    
    # apply gradient clipping 
    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_gradients, _global_norm = tf.clip_by_global_norm(gradients, clip_norm=GRADIENT_CLIPPING_NORM, use_norm=None, name=None)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
        
    # feed metrics
    train_loss(loss)
    train_accuracy(labels, logits)

@tf.function
def valid_step(sentence, labels):
    logits, preds = model(sentence)
    
    loss = loss_object(y_true=labels, y_pred=logits)
    
    valid_loss(loss)
    valid_accuracy(labels, logits)


# ## Training

# In[66]:


summary_writer = tf.summary.create_file_writer('./logs')

with summary_writer.as_default():
    
    word_to_index_table, index_to_word_table = build_vocab_lookup(PATH_VOCAB, "<unk>")
    ds_train = build_dataset(PATH_TRAIN, vocab=word_to_index_table)
    ds_train = ds_train.batch(BATCH_SIZE)
    
    ds_valid = build_dataset(PATH_VALID, vocab=word_to_index_table)
    ds_valid = ds_valid.batch(BATCH_SIZE)
    
    model = LanguageModel(vocab_size = VOCAB_SIZE, 
                          sentence_length =  SENTENCE_LENGTH, 
                          embedding_size = EMBEDDING_SIZE, 
                          hidden_state_size = LSTM_HIDDEN_STATE_SIZE, 
                          output_size = LSTM_OUTPUT_SIZE,
                          batch_size = BATCH_SIZE,
                          word_embeddings = WORD_EMBEDDINGS)

    
    loss_object = LanguageModelError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    

    for epoch in range(EPOCHS):
        
        # TODO: should the padded part be masked? (excluded from loss)
        
        for sentence, labels in ds_train:
            # sentence \in [batch_size, sentence_length]
            # labels \in [batch_size, sentence_length-1]
            # print(f"sentence = {sentence.shape}   labels = {labels.shape}")
            print("train_step")
            train_step(sentence, labels)
            
            # TODO figure out how to properly 2 log metrics to tensorboard
            tf.summary.scalar('train_loss', data=train_loss.result(), step=epoch)

            
        for sentence, labels in ds_valid:
            print("valid_step")
            valid_step(sentence, labels)
            




