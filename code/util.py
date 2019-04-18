'''
Natural Language Understanding 

Project 1: Neural Language Model
Task 1: RNN Language Modelling

Util file containing functions that are used to build data sets and vocabulary
files from the given inputs files throughout all exercises.

Authors: Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber
Date: 17.04.2019
Version: 4
'''

# Standard packages
import numpy as np
import tensorflow as tf
from collections import Counter

# Local modules
#from code.global_variable import SPECIAL, SENTENCE_LENGTH
from global_variable import SPECIAL, SENTENCE_LENGTH


def build_vocab(input_file, output_file, top_k=None, special=None):  
    '''
    Builds a vocubulary output_file of size top_k, taking the most frequent words 
    in the input_file and also adding the special symbols from the given dict.

    Arguments: 
        - input_file: Path to a file containing the a corpus that will be transformed into a vocabulary
        - output_file: Path to a file containing the built vocabulary
        - top_k: Most k common words contained in the vocabulary
        - special: Dictionary containing the special tags (<bos>, <eos>, <pad>)
    '''
    with open(input_file) as f:
        wordcount = Counter(f.read().split())
        wordcount = wordcount.most_common(top_k-len(special)-1) #TODO -1?
        
    with open(output_file, "w") as f:
        for symbol in special.values():
            f.write(f"{symbol}\n")
            
        for word, _ in wordcount:
            f.write(f"{word}\n")

def build_vocab_lookup(filename, unknown_value):
    '''
    Builds lookup tables for the mapping: word (str) <--> wordId (int)

    Arguments: 
        - filename: Path to the vocabulary text file
        - unknown_value: Replacement symbol for all words that are not contained in the vocabulary.

    Returns: 
        - word_to_index_table: vocabulary lookup table that maps words to indices
        - index_to_word_table: vocabulary lookup table that maps indices to words
        
    '''

    table_initializer = tf.lookup.TextFileInitializer(filename=filename,
                                                        key_dtype=tf.string,
                                                        key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                                                        value_dtype=tf.int64,
                                                        value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                                                        vocab_size=None,
                                                        delimiter=" ")

    word_to_index_table = tf.lookup.StaticVocabularyTable(table_initializer, num_oov_buckets=1)
    
    index_to_word_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(filename=filename,
                                                        value_dtype=tf.string,
                                                        value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                                                        key_dtype=tf.int64,
                                                        key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                                                        vocab_size=None,
                                                        delimiter=" "), unknown_value)
    return word_to_index_table, index_to_word_table

def build_dataset(filename, vocab):
    '''
    Builds a dataset from the given file and vocabulary

    Arguments: 
        filename: Path to a corpus consisting of lines of words (sentences)
        vocab: Path to the vocabulary text file

    Returns: 
        dataset: Parsed data set
    '''

    # load dataset from text file
    dataset = tf.data.TextLineDataset(filename)

    # tokenize sentence
    dataset = dataset.map(lambda sentence: tf.strings.split([sentence], sep=' ').values)

    #remove empty string words (Yeah it looks ugly)
    dataset=dataset.map(lambda sentence: tf.cond(tf.equal(tf.strings.length(sentence[-1]),tf.constant(0)),true_fn=lambda: sentence[:-1], false_fn=lambda: sentence))
    
    # add <bos> and <eos>
    dataset = dataset.map(lambda sentence: tf.concat([[SPECIAL['bos']], sentence, [SPECIAL['eos']]], axis=0))

    # filter out sentences longer than 30
    dataset = dataset.filter(lambda sentence: tf.shape(sentence)[0] <= SENTENCE_LENGTH)

    # pad all sentences to length 30
    dataset = dataset.map(lambda sentence: tf.pad(sentence, [[0,SENTENCE_LENGTH - tf.shape(sentence)[0]]], mode='CONSTANT', constant_values=SPECIAL['pad']))
    
    # map words to id
    dataset = dataset.map(lambda sentence: vocab.lookup(sentence))
    
    # map to sentence and labels
    dataset = dataset.map(lambda sentence: (sentence[:-1],sentence[1:]))
    
    return dataset

def build_continuation_dataset(filename, vocab):
    '''
    Builds a dataset from the given file and vocabulary

    Arguments: 
        filename: Path to a corpus consisting of lines of words (sentences)
        vocab: Path to the vocabulary text file

    Returns: 
        dataset: Parsed data set
    '''
    
    # load dataset from text file
    dataset = tf.data.TextLineDataset(filename)

    # tokenize sentence
    dataset = dataset.map(lambda sentence: tf.strings.split([sentence], sep=' ').values)
    
    #remove empty string words (Yeah it looks ugly)
    dataset=dataset.map(lambda sentence: tf.cond(tf.equal(tf.strings.length(sentence[-1]),tf.constant(0)),true_fn=lambda: sentence[:-1], false_fn=lambda: sentence))
    
    # add <bos> and <eos>
    dataset = dataset.map(lambda sentence: tf.concat([[SPECIAL['bos']], sentence], axis=0))

    # pad all sentences to length 30
    dataset = dataset.map(lambda sentence: (tf.pad(sentence, [[0,SENTENCE_LENGTH - tf.shape(sentence)[0]]], mode='CONSTANT', constant_values=SPECIAL['pad']),tf.shape(sentence)[0]))
    
    # map words to id
    dataset = dataset.map(lambda sentence,length: (vocab.lookup(sentence),length))
    
    # map to sentence and labels
    dataset = dataset.map(lambda sentence, length: (sentence[:-1],length))
    
    return dataset