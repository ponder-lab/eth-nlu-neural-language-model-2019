'''
Natural Language Understanding

Project 1: Neural Language Model
Task 1: RNN Language Modelling

Main file that takes arguments specifying which task and which experiment
should be run (python3 main.py -h for explanations).

Authors: Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber
Date: 17.04.2019
Version: 8
'''

# Standard packages
import numpy as np
import tensorflow as tf
import datetime
import argparse
import sys
sys.path.append('./')

# Other imports
from tensorflow.keras import Model

# Local modules
from dataset import build_dataset
from util import build_vocab, build_vocab_lookup
from model import LanguageModel, Perplexity
from embedding import load_embedding

from global_variable import *
from generate import conditional_generation

__author__ = 'Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber'
__email__ = 'kunicola@student.ethz.ch, pblatter@student.ethz.ch, brunelu@student.ethz.ch, faberf@student.ethz.ch'
__status__ = 'v9'


def main():
    '''
    Main function that coordinates the entire process. Parses arguments that specify the exercise and the
    experiment that should be run. Initializes the model and the checkpoint managers.
    '''

    string_suffix = '/2_'

    print(tf.__version__)

    build_vocab(input_file=PATH_TRAIN, output_file=PATH_VOCAB, top_k=VOCAB_SIZE, special=SPECIAL)
    word_to_index_table, index_to_word_table = build_vocab_lookup(PATH_VOCAB, "<unk>")

    LSTM_HIDDEN_STATE_SIZE = LSTM_HIDDEN_STATE_SIZE_B


    print(f'initialize model')
    model = LanguageModel(vocab_size = VOCAB_SIZE,
                        sentence_length =  SENTENCE_LENGTH,
                        embedding_size = EMBEDDING_SIZE,
                        hidden_state_size = LSTM_HIDDEN_STATE_SIZE,
                        output_size = LSTM_OUTPUT_SIZE,
                        batch_size = BATCH_SIZE,
                        word_embeddings = WORD_EMBEDDINGS,
                        index_to_word_table = index_to_word_table)


    print(f'initialize optimizer')
    optimizer = tf.keras.optimizers.Adam()

    # checkpoint object
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)

    # checkpoint manager
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts'+string_suffix, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint: # TODO maybe add option to specify from which checkpoint to restore
        print(f'Restored from {manager.latest_checkpoint}')
    else:
        print("Initializing from scratch.")

    option = string_suffix[1:]
    print(f'option: {option}')


    if option == '2_':
        #TODO -> the model should be the trained model from 1c -> checkpoint required
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts/1c', max_to_keep=5) # TODO in this folder there won't be a checkpoint?
        ckpt.restore(manager.latest_checkpoint) # TODO add option to choose checkpoint
        if manager.latest_checkpoint:
            print(f'Restored from {manager.latest_checkpoint}')
        else:
            #raise ValueError("No model found!")
            print(f'Initializing from scratch!')
        conditional_generation(word_to_index_table, index_to_word_table, model=model)



if __name__ == "__main__":
    main()
