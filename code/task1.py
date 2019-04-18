'''
Natural Language Understanding 

Project 1: Neural Language Model
Task 1: RNN Language Modelling

File used to compute the perplexity of given evaluation sentences. 

Authors: Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber
Date: 17.04.2019
Version: 2
'''

# Standard packages
import numpy as np
import pandas as pd
import tensorflow as tf

# Local modules
from global_variable import SPECIAL, SENTENCE_LENGTH,PATH_TEST, BATCH_SIZE, PATH_SUBMISSION,PATH_VOCAB
from util import build_dataset, build_vocab_lookup
from model import Perplexity

def run_task1(word_to_index_table, model, task_nr, manager, checkpoint):
    '''
    Restores the model parameters, loads the test set and then computes the perplexity 
    for the given sentences.
    
    Arguments: 
        - word_to_index_table: 
        - model: trained model used to compute the perplexity
        - task_nr: defines the experiment number
        - manager: Coordinates the storing and restoring of the latest model and optimizer parameters
        - checkpoint: Checkpoint object, used for storing and restoring the latest model and optimizer parameters
    '''

    # restore the model 
    checkpoint.restore(manager.latest_checkpoint)

    print('started: run_task1')
    dataset = build_dataset(PATH_TEST,word_to_index_table)
    dataset = dataset.batch(BATCH_SIZE)

    perp_res = []
    perplexity = Perplexity()

    for sentence, labels in dataset:
        #print(sentence)
        #handle last batch
        size_batch = sentence.shape[0]
        if (size_batch != 64):
            sentence = tf.concat([sentence,tf.zeros((BATCH_SIZE-size_batch,sentence.shape[1]),dtype=tf.int64)],axis=0)
            print(sentence)


        #print(sentence.shape, labels.shape)
        logits, preds = model(sentence)

        for i in range(size_batch):
            perplexity.update_state(labels[i,:], preds[i,:])
            perp_res.append(perplexity.result().numpy())
            perplexity.reset_states()
        
    perp_res = np.reshape(np.array(perp_res),newshape=(-1,1))
    print(f'perp_res.shape: {perp_res.shape}')
    pd.DataFrame(perp_res).to_csv(PATH_SUBMISSION+'group35.perplexity'+task_nr,sep=' ', header=False , index=False)
    return 
