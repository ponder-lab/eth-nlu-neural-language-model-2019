'''
Natural Language Understanding 

Project 1: Neural Language Model
Task 1: RNN Language Modelling

File used for task 1.2. Functions used to generate sentence endings. 

Authors: Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber
Date: 17.04.2019
Version: 3
'''
# Standard packages
import numpy as np
import pandas as pd
import tensorflow as tf

# Local modules
from global_variable import SPECIAL, SENTENCE_LENGTH, PATH_CONTINUATION, BATCH_SIZE, PATH_VOCAB, PATH_SUBMISSION
from util import build_continuation_dataset, build_vocab_lookup

def conditional_generation(word_to_index_table,index_to_word_table, model=None):
    '''
    Builds a dataset of the sentences that will be extended by our model. Used our model
    to predict sentences of length less or equal than 20. If the <eos> tag occurs 
    before the 20th position in the sentence, it cuts off the rest of the sentence. 
    Predictions are stored in the respective submission file. 

    Arguments: 
        - word_to_index_table: vocabulary lookup table that maps words to indices
        - index_to_word_table: vocabulary lookup table that maps indices to words
        - model: Language model that is used for predictions

    '''

    ds_continuation = build_continuation_dataset(PATH_CONTINUATION, vocab=word_to_index_table)
    ds_continuation = ds_continuation.batch(BATCH_SIZE)
    predicted_sentence = []

    for sentence, length in ds_continuation:
        size_batch = sentence.shape[0]
        if (size_batch != 64):
            sentence = tf.concat([sentence,tf.zeros((BATCH_SIZE-size_batch,sentence.shape[1]),dtype=tf.int64)],axis=0)

        print(sentence, length)
        break

        #make 20 predictions
        for i in range(20):
            logits, preds = model(sentence)
            preds = tf.argmax(preds, axis=2)

            #add new word to each input
            for i in range(BATCH_SIZE):
                sentence[i,length[i]]=preds[i,length[i]-1]
                #add one to the length
                length = tf.math.add(length,[tf.constant(1)])
                #make sure, we dont predict more than 20 words
                length = tf.map_fn((lambda x:21 if x>21 else x),length)


        for i in range(BATCH_SIZE):
            #slice to gett one sentence
            curr_sentence = sentence[i,:]

            #find position of <eos>
            eos = np.where(curr_sentence==1) #<eos> is 1 hardcoded
            eos = eos[0]+1 if len(eos)>0 else 20
            #take part to <eos> or at most 20 words
            curr_sentence = curr_sentence[:min(eos,20)]
            
            #TODO filter out end of sentence if it contains <eos>
            predicted_sentence.append(curr_sentence)
        #only one batch for now    
        break
    pred = np.array(predicted_sentence)
    pred = sentences_to_text(index_to_word_table,pred)
    pd.DataFrame(pred).to_csv(PATH_SUBMISSION+'group35.continuation',sep=' ', header=False , index=False)
    return 


def sentences_to_text(index_to_word_table, sentence):
    '''
    Transforms a sentence consisting of IDs into a sentence consisting of words.

    Arguments:
        - index_to_word_table: vocabulary lookup table that maps indices to words
        - sentence: input sentence

    Returns:
        - sentence where the IDs of the words are replaced with the actual words (strings)
    '''
    f = (lambda id:index_to_word_table.lookup(id))
    f = np.vectorize(f)
    #print(sentence)
    return f(sentence)