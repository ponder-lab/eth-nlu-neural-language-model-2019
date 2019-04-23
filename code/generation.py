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
from util import build_vocab_lookup
from dataset import build_continuation_dataset

def generate(word_to_index_table,index_to_word_table, model=None, path_submission=None):
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

    print(f'model: {model}')

    # sentence in \[batch_size, sentence_length-1]
    # length in \[batch_size]
    for sentence, length in ds_continuation:
        #print(f'sentence shape: {sentence.shape}')
        #print(f'length shape: {length.shape}')
        #print(f'length: {length}')
        size_batch = sentence.shape[0]

        # if last fraction is less than the batch size, apply zero padding
        if (size_batch != 64):
            sentence = tf.concat([sentence,tf.zeros((BATCH_SIZE-size_batch,sentence.shape[1]),dtype=tf.int64)],axis=0)

        # make 20 predictions
        for i in range(20):

            logits = model(sentence) # \in [batch_size, sentence_length-1, vocab_size]
            preds = tf.nn.softmax(logits, name=None)
            preds = tf.argmax(preds, axis=2) # \in [batch_size, sentence_length - 1]

            # both sentence and preds are of type <class 'tensorflow.python.framework.ops.EagerTensor'>

            # add new word to each input sentence
            for i in range(BATCH_SIZE):

                # print(f'appending a word to sentence: {i}')

                # casting applied as eager tensor doesn't support assigning -> even though I am pretty
                # sure that there is a cleaner way to do this
                sentence_np = sentence.numpy()
                sentence_np[i,length[i]]=preds[i,length[i]-1]
                sentence = tf.convert_to_tensor(sentence_np, dtype=tf.int64)
                #sentence[i,length[i]].assign(preds[i,length[i]-1])

            #add one to the length
            length = tf.math.add(length,[tf.constant(1)])
            #make sure, we dont predict more than 20 words
            length = tf.map_fn((lambda x:21 if x>21 else x),length)

        #return
        for i in range(BATCH_SIZE):

            #print(f'cutting sentence: {i}')

            #slice to get one sentence
            curr_sentence = sentence[i,:].numpy()
            #print(f'current sentence: {curr_sentence}')

            #find position of <eos>
            eos = np.where(curr_sentence==1) # <eos> is has index 1 (hardcoded)
            #print(f'type of eos: {type(eos)}')
            #print(f'eos: {eos}')
            #print(f'eos[0]: {eos[0]}')

            # if there is an <eos> pad in the sentence, compute the index of the first appearance
            # if no <eos> was predicted, use 20 as an upper bound
            idx = (eos[0])[0]+1 if len(eos[0])>0 else 21

            #print(f'idx: {idx}')

            #take part to <eos> or at most 20 words
            curr_sentence = curr_sentence[1:idx]

            #print(f'type of curr_sentence[0]: {type(curr_sentence[0])}')
            curr_sentence = sentences_to_text(index_to_word_table,curr_sentence)

            #TODO filter out end of sentence if it contains <eos>
            predicted_sentence.append(curr_sentence)


        #only one batch for now
        #break
    #print(predicted_sentence)
    #print(len(predicted_sentence[0]))
    pred = np.array(predicted_sentence)
    #pred = tf.convert_to_tensor(pred, dtype=tf.int64)
    #print(f'pred[0]: {pred[0]}')
    #print(f'type(pred[0]): {type(pred[0])}')
    #pred = sentences_to_text(index_to_word_table,pred)
    pd.DataFrame(pred).to_csv(path_submission, sep=' ', header=False , index=False)
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

    ## ======================================================================
    # not sure if this block is necessary..?
    sentence = tf.cast(sentence, dtype=tf.int64)
    #print(f'type of sentence [0]: {type(sentence[0])}')
    ## ======================================================================

    '''
    f = (lambda id:index_to_word_table.lookup(id))
    f = np.vectorize(f)
    #print(sentence)
    return f(sentence)
    '''

    result = []
    for idx in sentence:
        result.append(index_to_word_table.lookup(idx).numpy().decode('utf-8'))

    #print(f'result: {result}')
    return result
