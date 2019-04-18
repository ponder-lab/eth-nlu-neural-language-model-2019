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
from gensim import models

# Local modules
from util import build_vocab, build_vocab_lookup, build_dataset
from model import LanguageModel, LanguageModelError, Perplexity #, train_step , valid_step
from embedding import load_embedding
from predict_sentences import conditional_generation
from task1 import run_task1
from global_variable import * #@UnusedWildImport
from task1 import run_task1
from predict_sentences import conditional_generation

__author__ = 'Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber'
__email__ = 'kunicola@student.ethz.ch, pblatter@student.ethz.ch, brunelu@student.ethz.ch, faberf@student.ethz.ch'
__status__ = 'v9'


def main():
    '''
    Main function that coordinates the entire process. Parses arguments that specify the exercise and the 
    experiment that should be run. Initializes the model and the checkpoint managers. 
    '''

    parser = argparse.ArgumentParser(description='Define which exercise and which experiment you want to run (1a,1b,1c,2)!')
    parser.add_argument('exercise', type=int, help='Either 1 or 2')
    parser.add_argument('experiment', type=str, nargs='?', default='second', help='Indicate the experiment you want to run for exercise 1 - either a,b or c')
    parser.add_argument('training', type=str, nargs='?', default='second', help='Indicate if you want to train or predict - either \'train\' or \'predict\'')
    args = parser.parse_args()

    # exercise 1 train or predict
    training='both'
    if(len(sys.argv) == 4):
        _, exercise, experiment, training = sys.argv
        if (int(exercise) != 1):
            parser.print_help()
            exit(0)
        if experiment not in ['a', 'b', 'c']:
            parser.print_help()
            exit(0)
        if training not in ['train','predict']:
            parser.print_help()
            exit(0)

    # exercise 1 train and predict
    elif(len(sys.argv) == 3):
        _, exercise, experiment = sys.argv
        if (int(exercise) != 1):
            parser.print_help()
            exit(0)
        if experiment not in ['a', 'b', 'c']:
            parser.print_help()
            exit(0)
    
    # exercise 2
    else:
        _, exercise, = sys.argv
        if int(exercise) != 2:
            parser.print_help()
            exit(0)
        experiment = '_'
    
    print(f'indicated exercise: {exercise} \nindicated experiment: {experiment}')
    
    string_suffix = '/'+exercise+experiment

    print(tf.__version__)
    
    build_vocab(input_file=PATH_TRAIN, output_file=PATH_VOCAB, top_k=VOCAB_SIZE, special=SPECIAL)
    word_to_index_table, index_to_word_table = build_vocab_lookup(PATH_VOCAB, "<unk>")

    if experiment == 'a' or 'b':
        LSTM_HIDDEN_STATE_SIZE = LSTM_HIDDEN_STATE_SIZE_A
    else: 
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

    
    print(f'initialize loss and optimizer objects')
    loss_object = LanguageModelError()
    optimizer = tf.keras.optimizers.Adam()

    # checkpoint object
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)

    # checkpoint manager
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts'+string_suffix, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f'Restored from {manager.latest_checkpoint}')
    else:
        print("Initializing from scratch.")

    option = string_suffix[1:]
    print(f'option: {option}')
    
    # case distinction of string_suffix
    # TODO: check if model stores correct parameters or another checkpoint restoring is required
    if option == '1a':
        if training != 'predict':
            train(ckpt, manager, model, loss_object, optimizer, word_to_index_table, index_to_word_table, string_suffix)
        if training != 'train':
            run_task1(word_to_index_table, model, 'A', manager, ckpt)
    elif option == '1b':
        if training != 'predict':
            train(ckpt, manager, model, loss_object, optimizer, word_to_index_table, index_to_word_table, string_suffix)
        if training != 'train':
            run_task1(word_to_index_table, model, 'B', manager, ckpt)
    elif option == '1c':
        if training != 'predict':
            train(ckpt, manager, model, loss_object, optimizer, word_to_index_table, index_to_word_table, string_suffix)
        if training != 'train':
            run_task1(word_to_index_table, model, 'C', manager, ckpt)


    elif option == '2_':
        #TODO -> the model should be the trained model from 1c -> checkpoint required
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts/1c', max_to_keep=1)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print(f'Restored from {manager.latest_checkpoint}')
        else:
            print("No model found!")
        conditional_generation(word_to_index_table, index_to_word_table, model=model)




def train(ckpt, manager, model, loss_object, optimizer, word_to_index_table, index_to_word_table, suffix):
    '''
    Trains the model by unrolling the RNN, writes summary files for logging/plotting, stores the latest model to a checkpoint
    so that training can be resumed at a later point in time. 

    Arguments: 
        - ckpt: Checkpoint object, used for storing and restoring the latest model and optimizer parameters
        - manager: Coordinates the storing and restoring of the latest model and optimizer parameters
        - model: Language model that will be trained
        - loss_object: Used to compute the loss between the actual labels and the predictions
        - optimizer: Optimizer that will be assigned its old values when resuming training
        - word_to_index_table: vocabulary lookup table that maps words to indices
        - index_to_word_table: vocabulary lookup table that maps indices to words
        - suffix: in order to determine which logs to use
    '''

    # define time and summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'current time: {current_time}')
    #summary_writer = tf.summary.create_file_writer(PATH_LOG)
    train_log_dir = PATH_LOG + suffix + '/' + current_time + '/train'
    print(f'train_log_dir: {train_log_dir}')
    test_log_dir = PATH_LOG + suffix + '/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    train_summary_writer.flush()
    test_summary_writer.flush()

    ds_train = build_dataset(PATH_TRAIN, vocab=word_to_index_table)
    ds_train = ds_train.batch(BATCH_SIZE)
    
    ds_valid = build_dataset(PATH_VALID, vocab=word_to_index_table)
    ds_valid = ds_valid.batch(BATCH_SIZE)

    #define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_perplexity = Perplexity()

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    valid_perplexity = Perplexity()

    for epoch in range(EPOCHS):
        
        # TODO: should the padded part be masked? (excluded from loss)
        
        batch_index = 0
        for sentence, labels in ds_train:

            mask = labels.numpy() != 2
            print(f'shape of mask: {mask.shape}')

            masked_labels = tf.boolean_mask(labels, mask)
            print(f'shape of masked labels: {masked_labels.shape}')
            

            # sentence \in [batch_size, sentence_length]
            # labels \in [batch_size, sentence_length-1]
            # print(f"sentence = {sentence.shape}   labels = {labels.shape}")
            print("train_step")
            #train_step(sentence, labels) -->method is now inline
            with tf.GradientTape() as tape: 
                # within this context all ops are recorded =>
                # can calc gradient of any tensor computed in this context with respect to any trainable var
                
                #tf.summary.trace_on(graph=True, profiler=True)
                logits, preds = model(sentence)
                
                print(f"logits = {logits.shape}  preds = {preds.shape}, labels = {labels.shape}") 

                masked_logits = tf.boolean_mask(logits, mask)
                print(f'shape of masked logits: {masked_logits.shape}')

                # replaced labels with masked_labels and logits with masked_logits
                #loss = loss_object(y_true=labels, y_pred=logits)
                loss = loss_object(y_true=masked_labels, y_pred=masked_logits)

                print(f"Train loss  {loss}")
                print(f'Shape of loss: {loss.shape}')

            
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss {:1.2f}".format(loss.numpy()))

            
            # apply gradient clipping 
            gradients = tape.gradient(loss, model.trainable_variables)
            clipped_gradients, _global_norm = tf.clip_by_global_norm(gradients, clip_norm=GRADIENT_CLIPPING_NORM, use_norm=None, name=None)
            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
                
            # feed metrics
            train_loss(loss)
            train_accuracy(labels, logits)
            train_perplexity.update_state(labels[0,:], preds[0,:,:]) #perplexity only on one sentence
            #perplexity.update_state(labels, preds) #calulate perplexity on whole batch

            
            
            #print(f'before writing to files')
            # TODO: indentation level? -> write after every batch or after every epoch?
            with train_summary_writer.as_default():
                # TODO figure out how to properly 2 log metrics to tensorboard
                print(f'writing to files')

                tl_res = train_loss.result()
                ta_res = train_accuracy.result()
                tp_res = data=train_perplexity.result()

                tl = tf.summary.scalar('train_loss', data=tl_res, step=batch_index)
                ta = tf.summary.scalar('train_accuracy', data=ta_res, step=batch_index)
                tp = tf.summary.scalar('train_perplexity', data=tp_res, step=batch_index)
                #tf.summary.trace_export(
                 #   name='graph',
                 #   step=batch_index,
                 #   profiler_outdir=train_log_dir
                #)

                print(f'successfully written train_loss: {tl}, with val: {tl_res.numpy()}')
                print(f'successfully written train_accuracy: {ta}, with val: {ta_res.numpy()}')
                print(f'successfully written perplexity: {tp}, with val: {tp_res.numpy()}')

            batch_index += 1

            
        test_batch_ind = 0
        for sentence, labels in ds_valid:
            print("valid_step")
            #valid_step(sentence, labels) -->method is now inline
            logits, preds = model(sentence)
    
            loss = loss_object(y_true=labels, y_pred=logits)
            
            valid_loss(loss)
            valid_accuracy(labels, logits)
            valid_perplexity.update_state(labels, preds)

            # TODO: indentation level? -> write after every batch or after every epoch?
            with test_summary_writer.as_default():
                tf.summary.scalar('validation loss', valid_loss.result(), step=test_batch_ind)
                tf.summary.scalar('validation accuracy', valid_accuracy.result(), step=test_batch_ind)
                tf.summary.scalar('validation perplexity', valid_perplexity.result(), step=test_batch_ind)

            test_batch_ind += 1


        
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Perplexity: {}, Test Loss: {}, Test Accuracy: {}, Test Perplexity: {}'
        print (template.format(epoch+1,
                    train_loss.result(), 
                    train_accuracy.result()*100,
                    train_perplexity.result(),
                    valid_loss.result(), 
                    valid_accuracy.result()*100),
                    valid_perplexity.result())

        # Reset metrics every epoch
        train_loss.reset_states()
        valid_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
        train_perplexity.reset_states()
        valid_perplexity.reset_states()
    return




if __name__ == "__main__":
    main()
