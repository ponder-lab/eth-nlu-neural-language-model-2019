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

    # TODO proper usage of arg parse
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

    # case distinction of string_suffix
    # TODO: check if model stores correct parameters or another checkpoint restoring is required
    if option == '1a':
        if training != 'predict':
            train(ckpt, manager, model, optimizer, word_to_index_table, index_to_word_table, string_suffix)
        if training != 'train':
            run_task1(word_to_index_table, model, 'A', manager, ckpt)
    elif option == '1b':
        if training != 'predict':
            train(ckpt, manager, model, optimizer, word_to_index_table, index_to_word_table, string_suffix)
        if training != 'train':
            run_task1(word_to_index_table, model, 'B', manager, ckpt)
    elif option == '1c':
        if training != 'predict':
            train(ckpt, manager, model, optimizer, word_to_index_table, index_to_word_table, string_suffix)
        if training != 'train':
            run_task1(word_to_index_table, model, 'C', manager, ckpt)


    elif option == '2_':
        #TODO -> the model should be the trained model from 1c -> checkpoint required
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts/1c', max_to_keep=5) # TODO in this folder there won't be a checkpoint?
        ckpt.restore(manager.latest_checkpoint) # TODO add option to choose checkpoint
        if manager.latest_checkpoint:
            print(f'Restored from {manager.latest_checkpoint}')
        else:
            raise ValueError("No model found!")
        conditional_generation(word_to_index_table, index_to_word_table, model=model)


def train(ckpt, manager, model, optimizer, word_to_index_table, index_to_word_table, suffix):
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
    train_log_dir = PATH_LOG + suffix + '/' + current_time + '/train'
    valid_log_dir = PATH_LOG + suffix + '/' + current_time + '/valid'

    print(f'writing train summaries to: {train_log_dir}')
    print(f'writing valid summaries to: {valid_log_dir}')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)


    # Build Training and Validation Dataset
    ds_train = build_dataset(PATH_TRAIN, vocab=word_to_index_table)
    ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # ds_train = ds_train.take(2) # uncomment for demo purposes

    ds_valid = build_dataset(PATH_VALID, vocab=word_to_index_table)
    ds_valid = ds_valid.batch(BATCH_SIZE)
    ds_valid = ds_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # ds_valid = ds_valid.take(1)  # uncomment for demo purposes


    # Define Train Metrics
    metrics_train = {}
    metrics_train['train_loss'] = tf.metrics.Mean(name='train_loss')
    metrics_train['train_accuracy'] = tf.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    metrics_train['train_top5_accuracy'] = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name="train_top5_accuracy")
    metrics_train['train_perplexity'] = Perplexity(name="train_perplexity")

    # Define Valid Metrics
    metrics_valid = {}
    metrics_valid['valid_loss'] = tf.keras.metrics.Mean(name='valid_loss')
    metrics_valid['valid_accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    metrics_valid['valid_top5_accuracy'] = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name="valid_top5_accuracy")
    metrics_valid['valid_perplexity'] = Perplexity(name="valid_perplexity")

    # Training Loop
    best_validation_score = 0
    for epoch in range(EPOCHS):

        with train_summary_writer.as_default():
            print(f"Start Training of Epoch {epoch}")
            train_epoch(model=model, dataset=ds_train, optimizer=optimizer, metrics=metrics_train)

        with valid_summary_writer.as_default():
            print(f"Start Validation of Epoch {epoch}")
            validation_score = validate_epoch(model=model, dataset=ds_valid, step=optimizer.iterations, metrics=metrics_valid)

        ckpt.step.assign_add(1)

        if validation_score > best_validation_score:
            # current validation score is better than previous -> store checkpoints
            best_validation_score = validation_score
            save_path = manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")
            print(f"Validation Score {validation_score.numpy()}")



@tf.function
def train_epoch(model, dataset, optimizer, metrics):

    for sentence, labels, mask in dataset:

        # sentence, labels, mask \in [batch_size, sentence_length-1]

        with tf.GradientTape() as tape:

            # feed sentence to model to calculate
            logits = model(sentence)
            preds = tf.nn.softmax(logits, name=None)

            # mask out padding
            logits_masked = tf.boolean_mask(logits, mask) # logits_masked \in [number_words_in_batch, VOCAB_SIZE] where number_words_in_batch is number of unpadded words
            preds_masked = tf.boolean_mask(preds, mask) # logits_masked \in [number_words_in_batch, VOCAB_SIZE]
            labels_masked = tf.boolean_mask(labels, mask) # labels_masked \in [number_words_in_batch]

            # calculate masked loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_masked, logits=logits_masked)
            loss = tf.math.reduce_mean(loss)

        # apply gradient clipping
        gradients = tape.gradient(loss, model.trainable_variables)
        clipped_gradients, _global_norm = tf.clip_by_global_norm(gradients, clip_norm=GRADIENT_CLIPPING_NORM, use_norm=None, name=None)
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

        # calculate metrics
        metrics['train_loss'].update_state(loss)
        metrics['train_accuracy'].update_state(y_true=labels_masked, y_pred=logits_masked)
        metrics['train_top5_accuracy'].update_state(y_true=labels_masked, y_pred=logits_masked)
        metrics['train_perplexity'].update_state(y_true=labels_masked, y_pred=preds_masked)

        # log metrics to summary every summary_freq steps (aggregated over the last summary_freq steps)
        if tf.equal(optimizer.iterations % SUMMARY_FREQ, 0):
            tf.summary.scalar('train_loss', metrics['train_loss'].result(), step=optimizer.iterations)
            metrics['train_loss'].reset_states()

            tf.summary.scalar('train_accuracy', metrics['train_accuracy'].result(), step=optimizer.iterations)
            metrics['train_accuracy'].reset_states()

            tf.summary.scalar('train_top5_accuracy', metrics['train_top5_accuracy'].result(), step=optimizer.iterations)
            metrics['train_top5_accuracy'].reset_states()

            tf.summary.scalar('train_perplexity', metrics['train_perplexity'].result(), step=optimizer.iterations)
            metrics['train_perplexity'].reset_states()


@tf.function
def validate_epoch(model, dataset, step, metrics):

    metrics['valid_loss'].reset_states()
    metrics['valid_accuracy'].reset_states()
    metrics['valid_top5_accuracy'].reset_states()
    metrics['valid_perplexity'].reset_states()

    for sentence, labels, mask in dataset:
        logits = model(sentence)

        preds = tf.nn.softmax(logits, name=None)

        # mask out padding
        logits_masked = tf.boolean_mask(logits, mask)
        preds_masked = tf.boolean_mask(preds, mask)
        labels_masked = tf.boolean_mask(labels, mask)

        # calculate masked loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_masked, logits=logits_masked)
        loss = tf.math.reduce_mean(loss)

        # update metrics
        metrics['valid_loss'].update_state(loss)
        metrics['valid_accuracy'].update_state(y_true=labels_masked, y_pred=logits_masked)
        metrics['valid_top5_accuracy'].update_state(y_true=labels_masked, y_pred=logits_masked)
        metrics['valid_perplexity'].update_state(y_true=labels_masked, y_pred=preds_masked)

    # log metrics after complete pass through validation set
    tf.summary.scalar('valid_loss', metrics['valid_loss'].result(), step=step)
    tf.summary.scalar('valid_accuracy', metrics['valid_accuracy'].result(), step=step)
    tf.summary.scalar('valid_top5_accuracy', metrics['valid_top5_accuracy'].result(), step=step)
    tf.summary.scalar('valid_perplexity', metrics['valid_perplexity'].result(), step=step)

    # TODO define metric from which to choose if we want to save checkpoint
    return metrics['valid_accuracy'].result()

if __name__ == "__main__":
    main()
