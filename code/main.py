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
import os
sys.path.append('./')

# Other imports
from tensorflow.keras import Model

# Local modules
from util import build_vocab, build_vocab_lookup
from model import LanguageModel


from global_variable import *

from embedding import load_embedding
from training import train
from evaluation import evaluate
from generation import generate


__author__ = 'Nicolas Küchler, Philippe Blatter, Lucas Brunner, Fynn Faber'
__email__ = 'kunicola@student.ethz.ch, pblatter@student.ethz.ch, brunnelu@student.ethz.ch, faberf@student.ethz.ch'
__status__ = 'v9'


def main():
    '''
    Main function that coordinates the entire process. Parses arguments that specify the exercise and the
    experiment that should be run. Initializes the model and the checkpoint managers.
    '''

    parser = argparse.ArgumentParser(description='Define configuration of experiments')
    parser.add_argument('--mode', type=str, nargs='+', choices=['train', 'evaluate','generate'], required=True)
    parser.add_argument('--experiment', type=str, choices=['a','b','c'], required=True)
    parser.add_argument('--id', type=str, required=False)
    parser.add_argument('--epochs', type=int, default=EPOCHS, required=False)

    args = parser.parse_args()


    # Setting Experiment Id
    if args.id is None:
        exp_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"No Experiment Id Set, Creating New: {exp_id}")
    else:
        exp_id = args.id
        print(f"Using Experiment Id: {exp_id}")

    # Setting Directories
    base_dir = f"{OUTPUT_DIR}/exp_{args.experiment}/{exp_id}"
    log_dir = f"{base_dir}/logs"
    submission_dir = f"{base_dir}/submissions"
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    ckpt_dir = f"{base_dir}/ckpts"



    print(f"Experiment Directory: {base_dir}")

    print(f"Using Tensorflow Version: {tf.__version__}")
    print("Building Vocabulary...")
    build_vocab(input_file=PATH_TRAIN, output_file=PATH_VOCAB, top_k=VOCAB_SIZE, special=SPECIAL)
    word2id, id2word = build_vocab_lookup(PATH_VOCAB, "<unk>")

    # Setting Experiment Specific Configurations
    if args.experiment == 'a':
        lstm_hidden_state_size = 512
        word_embeddings = None

    elif args.experiment == 'b':
        lstm_hidden_state_size = 512
        word_embeddings = load_embedding(dim_embedding=EMBEDDING_SIZE, vocab_size=VOCAB_SIZE)

    elif args.experiment == 'c':
        lstm_hidden_state_size = 1024
        word_embeddings = load_embedding(dim_embedding=EMBEDDING_SIZE, vocab_size=VOCAB_SIZE)
    else:
        raise ValueError(f"Unknown Experiment {args.experiment}")


    print(f'Initializing Model...')
    model = LanguageModel(vocab_size = VOCAB_SIZE,
                            sentence_length =  SENTENCE_LENGTH,
                            embedding_size = EMBEDDING_SIZE,
                            hidden_state_size = lstm_hidden_state_size,
                            output_size = LSTM_OUTPUT_SIZE,
                            batch_size = BATCH_SIZE,
                            word_embeddings = word_embeddings,
                            index_to_word_table = id2word)

    print(f'Initializing Optimizer...')
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    if manager.latest_checkpoint:
        print(f"Restoring Model from {manager.latest_checkpoint}...")
        ckpt.restore(manager.latest_checkpoint)
        model_loaded = True
    else:
        print("Initializing Model from Scratch")
        model_loaded = False

    if "train" in args.mode:
        print(f"Starting Training...")
        train_summary_writer = tf.summary.create_file_writer(f"{log_dir}/train")
        with train_summary_writer.as_default():
            train(ckpt=ckpt,
                    manager=manager,
                    model=model,
                    optimizer=optimizer,
                    word2id=word2id,
                    id2word=id2word,
                    epochs=args.epochs)
        model_loaded = True

    if "evaluate" in args.mode:
        print(f"Starting Evaluation...")
        assert model_loaded, 'model must be loaded from checkpoint in order to be evaluated'

        test_summary_writer = tf.summary.create_file_writer(f"{log_dir}/evaluate")
        with test_summary_writer.as_default():
            evaluate(model=model, word2id=word2id, id2word=id2word, step=optimizer.iterations, path_submission=f"{submission_dir}/group35.perplexity{args.experiment.upper()}")


    if "generate" in args.mode:
        print(f"Starting Generation...")
        assert model_loaded, 'model must be loaded from checkpoint in order to start generation'

        generate_summary_writer = tf.summary.create_file_writer(f"{log_dir}/generate")
        with generate_summary_writer.as_default():
            generate(word2id, id2word, model=model, path_submission=f"{submission_dir}/group35.continuation")

if __name__ == "__main__":
    main()
