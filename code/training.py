import tensorflow as tf

from global_variable import *
from dataset import build_dataset
from evaluation import validate, format_to_text
from perplexity import Perplexity



def train(ckpt, manager, model, optimizer, word2id, id2word, epochs):
    '''
    Trains the model by unrolling the RNN, writes summary files for logging/plotting, stores the latest model to a checkpoint
    so that training can be resumed at a later point in time.

    Arguments:
        - ckpt: Checkpoint object, used for storing and restoring the latest model and optimizer parameters
        - manager: Coordinates the storing and restoring of the latest model and optimizer parameters
        - model: Language model that will be trained
        - optimizer: Optimizer that will be assigned its old values when resuming training
        - word2id: vocabulary lookup table that maps words to indices
        - id2word: vocabulary lookup table that maps indices to words
    '''


    # Build Training and Validation Dataset
    ds_train = build_dataset(PATH_TRAIN, vocab=word2id)
    ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE)
    ds_train = ds_train.batch(BATCH_SIZE,  drop_remainder=True)
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #ds_train = ds_train.take(5) # uncomment for demo purposes

    ds_valid = build_dataset(PATH_VALID, vocab=word2id)
    ds_valid = ds_valid.batch(BATCH_SIZE)
    ds_valid = ds_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #ds_valid = ds_valid.take(1)  # uncomment for demo purposes

    # Define Train Metrics
    metrics_train = {}
    metrics_train['loss'] = tf.metrics.Mean(name='train_loss')
    metrics_train['accuracy'] = tf.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    metrics_train['top5_accuracy'] = tf.metrics.SparseTopKCategoricalAccuracy(k=5, name="train_top5_accuracy")
    metrics_train['total_perplexity'] = Perplexity(name="train_perplexity")

    # Training Loop
    best_validation_score = 0
    for epoch in range(epochs):
        print(f"Start Training of Epoch {epoch}")
        for sentence, labels, mask in ds_train:
            # sentence, labels, mask \in [batch_size, sentence_length-1]
            train_step(model=model, optimizer=optimizer, metrics=metrics_train, sentence=sentence, labels=labels, mask=mask, id2word=id2word)

        print(f"Start Validation of Epoch {epoch}")
        validation_score, _ = validate(model=model, dataset=ds_valid, id2word=id2word, step=optimizer.iterations)

    ckpt.step.assign_add(1)

    if validation_score > best_validation_score:
        # current validation score is better than previous -> store checkpoints
        best_validation_score = validation_score
        save_path = manager.save()
        print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")
        print(f"Validation Score {validation_score.numpy()}")

@tf.function
def train_step(model, optimizer, metrics, sentence, labels, mask, id2word):
    with tf.GradientTape() as tape:
        # feed sentence to model to calculate
        logits = model(sentence)
        preds = tf.nn.softmax(logits)

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
    metrics['loss'].update_state(loss)
    metrics['accuracy'].update_state(y_true=labels_masked, y_pred=logits_masked)
    metrics['top5_accuracy'].update_state(y_true=labels_masked, y_pred=logits_masked)
    metrics['total_perplexity'].update_state(y_true=labels_masked, y_pred=preds_masked)

    # log metrics to summary every summary_freq steps (aggregated over the last summary_freq steps)
    if tf.equal(optimizer.iterations % SUMMARY_FREQ, 0):

        tf.summary.scalar('train/loss', metrics['loss'].result(), step=optimizer.iterations)
        metrics['loss'].reset_states()

        tf.summary.scalar('train/accuracy', metrics['accuracy'].result(), step=optimizer.iterations)
        metrics['accuracy'].reset_states()

        tf.summary.scalar('train/top5_accuracy', metrics['top5_accuracy'].result(), step=optimizer.iterations)
        metrics['top5_accuracy'].reset_states()

        tf.summary.scalar('train/perplexity', metrics['total_perplexity'].result(), step=optimizer.iterations)
        metrics['total_perplexity'].reset_states()

        predicted_words = tf.math.argmax(logits, axis=2)

        pred_batch_text = format_to_text(words=predicted_words, mask=mask, id2word=id2word)
        label_batch_text = format_to_text(words=labels, mask=mask, id2word=id2word)
        stacked_text = tf.stack([label_batch_text, pred_batch_text], axis=1)
        stacked_text = tf.concat([tf.constant(['**Ground Truth**', '**Prediction (argmax)**'], shape=[1,2]), stacked_text], axis=0)
        tf.summary.text('train/text_gt_pred', stacked_text, step=optimizer.iterations)
