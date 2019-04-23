import tensorflow as tf

from global_variable import SPECIAL, SENTENCE_LENGTH

def _build_base_dataset(filename, vocab):

    # load dataset from text file
    dataset = tf.data.TextLineDataset(filename)

    # tokenize sentence
    dataset = dataset.map(lambda sentence: tf.strings.split([sentence], sep=' ').values,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #remove empty string words (Yeah it looks ugly)
    dataset=dataset.map(lambda sentence: tf.cond(tf.equal(tf.strings.length(sentence[-1]),tf.constant(0)),true_fn=lambda: sentence[:-1], false_fn=lambda: sentence),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # add <bos> and <eos>
    dataset = dataset.map(lambda sentence: tf.concat([[SPECIAL['bos']], sentence, [SPECIAL['eos']]], axis=0),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # filter out sentences longer than 30
    dataset = dataset.filter(lambda sentence: tf.shape(sentence)[0] <= SENTENCE_LENGTH)

    # pad all sentences to length 30 and add length of sentence
    dataset = dataset.map(lambda sentence: (tf.pad(sentence, [[0,SENTENCE_LENGTH - tf.shape(sentence)[0]]], mode='CONSTANT', constant_values=SPECIAL['pad']), tf.shape(sentence)[0]),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # map words to id
    dataset = dataset.map(lambda sentence, length: (vocab.lookup(sentence), length),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def build_dataset(filename, vocab):
    '''
    Builds a dataset from the given file and vocabulary

    Arguments:
        filename: Path to a corpus consisting of lines of words (sentences)
        vocab: Path to the vocabulary text file

    Returns:
        dataset: Parsed data set
    '''

    dataset = _build_base_dataset(filename, vocab)

    # map to sentence, labels and mask
    dataset = dataset.map(lambda sentence, length: (sentence[:-1],sentence[1:], tf.concat([tf.fill([length-1],True), tf.fill([SENTENCE_LENGTH-length],False)], axis=0)),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

    dataset = _build_base_dataset(filename, vocab)

    dataset = dataset.map(lambda sentence, length: (sentence[:-1], length-1),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
