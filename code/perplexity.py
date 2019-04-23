import tensorflow as tf

class Perplexity(tf.metrics.Metric):

    def __init__(self, name='perplexity', **kwargs):
        super(Perplexity, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(dtype=tf.float64,name='perp_sum_log_probs', initializer='zeros')
        self.n = self.add_weight(dtype=tf.int32,name='perp_n', initializer='zeros')

        # pre define constant tensor [0,1,2,..., ] which is used at every update_state
        # as part of required index (think more efficient than creating new every time)
        self.range = tf.range((SENTENCE_LENGTH-1)*BATCH_SIZE, dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''

        Arguments:
            - y_true = masked labels for the true index of the word for every pos  ->
                        (y_true in \[n])
            - y_pred = masked probabilities (not logits!) for each
                        non-padding position in batch
                        (y_pred \in [n, VOCAB_SIZE])
        '''

        n = tf.shape(y_pred)[0]
        self.n.assign_add(n)

        # use only required slice of constant [0,1,2,..., n-1]
        range = self.range[:n]

        # build gather index by merging range [0,1,2, ...] with y_true
        indices = tf.stack([range, y_true], axis=1)

        # select for every sample the probability of the true word (label)
        probs = tf.gather_nd(params=y_pred, indices=indices) # probs \in [n]

        log_probs = log2(probs) # log_probs \in [n]

        sum_log_probs = tf.reduce_sum(log_probs) # sum_log_probs \in scalar

        self.sum.assign_add(tf.cast(sum_log_probs, dtype=tf.float64))

    def result(self):
        '''
        Computes the actual perplexity value.

        '''

        perp = tf.math.pow(tf.constant(2, dtype=tf.float64), -1/tf.cast(self.n, dtype=tf.float64) * self.sum)

        return perp

@tf.function
def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
