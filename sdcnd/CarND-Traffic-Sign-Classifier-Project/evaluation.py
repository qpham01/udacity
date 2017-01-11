"""
Model evaluation
"""
import tensorflow as tf

from lenet_hyper import FEATURES, LABELS, KEEP_PROB, ONE_HOT_LABELS

# Model import ... change this to change the model
from lenet3_simple import LOGITS, SOFTMAX

# Evaluation tensor functions
CORRECT_PREDICTION = tf.equal(tf.argmax(LOGITS, 1), tf.argmax(ONE_HOT_LABELS, 1))
ACCURACY_OPERATION = tf.reduce_mean(tf.cast(CORRECT_PREDICTION, tf.float32))

def evaluate(features, labels, batch_size):
    """
    Evaluate known features and labels versus model output
    """
    num_examples = len(features)
    total_accuracy = 0
    sess = tf.get_default_session()
    softmax_out = []
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = features[offset:offset + batch_size], \
            labels[offset:offset + batch_size]
        (softmax, accuracy) = sess.run((SOFTMAX, ACCURACY_OPERATION), \
            feed_dict={FEATURES: batch_x, LABELS: batch_y, KEEP_PROB: 1.0})
        total_accuracy += (accuracy * len(batch_x))
        softmax_out.extend(softmax)
    return (softmax_out, total_accuracy / num_examples)
