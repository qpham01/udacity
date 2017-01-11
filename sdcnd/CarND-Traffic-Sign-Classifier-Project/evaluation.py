"""
Model evaluation
"""
import tensorflow as tf

from lenet_hyper import FEATURES, LABELS, KEEP_PROB, ONE_HOT_LABELS

# Evaluation tensor functions

def evaluate(features, labels, logits, softmax, batch_size):
    """
    Evaluate known features and labels versus model output
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(ONE_HOT_LABELS, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_examples = len(features)
    total_accuracy = 0
    sess = tf.get_default_session()
    softmax_all = []
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = features[offset:offset + batch_size], \
            labels[offset:offset + batch_size]
        (softmax_out, accuracy) = sess.run((softmax, accuracy_operation), \
            feed_dict={FEATURES: batch_x, LABELS: batch_y, KEEP_PROB: 1.0})
        total_accuracy += (accuracy * len(batch_x))
        softmax_all.extend(softmax_out)
    return (softmax_all, total_accuracy / num_examples)
