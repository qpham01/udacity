# Solution is available in the other "solution.py" tab
import tensorflow as tf
import numpy as np

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
cross_entropy = -tf.reduce_sum(tf.mul(one_hot_data, tf.log(softmax_data)))

with tf.Session() as sess:
    # Feed in the logit data
    output = sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data})  
    print(output)