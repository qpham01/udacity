"""
Demonstrate tensorflow arithmethics.abs
"""
# Solution is available in the other "solution.py" tab
import tensorflow as tf

X = tf.constant(10.0)
Y = tf.constant(2.0)
Z = tf.sub(tf.div(X, Y), 1.0)

with tf.Session() as sess:
    # How multiple sensors are fed using feed_dict.
    OUTPUT = sess.run(Z)
    # Note that OUTPUT is the fetches parameter (Y in the above session run)
    print(OUTPUT)
