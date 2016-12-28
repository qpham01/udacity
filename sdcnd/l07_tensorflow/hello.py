"""
Simple tests of tensorflow's data interface.abs
"""
import tensorflow as tf

# Create TensorFlow object called hello_constant
HELLO_CONSTANT = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    OUTPUT = sess.run(HELLO_CONSTANT)
    print(OUTPUT)

STR = tf.placeholder(tf.string)

with tf.Session() as sess:
    # the feed_dict key STR has to be a placeholder tensor.
    OUTPUT = sess.run(STR, feed_dict={STR: 'Hello World'})
    print(OUTPUT)

X = tf.placeholder(tf.string)
Y = tf.placeholder(tf.int32)
Z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    # How multiple sensors are fed using feed_dict.
    OUTPUT = sess.run(Y, feed_dict={X: 'Test String', Y: 123, Z: 45.67})
    # Note that OUTPUT is the fetches parameter (Y in the above session run)
    print(OUTPUT)
