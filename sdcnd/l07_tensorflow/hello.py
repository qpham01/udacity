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
