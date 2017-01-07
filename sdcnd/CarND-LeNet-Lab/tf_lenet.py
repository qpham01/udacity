"""
TensorFlow implementation of the LeNet neural network
"""
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Convolutional. Input = 32x32x1. Output = 28x28x6.
    # truncated_normal inputs: (height, width, input_depth, output_depth)
    # height and width (5, 5, ...) are patch dimensions
    l1_weights = tf.Variable(tf.truncated_normal((5, 5, 1, 6), mean=mu, stddev=sigma))
    l1_bias = tf.Variable(tf.zeros(6))
    l1_strides = (1, 1, 1, 1)
    l1_padding = 'VALID'
    l1_conv = tf.nn.conv2d(x, l1_weights, l1_strides, l1_padding)
    l1_bias = tf.nn.bias_add(l1_conv, l1_bias)

    # Activation.
    l1_relu = tf.nn.relu(l1_bias)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    l1_pool = tf.nn.max_pool(l1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
        padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    # truncated_normal inputs: (height, width, input_depth, output_depth)
    # height and width (5, 5, ...) are patch dimensions
    l2_weights = tf.Variable(tf.truncated_normal((5, 5, 6, 16), mean=mu, stddev=sigma))
    l2_bias = tf.Variable(tf.zeros(16))
    l2_strides = (1, 1, 1, 1)
    l2_padding = 'VALID'
    l2_conv = tf.nn.conv2d(l1_pool, l2_weights, l2_strides, l2_padding)
    l2_bias = tf.nn.bias_add(l2_conv, l2_bias)

    # Activation.
    l2_relu = tf.nn.relu(l2_bias)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    l2_pool = tf.nn.max_pool(l2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
        padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    flat = tf.reshape(l2_pool, [-1, 400])

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    l3_weights = tf.Variable(tf.truncated_normal((400, 120), mean=mu, stddev=sigma))
    l3_bias = tf.Variable(tf.zeros(120))
    l3_linear = tf.add(tf.matmul(flat, l3_weights), l3_bias)

    # Activation.
    l3_relu = tf.nn.relu(l3_linear)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    l4_weights = tf.Variable(tf.truncated_normal((120, 84), mean=mu, stddev=sigma))
    l4_bias = tf.Variable(tf.zeros(84))
    l4_linear = tf.add(tf.matmul(l3_relu, l4_weights), l4_bias)

    # Activation.
    l4_relu = tf.nn.relu(l4_linear)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    l5_weights = tf.Variable(tf.truncated_normal((84, 10), mean=mu, stddev=sigma))
    l5_bias = tf.Variable(tf.zeros(10))
    logits = tf.add(tf.matmul(l4_relu, l5_weights), l5_bias)

    # logits = tf.nn.tanh(l5_linear)

    return logits

