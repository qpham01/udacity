"""
TensorFlow implementation of the LeNet neural network
"""
import tensorflow as tf
from tensorflow.contrib.layers import flatten

version_description = 'Added drop out, expanded network to 24 and 72 convolution depth, Adam optimizer with l2_reg_strength 1.0'

l2_reg_strength = 1.0

# Hyperparameters
mu = 0
sigma = 0.1

# Weights and biases
# Layer 1: Input = 32x32x3. Output = 28x28x24.
# truncated_normal inputs: (height, width, input_depth, output_depth)
# height and width (5, 5, ...) are patch dimensions
l1_depth = 24
l1_weights = tf.Variable(tf.truncated_normal((5, 5, 3, l1_depth), mean=mu, stddev=sigma), \
    name='w1')
l1_bias = tf.Variable(tf.zeros(l1_depth), name='b1')

# Layer 2: Convolutional. Output = 10x10x72.
# truncated_normal inputs: (height, width, input_depth, output_depth)
# height and width (5, 5, ...) are patch dimensions
l2_depth = 72
l2_size = 5 * 5 * l2_depth
l2_weights = tf.Variable(tf.truncated_normal((5, 5, l1_depth, l2_depth), mean=mu, \
    stddev=sigma), name='w2')
l2_bias = tf.Variable(tf.zeros(l2_depth), name='b2')

# Layer 3: Fully Connected. Input = 5x5x72 = 1800. Output = 1000.
l3_size = 1000
l3_weights = tf.Variable(tf.truncated_normal((l2_size, l3_size), mean=mu, \
    stddev=sigma), name='w3')
l3_bias = tf.Variable(tf.zeros(l3_size), name='b3')

# Layer 4: Fully Connected. Input = 1000. Output = 500.
l4_size = 500
l4_weights = tf.Variable(tf.truncated_normal((l3_size, l4_size), mean=mu, stddev=sigma), \
    name='w4')
l4_bias = tf.Variable(tf.zeros(l4_size), name='b4')

# Layer 5: Fully Connected. Input = 500. Output = 43.
l5_size = 43
l5_weights = tf.Variable(tf.truncated_normal((l4_size, l5_size), mean=mu, stddev=sigma), \
    name='w5')
l5_bias = tf.Variable(tf.zeros(l5_size), name='b5')

def LeNetTraffic(x, keep_prob):

    # Convolutional layer 1
    l1_strides = (1, 1, 1, 1)
    l1_padding = 'VALID'
    l1_conv = tf.nn.conv2d(x, l1_weights, l1_strides, l1_padding)
    l1_biases = tf.nn.bias_add(l1_conv, l1_bias)

    # Activation.
    l1_relu = tf.nn.relu(l1_biases)

    # Pooling. Input = 28x28x24. Output = 14x14x24.
    l1_pool = tf.nn.max_pool(l1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
        padding='VALID')

    # Convolutional layer 2
    l2_strides = (1, 1, 1, 1)
    l2_padding = 'VALID'
    l2_conv = tf.nn.conv2d(l1_pool, l2_weights, l2_strides, l2_padding)
    l2_biases = tf.nn.bias_add(l2_conv, l2_bias)

    # Activation.
    l2_relu = tf.nn.relu(l2_biases)

    # Pooling. Input = 10x10x72. Output = 5x5x72.
    l2_pool = tf.nn.max_pool(l2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
        padding='VALID')

    # Flatten. Input = 5x5x72. Output = 1800.
    flat = tf.reshape(l2_pool, [-1, l2_size])

    # Layer 3: Fully Connected. Input = 1800. Output = 1000.
    l3_linear = tf.add(tf.matmul(flat, l3_weights), l3_bias)

    # Activation.
    l3_relu = tf.nn.relu(l3_linear)
    l3_drop = tf.nn.dropout(l3_relu, keep_prob)

    # Layer 4: Fully Connected. Input = 1000. Output = 500.
    l4_linear = tf.add(tf.matmul(l3_drop, l4_weights), l4_bias)

    # Activation.
    l4_relu = tf.nn.relu(l4_linear)
    l4_drop = tf.nn.dropout(l4_relu, keep_prob)

    # Layer 5: Fully Connected. Input = 500. Output = 43.
    logits = tf.add(tf.matmul(l4_drop, l5_weights), l5_bias)

    return logits

