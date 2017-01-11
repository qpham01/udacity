"""
TensorFlow implementation of the LeNet neural network
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from traffic_sign_data import use_grayscale

version_description = 'Added drop out, expanded network to 24 and 72 convolution depth, Adam optimizer with l2_reg_strength 1.0'

l2_reg_strength = 1.0

# Hyperparameters
mu = 0
sigma = 0.1

# Define color depth
color_depth = 3
if use_grayscale:
    color_depth = 1

# Weights and biases
# Layer 1: Input = 32x32xcolor_depth. Output = 28x28x24.
# truncated_normal inputs: (height, width, input_depth, output_depth)
# height and width (5, 5, ...) are patch dimensions
l1_depth = 24
l1_weights = tf.Variable(tf.truncated_normal((5, 5, color_depth, l1_depth), \
    mean=mu, stddev=sigma), name='w1')
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


# Set up hyper parameters

EPOCHS = [50]
BATCH_SIZES = [64]
LEARNING_RATE = 0.0001
BETA = 0.01

# Features and labels

x = tf.placeholder(tf.float32, (None, 32, 32, color_depth))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Training pipeline

keep_prob = tf.placeholder(tf.float32)
logits = LeNetTraffic(x, keep_prob)
softmax = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy + l2_reg_strength * BETA * \
    (tf.nn.l2_loss(l3_weights) + tf.nn.l2_loss(l4_weights) + tf.nn.l2_loss(l5_weights)))
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def evaluate_extra(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        softmax_out = sess.run(softmax, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        print("Extra softmax:", softmax_out)
        print("Extra softmax max index:", np.argmax(softmax_out, axis=1))
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
