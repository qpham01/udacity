"""
TensorFlow implementation of the LeNet neural network
"""
from math import ceil
import tensorflow as tf
from tensorflow.contrib.layers import flatten

VERSION_DESCRIPTION = 'small network for traffic sign classification, l1 depth 25, l2 size 2500'

# Define color depth
COLOR_DEPTH = 1

# Hyperparameters

# initial weight distribution
MU = 0
SIGMA = 0.1

# L2 regularization
L2_REG_STRENGTH = 1.0
BETA = 0.01

# traing hyper parameters
EPOCHS = [100]
BATCH_SIZES = [64]
LEARNING_RATE = 0.0001

# Weights and biases
# Layer 1: Input = 32x32xCOLOR_DEPTH. Output = 14x14xL1_DEPTH.
# truncated_normal inputs: (height, width, input_depth, output_depth)
# height and width (5, 5, ...) are patch dimensions
L1_IN_DIM = 32
L1_DEPTH = 25
L1_CONV_KERNEL = 5
L1_CONV_STRIDE = 1.
L1_POOL_KERNEL = 2
L1_POOL_STRIDE = 2.
L1_PAD = 0
L1_OUT_CONV = int(ceil((L1_IN_DIM - L1_CONV_KERNEL + 2 * L1_PAD) / L1_CONV_STRIDE + 1))
print("L1_OUT_CONV", L1_OUT_CONV)
L1_OUT_DIM = int(ceil((L1_OUT_CONV - L1_POOL_KERNEL + 1) / L1_POOL_STRIDE))
print("L1_OUT_DIM", L1_OUT_DIM)
L1_SIZE = L1_OUT_DIM * L1_OUT_DIM * L1_DEPTH
L1_W = tf.Variable(tf.truncated_normal((L1_CONV_KERNEL, L1_CONV_KERNEL, COLOR_DEPTH, \
    L1_DEPTH), mean=MU, stddev=SIGMA), name='w1')
L1_B = tf.Variable(tf.zeros(L1_DEPTH), name='b1')

# Layer 2: Fully Connected. Input = 14x14xL1_DEPTH. Output = 400.
L2_SIZE = 2500
L2_W = tf.Variable(tf.truncated_normal((L1_SIZE, L2_SIZE), mean=MU, \
    stddev=SIGMA), name='w2')
L2_B = tf.Variable(tf.zeros(L2_SIZE), name='b2')

# Layer 3: Fully Connected. Input = 500. Output = 43.
L3_SIZE = 43
L3_W = tf.Variable(tf.truncated_normal((L2_SIZE, L3_SIZE), mean=MU, stddev=SIGMA), \
    name='w3')
L3_B = tf.Variable(tf.zeros(L3_SIZE), name='b3')

def lenet_traffic(features, keep_prob):
    """
    Define simple Lenet-like model with one convolution layer and three fully
    connected layers.
    """
    # Convolutional layer 1
    l1_strides = (1, 1, 1, 1)
    l1_padding = 'VALID'
    l1_conv = tf.nn.conv2d(features, L1_W, l1_strides, l1_padding)
    l1_biases = tf.nn.bias_add(l1_conv, L1_B)

    # Activation.
    l1_relu = tf.nn.relu(l1_biases)

    # Pooling. Input = 28x28xL1_DEPTH. Output = 14x14xL1_DEPTH.
    l1_pool = tf.nn.max_pool(l1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
        padding='VALID')

    # Flatten. Input = 14x14xL1_DEPTH. Output = L1_SIZE.
    flat = flatten(l1_pool)
    print("Flatten dimensions:", flat.get_shape())

    # Layer 2: Fully Connected. Input = L1_SIZE. Output = L2_SIZE.
    l2_linear = tf.add(tf.matmul(flat, L2_W), L2_B)

    # Activation.
    l2_relu = tf.nn.relu(l2_linear)
    l2_drop = tf.nn.dropout(l2_relu, keep_prob)

    # Layer 3: Fully Connected. Input = 500. Output = 43.
    return tf.add(tf.matmul(l2_drop, L3_W), L3_B)

# Features and labels

FEATURES = tf.placeholder(tf.float32, (None, 32, 32, COLOR_DEPTH))
LABELS = tf.placeholder(tf.int32, (None))
ONE_HOT_LABELS = tf.one_hot(LABELS, 43)

# Training pipeline

KEEP_PROB = tf.placeholder(tf.float32)
LOGITS = lenet_traffic(FEATURES, KEEP_PROB)
SOFTMAX = tf.nn.softmax(LOGITS)
CROSS_ENTROPY = tf.nn.softmax_cross_entropy_with_logits(LOGITS, ONE_HOT_LABELS)
LOSS_OPERATION = tf.reduce_mean(CROSS_ENTROPY + L2_REG_STRENGTH * BETA * \
    (tf.nn.l2_loss(L2_W) + tf.nn.l2_loss(L3_W) + tf.nn.l2_loss(L3_W)))
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
TRAIN_OPERATION = OPTIMIZER.minimize(LOSS_OPERATION)

# Model evaluation

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
