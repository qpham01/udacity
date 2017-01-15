import pickle
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from time import time
from alexnet import AlexNet

# Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

# Load the feature data to the variable X_train
X_train, y_train = data['features'], data['labels']
n_train = len(y_train)

# Shuffle the data.
X_shuffled, y_shuffled = shuffle(X_train, y_train)

# Split data into training and validation sets.
TRAIN_RATIO = 0.9
split_index = int(TRAIN_RATIO * n_train)

X_split = np.split(X_shuffled, [split_index, n_train])
y_split = np.split(y_shuffled, [split_index, n_train])

X_train = X_split[0]
X_valid = X_split[1]
y_train = y_split[0]
y_valid = y_split[1]

# Define placeholders and resize operation.
# One Hot encode the labels to the variable y_one_hot
y = tf.placeholder(tf.int64, (None))
one_hot = tf.one_hot(y, 43)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_one_hot = sess.run(one_hot, feed_dict={y: y_train})
# y_one_hot_test = sess.run(one_hot, feed_dict={label_placeholder: y_test})

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# Resize the images so they can be fed into AlexNet.
# HINT: Use `tf.image.resize_images` to resize the images
resized = tf.image.resize_images(x, [227, 227])

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
nb_classes = 43
fc7 = AlexNet(resized, feature_extract=True)
# TODO: Define a new fully connected layer followed by a softmax activation to classify
# the traffic signs. Assign the result of the softmax activation to `probs` below.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
mu = 0.0
sigma = 0.01
fc8W = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))
fc8b = tf.Variable(tf.zeros(shape[1]))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

learning_rate = 0.0001 # 10% worse validation accuracy (86% instead of 96%) than default.

# Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# Using sparse_softmax... means no need to use one_hot with logits, just the labels directly
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
train_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])
correct_prediction = tf.equal(tf.argmax(logits, 1), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
train_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

preds = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
"""

# Train and evaluate the feature extraction model.
def train_in_batches(sess, train_inputs, train_labels, batch_size):
    """
    Train network in batches of data
    """
    train_inputs, train_labels = shuffle(train_inputs, train_labels)
    num_examples = len(train_inputs)
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = train_inputs[offset:end], train_labels[offset:end]
        sess.run(train_operation, feed_dict={x: batch_x, y: batch_y})
    # Return cost of the last batch
    cost = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y})
    return cost

def evaluate_in_batches(sess, inputs, labels, batch_size):
    """
    Evaluate inputs in batches
    """
    num_examples = len(inputs)
    total_accuracy = 0

    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = inputs[offset:offset + batch_size], labels[offset:offset + \
            batch_size]
        accuracy = sess.run(accuracy_operation, \
            feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def train_with_validate(sess, train_inputs, train_labels, valid_inputs, valid_labels, \
    train_epochs, batch_size):
    """
    Train the network with validation
    """

    print("Training...")
    print()

    time0 = time()
    for i in range(train_epochs):
        time1 = time()
        cost = train_in_batches(sess, train_inputs, train_labels, batch_size)

        print("Epoch:", '%04d' % (i), "Cost =", "{:.9f}".format(cost), \
            "Time elapsed: {:.2f}".format(time() - time1))

        valid_accuracy = evaluate_in_batches(sess, valid_inputs, valid_labels, batch_size)
        print("Validation Accuracy = {:.3f}".format(valid_accuracy))
        print()
    print("Training Finished!")
    print("Total training time:", "{:.2f}".format(time() - time0))

epochs = 5
batch_size = 128

with tf.Session() as sess1:
    sess1.run(tf.global_variables_initializer())
    train_with_validate(sess1, X_train, y_train, X_valid, y_valid, epochs, batch_size)
