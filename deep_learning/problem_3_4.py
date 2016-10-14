from read_data import *
from time import time
import tensorflow as tf

# The knobs
batch_size = 128
hidden_nodes = 2048
hidden_nodes_l2 = 512
starter_learn_rate = 0.5
reg_beta = 2e-3
num_steps = 5001

# Flow the data sets thru the layers
def train_three_layer(X, use_dropout):
    # Training computation.

    y1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    y2 = None
    if use_dropout:
        y1d = tf.nn.dropout(y1, input_keep_prob)
        y2 = tf.nn.relu(tf.matmul(y1d, W2) + b2)
    else:
        y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

    y3 = None
    if use_dropout:
        y2d = tf.nn.dropout(y2, train_keep_prob)
        y3 = tf.nn.relu(tf.matmul(y2d, W3) + b3)
    else:
        y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)

    return y3

# Graph with stochastic gradient descent
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    input_keep_prob = tf.placeholder("float")
    train_keep_prob = tf.placeholder("float")

    W1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes], stddev=0.03))
    b1 = tf.Variable(tf.zeros([hidden_nodes]))

    W2 = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes_l2], stddev=0.01))
    b2 = tf.Variable(tf.zeros([hidden_nodes_l2]))

    W3 = tf.Variable(tf.truncated_normal([hidden_nodes_l2, num_labels], stddev=0.01))
    b3 = tf.Variable(tf.zeros([num_labels]))

    # Computation
    y_train = train_three_layer(tf_train_dataset, True)
    y_valid = train_three_layer(tf_valid_dataset, True)
    y_test = train_three_layer(tf_test_dataset, True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_train, tf_train_labels))

    # L2 regularization for the fully connected parameters.
    # https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/mnist/convolutional.py
    regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) +
                  tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))
    # Add the regularization term to the loss.
    loss += reg_beta * regularizers

    loss_summary = tf.scalar_summary("loss", loss)

    global_step = tf.Variable(0)  # count the number of steps taken.
    learn_rate = tf.train.exponential_decay(starter_learn_rate, global_step, 500, 0.96)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(y_train)
valid_prediction = tf.nn.softmax(y_valid)
test_prediction = tf.nn.softmax(y_test)

# Let's run it
with tf.Session(graph=graph) as session:

    writer = tf.train.SummaryWriter("/tmp/notmnist_logs", session.graph_def)

    tf.initialize_all_variables().run()
    print("Initialized")

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, input_keep_prob : 0.9, train_keep_prob : 0.5}
        _, l, predictions, merged_summary, lr = session.run([optimizer, loss, train_prediction, merged, learn_rate], feed_dict=feed_dict)
        writer.add_summary(merged_summary, step)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(train_prediction.eval(
                feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, input_keep_prob : 1.0, train_keep_prob: 1.0}), batch_labels))
            print("Learn rate: ", lr)

            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph dependencies.
            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(feed_dict={input_keep_prob : 1.0, train_keep_prob: 1.0}), valid_labels))

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={input_keep_prob : 1.0, train_keep_prob: 1.0}), test_labels))