from read_data import *
from time import time
import tensorflow as tf

beta = 0.002

# Two-layer training
def train_two_layer(data, weights1, biases1, weights2, biases2):  
  # two-layer training
  
  data_d = tf.nn.dropout(data, keep_prob=0.9)
  output1 = tf.matmul(data_d, weights1) + biases1
  relu1 = tf.nn.relu(output1)
  logits = tf.matmul(relu1, weights2) + biases2
  return logits


batch_size = 128
hidden_nodes = 1024
graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_nodes]))
  biases1 = tf.Variable(tf.zeros([hidden_nodes]))
  weights2 = tf.Variable(
    tf.truncated_normal([hidden_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  # training feed forward
  logits = train_two_layer(tf_train_dataset, weights1, biases1, weights2, biases2)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  #loss = np.add(loss, beta * tf.nn.l2_loss(weights1))
  loss = np.add(loss, beta * tf.nn.l2_loss(weights1))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(train_two_layer(tf_valid_dataset, weights1, biases1, weights2, biases2))
  test_prediction = tf.nn.softmax(train_two_layer(tf_test_dataset, weights1, biases1, weights2, biases2))

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  t0 = time()
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
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print('')
  print('Session elapsed time for %d steps:  %f seconds' % (num_steps, round(time() - t0, 3)))  
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))