from read_data import *
import tensorflow as tf
from time import time

batch_size = 50
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  

from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data, keep_prob): 
    # Reshape
    # x_image = tf.reshape(data, [-1,28,28,1])

    # First Convolutional Layer
    h_conv1 = tf.nn.relu(conv2d(data, layer1_weights) + layer1_biases)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, layer2_weights) + layer2_biases)  
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer 1
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * depth])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, layer3_weights) + layer3_biases)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout Layer with Softmax
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, layer4_weights) + layer4_biases)
    return y  

  
  # Training computation.
  keep_prob = 0.5
  logits = model(tf_train_dataset, keep_prob)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

num_steps = 5001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  t0 = time()
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print ('Elapsed time %.1f seconds' % (time() - t0))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))