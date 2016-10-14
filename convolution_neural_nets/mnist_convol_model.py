from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from time import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
To create this model, we're going to need to create a lot of weights and biases. One 
should generally initialize weights with a small amount of noise for symmetry breaking, 
and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to
initialize them with a slightly positive initial bias to avoid "dead neurons". Instead
of doing this repeatedly while we build the model, let's create two handy functions to 
do it for us.
"""

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

"""
TensorFlow also gives us a lot of flexibility in convolution and pooling operations. 
How do we handle the boundaries? What is our stride size? In this example, we're always 
going to choose the vanilla version. Our convolutions uses a stride of one and are zero 
padded so that the output is the same size as the input. Our pooling is plain old max 
pooling over 2x2 blocks. To keep our code cleaner, let's also abstract those operations 
into functions.
"""

def conv2d(x, W):
  with tf.device('/gpu:0'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)

def max_pool_2x2(x):
  with tf.device('/gpu:0'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

"""
Weights and biases
"""

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

def model(x, keep_prob):
  # Reshape
  x_image = tf.reshape(x, [-1,28,28,1])

  # First Convolutional Layer
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # Second Convolutional Layer
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
  h_pool2 = max_pool_2x2(h_conv2)

  # Densely Connected Layer 1
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
  # Readout Layer with Softmax
  y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  return y


y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

y_conv = model(x, keep_prob)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

config = config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

t0 = time()
with sess.as_default():
  for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      print('elapsed time %.1f seconds' % (time() - t0))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %.2f%%' % (100.0 * accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
  print('total elapsed time %.1f seconds' % (time() - t0))
