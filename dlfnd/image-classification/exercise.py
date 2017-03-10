import numpy as np
import tensorflow as tf
import problem_unittests as tests
import helper

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # Implement Function
    return tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], image_shape[2]), name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # Implement Function
    return tf.placeholder(tf.float32, shape=(None, n_classes), name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # Implement Function
    return tf.placeholder(tf.float32, name='keep_prob')

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # Implement Function
    input_shape = x_tensor.get_shape().as_list()
    vshape = [conv_ksize[0], conv_ksize[1], input_shape[3], conv_num_outputs]
    random = tf.truncated_normal(vshape)
    filters = tf.Variable(random)
    conv_stride_shape = [1, conv_strides[0], conv_strides[1], 1]
    pool_kernel_shape = [1, pool_ksize[0], pool_ksize[1], 1]
    pool_stride_shape = [1, pool_strides[0], pool_strides[1], 1]
    conv = tf.nn.conv2d(x_tensor, filters, conv_stride_shape, "SAME");
    pool = tf.nn.max_pool(conv, pool_kernel_shape, pool_stride_shape, "SAME");
    return pool 

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # Implement Function
    shape = x_tensor.get_shape().as_list()
    dim = np.prod(shape[1:])
    return tf.reshape(x_tensor, [-1, dim])

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # Implement Function
    input_shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([input_shape[1], num_outputs]))
    biases = tf.Variable(tf.zeros([num_outputs]))
    return tf.add(tf.matmul(x_tensor, weights), biases)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # Implement Function
    input_shape = x_tensor.get_shape().as_list()
    weights = tf.Variable(tf.truncated_normal([input_shape[1], num_outputs]))
    biases = tf.Variable(tf.zeros([num_outputs]))
    return tf.add(tf.matmul(x_tensor, weights), biases)

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    out = conv2d_maxpool(x, 32, (3, 3), (1, 1), (2, 2), (1, 1))
    out = conv2d_maxpool(out, 16, (3, 3), (1, 1), (2, 2), (1, 1))
    out = conv2d_maxpool(out, 8, (3, 3), (1, 1), (2, 2), (1, 1))

    # Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    out = flatten(out)

    # Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    out = fully_conn(out, 2048)
    out = tf.nn.dropout(out, keep_prob)
    out = fully_conn(out, 512)
    out = tf.nn.dropout(out, keep_prob)
    out = fully_conn(out, 128)
    out = tf.nn.dropout(out, keep_prob)

    # Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(out, 10)

    # return output
    return out

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement 
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    output_cost = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
    output_accuracy = session.run(accuracy, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
    print("Cost: {:.0f}, Accuracy {:.3f}".format(output_cost, output_accuracy)) 

def run_tests():
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tf.reset_default_graph()
    tests.test_nn_image_inputs(neural_net_image_input)
    tests.test_nn_label_inputs(neural_net_label_input)
    tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_con_pool(conv2d_maxpool)
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_flatten(flatten)
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_fully_conn(fully_conn)
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    tests.test_output(output)
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """

    ##############################
    ## Build the Neural Network ##
    ##############################

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = neural_net_image_input((32, 32, 3))
    y = neural_net_label_input(10)
    keep_prob = neural_net_keep_prob_input()

    # Model
    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    tests.test_conv_net(conv_net)
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    x = neural_net_image_input((32, 32, 3))
    y = neural_net_label_input(10)
    tests.test_train_nn(train_neural_network)

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

#run_tests()

# TODO: Tune Parameters
epochs = 20
batch_size = 256
keep_probability = 0.8

def train_single_batch():
    """
    DON'T MODIFY ANYTHING IN THIS CELL
    """
    print('Checking the Training on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

train_single_batch()