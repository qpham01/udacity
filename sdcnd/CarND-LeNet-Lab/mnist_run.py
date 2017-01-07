"""
Train Lenet to recognize MNIST data
"""
import tensorflow as tf
from sklearn.utils import shuffle
from tf_lenet import LeNet
from mnist_data import X_train, X_validation, X_test, y_train, y_validation, y_test
from dldata import dllog as log

# Set up hyper parameters

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Features and labels

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

# Training pipeline

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Setup run logging

dl_run = "Lenet for MNIST"
dl_network = 'Lenet'
dl_model_file_path = 'tf_lenet.py'
dl_data = ('TF_MNIST', 'Train')
dl_environment = 'Default'
hyper_dict = { 'epochs' : EPOCHS, 'batch size' : BATCH_SIZE, 'learning rate' : \
    LEARNING_RATE }

# Train the model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")

    run_id = log.dl_run_start(dl_run, dl_network, dl_model_file_path, dl_data, hyper_dict)

    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Model saved")
    log.dl_run_end(run_id, validation_accuracy)

# Evaluate the model with test data
dl_data = ('TF_MNIST', 'Test')

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    run_id = log.dl_run_start(dl_run, dl_network, dl_model_file_path, dl_data, hyper_dict)

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    log.dl_run_end(run_id, test_accuracy)