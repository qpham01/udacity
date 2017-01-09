"""
Train Lenet to recognize MNIST data
"""
from time import sleep
import tensorflow as tf
from sklearn.utils import shuffle
from traffic_sign_model import LeNetTraffic, version_description, l2_reg_strength, \
    l1_weights, l2_weights, l3_weights, l4_weights, l5_weights, mu, sigma
from traffic_sign_data import X_train, X_validation, X_test, y_train, y_validation, y_test, \
    train_validate_ratio
from traffic_sign_log import log_run_start, log_run_end

# Set up hyper parameters

EPOCHS = [100, 100, 100]
BATCH_SIZES = [64]
LEARNING_RATE = 0.0001
BETA = 0.01

# Features and labels

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Training pipeline

keep_prob = tf.placeholder(tf.float32)
logits = LeNetTraffic(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy + l2_reg_strength * BETA * \
    (tf.nn.l2_loss(l3_weights) + tf.nn.l2_loss(l4_weights) + tf.nn.l2_loss(l5_weights)))
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

save_file = 'traffic_signs.ckpt'

for epochs in EPOCHS:
    for batch_size in BATCH_SIZES:
        # Train the model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")

            # Setup run logging
            hyper_dict = {'epochs' : epochs, 'batch size' : batch_size, 'learning rate' : \
                LEARNING_RATE, 'train validate ratio' : train_validate_ratio, 'l2 strength': \
                l2_reg_strength, 'beta': BETA, 'mu': mu, 'sigma': sigma}

            run_id = log_run_start('Train', hyper_dict, version_description)

            print()
            for i in range(epochs):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    _, loss = sess.run((training_operation, loss_operation), \
                        feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                if train_validate_ratio < 1.0:
                    validation_accuracy = evaluate(X_validation, y_validation, batch_size)
                else:
                    validation_accuracy = None
                print("EPOCH {} ...".format(i+1))
                print("Training Loss = {:.3f}".format(loss))
                if validation_accuracy is not None:
                    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            saver.save(sess, save_file)
            print("Model saved")
            log_run_end(run_id, validation_accuracy, loss)

        # Evaluate the model with test data
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('.'))

            run_id = log_run_start('Test', hyper_dict, version_description,)

            test_accuracy = evaluate(X_test, y_test, batch_size)
            print("Test Accuracy = {:.3f}".format(test_accuracy))

            log_run_end(run_id, test_accuracy, 0.0)
        sess.close()

        # A delay to make sure there's a second of difference between timestamps of runs
        sleep(1.5)

