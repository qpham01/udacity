"""
Train Lenet to recognize MNIST data
"""
from time import sleep, time
import statistics as st
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('svg')
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from traffic_sign_model import version_description, l2_reg_strength, mu, sigma, evaluate, \
    LEARNING_RATE, EPOCHS, BATCH_SIZES, BETA, x, y, training_operation, \
    loss_operation, keep_prob, softmax, accuracy_operation

from traffic_sign_data import X_train, X_validation, X_test, y_train, y_validation, y_test, \
    train_validate_ratio, rgb_to_gray, use_grayscale
from traffic_sign_log import log_run_start, log_run_end

save_file = 'traffic_signs.ckpt'
saver = tf.train.Saver()

for batch_size in BATCH_SIZES:
    t0 = time()
    for epochs in EPOCHS:
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
            epoch_train_time_list = []
            training_losses = []
            validate_accuracies = []
            for i in range(epochs):
                t1 = time()
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
                epoch_train_time_list.append(time() - t1)
                training_losses.append(loss)
                validate_accuracies.append(validation_accuracy)

            saver.save(sess, save_file)
            print("Model saved")
            log_run_end(run_id, validation_accuracy, loss)
        print()
        print("Total training time:        {:.2f}".format(time() - t0))
        print("Mean epoch training time:   {:.2f}".format(st.mean(epoch_train_time_list)))
        print("Epoch training time stdev:  {:.2f}".format(st.pstdev(epoch_train_time_list)))
        print()

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

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
from matplotlib import image as mpimg
from math import ceil

def plot_images(X, rows, c_map):
    img_count = len(X)
    plot_cols = int(ceil(img_count / rows))
    f, img_plots = plt.subplots(rows, plot_cols, sharex=True)

    index = 0
    for r in range(rows):
        for c in range(plot_cols):
            img_plots[r, c].imshow(X[index].squeeze(), cmap=c_map)
            index += 1
            if index >= len(X):
                break

def plot_probabilities(X, rows, color):
    plot_cols = ceil(len(X) / rows)
    f, softmax_plots = plt.subplots(rows, plot_cols, sharex=True)
    index = 0
    for r in range(rows):
        for c in range(plot_cols):
            softmax_plots[r, c].plot(X[index], color)
            softmax_plots[r, c].set_ylim([0.0, 1.0])
            index += 1
            if index >= len(X):
                break
    plt.show()

X_extra = None
img_count = 10
for i in range(1, img_count + 1):
    fname = 'sign{:02d}.png'.format(i)
    img = mpimg.imread(fname)

    img = np.expand_dims(img, axis=0)
    print("image %s shape" % (i), img.shape)
    if i == 1:
        X_extra = img
    else:
        X_extra = np.append(X_extra, img, axis=0)
        
# grayscale
c_map = 'viridis'

row_count = 2
plot_images(X_extra, row_count, c_map)

if use_grayscale:
    X_extra = rgb_to_gray(X_extra)
    c_map = plt.cm.gray    
    
# normalize
X_extra = np.subtract(X_extra, 128.0)
extra_mean = np.mean(X_extra, axis=(0,1,2))
X_extra = np.subtract(X_extra, extra_mean)

row_count = 2
plot_images(X_extra, row_count, c_map)

#printing out some stats and plotting
print('The extra data shape is:', X_extra.shape)

# labels
y_extra = [14, 28, 13, 27, 17, 26, 2, 33, 5, 5]

saver = tf.train.Saver()

save_file = 'traffic_signs.ckpt'

def evaluate_extra(X_data, y_data, batch_size):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
        softmax_out = sess.run(softmax, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        # print ("Extra softmax:", softmax_out)
        print ("Extra softmax max index:", np.argmax(softmax_out, axis=1))
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return (total_accuracy / num_examples, softmax_out)

# Evaluate the model with test data
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    (test_accuracy, softmax_prob) = evaluate_extra(X_extra, y_extra, len(y_extra))
    print("Extra image test accuracy = {:.3f}".format(test_accuracy))

    row_count = 2
    plot_probabilities(softmax_prob, row_count, 'r')

### Run the predictions here.
### Feel free to use as many code cells as needed.
# Show some specified number of images and also save them to the signs directory to look at.
select_list = range(100, 120)
y_select = [i for i in y_test[select_list]]
print("Selected Labels:", y_select)

for i in range(0, len(select_list)):
    index = select_list[i]
    image = X_test[index].squeeze()

    image = np.expand_dims(image, axis=0)
    if use_grayscale:
        image = np.reshape(image, (-1, 32, 32, 1))
    print("image %s shape" % (i), image.shape)
    if i == 0:
        X_select = image
    else:
        X_select = np.append(X_select, image, axis=0)
    
# Evaluate the model with selected images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    (select_accuracy, softmax_prob_test) = evaluate_extra(X_select, y_select, len(y_select))
    print("Extra image test accuracy = {:.3f}".format(select_accuracy))

plot_images(X_select, 5, c_map)