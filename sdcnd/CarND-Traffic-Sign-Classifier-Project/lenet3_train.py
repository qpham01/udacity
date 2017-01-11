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
from sklearn.utils import shuffle
from lenet5_simple import VERSION_DESCRIPTION, L2_REG_STRENGTH, MU, SIGMA, \
    LEARNING_RATE, EPOCHS, BATCH_SIZES, BETA, FEATURES, LABELS, TRAIN_OPERATION, \
    LOSS_OPERATION, KEEP_PROB, evaluate
from traffic_sign_data import X_TRAIN, X_VALID, X_TEST, Y_TRAIN, Y_VALID, \
    Y_TEST, TRAIN_RATIO, USE_GRAYSCALE, rgb_to_gray
from traffic_sign_log import log_run_start, log_run_end
from plots import plot_images, plot_probabilities

SAVE_FILE = 'lenet5_simple.ckpt'
SAVER = tf.train.Saver()

for batch_size in BATCH_SIZES:
    t0 = time()
    for epochs in EPOCHS:
        # Train the model
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_TRAIN)

            print("Training...")
            print()

            # Setup run logging
            hyper_dict = {'epochs' : epochs, 'batch size' : batch_size, 'learning rate' : \
                LEARNING_RATE, 'train validate ratio' : TRAIN_RATIO, 'l2 strength': \
                L2_REG_STRENGTH, 'beta': BETA, 'mu': MU, 'sigma': SIGMA}

            run_id = log_run_start('Train', hyper_dict, VERSION_DESCRIPTION)
            epoch_train_time_list = []
            training_losses = []
            validate_accuracies = []
            for i in range(epochs):
                t1 = time()
                X_TRAIN, Y_TRAIN = shuffle(X_TRAIN, Y_TRAIN)
                for offset in range(0, num_examples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = X_TRAIN[offset:end], Y_TRAIN[offset:end]
                    _, loss = sess.run((TRAIN_OPERATION, LOSS_OPERATION), \
                        feed_dict={FEATURES: batch_x, LABELS: batch_y, KEEP_PROB: 0.5})

                if TRAIN_RATIO < 1.0:
                    _, validation_accuracy = evaluate(X_VALID, Y_VALID, batch_size)
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

            SAVER.save(sess, SAVE_FILE)
            print("Model saved")
            log_run_end(run_id, validation_accuracy, loss)
        print()
        print("Total training time:        {:.2f}".format(time() - t0))
        print("Mean epoch training time:   {:.2f}".format(st.mean(epoch_train_time_list)))
        print("Epoch training time stdev:  {:.2f}".format(st.pstdev(epoch_train_time_list)))
        print()

    # Evaluate the model with test data
    with tf.Session() as sess:
        SAVER.restore(sess, tf.train.latest_checkpoint('.'))

        run_id = log_run_start('Test', hyper_dict, VERSION_DESCRIPTION,)

        _, TEST_ACCURACY = evaluate(X_TEST, Y_TEST, batch_size)
        print("Test Accuracy = {:.3f}".format(TEST_ACCURACY))

        log_run_end(run_id, TEST_ACCURACY, 0.0)
    sess.close()

    # A delay to make sure there's a second of difference between timestamps of runs
    sleep(1.5)

X_EXTRA = None
EXTRA_COUNT = 10
for i in range(1, EXTRA_COUNT + 1):
    fname = 'sign{:02d}.png'.format(i)
    img = mpimg.imread(fname)

    img = np.expand_dims(img, axis=0)
    print("image %s shape" % (i), img.shape)
    if i == 1:
        X_EXTRA = img
    else:
        X_EXTRA = np.append(X_EXTRA, img, axis=0)

# grayscale
C_MAP = 'viridis'

ROW_COUNT = 2
plot_images(X_EXTRA, ROW_COUNT, C_MAP)

if USE_GRAYSCALE:
    X_EXTRA = rgb_to_gray(X_EXTRA)
    C_MAP = 'gray'

# normalize
X_EXTRA = np.subtract(X_EXTRA, 128.0)
EXTRA_MEAN = np.mean(X_EXTRA, axis=(0, 1, 2))
X_EXTRA = np.subtract(X_EXTRA, EXTRA_MEAN)

plot_images(X_EXTRA, ROW_COUNT, C_MAP)

#printing out some stats and plotting
print('The extra data shape is:', X_EXTRA.shape)

# labels
Y_EXTRA = [14, 28, 13, 27, 17, 26, 2, 33, 5, 5]

SAVER = tf.train.Saver()

# Evaluate the model with test data
with tf.Session() as sess:
    SAVER.restore(sess, tf.train.latest_checkpoint('.'))

    (SOFTMAX_PROB, TEST_ACCURACY) = evaluate(X_EXTRA, Y_EXTRA, len(Y_EXTRA))
    print("Extra image test accuracy = {:.3f}".format(TEST_ACCURACY))

    ROW_COUNT = 2
    plot_probabilities(SOFTMAX_PROB, ROW_COUNT, 'r')

### Run the predictions here.
### Feel free to use as many code cells as needed.
# Show some specified number of images and also save them to the signs directory to look at.
SELECT_LIST = range(100, 120)
Y_SELECT = [i for i in Y_TEST[SELECT_LIST]]
print("Selected Labels:", Y_SELECT)

for i in range(0, len(SELECT_LIST)):
    index = SELECT_LIST[i]
    image = X_TEST[index].squeeze()

    image = np.expand_dims(image, axis=0)
    if USE_GRAYSCALE:
        image = np.reshape(image, (-1, 32, 32, 1))
    print("image %s shape" % (i), image.shape)
    if i == 0:
        X_select = image
    else:
        X_select = np.append(X_select, image, axis=0)

# Evaluate the model with selected images
with tf.Session() as sess:
    SAVER.restore(sess, tf.train.latest_checkpoint('.'))

    (SOFTMAX_PROB_TEST, SELECT_ACCURACY) = evaluate(X_select, Y_SELECT, len(Y_SELECT))
    print("Extra image test accuracy = {:.3f}".format(SELECT_ACCURACY))

# plot_images(X_select, 5, c_map)
