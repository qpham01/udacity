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
    loss_operation, keep_prob, evaluate_extra, softmax, accuracy_operation

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

    i = 1
    fname = 'sign{:02d}.png'.format(i)
    img = mpimg.imread(fname)

    plt.figure(figsize=(1,1))
    plt.imshow(img)

    img_count = 10
    X_extra = np.expand_dims(img, axis=0)
    print("Image shape", X_extra.shape)
    for i in range(2, img_count + 1):
        fname = 'sign{:02d}.png'.format(i)
        img = mpimg.imread(fname)

        plt.figure(figsize=(1,1))
        plt.imshow(img)

        img = np.expand_dims(img, axis=0)
        print("image %s shape" % (i), img.shape)
        X_extra = np.append(X_extra, img, axis=0)


    # normalize
    if use_grayscale:
        X_extra = rgb_to_gray(X_extra)
        X_extra = np.subtract(X_extra, 128.0)

    extra_mean = np.mean(X_extra, axis=(0, 1, 2))
    X_extra = np.subtract(X_extra, extra_mean)

    #printing out some stats and plotting
    print('The extra data shape is:', X_extra.shape)

    # labels
    y_extra = [14, 28, 13, 27, 17, 26, 2, 33, 5, 5]

    saver = tf.train.Saver()

    save_file = 'traffic_signs.ckpt'

    # Evaluate the model with test data
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        test_accuracy = evaluate_extra(X_extra, y_extra, len(y_extra))
        print("Extra image test accuracy = {:.3f}".format(test_accuracy))
