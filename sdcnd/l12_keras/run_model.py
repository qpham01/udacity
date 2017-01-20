"""
Get the data, build the model, train model and evaluate accuracy.
"""
import pickle
import tensorflow as tf
from keras.optimizers import Adam
from data_prepare import prepare_training_data, normalize
from keras_model import make_linear_model, make_convolutional_model

def run_models():
    """
    Run the two models
    """
    (x_normalized, y_one_hot, label_placeholder, one_hot) = prepare_training_data()

    # Make the model
    model = make_linear_model()

    # Compile and train the model here.
    # Configures the learning process and metrics
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    # Train the model
    # History is a record of training loss and metrics
    history = model.fit(x_normalized, y_one_hot, batch_size=128, nb_epoch=10, \
        validation_split=0.2, verbose=True)

    # Calculate test score
    test_score = model.evaluate(x_normalized, y_one_hot)
    print()
    print("Test score: {}".format(test_score[1]))

    assert model.loss == 'categorical_crossentropy', \
        'Not using categorical_crossentropy loss function'
    assert isinstance(model.optimizer, Adam), 'Not using adam optimizer'
    assert len(history.history['acc']) == 10, \
        'You\'re using {} epochs when you need to use 10 epochs.'.format(len(history.\
        history['acc']))
    assert history.history['acc'][-1] > 0.92, \
        'The training accuracy was: %.3f. It shoud be greater than 0.92' % history.\
        history['acc'][-1]
    assert history.history['val_acc'][-1] > 0.85, \
        'The validation accuracy is: %.3f. It shoud be greater than 0.85' % history.\
        history['val_acc'][-1]
    print('Tests passed.')

    # Make the model
    model = make_convolutional_model()

    # Compile and train the model here.
    # Configures the learning process and metrics
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    # Train the model
    # History is a record of training loss and metrics
    history = model.fit(x_normalized, y_one_hot, batch_size=128, nb_epoch=10, \
        validation_split=0.2, verbose=True)

    # Calculate test score
    test_score = model.evaluate(x_normalized, y_one_hot)
    print()
    print("Validation accuracy:", history.history['val_acc'][-1])

    # Load test data
    with open('test.p', 'rb') as file_handle:
        test_data = pickle.load(file_handle)
    x_test, y_test = test_data['features'], test_data['labels']

    # Preprocess data & one-hot encode the labels
    x_test_normalized = normalize(x_test)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        y_test_one_hot = sess.run(one_hot, feed_dict={label_placeholder: y_test})

    # Evaluate model on test data
    # Calculate test accuracy
    _, test_accuracy = model.evaluate(x_test_normalized, y_test_one_hot)
    print()
    print("Test accuracy:", test_accuracy)

run_models()
