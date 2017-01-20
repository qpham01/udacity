"""
Build a Keras model
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.activations import relu, softmax

def check_layers(layers, true_layers):
    """
    Verify layer correctness.
    """
    assert len(true_layers) != 0, 'No layers found'
    for layer_i, _ in enumerate(layers):
        assert isinstance(true_layers[layer_i], layers[layer_i]), \
            'Layer {} is not a {} layer'.format(layer_i+1, layers[layer_i].__name__)
    assert len(true_layers) == len(layers), '{} layers found, should be {} layers'.\
        format(len(true_layers), len(layers))

def make_linear_model(input_shape=(32, 32, 3)):
    """
    Create a linear model and its layers
    """
    # Create the Sequential model
    model = Sequential()

    # Add a flatten layer
    model.add(Flatten(input_shape=input_shape))

    # Add a fully connected layer
    model.add(Dense(128))

    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a fully connected layer
    model.add(Dense(43))

    # Add a softmax activation layer
    model.add(Activation('softmax'))

    # STOP: Do not change the tests below. Your implementation should pass these tests.
    check_layers([Flatten, Dense, Activation, Dense, Activation], model.layers)

    assert model.layers[0].input_shape == (None, 32, 32, 3), \
        'First layer input shape is wrong, it should be (32, 32, 3)'
    assert model.layers[1].output_shape == (None, 128), \
        'Second layer output is wrong, it should be (128)'
    assert model.layers[2].activation == relu, \
        'Third layer not a relu activation layer'
    assert model.layers[3].output_shape == (None, 43), \
        'Fourth layer output is wrong, it should be (43)'
    assert model.layers[4].activation == softmax, \
        'Fifth layer not a softmax activation layer'
    print('Tests passed.')

    return model

def make_convolutional_model(input_shape=(32, 32, 3)):
    """
    Create a convolutional model and its layers
    """
    # Build a model
    # Create the Sequential model
    model = Sequential()

    # Add a convolution layer with a 3x3 kernel and valid padding
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=(32, 32, 3)))

    # Add a pooling layer with 2x2 pool size, strides defaulting to pool size.
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))

    # Add a dropout layer
    model.add(Dropout(0.5))

    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a convolution layer with a 3x3 kernel and valid padding
    model.add(Convolution2D(72, 5, 5, border_mode='valid'))

    # Add a pooling layer with 2x2 pool size, strides defaulting to pool size.
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))

    # Add a dropout layer
    model.add(Dropout(0.5))

    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a flatten layer
    model.add(Flatten(input_shape=(5, 5, 72)))

    # Add a fully connected layer
    model.add(Dense(1000))

    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a fully connected layer
    model.add(Dense(500))

    # Add a ReLU activation layer
    model.add(Activation('relu'))

    # Add a fully connected layer
    model.add(Dense(43))

    # Add a softmax activation layer
    model.add(Activation('softmax'))

    return model
