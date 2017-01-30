"""
Defines deep learning model for behavioral cloning
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.activations import relu, softmax

def add_conv_type1(model, depth, input_shape=None):
    """
    Create type 1 model
    """
    if input_shape is not None:
        model.add(Convolution2D(depth, 5, 5, subsample=(2, 2), input_shape=input_shape))
    else:
        model.add(Convolution2D(depth, 5, 5, subsample=(2, 2)))

def add_conv_type2(model, depth):
    """
    Create type 2 model
    """
    model.add(Convolution2D(depth, 3, 3, subsample=(1, 1)))

def make_cloning_model(input_shape=(66, 200, 3)):
    """
    Create a convolutional model and its layers
    """
    # Create the Sequential model
    model = Sequential()

    add_conv_type1(model, 24, input_shape)
    add_conv_type1(model, 36)
    add_conv_type1(model, 48)
    add_conv_type2(model, 64)
    add_conv_type2(model, 64)
    model.add(Flatten(input_shape=(1, 18, 64)))
    model.add(Dense(500))
    model.add(Dense(100))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = make_cloning_model(input_shape=(80,160,3))

def print_layer_io_shapes(model):
    """
    Print input and output shapes for each layer in the model.
    """
    for i, layer in enumerate(model.layers):
        print("layer {} input: ".format(i), model.layers[i].input_shape)
        print("layer {} output:".format(i), model.layers[i].output_shape)

print_layer_io_shapes(model)
