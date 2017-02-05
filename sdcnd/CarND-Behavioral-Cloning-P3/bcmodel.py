"""
Defines deep learning model for behavioral cloning
"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.activations import relu, softmax
from keras.regularizers import l2

def add_conv_type1(model, depth, input_shape=None):
    """
    Create type 1 model
    """
    if input_shape is not None:
        model.add(Convolution2D(depth, 5, 5, subsample=(2, 2), \
            input_shape=input_shape))
    else:
        model.add(Convolution2D(depth, 5, 5, subsample=(2, 2), \
            activation='relu', W_regularizer=l2(0.05)))

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
    print("input shape", input_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x / 128. - 1., output_shape=input_shape, input_shape=input_shape))
    add_conv_type1(model, 12, input_shape)
    add_conv_type1(model, 18)
    add_conv_type1(model, 24)
    add_conv_type2(model, 30)
    add_conv_type2(model, 30)
    model.add(Flatten(input_shape=(13, 33, 30)))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

model = make_cloning_model(input_shape=(160, 320, 1))

def print_layer_io_shapes(model):
    """
    Print input and output shapes for each layer in the model.
    """
    for i, layer in enumerate(model.layers):
        print("layer {} input: ".format(i), model.layers[i].input_shape)
        print("layer {} output:".format(i), model.layers[i].output_shape)

print_layer_io_shapes(model)
