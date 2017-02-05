"""
Defines deep learning model for behavioral cloning
"""
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data', 'train.p', "File name the pickled training data")
flags.DEFINE_string('model_name', 'model', "File name of the output model file")
flags.DEFINE_integer('epochs', 6, 'Training epoch count')

def load_data(data_file):
    """ Load pickled training data """
    data = pickle.load(open(data_file, "rb"))
    images = data["images"]
    labels = data["labels"]

    return images, labels

def add_conv_type1(model, depth, input_shape=None):
    """
    Create type 1 convolutional layers
    """
    if input_shape is not None:
        model.add(Convolution2D(depth, 5, 5, subsample=(2, 2), \
            input_shape=input_shape))
    else:
        model.add(Convolution2D(depth, 5, 5, subsample=(2, 2), \
            activation='relu', W_regularizer=l2(0.05)))

def add_conv_type2(model, depth):
    """
    Create type 2 convolutional layers
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

def print_layer_io_shapes(model):
    """
    Print input and output shapes for each layer in the model.
    """
    for i, _ in enumerate(model.layers):
        print("layer {} input: ".format(i), model.layers[i].input_shape)
        print("layer {} output:".format(i), model.layers[i].output_shape)

def main(_):
    """ Main method """
    print("Inputs:", FLAGS.train_data, FLAGS.model_name, FLAGS.epochs)

    input_shape = (160, 320, 1)

    model = make_cloning_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train model
    #model.fit_generator(image_generator(good_data), 1, nb_epoch=epochs)
    images, labels = load_data(FLAGS.train_data)
    model.fit(images, labels, nb_epoch=FLAGS.epochs, validation_split=0.2)

    with open(FLAGS.model_name + '.json', mode='w', encoding='utf8') as file:
        file.write(model.to_json())

    model.save(FLAGS.model_name + '.h5')

if __name__ == '__main__':
    tf.app.run()
