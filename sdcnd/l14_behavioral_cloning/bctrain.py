"""
Load in the images and labels and train the network.
"""
import pickle
import tensorflow as tf
from bcmodel import make_cloning_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_data', 'train.p', "Name of training data pickle file")
flags.DEFINE_string('model_name', 'model', "File name of the output model file")
flags.DEFINE_integer('epochs', 15, 'Training epoch count')

def load_data(data_file):
    data = pickle.load(open("train.p", "rb"))
    images = data["images"]
    labels = data["labels"]

    return images, labels

def main(_):
    epochs = FLAGS.epochs
    train_data = FLAGS.train_data
    model_name = FLAGS.model_name

    print ("Inputs:", train_data, model_name, epochs)

    images, labels = load_data(train_data)

    input_shape = images.shape[1:]

    model = make_cloning_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # train model
    model.fit(images, labels, nb_epoch=epochs)

    model.save(model_name + '.h5')

if __name__ == '__main__':
    tf.app.run()
