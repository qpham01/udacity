"""
Contains traffic_signs class which encapsulates traffic sign data
"""
import pickle
import numpy as np
from sklearn.utils import shuffle
from image_util import rgb_to_gray

class TrafficSignData:
    """
    Encapsulates traffic sign data
    """
    def __init__(self, train_data_file, test_data_file, use_grayscale=False, train_ratio=0.8):
        with open(train_data_file, mode='rb') as train_file:
            self.train = pickle.load(train_file)
        with open(test_data_file, mode='rb') as test_file:
            self.test = pickle.load(test_file)

        self.x_train_raw, self.y_train = self.train['features'], self.train['labels']
        self.x_test_raw, self.y_test = self.test['features'], self.test['labels']

        # How many data points for training and testing
        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        # What's the shape of an traffic sign image?
        self.image_shape = (self.x_train_raw.shape[1], self.x_train_raw.shape[2])

        # How many unique classes/labels there are in the dataset.
        self.n_classes = len(set(self.y_train))

        # Grayscale
        self.use_grayscale = use_grayscale
        if self.use_grayscale:
            self.colormap = 'gray'
            self.colordepth = 1
            self.x_train_img = rgb_to_gray(self.x_train_raw)
            self.x_test_img = rgb_to_gray(self.x_test_raw)
        else:
            self.colormap = 'viridis'
            self.colordepth = 3
            self.x_train_img = self.x_train_raw
            self.x_test_img = self.x_test_raw

        # Normalize by first subtracting 128 to center the color scale.
        self.x_train = np.subtract(self.x_train_img, 128.0)
        self.x_test = np.subtract(self.x_test_img, 128.0)

        # Then calculate the RGB means...
        train_mean = np.mean(self.x_train, axis=(0, 1, 2))
        test_mean = np.mean(self.x_test, axis=(0, 1, 2))

        # And subtract them from the train/test data to center the dataset at 0 RGB.
        self.x_train = np.subtract(self.x_train, train_mean)
        self.x_test = np.subtract(self.x_test, test_mean)

        # Shuffle training data.
        x_shuffled, y_shuffled = shuffle(self.x_train, self.y_train)

        # Divide randomized training data between training and validation
        self.train_ratio = train_ratio
        split_index = int(self.train_ratio * self.n_train)

        x_split = np.split(x_shuffled, [split_index, self.n_train])
        y_split = np.split(y_shuffled, [split_index, self.n_train])

        self.x_train = x_split[0]
        self.x_valid = x_split[1]
        self.y_train = y_split[0]
        self.y_valid = y_split[1]

    def print_data_dimensions(self):
        """
        Print out the shape and sizes of the data.
        """
        print("Number of training examples =", self.n_train)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)
        print("x_train shape", self.x_train.shape)
        print("y_train shape", self.y_train.shape)
        print("x_valid shape", self.x_valid.shape)
        print("y_valid shape", self.y_valid.shape)
