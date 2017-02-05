"""
Load in the images and labels and train the network.
"""
import os
from pathlib import Path
from time import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from bcmodel import make_cloning_model
from image_util import rgb_to_gray

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'Simulator/Training04', "Name of training data pickle file")
flags.DEFINE_string('model_name', 'model', "File name of the output model file")
flags.DEFINE_integer('epochs', 5, 'Training epoch count')
flags.DEFINE_boolean('small_images', False, 'If true, use half-size images in img folder')
flags.DEFINE_integer('keep_interval', 5, 'The interval of zero angle data point to keep')
flags.DEFINE_string('train_data', 'train.p', 'Pickle file containing training data')

def load_data(data_file):
    data = pickle.load(open("train.p", "rb"))
    images = data["images"]
    labels = data["labels"]

    return images, labels

def read_lines_from_file(file_name):
    """ Read lines from a file """
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return lines

def parse_drive_log_line(line):
    """ Parse image file name and steering angle from a drive log line """
    splitted = line.split(',')
    if FLAGS.small_images:
        parts = [part.strip().replace('IMG', 'img') for part in splitted]
    else:
        parts = [part.strip() for part in splitted]
    return parts[0], float(parts[3])

def image_generator(data, zero_angle_keep_interval=5):
    # img_dict = {}
    images = None
    labels = None
    while True:
        t0 = time()
        count = 0
        if images is None:
            for img_fname, angle in data:
                img = rgb_to_gray(mpimg.imread(img_fname))
                img = np.expand_dims(img, axis=0)
                if images is None:
                    images = img
                else:
                    images = np.append(images, img, axis=0)
                label = np.array(angle)
                label = np.expand_dims(label, axis=0)
                if labels is None:
                    labels = label
                else:
                    labels = np.append(labels, label, axis=0)
                count += 1
                if count % 100 == 0:
                    print("processed {} items".format(count))
            yield (images, labels)

def generate_fit_data(drive_log):
    epochs = FLAGS.epochs
    model_name = FLAGS.model_name
    keep_index = 0
    lines = read_lines_from_file(drive_log)
    good_data = []
    keep_index = 0
    neg_count = 0
    pos_count = 0
    not_found_count = 0
    for line in lines:
        if 'center_' not in line:
            continue
        # if count % update_interval == 0:
        #    print("{} lines out of {} processed".format(count, line_count))
        fname, angle = parse_drive_log_line(line)

        # Decide to whether throw out or keep zero steering angle data.
        if angle == 0.0 and keep_index != FLAGS.keep_interval:
            keep_index += 1
            continue
        if angle == 0.0 and keep_index == FLAGS.keep_interval:
            keep_index = 0

        if (FLAGS.small_images and fname.startswith('img')) or \
            (not FLAGS.small_images and fname.startswith('IMG')):
            fname = os.path.join(FLAGS.sim_dir, fname)

        img_file = Path(fname)
        if img_file.is_file():
            if angle < 0:
                neg_count += 1
            if angle > 0:
                pos_count += 1
            good_data.append((fname, angle))
        else:
            not_found_count += 1
    print("Could not find {} file".format(not_found_count))
    print("Positive Count", pos_count)
    print("Negative Count", neg_count)
    return good_data

def main(_):
    print("Inputs:", FLAGS.data_dir, FLAGS.model_name, FLAGS.epochs)
    drive_log = os.path.join(FLAGS.data_dir, 'driving_log.csv')

    input_shape = (160, 320, 1)

    model = make_cloning_model(input_shape)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train model
    #model.fit_generator(image_generator(good_data), 1, nb_epoch=epochs)
    images, labels = load_data(FLAGS.train_data)
    model.fit(images, labels, nb_epoch=FLAGS.epochs)

    with open(FLAGS.model_name + '.json', mode='w', encoding='utf8') as f:
        f.write(model.to_json())

    model.save(FLAGS.model_name + '.h5')


if __name__ == '__main__':
    tf.app.run()
