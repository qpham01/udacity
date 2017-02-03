"""
Reads in and process the image data for training
"""
import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

SIM_DIR = 'Simulator/Training02'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('sim_dir', SIM_DIR, "The simulation directory with the training data")
flags.DEFINE_string('pickle_name', 'train.p', "Name of the output pickle file")
flags.DEFINE_integer('keep_interval', 5, 'The interval of zero angle data point to keep')

def read_lines_from_file(file_name):
    """ Read lines from a file """
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return lines

def parse_drive_log_line(line):
    """ Parse image file name and steering angle from a drive log line """
    splitted = line.split(',')
    parts = [part.strip().replace('IMG', 'img') for part in splitted]
    return parts[0], float(parts[3])

def read_drive_data(drive_log, zero_angle_keep_interval=5):
    """ Read the driving data from the log file and the driving images """
    lines = read_lines_from_file(drive_log)
    keep_index = 0
    images = None
    labels = []
    not_found_count = 0
    for line in lines:
        if 'center' not in line:
            continue
        fname, angle = parse_drive_log_line(line)

        if fname.startswith('img'):
            fname = os.path.join(FLAGS.sim_dir, fname)

        # Decide to whether throw out or keep zero steering angle data.
        if angle == 0.0 and keep_index != zero_angle_keep_interval:
            keep_index += 1
            continue
        if angle == 0.0 and keep_index == zero_angle_keep_interval:
            keep_index = 0

        img_file = Path(fname)
        if img_file.is_file():
            # file exists
            img = mpimg.imread(fname)
            img = np.expand_dims(img, axis=0)

            if images is None:
                images = img
            else:
                images = np.append(images, img, axis=0)
            labels.append(angle)
        else:
            not_found_count += 1
            print("Could not file file", fname)
    print("Could not find {} file".format(not_found_count))

    return images, labels

def main(_):
    print("Inputs:", FLAGS.sim_dir, FLAGS.pickle_name, FLAGS.keep_interval)
    drive_log = os.path.join(FLAGS.sim_dir, 'driving_log.csv')

    images, labels = read_drive_data(drive_log)
    data_pickle = {}
    data_pickle["images"] = images
    data_pickle["labels"] = labels
    pickle.dump(data_pickle, open(FLAGS.pickle_name, "wb"))

if __name__ == '__main__':
    tf.app.run()
