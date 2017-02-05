"""
Reads in and process the image data for training
"""
import os
from pathlib import Path
from time import time
import pickle
import numpy as np
import cv2
import matplotlib.image as mpimg
import tensorflow as tf
from image_util import rgb_to_gray

SIM_DIR = 'Simulator/Training06'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('sim_dir', SIM_DIR, "The simulation directory with the training data")
flags.DEFINE_string('pickle_name', 'train.p', "Name of the output pickle file")
flags.DEFINE_integer('keep_interval', 5, 'The interval of zero angle data point to keep')
flags.DEFINE_boolean('small_images', False, 'If true, use half-size images in img folder')

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

def read_drive_data(images, labels, fnames, drive_log, zero_angle_keep_interval=5):
    """ Read the driving data from the log file and the driving images """
    lines = read_lines_from_file(drive_log)
    keep_index = 0
    not_found_count = 0
    line_count = len(lines)
    update_interval = 100
    count = 0
    neg_count = 0
    pos_count = 0
    skip_count = 0
    for line in lines:
        if 'center' not in line:
            continue
        fname, angle = parse_drive_log_line(line)

        if (FLAGS.small_images and fname.startswith('img')) or \
            (not FLAGS.small_images and fname.startswith('IMG')):
            fname = os.path.join(FLAGS.sim_dir, fname)
        if fname in fnames:
            skip_count += 1
            if skip_count % update_interval == 0:
                print("Skipped {} lines".format(skip_count))
            continue

        # Decide to whether throw out or keep zero steering angle data.
        if angle == 0.0 and keep_index != zero_angle_keep_interval:
            keep_index += 1
            continue
        if angle == 0.0 and keep_index == zero_angle_keep_interval:
            keep_index = 0

        count += 1
        if count % update_interval == 0:
            print("{} lines out of {} processed".format(count, line_count))

        img_file = Path(fname)
        if img_file.is_file():
            # file exists
            # img = cv2.imread(fname)
            img = rgb_to_gray(mpimg.imread(fname))
            img = np.expand_dims(img, axis=0)

            if images is None:
                images = img
            else:
                images = np.append(images, img, axis=0)
            labels.append(angle)
            fnames.append(fname)
            if angle < 0:
                neg_count += 1
            if angle > 0:
                pos_count += 1
        else:
            not_found_count += 1
            print("Could not file file", fname)
    print("Could not find {} file".format(not_found_count))
    print("Positive Count", pos_count)
    print("Negative Count", neg_count)
    return images, labels, fnames

def main(_):
    print("Inputs:", FLAGS.sim_dir, FLAGS.pickle_name, FLAGS.keep_interval)
    drive_log = os.path.join(FLAGS.sim_dir, 'driving_log.csv')

    images = None
    labels = []
    fnames = []
    pickle_file = Path(FLAGS.pickle_name)
    if pickle_file.is_file():
        data_pickle = pickle.load(open(FLAGS.pickle_name, "rb"))
        images = data_pickle['images']
        labels = data_pickle['labels']
        fnames = data_pickle['fnames']

    images, labels, fnames = read_drive_data(images, labels, fnames, drive_log, FLAGS.keep_interval)
    print("Shape of training images", images.shape)
    data_pickle = {}
    data_pickle["images"] = images
    data_pickle["labels"] = labels
    data_pickle["fnames"] = fnames
    pickle.dump(data_pickle, open(FLAGS.pickle_name, "wb"))

if __name__ == '__main__':
    tf.app.run()
