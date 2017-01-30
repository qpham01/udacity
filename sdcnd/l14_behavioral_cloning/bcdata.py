"""
Reads in and process the image data for training
"""
import pickle
import numpy as np
import matplotlib.image as mpimg

DRIVE_LOG = '/home/quoc/Simulator/Training02/driving_log.csv'

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
    for line in lines:
        fname, angle = parse_drive_log_line(line)

        # Decide to whether throw out or keep zero steering angle data.
        if angle == 0.0 and keep_index != zero_angle_keep_interval:
            keep_index += 1
            continue
        if angle == 0.0 and keep_index == zero_angle_keep_interval:
            keep_index = 0

        img = mpimg.imread(fname)
        img = np.expand_dims(img, axis=0)

        if images is None:
            images = img
        else:
            images = np.append(images, img, axis=0)
        labels.append(angle)

    return images, labels

IMAGES, LABELS = read_drive_data(DRIVE_LOG)
DATA_PICKLE = {}
DATA_PICKLE["images"] = IMAGES
DATA_PICKLE["labels"] = LABELS
pickle.dump(DATA_PICKLE, open("train.p", "wb"))
