"""
Code loading and analyzing SVHN images and data
"""
import os
import numpy as np
from PIL import Image

print('All modules imported.')

# Wait until you see that all files have been downloaded.
print('All files downloaded.')

def load_svhn_images(folder_path):
    """
    Load in all images from a folder

    :param folder_path: Path to folder containing
    :return a numpy array of all the images
    """
    images = []
    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            image = Image.open(file)
            image.load()
            # Load image data as 1 dimensional array
            # We're using float32 to save on memory space
            feature = np.array(image, dtype=np.float32)
            images.append(feature)

    return images

IMAGES = load_svhn_images('data/train/')
HEIGHTS = [image.shape[0] for image in IMAGES]
WIDTHS = [image.shape[1] for image in IMAGES]

#---

MAX_HEIGHT, MIN_HEIGHT = max(HEIGHTS), min(HEIGHTS)
MAX_WIDTH, MIN_WIDTH = max(WIDTHS), min(WIDTHS)
print()
print("Max Height:", MAX_HEIGHT, "Min Height:", MIN_HEIGHT)
print("Max Width:", MAX_WIDTH, "Min Width:", MIN_WIDTH)

#---

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
# %matplotlib inline

# setup heights histogram
fig = plt.figure()
height_plot = fig.add_subplot(111)

l = height_plot.hist(HEIGHTS, 50, normed=1, facecolor='green', alpha=0.75)

height_plot.set_xlabel('Image Height')
height_plot.set_ylabel('Fraction')
height_plot.set_title('Height Distribution')
height_plot.set_xlim(MIN_HEIGHT, MAX_HEIGHT)
height_plot.set_ylim(0, max(n))
height_plot.grid(True)

width_plot = fig.add_subplot(112)

l = width_plot.hist(HEIGHTS, 50, normed=1, facecolor='green', alpha=0.75)

width_plot.set_xlabel('Image Height')
width_plot.set_ylabel('Fraction')
width_plot.set_title('Height Distribution')
width_plot.set_xlim(MIN_HEIGHT, MAX_HEIGHT)
width_plot.set_ylim(0, max(n))
width_plot.grid(True)

plt.show()

#---
from digitStruct import DigitStruct, yieldNextDigitStruct
from tdqm import tdqm

def read_labels(digitstruct_file):
    """
    Read in labels from digitStruct.mat file to create a dict of image file name and 
    corresponding labels
    """        
    labels = dict()
    for dsObj in tdqm(yieldNextDigitStruct(digitstruct_file), ncols=50):
        image_labels = []
        for bbox in dsObj.bboxList:
            image_labels.append(bbox.label)            
        labels[dsObj.name] = image_labels
    return labels

DSFILE = 'data/train/digitStruct.mat'
LABELS = read_labels(DSFILE)
#---
# View first few lables
for index in range(3):
    image_file = '{}.png'.format(index)
    print(image_file, labels(image_file))
