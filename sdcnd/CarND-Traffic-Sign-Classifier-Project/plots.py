"""
Contain plotting functions for data visualization
"""
from math import ceil
import matplotlib
matplotlib.use('svg')
from matplotlib import pyplot as plt

def plot_images(images, rows, c_map):
    """
    Plot images
    """
    img_count = len(images)
    plot_cols = int(ceil(img_count / rows))
    f, img_plots = plt.subplots(rows, plot_cols, sharex=True)

    index = 0
    for row in range(rows):
        for col in range(plot_cols):
            img_plots[row, col].imshow(images[index].squeeze(), cmap=c_map)
            index += 1
            if index >= len(images):
                break

def plot_probabilities(probabilities, rows, color):
    """
    Plot probabilities
    """
    plot_cols = int(ceil(len(probabilities) / rows))
    f, softmax_plots = plt.subplots(rows, plot_cols, sharex=True)    
    index = 0
    for row in range(rows):
        for col in range(plot_cols):
            softmax_plots[row, col].plot(probabilities[index], color)
            softmax_plots[row, col].set_ylim([0.0, 1.0])
            index += 1
            if index >= len(probabilities):
                break
    plt.show()
