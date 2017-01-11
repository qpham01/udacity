"""
Contain utility methods for working with images.
"""
import numpy as np

# Convert to grayscale
def rgb_to_gray(rgb):
    """
    Convert an RGB image array in the form of (?, w, h, 3) to grayscale (?, w, h, 1)
    """
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return np.reshape(gray, (-1, 32, 32, 1))
