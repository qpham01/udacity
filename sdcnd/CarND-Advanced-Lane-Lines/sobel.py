"""
Various uses of the Sobel operator for edge gradient thresholding.
"""
import numpy as np
import cv2

def gray_rgb(img):
    """
    # Convert to grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def absolute_threshold(img, orient='x', kernel=3, thresh=(0, 255)):
    """
    Define a function that applies Sobel x or y,
    then takes an absolute value and applies a threshold.
    Note: calling your function with orient='x', thresh_min=5, thresh_max=100
    should produce output like the example image shown above this quiz.
    """
    # Apply the following steps to img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    else:
        raise ValueError("orient must be in ['x', 'y']")
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def magnitude_threshold(img, orient='x', kernel=3, thresh=(0, 255)):
    """
    Define a function that applies Sobel x and y, then computes the magnitude of the gradient
    and applies a threshold
    """
    # Apply the following steps to img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # 3) Calculate the magnitude
    if orient == 'x':
        magnitude = np.sqrt(sobel_x**2)
    elif orient == 'y':
        magnitude = np.sqrt(sobel_y**2)
    elif orient == 'b':
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    else:
        raise ValueError("orient must be in ['x', 'y', 'b']")

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(magnitude) / 255
    magnitude = (magnitude/scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met, zeros otherwise
    binary_output = np.zeros_like(magnitude)
    binary_output[(magnitude >= thresh[0]) & (magnitude <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def direction_threshold(img, kernel=3, thresh=(0, np.pi/2)):
    """
    Define a function that applies Sobel x and y, then computes the direction of the gradient
    and applies a threshold.
    """
    # Apply the following steps to img
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.sqrt(sobel_x**2)
    abs_sobely = np.sqrt(sobel_y**2)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output
