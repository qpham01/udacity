#importing some useful packages
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from lane_segment import LaneSegment

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
#plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            lane_segment = LaneSegment(x1, y1, x2, y2)
            # Ignore segments that don't looks like lane lines in slope and length
            if math.isnan(lane_segment.slope) or lane_segment.abs_d_y < MIN_DY or \
                lane_segment.abs_slope < MIN_ABS_SLOPE or lane_segment.abs_slope > MAX_ABS_SLOPE:
                continue
            if lane_segment.slope > 0:
                if lane_segment.min_x < MID_X:
                    # Ignore segments that are on the wrong side of the screen based on slope
                    continue
                right_lane_lines.append(lane_segment)
            if lane_segment.slope < 0:
                if lane_segment.max_x > MID_X:
                    # Ignore segments that are on the wrong side of the screen based on slope
                    continue
                left_lane_lines.append(lane_segment)

    left_max_x = sorted(left_lane_lines, key=lambda x: x.max_x, reverse=True)
    right_min_x = sorted(right_lane_lines, key=lambda x: x.min_x, reverse=False)
    left_top = (left_max_x[0].max_x, left_max_x[0].min_y)
    right_top = (right_min_x[0].min_x, right_min_x[0].min_y)
    left_len = len(left_max_x)
    right_len = len(right_min_x)
    left_bottom1 = (left_max_x[left_len - 2].min_x, left_max_x[left_len - 2].max_y)
    left_bottom2 = (left_max_x[left_len - 1].min_x, left_max_x[left_len - 1].max_y)
    right_bottom1 = (right_min_x[right_len - 2].max_x, right_min_x[right_len - 2].max_y)
    right_bottom2 = (right_min_x[right_len - 1].max_x, right_min_x[right_len - 1].max_y)

    cv2.line(img, left_bottom1, left_top, color, thickness)
    cv2.line(img, left_bottom2, left_top, color, thickness)
    cv2.line(img, right_bottom1, right_top, color, thickness)
    cv2.line(img, right_bottom2, right_top, color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=5)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

# define blur parameters
KERNEL_SIZE = 5

# Define the canny parameters
LOW_THRESHOLD=50
HIGH_THRESHOLD=150

# Define the Hough transform parameters
RHO = 1 # distance resolution in pixels of the Hough grid
THETA = np.pi/180 # angular resolution in radians of the Hough grid
THRESHOLD = 10    # minimum number of votes (intersections in Hough grid cell)
MIN_LINE_LENGTH = 20 #minimum number of pixels making up a line
MAX_LINE_GAP = 10    # maximum gap in pixels between connectable line segments

# Define expected image size
XSIZE = 960
YSIZE = 540
MID_X = XSIZE / 2

# Define area of interest parameters
DX1 = 60   # Pixels from left/right borders of bottom edge
DX2 = 400  # Pixels from left/right borders of top edge
DY = 300   # Pixels from top border of top edge

# Lane filtering parameters
MIN_DY = 5
MIN_ABS_SLOPE = 0.5
MAX_ABS_SLOPE = 3.0


import os
test_images = os.listdir("test_images/")
for test_image in test_images:
    # load in image
    initial_img = mpimg.imread('test_images/' + test_image)
    xsize = image.shape[1]
    ysize = image.shape[0]
    if xsize != XSIZE or ysize != YSIZE:
        raise Exception("Incorrect image size", xsize, ysize)

    # step 1: convert to grayscale
    img = grayscale(initial_img)

    # step 2: blur
    img = gaussian_blur(img, KERNEL_SIZE)

    # step 3: canny
    img = canny(img, LOW_THRESHOLD, HIGH_THRESHOLD)

    # step 4: calculate mask for region of interest

    # calculate vertices for region of interest
    vertices = np.array([[(DX1, YSIZE), (DX2, DY), (XSIZE - DX2, DY), (XSIZE - DX1, YSIZE)]],
                        dtype=np.int32)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(img)
    ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    img = cv2.bitwise_and(img, mask)

    # step 5: apply Hough to find lines
    img = hough_lines(img, RHO, THETA, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)

    img = weighted_img(img, initial_img)

    # save to output
    mpimg.imsave("output/" + test_image, img)
