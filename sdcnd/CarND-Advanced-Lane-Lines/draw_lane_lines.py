"""
Detect and draw lane line on images and movies
"""
from math import floor
import matplotlib.image as mpimg
from detect_lane import undistort, draw_lane_polygon, draw_curvature_text, M_INV
from lane_line import LaneLine

# Draw parameters
TEXT_COLOR = (255, 255, 128)
LANE_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
THICKNESS = 50

def draw_lane_image(image):
    """
    Given an undistorted image, detect lane lines in it, and draw the lane lines and
    radius of curvature on an output image.abs
    """
    width = image.shape[1]

    left = LaneLine()
    left.detect_lane_line(image, (left.side_margin, floor(width / 2)))

    right = LaneLine()
    right.detect_lane_line(image, (floor(width / 2), width - left.side_margin))

    output = draw_lane_polygon(image, left.fit_x, left.ally, right.fit_x, right.ally, \
        M_INV, LANE_COLOR, LINE_COLOR, THICKNESS)
    draw_curvature_text(output, left.radius_of_curvature, right.radius_of_curvature, TEXT_COLOR)

    return output

def draw_lane_test_image(image_name):
    """
    Draw lane lines on a test image
    """
    undist = undistort(mpimg.imread('test_images/' + image_name))
    output = draw_lane_image(undist)
    mpimg.imsave('output_images/' + image_name.replace(image_name, 'output_' + image_name), output)

image_names = ['straight_lines1.jpg', 'straight_lines2.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg',\
    'test4.jpg', 'test5.jpg', 'test6.jpg']
for name in image_names:
    draw_lane_test_image(name)
