"""
Contains code to detect lane lines.
"""

import pickle
from math import ceil, floor
import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
from pipeline import pipeline

CAMERA_CALIBRATION_PICKLE = pickle.load(open("camera_cal_pickle.p", "rb"))
CALIBRATION_MATRIX = CAMERA_CALIBRATION_PICKLE["mtx"]
DISTORTION_PARAMS = CAMERA_CALIBRATION_PICKLE["dist"]

def undistort(image):
    """
    Undistort an image using default camera calibration parameters
    """
    return cv2.undistort(image, CALIBRATION_MATRIX, DISTORTION_PARAMS, None, CALIBRATION_MATRIX)

LANE_PARAMETERS = [ \
    {'color_s': (140, 255), 'sobel_abs_x_l': (25, 100, 9), 'sobel_abs_y_l': (25, 100, 5), \
       'combine': [['color_s', 'color_s'], ['sobel_abs_x_l', 'sobel_abs_x_l']]},
    {'color_s': (180, 255), 'single': 'color_s'}, \
    {'color_h': (0, 100), 'single': 'color_h'}, \
    {'color_l': (50, 150), 'single': 'color_l'}, \
    {'sobel_abs_x_l': (25, 100, 9), 'single': 'sobel_abs_x_l'}, \
    {'sobel_abs_y_l': (25, 100, 9), 'single': 'sobel_abs_y_l'}, \
    {'sobel_mag_b_l': (50, 100, 9), 'single': 'sobel_mag_b_l'}, \
    {'sobel_dir_l': (0.7, 1.2, 15), 'single': 'sobel_dir_l'}, \
    {'color_s': (140, 255), 'sobel_abs_x_l': (25, 100, 9), 'sobel_dir_l': (0.8, 1.2, 15), \
       'combine': [['color_s', 'color_s'], ['sobel_abs_x_l', 'sobel_dir_l']]}]

# Coordinates from forward view and topdown view for making perspective transform
FORWARD_COORDS = np.float32([[590, 450], [700, 450], [1206, 670], [278, 670]])
TOPDOWN_COORDS = np.float32([[200, 0], [1080, 0], [1080, 670], [200, 670]])

# Get perspective tranform
M = cv2.getPerspectiveTransform(FORWARD_COORDS, TOPDOWN_COORDS)
M_INV = cv2.getPerspectiveTransform(TOPDOWN_COORDS, FORWARD_COORDS)

def make_topdown_binary(undistorted_image, index):
    """
    Make a binary image using the index of thresholding method in the LANE_PARAMETERS list.
    """
    # Make cv2 format image size.
    image_size = undistorted_image.shape[0:2]
    cv2_image_size = (image_size[1], image_size[0]) # cv2 size is (width, height)

    # Warp the undistorted image with the perspective transform
    warped = cv2.warpPerspective(undistorted_image, M, cv2_image_size, flags=cv2.INTER_LINEAR)

    # Now apply the pipeline to get the binary topdown lane image
    binary = pipeline(warped, 'topdown', LANE_PARAMETERS[index], None)

    return binary, warped

def create_lane_histogram_data(image, top, bottom, left, right):
    """
    Create the lane histogram data from some portion of the image.
    :param image: The binary image containing the topdown lane pixels
    :param image_fraction: How much of the image to use, from the bottom up.
    """
    return np.sum(image[top:bottom, left:right], axis=0)

def get_lane_pixels(binary_topdown_image, box_half_width, box_height, side_margin):
    """
    Returns the lane line pixels detected from the topdown binary image.
    """
    image_height, image_width = binary_topdown_image.shape
    leftx = []
    lefty = []
    rightx = []
    righty = []
    box_top = image_height - box_height

    # start with a border at the middle of the image... will adjust when lane lines found.
    border = floor(image_width / 2)

    histogram_values = create_lane_histogram_data(binary_topdown_image, 0, image_height, 0, \
        image_width)

    # start with the assumption that lane lines will be found on the two halves of the binary image.
    # pixel column of left lane line.
    left_lane = histogram_values[side_margin:border].argmax() + side_margin

    # right lane line pixel column.
    right_lane = histogram_values[border:image_width - side_margin].argmax() + border

    while box_top >= 0:
        # Find lane line maximum value when box is slide up.  Will use for new lane line
        # centerseach iteration.
        histogram_values = create_lane_histogram_data(binary_topdown_image, box_top, box_top + \
            box_height, 0, image_width)

        # print("left lane:", left_lane)
        # print("right lane:", right_lane)

        # Left lane
        box_left = max(0, left_lane - box_half_width)
        box_right = min(image_width, left_lane + box_half_width - 1)
        box_bottom = box_top + box_height
        for row in range(box_top, box_bottom):
            for col in range(box_left, box_right):
                if binary_topdown_image[row, col] > 0:
                    leftx.append(col)
                    lefty.append(row)
        # Slide left box to center of bright pixels
        last_left_lane = left_lane
        left_lane = histogram_values[box_left:box_right].argmax()
        if left_lane == 0:
            left_lane = last_left_lane
        else:
            left_lane += box_left

        # If new lane x coordinate is too far away, don't use it.
        if abs(left_lane - last_left_lane) > box_half_width:
            left_lane = last_left_lane

        # Right lane
        box_left = max(0, right_lane - box_half_width)
        box_right = min(image_width, right_lane + box_half_width - 1)
        box_bottom = box_top + box_height
        for row in range(box_top, box_bottom):
            for col in range(box_left, box_right):
                if binary_topdown_image[row, col] > 0:
                    rightx.append(col)
                    righty.append(row)

        # Slide right box to center of bright pixels
        last_right_lane = right_lane
        right_lane = histogram_values[box_left:box_right].argmax()
        if right_lane == 0:
            right_lane = last_right_lane
        else:
            right_lane += box_left

        if abs(right_lane - last_right_lane) > box_half_width:
            right_lane = last_right_lane

        # Slide box up
        box_top -= box_height

        #print(left_lane, right_lane, box_left, box_right)
    return np.array(leftx), np.array(lefty), np.array(rightx), np.array(righty)


# Convert to real world coordinates
YM_PER_PIX = 30/720 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meteres per pixel in x dimension

def convert_pixel_to_world(x, y):
    """
    Convert pixel coordinates from world coordinates.
    """
    y = y * YM_PER_PIX
    x = x * XM_PER_PIX
    return x, y

def fit_lane_line(x, y):
    """
    Fit a second order polynomial to each fake lane line
    """
    fit = np.polyfit(y, x, 2)
    fitx = fit[0]*y**2 + fit[1]*y + fit[2]
    return fit, fitx

# Calculate curve radius
def curve_radius(y, fit):
    """
    Calculate the radius of curvature from y coordinates and quadratic fit parameters
    """
    max_y = np.max(y)
    radius = ((1 + (2*fit[0]*max_y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return radius

def draw_lane_line(image, lane_points, color, thickness):
    """
    Draw the lane line through consecutive pixel points on a lane,
    """
    last_point = None
    for point in lane_points:
        if last_point is not None:
            cv2.line(image, last_point, point, color, thickness)
        last_point = point

def clean_lane_points(lane_points):
    """
    Remove lane points with same values
    """
    points = lane_points[0]
    last_point = []
    cleaned_points = []
    for point in points:
        if len(last_point) > 0 and point[0] == last_point[0] and point[1] == last_point[1]:
            continue
        last_point = point
        cleaned_points.append((int(point[0]), int(point[1])))
    return cleaned_points

def sample_lane_points(points, point_count):
    """
    Uniformly sample the specified point count from a long list of points.
    """
    interval = floor(len(points) / point_count)
    sample = []
    for index in range(point_count):
        sample.append(points[index * interval])
    return sample

def draw_lane_polygon(image, fit_x_left, y_left, fit_x_right, y_right, p_inv, lane_color, \
    line_color, thickness):
    """
    Draw the lane polygon onto an image.
    """
    # Create an image to draw the lines on
    lane_image = np.zeros_like(image).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # There are often too many lane points so we'll just uniformly sample n lane_points for
    # visualization
    sampled_lx = fit_x_left # sample_lane_points(fit_x_left, 100) #
    sampled_ly = y_left # sample_lane_points(y_left, 100) #

    sampled_rx = sample_lane_points(fit_x_right, 100) # fit_x_right #
    sampled_ry = sample_lane_points(y_right, 100) # y_right #

    # Recast the x and y points into usable format for cv2.fillPoly() and transform back to
    # pixel coords.
    pts_left = np.array([np.transpose(np.vstack([sampled_lx, sampled_ly]))]) /\
        np.array([XM_PER_PIX, YM_PER_PIX])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([sampled_rx, sampled_ry])))]) / \
        np.array([XM_PER_PIX, YM_PER_PIX])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lane_image, np.int_([pts]), lane_color)

    # Draw the lane lines
    pts_left = clean_lane_points(pts_left)
    pts_right = clean_lane_points(pts_right)

    draw_lane_line(lane_image, pts_left, line_color, thickness)
    draw_lane_line(lane_image, pts_right, line_color, thickness)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(lane_image, p_inv, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

def draw_text(image, text, position, color):
    """
    Draw text onto an image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, 1, color, 2, cv2.LINE_AA)

def draw_curvature_text(image, left_radius, right_radius, text_color):
    """
    Draw curvature text onto an image
    """
    left_radius_text = "Left Curvature Radius:  {:.2f} m".format(left_radius)
    right_radius_text = "Right Curvature Radius: {:.2f} m".format(right_radius)

    draw_text(image, left_radius_text, (700, 50), text_color)
    draw_text(image, right_radius_text, (700, 90), text_color)

# Draw parameters
TEXT_COLOR = (255, 255, 128)
LANE_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
THICKNESS = 50

def find_and_draw_lanes(image, pipeline_index, box_half_width, box_height, side_margin):
    """
    In one step, find the lane lines in an image and draw them onto the image.abs
    """
    binary, _ = make_topdown_binary(image, pipeline_index)
    # Extract lane box_half_width, box_height, side_margin
    left_x, left_y, right_x, right_y = get_lane_pixels(binary, box_half_width, box_height, \
        side_margin)
    # Convert to world coordinates
    left_x, left_y = convert_pixel_to_world(left_x, left_y)
    right_x, right_y = convert_pixel_to_world(right_x, right_y)

    # Polynomial fit
    left_fit, left_fit_x = fit_lane_line(left_x, left_y)
    right_fit, right_fit_x = fit_lane_line(right_x, right_y)

    left_curverad = curve_radius(left_y, left_fit)
    right_curverad = curve_radius(right_y, right_fit)

    # Draw lane data on image and show
    output = draw_lane_polygon(image, left_fit_x, left_y, right_fit_x, right_y, M_INV, \
        LANE_COLOR, LINE_COLOR, THICKNESS)
    draw_curvature_text(output, left_curverad, right_curverad, TEXT_COLOR)

    return output, binary

# The index into the LANE_PARAMETERS of the thresholding method to use.
PIPELINE_INDEX = 8

# Hack here to disregard noise from other cars at edge of the image at cost of reduced FOV.
SIDE_MARGIN = 100

# Will call our sliding window 'box' for short in variable names.
BOX_HALF_WIDTH = 50
BOX_HEIGHT = 60

NAME = "test2.jpg"
UNDIST = undistort(mpimg.imread('test_images/' + NAME))
OUTPUT, _ = find_and_draw_lanes(UNDIST, PIPELINE_INDEX, BOX_HALF_WIDTH, BOX_HEIGHT, \
    SIDE_MARGIN)
mpimg.imsave('output_images/' + NAME.replace(NAME, 'output_' + NAME), OUTPUT)
