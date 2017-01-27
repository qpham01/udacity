"""
Detect and draw lane line on images and movies
"""
from math import floor
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.image as mpimg
from detect_lane import undistort, draw_lane_polygon, draw_curvature_text, draw_lane_image

from lane_line import LaneLine

# Draw parameters
TEXT_COLOR = (255, 255, 128)
LANE_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
THICKNESS = 50
SIDE_MARGIN = 100

global LEFT, RIGHT, LEFT_REGION, RIGHT_REGION
LEFT = LaneLine()
RIGHT = LaneLine()
LEFT_REGION = (0, 0)
RIGHT_REGION = (0, 0)

def init_lane_lines():
    width = 1280
    left = LaneLine()
    right = LaneLine()
    left_region = (SIDE_MARGIN, floor(width / 2))
    right_region = (floor(width / 2), width - SIDE_MARGIN)
    return left, right, left_region, right_region

def draw_lane_test_image(image_name):
    """
    Draw lane lines on a test image
    """
    undist = undistort(mpimg.imread('test_images/' + image_name))
    left, right, left_region, right_region = init_lane_lines()
    output = draw_lane_image(undist, left, right, left_region, right_region)
    mpimg.imsave('output_images/' + image_name.replace(image_name, 'output_' + image_name), output)
    return output

def process_movie(movie_name):
    """
    Detect lane lines in a movie.
    """
    global LEFT, RIGHT, LEFT_REGION, RIGHT_REGION
    LEFT, RIGHT, LEFT_REGION, RIGHT_REGION = init_lane_lines()
    clip1 = VideoFileClip(movie_name)
    output = clip1.fl_image(process_image) 
    output_name = 'output_' + movie_name
    print("Writing movie file", output_name)
    output.write_videofile(output_name, audio=False)

def process_image(image):
    global LEFT, RIGHT, LEFT_REGION, RIGHT_REGION
    output = draw_lane_image(image, LEFT, RIGHT, LEFT_REGION, RIGHT_REGION)
    # LEFT_REGION = (LEFT.pixel_pos - LEFT.box_half_width, LEFT.line_base_pos + \
    #    LEFT.box_half_width)
    # RIGHT_REGION = (RIGHT.pixel_pos - RIGHT.box_half_width, RIGHT.line_base_pos + \
    #    RIGHT.box_half_width)
    return output

PROCESS_ONE_IMAGE = False
if PROCESS_ONE_IMAGE:
    draw_lane_test_image('test2.jpg')

PROCESS_TEST_IMAGES = False
if PROCESS_TEST_IMAGES:
    IMAGE_NAMES = ['straight_lines1.jpg', 'straight_lines2.jpg', 'test1.jpg', 'test2.jpg', \
        'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
    for name in IMAGE_NAMES:
        draw_lane_test_image(name)

PROCESS_PROJECT_MOVIE = False
if PROCESS_PROJECT_MOVIE:
    process_movie('project_video.mp4')

PROCESS_CHALLENGE_MOVIE = True
if PROCESS_CHALLENGE_MOVIE:
    process_movie('challenge_video.mp4')

PROCESS_HARDER_MOVIE = False
if PROCESS_HARDER_MOVIE:
    process_movie('harder_challenge_video.mp4')
