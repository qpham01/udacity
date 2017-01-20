"""
Color and gradient pipeline for lane detection
"""
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sobel import gray_rgb, absolute_threshold, magnitude_threshold, direction_threshold

hard_test_images = ['test1.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']

calibration_matrix = None
distortion_params = None

def load_camera_calibration():
    """
    Initialize calibration data
    """
    camera_calibration_pickle = pickle.load(open("camera_cal_pickle.p", "rb"))
    mtx = camera_calibration_pickle["mtx"]
    dist = camera_calibration_pickle["dist"]
    return mtx, dist

def threshold_string(threshold):
    """
    Build a string from the threshold values
    """
    size = len(threshold)
    if size == 2:
        return "{}_{}".format(threshold[0], threshold[1])
    if size == 3:
        return "{}_{}_{}".format(threshold[0], threshold[1], threshold[2])
    raise ValueError("Unexpected threshold list size {}".format(size))

def output_file_name(name, param_key, threshold, image_file_name):
    """
    Builds a new file name from the parameter key, the threshold values, and the
    image file name.
    """
    return 'output_images/' + name + '/' + param_key + '_' + threshold_string(threshold) + \
        '_' + image_file_name

# Edit this function to create your own pipeline.
def pipeline(img, name, test_params, file_name):
    """
    An image processing pipeline that acts on a dictionary of test parameters,
    saving an output image for each processing step in the pipeline to a file
    named after the parameters that was applied to the input image.
    """
    global calibration_matrix, distortion_params
    if calibration_matrix is None:
        calibration_matrix, distortion_params = load_camera_calibration()

    output_dir = 'output_images/' + name
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Undistort input image as first step.
    img = cv2.undistort(img, calibration_matrix, distortion_params, None, calibration_matrix)
    parameters = test_params.items()
    img_layers = dict()

    # Convert to HLS color space and separate the L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    binary = None
    single = None
    stack = None
    combine = None

    for key, value in parameters:
        layer = None
        if key == 'sobel_abs_x_l':
            layer = absolute_threshold(l_channel, orient='x', kernel=value[2], thresh=value)

        if key == 'sobel_abs_y_l':
            layer = absolute_threshold(l_channel, orient='y', kernel=value[2], thresh=value)

        if key == 'sobel_mag_b_l':
            layer = magnitude_threshold(l_channel, orient='b', kernel=value[2], thresh=value)

        if key == 'sobel_mag_x_l':
            layer = magnitude_threshold(l_channel, orient='x', kernel=value[2], thresh=value)

        if key == 'sobel_mag_y_l':
            layer = magnitude_threshold(l_channel, orient='y', kernel=value[2], thresh=value)

        if key == 'sobel_dir_l':
            layer = direction_threshold(l_channel, kernel=value[2], thresh=value)

        if key == 'color_s':
            layer = np.zeros_like(s_channel)
            layer[(s_channel >= value[0]) & (s_channel <= value[1])] = 1

        if key == 'color_l':
            layer = np.zeros_like(s_channel)
            layer[(l_channel >= value[0]) & (l_channel <= value[1])] = 1

        if key == 'color_h':
            layer = np.zeros_like(s_channel)
            layer[(h_channel >= value[0]) & (h_channel <= value[1])] = 1

        if layer is not None:
            img_layers[key] = layer

        if key == 'single' and len(value) > 0:
            single = value

        if key == 'stack' and len(value) > 0:
            stack = value

        if key == 'combine' and len(value) > 0:
            combine = value

        # Save the file
        if layer is not None:
            output_name = output_file_name(name, key, value, file_name)
            mpimg.imsave(output_name, layer, cmap="gray")

    if single is not None:
        binary = img_layers[single]
        output_name = output_file_name(name, 'single', (0, 0), file_name)
        mpimg.imsave(output_name, binary, cmap="gray")

    # Stack the layers specified
    if stack is not None:
        binary = np.dstack((np.zeros_like(img_layers[stack[0]]), img_layers[stack[0]], \
            img_layers[stack[1]]))
        output_name = output_file_name(name, 'stack', stack, file_name)
        mpimg.imsave(output_name, binary, cmap="gray")

    if combine is not None:
        binary = np.zeros_like(img_layers[combine[0][0]])
        binary[((img_layers[combine[0][0]] == 1) & (img_layers[combine[0][1]] == 1)) |\
            ((img_layers[combine[1][0]] == 1) & (img_layers[combine[1][1]] == 1))] = 1
        output_name = output_file_name(name, 'combine', (0, 0), file_name)
        mpimg.imsave(output_name, binary, cmap="gray")

    return binary
"""
for i, param in enumerate(test_parameters):
    for file_name in hard_test_images:
        image = mpimg.imread('test_images/' + file_name)
        result = pipeline(image, 'param{}'.format(i), param, file_name)
"""

test_parameters = [\
    {'color_s': (170, 255), 'sobel_abs_x_l': (50, 150, 5), 'stack': ['color_s', 'sobel_abs_x_l']},\
    {'color_s': (170, 255), 'sobel_mag_b_l': (40, 100, 9), 'stack': ['color_s', 'sobel_mag_b_l']},\
    {'color_s': (170, 255), 'sobel_mag_x_l': (40, 100, 9), 'stack': ['color_s', 'sobel_mag_x_l']},\
    {'color_s': (170, 255), 'sobel_dir_l': (0.7, 1.2, 15), 'stack': ['color_s', 'sobel_dir_l']},\
    {'sobel_abs_x_l': (20, 100, 3), 'sobel_abs_y_l': (20, 100, 3), 'sobel_mag_x_l': (40, 100, 9),\
        'sobel_dir_l': (0.7, 1.2, 15), 'combine': [['sobel_abs_x_l', 'sobel_abs_y_l'],\
        ['sobel_mag_x_l', 'sobel_dir_l']]},
    {'sobel_abs_x_l': (20, 100, 3), 'sobel_abs_y_l': (20, 100, 3), 'sobel_mag_x_l': (40, 100, 9),\
        'color_s': (170, 255), 'combine': [['sobel_abs_x_l', 'sobel_abs_y_l'],\
        ['color_s', 'color_s']]},
    {'color_s': (140,255), 'single': 'color_s'}, 
    {'color_l': (140,255), 'single': 'color_l'},
    {'color_h': (140,255), 'single': 'color_h'},
    {'sobel_mag_x_l': (20,100, 5), 'single': 'sobel_mag_x_l'},
    {'sobel_abs_x_l': (20,100, 9), 'single': 'sobel_abs_x_l'},
    {'sobel_mag_y_l': (30,100, 3), 'single': 'sobel_mag_y_l'},
    {'sobel_abs_y_l': (20,100, 3), 'single': 'sobel_abs_y_l'},
    {'sobel_mag_b_l': (20,100, 3), 'single': 'sobel_mag_b_l'},
    {'sobel_dir_l': (0.7, 1.2, 15), 'single': 'sobel_dir_l'},
    {'color_s': (170, 255), 'sobel_dir_l': (0.7, 1.2, 15), 'sobel_mag_x_l': (50, 150, 5),\
        'combine': [['color_s', 'sobel_dir_l'],['sobel_mag_x_l', 'sobel_mag_x_l']]}
    ]

i = 13
for file_name in hard_test_images:
    image = mpimg.imread('test_images/' + file_name)
    result = pipeline(image, 'param{}'.format(i), test_parameters[i], file_name)

# Plot the result
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result)
ax2.set_title('Pipeline Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
"""
# mpimg.imsave("result.png", result)