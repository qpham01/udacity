
# Advanced Lane Finding Project
---
The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
9. Discussions of limitations and potential improvements.

---

## 1. Compute Camera Calibration and Correct Lens Distortion

Camera calibration is the process of undoing the light distortion by the camera lens that warps certain parts of the image.  The magnitude of distortion depends on the physical shape of the lens.  The following steps are involved in camera calibration.

1. A number of images need to be taken of a chessboard pattern by the camera which we want to calibrate.
    * The chessboard is essentially a grid of known corners where the chess spaces intersect.  These generated corners are known objective points on a grid that can be algorithmically generated for correlation with the corresponding corners detected from the camera images (called image points) to measure the distortion produced by the camera lens.  
    * The chessboard pattern and the camera images are provided for this project.
2. Generate the objective points on the chess board.
3. Detect the chessboard corners (image points) on the camera images.
4. Feed objective points and image points into the OpenCV cv2.calibrateCamera method to compute the camera calibration matrix and distortion coefficients.
5. Call cv2.undistort method with the camera calibration matrix and the distortion coefficients to undo the distortion effect of the camera lens on a test camera image to verify correctness.
6. Save the camera calibration matrix and the distortion coefficients to a pickle file for later use in correcting other images taken from the same camera.

The code cell below performs steps 1 through 3, above.  It also saves the images of the detected corners in _output_images/corners_ for later review.


```python
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        corner_file = fname.replace('camera_cal/', 'output_images/corners/corners_')
        mpimg.imsave(corner_file, img)
```

Now that we have the objective points and the image points, we are ready to compute the camera calibration matrix and distortion parameters.  The code cell below performs steps 3 to 6, above, as well as plotting the last test camera image and its undistorted image for visual comparison.  The undistorted images can also be seen in the output_images folder.  As shown below, the most visible sign of removing the camera lens distortion is the straigtening of the curve lines near the edges of the image.


```python
import pickle

# Do camera calibration given object points and image points
img_size = img.shape[0:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

# Undistort three test images and write them out to disk for review.
calibration_test_images = ['calibration2', 'calibration3', 'calibration5']
for test_name in calibration_test_images:
    img = mpimg.imread('camera_cal/' + test_name + '.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/' + test_name + '_undistorted.jpg', dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open("camera_cal_pickle.p", "wb") )

# Plot the last distorted and undistorted test images for visual comparison
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_4_0.png)


---

## 2. Undistort the Camera Image

We'll start with a method to undistort the road image. In the code cell below we load the camera calibration parameters that were saved earlier and use them in the **undistort** method to remove camera lens distortions.


```python
camera_calibration_pickle = pickle.load( open("camera_cal_pickle.p", "rb") )
calibration_matrix = camera_calibration_pickle["mtx"]
distortion_params = camera_calibration_pickle["dist"]

def undistort(image):
    return cv2.undistort(image, calibration_matrix, distortion_params, None, calibration_matrix)
```


```python
# Read in test image and undistort it.
test_name = 'test_images/test5.jpg'
img = mpimg.imread(test_name)
dst = undistort(img)

# Save output image to disk for review
output_name = test_name.replace('test_images/', 'output_images/undistorted_')
mpimg.imsave(output_name, dst)

# Visualize the before and after images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_7_0.png)


Looks like the lens distortion of this image is not too bad.  In the undistorted image, less of the car's hood is visible, and yellow and white lane lines are stretched slightly more toward the edge of the image.  This can most easily be seen by looking at where the yellow lane meets the hood: at slightly more than 200 pixels in the original image, but at slightly less than 200 pixels in the undistorted image.

---
## 3. Create a Thresholded Binary Image to Reveal Lane Lines from Forward Camera View

Reliably detecting lane lines and curvature to guide the steering of self-driving cars is a challenging proposition due to varying light conditions, road and lane line coloration, weather effects and shadows, and various other visual artifacts and interferences.  Fortunately, we do have a number of computer vision methods available to help, all thanks to OpenCV.

In this section the task is to transform an image of the road from a forward facing camera into a binary image where the lane lines are white and everything else around them are black, thus effectively detecting them.  We'll do this by taking the following steps:

1. Define the image processing methods which will be used in the process.
2. Create a pipeline through which we'll apply the above image processing methods in sequence.
3. Run the pipeline with a variety of parameters to see which parameter combination works best for detecting lane lines.

### Using Color and Gradient to Produce Thresholded Binary Image

In this step we'll apply a number of techniques to try to isolate the pixels making up the lane lines from the remainder of the road image.  We'll do this by creating a pipeline where one or more image processing/computer vision methods will be applied to the image's pixel values.  I expect a lot of trial and errors, and I want to specify the various methods and parameter values succinctly, so the pipeline that I created will be defined by a dictionary of methods and parameters.  That way, to try something different I just change content of the dictionary and pass it through the pipeline, for I can even generate the dictionary procedurally to try out a range of settings and/or a variety of method combinations.  The pipeline code saves images for both the 

The pipeline code is in the file **pipeline.py** in the same folder as this notebook. I've run over a dozen method combinations, often with multiple parameter values each, resulting in hundreds of processed road images for the following four test images that I considered most challening due to shadows and varying light conditions on the road.  I started with some of the suggested starting points during the lectures.  Then when I found some effective outputs, I run experiments on the individual methods of those approaches to see what they look like.  After I've seen those outputs, I proceeded to try out some new combinations guided by what I've learned.  The code cell below contains the four _hard_ images that I used for testing as well the list of pipeline parameter dictionaries that were part of my experiments, in the order that I ran them.


```python
hard_test_images = ['test1.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']

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
    {'color_s': (150,255), 'single': 'color_s'}, 
    {'color_l': (100,200), 'single': 'color_l'},
    {'color_h': (15,100), 'single': 'color_h'},
    {'sobel_mag_x_l': (30,200, 5), 'single': 'sobel_mag_x_l'},
    {'sobel_abs_x_l': (25,100, 5), 'single': 'sobel_abs_x_l'},
    {'sobel_mag_y_l': (30,100, 3), 'single': 'sobel_mag_y_l'},
    {'sobel_abs_y_l': (20,100, 3), 'single': 'sobel_abs_y_l'},
    {'sobel_mag_b_l': (50,100, 9), 'single': 'sobel_mag_b_l'},
    {'sobel_dir_l': (0.7, 1.2, 15), 'single': 'sobel_dir_l'},
    {'color_s': (150, 255), 'sobel_abs_x_l': (25, 100, 5),\
        'combine': [['color_s', 'color_s'],['sobel_abs_x_l', 'sobel_abs_x_l']]},
    ]
```

The key in each entry in the parameter dictionaries above define a color or gradient operation.  The value in each entry is a tuple containing the threshold parameters (first two elements), and for the sobel operations also the kernel size (third element).

Then there are the three operations for outputing the final thresholded binary image with the keys 'single', 'stack', and 'combine'. The key 'single' means apply the single thresholding operation specified in the value of the entry to produce the thresholded binary image.  The key 'stack' means that the value will be a list of thresholding operations that should be stacked in different color channels to produce the final thresholded binary image.  The key 'combine' means that the value will be a list of four operations that are combined as follows:

    ouptut_pixel = (operation1 == 1 & operation2 == 1) | (operation3 == 1 & operation4 == 1) == 1

Up to four operation can be thus combined, though 2 or 3 operations can also be specified by simply duplicating one or two operations in one of the two parenthesized & clauses above.  While this is not very flexible and have some potential duplication, it's simple and effective.

Stack and combine are thus two ways in much multiple color and/or gradient operations can be used together to better detect lane line pixels.  This is important because no one method will be sufficiently robust over all road, lighting, and weather conditions.  The key to effective lane finding is to use multiple complementary methods together.

The above list captures only a subset of the experiments that I ran, since I often modified the thresholding parameter values without creating a new entry in the above list of dictionaries.  However, the pipe_line code save images with the parameter values encoded in the image file names.  These images are saved to the **param[X]** subfolders of the **output_images** folder, where [X] is the list index of the _test_parameters_ list in the above code cell, and so nearly all the results of all my experiments should have been preserved in those **output_images/param[X]** folders.  There are hundreds of images in these folders from the experiments I ran, and I've included them all in the GitHub repository for review.

From these experiments I found that the saturation value of the HLS color space is quite effective and is well complemented by the Sobel magnitude calculation with the x orientation.  In the code cells below I will run a few of the more promising entries to show how I selected the methods for binary thresholding.


```python
from pipeline import pipeline
from math import ceil

# Load hard test images into a dictionary with file name keys, undistorting them before storing them.
# hard_images will contain undistorted images.
hard_images = []
for file_name in hard_test_images:
    hard_images.append((file_name, undistort(mpimg.imread('test_images/' + file_name))))

# A method to plot the thresholded test images after a pipeline run.
def plot_images(named_image_list, images_per_row, figure_size=(10,4)):
    """
    named_image_list: A list of tuples in the form (image_name, image)
    """
    rows = ceil(len(named_image_list) / images_per_row)
    f, axes = plt.subplots(rows, images_per_row, figsize=(figure_size[0], figure_size[1] * rows))
    f.tight_layout()
    index = 0
    for key, value in named_image_list:
        axes[index].imshow(value, cmap="gray")
        axes[index].set_title(key, fontsize=20)
        index += 1
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Run the pipeline on the nth parameter dictionary (0-based) in the test_parameters list.
def run_pipeline_with_test_parameter_index(index, named_image_list):
    binary_list = []
    for key, value in named_image_list:
        binary_list.append((key, pipeline(value, 'param{}'.format(index), test_parameters[index], key)))
    plot_images(binary_list, 4)
```

For starters, let just show the test images that will be used to judge the effectiveness of different methods and parameters.


```python
plot_images(hard_images, 4)
```


![png](output_14_0.png)



```python
# Run the pipeline on index 6 to show the effect of just thresholding the saturation color channel
run_pipeline_with_test_parameter_index(6, hard_images)
```


![png](output_15_0.png)


As shown above, just using thresholding the HLS saturation channel did a great job of identifying the solid left yellow lane, but not a very good job on the intermittent while lane lines that are far away, near the horizon.  Below I tried just using the sobel absolute thresholding with x orientation (index = 10).


```python
# Run the pipeline on index 10 to show the effect of absolute gradient thresholding with x orientation.
run_pipeline_with_test_parameter_index(10, hard_images)
```


![png](output_17_0.png)


The above didn't do a good job on the left yellow lane but in test5.jpg and test6.jpg that had shadows further out, it did detect the farther white lane lines better that the saturation thresholding.  At least on these test images these two methods seems to be good complements, so below they are _OR_ combined (method index 15)


```python
# Run the pipeline on index 15 to combine color_s and sobel_abs_x_l
run_pipeline_with_test_parameter_index(15, hard_images)
```


![png](output_19_0.png)



```python
# Now test with the remaining images to see if they look OK.
easy_test_images = ['straight_lines1.jpg', 'straight_lines2.jpg', 'test2.jpg', 'test3.jpg']

# easy_images will contain undistorted images.
easy_images = []
for file_name in easy_test_images:
    easy_images.append((file_name, undistort(mpimg.imread('test_images/' + file_name))))
```


```python
plot_images(easy_images, 4)
```


![png](output_21_0.png)



```python
# Run the pipeline on index 15 to combine color_s and sobel_abs_x_l
run_pipeline_with_test_parameter_index(15, easy_images)
```


![png](output_22_0.png)


As expected what works well for the hard test images will also work good with the easy test images.

## 4. Applying a Perspective Transform

The next task is to project the detected lane lines to a top down view in order to quantify curvature in the road.  We'll do this by mapping some pixel coordinates from an undistorted road image where the lanes are straight (as far as we can tell) to the corresponding pixel coordinates in a top down view of the road.  In the code below, we'll undistort the straight_lines1.jpg test image for this purpose.

Note that the first 3 lines in the code cell below contains the list of coordinates in the forward-looking view (**forward_coords**) and the list of corresponding coordinates in the top-down view (**topdown_coords**).


```python
# Coordinates from forward view and topdown view for making perspective transform
forward_coords = np.float32([[590, 450], [700, 450], [1206, 670], [278, 670]])
topdown_coords = np.float32([[200, 0], [1080, 0], [1080, 670], [200, 670]])

# Get perspective tranform
M = cv2.getPerspectiveTransform(forward_coords, topdown_coords)

def make_topdown_binary(undistorted_image, index):
    # Make cv2 format image size.
    image_size = undistorted_image.shape[0:2]
    cv2_image_size = (image_size[1], image_size[0]) # cv2 size is (width, height)

    # Warp the undistorted image with the perspective transform 
    warped = cv2.warpPerspective(undistorted_image, M, cv2_image_size, flags=cv2.INTER_LINEAR)

    # Now apply the pipeline to get the binary topdown lane image
    binary = pipeline(warped, 'topdown', lane_parameters[index], None)    
    
    return binary, warped

def visualize_topdown(undistorted_image, index):
    binary, warped = make_topdown_binary(undistorted_image, index)
    
    # Visualize
    perspective_visualize = [('Undistorted', undistorted_image), ('Warped', warped), ('Binary', binary)]
    plot_images(perspective_visualize, 3)


# Let's copy down the parameters used to find lane lines in the forward image and adjust it to 
# work with the top-down image
# lane_parameters = {'color_s': (150, 255), 'sobel_abs_x_l': (25, 100, 5),\
#        'combine': [['color_s', 'color_s'],['sobel_abs_x_l', 'sobel_abs_x_l']]}
lane_parameters = [ \
    {'color_s': (140, 255), 'sobel_abs_x_l': (25, 100, 9), 'sobel_abs_y_l': (25, 100, 5), \
       'combine': [['color_s', 'color_s'],['sobel_abs_x_l', 'sobel_abs_x_l']]},
    {'color_s': (180, 255), 'single': 'color_s' }, \
    {'color_h': (0, 100), 'single': 'color_h' }, \
    {'color_l': (50, 150), 'single': 'color_l' }, \
    {'sobel_abs_x_l': (25, 100, 9), 'single': 'sobel_abs_x_l' }, \
    {'sobel_abs_y_l': (25, 100, 9), 'single': 'sobel_abs_y_l' }, \
    {'sobel_mag_b_l': (50,100, 9), 'single': 'sobel_mag_b_l'}, \
    {'sobel_dir_l': (0.7, 1.2, 15), 'single': 'sobel_dir_l'}, \
    {'color_s': (140, 255), 'sobel_abs_x_l': (25, 100, 9), 'sobel_dir_l': (0.8, 1.2, 15), \
       'combine': [['color_s', 'color_s'],['sobel_abs_x_l', 'sobel_dir_l']]} \
    ]

# Undistort straight line image
straight_name = 'test_images/straight_lines1.jpg'
straight = mpimg.imread(straight_name)
undist = undistort(straight)

# Save output image to disk for review
output_name = straight_name.replace('test_images/straight', 'output_images/undistorted_straight')
mpimg.imsave(output_name, undist)

visualize_topdown(undist, 0)
```


![png](output_25_0.png)


**Note:** 

The lane_parameters list in the above code cell looks different than the test_parameters list used for identifying lanes in the forward-looking images.  This is because performing binary thresholding on top-down view images are an entirely different challenge than forward looking images.  The final set of lane parameters (index = 8) also adds a sobel direction gradient which helped to eliminate the shoulder lines from showing up prominently in the binary image. The code cell below produces the top-down binary images without the direction gradient.  Notice the prominence of the shoulder line on the top left corners of the test1.jpg, test4.jpg, and test6.jpg images.


```python
# No direction gradient (lane_parameters index = 0)
binary_images = []
for file_name, image in hard_images:
    binary, warped = make_topdown_binary(image, 0)
    binary_images.append((file_name, binary))
plot_images(binary_images, 4)
```


![png](output_27_0.png)


Now we'll also apply the direction gradient (parameter index 8 instead of 0)


```python
# with direction gradient (lane_parameters index = 8), the shoulder lines are removed.
binary_images = []
for file_name, image in hard_images:
    binary, warped = make_topdown_binary(image, 8)
    binary_images.append((file_name, binary))
plot_images(binary_images, 4)
```


![png](output_29_0.png)


Note that while the shoulder lines were removed above, we still have some white pixels from the car in the next lane as well as the stretch of shadows across the test5.jpg image.  These artifacts will interfere with trying to find the lane line pixels using the histogram method in the next step.  One way to reduce/remove the shadow effect in test5.jpg is to increase the lower threshold of the saturation channel thresholding (color_s), say to 170 or more (try index = 1).  However, this also drastically reduce the lane line detected by the saturation thresholding.  I have not found a way to remove these artifacts without also massively reducing lane line detection in other instances and so had to settle on a compromise lower saturation thresholding value.


```python
# now show easy test images with lane_parameters index = 8 (with direction gradient)
# easy test images
binary_images = []
for file_name, image in easy_images:
    binary, warped = make_topdown_binary(image, 8)
    binary_images.append((file_name, binary))
plot_images(binary_images, 4)
```


![png](output_31_0.png)


Now we have the perspective transformation matrix **M** to use for transforming forward-looking camera images to topdown views in order to find the lane line pixels on the road for measuring lane curvatures.  The inverse transformation **Minv** is obtained by passing the coordinates in the reverse order to the cv2 getPerspectiveTransform method.

---
## 5. Lane Pixel Extraction

With the above binary images we now have the raw data for finding the pixels that represents the lane lines.  The next step is to extract them from the image.  We do this by looking at the histogram of pixel values near the bottom of the screen and then slide up the screen to follow the lane pixels.


```python
from math import floor

def create_lane_histogram_data(image, top, bottom, left, right):
    """
    Create the lane histogram data from some portion of the image.
    :param image: The binary image containing the topdown lane pixels
    :param image_fraction: How much of the image to use, from the bottom up.
    """
    return np.sum(image[top:bottom,left:right], axis=0)

def plot_lane_histograms(histogram_data_list, images_per_row, figure_size=(10,2)):
    """
    Plot the lane histogram data
    :param named_image_list: A list of tuples in the form (image_name, histogram_data)
    """
    rows = ceil(len(histogram_data_list) / images_per_row)
    f, axes = plt.subplots(rows, images_per_row, figsize=(figure_size[0], figure_size[1] * rows))
    f.tight_layout()
    index = 0
    for key, value in histogram_data_list:
        if rows == 1:
            axes[index].plot(value)        
            axes[index].set_title(key, fontsize=20)
        else:
            row = floor(index / images_per_row)
            col = index % images_per_row
            axes[row, col].plot(value)
            axes[row, col].set_title(key, fontsize=20)
        index += 1
    plt.subplots_adjust(left=0., right=1, top=1, bottom=0.)

image_size = hard_images[0][1].shape
top = floor(image_size[0] * 0.5)
bottom = image_size[0]
left = 0
right = image_size[1]
histogram_data = []
binaries = dict()
warps = dict()
for name, value in hard_images:
    binary, warped = make_topdown_binary(value, 8)
    histogram_data.append((name, create_lane_histogram_data(binary, top, bottom, left, right)))
    binaries[name] = binary
    warps[name] = warped
    
for name, value in easy_images:
    binary, warped = make_topdown_binary(value, 8)
    histogram_data.append((name, create_lane_histogram_data(binary, top, bottom, left, right)))
    binaries[name] = binary
    warps[name] = warped
plot_lane_histograms(histogram_data, 4)

```


![png](output_34_0.png)


As can be seen above, the lower row of easy images produced much cleaner histograms showing the lane locations than the harder images in the top row.  Now we need to extract the lane pixels using the sliding window approach and then convert them from pixel coordinates to world coordinates, which is done in the code cell below.

The key function **get_lane_pixels()** method takes as input a binary top-down view image and goes through the following steps:

1. Look for initial lane position by calculating the pixel intensity histograms on the left and right half of the image.
2. Using argmax(), finds the column of the maximum histogram value and use those for initial guesses of the left and right lane line positions.
3. Creates two sliding windows (called box for short) of dimensions box_height by box_half_width * 2 starting at the bottom of the image and centered on the initial left and right lane line position guesses in step 2, above.
4. Extract all non-zero pixels in the two sliding windows as 'lane pixels', the pixels that represent the lane lines.
5. Calculate the maximum histogram value like in steps 1 & 2 but only within the two sliding windows to determine the center for the next set of sliding windows.
    * Note that when no bright pixel is detected (argmax() returns 0), the previous lane position is used.
6. Move the two sliding windows up by their height, centered around the maximum histogram values determined in step 5.
7. Iterate steps 4 through 6 until the sliding window reaches the top of the image.
8. Finally, returns the arrays of x and y coordinates for the left and right lane lines.


```python
# start with the assumption that lane lines will be found on the two halves of the binary image.
image_height = image_size[0]
image_width = image_size[1]
side_margin = 100  # Hack here to disregard noise from other cars at edge of the image at cost of reduced FOV.
border = floor(image_width / 2)  # start with a border at the middle of the image... will adjust when lane lines found.
left_lane = histogram_data[3][1][side_margin:border].argmax() + side_margin  # pixel column of left lane line.
right_lane = histogram_data[3][1][border:image_width - side_margin].argmax() + border # right lane line pixel column.
# print(left_lane, right_lane)

# Will call our sliding window 'box' for short in variable names.
box_half_width = 50
box_height = 60

def get_lane_pixels(binary_topdown_image):
    leftx = []
    lefty = []
    rightx = []
    righty = []
    box_top = image_size[0] - box_height
    
    histogram_values = create_lane_histogram_data(binary_topdown_image, 0, image_size[0], 0, image_size[1])
    
    # start with the assumption that lane lines will be found on the two halves of the binary image.
    left_lane = histogram_values[side_margin:border].argmax() + side_margin  # pixel column of left lane line.
    right_lane = histogram_values[border:image_width - side_margin].argmax() + border # right lane line pixel column.
    
    while box_top >= 0:
        # Find lane line maximum value when box is slide up.  Will use for new lane line centers each iteration.
        histogram_values = create_lane_histogram_data(binary_topdown_image, box_top, box_top + box_height, 0, \
            image_size[1])
       
        # print("left lane:", left_lane)
        # print("right lane:", right_lane)
        
        # Left lane
        box_left = max(0, left_lane - box_half_width)
        box_right = min(image_width, left_lane + box_half_width - 1)
        box_bottom = box_top + box_height
        for row in range(box_top,box_bottom):            
            for col in range(box_left,box_right):
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
        for row in range(box_top,box_bottom):
            for col in range(box_left,box_right):
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
```

## 6.  Lane Curvature Calculation

Once we have the arrays of lane pixel x,y coordinates, we can apply a polynomial fit to calculate the radius of curvature.  Before that, however, we need to convert from pixel coordinates to world coordinates, using the conversion factors provided in the project guidelines.  The codes to convert coordinates, apply the polynomial fit using **numpy.polyfit**, and calculating the curvature radius are in the code cell below.


```python
# Convert to real world coordinates
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

test_image_index = 3

left_x, left_y, right_x, right_y = get_lane_pixels(binaries[hard_test_images[test_image_index]])

def convert_pixel_to_world(x, y):
    y = y * ym_per_pix
    x = x * xm_per_pix
    return x, y

def fit_lane_line(x, y):
    """
    Fit a second order polynomial to each fake lane line
    """
    fit = np.polyfit(y, x, 2)
    fitx = fit[0]*y**2 + fit[1]*y + fit[2]
    return fit, fitx

left_x, left_y = convert_pixel_to_world(left_x, left_y)
right_x, right_y = convert_pixel_to_world(right_x, right_y)   

left_fit, left_fit_x = fit_lane_line(left_x, left_y)
right_fit, right_fit_x = fit_lane_line(right_x, right_y)

# Calculate curve radius
def curve_radius(y, fit):
    max_y = np.max(y)
    radius = ((1 + (2*fit[0]*max_y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return radius

left_curverad = curve_radius(left_y, left_fit)
right_curverad = curve_radius(right_y, right_fit)

# Now our radius of curvature is in meters
print("Curvature radius in meters: {:.2f} m  {:.2f} m".format(left_curverad, right_curverad))
print("Curvature radius in feet: {:.2f} ft {:.2f} ft".format(left_curverad * 3, right_curverad * 3))
```

    Curvature radius in meters: 815.71 m  529.09 m
    Curvature radius in feet: 2447.12 ft 1587.26 ft


The above radii in feet falls within the 800 ft through 4500 ft range of highway curvature radii specified at http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC.

Now let's plot the polygon fit to see what it looks like.


```python
# Plot up the fake data
plt.plot(left_x, left_y, 'o', color='red')
plt.plot(right_x, right_y, 'o', color='blue')
plt.plot(left_fit_x, left_y, color='green', linewidth=3)
plt.plot(right_fit_x, right_y, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images
```


![png](output_40_0.png)


## 7. Drawing the Lane Lines back on the Road Images

Now we're ready to draw the lane lines back on the road images by reversing the perspective transform.


```python
# Get the inverse perspective tranform
Minv = cv2.getPerspectiveTransform(topdown_coords, forward_coords)
yvals = np.array([y for y in range(floor(image_height * 0.5), image_height, 3)])
_, yvals = convert_pixel_to_world(np.array([0]), yvals)

def draw_lane_line(image, lane_points, color, thickness):
    last_point = None
    for point in lane_points:
        if last_point is not None:
            cv2.line(image, last_point, point, color, thickness)
        last_point = point

def clean_lane_points(lane_points):
    points = lane_points[0]
    last_point = None
    cleaned_points = []
    for point in points:
        if last_point is not None and point[0] == last_point[0] and point[1] == last_point[1]:
            continue
        last_point = point
        cleaned_points.append((int(point[0]), int(point[1])))
    return cleaned_points

def draw_lane_polygon(image, left_fit, right_fit, Pinv, lane_color, line_color, thickness):
    # Create an image to draw the lines on
    lane_image = np.zeros_like(image).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    draw_left_x = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    draw_right_x = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly() and transform back to pixel coords.
    pts_left = np.array([np.transpose(np.vstack([draw_left_x, yvals]))]) / np.array([xm_per_pix, ym_per_pix])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([draw_right_x, yvals])))]) / \
        np.array([xm_per_pix, ym_per_pix])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lane_image, np.int_([pts]), lane_color)

    # Draw the lane lines
    pts_left = clean_lane_points(pts_left)
    pts_right = clean_lane_points(pts_right)    
    
    draw_lane_line(lane_image, pts_left, line_color, thickness)
    draw_lane_line(lane_image, pts_right, line_color, thickness)   

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(lane_image, Pinv, (image.shape[1], image.shape[0]))
    
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

text_color = (255, 255, 128)
lane_color = (0, 255, 0)
line_color = (255, 0, 0)
thickness = 50

image = hard_images[test_image_index][1]
# Combine the result with the original image
result = draw_lane_polygon(image, left_fit, right_fit, Minv, lane_color, line_color, thickness)

plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7feec00fd6a0>




![png](output_42_1.png)


## 8. Visual Display of Lane Curvature and Vehicle 

Now all that's left is to write the curvature radii and the vehicle position to the display.  From the variations in the lane detection, the left and right lane will usually have different curvatures.  For steering purposes, we can use the average of the two curvatures.  However, for display purposes I will show the two individual lane curvatures.




```python
       
def draw_text(image, text, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text , position, font, 1, color, 2,cv2.LINE_AA)

def draw_curvature_text(image, left_radius, right_radius):
    left_radius_text  = "Left Curvature Radius:  {:.2f} m".format(left_radius) 
    right_radius_text = "Right Curvature Radius: {:.2f} m".format(right_radius) 

    draw_text(image, left_radius_text, (700, 50), text_color)    
    draw_text(image, right_radius_text, (700, 90), text_color)    
    
def draw_center_distance_text(image, distance, text_color):
    distance_text = "Distance From Center:   {:.2f} m".format(distance)
    draw_text(image, distance_text, (700, 130), text_color)
    
draw_curvature_text(result, left_curverad, right_curverad)

def distance_from_center(left_fit, right_fit):
    # Establish the base for vehical position at the bottom of the camera image.
    base = image.shape[0] * ym_per_pix
    
    # Apply polynomial fit to the base to determine lane position at the vehicle position.
    left_base_pos = (left_fit[0]*base**2 + left_fit[1]*base + left_fit[2])
    right_base_pos = (right_fit[0]*base**2 + right_fit[1]*base + right_fit[2])

    # lane center in pixel coordinates
    pixel_lane_center = ((right_base_pos - left_base_pos) / 2.0 + left_base_pos) / xm_per_pix

    # vehicle center assumed to be center of camera image.  Use that to compute distance from lane center.
    world_distance_from_center = (pixel_lane_center - (image.shape[1] / 2.0)) * xm_per_pix
    return world_distance_from_center

center_distance = distance_from_center(left_fit, right_fit)
draw_center_distance_text(result, center_distance, text_color)

plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x7feec0306240>




![png](output_44_1.png)


The lane lines, the lane polygon, and the curvature radii of the lanes are all superimposed over the above image.  This concludes all the tasks on the list at the beginning of this notebook.  This once-through established all the code to perform these tasks.  We can also put all the fitting code and drawing code into a single method, __find_and_draw_lanes()__, that takes just the road image as input to simply the use of this function later:


```python
pipeline_index = 8

def find_and_draw_lanes(image):
    binary, _ = make_topdown_binary(image, pipeline_index)
    # Extract lane pixesl
    left_x, left_y, right_x, right_y = get_lane_pixels(binary)
    # Convert to world coordinates
    left_x, left_y = convert_pixel_to_world(left_x, left_y)
    right_x, right_y = convert_pixel_to_world(right_x, right_y)

    # Polynomial fit
    left_fit, left_fit_x = fit_lane_line(left_x, left_y)
    right_fit, right_fit_x = fit_lane_line(right_x, right_y)

    # Draw lane data on image and show
    result = draw_lane_polygon(image , left_fit, right_fit, Minv, lane_color, line_color, thickness)
    draw_curvature_text(result, left_curverad, right_curverad)
    draw_center_distance_text(result, distance_from_center(left_fit, right_fit), text_color)
    return result, binary
```

Finally, let's look at what all the test images look like with the lane detection applied.


```python
final_hard_images = []
final_hard_binaries = []
for name, value in hard_images:
    result, binary = find_and_draw_lanes(value)
    final_hard_images.append((name, result))
    final_hard_binaries.append((name, binary))
    
plot_images(final_hard_images, len(final_hard_images))
plot_images(final_hard_binaries, len(final_hard_binaries))

final_easy_images = []
final_easy_binaries = []
for name, value in easy_images:
    result, binary = find_and_draw_lanes(value)
    final_easy_images.append((name, result))
    final_easy_binaries.append((name, binary))
    
plot_images(final_easy_images, len(final_easy_images))
plot_images(final_easy_binaries, len(final_easy_binaries))
```


![png](output_48_0.png)



![png](output_48_1.png)



![png](output_48_2.png)



![png](output_48_3.png)


### Challenge Movies

Per the above images the lane detection for all the test images look decent.  It's time to apply the code in this notebook to the challenge movies.  A new **LaneLine** class, whose constructor is shown below, will be used to contain the results of the polynomial fit and other calculations, including moving averages of certain values, to help improve lane detection, smooth out rough changes, and recover from detection failure in certain challenging frames.  Note that the full code for this class is in the *lane_line.py* file.  All of the above functions that help with the process of identify lane lines from the forward camera images are store in the *detect_lane.py* file.  

The **LaneLine** class has two member functions, **get_lane_pixels()** and **detect_lane_line()**.  The **get_lane_pixels()** member function refactors the code from the method of the same name in this notebook to apply to a single lane instead of both lane lines and adds features to better handle pixel detection failures.  The **detect_lane_line()** member function obtains the lane pixels, applies the polynomial fit, stores results in the data members of the **LaneLine** class, and looks for and ignores big changes from previous successful detections.

The above approach runs very well with the project video.  However, there were still some instances of drastic line warping.  I added two functionalities that helped.  The first is to use an average of the last 15 fits, and the second is to just throw out the fit if the quadratic and linear coefficients change by for than 0.5.  I added a print statement whenever a fit is thrown out, so it can be seen during video generation, as shown in the console log of generating the outpout for *project_video.mp4*.

```
[MoviePy] Writing video output_project_video.mp4
 55%|████████████████████████████████████▋                              | 691/1261 [04:47<03:58,  2.38it/s]
[ -0.02087919   1.15049239 -15.7316698 ] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▌           | 1043/1261 [07:14<01:32,  2.36it/s]
[ 0.01643206 -0.76898005  9.04306726] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▋           | 1044/1261 [07:14<01:32,  2.35it/s]
[ 0.014073   -0.62240506  6.90326728] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▋           | 1045/1261 [07:14<01:32,  2.34it/s]
[ -0.01730299   0.89240031 -11.10001248] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▋           | 1046/1261 [07:15<01:31,  2.34it/s]
[ 0.01161201 -0.5835024   7.37099504] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▊           | 1047/1261 [07:15<01:31,  2.35it/s]
[ 0.01184795 -0.60753454  7.86526535] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▊           | 1048/1261 [07:16<01:30,  2.35it/s]
[ -0.02246881   1.20278833 -15.93235145] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▉           | 1049/1261 [07:16<01:30,  2.35it/s]
[  0.11002469  -5.76502903  76.07735777] Fit changed too much, detected = false
 83%|██████████████████████████████████████████████████████▉           | 1050/1261 [07:17<01:29,  2.36it/s]
[ -0.04310725   2.28777841 -29.5075999 ] Fit changed too much, detected = false
100%|█████████████████████████████████████████████████████████████████▉| 1260/1261 [08:46<00:00,  2.29it/s]
[MoviePy] Done.
```

With the above change there is only one problematic area during the run through project_video.mp4, between the 42 and 43 second mark where the shadow of a tree causes the right lane to be detected about a meter to the left of where it should be for a second.  The project video can be viewed at https://youtu.be/jMebxa4BiXU.

Note that in the project video I also displayed the fit parameters and their differences between frames on the video to help me work on improving lane line detections through rough patches.

## 9. Discussions

### Camera Calibration

This part has some still unavoidable manual work to put up a chessboard pattern and taking a bunch of photos, but after that OpenCV pretty much handle the rest.  The only area to potentially improve is to somehow make the whole process fully automated or require only a few images instead of 20-30.

### Color and Gradient Thresholding

This is where I spent a lot of time in this project, learning the behaviors of different computer vision techniques.  I am still a beginner here, and did found a combination of techniques that works well in the tested lighting condition and lane colors.  This one static combination of color and gradient filtering would not work for all lighting or road conditions, like at night with head lights or when the road is wet with rain. One particular weakness of the current setting is with dark areas, like with the tree's shadow at the 42 second mark in the project_video.mp4 file and also with the dark lines from cracks or uneven paving along the highway's direction or from a center barrier in the challenge_video.mp4 file.  These dark runs are often mistaken for lane lines, when only the lighter lane markings should be detected as such.  There should be some kind of color thresholding that can suppress dark areas and lines so that only bright markings will be recognized as potential lane lines.

Another improvement could be to adaptively select from among several methods and/or use varying parameters to find the best detection method for any given road condition.  To this there an objective function of what good lane line detection would look like and then see which among several filtering combinations work well.  Another dynamic approach is to somehow find good complementary approaches dynamically, instead of just matching two complementary approaches like was done above.

### Perspective Transform Limitations and Improvements

Flat road approximation for transform points selection is one limitation of this implementation.  If the road is sloping up or down, the perspective transform will be incorrect and also the resulting radius of curvature calculations.  I wonder if information from the drive train with regards to more or less resistance from going uphill or downhill could be calibrated to produce the slope grade of where the car is driving, or some other techiques that could yield the same information to make this step more robust to the sloping of the road.

### Lane Curvature Calculations

Once the lane pixels are identified from the top-down view, applying polynomial fit to measure radius of curvature is clever.  However, there are stretches of roads where the measured radius of curvature is substantially different between the two lane lines. In the project video, the solid yellow lane line on the left allowed for much better detection than the intermittent white lane lines.  This means more improvements need to be made to the gradient and color thresholding so that there's more data to produce a more robust fit.

This step also yielded information like differences in the fit parameters from one frame to the next that can be used to quantify erroneous lane detection.  Averaging fit parameters from several previous frames also helped to smooth out the lane detection.  I also experimented with changes in the curvature radius here, but I couldn't find a consistent pattern that would yield consistent and accurate result.

### Visualization

Drawing the detected lane line and lane polygon back on the forward-looking road image is a powerful way to demonstrate the effectiveness (and lack thereof) of lane line detection.  It's easy to see when things works reasonably well, like in the project video and when thing don't work well at all, like in the challenge video.  Shortcomings of the current technique are revealed in helpful ways, with exactly what road or lighting condition is causing the problem.  This is very helpful for debugging.

Displaying various parameters related to lane detection on the forward-looking video image is also helpful with debugging, since the parameters affecting a particular frame can be seen with that frame.  I also find being able to see the binary thresholding images very helpful.  I would actually like to make a video with the forward-looking images and their corresponding binary thresholding images right next to them in a single video.  If there's any tip to do that I would very much appreciate it.

### Summary

This is a very interesting project.  However, manually working through the various binary thresholding options was very time consuming.  The intuition provided in the project was helpful, but there are many numeric parameters to play with in addition to the various techniques.  I probably need to invest some time in the computer vision course to have more experience with how these techniques work, particularly the color thresholding techniques.



```python

```
