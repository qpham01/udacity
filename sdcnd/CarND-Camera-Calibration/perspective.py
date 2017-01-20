import pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "calibration_wide/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('calibration_wide/GOPR0042.jpg')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    #print(corners.shape)
    #print(corners[0:4][0:4])
    # If found, draw corners
    warped = None
    M = None
    print(ret)
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)    
        # 4) If corners found: 
        # a) draw corners
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        source_list = []
        source_list.append(corners[0][0])  # Add first corner
        source_list.append(corners[nx - 1][0])  # Add first corner of last row
        source_list.append(corners[-1][0])  # Add very last corner (same as nx * ny - 1)
        source_list.append(corners[-nx][0]) # Add last corner on first row (same as nx * (ny - 1)
        source_points = np.float32(source_list)
        #print(source_points)
         #Note: you could pick any four of the detected corners 
         # as long as those four corners define a rectangle
         #One especially smart way to do this would be to use four well-chosen
         # corners that were automatically detected during the undistortion steps
         #We recommend using the automatic detection of corners in your code
                     
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        image_size = undist.shape[0:2]
        square_height = image_size[0] / ny
        square_width = image_size[1] / nx
        # Note that corner coordinates need to be in cv2 (x, y) order
        offset = 100
        destination_list = [[offset, offset], \
            [image_size[1] - offset, offset], \
            [image_size[1] - offset, image_size[0] - offset], \
            [offset, image_size[0] - offset]]
        destination_points = np.float32(destination_list)
        #print(destination_points)
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(source_points, destination_points)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
        cv2_image_size = (image_size[1], image_size[0]) # cv2 size is (width, height)
        warped = cv2.warpPerspective(undist, M, cv2_image_size, flags=cv2.INTER_LINEAR)
        #print(warped)
    return warped, M
    
top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
#print(top_down)
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
"""
mpimg.imsave('topdown.png', top_down)