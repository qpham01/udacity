# Vehicle Detection Project

## Project Goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./examples/car.png
[notcar]: ./examples/notcar.png
[hls]: ./examples/hls.png
[luvhist]: ./examples/luvhist.png
[carnot]: ./examples/cars_and_notcars.png
[colclas]: ./examples/color_classify.png
[hogclas]: ./examples/hog_classify.png
[slidewin]: ./examples/sliding_windows.png
[rgbclas]: ./examples/rgb_classify.png
[testdet]: ./examples/test_detections.png
[test860]: ./examples/detect_8_60.png
[multiscale]: ./examples/multiscale.png
[heatmap]: ./examples/heatmap.png
[boximgs]: ./examples/box_images.png
[heatimgs]: ./examples/heat_images.png
[labelimg]: ./examples/label_image.png
[bboximg]: ./examples/bbox_image.png

## Overview

### Work Log

The Jupyter notebook **vehicle_detection.ipynb** contains a full log of my work through the various lessons leading up to the implementation of this project.  This write up is a section-by-section summary of the work done and the results from that notebook.  Please refer to the notebook as necessary to see more details of the work described herein.

### Example Images

The output_images folder contains various processings of the test images test1.jpg through test6.jpg, including multi-scale windows, heat map, and final bounding boxes.

### Training Data

I am using the vehicles and non-vehicles datasets provided for this project.

### Project Video

Here is the URL to the project video: 

## Color Space Exploration

The first section of the **vehicle_detection.ipynb** notebook is *Color Space Exploration*. In this section I plotted the color space representation of various color spaces for 3 car images (top row) and 3 non-car images (bottom row).  These images are shown here:

![carnot][carnot]

Below is an example plot of the HLS color space for the above images:

![hls][hls]

I also plotted the color histograms of these same 6 images.  The LUV example is shown below:

![luvhist][luvhist]

## Histogram of Oriented Gradients (HOG)

In the **HOG** section of **vehicle_detection.ipynb**, I extracted HOG features from a random car and non-car training images and plot the images with their gradient visualization and feature histogram.  Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![car][car]

![notcar][notcar]

The **get_hog_features()** method in the first code cell of the **HOG** section contains the code to extract the HOG features.

## Color Classify

In this section I tried different values for spatial and histogram binnings, starting with (32, 32) for spatial bins and 32 for histogram bins and varying from there.  I found that using (8, 8) spatial bins and 60 histogram bins resulted in similar training accuracy but is quicker by more than a factor of 10, 0.17 seconds as opposed to 2.58 seconds, as shown in the results here:

![color classify][colclas]

Later on I will user the (8, 8) spatial bins and 60 histogram bins in the training of the classifier.

## HOG Classify

In this section I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

The best performing color space and parameter combination is 9 orientations, 8 pixels_per_cell, and 2 cells_per_block with all hog channels, as shown below.

![hog classify][hogclas]

The classification accuracy at 1.0 is the best among all color space and parameter combinations I tried, and it also has among the smallest feature count at 5292 for more rapid predictions, clocking in at just over 2 milliseconds per prediction.

I trained a linear SVM using...

## Sliding Windows

In the sliding windows section I just implemented the sliding windows approach and drew rectangles on a test image to test it out.

![sliding windows][slidewin]

Nothing exciting here as the car detection with the classifier comes later.

## Search Classify

In this section I tried out car detection with different combinations of HOG features and other features like spatial features and histogram features.  For speed I am still using only 500 and 3000 data points among the provided data set.  This work here is mainly about testing out the full classification pipeline code.  Below is an image of classification using the RGB color space and all feature sets.  There are more combinations and results in the **vehicle_detection.ipynb** notebook.

![rgb classify][rgbclas]

## Hog Sub-Sampling Window Search

Now its time to do the full training of the classifier using the entire training dataset.  I trained the classifier with all training data and tried out the sub-sampling approach suggested in the lessons using the provided **find_cars()** method with the LUV color space and good parameter values discovered in my previous explorations.  I applied the classification to all 6 test images (test1.jpg through test6.jpg in the folder test_images) with the following results:

![test detections][testdet]

I tried training with a couple of different paramater sets, including (8, 8) spatial bins and 60 histogram bins that I found to be efficient and good in earlier exploration.  I saved the LinearSVC classifier and the X_scaler with these parameters to use in the project video.  The result from this classifier is here:

![detect 8 60][test860]

## Multi-Scale Windows

In this section I implemented multiple window scales.  I adjusted the **find_cars()** method to return rectangles (now renamed **find_car_rects()**).  Here is the result, with different colors showing detection windows at different scale:

![multiscale][multiscale]

## Heat Map

This section implements the heat map and bounding boxes.  This is the final step before video implementation.  Here are the results for the six test images:

![heatmap][heatmap]

### Video Implementation

The code of the video implementation is in **vehicle_detector.py**.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

The parameterizations of the different window scales are between lines 159 and 163, reproduced here:

    xstarts = [0, 0, 200]
    xstops = [1280, 1280, 1080]
    ystarts = [360, 360, 400]
    ystops = [720, 600, 480]
    scales = [2, 1.3, 0.6]

I also applied the heat map to detection boxes over multiple frames to counter false positives.  Below, I set the frame count to use for heatmap to 10 and the heat_threshold to 12 (lines 165-166), so that only areas with slightly more than one detection per frame will be counted to minimize false positives.  This higher bar could also result in the occasional detection dropout that needed further parameter tunings to fix.

    heat_threshold = 12
    frame_count = 10

The above numbers were arrived at after numerous trials and errors on various snippets of video.  The main issues are false positives and the occasional missing detection of the white car which lasted up to 2-3 seconds.  To address the missing detections I needed to reduce the lowest scale to 0.6.  This results in the frames being blown up to a larger size with a smaller relative scanning window and thus reduces the overall processing speed, but was necessary to keep up the detection of the cars through the entire video.  It also produces more false positives along the image edges in the tree lines, which are the most likely of all the terrains to be mistaken for cars.  To address the latter problem I had to add 200 pixels of margin on the left and right edges of the detection area for the 0.6 scale.

Here's a [link to my video result](https://youtu.be/WdKpBuRJjkA)

Below are example results showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video.  

Here are six frames and their corresponding heatmaps:

![heat images][heatimgs]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![label image][labelimg]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![bbox][bboximg]

Note that the above images were displayed by the notebook **process_images.ipynb**, but their generation is part of the video creation process at lines 211 through 225 in **vehicle_detector.py**.

---

## Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I basically followed the lessons on vehicle detection, used the provided data, and applied the lesson code with minor modifications.  Exploring the parameter and color spaces was instructive on seeing what worked and what didn't, as well as the relative effectiveness and computational efficiencies of various approaches.  There still a few false positives in the project video, as well as detection drop outs here and there, but the implemented version works well for nearly all of the video.

I really liked the approach in this project of collecting various color and image features from known labeled data, combining them into a single normalized feature vector, and then train a binary classifer to detect the vehicles on differences in this feature vector.  It's good to know the approach of combining classical computer vision and machine learning for image classification after my having used deep learning for much of this kind of work.

This is a great **first pass** at the problem of vehicle detection, but it seems we're just scratching the surface. I have to try quite a few sets of ystart/ystop/scaling parameters to get full detection of two cars throughout this 50 second project video with not much terrain or car variety.  This seems to be an indication of brittleness of what I've implemented and the data I've used so far.

To reliably detect vehicles across various terrains, road types, road conditions, slopes and elevations, lighting and weather conditions, etc., massive amount of training data will be needed, as well as some level of automatic adaptability.  Also, it really unclear how well a particular feature combination will work across all driving conditions, even with lots of data. Performance is also an issue with our python implementation for learning purposes.  Since we're combining results from multiple frames for more reliable detetion, and real driving will need multiple results per second for proper control, I imagine this whole pipeline will need to run at 30 iterations per second or more in the real world.

It would be cool to combine vehicle detection and lane detection from this course with pedestrian and road sign detection.  The combined system needs to run at multiple full updates per second so will provide an interesting optimization and computation challenge.
