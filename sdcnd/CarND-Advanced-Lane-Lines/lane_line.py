"""
Define a class to receive the characteristics of each line detection
"""
import numpy as np
from detect_lane import make_topdown_binary, convert_pixel_to_world, fit_lane_line, curve_radius, \
    create_lane_histogram_data

class LaneLine():
    """
    Encapsulates a lane line
    """
    def __init__(self, box_half_width=50, box_height=60, side_margin=100):
        """
        :param box_half_width: half the width of the detection area box.
        :param box_height: heigth of the detection area box.
        :param side_margin: a margin on the edge of the input image to ignore for just the first
            lane detection attempt along the bottom of the image.
        """
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # x value of the last fit
        self.fit_x = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        # recent fits
        self.recent_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # sliding detection box half width
        self.box_half_width = box_half_width
        # sliding detection box height
        self.box_height = box_height
        # side margin
        self.side_margin = side_margin
        # length of moving average lists
        self.list_length = 5

    def detect_lane_line(self, image, region):
        """
        Detect lane line within an undistorted image and initial region within the image.  This
        method updates data members of the LaneLine class with new detection data from the provided
        image.

        :param image: an undistorted forward facing image.
        :param region: a tuple (left, right) bounding the starting horizontal detection area.
        :return None
        """
        binary, _ = make_topdown_binary(image)

        # Extract lane box_half_width, box_height, side_margin
        self.get_lane_pixels(binary, region)

        if self.detected:
            # Convert to world coordinates
            self.allx, self.ally = convert_pixel_to_world(self.allx, self.ally)
            if len(self.allx) == 0:
                self.detected = False
                return

            # Polynomial fit
            fit, fit_x = fit_lane_line(self.allx, self.ally)
            diff = fit - self.current_fit
            # First coefficient varies by more than 1 means detection failure so
            # ignore current detection.
            if len(self.current_fit) == 3 and abs(diff[0][0]) > 1.0 or abs(diff[0][1]) > 2.0:
                self.detected = False
            else:
                self.current_fit = fit
                self.fit_x = fit_x
                self.recent_xfitted.append(fit_x)
                if len(self.recent_xfitted) > self.list_length:
                    del self.recent_xfitted[0]

                self.recent_fits.append(fit)
                if len(self.recent_fits) > self.list_length:
                    del self.recent_fits[0]

                #print("Current fit", self.current_fit)
                # Calculate best fit as average of last N fits
                self.best_fit = sum(self.recent_fits) / len(self.recent_fits)

                #print("Best fit", self.best_fit)
                self.radius_of_curvature = curve_radius(self.ally, self.current_fit)

    def get_lane_pixels(self, binary, region):
        """
        Extract the lane pixels from the topdown binary image.

        :param image: a topdown binary image of the road.
        :param region: a tuple (left, right) bounding the starting horizontal detection area.
        """
        image_height, image_width = binary.shape
        self.allx = []
        self.ally = []
        box_top = image_height - self.box_height

        # start with a border at the middle of the image... will adjust when lane lines found.
        left, right = region

        histogram_values = create_lane_histogram_data(binary, 0, image_height, 0, image_width)

        # pixel column of left lane line.
        lane = histogram_values[left:right].argmax()
        if lane == 0:
            # Couldn't detect the lane within the specified region this time.
            # Fall back on the previous detection location.
            self.detected = False
            lane = self.bestx
        else:
            self.detected = True
            lane += left

        self.line_base_pos = (image_height, self.bestx)

        while box_top >= 0:
            # Find lane line maximum value when box is slide up.  Will use for new lane line
            # centerseach iteration.
            histogram_values = create_lane_histogram_data(binary, box_top, box_top + \
                self.box_height, 0, image_width)

            # Left lane
            box_left = max(0, lane - self.box_half_width)
            box_right = min(image_width, lane + self.box_half_width - 1)
            box_bottom = box_top + self.box_height
            for row in range(box_top, box_bottom):
                for col in range(box_left, box_right):
                    if binary[row, col] > 0:
                        self.allx.append(col)
                        self.ally.append(row)
            # Slide left box to center of bright pixels
            last_lane = lane
            lane = histogram_values[box_left:box_right].argmax()
            if lane == 0:
                lane = last_lane
            else:
                lane += box_left

            # If new lane x coordinate is too far away, don't use it.
            if abs(lane - last_lane) > self.box_half_width:
                lane = last_lane

            # Slide box up
            box_top -= self.box_height

            # Moving average accounting
            if len(self.allx) > 0:
                self.recent_xfitted.append(self.allx)

            if len(self.recent_xfitted) > self.list_length:
                del self.recent_xfitted[0]
