"""
LaneSegment class
"""
import math

class LaneSegment:
    """
    Encapsultes a lane segment defined by a raster image
    """
    def __init__(self, px1, py1, px2, py2):
        self.set_endpoints(px1, py1, px2, py2)

    def set_endpoints(self, px1, py1, px2, py2):
        """
        Set new endpoints for line segment.
        """
        self.px1 = px1
        self.py1 = py1
        self.px2 = px2
        self.py2 = py2
        self.d_x = px2 - px1
        self.d_y = py2 - py1
        self.abs_d_y = math.fabs(self.d_y)
        if self.d_x == 0:
            self.slope = float('NaN')
            self.abs_slope = float('NaN')
        else:
            self.slope = self.d_y / self.d_x
            self.abs_slope = math.fabs(self.slope)
        self.min_x = min(self.px1, self.px2)
        self.min_y = min(self.py1, self.py2)
        self.max_x = max(self.px1, self.px2)
        self.max_y = max(self.py1, self.py2)
