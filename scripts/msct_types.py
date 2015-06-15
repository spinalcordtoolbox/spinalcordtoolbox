#!/usr/bin/env python
#########################################################################################
#
# msct_types
# This file contains many useful (and tiny) classes corresponding to data types.
# Large data types with many options have their own file (e.g., msct_image)
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2015-02-10
# Last modified: 2015-02-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from math import sqrt

class Point(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    # Euclidean distance
    def euclideanDistance(self, other_point):
        return sqrt(pow((self.x - self.x), 2) + pow((self.y - self.y), 2) + pow((self.z - self.z), 2))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

class Coordinate(Point):
    def __init__(self, coord=None):
        super(Coordinate, self).__init__()
        if coord is None:
            self.value = 0
            return

        if not isinstance(coord, list) and not isinstance(coord, str):
            raise TypeError("Coordinates parameter must be a list with coordinates [x, y, z] or [x, y, z, value] or a string with coordinates delimited by commas.")

        if isinstance(coord, str):
            # coordinate as a string. Values delimited by a comma.
            coord = coord.split(',')

        if len(coord) not in [3,4]:
            raise TypeError("Parameter must be a list with coordinates [x, y, z] or [x, y, z, value].")

        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        if len(coord) == 4:
            self.value = coord[3]
        else:
            self.value = 0
        # coordinates and value must be digits:
        try:
            int(self.x), int(self.y), int(self.z), int(self.value)
        except ValueError:
            raise TypeError("All coordinates and value must be digits. x="+str(self.x)+", y="+str(self.y)+", z="+str(self.z)+", value="+str(self.value))


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def hasEqualValue(self, other):
        return self.value == other.value
