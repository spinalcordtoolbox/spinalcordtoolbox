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

from __future__ import division
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
    def __init__(self, coord=None, mode='continuous'):
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
            if mode == 'index':
                int(self.x), int(self.y), int(self.z), float(self.value)
            else:
                float(self.x), float(self.y), float(self.z), float(self.value)
        except ValueError:
            raise TypeError("All coordinates must be int and the value can be a float or a int. x="+str(self.x)+", y="+str(self.y)+", z="+str(self.z)+", value="+str(self.value))

    def __repr__(self):
        return "("+str(self.x)+", "+str(self.y)+", "+str(self.z)+", "+str(self.value)+")"

    def __str__(self):
        return "("+str(self.x)+", "+str(self.y)+", "+str(self.z)+", "+str(self.value)+")"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def hasEqualValue(self, other):
        return self.value == other.value

    def __add__(self, other):
        return Coordinate([self.x + other.x, self.y + other.y, self.z + other.z, self.value])

    def __div__(self, scalar):
        return Coordinate([self.x / float(scalar), self.y / float(scalar), self.z / float(scalar), self.value])

    def __truediv__(self, scalar):
        return Coordinate([self.x / float(scalar), self.y / float(scalar), self.z / float(scalar), self.value])


class CoordinateValue(Coordinate):
    def __init__(self, coord=None, mode='index'):
        super(CoordinateValue, self).__init__(coord, mode)


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return float(self.value) == float(other.value)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.value)