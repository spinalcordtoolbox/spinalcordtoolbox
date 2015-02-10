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

class Coordinate(object):
    def __init__(self, coord=None):#x=0, y=0, z=0, value=0):
        if coord is None:
            self.x = 0
            self.y = 0
            self.z = 0
            self.value = 0
            return

        if not isinstance(coord, list) or len(coord) not in [3,4]:
            raise TypeError("Parameter must be a list with coordinates [x, y, z] or [x, y, z, value].")

        self.x = coord[0]
        self.y = coord[1]
        self.z = coord[2]
        if len(coord) == 4:
            self.value = coord[3]
        else:
            self.value = 0
        # coordinates and value must be digits:
        if not (isinstance(self.x, (int, long, float)) and isinstance(self.y, (int, long, float)) and isinstance(self.z, (int, long, float)) and isinstance(self.value, (int, long, float))):
            raise TypeError("All coordinates and value must be digits.")


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)