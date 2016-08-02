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
from numpy import dot, cross, array, dstack, einsum, tile, multiply, stack, rollaxis
from numpy.linalg import norm, inv

class Point(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    # Euclidean distance
    def euclideanDistance(self, other_point):
        return sqrt(pow((self.x - other_point.x), 2) + pow((self.y - other_point.y), 2) + pow((self.z - other_point.z), 2))

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
        if other == 0:  # this check is necessary for using the function sum() of list
            other = Coordinate()
        return Coordinate([self.x + other.x, self.y + other.y, self.z + other.z, self.value])

    def __radd__(self, other):
        return self + other

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


class Centerline:
    """
    This class represents a centerline in an image. Its coordinates can be in voxel space as well as in physical space.
    A centerline is defined by its points and the derivatives of each point.
    When initialized, the lenght of the centerline is computed as well as the coordinate reference system of each plane.
    """
    def __init__(self, points_x, points_y, points_z, deriv_x, deriv_y, deriv_z):
        from math import sqrt
        #self.points = []
        self.derivatives = []
        self.length = 0.0
        self.progressive_length = [0.0]
        self.progressive_length_inverse = [0.0]

        self.points = array(zip(points_x, points_y, points_z))
        self.derivatives = array(zip(deriv_x, deriv_y, deriv_z))
        self.number_of_points = len(self.points)

        for i in range(0, self.number_of_points - 1):
            self.length += sqrt((points_x[i] - points_x[i + 1]) ** 2 +
                                (points_y[i] - points_y[i + 1]) ** 2 +
                                (points_z[i] - points_z[i + 1]) ** 2)

        self.coordinate_system = [self.compute_coordinate_system(index) for index in range(0, self.number_of_points)]
        self.plans_parameters = [self.get_plan_parameters(index) for index in range(0, self.number_of_points)]

        self.matrices = stack([item[4] for item in self.coordinate_system])
        self.inverse_matrices = stack([item[5] for item in self.coordinate_system])
        self.offset_plans = array([item[3] for item in self.plans_parameters])

        from scipy.spatial import cKDTree
        self.tree_points = cKDTree(dstack([points_x, points_y, points_z])[0])

    def compute_length(self, points_x, points_y, points_z):
        for i in range(0, self.number_of_points - 1):
            distance = sqrt((points_x[i] - points_x[i + 1]) ** 2 +
                            (points_y[i] - points_y[i + 1]) ** 2 +
                            (points_z[i] - points_z[i + 1]) ** 2)
            self.length += distance
            self.progressive_length.append(distance)
        for i in range(self.number_of_points, 1, -1):
            distance = sqrt((points_x[i] - points_x[i - 1]) ** 2 +
                            (points_y[i] - points_y[i - 1]) ** 2 +
                            (points_z[i] - points_z[i - 1]) ** 2)
            self.progressive_length_inverse.append(distance)

    def find_nearest_index(self, coord):
        """
        This function returns the index of the nearest point from centerline.
        Returns None if list of centerline points is empty.
        Raise an exception is the input coordinate has wrong format.
        :param coord: must be a numpy array [x, y, z]
        :return: index
        """
        if not self.points:
            return None

        dist, result_index = self.tree_points.query(coord)

        return result_index

    def find_nearest_indexes(self, array_coordinates):
        dist, result_indexes = self.tree_points.query(array_coordinates)
        return result_indexes

    def get_point_from_index(self, index):
        """
        Returns the coordinate of centerline at specified index.
        Raise an index error if the index is not in the list.
        :param index: int
        :return:
        """
        return self.points[index]

    def get_plan_parameters(self, index):
        """
        This function returns the parameters of the parametric equation of the plane at index.
        :param index: int
        :return: List of parameters [a, b, c, d], corresponding to plane parametric equation a*x + b*y + c*z + d = 0
        """
        if 0 <= index < self.number_of_points:
            a = self.derivatives[index][0]
            b = self.derivatives[index][1]
            c = self.derivatives[index][2]
            d = - (a * self.points[index][0] + b * self.points[index][1] + c * self.points[index][2])
        else:
            raise IndexError('ERROR in msct_types.Centerline.get_plan_parameters: index (' + str(index) + ') should be '
                             'within [' + str(0) + ', ' + str(self.number_of_points) + '[.')

        return [a, b, c, d]

    def get_distance_from_plane(self, coord, index, plane_params=None):
        """
        This function returns the distance between a coordinate and the plan at index position.
        If the derivative at index is nul (a, b, c = 0, 0, 0), a ValueError exception is raised.
        :param coord: must be a numpy array [x, y, z]
        :param index: int
        :param plane_params: list [a, b, c, d] with plane parameters. If not provided, these parameters are computed
        from index.
        :return:
        """
        if plane_params:
            [a, b, c, d] = plane_params
        else:
            [a, b, c, d] = self.plans_parameters[index]

        if a == 0 and b == 0 and c == 0:
            raise ValueError('ERROR in msct_types.Centerline.get_distance_from_plane: derivative at this location is '
                             'nul. Impossible to compute plane distance.')

        return (a * coord[0] + b * coord[1] + c * coord[2] + d) / sqrt(a * a + b * b + c * c)

    def get_distances_from_planes(self, coordinates, indexes):
        return (einsum('ij,ij->i', self.derivatives[indexes], coordinates) + self.offset_plans[indexes]) / norm(self.derivatives[indexes], axis=1)

    def get_nearest_plane(self, coord, index=None):
        """
        This function computes the nearest plane from the point and returns the parameters of its parametric equation
        [a, b, c, d] and the distance between the point and the plane.
        :param coord: must be a numpy array [x, y, z]
        :return: index, plane_parameters [a, b, c, d|, distance_from_plane
        """
        if index is None:
            index = self.find_nearest_index(coord)
        plane_params = self.plans_parameters[index]
        distance = self.get_distance_from_plane(coord, index, plane_params=plane_params)

        return index, plane_params, distance

    def compute_coordinate_system(self, index):
        """
        This function computes the cordinate reference system (X, Y, and Z axes) for a given index of centerline.
        :param index: int
        :return:
        """
        if 0 <= index < self.number_of_points:
            origin = self.points[index]
            z_prime_axis = self.derivatives[index]
            z_prime_axis /= norm(z_prime_axis)
            y_axis = array([0, 1, 0])
            y_prime_axis = (y_axis - dot(y_axis, z_prime_axis) * z_prime_axis)
            y_prime_axis /= norm(y_prime_axis)
            x_prime_axis = cross(y_prime_axis, z_prime_axis)
            x_prime_axis /= norm(x_prime_axis)

            matrix_base = array([[x_prime_axis[0], y_prime_axis[0], z_prime_axis[0]],
                                 [x_prime_axis[1], y_prime_axis[1], z_prime_axis[1]],
                                 [x_prime_axis[2], y_prime_axis[2], z_prime_axis[2]]])

            inverse_matrix = inv(matrix_base)
        else:
            raise IndexError('ERROR in msct_types.Centerline.compute_coordinate_system: index (' + str(index) + ') '
                             'should be within [' + str(0) + ', ' + str(self.number_of_points) + '[.')

        return origin, x_prime_axis, y_prime_axis, z_prime_axis, matrix_base, inverse_matrix

    def get_projected_coordinates_on_plane(self, coord, index, plane_params=None):
        """
        This function returns the coordinates of
        :param coord: must be a numpy array [x, y, z]
        :param index: int
        :param plane_params:
        :return:
        """
        if plane_params:
            [a, b, c, d] = plane_params
        else:
            [a, b, c, d] = self.plans_parameters[index]

        n = array([a, b, c])
        return coord - dot(coord - self.points[index], n) * n

    def get_projected_coordinates_on_planes(self, coordinates, indexes):
        return coordinates - multiply(tile(einsum('ij,ij->i', coordinates - self.points[indexes], self.derivatives[indexes]), (3, 1)).transpose(), self.derivatives[indexes])

    def get_in_plane_coordinates(self, coord, index):
        """
        This function returns the coordinate of the point from coord in the coordinate system of the plane.
        The point must be in the plane (you can use the function get_projected_coordinate_on_plane() to get it.
        :param coord: must be a numpy array [x, y, z]
        :param index: int
        :return:
        """
        if 0 <= index < self.number_of_points:
            origin, x_prime_axis, y_prime_axis, z_prime_axis, matrix_base, inverse_matrix = self.coordinate_system[index]
            return inverse_matrix.dot(coord - origin)
        else:
            raise IndexError('ERROR in msct_types.Centerline.compute_coordinate_system: index (' + str(index) + ') '
                             'should be within [' + str(0) + ', ' + str(self.number_of_points) + '[.')

    def get_in_plans_coordinates(self, coordinates, indexes):
        return einsum('mnr,nr->mr', rollaxis(self.inverse_matrices[indexes], 0, 3), (coordinates - self.points[indexes]).transpose()).transpose()

    def get_inverse_plans_coordinates(self, coordinates, indexes):
        return einsum('mnr,nr->mr', rollaxis(self.matrices[indexes], 0, 3), coordinates.transpose()).transpose() + self.points[indexes]
