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
from numpy import dot, cross, array, dstack, einsum, tile, multiply, stack, rollaxis, zeros
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

        self.compute_length(points_x, points_y, points_z)

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
        for i in range(self.number_of_points-1, 0, -1):
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
        if len(self.points) == 0:
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

    def compute_vertebral_distribution(self, disks_levels):
        """

        Parameters
        ----------
        vertebral_levels: list of coordinates with value [[x, y, z, value], [x, y, z, value], ...]
        the value correspond to the vertebral (disk) level label

        Returns
        -------

        """
        labels_regions = {'PONS': 50, 'MO': 51,
                          'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                          'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
                          'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                          'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                          'Co': 30}
        regions_labels = {'50': 'PONS', '51': 'MO',
                          '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6', '7': 'C7',
                          '8': 'T1', '9': 'T2', '10': 'T3', '11': 'T4', '12': 'T5', '13': 'T6', '14': 'T7', '15': 'T8', '16': 'T9', '17': 'T10', '18': 'T11', '19': 'T12',
                          '20': 'L1', '21': 'L2', '22': 'L3', '23': 'L4', '24': 'L5',
                          '25': 'S1', '26': 'S2', '27': 'S3', '28': 'S4', '29': 'S5',
                          '30': 'Co'}
        list_labels = [50, 51, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

        average_vert_length = {'T8': 24.137548673859101, 'T9': 24.791416310898683, 'T6': 23.57871316317943,
                               'T7': 24.390732464985319, 'T4': 21.88296642371796, 'T5': 22.894373460721738,
                               'T2': 20.605686554924716, 'T3': 20.941237157709139, 'T1': 19.512706875626321,
                               'T10': 26.730082218762107, 'T11': 29.065554485468454, 'T12': 29.785793712133625,
                               'C3': 17.279530661087819, 'C2': 60.263721005912174, 'C1': 27.820552850602741,
                               'C7': 16.803365329979044, 'C6': 15.773148081455446, 'C5': 16.234540144242303,
                               'C4': 16.13505684278411, 'L1': 30, 'L2': 30, 'L3': 30, 'PONS': 50, 'MO': 30}

        labels_points = [0] * self.number_of_points
        self.l_points = [0] * self.number_of_points
        self.dist_points = [0] * self.number_of_points
        self.dist_points_rel = [0] * self.number_of_points
        self.index_disk, index_disk_inv = {}, []
        for level in disks_levels:
            coord_level = [level[0], level[1], level[2]]
            disk = regions_labels[str(level[3])]
            nearest_index = self.find_nearest_index(coord_level)
            labels_points[nearest_index] = disk + '-0.0'
            self.index_disk[disk] = nearest_index
            index_disk_inv.append([nearest_index, disk])

        from operator import itemgetter
        index_disk_inv.append([0, 'bottom'])
        index_disk_inv = sorted(index_disk_inv, key=itemgetter(0))

        progress_length = zeros(self.number_of_points)
        for i in range(self.number_of_points - 1):
            progress_length[i+1] = progress_length[i] + self.progressive_length[i]

        self.distance_from_C1label = {}
        for disk in self.index_disk:
            self.distance_from_C1label[disk] = progress_length[self.index_disk['C1']] - progress_length[self.index_disk[disk]]

        for i in range(1, len(index_disk_inv)):
            for j in range(index_disk_inv[i - 1][0], index_disk_inv[i][0]):
                self.l_points[j] = index_disk_inv[i][1]

        for i in range(self.number_of_points):
            self.dist_points[i] = progress_length[self.index_disk['C1']] - progress_length[i]
        for i in range(self.number_of_points):
            current_label = self.l_points[i]
            if current_label == 'bottom' or current_label == 0:
                continue
            elif current_label in ['MO', 'PONS']:
                next_label = regions_labels[str(list_labels[list_labels.index(labels_regions[self.l_points[i]]) + 1])]
                if next_label in self.index_disk:
                    self.dist_points_rel[i] = - (self.dist_points[self.index_disk[next_label]] - self.dist_points[i]) / abs(self.dist_points[self.index_disk[next_label]] - self.dist_points[self.index_disk[current_label]])
                else:
                    self.dist_points_rel[i] = - (self.dist_points[self.index_disk[next_label]] - self.dist_points[i]) / average_vert_length[current_label]
            else:
                next_label = regions_labels[str(list_labels[list_labels.index(labels_regions[self.l_points[i]]) + 1])]
                if next_label in self.index_disk:
                    self.dist_points_rel[i] = (self.dist_points[i] - self.dist_points[self.index_disk[current_label]]) / abs(self.dist_points[self.index_disk[next_label]] - self.dist_points[self.index_disk[current_label]])
                else:
                    self.dist_points_rel[i] = (self.dist_points[i] - self.dist_points[self.index_disk[current_label]]) / average_vert_length[current_label]

        """
        for i in range(self.number_of_points):
            print l_points[i], dist_points_rel[i]
        """

    def get_closest_to_relative_position(self, vertebral_level, relative_position):
        import numpy as np
        indexes_vert = np.argwhere(np.array(self.l_points) == vertebral_level)
        if len(indexes_vert) == 0:
            return None

        # find closest
        arr_dist_rel = np.array(self.dist_points_rel)
        idx = np.argmin(np.abs(arr_dist_rel[indexes_vert] - relative_position))

        return indexes_vert[idx]

