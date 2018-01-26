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
import numpy as np
from scipy.spatial import cKDTree


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

        if len(coord) not in [3, 4]:
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
            raise TypeError("All coordinates must be int and the value can be a float or a int. x=" + str(self.x) + ", y=" + str(self.y) + ", z=" + str(self.z) + ", value=" + str(self.value))

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ", " + str(self.value) + ")"

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z) + "," + str(self.value)

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
    labels_regions = {'PMJ': 50, 'PMG': 49,
                      'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                      'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16,
                      'T10': 17, 'T11': 18, 'T12': 19,
                      'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                      'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                      'Co': 30}

    convert_vertlabel2disklabel = {'PMJ': 'PMJ', 'PMG': 'PMG',
                                   'C1': 'PMG-C1', 'C2': 'C1-C2', 'C3': 'C2-C3', 'C4': 'C3-C4', 'C5': 'C4-C5',
                                   'C6': 'C5-C6', 'C7': 'C6-C7',
                                   'T1': 'C7-T1', 'T2': 'T1-T2', 'T3': 'T2-T3', 'T4': 'T3-T4', 'T5': 'T4-T5',
                                   'T6': 'T5-T6', 'T7': 'T6-T7', 'T8': 'T7-T8', 'T9': 'T8-T9',
                                   'T10': 'T9-T10', 'T11': 'T10-T11', 'T12': 'T11-T12',
                                   'L1': 'T12-L1', 'L2': 'L1-L2', 'L3': 'L2-L3', 'L4': 'L3-L4', 'L5': 'L4-L5',
                                   'S1': 'L5-S1', 'S2': 'S1-S2', 'S3': 'S2-S3', 'S4': 'S3-S4', 'S5': 'S4-S5',
                                   'Co': 'S5-Co'}

    regions_labels = {'50': 'PMJ', '49': 'PMG',
                      '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6', '7': 'C7',
                      '8': 'T1', '9': 'T2', '10': 'T3', '11': 'T4', '12': 'T5', '13': 'T6', '14': 'T7',
                      '15': 'T8', '16': 'T9', '17': 'T10', '18': 'T11', '19': 'T12',
                      '20': 'L1', '21': 'L2', '22': 'L3', '23': 'L4', '24': 'L5',
                      '25': 'S1', '26': 'S2', '27': 'S3', '28': 'S4', '29': 'S5',
                      '30': 'Co'}

    average_vert_length = {'PMJ': 30.0, 'PMG': 15.0, 'C1': 0.0,
                           'C2': 20.176514191661337, 'C3': 17.022090519403065, 'C4': 17.842111671016056,
                           'C5': 16.800356992319429, 'C6': 16.019212889311383, 'C7': 15.715854192723905,
                           'T1': 16.84466163681078, 'T2': 19.865049296865475, 'T3': 21.550165130933905,
                           'T4': 21.761237991438083, 'T5': 22.633281372803687, 'T6': 23.801974227738132,
                           'T7': 24.358357813758332, 'T8': 25.200266294477885, 'T9': 25.315272064638506,
                           'T10': 25.501856729317133, 'T11': 27.619238824308123, 'T12': 29.465119270009946,
                            'L1': 31.89272719870084, 'L2': 33.511890474486449, 'L3': 35.721413718617441}

    """
    {'T10': ['T10', 25.543101799896391, 2.0015883550878457], 'T11': ['T11', 27.192970855618441, 1.9996136135271434], 'T12': ['T12', 29.559890137292335, 2.0204112073304121], 'PMG': ['PMG', 12.429867526011929, 2.9899172582983007], 'C3': ['C3', 18.229087873095388, 1.3299710200291315], 'C2': ['C2', 18.859365127066937, 1.5764843286826156], 'C1': ['C1', 0.0, 0.0], 'C7': ['C7', 15.543004729447034, 1.5597730786882851], 'C6': ['C6', 15.967482996580138, 1.4698898678270345], 'PMJ': ['PMJ', 11.38265467206886, 1.5641456310519117], 'C4': ['C4', 17.486130819790912, 1.5888243108648978], 'T8': ['T8', 25.649136105105754, 4.6835454011234718], 'T9': ['T9', 25.581999112288241, 1.9565018840832449], 'T6': ['T6', 23.539740893750668, 1.9073272889977211], 'T7': ['T7', 24.388589291326571, 1.828160893366733], 'T4': ['T4', 22.076131620822075, 1.726133989579701], 'T5': ['T5', 22.402770293433733, 2.0157113843189087], 'T2': ['T2', 19.800131846755267, 1.7600195442391204], 'T3': ['T3', 21.287064228802027, 1.8123109081532691], 'T1': ['T1', 16.525065003339993, 1.6130238001641826], 'L2': ['L2', 34.382300279977912, 2.378543023223767], 'L3': ['L3', 34.804841812064133, 2.7878124182683481], 'L1': ['L1', 32.02934490161379, 2.7697447948338381], 'C5': ['C5', 16.878263370935201, 1.6952832966782569]}
    """

    list_labels = [50, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    potential_list_labels = [50, 49, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                             23, 24, 25, 26, 27, 28, 29, 30]

    def __init__(self, points_x=None, points_y=None, points_z=None, deriv_x=None, deriv_y=None, deriv_z=None, fname=None):
        # initialization of variables
        self.length = 0.0
        self.progressive_length = [0.0]
        self.progressive_length_inverse = [0.0]
        self.incremental_length = [0.0]
        self.incremental_length_inverse = [0.0]

        # variables used for vertebral distribution
        self.first_label, self.last_label = None, None
        self.disks_levels = None
        self.label_reference = None

        self.compute_init_distribution = False

        if fname is not None:
            # Load centerline data from file
            centerline_file = np.load(fname)

            self.points = centerline_file['points']
            self.derivatives = centerline_file['derivatives']

            if 'disks_levels' in centerline_file:
                self.disks_levels = centerline_file['disks_levels'].tolist()
                # convertion of levels to int for future use
                for i in range(len(self.disks_levels)):
                    self.disks_levels[i][3] = int(self.disks_levels[i][3])
                self.label_reference = str(centerline_file['label_reference'])
                self.compute_init_distribution = True
        else:
            # Load centerline data from points and derivatives in parameters
            if points_x is None or points_y is None or points_z is None or deriv_x is None or deriv_y is None or deriv_z is None:
                raise ValueError('Data must be provided to centerline to be initialized')
            self.points = np.array(list(zip(points_x, points_y, points_z)))
            self.derivatives = array(list(zip(deriv_x, deriv_y, deriv_z)))

        self.number_of_points = len(self.points)

        # computation of centerline features, based on points and derivatives
        self.compute_length()
        self.coordinate_system = [self.compute_coordinate_system(index) for index in range(0, self.number_of_points)]
        self.plans_parameters = [self.get_plan_parameters(index) for index in range(0, self.number_of_points)]

        self.matrices = stack([item[4] for item in self.coordinate_system])
        self.inverse_matrices = stack([item[5] for item in self.coordinate_system])
        self.offset_plans = array([item[3] for item in self.plans_parameters])

        # initialization of KDTree for enabling computation of nearest points in centerline
        self.tree_points = cKDTree(self.points)

        if self.compute_init_distribution:
            self.compute_vertebral_distribution(disks_levels=self.disks_levels, label_reference=self.label_reference)

    def compute_length(self):
        for i in range(0, self.number_of_points - 1):
            distance = sqrt((self.points[i][0] - self.points[i + 1][0]) ** 2 +
                            (self.points[i][1] - self.points[i + 1][1]) ** 2 +
                            (self.points[i][2] - self.points[i + 1][2]) ** 2)
            self.length += distance
            self.progressive_length.append(distance)
            self.incremental_length.append(self.incremental_length[-1] + distance)
        for i in range(self.number_of_points - 1, 0, -1):
            distance = sqrt((self.points[i][0] - self.points[i - 1][0]) ** 2 +
                            (self.points[i][1] - self.points[i - 1][1]) ** 2 +
                            (self.points[i][2] - self.points[i - 1][2]) ** 2)
            self.progressive_length_inverse.append(distance)
            self.incremental_length_inverse.append(self.incremental_length_inverse[-1] + distance)

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

    def compute_vertebral_distribution(self, disks_levels, label_reference='C1'):
        """
        This function computes the vertebral distribution along the centerline, based on the position of
        intervertebral disks in space. A reference label can be provided (default is top of C1) so that relative
        distances are computed from this reference.

        Parameters
        ----------
        disks_levels: list of coordinates with value [[x, y, z, value], [x, y, z, value], ...]
                        the value correspond to the intervertebral disk label

        label_reference: reference label from which relative position will be computed.
                        Must be on of self.labels_regions

        """
        self.disks_levels = disks_levels
        self.label_reference = label_reference

        # special case for C2, which might not be present because it is difficult to identify
        is_C2_here = False
        C1, C3 = None, None
        for level in disks_levels:
            if level[3] == 2:
                is_C2_here = True
            elif level[3] == 1:
                C1 = level
            elif level[3] == 3:
                C3 = level
        if not is_C2_here and C1 is not None and C3 is not None:
            disks_levels.append([(C1[0] + C3[0]) / 2.0, (C1[1] + C3[1]) / 2.0, (C1[2] + C3[2]) / 2.0, 2])

        labels_points = [0] * self.number_of_points
        self.l_points = [0] * self.number_of_points
        self.dist_points = [0] * self.number_of_points
        self.dist_points_rel = [0] * self.number_of_points
        self.index_disk, index_disk_inv = {}, []

        # extracting each level based on position and computing its nearest point along the centerline
        first_label, last_label = None, None
        for level in disks_levels:
            if level[3] in self.list_labels:
                coord_level = [level[0], level[1], level[2]]
                disk = self.regions_labels[str(int(level[3]))]
                nearest_index = self.find_nearest_index(coord_level)
                labels_points[nearest_index] = disk + '-0.0'
                self.index_disk[disk] = nearest_index
                index_disk_inv.append([nearest_index, disk])

                # Finding minimum and maximum label, based on list_labels, which is ordered from top to bottom.
                index_label = self.list_labels.index(int(level[3]))
                if first_label is None:
                    first_label = index_label
                if index_label < first_label:
                    first_label = index_label
                if last_label is None:
                    last_label = index_label
                if index_label > last_label:
                    last_label = index_label

        if first_label is not None:
            self.first_label = self.list_labels[first_label]
        if last_label is not None:
            if last_label == len(self.list_labels):
                last_label -= 1
            self.last_label = self.list_labels[last_label]

        from operator import itemgetter
        index_disk_inv.append([0, 'bottom'])
        index_disk_inv = sorted(index_disk_inv, key=itemgetter(0))

        progress_length = zeros(self.number_of_points)
        for i in range(self.number_of_points - 1):
            progress_length[i + 1] = progress_length[i] + self.progressive_length[i]

        self.label_reference = label_reference
        if self.label_reference not in self.index_disk:
            upper = 31
            label_reference = ''
            for l in self.index_disk:
                if self.labels_regions[l] < upper:
                    label_reference = l
                    upper = self.labels_regions[l]
            self.label_reference = label_reference

        self.distance_from_C1label = {}
        for disk in self.index_disk:
            self.distance_from_C1label[disk] = progress_length[self.index_disk[self.label_reference]] - progress_length[self.index_disk[disk]]

        for i in range(1, len(index_disk_inv)):
            for j in range(index_disk_inv[i - 1][0], index_disk_inv[i][0]):
                self.l_points[j] = index_disk_inv[i][1]

        for i in range(self.number_of_points):
            self.dist_points[i] = progress_length[self.index_disk[self.label_reference]] - progress_length[i]

        for i in range(self.number_of_points):
            current_label = self.l_points[i]

            if current_label == 0:
                if 'PMG' in self.index_disk:
                    self.dist_points_rel[i] = self.dist_points[i] - self.dist_points[self.index_disk['PMG']]
                    continue
                else:
                    current_label = 'PMG'

            if self.list_labels.index(self.labels_regions[current_label]) < self.list_labels.index(self.first_label):
                reference_level_position = self.dist_points[self.index_disk[self.regions_labels[str(self.first_label)]]]
                self.dist_points_rel[i] = self.dist_points[i] - reference_level_position

            elif self.list_labels.index(self.labels_regions[current_label]) >= self.list_labels.index(self.last_label):
                reference_level_position = self.dist_points[self.index_disk[self.regions_labels[str(self.last_label)]]]
                self.dist_points_rel[i] = self.dist_points[i] - reference_level_position

            else:
                index_current_label = self.list_labels.index(self.labels_regions[self.l_points[i]])

                if self.list_labels.index(self.first_label) <= index_current_label < self.list_labels.index(self.last_label):
                    next_label = self.regions_labels[str(self.list_labels[index_current_label + 1])]

                    if current_label in ['PMJ', 'PMG']:
                        if next_label in self.index_disk:
                            self.dist_points_rel[i] = - (self.dist_points[i] - self.dist_points[self.index_disk[next_label]]) / abs(self.dist_points[self.index_disk[next_label]] - self.dist_points[self.index_disk[current_label]])
                        else:
                            self.dist_points_rel[i] = (self.average_vert_length[current_label] - self.dist_points[i] + self.dist_points[self.index_disk[current_label]]) / self.average_vert_length[current_label]
                    else:
                        next_label = self.regions_labels[str(self.list_labels[self.list_labels.index(self.labels_regions[self.l_points[i]]) + 1])]
                        if next_label in self.index_disk:
                            self.dist_points_rel[i] = (self.dist_points[i] - self.dist_points[self.index_disk[current_label]]) / abs(self.dist_points[self.index_disk[next_label]] - self.dist_points[self.index_disk[current_label]])
                        else:
                            self.dist_points_rel[i] = (self.dist_points[i] - self.dist_points[self.index_disk[current_label]]) / self.average_vert_length[current_label]

    def get_closest_to_relative_position(self, vertebral_level, relative_position, mode='levels'):
        """
        Args:
            vertebral_level: 
            relative_position: 
                if mode is 'levels', it is the relative position [0, 1] from upper disk
                if mode is 'length', it is the relative position [mm] from C1 top
            mode: {'levels', 'length'}

        Returns:
        """
        if mode == 'levels':
            indexes_vert = np.argwhere(np.array(self.l_points) == vertebral_level)
            if len(indexes_vert) == 0:
                return None
            # find closest
            arr_dist_rel = np.array(self.dist_points_rel)
            idx = np.argmin(np.abs(arr_dist_rel[indexes_vert] - relative_position))
            result = indexes_vert[idx]

        elif mode == 'length':
            result = np.argmin(np.abs(np.array(self.dist_points) - relative_position))
        else:
            raise ValueError("Mode must be either 'levels' or 'length'.")

        if isinstance(result, list) or isinstance(result, np.ndarray):
            result = result[0]
        return result

    def get_closest_to_absolute_position(self, vertebral_level, relative_position, backup_index=None, backup_centerline=None, mode='levels'):
        if mode == 'levels':
            if vertebral_level == 0:  # above the C1 vertebral level, the method used is length
                if backup_centerline is not None:
                    position_reference_backup = backup_centerline.dist_points[backup_centerline.index_disk[backup_centerline.regions_labels[str(self.first_label)]]]
                    position_reference_self = self.dist_points[self.index_disk[self.regions_labels[str(self.first_label)]]]
                    relative_position_from_reference_backup = backup_centerline.dist_points[backup_index] - position_reference_backup
                    result = np.argmin(np.abs(np.array(self.dist_points) - position_reference_self - relative_position_from_reference_backup))
                else:
                    result = np.argmin(np.abs(np.array(self.dist_points) - relative_position))
            else:
                vertebral_number = self.labels_regions[vertebral_level]
                if self.potential_list_labels.index(vertebral_number) < self.list_labels.index(self.first_label):
                    if backup_centerline is not None:
                        position_reference_backup = backup_centerline.dist_points[backup_centerline.index_disk[backup_centerline.regions_labels[str(self.first_label)]]]
                        position_reference_self = self.dist_points[self.index_disk[self.regions_labels[str(self.first_label)]]]
                        relative_position_from_reference_backup = backup_centerline.dist_points[backup_index] - position_reference_backup
                        result = np.argmin(np.abs(np.array(self.dist_points) - position_reference_self - relative_position_from_reference_backup))
                    else:
                        reference_level_position = self.dist_points[self.index_disk[self.regions_labels[str(self.first_label)]]]
                        result = np.argmin(np.abs(np.array(self.dist_points) - reference_level_position - relative_position))
                elif self.potential_list_labels.index(vertebral_number) >= self.list_labels.index(self.last_label):
                    if backup_centerline is not None:
                        position_reference_backup = backup_centerline.dist_points[backup_centerline.index_disk[backup_centerline.regions_labels[str(self.last_label)]]]
                        position_reference_self = self.dist_points[self.index_disk[self.regions_labels[str(self.last_label)]]]
                        relative_position_from_reference_backup = backup_centerline.dist_points[backup_index] - position_reference_backup
                        result = np.argmin(np.abs(np.array(self.dist_points) - position_reference_self - relative_position_from_reference_backup))
                    else:
                        reference_level_position = self.dist_points[self.index_disk[self.regions_labels[str(self.last_label)]]]
                        result = np.argmin(np.abs(np.array(self.dist_points) - reference_level_position - relative_position))
                else:
                    result = self.get_closest_to_relative_position(vertebral_level=vertebral_level, relative_position=relative_position)

        elif mode == 'length':
            result = self.get_closest_to_relative_position(vertebral_level=vertebral_level, relative_position=relative_position)
        else:
            raise ValueError("Mode must be either 'levels' or 'length'.")

        if isinstance(result, list) or isinstance(result, np.ndarray):
            result = result[0]
        return result

    def get_coordinate_interpolated(self, vertebral_level, relative_position, backup_index=None, backup_centerline=None, mode='levels'):
        index_closest = self.get_closest_to_absolute_position(vertebral_level, relative_position, backup_index=backup_index, backup_centerline=backup_centerline, mode=mode)
        if index_closest is None:
            return [np.nan, np.nan, np.nan]

        relative_position_closest = self.dist_points_rel[index_closest]
        coordinate_closest = self.get_point_from_index(index_closest)

        if relative_position < relative_position_closest:
            index_next = index_closest + 1
        else:
            index_next = index_closest - 1
        relative_position_next = self.dist_points_rel[index_next]
        coordinate_next = self.get_point_from_index(index_next)

        weight_closest = abs(relative_position - relative_position_closest) / abs(relative_position_next - relative_position_closest)
        weight_next = abs(relative_position - relative_position_next) / abs(relative_position_next - relative_position_closest)
        coordinate_result = [weight_closest * coordinate_closest[0] + weight_next * coordinate_next[0],
                             weight_closest * coordinate_closest[1] + weight_next * coordinate_next[1],
                             weight_closest * coordinate_closest[2] + weight_next * coordinate_next[2]]

        return coordinate_result

    def extract_perpendicular_square(self, image, index, size=20, resolution=0.5, interpolation_mode=0, border='constant', cval=0.0):
        x_grid, y_grid, z_grid = np.mgrid[-size:size:resolution, -size:size:resolution, 0:1]
        coordinates_grid = np.array(list(zip(x_grid.ravel(), y_grid.ravel(), z_grid.ravel())))
        coordinates_phys = self.get_inverse_plans_coordinates(coordinates_grid, np.array([index] * len(coordinates_grid)))
        coordinates_im = np.array(image.transfo_phys2continuouspix(coordinates_phys))
        square = image.get_values(coordinates_im.transpose(), interpolation_mode=interpolation_mode, border=border, cval=cval)
        return square.reshape((len(x_grid), len(x_grid)))

    def save_centerline(self, image=None, fname_output='centerline.sct'):
        if image is not None:
            image_output = image.copy()
            image_output.data = image_output.data.astype(np.float32)
            image_output.data *= 0.0

            for i in range(self.number_of_points):
                current_label = self.l_points[i]
                current_coord = self.points[i]
                current_dist_rel = self.dist_points_rel[i]
                if current_label in self.labels_regions:
                    coord_pix = image.transfo_phys2pix([current_coord])[0]
                    image_output.data[int(coord_pix[0]), int(coord_pix[1]), int(coord_pix[2])] = float(self.labels_regions[current_label]) + current_dist_rel

            image_output.setFileName(fname_output)
            image_output.save(type='float32')
        else:
            # save a .centerline file containing the centerline
            if self.disks_levels is None:
                np.savez(fname_output, points=self.points, derivatives=self.derivatives)
            else:
                np.savez(fname_output, points=self.points, derivatives=self.derivatives,
                         disks_levels=self.disks_levels, label_reference=self.label_reference)

    def average_coordinates_over_slices(self, image):
        # extracting points information for each coordinates
        P_x = np.array([point[0] for point in self.points])
        P_y = np.array([point[1] for point in self.points])
        P_z = np.array([point[2] for point in self.points])
        P_z_vox = np.array([coord[2] for coord in image.transfo_phys2pix(self.points)])
        P_x_d = np.array([deriv[0] for deriv in self.derivatives])
        P_y_d = np.array([deriv[1] for deriv in self.derivatives])
        P_z_d = np.array([deriv[2] for deriv in self.derivatives])

        P_z_vox = np.array([int(np.round(P_z_vox[i])) for i in range(0, len(P_z_vox))])
        # not perfect but works (if "enough" points), in order to deal with missing z slices
        for i in range(min(P_z_vox), max(P_z_vox) + 1, 1):
            if i not in P_z_vox:
                from bisect import bisect_right
                idx_closest = bisect_right(P_z_vox, i)
                z_min, z_max = P_z_vox[idx_closest - 1], P_z_vox[idx_closest]
                if z_min == z_max:
                    weight_min = weight_max = 0.5
                else:
                    weight_min, weight_max = abs((z_min - i) / (z_max - z_min)), abs((z_max - i) / (z_max - z_min))
                P_x_temp = np.insert(P_x, idx_closest, weight_min * P_x[idx_closest - 1] + weight_max * P_x[idx_closest])
                P_y_temp = np.insert(P_y, idx_closest, weight_min * P_y[idx_closest - 1] + weight_max * P_y[idx_closest])
                P_z_temp = np.insert(P_z, idx_closest, weight_min * P_z[idx_closest - 1] + weight_max * P_z[idx_closest])
                P_x_d_temp = np.insert(P_x_d, idx_closest, weight_min * P_x_d[idx_closest - 1] + weight_max * P_x_d[idx_closest])
                P_y_d_temp = np.insert(P_y_d, idx_closest, weight_min * P_y_d[idx_closest - 1] + weight_max * P_y_d[idx_closest])
                P_z_d_temp = np.insert(P_z_d, idx_closest, weight_min * P_z_d[idx_closest - 1] + weight_max * P_z_d[idx_closest])
                P_z_vox_temp = np.insert(P_z_vox, idx_closest, i)
                P_x, P_y, P_z, P_x_d, P_y_d, P_z_d, P_z_vox = P_x_temp, P_y_temp, P_z_temp, P_x_d_temp, P_y_d_temp, P_z_d_temp, P_z_vox_temp

        coord_mean = np.array([[np.mean(P_x[P_z_vox == i]), np.mean(P_y[P_z_vox == i]), np.mean(P_z[P_z_vox == i])] for i in range(min(P_z_vox), max(P_z_vox) + 1, 1)])
        x_centerline_fit = coord_mean[:, :][:, 0]
        y_centerline_fit = coord_mean[:, :][:, 1]
        coord_mean_d = np.array([[np.mean(P_x_d[P_z_vox == i]), np.mean(P_y_d[P_z_vox == i]), np.mean(P_z_d[P_z_vox == i])] for i in range(min(P_z_vox), max(P_z_vox) + 1, 1)])
        z_centerline = coord_mean[:, :][:, 2]
        x_centerline_deriv = coord_mean_d[:, :][:, 0]
        y_centerline_deriv = coord_mean_d[:, :][:, 1]
        z_centerline_deriv = coord_mean_d[:, :][:, 2]

        return x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv

    def display(self, mode='absolute'):
        """
        This function display the centerline using matplotlib. Two modes are available: absolute and relative.
        The absolute mode display the absolute position of centerline points.
        The relative mode display the centerline position relative to the reference label (default is C1). This mode
        requires the computation of vertebral distribution beforehand.
        Args:
            mode: {absolute, relative} see description of function for details
        """

        import matplotlib.pyplot as plt

        plt.figure(1)
        ax = plt.subplot(211)

        if mode is 'absolute':
            plt.plot([coord[2] for coord in self.points], [coord[0] for coord in self.points])
        else:
            position_reference = self.points[self.index_disk[self.label_reference]]
            plt.plot([coord[2] - position_reference[2] for coord in self.points],
                     [coord[0] - position_reference[0] for coord in self.points])
            for label_disk in self.labels_regions:
                if label_disk in self.index_disk:
                    point = self.points[self.index_disk[label_disk]]
                    plt.scatter(point[2] - position_reference[2], point[0] - position_reference[0], s=5)

        plt.grid()
        plt.title("X")
        #ax.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('x')
        ax = plt.subplot(212)

        if mode is 'absolute':
            plt.plot([coord[2] for coord in self.points], [coord[1] for coord in self.points])
        else:
            position_reference = self.points[self.index_disk[self.label_reference]]
            plt.plot([coord[2] - position_reference[2] for coord in self.points],
                     [coord[1] - position_reference[1] for coord in self.points])
            for label_disk in self.labels_regions:
                if label_disk in self.index_disk:
                    point = self.points[self.index_disk[label_disk]]
                    plt.scatter(point[2] - position_reference[2], point[1] - position_reference[1], s=5)

        plt.grid()
        plt.title("Y")
        #ax.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('y')
        plt.show()

    def get_lookup_coordinates(self, reference_image):
        nx, ny, nz, nt, px, py, pz, pt = reference_image.dim

        x, y, z, xd, yd, zd = self.average_coordinates_over_slices(reference_image)
        z_cov, coordinates = [], []
        for i in range(len(z)):
            nearest_index = self.find_nearest_indexes([[x[i], y[i], z[i]]])[0]
            disk_label = self.l_points[nearest_index]
            relative_position = self.dist_points_rel[nearest_index]
            if disk_label != 0:
                z_cov.append(int(reference_image.transfo_phys2pix([[x[i], y[i], z[i]]])[0][2]))
                if self.labels_regions[disk_label] > self.last_label and self.labels_regions[disk_label] not in [49, 50]:
                    coordinates.append(float(self.labels_regions[disk_label]) + relative_position / self.average_vert_length[disk_label])
                else:
                    coordinates.append(float(self.labels_regions[disk_label]) + relative_position)

        # concatenate results
        lookuptable_coordinates = []
        for zi in range(nz):
            if zi in z_cov:
                corresponding_values = z_cov.index(zi)
                lookuptable_coordinates.append(coordinates[corresponding_values])
            else:
                lookuptable_coordinates.append(None)

        return lookuptable_coordinates

    def compare_centerline(self, other, reference_image=None):
        """
        This function compute the mean square error and the maximum distance between two centerlines.
        If a reference image is provided, the distance metrics are computed on each slices where the both centerlines
        are present.
        Args:
            other: Centerline object
            reference_image: Image object

        Returns:
            mse, mean, std, max
        """
        distances = []
        mse = 0.0
        count_mean = 0

        if reference_image is not None:
            x, y, z, xd, yd, zd = self.average_coordinates_over_slices(reference_image)
            xo, yo, zo, xdo, ydo, zdo = other.average_coordinates_over_slices(reference_image)

            z_self = [reference_image.transfo_phys2pix([[x[i], y[i], z[i]]])[0][2] for i in range(len(z))]
            z_other = [reference_image.transfo_phys2pix([[xo[i], yo[i], zo[i]]])[0][2] for i in range(len(zo))]
            min_other, max_other = np.min(z_other), np.max(z_other)

            for index in range(len(z)):
                slice = z_self[index]

                if min_other <= slice <= max_other:
                    index_other = other.find_nearest_index([x[index], y[index], z[index]])
                    coord_other = other.points[index_other]
                    distance = (x[index] - coord_other[0])**2 + (y[index] - coord_other[1])**2 + (z[index] - coord_other[2])**2
                    distances.append(sqrt(distance))
                    mse += distance
                    count_mean += 1

        else:
            raise ValueError('Computation of centerline validation metrics without reference images is not yet '
                             'available. Please provide a reference image.')

        mse = sqrt(mse / float(count_mean))
        mean = np.mean(distances)
        std = np.std(distances)
        max = np.max(distances)
        return mse, mean, std, max, distances
