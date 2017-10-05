#!/usr/bin/env python
#########################################################################################
#
# All sort of utilities for labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2015-02-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: for vert-disc: make it faster! currently the module display-voxel is very long (esp. when ran on PAM50). We can find an alternative approach by sweeping through centerline voxels.
# TODO: label_disc: for top vertebrae, make label at the center of the cord (currently it's at the tip)
# TODO: check if use specified several processes.
# TODO: currently it seems like cross_radius is given in pixel instead of mm

import sys
import math
import numpy as np
from scipy import ndimage

from msct_parser import Parser
from msct_image import Image
import sct_utils as sct


class Param:
    def __init__(self):
        self.debug = 0
        self.fname_label_output = 'labels.nii.gz'
        self.labels = []
        self.cross_size = 5  # cross size in mm
        self.verbose = '1'


class ProcessLabels(object):
    def __init__(self, fname_label, fname_output=None, fname_ref=None, cross_radius=5, dilate=False,
                 coordinates=None, verbose=1, vertebral_levels=None, value=None):
        self.image_input = Image(fname_label, verbose=verbose)
        self.image_ref = None
        if fname_ref is not None:
            self.image_ref = Image(fname_ref, verbose=verbose)

        if isinstance(fname_output, list):
            if len(fname_output) == 1:
                self.fname_output = fname_output[0]
            else:
                self.fname_output = fname_output
        else:
            self.fname_output = fname_output
        self.cross_radius = cross_radius
        self.vertebral_levels = vertebral_levels
        self.dilate = dilate
        self.coordinates = coordinates
        self.verbose = verbose
        self.value = value

    def process(self, type_process):
        # for some processes, change orientation of input image to RPI
        change_orientation = False
        if type_process in ['vert-body', 'vert-disc', 'vert-continuous']:
            # get orientation of input image
            input_orientation = self.image_input.orientation
            # change orientation
            self.image_input.change_orientation('RPI')
            change_orientation = True
        if type_process == 'add':
            self.output_image = self.add(self.value)
        if type_process == 'cross':
            self.output_image = self.cross()
        if type_process == 'plan':
            self.output_image = self.plan(self.cross_radius, 100, 5)
        if type_process == 'plan_ref':
            self.output_image = self.plan_ref()
        if type_process == 'increment':
            self.output_image = self.increment_z_inverse()
        if type_process == 'disks':
            self.output_image = self.labelize_from_disks()
        if type_process == 'MSE':
            self.MSE()
            self.fname_output = None
        if type_process == 'remove':
            self.output_image = self.remove_label()
        if type_process == 'remove-symm':
            self.output_image = self.remove_label(symmetry=True)
        if type_process == 'centerline':
            self.extract_centerline()
        if type_process == 'create':
            self.output_image = self.create_label()
        if type_process == 'create-add':
            self.output_image = self.create_label(add=True)
        if type_process == 'create-seg':
            self.output_image = self.create_label_along_segmentation()
        if type_process == 'display-voxel':
            self.display_voxel()
            self.fname_output = None
        if type_process == 'diff':
            self.diff()
            self.fname_output = None
        if type_process == 'dist-inter':  # second argument is in pixel distance
            self.distance_interlabels(5)
            self.fname_output = None
        if type_process == 'cubic-to-point':
            self.output_image = self.cubic_to_point()
        if type_process == 'vert-body':
            self.output_image = self.label_vertebrae(self.vertebral_levels)
        if type_process == 'vert-continuous':
            self.output_image = self.continuous_vertebral_levels()
        if type_process == 'create-viewer':
            self.output_image = self.launch_sagittal_viewer(self.value)

        # save the output image as minimized integers
        if self.fname_output is not None:
            self.output_image.setFileName(self.fname_output)
            if change_orientation:
                self.output_image.change_orientation(input_orientation)
            if type_process == 'vert-continuous':
                self.output_image.save('float32')
            elif type_process != 'plan_ref':
                self.output_image.save('minimize_int')
            else:
                self.output_image.save()

    def add(self, value):
        """
        This function add a specified value to all non-zero voxels.
        """
        image_output = Image(self.image_input, self.verbose)
        # image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i, coord in enumerate(coordinates_input):
            image_output.data[int(coord.x), int(coord.y), int(coord.z)] = image_output.data[int(coord.x), int(coord.y), int(coord.z)] + float(value)
        return image_output

    def create_label(self, add=False):
        """
        Create an image with labels listed by the user.
        This method works only if the user inserted correct coordinates.

        self.coordinates is a list of coordinates (class in msct_types).
        a Coordinate contains x, y, z and value.
        If only one label is to be added, coordinates must be completed with '[]'
        examples:
        For one label:  object_define=ProcessLabels( fname_label, coordinates=[coordi]) where coordi is a 'Coordinate' object from msct_types
        For two labels: object_define=ProcessLabels( fname_label, coordinates=[coordi1, coordi2]) where coordi1 and coordi2 are 'Coordinate' objects from msct_types
        """
        image_output = self.image_input.copy()
        if not add:
            image_output.data *= 0

        # loop across labels
        for i, coord in enumerate(self.coordinates):
            # display info
            sct.printv('Label #' + str(i) + ': ' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ' --> ' +
                       str(coord.value), 1)
            if len(image_output.data.shape) == 3:
                image_output.data[int(coord.x), int(coord.y), int(coord.z)] = coord.value
            elif len(image_output.data.shape) == 2:
                assert str(coord.z) == '0', "ERROR: 2D coordinates should have a Z value of 0. Z coordinate is :" + str(coord.z)
                image_output.data[int(coord.x), int(coord.y)] = coord.value
        return image_output

    def create_label_along_segmentation(self):
        """
        Create an image with labels defined along the spinal cord segmentation (or centerline)
        Example:
        object_define=ProcessLabels(fname_segmentation, coordinates=[coord_1, coord_2, coord_i]), where coord_i='z,value'. If z=-1, then use z=nz/2 (i.e. center of FOV in superior-inferior direction)
        Returns
        -------
        image_output: Image object with labels.
        """
        # copy input Image object (will use the same header)
        image_output = self.image_input.copy()
        # set all voxels to 0
        image_output.data *= 0
        # loop across labels
        for i, coord in enumerate(self.coordinates):
            # split coord string
            list_coord = coord.split(',')
            # convert to int() and assign to variable
            z, value = [int(i) for i in list_coord]
            # if z=-1, replace with nz/2
            if z == -1:
                z = int(round(image_output.dim[2] / 2.0))
            # get center of mass of segmentation at given z
            x, y = ndimage.measurements.center_of_mass(np.array(self.image_input.data[:, :, z]))
            # round values to make indices
            x, y = int(round(x)), int(round(y))
            # display info
            sct.printv('Label #' + str(i) + ': ' + str(x) + ',' + str(y) + ',' + str(z) + ' --> ' + str(value), 1)
            if len(image_output.data.shape) == 3:
                image_output.data[x, y, z] = value
            elif len(image_output.data.shape) == 2:
                assert str(z) == '0', "ERROR: 2D coordinates should have a Z value of 0. Z coordinate is :" + str(z)
                image_output.data[x, y] = value
        return image_output

    def cross(self):
        """
        create a cross.
        :return:
        """
        output_image = Image(self.image_input, self.verbose)
        nx, ny, nz, nt, px, py, pz, pt = Image(self.image_input.absolutepath).dim

        coordinates_input = self.image_input.getNonZeroCoordinates()
        d = self.cross_radius  # cross radius in pixel
        dx = d / px  # cross radius in mm
        dy = d / py

        # clean output_image
        output_image.data *= 0

        cross_coordinates = self.get_crosses_coordinates(coordinates_input, dx, self.image_ref, self.dilate)

        for coord in cross_coordinates:
            output_image.data[int(round(coord.x)), int(round(coord.y)), int(round(coord.z))] = coord.value

        return output_image

    @staticmethod
    def get_crosses_coordinates(coordinates_input, gapxy=15, image_ref=None, dilate=False):
        from msct_types import Coordinate

        # if reference image is provided (segmentation), we draw the cross perpendicular to the centerline
        if image_ref is not None:
            # smooth centerline
            from sct_straighten_spinalcord import smooth_centerline
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(self.image_ref, verbose=self.verbose)

        # compute crosses
        cross_coordinates = []
        for coord in coordinates_input:
            if image_ref is None:
                from sct_straighten_spinalcord import compute_cross
                cross_coordinates_temp = compute_cross(coord, gapxy)
            else:
                from sct_straighten_spinalcord import compute_cross_centerline
                from numpy import where
                index_z = where(z_centerline == coord.z)
                deriv = Coordinate([x_centerline_deriv[index_z][0], y_centerline_deriv[index_z][0], z_centerline_deriv[index_z][0], 0.0])
                cross_coordinates_temp = compute_cross_centerline(coord, deriv, gapxy)

            for i, coord_cross in enumerate(cross_coordinates_temp):
                coord_cross.value = coord.value * 10 + i + 1

            # dilate cross to 3x3x3
            if dilate:
                additional_coordinates = []
                for coord_temp in cross_coordinates_temp:
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y, coord_temp.z - 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y + 1.0, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y + 1.0, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y + 1.0, coord_temp.z - 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y - 1.0, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y - 1.0, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x, coord_temp.y - 1.0, coord_temp.z - 1.0, coord_temp.value]))

                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y, coord_temp.z - 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y + 1.0, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y + 1.0, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y + 1.0, coord_temp.z - 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y - 1.0, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y - 1.0, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x + 1.0, coord_temp.y - 1.0, coord_temp.z - 1.0, coord_temp.value]))

                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y, coord_temp.z - 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y + 1.0, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y + 1.0, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y + 1.0, coord_temp.z - 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y - 1.0, coord_temp.z, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y - 1.0, coord_temp.z + 1.0, coord_temp.value]))
                    additional_coordinates.append(Coordinate([coord_temp.x - 1.0, coord_temp.y - 1.0, coord_temp.z - 1.0, coord_temp.value]))

                cross_coordinates_temp.extend(additional_coordinates)

            cross_coordinates.extend(cross_coordinates_temp)

        cross_coordinates = sorted(cross_coordinates, key=lambda obj: obj.value)
        return cross_coordinates

    def plan(self, width, offset=0, gap=1):
        """
        Create a plane of thickness="width" and changes its value with an offset and a gap between labels.
        """
        image_output = Image(self.image_input, self.verbose)
        image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for coord in coordinates_input:
            image_output.data[:, :, int(coord.z) - width:int(coord.z) + width] = offset + gap * coord.value

        return image_output

    def plan_ref(self):
        """
        Generate a plane in the reference space for each label present in the input image
        """

        image_output = Image(self.image_ref, self.verbose)
        image_output.data *= 0

        image_input_neg = Image(self.image_input, self.verbose).copy()
        image_input_pos = Image(self.image_input, self.verbose).copy()
        image_input_neg.data *= 0
        image_input_pos.data *= 0
        X, Y, Z = (self.image_input.data < 0).nonzero()
        for i in range(len(X)):
            image_input_neg.data[X[i], Y[i], Z[i]] = -self.image_input.data[X[i], Y[i], Z[i]]  # in order to apply getNonZeroCoordinates
        X_pos, Y_pos, Z_pos = (self.image_input.data > 0).nonzero()
        for i in range(len(X_pos)):
            image_input_pos.data[X_pos[i], Y_pos[i], Z_pos[i]] = self.image_input.data[X_pos[i], Y_pos[i], Z_pos[i]]

        coordinates_input_neg = image_input_neg.getNonZeroCoordinates()
        coordinates_input_pos = image_input_pos.getNonZeroCoordinates()

        image_output.changeType('float32')
        for coord in coordinates_input_neg:
            image_output.data[:, :, int(coord.z)] = -coord.value  # PB: takes the int value of coord.value
        for coord in coordinates_input_pos:
            image_output.data[:, :, int(coord.z)] = coord.value

        return image_output

    def cubic_to_point(self):
        """
        Calculate the center of mass of each group of labels and returns a file of same size with only a
        label by group at the center of mass of this group.
        It is to be used after applying homothetic warping field to a label file as the labels will be dilated.
        Be careful: this algorithm computes the center of mass of voxels with same value, if two groups of voxels with
         the same value are present but separated in space, this algorithm will compute the center of mass of the two
         groups together.
        :return: image_output
        """

        # 0. Initialization of output image
        output_image = self.image_input.copy()
        output_image.data *= 0

        # 1. Extraction of coordinates from all non-null voxels in the image. Coordinates are sorted by value.
        coordinates = self.image_input.getNonZeroCoordinates(sorting='value')

        # 2. Separate all coordinates into groups by value
        groups = dict()
        for coord in coordinates:
            if coord.value in groups:
                groups[coord.value].append(coord)
            else:
                groups[coord.value] = [coord]

        # 3. Compute the center of mass of each group of voxels and write them into the output image
        for value, list_coord in groups.iteritems():
            center_of_mass = sum(list_coord) / float(len(list_coord))
            sct.printv("Value = " + str(center_of_mass.value) + " : (" + str(center_of_mass.x) + ", " + str(center_of_mass.y) + ", " + str(center_of_mass.z) + ") --> ( " + str(round(center_of_mass.x)) + ", " + str(round(center_of_mass.y)) + ", " + str(round(center_of_mass.z)) + ")", verbose=self.verbose)
            output_image.data[int(round(center_of_mass.x)), int(round(center_of_mass.y)), int(round(center_of_mass.z))] = center_of_mass.value

        return output_image

    def increment_z_inverse(self):
        """
        Take all non-zero values, sort them along the inverse z direction, and attributes the values 1,
        2, 3, etc. This function assuming RPI orientation.
        """
        image_output = Image(self.image_input, self.verbose)
        image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z', reverse_coord=True)

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i, coord in enumerate(coordinates_input):
            image_output.data[int(coord.x), int(coord.y), int(coord.z)] = i + 1

        return image_output

    def labelize_from_disks(self):
        """
        Create an image with regions labelized depending on values from reference.
        Typically, user inputs a segmentation image, and labels with disks position, and this function produces
        a segmentation image with vertebral levels labelized.
        Labels are assumed to be non-zero and incremented from top to bottom, assuming a RPI orientation
        """
        image_output = Image(self.image_input, self.verbose)
        image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates(sorting='value')

        # for all points in input, find the value that has to be set up, depending on the vertebral level
        for i, coord in enumerate(coordinates_input):
            for j in range(0, len(coordinates_ref) - 1):
                if coordinates_ref[j + 1].z < coord.z <= coordinates_ref[j].z:
                    image_output.data[int(coord.x), int(coord.y), int(coord.z)] = coordinates_ref[j].value

        return image_output

    def label_vertebrae(self, levels_user=None):
        """
        Find the center of mass of vertebral levels specified by the user.
        :return: image_output: Image with labels.
        """
        # get center of mass of each vertebral level
        image_cubic2point = self.cubic_to_point()
        # get list of coordinates for each label
        list_coordinates = image_cubic2point.getNonZeroCoordinates(sorting='value')
        # if user did not specify levels, include all:
        if levels_user[0] == 0:
            levels_user = [int(i.value) for i in list_coordinates]
        # loop across labels and remove those that are not listed by the user
        for i_label in range(len(list_coordinates)):
            # check if this level is NOT in levels_user
            if not levels_user.count(int(list_coordinates[i_label].value)):
                # if not, set value to zero
                image_cubic2point.data[int(list_coordinates[i_label].x), int(list_coordinates[i_label].y), int(list_coordinates[i_label].z)] = 0
        # list all labels
        return image_cubic2point

    def MSE(self, threshold_mse=0):
        """
        Compute the Mean Square Distance Error between two sets of labels (input and ref).
        Moreover, a warning is generated for each label mismatch.
        If the MSE is above the threshold provided (by default = 0mm), a log is reported with the filenames considered here.
        """
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates()

        # check if all the labels in both the images match
        if len(coordinates_input) != len(coordinates_ref):
            sct.printv('ERROR: labels mismatch', 1, 'warning')
        for coord in coordinates_input:
            if round(coord.value) not in [round(coord_ref.value) for coord_ref in coordinates_ref]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')
        for coord_ref in coordinates_ref:
            if round(coord_ref.value) not in [round(coord.value) for coord in coordinates_input]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')

        result = 0.0
        for coord in coordinates_input:
            for coord_ref in coordinates_ref:
                if round(coord_ref.value) == round(coord.value):
                    result += (coord_ref.z - coord.z) ** 2
                    break
        result = math.sqrt(result / len(coordinates_input))
        sct.printv('MSE error in Z direction = ' + str(result) + ' mm')

        if result > threshold_mse:
            f = open(self.image_input.path + 'error_log_' + self.image_input.file_name + '.txt', 'w')
            f.write(
                'The labels error (MSE) between ' + self.image_input.file_name + ' and ' + self.image_ref.file_name + ' is: ' + str(
                    result))
            f.close()

        return result

    @staticmethod
    def remove_label_coord(coord_input, coord_ref, symmetry=False):
        """
        coord_input and coord_ref should be sets of CoordinateValue in order to improve speed of intersection
        :param coord_input: set of CoordinateValue
        :param coord_ref: set of CoordinateValue
        :param symmetry: boolean,
        :return: intersection of CoordinateValue: list
        """
        from msct_types import CoordinateValue
        if isinstance(coord_input[0], CoordinateValue) and isinstance(coord_ref[0], CoordinateValue) and symmetry:
            coord_intersection = list(set(coord_input).intersection(set(coord_ref)))
            result_coord_input = [coord for coord in coord_input if coord in coord_intersection]
            result_coord_ref = [coord for coord in coord_ref if coord in coord_intersection]
        else:
            result_coord_ref = coord_ref
            result_coord_input = [coord for coord in coord_input if filter(lambda x: x.value == coord.value, coord_ref)]
            if symmetry:
                result_coord_ref = [coord for coord in coord_ref if filter(lambda x: x.value == coord.value, result_coord_input)]

        return result_coord_input, result_coord_ref

    def remove_label(self, symmetry=False):
        """
        Compare two label images and remove any labels in input image that are not in reference image.
        The symmetry option enables to remove labels from reference image that are not in input image
        """
        # image_output = Image(self.image_input.dim, orientation=self.image_input.orientation, hdr=self.image_input.hdr, verbose=self.verbose)
        image_output = Image(self.image_input, verbose=self.verbose)
        image_output.data *= 0  # put all voxels to 0

        result_coord_input, result_coord_ref = self.remove_label_coord(self.image_input.getNonZeroCoordinates(coordValue=True),
                                                                       self.image_ref.getNonZeroCoordinates(coordValue=True), symmetry)

        for coord in result_coord_input:
            image_output.data[int(coord.x), int(coord.y), int(coord.z)] = int(round(coord.value))

        if symmetry:
            # image_output_ref = Image(self.image_ref.dim, orientation=self.image_ref.orientation, hdr=self.image_ref.hdr, verbose=self.verbose)
            image_output_ref = Image(self.image_ref, verbose=self.verbose)
            for coord in result_coord_ref:
                image_output_ref.data[int(coord.x), int(coord.y), int(coord.z)] = int(round(coord.value))
            image_output_ref.setFileName(self.fname_output[1])
            image_output_ref.save('minimize_int')

            self.fname_output = self.fname_output[0]

        return image_output

    def extract_centerline(self):
        """
        Write a text file with the coordinates of the centerline.
        The image is suppose to be RPI
        """
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z')

        fo = open(self.fname_output, "wb")
        for coord in coordinates_input:
            line = (coord.x, coord.y, coord.z)
            fo.write("%i %i %i\n" % line)
        fo.close()

    def display_voxel(self):
        """
        Display all the labels that are contained in the input image.
        The image is suppose to be RPI to display voxels. But works also for other orientations
        """
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z')
        useful_notation = ''
        for coord in coordinates_input:
            sct.printv('Position=(' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ') -- Value= ' + str(coord.value))
            if useful_notation:
                useful_notation = useful_notation + ':'
            useful_notation = useful_notation + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
        sct.printv('All labels (useful syntax):')
        sct.printv(useful_notation)
        return coordinates_input

    def diff(self):
        """
        Detect any label mismatch between input image and reference image
        """
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates()

        sct.printv("Label in input image that are not in reference image:")
        for coord in coordinates_input:
            isIn = False
            for coord_ref in coordinates_ref:
                if coord.value == coord_ref.value:
                    isIn = True
                    break
            if not isIn:
                sct.printv(coord.value)

        sct.printv("Label in ref image that are not in input image:")
        for coord_ref in coordinates_ref:
            isIn = False
            for coord in coordinates_input:
                if coord.value == coord_ref.value:
                    isIn = True
                    break
            if not isIn:
                sct.printv(coord_ref.value)

    def distance_interlabels(self, max_dist):
        """
        Calculate the distances between each label in the input image.
        If a distance is larger than max_dist, a warning message is displayed.
        """
        coordinates_input = self.image_input.getNonZeroCoordinates()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, len(coordinates_input) - 1):
            dist = math.sqrt((coordinates_input[i].x - coordinates_input[i + 1].x)**2 + (coordinates_input[i].y - coordinates_input[i + 1].y)**2 + (coordinates_input[i].z - coordinates_input[i + 1].z)**2)
            if dist < max_dist:
                sct.printv('Warning: the distance between label ' + str(i) + '[' + str(coordinates_input[i].x) + ',' + str(coordinates_input[i].y) + ',' + str(
                    coordinates_input[i].z) + ']=' + str(coordinates_input[i].value) + ' and label ' + str(i + 1) + '[' + str(
                    coordinates_input[i + 1].x) + ',' + str(coordinates_input[i + 1].y) + ',' + str(coordinates_input[i + 1].z) + ']=' + str(
                    coordinates_input[i + 1].value) + ' is larger than ' + str(max_dist) + '. Distance=' + str(dist))

    def continuous_vertebral_levels(self):
        """
        This function transforms the vertebral levels file from the template into a continuous file.
        Instead of having integer representing the vertebral level on each slice, a continuous value that represents
        the position of the slice in the vertebral level coordinate system.
        The image must be RPI
        :return:
        """
        im_input = Image(self.image_input, self.verbose)
        im_output = Image(self.image_input, self.verbose)
        im_output.data *= 0

        # 1. extract vertebral levels from input image
        #   a. extract centerline
        #   b. for each slice, extract corresponding level
        nx, ny, nz, nt, px, py, pz, pt = im_input.dim
        from sct_straighten_spinalcord import smooth_centerline
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(self.image_input, algo_fitting='nurbs', verbose=0)
        value_centerline = np.array([im_input.data[int(x_centerline_fit[it]), int(y_centerline_fit[it]), int(z_centerline_fit[it])] for it in range(len(z_centerline_fit))])

        # 2. compute distance for each vertebral level --> Di for i being the vertebral levels
        vertebral_levels = {}
        for slice_image, level in enumerate(value_centerline):
            if level not in vertebral_levels:
                vertebral_levels[level] = slice_image

        length_levels = {}
        for level in vertebral_levels:
            indexes_slice = np.where(value_centerline == level)
            length_levels[level] = np.sum([math.sqrt(((x_centerline_fit[indexes_slice[0][index_slice + 1]] - x_centerline_fit[indexes_slice[0][index_slice]]) * px)**2 +
                                                     ((y_centerline_fit[indexes_slice[0][index_slice + 1]] - y_centerline_fit[indexes_slice[0][index_slice]]) * py)**2 +
                                                     ((z_centerline_fit[indexes_slice[0][index_slice + 1]] - z_centerline_fit[indexes_slice[0][index_slice]]) * pz)**2)
                                           for index_slice in range(len(indexes_slice[0]) - 1)])

        # 2. for each slice:
        #   a. identify corresponding vertebral level --> i
        #   b. calculate distance of slice from upper vertebral level --> d
        #   c. compute relative distance in the vertebral level coordinate system --> d/Di
        continuous_values = {}
        for it, iz in enumerate(z_centerline_fit):
            level = value_centerline[it]
            indexes_slice = np.where(value_centerline == level)
            indexes_slice = indexes_slice[0][indexes_slice[0] >= it]
            distance_from_level = np.sum([math.sqrt(((x_centerline_fit[indexes_slice[index_slice + 1]] - x_centerline_fit[indexes_slice[index_slice]]) * px * px) ** 2 +
                                                    ((y_centerline_fit[indexes_slice[index_slice + 1]] - y_centerline_fit[indexes_slice[index_slice]]) * py * py) ** 2 +
                                                    ((z_centerline_fit[indexes_slice[index_slice + 1]] - z_centerline_fit[indexes_slice[index_slice]]) * pz * pz) ** 2)
                                          for index_slice in range(len(indexes_slice) - 1)])
            continuous_values[iz] = level + 2.0 * distance_from_level / float(length_levels[level])

        # 3. saving data
        # for each slice, get all non-zero pixels and replace with continuous values
        coordinates_input = self.image_input.getNonZeroCoordinates()
        im_output.changeType('float32')
        # for all points in input, find the value that has to be set up, depending on the vertebral level
        for i, coord in enumerate(coordinates_input):
            im_output.data[int(coord.x), int(coord.y), int(coord.z)] = continuous_values[coord.z]

        return im_output

    def launch_sagittal_viewer(self, labels):
        from spinalcordtoolbox.gui import base
        from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

        params = base.AnatomicalParams()
        params.vertebraes = labels
        params.input_file_name = self.image_input.file_name
        params.output_file_name = self.fname_output
        output = self.image_input.copy()
        output.data *= 0
        output.setFileName(self.fname_output)
        launch_sagittal_dialog(self.image_input, output, params)

        return output


def get_parser():
    # initialize default param
    param_default = Param()
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for label image.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input image.",
                      mandatory=True,
                      example="t2_labels.nii.gz")
    parser.add_option(name="-o",
                      type_value=[[','], "file_output"],
                      description="Output image(s).",
                      mandatory=False,
                      example="t2_labels_cross.nii.gz",
                      default_value="labels.nii.gz")
    parser.add_option(name='-add',
                      type_value='int',
                      description='Add value to all labels. Value can be negative.',
                      mandatory=False)
    parser.add_option(name='-create',
                      type_value=[[':'], 'Coordinate'],
                      description='Create labels in a new image. List labels as: x1,y1,z1,value1:x2,y2,z2,value2, ...',
                      example='12,34,32,1:12,35,33,2',
                      mandatory=False)
    parser.add_option(name='-create-add',
                      type_value=[[':'], 'Coordinate'],
                      description='Same as "-create", but add labels to the input image instead of creating a new image.',
                      example='12,34,32,1:12,35,33,2',
                      mandatory=False)
    parser.add_option(name='-create-seg',
                      type_value=[[':'], 'str'],
                      description='Create labels along cord segmentation (or centerline) defined by "-i". First value is "z", second is the value of the label. Separate labels with ":". Example: 5,1:14,2:23,3. To select the mid-point in the superior-inferior direction, set z to "-1". For example if you know that C2-C3 disc is centered in the S-I direction, then enter: -1,3',
                      mandatory=False)
    parser.add_option(name='-cross',
                      type_value='int',
                      description='Create a cross around each non-zero value. Input cross radius in mm.',
                      example=param_default.cross_size,
                      mandatory=False)
    parser.add_option(name='-cubic-to-point',
                      type_value=None,
                      description='Compute the center-of-mass for each label value.',
                      mandatory=False)
    parser.add_option(name='-display',
                      type_value=None,
                      description='Display all labels (i.e. non-zero values).',
                      mandatory=False)
    parser.add_option(name='-increment',
                      type_value=None,
                      description='Takes all non-zero values, sort them along the inverse z direction, and attributes the values 1, 2, 3, etc.',
                      mandatory=False)
    parser.add_option(name='-vert-body',
                      type_value=[[','], 'int'],
                      description='From vertebral labeling, create points that are centered at the mid-vertebral levels. Separate desired levels with ",". To get all levels, enter "0".',
                      example='3,8',
                      mandatory=False)
    parser.add_option(name='-vert-continuous',
                      type_value=None,
                      description='Convert discrete vertebral labeling to continuous vertebral labeling.',
                      mandatory=False)
    parser.add_option(name='-MSE',
                      type_value='file',
                      description='Compute Mean Square Error between labels from input and reference image. Specify reference image here.',
                      mandatory=False)
    parser.add_option(name='-remove',
                      type_value='file',
                      description='Remove labels from input image (-i) that are not in reference image (specified here).',
                      mandatory=False)
    parser.add_option(name='-remove-sym',
                      type_value='file',
                      description='Remove labels from input image (-i) and reference image (specified here) that don\'t match. You must provide two output names separated by ",".',
                      mandatory=False)
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description='Verbose. 0: nothing. 1: basic. 2: extended.',
                      mandatory=False,
                      default_value=param_default.verbose,
                      example=['0', '1', '2'])
    parser.add_option(name='-create-viewer',
                      type_value=[[','], 'int'],
                      description='Manually label from a GUI a list of labels IDs, separated with ",". Example: 2,3,4,5',
                      mandatory=False)
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    input_filename = arguments['-i']
    input_fname_output = None
    input_fname_ref = None
    input_cross_radius = 5
    input_dilate = False
    input_coordinates = None
    vertebral_levels = None
    value = None
    if '-add' in arguments:
        process_type = 'add'
        value = arguments['-add']
    elif '-create' in arguments:
        process_type = 'create'
        input_coordinates = arguments['-create']
    elif '-create-add' in arguments:
        process_type = 'create-add'
        input_coordinates = arguments['-create-add']
    elif '-create-seg' in arguments:
        process_type = 'create-seg'
        input_coordinates = arguments['-create-seg']
    elif '-cross' in arguments:
        process_type = 'cross'
        input_cross_radius = arguments['-cross']
    elif '-cubic-to-point' in arguments:
        process_type = 'cubic-to-point'
    elif '-display' in arguments:
        process_type = 'display-voxel'
    elif '-increment' in arguments:
        process_type = 'increment'
    elif '-vert-body' in arguments:
        process_type = 'vert-body'
        vertebral_levels = arguments['-vert-body']
    # elif '-vert-disc' in arguments:
    #     process_type = 'vert-disc'
    #     vertebral_levels = arguments['-vert-disc']
    elif '-vert-continuous' in arguments:
        process_type = 'vert-continuous'
    elif '-MSE' in arguments:
        process_type = 'MSE'
        input_fname_ref = arguments['-r']
    elif '-remove' in arguments:
        process_type = 'remove'
        input_fname_ref = arguments['-remove']
    elif '-remove-symm' in arguments:
        process_type = 'remove-symm'
        input_fname_ref = arguments['-r']
    elif '-create-viewer' in arguments:
        process_type = 'create-viewer'
        value = arguments['-create-viewer']
    else:
        # no process chosen
        sct.printv('ERROR: No process was chosen.', 1, 'error')
    if '-o' in arguments:
        input_fname_output = arguments['-o']
    input_verbose = int(arguments['-v'])

    processor = ProcessLabels(input_filename, fname_output=input_fname_output, fname_ref=input_fname_ref,
                              cross_radius=input_cross_radius, dilate=input_dilate, coordinates=input_coordinates,
                              verbose=input_verbose, vertebral_levels=vertebral_levels, value=value)
    processor.process(process_type)


if __name__ == "__main__":
    sct.start_stream_logger()
    # call main function
    main()
