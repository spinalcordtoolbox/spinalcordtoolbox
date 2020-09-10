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

from __future__ import division, absolute_import

import os
import sys
import argparse

import numpy as np
from scipy import ndimage

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.types import Coordinate, CoordinateValue
from spinalcordtoolbox.reports.qc import generate_qc

# TODO: Properly test when first PR (that includes list_type) gets merged
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder, list_type
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
                 coordinates=None, verbose=1, vertebral_levels=None, value=None, msg="", fname_previous=None):
        """
        Collection of processes that deal with label creation/modification.
        :param fname_label:
        :param fname_output:
        :param fname_ref:
        :param cross_radius:
        :param dilate:  # TODO: remove dilate (does not seem to be used)
        :param coordinates:
        :param verbose:
        :param vertebral_levels:
        :param value:
        :param msg: string. message to display to the user.
        """
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
        self.fname_previous = fname_previous
        self.cross_radius = cross_radius
        self.vertebral_levels = vertebral_levels
        self.dilate = dilate
        self.coordinates = coordinates
        self.verbose = verbose
        self.value = value
        self.msg = msg
        self.output_image = None
        self.previous_image = None

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
        if type_process == 'remove-reference':
            self.output_image = self.remove_label()
        if type_process == 'remove-symm':
            self.output_image = self.remove_label(symmetry=True)
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
            if self.fname_previous is not None:
                previous_lab = Image(self.fname_previous)
                # the input image is reoriented to 'SAL' when open by the GUI
                previous_lab.change_orientation('SAL')
                mid = int(np.round(previous_lab.data.shape[2]/2))
                previous_points = previous_lab.getNonZeroCoordinates()
                # boolean used to mark first element to initiate the list.
                first = True
                for i in range(len(previous_points)):
                    if int(previous_points[i].value) in self.value:
                        pass
                    else:
                        self.value.append(int(previous_points[i].value))
                    if first:
                        points = np.array([previous_points[i]. x, previous_points[i].y, previous_points[i].z, previous_points[i].value])
                        points = np.reshape(points, (1, 4))
                        previous_label = points
                        first = False
                    else:
                        points = np.array([previous_points[i].x, previous_points[i].y, previous_points[i].z, previous_points[i].value])
                        points = np.reshape(points, (1, 4))
                        previous_label = np.append(previous_label, points, axis=0)
                    self.value.sort()

                # check if variable was created which means the file was not empty and contains some points asked in self.value
                if 'previous_label' in locals():
                    # project onto mid sagittal plane
                    for i in range(len(previous_label)):
                        previous_label[i][2] = mid
                    self.output_image = self.launch_sagittal_viewer(self.value, previous_points=previous_label)
                else:
                    self.output_image = self.launch_sagittal_viewer(self.value)
            else:
                self.output_image = self.launch_sagittal_viewer(self.value)

        if type_process in ['remove', 'keep']:
            self.output_image = self.remove_or_keep_labels(self.value, action=type_process)

        # TODO: do not save here. Create another function save() for that
        if self.fname_output is not None:
            if change_orientation:
                self.output_image.change_orientation(input_orientation)
            self.output_image.absolutepath = self.fname_output
            if type_process == 'vert-continuous':
                self.output_image.save(dtype='float32')
            elif type_process != 'plan_ref':
                self.output_image.save(dtype='minimize_int')
            else:
                self.output_image.save()
        return self.output_image

    def add(self, value):
        """
        This function add a specified value to all non-zero voxels.
        """
        image_output = self.image_input.copy()
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

        self.coordinates is a list of coordinates (class in spinalcordtoolbox.types).
        a Coordinate contains x, y, z and value.
        If only one label is to be added, coordinates must be completed with '[]'
        examples:
        For one label:  object_define=ProcessLabels( fname_label, coordinates=[coordi]) where coordi is a 'Coordinate'
          object from spinalcordtoolbox.types
        For two labels: object_define=ProcessLabels( fname_label, coordinates=[coordi1, coordi2]) where coordi1 and
          coordi2 are 'Coordinate' objects from spinalcordtoolbox.types
        """
        image_output = self.image_input.copy() if add else msct_image.zeros_like(self.image_input)

        # loop across labels
        for i, coord in enumerate(self.coordinates):
            if len(image_output.data.shape) == 3:
                image_output.data[int(coord.x), int(coord.y), int(coord.z)] = coord.value
            elif len(image_output.data.shape) == 2:
                assert str(coord.z) == '0', "ERROR: 2D coordinates should have a Z value of 0. Z coordinate is :" + str(coord.z)
                image_output.data[int(coord.x), int(coord.y)] = coord.value
            else:
                sct.printv('ERROR: Data should be 2D or 3D. Current shape is: ' + str(image_output.data.shape), 1, 'error')
            # display info
            sct.printv('Label #' + str(i) + ': ' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ' --> ' +
                       str(coord.value), 1)
        return image_output

    def create_label_along_segmentation(self):
        """
        Create an image with labels defined along the spinal cord segmentation (or centerline).
        Input image does **not** need to be RPI (re-orientation is done within this function).
        Example:
        object_define=ProcessLabels(fname_segmentation, coordinates=[coord_1, coord_2, coord_i]), where coord_i='z,value'. If z=-1, then use z=nz/2 (i.e. center of FOV in superior-inferior direction)
        Returns
        """
        # reorient input image to RPI
        im_rpi = self.image_input.copy().change_orientation('RPI')
        im_output_rpi = zeros_like(im_rpi)
        # loop across labels
        for ilabel, coord in enumerate(self.coordinates):
            # split coord string
            list_coord = coord.split(',')
            # convert to int() and assign to variable
            z, value = [int(i) for i in list_coord]
            # update z based on native image orientation (z should represent superior-inferior axis)
            coord = Coordinate([z, z, z])  # since we don't know which dimension corresponds to the superior-inferior
            # axis, we put z in all dimensions (we don't care about x and y here)
            _, _, z_rpi = coord.permute(self.image_input, 'RPI')
            # if z=-1, replace with nz/2
            if z == -1:
                z_rpi = int(np.round(im_output_rpi.dim[2] / 2.0))
            # get center of mass of segmentation at given z
            x, y = ndimage.measurements.center_of_mass(np.array(im_rpi.data[:, :, z_rpi]))
            # round values to make indices
            x, y = int(np.round(x)), int(np.round(y))
            # display info
            sct.printv('Label #' + str(ilabel) + ': ' + str(x) + ',' + str(y) + ',' + str(z_rpi) + ' --> ' + str(value), 1)
            if len(im_output_rpi.data.shape) == 3:
                im_output_rpi.data[x, y, z_rpi] = value
            elif len(im_output_rpi.data.shape) == 2:
                assert str(z) == '0', "ERROR: 2D coordinates should have a Z value of 0. Z coordinate is :" + str(z)
                im_output_rpi.data[x, y] = value
        # change orientation back to native
        return im_output_rpi.change_orientation(self.image_input.orientation)

    def plan(self, width, offset=0, gap=1):
        """
        Create a plane of thickness="width" and changes its value with an offset and a gap between labels.
        """
        image_output = msct_image.zeros_like(self.image_input)

        coordinates_input = self.image_input.getNonZeroCoordinates()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for coord in coordinates_input:
            image_output.data[:, :, int(coord.z) - width:int(coord.z) + width] = offset + gap * coord.value

        return image_output

    def plan_ref(self):
        """
        Generate a plane in the reference space for each label present in the input image
        """

        image_output = msct_image.zeros_like(Image(self.image_ref))

        image_input_neg = msct_image.zeros_like(Image(self.image_input))
        image_input_pos = msct_image.zeros_like(Image(self.image_input))

        X, Y, Z = (self.image_input.data < 0).nonzero()
        for i in range(len(X)):
            image_input_neg.data[X[i], Y[i], Z[i]] = -self.image_input.data[X[i], Y[i], Z[i]]  # in order to apply getNonZeroCoordinates
        X_pos, Y_pos, Z_pos = (self.image_input.data > 0).nonzero()
        for i in range(len(X_pos)):
            image_input_pos.data[X_pos[i], Y_pos[i], Z_pos[i]] = self.image_input.data[X_pos[i], Y_pos[i], Z_pos[i]]

        coordinates_input_neg = image_input_neg.getNonZeroCoordinates()
        coordinates_input_pos = image_input_pos.getNonZeroCoordinates()

        image_output.change_type('float32')
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
        output_image = msct_image.zeros_like(self.image_input)

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
        for value, list_coord in groups.items():
            center_of_mass = sum(list_coord) / float(len(list_coord))
            sct.printv("Value = " + str(center_of_mass.value) + " : (" + str(center_of_mass.x) + ", " + str(center_of_mass.y) + ", " + str(center_of_mass.z) + ") --> ( " + str(np.round(center_of_mass.x)) + ", " + str(np.round(center_of_mass.y)) + ", " + str(np.round(center_of_mass.z)) + ")", verbose=self.verbose)
            output_image.data[int(np.round(center_of_mass.x)), int(np.round(center_of_mass.y)), int(np.round(center_of_mass.z))] = center_of_mass.value

        return output_image

    def increment_z_inverse(self):
        """
        Take all non-zero values, sort them along the inverse z direction, and attributes the values 1,
        2, 3, etc. This function assuming RPI orientation.
        """
        image_output = msct_image.zeros_like(self.image_input)

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
        image_output = msct_image.zeros_like(self.image_input)

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
            if np.round(coord.value) not in [np.round(coord_ref.value) for coord_ref in coordinates_ref]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')
        for coord_ref in coordinates_ref:
            if np.round(coord_ref.value) not in [np.round(coord.value) for coord in coordinates_input]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')

        result = 0.0
        for coord in coordinates_input:
            for coord_ref in coordinates_ref:
                if np.round(coord_ref.value) == np.round(coord.value):
                    result += (coord_ref.z - coord.z) ** 2
                    break
        result = np.sqrt(result / len(coordinates_input))
        sct.printv('MSE error in Z direction = ' + str(result) + ' mm')

        if result > threshold_mse:
            parent, stem, ext = sct.extract_fname(self.image_input.absolutepath)
            fname_report = os.path.join(parent, 'error_log_{}.txt'.format(stem))
            with open(fname_report, 'w') as f:
                f.write('The labels error (MSE) between {} and {} is: {}\n'.format(
                 os.path.relpath(self.image_input.absolutepath, os.path.dirname(fname_report)),
                 os.path.relpath(self.image_ref.absolutepath, os.path.dirname(fname_report)),
                 result))

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
        from spinalcordtoolbox.types import CoordinateValue
        if isinstance(coord_input[0], CoordinateValue) and isinstance(coord_ref[0], CoordinateValue) and symmetry:
            coord_intersection = list(set(coord_input).intersection(set(coord_ref)))
            result_coord_input = [coord for coord in coord_input if coord in coord_intersection]
            result_coord_ref = [coord for coord in coord_ref if coord in coord_intersection]
        else:
            result_coord_ref = coord_ref
            result_coord_input = [coord for coord in coord_input if list(filter(lambda x: x.value == coord.value, coord_ref))]
            if symmetry:
                result_coord_ref = [coord for coord in coord_ref if list(filter(lambda x: x.value == coord.value, result_coord_input))]

        return result_coord_input, result_coord_ref

    def remove_label(self, symmetry=False):
        """
        Compare two label images and remove any labels in input image that are not in reference image.
        The symmetry option enables to remove labels from reference image that are not in input image
        """
        # image_output = Image(self.image_input.dim, orientation=self.image_input.orientation, hdr=self.image_input.hdr, verbose=self.verbose)
        image_output = msct_image.zeros_like(self.image_input)

        result_coord_input, result_coord_ref = self.remove_label_coord(self.image_input.getNonZeroCoordinates(coordValue=True),
                                                                       self.image_ref.getNonZeroCoordinates(coordValue=True), symmetry)

        for coord in result_coord_input:
            image_output.data[int(coord.x), int(coord.y), int(coord.z)] = int(np.round(coord.value))

        if symmetry:
            # image_output_ref = Image(self.image_ref.dim, orientation=self.image_ref.orientation, hdr=self.image_ref.hdr, verbose=self.verbose)
            image_output_ref = Image(self.image_ref, verbose=self.verbose)
            for coord in result_coord_ref:
                image_output_ref.data[int(coord.x), int(coord.y), int(coord.z)] = int(np.round(coord.value))
            image_output_ref.absolutepath = self.fname_output[1]
            image_output_ref.save('minimize_int')

            self.fname_output = self.fname_output[0]

        return image_output

    def display_voxel(self):
        """
        Display all the labels that are contained in the input image.
        The image is suppose to be RPI to display voxels. But works also for other orientations
        """
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='value')
        self.useful_notation = ''
        for coord in coordinates_input:
            sct.printv('Position=(' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ') -- Value= ' + str(coord.value), verbose=self.verbose)
            if self.useful_notation:
                self.useful_notation = self.useful_notation + ':'
            self.useful_notation += str(coord)
        sct.printv('All labels (useful syntax):', verbose=self.verbose)
        sct.printv(self.useful_notation, verbose=self.verbose)
        return coordinates_input

    def get_physical_coordinates(self):
        """
        This function returns the coordinates of the labels in the physical referential system.
        :return: a list of CoordinateValue, in the physical (scanner) space
        """
        coord = self.image_input.getNonZeroCoordinates(sorting='value')
        phys_coord = []
        for c in coord:
            # convert pixelar coordinates to physical coordinates
            c_p = self.image_input.transfo_pix2phys([[c.x, c.y, c.z]])[0]
            phys_coord.append(CoordinateValue([c_p[0], c_p[1], c_p[2], c.value]))
        return phys_coord

    def get_coordinates_in_destination(self, im_dest, type='discrete'):
        """
        This function calculate the position of labels in the pixelar space of a destination image
        :param im_dest: Object Image
        :param type: 'discrete' or 'continuous'
        :return: a list of CoordinateValue, in the pixelar (image) space of the destination image
        """
        phys_coord = self.get_physical_coordinates()
        dest_coord = []
        for c in phys_coord:
            if type is 'discrete':
                c_p = im_dest.transfo_phys2pix([[c.x, c.y, c.y]])[0]
            elif type is 'continuous':
                c_p = im_dest.transfo_phys2pix([[c.x, c.y, c.y]], real=False)[0]
            else:
                raise ValueError("The value of 'type' should either be 'discrete' or 'continuous'.")
            dest_coord.append(CoordinateValue([c_p[0], c_p[1], c_p[2], c.value]))
        return dest_coord

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
            dist = np.sqrt((coordinates_input[i].x - coordinates_input[i + 1].x)**2 + (coordinates_input[i].y - coordinates_input[i + 1].y)**2 + (coordinates_input[i].z - coordinates_input[i + 1].z)**2)
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
        im_output = msct_image.zeros_like(self.image_input)

        # 1. extract vertebral levels from input image
        #   a. extract centerline
        #   b. for each slice, extract corresponding level
        nx, ny, nz, nt, px, py, pz, pt = im_input.dim
        from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
        _, arr_ctl, _, _ = get_centerline(self.image_input, param=ParamCenterline())
        x_centerline_fit, y_centerline_fit, z_centerline = arr_ctl
        value_centerline = np.array(
            [im_input.data[int(x_centerline_fit[it]), int(y_centerline_fit[it]), int(z_centerline[it])]
             for it in range(len(z_centerline))])

        # 2. compute distance for each vertebral level --> Di for i being the vertebral levels
        vertebral_levels = {}
        for slice_image, level in enumerate(value_centerline):
            if level not in vertebral_levels:
                vertebral_levels[level] = slice_image

        length_levels = {}
        for level in vertebral_levels:
            indexes_slice = np.where(value_centerline == level)
            length_levels[level] = np.sum([np.sqrt(((x_centerline_fit[indexes_slice[0][index_slice + 1]] - x_centerline_fit[indexes_slice[0][index_slice]]) * px)**2 +
                                                     ((y_centerline_fit[indexes_slice[0][index_slice + 1]] - y_centerline_fit[indexes_slice[0][index_slice]]) * py)**2 +
                                                     ((z_centerline[indexes_slice[0][index_slice + 1]] - z_centerline[indexes_slice[0][index_slice]]) * pz)**2)
                                           for index_slice in range(len(indexes_slice[0]) - 1)])

        # 2. for each slice:
        #   a. identify corresponding vertebral level --> i
        #   b. calculate distance of slice from upper vertebral level --> d
        #   c. compute relative distance in the vertebral level coordinate system --> d/Di
        continuous_values = {}
        for it, iz in enumerate(z_centerline):
            level = value_centerline[it]
            indexes_slice = np.where(value_centerline == level)
            indexes_slice = indexes_slice[0][indexes_slice[0] >= it]
            distance_from_level = np.sum([np.sqrt(((x_centerline_fit[indexes_slice[index_slice + 1]] - x_centerline_fit[indexes_slice[index_slice]]) * px * px) ** 2 +
                                                    ((y_centerline_fit[indexes_slice[index_slice + 1]] - y_centerline_fit[indexes_slice[index_slice]]) * py * py) ** 2 +
                                                    ((z_centerline[indexes_slice[index_slice + 1]] - z_centerline[indexes_slice[index_slice]]) * pz * pz) ** 2)
                                          for index_slice in range(len(indexes_slice) - 1)])
            continuous_values[iz] = level + 2.0 * distance_from_level / float(length_levels[level])

        # 3. saving data
        # for each slice, get all non-zero pixels and replace with continuous values
        coordinates_input = self.image_input.getNonZeroCoordinates()
        im_output.change_type(np.float32)
        # for all points in input, find the value that has to be set up, depending on the vertebral level
        for i, coord in enumerate(coordinates_input):
            im_output.data[int(coord.x), int(coord.y), int(coord.z)] = continuous_values[coord.z]

        return im_output

    def launch_sagittal_viewer(self, labels, previous_points=None):
        from spinalcordtoolbox.gui import base
        from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

        params = base.AnatomicalParams()
        params.vertebraes = labels
        params.input_file_name = self.image_input.absolutepath
        params.output_file_name = self.fname_output
        params.subtitle = self.msg
        if previous_points is not None: 
            params.message_warn = 'Please select the label you want to add \nor correct in the list below before clicking \non the image'
        output = msct_image.zeros_like(self.image_input)
        output.absolutepath = self.fname_output
        launch_sagittal_dialog(self.image_input, output, params, previous_points)

        return output

    def remove_or_keep_labels(self, labels, action):
        """
        Create or remove labels from self.image_input
        :param list(int): Labels to keep or remove
        :param str: 'remove': remove specified labels (i.e. set to zero), 'keep': keep specified labels and remove the others
        """
        if action == 'keep':
            image_output = msct_image.zeros_like(self.image_input)
        elif action == 'remove':
            image_output = self.image_input.copy()
        coordinates_input = self.image_input.getNonZeroCoordinates()

        for labelNumber in labels:
            isInLabels = False
            for coord in coordinates_input:
                if labelNumber == coord.value:
                    new_coord = coord
                    isInLabels = True
            if isInLabels:
                if action == 'keep':
                    image_output.data[int(new_coord.x), int(new_coord.y), int(new_coord.z)] = new_coord.value
                elif action == 'remove':
                    image_output.data[int(new_coord.x), int(new_coord.y), int(new_coord.z)] = 0.0
            else:
                sct.printv("WARNING: Label " + str(float(labelNumber)) + " not found in input image.", type='warning')

        return image_output

def get_parser():
    # initialize default param
    param_default = Param()
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Utility function for label images.",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input image. Example: t2_labels.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-add',
        metavar=Metavar.int,
        type=int,
        help="Add value to all labels. Value can be negative."
    )
    optional.add_argument(
        '-create',
        metavar=Metavar.list,
        type=list_type(':', Coordinate),
        help="Create labels in a new image. List labels as: x1,y1,z1,value1:x2,y2,z2,value2. "
             "Example: 12,34,32,1:12,35,33,2"
    )
    optional.add_argument(
        '-create-add',
        metavar=Metavar.list,
        type=list_type(':', Coordinate),
        help="Same as '-create', but add labels to the input image instead of creating a new image. "
             "Example: 12,34,32,1:12,35,33,2"
    )
    optional.add_argument(
        '-create-seg',
        metavar=Metavar.list,
        type=list_type(':', str),
        help="R|Create labels along cord segmentation (or centerline) defined by '-i'. First value is 'z', second is "
             "the value of the label. Separate labels with ':'. Example: 5,1:14,2:23,3. \n"
             "To select the mid-point in the superior-inferior direction, set z to '-1'. For example if you know that "
             "C2-C3 disc is centered in the S-I direction, then enter: -1,3"
    )
    optional.add_argument(
        '-create-viewer',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Manually label from a GUI a list of labels IDs, separated with ','. Example: 2,3,4,5"
    )
    optional.add_argument(
        '-ilabel',
        metavar=Metavar.file,
        help="File that contain labels that you want to correct. It is possible to add new points with this option. "
             "Use with -create-viewer. Example: t2_labels_auto.nii.gz"
    )
    optional.add_argument(
        '-cubic-to-point',
        action="store_true",
        help="Compute the center-of-mass for each label value."
    )
    optional.add_argument(
        '-display',
        action="store_true",
        help="Display all labels (i.e. non-zero values)."
    )
    optional.add_argument(
        '-increment',
        action="store_true",
        help="Takes all non-zero values, sort them along the inverse z direction, and attributes the values "
             "1, 2, 3, etc."
    )
    optional.add_argument(
        '-vert-body',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="R|From vertebral labeling, create points that are centered at the mid-vertebral levels. Separate "
             "desired levels with ','. Example: 3,8\n"
             "To get all levels, enter 0."
    )
    optional.add_argument(
        '-vert-continuous',
        action="store_true",
        help="Convert discrete vertebral labeling to continuous vertebral labeling.",
    )
    optional.add_argument(
        '-MSE',
        metavar=Metavar.file,
        help="Compute Mean Square Error between labels from input and reference image. Specify reference image here."
    )
    optional.add_argument(
        '-remove-reference',
        metavar=Metavar.file,
        help="Remove labels from input image (-i) that are not in reference image (specified here)."
    )
    optional.add_argument(
        '-remove-sym',
        metavar=Metavar.file,
        help="Remove labels from input image (-i) and reference image (specified here) that don't match. You must "
             "provide two output names separated by ','."
    )
    optional.add_argument(
        '-remove',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Remove labels of specific value (specified here) from reference image."
    )
    optional.add_argument(
        '-keep',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Keep labels of specific value (specified here) from reference image."
    )
    optional.add_argument(
        '-msg',
        metavar=Metavar.str,
        help="Display a message to explain the labeling task. Use with -create-viewer"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.list,
        type=list_type(',', str),
        default="labels.nii.gz",
        help="Output image(s). t2_labels_cross.nii.gz"
    )
    optional.add_argument(
        '-v',
        choices=['0', '1', '2'],
        default=param_default.verbose,
        help="Verbose. 0: nothing. 1: basic. 2: extended."
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )

    return parser


# MAIN
# ==========================================================================================
def main(args=None):
    parser = get_parser()
    if args:
        arguments = parser.parse_args(args)
    else:
        arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    input_filename = arguments.i
    input_fname_output = None
    input_fname_ref = None
    input_cross_radius = 5
    input_dilate = False
    input_coordinates = None
    vertebral_levels = None
    value = None
    if arguments.add is not None:
        process_type = 'add'
        value = arguments.add
    elif arguments.create is not None:
        process_type = 'create'
        input_coordinates = arguments.create
    elif arguments.create_add is not None:
        process_type = 'create-add'
        input_coordinates = arguments.create_add
    elif arguments.create_seg is not None:
        process_type = 'create-seg'
        input_coordinates = arguments.create_seg
    # TODO: This argument was not present in argparse. Should it be included?
    # elif arguments.cross is not None:
    #     process_type = 'cross'
    #     input_cross_radius = arguments.cross
    elif arguments.cubic_to_point:
        process_type = 'cubic-to-point'
    elif arguments.display:
        process_type = 'display-voxel'
    elif arguments.increment:
        process_type = 'increment'
    elif arguments.vert_body is not None:
        process_type = 'vert-body'
        vertebral_levels = arguments.vert_body
    # elif '-vert-disc' in arguments:
    #     process_type = 'vert-disc'
    #     vertebral_levels = arguments['-vert-disc']
    elif arguments.vert_continuous:
        process_type = 'vert-continuous'
    elif arguments.MSE is not None:
        process_type = 'MSE'
        input_fname_ref = arguments.r
    elif arguments.remove_reference is not None:
        process_type = 'remove-reference'
        input_fname_ref = arguments.remove_reference
    elif arguments.remove_symm is not None:
        process_type = 'remove-symm'
        input_fname_ref = arguments.r
    elif arguments.create_viewer is not None:
        process_type = 'create-viewer'
        value = arguments.create_viewer
    elif arguments.remove is not None:
        process_type = 'remove'
        value = arguments.remove
    elif arguments.keep is not None:
        process_type = 'keep'
        value = arguments.keep
    else:
        # no process chosen
        sct.printv('ERROR: No process was chosen.', 1, 'error')
    if arguments.msg is not None:
        msg = arguments.msg+"\n"
    else:
        msg = ""
    if arguments.o is not None:
        input_fname_output = arguments.o
    if arguments.ilabel is not None:
        input_fname_previous = arguments.ilabel
    else:
        input_fname_previous = None
    verbose = int(arguments.v)
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    processor = ProcessLabels(input_filename, fname_output=input_fname_output, fname_ref=input_fname_ref,
                              cross_radius=input_cross_radius, dilate=input_dilate, coordinates=input_coordinates,
                              verbose=verbose, vertebral_levels=vertebral_levels, value=value, msg=msg, fname_previous=input_fname_previous)
    processor.process(process_type)

    # return based on process type
    if process_type == 'display-voxel':
        return processor.useful_notation

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    if path_qc is not None:
        generate_qc(fname_in1=input_filename, fname_seg=input_fname_output[0], args=args,
                    path_qc=os.path.abspath(path_qc), dataset=qc_dataset, subject=qc_subject, process='sct_label_utils')


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
