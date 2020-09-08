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

import logging
from typing import *

import numpy as np
from scipy import ndimage

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.types import Coordinate, CoordinateValue

logger = logging.getLogger(__name__)

# TODO: for vert-disc: make it faster! currently the module display-voxel is very long (esp. when ran on PAM50). We can find an alternative approach by sweeping through centerline voxels.
# TODO: label_disc: for top vertebrae, make label at the center of the cord (currently it's at the tip)
# TODO: check if use specified several processes.
# TODO: currently it seems like cross_radius is given in pixel instead of mm


def add(img: Image, value: float) -> Image:
    """
    This function adds a specified value to all non-zero voxels.
    :param img: source image
    :param value: numeric value to add
    :returns new image with value added
    """
    out = img.copy()
    coordinates_input = img.getNonZeroCoordinates()

    # for all points with non-zeros neighbors, force the neighbors to 0
    for coord in coordinates_input:
        out.data[int(coord.x), int(coord.y), int(coord.z)] = out.data[int(coord.x), int(coord.y), int(coord.z)] + float(value)

    return out


def create_label(add=False):
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
    image_output = self.image_input.copy() if add else zeros_like(self.image_input)

    # loop across labels
    for i, coord in enumerate(self.coordinates):
        if len(image_output.data.shape) == 3:
            image_output.data[int(coord.x), int(coord.y), int(coord.z)] = coord.value
        elif len(image_output.data.shape) == 2:
            assert str(coord.z) == '0', "ERROR: 2D coordinates should have a Z value of 0. Z coordinate is :" + str(coord.z)
            image_output.data[int(coord.x), int(coord.y)] = coord.value
        else:
            raise ValueError(f"Data should be 2D or 3D. Current shape is: {str(image_output.data.shape)}")
        # display info
        logger.info(f"Label #{str(i)}: {str(coord.x)}, {str(coord.y)}, {str(coord.z)} --> {str(coord.value)}")

    return image_output


def create_label_along_segmentation():
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
        logger.info(f"Label # {str(ilabel)}: {str(x)}, {str(y)}. {str(z_rpi)} --> {str(value)}")

        if len(im_output_rpi.data.shape) == 3:
            im_output_rpi.data[x, y, z_rpi] = value
        elif len(im_output_rpi.data.shape) == 2:
            assert str(z) == '0', "ERROR: 2D coordinates should have a Z value of 0. Z coordinate is :" + str(z)
            im_output_rpi.data[x, y] = value

    # change orientation back to native
    return im_output_rpi.change_orientation(self.image_input.orientation)


def plan(width, offset=0, gap=1):
    """
    Create a plane of thickness="width" and changes its value with an offset and a gap between labels.
    """
    image_output = zeros_like(self.image_input)

    coordinates_input = self.image_input.getNonZeroCoordinates()

    # for all points with non-zeros neighbors, force the neighbors to 0
    for coord in coordinates_input:
        image_output.data[:, :, int(coord.z) - width:int(coord.z) + width] = offset + gap * coord.value

    return image_output


def plan_ref():
    """
    Generate a plane in the reference space for each label present in the input image
    """

    image_output = zeros_like(Image(self.image_ref))

    image_input_neg = zeros_like(Image(self.image_input))
    image_input_pos = zeros_like(Image(self.image_input))

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


def cubic_to_point():
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
    output_image = zeros_like(self.image_input)

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


def increment_z_inverse():
    """
    Take all non-zero values, sort them along the inverse z direction, and attributes the values 1,
    2, 3, etc. This function assuming RPI orientation.
    """
    image_output = zeros_like(self.image_input)

    coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z', reverse_coord=True)

    # for all points with non-zeros neighbors, force the neighbors to 0
    for i, coord in enumerate(coordinates_input):
        image_output.data[int(coord.x), int(coord.y), int(coord.z)] = i + 1

    return image_output

def labelize_from_disks():
    """
    Create an image with regions labelized depending on values from reference.
    Typically, user inputs a segmentation image, and labels with disks position, and this function produces
    a segmentation image with vertebral levels labelized.
    Labels are assumed to be non-zero and incremented from top to bottom, assuming a RPI orientation
    """
    image_output = zeros_like(self.image_input)

    coordinates_input = self.image_input.getNonZeroCoordinates()
    coordinates_ref = self.image_ref.getNonZeroCoordinates(sorting='value')

    # for all points in input, find the value that has to be set up, depending on the vertebral level
    for i, coord in enumerate(coordinates_input):
        for j in range(0, len(coordinates_ref) - 1):
            if coordinates_ref[j + 1].z < coord.z <= coordinates_ref[j].z:
                image_output.data[int(coord.x), int(coord.y), int(coord.z)] = coordinates_ref[j].value

    return image_output

def label_vertebrae(levels_user=None):
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

    return image_cubic2point

def MSE(threshold_mse=0):
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

def remove_label(symmetry=False):
    """
    Compare two label images and remove any labels in input image that are not in reference image.
    The symmetry option enables to remove labels from reference image that are not in input image
    """

    image_output = zeros_like(self.image_input)
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

def display_voxel():
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

def get_physical_coordinates():
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

def get_coordinates_in_destination(im_dest, type='discrete'):
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

def diff():
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

def distance_interlabels(max_dist):
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

def continuous_vertebral_levels():
    """
    This function transforms the vertebral levels file from the template into a continuous file.
    Instead of having integer representing the vertebral level on each slice, a continuous value that represents
    the position of the slice in the vertebral level coordinate system.
    The image must be RPI
    :return:
    """
    im_input = Image(self.image_input, self.verbose)
    im_output = zeros_like(self.image_input)

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

def launch_sagittal_viewer(labels, previous_points=None):
    from spinalcordtoolbox.gui import base
    from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

    params = base.AnatomicalParams()
    params.vertebraes = labels
    params.input_file_name = self.image_input.absolutepath
    params.output_file_name = self.fname_output
    params.subtitle = self.msg
    if previous_points is not None:
        params.message_warn = 'Please select the label you want to add \nor correct in the list below before clicking \non the image'
    output = zeros_like(self.image_input)
    output.absolutepath = self.fname_output
    launch_sagittal_dialog(self.image_input, output, params, previous_points)

    return output

def remove_or_keep_labels(labels, action):
    """
    Create or remove labels from self.image_input
    :param list(int): Labels to keep or remove
    :param str: 'remove': remove specified labels (i.e. set to zero), 'keep': keep specified labels and remove the others
    """
    if action == 'keep':
        image_output = zeros_like(self.image_input)
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
