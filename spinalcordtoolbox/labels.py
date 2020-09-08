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
from typing import Sequence

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

# FIXME [AJ] is it really needed? Haven't seen a caller use sct_label_utils -create without add


def create_labels_empty(img: Image, coordinates: Sequence[Coordinate]) -> Image:
    """
    Create an empty image with labels listed by the user.
    This method works only if the user inserted correct coordinates.
    If only one label is to be added, coordinates must be completed with '[]'
    :param img: source image
    :param coordinates: list of Coordinate objects (see spinalcordtoolbox.types)
    :returns: empty image with labels
    """
    out = zeros_like(img)
    out = _add_labels(img, coordinates)

    return out


def create_labels(img: Image, coordinates: Sequence[Coordinate]) -> Image:
    """
    Add labels provided by a user to the image.
    This method works only if the user inserted correct coordinates.
    If only one label is to be added, coordinates must be completed with '[]'
    :param img: source image
    :param coordinates: list of Coordinate objects (see spinalcordtoolbox.types)
    :returns: labeled source image
    """
    out = img.copy()
    out = _add_labels(img, coordinates)

    return out


def _add_labels(img: Image, coordinates: Sequence[Coordinate]) -> Image:
    """
    Given an image and list of coordinates, add the labels to the image and return it.
    :param img: source image
    :param coordinates: list of Coordinate objects (see spinalcordtoolbox.types)
    :returns: labeled source image
    """
    for i, coord in enumerate(coordinates):
        if len(img.data.shape) == 3:
            img.data[int(coord.x), int(coord.y), int(coord.z)] = coord.value
        elif len(img.data.shape) == 2:
            if str(coord.z) != '0':
                raise ValueError(f"2D coordinates should have a Z value of 0! Current value: {str(coord.z)}")
            img.data[int(coord.x), int(coord.y)] = coord.value
        else:
            raise ValueError(f"Data should be 2D or 3D. Current shape is: {str(img.data.shape)}")

        logger.info(f"Label #{str(i)}: {str(coord.x)}, {str(coord.y)}, {str(coord.z)} --> {str(coord.value)}")

    return img


def create_labels_along_segmentation(img: Image, coordinates: Sequence[Coordinate]) -> Image:
    """
    Create an image with labels defined along the spinal cord segmentation (or centerline).
    Input image does **not** need to be RPI (re-orientation is done within this function).
    :param img: source segmentation
    :param coordinates: list of Coordinate objects (see spinalcordtoolbox.types)
    :returns: labeled segmentation (Image)
    """

    if img.orientation != "RPI":
        img_rpi = img.copy().change_orientation('RPI')
    else:
        img_rpi = img.copy()

    out = zeros_like(img_rpi)

    for ilabel, coord in enumerate(coordinates):

        # split coord string
        list_coord = coord.split(',')

        # convert to int() and assign to variable
        z, value = [int(i) for i in list_coord]

        # update z based on native image orientation (z should represent superior-inferior axis)
        coord = Coordinate([z, z, z])  # since we don't know which dimension corresponds to the superior-inferior

        # axis, we put z in all dimensions (we don't care about x and y here)
        _, _, z_rpi = coord.permute(img, 'RPI')

        # if z=-1, replace with nz/2
        if z == -1:
            z_rpi = int(np.round(out.dim[2] / 2.0))

        # get center of mass of segmentation at given z
        x, y = ndimage.measurements.center_of_mass(np.array(img_rpi.data[:, :, z_rpi]))

        # round values to make indices
        x, y = int(np.round(x)), int(np.round(y))

        # display info
        logger.info(f"Label # {str(ilabel)}: {str(x)}, {str(y)}. {str(z_rpi)} --> {str(value)}")

        if len(out.data.shape) == 3:
            out.data[x, y, z_rpi] = value
        elif len(out.data.shape) == 2:
            if str(z) != '0':
                raise ValueError(f"2D coordinates should have a Z value of 0! Current value: {str(coord.z)}")

            out.data[x, y] = value

    # change orientation back to native
    return out.change_orientation(img.orientation)


# FIXME [AJ] better name for this? insert_plane_between_labels() ?
def plan(img: Image, width: int, offset: int = 0, gap: int = 1) -> Image:
    """
    Create a plane of thickness="width" and changes its value with an offset and a gap between labels.
    :param img: Source image
    :param width: thickness
    :param offset: offset
    :param gap: gap
    :returns: image with plane
    """
    out = zeros_like(img)
    coordinates = img.getNonZeroCoordinates()

    for coord in coordinates:
        out.data[:, :, int(coord.z) - width:int(coord.z) + width] = offset + gap * coord.value

    return out


# FIXME [AJ] better name for this? generate_plane_for_label() ?
def plan_ref(img: Image, ref: Image) -> Image:
    """
    Generate a plane in the reference space for each label present in the input image
    :param img: source image
    :param ref: reference image
    :returns: new image with plane in ref space for each label
    """

    out = zeros_like(ref)

    image_input_neg = zeros_like(img)
    image_input_pos = zeros_like(img)

    X, Y, Z = (img.data < 0).nonzero()
    for i in range(len(X)):
        image_input_neg.data[X[i], Y[i], Z[i]] = -img.data[X[i], Y[i], Z[i]]  # in order to apply getNonZeroCoordinates

    X_pos, Y_pos, Z_pos = (img.data > 0).nonzero()
    for i in range(len(X_pos)):
        image_input_pos.data[X_pos[i], Y_pos[i], Z_pos[i]] = img.data[X_pos[i], Y_pos[i], Z_pos[i]]

    coordinates_input_neg = image_input_neg.getNonZeroCoordinates()
    coordinates_input_pos = image_input_pos.getNonZeroCoordinates()

    out.change_type('float32')
    for coord in coordinates_input_neg:
        out.data[:, :, int(coord.z)] = -coord.value  # PB: takes the int value of coord.value
    for coord in coordinates_input_pos:
        out.data[:, :, int(coord.z)] = coord.value

    return out


def cubic_to_point(img: Image) -> Image:
    """
    Calculate the center of mass of each group of labels and returns a file of same size with only a
    label by group at the center of mass of this group.
    It is to be used after applying homothetic warping field to a label file as the labels will be dilated.

    .. note::
        Be careful: this algorithm computes the center of mass of voxels with same value, if two groups of voxels with
        the same value are present but separated in space, this algorithm will compute the center of mass of the two
        groups together.

    :param img: source image
    :return: image with labels at center of mass
    """

    # 0. Initialization of output image
    out = zeros_like(img)

    # 1. Extraction of coordinates from all non-null voxels in the image. Coordinates are sorted by value.
    coordinates = img.getNonZeroCoordinates(sorting='value')

    # 2. Separate all coordinates into groups by value
    groups = dict()
    for coord in coordinates:
        if coord.value in groups:
            groups[coord.value].append(coord)
        else:
            groups[coord.value] = [coord]

    # 3. Compute the center of mass of each group of voxels and write them into the output image
    for _, list_coord in groups.items():
        center_of_mass = sum(list_coord) / float(len(list_coord))
        logger.info(f"Value = {str(center_of_mass.value)} : ({str(center_of_mass.x)},\
         {str(center_of_mass.y) }, {str(center_of_mass.z)}) --> ({str(np.round(center_of_mass.x))}, \
         {str(np.round(center_of_mass.y))}, {str(np.round(center_of_mass.z))})")

        out.data[int(np.round(center_of_mass.x)), int(np.round(center_of_mass.y)), int(np.round(center_of_mass.z))] = center_of_mass.value

    return out


def increment_z_inverse(img: Image) -> Image:
    """
    Take all non-zero values, sort them along the inverse z direction, and attributes the values 1,
    2, 3, etc. This function assumes RPI orientation.
    :param img: source image
    :returns: image with non-zero balue sorted along inverse z
    """
    if img.orientation != "RPI":
        raise ValueError("Source image needs to be in RPI orientation!")

    out = zeros_like(img)
    coordinates_input = img.getNonZeroCoordinates(sorting='z', reverse_coord=True)

    # for all points with non-zeros neighbors, force the neighbors to 0
    for i, coord in enumerate(coordinates_input):
        out.data[int(coord.x), int(coord.y), int(coord.z)] = i + 1

    return out


def labelize_from_disks(img: Image, ref: Image) -> Image:
    """
    Create an image with regions labelized depending on values from reference.
    Typically, user inputs a segmentation image, and labels with disks position, and this function produces
    a segmentation image with vertebral levels labelized.
    Labels are assumed to be non-zero and incremented from top to bottom, assuming a RPI orientation
    :param img: segmentation
    :param ref: reference labels
    :returns: segmentation image with vertebral levels labelized
    """
    out = zeros_like(img)

    coordinates_input = img.getNonZeroCoordinates()
    coordinates_ref = ref.getNonZeroCoordinates(sorting='value')

    # for all points in input, find the value that has to be set up, depending on the vertebral level
    for coord in coordinates_input:
        for j in range(0, len(coordinates_ref) - 1):
            if coordinates_ref[j + 1].z < coord.z <= coordinates_ref[j].z:
                out.data[int(coord.x), int(coord.y), int(coord.z)] = coordinates_ref[j].value

    return out


def label_vertebrae(img: Image, vertebral_levels: Sequence[int] = None) -> Image:
    """
    Find the center of mass of vertebral levels specified by the user.
    :param img: source image
    :param vertebral_levels: list of vertebral levels
    :returns: image with labels
    """
    # get center of mass of each vertebral level
    out = cubic_to_point(img)

    # get list of coordinates for each label
    list_coordinates = out.getNonZeroCoordinates(sorting='value')

    # if user did not specify levels, include all:
    if not vertebral_levels:
        vertebral_levels = [int(i.value) for i in list_coordinates]

    # loop across labels and remove those that are not listed by the user
    for i in range(len(list_coordinates)):
        # check if this level is NOT in vertebral_levels. if not set value to zero
        if not vertebral_levels.count(int(list_coordinates[i].value)):
            out.data[int(list_coordinates[i].x), int(list_coordinates[i].y), int(list_coordinates[i].z)] = 0

    return out


def compute_mean_squared_error(img: Image, ref: Image) -> float:
    """
    Compute the Mean Squared Distance Error between two sets of labels (input and ref).
    Moreover, a warning is generated for each label mismatch.
    If the MSE is above the threshold provided (by default = 0mm), a log is reported with the filenames considered here.
    :param img: source image
    :param ref: reference image
    :returns: computed MSE
    """
    coordinates_input = img.getNonZeroCoordinates()
    coordinates_ref = ref.getNonZeroCoordinates()

    # check if all the labels in both the images match
    if len(coordinates_input) != len(coordinates_ref):
        raise ValueError(f"Input and reference image don't have the same number of labels!")

    rounded_coord_ref_values = [np.round(coord_ref.value) for coord_ref in coordinates_ref]
    rounded_coord_in_values = [np.round(coord.value) for coord in coordinates_input]

    for coord in coordinates_input:
        if np.round(coord.value) not in rounded_coord_ref_values:
            raise ValueError("Labels mismatch")  # FIXME [AJ] be more specific

    for coord_ref in coordinates_ref:
        if np.round(coord_ref.value) not in rounded_coord_in_values:
            raise ValueError("Labels mismatch")  # FIXME [AJ] be more specific

    result = 0.0

    for coord, coord_ref in zip(coordinates_input, coordinates_ref):
        if np.round(coord_ref.value) == np.round(coord.value):
            result += (coord_ref.z - coord.z) ** 2
            break

    result = np.sqrt(result / len(coordinates_input))
    logger.info(f"MSE error in Z direction = {result}mm")

    return result


def remove_missing_labels(img: Image, ref: Image) -> Image:
    """
    Compare an input image and a reference image. Remove any label from the input image that doesn't exist in the reference image.
    :param img: source image
    :param ref: reference image
    :returns: image with labels missing from reference removed
    """

    out = zeros_like(img)

    coord_input = img.getNonZeroCoordinates(coordValue=True)
    coord_ref = ref.getNonZeroCoordinates(coordValue=True)

    coord_intersection = list(set(coord_input.intersection(set(coord_ref))))

    for coord in coord_intersection:
        out.data[int(coord.x), int(coord.y), int(coord.z)] = int(np.round(coord.value))

    return out


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
