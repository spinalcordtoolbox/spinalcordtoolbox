"""
Utilities for labels

Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import center_of_mass

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline

logger = logging.getLogger(__name__)

# TODO: for vert-disc: make it faster!
#       currently the module display-voxel is very long (esp. when ran on PAM50).
#       We can find an alternative approach by sweeping through centerline voxels.
# TODO: label_disc: for top vertebrae, make label at the center of the cord (currently it's at the tip)


def add(img: Image, value: int) -> Image:
    """
    This function adds a specified value to all non-zero voxels.

    :param img: source image
    :param value: numeric value to add
    :returns: new image with value added
    """
    out = img.copy()
    out.data[np.where(out.data != 0)] += value

    return out


def create_labels_empty(img: Image, coordinates: Sequence[Coordinate]) -> Image:
    """
    Create an empty image with labels listed by the user.
    This method works only if the user inserted correct coordinates.
    If only one label is to be added, coordinates must be completed with '[]'

    :param img: source image
    :param coordinates: list of Coordinate objects (see spinalcordtoolbox.types)
    :returns: empty image with labels
    """
    out = _add_labels(zeros_like(img), coordinates)

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
    out = _add_labels(img.copy(), coordinates)

    return out


def _add_labels(img: Image, coordinates: Sequence[Coordinate]) -> Image:
    """
    Given an image and list of coordinates, add the labels to the image and return it.

    :param img: source image
    :param coordinates: list of Coordinate objects (see spinalcordtoolbox.types)
    :returns: labeled source image
    """

    for i, (x, y, z, v) in enumerate(coordinates):
        if len(img.data.shape) == 3:
            img.data[int(x), int(y), int(z)] = v
        elif len(img.data.shape) == 2:
            if z != 0:
                raise ValueError(f"2D coordinates should have a Z value of 0! Current value: {z}")
            img.data[x, y] = v
        else:
            raise ValueError(f"Data should be 2D or 3D. Current shape is: {img.data.shape}")

        logger.debug(f"Label #{i}: {x}, {y}, {z} --> {v}")

    return img


def create_labels_along_segmentation(img: Image, labels: Sequence[Tuple[int, int]]) -> Image:
    """
    Create an image with labels defined along the spinal cord segmentation (or centerline).
    Input image does **not** need to be RPI (re-orientation is done within this function).

    :param img: source segmentation
    :param labels: list of label tuples as (z_value, label_value)
    :returns: labeled segmentation (Image)
    """
    og_orientation = img.orientation

    if og_orientation != "RPI":
        img.change_orientation("RPI")

    out = zeros_like(img)

    for idx_label, label in enumerate(labels):
        z, value = label

        # update z based on native image orientation (z should represent superior-inferior axis)
        coord = Coordinate([z, z, z])  # since we don't know which dimension corresponds to the superior-inferior

        # axis, we put z in all dimensions (we don't care about x and y here)
        _, _, z_rpi = coord.permute(img, 'RPI')

        # if z=-1, replace with nz/2
        if z == -1:
            z_rpi = int(np.round(out.dim[2] / 2.0))

        # get center of mass of segmentation at given z
        x, y = center_of_mass(np.array(img.data[:, :, z_rpi]))

        # round values to make indices
        x, y = int(np.round(x)), int(np.round(y))

        # display info
        logger.debug(f"Label # {idx_label}: {x}, {y}. {z_rpi} --> {value}")

        if len(out.data.shape) == 3:
            out.data[x, y, z_rpi] = value
        elif len(out.data.shape) == 2:
            if z != 0:
                raise ValueError(f"2D coordinates should have a Z value of 0! Current value: {coord.z}")

            out.data[x, y] = value

    if out.orientation != og_orientation:
        out.change_orientation(og_orientation)
        img.change_orientation(og_orientation)

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
        logger.debug(f"Value = {center_of_mass.value} : ({center_of_mass.x},\
         {center_of_mass.y}, {center_of_mass.z}) --> ({np.round(center_of_mass.x)}, \
         {np.round(center_of_mass.y)}, {np.round(center_of_mass.z)})")

        out.data[int(np.round(center_of_mass.x)), int(np.round(center_of_mass.y)), int(np.round(center_of_mass.z))] = center_of_mass.value

    return out


def increment_z_inverse(img: Image) -> Image:
    """
    Take all non-zero values, sort them along the inverse z direction, and attributes the values 1,
    2, 3, etc.

    :param img: source image
    :returns: image with non-zero values sorted along inverse z
    """
    og_orientation = img.orientation
    if og_orientation != "RPI":
        img.change_orientation("RPI")

    out = zeros_like(img)
    coordinates_input = img.getNonZeroCoordinates(sorting='z', reverse_coord=True)

    # for all points with non-zeros neighbors, force the neighbors to 0
    for i, (x, y, z, _) in enumerate(coordinates_input):
        out.data[int(x), int(y), int(z)] = i + 1

    if out.orientation != og_orientation:
        out.change_orientation(og_orientation)
        img.change_orientation(og_orientation)

    return out


def label_regions_from_reference(img: Image, ref: Image, centerline: bool = False) -> Image:
    """
    Create an image with regions labelized depending on values from reference.
    Typically, user inputs a segmentation image, and labels with discs position, and this function produces
    a segmentation image with vertebral levels labelized.
    Note that no straightening is done. The labelization is done by projecting the labels onto the centerline (extracted from the segmentation).
    Input images do **not** need to be RPI (re-orientation is done within this function).

    :param img: segmentation
    :param ref: reference labels
    :param centerline: boolean, if True the centerline is returned instead of the segmentation (default to false)
    :returns: segmentation image with vertebral levels labelized
    """

    og_img_orientation = img.orientation
    if og_img_orientation != "RPI":
        img.change_orientation("RPI")

    og_ref_orientation = ref.orientation
    if og_ref_orientation != "RPI":
        ref.change_orientation("RPI")

    out = zeros_like(img)

    # Extract centerline from segmentation
    _, arr_ctl, _, _ = get_centerline(img)
    centerline_arr = arr_ctl.T

    if centerline:
        coordinates_input = np.concatenate((centerline_arr, np.ones((centerline_arr.shape[0], 1))), axis=1)
    else:
        coordinates_input = img.getNonZeroCoordinates()

    # Extract reference coordinates
    coordinates_ref = ref.getNonZeroCoordinates(sorting='value')

    # for all points in the segmentation project on the centerline and compare the `z` coordinate to the appropriate vertebral level
    for x, y, z, _ in coordinates_input:
        projection = project_point_on_line(point=np.array([x, y, z]), line=centerline_arr)
        _, _, z_proj = np.rint(projection).astype(int)
        # case 1: `z` is above the top-most disc label
        if z_proj > coordinates_ref[0].z:
            out.data[int(x), int(y), int(z)] = coordinates_ref[0].value - 1
        # case 2: `z` is at or below the bottom-most disc label
        elif z_proj <= coordinates_ref[-1].z:
            out.data[int(x), int(y), int(z)] = coordinates_ref[-1].value
        # case 3: `z` is between two disc labels, so find the correct vertebral level
        else:
            for j in range(len(coordinates_ref) - 1):
                if coordinates_ref[j + 1].z < z_proj <= coordinates_ref[j].z:
                    out.data[int(x), int(y), int(z)] = coordinates_ref[j].value

    # Set back the original orientation
    if img.orientation != og_img_orientation:
        img.change_orientation(og_img_orientation)
        out.change_orientation(og_img_orientation)

    if ref.orientation != og_ref_orientation:
        ref.change_orientation(og_ref_orientation)

    return out


def label_vertebrae(img: Image, vertebral_levels: Sequence[int] = None) -> Image:
    """
    Find the center of mass of vertebral levels specified by the user.

    :param img: source image
    :param vertebral_levels: list of vertebral levels
    :returns: image with labels
    """

    og_orientation = img.orientation
    if og_orientation != "RPI":
        img.change_orientation("RPI")

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

    if out.orientation != og_orientation:
        out.change_orientation(og_orientation)
        img.change_orientation(og_orientation)

    return out


def check_missing_label(img, ref):
    """
    Function that return the list of label that are present in ref and not in img.
    This is useful to find label that are in img and not in the ref (first output) and
    labels that are present in the ref and not in img (second output)

    :param img: source image
    :param ref: reference image
    :return: two lists. The first one is the list of label present in the input and not in the ref image, \
    the second one gives the labels presents in the ref and not in the input.
    """
    coordinates_input = img.getNonZeroCoordinates()
    coordinates_ref = ref.getNonZeroCoordinates()

    rounded_coord_ref_values = [np.round(c.value) for c in coordinates_ref]
    rounded_coord_in_values = [np.round(c.value) for c in coordinates_input]

    missing_labels_ref = [x for x in rounded_coord_in_values if x not in rounded_coord_ref_values]
    missing_labels_inp = [x for x in rounded_coord_ref_values if x not in rounded_coord_in_values]

    if missing_labels_ref:
        logger.warning(f"Label mismatch: Labels {missing_labels_ref} present in input image but missing from reference image.")

    if missing_labels_inp:
        logger.warning(f"Label mismatch: Labels {missing_labels_inp} present in reference image but missing from input image.")

    return missing_labels_ref, missing_labels_inp


# FIXME [AJ]: this is slow on large images
def compute_mean_squared_error(img: Image, ref: Image) -> float:
    """
    Compute the Mean Squared Distance Error between two sets of labels (input and ref).
    Moreover, a warning is generated for each label mismatch.

    :param img: source image
    :param ref: reference image
    :returns: computed MSE
    """
    coordinates_input = img.getNonZeroCoordinates()
    coordinates_ref = ref.getNonZeroCoordinates()
    result = 0.0

    # This line will add warning in the log if there are missing label.
    _, _ = check_missing_label(img, ref)

    for coord in coordinates_input:
        for coord_ref in coordinates_ref:
            if np.round(coord_ref.value) == np.round(coord.value):
                result += (coord_ref.z - coord.z) ** 2
                break

    result = np.sqrt(result / len(coordinates_input))
    logger.info(f"MSE error in Z direction = {result}mm")

    return result


def remove_missing_labels(img: Image, ref: Image):
    """
    Compare an input image and a reference image. Remove any label from the input image that doesn't exist in the reference image.

    :param img: source image
    :param ref: reference image
    :returns: image with labels missing from reference removed
    """
    # Get all label values in the reference image
    ref_values = set(coord.value for coord in ref.getNonZeroCoordinates())
    # Transfer only the source labels that match values present in the reference image
    out = zeros_like(img)
    for src_coord in img.getNonZeroCoordinates():
        if src_coord.value in ref_values:
            out.data[src_coord.x, src_coord.y, src_coord.z] = src_coord.value
    return out


def continuous_vertebral_levels(img: Image) -> Image:
    """
    This function transforms the vertebral levels file from the template into a continuous file.
    Instead of having integer representing the vertebral level on each slice, a continuous value that represents
    the position of the slice in the vertebral level coordinate system. The image must be RPI

    :param img: input image
    :returns: image with continuous vertebral levels
    """
    og_orientation = img.orientation

    if og_orientation != "RPI":
        img.change_orientation("RPI")

    out = zeros_like(img)

    # 1. extract vertebral levels from input image
    #   a. extract centerline
    #   b. for each slice, extract corresponding level
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    _, arr_ctl, _, _ = get_centerline(img, param=ParamCenterline())
    x_centerline_fit, y_centerline_fit, z_centerline = arr_ctl

    value_centerline = np.array(
        [img.data[int(x_centerline_fit[it]), int(y_centerline_fit[it]), int(z_centerline[it])]
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
    coordinates_input = img.getNonZeroCoordinates()
    out.change_type(np.float32)

    # for all points in input, find the value that has to be set up, depending on the vertebral level
    for x, y, z, v in coordinates_input:
        out.data[int(x), int(y), int(z)] = continuous_values[z]

    if out.orientation != og_orientation:
        out.change_orientation(og_orientation)
        img.change_orientation(og_orientation)

    return out


def remove_labels_from_image(img: Image, labels: Sequence[int]) -> Image:
    """
    Remove specified labels (set to 0) from an image.

    :param img: source image
    :param labels: list of specified labels to remove
    :returns: image with labels specified removed
    """
    out = img.copy()

    for label in labels:
        for x, y, z, v in img.getNonZeroCoordinates():
            if label == v:
                out.data[int(x), int(y), int(z)] = 0.0
                break
        else:
            logger.warning(f"Label {label} not found in input image!")

    return out


def remove_other_labels_from_image(img: Image, labels: Sequence[int]) -> Image:
    """
    Remove labels other than specified from an image

    :param img: source image
    :param labels: list of specified labels to keep
    :returns: image with labels specified kept only
    """
    out = zeros_like(img)

    for label in labels:
        for x, y, z, v in img.getNonZeroCoordinates():
            if label == v:
                out.data[int(x), int(y), int(z)] = v
                break
        else:
            logger.warning(f"Label {label} not found in input image!")

    return out


class ShapeMismatchError(ValueError):
    """Custom exception to distinguish between general ValueErrors. Stands for a shape mismatch."""
    def __init__(self, message, **dims):
        self.dims = dims
        super().__init__(f"{message}: {dims}")


def project_centerline(img: Image, ref: Image) -> Image:
    """
    Project an input image on the spinal cord centerline. This projection is obtained by iterating along
    the centerline to identify the shortest distance with each referenced coordinates.
    Typically, user inputs a segmentation image, and labels with disks position, and this function computes
    the identification to the closest coordinates of each labels on the centerline.

    :param img: segmentation
    :param ref: reference labels
    :returns: image with the new projected labels on the centerline
    """

    # Checking orientation
    og_img_orientation = img.orientation
    if img.orientation != "RPI":
        img.change_orientation("RPI")

    og_ref_orientation = ref.orientation
    if ref.orientation != "RPI":
        ref.change_orientation("RPI")

    # Checking input dimensions
    if img.data.shape != ref.data.shape:
        raise ShapeMismatchError(
            "Input image and referenced labels should have the same dimension",
            img=img.data.shape,
            ref=ref.data.shape)

    # Extract centerline from segmentation
    _, arr_ctl, _, _ = get_centerline(img)
    centerline = arr_ctl.T

    # Extract referenced coordinates
    coordinates_ref = ref.getNonZeroCoordinates(sorting='value')

    # Create the output image
    out = zeros_like(ref)

    # Compute the shortest distance for each referenced points on the centerline
    for x, y, z, value in coordinates_ref:
        projection = project_point_on_line(point=np.array([x, y, z]), line=centerline)
        x, y, z = np.rint(projection).astype(int)
        if out.data[x, y, z] != 0:
            if out.data[x, y, z] == value:
                logger.warning("Two labels with the same value were projected on the same coordinate")
            else:
                # overwrite with the highest value because referenced points are sorted
                logger.warning("Two labels were projected on the same coordinate, the highest value was kept")
        out.data[x, y, z] = value

    if out.orientation != og_img_orientation:
        img.change_orientation(og_img_orientation)

    if ref.orientation != og_ref_orientation:
        ref.change_orientation(og_ref_orientation)
        out.change_orientation(og_ref_orientation)

    return out


def project_point_on_line(point, line):
    """
    Project the input point on the referenced line by finding the minimal distance

    :param point: coordinates of a point and its value: point = numpy.array([x y z])
    :param line: list of points coordinates which composes the line
    :returns: closest coordinate to the referenced point on the line: projected_point = numpy.array([X Y Z])
    """
    # Calculate distances between the referenced point and the line then keep the closest point
    dist = np.sum((line - point) ** 2, axis=1)

    return line[np.argmin(dist)]
