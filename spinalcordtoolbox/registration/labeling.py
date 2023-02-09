#########################################################################################
#
# Module containing labeling functions used during registration.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Julien Cohen-Adad, Augustin Roux
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import numpy as np

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.utils.shell import printv
import spinalcordtoolbox.image as msct_image
import spinalcordtoolbox.labels as sct_labels


def add_orthogonal_label(fname_label):
    """
    Add one label of value=99 at the axial slice that contains the label with the lowest value, 10 pixels to the right.
    :param fname_label:
    :return:
    """
    im_label = Image(fname_label)
    orient_orig = im_label.orientation
    # For some reasons (#3304) calling self.change_orientation() replaces self.absolutepath with Null so we need to
    # save it.
    path_label = im_label.absolutepath
    im_label.change_orientation('RPI')
    coord_label = im_label.getCoordinatesAveragedByValue()  # N.B. landmarks are sorted by value
    # Create new label
    from copy import deepcopy
    new_label = deepcopy(coord_label[0])
    # move it 5mm to the left (orientation is RAS)
    nx, ny, nz, nt, px, py, pz, pt = im_label.dim
    new_label.x = np.round(coord_label[0].x + 5.0 / px)  # TODO change to 10 pixels
    # assign value 99
    new_label.value = 99
    # Add to existing image
    im_label.data[int(new_label.x), int(new_label.y), int(new_label.z)] = new_label.value
    # Overwrite label file
    im_label.change_orientation(orient_orig)
    im_label.save(path_label)


def project_labels_on_spinalcord(fname_label, fname_seg, param_centerline, remove_temp_files):
    """
    Project labels orthogonally on the spinal cord centerline. The algorithm works by finding the smallest distance
    between each label and the spinal cord center of mass.
    :param fname_label: file name of labels
    :param fname_seg: file name of cord segmentation (could also be of centerline)
    :param remove_temp_files: int: Whether to remove temporary files. 0 = no, 1 = yes.
    :return: file name of projected labels
    """
    # build output name
    fname_label_projected = add_suffix(fname_label, "_projected")
    # open labels and segmentation
    im_label = Image(fname_label).change_orientation("RPI")
    im_seg = Image(fname_seg)
    native_orient = im_seg.orientation
    im_seg.change_orientation("RPI")

    # smooth centerline and return fitted coordinates in voxel space
    _, centerline, _ = get_centerline(im_seg, param_centerline, remove_temp_files=remove_temp_files)
    arr_ctl = centerline.arr_ctl
    x_centerline_fit, y_centerline_fit, z_centerline = arr_ctl
    # convert pixel into physical coordinates
    centerline_xyz_transposed = \
        [im_seg.transfo_pix2phys([[x_centerline_fit[i], y_centerline_fit[i], z_centerline[i]]])[0]
         for i in range(len(x_centerline_fit))]
    # transpose list
    centerline_phys_x = [i[0] for i in centerline_xyz_transposed]
    centerline_phys_y = [i[1] for i in centerline_xyz_transposed]
    centerline_phys_z = [i[2] for i in centerline_xyz_transposed]
    # get center of mass of label
    labels = im_label.getCoordinatesAveragedByValue()
    # initialize image of projected labels. Note that we use the space of the seg (not label).
    im_label_projected = msct_image.zeros_like(im_seg, dtype=np.uint8)

    # loop across label values
    for label in labels:
        # convert pixel into physical coordinates for the label
        label_phys_x, label_phys_y, label_phys_z = im_label.transfo_pix2phys([[label.x, label.y, label.z]])[0]
        # calculate distance between label and each point of the centerline
        distance_centerline = [np.linalg.norm([centerline_phys_x[i] - label_phys_x,
                                               centerline_phys_y[i] - label_phys_y,
                                               centerline_phys_z[i] - label_phys_z])
                               for i in range(len(x_centerline_fit))]
        # get the index corresponding to the min distance
        ind_min_distance = np.argmin(distance_centerline)
        # get centerline coordinate (in physical space)
        [min_phy_x, min_phy_y, min_phy_z] = [centerline_phys_x[ind_min_distance],
                                             centerline_phys_y[ind_min_distance],
                                             centerline_phys_z[ind_min_distance]]
        # convert coordinate to voxel space
        minx, miny, minz = im_seg.transfo_phys2pix([[min_phy_x, min_phy_y, min_phy_z]])[0]
        # use that index to assign projected label in the centerline
        im_label_projected.data[minx, miny, minz] = label.value
    # re-orient projected labels to native orientation and save
    im_label_projected.change_orientation(native_orient).save(fname_label_projected)
    return fname_label_projected


# Resample labels
# ==========================================================================================
def resample_labels(fname_labels, fname_dest, fname_output):
    """
    This function re-create labels into a space that has been resampled. It works by re-defining the location of each
    label using the old and new voxel size.
    IMPORTANT: this function assumes that the origin and FOV of the two images are the SAME.
    """
    # get dimensions of input and destination files
    nx, ny, nz, _, _, _, _, _ = Image(fname_labels).dim
    nxd, nyd, nzd, _, _, _, _, _ = Image(fname_dest).dim
    sampling_factor = [float(nx) / nxd, float(ny) / nyd, float(nz) / nzd]

    og_labels = Image(fname_labels).getNonZeroCoordinates()
    new_labels = [Coordinate([int(np.round(int(x) / sampling_factor[0])),
                              int(np.round(int(y) / sampling_factor[1])),
                              int(np.round(int(z) / sampling_factor[2])),
                              int(float(v))])
                  for x, y, z, v in og_labels]

    sct_labels.create_labels_empty(Image(fname_dest).change_type('uint8'), new_labels).save(path=fname_output)


def check_labels(fname_landmarks, label_type='body'):
    """
    Make sure input labels are consistent
    Parameters
    ----------
    fname_landmarks: file name of input labels
    label_type: 'body', 'disc', 'spinal'
    Returns
    -------
    none
    """
    printv('\nCheck input labels...')
    # open label file
    image_label = Image(fname_landmarks)
    # -> all labels must be different
    labels = image_label.getNonZeroCoordinates(sorting='value')
    # check if labels are integer
    for label in labels:
        if not int(label.value) == label.value:
            printv('ERROR: Label should be integer.', 1, 'error')
    # check if there are duplicates in label values
    n_labels = len(labels)
    list_values = [labels[i].value for i in range(0, n_labels)]
    list_duplicates = [x for x in list_values if list_values.count(x) > 1]
    if not list_duplicates == []:
        printv('ERROR: Found two labels with same value.', 1, 'error')
    return labels
