#!/usr/bin/env python
# -*- coding: utf-8
# Functions that utilize the template (e.g., PAM50)

def get_slices_from_vertebral_levels(im_vertlevel, level):
    """
    Find the slices of the corresponding vertebral level.
    Important: This function assumes that the 3rd dimension is Z.
    :param im_vertlevel: image object of vertebral labeling (e.g., label/template/PAM50_levels.nii.gz)
    :param level: int: vertebral level
    :return: list of int: slices
    """

    vert_levels_list = [int(x) for x in vertebral_levels.split(':')]

    # If only one vertebral level was selected (n), consider as n:n
    if len(vert_levels_list) == 1:
        vert_levels_list = [vert_levels_list[0], vert_levels_list[0]]

    # Check if there are only two values [start_level, end_level] and if the end level is higher than the start level
    if (len(vert_levels_list) > 2) or (vert_levels_list[0] > vert_levels_list[1]):
        sct.printv('\nERROR:  "' + vertebral_levels + '" is not correct. Enter format "1:4". Exit program.\n')
        sys.exit(2)

    # Extract the vertebral levels available in the metric image
    vertebral_levels_available = np.array(list(set(data_vertebral_labeling[data_vertebral_labeling > 0])))
