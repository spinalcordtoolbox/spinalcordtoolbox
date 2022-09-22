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

import os
import sys
from typing import Sequence

import numpy as np

import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils import (SCTArgumentParser, Metavar, ActionCreateFolder, list_type, init_sct, printv,
                                     parse_num_list, set_loglevel)
from spinalcordtoolbox.utils.shell import display_viewer_syntax


def get_parser():
    parser = SCTArgumentParser(
        description="Utility functions for label images."
    )

    req_group = parser.add_argument_group("\nREQUIRED I/O")
    req_group.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input image (Required) Example: t2_labels.nii.gz"
    )

    io_group = parser.add_argument_group("\nOPTIONAL I/O")

    io_group.add_argument(
        '-o',
        metavar=Metavar.file,
        default='labels.nii.gz',
        help=("Output image. Note: Only some label utilities create an output image. Example: t2_labels.nii.gz")
    )

    io_group.add_argument(
        '-ilabel',
        metavar=Metavar.file,
        help="File that contain labels that you want to correct. It is possible to add new points with this option. "
             "Use with -create-viewer. Example: t2_labels_auto.nii.gz"
    )

    functions = parser.add_argument_group("\nLABEL FUNCTIONS")
    func_group = functions.add_mutually_exclusive_group(required=True)

    func_group.add_argument(
        '-add',
        metavar=Metavar.int,
        type=int,
        help="Add value to all labels. Value can be negative."
    )

    func_group.add_argument(
        '-create',
        metavar=Metavar.list,
        type=list_type(':', Coordinate),
        help="Create labels in a new image. List labels as: x1,y1,z1,value1:x2,y2,z2,value2. "
             "Example: 12,34,32,1:12,35,33,2"
    )

    func_group.add_argument(
        '-create-add',
        metavar=Metavar.list,
        type=list_type(':', Coordinate),
        help="Same as '-create', but add labels to the input image instead of creating a new image. "
             "Example: 12,34,32,1:12,35,33,2"
    )

    func_group.add_argument(
        '-create-seg',
        metavar=Metavar.list,
        type=list_type(':', list_type(',', int)),
        help="Create labels on a cord segmentation (or centerline) image defined by '-i'. Each label should be "
             "specified using the form 'v1,v2' where 'v1' is value of the slice index along the inferior-superior "
             "axis, and 'v2' is the value of the label. Separate each label with ':'. \n"
             "Example: '-create-seg 5,1:14,2:23,3' adds three labels at the axial slices 5, 14, and 23 (starting from "
             "the most inferior slice)."
    )

    func_group.add_argument(
        '-create-seg-mid',
        metavar=Metavar.int,
        type=int,
        help="Similar to '-create-seg'. This option takes a single label value, and will automatically select the "
             "mid-point slice in the inferior-superior direction (so there is no need for a slice index).\n"
             "This is useful for when you have centered the field of view of your data at a specific location. "
             "For example, if you already know that the C2-C3 disc is centered in the I-S direction, then "
             "you can enter '-create-seg-mid 3' for that label. This saves you the trouble of having to manually "
             "specify a slice index using '-create-seg'."
    )

    func_group.add_argument(
        '-create-viewer',
        metavar=Metavar.list,
        help="Manually label from a GUI a list of labels IDs. Provide a comma-separated list "
             "containing individual values and/or intervals. Example: '-create-viewer 1:4,6,8' "
             "will allow you to add labels [1,2,3,4,6,8] using the GUI."
    )

    func_group.add_argument(
        '-cubic-to-point',
        action="store_true",
        help="Compute the center-of-mass for each label value."
    )

    func_group.add_argument(
        '-disc',
        metavar=Metavar.file,
        help="Create an image with regions labelized depending on values from reference"
    )

    func_group.add_argument(
        '-display',
        action="store_true",
        help="Display all labels (i.e. non-zero values)."
    )
    func_group.add_argument(
        '-increment',
        action="store_true",
        help="Takes all non-zero values, sort them along the inverse z direction, and attributes the values "
             "1, 2, 3, etc."
    )
    func_group.add_argument(
        '-vert-body',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="From vertebral labeling, create points that are centered at the mid-vertebral levels. Separate "
             "desired levels with ','. Example: 3,8\n"
             "To get all levels, enter 0."
    )

    func_group.add_argument(
        '-vert-continuous',
        action="store_true",
        help="Convert discrete vertebral labeling to continuous vertebral labeling.",
    )
    func_group.add_argument(
        '-MSE',
        metavar=Metavar.file,
        help="Compute Mean Square Error between labels from input and reference image. Specify reference image here."
    )
    func_group.add_argument(
        '-remove-reference',
        metavar=Metavar.file,
        help="Remove labels from input image (-i) that are not in reference image (specified here)."
    )
    func_group.add_argument(
        '-remove-sym',
        metavar=Metavar.file,
        help="Remove labels from input image (-i) and reference image (specified here) that don't match. You must "
             "provide two output names separated by ','."
    )
    func_group.add_argument(
        '-remove',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Remove labels of specific value (specified here) from reference image."
    )
    func_group.add_argument(
        '-keep',
        metavar=Metavar.list,
        type=list_type(',', int),
        help="Keep labels of specific value (specified here) from reference image."
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )

    optional.add_argument(
        '-msg',
        metavar=Metavar.str,
        help="Display a message to explain the labeling task. Use with -create-viewer"
    )

    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
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
def main(argv: Sequence[str]):
    for i, arg in enumerate(argv):
        if arg == '-create-seg' and len(argv) > i+1 and '-1,' in argv[i+1]:
            raise DeprecationWarning("The use of '-1' for '-create-seg' has been deprecated. Please use "
                                     "'-create-seg-mid' instead.")

    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    input_filename = arguments.i
    output_fname = arguments.o

    img = Image(input_filename)
    dtype = None

    if arguments.add is not None:
        value = arguments.add
        out = sct_labels.add(img, value)
    elif arguments.create is not None:
        labels = arguments.create
        out = sct_labels.create_labels_empty(img, labels)
    elif arguments.create_add is not None:
        labels = arguments.create_add
        out = sct_labels.create_labels(img, labels)
    elif arguments.create_seg is not None:
        labels = arguments.create_seg
        out = sct_labels.create_labels_along_segmentation(img, labels)
    elif arguments.create_seg_mid is not None:
        labels = [(-1, arguments.create_seg_mid)]
        out = sct_labels.create_labels_along_segmentation(img, labels)
    elif arguments.cubic_to_point:
        out = sct_labels.cubic_to_point(img)
    elif arguments.display:
        display_voxel(img, verbose)
        return
    elif arguments.increment:
        out = sct_labels.increment_z_inverse(img)
    elif arguments.disc is not None:
        ref = Image(arguments.disc)
        out = sct_labels.labelize_from_discs(img, ref)
    elif arguments.vert_body is not None:
        levels = arguments.vert_body
        if len(levels) == 1 and levels[0] == 0:
            levels = None  # all levels
        out = sct_labels.label_vertebrae(img, levels)
    elif arguments.vert_continuous:
        out = sct_labels.continuous_vertebral_levels(img)
        dtype = 'float32'
    elif arguments.MSE is not None:
        ref = Image(arguments.MSE)
        mse = sct_labels.compute_mean_squared_error(img, ref)
        printv(f"Computed MSE: {mse}")
        return
    elif arguments.remove_reference is not None:
        ref = Image(arguments.remove_reference)
        out = sct_labels.remove_missing_labels(img, ref)
    elif arguments.remove_sym is not None:
        # first pass use img as source
        ref = Image(arguments.remove_reference)
        out = sct_labels.remove_missing_labels(img, ref)

        # second pass use previous pass result as reference
        ref_out = sct_labels.remove_missing_labels(ref, out)
        ref_out.save(path=ref.absolutepath)
    elif arguments.remove is not None:
        labels = arguments.remove
        out = sct_labels.remove_labels_from_image(img, labels)
    elif arguments.keep is not None:
        labels = arguments.keep
        out = sct_labels.remove_other_labels_from_image(img, labels)
    elif arguments.create_viewer is not None:
        msg = "" if arguments.msg is None else f"{arguments.msg}\n"
        if arguments.ilabel is not None:
            input_labels_img = Image(arguments.ilabel)
            out = launch_manual_label_gui(img, input_labels_img, parse_num_list(arguments.create_viewer), msg)
        else:
            out = launch_sagittal_viewer(img, parse_num_list(arguments.create_viewer), msg)

    printv("Generating output files...")
    out.save(path=output_fname, dtype=dtype)
    display_viewer_syntax([input_filename, output_fname], verbose=verbose)

    if arguments.qc is not None:
        generate_qc(fname_in1=input_filename, fname_seg=output_fname, args=argv,
                    path_qc=os.path.abspath(arguments.qc), dataset=arguments.qc_dataset,
                    subject=arguments.qc_subject, process='sct_label_utils')


def display_voxel(img: Image, verbose: int = 1) -> Sequence[Coordinate]:
    """
    Display all the labels that are contained in the input image.
    :param img: source image
    :param verbose: verbosity level
    """

    coordinates_input = img.getNonZeroCoordinates(sorting='value')
    useful_notation = ''

    for coord in coordinates_input:
        printv('Position=(' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ') -- Value= ' + str(coord.value), verbose=verbose)
        if useful_notation:
            useful_notation = useful_notation + ':'
        useful_notation += str(coord)

    printv('All labels (useful syntax):', verbose=verbose)
    printv(useful_notation, verbose=verbose)


def launch_sagittal_viewer(img: Image, labels: Sequence[int], msg: str, previous_points: Sequence[Coordinate] = None, output_img: Image = None) -> Image:
    from spinalcordtoolbox.gui import base
    from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog
    params = base.AnatomicalParams()
    params.vertebraes = labels
    params.input_file_name = img.absolutepath

    if output_img is not None:
        params.output_file_name = output_img.absolutepath
    else:
        params.output_file_name = img.absolutepath

    params.subtitle = msg

    if previous_points is not None:
        params.message_warn = 'Please select the label you want to add \nor correct in the list below before clicking \non the image'

    out = zeros_like(img, dtype='uint8')
    out.absolutepath = params.output_file_name
    launch_sagittal_dialog(img, out, params, previous_points)

    return out


def launch_manual_label_gui(img: Image, input_labels_img: Image, labels: Sequence[int], msg):
    # the input image is reoriented to 'SAL' when open by the GUI
    input_labels_img.change_orientation('SAL')
    mid = int(np.round(input_labels_img.data.shape[2] / 2))
    previous_points = input_labels_img.getNonZeroCoordinates()

    # boolean used to mark first element to initiate the list.
    first = True

    previous_label = None

    for i in range(len(previous_points)):
        if int(previous_points[i].value) in labels:
            pass
        else:
            labels.append(int(previous_points[i].value))
        if first:
            points = np.array([previous_points[i]. x, previous_points[i].y, previous_points[i].z, previous_points[i].value])
            points = np.reshape(points, (1, 4))
            previous_label = points
            first = False
        else:
            points = np.array([previous_points[i].x, previous_points[i].y, previous_points[i].z, previous_points[i].value])
            points = np.reshape(points, (1, 4))
            previous_label = np.append(previous_label, points, axis=0)
        labels.sort()

    # check if variable was created which means the file was not empty and contains some points asked in labels
    if previous_label is not None:
        # project onto mid sagittal plane
        for i in range(len(previous_label)):
            previous_label[i][2] = mid

    out = launch_sagittal_viewer(img, labels, msg, previous_points=previous_label)

    return out


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
