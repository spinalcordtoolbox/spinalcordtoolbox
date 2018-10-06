#!/usr/bin/env python

from __future__ import division, absolute_import

import os
import sys

import numpy as np

import sct_utils as sct
from msct_parser import Parser
from spinalcordtoolbox.centerline import optic
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image


def _call_viewer_centerline(fname_in):
    from spinalcordtoolbox.gui.base import AnatomicalParams
    from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

    params = AnatomicalParams()
    # setting maximum number of points to a reasonable value
    params.num_points = 20
    params.interval_in_mm = 30
    params.starting_slice = 'top'
    im_data = Image(fname_in)

    im_mask_viewer = msct_image.zeros_like(im_data)
    im_mask_viewer.absolutepath = sct.add_suffix(fname_in, '_labels_viewer')
    controller = launch_centerline_dialog(im_data, im_mask_viewer, params)
    fname_labels_viewer = sct.add_suffix(fname_in, '_labels_viewer')

    if not controller.saved:
        sct.log.error('The viewer has been closed before entering all manual points. Please try again.')
        sys.exit(1)
    # save labels
    controller.as_niftii(fname_labels_viewer)

    return fname_labels_viewer


def _from_viewerLabels_to_centerline(fname_labels, fname_out):
    from sct_straighten_spinalcord import smooth_centerline
    from msct_types import Centerline

    image_labels = Image(fname_labels)
    orientation = image_labels.orientation
    if orientation != 'RPI':  # RPI orientation assumed in this function
        image_labels.change_orientation('RPI')

    # fit centerline, smooth it and return the first derivative (in physical space)
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(image_labels, algo_fitting='nurbs', nurbs_pts_number=10000, phys_coordinates=True, verbose=False, all_slices=False)
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

    # average centerline coordinates over slices of the image
    x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(image_labels)

    # compute z_centerline in image coordinates with continuous precision
    voxel_coordinates = image_labels.transfo_phys2pix([[x_centerline_fit_rescorr[i], y_centerline_fit_rescorr[i], z_centerline_rescorr[i]] for i in range(len(z_centerline_rescorr))], real=False)
    x_centerline_voxel_cont = [coord[0] for coord in voxel_coordinates]
    y_centerline_voxel_cont = [coord[1] for coord in voxel_coordinates]
    z_centerline_voxel_cont = [coord[2] for coord in voxel_coordinates]

    # Create an image with the centerline
    image_centerline = msct_image.zeros_like(image_labels)
    min_z_index, max_z_index = int(np.round(min(z_centerline_voxel_cont))), int(np.round(max(z_centerline_voxel_cont)))
    for iz in range(min_z_index, max_z_index + 1):
        image_centerline.data[int(np.round(x_centerline_voxel_cont[iz - min_z_index])), int(np.round(y_centerline_voxel_cont[iz - min_z_index])), int(iz)] = 1  # if index is out of bounds here for hanning: either the segmentation has holes or labels have been added to the file

    if orientation != 'RPI':
        image_centerline.change_orientation(orientation)

    # Write the centerline image
    image_centerline.change_type(np.uint8).save(fname_out)


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This function allows the extraction of the spinal cord centerline. Two methods are available: OptiC (automatic) and Viewer (manual).\n\nReference: C Gros, B De Leener, et al. Automatic spinal cord localization, robust to MRI contrasts using global curve optimization (2017). doi.org/10.1016/j.media.2017.12.001""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=False,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name="-roi",
                      type_value="multiple_choice",
                      description="outputs a ROI file, compatible with JIM software.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')
    parser.add_option(name="-method",
                      type_value="multiple_choice",
                      description="Method used for extracting the centerline.\n"
                                  "optic: automatic spinal cord detection method\n"
                                  "viewer: manually selected a few points, approximation with NURBS",
                      mandatory=False,
                      example=['optic', 'viewer'],
                      default_value='optic')
    parser.add_option(name="-gap",
                      type_value="float",
                      description="Gap in mm between manually selected points when using the Viewer method.",
                      mandatory=False,
                      default_value='10.0')
    parser.add_option(name="-igt",
                      type_value="image_nifti",
                      description="File name of ground-truth centerline or segmentation (binary nifti).",
                      mandatory=False)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    return parser

def run_main():
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    # Input filename
    fname_input_data = arguments["-i"]
    fname_data = os.path.abspath(fname_input_data)

    # Method used
    method = 'optic'
    if "-method" in arguments:
        method = arguments["-method"]

    # Contrast type
    contrast_type = ''
    if "-c" in arguments:
        contrast_type = arguments["-c"]
    if method == 'optic' and not contrast_type:
        # Contrast must be
        error = 'ERROR: -c is a mandatory argument when using Optic method.'
        sct.printv(error, type='error')
        return

    # Ga between slices
    interslice_gap = 10.0
    if "-gap" in arguments:
        interslice_gap = float(arguments["-gap"])

    # Output folder
    if "-ofolder" in arguments:
        folder_output = arguments["-ofolder"]
    else:
        folder_output = '.'

    # Remove temporary files
    remove_temp_files = True
    if "-r" in arguments:
        remove_temp_files = bool(int(arguments["-r"]))

    # Outputs a ROI file
    output_roi = False
    if "-roi" in arguments:
        output_roi = bool(int(arguments["-roi"]))

    # Verbosity
    verbose = 0
    if "-v" in arguments:
        verbose = int(arguments["-v"])

    if method == 'viewer':
        centerline_filename = sct.add_suffix(fname_data, "_centerline")
        fname_labels_viewer = _call_viewer_centerline(fname_in=fname_data)
        _from_viewerLabels_to_centerline(fname_labels=fname_labels_viewer, fname_out=centerline_filename)

    else:
        # condition on verbose when using OptiC
        if verbose == 1:
            verbose = 2

        # OptiC models
        path_script = os.path.dirname(__file__)
        path_sct = os.path.dirname(path_script)
        optic_models_path = os.path.join(path_sct, 'data', 'optic_models', '{}_model'.format(contrast_type))

        # Execute OptiC binary
        _, centerline_filename = optic.detect_centerline(image_fname=fname_data,
                                                    contrast_type=contrast_type,
                                                    optic_models_path=optic_models_path,
                                                    folder_output=folder_output,
                                                    remove_temp_files=remove_temp_files,
                                                    output_roi=output_roi,
                                                    verbose=verbose)

    sct.display_viewer_syntax([fname_input_data, centerline_filename], colormaps=['gray', 'red'], opacities=['', '1'])

if __name__ == '__main__':
    run_main()
