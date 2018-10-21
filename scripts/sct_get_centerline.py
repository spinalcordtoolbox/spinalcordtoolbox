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
from sct_process_segmentation import extract_centerline


def _call_viewer_centerline(fname_in, interslice_gap=20.0):
    from spinalcordtoolbox.gui.base import AnatomicalParams
    from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

    im_data = Image(fname_in)

    # Get the number of slice along the (IS) axis
    im_tmp = msct_image.change_orientation(im_data, 'RPI')
    _, _, nz, _, _, _, pz, _ = im_tmp.dim
    del im_tmp

    params = AnatomicalParams()
    # setting maximum number of points to a reasonable value
    params.num_points = np.ceil(nz * pz / interslice_gap) + 2
    params.interval_in_mm = interslice_gap
    params.starting_slice = 'top'

    im_mask_viewer = msct_image.zeros_like(im_data)
    controller = launch_centerline_dialog(im_data, im_mask_viewer, params)
    fname_labels_viewer = sct.add_suffix(fname_in, '_viewer')

    if not controller.saved:
        sct.log.error('The viewer has been closed before entering all manual points. Please try again.')
        sys.exit(1)
    # save labels
    controller.as_niftii(fname_labels_viewer)

    return fname_labels_viewer


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
                      default_value='20.0')
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
                      example=["0", "1", "2"],
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
        fname_labels_viewer = _call_viewer_centerline(fname_in=fname_data, interslice_gap=interslice_gap)
        centerline_filename = extract_centerline(fname_labels_viewer, remove_temp_files=remove_temp_files,
                                                 verbose=verbose, algo_fitting='nurbs', nurbs_pts_number=8000)

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
