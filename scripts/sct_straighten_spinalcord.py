#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Julien Touati
# Modified: 2014-09-01
#
# License: see the LICENSE.TXT
# ======================================================================================================================


from __future__ import division, absolute_import

import sys, os

from spinalcordtoolbox.straightening import SpinalCordStraightener
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.reports.qc import generate_qc

from msct_parser import Parser
import sct_utils as sct


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("This program takes as input an anatomic image and the spinal cord centerline (or "
                                 "segmentation), and returns the an image of a straightened spinal cord. Reference: "
                                 "De Leener B, Mangeat G, Dupont S, Martin AR, Callot V, Stikov N, Fehlings MG, "
                                 "Cohen-Adad J. Topologically-preserving straightening of spinal cord MRI. J Magn "
                                 "Reson Imaging. 2017 Oct;46(4):1209-1219")
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="Input image with curved spinal cord.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-s",
                      type_value="image_nifti",
                      description="Spinal cord centerline (or segmentation) of the input image. To obtain the centerline"
                                  "you can use sct_get_centerline. To obtain the segmentation you can use sct_propseg"
                                  "or sct_deepseg_sc.",
                      mandatory=True,
                      example="centerline.nii.gz")
    parser.add_option(name="-c",
                      type_value=None,
                      description="centerline or segmentation.",
                      mandatory=False,
                      deprecated_by='-s')
    parser.add_option(name="-dest",
                      type_value="image_nifti",
                      description="Spinal cord centerline (or segmentation) of a destination image (which could be "
                                  "straight or curved). An "
                                  "algorithm scales the length of the input centerline to match that of the "
                                  "destination centerline. If using -ldisc_input and -ldisc_dest with this parameter, "
                                  "instead of linear scaling, the source centerline will be non-linearly matched so "
                                  "that the inter-vertebral discs of the input image will match that of the "
                                  "destination image. This feature is particularly useful for registering to a "
                                  "template while accounting for disc alignment.",
                      mandatory=False,
                      example="centerline.nii.gz")
    parser.add_option(name="-ldisc_input",
                      type_value="image_nifti",
                      description="Labels located at the posterior edge of the intervertebral discs, for the input "
                                  "image (-i). All disc covering the region of interest should be provided. E.g., if "
                                  "you are interested in levels C2 to C7, then you should provide disc labels 2,3,4,5,"
                                  "6,7). More details about label creation at "
                                  "http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/.\n"  # TODO (Julien) update this link
                                  "This option must be used with the -ldisc_dest parameter.",
                      mandatory=False,
                      example="ldisc_input.nii.gz")
    parser.add_option(name="-ldisc_dest",
                      type_value="image_nifti",
                      description="Labels located at the posterior edge of the intervertebral discs, for the "
                                  "destination file (-dest). The same comments as in -ldisc_input apply.\n"
                                  "This option must be used with the -ldisc_input parameter.",
                      mandatory=False,
                      example="ldisc_dest.nii.gz")
    parser.add_option(name="-disable-straight2curved",
                      type_value=None,
                      description="Disable straight to curved transformation computation, in case you do not need the "
                                  "output warping field straight-->curve (faster).",
                      mandatory=False)
    parser.add_option(name="-disable-curved2straight",
                      type_value=None,
                      description="Disable curved to straight transformation computation, in case you do not need the "
                                  "output warping field curve-->straight (faster).",
                      mandatory=False)
    parser.add_option(name="-speed_factor",
                      type_value='float',
                      description='Acceleration factor for the calculation of the straightening warping field.'
                                  ' This speed factor enables an intermediate resampling to a lower resolution, which '
                                  'decreases the computational time at the cost of lower accuracy.'
                                  ' A speed factor of 2 means that the input image will be downsampled by a factor 2 '
                                  'before calculating the straightening warping field. For example, a 1x1x1 mm^3 image '
                                  'will be downsampled to 2x2x2 mm3, providing a speed factor of approximately 8.'
                                  ' Note that accelerating the straightening process reduces the precision of the '
                                  'algorithm, and induces undesirable edges effects. Default=1 (no downsampling).',
                      mandatory=False,
                      default_value=1)
    parser.add_option(name="-xy_size",
                      type_value='float',
                      description='Size of the output FOV in the RL/AP plane, in mm. The resolution of the destination '
                                  'image is the same as that of the source image (-i).\n',
                      mandatory=False,
                      default_value=35.0)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="straightened file",
                      mandatory=False,
                      default_value='',
                      example="data_straight.nii.gz")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder (all outputs will go there).",
                      mandatory=False,
                      default_value='')

    parser.add_option(name='-centerline-algo',
                      type_value='multiple_choice',
                      description='Algorithm for centerline fitting.',
                      mandatory=False,
                      example=['bspline', 'linear', 'nurbs'],
                      default_value='nurbs')
    parser.add_option(name='-centerline-smooth',
                      type_value='int',
                      description='Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}.',
                      mandatory=False,
                      default_value=10)

    parser.add_option(name="-param",
                      type_value=[[','], 'str'],
                      description="Parameters for spinal cord straightening. Separate arguments with ','."
                                  "\nprecision: [1.0,inf[. Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with increased computational time. Not taken into account with hanning fitting method. Default=2"
                                  "\nthreshold_distance: [0.0,inf[. Threshold at which voxels are not considered into displacement. Increase this threshold if the image is blackout around the spinal cord too much. Default=10"
                                  "\naccuracy_results: {0, 1} Disable/Enable computation of accuracy results after straightening. Default=0"
                                  "\ntemplate_orientation: {0, 1} Disable/Enable orientation of the straight image to be the same as the template. Default=0",
                      mandatory=False)

    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="Final interpolation.",
                      mandatory=False,
                      example=["nn", "linear", "spline"],
                      default_value="spline")
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-qc-dataset',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the dataset the '
                                  'process was run on',
                      )
    parser.add_option(name='-qc-subject',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the subject the '
                                  'process was run on',
                      )
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing, 1: basic, 2: extended.",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser


# MAIN
# ==========================================================================================
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)

    # assigning variables to arguments
    input_filename = arguments["-i"]
    centerline_file = arguments["-s"]

    sc_straight = SpinalCordStraightener(input_filename, centerline_file)

    if "-dest" in arguments:
        sc_straight.use_straight_reference = True
        sc_straight.centerline_reference_filename = str(arguments["-dest"])

    if "-ldisc_input" in arguments:
        if not sc_straight.use_straight_reference:
            sct.printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_input_filename = str(arguments["-ldisc_input"])
            sc_straight.precision = 4.0
    if "-ldisc_dest" in arguments:
        if not sc_straight.use_straight_reference:
            sct.printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_ref_filename = str(arguments["-ldisc_dest"])
            sc_straight.precision = 4.0

    # Handling optional arguments
    if "-r" in arguments:
        sc_straight.remove_temp_files = int(arguments["-r"])
    if "-x" in arguments:
        sc_straight.interpolation_warp = str(arguments["-x"])
    if "-o" in arguments:
        sc_straight.output_filename = str(arguments["-o"])
    if '-ofolder' in arguments:
        sc_straight.path_output = arguments['-ofolder']
    else:
        sc_straight.path_output = './'
    path_qc = arguments.get("-qc", None)
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    sc_straight.verbose = verbose

    # if "-cpu-nb" in arguments:
    #     sc_straight.cpu_number = int(arguments["-cpu-nb"])

    if '-disable-straight2curved' in arguments:
        sc_straight.straight2curved = False
    if '-disable-curved2straight' in arguments:
        sc_straight.curved2straight = False

    if '-speed_factor' in arguments:
        sc_straight.speed_factor = arguments['-speed_factor']

    if '-xy_size' in arguments:
        sc_straight.xy_size = arguments['-xy_size']

    sc_straight.param_centerline = ParamCenterline(
        algo_fitting=arguments['-centerline-algo'],
        degree=arguments['-centerline-degree'],
        smooth=arguments['-centerline-smooth'])
    if "-param" in arguments:
        params_user = arguments['-param']
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'precision':
                sc_straight.precision = float(param_split[1])
            if param_split[0] == 'threshold_distance':
                sc_straight.threshold_distance = float(param_split[1])
            if param_split[0] == 'accuracy_results':
                sc_straight.accuracy_results = int(param_split[1])
            if param_split[0] == 'template_orientation':
                sc_straight.template_orientation = int(param_split[1])

    fname_straight = sc_straight.straighten()

    sct.printv("\nFinished! Elapsed time: {} s".format(sc_straight.elapsed_time), verbose)

    # Generate QC report
    if path_qc is not None:
        path_qc = os.path.abspath(path_qc)
        qc_dataset = arguments.get("-qc-dataset", None)
        qc_subject = arguments.get("-qc-subject", None)
        generate_qc(fname_straight, args=args, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process=os.path.basename(__file__.strip('.py')))

    sct.display_viewer_syntax([fname_straight], verbose=verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
