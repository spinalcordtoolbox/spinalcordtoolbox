#!/usr/bin/env python
#########################################################################################
# Register a volume (e.g., EPI from fMRI or DTI scan) to an anatomical image.
#
# See Usage() below for more information.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add flag -owarpinv
# TODO: if user specified -param, then ignore the default paramreg
# TODO: check syn with shrink=4
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: testing script for all cases
# TODO: add following feature:
# -r of isct_antsRegistration at the initial step (step 0).
# -r [' dest ',' src ',0] --> align the geometric center of the two images
# -r [' dest ',' src ',1] --> align the maximum intensities of the two images I use that quite often...
# TODO: output reg for ants2d and centermass (2016-02-25)

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherwise inverse warping field is not output

# TODO: make three possibilities:
# - one-step registration, using only image registration (by sliceReg or antsRegistration)
# - two-step registration, using first segmentation-based registration (based on sliceReg or antsRegistration) and
# second the image registration (and allow the choice of algo, metric, etc.)
# - two-step registration, using only segmentation-based registration

import sys
import os
import time
import argparse

import numpy as np

from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.registration.register import Paramreg, ParamregMultiStep
from spinalcordtoolbox.utils.shell import Metavar, SmartFormatter, ActionCreateFolder, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv
from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.image import check_dim

from sct_register_to_template import register_wrapper


def get_parser(paramregmulti=None):
    # Initialize the parser

    if paramregmulti is None:
        step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5',
                         slicewise='0', dof='Tx_Ty_Tz_Rx_Ry_Rz')  # only used to put src into dest space
        step1 = Paramreg(step='1', type='im')
        paramregmulti = ParamregMultiStep([step0, step1])

    parser = argparse.ArgumentParser(
        description="This program co-registers two 3D volumes. The deformation is non-rigid and is constrained along "
                    "Z direction (i.e., axial plane). Hence, this function assumes that orientation of the destination "
                    "image is axial (RPI). If you need to register two volumes with large deformations and/or "
                    "different contrasts, it is recommended to input spinal cord segmentations (binary mask) in order "
                    "to achieve maximum robustness. The program outputs a warping field that can be used to register "
                    "other images to the destination image. To apply the warping field to another image, use "
                    "'sct_apply_transfo'\n"
                    "\n"
                    "Tips:\n"
                    " - For a registration step using segmentations, use the MeanSquares metric. Also, simple "
                    "algorithm will be very efficient, for example centermass as a 'preregistration'.\n"
                    " - For a registration step using images of different contrast, use the Mutual Information (MI) "
                    "metric.\n"
                    " - Combine the steps by increasing the complexity of the transformation performed in each step, "
                    "for example: -param step=1,type=seg,algo=slicereg,metric=MeanSquares:"
                    "step=2,type=seg,algo=affine,metric=MeanSquares,gradStep=0.2:"
                    "step=3,type=im,algo=syn,metric=MI,iter=5,shrink=2\n"
                    " - When image contrast is low, a good option is to perform registration only based on the image "
                    "segmentation, i.e. using type=seg\n"
                    " - Columnwise algorithm needs to be applied after a translation and rotation such as centermassrot "
                    "algorithm. For example: -param step=1,type=seg,algo=centermassrot,metric=MeanSquares:"
                    "step=2,type=seg,algo=columnwise,metric=MeanSquares",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Image source. Example: src.nii.gz"
    )
    mandatory.add_argument(
        '-d',
        metavar=Metavar.file,
        required=True,
        help="Image destination. Example: dest.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-iseg',
        metavar=Metavar.file,
        help="Segmentation source. Example: src_seg.nii.gz"
    )
    optional.add_argument(
        '-dseg',
        metavar=Metavar.file,
        help="Segmentation destination. Example: dest_seg.nii.gz"
    )
    optional.add_argument(
        '-ilabel',
        metavar=Metavar.file,
        help="Labels source."
    )
    optional.add_argument(
        '-dlabel',
        metavar=Metavar.file,
        help="Labels destination."
    )
    optional.add_argument(
        '-initwarp',
        metavar=Metavar.file,
        help="Initial warping field to apply to the source image."
    )
    optional.add_argument(
        '-initwarpinv',
        metavar=Metavar.file,
        help="Initial inverse warping field to apply to the destination image (only use if you wish to generate the "
             "dest->src warping field)"
    )
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        help="Mask that can be created with sct_create_mask to improve accuracy over region of interest. This mask "
             "will be used on the destination image. Example: mask.nii.gz"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="Name of output file. Example: src_reg.nii.gz"
    )
    optional.add_argument(
        '-owarp',
        metavar=Metavar.file,
        help="Name of output forward warping field."
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(':', str),
        help=(f"R|Parameters for registration. Separate arguments with \",\". Separate steps with \":\".\n"
              f"Example: step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,metric=MI,iter=5,"
              f"shrink=2\n"
              f"  - step: <int> Step number (starts at 1, except for type=label).\n"
              f"  - type: {{im, seg, imseg, label}} type of data used for registration. Use type=label only at "
              f"step=0.\n"
              f"  - algo: The algorithm used to compute the transformation. Default={paramregmulti.steps['1'].algo}\n"
              f"    * translation: translation in X-Y plane (2dof)\n"
              f"    * rigid: translation + rotation in X-Y plane (4dof)\n"
              f"    * affine: translation + rotation + scaling in X-Y plane (6dof)\n"
              f"    * syn: non-linear symmetric normalization\n"
              f"    * bsplinesyn: syn regularized with b-splines\n"
              f"    * slicereg: regularized translations (see: goo.gl/Sj3ZeU)\n"
              f"    * centermass: slicewise center of mass alignment (seg only).\n"
              f"    * centermassrot: slicewise center of mass and rotation alignment using method specified in "
              f"'rot_method'\n"
              f"    * columnwise: R-L scaling followed by A-P columnwise alignment (seg only).\n"
              f"  - slicewise: <int> Slice-by-slice 2d transformation. "
              f"Default={paramregmulti.steps['1'].slicewise}.\n"
              f"  - metric: {{CC, MI, MeanSquares}}. Default={paramregmulti.steps['1'].metric}.\n"
              f"    * CC: The cross correlation metric compares the images based on their intensities but with a small "
              f"normalization. It can be used with images with the same contrast (for ex. T2-w with T2-w). In this "
              f"case it is very efficient but the computation time can be very long.\n"
              f"    * MI: the mutual information metric compares the images based on their entropy, therefore the "
              f"images need to be big enough to have enough information. It works well for images with different "
              f"contrasts (for example T2-w with T1-w) but not on segmentations.\n"
              f"    * MeanSquares: The mean squares metric compares the images based on their intensities. It can be "
              f"used only with images that have exactly the same contrast (with the same intensity range) or with "
              f"segmentations.\n"
              f"  - iter: <int> Number of iterations. Default={paramregmulti.steps['1'].iter}.\n"
              f"  - shrink: <int> Shrink factor. A shrink factor of 2 will down sample the images by a factor of 2 to "
              f"do the registration, and thus allow bigger deformations (and be faster to compute). It is usually "
              f"combined with a smoothing. (only for syn/bsplinesyn). Default={paramregmulti.steps['1'].shrink}.\n"
              f"  - smooth: <int> Smooth factor (in mm). Note: if algo={{centermassrot,columnwise}} the smoothing "
              f"kernel is: SxSx0. Otherwise it is SxSxS. Default={paramregmulti.steps['1'].smooth}.\n"
              f"  - laplacian: <int> Laplace filter using Gaussian second derivatives, applied before registration. "
              f"The input number correspond to the standard deviation of the Gaussian filter. "
              f"Default={paramregmulti.steps['1'].laplacian}.\n"
              f"  - gradStep: <float> The gradient step used by the function opitmizer. A small gradient step can lead "
              f"to a more accurate registration but will take longer to compute, with the risk to not reach "
              f"convergence. A bigger gradient step will make the registration faster but the result can be far from "
              f"an optimum. Default={paramregmulti.steps['1'].gradStep}.\n"
              f"  - deformation: ?x?x?: Restrict deformation (for ANTs algo). Replace ? by 0 (no deformation) or 1 "
              f"(deformation). Default={paramregmulti.steps['1'].deformation}.\n"
              f"  - init: Initial translation alignment based on:\n"
              f"    * geometric: Geometric center of images\n"
              f"    * centermass: Center of mass of images\n"
              f"    * origin: Physical origin of images\n"
              f"  - poly: <int> Polynomial degree of regularization (only for algo=slicereg). "
              f"Default={paramregmulti.steps['1'].poly}.\n"
              f"  - filter_size: <float> Filter size for regularization (only for algo=centermassrot). "
              f"Default={paramregmulti.steps['1'].filter_size}.\n"
              f"  - smoothWarpXY: <int> Smooth XY warping field (only for algo=columnwize). "
              f"Default={paramregmulti.steps['1'].smoothWarpXY}.\n"
              f"  - pca_eigenratio_th: <int> Min ratio between the two eigenvalues for PCA-based angular adjustment "
              f"(only for algo=centermassrot and rot_method=pca). "
              f"Default={paramregmulti.steps['1'].pca_eigenratio_th}.\n"
              f"  - dof: <str> Degree of freedom for type=label. Separate with '_'. T stands for translation and R "
              f"stands for rotation, x, y, and z indicating the direction. For example, Tx_Ty_Tz_Rx_Ry_Rz would allow "
              f"translation on x, y and z axes and rotation on x, y and z axes. "
              f"Default={paramregmulti.steps['0'].dof}.\n"
              f"  - rot_method {{pca, hog, pcahog}}: rotation method to be used with algo=centermassrot. If using hog "
              f"or pcahog, type should be set to imseg. Default={paramregmulti.steps['1'].rot_method}\n"
              f"    * pca: approximate cord segmentation by an ellipse and finds it orientation using PCA's "
              f"eigenvectors\n"
              f"    * hog: finds the orientation using the symmetry of the image\n"
              f"    * pcahog: tries method pca and if it fails, uses method hog.\n")
    )
    optional.add_argument(
        '-identity',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Just put source into destination (no optimization)."
    )
    optional.add_argument(
        '-z',
        metavar=Metavar.int,
        type=int,
        default=Param().padding,
        help="Size of z-padding to enable deformation at edges when using SyN."
    )
    optional.add_argument(
        '-x',
        choices=['nn', 'linear', 'spline'],
        default='linear',
        help="Final interpolation."
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="Output folder. Example: reg_results/"
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
    optional.add_argument(
        '-r',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to remove temporary files. 0 = no, 1 = yes"
    )
    optional.add_argument(
        '-v',
        choices=['0', '1', '2'],
        default='1',
        help="Verbose. 0: nothing, 1: basic, 2: extended."
    )
    return parser


# DEFAULT PARAMETERS

class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.outSuffix = "_reg"
        self.padding = 5
        self.remove_temp_files = 1


# MAIN
# ==========================================================================================
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # initialize parameters
    param = Param()

    # Initialization
    fname_output = ''
    path_out = ''
    fname_src_seg = ''
    fname_dest_seg = ''
    fname_src_label = ''
    fname_dest_label = ''
    generate_warpinv = 1

    start_time = time.time()

    # get default registration parameters
    # step1 = Paramreg(step='1', type='im', algo='syn', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5')
    step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5',
                     slicewise='0', dof='Tx_Ty_Tz_Rx_Ry_Rz')  # only used to put src into dest space
    step1 = Paramreg(step='1', type='im')
    paramregmulti = ParamregMultiStep([step0, step1])

    parser = get_parser(paramregmulti=paramregmulti)

    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # get arguments
    fname_src = arguments.i
    fname_dest = arguments.d
    if arguments.iseg is not None:
        fname_src_seg = arguments.iseg
    if arguments.dseg is not None:
        fname_dest_seg = arguments.dseg
    if arguments.ilabel is not None:
        fname_src_label = arguments.ilabel
    if arguments.dlabel is not None:
        fname_dest_label = arguments.dlabel
    if arguments.o is not None:
        fname_output = arguments.o
    if arguments.ofolder is not None:
        path_out = arguments.ofolder
    if arguments.owarp is not None:
        fname_output_warp = arguments.owarp
    else:
        fname_output_warp = ''
    if arguments.initwarp is not None:
        fname_initwarp = os.path.abspath(arguments.initwarp)
    else:
        fname_initwarp = ''
    if arguments.initwarpinv is not None:
        fname_initwarpinv = os.path.abspath(arguments.initwarpinv)
    else:
        fname_initwarpinv = ''
    if arguments.m is not None:
        fname_mask = arguments.m
    else:
        fname_mask = ''
    padding = arguments.z
    if arguments.param is not None:
        paramregmulti_user = arguments.param
        # update registration parameters
        for paramStep in paramregmulti_user:
            paramregmulti.addStep(paramStep)
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject

    identity = arguments.identity
    interp = arguments.x
    remove_temp_files = arguments.r
    verbose = int(arguments.v)
    init_sct(log_level=verbose, update=True)  # Update log level

    # printv(arguments)
    printv('\nInput parameters:')
    printv('  Source .............. ' + fname_src)
    printv('  Destination ......... ' + fname_dest)
    printv('  Init transfo ........ ' + fname_initwarp)
    printv('  Mask ................ ' + fname_mask)
    printv('  Output name ......... ' + fname_output)
    # printv('  Algorithm ........... '+paramregmulti.algo)
    # printv('  Number of iterations  '+paramregmulti.iter)
    # printv('  Metric .............. '+paramregmulti.metric)
    printv('  Remove temp files ... ' + str(remove_temp_files))
    printv('  Verbose ............. ' + str(verbose))

    # update param
    param.verbose = verbose
    param.padding = padding
    param.fname_mask = fname_mask
    param.remove_temp_files = remove_temp_files

    # Get if input is 3D
    printv('\nCheck if input data are 3D...', verbose)
    check_dim(fname_src, dim_lst=[3])
    check_dim(fname_dest, dim_lst=[3])

    # Check if user selected type=seg, but did not input segmentation data
    if 'paramregmulti_user' in locals():
        if True in ['type=seg' in paramregmulti_user[i] for i in range(len(paramregmulti_user))]:
            if fname_src_seg == '' or fname_dest_seg == '':
                printv('\nERROR: if you select type=seg you must specify -iseg and -dseg flags.\n', 1, 'error')

    # Put source into destination space using header (no estimation -- purely based on header)
    # TODO: Check if necessary to do that
    # TODO: use that as step=0
    # printv('\nPut source into destination space using header...', verbose)
    # run_proc('isct_antsRegistration -d 3 -t Translation[0] -m MI[dest_pad.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o
    # [regAffine,src_regAffine.nii] -n BSpline[3]', verbose)
    # if segmentation, also do it for seg

    fname_src2dest, fname_dest2src, _, _ = \
        register_wrapper(fname_src, fname_dest, param, paramregmulti, fname_src_seg=fname_src_seg,
                         fname_dest_seg=fname_dest_seg, fname_src_label=fname_src_label,
                         fname_dest_label=fname_dest_label, fname_mask=fname_mask, fname_initwarp=fname_initwarp,
                         fname_initwarpinv=fname_initwarpinv, identity=identity, interp=interp,
                         fname_output=fname_output,
                         fname_output_warp=fname_output_warp,
                         path_out=path_out)

    # display elapsed time
    elapsed_time = time.time() - start_time
    printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', verbose)

    if path_qc is not None:
        if fname_dest_seg:
            generate_qc(fname_src2dest, fname_in2=fname_dest, fname_seg=fname_dest_seg, args=args,
                        path_qc=os.path.abspath(path_qc), dataset=qc_dataset, subject=qc_subject,
                        process='sct_register_multimodal')
        else:
            printv('WARNING: Cannot generate QC because it requires destination segmentation.', 1, 'warning')

    if generate_warpinv:
        display_viewer_syntax([fname_src, fname_dest2src], verbose=verbose)
    display_viewer_syntax([fname_dest, fname_src2dest], verbose=verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    init_sct()
    # call main function
    main()
