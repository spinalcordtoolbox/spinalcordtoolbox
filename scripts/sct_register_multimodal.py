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

from __future__ import division, absolute_import

import sys, os, time
import numpy as np

from spinalcordtoolbox.reports.qc import generate_qc

import sct_utils as sct
from msct_parser import Parser
from msct_register import Paramreg, ParamregMultiStep, register_wrapper


def get_parser(paramregmulti=None):
    # Initialize the parser

    if paramregmulti is None:
        step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5',
                         slicewise='0', dof='Tx_Ty_Tz_Rx_Ry_Rz')  # only used to put src into dest space
        step1 = Paramreg(step='1', type='im')
        paramregmulti = ParamregMultiStep([step0, step1])

    parser = Parser(__file__)
    parser.usage.set_description('This program co-registers two 3D volumes. The deformation is non-rigid and is '
                                 'constrained along Z direction (i.e., axial plane). Hence, this function assumes '
                                 'that orientation of the destination image is axial (RPI). If you need to register '
                                 'two volumes with large deformations and/or different contrasts, it is recommended to '
                                 'input spinal cord segmentations (binary mask) in order to achieve maximum robustness.'
                                 ' The program outputs a warping field that can be used to register other images to the'
                                 ' destination image. To apply the warping field to another image, use '
                                 'sct_apply_transfo')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image source.",
                      mandatory=True,
                      example="src.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="Image destination.",
                      mandatory=True,
                      example="dest.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation source.",
                      mandatory=False,
                      example="src_seg.nii.gz")
    parser.add_option(name="-dseg",
                      type_value="file",
                      description="Segmentation destination.",
                      mandatory=False,
                      example="dest_seg.nii.gz")
    parser.add_option(name="-ilabel",
                      type_value="file",
                      description="Labels source.",
                      mandatory=False)
    parser.add_option(name="-dlabel",
                      type_value="file",
                      description="Labels destination.",
                      mandatory=False)
    parser.add_option(name='-initwarp',
                      type_value='file',
                      description='Initial warping field to apply to the source image.',
                      mandatory=False)
    parser.add_option(name='-initwarpinv',
                      type_value='file',
                      description='Initial inverse warping field to apply to the destination image (only use if you wish to generate the dest->src warping field).',
                      mandatory=False)
    parser.add_option(name="-m",
                      type_value="file",
                      description="Mask that can be created with sct_create_mask to improve accuracy over region of interest. "
                                  "This mask will be used on the destination image.",
                      mandatory=False,
                      example="mask.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of output file.",
                      mandatory=False,
                      example="src_reg.nii.gz")
    parser.add_option(name='-owarp',
                      type_value="file_output",
                      description="Name of output forward warping field.",
                      mandatory=False)
    parser.add_option(name="-param",
                      type_value=[[':'], 'str'],
                      description="Parameters for registration. Separate arguments with \",\". Separate steps with \":\".\n"
                                  "step: <int> Step number (starts at 1, except for type=label).\n"
                                  "type: {im, seg, imseg, label} type of data used for registration. Use type=label only at step=0.\n"
                                  "algo: Default=" + paramregmulti.steps['1'].algo + "\n"
                                                                                "  translation: translation in X-Y plane (2dof)\n"
                                                                                "  rigid: translation + rotation in X-Y plane (4dof)\n"
                                                                                "  affine: translation + rotation + scaling in X-Y plane (6dof)\n"
                                                                                "  syn: non-linear symmetric normalization\n"
                                                                                "  bsplinesyn: syn regularized with b-splines\n"
                                                                                "  slicereg: regularized translations (see: goo.gl/Sj3ZeU)\n"
                                                                                "  centermass: slicewise center of mass alignment (seg only).\n"
                                                                                "  centermassrot: slicewise center of mass and rotation alignment using method specified in 'rot_method'\n"
                                                                                "  columnwise: R-L scaling followed by A-P columnwise alignment (seg only).\n"
                                                                                "slicewise: <int> Slice-by-slice 2d transformation. Default=" +
                                  paramregmulti.steps['1'].slicewise + "\n"
                                                                  "metric: {CC,MI,MeanSquares}. Default=" +
                                  paramregmulti.steps['1'].metric + "\n"
                                                               "iter: <int> Number of iterations. Default=" +
                                  paramregmulti.steps['1'].iter + "\n"
                                                             "shrink: <int> Shrink factor (only for syn/bsplinesyn). Default=" +
                                  paramregmulti.steps['1'].shrink + "\n"
                                                               "smooth: <int> Smooth factor (in mm). Note: if algo={centermassrot,columnwise} the smoothing kernel is: SxSx0. Otherwise it is SxSxS. Default=" +
                                  paramregmulti.steps['1'].smooth + "\n"
                                                               "laplacian: <int> Laplacian filter. Default=" +
                                  paramregmulti.steps['1'].laplacian + "\n"
                                                                  "gradStep: <float> Gradient step. Default=" +
                                  paramregmulti.steps['1'].gradStep + "\n"
                                                                 "deformation: ?x?x?: Restrict deformation (for ANTs algo). Replace ? by 0 (no deformation) or 1 (deformation). Default=" +
                                  paramregmulti.steps['1'].deformation + "\n"
                                                                    "init: Initial translation alignment based on:\n"
                                                                    "  geometric: Geometric center of images\n"
                                                                    "  centermass: Center of mass of images\n"
                                                                    "  origin: Physical origin of images\n"
                                                                    "poly: <int> Polynomial degree of regularization (only for algo=slicereg). Default=" +
                                  paramregmulti.steps['1'].poly + "\n"
                                                                    "filter_size: <float> Filter size for regularization (only for algo=centermassrot). Default=" +
                                  str(paramregmulti.steps['1'].filter_size) + "\n"
                                                             "smoothWarpXY: <int> Smooth XY warping field (only for algo=columnwize). Default=" +
                                  paramregmulti.steps['1'].smoothWarpXY + "\n"
                                                                     "pca_eigenratio_th: <int> Min ratio between the two eigenvalues for PCA-based angular adjustment (only for algo=centermassrot and rot_method=pca). Default=" +
                                  paramregmulti.steps['1'].pca_eigenratio_th + "\n"
                                                                          "dof: <str> Degree of freedom for type=label. Separate with '_'. Default=" +
                                  paramregmulti.steps['0'].dof + "\n" +
                                  paramregmulti.steps['1'].rot_method + "\n"
                                                                    "rot_method {pca, hog, pcahog}: rotation method to be used with algo=centermassrot. pca: approximate cord segmentation by an ellipse and finds it orientation using PCA's eigenvectors; hog: finds the orientation using the symmetry of the image; pcahog: tries method pca and if it fails, uses method hog. If using hog or pcahog, type should be set to imseg.",
                      mandatory=False,
                      example="step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,metric=MI,iter=5,shrink=2")
    parser.add_option(name="-identity",
                      type_value="multiple_choice",
                      description="just put source into destination (no optimization).",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-z",
                      type_value="int",
                      description="""size of z-padding to enable deformation at edges when using SyN.""",
                      mandatory=False,
                      default_value=Param().padding)
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="""Final interpolation.""",
                      mandatory=False,
                      default_value='linear',
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      example='reg_results/')
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-qc-dataset',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
                      )
    parser.add_option(name='-qc-subject',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the subject the process was run on',
                      )
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])

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

    arguments = parser.parse(args)

    # get arguments
    fname_src = arguments['-i']
    fname_dest = arguments['-d']
    if '-iseg' in arguments:
        fname_src_seg = arguments['-iseg']
    if '-dseg' in arguments:
        fname_dest_seg = arguments['-dseg']
    if '-ilabel' in arguments:
        fname_src_label = arguments['-ilabel']
    if '-dlabel' in arguments:
        fname_dest_label = arguments['-dlabel']
    if '-o' in arguments:
        fname_output = arguments['-o']
    if '-ofolder' in arguments:
        path_out = arguments['-ofolder']
    if '-owarp' in arguments:
        fname_output_warp = arguments['-owarp']
    else:
        fname_output_warp = ''
    if '-initwarp' in arguments:
        fname_initwarp = os.path.abspath(arguments['-initwarp'])
    else:
        fname_initwarp = ''
    if '-initwarpinv' in arguments:
        fname_initwarpinv = os.path.abspath(arguments['-initwarpinv'])
    else:
        fname_initwarpinv = ''
    if '-m' in arguments:
        fname_mask = arguments['-m']
    else:
        fname_mask = ''
    padding = arguments['-z']
    if "-param" in arguments:
        paramregmulti_user = arguments['-param']
        # update registration parameters
        for paramStep in paramregmulti_user:
            paramregmulti.addStep(paramStep)
    path_qc = arguments.get("-qc", None)
    qc_dataset = arguments.get("-qc-dataset", None)
    qc_subject = arguments.get("-qc-subject", None)

    identity = int(arguments['-identity'])
    interp = arguments['-x']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # sct.printv(arguments)
    sct.printv('\nInput parameters:')
    sct.printv('  Source .............. ' + fname_src)
    sct.printv('  Destination ......... ' + fname_dest)
    sct.printv('  Init transfo ........ ' + fname_initwarp)
    sct.printv('  Mask ................ ' + fname_mask)
    sct.printv('  Output name ......... ' + fname_output)
    # sct.printv('  Algorithm ........... '+paramregmulti.algo)
    # sct.printv('  Number of iterations  '+paramregmulti.iter)
    # sct.printv('  Metric .............. '+paramregmulti.metric)
    sct.printv('  Remove temp files ... ' + str(remove_temp_files))
    sct.printv('  Verbose ............. ' + str(verbose))

    # update param
    param.verbose = verbose
    param.padding = padding
    param.fname_mask = fname_mask
    param.remove_temp_files = remove_temp_files

    # Get if input is 3D
    sct.printv('\nCheck if input data are 3D...', verbose)
    sct.check_dim(fname_src, dim_lst=[3])
    sct.check_dim(fname_dest, dim_lst=[3])

    # Check if user selected type=seg, but did not input segmentation data
    if 'paramregmulti_user' in locals():
        if True in ['type=seg' in paramregmulti_user[i] for i in range(len(paramregmulti_user))]:
            if fname_src_seg == '' or fname_dest_seg == '':
                sct.printv('\nERROR: if you select type=seg you must specify -iseg and -dseg flags.\n', 1, 'error')

    # Put source into destination space using header (no estimation -- purely based on header)
    # TODO: Check if necessary to do that
    # TODO: use that as step=0
    # sct.printv('\nPut source into destination space using header...', verbose)
    # sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[dest_pad.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o
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
    sct.printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', verbose)

    if path_qc is not None:
        if fname_dest_seg:
            generate_qc(fname_src2dest, fname_in2=fname_dest, fname_seg=fname_dest_seg, args=args,
                        path_qc=os.path.abspath(path_qc), dataset=qc_dataset, subject=qc_subject,
                        process='sct_register_multimodal')
        else:
            sct.printv('WARNING: Cannot generate QC because it requires destination segmentation.', 1, 'warning')

    if generate_warpinv:
        sct.display_viewer_syntax([fname_src, fname_dest2src], verbose=verbose)
    sct.display_viewer_syntax([fname_dest, fname_src2dest], verbose=verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
