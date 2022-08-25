#!/usr/bin/env python
#########################################################################################
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Julien Cohen-Adad, Augustin Roux
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: for -ref subject, crop data, otherwise registration is too long
# TODO: testing script for all cases
# TODO: enable vertebral alignment with -ref subject

import sys
import os
import time
from typing import Sequence

import numpy as np

from spinalcordtoolbox.registration.core import register_wrapper
from spinalcordtoolbox.registration.algorithms import Paramreg, ParamregMultiStep
from spinalcordtoolbox.registration.labeling import (add_orthogonal_label, check_labels,
                                                     project_labels_on_spinalcord, resample_labels)
from spinalcordtoolbox.registration.landmarks import register_landmarks

from spinalcordtoolbox.metadata import get_file_label
from spinalcordtoolbox.image import Image, add_suffix, generate_output_file
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.resampling import resample_file
from spinalcordtoolbox.math import dilate, binarize
from spinalcordtoolbox.utils.fs import (copy, extract_fname, check_file_exist, rmtree,
                                        cache_save, cache_signature, cache_valid, tmp_create)
from spinalcordtoolbox.utils.shell import (SCTArgumentParser, ActionCreateFolder, Metavar, list_type,
                                           printv, display_viewer_syntax)
from spinalcordtoolbox.utils.sys import set_loglevel, init_sct, run_proc
from spinalcordtoolbox import __data_dir__
import spinalcordtoolbox.image as msct_image
import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.scripts import sct_apply_transfo


class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1  # remove temporary files
        self.fname_mask = ''  # this field is needed in the function register@sct_register_multimodal
        self.padding = 10  # this field is needed in the function register@sct_register_multimodal
        self.verbose = 1  # verbose
        self.path_template = os.path.join(__data_dir__, 'PAM50')
        self.path_qc = None
        self.zsubsample = '0.25'
        self.rot_src = None
        self.rot_dest = None


# get default parameters
# Note: step0 is used as pre-registration
step0 = Paramreg(step='0', type='label', dof='Tx_Ty_Tz_Sz')  # if ref=template, we only need translations and z-scaling because the cord is already straight
step1 = Paramreg(step='1', type='imseg', algo='centermassrot', rot_method='pcahog')
step2 = Paramreg(step='2', type='seg', algo='bsplinesyn', metric='MeanSquares', iter='3', smooth='1', slicewise='0')
paramregmulti = ParamregMultiStep([step0, step1, step2])


# PARSER
# ==========================================================================================
def get_parser():
    param = Param()
    parser = SCTArgumentParser(
        description=(
            "Register an anatomical image to the spinal cord MRI template (default: PAM50).\n"
            "\n"
            "The registration process includes three main registration steps:\n"
            "  1. straightening of the image using the spinal cord segmentation (see sct_straighten_spinalcord for "
            "details);\n"
            "  2. vertebral alignment between the image and the template, using labels along the spine;\n"
            "  3. iterative slice-wise non-linear registration (see sct_register_multimodal for details)\n"
            "\n"
            "To register a subject to the template, try the default command:\n"
            "  sct_register_to_template -i data.nii.gz -s data_seg.nii.gz -l data_labels.nii.gz\n"
            "\n"
            "If this default command does not produce satisfactory results, the '-param' "
            "argument should be tweaked according to the tips given here:\n"
            "  https://spinalcordtoolbox.com/en/latest/user_section/command-line.html#sct-register-multimodal\n"
            "\n"
            "The default registration method brings the subject image to the template, which can be problematic with "
            "highly non-isotropic images as it would induce large interpolation errors during the straightening "
            "procedure. Although the default method is recommended, you may want to register the template to the "
            "subject (instead of the subject to the template) by skipping the straightening procedure. To do so, use "
            "the parameter '-ref subject'. Example below:\n"
            "  sct_register_to_template -i data.nii.gz -s data_seg.nii.gz -l data_labels.nii.gz -ref subject -param "
            "step=1,type=seg,algo=centermassrot,smooth=0:step=2,type=seg,algo=columnwise,smooth=0,smoothWarpXY=2\n"
            "\n"
            "Vertebral alignment (step 2) consists in aligning the vertebrae between the subject and the template. "
            "Two types of labels are possible:\n"
            "  - Vertebrae mid-body labels, created at the center of the spinal cord using the parameter '-l';\n"
            "  - Posterior edge of the intervertebral discs, using the parameter '-ldisc'.\n"
            "\n"
            "If only one label is provided, a simple translation will be applied between the subject label and the "
            "template label. No scaling will be performed. \n"
            "\n"
            "If two labels are provided, a linear transformation (translation + rotation + superior-inferior linear "
            "scaling) will be applied. The strategy here is to define labels that cover the region of interest. For "
            "example, if you are interested in studying C2 to C6 levels, then provide one label at C2 and another at "
            "C6. However, note that if the two labels are very far apart (e.g. C2 and T12), there might be a "
            "mis-alignment of discs because a subject's intervertebral discs distance might differ from that of the "
            "template.\n"
            "\n"
            "If more than two labels are used, a non-linear registration will be applied to align the each "
            "intervertebral disc between the subject and the template, as described in "
            "sct_straighten_spinalcord. This the most accurate method, however it has some serious caveats: \n"
            "  - This feature is not compatible with the parameter '-ref subject', where only a rigid registration is performed.\n"
            "  - Due to the non-linear registration in the S-I direction, the warping field will be cropped above the top label and below the bottom label. Applying this warping field will result in a strange-looking registered image that has the same value above the top label and below the bottom label. But if you are not interested in these regions, you do not need to worry about it.\n"
            "\n"
            "We recommend starting with 2 labels, then trying the other "
            "options on a case-by-case basis depending on your data.\n"
            "\n"
            "More information about label creation can be found at "
            "https://spinalcordtoolbox.com/user_section/tutorials/registration-to-template/vertebral-labeling.html"
        )
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input anatomical image. Example: anat.nii.gz"
    )
    mandatory.add_argument(
        '-s',
        metavar=Metavar.file,
        required=True,
        help="Spinal cord segmentation. Example: anat_seg.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-s-template-id',
        metavar=Metavar.int,
        type=int,
        help="Segmentation file ID to use for registration. The ID is an integer indicated in the file "
             "'template/info_label.txt'. This 'info_label.txt' file corresponds to the template indicated by the flag "
             "'-t'. By default, the spinal cord segmentation is used (ID=3), but if available, a different segmentation"
             " such as white matter segmentation could produce better registration results.",
        default=3
        )
    optional.add_argument(
        '-l',
        metavar=Metavar.file,
        help="One or two labels (preferred) located at the center of the spinal cord, on the mid-vertebral slice. "
             "Example: anat_labels.nii.gz\n"
             "For more information about label creation, please see: "
             "https://spinalcordtoolbox.com/user_section/tutorials/registration-to-template/vertebral-labeling.html"
    )
    optional.add_argument(
        '-ldisc',
        metavar=Metavar.file,
        help="File containing disc labels. Labels can be located either at the posterior edge "
             "of the intervertebral discs, or at the orthogonal projection of each disc onto "
             "the spinal cord (e.g.: the file 'xxx_seg_labeled_discs.nii.gz' output by sct_label_vertebrae).\n"
             "If you are using more than 2 labels, all discs covering the region of interest should be provided. "
             "E.g., if you are interested in levels C2 to C7, then you should provide disc labels 2,3,4,5,6,7. "
             "For more information about label creation, please refer to "
             "https://spinalcordtoolbox.com/user_section/tutorials/registration-to-template/vertebral-labeling.html"
    )
    optional.add_argument(
        '-lspinal',
        metavar=Metavar.file,
        help="Labels located in the center of the spinal cord, at the superior-inferior level corresponding to the "
             "mid-point of the spinal level. Example: anat_labels.nii.gz\n"
             "Each label is a single voxel, which value corresponds to the spinal level (e.g.: 2 for spinal level 2). "
             "If you are using more than 2 labels, all spinal levels covering the region of interest should be "
             "provided (e.g., if you are interested in levels C2 to C7, then you should provide spinal level labels "
             "2,3,4,5,6,7)."
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="Output folder."
    )
    optional.add_argument(
        '-t',
        metavar=Metavar.folder,
        default=param.path_template,
        help="Path to template"
    )
    optional.add_argument(
        '-c',
        choices=['t1', 't2', 't2s'],
        default='t2',
        help="Contrast to use for registration."
    )
    optional.add_argument(
        '-ref',
        choices=['template', 'subject'],
        default='template',
        help="Reference for registration: template: subject->template, subject: template->subject."
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(':', str),
        help=(f"Parameters for registration (see sct_register_multimodal). Default:"
              f"\n"
              f"step=0\n"
              f"  - type={paramregmulti.steps['0'].type}\n"
              f"  - dof={paramregmulti.steps['0'].dof}\n"
              f"\n"
              f"step=1\n"
              f"  - type={paramregmulti.steps['1'].type}\n"
              f"  - algo={paramregmulti.steps['1'].algo}\n"
              f"  - metric={paramregmulti.steps['1'].metric}\n"
              f"  - iter={paramregmulti.steps['1'].iter}\n"
              f"  - smooth={paramregmulti.steps['1'].smooth}\n"
              f"  - gradStep={paramregmulti.steps['1'].gradStep}\n"
              f"  - slicewise={paramregmulti.steps['1'].slicewise}\n"
              f"  - smoothWarpXY={paramregmulti.steps['1'].smoothWarpXY}\n"
              f"  - pca_eigenratio_th={paramregmulti.steps['1'].pca_eigenratio_th}\n"
              f"\n"
              f"step=2\n"
              f"  - type={paramregmulti.steps['2'].type}\n"
              f"  - algo={paramregmulti.steps['2'].algo}\n"
              f"  - metric={paramregmulti.steps['2'].metric}\n"
              f"  - iter={paramregmulti.steps['2'].iter}\n"
              f"  - smooth={paramregmulti.steps['2'].smooth}\n"
              f"  - gradStep={paramregmulti.steps['2'].gradStep}\n"
              f"  - slicewise={paramregmulti.steps['2'].slicewise}\n"
              f"  - smoothWarpXY={paramregmulti.steps['2'].smoothWarpXY}\n"
              f"  - pca_eigenratio_th={paramregmulti.steps['1'].pca_eigenratio_th}")
    )
    optional.add_argument(
        '-centerline-algo',
        choices=['polyfit', 'bspline', 'linear', 'nurbs'],
        default=ParamCenterline().algo_fitting,
        help="Algorithm for centerline fitting (when straightening the spinal cord)."
    )
    optional.add_argument(
        '-centerline-smooth',
        metavar=Metavar.int,
        type=int,
        default=ParamCenterline().smooth,
        help="Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}."
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default=param.path_qc,
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
        '-igt',
        metavar=Metavar.file,
        help="File name of ground-truth template cord segmentation (binary nifti)."
    )
    optional.add_argument(
        '-r',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=param.remove_temp_files,
        help="Whether to remove temporary files. 0 = no, 1 = yes"
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

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # initializations
    param = Param()

    fname_data = arguments.i
    fname_seg = arguments.s
    if arguments.l is not None:
        fname_landmarks = arguments.l
        label_type = 'body'
    elif arguments.ldisc is not None:
        fname_landmarks = arguments.ldisc
        label_type = 'disc'
    elif arguments.lspinal is not None:
        fname_landmarks = arguments.lspinal
        label_type = 'spinal'
    else:
        printv('ERROR: Labels should be provided.', 1, 'error')

    if arguments.ofolder is not None:
        path_output = arguments.ofolder
    else:
        path_output = ''

    param.path_qc = arguments.qc

    path_template = arguments.t
    contrast_template = arguments.c
    ref = arguments.ref
    param.remove_temp_files = arguments.r
    param.verbose = verbose  # TODO: not clean, unify verbose or param.verbose in code, but not both
    param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth)
    # registration parameters
    if arguments.param is not None:
        # reset parameters but keep step=0 (might be overwritten if user specified step=0)
        paramregmulti = ParamregMultiStep([step0])
        if ref == 'subject':
            paramregmulti.steps['0'].dof = 'Tx_Ty_Tz_Rx_Ry_Rz_Sz'
        # add user parameters
        for paramStep in arguments.param:
            paramregmulti.addStep(paramStep)
    else:
        paramregmulti = ParamregMultiStep([step0, step1, step2])
        # if ref=subject, initialize registration using different affine parameters
        if ref == 'subject':
            paramregmulti.steps['0'].dof = 'Tx_Ty_Tz_Rx_Ry_Rz_Sz'

    # initialize other parameters
    zsubsample = param.zsubsample

    # retrieve template file names
    if label_type == 'spinal':
        # point-wise spinal level labels
        file_template_labeling = get_file_label(os.path.join(path_template, 'template'), id_label=14)
    elif label_type == 'disc':
        # point-wise intervertebral disc labels
        file_template_labeling = get_file_label(os.path.join(path_template, 'template'), id_label=10)
    else:
        # spinal cord mask with discrete vertebral levels
        file_template_labeling = get_file_label(os.path.join(path_template, 'template'), id_label=7)

    id_label_dct = {'T1': 0, 'T2': 1, 'T2S': 2}
    file_template = get_file_label(os.path.join(path_template, 'template'), id_label=id_label_dct[contrast_template.upper()])  # label = *-weighted template
    file_template_seg = get_file_label(os.path.join(path_template, 'template'), id_label=arguments.s_template_id)

    # start timer
    start_time = time.time()

    # get fname of the template + template objects
    fname_template = os.path.join(path_template, 'template', file_template)
    fname_template_labeling = os.path.join(path_template, 'template', file_template_labeling)
    fname_template_seg = os.path.join(path_template, 'template', file_template_seg)

    # check file existence
    # TODO: no need to do that!
    printv('\nCheck template files...')
    check_file_exist(fname_template, verbose)
    check_file_exist(fname_template_labeling, verbose)
    check_file_exist(fname_template_seg, verbose)
    path_data, file_data, ext_data = extract_fname(fname_data)

    # printv(arguments)
    printv('\nCheck parameters:', verbose)
    printv('  Data:                 ' + fname_data, verbose)
    printv('  Landmarks:            ' + fname_landmarks, verbose)
    printv('  Segmentation:         ' + fname_seg, verbose)
    printv('  Path template:        ' + path_template, verbose)
    printv('  Remove temp files:    ' + str(param.remove_temp_files), verbose)

    # check input labels
    labels = check_labels(fname_landmarks, label_type=label_type)

    level_alignment = False
    if len(labels) > 2 and label_type in ['disc', 'spinal']:
        level_alignment = True

    path_tmp = tmp_create(basename="register_to_template")

    # set temporary file names
    ftmp_data = 'data.nii'
    ftmp_seg = 'seg.nii.gz'
    ftmp_label = 'label.nii.gz'
    ftmp_template = 'template.nii'
    ftmp_template_seg = 'template_seg.nii.gz'
    ftmp_template_label = 'template_label.nii.gz'

    # copy files to temporary folder
    printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    Image(fname_data).save(os.path.join(path_tmp, ftmp_data))
    Image(fname_seg).save(os.path.join(path_tmp, ftmp_seg))
    Image(fname_landmarks).save(os.path.join(path_tmp, ftmp_label))
    Image(fname_template).save(os.path.join(path_tmp, ftmp_template))
    Image(fname_template_seg).save(os.path.join(path_tmp, ftmp_template_seg))
    Image(fname_template_labeling).save(os.path.join(path_tmp, ftmp_template_label))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Generate labels from template vertebral labeling
    if label_type == 'body':
        printv('\nGenerate labels from template vertebral labeling', verbose)
        ftmp_template_label_, ftmp_template_label = ftmp_template_label, add_suffix(ftmp_template_label, "_body")
        sct_labels.label_vertebrae(Image(ftmp_template_label_)).save(path=ftmp_template_label)

    # check if provided labels are available in the template
    printv('\nCheck if provided labels are available in the template', verbose)
    image_label_template = Image(ftmp_template_label)

    labels_template = image_label_template.getNonZeroCoordinates(sorting='value')
    if labels[-1].value > labels_template[-1].value:
        printv('ERROR: Wrong landmarks input. Labels must have correspondence in template space. \nLabel max '
               'provided: ' + str(labels[-1].value) + '\nLabel max from template: ' +
               str(labels_template[-1].value), verbose, 'error')

    # if only one label is present, force affine transformation to be Tx,Ty,Tz only (no scaling)
    if len(labels) == 1:
        paramregmulti.steps['0'].dof = 'Tx_Ty_Tz'
        printv('WARNING: Only one label is present. Forcing initial transformation to: ' + paramregmulti.steps['0'].dof,
               1, 'warning')

    # Project labels onto the spinal cord centerline because later, an affine transformation is estimated between the
    # template's labels (centered in the cord) and the subject's labels (assumed to be centered in the cord).
    # If labels are not centered, mis-registration errors are observed (see issue #1826)
    ftmp_label = project_labels_on_spinalcord(ftmp_label, ftmp_seg, param_centerline)

    # binarize segmentation (in case it has values below 0 caused by manual editing)
    printv('\nBinarize segmentation', verbose)
    ftmp_seg_, ftmp_seg = ftmp_seg, add_suffix(ftmp_seg, "_bin")
    img = Image(ftmp_seg_)
    out = img.copy()
    out.data = binarize(out.data, 0.5)
    out.save(path=ftmp_seg)

    # Switch between modes: subject->template or template->subject
    if ref == 'template':

        # resample data to 1mm isotropic
        printv('\nResample data to 1mm isotropic...', verbose)
        resample_file(ftmp_data, add_suffix(ftmp_data, '_1mm'), '1.0x1.0x1.0', 'mm', 'linear', verbose)
        ftmp_data = add_suffix(ftmp_data, '_1mm')
        resample_file(ftmp_seg, add_suffix(ftmp_seg, '_1mm'), '1.0x1.0x1.0', 'mm', 'linear', verbose)
        ftmp_seg = add_suffix(ftmp_seg, '_1mm')
        # N.B. resampling of labels is more complicated, because they are single-point labels, therefore resampling
        # with nearest neighbour can make them disappear.
        resample_labels(ftmp_label, ftmp_data, add_suffix(ftmp_label, '_1mm'))
        ftmp_label = add_suffix(ftmp_label, '_1mm')

        # Change orientation of input images to RPI
        printv('\nChange orientation of input images to RPI...', verbose)

        img_tmp_data = Image(ftmp_data).change_orientation("RPI")
        ftmp_data = add_suffix(img_tmp_data.absolutepath, "_rpi")
        img_tmp_data.save(path=ftmp_data, mutable=True)

        img_tmp_seg = Image(ftmp_seg).change_orientation("RPI")
        ftmp_seg = add_suffix(img_tmp_seg.absolutepath, "_rpi")
        img_tmp_seg.save(path=ftmp_seg, mutable=True)

        img_tmp_label = Image(ftmp_label).change_orientation("RPI")
        ftmp_label = add_suffix(img_tmp_label.absolutepath, "_rpi")
        img_tmp_label.save(ftmp_label, mutable=True)

        ftmp_seg_, ftmp_seg = ftmp_seg, add_suffix(ftmp_seg, '_crop')
        if level_alignment:
            # cropping the segmentation based on the label coverage to ensure good registration with level alignment
            # See https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1669 for details
            image_labels = Image(ftmp_label)
            coordinates_labels = image_labels.getNonZeroCoordinates(sorting='z')
            nx, ny, nz, nt, px, py, pz, pt = image_labels.dim
            offset_crop = 10.0 * pz  # cropping the image 10 mm above and below the highest and lowest label
            cropping_slices = [coordinates_labels[0].z - offset_crop, coordinates_labels[-1].z + offset_crop]
            # make sure that the cropping slices do not extend outside of the slice range (issue #1811)
            if cropping_slices[0] < 0:
                cropping_slices[0] = 0
            if cropping_slices[1] > nz:
                cropping_slices[1] = nz
            msct_image.spatial_crop(Image(ftmp_seg_), dict(((2, np.int32(np.round(cropping_slices))),))).save(ftmp_seg)
        else:
            # if we do not align the vertebral levels, we crop the segmentation from top to bottom
            im_seg_rpi = Image(ftmp_seg_)
            bottom = 0
            for data in msct_image.SlicerOneAxis(im_seg_rpi, "IS"):
                if (data != 0).any():
                    break
                bottom += 1
            top = im_seg_rpi.data.shape[2]
            for data in msct_image.SlicerOneAxis(im_seg_rpi, "SI"):
                if (data != 0).any():
                    break
                top -= 1
            msct_image.spatial_crop(im_seg_rpi, dict(((2, (bottom, top)),))).save(ftmp_seg)

        # straighten segmentation
        printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)

        # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
        fn_warp_curve2straight = os.path.join(curdir, "warp_curve2straight.nii.gz")
        fn_warp_straight2curve = os.path.join(curdir, "warp_straight2curve.nii.gz")
        fn_straight_ref = os.path.join(curdir, "straight_ref.nii.gz")

        cache_input_files = [ftmp_seg]
        if level_alignment:
            cache_input_files += [
                ftmp_template_seg,
                ftmp_label,
                ftmp_template_label,
            ]
        cache_sig = cache_signature(
            input_files=cache_input_files,
        )
        cachefile = os.path.join(curdir, "straightening.cache")
        if cache_valid(cachefile, cache_sig) and os.path.isfile(fn_warp_curve2straight) and os.path.isfile(fn_warp_straight2curve) and os.path.isfile(fn_straight_ref):
            printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
            copy(fn_warp_curve2straight, 'warp_curve2straight.nii.gz')
            copy(fn_warp_straight2curve, 'warp_straight2curve.nii.gz')
            copy(fn_straight_ref, 'straight_ref.nii.gz')
            # apply straightening
            sct_apply_transfo.main(argv=[
                '-i', ftmp_seg,
                '-w', 'warp_curve2straight.nii.gz',
                '-d', 'straight_ref.nii.gz',
                '-o', add_suffix(ftmp_seg, '_straight')])
        else:
            from spinalcordtoolbox.straightening import SpinalCordStraightener
            sc_straight = SpinalCordStraightener(ftmp_seg, ftmp_seg)
            sc_straight.param_centerline = param_centerline
            sc_straight.output_filename = add_suffix(ftmp_seg, '_straight')
            sc_straight.path_output = '.'
            sc_straight.qc = '0'
            sc_straight.remove_temp_files = param.remove_temp_files
            sc_straight.verbose = verbose

            if level_alignment:
                sc_straight.centerline_reference_filename = ftmp_template_seg
                sc_straight.use_straight_reference = True
                sc_straight.discs_input_filename = ftmp_label
                sc_straight.discs_ref_filename = ftmp_template_label

            sc_straight.straighten()
            cache_save(cachefile, cache_sig)

        # N.B. DO NOT UPDATE VARIABLE ftmp_seg BECAUSE TEMPORARY USED LATER
        # re-define warping field using non-cropped space (to avoid issue #367)

        dimensionality = len(Image(ftmp_data).hdr.get_data_shape())
        cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_straight2curve.nii.gz', '-R', ftmp_data, 'warp_straight2curve.nii.gz']
        status, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)
        if status != 0:
            raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

        if level_alignment:
            copy('warp_curve2straight.nii.gz', 'warp_curve2straightAffine.nii.gz')
        else:
            # Label preparation:
            # --------------------------------------------------------------------------------
            # Remove unused label on template. Keep only label present in the input label image
            printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
            sct_labels.remove_missing_labels(Image(ftmp_template_label), Image(ftmp_label)).save(path=ftmp_template_label)

            # Dilating the input label so they can be straighten without losing them
            printv('\nDilating input labels using 3vox ball radius')
            dilate(Image(ftmp_label), 3, 'ball').save(add_suffix(ftmp_label, '_dilate'))
            ftmp_label = add_suffix(ftmp_label, '_dilate')

            # Apply straightening to labels
            printv('\nApply straightening to labels...', verbose)
            sct_apply_transfo.main(argv=[
                '-i', ftmp_label,
                '-o', add_suffix(ftmp_label, '_straight'),
                '-d', add_suffix(ftmp_seg, '_straight'),
                '-w', 'warp_curve2straight.nii.gz',
                '-x', 'nn'])
            ftmp_label = add_suffix(ftmp_label, '_straight')

            # Compute rigid transformation straight landmarks --> template landmarks
            printv('\nEstimate transformation for step #0...', verbose)
            try:
                register_landmarks(ftmp_label, ftmp_template_label, paramregmulti.steps['0'].dof,
                                   fname_affine='straight2templateAffine.txt', verbose=verbose)
            except RuntimeError:
                raise('Input labels do not seem to be at the right place. Please check the position of the labels. '
                      'See documentation for more details: https://spinalcordtoolbox.com/user_section/tutorials/registration-to-template/vertebral-labeling.html')

            # Concatenate transformations: curve --> straight --> affine
            printv('\nConcatenate transformations: curve --> straight --> affine...', verbose)

            dimensionality = len(Image("template.nii").hdr.get_data_shape())
            cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_curve2straightAffine.nii.gz', '-R', 'template.nii', 'straight2templateAffine.txt', 'warp_curve2straight.nii.gz']
            status, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)
            if status != 0:
                raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

        # Apply transformation
        printv('\nApply transformation...', verbose)
        sct_apply_transfo.main(argv=[
            '-i', ftmp_data,
            '-o', add_suffix(ftmp_data, '_straightAffine'),
            '-d', ftmp_template,
            '-w', 'warp_curve2straightAffine.nii.gz'])
        ftmp_data = add_suffix(ftmp_data, '_straightAffine')
        sct_apply_transfo.main(argv=[
            '-i', ftmp_seg,
            '-o', add_suffix(ftmp_seg, '_straightAffine'),
            '-d', ftmp_template,
            '-w', 'warp_curve2straightAffine.nii.gz',
            '-x', 'linear'])
        ftmp_seg = add_suffix(ftmp_seg, '_straightAffine')

        """
        # Benjamin: Issue from Allan Martin, about the z=0 slice that is screwed up, caused by the affine transform.
        # Solution found: remove slices below and above landmarks to avoid rotation effects
        points_straight = []
        for coord in landmark_template:
            points_straight.append(coord.z)
        min_point, max_point = int(np.round(np.min(points_straight))), int(np.round(np.max(points_straight)))
        ftmp_seg_, ftmp_seg = ftmp_seg, add_suffix(ftmp_seg, '_black')
        msct_image.spatial_crop(Image(ftmp_seg_), dict(((2, (min_point,max_point)),))).save(ftmp_seg)

        """
        # open segmentation
        im = Image(ftmp_seg)
        im_new = msct_image.empty_like(im)
        # binarize
        im_new.data = im.data > 0.5
        # find min-max of anat2template (for subsequent cropping)
        zmin_template, zmax_template = msct_image.find_zmin_zmax(im_new, threshold=0.5)
        # save binarized segmentation
        im_new.save(add_suffix(ftmp_seg, '_bin'))  # unused?
        # crop template in z-direction (for faster processing)
        # TODO: refactor to use python module instead of doing i/o
        printv('\nCrop data in template space (for faster processing)...', verbose)
        ftmp_template_, ftmp_template = ftmp_template, add_suffix(ftmp_template, '_crop')
        msct_image.spatial_crop(Image(ftmp_template_), dict(((2, (zmin_template, zmax_template)),))).save(ftmp_template)

        ftmp_template_seg_, ftmp_template_seg = ftmp_template_seg, add_suffix(ftmp_template_seg, '_crop')
        msct_image.spatial_crop(Image(ftmp_template_seg_), dict(((2, (zmin_template, zmax_template)),))).save(ftmp_template_seg)

        ftmp_data_, ftmp_data = ftmp_data, add_suffix(ftmp_data, '_crop')
        msct_image.spatial_crop(Image(ftmp_data_), dict(((2, (zmin_template, zmax_template)),))).save(ftmp_data)

        ftmp_seg_, ftmp_seg = ftmp_seg, add_suffix(ftmp_seg, '_crop')
        msct_image.spatial_crop(Image(ftmp_seg_), dict(((2, (zmin_template, zmax_template)),))).save(ftmp_seg)

        # sub-sample in z-direction
        # TODO: refactor to use python module instead of doing i/o
        printv('\nSub-sample in z-direction (for faster processing)...', verbose)
        run_proc(['sct_resample', '-i', ftmp_template, '-o', add_suffix(ftmp_template, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_template = add_suffix(ftmp_template, '_sub')
        run_proc(['sct_resample', '-i', ftmp_template_seg, '-o', add_suffix(ftmp_template_seg, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_template_seg = add_suffix(ftmp_template_seg, '_sub')
        run_proc(['sct_resample', '-i', ftmp_data, '-o', add_suffix(ftmp_data, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_data = add_suffix(ftmp_data, '_sub')
        run_proc(['sct_resample', '-i', ftmp_seg, '-o', add_suffix(ftmp_seg, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_seg = add_suffix(ftmp_seg, '_sub')

        # Registration straight spinal cord to template
        printv('\nRegister straight spinal cord to template...', verbose)

        # TODO: find a way to input initwarp, corresponding to straightening warp
        # Set the angle of the template orientation to 0 (destination image)
        for key in list(paramregmulti.steps.keys()):
            paramregmulti.steps[key].rot_dest = 0
        fname_src2dest, fname_dest2src, warp_forward, warp_inverse = register_wrapper(
            ftmp_data, ftmp_template, param, paramregmulti, fname_src_seg=ftmp_seg, fname_dest_seg=ftmp_template_seg,
            same_space=True)

        # Concatenate transformations: anat --> template
        printv('\nConcatenate transformations: anat --> template...', verbose)

        dimensionality = len(Image("template.nii").hdr.get_data_shape())
        cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_anat2template.nii.gz', '-R', 'template.nii', warp_forward, 'warp_curve2straightAffine.nii.gz']
        status, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)
        if status != 0:
            raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

        # Concatenate transformations: template --> anat
        printv('\nConcatenate transformations: template --> anat...', verbose)
        # TODO: make sure the commented code below is consistent with the new implementation
        # warp_inverse.reverse()
        if level_alignment:
            dimensionality = len(Image("data.nii").hdr.get_data_shape())
            cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_template2anat.nii.gz', '-R', 'data.nii', 'warp_straight2curve.nii.gz', warp_inverse]
            status, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)
            if status != 0:
                raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

        else:
            dimensionality = len(Image("data.nii").hdr.get_data_shape())
            cmd = ['isct_ComposeMultiTransform', f"{dimensionality}", 'warp_template2anat.nii.gz', '-R', 'data.nii', 'warp_straight2curve.nii.gz', '-i', 'straight2templateAffine.txt', warp_inverse]
            status, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)
            if status != 0:
                raise RuntimeError(f"Subprocess call {cmd} returned non-zero: {output}")

    # register template->subject
    elif ref == 'subject':

        # Change orientation of input images to RPI
        printv('\nChange orientation of input images to RPI...', verbose)

        img_tmp_data = Image(ftmp_data).change_orientation("RPI")
        ftmp_data = add_suffix(img_tmp_data.absolutepath, "_rpi")
        img_tmp_data.save(path=ftmp_data, mutable=True)

        img_tmp_seg = Image(ftmp_seg).change_orientation("RPI")
        ftmp_seg = add_suffix(img_tmp_seg.absolutepath, "_rpi")
        img_tmp_seg.save(path=ftmp_seg, mutable=True)

        img_tmp_label = Image(ftmp_label).change_orientation("RPI")
        ftmp_label = add_suffix(img_tmp_label.absolutepath, "_rpi")
        img_tmp_label.save(ftmp_label, mutable=True)

        # Remove unused label on template. Keep only label present in the input label image
        printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
        sct_labels.remove_missing_labels(Image(ftmp_template_label), Image(ftmp_label)).save(path=ftmp_template_label)

        # Add one label because at least 3 orthogonal labels are required to estimate an affine transformation.
        add_orthogonal_label(ftmp_label)
        add_orthogonal_label(ftmp_template_label)

        # Set the angle of the template orientation to 0 (source image)
        for key in list(paramregmulti.steps.keys()):
            paramregmulti.steps[key].rot_src = 0
        fname_src2dest, fname_dest2src, warp_forward, warp_inverse = register_wrapper(
            ftmp_template, ftmp_data, param, paramregmulti, fname_src_seg=ftmp_template_seg, fname_dest_seg=ftmp_seg,
            fname_src_label=ftmp_template_label, fname_dest_label=ftmp_label, same_space=False)
        # Renaming for code compatibility
        os.rename(warp_forward, 'warp_template2anat.nii.gz')
        os.rename(warp_inverse, 'warp_anat2template.nii.gz')

    # Apply warping fields to anat and template
    run_proc(['sct_apply_transfo', '-i', 'template.nii', '-o', 'template2anat.nii.gz', '-d', 'data.nii', '-w', 'warp_template2anat.nii.gz', '-crop', '0'], verbose)
    run_proc(['sct_apply_transfo', '-i', 'data.nii', '-o', 'anat2template.nii.gz', '-d', 'template.nii', '-w', 'warp_anat2template.nii.gz', '-crop', '0'], verbose)

    # come back
    os.chdir(curdir)

    # Generate output files
    printv('\nGenerate output files...', verbose)
    fname_template2anat = os.path.join(path_output, 'template2anat' + ext_data)
    fname_anat2template = os.path.join(path_output, 'anat2template' + ext_data)
    generate_output_file(os.path.join(path_tmp, "warp_template2anat.nii.gz"), os.path.join(path_output, "warp_template2anat.nii.gz"), verbose=verbose)
    generate_output_file(os.path.join(path_tmp, "warp_anat2template.nii.gz"), os.path.join(path_output, "warp_anat2template.nii.gz"), verbose=verbose)
    generate_output_file(os.path.join(path_tmp, "template2anat.nii.gz"), fname_template2anat, verbose=verbose)
    generate_output_file(os.path.join(path_tmp, "anat2template.nii.gz"), fname_anat2template, verbose=verbose)
    if ref == 'template':
        # copy straightening files in case subsequent SCT functions need them
        generate_output_file(os.path.join(path_tmp, "warp_curve2straight.nii.gz"), os.path.join(path_output, "warp_curve2straight.nii.gz"), verbose=verbose)
        generate_output_file(os.path.join(path_tmp, "warp_straight2curve.nii.gz"), os.path.join(path_output, "warp_straight2curve.nii.gz"), verbose=verbose)
        generate_output_file(os.path.join(path_tmp, "straight_ref.nii.gz"), os.path.join(path_output, "straight_ref.nii.gz"), verbose=verbose)

    # Delete temporary files
    if param.remove_temp_files:
        printv('\nDelete temporary files...', verbose)
        rmtree(path_tmp, verbose=verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', verbose)

    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    if param.path_qc is not None:
        generate_qc(fname_data, fname_in2=fname_template2anat, fname_seg=fname_seg, args=argv,
                    path_qc=os.path.abspath(param.path_qc), dataset=qc_dataset, subject=qc_subject,
                    process='sct_register_to_template')
    display_viewer_syntax([fname_data, fname_template2anat], verbose=verbose)
    display_viewer_syntax([fname_template, fname_anat2template], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
