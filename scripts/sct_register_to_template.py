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

from __future__ import division, absolute_import

import sys, os, time
import numpy as np

from spinalcordtoolbox.metadata import get_file_label
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.resampling import resample_file
from spinalcordtoolbox.math import dilate

import sct_utils as sct
import sct_maths
import sct_label_utils
from sct_utils import add_suffix
from msct_register import Paramreg, ParamregMultiStep, register_wrapper
from msct_parser import Parser
from msct_register_landmarks import register_landmarks
import sct_apply_transfo
import sct_concat_transfo


class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1  # remove temporary files
        self.fname_mask = ''  # this field is needed in the function register@sct_register_multimodal
        self.padding = 10  # this field is needed in the function register@sct_register_multimodal
        self.verbose = 1  # verbose
        self.path_template = os.path.join(sct.__data_dir__, 'PAM50')
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
    parser = Parser(__file__)
    parser.usage.set_description('Register an anatomical image to the spinal cord MRI template (default: PAM50).\n\n'
                                 'The registration process includes three main registration steps:\n'
                                   '1. straightening of the image using the spinal cord segmentation (see sct_straighten_spinalcord for details);\n'
                                   '2. vertebral alignment between the image and the template, using labels along the spine;\n'
                                   '3. iterative slice-wise non-linear registration (see sct_register_multimodal for details)\n\n'
                                 'To register a subject to the template, try the default command:\n'
                                   'sct_register_to_template -i data.nii.gz -s data_seg.nii.gz -l data_labels.nii.gz\n\n'
                                 'If this default command does not produce satisfactory results, please refer to:\n'
                                   'https://sourceforge.net/p/spinalcordtoolbox/wiki/registration_tricks/\n\n'
                                 'The default registration method brings the subject image to the template, which can be problematic with highly non-isotropic images as it would induce large interpolation errors during the straightening procedure. Although the default method is recommended, you may want to register the template to the subject (instead of the subject to the template) by skipping the straightening procedure. To do so, use the parameter "-ref subject". Example below:\n'
                                   'sct_register_to_template -i data.nii.gz -s data_seg.nii.gz -l data_labels.nii.gz -ref subject -param step=1,type=seg,algo=centermassrot,smooth=0:step=2,type=seg,algo=columnwise,smooth=0,smoothWarpXY=2\n\n'
                                 'Vertebral alignment (step 2) consists in aligning the vertebrae between the subject and the template. Two types of labels are possible:\n'
                                   '- Vertebrae mid-body labels, created at the center of the spinal cord using the parameter "-l";\n'
                                   '- Posterior edge of the intervertebral discs, using the parameter "-ldisc".\n\n'
                                 'If only one label is provided, a simple translation will be applied between the subject label and the template label. No scaling will be performed. \n\n'
                                 'If two labels are provided, a linear transformation (translation + rotation + superior-inferior linear scaling) will be applied. The strategy here is to defined labels that cover the region of interest. For example, if you are interested in studying C2 to C6 levels, then provide one label at C2 and another at C6. However, note that if the two labels are very far apart (e.g. C2 and T12), there might be a mis-alignment of discs because a subject''s intervertebral discs distance might differ from that of the template.\n\n'
                                 'If more than two labels (only with the parameter "-disc") are used, a non-linear registration will be applied to align the each intervertebral disc between the subject and the template, as described in sct_straighten_spinalcord. This the most accurate and preferred method. This feature does not work with the parameter "-ref subject".\n\n'
                                 'More information about label creation can be found at https://www.slideshare.net/neuropoly/sct-course-20190121/42'
      )
    parser.add_option(name="-i",
                      type_value="file",
                      description="Anatomical image.",
                      mandatory=True,
                      example="anat.nii.gz")
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord segmentation.",
                      mandatory=True,
                      example="anat_seg.nii.gz")
    parser.add_option(name="-l",
                      type_value="file",
                      description="One or two labels (preferred) located at the center of the spinal cord, on the "
                                  "mid-vertebral slice. For more information about label creation, please see: "
                                  "https://www.slideshare.net/neuropoly/sct-course-20190121/42",
                      mandatory=False,
                      default_value='',
                      example="anat_labels.nii.gz")
    parser.add_option(name="-ldisc",
                      type_value="file",
                      description="Labels located at the posterior edge of the intervertebral discs. If you are using "
                                  "more than 2 labels, all disc covering the region of interest should be provided. "
                                  "E.g., if you are interested in levels C2 to C7, then you should provide disc labels "
                                  "2,3,4,5,6,7). For more information about label creation, please refer to "
                                  "https://www.slideshare.net/neuropoly/sct-course-20190121/42",  # TODO: update URL
                      mandatory=False,
                      default_value='',
                      example="anat_labels.nii.gz")
    parser.add_option(name="-lspinal",
                      type_value="file",
                      description="Labels located in the center of the spinal cord, at the superior-inferior level " 
                                  "corresponding to the mid-point of the spinal level. Each label is a single voxel, " 
                                  "which value corresponds to the spinal level (e.g.: 2 for spinal level 2). If you " 
                                  "are using more than 2 labels, all spinal levels covering the region of interest " 
                                  "should be provided (e.g., if you are interested in levels C2 to C7, then you " 
                                  "should provide spinal level labels 2,3,4,5,6,7).",
                      mandatory=False,
                      default_value='',
                      example="anat_labels.nii.gz")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder.",
                      mandatory=False,
                      default_value='')
    parser.add_option(name="-t",
                      type_value="folder",
                      description="Path to template.",
                      mandatory=False,
                      default_value=param.path_template)
    parser.add_option(name='-c',
                      type_value='multiple_choice',
                      description='Contrast to use for registration.',
                      mandatory=False,
                      default_value='t2',
                      example=['t1', 't2', 't2s'])
    parser.add_option(name='-ref',
                      type_value='multiple_choice',
                      description='Reference for registration: template: subject->template, subject: template->subject.',
                      mandatory=False,
                      default_value='template',
                      example=['template', 'subject'])
    parser.add_option(name="-param",
                      type_value=[[':'], 'str'],
                      description='Parameters for registration (see sct_register_multimodal). Default: \
                      \n--\nstep=0\ntype=' + paramregmulti.steps['0'].type + '\ndof=' + paramregmulti.steps['0'].dof + '\
                      \n--\nstep=1\ntype=' + paramregmulti.steps['1'].type + '\nalgo=' + paramregmulti.steps['1'].algo + '\nmetric=' + paramregmulti.steps['1'].metric + '\niter=' + paramregmulti.steps['1'].iter + '\nsmooth=' + paramregmulti.steps['1'].smooth + '\ngradStep=' + paramregmulti.steps['1'].gradStep + '\nslicewise=' + paramregmulti.steps['1'].slicewise + '\nsmoothWarpXY=' + paramregmulti.steps['1'].smoothWarpXY + '\npca_eigenratio_th=' + paramregmulti.steps['1'].pca_eigenratio_th + '\
                      \n--\nstep=2\ntype=' + paramregmulti.steps['2'].type + '\nalgo=' + paramregmulti.steps['2'].algo + '\nmetric=' + paramregmulti.steps['2'].metric + '\niter=' + paramregmulti.steps['2'].iter + '\nsmooth=' + paramregmulti.steps['2'].smooth + '\ngradStep=' + paramregmulti.steps['2'].gradStep + '\nslicewise=' + paramregmulti.steps['2'].slicewise + '\nsmoothWarpXY=' + paramregmulti.steps['2'].smoothWarpXY + '\npca_eigenratio_th=' + paramregmulti.steps['1'].pca_eigenratio_th,
                      mandatory=False)
    parser.add_option(name='-centerline-algo',
                      type_value='multiple_choice',
                      description='Algorithm for centerline fitting (when straightening the spinal cord).',
                      mandatory=False,
                      example=['polyfit', 'bspline', 'linear', 'nurbs'],
                      default_value=ParamCenterline().algo_fitting)
    parser.add_option(name='-centerline-smooth',
                      type_value='int',
                      description='Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}.',
                      mandatory=False,
                      default_value=ParamCenterline().smooth)
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=param.path_qc)
    parser.add_option(name='-qc-dataset',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
                      )
    parser.add_option(name='-qc-subject',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the subject the process was run on',
                      )
    parser.add_option(name="-igt",
                      type_value="image_nifti",
                      description="File name of ground-truth template cord segmentation (binary nifti).",
                      mandatory=False)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value=param.remove_temp_files,
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1', '2'])
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # initializations
    param = Param()

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    fname_data = arguments['-i']
    fname_seg = arguments['-s']
    if '-l' in arguments:
        fname_landmarks = arguments['-l']
        label_type = 'body'
    elif '-ldisc' in arguments:
        fname_landmarks = arguments['-ldisc']
        label_type = 'disc'
    elif '-lspinal' in arguments:
        fname_landmarks = arguments['-lspinal']
        label_type = 'spinal'
    else:
        sct.printv('ERROR: Labels should be provided.', 1, 'error')
    if '-ofolder' in arguments:
        path_output = arguments['-ofolder']
    else:
        path_output = ''

    param.path_qc = arguments.get("-qc", None)

    path_template = arguments['-t']
    contrast_template = arguments['-c']
    ref = arguments['-ref']
    param.remove_temp_files = int(arguments.get('-r'))
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    param.verbose = verbose  # TODO: not clean, unify verbose or param.verbose in code, but not both
    param_centerline = ParamCenterline(
        algo_fitting=arguments['-centerline-algo'],
        smooth=arguments['-centerline-smooth'])
    # registration parameters
    if '-param' in arguments:
        # reset parameters but keep step=0 (might be overwritten if user specified step=0)
        paramregmulti = ParamregMultiStep([step0])
        if ref == 'subject':
            paramregmulti.steps['0'].dof = 'Tx_Ty_Tz_Rx_Ry_Rz_Sz'
        # add user parameters
        for paramStep in arguments['-param']:
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
        file_template_labeling = get_file_label(os.path.join(path_template, 'template'), id_label=14)  # label = point-wise spinal level labels
    else:
        file_template_labeling = get_file_label(os.path.join(path_template, 'template'), id_label=7)  # label = spinal cord mask with discrete vertebral levels
    id_label_dct = {'T1': 0, 'T2': 1, 'T2S': 2}
    file_template = get_file_label(os.path.join(path_template, 'template'), id_label=id_label_dct[contrast_template.upper()])  # label = *-weighted template
    file_template_seg = get_file_label(os.path.join(path_template, 'template'), id_label=3)  # label = spinal cord mask (binary)

    # start timer
    start_time = time.time()

    # get fname of the template + template objects
    fname_template = os.path.join(path_template, 'template', file_template)
    fname_template_labeling = os.path.join(path_template, 'template', file_template_labeling)
    fname_template_seg = os.path.join(path_template, 'template', file_template_seg)
    fname_template_disc_labeling = os.path.join(path_template, 'template', 'PAM50_label_disc.nii.gz')

    # check file existence
    # TODO: no need to do that!
    sct.printv('\nCheck template files...')
    sct.check_file_exist(fname_template, verbose)
    sct.check_file_exist(fname_template_labeling, verbose)
    sct.check_file_exist(fname_template_seg, verbose)
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # sct.printv(arguments)
    sct.printv('\nCheck parameters:', verbose)
    sct.printv('  Data:                 ' + fname_data, verbose)
    sct.printv('  Landmarks:            ' + fname_landmarks, verbose)
    sct.printv('  Segmentation:         ' + fname_seg, verbose)
    sct.printv('  Path template:        ' + path_template, verbose)
    sct.printv('  Remove temp files:    ' + str(param.remove_temp_files), verbose)

    # check input labels
    labels = check_labels(fname_landmarks, label_type=label_type)

    level_alignment = False
    if len(labels) > 2 and label_type in ['disc', 'spinal']:
        level_alignment = True

    path_tmp = sct.tmp_create(basename="register_to_template", verbose=verbose)

    # set temporary file names
    ftmp_data = 'data.nii'
    ftmp_seg = 'seg.nii.gz'
    ftmp_label = 'label.nii.gz'
    ftmp_template = 'template.nii'
    ftmp_template_seg = 'template_seg.nii.gz'
    ftmp_template_label = 'template_label.nii.gz'

    # copy files to temporary folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    Image(fname_data).save(os.path.join(path_tmp, ftmp_data))
    Image(fname_seg).save(os.path.join(path_tmp, ftmp_seg))
    Image(fname_landmarks).save(os.path.join(path_tmp, ftmp_label))
    Image(fname_template).save(os.path.join(path_tmp, ftmp_template))
    Image(fname_template_seg).save(os.path.join(path_tmp, ftmp_template_seg))
    Image(fname_template_labeling).save(os.path.join(path_tmp, ftmp_template_label))
    if label_type == 'disc':
        Image(fname_template_disc_labeling).save(os.path.join(path_tmp, ftmp_template_label))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Generate labels from template vertebral labeling
    if label_type == 'body':
        sct.printv('\nGenerate labels from template vertebral labeling', verbose)
        ftmp_template_label_, ftmp_template_label = ftmp_template_label, sct.add_suffix(ftmp_template_label, "_body")
        sct_label_utils.main(args=['-i', ftmp_template_label_, '-vert-body', '0', '-o', ftmp_template_label])

    # check if provided labels are available in the template
    sct.printv('\nCheck if provided labels are available in the template', verbose)
    image_label_template = Image(ftmp_template_label)
    labels_template = image_label_template.getNonZeroCoordinates(sorting='value')
    if labels[-1].value > labels_template[-1].value:
        sct.printv('ERROR: Wrong landmarks input. Labels must have correspondence in template space. \nLabel max '
                   'provided: ' + str(labels[-1].value) + '\nLabel max from template: ' +
                   str(labels_template[-1].value), verbose, 'error')

    # if only one label is present, force affine transformation to be Tx,Ty,Tz only (no scaling)
    if len(labels) == 1:
        paramregmulti.steps['0'].dof = 'Tx_Ty_Tz'
        sct.printv('WARNING: Only one label is present. Forcing initial transformation to: ' + paramregmulti.steps['0'].dof,
                   1, 'warning')

    # Project labels onto the spinal cord centerline because later, an affine transformation is estimated between the
    # template's labels (centered in the cord) and the subject's labels (assumed to be centered in the cord).
    # If labels are not centered, mis-registration errors are observed (see issue #1826)
    ftmp_label = project_labels_on_spinalcord(ftmp_label, ftmp_seg, param_centerline)

    # binarize segmentation (in case it has values below 0 caused by manual editing)
    sct.printv('\nBinarize segmentation', verbose)
    ftmp_seg_, ftmp_seg = ftmp_seg, sct.add_suffix(ftmp_seg, "_bin")
    sct_maths.main(['-i', ftmp_seg_,
                    '-bin', '0.5',
                    '-o', ftmp_seg])

    # Switch between modes: subject->template or template->subject
    if ref == 'template':

        # resample data to 1mm isotropic
        sct.printv('\nResample data to 1mm isotropic...', verbose)
        resample_file(ftmp_data, add_suffix(ftmp_data, '_1mm'), '1.0x1.0x1.0', 'mm', 'linear', verbose)
        ftmp_data = add_suffix(ftmp_data, '_1mm')
        resample_file(ftmp_seg, add_suffix(ftmp_seg, '_1mm'), '1.0x1.0x1.0', 'mm', 'linear', verbose)
        ftmp_seg = add_suffix(ftmp_seg, '_1mm')
        # N.B. resampling of labels is more complicated, because they are single-point labels, therefore resampling
        # with nearest neighbour can make them disappear.
        resample_labels(ftmp_label, ftmp_data, add_suffix(ftmp_label, '_1mm'))
        ftmp_label = add_suffix(ftmp_label, '_1mm')

        # Change orientation of input images to RPI
        sct.printv('\nChange orientation of input images to RPI...', verbose)

        ftmp_data = Image(ftmp_data).change_orientation("RPI", generate_path=True).save().absolutepath
        ftmp_seg = Image(ftmp_seg).change_orientation("RPI", generate_path=True).save().absolutepath
        ftmp_label = Image(ftmp_label).change_orientation("RPI", generate_path=True).save().absolutepath


        ftmp_seg_, ftmp_seg = ftmp_seg, add_suffix(ftmp_seg, '_crop')
        if level_alignment:
            # cropping the segmentation based on the label coverage to ensure good registration with level alignment
            # See https://github.com/neuropoly/spinalcordtoolbox/pull/1669 for details
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
        sct.printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)

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
        cache_sig = sct.cache_signature(
         input_files=cache_input_files,
        )
        cachefile = os.path.join(curdir, "straightening.cache")
        if sct.cache_valid(cachefile, cache_sig) and os.path.isfile(fn_warp_curve2straight) and os.path.isfile(fn_warp_straight2curve) and os.path.isfile(fn_straight_ref):
            sct.printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
            sct.copy(fn_warp_curve2straight, 'warp_curve2straight.nii.gz')
            sct.copy(fn_warp_straight2curve, 'warp_straight2curve.nii.gz')
            sct.copy(fn_straight_ref, 'straight_ref.nii.gz')
            # apply straightening
            sct_apply_transfo.main(args=[
                '-i', ftmp_seg,
                '-w', 'warp_curve2straight.nii.gz',
                '-d', 'straight_ref.nii.gz',
                '-o', add_suffix(ftmp_seg, '_straight')])
        else:
            from spinalcordtoolbox.straightening import SpinalCordStraightener
            sc_straight = SpinalCordStraightener(ftmp_seg, ftmp_seg)
            sc_straight.param_centerline = param_centerline
            sc_straight.output_filename = add_suffix(ftmp_seg, '_straight')
            sc_straight.path_output = './'
            sc_straight.qc = '0'
            sc_straight.remove_temp_files = param.remove_temp_files
            sc_straight.verbose = verbose

            if level_alignment:
                sc_straight.centerline_reference_filename = ftmp_template_seg
                sc_straight.use_straight_reference = True
                sc_straight.discs_input_filename = ftmp_label
                sc_straight.discs_ref_filename = ftmp_template_label

            sc_straight.straighten()
            sct.cache_save(cachefile, cache_sig)

        # N.B. DO NOT UPDATE VARIABLE ftmp_seg BECAUSE TEMPORARY USED LATER
        # re-define warping field using non-cropped space (to avoid issue #367)
        sct_concat_transfo.main(args=[
            '-w', 'warp_straight2curve.nii.gz',
            '-d', ftmp_data,
            '-o', 'warp_straight2curve.nii.gz'])

        if level_alignment:
            sct.copy('warp_curve2straight.nii.gz', 'warp_curve2straightAffine.nii.gz')
        else:
            # Label preparation:
            # --------------------------------------------------------------------------------
            # Remove unused label on template. Keep only label present in the input label image
            sct.printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
            sct.run(['sct_label_utils', '-i', ftmp_template_label, '-o', ftmp_template_label, '-remove-reference', ftmp_label])

            # Dilating the input label so they can be straighten without losing them
            sct.printv('\nDilating input labels using 3vox ball radius')
            dilate(Image(ftmp_label), 3, 'ball').save(add_suffix(ftmp_label, '_dilate'))
            ftmp_label = add_suffix(ftmp_label, '_dilate')

            # Apply straightening to labels
            sct.printv('\nApply straightening to labels...', verbose)
            sct_apply_transfo.main(args=[
                '-i', ftmp_label,
                '-o', add_suffix(ftmp_label, '_straight'),
                '-d', add_suffix(ftmp_seg, '_straight'),
                '-w', 'warp_curve2straight.nii.gz',
                '-x', 'nn'])
            ftmp_label = add_suffix(ftmp_label, '_straight')

            # Compute rigid transformation straight landmarks --> template landmarks
            sct.printv('\nEstimate transformation for step #0...', verbose)
            try:
                register_landmarks(ftmp_label, ftmp_template_label, paramregmulti.steps['0'].dof,
                                   fname_affine='straight2templateAffine.txt', verbose=verbose)
            except RuntimeError:
                raise('Input labels do not seem to be at the right place. Please check the position of the labels. '
                      'See documentation for more details: https://www.slideshare.net/neuropoly/sct-course-20190121/42')

            # Concatenate transformations: curve --> straight --> affine
            sct.printv('\nConcatenate transformations: curve --> straight --> affine...', verbose)
            sct_concat_transfo.main(args=[
                '-w', ['warp_curve2straight.nii.gz', 'straight2templateAffine.txt'],
                '-d', 'template.nii',
                '-o', 'warp_curve2straightAffine.nii.gz'])

        # Apply transformation
        sct.printv('\nApply transformation...', verbose)
        sct_apply_transfo.main(args=[
            '-i', ftmp_data,
            '-o', add_suffix(ftmp_data, '_straightAffine'),
            '-d', ftmp_template,
            '-w', 'warp_curve2straightAffine.nii.gz'])
        ftmp_data = add_suffix(ftmp_data, '_straightAffine')
        sct_apply_transfo.main(args=[
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
        im_new.save(add_suffix(ftmp_seg, '_bin')) # unused?
        # crop template in z-direction (for faster processing)
        # TODO: refactor to use python module instead of doing i/o
        sct.printv('\nCrop data in template space (for faster processing)...', verbose)
        ftmp_template_, ftmp_template = ftmp_template, add_suffix(ftmp_template, '_crop')
        msct_image.spatial_crop(Image(ftmp_template_), dict(((2, (zmin_template,zmax_template)),))).save(ftmp_template)

        ftmp_template_seg_, ftmp_template_seg = ftmp_template_seg, add_suffix(ftmp_template_seg, '_crop')
        msct_image.spatial_crop(Image(ftmp_template_seg_), dict(((2, (zmin_template,zmax_template)),))).save(ftmp_template_seg)

        ftmp_data_, ftmp_data = ftmp_data, add_suffix(ftmp_data, '_crop')
        msct_image.spatial_crop(Image(ftmp_data_), dict(((2, (zmin_template,zmax_template)),))).save(ftmp_data)

        ftmp_seg_, ftmp_seg = ftmp_seg, add_suffix(ftmp_seg, '_crop')
        msct_image.spatial_crop(Image(ftmp_seg_), dict(((2, (zmin_template,zmax_template)),))).save(ftmp_seg)

        # sub-sample in z-direction
        # TODO: refactor to use python module instead of doing i/o
        sct.printv('\nSub-sample in z-direction (for faster processing)...', verbose)
        sct.run(['sct_resample', '-i', ftmp_template, '-o', add_suffix(ftmp_template, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_template = add_suffix(ftmp_template, '_sub')
        sct.run(['sct_resample', '-i', ftmp_template_seg, '-o', add_suffix(ftmp_template_seg, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_template_seg = add_suffix(ftmp_template_seg, '_sub')
        sct.run(['sct_resample', '-i', ftmp_data, '-o', add_suffix(ftmp_data, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_data = add_suffix(ftmp_data, '_sub')
        sct.run(['sct_resample', '-i', ftmp_seg, '-o', add_suffix(ftmp_seg, '_sub'), '-f', '1x1x' + zsubsample], verbose)
        ftmp_seg = add_suffix(ftmp_seg, '_sub')

        # Registration straight spinal cord to template
        sct.printv('\nRegister straight spinal cord to template...', verbose)

        # TODO: find a way to input initwarp, corresponding to straightening warp
        # Set the angle of the template orientation to 0 (destination image)
        for key in list(paramregmulti.steps.keys()) :
            paramregmulti.steps[key].rot_dest = 0
        fname_src2dest, fname_dest2src, warp_forward, warp_inverse = register_wrapper(
            ftmp_data, ftmp_template, param, paramregmulti, fname_src_seg=ftmp_seg, fname_dest_seg=ftmp_template_seg,
            same_space=True)

        # Concatenate transformations: anat --> template
        sct.printv('\nConcatenate transformations: anat --> template...', verbose)
        sct_concat_transfo.main(args=[
            '-w', ['warp_curve2straightAffine.nii.gz', warp_forward],
            '-d', 'template.nii',
            '-o', 'warp_anat2template.nii.gz'])

        # Concatenate transformations: template --> anat
        sct.printv('\nConcatenate transformations: template --> anat...', verbose)
        # TODO: make sure the commented code below is consistent with the new implementation
        # warp_inverse.reverse()
        if level_alignment:
            sct_concat_transfo.main(args=[
                '-w', [warp_inverse, 'warp_straight2curve.nii.gz'],
                '-d', 'data.nii',
                '-o', 'warp_template2anat.nii.gz'])
        else:
            sct_concat_transfo.main(args=[
                '-w', [warp_inverse, 'straight2templateAffine.txt', 'warp_straight2curve.nii.gz'],
                '-winv', ['straight2templateAffine.txt'],
                '-d', 'data.nii',
                '-o', 'warp_template2anat.nii.gz'])

    # register template->subject
    elif ref == 'subject':

        # Change orientation of input images to RPI
        sct.printv('\nChange orientation of input images to RPI...', verbose)
        ftmp_data = Image(ftmp_data).change_orientation("RPI", generate_path=True).save().absolutepath
        ftmp_seg = Image(ftmp_seg).change_orientation("RPI", generate_path=True).save().absolutepath
        ftmp_label = Image(ftmp_label).change_orientation("RPI", generate_path=True).save().absolutepath

        # Remove unused label on template. Keep only label present in the input label image
        sct.printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
        sct.run(['sct_label_utils', '-i', ftmp_template_label, '-o', ftmp_template_label, '-remove-reference', ftmp_label])

        # Add one label because at least 3 orthogonal labels are required to estimate an affine transformation. This
        # new label is added at the level of the upper most label (lowest value), at 1cm to the right.
        for i_file in [ftmp_label, ftmp_template_label]:
            im_label = Image(i_file)
            coord_label = im_label.getCoordinatesAveragedByValue()  # N.B. landmarks are sorted by value
            # Create new label
            from copy import deepcopy
            new_label = deepcopy(coord_label[0])
            # move it 5mm to the left (orientation is RAS)
            nx, ny, nz, nt, px, py, pz, pt = im_label.dim
            new_label.x = np.round(coord_label[0].x + 5.0 / px)
            # assign value 99
            new_label.value = 99
            # Add to existing image
            im_label.data[int(new_label.x), int(new_label.y), int(new_label.z)] = new_label.value
            # Overwrite label file
            # im_label.absolutepath = 'label_rpi_modif.nii.gz'
            im_label.save()
        # Set the angle of the template orientation to 0 (source image)
        for key in list(paramregmulti.steps.keys()) :
            paramregmulti.steps[key].rot_src = 0
        fname_src2dest, fname_dest2src, warp_forward, warp_inverse = register_wrapper(
            ftmp_template, ftmp_data, param, paramregmulti, fname_src_seg=ftmp_template_seg, fname_dest_seg=ftmp_seg,
            fname_src_label=ftmp_template_label, fname_dest_label=ftmp_label, same_space=False)
        # Renaming for code compatibility
        os.rename(warp_forward, 'warp_template2anat.nii.gz')
        os.rename(warp_inverse, 'warp_anat2template.nii.gz')

    # Apply warping fields to anat and template
    sct.run(['sct_apply_transfo', '-i', 'template.nii', '-o', 'template2anat.nii.gz', '-d', 'data.nii', '-w', 'warp_template2anat.nii.gz', '-crop', '0'], verbose)
    sct.run(['sct_apply_transfo', '-i', 'data.nii', '-o', 'anat2template.nii.gz', '-d', 'template.nii', '-w', 'warp_anat2template.nii.gz', '-crop', '0'], verbose)

    # come back
    os.chdir(curdir)

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    fname_template2anat = os.path.join(path_output, 'template2anat' + ext_data)
    fname_anat2template = os.path.join(path_output, 'anat2template' + ext_data)
    sct.generate_output_file(os.path.join(path_tmp, "warp_template2anat.nii.gz"), os.path.join(path_output, "warp_template2anat.nii.gz"), verbose)
    sct.generate_output_file(os.path.join(path_tmp, "warp_anat2template.nii.gz"), os.path.join(path_output, "warp_anat2template.nii.gz"), verbose)
    sct.generate_output_file(os.path.join(path_tmp, "template2anat.nii.gz"), fname_template2anat, verbose)
    sct.generate_output_file(os.path.join(path_tmp, "anat2template.nii.gz"), fname_anat2template, verbose)
    if ref == 'template':
        # copy straightening files in case subsequent SCT functions need them
        sct.generate_output_file(os.path.join(path_tmp, "warp_curve2straight.nii.gz"), os.path.join(path_output, "warp_curve2straight.nii.gz"), verbose)
        sct.generate_output_file(os.path.join(path_tmp, "warp_straight2curve.nii.gz"), os.path.join(path_output, "warp_straight2curve.nii.gz"), verbose)
        sct.generate_output_file(os.path.join(path_tmp, "straight_ref.nii.gz"), os.path.join(path_output, "straight_ref.nii.gz"), verbose)

    # Delete temporary files
    if param.remove_temp_files:
        sct.printv('\nDelete temporary files...', verbose)
        sct.rmtree(path_tmp, verbose=verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', verbose)

    qc_dataset = arguments.get("-qc-dataset", None)
    qc_subject = arguments.get("-qc-subject", None)
    if param.path_qc is not None:
        generate_qc(fname_data, fname_in2=fname_template2anat, fname_seg=fname_seg, args=args,
                    path_qc=os.path.abspath(param.path_qc), dataset=qc_dataset, subject=qc_subject,
                    process='sct_register_to_template')
    sct.display_viewer_syntax([fname_data, fname_template2anat], verbose=verbose)
    sct.display_viewer_syntax([fname_template, fname_anat2template], verbose=verbose)


def project_labels_on_spinalcord(fname_label, fname_seg, param_centerline):
    """
    Project labels orthogonally on the spinal cord centerline. The algorithm works by finding the smallest distance
    between each label and the spinal cord center of mass.
    :param fname_label: file name of labels
    :param fname_seg: file name of cord segmentation (could also be of centerline)
    :return: file name of projected labels
    """
    # build output name
    fname_label_projected = sct.add_suffix(fname_label, "_projected")
    # open labels and segmentation
    im_label = Image(fname_label).change_orientation("RPI")
    im_seg = Image(fname_seg)
    native_orient = im_seg.orientation
    im_seg.change_orientation("RPI")

    # smooth centerline and return fitted coordinates in voxel space
    _, arr_ctl, _, _ = get_centerline(im_seg, param_centerline)
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
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_labels).dim
    nxd, nyd, nzd, ntd, pxd, pyd, pzd, ptd = Image(fname_dest).dim
    sampling_factor = [float(nx) / nxd, float(ny) / nyd, float(nz) / nzd]
    # read labels
    processor = sct_label_utils.ProcessLabels(fname_labels)
    label_list = processor.display_voxel()
    label_new_list = []
    for label in label_list:
        label_sub_new = [str(int(np.round(int(label.x) / sampling_factor[0]))),
                         str(int(np.round(int(label.y) / sampling_factor[1]))),
                         str(int(np.round(int(label.z) / sampling_factor[2]))),
                         str(int(float(label.value)))]
        label_new_list.append(','.join(label_sub_new))
    label_new_list = ':'.join(label_new_list)
    # create new labels
    sct.run(['sct_label_utils', '-i', fname_dest, '-create', label_new_list, '-v', '1', '-o', fname_output])


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
    sct.printv('\nCheck input labels...')
    # open label file
    image_label = Image(fname_landmarks)
    # -> all labels must be different
    labels = image_label.getNonZeroCoordinates(sorting='value')
    # check if there is two labels
    if label_type == 'body' and not len(labels) <= 2:
        sct.printv('ERROR: Label file has ' + str(len(labels)) + ' label(s). It must contain one or two labels.', 1,
                   'error')
    # check if labels are integer
    for label in labels:
        if not int(label.value) == label.value:
            sct.printv('ERROR: Label should be integer.', 1, 'error')
    # check if there are duplicates in label values
    n_labels = len(labels)
    list_values = [labels[i].value for i in range(0,n_labels)]
    list_duplicates = [x for x in list_values if list_values.count(x) > 1]
    if not list_duplicates == []:
        sct.printv('ERROR: Found two labels with same value.', 1, 'error')
    return labels


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
