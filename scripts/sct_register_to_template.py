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

# TODO: testing script for all cases

import sys
import os
import commands
import time
from glob import glob
import sct_utils as sct
from sct_utils import add_suffix
from sct_image import set_orientation
from sct_register_multimodal import Paramreg, ParamregMultiStep, register
from msct_parser import Parser
from msct_image import Image, find_zmin_zmax
from shutil import move
from sct_label_utils import ProcessLabels
import numpy as np


# get path of the toolbox
path_script = os.path.dirname(__file__)
path_sct = os.path.dirname(path_script)

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1  # remove temporary files
        self.fname_mask = ''  # this field is needed in the function register@sct_register_multimodal
        self.padding = 10  # this field is needed in the function register@sct_register_multimodal
        # self.speed = 'fast'  # speed of registration. slow | normal | fast
        # self.nb_iterations = '5'
        # self.algo = 'SyN'
        # self.gradientStep = '0.5'
        # self.metric = 'MI'
        self.verbose = 1  # verbose
        # self.folder_template = 'template/'  # folder where template files are stored (MNI-Poly-AMU_T2.nii.gz, etc.)
        self.path_template = path_sct+'/data/PAM50'
        # self.file_template_label = 'landmarks_center.nii.gz'
        self.zsubsample = '0.25'
        self.param_straighten = ''
        # self.smoothing_sigma = 5  # Smoothing along centerline to improve accuracy and remove step effects


# get default parameters
step1 = Paramreg(step='1', type='seg', algo='rigid', metric='MeanSquares', slicewise='1', smooth='5')
step2 = Paramreg(step='2', type='seg', algo='bsplinesyn', metric='MeanSquares', iter='5', smooth='1')
step3 = Paramreg(step='3', type='im', algo='syn', metric='CC', iter='3')
paramreg = ParamregMultiStep([step1, step2, step3])


# PARSER
# ==========================================================================================
def get_parser():
    param = Param()
    parser = Parser(__file__)
    parser.usage.set_description('Register anatomical image to the template.')
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
                      description="Labels. See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/",
                      mandatory=True,
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
                      example=['t1', 't2'])
    parser.add_option(name="-param",
                      type_value=[[':'], 'str'],
                      description='Parameters for registration (see sct_register_multimodal). Default: \
                      \n--\nstep=1\ntype=' + paramreg.steps['1'].type + '\nalgo=' + paramreg.steps['1'].algo + '\nmetric=' + paramreg.steps['1'].metric + '\niter=' + paramreg.steps['1'].iter + '\nsmooth=' + paramreg.steps['1'].smooth + '\ngradStep=' + paramreg.steps['1'].gradStep + '\nslicewise=' + paramreg.steps['1'].slicewise + '\
                      \n--\nstep=2\ntype=' + paramreg.steps['2'].type + '\nalgo=' + paramreg.steps['2'].algo + '\nmetric=' + paramreg.steps['2'].metric + '\niter=' + paramreg.steps['2'].iter + '\nsmooth=' + paramreg.steps['2'].smooth + '\ngradStep=' + paramreg.steps['2'].gradStep + '\
                      \n--\nstep=3\ntype=' + paramreg.steps['3'].type + '\nalgo=' + paramreg.steps['3'].algo + '\nmetric=' + paramreg.steps['3'].metric + '\niter=' + paramreg.steps['3'].iter + '\nsmooth=' + paramreg.steps['3'].smooth + '\ngradStep=' + paramreg.steps['3'].gradStep + '\n',
                      mandatory=False,
                      example="step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=5,shrink=2:step=3,type=im,algo=syn,metric=MI,iter=5,shrink=1,gradStep=0.3")
    parser.add_option(name="-p",
                      type_value=None,
                      description='Parameters for registration (see sct_register_multimodal). Default: \
                      \n--\nstep=1\ntype=' + paramreg.steps['1'].type + '\nalgo=' + paramreg.steps['1'].algo + '\nmetric=' + paramreg.steps['1'].metric + '\iter=' + paramreg.steps['1'].iter + '\smooth=' + paramreg.steps['1'].smooth + '\gradStep=' + paramreg.steps['1'].gradStep + '\slicewise=' + paramreg.steps['1'].slicewise + '\
                      \n--\nstep=1\ntype=' + paramreg.steps['2'].type + '\nalgo=' + paramreg.steps['2'].algo + '\nmetric=' + paramreg.steps['2'].metric + '\iter=' + paramreg.steps['2'].iter + '\smooth=' + paramreg.steps['2'].smooth + '\gradStep=' + paramreg.steps['2'].gradStep + '\slicewise=' + paramreg.steps['2'].slicewise + '\
                      \n--\nstep=1\ntype=' + paramreg.steps['3'].type + '\nalgo=' + paramreg.steps['3'].algo + '\nmetric=' + paramreg.steps['3'].metric + '\iter=' + paramreg.steps['3'].iter + '\smooth=' + paramreg.steps['3'].smooth + '\gradStep=' + paramreg.steps['3'].gradStep + '\slicewise=' + paramreg.steps['3'].slicewise + '\n',
                      mandatory=False,
                      deprecated_by='-param')
    parser.add_option(name="-param-straighten",
                      type_value='str',
                      description="""Parameters for straightening (see sct_straighten_spinalcord).""",
                      mandatory=False,
                      default_value='')
    # parser.add_option(name="-cpu-nb",
    #                   type_value="int",
    #                   description="Number of CPU used for straightening. 0: no multiprocessing. By default, uses all the available cores.",
    #                   mandatory=False,
    #                   example="8")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
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
def main():
    parser = get_parser()
    param = Param()

    arguments = parser.parse(sys.argv[1:])

    # get arguments
    fname_data = arguments['-i']
    fname_seg = arguments['-s']
    fname_landmarks = arguments['-l']
    if '-ofolder' in arguments:
        path_output = arguments['-ofolder']
    else:
        path_output = ''
    path_template = sct.slash_at_the_end(arguments['-t'], 1)
    contrast_template = arguments['-c']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])
    param.verbose = verbose  # TODO: not clean, unify verbose or param.verbose in code, but not both
    if '-param-straighten' in arguments:
        param.param_straighten = arguments['-param-straighten']
    # if '-cpu-nb' in arguments:
    #     arg_cpu = ' -cpu-nb '+str(arguments['-cpu-nb'])
    # else:
    #     arg_cpu = ''
    if '-param' in arguments:
        paramreg_user = arguments['-param']
        # update registration parameters
        for paramStep in paramreg_user:
            paramreg.addStep(paramStep)

    # initialize other parameters
    # file_template_label = param.file_template_label
    zsubsample = param.zsubsample
    template = os.path.basename(os.path.normpath(path_template))
    # smoothing_sigma = param.smoothing_sigma

    # retrieve template file names
    from sct_warp_template import get_file_label
    file_template_vertebral_labeling = get_file_label(path_template+'template/', 'vertebral')
    file_template = get_file_label(path_template+'template/', contrast_template.upper()+'-weighted')
    file_template_seg = get_file_label(path_template+'template/', 'spinal cord')

    # # adjust file names for old versions of template
    # if any(substring in path_template for substring in ['MNI-Poly-AMU', 'sct_testing_data']):
    #     # template name
    #     contrast_template = contrast_template.upper()
    #     # label name
    #     file_template_label = 'landmarks_center.nii.gz'
    # else:
    #     file_template_label = template + '_label_body.nii.gz'
    #
    # # retrieve file_template based on contrast
    # try:
    #     fname_template_list = glob(path_template + 'template/*' + contrast_template + '.nii.gz')
    #     fname_template = fname_template_list[0]
    # except IndexError:
    #     sct.printv('\nERROR: No template found in '+path_template+'template/*'+contrast_template+'.nii.gz', 1, 'error')
    #
    # # retrieve file_template_seg
    # try:
    #     fname_template_seg_list = glob(path_template + 'template/*cord.nii.gz')
    #     fname_template_seg = fname_template_seg_list[0]
    # except IndexError:
    #     sct.printv('\nERROR: No template cord segmentation found. Please check the provided path.', 1, 'error')

    # start timer
    start_time = time.time()

    # get absolute path - TO DO: remove! NEVER USE ABSOLUTE PATH...
    # path_template = os.path.abspath(path_template+'template/')

    # get fname of the template + template objects
    fname_template = path_template+'template/'+file_template
    fname_template_vertebral_labeling = path_template+'template/'+file_template_vertebral_labeling
    fname_template_seg = path_template+'template/'+file_template_seg

    # check file existence
    # TODO: no need to do that!
    sct.printv('\nCheck template files...')
    sct.check_file_exist(fname_template, verbose)
    sct.check_file_exist(fname_template_vertebral_labeling, verbose)
    sct.check_file_exist(fname_template_seg, verbose)

    # print arguments
    sct.printv('\nCheck parameters:', verbose)
    sct.printv('.. Data:                 '+fname_data, verbose)
    sct.printv('.. Landmarks:            '+fname_landmarks, verbose)
    sct.printv('.. Segmentation:         '+fname_seg, verbose)
    sct.printv('.. Path template:        '+path_template, verbose)
    sct.printv('.. Remove temp files:    '+str(remove_temp_files), verbose)

    sct.printv('\nParameters for registration:')
    for pStep in range(1, len(paramreg.steps)+1):
        sct.printv('Step #'+paramreg.steps[str(pStep)].step, verbose)
        sct.printv('.. Type #'+paramreg.steps[str(pStep)].type, verbose)
        sct.printv('.. Algorithm................ '+paramreg.steps[str(pStep)].algo, verbose)
        sct.printv('.. Metric................... '+paramreg.steps[str(pStep)].metric, verbose)
        sct.printv('.. Number of iterations..... '+paramreg.steps[str(pStep)].iter, verbose)
        sct.printv('.. Shrink factor............ '+paramreg.steps[str(pStep)].shrink, verbose)
        sct.printv('.. Smoothing factor......... '+paramreg.steps[str(pStep)].smooth, verbose)
        sct.printv('.. Gradient step............ '+paramreg.steps[str(pStep)].gradStep, verbose)
        sct.printv('.. Degree of polynomial..... '+paramreg.steps[str(pStep)].poly, verbose)

    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    sct.printv('\nCheck if data, segmentation and landmarks are in the same space...')
    if not sct.check_if_same_space(fname_data, fname_seg):
        sct.printv('ERROR: Data image and segmentation are not in the same space. Please check space and orientation of your files', verbose, 'error')
    if not sct.check_if_same_space(fname_data, fname_landmarks):
        sct.printv('ERROR: Data image and landmarks are not in the same space. Please check space and orientation of your files', verbose, 'error')

    sct.printv('\nCheck input labels...')
    # check if label image contains coherent labels
    image_label = Image(fname_landmarks)
    # -> all labels must be different
    labels = image_label.getNonZeroCoordinates(sorting='value')
    hasDifferentLabels = True
    for lab in labels:
        for otherlabel in labels:
            if lab != otherlabel and lab.hasEqualValue(otherlabel):
                hasDifferentLabels = False
                break
    if not hasDifferentLabels:
        sct.printv('ERROR: Wrong landmarks input. All labels must be different.', verbose, 'error')

    # create temporary folder
    path_tmp = sct.tmp_create(verbose=verbose)

    # set temporary file names
    ftmp_data = 'data.nii'
    ftmp_seg = 'seg.nii.gz'
    ftmp_label = 'label.nii.gz'
    ftmp_template = 'template.nii'
    ftmp_template_seg = 'template_seg.nii.gz'
    ftmp_template_label = 'template_label.nii.gz'

    # copy files to temporary folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.run('sct_convert -i '+fname_data+' -o '+path_tmp+ftmp_data)
    sct.run('sct_convert -i '+fname_seg+' -o '+path_tmp+ftmp_seg)
    sct.run('sct_convert -i '+fname_landmarks+' -o '+path_tmp+ftmp_label)
    sct.run('sct_convert -i '+fname_template+' -o '+path_tmp+ftmp_template)
    sct.run('sct_convert -i '+fname_template_seg+' -o '+path_tmp+ftmp_template_seg)
    # sct.run('sct_convert -i '+fname_template_label+' -o '+path_tmp+ftmp_template_label)

    # go to tmp folder
    os.chdir(path_tmp)

    # Generate labels from template vertebral labeling
    sct.printv('\nGenerate labels from template vertebral labeling', verbose)
    sct.run('sct_label_utils -i '+fname_template_vertebral_labeling+' -vert-body 0 -o '+ftmp_template_label)

    # check if provided labels are available in the template
    sct.printv('\nCheck if provided labels are available in the template', verbose)
    image_label_template = Image(ftmp_template_label)
    labels_template = image_label_template.getNonZeroCoordinates(sorting='value')
    if labels[-1].value > labels_template[-1].value:
        sct.printv('ERROR: Wrong landmarks input. Labels must have correspondence in template space. \nLabel max '
                   'provided: ' + str(labels[-1].value) + '\nLabel max from template: ' +
                   str(labels_template[-1].value), verbose, 'error')

    # smooth segmentation (jcohenadad, issue #613)
    sct.printv('\nSmooth segmentation...', verbose)
    # sct.run('sct_maths -i '+ftmp_seg+' -smooth 1.5 -o '+add_suffix(ftmp_seg, '_smooth'))
    # jcohenadad: updated 2016-06-16: DO NOT smooth the seg anymore. Issue #
    sct.run('sct_maths -i '+ftmp_seg+' -smooth 0 -o '+add_suffix(ftmp_seg, '_smooth'))
    ftmp_seg = add_suffix(ftmp_seg, '_smooth')

    # resample data to 1mm isotropic
    sct.printv('\nResample data to 1mm isotropic...', verbose)
    sct.run('sct_resample -i '+ftmp_data+' -mm 1.0x1.0x1.0 -x linear -o '+add_suffix(ftmp_data, '_1mm'))
    ftmp_data = add_suffix(ftmp_data, '_1mm')
    sct.run('sct_resample -i '+ftmp_seg+' -mm 1.0x1.0x1.0 -x linear -o '+add_suffix(ftmp_seg, '_1mm'))
    ftmp_seg = add_suffix(ftmp_seg, '_1mm')
    # N.B. resampling of labels is more complicated, because they are single-point labels, therefore resampling with neighrest neighbour can make them disappear. Therefore a more clever approach is required.
    resample_labels(ftmp_label, ftmp_data, add_suffix(ftmp_label, '_1mm'))
    ftmp_label = add_suffix(ftmp_label, '_1mm')

    # Change orientation of input images to RPI
    sct.printv('\nChange orientation of input images to RPI...', verbose)
    sct.run('sct_image -i '+ftmp_data+' -setorient RPI -o '+add_suffix(ftmp_data, '_rpi'))
    ftmp_data = add_suffix(ftmp_data, '_rpi')
    sct.run('sct_image -i '+ftmp_seg+' -setorient RPI -o '+add_suffix(ftmp_seg, '_rpi'))
    ftmp_seg = add_suffix(ftmp_seg, '_rpi')
    sct.run('sct_image -i '+ftmp_label+' -setorient RPI -o '+add_suffix(ftmp_label, '_rpi'))
    ftmp_label = add_suffix(ftmp_label, '_rpi')

    # get landmarks in native space
    # crop segmentation
    # output: segmentation_rpi_crop.nii.gz
    status_crop, output_crop = sct.run('sct_crop_image -i '+ftmp_seg+' -o '+add_suffix(ftmp_seg, '_crop')+' -dim 2 -bzmax', verbose)
    ftmp_seg = add_suffix(ftmp_seg, '_crop')
    cropping_slices = output_crop.split('Dimension 2: ')[1].split('\n')[0].split(' ')

    # straighten segmentation
    sct.printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)
    sct.run('sct_straighten_spinalcord -i '+ftmp_seg+' -s '+ftmp_seg+' -o '+add_suffix(ftmp_seg, '_straight')+' -qc 0 -r 0 -v '+str(verbose), verbose)
    # N.B. DO NOT UPDATE VARIABLE ftmp_seg BECAUSE TEMPORARY USED LATER
    # re-define warping field using non-cropped space (to avoid issue #367)
    sct.run('sct_concat_transfo -w warp_straight2curve.nii.gz -d '+ftmp_data+' -o warp_straight2curve.nii.gz')

    # Label preparation:
    # --------------------------------------------------------------------------------
    # Remove unused label on template. Keep only label present in the input label image
    sct.printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
    sct.run('sct_label_utils -i '+ftmp_template_label+' -o '+ftmp_template_label+' -remove '+ftmp_label)

    # Dilating the input label so they can be straighten without losing them
    sct.printv('\nDilating input labels using 3vox ball radius')
    sct.run('sct_maths -i '+ftmp_label+' -o '+add_suffix(ftmp_label, '_dilate')+' -dilate 3')
    ftmp_label = add_suffix(ftmp_label, '_dilate')

    # Apply straightening to labels
    sct.printv('\nApply straightening to labels...', verbose)
    sct.run('sct_apply_transfo -i '+ftmp_label+' -o '+add_suffix(ftmp_label, '_straight')+' -d '+add_suffix(ftmp_seg, '_straight')+' -w warp_curve2straight.nii.gz -x nn')
    ftmp_label = add_suffix(ftmp_label, '_straight')

    # Compute rigid transformation between straight landmarks and template landmarks
    sct.printv('\nComputing rigid transformation (algo=translation-scaling-z) ...', verbose)
    # open template label
    template_image = Image(ftmp_template_label)
    coordinates_input = template_image.getNonZeroCoordinates(sorting='value')
    # jcohenadad, issue #628 <<<<<
    # landmark_template = ProcessLabels.get_crosses_coordinates(coordinates_input, gapxy=15)
    landmark_template = coordinates_input
    # >>>>>
    # open data label
    label_straight_image = Image(ftmp_label)
    coordinates_input = label_straight_image.getCoordinatesAveragedByValue()  # landmarks are sorted by value
    # jcohenadad, issue #628 <<<<<
    # landmark_straight = ProcessLabels.get_crosses_coordinates(coordinates_input, gapxy=15)
    landmark_straight = coordinates_input
    # >>>>>
    # Reorganize landmarks
    points_fixed, points_moving = [], []
    for coord in landmark_straight:
        point_straight = label_straight_image.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        points_moving.append([point_straight[0][0], point_straight[0][1], point_straight[0][2]])
    for coord in landmark_template:
        point_template = template_image.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        points_fixed.append([point_template[0][0], point_template[0][1], point_template[0][2]])

    import msct_register_landmarks
    # for some reason, the moving and fixed points are inverted between ITK transform and our python-based transform.
    # and for another unknown reason, x and y dimensions have a negative sign (at least for translation and center of rotation).
    if verbose == 2:
        show_transfo = True
    else:
        show_transfo = False
    (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = msct_register_landmarks.getRigidTransformFromLandmarks(points_moving, points_fixed, constraints='translation-scaling-z', show=show_transfo)
    # writing rigid transformation file
    text_file = open("straight2templateAffine.txt", "w")
    text_file.write("#Insight Transform File V1.0\n")
    text_file.write("#Transform 0\n")
    text_file.write("Transform: AffineTransform_double_3_3\n")
    text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
        rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
        rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
        rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2],
        -translation_array[0, 0], -translation_array[0, 1], translation_array[0, 2]))
    text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (-points_moving_barycenter[0],
                                                           -points_moving_barycenter[1],
                                                           points_moving_barycenter[2]))
    text_file.close()

    # Concatenate transformations: curve --> straight --> affine
    sct.printv('\nConcatenate transformations: curve --> straight --> affine...', verbose)
    sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt -d template.nii -o warp_curve2straightAffine.nii.gz')

    # Apply transformation
    sct.printv('\nApply transformation...', verbose)
    sct.run('sct_apply_transfo -i '+ftmp_data+' -o '+add_suffix(ftmp_data, '_straightAffine')+' -d '+ftmp_template+' -w warp_curve2straightAffine.nii.gz')
    ftmp_data = add_suffix(ftmp_data, '_straightAffine')
    sct.run('sct_apply_transfo -i '+ftmp_seg+' -o '+add_suffix(ftmp_seg, '_straightAffine')+' -d '+ftmp_template+' -w warp_curve2straightAffine.nii.gz -x linear')
    ftmp_seg = add_suffix(ftmp_seg, '_straightAffine')

    """
    # Benjamin: Issue from Allan Martin, about the z=0 slice that is screwed up, caused by the affine transform.
    # Solution found: remove slices below and above landmarks to avoid rotation effects
    points_straight = []
    for coord in landmark_template:
        points_straight.append(coord.z)
    min_point, max_point = int(round(np.min(points_straight))), int(round(np.max(points_straight)))
    sct.run('sct_crop_image -i ' + ftmp_seg + ' -start ' + str(min_point) + ' -end ' + str(max_point) + ' -dim 2 -b 0 -o ' + add_suffix(ftmp_seg, '_black'))
    ftmp_seg = add_suffix(ftmp_seg, '_black')
    """

    # threshold and binarize
    sct.printv('\nBinarize segmentation...', verbose)
    sct.run('sct_maths -i '+ftmp_seg+' -thr 0.4 -o '+add_suffix(ftmp_seg, '_thr'))
    sct.run('sct_maths -i '+add_suffix(ftmp_seg, '_thr')+' -bin -o '+add_suffix(ftmp_seg, '_thr_bin'))
    ftmp_seg = add_suffix(ftmp_seg, '_thr_bin')

    # find min-max of anat2template (for subsequent cropping)
    zmin_template, zmax_template = find_zmin_zmax(ftmp_seg)

    # crop template in z-direction (for faster processing)
    sct.printv('\nCrop data in template space (for faster processing)...', verbose)
    sct.run('sct_crop_image -i '+ftmp_template+' -o '+add_suffix(ftmp_template, '_crop')+' -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    ftmp_template = add_suffix(ftmp_template, '_crop')
    sct.run('sct_crop_image -i '+ftmp_template_seg+' -o '+add_suffix(ftmp_template_seg, '_crop')+' -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    ftmp_template_seg = add_suffix(ftmp_template_seg, '_crop')
    sct.run('sct_crop_image -i '+ftmp_data+' -o '+add_suffix(ftmp_data, '_crop')+' -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    ftmp_data = add_suffix(ftmp_data, '_crop')
    sct.run('sct_crop_image -i '+ftmp_seg+' -o '+add_suffix(ftmp_seg, '_crop')+' -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    ftmp_seg = add_suffix(ftmp_seg, '_crop')

    # sub-sample in z-direction
    sct.printv('\nSub-sample in z-direction (for faster processing)...', verbose)
    sct.run('sct_resample -i '+ftmp_template+' -o '+add_suffix(ftmp_template, '_sub')+' -f 1x1x'+zsubsample, verbose)
    ftmp_template = add_suffix(ftmp_template, '_sub')
    sct.run('sct_resample -i '+ftmp_template_seg+' -o '+add_suffix(ftmp_template_seg, '_sub')+' -f 1x1x'+zsubsample, verbose)
    ftmp_template_seg = add_suffix(ftmp_template_seg, '_sub')
    sct.run('sct_resample -i '+ftmp_data+' -o '+add_suffix(ftmp_data, '_sub')+' -f 1x1x'+zsubsample, verbose)
    ftmp_data = add_suffix(ftmp_data, '_sub')
    sct.run('sct_resample -i '+ftmp_seg+' -o '+add_suffix(ftmp_seg, '_sub')+' -f 1x1x'+zsubsample, verbose)
    ftmp_seg = add_suffix(ftmp_seg, '_sub')

    # Registration straight spinal cord to template
    sct.printv('\nRegister straight spinal cord to template...', verbose)

    # loop across registration steps
    warp_forward = []
    warp_inverse = []
    for i_step in range(1, len(paramreg.steps)+1):
        sct.printv('\nEstimate transformation for step #'+str(i_step)+'...', verbose)
        # identify which is the src and dest
        if paramreg.steps[str(i_step)].type == 'im':
            src = ftmp_data
            dest = ftmp_template
            interp_step = 'linear'
        elif paramreg.steps[str(i_step)].type == 'seg':
            src = ftmp_seg
            dest = ftmp_template_seg
            interp_step = 'nn'
        else:
            sct.printv('ERROR: Wrong image type.', 1, 'error')
        # if step>1, apply warp_forward_concat to the src image to be used
        if i_step > 1:
            # sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+sct.add_suffix(src, '_reg')+' -x '+interp_step, verbose)
            # apply transformation from previous step, to use as new src for registration
            sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+add_suffix(src, '_regStep'+str(i_step-1))+' -x '+interp_step, verbose)
            src = add_suffix(src, '_regStep'+str(i_step-1))
        # register src --> dest
        # TODO: display param for debugging
        param.verbose
        warp_forward_out, warp_inverse_out = register(src, dest, paramreg, param, str(i_step))
        warp_forward.append(warp_forward_out)
        warp_inverse.append(warp_inverse_out)

    # Concatenate transformations:
    sct.printv('\nConcatenate transformations: anat --> template...', verbose)
    sct.run('sct_concat_transfo -w warp_curve2straightAffine.nii.gz,'+','.join(warp_forward)+' -d template.nii -o warp_anat2template.nii.gz', verbose)
    # sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt,'+','.join(warp_forward)+' -d template.nii -o warp_anat2template.nii.gz', verbose)
    sct.printv('\nConcatenate transformations: template --> anat...', verbose)
    warp_inverse.reverse()
    sct.run('sct_concat_transfo -w '+','.join(warp_inverse)+',-straight2templateAffine.txt,warp_straight2curve.nii.gz -d data.nii -o warp_template2anat.nii.gz', verbose)

    # Apply warping fields to anat and template
    sct.run('sct_apply_transfo -i template.nii -o template2anat.nii.gz -d data.nii -w warp_template2anat.nii.gz -crop 1', verbose)
    sct.run('sct_apply_transfo -i data.nii -o anat2template.nii.gz -d template.nii -w warp_anat2template.nii.gz -crop 1', verbose)

    # come back to parent folder
    os.chdir('..')

   # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'warp_template2anat.nii.gz', path_output+'warp_template2anat.nii.gz', verbose)
    sct.generate_output_file(path_tmp+'warp_anat2template.nii.gz', path_output+'warp_anat2template.nii.gz', verbose)
    sct.generate_output_file(path_tmp+'template2anat.nii.gz', path_output+'template2anat'+ext_data, verbose)
    sct.generate_output_file(path_tmp+'anat2template.nii.gz', path_output+'anat2template'+ext_data, verbose)

    # Delete temporary files
    if remove_temp_files:
        sct.printv('\nDelete temporary files...', verbose)
        sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)

    # to view results
    sct.printv('\nTo view results, type:', verbose)
    sct.printv('fslview '+fname_data+' '+path_output+'template2anat -b 0,4000 &', verbose, 'info')
    sct.printv('fslview '+fname_template+' -b 0,5000 '+path_output+'anat2template &\n', verbose, 'info')


# Resample labels
# ==========================================================================================
def resample_labels(fname_labels, fname_dest, fname_output):
    """
    This function re-create labels into a space that has been resampled. It works by re-defining the location of each
    label using the old and new voxel size.
    """
    # get dimensions of input and destination files
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_labels).dim
    nxd, nyd, nzd, ntd, pxd, pyd, pzd, ptd = Image(fname_dest).dim
    sampling_factor = [float(nx)/nxd, float(ny)/nyd, float(nz)/nzd]
    # read labels
    from sct_label_utils import ProcessLabels
    processor = ProcessLabels(fname_labels)
    label_list = processor.display_voxel()
    label_new_list = []
    for label in label_list:
        label_sub_new = [str(int(round(int(label.x)/sampling_factor[0]))),
                         str(int(round(int(label.y)/sampling_factor[1]))),
                         str(int(round(int(label.z)/sampling_factor[2]))),
                         str(int(float(label.value)))]
        label_new_list.append(','.join(label_sub_new))
    label_new_list = ':'.join(label_new_list)
    # create new labels
    sct.run('sct_label_utils -i '+fname_dest+' -create '+label_new_list+' -v 1 -o '+fname_output)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()
