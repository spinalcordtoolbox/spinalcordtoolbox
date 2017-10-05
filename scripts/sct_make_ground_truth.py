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

import sys
import os
import shutil
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
        self.verbose = 1  # verbose
        self.path_template = path_sct+'/data/PAM50'
        self.path_qc = os.path.abspath(os.curdir)+'/qc/'
        self.zsubsample = '0.25'
        self.param_straighten = ''


# get default parameters
# Note: step0 is used as pre-registration
step0 = Paramreg(step='0', type='label', dof='Tx_Ty_Tz_Sz')  # if ref=template, we only need translations and z-scaling because the cord is already straight
step1 = Paramreg(step='1', type='seg', algo='centermassrot', smooth='2')
# step2 = Paramreg(step='2', type='seg', algo='columnwise', smooth='0', smoothWarpXY='2')
step2 = Paramreg(step='2', type='seg', algo='bsplinesyn', metric='MeanSquares', iter='3', smooth='1')
# step3 = Paramreg(step='3', type='im', algo='syn', metric='CC', iter='1')
paramreg = ParamregMultiStep([step0, step1, step2])


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
                      mandatory=False,
                      example="anat_seg.nii.gz")
    parser.add_option(name="-l",
                      type_value="file",
                      description="Labels. See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels\n",
                      mandatory=False,
                      default_value='labels_ground_truth.nii.gz',
                      example="labels_ground_truth.nii.gz")
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
    parser.add_option(name='-first',
                      type_value='int',
                      description='Define the label from which you wish to start. You can choose among the following labels : {50,49,1,3,4,...,25}. \n ',
                      mandatory=False,
                      default_value=50,
                      example= 50)
    parser.add_option(name='-slice-to-mean',
                      type_value='int',
                      description='Define the number of slice you want to average.',
                      mandatory=False,
                      default_value=3,
                      example=2)
    parser.add_option(name="-param",
                      type_value=[[':'], 'str'],
                      description='Parameters for registration (see sct_register_multimodal). Default: \
                      \n--\nstep=0\ntype=' + paramreg.steps['0'].type + '\ndof=' + paramreg.steps['0'].dof + '\
                      \n--\nstep=1\ntype=' + paramreg.steps['1'].type + '\nalgo=' + paramreg.steps['1'].algo + '\nmetric=' + paramreg.steps['1'].metric + '\niter=' + paramreg.steps['1'].iter + '\nsmooth=' + paramreg.steps['1'].smooth + '\ngradStep=' + paramreg.steps['1'].gradStep + '\nslicewise=' + paramreg.steps['1'].slicewise + '\nsmoothWarpXY=' + paramreg.steps['1'].smoothWarpXY + '\npca_eigenratio_th=' + paramreg.steps['1'].pca_eigenratio_th + '\
                      \n--\nstep=2\ntype=' + paramreg.steps['2'].type + '\nalgo=' + paramreg.steps['2'].algo + '\nmetric=' + paramreg.steps['2'].metric + '\niter=' + paramreg.steps['2'].iter + '\nsmooth=' + paramreg.steps['2'].smooth + '\ngradStep=' + paramreg.steps['2'].gradStep + '\nslicewise=' + paramreg.steps['2'].slicewise + '\nsmoothWarpXY=' + paramreg.steps['2'].smoothWarpXY + '\npca_eigenratio_th=' + paramreg.steps['1'].pca_eigenratio_th,
                      mandatory=False)
    parser.add_option(name="-param-straighten",
                      type_value='str',
                      description="""Parameters for straightening (see sct_straighten_spinalcord).""",
                      mandatory=False,
                      default_value='')
    parser.add_option(name="-init-labels",
                      type_value="multiple_choice",
                      description="You can create your own labels using a interactive viewer using option 'viewer",
                      mandatory=False,
                      default_value='none',
                      example=['none', 'viewer'])
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

def rewrite_arguments(arguments):
    fname_data = arguments['-i']
    #fname_seg = arguments['-s']
    fname_landmarks = arguments['-l']
    if '-ofolder' in arguments:
        path_output = arguments['-ofolder']
    else:
        path_output = ''
    first_label=arguments['-first']
    path_template = sct.slash_at_the_end(arguments['-t'], 1)
    contrast_template = arguments['-c']
    ref = arguments['-ref']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])
    init_labels= correct_init_labels(arguments['-init-labels'])
    nb_slice_to_mean=correct_nb_slice_to_mean(int(arguments['-slice-to-mean']))

    return (fname_data,fname_landmarks,path_output,path_template,contrast_template,ref,remove_temp_files,verbose,init_labels,first_label,nb_slice_to_mean)

def correct_init_labels(s):
    if s=='viewer':
        return True
    else:
        return False

def correct_nb_slice_to_mean(i):
    if i == 0:
        return 1
    elif i%2==1:
        return i
    elif i%2 == 0:
        return i-1

def write_paramaters(arguments,param,ref,verbose):
    param.verbose = verbose
    if '-param-straighten' in arguments:
        param.param_straighten = arguments['-param-straighten']

    """
    if '-cpu-nb' in arguments:
         arg_cpu = ' -cpu-nb '+str(arguments['-cpu-nb'])
    else:
         arg_cpu = ''
     registration parameters
    """
    if '-param' in arguments:
        # reset parameters but keep step=0 (might be overwritten if user specified step=0)
        paramreg = ParamregMultiStep([step0])
        if ref == 'subject':
            paramreg.steps['0'].dof = 'Tx_Ty_Tz_Rx_Ry_Rz_Sz'
        # add user parameters
        for paramStep in arguments['-param']:
            paramreg.addStep(paramStep)
    else:
        paramreg = ParamregMultiStep([step0, step1, step2])
        # if ref=subject, initialize registration using different affine parameters
        if ref == 'subject':
            paramreg.steps['0'].dof = 'Tx_Ty_Tz_Rx_Ry_Rz_Sz'

    return (param,paramreg)

def check_do_files_exist(fname_template,fname_template_vertebral_labeling,fname_template_seg,verbose):
    # TODO: no need to do that!
    sct.printv('\nCheck template files...')
    sct.check_file_exist(fname_template, verbose)
    sct.check_file_exist(fname_template_vertebral_labeling, verbose)
    sct.check_file_exist(fname_template_seg, verbose)

def make_fname_of_templates(file_template,path_template,file_template_vertebral_labeling,file_template_seg):
    fname_template = path_template+'template/'+file_template
    fname_template_vertebral_labeling = path_template+'template/'+file_template_vertebral_labeling
    fname_template_seg = path_template+'template/'+file_template_seg
    return(fname_template,fname_template_vertebral_labeling,fname_template_seg)

def sct.printv(arguments(verbose,fname_data,fname_landmarks,fname_seg,path_template,remove_temp_files):)
    sct.printv('\nCheck parameters:', verbose)
    sct.printv('  Data:                 '+fname_data, verbose)
    sct.printv('  Landmarks:            '+fname_landmarks, verbose)
    sct.printv('  Segmentation:         '+fname_seg, verbose)
    sct.printv('  Path template:        '+path_template, verbose)
    sct.printv('  Remove temp files:    '+str(remove_temp_files), verbose)

def check_data_segmentation_landmarks_same_space(fname_data,fname_seg,fname_landmarks,verbose):
    sct.printv('\nCheck if data, segmentation and landmarks are in the same space...')
    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    if not sct.check_if_same_space(fname_data, fname_seg):
        sct.printv('ERROR: Data image and segmentation are not in the same space. Please check space and orientation of your files', verbose, 'error')
    if not sct.check_if_same_space(fname_data, fname_landmarks):
        sct.printv('ERROR: Data image and landmarks are not in the same space. Please check space and orientation of your files', verbose, 'error')
    return (ext_data,path_data,file_data)

def set_temporary_files():
    ftmp_data = 'data.nii'
    ftmp_seg = 'seg.nii.gz'
    ftmp_label = 'label.nii.gz'
    ftmp_template = 'template.nii'
    ftmp_template_seg = 'template_seg.nii.gz'
    ftmp_template_label = 'template_label.nii.gz'
    return(ftmp_data,ftmp_seg,ftmp_label,ftmp_template,ftmp_template_seg,ftmp_template_label)

def copy_files_to_temporary_files(verbose,fname_data,path_tmp,ftmp_seg,ftmp_data,fname_seg,fname_landmarks,ftmp_label,fname_template,ftmp_template,fname_template_seg,ftmp_template_seg):
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.run('sct_convert -i '+fname_data+' -o '+path_tmp+ftmp_data)
    sct.run('sct_convert -i '+fname_seg+' -o '+path_tmp+ftmp_seg)
    sct.run('sct_convert -i '+fname_landmarks+' -o '+path_tmp+ftmp_label)
    sct.run('sct_convert -i '+fname_template+' -o '+path_tmp+ftmp_template)
    sct.run('sct_convert -i '+fname_template_seg+' -o '+path_tmp+ftmp_template_seg)
    # sct.run('sct_convert -i '+fname_template_label+' -o '+path_tmp+ftmp_template_label)

def set_viewer_parameters(viewer,nb_slices_to_mean):
    viewer.number_of_slices = 1
    pz = 1
    viewer.gap_inter_slice = int(10 / pz)
    viewer.calculate_list_slices()
    viewer.number_of_slices_to_mean=nb_slices_to_mean
    viewer.show_image_mean()

def prepare_input_image_for_viewer(fname_data):
    # reorient image to SAL to be compatible with viewer
    im_input = Image(fname_data)
    im_input_SAL = im_input.copy()
    im_input_SAL.change_orientation('SAL')
    return(im_input_SAL)

def check_mask_point_not_empty(mask_points):
    if(mask_points):
        return True
    else:
        sct.printv('\nERROR: the viewer has been closed before entering all manual points. Please try again.', 1,
                   type='error')
        return False

def make_labels_image_from_list_points(mask_points,reoriented_image_filename,image_input_orientation):
    if check_mask_point_not_empty(mask_points):
        import sct_image
        # create the mask containing either the three-points or centerline mask for initialization
        sct.run("sct_label_utils -i " + reoriented_image_filename + " -create " + mask_points ,verbose=False)
        sct.run('sct_image -i ' + 'labels.nii.gz'+ ' -o ' + 'labels_ground_truth.nii.gz' + ' -setorient ' + image_input_orientation + ' -v 0',verbose=False)
        sct.run('rm -rf ' + 'labels.nii.gz')

def use_viewer_to_define_labels(fname_data,first_label,nb_of_slices_to_mean):
    from sct_viewer import ClickViewerGroundTruth
    from msct_image import Image
    import sct_image

    image_input = Image(fname_data)

    image_input_orientation = sct_image.orientation(image_input, get=True, verbose=False)
    reoriented_image_filename = 'reoriented_image_source.nii.gz'
    path_tmp_viewer = sct.tmp_create(verbose=False)
    cmd_image = 'sct_image -i "%s" -o "%s" -setorient SAL -v 0' % (
    fname_data, reoriented_image_filename)
    sct.run(cmd_image, verbose=False)


    im_input_SAL=prepare_input_image_for_viewer(fname_data)
    viewer = ClickViewerGroundTruth(im_input_SAL, first_label,orientation_subplot=['sag', 'ax'])
    set_viewer_parameters(viewer,nb_of_slices_to_mean)

    mask_points = viewer.start()
    if not mask_points and viewer.closed:
        mask_points = viewer.list_points_useful_notation
    make_labels_image_from_list_points(mask_points,reoriented_image_filename,image_input_orientation)



# MAIN
# ==========================================================================================
def main():
    parser = get_parser()
    param = Param()

    """ Rewrite arguments and set parameters"""
    arguments = parser.parse(sys.argv[1:])
    (fname_data, fname_landmarks, path_output, path_template, contrast_template, ref, remove_temp_files,
     verbose, init_labels, first_label,nb_slice_to_mean)=rewrite_arguments(arguments)
    (param, paramreg)=write_paramaters(arguments,param,ref,verbose)

    if(init_labels):
        use_viewer_to_define_labels(fname_data,first_label,nb_slice_to_mean)
    # initialize other parameters
    # file_template_label = param.file_template_label
    zsubsample = param.zsubsample
    template = os.path.basename(os.path.normpath(pth_template))
    # smoothing_sigma = param.smoothing_sigma

    # retrieve template file names

    from sct_warp_template import get_file_label
    file_template_vertebral_labeling = get_file_label(path_template+'template/', 'vertebral')
    file_template = get_file_label(path_template+'template/', contrast_template.upper()+'-weighted')
    file_template_seg = get_file_label(path_template+'template/', 'spinal cord')


    """ Start timer"""
    start_time = time.time()

    """ Manage file of templates"""
    (fname_template, fname_template_vertebral_labeling, fname_template_seg)=make_fname_of_templates(file_template,path_template,file_template_vertebral_labeling,file_template_seg)
    check_do_files_exist(fname_template,fname_template_vertebral_labeling,fname_template_seg,verbose)
    sct.printv(arguments(verbose, fname_data, fname_landmarks, fname_seg, path_template, remove_temp_files))

    """ Create QC folder """
    sct.create_folder(param.path_qc)

    """ Check if data, segmentation and landmarks are in the same space"""
    (ext_data, path_data, file_data)=check_data_segmentation_landmarks_same_space(fname_data, fname_seg, fname_landmarks,verbose)

    ''' Check input labels'''
    labels = check_labels(fname_landmarks)

    """ Create temporary folder, set temporary file names, copy files into it and go in it """
    path_tmp = sct.tmp_create(verbose=verbose)
    (ftmp_data, ftmp_seg, ftmp_label, ftmp_template, ftmp_template_seg, ftmp_template_label)=set_temporary_files()
    copy_files_to_temporary_files(verbose, fname_data, path_tmp, ftmp_seg, ftmp_data, fname_seg, fname_landmarks,
                                  ftmp_label, fname_template, ftmp_template, fname_template_seg, ftmp_template_seg)
    os.chdir(path_tmp)

    ''' Generate labels from template vertebral labeling'''
    sct.printv('\nGenerate labels from template vertebral labeling', verbose)
    sct.run('sct_label_utils -i '+fname_template_vertebral_labeling+' -vert-body 0 -o '+ftmp_template_label)

    ''' Check if provided labels are available in the template'''
    sct.printv('\nCheck if provided labels are available in the template', verbose)
    image_label_template = Image(ftmp_template_label)
    labels_template = image_label_template.getNonZeroCoordinates(sorting='value')
    if labels[-1].value > labels_template[-1].value:
        sct.printv('ERROR: Wrong landmarks input. Labels must have correspondence in template space. \nLabel max '
                   'provided: ' + str(labels[-1].value) + '\nLabel max from template: ' +
                   str(labels_template[-1].value), verbose, 'error')

    ''' Binarize segmentation (in case it has values below 0 caused by manual editing)'''
    sct.printv('\nBinarize segmentation', verbose)
    sct.run('sct_maths -i seg.nii.gz -bin 0.5 -o seg.nii.gz')

    # smooth segmentation (jcohenadad, issue #613)
    # sct.printv('\nSmooth segmentation...', verbose)
    # sct.run('sct_maths -i '+ftmp_seg+' -smooth 1.5 -o '+add_suffix(ftmp_seg, '_smooth'))
    # jcohenadad: updated 2016-06-16: DO NOT smooth the seg anymore. Issue #
    # sct.run('sct_maths -i '+ftmp_seg+' -smooth 0 -o '+add_suffix(ftmp_seg, '_smooth'))
    # ftmp_seg = add_suffix(ftmp_seg, '_smooth')

    # Switch between modes: subject->template or template->subject
    if ref == 'template':

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
        # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
        if os.path.isfile('../warp_curve2straight.nii.gz') and os.path.isfile('../warp_straight2curve.nii.gz') and os.path.isfile('../straight_ref.nii.gz'):
            # if they exist, copy them into current folder
            sct.printv('WARNING: Straightening was already run previously. Copying warping fields...', verbose, 'warning')
            shutil.copy('../warp_curve2straight.nii.gz', 'warp_curve2straight.nii.gz')
            shutil.copy('../warp_straight2curve.nii.gz', 'warp_straight2curve.nii.gz')
            shutil.copy('../straight_ref.nii.gz', 'straight_ref.nii.gz')
            # apply straightening
            sct.run('sct_apply_transfo -i '+ftmp_seg+' -w warp_curve2straight.nii.gz -d straight_ref.nii.gz -o '+add_suffix(ftmp_seg, '_straight'))
        else:
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

        # Compute rigid transformation straight landmarks --> template landmarks
        sct.printv('\nEstimate transformation for step #0...', verbose)
        from msct_register_landmarks import register_landmarks
        try:
            register_landmarks(ftmp_label, ftmp_template_label, paramreg.steps['0'].dof, fname_affine='straight2templateAffine.txt', verbose=verbose)
        except Exception:
            sct.printv('ERROR: input labels do not seem to be at the right place. Please check the position of the labels. See documentation for more details: https://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/', verbose=verbose, type='error')

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

        # binarize
        sct.printv('\nBinarize segmentation...', verbose)
        sct.run('sct_maths -i '+ftmp_seg+' -bin 0.5 -o '+add_suffix(ftmp_seg, '_bin'))
        ftmp_seg = add_suffix(ftmp_seg, '_bin')

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
        for i_step in range(1, len(paramreg.steps)):
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

    # register template->subject
    elif ref == 'subject':

        # Change orientation of input images to RPI
        sct.printv('\nChange orientation of input images to RPI...', verbose)
        sct.run('sct_image -i ' + ftmp_data + ' -setorient RPI -o ' + add_suffix(ftmp_data, '_rpi'))
        ftmp_data = add_suffix(ftmp_data, '_rpi')
        sct.run('sct_image -i ' + ftmp_seg + ' -setorient RPI -o ' + add_suffix(ftmp_seg, '_rpi'))
        ftmp_seg = add_suffix(ftmp_seg, '_rpi')
        sct.run('sct_image -i ' + ftmp_label + ' -setorient RPI -o ' + add_suffix(ftmp_label, '_rpi'))
        ftmp_label = add_suffix(ftmp_label, '_rpi')

        # Remove unused label on template. Keep only label present in the input label image
        sct.printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
        sct.run('sct_label_utils -i '+ftmp_template_label+' -o '+ftmp_template_label+' -remove '+ftmp_label)

        # Add one label because at least 3 orthogonal labels are required to estimate an affine transformation. This new label is added at the level of the upper most label (lowest value), at 1cm to the right.
        for i_file in [ftmp_label, ftmp_template_label]:
            im_label = Image(i_file)
            coord_label = im_label.getCoordinatesAveragedByValue()  # N.B. landmarks are sorted by value
            # Create new label
            from copy import deepcopy
            new_label = deepcopy(coord_label[0])
            # move it 5mm to the left (orientation is RAS)
            nx, ny, nz, nt, px, py, pz, pt = im_label.dim
            new_label.x = round(coord_label[0].x + 5.0 / px)
            # assign value 99
            new_label.value = 99
            # Add to existing image
            im_label.data[int(new_label.x), int(new_label.y), int(new_label.z)] = new_label.value
            # Overwrite label file
            # im_label.setFileName('label_rpi_modif.nii.gz')
            im_label.save()

        # Bring template to subject space using landmark-based transformation
        sct.printv('\nEstimate transformation for step #0...', verbose)
        from msct_register_landmarks import register_landmarks
        warp_forward = ['template2subjectAffine.txt']
        warp_inverse = ['-template2subjectAffine.txt']
        try:
            register_landmarks(ftmp_template_label, ftmp_label, paramreg.steps['0'].dof, fname_affine=warp_forward[0], verbose=verbose, path_qc=param.path_qc)
        except Exception:
            sct.printv('ERROR: input labels do not seem to be at the right place. Please check the position of the labels. See documentation for more details: https://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/', verbose=verbose, type='error')

        # loop across registration steps
        for i_step in range(1, len(paramreg.steps)):
            sct.printv('\nEstimate transformation for step #'+str(i_step)+'...', verbose)
            # identify which is the src and dest
            if paramreg.steps[str(i_step)].type == 'im':
                src = ftmp_template
                dest = ftmp_data
                interp_step = 'linear'
            elif paramreg.steps[str(i_step)].type == 'seg':
                src = ftmp_template_seg
                dest = ftmp_seg
                interp_step = 'nn'
            else:
                sct.printv('ERROR: Wrong image type.', 1, 'error')
            # apply transformation from previous step, to use as new src for registration
            sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+add_suffix(src, '_regStep'+str(i_step-1))+' -x '+interp_step, verbose)
            src = add_suffix(src, '_regStep'+str(i_step-1))
            # register src --> dest
            # TODO: display param for debugging
            warp_forward_out, warp_inverse_out = register(src, dest, paramreg, param, str(i_step))
            warp_forward.append(warp_forward_out)
            warp_inverse.insert(0, warp_inverse_out)

        # Concatenate transformations:
        sct.printv('\nConcatenate transformations: template --> subject...', verbose)
        sct.run('sct_concat_transfo -w '+','.join(warp_forward)+' -d data.nii -o warp_template2anat.nii.gz', verbose)
        sct.printv('\nConcatenate transformations: subject --> template...', verbose)
        sct.run('sct_concat_transfo -w '+','.join(warp_inverse)+' -d template.nii -o warp_anat2template.nii.gz', verbose)

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
    if ref == 'template':
        # copy straightening files in case subsequent SCT functions need them
        sct.generate_output_file(path_tmp+'warp_curve2straight.nii.gz', path_output+'warp_curve2straight.nii.gz', verbose)
        sct.generate_output_file(path_tmp+'warp_straight2curve.nii.gz', path_output+'warp_straight2curve.nii.gz', verbose)
        sct.generate_output_file(path_tmp+'straight_ref.nii.gz', path_output+'straight_ref.nii.gz', verbose)

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

def check_labels(fname_landmarks):
    """
    Make sure input labels are consistent
    Parameters
    ----------
    fname_landmarks: file name of input labels

    Returns
    -------
    none
    """

    if(fname_landmarks=='viewer'):
        sct.printv('\nCheck input labels...')
    # open label file
    image_label = Image(fname_landmarks)
    # -> all labels must be different
    labels = image_label.getNonZeroCoordinates(sorting='value')
    # check if there is two labels
    if not len(labels) == 2:
        sct.printv('ERROR: Label file has ' + str(len(labels)) + ' label(s). It must contain exactly two labels.', 1, 'error')
    # check if the two labels are integer
    for label in labels:
        if not int(label.value) == label.value:
            sct.printv('ERROR: Label should be integer.', 1, 'error')
    # check if the two labels are different
    if labels[0].value == labels[1].value:
        sct.printv('ERROR: The two labels must be different.', 1, 'error')
    return labels


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # call main function
    main()
