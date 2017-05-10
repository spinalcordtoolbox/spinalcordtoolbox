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
    parser.add_option(name="-o",
                      type_value="str",
                      description="Prefix for output files.",
                      mandatory=False,
                      default_value='')
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
    parser.add_option(name='-nb-pts',
                      type_value='int',
                      description='Number of points you want to make before auto leave \n'
                                  'Warning : the window will close as soon as you made the number of points you requested \n',
                      default_value=-1,
                      example= 2)
    parser.add_option(name='-save-as',
                      type_value='multiple_choice',
                      description='Define how you wish to save labels',
                      default_value='png_txt',
                      example= ['png_txt', 'niftii'])
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
    parser.add_option(name="-param",
                      type_value=[[':'], 'str'],
                      description='Parameters for registration (see sct_register_multimodal). Default: \
                      \n--\nstep=0\ntype=' + paramreg.steps['0'].type + '\ndof=' + paramreg.steps['0'].dof + '\
                      \n--\nstep=1\ntype=' + paramreg.steps['1'].type + '\nalgo=' + paramreg.steps['1'].algo + '\nmetric=' + paramreg.steps['1'].metric + '\niter=' + paramreg.steps['1'].iter + '\nsmooth=' + paramreg.steps['1'].smooth + '\ngradStep=' + paramreg.steps['1'].gradStep + '\nslicewise=' + paramreg.steps['1'].slicewise + '\nsmoothWarpXY=' + paramreg.steps['1'].smoothWarpXY + '\npca_eigenratio_th=' + paramreg.steps['1'].pca_eigenratio_th + '\
                      \n--\nstep=2\ntype=' + paramreg.steps['2'].type + '\nalgo=' + paramreg.steps['2'].algo + '\nmetric=' + paramreg.steps['2'].metric + '\niter=' + paramreg.steps['2'].iter + '\nsmooth=' + paramreg.steps['2'].smooth + '\ngradStep=' + paramreg.steps['2'].gradStep + '\nslicewise=' + paramreg.steps['2'].slicewise + '\nsmoothWarpXY=' + paramreg.steps['2'].smoothWarpXY + '\npca_eigenratio_th=' + paramreg.steps['1'].pca_eigenratio_th,
                      mandatory=False)

    return parser

def rewrite_arguments(arguments):
    def rewrite_output_path(arguments):
        if '-o' in arguments:
            return arguments['-o']
        else:
            return ''
    def rewrite_save_as(arguments):
        s=arguments['-save-as']
        if s=='png_txt':
            return True
        elif s=='niftii':
            return False
        else:
            return True


    fname_data = arguments['-i']
    output_path=rewrite_output_path(arguments)
    first_label=arguments['-first']
    ref = arguments['-ref']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])
    nb_pts=int(arguments['-nb-pts'])
    bool_save_as_png=rewrite_save_as(arguments)


    return (fname_data,output_path,ref,remove_temp_files,verbose,first_label,nb_pts,bool_save_as_png)

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

def print_arguments(verbose,fname_data,fname_landmarks,fname_seg,path_template,remove_temp_files):
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
        sct.run('sct_image -i ' + 'labels.nii.gz'+ ' -o ' + 'labels_ground_truth.nii.gz' + ' -setorient ' + image_input_orientation + ' -v 0',verbose=True)
        sct.run('rm -rf ' + 'labels.nii.gz')

def save_niftii(mask_points,reoriented_image_filename,image_input_orientation):
    print(mask_points)
    make_labels_image_from_list_points(mask_points, reoriented_image_filename, image_input_orientation)

def use_viewer_to_define_labels(fname_data,first_label,output_path,nb_pts,bool_save_as_png):
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

    from viewer2 import WindowGroundTruth
    im_input_SAL=prepare_input_image_for_viewer(fname_data)
    viewer = WindowGroundTruth(im_input_SAL,first_label=first_label,
                               file_name=fname_data,
                               output_path=output_path,
                               dic_save_niftii={'save_function':save_niftii,
                                                'reoriented_image_filename':reoriented_image_filename,
                                                'image_input_orientation':image_input_orientation},
                               nb_pts=nb_pts,
                               bool_save_as_png=bool_save_as_png)

    #mask_points = viewer.start()
    #if not mask_points and viewer.closed:
    #    mask_points = viewer.list_points_useful_notation
    #make_labels_image_from_list_points(mask_points,reoriented_image_filename,image_input_orientation)


# MAIN
# ==========================================================================================
def main():
    parser = get_parser()
    param = Param()

    """ Rewrite arguments and set parameters"""
    arguments = parser.parse(sys.argv[1:])
    (fname_data, output_path, ref, remove_temp_files, verbose, first_label,nb_pts,bool_save_as_png)=rewrite_arguments(arguments)
    (param, paramreg)=write_paramaters(arguments,param,ref,verbose)

    use_viewer_to_define_labels(fname_data,first_label,output_path=output_path,nb_pts=nb_pts,bool_save_as_png=bool_save_as_png)


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
    # call main function
    main()
