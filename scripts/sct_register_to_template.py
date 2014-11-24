#!/usr/bin/env python
#########################################################################################
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad, Augustin Roux
# Modified: 2014-08-29
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: make function sct_convert_binary_to_trilinear
# TODO: testing script for all cases
# TODO: try to combine seg and image based for 2nd stage
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: flag to output warping field
# TODO: check if destination is axial orientation
# TODO: set gradient-step-length in mm instead of vox size.

import sys
import getopt
import os
import commands
import time
import sct_utils as sct
from sct_orientation import get_orientation, set_orientation

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1  # remove temporary files
        self.output_type = 1
        self.speed = 'fast'  # speed of registration. slow | normal | fast
        self.algo = 'SyN'
        self.gradientStep = '0.5'
        self.metric = 'MI'
        self.verbose = 1  # verbose
        self.path_template = path_sct+'/data/template'
        self.file_template = 'MNI-Poly-AMU_T2.nii.gz'
        self.file_template_label = 'landmarks_center.nii.gz'
        self.file_template_seg = 'MNI-Poly-AMU_cord.nii.gz'
        self.smoothing_sigma = 5  # Smoothing along centerline to improve accuracy and remove step effects



# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_data = ''
    fname_landmarks = ''
    fname_seg = ''
    path_template = param.path_template
    file_template = param.file_template
    file_template_label = param.file_template_label
    file_template_seg = param.file_template_seg
    output_type = param.output_type
    param_reg = ''
    speed, algo, gradientStep, metric = param.speed, param.algo, param.gradientStep, param.metric
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    smoothing_sigma = param.smoothing_sigma
    # start timer
    start_time = time.time()

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = '/Volumes/users_hd2-1/slevy/data/criugm/errsm_23/t2/t2_crop.nii.gz'
        fname_landmarks = '/Volumes/users_hd2-1/slevy/data/criugm/errsm_23/t2/t2_crop_landmarks.nii.gz'
        fname_seg = '/Volumes/users_hd2-1/slevy/data/criugm/errsm_23/t2/t2_crop_seg.nii.gz'
        speed = 'superfast'
        #param_reg = '2,BSplineSyN,0.6,MeanSquares'
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:l:m:o:p:r:s:t:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ("-i"):
                fname_data = arg
            elif opt in ('-l'):
                fname_landmarks = arg
            elif opt in ("-m"):
                fname_seg = arg
            elif opt in ("-o"):
                output_type = int(arg)
            elif opt in ("-p"):
                param_reg = arg
            elif opt in ("-r"):
                remove_temp_files = int(arg)
            elif opt in ("-s"):
                speed = arg
            elif opt in ("-t"):
                path_template = arg

    # display usage if a mandatory argument is not provided
    if fname_data == '' or fname_landmarks == '' or fname_seg == '':
        usage()

    # get absolute path
    path_template = os.path.abspath(path_template)

    # get fname of the template + template objects
    fname_template = sct.slash_at_the_end(path_template, 1)+file_template
    fname_template_label = sct.slash_at_the_end(path_template, 1)+file_template_label
    fname_template_seg = sct.slash_at_the_end(path_template, 1)+file_template_seg

    # check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data, verbose)
    sct.check_file_exist(fname_landmarks, verbose)
    sct.check_file_exist(fname_seg, verbose)
    sct.check_file_exist(fname_template, verbose)
    sct.check_file_exist(fname_template_label, verbose)
    sct.check_file_exist(fname_template_seg, verbose)

    # print arguments
    print '\nCheck parameters:'
    print '.. Data:                 '+fname_data
    print '.. Landmarks:            '+fname_landmarks
    print '.. Segmentation:         '+fname_seg
    print '.. Path template:        '+path_template
    print '.. Output type:          '+str(output_type)
    print '.. Speed:                '+speed
    print '.. Remove temp files:    '+str(remove_temp_files)

    # Check speed parameter and create registration mode: slow 50x30, normal 50x15, fast 10x3 (default)
    if speed == "slow":
        nb_iterations = "50"
    elif speed == "normal":
        nb_iterations = "15"
    elif speed == "fast":
        nb_iterations = "5"
    elif speed == "superfast":
        nb_iterations = "1"  # only for debugging purpose-- do not inform the user about this option
    else:
        print 'ERROR: Wrong input registration speed {slow, normal, fast}.'
        sys.exit(2)

    # Check registration parameters selected by user
    if param_reg:
        nb_iterations, algo, gradientStep, metric = param_reg.split(',')

    sct.printv('\nParameters for registration:')
    sct.printv('.. Number of iterations..... '+nb_iterations)
    sct.printv('.. Algorithm................ '+algo)
    sct.printv('.. Gradient step............ '+gradientStep)
    sct.printv('.. Metric................... '+metric)

    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    status, output = sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    print('\nCopy files...')
    status, output = sct.run('sct_c3d '+fname_data+' -o '+path_tmp+'/data.nii')
    status, output = sct.run('sct_c3d '+fname_landmarks+' -o '+path_tmp+'/landmarks.nii.gz')
    status, output = sct.run('sct_c3d '+fname_seg+' -o '+path_tmp+'/segmentation.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of input images to RPI
    print('\nChange orientation of input images to RPI...')
    set_orientation('data.nii', 'RPI', 'data_rpi.nii')
    set_orientation('landmarks.nii.gz', 'RPI', 'landmarks_rpi.nii.gz')
    set_orientation('segmentation.nii.gz', 'RPI', 'segmentation_rpi.nii.gz')

    # Straighten the spinal cord using centerline/segmentation
    print('\nStraighten the spinal cord using centerline/segmentation...')
    sct.run('sct_straighten_spinalcord -i data_rpi.nii -c segmentation_rpi.nii.gz -r 0')

    # Apply straightening to segmentation
    print('\nApply straightening to segmentation...')
    sct.run('sct_apply_transfo -i segmentation_rpi.nii.gz -o segmentation_rpi_straight.nii.gz -d data_rpi_straight.nii -w warp_curve2straight.nii.gz')

    # Smoothing along centerline to improve accuracy and remove step effects
    print('\nSmoothing along centerline to improve accuracy and remove step effects...')
    sct.run('sct_c3d data_rpi_straight.nii -smooth 0x0x'+str(smoothing_sigma)+'vox -o data_rpi_straight.nii')
    sct.run('sct_c3d segmentation_rpi_straight.nii.gz -smooth 0x0x'+str(smoothing_sigma)+'vox -o segmentation_rpi_straight.nii.gz')

    # Label preparation:
    # --------------------------------------------------------------------------------
    # Remove unused label on template. Keep only label present in the input label image
    print('\nRemove unused label on template. Keep only label present in the input label image...')
    status, output = sct.run('sct_label_utils -t remove -i '+fname_template_label+' -o template_label.nii.gz -r landmarks_rpi.nii.gz')

    # Make sure landmarks are INT
    print '\nConvert landmarks to INT...'
    sct.run('sct_c3d template_label.nii.gz -type int -o template_label.nii.gz')

    # Create a cross for the template labels - 5 mm
    print('\nCreate a 5 mm cross for the template labels...')
    status, output = sct.run('sct_label_utils -t cross -i template_label.nii.gz -o template_label_cross.nii.gz -c 5')

    # Create a cross for the input labels and dilate for straightening preparation - 5 mm
    print('\nCreate a 5mm cross for the input labels and dilate for straightening preparation...')
    status, output = sct.run('sct_label_utils -t cross -i landmarks_rpi.nii.gz -o landmarks_rpi_cross3x3.nii.gz -c 5 -d')

    # Push the input labels in the template space
    print('\nPush the input labels to the straight space...')
    status, output = sct.run('sct_apply_transfo -i landmarks_rpi_cross3x3.nii.gz -o landmarks_rpi_cross3x3_straight.nii.gz -d data_rpi_straight.nii -w warp_curve2straight.nii.gz -p nn')

    # Convert landmarks from FLOAT32 to INT
    print '\nConvert landmarks from FLOAT32 to INT...'
    sct.run('sct_c3d landmarks_rpi_cross3x3_straight.nii.gz -type int -o landmarks_rpi_cross3x3_straight.nii.gz')

    # Estimate affine transfo: straight --> template (landmark-based)'
    print '\nEstimate affine transfo: straight anat --> template (landmark-based)...'
    sct.run('sct_ANTSUseLandmarkImagesToGetAffineTransform template_label_cross.nii.gz landmarks_rpi_cross3x3_straight.nii.gz affine straight2templateAffine.txt')

    # Apply affine transformation: straight --> template
    print '\nApply affine transformation: straight --> template...'
    status, output = sct.run('sct_apply_transfo -i data_rpi_straight.nii -o data_rpi_straight2templateAffine.nii -d '+fname_template+' -w straight2templateAffine.txt')
    status, output = sct.run('sct_apply_transfo -i segmentation_rpi_straight.nii.gz -o segmentation_rpi_straight2templateAffine.nii.gz -d '+fname_template+' -w straight2templateAffine.txt')

    # now threshold at 0.5 (for partial volume interpolation)
    # do not do that anymore-- better to estimate transformation using trilinear interp image to avoid step effect. See issue #31 on github.
    # sct.run('sct_c3d segmentation_rpi_straight2templateAffine.nii.gz -threshold -inf 0.5 0 1 -o segmentation_rpi_straight2templateAffine.nii.gz')

    # Registration straight spinal cord to template
    print('\nRegister straight spinal cord to template...')
    sct.run('sct_register_multimodal -i data_rpi_straight2templateAffine.nii -d '+fname_template+' -s segmentation_rpi_straight2templateAffine.nii.gz -t '+fname_template_seg+' -r 0 -p '+nb_iterations+','+algo+','+gradientStep+','+metric+' -v '+str(verbose)+' -x spline -z 10', verbose)

    # Concatenate warping fields: template2anat & anat2template
    print('\nConcatenate transformations: template --> straight --> anat...')
    sct.run('sct_concat_transfo -w warp_dest2src.nii.gz,-straight2templateAffine.txt,warp_straight2curve.nii.gz -d data.nii -o warp_template2anat.nii.gz')
    print('\nConcatenate transformations: anat --> straight --> template...')
    sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt,warp_src2dest.nii.gz -d '+fname_template+' -o warp_anat2template.nii.gz')
    # cmd = 'sct_ComposeMultiTransform 3 warp_anat2template.nii.gz -R '+fname_template+' warp_src2dest.nii.gz straight2templateAffine.txt warp_curve2straight.nii.gz'
    # print '>> '+cmd
    # commands.getstatusoutput(cmd)

# sct_ComposeMultiTransform 3 warp_final.nii.gz -R data.nii warp_dest2src.nii.gz -i straight2templateAffine.txt warp_straight2curve.nii.gz

    # Apply warping fields to anat and template
    if output_type == 1:
        sct.run('sct_apply_transfo -i '+fname_template+' -o template2anat.nii.gz -d data.nii -w warp_template2anat.nii.gz')
        sct.run('sct_apply_transfo -i data.nii  -o anat2template.nii.gz -d '+fname_template+' -w warp_anat2template.nii.gz')

    # come back to parent folder
    os.chdir('..')

   # Generate output files
    print('\nGenerate output files...')
    sct.generate_output_file(path_tmp+'/warp_template2anat.nii.gz', 'warp_template2anat.nii.gz')
    sct.generate_output_file(path_tmp+'/warp_anat2template.nii.gz', 'warp_anat2template.nii.gz')
    if output_type == 1:
        sct.generate_output_file(path_tmp+'/template2anat.nii.gz', 'template2anat'+ext_data)
        sct.generate_output_file(path_tmp+'/anat2template.nii.gz', 'anat2template'+ext_data)

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview template2anat -b 0,4000 '+fname_data+' &'
    print 'fslview anat2template '+fname_template+' -b 0,4000 &\n'



# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Register anatomical image to the template.

USAGE
  """+os.path.basename(__file__)+""" -i <anat> -m <segmentation> -l <landmarks>

MANDATORY ARGUMENTS
  -i <anat>                    anatomical image
  -m <segmentation>            spinal cord segmentation.
  -l <landmarks>               landmarks at spinal cord center.
                               See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/

OPTIONAL ARGUMENTS
  -o {0, 1}                    output type. 0: warp, 1: warp+images. Default="""+str(param_default.output_type)+"""
  -p <param>                   parameters for registration.
                               ALL ITEMS MUST BE LISTED IN ORDER. Separate with comma WITHOUT WHITESPACE IN BETWEEN.
                               Default=10,SyN,0.5,MeanSquares
                                 1) number of iterations for last stage.
                                 2) algo: {SyN, BSplineSyN, sliceReg}
                                    N.B. if you use sliceReg, then you should set -z 0. Also, the two input
                                    volumes should have same the same dimensions.
                                    For more info about sliceReg, type: sct_antsSliceRegularizedRegistration
                                 3) gradient step. The larger the more deformation.
                                 4) metric: {MI,MeanSquares}.
                                    If you find very large deformations, switching to MeanSquares can help.
  -t <path_template>           Specify path to template. Default="""+str(param_default.path_template)+"""
  -s {slow, normal, fast}      Speed of registration. Slow gives the best results. Default="""+str(param_default.speed)+"""
  -r {0, 1}                    remove temporary files. Default="""+str(param_default.remove_temp_files)+"""
  -h                           help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i t2.nii.gz -l labels.nii.gz -m t2_seg.nii.gz -s normal\n"""

    # exit program
    sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
