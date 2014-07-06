#!/usr/bin/env python
#########################################################################################
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-06-03
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


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug               = 0
        self.remove_temp_files   = 1 # remove temporary files
        self.output_type         = 1
        self.speed               = 'fast' # speed of registration. slow | normal | fast
        self.verbose             = 1 # verbose
        self.folder_template     = '/data/template'
        self.file_template       = 'MNI-Poly-AMU_T2.nii.gz'
        self.file_template_label = 'landmarks_center.nii.gz'
        self.file_template_seg   = 'MNI-Poly-AMU_cord.nii.gz'


import sys
import getopt
import os
import commands
import time
import sct_utils as sct

# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_data = ''
    fname_landmarks = ''
    fname_seg = ''
    folder_template = param.folder_template
    file_template = param.file_template
    file_template_label = param.file_template_label
    file_template_seg = param.file_template_seg
    output_type = param.output_type
    speed = param.speed
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    start_time = time.time()

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # # get path of the template
    # path_template = path_sct+folder_template

    # get fname of the template + template objects
    fname_template = path_sct+folder_template+'/'+file_template
    fname_template_label = path_sct+folder_template+'/'+file_template_label
    fname_template_seg = path_sct+folder_template+'/'+file_template_seg

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = path_sct+'/testing/data/errsm_23/t2/t2.nii.gz'
        fname_landmarks = path_sct+'/testing/data/errsm_23/t2/t2_landmarks_C2_T2_center.nii.gz'
        fname_seg = path_sct+'/testing/data/errsm_23/t2/t2_segmentation_PropSeg.nii.gz'
        speed = 'superfast'

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:l:m:o:r:s:')
    except getopt.GetoptError:
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
        elif opt in ("-r"):
            remove_temp_files = int(arg)
        elif opt in ("-s"):
            speed = arg

    # display usage if a mandatory argument is not provided
    if fname_data == '' or fname_landmarks == '' or fname_seg == '':
        usage()

    # print arguments
    print '\nCheck parameters:'
    print '.. Data:                 '+fname_data
    print '.. Landmarks:            '+fname_landmarks
    print '.. Segmentation:         '+fname_seg
    print '.. Output type:          '+str(output_type)
    print '.. Speed:                '+speed
    print '.. Remove temp files:    '+str(remove_temp_files)

    # Check speed parameter and create registration mode: slow 50x30, normal 50x15, fast 10x3 (default)
    print('\nAssign number of iterations based on speed...')
    if speed == "slow":
        nb_iterations = "50x30"
    elif speed == "normal":
        nb_iterations = "50x15"
    elif speed == "fast":
        nb_iterations = "10x3"
    elif speed == "superfast":
        nb_iterations = "3x1" # only for debugging purpose-- do not inform the user about this option
    else:
        print 'ERROR: Wrong input registration speed {slow, normal, fast}.'
        sys.exit(2)
    print '.. '+nb_iterations

    # Get full path
    # fname_data = os.path.abspath(fname_data)
    # fname_landmarks = os.path.abspath(fname_landmarks)
    # fname_seg = os.path.abspath(fname_seg)

    # check existence of input files
    print('\nCheck existence of input files...')
    sct.check_file_exist(fname_data,verbose)
    sct.check_file_exist(fname_landmarks,verbose)
    sct.check_file_exist(fname_seg,verbose)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    status, output = sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    print('\nCopy files...')
    status, output = sct.run('c3d '+fname_data+' -o '+path_tmp+'/data.nii')
    status, output = sct.run('c3d '+fname_landmarks+' -o '+path_tmp+'/landmarks.nii.gz')
    status, output = sct.run('c3d '+fname_seg+' -o '+path_tmp+'/segmentation.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of input images to RPI
    print('\nChange orientation of input images to RPI...')
    status, output = sct.run('sct_orientation -i data.nii -o data_rpi.nii.gz -orientation RPI')
    status, output = sct.run('sct_orientation -i landmarks.nii.gz -o landmarks_rpi.nii.gz -orientation RPI')
    status, output = sct.run('sct_orientation -i segmentation.nii.gz -o segmentation_rpi.nii.gz -orientation RPI')

    # Straighten the spinal cord using centerline/segmentation
    print('\nStraighten the spinal cord using centerline/segmentation...')
    status, output = sct.run('sct_straighten_spinalcord.py -i data_rpi.nii.gz -c segmentation_rpi.nii.gz -r 1')

    # Label preparation:
    # --------------------------------------------------------------------------------
    # Remove unused label on template. Keep only label present in the input label image
    print('\nRemove unused label on template. Keep only label present in the input label image...')
    status, output = sct.run('sct_label_utils.py -t remove -i '+fname_template_label+' -o template_label.nii.gz -r landmarks_rpi.nii.gz')

    # Create a cross for the template labels - 5 mm
    print('\nCreate a 5 mm cross for the template labels...')
    status, output = sct.run('sct_label_utils.py -t cross -i template_label.nii.gz -o template_label_cross.nii.gz -c 5')

    # Create a cross for the input labels and dilate for straightening preparation - 5 mm
    print('\nCreate a 5mm cross for the input labels and dilate for straightening preparation...')
    status, output = sct.run('sct_label_utils.py -t cross -i landmarks_rpi.nii.gz -o landmarks_rpi_cross3x3.nii.gz -c 5 -d')

    # Push the input labels in the template space
    print('\nPush the input labels to the straight space...')
    status, output = sct.run('WarpImageMultiTransform 3 landmarks_rpi_cross3x3.nii.gz landmarks_rpi_cross3x3_straight.nii.gz -R data_rpi_straight.nii.gz warp_curve2straight.nii.gz --use-NN')

    # Convert landmarks from FLOAT32 to INT
    print '\nConvert landmarks from FLOAT32 to INT...'
    sct.run('c3d landmarks_rpi_cross3x3_straight.nii.gz -type int -o landmarks_rpi_cross3x3_straight.nii.gz')

    # Estimate affine transfo: straight --> template (landmark-based)'
    print '\nEstimate affine transfo: straight anat --> template (landmark-based)...'
    sct.run('ANTSUseLandmarkImagesToGetAffineTransform template_label_cross.nii.gz landmarks_rpi_cross3x3_straight.nii.gz affine straight2templateAffine.txt')

    # Apply affine transformation: straight --> template
    print '\nApply affine transformation to data: straight --> template...'
    sct.run('WarpImageMultiTransform 3 data_rpi_straight.nii.gz data_rpi_straight2templateAffine.nii.gz straight2templateAffine.txt -R '+fname_template)

    # Apply straightening + affine to segmentation
    print('\nApply straightening + affine to segmentation...')
    #TODO: combine the two transfo into one
    sct.run('WarpImageMultiTransform 3 segmentation_rpi.nii.gz segmentation_rpi_straight.nii.gz -R data_rpi_straight.nii.gz warp_curve2straight.nii.gz')
    sct.run('WarpImageMultiTransform 3 segmentation_rpi_straight.nii.gz segmentation_rpi_straight2templateAffine.nii.gz straight2templateAffine.txt -R '+fname_template)
    # now threshold at 0.5 (for partial volume interpolation)
    # do not do that anymore-- better to estimate transformation using trilinear interp image to avoid step effect. See issue #31 on github.
    # sct.run('c3d segmentation_rpi_straight2templateAffine.nii.gz -threshold -inf 0.5 0 1 -o segmentation_rpi_straight2templateAffine.nii.gz')

    # Registration of straight spinal cord to template
    print('\nRegister straight spinal cord to template...')
    nb_iterations = '50x15'
    # TODO: nb iteration for step 2
    sct.run('sct_register_multimodal.py -i data_rpi_straight2templateAffine.nii.gz -d '+fname_template+' -s segmentation_rpi_straight2templateAffine.nii.gz -t '+fname_template_seg+' -r '+str(remove_temp_files)+' -n '+nb_iterations+' -v '+str(verbose),verbose)
    # status, output = sct.run('sct_register_straight_spinalcord_to_template.py -i data_rpi_straight.nii.gz -l landmarks_rpi_cross3x3_straight.nii.gz -t '+path_template+'/MNI-Poly-AMU_T2.nii.gz -f template_label_cross.nii.gz -m '+path_template+'/mask_gaussian_templatespace_sigma20.nii.gz -r 1 -n '+nb_iterations+' -v 1')

    # Concatenate warping fields: template2anat & anat2template
    print('\nConcatenate warping fields: template2anat & anat2template...')
    commands.getstatusoutput('ComposeMultiTransform 3 warp_template2anat.nii.gz -R data.nii.gz warp_straight2curve.nii.gz straight2templateAffine.txt warp_dest2src.nii.gz')
    commands.getstatusoutput('ComposeMultiTransform 3 warp_anat2template.nii.gz -R '+fname_template+' warp_src2dest.nii.gz -i straight2templateAffine.txt warp_curve2straight.nii.gz')

    # Apply warping fields to anat and template
    if output_type == 1:
        sct.run('WarpImageMultiTransform 3 '+fname_template+' template2anat.nii.gz -R data.nii.gz warp_template2anat.nii.gz')
        sct.run('WarpImageMultiTransform 3 data.nii.gz anat2template.nii.gz -R '+fname_template+' warp_anat2template.nii.gz')

    # come back to parent folder
    os.chdir('..')

   # Generate output files
    print('\nGenerate output files...')
    sct.generate_output_file(path_tmp+'/warp_template2anat.nii.gz','','warp_template2anat','.nii.gz')
    sct.generate_output_file(path_tmp+'/warp_anat2template.nii.gz','','warp_anat2template','.nii.gz')
    if output_type == 1:
        sct.generate_output_file(path_tmp+'/template2anat.nii.gz','','template2anat','.nii.gz')
        sct.generate_output_file(path_tmp+'/anat2template.nii.gz','','anat2template','.nii.gz')

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview template2anat '+fname_data+' &'
    print 'fslview anat2template '+path_template+'/MNI-Poly-AMU_T2.nii.gz &\n'



# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Register anatomical image to the template.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <anat> -l <landmarks> -m <segmentation>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <anat>                    anatomical image\n' \
        '  -l <landmarks>               landmarks at spinal cord center. See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/\n' \
        '  -m <segmentation>            spinal cord segmentation. \n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -o {0, 1}                    output type. 0: warp, 1: warp+images. Default='+str(param.output_type)+'\n' \
        '  -s {slow, normal, fast}      Speed of registration. Slow gives the best results. Default='+param.speed+'\n' \
        '  -r {0, 1}                    remove temporary files. Default='+str(param.remove_temp_files)+'\n'


    # exit program
    sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
