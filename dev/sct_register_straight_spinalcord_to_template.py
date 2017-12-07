#!/usr/bin/env python

## @package sct_register_straight_spinalcord_to_template
#
# Estimate deformation field between straightened spinal cord and template.
#
#
#
# USAGE
# ---------------------------------------------------------------------------------------
# TODO
#
# INPUT
# ---------------------------------------------------------------------------------------
# - anat_straight
# - landmarks on anat_straight
# - template
# - landmarks on template
# - (gaussian mask on template)
# - (gaussian mask on anat_straight)
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# - warping field: anat_straight --> template
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# none
#
# EXTERNAL SOFTWARE
# - FSL: <http://fsl.fmrib.ox.ac.uk/fsl/>
# - ANTS
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-06
#
# License: see the LICENSE.TXT
#=======================================================================================================================

# TODO: register using intermediate step from segmentations
# TODO: try to register template 2 straight using MI
# TODO: don't apply affine transfo-- estimate with init option uner isct_antsRegistration
# TODO: the user would only have to select at two locations, e.g., C2 and T2, and would put values = 2 and 9 (7+2). Then add a function that would remove unused labels on the template, to keep only 2 and 9. Then, generate a cross in order to have the proper affine transfo.
# TODO: mask template
# TODO: add number of thread: ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NUMBEROFTHREADS // export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS
# TODO: don't output inverse transfo if not necessary (i.e., could be inversed later in the worksflow)


## Create a structure to pass important user parameters to the main function
class param:
    ## The constructor
    def __init__(self):
        self.remove_temp_files = 1 # remove temporary files
        self.debug = 0 # debug mode
        self.verbose             = 1 # verbose
        self.number_iterations    = "50x20" # number of iterations

# check if needed Python libraries are already installed or not
import os
import getopt
import commands
import sys
import time
import sct_utils as sct




#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_anat = ''
    fname_landmark_anat = ''
    fname_template = ''
    fname_landmark_template = ''
    fname_mask = ''
    remove_temp_files = param.remove_temp_files
    number_iterations = param.number_iterations
    verbose = param.verbose
    start_time = time.time()

    # extract path of the script
    path_script = os.path.dirname(__file__)+'/'

    # Parameters for debug mode
    if param.debug == 1:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_anat = path_script+'../testing/sct_register_straight_spinalcord_to_template/data/errsm_22_t2_cropped_rpi_straight.nii.gz'
        fname_landmark_anat = path_script+'../testing/sct_register_straight_spinalcord_to_template/data/landmarks_C2_T5.nii.gz'
        fname_seg_anat = path_script+'../testing/sct_register_straight_spinalcord_to_template/data/landmarks_C2_T5.nii.gz'
        fname_template = path_script+'../data/template/MNI-Poly-AMU_T2.nii.gz'
        fname_landmark_template = path_script+'../data/template/landmarks_C2_T5.nii.gz'

    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:f:l:m:n:o:r:s:t:v:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-f'):
            fname_landmark_template = arg
        elif opt in ('-i'):
            fname_anat = arg
        elif opt in ('-l'):
            fname_landmark_anat = arg
        elif opt in ('-m'):
            fname_mask = arg
        elif opt in ('-n'):
            number_iterations = arg
        elif opt in ("-o"):
            fname_template_seg = arg
        elif opt in ('-r'):
            remove_temp_files = int(arg)
        elif opt in ("-s"):
            fname_anat_seg = arg
        elif opt in ('-t'):
            fname_template = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_landmark_anat == '' or fname_template == '' or fname_landmark_template == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_landmark_anat)
    sct.check_file_exist(fname_template)
    sct.check_file_exist(fname_landmark_template)
    sct.check_file_exist(fname_seg_template)

    # Display arguments
    print '\nCheck input arguments:'
    print '  straight anatomic:    '+fname_anat
    print '  landmarks anatomic:   '+fname_landmark_anat
    print '  template T2:          '+fname_template
    print '  template landmarks:   '+fname_landmark_template
    print '  template segmentation:'+fname_landmark_template
    print '  number of iterations: '+str(number_iterations)
    print '  mask anatomic:        '+fname_mask
    print '  Verbose:              '+str(verbose)

    # Get full path
    fname_anat = os.path.abspath(fname_anat)
    fname_landmark_anat = os.path.abspath(fname_landmark_anat)
    fname_template = os.path.abspath(fname_template)
    fname_landmark_template = os.path.abspath(fname_landmark_template)

    # extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_template, file_template, ext_template = sct.extract_fname(fname_template)

    path_tmp = sct.tmp_create()

    # go to tmp folder
    os.chdir(path_tmp)

    # Estimate transfo: straight --> template (affine landmark-based)'
    print '\nEstimate transfo: straight anat --> template (affine landmark-based)...'
    sct.run('ANTSUseLandmarkImagesToGetAffineTransform '+fname_landmark_template+' '+fname_landmark_anat+' affine tmp.straight2templateAffine.txt')

    # Apply transformation: straight --> template
    print '\nApply transformation straight --> template...'
    sct.run('WarpImageMultiTransform 3 '+fname_anat+' tmp.straight2templateAffine.nii tmp.straight2templateAffine.txt -R '+fname_template)

    # Estimate transformation: straight --> template (deformation)
    print '\nEstimate transformation: straight --> template (diffeomorphic transformation). Takes ~15-45 minutes...'
    cmd = 'isct_antsRegistration \
--dimensionality 3 \
--transform SyN[0.2,3] \
--metric MI['+fname_template+',tmp.straight2templateAffine.nii,1,32] \
--convergence '+number_iterations+' \
--shrink-factors 4x1 \
--smoothing-sigmas 1x0mm \
--Restrict-Deformation 1x1x0 \
--output [tmp.straight2template,tmp.straight2template.nii.gz] \
--collapse-output-transforms 1 \
--interpolation BSpline[3] \
--winsorize-image-intensities [0.005,0.995]'

    if fname_mask != '':
        # TODO: check if mask exist
        cmd = cmd+' -x '+fname_mask

    # run command
    status, output = sct.run(cmd)
    if verbose:
        print output

    # Concatenate affine and non-linear transformations...
    print '\nConcatenate affine and non-linear transformations: straight --> template...'
    # NB: cannot use sct.run() because output of ComposeMultiTransform is not 0, even if there is no error (bug in ANTS-- already reported on 2013-12-30)
    cmd = 'ComposeMultiTransform 3 tmp.warp_straight2template.nii.gz -R '+fname_template+' tmp.straight2template0Warp.nii.gz tmp.straight2templateAffine.txt'
    print('>> '+cmd)
    commands.getstatusoutput(cmd)

    # Concatenate affine and non-linear transformations...
    print '\nConcatenate affine and non-linear transformations: template --> straight...'
    # NB: cannot use sct.run() because output of ComposeMultiTransform is not 0, even if there is no error (bug in ANTS-- already reported on 2013-12-30)
    cmd = 'ComposeMultiTransform 3 tmp.warp_template2straight.nii.gz -R '+fname_anat+' -i tmp.straight2templateAffine.txt tmp.straight2template0InverseWarp.nii.gz'
    print('>> '+cmd)
    commands.getstatusoutput(cmd)

    # Apply transformation: template --> straight
    print '\nApply transformation: template --> straight...'
    sct.run('WarpImageMultiTransform 3 '+fname_template+' tmp.template2straight.nii.gz'+' -R '+fname_anat+' tmp.warp_template2straight.nii.gz')



# THIS CODE USES 2-STEP METHOD WITH SEGMENTATION

#     # Estimate transfo: straight --> template (affine landmark-based)'
#     print '\nEstimate transfo: straight anat --> template (affine landmark-based)...'
#     sct.run('ANTSUseLandmarkImagesToGetAffineTransform '+fname_landmark_template+' '+fname_landmark_anat+' affine tmp.straight2templateAffine.txt')
#
#     # Apply transformation: straight --> template
#     print '\nApply transformation straight --> template...'
#     sct.run('WarpImageMultiTransform 3 '+fname_anat+' tmp.straight2templateAffine.nii tmp.straight2templateAffine.txt -R '+fname_template)
#     sct.run('WarpImageMultiTransform 3 '+fname_anat_seg+' tmp.straightSeg2templateAffine.nii tmp.straight2templateAffine.txt -R '+fname_template)
#
#     # Estimate transformation using ANTS
#     print('\nStep #1: Estimate transformation using spinal cord segmentations...')
#
#     cmd = 'isct_antsRegistration \
# --dimensionality 3 \
# --transform SyN[0.2,3,0] \
# --metric MI['+fname_template_seg+',tmp.straightSeg2templateAffine.nii,1,32] \
# --convergence 50x10 \
# --shrink-factors 2x1 \
# --smoothing-sigmas 2x1mm \
# --Restrict-Deformation 1x1x0 \
# --output [tmp.regSeg,tmp.straightSeg2template.nii.gz]'
#
#     # run command
#     status, output = sct.run(cmd)
#     if verbose:
#         print output
#
#     # Apply warping field: seg --> template_seg
#     print '\nApply transformation anat_seg --> template_seg...'
#     sct.run('WarpImageMultiTransform 3 '+fname_anat+' tmp.straight2templateStep1.nii tmp.regSeg0Warp.nii.gz -R '+fname_template)
#
#     print('\nStep #2: Improve local deformation using images (start from previous transformation)...')
#
#     # Estimate transformation: straight --> template (deformation)
#     print '\nEstimate transformation: straight --> template (diffeomorphic transformation). Takes 10-45 minutes...'
#     cmd = 'isct_antsRegistration \
# --dimensionality 3 \
# --transform SyN[0.1,1,0] \
# --metric CC['+fname_template+',tmp.straight2templateStep1.nii,1,4] \
# --convergence 20 \
# --shrink-factors 1 \
# --smoothing-sigmas 0mm \
# --Restrict-Deformation 1x1x0 \
# --output [tmp.straight2template,tmp.straight2template.nii.gz] \
# --interpolation BSpline[3]'
#
#     # use mask (if provided by user)
#     if fname_mask != '':
#         # TODO: check if mask exist
#         cmd = cmd+' -x '+fname_mask
#
#     # run command
#     status, output = sct.run(cmd)
#     if verbose:
#         print output
#
#     # Concatenate affine and non-linear transformations...
#     print '\nConcatenate affine and non-linear transformations: straight --> template...'
#     # NB: cannot use sct.run() because output of ComposeMultiTransform is not 0, even if there is no error (bug in ANTS-- already reported on 2013-12-30)
#     cmd = 'ComposeMultiTransform 3 tmp.warp_straight2template.nii.gz -R '+fname_template+' tmp.straight2template0Warp.nii.gz tmp.regSeg0Warp.nii.gz tmp.straight2templateAffine.txt'
#     print('>> '+cmd)
#     commands.getstatusoutput(cmd)
#
#     # Concatenate affine and non-linear transformations...
#     print '\nConcatenate affine and non-linear transformations: template --> straight...'
#     # NB: cannot use sct.run() because output of ComposeMultiTransform is not 0, even if there is no error (bug in ANTS-- already reported on 2013-12-30)
#     cmd = 'ComposeMultiTransform 3 tmp.warp_template2straight.nii.gz -R '+fname_anat+' -i tmp.straight2templateAffine.txt tmp.straight2template0InverseWarp.nii.gz'
#     print('>> '+cmd)
#     commands.getstatusoutput(cmd)
#
#     # Apply transformation: template --> straight
#     print '\nApply transformation: template --> straight...'
#     sct.run('WarpImageMultiTransform 3 '+fname_template+' tmp.template2straight.nii.gz'+' -R '+fname_anat+' tmp.warp_template2straight.nii.gz')
#




    # Generate output file (in current folder)
    print '\nGenerate output file...'
    sct.generate_output_file('tmp.warp_template2straight.nii.gz','./','warp_template2straight',ext_anat) # warping field template --> straight
    sct.generate_output_file('tmp.warp_straight2template.nii.gz','./','warp_straight2template',ext_anat) # warping field straight --> template
    sct.generate_output_file('tmp.straight2template.nii.gz','./',file_anat+'2template',ext_anat) # anat --> template
    sct.generate_output_file('tmp.template2straight.nii.gz','./',file_template+'2straight',ext_anat) # anat --> template

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm tmp.*')

    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'



#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This program estimates a deformation field between a straightened spinal cord and the template. First,\n' \
        '  an affine transformation is calculated (based on landmarks set by the user on the straight spinal cord).\n' \
        '  Second, a warping field is estimated. The program outputs the forward (straight --> template) and the inverse\n' \
        '  (template --> straight) transformations. \n' \
        '\n'\
        'USAGE \n' \
        '  '+os.path.basename(__file__)+' -i <anat> -l <landmarks_anat> -t <template> -f <landmarks_template>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i <anat>                   straight anatomic (generated with sct_straighted_spinalcord).\n' \
        '  -l <anat_labels>         anatomical landmarks.\n' \
        '  -m <anat_seg>         anatomical landmarks.\n' \

        '  -t <template>               template.\n' \
        '  -f <landmarks_template>     template landmarks (should match the anatomical landmarks).\n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -m <mask>                   mask on anatomical image.\n' \
        '  -r <0,1>                    remove temporary files. Default='+str(param.remove_temp_files)+'. \n' \
        '  -n <nxm>                    change the iteration number of the registration (isct_antsRegistration).\n' \
        '  -v <0,1>                    verbose. Default='+str(param.verbose)+'\n'

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()

