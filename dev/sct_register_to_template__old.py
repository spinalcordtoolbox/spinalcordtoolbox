#!/usr/bin/env python
# ==========================================================================================
# Register to template.
#
#
# USAGE
# ---------------------------------------------------------------------------------------
# To estimate transformations:
#   sct_register_to_template.py -s <source> -t <template> -f <landmark_template> -m <mask_template> -i <anat> -l <landmark_anat>
#
# To apply transformations:
#   sct_register_to_template.py -s <source> -t <template> -n
#
#
# INPUT
# ---------------------------------------------------------------------------------------
#  -s       Source image to map to the template. For estimation, use similar contrast (e.g., b=0 image from DTI scan).
#  -t       Template image.
#  -f       Landmarks on the template.
#  -m       Mask encompassing the spinal cord on the template
#  -i       Anatomical image of the subject.
#  -l       Landmarks on the anatomic (!! should be in the same order as the landmarks in the template)
#  -n       No estimation: only apply transformations. Warning: previously-estimated should be local!!
#           If using this flag, mandatory arguments are: -s -t.
#  -h       help. Show this message.
#
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# TODO
#
#
# EXAMPLES:
# ---------------------------------------------------------------------------------------
#
#- To estimate transformations (only run once):
#     sct_register_to_template.py -i t2.nii.gz -t /users/bob/template/MNI-POLY-AMU_v1__T2.nii.gz -f /users/bob/template/landmarks_C3.nii.gz -l landmarks_C3.nii.gz -m /users/bob/template/mask_C3.nii.gz -b b0_mean.nii.gz
#
#- If transformations have already been computed. The following command will register FA into template:
#     sct_register_to_template_python.py -n -b dti_FA.nii.gz -t /users/bob/template/MNI-POLY-AMU_v1__T2.nii.gz
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# none
#
# EXTERNAL SOFTWARE
# - ants <http://stnava.github.io/ANTs/>
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Marc Benhamou
# Modified: 2013-11-10
#
# TODO: copy src locally
# TODO: unpad inverse Warp
# TODO: !!! crop the warping field after padding!!!! cannot do it directly with isct_c3d because it's a composite image:
# TODO: the use of a gaussian mask gives the following error: itk::ERROR: Image(0x7fb849404ce0): A spacing of 0 is not allowed: Spacing is [0, 0.0357143].
# TODO: make fname_anat_mask optional
# TODO: adjust gradient step based on native resolution
# TODO: log file with 'result' from each os command
# TODO: add usage
# TODO: enable relative path
# TODO: check input/output at each step.
# TODO: is import os needed?
#
# About the license: see the file LICENSE.TXT
# ==========================================================================================

import sys
import commands
import getopt
import os


# PARAMETERS
debugging           = 0 # automatic file names for debugging



# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_anat = ''
    fname_template = ''
    fname_landmark_template = ''
    fname_landmark_anat = ''
    fname_anat_mask = '' # optional
    fname_src = ''
    apply_only = 0
    padding = 5 # Pad the source image (because ants doesn't deform the extremities)

    # Check input parameters
    if debugging:
        path_spinalcordtoolbox = '~/code/spinalcordtoolbox_dev'
        fname_src = path_spinalcordtoolbox+'/dev/sct_register_to_template/mtc1.nii.gz'
        fname_template = path_spinalcordtoolbox+'data/template/MNI-POLY-AMU_v1__T2.nii.gz'
        fname_landmark_template = path_spinalcordtoolbox+'data/template/landmarks_C3.nii.gz'
        fname_anat  = path_spinalcordtoolbox+'dev/sct_register_to_template/t2.nii.gz'
        fname_landmark_anat = path_spinalcordtoolbox+'dev/sct_register_to_template/t2_landmarks.nii.gz'
        fname_anat_mask = path_spinalcordtoolbox+'dev/sct_register_to_template/t2_mask.nii.gz'
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:t:f:l:m:s:n')
        except getopt.GetoptError:
            usage()
        if not opts:
            # no option supplied
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_anat = arg
                exist_image(fname_anat)
            elif opt in ('-t'):
                fname_template = arg
                exist_image(fname_template)
            elif opt in ('-f'):
                fname_landmark_template = arg
                exist_image(fname_landmark_template)
            elif opt in ('-l'):
                fname_landmark_anat = arg
                exist_image(fname_landmark_anat)
            elif opt in ('-m'):
                fname_anat_mask = arg
                exist_image(fname_anat_mask)
            elif opt in ('-s'):
                fname_src = arg
                exist_image(fname_src)
            elif opt in ('-n'):
                apply_only = 1


    # print arguments
    #print 'Check input parameters...'
    #print '.. Image:    '+fname_src
    #print '.. Mask:     '+fname_mask

    # Extract path, file and extension
    #path_src, file_src, ext_src = extract_fname(fname_src)
    #path_mask, file_mask, ext_mask = extract_fname(fname_mask)

    # If transformations have already been estimated before: only source --> template transformation
    if apply_only:
        print '\n*** NO ESTIMATION (use previously-estimated transformations) ***'

        # check mandatory arguments
        if fname_template == '' or fname_src == '':
            usage()

        # Extract path, file and extension
        path_src, file_src, ext_src = extract_fname(fname_src)

        # Apply transformation: source --> anat --> template
        fname_src2template = path_src+file_src+'_reg2template'+ext_src
        print '\nApply transformation: source --> anat --> template...'
        cmd = 'WarpImageMultiTransform 3 '+fname_src+' '+fname_src2template+' -i tmp.template2anatAffineLandmark.txt tmp.template2anatInverseWarp.nii.gz tmp.src2anatAffine.txt tmp.src2anatWarp.nii.gz -R '+fname_template
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)

        # display created file
        print('\nCreated file:')
        print fname_src2template+'\n'

        # exit program
        sys.exit()

    # check mandatory arguments
    if fname_anat == '' or fname_template == '' or fname_landmark_template == '' or fname_landmark_anat == '' or fname_src == '':
        usage()

    ## Copy files locally
    #print '\nCopy files locally...'
    #cmd = 'cp '+fname_src+' tmp.src.'
    #print('>> '+cmd)
    #status, output = commands.getstatusoutput(cmd)
    #if debugging:
    #    print output


    # Estimate transfo: template --> anat (affine landmark-based)'
    print '\nEstimate transfo: template --> anat (affine landmark-based)...'
    cmd = 'ANTSUseLandmarkImagesToGetAffineTransform '+fname_landmark_anat+' '+fname_landmark_template+' affine tmp.template2anatAffineLandmark.txt'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    ## Apply transformation: mask --> anat
    #print '\nApply transformation: mask --> anat...'
    #cmd = 'WarpImageMultiTransform 3 '+fname_mask+' tmp.mask2anat.nii.gz'+' tmp.template2anatAffineLandmark.txt -R '+fname_anat
    #print('>> '+cmd)
    #status, output = commands.getstatusoutput(cmd)
    #if debugging:
    #    print output

   # Apply transformation: template --> anat
    print '\nApply transformation: template --> anat...'
    cmd = 'WarpImageMultiTransform 3 '+fname_template+' tmp.template2anatAffineLandmark.nii.gz tmp.template2anatAffineLandmark.txt -R '+fname_anat
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Estimate transformation: template --> anat (deformation)
    print '\nEstimate transformation: template --> anat (diffeomorphic transformation). Might take a couple of minutes...'
    #cmd = 'ants 3 -m CC['+fname_anat+',tmp.template2anatAffineLandmark.nii.gz,2,4] -i 10x5x1 -t SyN[0.5] -r Gauss[0,2] --Restrict-Deformation 1x1x0 --number-of-affine-iterations 0x0x0 -o tmp.template2anat.nii.gz'
    cmd = 'ants 3 -m CC['+fname_anat+',tmp.template2anatAffineLandmark.nii.gz,2,4] -i 50x5x1 -t SyN[2] -r Gauss[0,2] --Restrict-Deformation 1x1x1 --number-of-affine-iterations 0x0x0 -o tmp.template2anat.nii.gz'
    if fname_anat_mask != '':
        cmd = cmd+' -x '+fname_anat_mask
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Apply transformation: template --> anat
    print '\nApply transformation: template --> anat...'
    cmd = 'WarpImageMultiTransform 3 '+fname_template+' tmp.template2anat.nii.gz'+' tmp.template2anatWarp.nii.gz tmp.template2anatAffineLandmark.txt -R '+fname_anat
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # If mask exist, put it in the space of the source for more robust registration
    if fname_anat_mask != '':
        print '\nReslice: mask --> source...'
        cmdisct_c3d'c3d '+fname_src+' '+fname_anat_mask+' -reslice-identity tmp.mask2src.nii.gz'
        print('>> '+cmd)
        status, output = commands.getstatusoutput(cmd)
        if debugging:
            print output

    # Pad the source image (because ants doesn't deform the extremities)
    print '\nPad source image and mask...'
   isct_c3dd = 'c3d '+fname_src+' -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o tmp.src_padded.nii.gz'
    print(">> "+cmd)
    status, output = commands.getstatusoutput(cmd)
    cmd = 'c3d tmp.mask2src.nii.gz -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o tmp.mask2src_padded.nii.gz'
    print(">> "+cmd)
    status, output = commands.getstatusoutput(cmd)

    # estimate non-affine transformation between template2anat and source image
    print '\nEstimate transformation: anat --> source (diffeomorphic transformation). Might take a couple a minutes...'
    #cmd = 'ants 3 -m CC['+fname_src+','+fname_anat+',2,4] -i 10x5x1 -t SyN[1] -r Gauss[0,2] --Restrict-Deformation 1x1x0 --number-of-affine-iterations 0x0x0 -o tmp.anat2src'
    cmd = 'ants 3 -m MI[tmp.src_padded.nii.gz,tmp.template2anat.nii.gz,2,4] -i 30x10x3 -t SyN[5] -r Gauss[0,0.5] --Restrict-Deformation 1x1x0 --number-of-affine-iterations 0x0x0 -o tmp.anat2src_padded'
    if fname_anat_mask != '':
        cmd = cmd+' -x tmp.mask2src_padded.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Convert the composite warping field into XYZ components
    print '\nConvert the composite warping field into XYZ components...'
    cmd = 'c3d -mcs tmp.anat2src_paddedWarp.nii.gz -oo tmp.anat2srcWarpX_padded.nii.gz tmp.anat2srcWarpY_padded.nii.gz tmp.anat2srcWarpZ_padded.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d -mcs tmp.anat2src_paddedInverseWarp.nii.gz -oo tmp.anat2srcInverseWarpX_padded.nii.gz tmp.anat2srcInverseWarpY_padded.nii.gz tmp.anat2srcInverseWarpZ_padded.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Unpad the warping fields (register to native source space)
    print '\nRemove padding on the warping fields...'
    cmd = 'c3d '+fname_src+' tmp.anat2srcWarpX_padded.nii.gz -reslice-identity tmp.anat2srcWarpX.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d '+fname_src+' tmp.anat2srcWarpY_padded.nii.gz -reslice-identity tmp.anat2srcWarpY.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d '+fname_src+' tmp.anat2srcWarpZ_padded.nii.gz -reslice-identity tmp.anat2srcWarpZ.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d '+fname_src+' tmp.anat2srcInverseWarpX_padded.nii.gz -reslice-identity tmp.anat2srcInverseWarpX.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d '+fname_src+' tmp.anat2srcInverseWarpY_padded.nii.gz -reslice-identity tmp.anat2srcInverseWarpY.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d '+fname_src+' tmp.anat2srcInverseWarpZ_padded.nii.gz -reslice-identity tmp.anat2srcInverseWarpZ.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # convert the unpadded component warping fields into composite (readable by ANTS)
    print '\nConvert the component warping fields into composite...'
    cmd = 'c3d tmp.anat2srcWarpX.nii.gz tmp.anat2srcWarpY.nii.gz tmp.anat2srcWarpZ.nii.gz -omc 3 tmp.anat2srcWarp.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output
    cmd = 'c3d tmp.anat2srcInverseWarpX.nii.gz tmp.anat2srcInverseWarpY.nii.gz tmp.anat2srcInverseWarpZ.nii.gz -omc 3 tmp.anat2srcInverseWarp.nii.gz'
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Apply transformation: anat --> source
    print '\nApply transformation: anat --> source...'
    cmd = 'WarpImageMultiTransform 3 '+fname_anat+' tmp.anat2src.nii.gz'+' tmp.anat2srcWarp.nii.gz -R '+fname_src
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Apply transformation: template --> anat --> source
    print '\nApply transformation: template --> anat --> source...'
    cmd = 'WarpImageMultiTransform 3 '+fname_template+' '+'tmp.template2src.nii.gz'+' tmp.anat2srcWarp.nii.gz tmp.template2anatWarp.nii.gz tmp.template2anatAffineLandmark.txt -R '+fname_src
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Apply transformation: source --> anat --> template
    print '\nApply transformation: source --> anat --> template...'
    cmd = 'WarpImageMultiTransform 3 '+fname_src+' '+'tmp.src2template.nii.gz'+' -i tmp.template2anatAffineLandmark.txt tmp.template2anatInverseWarp.nii.gz tmp.anat2srcInverseWarp.nii.gz -R '+fname_template
    print('>> '+cmd)
    status, output = commands.getstatusoutput(cmd)
    if debugging:
        print output

    # Display result
    print '\nCheck results:\n'
    print 'fslview tmp.src2template.nii.gz '+fname_template

# Extracts path, file and extension
# ==========================================================================================
def extract_fname(fname):
    # extract path
    path_fname = os.path.dirname(fname)+'/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname,'')
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"
    return path_fname, file_fname, ext_fname


# Print usage
# ==========================================================================================
def usage():
    path_func, file_func, ext_func = extract_fname(sys.argv[0])
    print 'USAGE: \n' \
        'Estimate transformations (only run it once):\n' \
        '    sct_register_to_template.py -s <source> -t <template> -f <landmark_template> -i <anat> -l <landmark_anat>\n' \
        '\n'\
        'Apply transformations:\n' \
        '    sct_register_to_template.py -s <source> -t <template> -n\n' \
        '\n'\
        '  -s       Source image to map to the template. For estimation, use similar contrast (e.g., b=0 image from DTI scan).\n' \
        '  -t       Template image.\n' \
        '  -f       Landmarks on the template.\n' \
        '  -i       Anatomical image of the subject.\n' \
        '  -l       Landmarks on the anatomic (!! should be in the same order as the landmarks in the template)\n' \
        '  -m       Mask encompassing the spinal cord on the anatomic (optional)\n' \
        '  -n       No estimation: only apply transformations. Warning! transformation should be present in the current folder.\n' \
        '           If using this flag, mandatory arguments are: -s -t.\n' \
        '  -h       help. Show this message.\n' \
        '\n'\
        'EXAMPLES:\n' \
        '\n'\
        'Estimate transformations (only run once):\n' \
        '     sct_register_to_template.py -i t2.nii.gz -t /users/bob/template/MNI-POLY-AMU_v1__T2.nii.gz -f /users/bob/template/landmarks_C3.nii.gz -l landmarks_C3.nii.gz -m /users/bob/template/mask_C3.nii.gz -b b0_mean.nii.gz\n' \
        '\n'\
        'Apply transformations. For example the following command will register FA to template:\n' \
        '     sct_register_to_template_python.py -n -b dti_FA.nii.gz -t /users/bob/template/MNI-POLY-AMU_v1__T2.nii.gz\n'
    sys.exit(2)

# Check existence of a file
# ==========================================================================================
def exist_image(fname):
    if os.path.isfile(fname) or os.path.isfile(fname+'.nii') or os.path.isfile(fname+'.nii.gz'):
        pass
    else:
        print('\nERROR: '+fname+' does not exist. Exit program.\n')
        sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    main()
