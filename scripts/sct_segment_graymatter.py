#!/usr/bin/env python
#
# This program returns the grey matter segmentation given anatomical, landmarks and t2star images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Augustin Roux
# Created: 2014-10-18
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sct_utils as sct
import os
import time
import sys
import getopt


def main():
    fname_ref = ''
    fname_moving = ''
    transformation = 'SyN'
    gradient_step = '0.25'
    fname_seg_fixed = ''
    fname_seg_moving = ''
    fname_output = ''
    padding = '5'

    moving_name = 'moving'
    fixed_name = 'fixed'
    moving_seg_name = 'moving_seg'
    fixed_seg_name = 'fixed_seg'

    remove_temp = 1

    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:d:t:s:g:o:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-d'):
            fname_moving = arg
        elif opt in ('-i'):
            fname_ref = arg
        elif opt in ('-t'):
            transformation = arg
        elif opt in ('-s'):
            fname_seg_fixed = arg
        elif opt in ('-g'):
            fname_seg_moving = arg
        elif opt in ('-o'):
            fname_output = arg

    if fname_moving == '' or fname_ref == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_ref)
    sct.check_file_exist(fname_moving)
    if (fname_seg_moving != '' and fname_seg_moving == '') or (fname_seg_moving == '' and fname_seg_moving != ''):
        print('\nERROR: You need to provide one mask for each image (moving and fixed)')
        usage()
    if fname_seg_moving != '':
        sct.check_file_exist(fname_seg_moving)
    if fname_seg_fixed != '':
        sct.check_file_exist(fname_seg_fixed)

    # Extract path/file/extension
    path_output, file_output, ext_output = sct.extract_fname(fname_output)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    print('\nCopy files...')
    sct.run("sct_c3d "+fname_moving+" -o "+path_tmp+"/"+moving_name+".nii")
    sct.run("sct_c3d "+fname_ref+" -o "+path_tmp+"/"+fixed_name+".nii")
    if fname_seg_moving != '':
        sct.run("sct_c3d "+fname_moving+" -o "+path_tmp+"/"+moving_seg_name+".nii")
    if fname_seg_fixed != '':
        sct.run("sct_c3d "+fname_moving+" -o "+path_tmp+"/"+fixed_seg_name+".nii")

    # go to tmp folder
    os.chdir(path_tmp)

    # denoising the fixed image using non-local means from dipy
    # file = nibabel.load(fixed_name+".nii")
    # data = file.get_data()
    # hdr = file.get_header()
    # fixed_name_temp = fixed_name+"_denoised"
    # data_denoised = nlmeans(data,3)
    # fixed_name = fixed_name_temp
    # hdr.set_data_dtype('uint32') # set imagetype to uint32
    # img = nibabel.Nifti1Image(data_denoised, None, hdr)
    # nibabel.save(img, fixed_name+".nii.gz")

    # padding the images
    moving_name_temp = moving_name+"_pad"
    fixed_name_temp = fixed_name+"_pad"
    sct.run("sct_c3d "+moving_name+".nii -pad 0x0x"+padding+"vox 0x0x"+padding+"vox 0 -o "+moving_name_temp+".nii")
    sct.run("sct_c3d "+fixed_name+".nii -pad 0x0x"+padding+"vox 0x0x"+padding+"vox 0 -o "+fixed_name_temp+".nii")
    moving_name = moving_name_temp
    fixed_name = fixed_name_temp
    if fname_seg_moving != '':
        moving_seg_name_temp = moving_seg_name+"_pad"
        sct.run("sct_c3d "+moving_seg_name+".nii -pad 0x0x"+padding+"vox 0x0x"+padding+"vox 0 -o "+moving_seg_name_temp+".nii")
        moving_seg_name = moving_seg_name_temp
    if fname_seg_fixed != '':
        fixed_seg_name_temp = fixed_seg_name+"_pad"
        sct.run("sct_c3d "+fixed_seg_name+".nii -pad 0x0x"+padding+"vox 0x0x"+padding+"vox 0 -o "+fixed_seg_name_temp+".nii")
        fixed_seg_name = fixed_seg_name_temp


    # register template to anat file: this generate warp_template2anat.nii
    # cmd = "sct_register_to_template -i " + fname_anat + " -l " + fname_landmarks + " -m " + fname_seg + " -s normal"
    # sct.run(cmd)
    # warp_template2anat = ""

    # # register anat file to t2star: generate  warp_anat2t2star
    # cmd = "sct_register_multimodal -i " + fname_anat + " -d " + fname_t2star
    # sct.run(cmd)
    # warp_anat2t2star = ""

    # # concatenation of the two warping fields
    # warp_template2t2star = "warp_template2t2star.nii.gz"
    # cmd = "sct_concat_transfo -w " + warp_template2anat + ',' + warp_anat2t2star + " -d " + fname_t2star + " -o " + warp_template2t2star
    # sct.run(cmd)

    # # apply the concatenated warping field to the template
    # cmd = "sct_warp_template -d " + fname_t2star + " -w " + warp_template2anat + " -s 1 -o template_in_t2star_space"
    # sct.run(cmd)


    #sct_register_to_template --> warp_template2anat
    # register anat file to t2star
    #sct_register_multimodal --> warp_anat2t2star
    # concatenate warp_template2anat with warp_anat2t2star
    #--> warp_template2t2star


    # registration of the grey matter
    print('\nDeforming the image...')
    moving_name_temp = moving_name+"_deformed"
    cmd = "sct_antsRegistration --dimensionality 3 --transform "+ transformation +"["+gradient_step+",3,0] --metric CC["+moving_name+".nii,"+fixed_name+".nii,1,5] --convergence 20x15 --shrink-factors 2x1 --smoothing-sigmas 0mm --Restrict-Deformation 1x1x0 --output ["+moving_name_temp+","+moving_name_temp+".nii]"
    if fname_seg_moving != '':
        cmd += " --masks["+moving_seg_name+".nii,"+fixed_seg_name+".nii]"
    sct.run(cmd)
    moving_name = moving_name_temp

    moving_name_temp = file_output+ext_output
    sct.run("sct_crop_image -i "+moving_name+".nii -dim 1 -start "+padding+" -end -"+padding+" -o "+file_output+ext_output)
    moving_name = moving_name_temp

    # move output files to initial folder
    sct.run("cp "+file_output+"* ../")

    # remove temporary file
    if remove_temp == 1:
        print('\nRemove temporary files...')
        sct.run("rm -rf "+path_tmp)

    return


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
        '  This function returns the grey matter segmentation.\n' \
        '\n'\
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <anat volume> -t <t2star volume> -l <landmarks>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i                   input image (with white/gray matter contrast).\n' \
        '  -d                   moving image.\n' \
        '  -o                   output name (for warping field and image).\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -t                   transformation {SyN, BSplineSyN}.\n' \
        '  -s                   input image segmentation.\n' \
        '  -g                   moving image segmentation.\n'
    sys.exit(2)



if __name__ == "__main__":
    # initialize parameters
    # param = param()
    # call main function
    main()


