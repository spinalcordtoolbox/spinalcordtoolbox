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
from msct_parser import Parser
import os
import time
import sys
import getopt


class Param:
    def __init__(self):
        self.debug = False
        self.fname_ref = ''
        self.fname_moving = ''
        self.transformation = 'SyN'
        self.metric = 'MI'  # 'MeanSquares'  # 'CC'
        self.gradient_step = '0.5'
        self.radius = '4'
        self.iteration='20x15'
        self.fname_seg_fixed = ''
        self.fname_seg_moving = ''
        self.fname_output = ''
        self.padding = '10'

        self.remove_temp = 1

def main(param):

    moving_name = 'moving'
    fixed_name = 'fixed'
    moving_seg_name = 'moving_seg'
    fixed_seg_name = 'fixed_seg'

    # Extract path/file/extension
    path_output, file_output, ext_output = sct.extract_fname(param.fname_output)

    path_tmp = sct.tmp_create()

    # copy files to temporary folder
    print('\nCopy files...')
    sct.run("c3d "+param.fname_moving+" -o "+path_tmp+"/"+moving_name+".nii")
    sct.run("c3d "+param.fname_ref+" -o "+path_tmp+"/"+fixed_name+".nii")
    if param.fname_seg_moving != '':
        sct.run("c3d "+param.fname_seg_moving+" -o "+path_tmp+"/"+moving_seg_name+".nii")
    if param.fname_seg_fixed != '':
        sct.run("c3d "+param.fname_seg_fixed+" -o "+path_tmp+"/"+fixed_seg_name+".nii")

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

    # cropping in x & y directions
    fixed_name_temp = fixed_name + "_crop"
    cmd = "sct_crop_image -i " + fixed_name + ".nii -o " + fixed_name_temp + ".nii -m " + fixed_seg_name + ".nii -shift 10,10 -dim 0,1"
    sct.run(cmd)
    fixed_name = fixed_name_temp
    if param.fname_seg_fixed != '':
        fixed_seg_name_temp = fixed_seg_name+"_crop"
        sct.run("sct_crop_image -i " + fixed_seg_name + ".nii -o " + fixed_seg_name_temp + ".nii -m " + fixed_seg_name + ".nii -shift 10,10 -dim 0,1")
        fixed_seg_name = fixed_seg_name_temp

    #sct_crop_image -i t2star_denoised.nii -o t2star_denoised_crop.nii -m ../t2star_seg.nii.gz -shift 10,10 -dim 0,1

    # padding the images
    moving_name_temp = moving_name+"_pad"
    fixed_name_temp = fixed_name+"_pad"
    sct.run("c3d "+moving_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+moving_name_temp+".nii")
    sct.run("c3d "+fixed_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+fixed_name_temp+".nii")
    moving_name = moving_name_temp
    fixed_name = fixed_name_temp
    if param.fname_seg_moving != '':
        moving_seg_name_temp = moving_seg_name+"_pad"
        sct.run("c3d "+moving_seg_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+moving_seg_name_temp+".nii")
        moving_seg_name = moving_seg_name_temp
    if param.fname_seg_fixed != '':
        fixed_seg_name_temp = fixed_seg_name+"_pad"
        sct.run("c3d "+fixed_seg_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+fixed_seg_name_temp+".nii")
        fixed_seg_name = fixed_seg_name_temp

    # binarise the moving image
    # moving_name_temp_bin = moving_name_temp + "_bin"
    # cmd = 'fslmaths ' + moving_name_temp + '.nii -thr 0.25 ' + moving_name_temp_bin + '.nii'
    # sct.run(cmd, 1)
    #
    # cmd = 'fslmaths ' + moving_name_temp_bin + '.nii -bin ' + moving_name_temp_bin + '.nii'
    # sct.run(cmd, 1)
    #
    # moving_name = moving_name_temp_bin


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
    cmd = "antsRegistration --dimensionality 3 --transform "+ param.transformation +"["+param.gradient_step+",3,0] --metric "+param.metric+"["+fixed_name+".nii,"+moving_name+".nii,1,"+param.radius+"] --convergence "+param.iteration+" --shrink-factors 2x1 --smoothing-sigmas 0mm --Restrict-Deformation 1x1x0 --output ["+moving_name_temp+","+moving_name_temp+".nii]"
    if param.fname_seg_moving != '':
        cmd += " --masks ["+fixed_seg_name+".nii,"+moving_seg_name+".nii]"
    sct.run(cmd)
    moving_name = moving_name_temp

    moving_name_temp = moving_name+"_unpadded"
    sct.run("sct_crop_image -i "+moving_name+".nii -dim 2 -start "+param.padding+" -end -"+param.padding+" -o "+moving_name_temp+".nii")
    sct.run("mv "+moving_name+"0Warp.nii.gz "+file_output+"0Warp"+ext_output)
    sct.run("mv "+moving_name+"0InverseWarp.nii.gz "+file_output+"0InverseWarp"+ext_output)
    moving_name = moving_name_temp

    # TODO change "fixed.nii"
    moving_name_temp = file_output+ext_output
    #sct.run("c3d "+fixed_name+".nii "+file_output+ext_output+" -reslice-identity  -o "+file_output+'_register'+ext_output)
    sct.run("c3d fixed.nii "+moving_name+".nii -reslice-identity -o "+file_output+ext_output)

    # move output files to initial folder
    sct.run("cp "+file_output+"* ../")

    # remove temporary file
    if param.remove_temp == 1:
        os.chdir('../')
        print('\nRemove temporary files...')
        sct.run("rm -rf "+path_tmp)

    return


if __name__ == "__main__":
    param = Param()
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        param_default = Param()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Register the template on a gray matter segmentation')
        parser.add_option(name="-i",
                          type_value="file",
                          description="Fixed image",  # TODO : specify which image to use
                          mandatory=True,
                          example='wm_seg.nii.gz')
        parser.add_option(name="-d",
                          type_value="file",
                          description="moving image",  # TODO : specify which image to use
                          mandatory=True,
                          example='MNI-Poly-AMU_WM.nii.gz')
        parser.add_option(name="-o",
                          type_value="str",
                          description="output name",  # TODO : specify which image to use
                          mandatory=False,
                          example='moving_to_fixed.nii.gz')
        parser.add_option(name="-s",
                          type_value="file",
                          description="Spinal cord segmentation of the fixed image",  # TODO : specify which image to use
                          mandatory=True,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-g",
                          type_value="file",
                          description="Spinal cord segmentation of the moving image",  # TODO : specify which image to use
                          mandatory=True,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-t",
                          type_value='multiple_choice',
                          description="type of transformation",
                          mandatory=False,
                          default_value='SyN',
                          example=['SyN', 'BSplineSyN'])
        parser.add_option(name="-m",
                          type_value='multiple_choice',
                          description="Metric used for the registration",
                          mandatory=False,
                          default_value='MeanSquares',
                          example=['CC', 'MeanSquares'])
        parser.add_option(name="-r",
                          type_value='multiple_choice',
                          description="Remove temporary files",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])

        arguments = parser.parse(sys.argv[1:])


        param.fname_ref = arguments["-i"]
        param.fname_seg_fixed = arguments["-s"]
        param.fname_moving = arguments["-d"]
        param.fname_seg_moving = arguments["-g"]

        if "-o" in arguments:
            param.fname_output = arguments["-o"]
        else:
            param.fname_output = sct.extract_fname(param.fname_moving)[1] + '_to_' + sct.extract_fname(param.fname_ref)[1] +  sct.extract_fname(param.fname_ref)[2]
        if "-t" in arguments:
            param.transformation = arguments["-t"]
        if "-m" in arguments:
            param.metric = arguments["-m"]
        if "-r" in arguments:
            param.remove_temp = int(arguments["-r"])

    main(param)


