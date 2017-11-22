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
from msct_image import Image
from msct_multiatlas_seg import SegmentationParam
import os
import time
import sys
import getopt


class RegistrationParam:
    def __init__(self):
        self.debug = False
        self.fname_fixed = ''
        self.fname_moving = ''
        self.transformation = 'BSplineSyN'  # 'SyN'
        self.metric = 'CC'  # 'MeanSquares'
        self.gradient_step = '0.5'
        self.radius = '2'  # '4'
        self.interpolation = 'BSpline'
        self.iteration = '10x5' # '20x15'
        self.fname_seg_fixed = ''
        self.fname_seg_moving = ''
        self.fname_output = 'wm_registration.nii.gz'
        self.padding = '10'

        self.verbose = 1
        self.remove_temp = 1


def wm_registration(param):
    path, fixed_name, ext = sct.extract_fname(param.fname_fixed)
    path_moving, moving_name, ext = sct.extract_fname(param.fname_moving)
    path, fixed_seg_name, ext = sct.extract_fname(param.fname_seg_fixed)
    path_moving_seg, moving_seg_name, ext = sct.extract_fname(param.fname_seg_moving)


    # cropping in x & y directions
    fixed_name_temp = fixed_name + "_crop"
    cmd = "sct_crop_image -i " + fixed_name + ext + " -o " + fixed_name_temp + ext + " -m " + fixed_seg_name + ext + " -shift 10,10 -dim 0,1"
    sct.run(cmd)
    fixed_name = fixed_name_temp

    fixed_seg_name_temp = fixed_name+"_crop"
    sct.run("sct_crop_image -i " + fixed_seg_name + ext + " -o " + fixed_seg_name_temp + ext + " -m " + fixed_seg_name + ext + " -shift 10,10 -dim 0,1")
    fixed_seg_name = fixed_seg_name_temp

    # padding the images
    moving_name_pad = moving_name+"_pad"
    fixed_name_pad = fixed_name+"_pad"
    sct.run("c3d "+path_moving+moving_name+ext+" -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+moving_name_pad+ext)
    sct.run("c3d "+fixed_name+ext+" -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+fixed_name_pad+ext)
    moving_name = moving_name_pad
    fixed_name = fixed_name_pad

    moving_seg_name_pad = moving_seg_name+"_pad"
    sct.run("c3d "+path_moving_seg+moving_seg_name+ext+" -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+moving_seg_name_pad+ext)
    moving_seg_name = moving_seg_name_pad
    fixed_seg_name_pad = fixed_seg_name+"_pad"
    sct.run("c3d "+fixed_seg_name+ext+" -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+fixed_seg_name_pad+ext)
    fixed_seg_name = fixed_seg_name_pad

    # offset
    old_min = 0
    old_max = 1
    new_min = 100
    new_max = 200

    fixed_im = Image(fixed_name+ext)
    fixed_im.data = (fixed_im.data - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
    fixed_im.save()

    moving_im = Image(moving_name+ext)
    moving_im.data = (moving_im.data - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
    moving_im.save()

    # registration of the gray matter
    sct.printv('\nDeforming the image...', reg_param.verbose, 'normal')
    moving_name_reg = moving_name+"_deformed"

    if param.transformation == 'BSplineSyN':
        transfo_params = ',3,0'
    elif param.transforlation == 'SyN':     # SyN gives bad results...
        transfo_params = ',1,1'
    else:
        transfo_params = ''

    cmd = 'isct_antsRegistration --dimensionality 3 --interpolation '+param.interpolation+' --transform '+param.transformation+'['+param.gradient_step+transfo_params+'] --metric '+param.metric+'['+fixed_name+ext+','+moving_name+ext+',1,4] --output ['+moving_name_reg+','+moving_name_reg+ext+']  --convergence '+param.iteration+' --shrink-factors 2x1 --smoothing-sigmas 0x0 '

    cmd += " --masks ["+fixed_seg_name+ext+","+moving_seg_name + ext + "]"
    # cmd += " -m ["+fixed_seg_name+".nii,"+moving_seg_name+".nii]"
    sct.run(cmd)
    moving_name = moving_name_reg

    # removing offset
    fixed_im = Image(fixed_name+ext)
    fixed_im.data = (fixed_im.data - new_min)*(old_max - old_min)/(new_max - new_min) + old_min
    fixed_im.save()

    moving_im = Image(moving_name+ext)
    moving_im.data = (moving_im.data - new_min)*(old_max - old_min)/(new_max - new_min) + old_min
    moving_im.save()

    # un-padding the images
    moving_name_unpad = moving_name+"_unpadded"
    sct.run("sct_crop_image -i "+moving_name+ext+" -dim 2 -start "+str(int(param.padding)-1)+" -end -"+param.padding+" -o "+moving_name_unpad+ext)

    path_output, file_output, ext_output = sct.extract_fname(param.fname_output)
    warp_output = file_output+"0Warp"+ext_output
    inverse_warp_output = file_output+"0InverseWarp"+ext_output
    sct.run("mv "+moving_name+"0Warp.nii.gz "+warp_output)
    sct.run("mv "+moving_name+"0InverseWarp.nii.gz "+inverse_warp_output)
    moving_name = moving_name_unpad

    moving_name_out = file_output+ext_output
    sct.run("c3d "+fixed_name+ext+" "+moving_name+ext+" -reslice-identity -o "+moving_name_out+ext)

    return warp_output, inverse_warp_output


def segment_gm(target_fname='', sc_seg_fname='', path_to_label='', param=None):
    from sct_segment_graymatter import FullGmSegmentation
    level_fname = path_to_label + '/template/MNI-Poly-AMU_level.nii.gz'
    gmsegfull = FullGmSegmentation(target_fname, sc_seg_fname, None, level_fname, param=param)

    return gmsegfull.res_names['corrected_wm_seg'], gmsegfull.res_names['gm_seg']


def main(seg_params, reg_param, target_fname='', sc_seg_fname='', path_to_label=''):
    path_tmp = sct.tmp_create(basename="register_graymatter_old", verbose=verbose) + "/"

    target = 'target.nii.gz'
    sc_seg = 'sc_seg.nii.gz'
    label = 'label'

    sct.run('cp '+target_fname+' '+path_tmp+target)
    sct.run('cp '+sc_seg_fname+' '+path_tmp+sc_seg)
    sct.run('cp -r '+path_to_label+' '+path_tmp+label)
    sct.run('cp '+reg_param.template2anat2wm_gm_warp+' '+path_tmp+''.join(sct.extract_fname(reg_param.template2anat2wm_gm_warp)[1:]))
    sct.run('cp '+reg_param.wm_gm2template2anat_warp+' '+path_tmp+''.join(sct.extract_fname(reg_param.wm_gm2template2anat_warp)[1:]))

    os.chdir(path_tmp)

    wm_fname, gm_fname = segment_gm(target_fname=target, sc_seg_fname=sc_seg, path_to_label=label, param=seg_params)

    reg_param.fname_fixed = wm_fname
    reg_param.fname_moving = label + '/template/MNI-Poly-AMU_WM.nii.gz'
    reg_param.fname_seg_fixed = sc_seg
    reg_param.fname_seg_moving = label + '/template/MNI-Poly-AMU_cord.nii.gz'

    warp, inverse_warp = wm_registration(reg_param)

    # warp the T2star = output
    # sct.run('sct_apply_transfo -i ' + target_fname + ' -d ' + target_fname + ' -w ' + inverse_warp + ' -o ' + sct.extract_fname(target_fname)[1] + '_reg.nii.gz')
    # sct.run('sct_apply_transfo -i ' + sc_seg_fname + ' -d ' + sc_seg_fname + ' -w ' + inverse_warp + ' -o ' + sct.extract_fname(sc_seg_fname)[1] + '_reg.nii.gz  -x nn ')

    # concatenate transformations
    sct.run('sct_concat_transfo -w '+''.join(sct.extract_fname(reg_param.template2anat2wm_gm_warp)[1:])+','+warp+' -d '+target+' -o warp_template2anat2wm_gm')
    sct.run('sct_concat_transfo -w '+inverse_warp+','+''.join(sct.extract_fname(reg_param.wm_gm2template2anat_warp)[1:])+' -d '+target+' -o warp_wm_gm2template2anat')

    sct.run('cp warp_template2anat2wm_gm* ../'+sct.extract_fname(reg_param.template2anat2wm_gm_warp)[1]+'_wm.nii.gz')
    sct.run('cp warp_wm_gm2template2anat* ../'+sct.extract_fname(reg_param.wm_gm2template2anat_warp)[1]+'_wm.nii.gz')

    os.chdir('..')

    sct.printv('Done!\n You can apply the following commands to correct the registration of the template to :\n'
               'sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,'+sct.extract_fname(reg_param.wm_gm2template2anat_warp)[1]+'_wm.nii.gz  -d '+target_fname+' -o warp_template2t2star_with_wm.nii.gz\n'
               ' sct_warp_template -d '+target_fname+' -w warp_template2t2star_with_wm.nii.gz', verbose, 'info')
    # remove temporary file
    if reg_param.remove_temp == 1:
        sct.printv('\nRemove temporary files...', verbose, 'normal')
        sct.run("rm -rf "+path_tmp)


if __name__ == "__main__":
    reg_param = RegistrationParam()
    gm_seg_param = SegmentationParam()
    input_target_fname = ''
    input_sc_seg_fname = ''
    path_to_label = ''

    if reg_param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        param_default = RegistrationParam()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Register the template on a gray matter segmentation')  # TODO: change description
        parser.usage.addSection('Segmentation parameters')
        parser.add_option(name="-i",
                          type_value="file",
                          description="T2star target image (or image with a white/gray matter contrast)",
                          mandatory=True,
                          example='t2star.nii.gz')
        parser.add_option(name="-iseg",
                          type_value="file",
                          description="Spinal cord segmentation of the T2star target",
                          mandatory=True,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-anat",
                          type_value="file",
                          description="Anatomic image or template registered on the anatomical space (better)",
                          mandatory=True,
                          example='../t2/template2anat.nii.gz')
        parser.add_option(name="-anat-seg",
                          type_value="file",
                          description="Segmentation of the spinal cord on the anatomic image or on the template registered on the anatomical space (better)",
                          mandatory=True,
                          example='../t2/label/template/MNI-Poly-AMU_cord.nii.gz')
        parser.add_option(name="-model",
                          type_value="folder",
                          description="Path to the model data",
                          mandatory=False,
                          example='/home/jdoe/gm_seg_model_data/')
        '''
        parser.add_option(name="-i",
                          type_value="file",
                          description="Fixed image : the white matter automatic segmentation (should be probabilistic)",
                          mandatory=True,
                          example='wm_seg.nii.gz')
        parser.add_option(name="-d",
                          type_value="file",
                          description="Moving image: the white matter probabilistic segmentation from the template",
                          mandatory=True,
                          example='MNI-Poly-AMU_WM.nii.gz')
        parser.add_option(name="-o",
                          type_value="str",
                          description="Output image name",
                          mandatory=False,
                          example='moving_to_fixed.nii.gz')
        parser.add_option(name="-iseg",
                          type_value="file",
                          description="Spinal cord segmentation of the fixed image",
                          mandatory=False,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-dseg",
                          type_value="file",
                          description="Spinal cord segmentation of the moving image (should be the same)",
                          mandatory=False,
                          example='sc_seg.nii.gz')
        '''
        parser.usage.addSection('Registration parameters')
        parser.add_option(name="-warp",
                          type_value=[[','], 'file'],
                          description="Warping field from the template2anat to the WM/GM contrasted image and inverse that was used to register the template to the WM/GM contrasted image",
                          mandatory=True,
                          example="warp_template2anat2t2star.nii,warp_t2star2template2anat.nii")
        parser.add_option(name="-t",
                          type_value='multiple_choice',
                          description="type of transformation",
                          mandatory=False,
                          default_value='BSplineSyN',
                          example=['SyN', 'BSplineSyN'])
        parser.add_option(name="-m",
                          type_value='multiple_choice',
                          description="Metric used for the registration",
                          mandatory=False,
                          default_value='CC',
                          example=['CC', 'MeanSquares'])
        parser.add_option(name="-reg-o",
                          type_value="str",
                          description="Output image name",
                          mandatory=False,
                          example='moving_to_fixed.nii.gz')
        parser.usage.addSection("Misc")
        parser.add_option(name="-r",
                          type_value='multiple_choice',
                          description="Remove temporary files",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-v",
                          type_value='multiple_choice',
                          description="Verbose",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1', '2'])

        arguments = parser.parse(sys.argv[1:])

        input_target_fname = arguments["-i"]
        input_sc_seg_fname = arguments["-iseg"]
        gm_seg_param.todo_model = 'load'
        path_to_label = arguments["-label"]
        verbose = 1
        if "-model" in arguments:
            gm_seg_param.path_model = arguments["-model"]
        '''
        reg_param.fname_ref = arguments["-i"]
        if "-iseg" in arguments:
            reg_param.fname_seg_fixed = arguments["-iseg"]
        reg_param.fname_moving = arguments["-d"]
        if "-dseg" in arguments:
            reg_param.fname_seg_moving = arguments["-dseg"]

        if "-o" in arguments:
            reg_param.fname_output = arguments["-o"]
        else:
            reg_param.fname_output = sct.extract_fname(reg_param.fname_moving)[1] + '_to_' + sct.extract_fname(reg_param.fname_ref)[1] +  sct.extract_fname(reg_param.fname_ref)[2]
        '''
        reg_param.template2anat2wm_gm_warp, reg_param.wm_gm2template2anat_warp = arguments["-warp"]
        if "-t" in arguments:
            reg_param.transformation = arguments["-t"]
        if "-m" in arguments:
            reg_param.metric = arguments["-m"]
        if "-r" in arguments:
            reg_param.remove_temp = int(arguments["-r"])
        if "-v" in arguments:
            verbose = int(arguments["-v"])

        gm_seg_param.verbose = verbose
        reg_param.verbose = verbose

    main(gm_seg_param, reg_param, target_fname=input_target_fname, sc_seg_fname=input_sc_seg_fname, path_to_label=path_to_label)


