#!/usr/bin/env python
#
# This program register the template to the GM/WM contrasted image, segment the gray matter on it,
#  and correct the template registration using the graymatter automatic segmentation
# returns the grey matter segmentation given anatomical, landmarks and t2star images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Augustin Roux, Sara Dupont
# Created: 2014-10-18
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from msct_parser import Parser
from msct_image import Image
from msct_multiatlas_seg import SegmentationParam
import sct_utils as sct
import os
import sys
import time


class MultimodalRegistrationParam:
    def __init__(self):
        self.target = ''
        self.anat = ''
        self.target_seg = ''
        self.anat_seg = ''
        self.output = 'anat2target'
        self.warp_template2anat = ''
        self.p = 'step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,metric=MI,iter=5,shrink=2'
            # 'step=1,type=seg,algo=syn,metric=CC,iter=5:step=2,type=im,algo=slicereg,metric=MeanSquares,iter=10'


class WmRegistrationParam:
    def __init__(self):
        self.debug = False
        self.fname_fixed = ''
        self.fname_moving = ''
        self.fname_seg_fixed = ''
        self.fname_seg_moving = ''
        self.transformation = 'BSplineSyN' # 'slicereg'  # 'BSplineSyN'  # 'SyN'
        self.metric = 'MeanSquares'  # 'CC'
        self.gradient_step = '0.5'
        self.radius = '2'  # '4'
        self.interpolation = 'BSpline'  # 'NearestNeighbor'
        self.iteration = '5x10'  # '20x15'
        self.fname_output = 'wm_registration.nii.gz'
        self.padding = '10'


def reg_multimodal_warp(anat, target, anat_seg, target_seg, warp_template2anat, param):
    status, output = sct.run('sct_register_multimodal -i '+anat+' -d '+target+' -iseg '+anat_seg+' -dseg '+target_seg+' -p '+param.p)
    if status != 0:
        sct.printv('WARNING: an error occurred ...', verbose, 'warning')
        sct.printv(output, verbose, 'normal')
        return None
    else:
        warp_template2target = 'warp_template2target.nii.gz'
        warp_anat2target = 'warp_'+sct.extract_fname(anat)[1]+'2'+sct.extract_fname(target)[1]+'.nii.gz'
        warp_target2anat = 'warp_'+sct.extract_fname(target)[1]+'2'+sct.extract_fname(anat)[1]+'.nii.gz'

        sct.run('sct_concat_transfo -w '+warp_template2anat+','+warp_anat2target+'  -d '+target+' -o  '+warp_template2target)
        sct.run('sct_warp_template -d '+target+' -w '+warp_template2target)
        label_folder = 'label_original_reg'
        sct.run('mv label/ '+label_folder)
        return warp_anat2target, warp_target2anat, label_folder


def segment_gm(target_fname='', sc_seg_fname='', path_to_label='', param=None):
    from sct_segment_graymatter import FullGmSegmentation
    level_fname = path_to_label + '/template/MNI-Poly-AMU_level.nii.gz'
    gmsegfull = FullGmSegmentation(target_fname, sc_seg_fname, None, level_fname, param=param)

    return gmsegfull.res_names['corrected_wm_seg'], gmsegfull.res_names['gm_seg']


def wm_registration(param, smooth=False, init=False):
    path, fixed_name, ext = sct.extract_fname(param.fname_fixed)
    path_moving, moving_name, ext = sct.extract_fname(param.fname_moving)
    path, fixed_seg_name, ext = sct.extract_fname(param.fname_seg_fixed)
    path_moving_seg, moving_seg_name, ext = sct.extract_fname(param.fname_seg_moving)

    # cropping in x & y directions
    fixed_name_temp = fixed_name + "_crop"
    cmd = "sct_crop_image -i " + fixed_name + ext + " -o " + fixed_name_temp + ext + " -m " + fixed_seg_name + ext + " -shift 5,5 -dim 0,1"
    sct.run(cmd)
    fixed_name = fixed_name_temp

    fixed_seg_name_temp = fixed_seg_name+"_crop"
    sct.run("sct_crop_image -i " + fixed_seg_name + ext + " -o " + fixed_seg_name_temp + ext + " -m " + fixed_seg_name + ext + " -shift 5,5 -dim 0,1")
    fixed_seg_name = fixed_seg_name_temp

    # padding the images
    moving_name_pad = moving_name+"_pad"
    fixed_name_pad = fixed_name+"_pad"
    sct.run('sct_maths -i '+path_moving+moving_name+ext+' -o '+moving_name_pad+ext+' -pad 0,0,'+str(param.padding))
    sct.run('sct_maths -i '+fixed_name+ext+' -o '+fixed_name_pad+ext+' -pad 0,0,'+str(param.padding))
    moving_name = moving_name_pad
    fixed_name = fixed_name_pad

    moving_seg_name_pad = moving_seg_name+"_pad"
    sct.run('sct_maths -i '+path_moving_seg+moving_seg_name+ext+' -o '+moving_seg_name_pad+ext+' -pad 0,0,'+str(param.padding))
    moving_seg_name = moving_seg_name_pad

    fixed_seg_name_pad = fixed_seg_name+"_pad"
    sct.run('sct_maths -i '+fixed_seg_name+ext+' -o '+fixed_seg_name_pad+ext+' -pad 0,0,'+str(param.padding))
    fixed_seg_name = fixed_seg_name_pad

    if param.metric == 'MeanSquares':
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

    if smooth:
        # smoothing the images to register
        sct.printv('\nSmoothing the images...', verbose, 'normal')
        fixed_smooth = fixed_name+'_smooth'
        moving_smooth = moving_name+'_smooth'
        sct.run('sct_maths -i '+fixed_name+ext+' -smooth 1 -o '+fixed_smooth+ext)
        sct.run('sct_maths -i '+moving_name+ext+' -smooth 1 -o '+moving_smooth+ext)
        fixed_name = fixed_smooth
        moving_name = moving_smooth

    # registration of the gray matter
    sct.printv('\nDeforming the image...', verbose, 'normal')
    moving_name_reg = moving_name+"_deformed"

    param_multimodal_reg = 'step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,shrink=2'
    if param.transformation == 'BSplineSyN':
        transfo_params = ',3,0'
    elif param.transformation == 'SyN':     # SyN gives bad results...
        transfo_params = ',1,1'
    else:
        transfo_params = ''

    cmd = 'sct_register_multimodal -i '+moving_name+ext+' -d '+fixed_name+ext+' -iseg '+moving_seg_name+ext+' -dseg '+fixed_seg_name+ext+' -p '+param_multimodal_reg+' -o '+moving_name_reg
    ext_multimodal = '.nii'
    '''
    cmd = 'isct_antsRegistration --dimensionality 3 --interpolation '+param.interpolation+' --transform '+param.transformation+'['+param.gradient_step+transfo_params+'] --metric '+param.metric+'['+fixed_name+ext+','+moving_name+ext+',1,4] --output ['+moving_name_reg+','+moving_name_reg+ext+']  --convergence '+param.iteration+' --shrink-factors 1x1 --smoothing-sigmas 0x0 '
    if init:
        # initialize the transformation using the intensity
        cmd += ' -r ['+fixed_name+ext+','+moving_name+ext+',1] '

    cmd += " --masks ["+fixed_seg_name+ext+","+moving_seg_name + ext + "]"
    # cmd += " -m ["+fixed_seg_name+".nii,"+moving_seg_name+".nii]"
    '''
    sct.run(cmd)

    if init:
        # transform the .mat warping field into a nifti file:
        warp0_mat = moving_name_reg+'0GenericAffine.mat'
        warp0, warp0_inv = transform_mat_to_warp(warp0_mat, moving_name_reg+ext, fixed_name+ext)
        if param.transformation == 'Affine':
            warp_tot = warp0
            inverse_warp_tot = warp0_inv
        else:
            warp1 = moving_name_reg+'1Warp.nii.gz'
            warp1_inv = moving_name_reg+'1InverseWarp.nii.gz'

            # Concatenate the warping fields from the 2 steps into a warp total and an inverse
            warp_tot = moving_name_reg+'_warp_tot'+ext
            inverse_warp_tot = moving_name_reg+'_inverse_warp_tot'+ext

            sct.run('sct_concat_transfo -w '+warp0+','+warp1+' -d '+fixed_name+ext+' -o '+warp_tot)
            sct.run('sct_concat_transfo -w '+warp1_inv+','+warp0_inv+' -d '+moving_name+ext+' -o '+inverse_warp_tot)
    else:
        warp_tot = moving_name_reg+'0Warp.nii.gz'
        inverse_warp_tot = moving_name_reg+'0InverseWarp.nii.gz'


    if param.metric == 'MeanSquares':
        # removing offset
        fixed_im = Image(fixed_name+ext)
        fixed_im.data = (fixed_im.data - new_min)*(old_max - old_min)/(new_max - new_min) + old_min
        fixed_im.save()

        moving_im = Image(moving_name_reg+ext_multimodal)
        moving_im.data = (moving_im.data - new_min)*(old_max - old_min)/(new_max - new_min) + old_min
        moving_im.save()


    # un-padding the images
    moving_name_unpad = moving_name_reg+"_unpadded"
    fixed_name_unpad = fixed_name+"_unpadded"
    sct.run("sct_crop_image -i "+moving_name_reg+ext+" -dim 2 -start "+str(int(param.padding))+" -end -"+param.padding+" -o "+moving_name_unpad+ext)
    sct.run("sct_crop_image -i "+fixed_name+ext+" -dim 2 -start "+str(int(param.padding))+" -end -"+param.padding+" -o "+fixed_name_unpad+ext)

    '''
    # warp_tot = moving_name_reg+"0Warp.nii.gz"
    # inverse_warp_tot = moving_name_reg+"0InverseWarp.nii.gz"
    warp_unpad = sct.add_suffix(warp_tot, '_unpad')  # moving_name_reg+"0Warp_unpad.nii.gz"
    inverse_warp_unpad =  sct.add_suffix(inverse_warp_tot, '_unpad')  # moving_name_reg+"0InverseWarp_unpad.nii.gz"
    sct.run("sct_crop_image -i "+warp_tot+" -dim 2 -start "+str(int(param.padding))+" -end -"+param.padding+" -o "+warp_unpad)
    sct.run("sct_crop_image -i "+inverse_warp_tot+" -dim 2 -start "+str(int(param.padding))+" -end -"+param.padding+" -o "+inverse_warp_unpad)
    '''


    path_output, file_output, ext_output = sct.extract_fname(param.fname_output)
    warp_output = file_output+"_Warp"+ext_output
    inverse_warp_output = file_output+"_InverseWarp"+ext_output


    # with multimodal_reg
    sct.run("mv warp_"+moving_name+"2"+fixed_name+".nii.gz "+warp_output)
    sct.run("mv warp_"+fixed_name+"2"+moving_name+".nii.gz "+inverse_warp_output)
    '''
    # with antsReg
    sct.run("mv "+warp_unpad+" "+warp_output)
    sct.run("mv "+inverse_warp_unpad+" "+inverse_warp_output)
    '''

    moving_name_out = file_output+ext_output
    # put the result and the reference in the same space using a registration with ANTs with no iteration:
    sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+fixed_name_unpad+ext+','+moving_name_unpad+ext+',1,16] -o [regAffine,'+moving_name_out+'] -n BSpline[3] -c 0 -f 1 -s 0')

    return warp_output, inverse_warp_output


def transform_mat_to_warp(warp_mat_name, src, dest, dim=3):
    from msct_register_regularized import generate_warping_field
    path_warp, name_warp, ext_warp = sct.extract_fname(warp_mat_name)
    new_ext = '.nii.gz'
    name_output_warp = name_warp+new_ext
    name_output_warp_inverse = name_warp+'_inverse'+new_ext
    name_warp_null = 'warp_null.nii.gz'
    name_warp_null_inv = 'warp_null_inv.nii.gz'

    # Generating null nifti warping fields
    nx, ny, nz, nt, px, py, pz, pt = Image(src).dim
    nx_d, ny_d, nz_d, nt_d, px_d, py_d, pz_d, pt_d = Image(dest).dim
    x_trans = [0 for i in range(nz)]
    x_trans_d = [0 for i in range(nz_d)]
    y_trans= [0 for i in range(nz)]
    y_trans_d = [0 for i in range(nz_d)]
    generate_warping_field(src, x_trans=x_trans, y_trans=y_trans, fname=name_warp_null, verbose=0)
    generate_warping_field(dest, x_trans=x_trans_d, y_trans=y_trans_d, fname=name_warp_null_inv, verbose=0)
    # Concatenating mat wrp and null nifti warp to obtain equivalent nifti warp to mat warp
    #if fails, try to inverse src and dest
    sct.run('isct_ComposeMultiTransform '+str(dim)+' ' + name_output_warp + ' -R ' + src + ' ' + name_warp_null + ' ' + warp_mat_name) # after output :
    sct.run('isct_ComposeMultiTransform '+str(dim)+' ' + name_output_warp_inverse + ' -R ' + dest + ' ' + name_warp_null_inv + ' -i ' + warp_mat_name) # after output :
    return name_output_warp, name_output_warp_inverse


########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################
def main():
    path_tmp = sct.tmp_create(basename="register_graymatter", verbose=verbose) + "/"

    target = 'target.nii.gz'
    target_seg = 'target_seg.nii.gz'
    anat = 'anat.nii.gz'
    anat_seg = 'anat_seg.nii.gz'
    warp_template2anat = 'warp_template2anat.nii.gz'

    sct.run('cp '+multimodal_reg_param.anat+' '+path_tmp+anat)
    sct.run('cp '+multimodal_reg_param.target+' '+path_tmp+target)
    sct.run('cp '+multimodal_reg_param.anat_seg+' '+path_tmp+anat_seg)
    sct.run('cp '+multimodal_reg_param.target_seg+' '+path_tmp+target_seg)
    sct.run('cp '+multimodal_reg_param.warp_template2anat+' '+path_tmp+warp_template2anat)

    os.chdir(path_tmp)
    # registration of the template to the target
    warp_anat2target, warp_target2anat, label_original = reg_multimodal_warp(anat, target, anat_seg, target_seg, warp_template2anat, multimodal_reg_param)

    # segmentation of the gray matter
    gm_seg_param.res_type = 'binary'
    wm_fname, gm_fname = segment_gm(target_fname=target, sc_seg_fname=target_seg, path_to_label=label_original, param=gm_seg_param)
    wm_seg_out = sct.extract_fname(multimodal_reg_param.target)[1]+'_wm_seg.nii.gz'
    gm_seg_out = sct.extract_fname(multimodal_reg_param.target)[1]+'_gm_seg.nii.gz'
    sct.run('cp '+wm_fname+' ../'+wm_seg_out)
    sct.run('cp '+gm_fname+' ../'+gm_seg_out)

    # registration of the template WM to the automatic Wm segmentation
    wm_reg_param.fname_fixed = wm_fname
    wm_reg_param.fname_moving = label_original + '/template/MNI-Poly-AMU_WM.nii.gz'
    wm_reg_param.fname_seg_fixed = target_seg
    wm_reg_param.fname_seg_moving = label_original + '/template/MNI-Poly-AMU_cord.nii.gz'

    if gm_seg_param.res_type == 'binary':
        # binarize template WM
        template_wm = Image(wm_reg_param.fname_moving)
        template_wm.data = (template_wm.data > 0.5).astype(int)
        template_wm.save()

    warp, inverse_warp = wm_registration(wm_reg_param)

    warp_anat2target_corrected = 'warp_'+sct.extract_fname(multimodal_reg_param.anat)[1]+'2'+sct.extract_fname(multimodal_reg_param.target)[1]+'_corrected_wm.nii.gz'
    warp_target2anat_corrected = 'warp_'+sct.extract_fname(multimodal_reg_param.target)[1]+'2'+sct.extract_fname(multimodal_reg_param.anat)[1]+'_corrected_wm.nii.gz'

    sct.run('sct_concat_transfo -w '+warp_anat2target+','+warp+' -d '+target+' -o '+warp_anat2target_corrected)
    sct.run('sct_concat_transfo -w '+inverse_warp+','+warp_target2anat+' -d '+anat+' -o '+warp_target2anat_corrected)

    target_reg = sct.extract_fname(multimodal_reg_param.target)[1]+'_reg_corrected.nii.gz'
    anat_reg = sct.extract_fname(multimodal_reg_param.anat)[1]+'_reg_corrected.nii.gz'
    sct.run('sct_apply_transfo -i '+target+' -d '+anat+' -w '+warp_target2anat_corrected+' -o '+target_reg)
    sct.run('sct_apply_transfo -i '+anat+' -d '+target+' -w '+warp_anat2target_corrected+' -o '+anat_reg)

    sct.run('cp '+target_reg+' ../'+target_reg)
    sct.run('cp '+anat_reg+' ../'+anat_reg)
    sct.run('cp '+warp_anat2target_corrected+' ../'+warp_anat2target_corrected)
    sct.run('cp '+warp_target2anat_corrected+' ../'+warp_target2anat_corrected)

    sct.printv('Done!\n'
               'You can warp the template to the target using the following command lines:\n'
               'sct_concat_transfo -w '+multimodal_reg_param.warp_template2anat+','+warp_anat2target_corrected+' -d '+multimodal_reg_param.target+' -o warp_template2'+sct.extract_fname(multimodal_reg_param.target)[1]+'.nii.gz\n'
               'sct_warp_template -d '+multimodal_reg_param.target+' -w warp_template2'+sct.extract_fname(multimodal_reg_param.target)[1]+'.nii.gz', verbose, 'info')
    os.chdir('..')

    if remove:
        sct.printv('\nRemove temporary files...', verbose, 'normal')
        sct.run("rm -rf "+path_tmp)



if __name__ == "__main__":
    multimodal_reg_param = MultimodalRegistrationParam()
    wm_reg_param = WmRegistrationParam()
    gm_seg_param = SegmentationParam()
    remove = True
    verbose = 1

    if wm_reg_param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        param_default = WmRegistrationParam()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Register the template on an image with a white/gray matter contrast according to the internal structure:\n'
                                     ' - register multimodal of the template to the target image\n'
                                     ' - automatic segmentation of the white and gray matter in the target image\n'
                                     ' - correction of the registration of the template to the target according to the WM/GM segmentation')  # TODO: change description
        parser.usage.addSection('General parameters')
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
        parser.add_option(name="-warp",
                          type_value="file",
                          description="Warping field of the template to the anatomical image",
                          mandatory=True,
                          example='../t2/warp_template2anat.nii.gz')
        parser.usage.addSection('Registration parameters')
        parser.add_option(name="-t",
                          type_value='multiple_choice',
                          description="type of transformation",
                          mandatory=False,
                          default_value='BSplineSyN',
                          example=['BSplineSyN', 'SyN', 'Affine'])
        parser.add_option(name="-m",
                          type_value='multiple_choice',
                          description="Metric used for the registration",
                          mandatory=False,
                          default_value='MeanSquares',
                          example=['CC', 'MeanSquares'])
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
        multimodal_reg_param.target = arguments["-i"]
        multimodal_reg_param.target_seg = arguments["-iseg"]
        multimodal_reg_param.anat = arguments["-anat"]
        multimodal_reg_param.anat_seg = arguments["-anat-seg"]
        multimodal_reg_param.warp_template2anat = arguments["-warp"]
        if "-t" in arguments:
            wm_reg_param.transformation = arguments["-t"]
        if "-m" in arguments:
            wm_reg_param.metric = arguments["-m"]
        if "-r" in arguments:
            remove = bool(int(arguments["-r"]))
        if "-v" in arguments:
            verbose = arguments["-v"]
        main()
