#!/usr/bin/env python
#########################################################################################
# Register a volume (e.g., EPI from fMRI or DTI scan) to an anatomical image.
#
# See Usage() below for more information.
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# none
#
# EXTERNAL SOFTWARE
# - itksnap/isct_c3d <http://www.itksnap.org/pmwiki/pmwiki.php?n=Main.HomePage>
# - ants <http://stnava.github.io/ANTs/>
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: testing script for all cases

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherwise inverse warping field is not output

# TODO: make three possibilities:
# - one-step registration, using only image registration (by sliceReg or antsRegistration)
# - two-step registration, using first segmentation-based registration (based on sliceReg or antsRegistration) and second the image registration (and allow the choice of algo, metric, etc.)
# - two-step registration, using only segmentation-based registration


import sys
import os
import commands
import time

import sct_utils as sct
from msct_parser import Parser


# DEFAULT PARAMETERS

class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.outSuffix  = "_reg"
        self.fname_mask = ''
        self.padding = 5
        self.outlier_factor = 2

# Parameters for registration
class Paramreg(object):
    def __init__(self, step='1', type='im', algo='syn', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5', poly='3', window_length = '31'):
        self.step = step
        self.type = type
        self.algo = algo
        self.metric = metric
        self.iter = iter
        self.shrink = shrink
        self.smooth = smooth
        self.gradStep = gradStep
        self.poly = poly  # slicereg only
        self.window_length = window_length

    # update constructor with user's parameters
    def update(self, paramreg_user):
        list_objects = paramreg_user.split(',')
        for object in list_objects:
            if len(object)<2:
                sct.printv('Please check parameter -p (usage changed)',1,type='error')

            obj = object.split('=')
            setattr(self, obj[0], obj[1])

class ParamregMultiStep:
    '''
    This class contains a dictionary with the params of multiple steps
    '''
    def __init__(self, listParam=[]):
        self.steps = dict()
        for stepParam in listParam:
            if isinstance(stepParam, Paramreg):
                self.steps[stepParam.step] = stepParam
            else:
                self.addStep(stepParam)

    def addStep(self, stepParam):
        # this function checks if the step is already present. If it is present, it must update it. If it is not, it must add it.
        param_reg = Paramreg()
        param_reg.update(stepParam)
        if param_reg.step != 0:
            if param_reg.step in self.steps:
                self.steps[param_reg.step].update(stepParam)
            else:
                self.steps[param_reg.step] = param_reg
        else:
            sct.printv("ERROR: parameters must contain 'step'", 1, 'error')


# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_output = ''
    fname_mask = param.fname_mask
    fname_src_seg = ''
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI'

    start_time = time.time()
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # get default registration parameters
    # step1 = Paramreg(step='1', type='im', algo='syn', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5')
    step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5')  # only used to put src into dest space
    step1 = Paramreg()
    paramreg = ParamregMultiStep([step0, step1])

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('This program co-registers two 3D volumes. The deformation is non-rigid and is '
                                 'constrained along Z direction (i.e., axial plane). Hence, this function assumes '
                                 'that orientation of the destination image is axial (RPI). If you need to register '
                                 'two volumes with large deformations and/or different contrasts, it is recommended to '
                                 'input spinal cord segmentations (binary mask) in order to achieve maximum robustness.'
                                 ' The program outputs a warping field that can be used to register other images to the'
                                 ' destination image. To apply the warping field to another image, use '
                                 'sct_apply_transfo')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image source.",
                      mandatory=True,
                      example="src.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="Image destination.",
                      mandatory=True,
                      example="dest.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation source.",
                      mandatory=False,
                      example="src_seg.nii.gz")
    parser.add_option(name="-dseg",
                      type_value="file",
                      description="Segmentation destination.",
                      mandatory=False,
                      example="dest_seg.nii.gz")
    parser.add_option(name="-m",
                      type_value="file",
                      description="Mask that can be created with sct_create_mask to improve accuracy over region of interest. "
                                  "This mask will be used on the destination image.",
                      mandatory=False,
                      example="mask.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of output file.",
                      mandatory=False,
                      example="src_reg.nii.gz")
    parser.add_option(name="-p",
                      type_value=[[':'],'str'],
                      description="""Parameters for registration. Separate arguments with ",". Separate steps with ":".\nstep: <int> Step number (starts at 1).\ntype: {im,seg} type of data used for registration.\nalgo: Default="""+paramreg.steps['1'].algo+"""\n  global registration: {rigid,  affine,  syn,  bsplinesyn}\n  Slice By Slice registration: {slicereg: regularized translations (see: goo.gl/Sj3ZeU),  slicereg2d_translation: regularized using moving average (Hanning window),  slicereg2d_rigid,  slicereg2d_affine,  slicereg2d_pointwise: registration based on the Center of Mass of each slice (use only with type:Seg. Designed for centerlines), slicereg2d_bsplinesyn, slicereg2d_syn}\nmetric: {CC,MI,MeanSquares}. Default="""+paramreg.steps['1'].metric+"""\niter: <int> Number of iterations. Default="""+paramreg.steps['1'].iter+"""\nshrink: <int> Shrink factor (only for SyN). Default="""+paramreg.steps['1'].shrink+"""\nsmooth: <int> Smooth factor (only for SyN). Default="""+paramreg.steps['1'].smooth+"""\ngradStep: <float> Gradient step. Default="""+paramreg.steps['1'].gradStep+"""\npoly: <int> Polynomial degree (only for slicereg). Default="""+paramreg.steps['1'].poly+"""\nwindow_length: <int> size of hanning window for smoothing along z for slicereg2d_pointwise, slicereg2d_translation, slicereg2d_rigid, slicereg2d_affine, slicereg2d_syn and slicereg2d_bsplinesyn.. Default="""+paramreg.steps['1'].window_length,
                      mandatory=False,
                      example="step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,metric=MI,iter=5,shrink=2")
    parser.add_option(name="-z",
                      type_value="int",
                      description="""size of z-padding to enable deformation at edges when using SyN.""",
                      mandatory=False,
                      default_value=param.padding)
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="""Final interpolation.""",
                      mandatory=False,
                      default_value='linear',
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])
    arguments = parser.parse(sys.argv[1:])

    # get arguments
    fname_src = arguments['-i']
    fname_dest = arguments['-d']
    if '-iseg' in arguments:
        fname_src_seg = arguments['-iseg']
    if '-dseg' in arguments:
        fname_dest_seg = arguments['-dseg']
    if '-o' in arguments:
        fname_output = arguments['-o']
    if "-m" in arguments:
        fname_mask = arguments['-m']
    padding = arguments['-z']
    if "-p" in arguments:
        paramreg_user = arguments['-p']
        # update registration parameters
        for paramStep in paramreg_user:
            paramreg.addStep(paramStep)

    interp = arguments['-x']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_dest = path_sct_data+'/mt/mt1.nii.gz'
        fname_src = path_sct_data+'/t2/t2.nii.gz'
        param_user = '10,syn,0.5,MI'
        remove_temp_files = '0'
        verbose = 1

    # print arguments
    print '\nInput parameters:'
    print '  Source .............. '+fname_src
    print '  Destination ......... '+fname_dest
    print '  Mask ................ '+fname_mask
    print '  Output name ......... '+fname_output
    # print '  Algorithm ........... '+paramreg.algo
    # print '  Number of iterations  '+paramreg.iter
    # print '  Metric .............. '+paramreg.metric
    print '  Remove temp files ... '+str(remove_temp_files)
    print '  Verbose ............. '+str(verbose)

    # update param
    param.verbose = verbose
    param.padding = padding
    param.fname_mask = fname_mask
    param.remove_temp_files = remove_temp_files

    # Get if input is 3D
    sct.printv('\nCheck if input data are 3D...', verbose)
    sct.check_if_3d(fname_src)
    sct.check_if_3d(fname_dest)

    # check if destination data is RPI
    sct.printv('\nCheck if destination data is RPI...', verbose)
    sct.check_if_rpi(fname_dest)

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)

    # define output folder and file name
    if fname_output == '':
        path_out = ''  # output in user's current directory
        file_out = file_src+"_reg"
        ext_out = ext_src
    else:
        path_out, file_out, ext_out = sct.extract_fname(fname_output)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    status, output = sct.run('mkdir '+path_tmp, verbose)

    # copy files to temporary folder
    sct.printv('\nCopy files...', verbose)
    sct.run('isct_c3d '+fname_src+' -o '+path_tmp+'/src.nii', verbose)
    sct.run('isct_c3d '+fname_dest+' -o '+path_tmp+'/dest.nii', verbose)
    if fname_src_seg:
        sct.run('isct_c3d '+fname_src_seg+' -o '+path_tmp+'/src_seg.nii', verbose)
        sct.run('isct_c3d '+fname_dest_seg+' -o '+path_tmp+'/dest_seg.nii', verbose)
    if not fname_mask == '':
        sct.run('isct_c3d '+fname_mask+' -o '+path_tmp+'/mask.nii.gz', verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # Put source into destination space using header (no estimation -- purely based on header)
    # TODO: use c3d?
    # TODO: Check if necessary to do that
    # TODO: use that as step=0
    # sct.printv('\nPut source into destination space using header...', verbose)
    # sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[dest_pad.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o [regAffine,src_regAffine.nii] -n BSpline[3]', verbose)
    # if segmentation, also do it for seg

    # loop across registration steps
    warp_forward = []
    warp_inverse = []
    for i_step in range(0, len(paramreg.steps)):
        sct.printv('\nEstimate transformation for step #'+str(i_step)+'...', param.verbose)
        # identify which is the src and dest
        if paramreg.steps[str(i_step)].type == 'im':
            src = 'src.nii'
            dest = 'dest.nii'
            interp_step = 'linear'
        elif paramreg.steps[str(i_step)].type == 'seg':
            src = 'src_seg.nii'
            dest = 'dest_seg.nii'
            interp_step = 'nn'
        else:
            sct.run('ERROR: Wrong image type.', 1, 'error')
        # if step>0, apply warp_forward_concat to the src image to be used
        if i_step > 0:
            sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+sct.add_suffix(src, '_reg')+' -x '+interp_step, verbose)
            src = sct.add_suffix(src, '_reg')
        # register src --> dest
        warp_forward_out, warp_inverse_out = register(src, dest, paramreg, param, str(i_step))
        warp_forward.append(warp_forward_out)
        warp_inverse.append(warp_inverse_out)

    # Put warp_forward_0 at the end of the list
    warp_forward_0 = warp_forward.pop(0)
    warp_forward.append(warp_forward_0)

    # Concatenate transformations
    sct.printv('\nConcatenate transformations...', verbose)
    sct.run('sct_concat_transfo -w '+','.join(warp_forward)+' -d dest.nii -o warp_src2dest.nii.gz', verbose)
    warp_inverse.reverse()
    sct.run('sct_concat_transfo -w '+','.join(warp_inverse)+' -d dest.nii -o warp_dest2src.nii.gz', verbose)

    # Apply warping field to src data
    sct.printv('\nApply transfo source --> dest...', verbose)
    sct.run('sct_apply_transfo -i src.nii -o src_reg.nii -d dest.nii -w warp_src2dest.nii.gz -x '+interp, verbose)
    sct.printv('\nApply transfo dest --> source...', verbose)
    sct.run('sct_apply_transfo -i dest.nii -o dest_reg.nii -d src.nii -w warp_dest2src.nii.gz -x '+interp, verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    fname_src2dest = sct.generate_output_file(path_tmp+'/src_reg.nii', path_out+file_out+ext_out, verbose)
    sct.generate_output_file(path_tmp+'/warp_src2dest.nii.gz', path_out+'warp_'+file_src+'2'+file_dest+'.nii.gz', verbose)
    fname_dest2src = sct.generate_output_file(path_tmp+'/dest_reg.nii', path_out+file_dest+'_reg'+ext_dest, verbose)
    sct.generate_output_file(path_tmp+'/warp_dest2src.nii.gz', path_out+'warp_'+file_dest+'2'+file_src+'.nii.gz', verbose)
    # sct.generate_output_file(path_tmp+'/warp_dest2src.nii.gz', path_out+'warp_dest2src.nii.gz')

    # Delete temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf '+path_tmp, verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)
    sct.printv('\nTo view results, type:', verbose)
    sct.printv('fslview '+fname_dest+' '+fname_src2dest+' &', verbose, 'info')
    sct.printv('fslview '+fname_src+' '+fname_dest2src+' &\n', verbose, 'info')



# register images
# ==========================================================================================
def register(src, dest, paramreg, param, i_step_str):

    # initiate default parameters of antsRegistration transformation
    ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
                                'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}

    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI'

    # set metricSize
    if paramreg.steps[i_step_str].metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # set masking
    if param.fname_mask:
        fname_mask = 'mask.nii.gz'
        masking = '-x mask.nii.gz'
    else:
        fname_mask = ''
        masking = ''

    if paramreg.steps[i_step_str].algo == 'slicereg':
        from msct_image import find_zmin_zmax
        # threshold images (otherwise, automatic crop does not work -- see issue #293)
        src_th = sct.add_suffix(src, '_th')
        sct.run(fsloutput+'fslmaths '+src+' -thr 0.1 '+src_th, param.verbose)
        dest_th = sct.add_suffix(dest, '_th')
        sct.run(fsloutput+'fslmaths '+dest+' -thr 0.1 '+dest_th, param.verbose)
        # find zmin and zmax
        zmin_src, zmax_src = find_zmin_zmax(src_th)
        zmin_dest, zmax_dest = find_zmin_zmax(dest_th)
        zmin_total = max([zmin_src, zmin_dest])
        zmax_total = min([zmax_src, zmax_dest])
        # crop data
        src_crop = sct.add_suffix(src, '_crop')
        sct.run('sct_crop_image -i '+src+' -o '+src_crop+' -dim 2 -start '+str(zmin_total)+' -end '+str(zmax_total), param.verbose)
        dest_crop = sct.add_suffix(dest, '_crop')
        sct.run('sct_crop_image -i '+dest+' -o '+dest_crop+' -dim 2 -start '+str(zmin_total)+' -end '+str(zmax_total), param.verbose)
        # update variables
        src = src_crop
        dest = dest_crop
        # estimate transfo
        cmd = ('isct_antsSliceRegularizedRegistration '
               '-t Translation[0.5] '
               '-m '+paramreg.steps[i_step_str].metric+'['+dest+','+src+',1,'+metricSize+',Regular,0.2] '
               '-p '+paramreg.steps[i_step_str].poly+' '
               '-i '+paramreg.steps[i_step_str].iter+' '
               '-f 1 '
               '-s '+paramreg.steps[i_step_str].smooth+' '
               '-v 1 '  # verbose (verbose=2 does not exist, so we force it to 1)
               '-o [step'+i_step_str+','+src+'_regStep'+i_step_str+'.nii] '  # here the warp name is stage10 because antsSliceReg add "Warp"
               +masking)
        warp_forward_out = 'step'+i_step_str+'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str+'InverseWarp.nii.gz'

    elif paramreg.steps[i_step_str].algo == 'slicereg2d_pointwise':
        from msct_register import register_slicereg2d_pointwise
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicereg2d_pointwise(src, dest, window_length=paramreg.steps[i_step_str].window_length, paramreg=Paramreg(step=paramreg.steps[i_step_str].step, type=paramreg.steps[i_step_str].type, algo='Translation', metric=paramreg.steps[i_step_str].metric, iter= paramreg.steps[i_step_str].iter, shrink=paramreg.steps[i_step_str].shrink, smooth=paramreg.steps[i_step_str].smooth, gradStep=paramreg.steps[i_step_str].gradStep),
                                      warp_forward_out=warp_forward_out, warp_inverse_out=warp_inverse_out, factor=param.outlier_factor, verbose=param.verbose)
        cmd = ('')

    elif paramreg.steps[i_step_str].algo == 'slicereg2d_translation':
        from msct_register import register_slicereg2d_translation
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicereg2d_translation(src, dest, window_length=paramreg.steps[i_step_str].window_length, paramreg=Paramreg(step=paramreg.steps[i_step_str].step, type=paramreg.steps[i_step_str].type, algo='Translation', metric=paramreg.steps[i_step_str].metric, iter= paramreg.steps[i_step_str].iter, shrink=paramreg.steps[i_step_str].shrink, smooth=paramreg.steps[i_step_str].smooth, gradStep=paramreg.steps[i_step_str].gradStep),
                                        fname_mask=fname_mask, warp_forward_out=warp_forward_out, warp_inverse_out=warp_inverse_out, factor=param.outlier_factor, remove_temp_files=param.remove_temp_files,
                                        verbose=param.verbose, ants_registration_params=ants_registration_params)
        cmd = ('')

    elif paramreg.steps[i_step_str].algo == 'slicereg2d_rigid':
        from msct_register import register_slicereg2d_rigid
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicereg2d_rigid(src, dest, window_length=paramreg.steps[i_step_str].window_length, paramreg=Paramreg(step=paramreg.steps[i_step_str].step, type=paramreg.steps[i_step_str].type, algo='Rigid', metric=paramreg.steps[i_step_str].metric, iter= paramreg.steps[i_step_str].iter, shrink=paramreg.steps[i_step_str].shrink, smooth=paramreg.steps[i_step_str].smooth, gradStep=paramreg.steps[i_step_str].gradStep),
                                  fname_mask=fname_mask, warp_forward_out=warp_forward_out, warp_inverse_out=warp_inverse_out, factor=param.outlier_factor, remove_temp_files=param.remove_temp_files,
                                  verbose=param.verbose, ants_registration_params=ants_registration_params)
        cmd = ('')

    elif paramreg.steps[i_step_str].algo == 'slicereg2d_affine':
        from msct_register import register_slicereg2d_affine
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicereg2d_affine(src, dest, window_length=paramreg.steps[i_step_str].window_length, paramreg=Paramreg(step=paramreg.steps[i_step_str].step, type=paramreg.steps[i_step_str].type, algo='Affine', metric=paramreg.steps[i_step_str].metric, iter= paramreg.steps[i_step_str].iter, shrink=paramreg.steps[i_step_str].shrink, smooth=paramreg.steps[i_step_str].smooth, gradStep=paramreg.steps[i_step_str].gradStep),
                                   fname_mask=fname_mask, warp_forward_out=warp_forward_out, warp_inverse_out=warp_inverse_out, factor=param.outlier_factor, remove_temp_files=param.remove_temp_files,
                                   verbose=param.verbose, ants_registration_params=ants_registration_params)
        cmd = ('')

    elif paramreg.steps[i_step_str].algo == 'slicereg2d_syn':
        from msct_register import register_slicereg2d_syn
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicereg2d_syn(src, dest, window_length=paramreg.steps[i_step_str].window_length, paramreg=Paramreg(step=paramreg.steps[i_step_str].step, type=paramreg.steps[i_step_str].type, algo='SyN', metric=paramreg.steps[i_step_str].metric, iter= paramreg.steps[i_step_str].iter, shrink=paramreg.steps[i_step_str].shrink, smooth=paramreg.steps[i_step_str].smooth, gradStep=paramreg.steps[i_step_str].gradStep),
                                fname_mask=fname_mask, warp_forward_out=warp_forward_out, warp_inverse_out=warp_inverse_out, factor=param.outlier_factor, remove_temp_files=param.remove_temp_files,
                                verbose=param.verbose, ants_registration_params=ants_registration_params)
        cmd = ('')

    elif paramreg.steps[i_step_str].algo == 'slicereg2d_bsplinesyn':
        from msct_register import register_slicereg2d_bsplinesyn
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicereg2d_bsplinesyn(src, dest, window_length=paramreg.steps[i_step_str].window_length, paramreg=Paramreg(step=paramreg.steps[i_step_str].step, type=paramreg.steps[i_step_str].type, algo='BSplineSyN', metric=paramreg.steps[i_step_str].metric, iter= paramreg.steps[i_step_str].iter, shrink=paramreg.steps[i_step_str].shrink, smooth=paramreg.steps[i_step_str].smooth, gradStep=paramreg.steps[i_step_str].gradStep),
                                       fname_mask=fname_mask, warp_forward_out=warp_forward_out, warp_inverse_out=warp_inverse_out, factor=param.outlier_factor, remove_temp_files=param.remove_temp_files,
                                       verbose=param.verbose, ants_registration_params=ants_registration_params)
        cmd = ('')

    elif paramreg.steps[i_step_str].algo.lower() in ants_registration_params:
        from msct_image import pad_image
        # Pad the destination image (because ants doesn't deform the extremities)
        # N.B. no need to pad if iter = 0
        if not paramreg.steps[i_step_str].iter == '0':
            dest_pad = sct.add_suffix(dest, '_pad')
            pad_image(dest, dest_pad, param.padding)
            dest = dest_pad

        cmd = ('isct_antsRegistration '
               '--dimensionality 3 '
               '--transform '+paramreg.steps[i_step_str].algo+'['+paramreg.steps[i_step_str].gradStep +
               ants_registration_params[paramreg.steps[i_step_str].algo.lower()]+'] '
               '--metric '+paramreg.steps[i_step_str].metric+'['+dest+','+src+',1,'+metricSize+'] '
               '--convergence '+paramreg.steps[i_step_str].iter+' '
               '--shrink-factors '+paramreg.steps[i_step_str].shrink+' '
               '--smoothing-sigmas '+paramreg.steps[i_step_str].smooth+'mm '
               '--restrict-deformation 1x1x0 '
               '--output [step'+i_step_str+','+src+'_regStep'+i_step_str+'.nii] '
               '--interpolation BSpline[3] '
               +masking)
        if param.verbose >= 1:
            cmd += ' --verbose 1'
        if paramreg.steps[i_step_str].algo in ['rigid', 'affine']:
            warp_forward_out = 'step'+i_step_str+'0GenericAffine.mat'
            warp_inverse_out = '-step'+i_step_str+'0GenericAffine.mat'
        else:
            warp_forward_out = 'step'+i_step_str+'0Warp.nii.gz'
            warp_inverse_out = 'step'+i_step_str+'0InverseWarp.nii.gz'
    else:
        sct.printv('\nERROR: algo '+paramreg.steps[i_step_str].algo+' does not exist. Exit program\n', 1, 'error')

    # run registration
    status, output = sct.run(cmd, param.verbose)

    if os.path.isfile(warp_forward_out):
        # rename warping fields
        if paramreg.steps[i_step_str].algo in ['rigid', 'affine']:
            warp_forward = 'warp_forward_'+i_step_str+'.mat'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = '-warp_forward_'+i_step_str+'.mat'
        else:
            warp_forward = 'warp_forward_'+i_step_str+'.nii.gz'
            warp_inverse = 'warp_inverse_'+i_step_str+'.nii.gz'
            os.rename(warp_forward_out, warp_forward)
            os.rename(warp_inverse_out, warp_inverse)
    else:
        sct.printv(output, 1, 'error')
        sct.printv('\nERROR: ANTs failed. Exit program.\n', 1, 'error')

    return warp_forward, warp_inverse


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()


# Convert deformation field to 4D volume (readable by fslview)
# DONE: clean code below-- right now it does not work
#===========
#if convertDeformation:
#    print('\nConvert deformation field...')
#    cmd = 'isct_c3d -mcs tmp.regWarp.nii -oo tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
#    print(">> "+cmd)
#    os.system(cmd)
#    cmd = 'fslmerge -t '+path_out+'warp_comp.nii tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
#    print(">> "+cmd)
#    os.system(cmd)
#===========
