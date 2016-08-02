#!/usr/bin/env python
#########################################################################################
# Register a volume (e.g., EPI from fMRI or DTI scan) to an anatomical image.
#
# See Usage() below for more information.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: check syn with shrink=4
# TODO: output name file for warp using "src" and "dest" file name, i.e. warp_filesrc2filedest.nii.gz
# TODO: testing script for all cases
# TODO: add following feature:
#-r of isct_antsRegistration at the initial step (step 0).
#-r [' dest ',' src ',0] --> align the geometric center of the two images
#-r [' dest ',' src ',1] --> align the maximum intensities of the two images I use that quite often...
# TODO: output reg for ants2d and centermass (2016-02-25)

# Note for the developer: DO NOT use --collapse-output-transforms 1, otherwise inverse warping field is not output

# TODO: make three possibilities:
# - one-step registration, using only image registration (by sliceReg or antsRegistration)
# - two-step registration, using first segmentation-based registration (based on sliceReg or antsRegistration) and second the image registration (and allow the choice of algo, metric, etc.)
# - two-step registration, using only segmentation-based registration


import sys
import time

import os
import commands
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

# Parameters for registration
class Paramreg(object):
    def __init__(self, step='1', type='im', algo='syn', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5', init='', poly='3', slicewise='0', laplacian='0'):
        self.step = step
        self.type = type
        self.algo = algo
        self.metric = metric
        self.iter = iter
        self.shrink = shrink
        self.smooth = smooth
        self.laplacian = laplacian
        self.gradStep = gradStep
        self.slicewise = slicewise
        self.init = init
        self.poly = poly  # slicereg only
        # self.window_length = window_length  # str
        # self.detect_outlier = detect_outlier  # str. detect outliers, for methods slicereg2d_xx

    # update constructor with user's parameters
    def update(self, paramreg_user):
        list_objects = paramreg_user.split(',')
        for object in list_objects:
            if len(object) < 2:
                sct.printv('Please check parameter -param (usage changed from previous version)', 1, type='error')
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
    fname_dest_seg = ''

    start_time = time.time()
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # get default registration parameters
    # step1 = Paramreg(step='1', type='im', algo='syn', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5')
    step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5', slicewise='0')  # only used to put src into dest space
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
    parser.add_option(name="-param",
                      type_value=[[':'],'str'],
                      description="Parameters for registration. Separate arguments with \",\". Separate steps with \":\".\n"
                                  "step: <int> Step number (starts at 1).\ntype: {im,seg} type of data used for registration.\n"
                                  "algo: Default="+paramreg.steps['1'].algo+"\n"
                                  "  translation: translation in X-Y plane (2dof)\n"
                                  "  rigid: translation + rotation in X-Y plane (4dof)\n"
                                  "  affine: translation + rotation + scaling in X-Y plane (6dof)\n"
                                  "  syn: non-linear symmetric normalization\n"
                                  "  bsplinesyn: syn regularized with b-splines\n"
                                  "  slicereg: regularized translations (see: goo.gl/Sj3ZeU)\n"
                                  "  centermass: slicewise center of mass alignment (seg only).\n"
                                  "  centermassrot: slicewise center of mass and PCA-based rotation alignment (seg only)\n"
                                  "  columnwise: R-L scaling followed by A-P columnwise alignment (seg only).\n"
                                  "slicewise: <int> Slice-by-slice 2d transformation. Default="+paramreg.steps['1'].slicewise+"\n"
                                  "metric: {CC,MI,MeanSquares}. Default="+paramreg.steps['1'].metric+"\n"
                                  "iter: <int> Number of iterations. Default="+paramreg.steps['1'].iter+"\n"
                                  "shrink: <int> Shrink factor (only for syn/bsplinesyn). Default="+paramreg.steps['1'].shrink+"\n"
                                  "smooth: <int> Smooth factor. Default="+paramreg.steps['1'].smooth+"\n"
                                  "laplacian: <int> Laplacian filter. Default="+paramreg.steps['1'].laplacian+"\n"
                                  "gradStep: <float> Gradient step. Default="+paramreg.steps['1'].gradStep+"\n"
                                  "init: <int> Initial translation alignment based on:\n"
                                  "  geometric: Geometric center of images\n"
                                  "  centermass: Center of mass of images\n"
                                  "  origin: Physical origin of images\n"
                                  "poly: <int> Polynomial degree (only for slicereg). Default="+paramreg.steps['1'].poly+"\n",
                      mandatory=False,
                      example="step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,metric=MI,iter=5,shrink=2")
    parser.add_option(name="-identity",
                      type_value="multiple_choice",
                      description="just put source into destination (no optimization).",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
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
    if "-param" in arguments:
        paramreg_user = arguments['-param']
        # update registration parameters
        for paramStep in paramreg_user:
            paramreg.addStep(paramStep)
    identity = int(arguments['-identity'])
    interp = arguments['-x']
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])

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

    # Check if user selected type=seg, but did not input segmentation data
    if 'paramreg_user' in locals():
        if True in ['type=seg' in paramreg_user[i] for i in range(len(paramreg_user))]:
            if fname_src_seg == '' or fname_dest_seg == '':
                sct.printv('\nERROR: if you select type=seg you must specify -iseg and -dseg flags.\n', 1, 'error')

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
    path_tmp = sct.tmp_create()

    # copy files to temporary folder
    from sct_convert import convert
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    convert(fname_src, path_tmp+'src.nii')
    convert(fname_dest, path_tmp+'dest.nii')

    if fname_src_seg:
        convert(fname_src_seg, path_tmp+'src_seg.nii')
        convert(fname_dest_seg, path_tmp+'dest_seg.nii')

    if fname_mask != '':
        convert(fname_mask, path_tmp+'mask.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # reorient destination to RPI
    sct.run('sct_image -i dest.nii -setorient RPI -o dest_RPI.nii')
    if fname_dest_seg:
        sct.run('sct_image -i dest_seg.nii -setorient RPI -o dest_seg_RPI.nii')

    if identity:
        # overwrite paramreg and only do one identity transformation
        step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5')
        paramreg = ParamregMultiStep([step0])

    # Put source into destination space using header (no estimation -- purely based on header)
    # TODO: Check if necessary to do that
    # TODO: use that as step=0
    # sct.printv('\nPut source into destination space using header...', verbose)
    # sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[dest_pad.nii,src.nii,1,16] -c 0 -f 1 -s 0 -o [regAffine,src_regAffine.nii] -n BSpline[3]', verbose)
    # if segmentation, also do it for seg

    # loop across registration steps
    warp_forward = []
    warp_inverse = []
    for i_step in range(0, len(paramreg.steps)):
        sct.printv('\n--\nESTIMATE TRANSFORMATION FOR STEP #'+str(i_step), param.verbose)
        # identify which is the src and dest
        if paramreg.steps[str(i_step)].type == 'im':
            src = 'src.nii'
            dest = 'dest_RPI.nii'
            interp_step = 'spline'
        elif paramreg.steps[str(i_step)].type == 'seg':
            src = 'src_seg.nii'
            dest = 'dest_seg_RPI.nii'
            interp_step = 'nn'
        else:
            src = dest = interp_step = None
            sct.run('ERROR: Wrong image type.', 1, 'error')
        # if step>0, apply warp_forward_concat to the src image to be used
        if i_step > 0:
            sct.printv('\nApply transformation from previous step', param.verbose)
            sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+sct.add_suffix(src, '_reg')+' -x '+interp_step, verbose)
            src = sct.add_suffix(src, '_reg')
        # register src --> dest
        warp_forward_out, warp_inverse_out = register(src, dest, paramreg, param, str(i_step))
        warp_forward.append(warp_forward_out)
        warp_inverse.insert(0, warp_inverse_out)

    # Concatenate transformations
    sct.printv('\nConcatenate transformations...', verbose)
    sct.run('sct_concat_transfo -w '+','.join(warp_forward)+' -d dest.nii -o warp_src2dest.nii.gz', verbose)
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
    fname_src2dest = sct.generate_output_file(path_tmp+'src_reg.nii', path_out+file_out+ext_out, verbose)
    sct.generate_output_file(path_tmp+'warp_src2dest.nii.gz', path_out+'warp_'+file_src+'2'+file_dest+'.nii.gz', verbose)
    fname_dest2src = sct.generate_output_file(path_tmp+'dest_reg.nii', path_out+file_dest+'_reg'+ext_dest, verbose)
    sct.generate_output_file(path_tmp+'warp_dest2src.nii.gz', path_out+'warp_'+file_dest+'2'+file_src+'.nii.gz', verbose)
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
    output = ''  # default output if problem

    # display arguments
    sct.printv('Registration parameters:', param.verbose)
    sct.printv('  type ........... '+paramreg.steps[i_step_str].type, param.verbose)
    sct.printv('  algo ........... '+paramreg.steps[i_step_str].algo, param.verbose)
    sct.printv('  slicewise ...... '+paramreg.steps[i_step_str].slicewise, param.verbose)
    sct.printv('  metric ......... '+paramreg.steps[i_step_str].metric, param.verbose)
    sct.printv('  iter ........... '+paramreg.steps[i_step_str].iter, param.verbose)
    sct.printv('  smooth ......... '+paramreg.steps[i_step_str].smooth, param.verbose)
    sct.printv('  laplacian ...... '+paramreg.steps[i_step_str].laplacian, param.verbose)
    sct.printv('  shrink ......... '+paramreg.steps[i_step_str].shrink, param.verbose)
    sct.printv('  gradStep ....... '+paramreg.steps[i_step_str].gradStep, param.verbose)
    sct.printv('  init ........... '+paramreg.steps[i_step_str].init, param.verbose)
    sct.printv('  poly ........... '+paramreg.steps[i_step_str].poly, param.verbose)

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
        from msct_image import Image
        nii = Image(src)
        data = nii.data
        data[data < 0.1] = 0
        nii.data = data
        nii.setFileName(src_th)
        nii.save()
        # sct.run(fsloutput+'fslmaths '+src+' -thr 0.1 '+src_th, param.verbose)
        dest_th = sct.add_suffix(dest, '_th')
        nii = Image(dest)
        data = nii.data
        data[data < 0.1] = 0
        nii.data = data
        nii.setFileName(dest_th)
        nii.save()
        # sct.run(fsloutput+'fslmaths '+dest+' -thr 0.1 '+dest_th, param.verbose)
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
        scr_regStep = sct.add_suffix(src, '_regStep'+i_step_str)
        # estimate transfo
        cmd = ('isct_antsSliceRegularizedRegistration '
               '-t Translation[0.5] '
               '-m '+paramreg.steps[i_step_str].metric+'['+dest+','+src+',1,'+metricSize+',Regular,0.2] '
               '-p '+paramreg.steps[i_step_str].poly+' '
               '-i '+paramreg.steps[i_step_str].iter+' '
               '-f 1 '
               '-s '+paramreg.steps[i_step_str].smooth+' '
               '-v 1 '  # verbose (verbose=2 does not exist, so we force it to 1)
               '-o [step'+i_step_str+','+scr_regStep+'] '  # here the warp name is stage10 because antsSliceReg add "Warp"
               +masking)
        warp_forward_out = 'step'+i_step_str+'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str+'InverseWarp.nii.gz'
        # run command
        status, output = sct.run(cmd, param.verbose)

    # ANTS 3d
    elif paramreg.steps[i_step_str].algo.lower() in ants_registration_params and paramreg.steps[i_step_str].slicewise == '0':
        # Pad the destination image (because ants doesn't deform the extremities)
        # N.B. no need to pad if iter = 0
        if not paramreg.steps[i_step_str].iter == '0':
            dest_pad = sct.add_suffix(dest, '_pad')
            sct.run('sct_image -i '+dest+' -o '+dest_pad+' -pad 0,0,'+str(param.padding))
            dest = dest_pad
        # apply Laplacian filter
        if not paramreg.steps[i_step_str].laplacian == '0':
            sct.printv('\nApply Laplacian filter', param.verbose)
            sct.run('sct_maths -i '+src+' -laplacian '+paramreg.steps[i_step_str].laplacian+','+paramreg.steps[i_step_str].laplacian+',0 -o '+sct.add_suffix(src, '_laplacian'))
            sct.run('sct_maths -i '+dest+' -laplacian '+paramreg.steps[i_step_str].laplacian+','+paramreg.steps[i_step_str].laplacian+',0 -o '+sct.add_suffix(dest, '_laplacian'))
            src = sct.add_suffix(src, '_laplacian')
            dest = sct.add_suffix(dest, '_laplacian')
        # Estimate transformation
        sct.printv('\nEstimate transformation', param.verbose)
        scr_regStep = sct.add_suffix(src, '_regStep' + i_step_str)
        cmd = ('isct_antsRegistration '
               '--dimensionality 3 '
               '--transform '+paramreg.steps[i_step_str].algo+'['+paramreg.steps[i_step_str].gradStep +
               ants_registration_params[paramreg.steps[i_step_str].algo.lower()]+'] '
               '--metric '+paramreg.steps[i_step_str].metric+'['+dest+','+src+',1,'+metricSize+'] '
               '--convergence '+paramreg.steps[i_step_str].iter+' '
               '--shrink-factors '+paramreg.steps[i_step_str].shrink+' '
               '--smoothing-sigmas '+paramreg.steps[i_step_str].smooth+'mm '
               '--restrict-deformation 1x1x0 '
               '--output [step'+i_step_str+','+scr_regStep+'] '
               '--interpolation BSpline[3] '
               +masking)
        # add verbose
        if param.verbose >= 1:
            cmd += ' --verbose 1'
        # add init translation
        if not paramreg.steps[i_step_str].init == '':
            init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
            cmd += ' -r ['+dest+','+src+','+init_dict[paramreg.steps[i_step_str].init]+']'
        # run command
        status, output = sct.run(cmd, param.verbose)
        # get appropriate file name for transformation
        if paramreg.steps[i_step_str].algo in ['rigid', 'affine', 'translation']:
            warp_forward_out = 'step'+i_step_str+'0GenericAffine.mat'
            warp_inverse_out = '-step'+i_step_str+'0GenericAffine.mat'
        else:
            warp_forward_out = 'step'+i_step_str+'0Warp.nii.gz'
            warp_inverse_out = 'step'+i_step_str+'0InverseWarp.nii.gz'

    # ANTS 2d
    elif paramreg.steps[i_step_str].algo.lower() in ants_registration_params and paramreg.steps[i_step_str].slicewise == '1':
        from msct_register import register_slicewise
        # if shrink!=1, force it to be 1 (otherwise, it generates a wrong 3d warping field). TODO: fix that!
        if not paramreg.steps[i_step_str].shrink == '1':
            sct.printv('\nWARNING: when using slicewise with SyN or BSplineSyN, shrink factor needs to be one. Forcing shrink=1.', 1, 'warning')
            paramreg.steps[i_step_str].shrink = '1'
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicewise(src,
                            dest,
                            paramreg=paramreg.steps[i_step_str],
                            fname_mask=fname_mask,
                            warp_forward_out=warp_forward_out,
                            warp_inverse_out=warp_inverse_out,
                            verbose=param.verbose,
                            ants_registration_params=ants_registration_params)

    # slice-wise transfo
    elif paramreg.steps[i_step_str].algo in ['centermass', 'centermassrot', 'columnwise']:
        # check if type=seg
        if not paramreg.steps[i_step_str].type == 'seg':
            sct.printv('\nWARNING: algo '+paramreg.steps[i_step_str].algo+' should be used with type=seg.\n', 1, 'warning')
        if not fname_mask == '':
            sct.printv('\nWARNING: algo '+paramreg.steps[i_step_str].algo+' will ignore the provided mask.\n', 1, 'warning')
        # smooth data
        if not paramreg.steps[i_step_str].smooth == '0':
            sct.printv('\nSmooth data', param.verbose)
            sct.run('sct_maths -i '+src+' -smooth '+paramreg.steps[i_step_str].smooth+','+paramreg.steps[i_step_str].smooth+',0 -o '+sct.add_suffix(src, '_smooth'))
            sct.run('sct_maths -i '+dest+' -smooth '+paramreg.steps[i_step_str].smooth+','+paramreg.steps[i_step_str].smooth+',0 -o '+sct.add_suffix(dest, '_smooth'))
            src = sct.add_suffix(src, '_smooth')
            dest = sct.add_suffix(dest, '_smooth')
        from msct_register import register_slicewise
        warp_forward_out = 'step'+i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step'+i_step_str + 'InverseWarp.nii.gz'
        register_slicewise(src,
                            dest,
                            paramreg=paramreg.steps[i_step_str],
                            fname_mask=fname_mask,
                            warp_forward_out=warp_forward_out,
                            warp_inverse_out=warp_inverse_out,
                            verbose=param.verbose,
                            ants_registration_params=ants_registration_params)

    else:
        sct.printv('\nERROR: algo '+paramreg.steps[i_step_str].algo+' does not exist. Exit program\n', 1, 'error')

    if not os.path.isfile(warp_forward_out):
        # no forward warping field for rigid and affine
        sct.printv('\nERROR: file '+warp_forward_out+' doesn\'t exist (or is not a file).\n' + output +
                   '\nERROR: ANTs failed. Exit program.\n', 1, 'error')
    elif not os.path.isfile(warp_inverse_out) and paramreg.steps[i_step_str].algo not in ['rigid', 'affine', 'translation']:
        # no inverse warping field for rigid and affine
        sct.printv('\nERROR: file '+warp_inverse_out+' doesn\'t exist (or is not a file).\n' + output +
                   '\nERROR: ANTs failed. Exit program.\n', 1, 'error')
    else:
        # rename warping fields
        if paramreg.steps[i_step_str].algo.lower() in ['rigid', 'affine', 'translation'] and paramreg.steps[i_step_str].slicewise == '0':
            warp_forward = 'warp_forward_'+i_step_str+'.mat'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = '-warp_forward_'+i_step_str+'.mat'
        else:
            warp_forward = 'warp_forward_'+i_step_str+'.nii.gz'
            warp_inverse = 'warp_inverse_'+i_step_str+'.nii.gz'
            os.rename(warp_forward_out, warp_forward)
            os.rename(warp_inverse_out, warp_inverse)

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
#    cmd = 'sct_image -i tmp.regWarp.nii -mcs  -o tmp.regWarp.nii'
#    print(">> "+cmd)
#    os.system(cmd)
#    cmd = 'fslmerge -t '+path_out+'warp_comp.nii tmp.regWarp_x.nii tmp.regWarp_y.nii tmp.regWarp_z.nii'
#    print(">> "+cmd)
#    os.system(cmd)
#===========
