#!/usr/bin/env python
#########################################################################################
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad, Augustin Roux
# Modified: 2015-03-31
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
import os
import commands
import time

import sct_utils as sct
from sct_orientation import set_orientation
from sct_register_multimodal import Paramreg
from msct_parser import Parser
from msct_image import Image





# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1  # remove temporary files
        self.output_type = 1
        # self.speed = 'fast'  # speed of registration. slow | normal | fast
        # self.nb_iterations = '5'
        # self.algo = 'SyN'
        # self.gradientStep = '0.5'
        # self.metric = 'MI'
        # self.verbose = 1  # verbose
        self.path_template = path_sct+'/data/template'
        self.file_template = 'MNI-Poly-AMU_T2.nii.gz'
        self.file_template_label = 'landmarks_center.nii.gz'
        self.file_template_seg = 'MNI-Poly-AMU_cord.nii.gz'
        # self.smoothing_sigma = 5  # Smoothing along centerline to improve accuracy and remove step effects

class Paramreg_step(Paramreg):
    def __init__(self, step='0', type='im', algo='syn', metric='MI', iter='10', shrink='2', smooth='0', poly='3', gradStep='0.5'):
        # additional parameters from class Paramreg
        # default step is zero to manage wrong input: if step=0, it is not a correct step
        self.step = step
        self.type = type
        # inheritate class Paramreg from sct_register_multimodal
        Paramreg.__init__(self, algo, metric, iter, shrink, smooth, poly, gradStep)

class ParamregMultiStep:
    '''
    This class contains a dictionary with the params of multiple steps
    '''
    def __init__(self, listParam=[]):
        self.steps = dict()
        for stepParam in listParam:
            if isinstance(stepParam, Paramreg_step):
                self.steps[stepParam.step] = stepParam
            else:
                self.addStep(stepParam)

    def addStep(self, stepParam):
        # this function must check if the step is already present or not. If it is present, it must update it. If it is not, it must add it.
        param_reg = Paramreg_step()
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

    # get default parameters
    step1 = Paramreg_step(step='1', type='seg', algo='bsplinesyn', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5')
    step2 = Paramreg_step(step='2', type='im', algo='syn', metric='MI', iter='10', shrink='1', smooth='0', gradStep='0.5')
    paramreg = ParamregMultiStep([step1, step2])

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Register anatomical image to the template.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Anatomical image.",
                      mandatory=True,
                      example="anat.nii.gz")
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord segmentation.",
                      mandatory=True,
                      example="anat_seg.nii.gz")
    parser.add_option(name="-l",
                      type_value="file",
                      description="Labels. See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/",
                      mandatory=True,
                      default_value='',
                      example="anat_labels.nii.gz")
    parser.add_option(name="-t",
                      type_value="folder",
                      description="Path to MNI-Poly-AMU template.",
                      mandatory=False,
                      default_value=param.path_template)
    parser.add_option(name="-p",
                      type_value=[[':'],'str'],
                      description="""Parameters for registration. Separate arguments with ",". Separate steps with ":".\nstep: <int> Step number (starts at 1).\ntype: {im,seg} type of data used for registration.\nalgo: {syn,bsplinesyn,slicereg}. Default="""+paramreg.steps['1'].algo+"""\nmetric: {CC,MI,MeanSquares}. Default="""+paramreg.steps['1'].metric+"""\niter: <int> Number of iterations. Default="""+paramreg.steps['1'].iter+"""\nshrink: <int> Shrink factor (only for SyN). Default="""+paramreg.steps['1'].shrink+"""\nsmooth: <int> Smooth factor (only for SyN). Default="""+paramreg.steps['1'].smooth+"""\ngradStep: <float> Gradient step (only for SyN). Default="""+paramreg.steps['1'].gradStep+"""\npoly: <int> Polynomial degree (only for slicereg). Default="""+paramreg.steps['1'].poly,
                      mandatory=False,
                      example="algo=slicereg,metric=MeanSquares,iter=20")
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
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = '/Users/julien/data/temp/sct_example_data/t2/t2.nii.gz'
        fname_landmarks = '/Users/julien/data/temp/sct_example_data/t2/labels.nii.gz'
        fname_seg = '/Users/julien/data/temp/sct_example_data/t2/t2_seg.nii.gz'
        path_template = param.path_template
        remove_temp_files = 0
        verbose = 2
        # speed = 'superfast'
        #param_reg = '2,BSplineSyN,0.6,MeanSquares'
    else:
        arguments = parser.parse(sys.argv[1:])

        # get arguments
        fname_data = arguments['-i']
        fname_seg = arguments['-s']
        fname_landmarks = arguments['-l']
        path_template = arguments['-t']
        remove_temp_files = int(arguments['-r'])
        verbose = int(arguments['-v'])
        paramreg_user = arguments['-p']
        # update parameters for registration
        for paramStep in paramreg_user:
            paramreg.addStep(paramStep)

    # initialize other parameters
    file_template = param.file_template
    file_template_label = param.file_template_label
    file_template_seg = param.file_template_seg
    output_type = param.output_type
    # smoothing_sigma = param.smoothing_sigma

    # start timer
    start_time = time.time()


    # get absolute path - TO DO: remove! NEVER USE ABSOLUTE PATH...
    path_template = os.path.abspath(path_template)

    # get fname of the template + template objects
    fname_template = sct.slash_at_the_end(path_template, 1)+file_template
    fname_template_label = sct.slash_at_the_end(path_template, 1)+file_template_label
    fname_template_seg = sct.slash_at_the_end(path_template, 1)+file_template_seg

    # check file existence
    # sct.printv('\nCheck file existence...', verbose)
    # sct.check_file_exist(fname_data, verbose)
    # sct.check_file_exist(fname_landmarks, verbose)
    # sct.check_file_exist(fname_seg, verbose)
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
    print '.. Remove temp files:    '+str(remove_temp_files)

    # # Check speed parameter and create registration mode: slow 50x30, normal 50x15, fast 10x3 (default)
    # if speed == "slow":
    #     nb_iterations = "50"
    # elif speed == "normal":
    #     nb_iterations = "15"
    # elif speed == "fast":
    #     nb_iterations = "5"
    # elif speed == "superfast":
    #     nb_iterations = "1"  # only for debugging purpose-- do not inform the user about this option
    # else:
    #     print 'ERROR: Wrong input registration speed {slow, normal, fast}.'
    #     sys.exit(2)

    sct.printv('\nParameters for registration:')
    for pStep in paramreg.steps:
        sct.printv('.. Step #'+paramreg.steps[pStep].step)
        sct.printv('.. Number of iterations..... '+paramreg.steps[pStep].iter)
        sct.printv('.. Algorithm................ '+paramreg.steps[pStep].algo)
        sct.printv('.. Gradient step............ '+paramreg.steps[pStep].gradStep)
        sct.printv('.. Metric................... '+paramreg.steps[pStep].metric)

    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    # check if label image contains coherent labels
    image_label = Image(fname_landmarks)
    # -> all labels must be different
    labels = image_label.getNonZeroCoordinates()
    hasDifferentLabels = True
    for lab in labels:
        for otherlabel in labels:
            if lab != otherlabel and lab.hasEqualValue(otherlabel):
                hasDifferentLabels = False
                break
    if not hasDifferentLabels:
        print 'ERROR: Wrong landmarks input. All labels must be different.'
        sys.exit(2)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    status, output = sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    print('\nCopy files...')
    sct.run('sct_c3d '+fname_data+' -o '+path_tmp+'/data.nii')
    sct.run('sct_c3d '+fname_landmarks+' -o '+path_tmp+'/landmarks.nii.gz')
    sct.run('sct_c3d '+fname_seg+' -o '+path_tmp+'/segmentation.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of input images to RPI
    print('\nChange orientation of input images to RPI...')
    set_orientation('data.nii', 'RPI', 'data_rpi.nii')
    set_orientation('landmarks.nii.gz', 'RPI', 'landmarks_rpi.nii.gz')
    set_orientation('segmentation.nii.gz', 'RPI', 'segmentation_rpi.nii.gz')

    # crop segmentation
    # output: segmentation_rpi_crop.nii.gz
    sct.run('sct_crop_image -i segmentation_rpi.nii.gz -o segmentation_rpi_crop.nii.gz -dim 2 -bzmax')

    # straighten segmentation
    print('\nStraighten the spinal cord using centerline/segmentation...')
    sct.run('sct_straighten_spinalcord -i segmentation_rpi_crop.nii.gz -c segmentation_rpi_crop.nii.gz -r 0')

    # Label preparation:
    # --------------------------------------------------------------------------------
    # Remove unused label on template. Keep only label present in the input label image
    print('\nRemove unused label on template. Keep only label present in the input label image...')
    sct.run('sct_label_utils -t remove -i '+fname_template_label+' -o template_label.nii.gz -r landmarks_rpi.nii.gz')

    # Make sure landmarks are INT
    print '\nConvert landmarks to INT...'
    sct.run('sct_c3d template_label.nii.gz -type int -o template_label.nii.gz')

    # Create a cross for the template labels - 5 mm
    print('\nCreate a 5 mm cross for the template labels...')
    sct.run('sct_label_utils -t cross -i template_label.nii.gz -o template_label_cross.nii.gz -c 5')

    # Create a cross for the input labels and dilate for straightening preparation - 5 mm
    print('\nCreate a 5mm cross for the input labels and dilate for straightening preparation...')
    sct.run('sct_label_utils -t cross -i landmarks_rpi.nii.gz -o landmarks_rpi_cross3x3.nii.gz -c 5 -d')

    # Apply straightening to labels
    print('\nApply straightening to labels...')
    sct.run('sct_apply_transfo -i landmarks_rpi_cross3x3.nii.gz -o landmarks_rpi_cross3x3_straight.nii.gz -d segmentation_rpi_crop_straight.nii.gz -w warp_curve2straight.nii.gz -x nn')

    # Convert landmarks from FLOAT32 to INT
    print '\nConvert landmarks from FLOAT32 to INT...'
    sct.run('sct_c3d landmarks_rpi_cross3x3_straight.nii.gz -type int -o landmarks_rpi_cross3x3_straight.nii.gz')

    # Estimate affine transfo: straight --> template (landmark-based)'
    print '\nEstimate affine transfo: straight anat --> template (landmark-based)...'
    sct.run('sct_ANTSUseLandmarkImagesToGetAffineTransform template_label_cross.nii.gz landmarks_rpi_cross3x3_straight.nii.gz affine straight2templateAffine.txt')

    # Apply affine transformation: straight --> template
    print '\nApply affine transformation: straight --> template...'
    sct.run('sct_apply_transfo -i data_rpi.nii -o data_rpi_straight2templateAffine.nii.gz -d '+fname_template+' -w warp_curve2straight.nii.gz,straight2templateAffine.txt')
    sct.run('sct_apply_transfo -i segmentation_rpi.nii.gz -o segmentation_rpi_straight2templateAffine.nii.gz -d '+fname_template+' -w warp_curve2straight.nii.gz,straight2templateAffine.txt -x nn')

    # Registration straight spinal cord to template
    print('\nRegister straight spinal cord to template...')

    # multi-step registration
    # here we only consider two modes: (im) -> registration on template anatomical image and (seg) -> registration on template segmentation
    file_multistepreg, interpolation, destination = dict(), dict(), dict()
    file_multistepreg['seg'], interpolation['seg'], destination['seg'] = 'segmentation_rpi_straight2templateAffine', 'nn', fname_template_seg
    file_multistepreg['im'], interpolation['im'], destination['im'] = 'data_rpi_straight2templateAffine', 'spline', fname_template

    path_template, f_template, ext_template = sct.extract_fname(fname_template)
    path_template_seg, f_template_seg, ext_template_seg = sct.extract_fname(fname_template_seg)
    list_warping_fields, list_inverse_warping_fields = [], []

    # at least one step is mandatory
    pStep = paramreg.steps['1']
    sct.run('sct_register_multimodal -i '+file_multistepreg[pStep.type]+'.nii.gz -o '+file_multistepreg[pStep.type]+'_step1.nii.gz -d '+destination[pStep.type]+' -p algo='+pStep.algo+',metric='+pStep.metric+',iter='+pStep.iter+',shrink='+pStep.shrink+',smooth='+pStep.smooth+',poly='+pStep.poly+',gradStep='+pStep.gradStep+' -r 0 -v '+str(verbose)+' -x '+interpolation[pStep.type]+' -z 10', verbose)
    # apply warping field on the other image
    if pStep.type == 'im':
        list_warping_fields.append('warp_'+file_multistepreg['im']+'2'+f_template+'.nii.gz')
        list_inverse_warping_fields.append('warp_'+f_template+'2'+file_multistepreg['im']+'.nii.gz')
        sct.run('sct_apply_transfo -i '+file_multistepreg['seg']+'.nii.gz -w warp_'+file_multistepreg['im']+'2'+f_template+'.nii.gz -d '+fname_template+' -o '+file_multistepreg['seg']+'_step'+pStep.step+'.nii.gz')
    else:
        list_warping_fields.append('warp_'+file_multistepreg['seg']+'2'+f_template_seg+'.nii.gz')
        list_inverse_warping_fields.append('warp_'+f_template_seg+'2'+file_multistepreg['seg']+'.nii.gz')
        sct.run('sct_apply_transfo -i '+file_multistepreg['im']+'.nii.gz -w warp_'+file_multistepreg['seg']+'2'+f_template_seg+'.nii.gz -d '+fname_template+' -o '+file_multistepreg['im']+'_step'+pStep.step+'.nii.gz')

    for i in range(2, len(paramreg.steps)+1):
        pStep = paramreg.steps[str(i)]
        if pStep is not '1': # first step is already done
            # compute warping field
            sct.run('sct_register_multimodal -i '+file_multistepreg[pStep.type]+'_step'+str(i-1)+'.nii.gz -o '+file_multistepreg[pStep.type]+'_step'+pStep.step+'.nii.gz -d '+destination[pStep.type]+' -p algo='+pStep.algo+',metric='+pStep.metric+',iter='+pStep.iter+',shrink='+pStep.shrink+',smooth='+pStep.smooth+',poly='+pStep.poly+',gradStep='+pStep.gradStep+' -r 0 -v '+str(verbose)+' -x '+interpolation[pStep.type]+' -z 10', verbose)

            # apply warping field on the other image and add new warping field to list
            if pStep.type == 'im':
                list_warping_fields.append('warp_'+file_multistepreg['im']+'_step'+str(i-1)+'2'+f_template+'.nii.gz')
                list_inverse_warping_fields.append('warp_'+f_template+'2'+file_multistepreg['im']+'_step'+str(i-1)+'.nii.gz')
                sct.run('sct_apply_transfo -i '+file_multistepreg['seg']+'_step'+str(i-1)+'.nii.gz -w warp_'+file_multistepreg['im']+'_step'+str(i-1)+'2'+f_template+'.nii.gz -d '+fname_template+' -o '+file_multistepreg['seg']+'_step'+pStep.step+'.nii.gz')
            else:
                list_warping_fields.append('warp_'+file_multistepreg['seg']+'_step'+str(i-1)+'2'+f_template_seg+'.nii.gz')
                list_inverse_warping_fields.append('warp_'+f_template_seg+'2'+file_multistepreg['seg']+'_step'+str(i-1)+'.nii.gz')
                sct.run('sct_apply_transfo -i '+file_multistepreg['im']+'_step'+str(i-1)+'.nii.gz -w warp_'+file_multistepreg['seg']+'_step'+str(i-1)+'2'+f_template_seg+'.nii.gz -d '+fname_template+' -o '+file_multistepreg['im']+'_step'+pStep.step+'.nii.gz')

    list_inverse_warping_fields.reverse()

    # Concatenate transformations: template2anat & anat2template
    sct.printv('\nConcatenate transformations: template --> straight --> anat...', verbose)
    sct.run('sct_concat_transfo -w '+','.join(list_inverse_warping_fields)+',-straight2templateAffine.txt,warp_straight2curve.nii.gz -d data.nii -o warp_template2anat.nii.gz')
    sct.printv('\nConcatenate transformations: anat --> straight --> template...', verbose)
    sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt,'+','.join(list_warping_fields)+' -d '+fname_template+' -o warp_anat2template.nii.gz')

    ############ OLD FASHION ############
    # # register using segmentations
    # sct.run('sct_register_multimodal -i segmentation_rpi_straight2templateAffine.nii.gz -d '+fname_template_seg+' -a bsplinesyn -p 10,1,0,0.5,MeanSquares -r 0 -v '+str(verbose)+' -x nn -z 10', verbose)
    # # apply to image
    # sct.run('sct_apply_transfo -i data_rpi_straight2templateAffine.nii -w warp_segmentation_rpi_straight2templateAffine2MNI-Poly-AMU_cord.nii.gz -d '+fname_template+' -o data_rpi_straight2templateAffine_step0.nii')
    # # register using images
    # sct.run('sct_register_multimodal -i data_rpi_straight2templateAffine_step0.nii -d '+fname_template+' -a syn -p 10,1,0,0.5,MI -r 0 -v '+str(verbose)+' -x spline -z 10', verbose)

    # # Concatenate transformations: template2anat & anat2template
    # sct.printv('\nConcatenate transformations: template --> straight --> anat...', verbose)
    # sct.run('sct_concat_transfo -w warp_MNI-Poly-AMU_T22data_rpi_straight2templateAffine_step0.nii.gz,warp_MNI-Poly-AMU_cord2segmentation_rpi_straight2templateAffine.nii.gz,-straight2templateAffine.txt,warp_straight2curve.nii.gz -d data.nii -o warp_template2anat.nii.gz')
    # sct.printv('\nConcatenate transformations: anat --> straight --> template...', verbose)
    # sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt,warp_segmentation_rpi_straight2templateAffine2MNI-Poly-AMU_cord.nii.gz,warp_data_rpi_straight2templateAffine_step02MNI-Poly-AMU_T2.nii.gz -d '+fname_template+' -o warp_anat2template.nii.gz')

    # Apply warping fields to anat and template
    if output_type == 1:
        sct.run('sct_apply_transfo -i '+fname_template+' -o template2anat.nii.gz -d data.nii -w warp_template2anat.nii.gz')
        sct.run('sct_apply_transfo -i data.nii  -o anat2template.nii.gz -d '+fname_template+' -w warp_anat2template.nii.gz')

    # come back to parent folder
    os.chdir('..')

   # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'/warp_template2anat.nii.gz', 'warp_template2anat.nii.gz', verbose)
    sct.generate_output_file(path_tmp+'/warp_anat2template.nii.gz', 'warp_anat2template.nii.gz', verbose)
    if output_type == 1:
        sct.generate_output_file(path_tmp+'/template2anat.nii.gz', 'template2anat'+ext_data, verbose)
        sct.generate_output_file(path_tmp+'/anat2template.nii.gz', 'anat2template'+ext_data, verbose)

    # Delete temporary files
    if remove_temp_files:
        sct.printv('\nDelete temporary files...', verbose)
        sct.run('rm -rf '+path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', verbose)

    # to view results
    sct.printv('\nTo view results, type:', verbose)
    sct.printv('fslview '+fname_data+' template2anat -b 0,4000 &', verbose, 'info')
    sct.printv('fslview '+fname_template+' -b 0,5000 anat2template &\n', verbose, 'info')



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
