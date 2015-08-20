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

# TODO: testing script for all cases

import sys
import os
import commands
import time

import sct_utils as sct
from sct_orientation import set_orientation
from sct_register_multimodal import Paramreg, ParamregMultiStep, register
from msct_parser import Parser
from msct_image import Image, find_zmin_zmax


# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.remove_temp_files = 1  # remove temporary files
        self.output_type = 1
        self.fname_mask = ''  # this field is needed in the function register@sct_register_multimodal
        self.padding = 10  # this field is needed in the function register@sct_register_multimodal
        # self.speed = 'fast'  # speed of registration. slow | normal | fast
        # self.nb_iterations = '5'
        # self.algo = 'SyN'
        # self.gradientStep = '0.5'
        # self.metric = 'MI'
        self.verbose = 1  # verbose
        self.path_template = path_sct+'/data/template'
        self.file_template = 'MNI-Poly-AMU_T2.nii.gz'
        self.file_template_label = 'landmarks_center.nii.gz'
        self.file_template_seg = 'MNI-Poly-AMU_cord.nii.gz'
        self.zsubsample = '0.25'
        # self.smoothing_sigma = 5  # Smoothing along centerline to improve accuracy and remove step effects



# MAIN
# ==========================================================================================
def main():

    # get default parameters
    step1 = Paramreg(step='1', type='seg', algo='slicereg', metric='MeanSquares', iter='10')
    step2 = Paramreg(step='2', type='im', algo='syn', metric='MI', iter='3')
    # step1 = Paramreg()
    paramreg = ParamregMultiStep([step1, step2])

    # step1 = Paramreg_step(step='1', type='seg', algo='bsplinesyn', metric='MeanSquares', iter='10', shrink='1', smooth='0', gradStep='0.5')
    # step2 = Paramreg_step(step='2', type='im', algo='syn', metric='MI', iter='10', shrink='1', smooth='0', gradStep='0.5')
    # paramreg = ParamregMultiStep([step1, step2])

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
                      type_value=[[':'], 'str'],
                      description="""Parameters for registration (see sct_register_multimodal). Default:\n--\nstep=1\ntype="""+paramreg.steps['1'].type+"""\nalgo="""+paramreg.steps['1'].algo+"""\nmetric="""+paramreg.steps['1'].metric+"""\npoly="""+paramreg.steps['1'].poly+"""\n--\nstep=2\ntype="""+paramreg.steps['2'].type+"""\nalgo="""+paramreg.steps['2'].algo+"""\nmetric="""+paramreg.steps['2'].metric+"""\niter="""+paramreg.steps['2'].iter+"""\nshrink="""+paramreg.steps['2'].shrink+"""\nsmooth="""+paramreg.steps['2'].smooth+"""\ngradStep="""+paramreg.steps['2'].gradStep+"""\n--""",
                      mandatory=False,
                      example="step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=5,shrink=2:step=3,type=im,algo=syn,metric=MI,iter=5,shrink=1,gradStep=0.3")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
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
        if '-p' in arguments:
            paramreg_user = arguments['-p']
            # update registration parameters
            for paramStep in paramreg_user:
                paramreg.addStep(paramStep)

    # initialize other parameters
    file_template = param.file_template
    file_template_label = param.file_template_label
    file_template_seg = param.file_template_seg
    output_type = param.output_type
    zsubsample = param.zsubsample
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
    sct.printv('\nCheck template files...')
    sct.check_file_exist(fname_template, verbose)
    sct.check_file_exist(fname_template_label, verbose)
    sct.check_file_exist(fname_template_seg, verbose)

    # print arguments
    sct.printv('\nCheck parameters:', verbose)
    sct.printv('.. Data:                 '+fname_data, verbose)
    sct.printv('.. Landmarks:            '+fname_landmarks, verbose)
    sct.printv('.. Segmentation:         '+fname_seg, verbose)
    sct.printv('.. Path template:        '+path_template, verbose)
    sct.printv('.. Output type:          '+str(output_type), verbose)
    sct.printv('.. Remove temp files:    '+str(remove_temp_files), verbose)

    sct.printv('\nParameters for registration:')
    for pStep in range(1, len(paramreg.steps)+1):
        sct.printv('Step #'+paramreg.steps[str(pStep)].step, verbose)
        sct.printv('.. Type #'+paramreg.steps[str(pStep)].type, verbose)
        sct.printv('.. Algorithm................ '+paramreg.steps[str(pStep)].algo, verbose)
        sct.printv('.. Metric................... '+paramreg.steps[str(pStep)].metric, verbose)
        sct.printv('.. Number of iterations..... '+paramreg.steps[str(pStep)].iter, verbose)
        sct.printv('.. Shrink factor............ '+paramreg.steps[str(pStep)].shrink, verbose)
        sct.printv('.. Smoothing factor......... '+paramreg.steps[str(pStep)].smooth, verbose)
        sct.printv('.. Gradient step............ '+paramreg.steps[str(pStep)].gradStep, verbose)
        sct.printv('.. Degree of polynomial..... '+paramreg.steps[str(pStep)].poly, verbose)

    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    sct.printv('\nCheck input labels...')
    # check if label image contains coherent labels
    image_label = Image(fname_landmarks)
    # -> all labels must be different
    labels = image_label.getNonZeroCoordinates(sorting='value')
    hasDifferentLabels = True
    for lab in labels:
        for otherlabel in labels:
            if lab != otherlabel and lab.hasEqualValue(otherlabel):
                hasDifferentLabels = False
                break
    if not hasDifferentLabels:
        sct.printv('ERROR: Wrong landmarks input. All labels must be different.', verbose, 'error')
    # all labels must be available in tempalte
    image_label_template = Image(fname_template_label)
    labels_template = image_label_template.getNonZeroCoordinates(sorting='value')
    if labels[-1].value > labels_template[-1].value:
        sct.printv('ERROR: Wrong landmarks input. Labels must have correspondance in tempalte space. \nLabel max '
                   'provided: ' + str(labels[-1].value) + '\nLabel max from template: ' +
                   str(labels_template[-1].value), verbose, 'error')


    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    status, output = sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    sct.printv('\nCopy files...', verbose)
    sct.run('isct_c3d '+fname_data+' -o '+path_tmp+'/data.nii')
    sct.run('isct_c3d '+fname_landmarks+' -o '+path_tmp+'/landmarks.nii.gz')
    sct.run('isct_c3d '+fname_seg+' -o '+path_tmp+'/segmentation.nii.gz')
    sct.run('isct_c3d '+fname_template+' -o '+path_tmp+'/template.nii')
    sct.run('isct_c3d '+fname_template_label+' -o '+path_tmp+'/template_labels.nii.gz')
    sct.run('isct_c3d '+fname_template_seg+' -o '+path_tmp+'/template_seg.nii.gz')

    # go to tmp folder
    os.chdir(path_tmp)

    # resample data to 1mm isotropic
    sct.printv('\nResample data to 1mm isotropic...', verbose)
    sct.run('isct_c3d data.nii -resample-mm 1.0x1.0x1.0mm -interpolation Linear -o datar.nii')
    sct.run('isct_c3d segmentation.nii.gz -resample-mm 1.0x1.0x1.0mm -interpolation NearestNeighbor -o segmentationr.nii.gz')
    # N.B. resampling of labels is more complicated, because they are single-point labels, therefore resampling with neighrest neighbour can make them disappear. Therefore a more clever approach is required.
    resample_labels('landmarks.nii.gz', 'datar.nii', 'landmarksr.nii.gz')
    # # TODO
    # sct.run('sct_label_utils -i datar.nii -t create -x 124,186,19,2:129,98,23,8 -o landmarksr.nii.gz')

    # Change orientation of input images to RPI
    sct.printv('\nChange orientation of input images to RPI...', verbose)
    set_orientation('datar.nii', 'RPI', 'data_rpi.nii')
    set_orientation('landmarksr.nii.gz', 'RPI', 'landmarks_rpi.nii.gz')
    set_orientation('segmentationr.nii.gz', 'RPI', 'segmentation_rpi.nii.gz')

    # # Change orientation of input images to RPI
    # sct.printv('\nChange orientation of input images to RPI...', verbose)
    # set_orientation('data.nii', 'RPI', 'data_rpi.nii')
    # set_orientation('landmarks.nii.gz', 'RPI', 'landmarks_rpi.nii.gz')
    # set_orientation('segmentation.nii.gz', 'RPI', 'segmentation_rpi.nii.gz')

    # get landmarks in native space
    # crop segmentation
    # output: segmentation_rpi_crop.nii.gz
    sct.run('sct_crop_image -i segmentation_rpi.nii.gz -o segmentation_rpi_crop.nii.gz -dim 2 -bzmax')

    # straighten segmentation
    sct.printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)
    sct.run('sct_straighten_spinalcord -i segmentation_rpi_crop.nii.gz -c segmentation_rpi_crop.nii.gz -r 0 -v '+str(verbose), verbose)
    # re-define warping field using non-cropped space (to avoid issue #367)
    sct.run('sct_concat_transfo -w warp_straight2curve.nii.gz -d data_rpi.nii -o warp_straight2curve.nii.gz')

    # Label preparation:
    # --------------------------------------------------------------------------------
    # Remove unused label on template. Keep only label present in the input label image
    sct.printv('\nRemove unused label on template. Keep only label present in the input label image...', verbose)
    sct.run('sct_label_utils -t remove -i template_labels.nii.gz -o template_label.nii.gz -r landmarks_rpi.nii.gz')

    # Make sure landmarks are INT
    sct.printv('\nConvert landmarks to INT...', verbose)
    sct.run('isct_c3d template_label.nii.gz -type int -o template_label.nii.gz', verbose)

    # Create a cross for the template labels - 5 mm
    sct.printv('\nCreate a 5 mm cross for the template labels...', verbose)
    sct.run('sct_label_utils -t cross -i template_label.nii.gz -o template_label_cross.nii.gz -c 5')

    # Create a cross for the input labels and dilate for straightening preparation - 5 mm
    sct.printv('\nCreate a 5mm cross for the input labels and dilate for straightening preparation...', verbose)
    sct.run('sct_label_utils -t cross -i landmarks_rpi.nii.gz -o landmarks_rpi_cross3x3.nii.gz -c 5 -d')

    # Apply straightening to labels
    sct.printv('\nApply straightening to labels...', verbose)
    sct.run('sct_apply_transfo -i landmarks_rpi_cross3x3.nii.gz -o landmarks_rpi_cross3x3_straight.nii.gz -d segmentation_rpi_crop_straight.nii.gz -w warp_curve2straight.nii.gz -x nn')

    # Convert landmarks from FLOAT32 to INT
    sct.printv('\nConvert landmarks from FLOAT32 to INT...', verbose)
    sct.run('isct_c3d landmarks_rpi_cross3x3_straight.nii.gz -type int -o landmarks_rpi_cross3x3_straight.nii.gz')

    # Remove unused label on template. Keep only label present in the input label image
    sct.printv('\nRemove labels that do not correspond with each others.', verbose)
    sct.run('sct_label_utils -t remove-symm -i landmarks_rpi_cross3x3_straight.nii.gz -o landmarks_rpi_cross3x3_straight.nii.gz,template_label_cross.nii.gz -r template_label_cross.nii.gz')

    # Estimate affine transfo: straight --> template (landmark-based)'
    sct.printv('\nEstimate affine transfo: straight anat --> template (landmark-based)...', verbose)
    # converting landmarks straight and curved to physical coordinates
    image_straight = Image('landmarks_rpi_cross3x3_straight.nii.gz')
    landmark_straight = image_straight.getNonZeroCoordinates(sorting='value')
    image_template = Image('template_label_cross.nii.gz')
    landmark_template = image_template.getNonZeroCoordinates(sorting='value')
    # Reorganize landmarks
    points_fixed, points_moving = [], []
    landmark_straight_mean = []
    for coord in landmark_straight:
        if coord.value not in [c.value for c in landmark_straight_mean]:
            temp_landmark = coord
            temp_number = 1
            for other_coord in landmark_straight:
                if coord.hasEqualValue(other_coord) and coord != other_coord:
                    temp_landmark += other_coord
                    temp_number += 1
            landmark_straight_mean.append(temp_landmark / temp_number)

    for coord in landmark_straight_mean:
        point_straight = image_straight.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        points_moving.append([point_straight[0][0], point_straight[0][1], point_straight[0][2]])
    for coord in landmark_template:
        point_template = image_template.transfo_pix2phys([[coord.x, coord.y, coord.z]])
        points_fixed.append([point_template[0][0], point_template[0][1], point_template[0][2]])

    # Register curved landmarks on straight landmarks based on python implementation
    sct.printv('\nComputing rigid transformation (algo=translation-scaling-z) ...', verbose)
    import msct_register_landmarks
    (rotation_matrix, translation_array, points_moving_reg, points_moving_barycenter) = \
        msct_register_landmarks.getRigidTransformFromLandmarks(
            points_fixed, points_moving, constraints='translation-scaling-z', show=False)

    # writing rigid transformation file
    text_file = open("straight2templateAffine.txt", "w")
    text_file.write("#Insight Transform File V1.0\n")
    text_file.write("#Transform 0\n")
    text_file.write("Transform: FixedCenterOfRotationAffineTransform_double_3_3\n")
    text_file.write("Parameters: %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % (
        1.0/rotation_matrix[0, 0], rotation_matrix[0, 1],     rotation_matrix[0, 2],
        rotation_matrix[1, 0],     1.0/rotation_matrix[1, 1], rotation_matrix[1, 2],
        rotation_matrix[2, 0],     rotation_matrix[2, 1],     1.0/rotation_matrix[2, 2],
        translation_array[0, 0],   translation_array[0, 1],   -translation_array[0, 2]))
    text_file.write("FixedParameters: %.9f %.9f %.9f\n" % (points_moving_barycenter[0],
                                                           points_moving_barycenter[1],
                                                           points_moving_barycenter[2]))
    text_file.close()

    # Apply affine transformation: straight --> template
    sct.printv('\nApply affine transformation: straight --> template...', verbose)
    sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt -d template.nii -o warp_curve2straightAffine.nii.gz')
    sct.run('sct_apply_transfo -i data_rpi.nii -o data_rpi_straight2templateAffine.nii -d template.nii -w warp_curve2straightAffine.nii.gz')
    sct.run('sct_apply_transfo -i segmentation_rpi.nii.gz -o segmentation_rpi_straight2templateAffine.nii.gz -d template.nii -w warp_curve2straightAffine.nii.gz -x linear')

    # find min-max of anat2template (for subsequent cropping)
    sct.run('export FSLOUTPUTTYPE=NIFTI; fslmaths segmentation_rpi_straight2templateAffine.nii.gz -thr 0.5 segmentation_rpi_straight2templateAffine_th.nii.gz', param.verbose)
    zmin_template, zmax_template = find_zmin_zmax('segmentation_rpi_straight2templateAffine_th.nii.gz')

    # crop template in z-direction (for faster processing)
    sct.printv('\nCrop data in template space (for faster processing)...', verbose)
    sct.run('sct_crop_image -i template.nii -o template_crop.nii -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    sct.run('sct_crop_image -i template_seg.nii.gz -o template_seg_crop.nii.gz -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    sct.run('sct_crop_image -i data_rpi_straight2templateAffine.nii -o data_rpi_straight2templateAffine_crop.nii -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    sct.run('sct_crop_image -i segmentation_rpi_straight2templateAffine.nii.gz -o segmentation_rpi_straight2templateAffine_crop.nii.gz -dim 2 -start '+str(zmin_template)+' -end '+str(zmax_template))
    # sub-sample in z-direction
    sct.printv('\nSub-sample in z-direction (for faster processing)...', verbose)
    sct.run('sct_resample -i template_crop.nii -o template_crop_r.nii -f 1x1x'+zsubsample, verbose)
    sct.run('sct_resample -i template_seg_crop.nii.gz -o template_seg_crop_r.nii.gz -f 1x1x'+zsubsample, verbose)
    sct.run('sct_resample -i data_rpi_straight2templateAffine_crop.nii -o data_rpi_straight2templateAffine_crop_r.nii -f 1x1x'+zsubsample, verbose)
    sct.run('sct_resample -i segmentation_rpi_straight2templateAffine_crop.nii.gz -o segmentation_rpi_straight2templateAffine_crop_r.nii.gz -f 1x1x'+zsubsample, verbose)

    # Registration straight spinal cord to template
    sct.printv('\nRegister straight spinal cord to template...', verbose)

    # loop across registration steps
    warp_forward = []
    warp_inverse = []
    for i_step in range(1, len(paramreg.steps)+1):
        sct.printv('\nEstimate transformation for step #'+str(i_step)+'...', verbose)
        # identify which is the src and dest
        if paramreg.steps[str(i_step)].type == 'im':
            src = 'data_rpi_straight2templateAffine_crop_r.nii'
            dest = 'template_crop_r.nii'
            interp_step = 'linear'
        elif paramreg.steps[str(i_step)].type == 'seg':
            src = 'segmentation_rpi_straight2templateAffine_crop_r.nii.gz'
            dest = 'template_seg_crop_r.nii.gz'
            interp_step = 'nn'
        else:
            sct.printv('ERROR: Wrong image type.', 1, 'error')
        # if step>1, apply warp_forward_concat to the src image to be used
        if i_step > 1:
            # sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+sct.add_suffix(src, '_reg')+' -x '+interp_step, verbose)
            sct.run('sct_apply_transfo -i '+src+' -d '+dest+' -w '+','.join(warp_forward)+' -o '+sct.add_suffix(src, '_reg')+' -x '+interp_step, verbose)
            src = sct.add_suffix(src, '_reg')
        # register src --> dest
        warp_forward_out, warp_inverse_out = register(src, dest, paramreg, param, str(i_step))
        warp_forward.append(warp_forward_out)
        warp_inverse.append(warp_inverse_out)

    # Concatenate transformations:
    sct.printv('\nConcatenate transformations: anat --> template...', verbose)
    sct.run('sct_concat_transfo -w warp_curve2straightAffine.nii.gz,'+','.join(warp_forward)+' -d template.nii -o warp_anat2template.nii.gz', verbose)
    # sct.run('sct_concat_transfo -w warp_curve2straight.nii.gz,straight2templateAffine.txt,'+','.join(warp_forward)+' -d template.nii -o warp_anat2template.nii.gz', verbose)
    warp_inverse.reverse()
    sct.run('sct_concat_transfo -w '+','.join(warp_inverse)+',-straight2templateAffine.txt,warp_straight2curve.nii.gz -d data.nii -o warp_template2anat.nii.gz', verbose)

    # Apply warping fields to anat and template
    if output_type == 1:
        sct.run('sct_apply_transfo -i template.nii -o template2anat.nii.gz -d data.nii -w warp_template2anat.nii.gz -c 1', verbose)
        sct.run('sct_apply_transfo -i data.nii -o anat2template.nii.gz -d template.nii -w warp_anat2template.nii.gz -c 1', verbose)

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


# Resample labels
# ==========================================================================================
def resample_labels(fname_labels, fname_dest, fname_output):
    """
    This function re-create labels into a space that has been resampled. It works by re-defining the location of each
    label using the old and new voxel size.
    """
    # get dimensions of input and destination files
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_labels)
    nxd, nyd, nzd, ntd, pxd, pyd, pzd, ptd = sct.get_dimension(fname_dest)
    sampling_factor = [float(nx)/nxd, float(ny)/nyd, float(nz)/nzd]
    # read labels
    from sct_label_utils import ProcessLabels
    processor = ProcessLabels(fname_labels)
    label_list = processor.display_voxel()
    label_new_list = []
    for label in label_list:
        label_sub_new = [str(int(round(int(label.x)/sampling_factor[0]))),
                         str(int(round(int(label.y)/sampling_factor[1]))),
                         str(int(round(int(label.z)/sampling_factor[2]))),
                         str(int(float(label.value)))]
        label_new_list.append(','.join(label_sub_new))
    label_new_list = ':'.join(label_new_list)
    # create new labels
    sct.run('sct_label_utils -i '+fname_dest+' -t create -x '+label_new_list+' -v 1 -o '+fname_output)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
