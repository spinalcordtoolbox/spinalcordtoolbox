#!/usr/bin/env python
#########################################################################################
#
# Motion correction of dMRI data.
#
# Inspired by Xu et al. Neuroimage 2013.
#
# Details of the algorithm:
# - grouping of DW data only (every n volumes, default n=5)
# - average all b0
# - average DWI data within each group
# - average DWI of all groups
# - moco on DWI groups
# - moco on b=0, using target volume: last b=0
# - moco on all dMRI data
# _ generating b=0 mean and DWI mean after motion correction
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-08-15
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: make sure slicewise not used with ants, eddy not used with ants
# TODO: if -f, we only need two plots. Plot 1: X params with fitted spline, plot 2: Y param with fitted splines. Each plot will have all Z slices (with legend Z=0, Z=1, ...) and labels: y; translation (mm), xlabel: volume #. Plus add grid.

from __future__ import division, absolute_import

import sys
import os
import time
import math
from tqdm import tqdm
import numpy as np
import operator
import functools

from spinalcordtoolbox.image import Image, concat_data
from spinalcordtoolbox.moco import ParamMoco, moco, spline, combine_matrix, copy_mat_files

import sct_utils as sct
import sct_dmri_separate_b0_and_dwi
from sct_convert import convert
from sct_image import split_data
from msct_parser import Parser


def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # initialize parameters
    param_default = ParamMoco()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description(
        '  Motion correction of dMRI data. Some of the features to improve robustness were proposed in Xu et al. (http://dx.doi.org/10.1016/j.neuroimage.2012.11.014) and include:\n'
        '- group-wise (-g)\n'
        '- slice-wise regularized along z using polynomial function (-param). For more info about the method, type: isct_antsSliceRegularizedRegistration\n'
        '- masking (-m)\n'
        '- iterative averaging of target volume\n')
    parser.add_option(name='-i',
                      type_value='file',
                      description='Diffusion data',
                      mandatory=True,
                      example='dmri.nii.gz')
    parser.add_option(name='-bvec',
                      type_value='file',
                      description='Bvecs file',
                      mandatory=True,
                      example='bvecs.nii.gz')
    parser.add_option(name='-bval',
                      type_value='file',
                      description='Bvals file',
                      mandatory=False,
                      example='bvals.nii.gz')
    parser.add_option(name='-bvalmin',
                      type_value='float',
                      description='B-value threshold (in s/mm2) below which data is considered as b=0.',
                      mandatory=False,
                      example='50')
    parser.add_option(name='-g',
                      type_value='int',
                      description='Group nvols successive dMRI volumes for more robustness.',
                      mandatory=False,
                      default_value=param_default.group_size,
                      example=['2'])
    parser.add_option(name='-m',
                      type_value='file',
                      description='Binary mask to limit voxels considered by the registration metric.',
                      mandatory=False,
                      example=['dmri_mask.nii.gz'])
    parser.add_option(name='-param',
                      type_value=[[','], 'str'],
                      description="Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
                                  "poly [int]: Degree of polynomial function used for regularization along Z. For no regularization set to 0. Default=" + param_default.poly + ".\n"
                                                "smooth [mm]: Smoothing kernel. Default=" + param_default.smooth + ".\n"
                                                  "metric {MI, MeanSquares, CC}: Metric used for registration. Default=" + param_default.metric + ".\n"
                                                  "gradStep [float]: Searching step used by registration algorithm. The higher the more deformation allowed. Default=" + param_default.gradStep + ".\n"
                                                    "sample [0-1]: Sampling rate used for registration metric. Default=" + param_default.sampling + ".\n",
                      mandatory=False)
    parser.add_option(name='-x',
                      type_value='multiple_choice',
                      description='Final Interpolation.',
                      mandatory=False,
                      default_value=param_default.interp,
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name='-ofolder',
                      type_value='folder_creation',
                      description='Output folder',
                      mandatory=False,
                      default_value='./',
                      example='dmri_moco_results/')
    parser.usage.addSection('MISC')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # initialization
    start_time = time.time()
    path_out = '.'
    param = ParamMoco()

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    param.fname_data = arguments['-i']
    param.fname_bvecs = arguments['-bvec']

    if '-bval' in arguments:
        param.fname_bvals = arguments['-bval']
    if '-bvalmin' in arguments:
        param.bval_min = arguments['-bvalmin']
    if '-g' in arguments:
        param.group_size = arguments['-g']
    if '-m' in arguments:
        param.fname_mask = arguments['-m']
    if '-param' in arguments:
        param.update(arguments['-param'])
    if '-x' in arguments:
        param.interp = arguments['-x']
    if '-ofolder' in arguments:
        path_out = arguments['-ofolder']
    if '-r' in arguments:
        param.remove_temp_files = int(arguments['-r'])
    param.verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=param.verbose, update=True)  # Update log level

    # Get full path
    param.fname_data = os.path.abspath(param.fname_data)
    param.fname_bvecs = os.path.abspath(param.fname_bvecs)
    if param.fname_bvals != '':
        param.fname_bvals = os.path.abspath(param.fname_bvals)
    if param.fname_mask != '':
        param.fname_mask = os.path.abspath(param.fname_mask)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    path_mask, file_mask, ext_mask = sct.extract_fname(param.fname_mask)

    path_tmp = sct.tmp_create(basename="dmri_moco", verbose=param.verbose)

    # names of files in temporary folder
    mask_name = 'mask'
    bvecs_fname = 'bvecs.txt'

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(param.fname_data, os.path.join(path_tmp, "dmri.nii"))
    sct.copy(param.fname_bvecs, os.path.join(path_tmp, bvecs_fname), verbose=param.verbose)
    if param.fname_mask != '':
        sct.copy(param.fname_mask, os.path.join(path_tmp, mask_name + ext_mask), verbose=param.verbose)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # update field in param (because used later).
    # TODO: make this cleaner...
    if param.fname_mask != '':
        param.fname_mask = mask_name + ext_mask

    # run moco
    fname_data_moco_tmp = dmri_moco(param)

    # generate b0_moco_mean and dwi_moco_mean
    args = ['-i', fname_data_moco_tmp, '-bvec', 'bvecs.txt', '-a', '1', '-v', '0']
    if not param.fname_bvals == '':
        # if bvals file is provided
        args += ['-bval', param.fname_bvals]
    fname_b0, fname_b0_mean, fname_dwi, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(args=args)

    # come back
    os.chdir(curdir)

    # Generate output files
    fname_dmri_moco = os.path.join(path_out, file_data + param.suffix + ext_data)
    fname_dmri_moco_b0_mean = sct.add_suffix(fname_dmri_moco, '_b0_mean')
    fname_dmri_moco_dwi_mean = sct.add_suffix(fname_dmri_moco, '_dwi_mean')
    sct.create_folder(path_out)
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(fname_data_moco_tmp, fname_dmri_moco, param.verbose)
    sct.generate_output_file(fname_b0_mean, fname_dmri_moco_b0_mean, param.verbose)
    sct.generate_output_file(fname_dwi_mean, fname_dmri_moco_dwi_mean, param.verbose)

    # Delete temporary files
    if param.remove_temp_files == 1:
        sct.printv('\nDelete temporary files...', param.verbose)
        sct.rmtree(path_tmp, verbose=param.verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', param.verbose)

    sct.display_viewer_syntax([fname_dmri_moco, file_data], mode='ortho,ortho')


def dmri_moco(param):

    file_data = 'dmri.nii'
    file_data_dirname, file_data_basename, file_data_ext = sct.extract_fname(file_data)
    file_b0 = 'b0.nii'
    file_dwi = 'dwi.nii'
    ext_data = '.nii.gz' # workaround "too many open files" by slurping the data
    mat_final = 'mat_final/'
    file_dwi_group = 'dwi_averaged_groups.nii'
    ext_mat = 'Warp.nii.gz'  # warping field

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    im_data = Image(file_data)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), param.verbose)

    # Get orientation
    sct.printv('\nData orientation: ' + im_data.orientation, param.verbose)
    if im_data.orientation[2] in 'LR':
        param.is_sagittal = True
        sct.printv('  Treated as sagittal')
    elif im_data.orientation[2] in 'IS':
        param.is_sagittal = False
        sct.printv('  Treated as axial')
    else:
        param.is_sagittal = False
        sct.printv('WARNING: Orientation seems to be neither axial nor sagittal.')

    # Adjust group size in case of sagittal scan
    if param.is_sagittal and param.group_size != 1:
        sct.printv('For sagittal data group_size should be one for more robustness. Forcing group_size=1.', 1, 'warning')
        param.group_size = 1

    # Identify b=0 and DWI images
    index_b0, index_dwi, nb_b0, nb_dwi = sct_dmri_separate_b0_and_dwi.identify_b0('bvecs.txt', param.fname_bvals, param.bval_min, param.verbose)

    # check if dmri and bvecs are the same size
    if not nb_b0 + nb_dwi == nt:
        sct.printv('\nERROR in ' + os.path.basename(__file__) + ': Size of data (' + str(nt) + ') and size of bvecs (' + str(nb_b0 + nb_dwi) + ') are not the same. Check your bvecs file.\n', 1, 'error')
        sys.exit(2)

    # ==================================================================================================================
    # Prepare data (mean/groups...)
    # ==================================================================================================================

    # Split into T dimension
    sct.printv('\nSplit along T dimension...', param.verbose)
    im_data_split_list = split_data(im_data, 3)
    for im in im_data_split_list:
        x_dirname, x_basename, x_ext = sct.extract_fname(im.absolutepath)
        im.absolutepath = os.path.join(x_dirname, x_basename + ".nii.gz")
        im.save()

    # Merge and average b=0 images
    sct.printv('\nMerge and average b=0 data...', param.verbose)
    im_b0_list = []
    for it in range(nb_b0):
        im_b0_list.append(im_data_split_list[index_b0[it]])
    im_b0 = concat_data(im_b0_list, 3).save(file_b0, verbose=0)
    # Average across time
    im_b0.mean(dim=3).save(sct.add_suffix(file_b0, '_mean'))

    # Number of DWI groups
    nb_groups = int(math.floor(nb_dwi / param.group_size))

    # Generate groups indexes
    group_indexes = []
    for iGroup in range(nb_groups):
        group_indexes.append(index_dwi[(iGroup * param.group_size):((iGroup + 1) * param.group_size)])

    # add the remaining images to the last DWI group
    nb_remaining = nb_dwi%param.group_size  # number of remaining images
    if nb_remaining > 0:
        nb_groups += 1
        group_indexes.append(index_dwi[len(index_dwi) - nb_remaining:len(index_dwi)])

    _, file_dwi_basename, file_dwi_ext = sct.extract_fname(file_dwi)
    # DWI groups
    file_dwi_mean = []
    for iGroup in tqdm(range(nb_groups), unit='iter', unit_scale=False, desc="Merge within groups", ascii=True, ncols=80):
        # get index
        index_dwi_i = group_indexes[iGroup]
        nb_dwi_i = len(index_dwi_i)
        # Merge DW Images
        file_dwi_merge_i = os.path.join(file_dwi_basename + '_' + str(iGroup) + ext_data)
        im_dwi_list = []
        for it in range(nb_dwi_i):
            im_dwi_list.append(im_data_split_list[index_dwi_i[it]])
        im_dwi_out = concat_data(im_dwi_list, 3).save(file_dwi_merge_i, verbose=0)
        # Average across time
        file_dwi_mean.append(os.path.join(file_dwi_basename + '_' + str(iGroup) + '_mean' + ext_data))
        im_dwi_out.mean(dim=3).save(file_dwi_mean[-1])

    # Merge DWI groups means
    sct.printv('\nMerging DW files...', param.verbose)
    # file_dwi_groups_means_merge = 'dwi_averaged_groups'
    im_dw_list = []
    for iGroup in range(nb_groups):
        im_dw_list.append(file_dwi_mean[iGroup])
    concat_data(im_dw_list, 3).save(file_dwi_group, verbose=0)

    # Cleanup
    del im, im_data_split_list

    # ==================================================================================================================
    # Estimate moco
    # ==================================================================================================================

    # Estimate moco on b0 groups
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion on b=0 images...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco = param
    param_moco.file_data = 'b0.nii'
    # identify target image
    if index_dwi[0] != 0:
        # If first DWI is not the first volume (most common), then there is a least one b=0 image before. In that case
        # select it as the target image for registration of all b=0
        param_moco.file_target = os.path.join(file_data_dirname, file_data_basename + '_T' + str(index_b0[index_dwi[0] - 1]).zfill(4) + ext_data)
    else:
        # If first DWI is the first volume, then the target b=0 is the first b=0 from the index_b0.
        param_moco.file_target = os.path.join(file_data_dirname, file_data_basename + '_T' + str(index_b0[0]).zfill(4) + ext_data)

    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_b0groups'
    file_mat_b0, im_b0_moco = moco(param_moco)

    # Estimate moco on dwi groups
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion on DW images...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_dwi_group
    param_moco.file_target = file_dwi_mean[0]  # target is the first DW image (closest to the first b=0)
    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_dwigroups'
    file_mat_dwi_group, im_dwi_moco = moco(param_moco)

    # Spline Regularization along T
    if param.spline_fitting:
        spline(mat_final, nt, nz, param.verbose, np.array(index_b0), param.plot_graph)

    # combine Eddy Matrices
    if param.run_eddy:
        param.mat_2_combine = 'mat_eddy'
        param.mat_final = mat_final
        combine_matrix(param)

    # ==================================================================================================================
    # Apply moco
    # ==================================================================================================================

    # If group_size>1, assign transformation to each individual ungrouped 3d volume
    # TODO: make sure this code works if the initial number of images is not divisible by group_size
    if param.group_size > 1:
        file_mat_dwi = []
        for iz in range(len(file_mat_b0)):
            # duplicate by factor group_size the transformation file for each it
            #  example: [mat.Z0000T0001Warp.nii] --> [mat.Z0000T0001Warp.nii, mat.Z0000T0001Warp.nii] for group_size=2
            file_mat_dwi.append(
                functools.reduce(operator.iconcat, [[i] * param.group_size for i in file_mat_dwi_group[iz]], []))
    else:
        file_mat_dwi = file_mat_dwi_group

    # Copy transformations to mat_final folder and rename them appropriately
    param.suffix_mat = '0GenericAffine.mat' if param.is_sagittal else 'Warp.nii.gz'
    copy_mat_files(nt, file_mat_b0, index_b0, mat_final, param)
    copy_mat_files(nt, file_mat_dwi, index_dwi, mat_final, param)

    # Apply moco on all dmri data
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Apply moco', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_data
    param_moco.file_target = file_dwi_mean[0]  # reference for reslicing into proper coordinate system
    param_moco.path_out = ''  # TODO not used in moco()
    param_moco.mat_moco = mat_final
    param_moco.todo = 'apply'
    _, im_dmri_moco = moco(param_moco)

    # copy geometric information from header
    # NB: this is required because WarpImageMultiTransform in 2D mode wrongly sets pixdim(3) to "1".
    # im_dmri_moco = Image(fname_data_moco)

    im_dmri_moco.header = im_data.header
    im_dmri_moco.save(verbose=0)

    # TODO: output im_dmri_moco.absolutepath
    # Build output file name
    fname_data_moco = os.path.join(file_data_dirname, file_data_basename + param.suffix + '.nii')
    return os.path.abspath(fname_data_moco)


if __name__ == "__main__":
    sct.init_sct()
    main()
