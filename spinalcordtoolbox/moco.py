#!/usr/bin/env python
# -*- coding: utf-8
# Tools for motion correction (moco)
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad

# TODO: Inform user if soft mask is used
# TODO: no need to pass absolute image path-- makes it difficult to read
# TODO: check the status of spline()
# TODO: check the status of combine_matrix()
# TODO: params for ANTS: CC/MI, shrink fact, nb_it
# TODO: ants: explore optin  --float  for faster computation


from copy import deepcopy
import sys
import os
from shutil import copyfile
import glob
import numpy as np
import math
import scipy.interpolate
import time
import functools
import operator
import csv

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import sct_progress_bar

import sct_utils as sct
import sct_dmri_separate_b0_and_dwi
from sct_convert import convert
from sct_image import split_data, concat_data, multicomponent_split
import sct_apply_transfo


class ParamMoco:
    """
    Class with a bunch of moco-specific parameters
    """
    # The constructor
    def __init__(self, is_diffusion=None, group_size=1, metric='MeanSquares', smooth='1'):
        """

        :param is_diffusion: Bool: If True, data will be treated as diffusion-MRI data (process slightly differs)
        :param group_size: int: Number of images averaged for 'dwi' method.
        :param metric: {MeanSquares, MI, CC}: metric to use for registration
        :param smooth: str: Smoothing sigma in mm # TODO: make it int
        """
        self.is_diffusion = is_diffusion
        self.debug = 0
        self.fname_data = ''
        self.fname_bvecs = ''
        self.fname_bvals = ''
        self.fname_target = ''
        self.fname_mask = ''
        self.path_out = ''
        self.mat_final = ''
        self.todo = ''
        self.group_size = group_size
        self.spline_fitting = 0
        self.remove_temp_files = 1
        self.verbose = 1
        self.plot_graph = 0
        self.suffix = '_moco'
        self.poly = '2'  # degree of polynomial function for moco
        self.smooth = smooth
        self.gradStep = '1'  # gradientStep for searching algorithm
        self.iter = '10'  # number of iterations
        self.metric = metric
        self.sampling = 'None'  # sampling rate used for registration metric; 'None' means use 'dense sampling'
        self.interp = 'spline'  # nn, linear, spline
        self.min_norm = 0.001
        self.swapXY = 0
        self.num_target = '0'
        self.suffix_mat = None  # '0GenericAffine.mat' or 'Warp.nii.gz' depending which transfo algo is used
        self.bval_min = 100  # in case user does not have min bvalues at 0, set threshold (where csf disapeared).
        self.iterAvg = 1  # iteratively average target image for more robust moco
        self.is_sagittal = False  # if True, then split along Z (right-left) and register each 2D slice (vs. 3D volume)
        self.output_motion_param = True  # if True, the motion parameters are outputted

    # update constructor with user's parameters
    def update(self, param_user):
        # list_objects = param_user.split(',')
        for object in param_user:
            if len(object) < 2:
                sct.printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


def copy_mat_files(nt, list_file_mat, index, folder_out, param):
    """
    Copy mat file from the grouped folder to the final folder (will be used by all individual ungrouped volumes)
    :param nt: int: Total number of volumes in native 4d data
    :param list_file_mat: list of list: File name of transformations
    :param index: list: Index to associate a given matrix file with a 3d volume (from the 4d native data)
    :param param: Param class
    :param folder_out: str: Output folder
    :return: None
    """
    # create final mat folder
    sct.create_folder(folder_out)
    # Loop across registration matrices and copy to mat_final folder
    # First loop is accross z. If axial orientation, there is only one z (i.e., len(file_mat)=1)
    for iz in range(len(list_file_mat)):
        # Second loop is across ALL volumes of the input dmri dataset (corresponds to its 4th dimension: time)
        for it in range(nt):
            # Check if this index corresponds to a volume listed in the index list
            if it in index:
                file_mat = list_file_mat[iz][index.index(it)]
                fsrc = os.path.join(file_mat + param.suffix_mat)
                # Build final transfo file name
                file_mat_final = os.path.basename(file_mat)[:-9] + str(iz).zfill(4) + 'T' + str(it).zfill(4)
                fdest = os.path.join(folder_out, file_mat_final + param.suffix_mat)
                copyfile(fsrc, fdest)


def moco_wrapper(param):
    """
    Wrapper that performs motion correction.
    :param param: ParamMoco class
    :return: None
    """
    file_data = 'data.nii'  # corresponds to the full input data (e.g. dmri or fmri)
    file_data_dirname, file_data_basename, file_data_ext = sct.extract_fname(file_data)
    file_b0 = 'b0.nii'
    file_datasub = 'datasub.nii'  # corresponds to the full input data minus the b=0 scans (if param.is_diffusion=True)
    file_datasubgroup = 'datasub-groups.nii'  # concatenation of the average of each file_datasub
    file_mask = 'mask.nii'
    file_moco_params_csv = 'moco_params.tsv'
    file_moco_params_x = 'moco_params_x.nii.gz'
    file_moco_params_y = 'moco_params_y.nii.gz'
    ext_data = '.nii.gz'  # workaround "too many open files" by slurping the data
    # TODO: check if .nii can be used
    mat_final = 'mat_final/'
    # ext_mat = 'Warp.nii.gz'  # warping field

    # Start timer
    start_time = time.time()

    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  Input file ............ ' + param.fname_data, param.verbose)
    sct.printv('  Group size ............ {}'.format(param.group_size), param.verbose)

    # Get full path
    # param.fname_data = os.path.abspath(param.fname_data)
    # param.fname_bvecs = os.path.abspath(param.fname_bvecs)
    # if param.fname_bvals != '':
    #     param.fname_bvals = os.path.abspath(param.fname_bvals)

    # Extract path, file and extension
    # path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    # path_mask, file_mask, ext_mask = sct.extract_fname(param.fname_mask)

    path_tmp = sct.tmp_create(basename="moco", verbose=param.verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(param.fname_data, os.path.join(path_tmp, file_data))
    if param.fname_mask != '':
        convert(param.fname_mask, os.path.join(path_tmp, file_mask), verbose=param.verbose)
        # Update field in param (because used later in another function, and param class will be passed)
        param.fname_mask = file_mask

    # Build absolute output path and go to tmp folder
    curdir = os.getcwd()
    path_out_abs = os.path.abspath(param.path_out)
    os.chdir(path_tmp)

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
        sct.printv('WARNING: Orientation seems to be neither axial nor sagittal. Treated as axial.')

    sct.printv("\nSet suffix of transformation file name, which depends on the orientation:")
    if param.is_sagittal:
        param.suffix_mat = '0GenericAffine.mat'
        sct.printv("Orientation is sagittal, suffix is '{}'. The image is split across the R-L direction, and the "
                   "estimated transformation is a 2D affine transfo.".format(param.suffix_mat))
    else:
        param.suffix_mat = 'Warp.nii.gz'
        sct.printv("Orientation is axial, suffix is '{}'. The estimated transformation is a 3D warping field, which is "
                   "composed of a stack of 2D Tx-Ty transformations".format(param.suffix_mat))

    # Adjust group size in case of sagittal scan
    if param.is_sagittal and param.group_size != 1:
        sct.printv('For sagittal data group_size should be one for more robustness. Forcing group_size=1.', 1,
                   'warning')
        param.group_size = 1

    if param.is_diffusion:
        # Identify b=0 and DWI images
        index_b0, index_dwi, nb_b0, nb_dwi = \
            sct_dmri_separate_b0_and_dwi.identify_b0(param.fname_bvecs, param.fname_bvals, param.bval_min,
                                                     param.verbose)

        # check if dmri and bvecs are the same size
        if not nb_b0 + nb_dwi == nt:
            sct.printv(
                '\nERROR in ' + os.path.basename(__file__) + ': Size of data (' + str(nt) + ') and size of bvecs (' + str(
                    nb_b0 + nb_dwi) + ') are not the same. Check your bvecs file.\n', 1, 'error')
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

    if param.is_diffusion:
        # Merge and average b=0 images
        sct.printv('\nMerge and average b=0 data...', param.verbose)
        im_b0_list = []
        for it in range(nb_b0):
            im_b0_list.append(im_data_split_list[index_b0[it]])
        im_b0 = concat_data(im_b0_list, 3).save(file_b0, verbose=0)
        # Average across time
        im_b0.mean(dim=3).save(sct.add_suffix(file_b0, '_mean'))

        n_moco = nb_dwi  # set number of data to perform moco on (using grouping)
        index_moco = index_dwi

    # If not a diffusion scan, we will motion-correct all volumes
    else:
        n_moco = nt
        index_moco = list(range(0, nt))

    nb_groups = int(math.floor(n_moco / param.group_size))

    # Generate groups indexes
    group_indexes = []
    for iGroup in range(nb_groups):
        group_indexes.append(index_moco[(iGroup * param.group_size):((iGroup + 1) * param.group_size)])

    # add the remaining images to a new last group (in case the total number of image is not divisible by group_size)
    nb_remaining = n_moco % param.group_size  # number of remaining images
    if nb_remaining > 0:
        nb_groups += 1
        group_indexes.append(index_moco[len(index_moco) - nb_remaining:len(index_moco)])

    _, file_dwi_basename, file_dwi_ext = sct.extract_fname(file_datasub)
    # Group data
    list_file_group = []
    for iGroup in sct_progress_bar(range(nb_groups), unit='iter', unit_scale=False, desc="Merge within groups", ascii=False,
                                   ncols=80):
        # get index
        index_moco_i = group_indexes[iGroup]
        n_moco_i = len(index_moco_i)
        # concatenate images across time, within this group
        file_dwi_merge_i = os.path.join(file_dwi_basename + '_' + str(iGroup) + ext_data)
        im_dwi_list = []
        for it in range(n_moco_i):
            im_dwi_list.append(im_data_split_list[index_moco_i[it]])
        im_dwi_out = concat_data(im_dwi_list, 3).save(file_dwi_merge_i, verbose=0)
        # Average across time
        list_file_group.append(os.path.join(file_dwi_basename + '_' + str(iGroup) + '_mean' + ext_data))
        im_dwi_out.mean(dim=3).save(list_file_group[-1])

    # Merge across groups
    sct.printv('\nMerge across groups...', param.verbose)
    # file_dwi_groups_means_merge = 'dwi_averaged_groups'
    im_dw_list = []
    for iGroup in range(nb_groups):
        im_dw_list.append(list_file_group[iGroup])
    concat_data(im_dw_list, 3).save(file_datasubgroup, verbose=0)

    # Cleanup
    del im, im_data_split_list

    # ==================================================================================================================
    # Estimate moco
    # ==================================================================================================================

    # Initialize another class instance that will be passed on to the moco() function
    param_moco = deepcopy(param)

    if param.is_diffusion:
        # Estimate moco on b0 groups
        sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
        sct.printv('  Estimating motion on b=0 images...', param.verbose)
        sct.printv('-------------------------------------------------------------------------------', param.verbose)
        param_moco.file_data = 'b0.nii'
        # Identify target image
        if index_moco[0] != 0:
            # If first DWI is not the first volume (most common), then there is a least one b=0 image before. In that
            # case select it as the target image for registration of all b=0
            param_moco.file_target = os.path.join(file_data_dirname,
                                                  file_data_basename + '_T' + str(index_b0[index_moco[0] - 1]).zfill(
                                                      4) + ext_data)
        else:
            # If first DWI is the first volume, then the target b=0 is the first b=0 from the index_b0.
            param_moco.file_target = os.path.join(file_data_dirname,
                                                  file_data_basename + '_T' + str(index_b0[0]).zfill(4) + ext_data)
        # Run moco
        param_moco.path_out = ''
        param_moco.todo = 'estimate_and_apply'
        param_moco.mat_moco = 'mat_b0groups'
        file_mat_b0, _ = moco(param_moco)

    # Estimate moco across groups
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion across groups...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_datasubgroup
    param_moco.file_target = list_file_group[0]  # target is the first volume (closest to the first b=0 if DWI scan)
    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_groups'
    file_mat_datasub_group, _ = moco(param_moco)

    # Spline Regularization along T
    if param.spline_fitting:
        # TODO: fix this scenario (haven't touched that code for a while-- it is probably buggy)
        raise NotImplementedError()
        # spline(mat_final, nt, nz, param.verbose, np.array(index_b0), param.plot_graph)

    # ==================================================================================================================
    # Apply moco
    # ==================================================================================================================

    # If group_size>1, assign transformation to each individual ungrouped 3d volume
    if param.group_size > 1:
        file_mat_datasub = []
        for iz in range(len(file_mat_datasub_group)):
            # duplicate by factor group_size the transformation file for each it
            #  example: [mat.Z0000T0001Warp.nii] --> [mat.Z0000T0001Warp.nii, mat.Z0000T0001Warp.nii] for group_size=2
            file_mat_datasub.append(
                functools.reduce(operator.iconcat, [[i] * param.group_size for i in file_mat_datasub_group[iz]], []))
    else:
        file_mat_datasub = file_mat_datasub_group

    # Copy transformations to mat_final folder and rename them appropriately
    copy_mat_files(nt, file_mat_datasub, index_moco, mat_final, param)
    if param.is_diffusion:
        copy_mat_files(nt, file_mat_b0, index_b0, mat_final, param)

    # Apply moco on all dmri data
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Apply moco', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_data
    param_moco.file_target = list_file_group[0]  # reference for reslicing into proper coordinate system
    param_moco.path_out = ''  # TODO not used in moco()
    param_moco.mat_moco = mat_final
    param_moco.todo = 'apply'
    file_mat_data, im_moco = moco(param_moco)

    # copy geometric information from header
    # NB: this is required because WarpImageMultiTransform in 2D mode wrongly sets pixdim(3) to "1".
    im_moco.header = im_data.header
    im_moco.save(verbose=0)

    # Average across time
    if param.is_diffusion:
        # generate b0_moco_mean and dwi_moco_mean
        args = ['-i', im_moco.absolutepath, '-bvec', param.fname_bvecs, '-a', '1', '-v', '0']
        if not param.fname_bvals == '':
            # if bvals file is provided
            args += ['-bval', param.fname_bvals]
        fname_b0, fname_b0_mean, fname_dwi, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(args=args)
    else:
        fname_moco_mean = sct.add_suffix(im_moco.absolutepath, '_mean')
        im_moco.mean(dim=3).save(fname_moco_mean)

    # Extract and output the motion parameters (doesn't work for sagittal orientation)
    sct.printv('Extract motion parameters...')
    if param.output_motion_param:
        if param.is_sagittal:
            sct.printv('Motion parameters cannot be generated for sagittal images.', 1, 'warning')
        else:
            files_warp_X, files_warp_Y = [], []
            moco_param = []
            for fname_warp in file_mat_data[0]:
                # Cropping the image to keep only one voxel in the XY plane
                im_warp = Image(fname_warp + param.suffix_mat)
                im_warp.data = np.expand_dims(np.expand_dims(im_warp.data[0, 0, :, :, :], axis=0), axis=0)

                # These three lines allow to generate one file instead of two, containing X, Y and Z moco parameters
                #fname_warp_crop = fname_warp + '_crop_' + ext_mat
                #files_warp.append(fname_warp_crop)
                #im_warp.save(fname_warp_crop)

                # Separating the three components and saving X and Y only (Z is equal to 0 by default).
                im_warp_XYZ = multicomponent_split(im_warp)

                fname_warp_crop_X = fname_warp + '_crop_X_' + param.suffix_mat
                im_warp_XYZ[0].save(fname_warp_crop_X)
                files_warp_X.append(fname_warp_crop_X)

                fname_warp_crop_Y = fname_warp + '_crop_Y_' + param.suffix_mat
                im_warp_XYZ[1].save(fname_warp_crop_Y)
                files_warp_Y.append(fname_warp_crop_Y)

                # Calculating the slice-wise average moco estimate to provide a QC file
                moco_param.append([np.mean(np.ravel(im_warp_XYZ[0].data)), np.mean(np.ravel(im_warp_XYZ[1].data))])

            # These two lines allow to generate one file instead of two, containing X, Y and Z moco parameters
            #im_warp_concat = concat_data(files_warp, dim=3)
            #im_warp_concat.save('fmri_moco_params.nii')

            # Concatenating the moco parameters into a time series for X and Y components.
            im_warp_concat = concat_data(files_warp_X, dim=3)
            im_warp_concat.save(file_moco_params_x)

            im_warp_concat = concat_data(files_warp_Y, dim=3)
            im_warp_concat.save(file_moco_params_y)

            # Writing a TSV file with the slicewise average estimate of the moco parameters. Useful for QC
            with open(file_moco_params_csv, 'wt') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(['X', 'Y'])
                for mocop in moco_param:
                    tsv_writer.writerow([mocop[0], mocop[1]])

    # Generate output files
    sct.printv('\nGenerate output files...', param.verbose)
    fname_moco = os.path.join(path_out_abs, sct.add_suffix(os.path.basename(param.fname_data), param.suffix))
    sct.generate_output_file(im_moco.absolutepath, fname_moco)
    if param.is_diffusion:
        sct.generate_output_file(fname_b0_mean, sct.add_suffix(fname_moco, '_b0_mean'))
        sct.generate_output_file(fname_dwi_mean, sct.add_suffix(fname_moco, '_dwi_mean'))
    else:
        sct.generate_output_file(fname_moco_mean, sct.add_suffix(fname_moco, '_mean'))
    if os.path.exists(file_moco_params_csv):
        sct.generate_output_file(file_moco_params_x, os.path.join(path_out_abs, file_moco_params_x),
                                 squeeze_data=False)
        sct.generate_output_file(file_moco_params_y, os.path.join(path_out_abs, file_moco_params_y),
                                 squeeze_data=False)
        sct.generate_output_file(file_moco_params_csv, os.path.join(path_out_abs, file_moco_params_csv))

    # Delete temporary files
    if param.remove_temp_files == 1:
        sct.printv('\nDelete temporary files...', param.verbose)
        sct.rmtree(path_tmp, verbose=param.verbose)

    # come back to working directory
    os.chdir(curdir)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', param.verbose)

    sct.display_viewer_syntax(
        [os.path.join(param.path_out, sct.add_suffix(os.path.basename(param.fname_data), param.suffix)),
         param.fname_data], mode='ortho,ortho')


def moco(param):
    """
    Main function that performs motion correction.
    :param param:
    :return:
    """
    # retrieve parameters
    file_data = param.file_data
    file_target = param.file_target
    folder_mat = param.mat_moco  # output folder of mat file
    todo = param.todo
    suffix = param.suffix
    verbose = param.verbose

    # other parameters
    file_mask = 'mask.nii'

    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  Input file ............ ' + file_data, param.verbose)
    sct.printv('  Reference file ........ ' + file_target, param.verbose)
    sct.printv('  Polynomial degree ..... ' + param.poly, param.verbose)
    sct.printv('  Smoothing kernel ...... ' + param.smooth, param.verbose)
    sct.printv('  Gradient step ......... ' + param.gradStep, param.verbose)
    sct.printv('  Metric ................ ' + param.metric, param.verbose)
    sct.printv('  Sampling .............. ' + param.sampling, param.verbose)
    sct.printv('  Todo .................. ' + todo, param.verbose)
    sct.printv('  Mask  ................. ' + param.fname_mask, param.verbose)
    sct.printv('  Output mat folder ..... ' + folder_mat, param.verbose)

    # create folder for mat files
    sct.create_folder(folder_mat)

    # Get size of data
    sct.printv('\nData dimensions:', verbose)
    im_data = Image(param.file_data)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    sct.printv(('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt)), verbose)

    # copy file_target to a temporary file
    sct.printv('\nCopy file_target to a temporary file...', verbose)
    file_target = "target.nii.gz"
    convert(param.file_target, file_target, verbose=0)

    # Check if user specified a mask
    if not param.fname_mask == '':
        # Check if this mask is soft (i.e., non-binary, such as a Gaussian mask)
        im_mask = Image(param.fname_mask)
        if not np.array_equal(im_mask.data, im_mask.data.astype(bool)):
            # If it is a soft mask, multiply the target by the soft mask.
            im = Image(file_target)
            im_masked = im.copy()
            im_masked.data = im.data * im_mask.data
            im_masked.save(verbose=0)  # silence warning about file overwritting

    # If scan is sagittal, split src and target along Z (slice)
    if param.is_sagittal:
        dim_sag = 2  # TODO: find it
        # z-split data (time series)
        im_z_list = split_data(im_data, dim=dim_sag, squeeze_data=False)
        file_data_splitZ = []
        for im_z in im_z_list:
            im_z.save(verbose=0)
            file_data_splitZ.append(im_z.absolutepath)
        # z-split target
        im_targetz_list = split_data(Image(file_target), dim=dim_sag, squeeze_data=False)
        file_target_splitZ = []
        for im_targetz in im_targetz_list:
            im_targetz.save(verbose=0)
            file_target_splitZ.append(im_targetz.absolutepath)
        # z-split mask (if exists)
        if not param.fname_mask == '':
            im_maskz_list = split_data(Image(file_mask), dim=dim_sag, squeeze_data=False)
            file_mask_splitZ = []
            for im_maskz in im_maskz_list:
                im_maskz.save(verbose=0)
                file_mask_splitZ.append(im_maskz.absolutepath)
        # initialize file list for output matrices
        file_mat = np.empty((nz, nt), dtype=object)

    # axial orientation
    else:
        file_data_splitZ = [file_data]  # TODO: make it absolute like above
        file_target_splitZ = [file_target]  # TODO: make it absolute like above
        # initialize file list for output matrices
        file_mat = np.empty((1, nt), dtype=object)

        # deal with mask
        if not param.fname_mask == '':
            convert(param.fname_mask, file_mask, squeeze_data=False, verbose=0)
            im_maskz_list = [Image(file_mask)]  # use a list with single element

    # Loop across file list, where each file is either a 2D volume (if sagittal) or a 3D volume (otherwise)
    # file_mat = tuple([[[] for i in range(nt)] for i in range(nz)])

    file_data_splitZ_moco = []
    sct.printv('\nRegister. Loop across Z (note: there is only one Z if orientation is axial)')
    for file in file_data_splitZ:
        iz = file_data_splitZ.index(file)
        # Split data along T dimension
        # sct.printv('\nSplit data along T dimension.', verbose)
        im_z = Image(file)
        list_im_zt = split_data(im_z, dim=3)
        file_data_splitZ_splitT = []
        for im_zt in list_im_zt:
            im_zt.save(verbose=0)
            file_data_splitZ_splitT.append(im_zt.absolutepath)
        # file_data_splitT = file_data + '_T'

        # Motion correction: initialization
        index = np.arange(nt)
        file_data_splitT_num = []
        file_data_splitZ_splitT_moco = []
        failed_transfo = [0 for i in range(nt)]

        # Motion correction: Loop across T
        for indice_index in sct_progress_bar(range(nt), unit='iter', unit_scale=False,
                                             desc="Z=" + str(iz) + "/" + str(len(file_data_splitZ)-1), ascii=False, ncols=80):

            # create indices and display stuff
            it = index[indice_index]
            file_mat[iz][it] = os.path.join(folder_mat, "mat.Z") + str(iz).zfill(4) + 'T' + str(it).zfill(4)
            file_data_splitZ_splitT_moco.append(sct.add_suffix(file_data_splitZ_splitT[it], '_moco'))
            # deal with masking (except in the 'apply' case, where masking is irrelevant)
            input_mask = None
            if not param.fname_mask == '' and not param.todo == 'apply':
                # Check if mask is binary
                if np.array_equal(im_maskz_list[iz].data, im_maskz_list[iz].data.astype(bool)):
                    # If it is, pass this mask into register() to be used
                    input_mask = im_maskz_list[iz]
                else:
                    # If not, do not pass this mask into register() because ANTs cannot handle non-binary masks.
                    #  Instead, multiply the input data by the Gaussian mask.
                    im = Image(file_data_splitZ_splitT[it])
                    im_masked = im.copy()
                    im_masked.data = im.data * im_maskz_list[iz].data
                    im_masked.save(verbose=0)  # silence warning about file overwritting

            # run 3D registration
            failed_transfo[it] = register(param, file_data_splitZ_splitT[it], file_target_splitZ[iz], file_mat[iz][it],
                                          file_data_splitZ_splitT_moco[it], im_mask=input_mask)

            # average registered volume with target image
            # N.B. use weighted averaging: (target * nb_it + moco) / (nb_it + 1)
            if param.iterAvg and indice_index < 10 and failed_transfo[it] == 0 and not param.todo == 'apply':
                im_targetz = Image(file_target_splitZ[iz])
                data_targetz = im_targetz.data
                data_mocoz = Image(file_data_splitZ_splitT_moco[it]).data
                data_targetz = (data_targetz * (indice_index + 1) + data_mocoz) / (indice_index + 2)
                im_targetz.data = data_targetz
                im_targetz.save(verbose=0)

        # Replace failed transformation with the closest good one
        fT = [i for i, j in enumerate(failed_transfo) if j == 1]
        gT = [i for i, j in enumerate(failed_transfo) if j == 0]
        for it in range(len(fT)):
            abs_dist = [np.abs(gT[i] - fT[it]) for i in range(len(gT))]
            if not abs_dist == []:
                index_good = abs_dist.index(min(abs_dist))
                sct.printv('  transfo #' + str(fT[it]) + ' --> use transfo #' + str(gT[index_good]), verbose)
                # copy transformation
                sct.copy(file_mat[iz][gT[index_good]] + 'Warp.nii.gz', file_mat[iz][fT[it]] + 'Warp.nii.gz')
                # apply transformation
                sct_apply_transfo.main(args=['-i', file_data_splitZ_splitT[fT[it]],
                                             '-d', file_target,
                                             '-w', file_mat[iz][fT[it]] + 'Warp.nii.gz',
                                             '-o', file_data_splitZ_splitT_moco[fT[it]],
                                             '-x', param.interp])
            else:
                # exit program if no transformation exists.
                sct.printv('\nERROR in ' + os.path.basename(__file__) + ': No good transformation exist. Exit program.\n', verbose, 'error')
                sys.exit(2)

        # Merge data along T
        file_data_splitZ_moco.append(sct.add_suffix(file, suffix))
        if todo != 'estimate':
            im_out = concat_data(file_data_splitZ_splitT_moco, 3)
            im_out.absolutepath = file_data_splitZ_moco[iz]
            im_out.save(verbose=0)

    # If sagittal, merge along Z
    if param.is_sagittal:
        # TODO: im_out.dim is incorrect: Z value is one
        im_out = concat_data(file_data_splitZ_moco, 2)
        dirname, basename, ext = sct.extract_fname(file_data)
        path_out = os.path.join(dirname, basename + suffix + ext)
        im_out.absolutepath = path_out
        im_out.save(verbose=0)

    return file_mat, im_out


def register(param, file_src, file_dest, file_mat, file_out, im_mask=None):
    """
    Register two images by estimating slice-wise Tx and Ty transformations, which are regularized along Z. This function
    uses ANTs' isct_antsSliceRegularizedRegistration.
    :param param:
    :param file_src:
    :param file_dest:
    :param file_mat:
    :param file_out:
    :param im_mask: Image of mask, could be 2D or 3D
    :return:
    """

    # TODO: deal with mask

    # initialization
    failed_transfo = 0  # by default, failed matrix is 0 (i.e., no failure)
    do_registration = True

    # get metric radius (if MeanSquares, CC) or nb bins (if MI)
    if param.metric == 'MI':
        metric_radius = '16'
    else:
        metric_radius = '4'
    file_out_concat = file_out

    kw = dict()
    im_data = Image(file_src)  # TODO: pass argument to use antsReg instead of opening Image each time

    # register file_src to file_dest
    if param.todo == 'estimate' or param.todo == 'estimate_and_apply':
        # If orientation is sagittal, use antsRegistration in 2D mode
        # Note: the parameter --restrict-deformation is irrelevant with affine transfo

        if param.sampling == 'None':
            # 'None' sampling means 'fully dense' sampling
            # see https://github.com/ANTsX/ANTs/wiki/antsRegistration-reproducibility-issues
            sampling = param.sampling
        else:
            # param.sampling should be a float in [0,1], and means the
            # samplingPercentage that chooses a subset of points to
            # estimate from. We always use 'Regular' (evenly-spaced)
            # mode, though antsRegistration offers 'Random' as well.
            # Be aware: even 'Regular' is not fully deterministic:
            # > Regular includes a random perturbation on the grid sampling
            # - https://github.com/ANTsX/ANTs/issues/976#issuecomment-602313884
            sampling = 'Regular,' + param.sampling

        if im_data.orientation[2] in 'LR':
            cmd = ['isct_antsRegistration',
                   '-d', '2',
                   '--transform', 'Affine[%s]' %param.gradStep,
                   '--metric', param.metric + '[' + file_dest + ',' + file_src + ',1,' + metric_radius + ',' + sampling + ']',
                   '--convergence', param.iter,
                   '--shrink-factors', '1',
                   '--smoothing-sigmas', param.smooth,
                   '--verbose', '1',
                   '--output', '[' + file_mat + ',' + file_out_concat + ']']
            cmd += sct.get_interpolation('isct_antsRegistration', param.interp)
            if im_mask is not None:
                # if user specified a mask, make sure there are non-null voxels in the image before running the registration
                if np.count_nonzero(im_mask.data):
                    cmd += ['--masks', im_mask.absolutepath]
                else:
                    # Mask only contains zeros. Copying the image instead of estimating registration.
                    sct.copy(file_src, file_out_concat, verbose=0)
                    do_registration = False
                    # TODO: create affine mat file with identity, in case used by -g 2
        # 3D mode
        else:
            cmd = ['isct_antsSliceRegularizedRegistration',
                   '--polydegree', param.poly,
                   '--transform', 'Translation[%s]' %param.gradStep,
                   '--metric', param.metric + '[' + file_dest + ',' + file_src + ',1,' + metric_radius + ',' + sampling + ']',
                   '--iterations', param.iter,
                   '--shrinkFactors', '1',
                   '--smoothingSigmas', param.smooth,
                   '--verbose', '1',
                   '--output', '[' + file_mat + ',' + file_out_concat + ']']
            cmd += sct.get_interpolation('isct_antsSliceRegularizedRegistration', param.interp)
            if im_mask is not None:
                cmd += ['--mask', im_mask.absolutepath]
        # run command
        if do_registration:
            kw.update(dict(is_sct_binary=True))
            # reducing the number of CPU used for moco (see issue #201 and #2642)
            env = {**os.environ, **{"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}}
            status, output = sct.run(cmd, verbose=1 if param.verbose == 2 else 0, env=env, **kw)

    elif param.todo == 'apply':
        sct_apply_transfo.main(args=['-i', file_src,
                                     '-d', file_dest,
                                     '-w', file_mat + param.suffix_mat,
                                     '-o', file_out_concat,
                                     '-x', param.interp,
                                     '-v', '0'])

    # check if output file exists
    # Note (from JCA): In the past, i've tried to catch non-zero output from ANTs function (via the 'status' variable),
    # but in some OSs, the function can fail while outputing zero. So as a pragmatic approach, I decided to go with
    # the "output file checking" approach, which is 100% sensitive.
    if not os.path.isfile(file_out_concat):
        # sct.printv(output, verbose, 'error')
        sct.printv('WARNING in ' + os.path.basename(__file__) + ': No output. Maybe related to improper calculation of '
                                                                'mutual information. Either the mask you provided is '
                                                                'too small, or the subject moved a lot. If you see too '
                                                                'many messages like this try with a bigger mask. '
                                                                'Using previous transformation for this volume (if it'
                                                                'exists).', param.verbose, 'warning')
        failed_transfo = 1

    # If sagittal, copy header (because ANTs screws it) and add singleton in 3rd dimension (for z-concatenation)
    if im_data.orientation[2] in 'LR' and do_registration:
        im_out = Image(file_out_concat)
        im_out.header = im_data.header
        im_out.data = np.expand_dims(im_out.data, 2)
        im_out.save(file_out, verbose=0)

    # return status of failure
    return failed_transfo


def spline(folder_mat, nt, nz, verbose, index_b0 = [], graph=0):

    sct.printv('\n\n\n------------------------------------------------------------------------------', verbose)
    sct.printv('Spline Regularization along T: Smoothing Patient Motion...', verbose)

    file_mat = [[[] for i in range(nz)] for i in range(nt)]
    for it in range(nt):
        for iz in range(nz):
            file_mat[it][iz] = os.path.join(folder_mat, "mat.T") + str(it) + '_Z' + str(iz) + '.txt'

    # Copying the existing Matrices to another folder
    old_mat = os.path.join(folder_mat, "old")
    if not os.path.exists(old_mat):
        os.makedirs(old_mat)
    # TODO
    for mat in glob.glob(os.path.join(folder_mat, '*.txt')):
        sct.copy(mat, old_mat)

    sct.printv('\nloading matrices...', verbose)
    X = [[[] for i in range(nt)] for i in range(nz)]
    Y = [[[] for i in range(nt)] for i in range(nz)]
    X_smooth = [[[] for i in range(nt)] for i in range(nz)]
    Y_smooth = [[[] for i in range(nt)] for i in range(nz)]
    for iz in range(nz):
        for it in range(nt):
            file =  open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            X[iz][it] = Matrix[0, 3]
            Y[iz][it] = Matrix[1, 3]

    # Generate motion splines
    sct.printv('\nGenerate motion splines...', verbose)
    T = np.arange(nt)
    if graph:
        import pylab as pl

    for iz in range(nz):

        spline = scipy.interpolate.UnivariateSpline(T, X[iz][:], w=None, bbox=[None, None], k=3, s=None)
        X_smooth[iz][:] = spline(T)

        if graph:
            pl.plot(T, X_smooth[iz][:], label='spline_smoothing')
            pl.plot(T, X[iz][:], marker='*', linestyle='None', label='original_val')
            if len(index_b0) != 0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                X_b0 = [X[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0, X_b0, marker='D', linestyle='None', color='k', label='b=0')
            pl.title('X')
            pl.grid()
            pl.legend()
            pl.show()

        spline = scipy.interpolate.UnivariateSpline(T, Y[iz][:], w=None, bbox=[None, None], k=3, s=None)
        Y_smooth[iz][:] = spline(T)

        if graph:
            pl.plot(T, Y_smooth[iz][:], label='spline_smoothing')
            pl.plot(T, Y[iz][:], marker='*', linestyle='None', label='original_val')
            if len(index_b0) != 0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                Y_b0 = [Y[iz][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0, Y_b0, marker='D', linestyle='None', color='k', label='b=0')
            pl.title('Y')
            pl.grid()
            pl.legend()
            pl.show()

    # Storing the final Matrices
    sct.printv('\nStoring the final Matrices...', verbose)
    for iz in range(nz):
        for it in range(nt):
            file =  open(file_mat[it][iz])
            Matrix = np.loadtxt(file)
            file.close()

            Matrix[0, 3] = X_smooth[iz][it]
            Matrix[1, 3] = Y_smooth[iz][it]

            file =  open(file_mat[it][iz], 'w')
            np.savetxt(file_mat[it][iz], Matrix, fmt="%s", delimiter='  ', newline='\n')
            file.close()

    sct.printv('\n...Done. Patient motion has been smoothed', verbose)
    sct.printv('------------------------------------------------------------------------------\n', verbose)
