#!/usr/bin/env python
#########################################################################################
#
# Motion correction of fMRI data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


from __future__ import absolute_import, division

import sys
import os
import time
import math
from tqdm import tqdm
import numpy as np
import sct_utils as sct
import msct_moco as moco
import sct_maths
from sct_convert import convert
from spinalcordtoolbox.image import Image
from sct_image import split_data, concat_data
from msct_parser import Parser


# PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_bvecs = ''
        self.fname_bvals = ''
        self.fname_target = ''
        self.fname_mask = ''
        self.mat_final = ''
        self.todo = ''
        self.group_size = 1  # number of images averaged for 'dwi' method.
        self.spline_fitting = 0
        self.remove_temp_files = 1
        self.verbose = 1
        self.plot_graph = 0
        self.suffix = '_moco'
        self.poly = '2'  # degree of polynomial function for moco
        self.smooth = '0'  # smoothing sigma in mm
        self.gradStep = '1'  # gradientStep for searching algorithm
        self.iter = '10'  # number of iterations
        self.metric = 'MeanSquares'  # metric: MI, MeanSquares, CC
        self.sampling = '0.2'  # sampling rate used for registration metric
        self.interp = 'spline'  # nn, linear, spline
        self.run_eddy = 0
        self.mat_eddy = ''
        self.min_norm = 0.001
        self.swapXY = 0
        self.bval_min = 100  # in case user does not have min bvalues at 0, set threshold (where csf disapeared).
        self.otsu = 0  # use otsu algorithm to segment dwi data for better moco. Value coresponds to data threshold. For no segmentation set to 0.
        self.iterAvg = 1  # iteratively average target image for more robust moco
        self.num_target = '0'
        self.is_sagittal = False  # if True, then split along Z (right-left) and register each 2D slice (vs. 3D volume)

    # update constructor with user's parameters
    def update(self, param_user):
        # list_objects = param_user.split(',')
        for object in param_user:
            if len(object) < 2:
                sct.printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            # set type based on default param field
            # isinstance(i, int)
            # isinstance(f, float)
            # isinstance(s, str)

            field = obj[1]
            if isinstance(getattr(self, obj[0]), int):
                field = int(field)
            setattr(self, obj[0], field)

# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # initialize parameters
    param_default = Param()

    parser.usage.set_description("""Motion correction of fMRI data. Some robust features include:
  - group-wise (-g)
  - slice-wise regularized along z using polynomial function (-p)
    For more info about the method, type: isct_antsSliceRegularizedRegistration
  - masking (-m)
  - iterative averaging of target volume""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='4D data',
                      mandatory=True,
                      example='fmri.nii.gz')
    parser.add_option(name='-g',
                      type_value='int',
                      description='Group nvols successive fMRI volumes for more robustness.',
                      mandatory=False,
                      default_value=param_default.group_size)
    parser.add_option(name='-m',
                      type_value='image_nifti',
                      description='Binary mask to limit voxels considered by the registration metric.',
                      mandatory=False)
    parser.add_option(name='-param',
                      type_value=[[','], 'str'],
                      description="Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
                                  "poly [int]: Degree of polynomial function used for regularization along Z. For no regularization set to 0. Default=" + param_default.poly + ".\n"
                                  "smooth [mm]: Smoothing kernel. Default=" + param_default.smooth + ".\n"
                                  "iter [int]: Number of iterations. Default=" + param_default.iter + ".\n"
                                  "metric {MI, MeanSquares, CC}: Metric used for registration. Default=" + param_default.metric + ".\n"
                                  "gradStep [float]: Searching step used by registration algorithm. The higher the more deformation allowed. Default=" + param_default.gradStep + ".\n"
                                  "sampling [0-1]: Sampling rate used for registration metric. Default=" + param_default.sampling + ".\n"
                                  "numTarget [int]: Target volume or group (starting with 0). Default=" + param_default.num_target + ".\n"
                                  "iterAvg [int]: Iterative averaging: Target volume is a weighted average of the previously-registered volumes. Default=" + str(param_default.iterAvg) + ".\n",
                      mandatory=False)
    parser.add_option(name='-ofolder',
                      type_value='folder_creation',
                      description='Output path.',
                      mandatory=False,
                      default_value='./')
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

    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # initialization
    start_time = time.time()
    param = Param()

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    param.fname_data = arguments['-i']
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
    if '-v' in arguments:
        param.verbose = int(arguments['-v'])

    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  input file ............' + param.fname_data, param.verbose)

    # Get full path
    param.fname_data = os.path.abspath(param.fname_data)
    if param.fname_mask != '':
        param.fname_mask = os.path.abspath(param.fname_mask)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)

    path_tmp = sct.tmp_create(basename="fmri_moco", verbose=param.verbose)

    # Copying input data to tmp folder and convert to nii
    # TODO: no need to do that (takes time for nothing)
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(param.fname_data, os.path.join(path_tmp, "fmri.nii"), squeeze_data=False)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # run moco
    fmri_moco(param)

    # come back
    os.chdir(curdir)

    # Generate output files
    fname_fmri_moco = os.path.join(path_out, file_data + param.suffix + ext_data)
    sct.create_folder(path_out)
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(os.path.join(path_tmp, "fmri" + param.suffix + '.nii'), fname_fmri_moco, param.verbose)
    sct.generate_output_file(os.path.join(path_tmp, "fmri" + param.suffix + '_mean.nii'), os.path.join(path_out, file_data + param.suffix + '_mean' + ext_data), param.verbose)

    # Delete temporary files
    if param.remove_temp_files == 1:
        sct.printv('\nDelete temporary files...', param.verbose)
        sct.rmtree(path_tmp, verbose=param.verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's', param.verbose)

    sct.display_viewer_syntax([fname_fmri_moco, file_data], mode='ortho,ortho')


def fmri_moco(param):

    file_data = "fmri.nii"
    mat_final = 'mat_final/'
    ext_mat = 'Warp.nii.gz'  # warping field

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    im_data = Image(param.fname_data)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), param.verbose)

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

    # Split into T dimension
    sct.printv('\nSplit along T dimension...', param.verbose)
    im_data_split_list = split_data(im_data, 3)
    for im in im_data_split_list:
        x_dirname, x_basename, x_ext = sct.extract_fname(im.absolutepath)
        # Make further steps slurp the data to avoid too many open files (#2149)
        im.absolutepath = os.path.join(x_dirname, x_basename + ".nii.gz")
        im.save()

    # assign an index to each volume
    index_fmri = list(range(0, nt))

    # Number of groups
    nb_groups = int(math.floor(nt / param.group_size))

    # Generate groups indexes
    group_indexes = []
    for iGroup in range(nb_groups):
        group_indexes.append(index_fmri[(iGroup * param.group_size):((iGroup + 1) * param.group_size)])

    # add the remaining images to the last fMRI group
    nb_remaining = nt%param.group_size  # number of remaining images
    if nb_remaining > 0:
        nb_groups += 1
        group_indexes.append(index_fmri[len(index_fmri) - nb_remaining:len(index_fmri)])

    # groups
    for iGroup in tqdm(range(nb_groups), unit='iter', unit_scale=False, desc="Merge within groups", ascii=True, ncols=80):
        # get index
        index_fmri_i = group_indexes[iGroup]
        nt_i = len(index_fmri_i)

        # Merge Images
        file_data_merge_i = sct.add_suffix(file_data, '_' + str(iGroup))
        # cmd = fsloutput + 'fslmerge -t ' + file_data_merge_i
        # for it in range(nt_i):
        #     cmd = cmd + ' ' + file_data + '_T' + str(index_fmri_i[it]).zfill(4)

        im_fmri_list = []
        for it in range(nt_i):
            im_fmri_list.append(im_data_split_list[index_fmri_i[it]])
        im_fmri_concat = concat_data(im_fmri_list, 3, squeeze_data=True).save(file_data_merge_i)

        file_data_mean = sct.add_suffix(file_data, '_mean_' + str(iGroup))
        if file_data_mean.endswith(".nii"):
            file_data_mean += ".gz" # #2149
        if param.group_size == 1:
            # copy to new file name instead of averaging (faster)
            # note: this is a bandage. Ideally we should skip this entire for loop if g=1
            convert(file_data_merge_i, file_data_mean)
        else:
            # Average Images
            sct.run(['sct_maths', '-i', file_data_merge_i, '-o', file_data_mean, '-mean', 't'], verbose=0)
        # if not average_data_across_dimension(file_data_merge_i+'.nii', file_data_mean+'.nii', 3):
        #     sct.printv('ERROR in average_data_across_dimension', 1, 'error')
        # cmd = fsloutput + 'fslmaths ' + file_data_merge_i + ' -Tmean ' + file_data_mean
        # sct.run(cmd, param.verbose)

    # Merge groups means. The output 4D volume will be used for motion correction.
    sct.printv('\nMerging volumes...', param.verbose)
    file_data_groups_means_merge = 'fmri_averaged_groups.nii'
    im_mean_list = []
    for iGroup in range(nb_groups):
        file_data_mean = sct.add_suffix(file_data, '_mean_' + str(iGroup))
        if file_data_mean.endswith(".nii"):
            file_data_mean += ".gz" # #2149
        im_mean_list.append(Image(file_data_mean))
    im_mean_concat = concat_data(im_mean_list, 3).save(file_data_groups_means_merge)

    # Estimate moco
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco = param
    param_moco.file_data = 'fmri_averaged_groups.nii'
    param_moco.file_target = sct.add_suffix(file_data, '_mean_' + param.num_target)
    if param_moco.file_target.endswith(".nii"):
        param_moco.file_target += ".gz" # #2149
    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_groups'
    file_mat = moco.moco(param_moco)

    # TODO: if g=1, no need to run the block below (already applied)
    if param.group_size == 1:
        # if flag g=1, it means that all images have already been corrected, so we just need to rename the file
        sct.mv('fmri_averaged_groups_moco.nii', 'fmri_moco.nii')
    else:
        # create final mat folder
        sct.create_folder(mat_final)

        # Copy registration matrices
        sct.printv('\nCopy transformations...', param.verbose)
        for iGroup in range(nb_groups):
            for data in range(len(group_indexes[iGroup])):  # we cannot use enumerate because group_indexes has 2 dim.
                # fetch all file_mat_z for given t-group
                list_file_mat_z = file_mat[:, iGroup]
                # loop across file_mat_z and copy to mat_final folder
                for file_mat_z in list_file_mat_z:
                    # we want to copy 'mat_groups/mat.ZXXXXTYYYYWarp.nii.gz' --> 'mat_final/mat.ZXXXXTYYYZWarp.nii.gz'
                    # Notice the Y->Z in the under the T index: the idea here is to use the single matrix from each group,
                    # and apply it to all images belonging to the same group.
                    sct.copy(file_mat_z + ext_mat,
                             mat_final + file_mat_z[11:20] + 'T' + str(group_indexes[iGroup][data]).zfill(4) + ext_mat)

        # Apply moco on all fmri data
        sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
        sct.printv('  Apply moco', param.verbose)
        sct.printv('-------------------------------------------------------------------------------', param.verbose)
        param_moco.file_data = 'fmri.nii'
        param_moco.file_target = sct.add_suffix(file_data, '_mean_' + str(0))
        if param_moco.file_target.endswith(".nii"):
            param_moco.file_target += ".gz"
        param_moco.path_out = ''
        param_moco.mat_moco = mat_final
        param_moco.todo = 'apply'
        moco.moco(param_moco)

    # copy geometric information from header
    # NB: this is required because WarpImageMultiTransform in 2D mode wrongly sets pixdim(3) to "1".
    im_fmri = Image('fmri.nii')
    im_fmri_moco = Image('fmri_moco.nii')
    im_fmri_moco.header = im_fmri.header
    im_fmri_moco.save()

    # Average volumes
    sct.printv('\nAveraging data...', param.verbose)
    sct_maths.main(args=['-i', 'fmri_moco.nii',
                         '-o', 'fmri_moco_mean.nii',
                         '-mean', 't',
                         '-v', '0'])


if __name__ == "__main__":
    sct.init_sct()
    main()
