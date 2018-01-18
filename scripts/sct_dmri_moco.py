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

# TODO: Do not merge per group if no group is asked.
# TODO: make sure slicewise not used with ants, eddy not used with ants
# TODO: make sure images are axial
# TDOD: if -f, we only need two plots. Plot 1: X params with fitted spline, plot 2: Y param with fitted splines. Each plot will have all Z slices (with legend Z=0, Z=1, ...) and labels: y; translation (mm), xlabel: volume #. Plus add grid.
# TODO (no priority): for sinc interp, use ANTs instead of flirt

import sys
import os

import time
import math
import numpy as np
import sct_utils as sct
import msct_moco as moco
from sct_dmri_separate_b0_and_dwi import identify_b0
import importlib
from sct_convert import convert
from msct_image import Image
from sct_image import copy_header, split_data, concat_data
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
        self.group_size = 3  # number of images averaged for 'dwi' method.
        self.spline_fitting = 0
        self.remove_tmp_files = 1
        self.verbose = 1
        self.plot_graph = 0
        self.suffix = '_moco'
        self.poly = '2'  # degree of polynomial function for moco
        self.smooth = '2'  # smoothing sigma in mm
        self.gradStep = '1'  # gradientStep for searching algorithm
        self.metric = 'MI'  # metric: MI, MeanSquares, CC
        self.sampling = '0.2'  # sampling rate used for registration metric
        self.interp = 'spline'  # nn, linear, spline
        self.run_eddy = 0
        self.mat_eddy = ''
        self.min_norm = 0.001
        self.swapXY = 0
        self.bval_min = 100  # in case user does not have min bvalues at 0, set threshold (where csf disapeared).
        self.otsu = 0  # use otsu algorithm to segment dwi data for better moco. Value coresponds to data threshold. For no segmentation set to 0.
        self.iterative_averaging = 1  # iteratively average target image for more robust moco

    # update constructor with user's parameters
    def update(self, param_user):
        # list_objects = param_user.split(',')
        for object in param_user:
            if len(object) < 2:
                sct.printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # initialize parameters
    param_default = Param()

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
    parser.add_option(name='-b',
                      type_value=None,
                      description='Bvecs file',
                      mandatory=False,
                      deprecated_by='-bvec')
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
    parser.add_option(name='-a',
                      type_value=None,
                      description='Bvals file',
                      mandatory=False,
                      deprecated_by='-bval')

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
    parser.add_option(name='-thr',
                      type_value='float',
                      description='Segment DW data using OTSU algorithm. Value corresponds to OTSU threshold. For no segmentation set to 0.',
                      mandatory=False,
                      default_value=param_default.otsu,
                      example=['25'])
    parser.add_option(name='-t',
                      type_value=None,
                      description='Segment DW data using OTSU algorithm. Value corresponds to OTSU threshold. For no segmentation set to 0.',
                      mandatory=False,
                      deprecated_by='-thr')
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
    parser.add_option(name='-o',
                      type_value=None,
                      description='Output folder.',
                      mandatory=False,
                      deprecated_by='-o')
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
    param = Param()

    # reducing the number of CPU used for moco (see issue #201)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

    # get path of the toolbox
    # status, param.path_sct = sct.run('echo $SCT_DIR')

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
    if '-thr' in arguments:
        param.otsu = arguments['-thr']
    if '-x' in arguments:
        param.interp = arguments['-x']
    if '-ofolder' in arguments:
        path_out = arguments['-ofolder']
    if '-r' in arguments:
        param.remove_tmp_files = int(arguments['-r'])
    if '-v' in arguments:
        param.verbose = int(arguments['-v'])

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
    ext = '.nii'
    dmri_name = 'dmri'
    mask_name = 'mask'
    bvecs_fname = 'bvecs.txt'

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(param.fname_data, os.path.join(path_tmp, dmri_name + ext))
    sct.run('cp ' + param.fname_bvecs + ' ' + os.path.join(path_tmp, bvecs_fname), param.verbose)
    if param.fname_mask != '':
        sct.run('cp ' + param.fname_mask + ' ' + os.path.join(path_tmp, mask_name + ext_mask), param.verbose)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # update field in param (because used later).
    # TODO: make this cleaner...
    if param.fname_mask != '':
        param.fname_mask = mask_name + ext_mask

    # run moco
    dmri_moco(param)

    # come back
    os.chdir(curdir)

    # Generate output files
    fname_dmri_moco = os.path.join(path_out, file_data + param.suffix + ext_data)
    sct.create_folder(path_out)
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(os.path.join(path_tmp, dmri_name + param.suffix + ext), os.path.join(path_out, file_data + param.suffix + ext_data), param.verbose)
    sct.generate_output_file(os.path.join(path_tmp, "b0_mean.nii"), os.path.join(path_out, 'b0' + param.suffix + '_mean' + ext_data), param.verbose)
    sct.generate_output_file(os.path.join(path_tmp, "dwi_mean.nii"), os.path.join(path_out, 'dwi' + param.suffix + '_mean' + ext_data), param.verbose)

    # Delete temporary files
    if param.remove_tmp_files == 1:
        sct.printv('\nDelete temporary files...', param.verbose)
        sct.run('rm -rf ' + path_tmp, param.verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's', param.verbose)

    sct.display_viewer_syntax([fname_dmri_moco, file_data], mode='ortho,ortho')


#=======================================================================================================================
# dmri_moco: motion correction specific to dmri data
#=======================================================================================================================
def dmri_moco(param):

    file_data = 'dmri'
    ext_data = '.nii'
    file_b0 = 'b0'
    file_dwi = 'dwi'
    mat_final = 'mat_final/'
    file_dwi_group = 'dwi_averaged_groups'  # no extension
    ext_mat = 'Warp.nii.gz'  # warping field

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    im_data = Image(file_data + ext_data)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), param.verbose)

    # Identify b=0 and DWI images
    sct.printv('\nIdentify b=0 and DWI images...', param.verbose)
    index_b0, index_dwi, nb_b0, nb_dwi = identify_b0('bvecs.txt', param.fname_bvals, param.bval_min, param.verbose)

    # check if dmri and bvecs are the same size
    if not nb_b0 + nb_dwi == nt:
        sct.printv('\nERROR in ' + os.path.basename(__file__) + ': Size of data (' + str(nt) + ') and size of bvecs (' + str(nb_b0 + nb_dwi) + ') are not the same. Check your bvecs file.\n', 1, 'error')
        sys.exit(2)

    # Prepare NIFTI (mean/groups...)
    #===================================================================================================================
    # Split into T dimension
    sct.printv('\nSplit along T dimension...', param.verbose)
    im_data_split_list = split_data(im_data, 3)
    for im in im_data_split_list:
        im.save()

    # Merge b=0 images
    sct.printv('\nMerge b=0...', param.verbose)
    im_b0_list = []
    for it in range(nb_b0):
        im_b0_list.append(im_data_split_list[index_b0[it]])
    im_b0_out = concat_data(im_b0_list, 3)
    im_b0_out.setFileName(file_b0 + ext_data)
    im_b0_out.save()
    sct.printv(('  File created: ' + file_b0), param.verbose)

    # Average b=0 images
    sct.printv('\nAverage b=0...', param.verbose)
    file_b0_mean = file_b0 + '_mean'
    sct.run('sct_maths -i ' + file_b0 + ext_data + ' -o ' + file_b0_mean + ext_data + ' -mean t', param.verbose)

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

    # DWI groups
    file_dwi_mean = []
    for iGroup in range(nb_groups):
        sct.printv('\nDWI group: ' + str((iGroup + 1)) + '/' + str(nb_groups), param.verbose)

        # get index
        index_dwi_i = group_indexes[iGroup]
        nb_dwi_i = len(index_dwi_i)

        # Merge DW Images
        sct.printv('Merge DW images...', param.verbose)
        file_dwi_merge_i = file_dwi + '_' + str(iGroup)

        im_dwi_list = []
        for it in range(nb_dwi_i):
            im_dwi_list.append(im_data_split_list[index_dwi_i[it]])
        im_dwi_out = concat_data(im_dwi_list, 3)
        im_dwi_out.setFileName(file_dwi_merge_i + ext_data)
        im_dwi_out.save()

        # Average DW Images
        sct.printv('Average DW images...', param.verbose)
        file_dwi_mean.append(file_dwi + '_mean_' + str(iGroup))
        sct.run('sct_maths -i ' + file_dwi_merge_i + ext_data + ' -o ' + file_dwi_mean[iGroup] + ext_data + ' -mean t', param.verbose)

    # Merge DWI groups means
    sct.printv('\nMerging DW files...', param.verbose)
    # file_dwi_groups_means_merge = 'dwi_averaged_groups'
    im_dw_list = []
    for iGroup in range(nb_groups):
        im_dw_list.append(file_dwi_mean[iGroup] + ext_data)
    im_dw_out = concat_data(im_dw_list, 3)
    im_dw_out.setFileName(file_dwi_group + ext_data)
    im_dw_out.save()

    # Average DW Images
    # TODO: USEFULL ???
    sct.printv('\nAveraging all DW images...', param.verbose)
    sct.run('sct_maths -i ' + file_dwi_group + ext_data + ' -o ' + file_dwi_group + '_mean' + ext_data + ' -mean t', param.verbose)

    # segment dwi images using otsu algorithm
    if param.otsu:
        sct.printv('\nSegment group DWI using OTSU algorithm...', param.verbose)
        # import module
        otsu = importlib.import_module('sct_otsu')
        # get class from module
        param_otsu = otsu.param()  #getattr(otsu, param)
        param_otsu.fname_data = file_dwi_group + ext_data
        param_otsu.threshold = param.otsu
        param_otsu.file_suffix = '_seg'
        # run otsu
        otsu.otsu(param_otsu)
        file_dwi_group = file_dwi_group + '_seg'

    # START MOCO
    #===================================================================================================================

    # Estimate moco on b0 groups
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion on b=0 images...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco = param
    param_moco.file_data = 'b0'
    # identify target image
    if index_dwi[0] != 0:
        # If first DWI is not the first volume (most common), then there is a least one b=0 image before. In that case
        # select it as the target image for registration of all b=0
        param_moco.file_target = file_data + '_T' + str(index_b0[index_dwi[0] - 1]).zfill(4)
    else:
        # If first DWI is the first volume, then the target b=0 is the first b=0 from the index_b0.
        param_moco.file_target = file_data + '_T' + str(index_b0[0]).zfill(4)
    param_moco.path_out = ''
    param_moco.todo = 'estimate'
    param_moco.mat_moco = 'mat_b0groups'
    moco.moco(param_moco)

    # Estimate moco on dwi groups
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion on DW images...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_dwi_group
    param_moco.file_target = file_dwi_mean[0]  # target is the first DW image (closest to the first b=0)
    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_dwigroups'
    moco.moco(param_moco)

    # create final mat folder
    sct.create_folder(mat_final)

    # Copy b=0 registration matrices
    sct.printv('\nCopy b=0 registration matrices...', param.verbose)

    for it in range(nb_b0):
        sct.run('cp ' + 'mat_b0groups/' + 'mat.T' + str(it) + ext_mat + ' ' + mat_final + 'mat.T' + str(index_b0[it]) + ext_mat, param.verbose)

    # Copy DWI registration matrices
    sct.printv('\nCopy DWI registration matrices...', param.verbose)
    for iGroup in range(nb_groups):
        for dwi in range(len(group_indexes[iGroup])):
            sct.run('cp ' + 'mat_dwigroups/' + 'mat.T' + str(iGroup) + ext_mat + ' ' + mat_final + 'mat.T' + str(group_indexes[iGroup][dwi]) + ext_mat, param.verbose)

    # Spline Regularization along T
    if param.spline_fitting:
        moco.spline(mat_final, nt, nz, param.verbose, np.array(index_b0), param.plot_graph)

    # combine Eddy Matrices
    if param.run_eddy:
        param.mat_2_combine = 'mat_eddy'
        param.mat_final = mat_final
        moco.combine_matrix(param)

    # Apply moco on all dmri data
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Apply moco', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = file_data
    param_moco.file_target = file_dwi + '_mean_' + str(0)  # reference for reslicing into proper coordinate system
    param_moco.path_out = ''
    param_moco.mat_moco = mat_final
    param_moco.todo = 'apply'
    moco.moco(param_moco)

    # copy geometric information from header
    # NB: this is required because WarpImageMultiTransform in 2D mode wrongly sets pixdim(3) to "1".
    im_dmri = Image(file_data + ext_data)
    im_dmri_moco = Image(file_data + param.suffix + ext_data)
    im_dmri_moco = copy_header(im_dmri, im_dmri_moco)
    im_dmri_moco.save()

    # generate b0_moco_mean and dwi_moco_mean
    cmd = 'sct_dmri_separate_b0_and_dwi -i ' + file_data + param.suffix + ext_data + ' -bvec bvecs.txt -a 1'
    if not param.fname_bvals == '':
        cmd = cmd + ' -m ' + param.fname_bvals
    sct.run(cmd, param.verbose)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    main()
