#!/usr/bin/env python
#########################################################################################
#
# Motion correction of fMRI data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-10-06
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
import commands
import getopt
import time
import math
import sct_utils as sct
import msct_moco as moco
from sct_convert import convert
from msct_image import Image
from sct_image import copy_header, split_data, concat_data
# from sct_average_data_across_dimension import average_data_across_dimension
from msct_parser import Parser


class Param:
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_target = ''
        self.fname_mask = ''
        self.mat_final = ''
        self.num_target = 0  # target volume (or group) for moco
        self.todo = ''
        self.group_size = 1  # number of consecutive images averaged
        self.remove_tmp_files = 1
        self.verbose = 1
        self.suffix = '_moco'
        self.param = ['2',  # degree of polynomial function for moco
                      '2',  # smoothing sigma in mm
                      '1',  # gradientStep
                      'MeanSquares'] # metric: MI,MeanSquares
        self.interp = 'spline'  # nn, linear, spline
        self.min_norm = 0.001
        self.iterative_averaging = 1  # iteratively average target image for more robust moco


#=======================================================================================================================
# main
#=======================================================================================================================
def main(path_out, param_user):

    # initialization
    start_time = time.time()

    # reducing the number of CPU used for moco (see issue #201)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/fmri/fmri.nii.gz'
        #param.fname_mask = path_sct_data+'/fmri/fmri.nii.gz'
        param.verbose = 1
        param.group_size = 3
        #param_user = '2,1,0.5'

    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  input file ............'+param.fname_data, param.verbose)

    # Get full path
    param.fname_data = os.path.abspath(param.fname_data)
    if param.fname_mask != '':
        param.fname_mask = os.path.abspath(param.fname_mask)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', param.verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    convert(param.fname_data, path_tmp+'fmri.nii')
    # sct.run('cp '+param.fname_data+' '+path_tmp+'fmri'+ext_data, param.verbose)
    #
    # go to tmp folder
    os.chdir(path_tmp)
    #
    # # convert fmri to nii format
    # convert('fmri'+ext_data, 'fmri.nii')

    # run moco
    fmri_moco(param)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    path_out = sct.slash_at_the_end(path_out, 1)
    sct.create_folder(path_out)
    sct.printv('\nGenerate output files...', param.verbose)
    if os.path.isfile(path_tmp+'fmri'+param.suffix+'.nii'):
        print path_tmp+'fmri'+param.suffix+'.nii'
        print path_out+file_data+param.suffix+ext_data
    sct.generate_output_file(path_tmp+'fmri'+param.suffix+'.nii', path_out+file_data+param.suffix+ext_data, param.verbose)
    sct.generate_output_file(path_tmp+'fmri'+param.suffix+'_mean.nii', path_out+file_data+param.suffix+'_mean'+ext_data, param.verbose)

    # Delete temporary files
    if param.remove_tmp_files == 1:
        sct.printv('\nDelete temporary files...', param.verbose)
        sct.run('rm -rf '+path_tmp, param.verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s', param.verbose)

    #To view results
    sct.printv('\nTo view results, type:', param.verbose)
    sct.printv('fslview -m ortho,ortho '+param.path_out+file_data+param.suffix+' '+file_data+' &\n', param.verbose, 'info')


#=======================================================================================================================
# fmri_moco: motion correction specific to fmri data
#=======================================================================================================================
def fmri_moco(param):

    file_data = 'fmri'
    ext_data = '.nii'
    mat_final = 'mat_final/'
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
    ext_mat = 'Warp.nii.gz'  # warping field

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(file_data+'.nii').dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), param.verbose)

    # Split into T dimension
    sct.printv('\nSplit along T dimension...', param.verbose)
    im_data = Image(file_data + ext_data)
    im_data_split_list = split_data(im_data, 3)
    for im in im_data_split_list:
        im.save()

    # assign an index to each volume
    index_fmri = range(0, nt)

    # Number of groups
    nb_groups = int(math.floor(nt/param.group_size))

    # Generate groups indexes
    group_indexes = []
    for iGroup in range(nb_groups):
        group_indexes.append(index_fmri[(iGroup*param.group_size):((iGroup+1)*param.group_size)])

    # add the remaining images to the last DWI group
    nb_remaining = nt%param.group_size  # number of remaining images
    if nb_remaining > 0:
        nb_groups += 1
        group_indexes.append(index_fmri[len(index_fmri)-nb_remaining:len(index_fmri)])

    # groups
    for iGroup in range(nb_groups):
        sct.printv('\nGroup: ' +str((iGroup+1))+'/'+str(nb_groups), param.verbose)

        # get index
        index_fmri_i = group_indexes[iGroup]
        nt_i = len(index_fmri_i)

        # Merge Images
        sct.printv('Merge consecutive volumes...', param.verbose)
        file_data_merge_i = file_data + '_' + str(iGroup)
        # cmd = fsloutput + 'fslmerge -t ' + file_data_merge_i
        # for it in range(nt_i):
        #     cmd = cmd + ' ' + file_data + '_T' + str(index_fmri_i[it]).zfill(4)

        im_fmri_list = []
        for it in range(nt_i):
            im_fmri_list.append(im_data_split_list[index_fmri_i[it]])
        im_fmri_concat = concat_data(im_fmri_list, 3)
        im_fmri_concat.setFileName(file_data_merge_i + ext_data)
        im_fmri_concat.save()

        # Average Images
        sct.printv('Average volumes...', param.verbose)
        file_data_mean = file_data + '_mean_' + str(iGroup)
        sct.run('sct_maths -i '+file_data_merge_i+'.nii -o '+file_data_mean+'.nii -mean t')
        # if not average_data_across_dimension(file_data_merge_i+'.nii', file_data_mean+'.nii', 3):
        #     sct.printv('ERROR in average_data_across_dimension', 1, 'error')
        # cmd = fsloutput + 'fslmaths ' + file_data_merge_i + ' -Tmean ' + file_data_mean
        # sct.run(cmd, param.verbose)

    # Merge groups means
    sct.printv('\nMerging volumes...', param.verbose)
    file_data_groups_means_merge = 'fmri_averaged_groups'
    # cmd = fsloutput + 'fslmerge -t ' + file_data_groups_means_merge
    # for iGroup in range(nb_groups):
    #     cmd = cmd + ' ' + file_data + '_mean_' + str(iGroup)
    im_mean_list = []
    for iGroup in range(nb_groups):
        im_mean_list.append(Image(file_data + '_mean_' + str(iGroup) + ext_data))
    im_mean_concat = concat_data(im_mean_list, 3)
    im_mean_concat.setFileName(file_data_groups_means_merge + ext_data)
    im_mean_concat.save()


    # Estimate moco on dwi groups
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Estimating motion...', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco = param
    param_moco.file_data = 'fmri_averaged_groups'
    param_moco.file_target = file_data + '_mean_' + str(param.num_target)
    param_moco.path_out = ''
    param_moco.todo = 'estimate_and_apply'
    param_moco.mat_moco = 'mat_groups'
    moco.moco(param_moco)

    # create final mat folder
    sct.create_folder(mat_final)

    # Copy registration matrices
    sct.printv('\nCopy transformations...', param.verbose)
    for iGroup in range(nb_groups):
        for data in range(len(group_indexes[iGroup])):
            # if param.slicewise:
            #     for iz in range(nz):
            #         sct.run('cp '+'mat_dwigroups/'+'mat.T'+str(iGroup)+'_Z'+str(iz)+ext_mat+' '+mat_final+'mat.T'+str(group_indexes[iGroup][dwi])+'_Z'+str(iz)+ext_mat, param.verbose)
            # else:
            sct.run('cp '+'mat_groups/'+'mat.T'+str(iGroup)+ext_mat+' '+mat_final+'mat.T'+str(group_indexes[iGroup][data])+ext_mat, param.verbose)

    # Apply moco on all fmri data
    sct.printv('\n-------------------------------------------------------------------------------', param.verbose)
    sct.printv('  Apply moco', param.verbose)
    sct.printv('-------------------------------------------------------------------------------', param.verbose)
    param_moco.file_data = 'fmri'
    param_moco.file_target = file_data+'_mean_'+str(0)
    param_moco.path_out = ''
    param_moco.mat_moco = mat_final
    param_moco.todo = 'apply'
    moco.moco(param_moco)

    # copy geometric information from header
    # NB: this is required because WarpImageMultiTransform in 2D mode wrongly sets pixdim(3) to "1".
    im_fmri = Image('fmri.nii')
    im_fmri_moco = Image('fmri_moco.nii')
    im_fmri_moco = copy_header(im_fmri, im_fmri_moco)
    im_fmri_moco.save()

    # Average volumes
    sct.printv('\nAveraging data...', param.verbose)
    sct.run('sct_maths -i fmri_moco.nii -o fmri_moco_mean.nii -mean t')
    # if not average_data_across_dimension('fmri_moco.nii', 'fmri_moco_mean.nii', 3):
    #     sct.printv('ERROR in average_data_across_dimension', 1, 'error')
    # cmd = fsloutput + 'fslmaths fmri_moco -Tmean fmri_moco_mean'
    # status, output = sct.run(cmd, param.verbose)


def get_parser():
    param_default = Param()
    parser = Parser(__file__)
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
                      default_value=param.group_size)
    parser.add_option(name='-m',
                      type_value='image_nifti',
                      description='Binary mask to limit voxels considered by the registration metric.',
                      mandatory=False)
    parser.add_option(name="-param",
                      type_value='str',
                      description="""Parameters. ALL ITEMS MUST BE LISTED IN ORDER. Separate with comma.
1) degree of polynomial function used for regularization along Z.
   For no regularization set to 0.
2) smoothing kernel size (in mm).
3) gradient step. The higher the more deformation allowed.
4) metric: {MI,MeanSquares}.
   If you find very large deformations, switching to MeanSquares can help.""",
                      mandatory=False,
                      example=param_default.param[0]+','+param_default.param[1]+','+param_default.param[2]+','+param_default.param[3])
    parser.add_option(name="-p",
                      type_value=None,
                      description="""ALL ITEMS MUST BE LISTED IN ORDER. Separate with comma.
1) degree of polynomial function used for regularization along Z.
   For no regularization set to 0.
2) smoothing kernel size (in mm).
3) gradient step. The higher the more deformation allowed.
4) metric: {MI,MeanSquares}.
   If you find very large deformations, switching to MeanSquares can help.""",
                      mandatory=False,
                      deprecated_by='-param')
    parser.add_option(name='-ofolder',
                      type_value='folder_creation',
                      description='Output path.',
                      mandatory=False,
                      default_value='./')
    parser.add_option(name='-o',
                      type_value='folder_creation',
                      description='Output path.',
                      mandatory=False,
                      default_value='./',
                      deprecated_by='-ofolder')
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


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    param_default = Param()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    param.fname_data = arguments['-i']
    if '-m' in arguments:
        param.fname_mask = arguments['-m']
    param.group_size = arguments['-g']
    path_out = arguments['-ofolder']
    param_user = ''
    if '-param' in arguments:
        param_user = arguments['-param']
    param.interp = arguments['-x']
    param.remove_tmp_files = arguments['-r']
    param.verbose = arguments['-v']

    main(path_out, param_user)
