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
import numpy as np
import sct_utils as sct
import msct_moco as moco
from sct_dmri_separate_b0_and_dwi import identify_b0

class param:
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_target = ''
        self.fname_centerline = ''
        # self.path_out = ''
        self.mat_final = ''
        self.num_target = 0  # target volume (or group) for moco
        self.todo = ''
        self.group_size = 3  # number of images averaged for 'dwi' method.
        self.spline_fitting = 0
        self.remove_tmp_files = 1
        self.verbose = 1
        self.plot_graph = 0
        # param for msct_moco
        self.slicewise = 0
        self.suffix = '_moco'
        self.mask_size = 0  # sigma of gaussian mask in mm --> std of the kernel. Default is 0
        self.program = 'slicereg'  # flirt, ants, ants_affine, slicereg
        self.file_schedule = '/flirtsch/schedule_TxTy.sch'  # /flirtsch/schedule_TxTy_2mm.sch, /flirtsch/schedule_TxTy.sch
        # self.cost_function_flirt = ''  # 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
        self.interp = 'spline'  # nn, linear, spline
        #Eddy Current Distortion Parameters:
        self.min_norm = 0.001


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # initialization
    start_time = time.time()
    path_out = '.'

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/fmri/fmri.nii.gz'
        param.verbose = 1
        param.slicewise = 0
        param.run_eddy = 0
        param.program = 'slicereg'  # ants_affine, flirt

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:c:d:f:g:l:m:o:p:r:s:v:z:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-c'):
            param.cost_function_flirt = arg
        elif opt in ('-d'):
            param.group_size = int(arg)
        elif opt in ('-f'):
            param.spline_fitting = int(arg)
        elif opt in ('-g'):
            param.plot_graph = int(arg)
        elif opt in ('-i'):
            param.fname_data = arg
        elif opt in ('-l'):
            param.fname_centerline = arg
        elif opt in ('-m'):
            param.program = arg
        elif opt in ('-o'):
            path_out = arg
        elif opt in ('-p'):
            param.interp = arg
        elif opt in ('-r'):
            param.remove_tmp_files = int(arg)
        elif opt in ('-s'):
            param.mask_size = float(arg)
        elif opt in ('-v'):
            param.verbose = int(arg)
        elif opt in ('-z'):
            param.slicewise = int(arg)

    # display usage if a mandatory argument is not provided
    if param.fname_data == '':
        sct.printv('ERROR: All mandatory arguments are not provided. See usage.', 1, 'error')
        usage()

    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  input file ............'+param.fname_data, param.verbose)

    # check existence of input files
    sct.printv('\nCheck file existence...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)

    # Get full path
    param.fname_data = os.path.abspath(param.fname_data)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', param.verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder and convert to nii
    # NB: cannot use c3d here because c3d cannot convert 4D data.
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    sct.run('cp '+param.fname_data+' '+path_tmp+'fmri'+ext_data, param.verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # convert fmri to nii format
    sct.run('fslchfiletype NIFTI fmri', param.verbose)

    # run moco
    fmri_moco(param)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    path_out = sct.slash_at_the_end(path_out, 1)
    sct.create_folder(path_out)
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(path_tmp+'fmri'+param.suffix+'.nii', path_out+file_data+param.suffix+ext_data, param.verbose)
    sct.generate_output_file(path_tmp+'fmri'+param.suffix+'_mean.nii', path_out+file_data+param.suffix+'_mean'+ext_data, param.verbose)

    # Delete temporary files
    if param.remove_tmp_files == 1:
        sct.printv('\nDelete temporary files...', param.verbose)
        sct.run('rm -rf '+path_tmp, param.verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    #To view results
    print '\nTo view results, type:'
    print 'fslview -m ortho,ortho '+param.path_out+file_data+param.suffix+' '+file_data+' &\n'


#=======================================================================================================================
# fmri_moco: motion correction specific to fmri data
#=======================================================================================================================
def fmri_moco(param):

    file_data = 'fmri'
    mat_final = 'mat_final/'
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(file_data+'.nii')
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), param.verbose)

    # Split into T dimension
    sct.printv('\nSplit along T dimension...', param.verbose)
    status, output = sct.run(fsloutput+'fslsplit ' + file_data + ' ' + file_data + '_T', param.verbose)

    # # Merge b=0 images
    # sct.printv('\nMerge b=0...', param.verbose)
    # cmd = fsloutput + 'fslmerge -t ' + file_b0
    # for it in range(nb_b0):
    #     cmd = cmd + ' ' + file_data + '_T' + str(index_b0[it]).zfill(4)
    # status, output = sct.run(cmd, param.verbose)
    # sct.printv(('  File created: ' + file_b0), param.verbose)
    #
    # # Average b=0 images
    # sct.printv('\nAverage b=0...', param.verbose)
    # file_b0_mean = file_b0+'_mean'
    # cmd = fsloutput + 'fslmaths ' + file_b0 + ' -Tmean ' + file_b0_mean
    # status, output = sct.run(cmd, param.verbose)

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
        cmd = fsloutput + 'fslmerge -t ' + file_data_merge_i
        for it in range(nt_i):
            cmd = cmd + ' ' + file_data + '_T' + str(index_fmri_i[it]).zfill(4)
        sct.run(cmd, param.verbose)

        # Average Images
        sct.printv('Average volumes...', param.verbose)
        file_data_mean = file_data + '_mean_' + str(iGroup)
        cmd = fsloutput + 'fslmaths ' + file_data_merge_i + ' -Tmean ' + file_data_mean
        sct.run(cmd, param.verbose)

    # Merge groups means
    sct.printv('\nMerging volumes...', param.verbose)
    file_data_groups_means_merge = 'fmri_averaged_groups'
    cmd = fsloutput + 'fslmerge -t ' + file_data_groups_means_merge
    for iGroup in range(nb_groups):
        cmd = cmd + ' ' + file_data + '_mean_' + str(iGroup)
    sct.run(cmd, param.verbose)

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
    #
    # # Copy b=0 registration matrices
    # sct.printv('\nCopy b=0 registration matrices...', param.verbose)
    # # first, use the right extension
    # # TODO: output param in moco so that we don't need to do the following twice
    #
    # for it in range(nb_b0):
    #     if param.slicewise:
    #         for iz in range(nz):
    #             sct.run('cp '+'mat_b0groups/'+'mat.T'+str(it)+'_Z'+str(iz)+ext_mat+' '+mat_final+'mat.T'+str(index_b0[it])+'_Z'+str(iz)+ext_mat, param.verbose)
    #     else:
    #         sct.run('cp '+'mat_b0groups/'+'mat.T'+str(it)+ext_mat+' '+mat_final+'mat.T'+str(index_b0[it])+ext_mat, param.verbose)

    # define type of tranformation depending of software used
    if param.program == 'flirt':
        ext_mat = '.txt'  # affine matrix
    elif param.program == 'ants':
        ext_mat = '0Warp.nii.gz'  # warping field
    elif param.program == 'slicereg':
        ext_mat = 'Warp.nii.gz'  # warping field
    elif param.program == 'ants_affine' or param.program == 'ants_rigid':
        ext_mat = '0GenericAffine.mat'  # ITK affine matrix

    # Copy DWI registration matrices
    sct.printv('\nCopy transformations...', param.verbose)
    for iGroup in range(nb_groups):
        for data in range(len(group_indexes[iGroup])):
            # if param.slicewise:
            #     for iz in range(nz):
            #         sct.run('cp '+'mat_dwigroups/'+'mat.T'+str(iGroup)+'_Z'+str(iz)+ext_mat+' '+mat_final+'mat.T'+str(group_indexes[iGroup][dwi])+'_Z'+str(iz)+ext_mat, param.verbose)
            # else:
            sct.run('cp '+'mat_groups/'+'mat.T'+str(iGroup)+ext_mat+' '+mat_final+'mat.T'+str(group_indexes[iGroup][data])+ext_mat, param.verbose)

    # Spline Regularization along T
    if param.spline_fitting:
        moco.spline(mat_final, nt, nz, param.verbose, np.array(index_b0), param.plot_graph)

    # # combine Eddy Matrices
    # if param.run_eddy:
    #     param.mat_2_combine = 'mat_eddy'
    #     param.mat_final = mat_final
    #     moco.combine_matrix(param)
    #
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
    sct.run(fsloutput+'fslcpgeom fmri fmri_moco')

    # Average volumes
    sct.printv('\nAveraging data...', param.verbose)
    cmd = fsloutput + 'fslmaths fmri_moco -Tmean fmri_moco_mean'
    status, output = sct.run(cmd, param.verbose)


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Motion correction of DWI data. Uses slice-by-slice and group-wise registration. Outputs are:
  - motion-corrected data (with suffix _moco)
  - mean b=0 data (b0_mean)
  - mean dwi data (dwi_mean)

USAGE
  """+os.path.basename(__file__)+""" -i <fmri>

MANDATORY ARGUMENTS
  -i <fmri>        diffusion data

OPTIONAL ARGUMENTS
  -d <nvols>       group nvols successive DWI volumes for more robustness. Default="""+str(param.group_size)+"""
  -s <int>         Size of Gaussian mask for more robust motion correction (in mm).
                   For no mask, put 0. Default=0
                   N.B. if centerline is provided, mask is centered on centerline. If not, mask
                   is centered in the middle of each slice.
  -l <centerline>  (requires -s). Centerline file to specify the centre of Gaussian Mask.
  -f {0,1}         spline regularization along T. Default="""+str(param.spline_fitting)+"""
                   N.B. Use only if you want to correct large drifts with time.
  -m {method}      Method for registration:
                     slicereg: slicewise regularized Tx and Ty transformations (based on ANTs). Disregard "-z"
                     ants: non-rigid deformation constrained in axial plane. HIGHLY EXPERIMENTAL!
                     ants_affine: affine transformation constrained in axial plane.
                     ants_rigid: rigid transformation constrained in axial plane.
                     flirt: FSL flirt with Tx and Ty transformations.
                     Default="""+str(param.program)+"""
  -o <path_out>    Output path.
  -p {nn,linear,spline}  Final Interpolation. Default="""+str(param.interp)+"""
  -g {0,1}         display graph of moco parameters. Default="""+str(param.plot_graph)+"""
  -v {0,1}         verbose. Default="""+str(param.verbose)+"""
  -r {0,1}         remove temporary files. Default="""+str(param.remove_tmp_files)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i fmri.nii.gz\n"""

    #Exit Program
    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = param()
    main()