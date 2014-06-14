#!/usr/bin/env python
#########################################################################################
#
# Motion correction of dMRI data.
#
# Inspired by Xu et al. Neuroimage 2013.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-06-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: spline and z regularization
# TODO: check if masking is effective (doesn't seem to make a difference)
# TODO: check inputs, e.g. cost function...
# TODO: add flag for selecting output images


import sys
import os
import commands
import getopt
import time
import math
from sct_moco import sct_moco

try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct



class param:
    def __init__(self):
    
        self.debug                     = 0
        #self.fname_data                = ''
        #self.fname_target              = ''
        self.mat_final                 = '' # TODO: remove that
        self.mat_moco                  = '' # TODO: remove that
        self.todo                      = '' # TODO: remove that
        self.dwi_group_size            = 4              # number of images averaged for 'dwi' method.
        self.suffix                    = '_moco'
        self.mask_size                 = 0               # sigma of gaussian mask in mm --> std of the kernel. Default is 0
        self.program                   = 'FLIRT'
        self.cost_function_flirt       = 'normcorr'              # 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. 
        self.interp                    = 'trilinear'     #  Default is 'trilinear'. Additional options: trilinear,nearestneighbour,sinc,spline.
        self.remove_temp_files         = 1 # remove temporary files
        self.merge_back                = 1  # TODO: remove that
        self.path_tmp                  = ''  # TODO: remove that
        #self.path_script               = ''  # TODO: remove that
        self.verbose            = 0 # verbose
        
#=======================================================================================================================
# main
#=======================================================================================================================

def main():
    
    # Initialization
    fname_data = ''
    fname_bvecs = ''
    mask_size = param.mask_size
    interp_final = param.interp
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    start_time = time.time()
    
    # Parameters for debug mode
    if param.debug:
        fname_data = path_sct+'/testing/data/errsm_23/dmri/dmri.nii.gz'
        fname_bvecs = path_sct+'/testing/data/errsm_23/dmri/bvecs.txt'
        remove_temp_files = 0
        param.mask_size = 10

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:b:g:s:c:p:v:r:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-b'):
            fname_bvecs = arg
        elif opt in ('-c'):
            param.cost_function_flirt = arg
        elif opt in ('-i'):
            fname_data = arg
        elif opt in ('-g'):
            param.dwi_group_size = int(arg)
        elif opt in ('-r'):
            remove_temp_files = int(arg)
        elif opt in ('-s'):
            mask_size = arg
        elif opt in ('-p'):
            param.interp = arg
        elif opt in ('-r'):
            remove_temp_files = int(arg)
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '' or fname_bvecs == '':
        print '\n\nAll mandatory arguments are not provided \n'
        usage()

    # print arguments
    print '\nCheck parameters:'
    print '.. dmri data:            '+fname_data
    print '.. bvecs file:           '+fname_bvecs
    print '.. DWI group size:       '+str(param.dwi_group_size)
    print '.. Gaussian mask size:   '+str(mask_size) + 'mm'
    print ''


    # Get full path
    fname_data = os.path.abspath(fname_data)
    fname_bvecs = os.path.abspath(fname_bvecs)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    #param.path_script = os.path.dirname(__file__)
    #param.path_script = os.path.abspath(param.path_script)

    # create temporary folder
    param.path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+ param.path_tmp)
    #param.path_tmp = param.path_script + '/' + param.path_tmp + '/'

    # go to tmp folder
    os.chdir(param.path_tmp)

    # run moco
    sct_moco_process_dmri(param, fname_data, fname_bvecs)

    # Generate output files
    print('\nGenerate output files...')
    fname_out = sct.generate_output_file('dmri_moco.nii', '../', file_data+param.suffix, ext_data)

    # come back to parent folder
    os.chdir('..')

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf '+ param.path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

    # to view results
    print '\nTo view results, type:'
    print 'fslview '+file_data+' '+file_data+'_moco &\n'

#=======================================================================================================================
# Function sct_moco_process_dmri - Preparing Data For MOCO
#=======================================================================================================================

def sct_moco_process_dmri(param, fname_data, fname_bvecs):
    
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    interp_final = param.interp

    dwi_group_size = param.dwi_group_size

    #path_tmp = param.path_tmp
    #path_script = param.path_script
    
    ## get path of the toolbox # TODO: no need to do that another time!
    #status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    ## append path that contains scripts, to be able to load modules
    #sys.path.append(path_sct + '/scripts')
    #import sct_utils as sct
    
    # check existence of input files
    sct.check_file_exist(fname_data)
    sct.check_file_exist(fname_bvecs)
    
    # Get full path
    fname_data = os.path.abspath(fname_data)
    fname_bvecs = os.path.abspath(fname_bvecs)
    
    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    
    file_b0 = 'b0'
    file_dwi = 'dwi'
    
    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)

    # Open bvecs file
    print '\nOpen bvecs file...'
    bvecs = []
    with open(fname_bvecs) as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)
    
    # Check if bvecs file is nx3
    if not len(bvecs[0][:]) == 3:
        print '.. WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.'
        print 'Transpose bvecs...'
        # transpose bvecs
        bvecs = zip(*bvecs)

    # Identify b=0 and DWI images
    print '\nIdentify b=0 and DWI images...'
    index_b0 = []
    index_dwi = []
    for it in xrange(0,nt):
        if math.sqrt(math.fsum([i**2 for i in bvecs[it]])) < 0.01:
            index_b0.append(it)
        else:
            index_dwi.append(it)
    n_b0 = len(index_b0)
    n_dwi = len(index_dwi)
    print '.. Index of b=0:'+str(index_b0)
    print '.. Index of DWI:'+str(index_dwi)

    # Split into T dimension
    print '\nSplit along T dimension...'
    status, output = sct.run(fsloutput + 'fslsplit ' + fname_data + ' data_splitT')
    numT = []
    for i in range(nt):
        if len(str(i))==1:
            numT.append('000' + str(i))
        elif len(str(i))==2:
            numT.append('00' + str(i))
        elif len(str(i))==3:
            numT.append('0' + str(i))
        else:
            numT.append(str(nt))

    # Merge b=0 images
    print '\nMerge b=0...'
    fname_b0_merge = file_b0    #+ suffix_data
    cmd = fsloutput + 'fslmerge -t ' + fname_b0_merge
    for iT in range(n_b0):
        cmd = cmd + ' data_splitT' + numT[index_b0[iT]]
    status, output = sct.run(cmd)
    print '.. File created: ', fname_b0_merge

    # Average b=0 images
    print '\nAverage b=0...'
    fname_b0_mean = file_b0 + '_mean'
    cmd = fsloutput + 'fslmaths ' + fname_b0_merge + ' -Tmean ' + fname_b0_mean
    status, output = sct.run(cmd)

    # Number of DWI groups
    nb_groups = int(math.floor(n_dwi/dwi_group_size))
    
    # Generate groups indexes
    group_indexes = []
    for iGroup in range(nb_groups):
        group_indexes.append(index_dwi[(iGroup*dwi_group_size):((iGroup+1)*dwi_group_size)])

    # add the remaining images to the last DWI group
    # TODO: fix the thing below
    #nb_remaining = n_dwi%dwi_group_size # number of remaining images
    nb_remaining = n_dwi - dwi_group_size * nb_groups # number of remaining images
    if nb_remaining > 0:
        #if nb_remaining < 3: # TODO: WHY 3?
        #    #group_indexes[nb_groups-1].append(index_dwi[len(index_dwi)-nb_remaining:len(index_dwi)])
        #    group_indexes.append(index_dwi[len(index_dwi)-nb_remaining:len(index_dwi)])
        #else:
        nb_groups += 1
        group_indexes.append(index_dwi[len(index_dwi)-nb_remaining:len(index_dwi)])

    # Size of dwi groups                        #SUFFIX
    for iGroup in range(nb_groups):
        print '\nGroup ', str((iGroup+1)), ' of DW images'
    
        index_dwi_i = group_indexes[iGroup]
        nb_dwi_i = len(index_dwi_i)
        
        # Merge DWI images
        print '\nMerge DW images...'        
        fname_dwi_merge_i = file_dwi + '_' + str(iGroup)
        cmd = fsloutput + 'fslmerge -t ' + fname_dwi_merge_i
        for iT in range(nb_dwi_i):
            cmd = cmd + ' data_splitT' + numT[index_dwi_i[iT]]
        status, output = sct.run(cmd)

        # Average DWI images
        print '\nAverage DW images...'
        fname_dwi_mean = file_dwi + '_mean' + '_' + str(iGroup)
        cmd = fsloutput + 'fslmaths ' + fname_dwi_merge_i + ' -Tmean ' + fname_dwi_mean
        status, output = sct.run(cmd)

    # Merge DWI groups means
    print '\nMerging DW files...'
    fname_dwi_groups_means_merge = 'dwi_averaged_groups'
    cmd = fsloutput + 'fslmerge -t ' + fname_dwi_groups_means_merge
    for iGroup in range(nb_groups):
        cmd = cmd + ' ' + file_dwi + '_mean_' + str(iGroup)
    status, output = sct.run(cmd)

    # Average DWI images
    print '\nAveraging all DW images...'
    fname_dwi_mean = 'dwi_mean'
    cmd = fsloutput + 'fslmaths ' + fname_dwi_groups_means_merge + ' -Tmean ' + fname_dwi_mean
    status, output = sct.run(cmd)

    # Estimate moco on dwi groups
    print '\n------------------------------------------------------------------------------'
    print 'Estimating motion based on DW groups...'
    print '------------------------------------------------------------------------------\n'
    param.fname_data =  fname_dwi_groups_means_merge
    param.fname_target =  fname_dwi_mean
    param.todo = 'estimate_and_apply'
    param.mat_moco = 'dwigroups_moco.mat'
    param.interp = 'trilinear'
    sct_moco(param)

    #Copy registration matrix for every dwi based on dwi_averaged_groups
    print '\n------------------------------------------------------------------------------'
    print 'Copy registration matrix for every dwi based on dwi_averaged_groups matrix...'
    print '------------------------------------------------------------------------------\n'
    mat_final = 'mat_final/'
    if not os.path.exists(mat_final):
        os.makedirs(mat_final)
        
    for b0 in range(len(index_b0)):
        for i_Z in range(nz):
            cmd = 'cp dwigroups_moco.mat/' + 'mat.T0' + '_Z' + str(i_Z) + '.txt' + ' ' + mat_final + 'mat.T' + str(index_b0[b0]) + '_Z' + str(i_Z) + '.txt'
            status, output = sct.run(cmd)
    
    for iGroup in range(nb_groups):
        for dwi in range(len(group_indexes[iGroup])):
            for i_Z in range(nz):
                cmd = 'cp dwigroups_moco.mat/' + 'mat.T' + str(iGroup) + '_Z' + str(i_Z) + '.txt' + ' ' + mat_final + 'mat.T' + str(group_indexes[iGroup][dwi]) + '_Z' + str(i_Z) + '.txt'
                status, output = sct.run(cmd)

    #Apply moco on all dmri data
    print '\n\n\n------------------------------------------------------------------------------'
    print 'Apply moco on all dmri data...'
    print '------------------------------------------------------------------------------\n'
    param.fname_data =  fname_data
    param.fname_target = fname_data
    param.mat_final = mat_final
    param.todo = 'apply'
    param.interp = interp_final
    sct_moco(param)



# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Motion correction of DWI data. Uses slice-by-slice and group-wise registration.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <dmri> -b <bvecs>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <dmri>                  diffusion data\n' \
        '  -b <bvecs>                 bvecs file\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -g <int>                   Number of volumes within groups. Default='+str(param.dwi_group_size)+'\n' \
        '  -s <num>                   Gaussian mask size to improve robustness (in mm). Default='+str(param.mask_size)+'\n' \
        '  -p {nearestneighbour, trilinear, sinc, spline}          Final interpolation. Default='+str(param.interp)+'\n' \
        '  -c {mutualinfo, woods, corratio, normcorr, normmi, leastsquares}   Cost function for FLIRT. Default='+str(param.cost_function_flirt)+'\n' \
        '  -h                         help. Show this message.\n' \
        '  -r {0, 1}                  remove temporary files. Default='+str(param.remove_temp_files)+'.\n' \
        '  -v {0, 1}                  verbose. Default='+str(param.verbose)+'.\n' \
        '\n'\
        'EXAMPLE\n' \
        '  '+os.path.basename(__file__)+' -i dmri.nii -b bvecs.txt \n'


    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
