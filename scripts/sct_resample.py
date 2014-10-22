#!/usr/bin/env python
#########################################################################################
#
# Resample data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-10-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################


#TODO: pad for c3d!!!!!!


import sys
import os
import getopt
import commands
import sct_utils as sct
import time


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.factor = ''
        self.file_suffix = 'r'  # output suffix
        self.verbose = 1
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main():

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/fmri/fmri.nii.gz'
        param.factor = '2' #'0.5x0.5x1'
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hf:i:r:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in '-f':
                param.factor = arg
            elif opt in '-i':
                param.fname_data = arg
            elif opt in '-r':
                param.remove_tmp_files = int(arg)
            elif opt in '-v':
                param.verbose = int(arg)

    # run main program
    resample()


# resample
#=======================================================================================================================
def resample():

    dim = 4  # by default, will be adjusted later
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
    ext = '.nii'

    # display usage if a mandatory argument is not provided
    if param.fname_data == '' or param.factor == '':
        sct.printv('\nERROR: All mandatory arguments are not provided. See usage (add -h).\n', 1, 'error')

    # check existence of input files
    sct.printv('\nCheck existence of input files...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)

    # extract resampling factor
    sct.printv('\nParse resampling factor...', param.verbose)
    factor_split = param.factor.split('x')
    factor = [float(factor_split[i]) for i in range(len(factor_split))]
    # check if it has three values
    if not len(factor) == 3:
        sct.printv('\nERROR: factor should have three dimensions. E.g., 2x2x1.\n', 1, 'error')
    else:
        fx, fy, fz = [float(factor_split[i]) for i in range(len(factor_split))]

    # display input parameters
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  data ..................'+param.fname_data, param.verbose)
    sct.printv('  resampling factor .....'+param.factor, param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    path_out, file_out, ext_out = '', file_data, ext_data

    # create temporary folder
    sct.printv('\nCreate temporary folder...', param.verbose)
    path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.run('mkdir '+path_tmp, param.verbose)

    # Copying input data to tmp folder and convert to nii
    # NB: cannot use c3d here because c3d cannot convert 4D data.
    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    sct.run('cp '+param.fname_data+' '+path_tmp+'data'+ext_data, param.verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # convert fmri to nii format
    sct.run('fslchfiletype NIFTI data', param.verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data.nii')
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)
    if nt == 1:
        dim == 3
    if nz == 1:
        dim == 2
        sct.run('ERROR (sct_resample): Dimension of input data is different from 3 or 4. Exit program', param.verbose, 'error')

    # Calculate new dimensions
    sct.printv('\nCalculate new dimensions...', param.verbose)
    nx_new = int(round(nx*fx))
    ny_new = int(round(ny*fy))
    nz_new = int(round(nz*fz))
    sct.printv('  ' + str(nx_new) + ' x ' + str(ny_new) + ' x ' + str(nz_new)+ ' x ' + str(nt), param.verbose)

    # if dim=4, split data
    if dim == 4:
        # Split into T dimension
        sct.printv('\nSplit along T dimension...', param.verbose)
        status, output = sct.run(fsloutput+'fslsplit data data_T', param.verbose)
    elif dim == 3:
        # rename file to have compatible code with 4d
        status, output = sct.run('cp data.nii data_T0000.nii', param.verbose)

    for it in range(nt):
        # identify current volume
        file_data_splitT = 'data_T'+str(it).zfill(4)
        file_data_splitT_resample = file_data_splitT+'r'

        # resample volume
        sct.printv(('\nResample volume '+str((it+1))+'/'+str(nt)+':'), param.verbose)
        sct.run('sct_c3d '+file_data_splitT+ext+' -resample '+str(nx_new)+'x'+str(ny_new)+'x'+str(nz_new)+'vox -o '+file_data_splitT_resample+ext)

        # pad data (for ANTs)
        # # TODO: check if need to pad also for the estimate_and_apply
        # if program == 'ants' and todo == 'estimate' and slicewise == 0:
        #     sct.run('sct_c3d '+file_data_splitT_num[it]+' -pad 0x0x3vox 0x0x3vox 0 -o '+file_data_splitT_num[it]+'_pad.nii')
        #     file_data_splitT_num[it] = file_data_splitT_num[it]+'_pad'

    # merge data back along T
    file_data_resample = file_data+param.file_suffix
    sct.printv('\nMerge data back along T...', param.verbose)
    cmd = fsloutput + 'fslmerge -t ' + file_data_resample
    for it in range(nt):
        cmd = cmd + ' ' + 'data_T'+str(it).zfill(4)+'r'
    sct.run(cmd, param.verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(path_tmp+file_data_resample+ext, path_out+file_out+param.file_suffix+ext_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp, param.verbose)

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+path_out+file_out+param.file_suffix+ext_out+' &', param.verbose, 'code')
    print


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Anisotropic resampling of 3D or 4D data.

USAGE
  """+os.path.basename(__file__)+""" -i <data> -r <factor>

MANDATORY ARGUMENTS
  -i <data>        image to segment. Can be 2D, 3D or 4D.
  -f <fxxfyxfz>    resampling factor in each of the first 3 dimensions (x,y,z). Separate with "x"
                   For 2x upsampling, set to 2. For 2x downsampling set to 0.5

OPTIONAL ARGUMENTS
  -r {0,1}         remove temporary files. Default="""+str(param_debug.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param_debug.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dwi.nii.gz -f 0.5x0.5x1\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_debug = Param()
    # call main function
    main()
