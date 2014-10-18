#!/usr/bin/env python
#########################################################################################
#
# Get or set orientation of nifti 3d or 4d data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-10-18
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
import getopt
import commands
import sct_utils as sct
import time

# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.orientation = ''
        self.verbose = 1
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main(param):

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/dmri/dmri.nii.gz'
        param.orientation = 'RPI'
        param.remove_tmp_files = 0
        param.verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:o:r:s:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in '-i':
            param.fname_data = arg
        elif opt in '-o':
            param.fname_out = arg
        elif opt in '-r':
            param.remove_tmp_files = int(arg)
        elif opt in '-s':
            param.orientation = arg
        elif opt in '-t':
            param.threshold = arg
        elif opt in '-v':
            param.verbose = int(arg)

    # run main program
    get_or_set_orientation(param)


# otsu
#=======================================================================================================================
def get_or_set_orientation(param):

    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

    # display usage if a mandatory argument is not provided
    if param.fname_data == '':
        sct.printv('ERROR: All mandatory arguments are not provided. See usage.', 1, 'error')
        usage()

    # check existence of input files
    sct.printv('\ncheck existence of input files...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)

    # display input parameters
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  data ..................'+param.fname_data, param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    if param.fname_out == '':
        # path_out, file_out, ext_out = '', file_data+'_'+param.orientation, ext_data
        fname_out = os.path.abspath(file_data+'_'+param.orientation+ext_data)
    else:
        fname_out = os.path.abspath(param.fname_out)

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

    # convert to nii format
    sct.run('fslchfiletype NIFTI data', param.verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('data.nii')
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)

    # if 4d, loop across the data
    if nt == 1:
        dim = 3

    else:
        # split along T dimension
        sct.printv('\nSplit along T dimension...', param.verbose)
        sct.run(fsloutput+'fslsplit data data_T', param.verbose)

        # change orientation
        sct.printv('\nChange orientation...', param.verbose)
        for it in range(nt):
            file_data_split = 'data_T'+str(it).zfill(4)+'.nii'
            file_data_split_orient = 'data_orient_T'+str(it).zfill(4)+'.nii'
            sct.run('isct_orientation3d -i '+file_data_split+' -orientation '+param.orientation+' -o '+file_data_split_orient, param.verbose)

        # Merge files back
        sct.printv('\nMerge file back...', param.verbose)
        cmd = fsloutput+'fslmerge -t data_orient'
        for it in range(nt):
            file_data_split_orient = 'data_orient_T'+str(it).zfill(4)+'.nii'
            cmd = cmd+' '+file_data_split_orient
        sct.run(cmd, param.verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', param.verbose)
    sct.generate_output_file(path_tmp+'data_orient.nii', fname_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp, param.verbose)

    # to view results
    sct.printv('\nDone! To view results, type:', param.verbose)
    sct.printv('fslview '+fname_out+' &', param.verbose, 'code')
    print


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Get or set orientation of 3D or 4D data. Available orientations are: XXX

USAGE
  Get orientation: """+os.path.basename(__file__)+""" -i <data>
  Set orientation: """+os.path.basename(__file__)+""" -i <data> -s <orient>

MANDATORY ARGUMENTS
  -i <data>        image to get or set orientation from. Can be 3D or 4D.

OPTIONAL ARGUMENTS
  -s <orient>      orientation. Default=None.
  -o <fname_out>   output file name. Default=None.
  -r {0,1}         remove temporary files. Default="""+str(param.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dwi.nii.gz -t 5\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main(param)
