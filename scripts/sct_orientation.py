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
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.orientation = ''
        self.list_of_correct_orientation = 'RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'
        self.verbose = 0
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main():

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        # get path of the testing data
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        param.fname_data = path_sct_data+'/dmri/dwi_moco_mean.nii.gz'
        param.orientation = ''
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:o:r:s:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
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
    get_or_set_orientation()


# get_or_set_orientation
#=======================================================================================================================
def get_or_set_orientation():

    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI

    # display usage if a mandatory argument is not provided
    if param.fname_data == '':
        sct.printv('ERROR: All mandatory arguments are not provided. See usage.', 1, 'error')

    # check existence of input files
    sct.printv('\ncheck existence of input files...', param.verbose)
    sct.check_file_exist(param.fname_data, param.verbose)

    # find what to do
    if param.orientation == '':
        todo = 'get_orientation'
    else:
        todo = 'set_orientation'
        # check if orientation is correct
        if check_orientation_input():
            sct.printv('\nERROR in '+os.path.basename(__file__)+': orientation is not recognized. Use one of the following orientation: '+param.list_of_correct_orientation+'\n', 1, 'error')
            sys.exit(2)

    # display input parameters
    sct.printv('\nInput parameters:', param.verbose)
    sct.printv('  data ..................'+param.fname_data, param.verbose)

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
    if param.fname_out == '':
        # path_out, file_out, ext_out = '', file_data+'_'+param.orientation, ext_data
        fname_out = path_data+file_data+'_'+param.orientation+ext_data
    else:
        fname_out = param.fname_out

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
        if todo == 'set_orientation':
            # set orientation
            sct.printv('\nChange orientation...', param.verbose)
            set_orientation('data.nii', param.orientation, 'data_orient.nii')
        elif todo == 'get_orientation':
            # get orientation
            sct.printv('\nGet orientation...', param.verbose)
            sct.printv(get_orientation('data.nii'), 1)

    else:
        # split along T dimension
        sct.printv('\nSplit along T dimension...', param.verbose)
        sct.run(fsloutput+'fslsplit data data_T', param.verbose)

        if todo == 'set_orientation':
            # set orientation
            sct.printv('\nChange orientation...', param.verbose)
            for it in range(nt):
                file_data_split = 'data_T'+str(it).zfill(4)+'.nii'
                file_data_split_orient = 'data_orient_T'+str(it).zfill(4)+'.nii'
                set_orientation(file_data_split, param.orientation, file_data_split_orient)
            # Merge files back
            sct.printv('\nMerge file back...', param.verbose)
            cmd = fsloutput+'fslmerge -t data_orient'
            for it in range(nt):
                file_data_split_orient = 'data_orient_T'+str(it).zfill(4)+'.nii'
                cmd = cmd+' '+file_data_split_orient
            sct.run(cmd, param.verbose)

        elif todo == 'get_orientation':
            sct.printv('\nGet orientation...', param.verbose)
            sct.printv(get_orientation('data_T0000.nii'), 1)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    if todo == 'set_orientation':
        sct.printv('\nGenerate output files...', param.verbose)
        sct.generate_output_file(path_tmp+'data_orient.nii', fname_out)

    # Remove temporary files
    if param.remove_tmp_files == 1:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.run('rm -rf '+path_tmp, param.verbose)

    # to view results
    if todo == 'set_orientation':
        sct.printv('\nDone! To view results, type:', param.verbose)
        sct.printv('fslview '+fname_out+' &', param.verbose, 'code')
        print


# check_orientation_input
# ==========================================================================================
def check_orientation_input():
    """check if orientation input by user is correct"""

    if param.orientation in param.list_of_correct_orientation:
        return 0
    else:
        return -1


# get_orientation (uses FSL)
# ==========================================================================================
def get_orientation(fname):
    status, output = sct.run('fslhd '+fname, 0)
    # status, output = sct.run('isct_orientation3d -i '+fname+' -get', 0)
    # orientation = output[26:]
    orientation = output[output.find('sform_xorient')+15:output.find('sform_xorient')+16]+ \
                  output[output.find('sform_yorient')+15:output.find('sform_yorient')+16]+ \
                  output[output.find('sform_zorient')+15:output.find('sform_zorient')+16]

    # check if orientation is specified in an other part of the header
    if orientation == 'UUU':
        orientation = output[output.find('qform_xorient')+15:output.find('qform_xorient')+16]+ \
                      output[output.find('qform_yorient')+15:output.find('qform_yorient')+16]+ \
                      output[output.find('qform_zorient')+15:output.find('qform_zorient')+16]
    return orientation


# set_orientation
# ==========================================================================================
def set_orientation(fname_in, orientation, fname_out):
    sct.run('isct_orientation3d -i '+fname_in+' -orientation '+orientation+' -o '+fname_out, 0)
    # return full path
    return os.path.abspath(fname_out)


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Get or set orientation of 3D or 4D data. Available orientations are:
  """+param_default.list_of_correct_orientation+"""

USAGE
  Get orientation: """+os.path.basename(__file__)+""" -i <data>
  Set orientation: """+os.path.basename(__file__)+""" -i <data> -s <orient>

MANDATORY ARGUMENTS
  -i <file>        image to get or set orientation from. Can be 3D or 4D.

OPTIONAL ARGUMENTS
  -s <orient>      orientation. Default=None.
  -o <fname_out>   output file name. Default=<file>_<orient>.<ext>.
  -r {0,1}         remove temporary files. Default="""+str(param_default.remove_tmp_files)+"""
  -v {0,1}         verbose. Default="""+str(param_default.verbose)+"""
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i dwi.nii.gz -s RPI\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
