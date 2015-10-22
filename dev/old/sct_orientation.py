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

# TODO: add exception in case set_orientation does not output file.

import sys
import os
import getopt
import commands
import sct_utils as sct
import time
from sct_convert import convert
from msct_image import Image
from msct_parser import Parser
from sct_image import split_data, concat_data


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.list_of_correct_orientation = 'RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'

# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # initialize parameters
    param = Param()
    param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Get or set orientation to NIFTI image (3D or 4D).')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to get or set orientation.",
                      mandatory=True,
                      example='data.nii.gz')
    parser.add_option(name="-s",
                      type_value="multiple_choice",
                      description="\nOrientation of header.",
                      default_value='',
                      mandatory=False,
                      example='RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'.split())
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="File output",
                      mandatory=False,
                      example=['data_orient.nii.gz'])
    parser.add_option(name="-a",
                      type_value="multiple_choice",
                      description="\nOrientation of image (use with care!).",
                      default_value='',
                      mandatory=False,
                      example='RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'.split())
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1', '2'])
    return parser


# get_orientation
# ==========================================================================================
def get_orientation(fname_in):
    """
    Get orientation from 3D data
    :param fname_in:
    :param remove_temp_files:
    :param verbose:
    :return:
    """
    string_out = 'Input image orientation : '
    # get orientation
    status, output = sct.run('isct_orientation3d -i '+fname_in+' -get ', 0)
    # check status
    if status != 0:
        from sct_utils import printv
        printv('ERROR in get_orientation.', 1, 'error')
    orientation = output[output.index(string_out)+len(string_out):]
    # orientation = output[26:]
    return orientation


# set_orientation
# ==========================================================================================
def set_orientation(fname_in, orientation, fname_out, inversion=False):
    if not inversion:
        sct.run('isct_orientation3d -i '+fname_in+' -orientation '+orientation+' -o '+fname_out, 0)
    else:
        from msct_image import Image
        input_image = Image(fname_in)
        input_image.change_orientation(orientation, True)
        input_image.setFileName(fname_out)
        input_image.save()
    # return full path
    return os.path.abspath(fname_out)


# MAIN
# ==========================================================================================
def main(args = None):

    orientation = ''
    change_header = ''
    fname_out = ''

    if not args:
        args = sys.argv[1:]

    # Building the command, do sanity checks
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments['-i']
    if '-o' in arguments:
        fname_out = arguments['-o']
    if '-s' in arguments:
        orientation = arguments['-s']
    if '-a' in arguments:
        change_header = arguments['-a']
    remove_tmp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])
    inversion = False  # change orientation

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S/")
    status, output = sct.run('mkdir '+path_tmp, verbose)

    # copy file in temp folder
    sct.printv('\nCopy files to tmp folder...', verbose)
    convert(fname_in, path_tmp+'data.nii', verbose=0)

    # go to temp folder
    os.chdir(path_tmp)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image('data.nii').dim
    sct.printv(str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), verbose)

    # if data are 3d, directly set or get orientation
    if nt == 1:
        if orientation != '':
            # set orientation
            sct.printv('\nChange orientation...', verbose)
            if change_header == '':
                set_orientation('data.nii', orientation, 'data_orient.nii')
            else:
                set_orientation('data.nii', change_header, 'data_orient.nii', True)
        else:
            # get orientation
            sct.printv('\nGet orientation...', verbose)
            print get_orientation('data.nii')
    else:
        # split along T dimension
        sct.printv('\nSplit along T dimension...', verbose)
        im = Image('data.nii')
        im_split_list = split_data(im, 3)
        for im_s in im_split_list:
            im_s.save()
        if orientation != '':
            # set orientation
            sct.printv('\nChange orientation...', verbose)
            for it in range(nt):
                file_data_split = 'data_T'+str(it).zfill(4)+'.nii'
                file_data_split_orient = 'data_orient_T'+str(it).zfill(4)+'.nii'
                set_orientation(file_data_split, orientation, file_data_split_orient)
            # Merge files back
            sct.printv('\nMerge file back...', verbose)
            from glob import glob
            im_data_list = [Image(fname) for fname in glob('data_orient_T*.nii')]
            im_concat = concat_data(im_data_list, 3)
            im_concat.setFileName('data_orient.nii')
            im_concat.save()

        else:
            sct.printv('\nGet orientation...', verbose)
            print get_orientation('data_T0000.nii')

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    if orientation != '':
        # Build fname_out
        if fname_out == '':
            path_data, file_data, ext_data = sct.extract_fname(fname_in)
            fname_out = path_data+file_data+'_'+orientation+ext_data
        sct.printv('\nGenerate output files...', verbose)
        sct.generate_output_file(path_tmp+'data_orient.nii', fname_out)

    # Remove temporary files
    if remove_tmp_files == 1:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf '+path_tmp, verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()

#
#
# # DEFAULT PARAMETERS
# class Param:
#     ## The constructor
#     def __init__(self):
#         self.debug = 0
#         self.fname_data = ''
#         self.fname_out = ''
#         self.orientation = ''
#         self.list_of_correct_orientation = 'RIP LIP RSP LSP RIA LIA RSA LSA IRP ILP SRP SLP IRA ILA SRA SLA RPI LPI RAI LAI RPS LPS RAS LAS PRI PLI ARI ALI PRS PLS ARS ALS IPR SPR IAR SAR IPL SPL IAL SAL PIR PSR AIR ASR PIL PSL AIL ASL'
#         self.change_header = ''
#         self.verbose = 0
#         self.remove_tmp_files = 1
#
#
# # main
# #=======================================================================================================================
# def main():
#
#     # Parameters for debug mode
#     if param.debug:
#         print '\n*** WARNING: DEBUG MODE ON ***\n'
#         # get path of the testing data
#         status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
#         param.fname_data = path_sct_data+'/dmri/dwi_moco_mean.nii.gz'
#         param.orientation = ''
#         param.change_header = ''
#         param.remove_tmp_files = 0
#         param.verbose = 1
#     else:
#         # Check input parameters
#         try:
#             opts, args = getopt.getopt(sys.argv[1:], 'hi:o:r:s:a:v:')
#         except getopt.GetoptError:
#             usage()
#         if not opts:
#             usage()
#         for opt, arg in opts:
#             if opt == '-h':
#                 usage()
#             elif opt in '-i':
#                 param.fname_data = arg
#             elif opt in '-o':
#                 param.fname_out = arg
#             elif opt in '-r':
#                 param.remove_tmp_files = int(arg)
#             elif opt in '-s':
#                 param.orientation = arg
#             elif opt in '-t':
#                 param.threshold = arg
#             elif opt in '-a':
#                 param.change_header = arg
#             elif opt in '-v':
#                 param.verbose = int(arg)
#
#     # run main program
#     get_or_set_orientation()
#
#
# # get_or_set_orientation
# #=======================================================================================================================
# def get_or_set_orientation():
#
#     fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
#
#     # display usage if a mandatory argument is not provided
#     if param.fname_data == '':
#         sct.printv('ERROR: All mandatory arguments are not provided. See usage.', 1, 'error')
#
#     # check existence of input files
#     sct.printv('\ncheck existence of input files...', param.verbose)
#     sct.check_file_exist(param.fname_data, param.verbose)
#
#     # find what to do
#     if param.orientation == '' and param.change_header is '':
#         todo = 'get_orientation'
#     else:
#         todo = 'set_orientation'
#         # check if orientation is correct
#         if check_orientation_input():
#             sct.printv('\nERROR in '+os.path.basename(__file__)+': orientation is not recognized. Use one of the following orientation: '+param.list_of_correct_orientation+'\n', 1, 'error')
#             sys.exit(2)
#
#     # display input parameters
#     sct.printv('\nInput parameters:', param.verbose)
#     sct.printv('  data ..................'+param.fname_data, param.verbose)
#
#     # Extract path/file/extension
#     path_data, file_data, ext_data = sct.extract_fname(param.fname_data)
#     if param.fname_out == '':
#         # path_out, file_out, ext_out = '', file_data+'_'+param.orientation, ext_data
#         fname_out = path_data+file_data+'_'+param.orientation+ext_data
#     else:
#         fname_out = param.fname_out
#
#     # create temporary folder
#     sct.printv('\nCreate temporary folder...', param.verbose)
#     path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
#     sct.run('mkdir '+path_tmp, param.verbose)
#
#     # Copying input data to tmp folder and convert to nii
#     sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
#     sct.run('cp '+param.fname_data+' '+path_tmp+'data'+ext_data, param.verbose)
#
#     # go to tmp folder
#     os.chdir(path_tmp)
#
#     # convert to nii format
#     convert('data'+ext_data, 'data.nii')
#
#     # Get dimensions of data
#     sct.printv('\nGet dimensions of data...', param.verbose)
#     nx, ny, nz, nt, px, py, pz, pt = Image('data.nii').dim
#     sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)+ ' x ' + str(nt), param.verbose)
#
#     # if 4d, loop across the data
#     if nt == 1:
#         if todo == 'set_orientation':
#             # set orientation
#             sct.printv('\nChange orientation...', param.verbose)
#             if param.change_header is '':
#                 set_orientation('data.nii', param.orientation, 'data_orient.nii')
#             else:
#                 set_orientation('data.nii', param.change_header, 'data_orient.nii', True)
#         elif todo == 'get_orientation':
#             # get orientation
#             sct.printv('\nGet orientation...', param.verbose)
#             sct.printv(get_orientation('data.nii'), 1)
#
#     else:
#         # split along T dimension
#         sct.printv('\nSplit along T dimension...', param.verbose)
#         from sct_image import split_data
#         im = Image('data.nii')
#         im_split_list = split_data(im, 3, '_T')
#         for im_s in im_split_list:
#             im_s.save()
#
#         if todo == 'set_orientation':
#             # set orientation
#             sct.printv('\nChange orientation...', param.verbose)
#             for it in range(nt):
#                 file_data_split = 'data_T'+str(it).zfill(4)+'.nii'
#                 file_data_split_orient = 'data_orient_T'+str(it).zfill(4)+'.nii'
#                 set_orientation(file_data_split, param.orientation, file_data_split_orient)
#             # Merge files back
#             sct.printv('\nMerge file back...', param.verbose)
#             from sct_image import concat_data
#             from glob import glob
#             im_data_list = [Image(fname) for fname in glob('data_orient_T*.nii')]
#             im_data_concat = concat_data(im_data_list, 3)
#             im_data_concat.setFileName('data_orient.nii')
#             im_data_concat.save()
#
#         elif todo == 'get_orientation':
#             sct.printv('\nGet orientation...', param.verbose)
#             sct.printv(get_orientation('data_T0000.nii'), 1)
#
#     # come back to parent folder
#     os.chdir('..')
#
#     # Generate output files
#     if todo == 'set_orientation':
#         sct.printv('\nGenerate output files...', param.verbose)
#         sct.generate_output_file(path_tmp+'data_orient.nii', fname_out)
#
#     # Remove temporary files
#     if param.remove_tmp_files == 1:
#         sct.printv('\nRemove temporary files...', param.verbose)
#         sct.run('rm -rf '+path_tmp, param.verbose)
#
#     # to view results
#     if todo == 'set_orientation':
#         sct.printv('\nDone! To view results, type:', param.verbose)
#         sct.printv('fslview '+fname_out+' &', param.verbose, 'code')
#         print
#
#
# # check_orientation_input
# # ==========================================================================================
# def check_orientation_input():
#     """check if orientation input by user is correct"""
#
#     if param.orientation in param.list_of_correct_orientation:
#         return 0
#     else:
#         return -1
#
#
# # get_orientation
# # ==========================================================================================
# def get_orientation(fname):
#     status, output = sct.run('isct_orientation3d -i '+fname+' -get ', 0)
#     if status != 0:
#         from sct_utils import printv
#         printv('ERROR in get_orientation.', 1, 'error')
#     orientation = output[26:]
#     return orientation
#
#
# # set_orientation
# # ==========================================================================================
# def set_orientation(fname_in, orientation, fname_out, inversion=False):
#     if not inversion:
#         sct.run('isct_orientation3d -i '+fname_in+' -orientation '+orientation+' -o '+fname_out, 0)
#     else:
#         from msct_image import Image
#         input_image = Image(fname_in)
#         input_image.change_orientation(orientation, True)
#         input_image.setFileName(fname_out)
#         input_image.save()
#     # return full path
#     return os.path.abspath(fname_out)
#
#
# # Print usage
# # ==========================================================================================
# def usage():
#     print """
# """+os.path.basename(__file__)+"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
#
# DESCRIPTION
#   Get or set orientation of 3D or 4D data. Available orientations are:
#   """+param_default.list_of_correct_orientation+"""
#
# USAGE
#   Get orientation: """+os.path.basename(__file__)+""" -i <data>
#   Set orientation: """+os.path.basename(__file__)+""" -i <data> -s <orient>
#
# MANDATORY ARGUMENTS
#   -i <file>        image to get or set orientation from. Can be 3D or 4D.
#
# OPTIONAL ARGUMENTS
#   -s <orient>      orientation. Default=None.
#   -o <fname_out>   output file name. Default=<file>_<orient>.<ext>.
#   -a <orient>      actual orientation of image data (for corrupted data). Change the data
#                      orientation to match orientation in the header.
#   -r {0,1}         remove temporary files. Default="""+str(param_default.remove_tmp_files)+"""
#   -v {0,1}         verbose. Default="""+str(param_default.verbose)+"""
#   -h               help. Show this message
#
# EXAMPLE
#   """+os.path.basename(__file__)+""" -i dwi.nii.gz -s RPI\n"""
#
#     # exit program
#     sys.exit(2)
#
#
# #=======================================================================================================================
# # Start program
# #=======================================================================================================================
# if __name__ == "__main__":
#     # initialize parameters
#     param = Param()
#     param_default = Param()
#     # call main function
#     main()
