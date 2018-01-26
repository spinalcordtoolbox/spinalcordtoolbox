#!/usr/bin/env python
#########################################################################################
#
# Module converting image files
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add output check in convert

from msct_parser import Parser
import sys
import sct_utils as sct


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.verbose = 1


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Convert image file to another type.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="File input",
                      mandatory=True,
                      example='data.nii.gz')
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="File output (indicate new extension)",
                      mandatory=True,
                      example=['data.nii'])
    parser.add_option(name="-squeeze",
                      type_value='multiple_choice',
                      description='Sueeze data dimension (remove unused dimension).',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    return parser


# conversion
# ==========================================================================================
def convert(fname_in, fname_out, squeeze_data=True, type=None, verbose=1):
    """
    Convert data
    :return True/False
    """
    from msct_image import Image
    from sct_utils import printv
    printv('sct_convert -i ' + fname_in + ' -o ' + fname_out, verbose, 'code')
    # Open file
    im = Image(fname_in)
    # Save file
    im.setFileName(fname_out)
    if type is not None:
        im.changeType(type=type)
    im.save(squeeze_data=squeeze_data)
    return im


# MAIN
# ==========================================================================================
def main(args=None):

    if not args:
        args = sys.argv[1:]

    # Building the command, do sanity checks
    parser = get_parser()
    arguments = parser.parse(args)
    fname_in = arguments["-i"]
    fname_out = arguments["-o"]
    squeeze_data = bool(int(arguments['-squeeze']))

    # convert file
    convert(fname_in, fname_out, squeeze_data=squeeze_data)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    # call main function
    main()


# import os
# import sys
# import getopt
# import sct_utils as sct
# import nibabel as nib
# from scipy.io import netcdf
#
#
# # DEFAULT PARAMETERS
# class Param:
#     ## The constructor
#     def __init__(self):
#         self.debug = 0
#         self.verbose = 1
#
# # main
# #=======================================================================================================================
# def main():
#
#     # Initialization
#     fname_data = ''
#     fname_out = ''
#     verbose = param.verbose
#
#     # Parameters for debug mode
#     if param.debug:
#         sct.printv('\n*** WARNING: DEBUG MODE ON ***\n')
#     else:
#         # Check input parameters
#         try:
#             opts, args = getopt.getopt(sys.argv[1:], 'hi:o:v:')
#         except getopt.GetoptError:
#             usage()
#         if not opts:
#             usage()
#         for opt, arg in opts:
#             if opt == '-h':
#                 usage()
#             elif opt in ('-i'):
#                 fname_data = arg
#             elif opt in ('-o'):
#                 fname_out = arg
#             elif opt in ('-v'):
#                 verbose = int(arg)
#
#     # display usage if a mandatory argument is not provided
#     if fname_data == '':
#         usage()
#
#     cmd = 'which mnc2nii'
#     status, output = sct.run(cmd)
#     if not output:
#         sct.printv('ERROR: minc-toolkit not installed...',1,'error')
#     if output != '/opt/minc/bin/mnc2nii':
#         sct.printv('ERROR: the minc-toolkit that you use is not the correct one. Please contact SCT administrator.')
#
#     # Check file existence
#     sct.printv('\nCheck file existence...', verbose)
#     sct.check_file_exist(fname_data, verbose)
#
#     # extract names
#     fname_data = os.path.abspath(fname_data)
#     path_in, file_in, ext_in = sct.extract_fname(fname_data)
#     if fname_out == '':
#         path_out, file_out, ext_out = '', file_in, '.nii'
#         fname_out = os.path.join(path_out, file_out+ext_out)
#     else:
#         fname_out = os.path.abspath(fname_out)
#         path_in, file_in, ext_in = sct.extract_fname(fname_data)
#         path_out, file_out, ext_out = sct.extract_fname(fname_out)
#
#     if ext_in=='.nii' and ext_out=='.mnc':
#         nii2mnc(fname_data,fname_out)
#     elif ext_in=='.nii.gz' and ext_out=='.mnc':
#         niigz2mnc(fname_data,fname_out)
#     elif ext_in=='.mnc' and ext_out=='.nii':
#         mnc2nii(fname_data,fname_out)
#     elif ext_in=='.mnc' and ext_out=='.nii.gz':
#         mnc2niigz(fname_data,fname_out)
#     elif ext_in=='.nii' and ext_out=='.header':
#         nii2volviewer(fname_data,fname_out)
#     elif ext_in=='.nii.gz' and ext_out=='.header':
#         niigz2volviewer(fname_data,fname_out)
#     elif ext_in=='.mnc' and ext_out=='.header':
#         mnc2volviewer(fname_data,fname_out)
#
#     # remove temp files
#     sct.run('rm -rf '+ os.path.join(path_in, 'tmp.*'), param.verbose)
#
#
# # Convert file from nifti to minc
# # ==========================================================================================
# def nii2mnc(fname_data,fname_out):
#     sct.printv("Converting from nifti to minc")
#     sct.run("nii2mnc "+fname_data+" "+fname_out)
#
# # Convert file from nifti to minc
# # ==========================================================================================
# def niigz2mnc(fname_data,fname_out):
#     sct.printv("Converting from nifti to minc")
#     path_in, file_in, ext_in = sct.extract_fname(fname_data)
#     fname_data_tmp=os.path.join(path_in, "tmp."+file_in+".nii")
#     sct.run("gunzip -c "+fname_data+" >"+fname_data_tmp)
#     sct.run("nii2mnc "+fname_data_tmp+" "+fname_out)
#
# # Convert file from minc to nifti
# # ==========================================================================================
# def mnc2nii(fname_data,fname_out):
#     sct.printv("Converting from minc to nifti")
#     sct.run("mnc2nii "+fname_data+" "+fname_out)
#
# # Convert file from minc to nifti
# # ==========================================================================================
# def mnc2niigz(fname_data,fname_out):
#     sct.printv("Converting from minc to nifti")
#     path_out, file_out, ext_out = sct.extract_fname(fname_out)
#     fname_data_tmp= os.path.join(path_out, file_out+".nii")
#     sct.run("mnc2nii "+fname_data+" "+fname_data_tmp)
#     sct.run("gzip "+fname_data_tmp)
#
# # Convert file from nifti to volumeviewer
# # ==========================================================================================
# def nii2volviewer(fname_data,fname_out):
#     sct.printv("Converting from nifti to volume viewer")
#     path_in, file_in, ext_in = sct.extract_fname(fname_data)
#     path_out, file_out, ext_out = sct.extract_fname(fname_out)
#     fname_data_nii = os.path.join(path_out, "tmp."+file_out+'.mnc')
#     nii2mnc(fname_data,fname_data_nii)
#     mnc2volviewer(fname_data_nii,os.path.join(path_out, file_out))
#
# # Convert file from nifti to volumeviewer
# # ==========================================================================================
# def niigz2volviewer(fname_data,fname_out):
#     sct.printv("Converting from nifti to volume viewer")
#     path_in, file_in, ext_in = sct.extract_fname(fname_data)
#     path_out, file_out, ext_out = sct.extract_fname(fname_out)
#     fname_data_mnc = os.path.join(path_out, "tmp."+file_out+'.mnc')
#     niigz2mnc(fname_data,fname_data_mnc)
#     mnc2volviewer(fname_data_mnc, os.path.join(path_out, file_out))
#
# # Convert file from minc to volumeviewer
# # ==========================================================================================
# def mnc2volviewer(fname_data,fname_out):
#     sct.printv("Converting from minc to volume viewer")
#     sct.run("isct_minc2volume-viewer "+fname_data+" -o "+fname_out)
#
#
# # sct.printv(usage)
# # ==========================================================================================
# def usage():
#     print("""
#         """+os.path.basename(__file__)+"""
#             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
#
#             DESCRIPTION
#             Convert files from nifti to minc, minc to nifti or nifti to volume viewer.
#
#             USAGE
#             """+os.path.basename(__file__)+""" -i <data>
#
#                 MANDATORY ARGUMENTS
#                 -i <data>             input volume
#
#                 OPTIONAL ARGUMENTS
#                 -o <output>           output volume. Add extension. Default="data".nii
#                 -v {0,1}              verbose. Default="""+str(param_default.verbose)+"""
#                 -h                    help. Show this message
#                 """
#     # exit program
#     sys.exit(2)
