#!/usr/bin/env python
#########################################################################################
#
# Compute magnetization transfer ratio (MTR).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-09-21
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import sys
import os

import sct_utils as sct
from msct_parser import Parser

# DEFAULT PARAMETERS


class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        # self.register = 1
        self.verbose = 1
        self.file_out = 'mtr'
        self.remove_temp_files = 1


# main
#=======================================================================================================================
def main(args=None):
    import numpy as np
    import spinalcordtoolbox.image as msct_image

    # Initialization
    fname_mt0 = ''
    fname_mt1 = ''
    file_out = param.file_out
    # register = param.register
    # remove_temp_files = param.remove_temp_files
    # verbose = param.verbose

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Check input parameters
    parser = get_parser()
    arguments = parser.parse(args)

    fname_mt0 = arguments['-mt0']
    fname_mt1 = arguments['-mt1']
    fname_mtr = arguments['-omtr']
    param.file_out, file_out = os.path.split(fname_mtr)[1], os.path.split(fname_mtr)[1]
    remove_temp_files = int(arguments['-r'])
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # Extract path/file/extension
    path_mt0, file_mt0, ext_mt0 = sct.extract_fname(fname_mt0)
    path_out, file_out, ext_out = '', file_out, ext_mt0

    # create temporary folder
    path_tmp = sct.tmp_create()

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    from sct_convert import convert
    convert(fname_mt0, os.path.join(path_tmp, "mt0.nii"), dtype=np.float32)
    convert(fname_mt1, os.path.join(path_tmp, "mt1.nii"), dtype=np.float32)

    # if changing output file name or location, create folder with mt0 and mt1 files at precised location
    curdir = os.getcwd()
    if os.path.split(fname_mtr)[0] != curdir:
        import shutil
        startdir = os.getcwd()
        os.chdir(os.path.split(fname_mtr)[0])
        shutil.copy(os.path.join(startdir,fname_mt0),os.path.join(os.path.split(fname_mtr)[0]))
        shutil.copy(os.path.join(startdir, fname_mt1), os.path.join(os.path.split(fname_mtr)[0]))

    # go to tmp folder
    os.chdir(path_tmp)

    # compute MTR
    sct.printv('\nCompute MTR...', verbose)
    nii_mt1 = msct_image.Image('mt1.nii')
    data_mt1 = nii_mt1.data
    data_mt0 = msct_image.Image('mt0.nii').data
    data_mtr = 100 * (data_mt0 - data_mt1) / data_mt0
    # save MTR file
    nii_mtr = nii_mt1
    nii_mtr.data = data_mtr
    nii_mtr.save(file_out + ".nii")
    # sct.run(fsloutput+'fslmaths -dt double mt0.nii -sub mt1.nii -mul 100 -div mt0.nii -thr 0 -uthr 100 file_out.nii', verbose)

    # come back
    os.chdir(os.path.split(fname_mtr)[0])

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(os.path.join(path_tmp, file_out + ".nii"), os.path.join(path_out, file_out + ext_out))

    # Remove temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...')
        sct.rmtree(path_tmp)

    # if output file location changed, notify user to move to file location
    if os.path.split(fname_mtr)[0] != curdir:
        sct.printv("\n\033[1;31mNotice: \033[0;0mOutput file location has changed. Before issuing the command to view the results, type:")
        sct.printv("\033[0;32mcd " + os.path.split(fname_mtr)[0])
    sct.display_viewer_syntax([fname_mt0, fname_mt1, file_out])


# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute magnetization transfer ratio (MTR). Output is given in percentage.')
    parser.add_option(name="-mt0",
                      type_value="file",
                      description="Image without MT pulse (MT0)",
                      mandatory=True,
                      example='mt0.nii.gz')
    parser.add_option(name="-i",
                      type_value=None,
                      description="Image without MT pulse (MT0)",
                      mandatory=False,
                      deprecated_by='-mt0')
    parser.add_option(name="-mt1",
                      type_value="file",
                      description="Image with MT pulse (MT1)",
                      mandatory=True,
                      example='mt1.nii.gz')
    parser.add_option(name="-j",
                      type_value=None,
                      description="Image with MT pulse (MT1)",
                      mandatory=False,
                      deprecated_by="-mt1")
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
    parser.add_option(name="-omtr",
                      type_value="str",
                      description="Creates output file with the specified path.",
                      mandatory=False,
                      example='My_File_Folder/My_New_File',
                      default_value=os.path.join(os.getcwd(),'mtr'))

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    # param_default = Param()
    # call main function
    main()
