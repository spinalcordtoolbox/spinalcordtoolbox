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
        self.remove_tmp_files = 1


# main
#=======================================================================================================================
def main():

    # Initialization
    fname_mt0 = ''
    fname_mt1 = ''
    file_out = param.file_out
    # register = param.register
    # remove_tmp_files = param.remove_tmp_files
    # verbose = param.verbose

    # get path of the toolbox
    # status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Check input parameters
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_mt0 = arguments['-mt0']
    fname_mt1 = arguments['-mt1']
    remove_tmp_files = int(arguments['-r'])
    verbose = int(arguments['-v'])

    # Extract path/file/extension
    path_mt0, file_mt0, ext_mt0 = sct.extract_fname(fname_mt0)
    path_out, file_out, ext_out = '', file_out, ext_mt0

    # create temporary folder
    path_tmp = sct.tmp_create()

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    from sct_convert import convert
    convert(fname_mt0, path_tmp + 'mt0.nii', type='float32')
    convert(fname_mt1, path_tmp + 'mt1.nii', type='float32')

    # go to tmp folder
    os.chdir(path_tmp)

    # compute MTR
    sct.printv('\nCompute MTR...', verbose)
    from msct_image import Image
    nii_mt1 = Image('mt1.nii')
    data_mt1 = nii_mt1.data
    data_mt0 = Image('mt0.nii').data
    data_mtr = 100 * (data_mt0 - data_mt1) / data_mt0
    # save MTR file
    nii_mtr = nii_mt1
    nii_mtr.data = data_mtr
    nii_mtr.setFileName('mtr.nii')
    nii_mtr.save()
    # sct.run(fsloutput+'fslmaths -dt double mt0.nii -sub mt1.nii -mul 100 -div mt0.nii -thr 0 -uthr 100 mtr.nii', verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp + 'mtr.nii', path_out + file_out + ext_out)

    # Remove temporary files
    if remove_tmp_files == 1:
        sct.printv('\nRemove temporary files...')
        sct.run('rm -rf ' + path_tmp)

    # to view results
    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv('fslview ' + fname_mt0 + ' ' + fname_mt1 + ' ' + file_out + ' &\n', verbose, 'info')


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

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    # param_default = Param()
    # call main function
    main()
