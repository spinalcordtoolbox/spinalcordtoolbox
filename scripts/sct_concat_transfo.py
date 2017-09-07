#!/usr/bin/env python
#########################################################################################
#
# Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: also enable to concatenate reversed transfo


import sys
import os
import getopt
from commands import getstatusoutput
import sct_utils as sct
from msct_parser import Parser
from msct_image import Image

# DEFAULT PARAMETERS


class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.fname_warp_final = 'warp_final.nii.gz'


# main
#=======================================================================================================================
def main():

    # Initialization
    fname_warp_list = ''  # list of warping fields
    fname_dest = ''  # destination image (fix)
    fname_warp_final = ''  # concatenated transformations
    verbose = 1

    # Parameters for debug mode
    if param.debug:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n')
        status, path_sct_data = getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_warp_list = path_sct_data + '/t2/warp_template2anat.nii.gz,-' + path_sct_data + '/mt/warp_template2mt.nii.gz'
        fname_dest = path_sct_data + '/mt/mtr.nii.gz'
        verbose = 1
    else:
        # Check input parameters
        parser = get_parser()
        arguments = parser.parse(sys.argv[1:])

        fname_dest = arguments['-d']
        fname_warp_list = arguments['-w']

        if '-o' in arguments:
            fname_warp_final = arguments['-o']
        verbose = int(arguments['-v'])

    # Parse list of warping fields
    sct.printv('\nParse list of transformations...', verbose)
    use_inverse = []
    fname_warp_list_invert = []
    for i in range(len(fname_warp_list)):
        # Check if inverse matrix is specified with '-' at the beginning of file name
        if fname_warp_list[i].find('-') == 0:
            use_inverse.append('-i ')
            fname_warp_list[i] = fname_warp_list[i][1:]  # remove '-'
        else:
            use_inverse.append('')
        sct.printv('  Transfo #' + str(i) + ': ' + use_inverse[i] + fname_warp_list[i], verbose)
        fname_warp_list_invert.append(use_inverse[i] + fname_warp_list[i])

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_dest, verbose)
    for i in range(len(fname_warp_list)):
        sct.check_file_exist(fname_warp_list[i], verbose)

    # Get output folder and file name
    if fname_warp_final == '':
        path_out, file_out, ext_out = sct.extract_fname(param.fname_warp_final)
    else:
        path_out, file_out, ext_out = sct.extract_fname(fname_warp_final)

    # Check dimension of data (cf. issue #1419)
    dimensionality = '3'
    path, file, ext = sct.extract_fname(fname_warp_list[0])
    if 'nii' in ext:
        im_warp = Image(fname_warp_list[0])
        if im_warp.data.shape[2] in (0, 1):
            dimensionality = '2'

    # Concatenate warping fields
    sct.printv('\nConcatenate warping fields...', verbose)
    # N.B. Here we take the inverse of the warp list
    fname_warp_list_invert.reverse()
    cmd = 'isct_ComposeMultiTransform '+dimensionality+' warp_final' + ext_out + ' -R ' + fname_dest + ' ' + ' '.join(fname_warp_list_invert)
    sct.printv('>> ' + cmd, verbose)
    status, output = getstatusoutput(cmd)  # here cannot use sct.run() because of wrong output status in isct_ComposeMultiTransform

    # check if output was generated
    if not os.path.isfile('warp_final' + ext_out):
        sct.printv('ERROR: Warping field was not generated.\n' + output, 1, 'error')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file('warp_final' + ext_out, path_out + file_out + ext_out)


# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform (ANTs). N.B. Order of input warping fields is important. For example, if you want to concatenate: A->B and B->C to yield A->C, then you have to input warping fields like that: A->B,B->C.')
    parser.add_option(name="-d",
                      type_value="file",
                      description="Destination image.",
                      mandatory=True,
                      example='mt.nii.gz')
    parser.add_option(name="-w",
                      type_value=[[','], 'file'],
                      description='List of affine matrix or warping fields separated with "," N.B. if you want to use the inverse matrix, add "-" before matrix file name. N.B. You should NOT use "-" with warping fields (only with matrices). If you want to use an inverse warping field, then input it directly (e.g., warp_template2anat.nii.gz instead of warp_anat2template.nii.gz) ',
                      mandatory=True,
                      example='warp_template2anat.nii.gz,warp_anat2mt.nii.gz')
    parser.add_option(name="-o",
                      type_value="file_output",
                      description='Name of output warping field.',
                      mandatory=False,
                      example='warp_template2mt.nii.gz')
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
    # call main function
    main()
