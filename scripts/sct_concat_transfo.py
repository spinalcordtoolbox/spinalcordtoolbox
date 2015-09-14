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
import commands
import sct_utils as sct

# DEFAULT PARAMETERS
class Param:
    ## The constructor
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
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_warp_list = path_sct_data+'/t2/warp_template2anat.nii.gz,-'+path_sct_data+'/mt/warp_template2mt.nii.gz'
        fname_dest = path_sct_data+'/mt/mtr.nii.gz'
        verbose = 1
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hw:d:o:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-w'):
                fname_warp_list = arg
            elif opt in ('-d'):
                fname_dest = arg
            elif opt in ('-o'):
                fname_warp_final = arg
            elif opt in ('-v'):
                verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_warp_list == '' or fname_dest == '':
        usage()

    # Parse list of warping fields
    sct.printv('\nParse list of transformations...', verbose)
    use_inverse = []
    fname_warp_list_invert = []
    fname_warp_list = fname_warp_list.replace(' ', '')  # remove spaces
    fname_warp_list = fname_warp_list.split(",")  # parse with comma
    for i in range(len(fname_warp_list)):
        # Check if inverse matrix is specified with '-' at the beginning of file name
        if fname_warp_list[i].find('-') == 0:
            use_inverse.append('-i ')
            fname_warp_list[i] = fname_warp_list[i][1:]  # remove '-'
        else:
            use_inverse.append('')
        sct.printv('  Transfo #'+str(i)+': '+use_inverse[i]+fname_warp_list[i], verbose)
        fname_warp_list_invert.append(use_inverse[i]+fname_warp_list[i])

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

    # Concatenate warping fields
    sct.printv('\nConcatenate warping fields...', verbose)
    # N.B. Here we take the inverse of the warp list
    fname_warp_list_invert.reverse()
    cmd = 'isct_ComposeMultiTransform 3 warp_final.nii.gz -R '+fname_dest+' '+' '.join(fname_warp_list_invert)
    sct.printv('>> '+cmd, verbose)
    commands.getstatusoutput(cmd)  # here cannot use sct.run() because of wrong output status in isct_ComposeMultiTransform

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file('warp_final.nii.gz', path_out+file_out+ext_out)

    print ''


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform (ANTs).
  N.B. Order of input warping fields is important. For example, if you want to concatenate: A->B and
  B->C to yield A->C, then you have to input warping fields like that: A->B,B->C

USAGE
  """+os.path.basename(__file__)+""" -w <warp_list> -d <dest>

MANDATORY ARGUMENTS
  -w <warp_list>        list of affine matrix or warping fields separated with ","
                        N.B. if you want to use the inverse matrix, add "-" before matrix file name.
  -d <dest>             destination image

OPTIONAL ARGUMENTS
  -o <warp_final>       name of output warping field
  -h                    help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -w warp_AtoB.nii.gz,warp_BtoC.nii.gz -d t1.nii.gz -o warp_AtoC.nii.gz\n"""

    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
