#!/usr/bin/env python
#########################################################################################
#
# This function test the integrity of ANTs output, given that some versions of ANTs give a wrong BSpline transform,
# notably when using sct_ANTSUseLandmarkImagesToGetBSplineDisplacementField.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys

import getopt
import os
import sct_utils as sct


# main
#=======================================================================================================================
def main():

    # Initialization
    remove_temp_files = 1
    verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hvr:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-v'):
            verbose = int(arg)
        elif opt in ('-r'):
            remove_temp_files = int(arg)    

    from spinalcordtoolbox.test_ants import script

    script.suf(verbose=verbose, remove_temp_files=remove_temp_files)

# sct.printv(usage)
# ==========================================================================================
def usage():
    print('\n' \
        '' + os.path.basename(__file__) + '\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This function test the integrity of ANTs output, given that some versions of ANTs give a wrong BSpline ' \
        '  transform notably when using sct_ANTSUseLandmarkImagesToGetBSplineDisplacementField..\n' \
        '\n' \
        'USAGE\n' \
        '  ' + os.path.basename(__file__) + '\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -h                         show this help\n' \
        '  -r {0, 1}                  remove temp files. Default=1\n' \
        '  -v {0, 1}                  verbose. Default=1\n' \
        '\n')

    # exit program
    sys.exit(2)


# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
