#!/usr/bin/env python
#########################################################################################
#
# Check the existence of external libraries that are required for sct modules
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.log                = 1

import os
import sys
import getopt
import commands
import logging



# MAIN
# ==========================================================================================
def main():

    # Initialization
    log = param.log
    fname_log = 'sct_check_library_existence.log'

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hl')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-l"):
            log = 1

    # open log file
    if log:
        logging.basicConfig(filename=fname_log, level=logging.INFO)
        logging.info('Started')
        # check if log file was created
        if os.path.isfile(fname_log):
            print '\nLog file was created: '+fname_log
        else:
            print '\n WARNING: Log file was not created.'

    # check installation packages
    try:
        print 'Check if numpy is installed...',
        import numpy
        print 'OK!'
        logging.info('numpy installed')
    except ImportError:
        print 'WARNING: numpy is not installed! Please install CANOPY (http://www.neuro.polymtl.ca/doku.php?id=tips_and_tricks:python:canopy).'
        logging.info('numpy NOT installed')

    try:
        print 'Check if scipy is installed...',
        import scipy
        print 'OK!'
        logging.info('scipy installed')
    except ImportError:
        print 'WARNING: scipy is not installed! Please install CANOPY (http://www.neuro.polymtl.ca/doku.php?id=tips_and_tricks:python:canopy).'
        logging.info('scipy NOT installed')

    try:
        print 'Check if nibabel is installed...',
        import nibabel
        print 'OK!'
        logging.info('nibabel installed')
    except ImportError:
        print 'WARNING: nibabel is not installed! Install it by typing in the Terminal: easy_install nibabel'
        logging.info('nibabel NOT installed')

    # check if FSL is installed
    print 'Check if FSL is installed...',
    (status, output) = commands.getstatusoutput('flirt -help')
    logging.info('FSL: status='+str(status))
    if status not in [0,256]:
        print 'WARNING: FSL is not installed! Install it from: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/'
        logging.info('FSL NOT installed')
    else:
        print 'OK!'
        logging.info('FSL installed')

    # check if ANTs is installed
    print 'Check if ANTS is installed...',
    status, output = commands.getstatusoutput('antsRegistration')
    logging.info('ANTS: status='+str(status))
    if status not in [0,256]:
        print 'WARNING: ANTS is not installed! Install it from: http://stnava.github.io/ANTs/ (You have to build the source!! Do not use the binaries)'
        logging.info('ANTS NOT installed')
    else:
        print 'OK!'
        logging.info('ANTS installed')

    # check if itksnap is installed
    print 'Check if itksnap/c3d is installed...',
    status, output = commands.getstatusoutput('c3d -h')
    logging.info('c3d: status='+str(status))
    if status not in [0,256]:
        print 'WARNING: itksnap/c3d is not installed! Install it from: http://www.itksnap.org'
        logging.info('c3d NOT installed')
    else:
        print 'OK!\n'
        logging.info('c3d installed')

    # logging info
    logging.info('Finished')


# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Check the existence of external libraries that are required for sct modules.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+'\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -h            print this help.\n' \
        '  -l            output log file.\n'

    # exit program
    sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()