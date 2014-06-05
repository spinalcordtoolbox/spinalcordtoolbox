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

# TODO: find another way to create log file. E.g. sct.print(). For color as well.
# TODO: manage .cshrc files


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.log                = 0

import os
import sys
#import getopt
import commands
import time
#import logging


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# MAIN
# ==========================================================================================
def main():

    # Initialization
#    log = param.log
#    fname_log = 'sct_check_library_existence.log'

    # old_stdout = sys.stdout
    # log_file = open(fname_log,"w")
    # sys.stdout = log_file

    # Check input parameters
#    try:
#        opts, args = getopt.getopt(sys.argv[1:],'hl')
#    except getopt.GetoptError:
#        usage()
#    for opt, arg in opts:
#        if opt == '-h':
#            usage()
#        elif opt in ("-l"):
#            log = 1

    # open log file
#    if log:
#        logging.basicConfig(filename=fname_log, level=logging.INFO)
#        logging.info('Started')
#        # check if log file was created
#        if os.path.isfile(fname_log):
#            print '\nLog file was created: '+fname_log
#        else:
#            print '\n WARNING: Log file was not created.'

    # initialization
    fsl_is_installed = 1
    ants_is_installed = 1
    c3d_is_installed = 1
    install_software = 0
    restart_terminal = 0
    os_running = ''
    print
        
    # check OS
    print
    print 'Check which OS is running ... '
    if (sys.platform == 'darwin'):
        os_running = 'osx'
    print '  '+os_running+' ('+sys.platform+')'

    # check installation packages
    print 'Check which Python is running ... '
    print '  '+sys.executable

    # check numpy
    print_line('Check if numpy is installed ................... ')
    try:
        import numpy
        print_ok()
    except ImportError:
        print_fail()
        print '  numpy is not installed! Please install ENTHOUGH CANOPY (https://www.enthought.com/downloads/)'
        install_software = 1

    # check scipy
    print_line('Check if scipy is installed ................... ')
    try:
        import scipy
        print_ok()
    except ImportError:
        print_fail()
        print '  scipy is not installed! Please install ENTHOUGH CANOPY (https://www.enthought.com/downloads/).'
        install_software = 1

    # check nibabel
    print_line('Check if nibabel is installed ................. ')
    try:
        import nibabel
        print_ok()
    except ImportError:
        print_fail()
        print '  nibabel is not installed! Install it by typing in the Terminal: easy_install nibabel'
        install_software = 1

    # check if FSL is installed
    print_line('Check if FSL is installed ..................... ')
    (status, output) = commands.getstatusoutput('find /usr -name "fsl" -type d -print -quit 2>/dev/null')
    if output:
        print_ok()
        path_fsl = output
        print '  '+path_fsl
    else:
        print_fail()
        print '  FSL is not installed! Install it from: http://fsl.fmrib.ox.ac.uk/'
        fsl_is_installed = 0
        install_software = 1

    # check if FSL is declared
    if fsl_is_installed:
        print_line('Check if FSL is declared ...................... ')
        (status, output) = commands.getstatusoutput('which fsl')
        if output:
            print_ok()
        else:
            print_warning()
            print '  FSL is not declared! Modifying .bash_profile ...'
            add_bash_profile('#FSL (added on '+time.strftime("%Y-%m-%d")+')\n' \
                'FSLDIR='+path_fsl+'\n' \
                '. ${FSLDIR}/etc/fslconf/fsl.sh\n' \
                'export FSLDIR PATH')
            restart_terminal = 1

    # check if ANTS is installed
    print_line('Check if ANTs is installed .................... ')
    (status, output) = commands.getstatusoutput('find /usr -name "antsRegistration" -type f -print -quit 2>/dev/null')
    if output:
        print_ok()
        path_ants = os.path.dirname(output)
        print '  '+path_ants
    else:
        print_fail()
        print '  ANTs is not installed! Follow instructions here: https://sourceforge.net/p/spinalcordtoolbox/wiki/ants_installation/'
        ants_is_installed = 0
        install_software = 1

    # check if ANTS is declared
    if ants_is_installed:
        print_line('Check if ANTs is declared ..................... ')
        (status, output) = commands.getstatusoutput('which antsRegistration')
        if output:
            print_ok()
        else:
            print_warning()
            print '  ANTs is not declared! Modifying .bash_profile ...'
            add_bash_profile('#ANTS (added on '+time.strftime("%Y-%m-%d")+')\n' \
                'PATH=${PATH}:'+path_ants)
            restart_terminal = 1

    # check if C3D is installed
    print_line('Check if c3d is installed ..................... ')
    output = ''
    if os_running == 'osx':
        # in OSX, c3d is typically installed under /Applications
        (status, output) = commands.getstatusoutput('find /Applications -name "c3d" -type f -print -quit 2>/dev/null')
    if not output:
        # if it is not OSX, or if it is OSX and it did not find c3d yet
        (status, output) = commands.getstatusoutput('find / -name "c3d" -type f -print -quit 2>/dev/null')
    if output:
        print_ok()
        path_c3d = os.path.dirname(output)
        print '  '+path_c3d
    else:
        print_fail()
        print '  Please install it from there: http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.C3D'
        c3d_is_installed = 0
        install_software = 1

    # check if C3D is declared
    if c3d_is_installed:
        print_line('Check if c3d is declared ...................... ')
        (status, output) = commands.getstatusoutput('which c3d')
        if output:
            print_ok()
        else:
            print_warning()
            print '  c3d is not declared! Modifying .bash_profile ...'
            add_bash_profile('#C3D (added on '+time.strftime("%Y-%m-%d")+')\n' \
                'PATH=${PATH}:'+path_c3d)
            restart_terminal = 1

    if install_software:
        print '\nDone! Please install the required software, then run this script again.'
    elif restart_terminal:
        print '\nDone! Please restart your Terminal for changes to take effect.'
    else:
        print '\nDone! Everything is in order :-)'
    print


# Print without new carriage return
# ==========================================================================================
def print_line(string):
    import sys
    sys.stdout.write(string)
    sys.stdout.flush()


def print_ok():
    print "[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]"


def print_warning():
    print "[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]"


def print_fail():
    print "[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]"
    

def add_bash_profile(string):
    from os.path import expanduser
    home = expanduser("~")
    with open(home+"/.bash_profile", "a") as file_bash:
    # with open("test.txt", "a") as file_bash:
        file_bash.write("\n"+string)


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