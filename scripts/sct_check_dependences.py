#!/usr/bin/env python
#########################################################################################
#
# Check the installation and environment variables of the toolbox and its dependences.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-07-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: check chmod of binaries
# TODO: find another way to create log file. E.g. sct.print(). For color as well.
# TODO: manage .cshrc files
# TODO: add linux distrib when checking OS


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.create_log_file = 0
        self.complete_test = 0

import os
import sys
import commands
import time
import platform
import getopt
import sct_utils as sct


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


    # initialization
    fsl_is_working = 1
    # ants_is_installed = 1
    # sct_c3d_is_installed = 1
    install_software = 0
    e = 0
    restart_terminal = 0
    create_log_file = param.create_log_file
    file_log = 'sct_check_dependencies.log'
    complete_test = param.complete_test
    os_running = 'not identified'
    print

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hlc')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-c'):
            complete_test = 1
        elif opt in ('-l'):
            create_log_file = 1

    # use variable "verbose" when calling sct.run for more clarity
    verbose = complete_test
    
    # redirect to log file
    if create_log_file:
        orig_stdout = sys.stdout
        handle_log = file(file_log, 'w')
        sys.stdout = handle_log

    # complete test
    if complete_test:
        print sct.run('date', verbose)
        print sct.run('whoami', verbose)
        print sct.run('pwd', verbose)
        (status, output) = sct.run('more ~/.bash_profile', verbose)
        print output
        (status, output) = sct.run('more ~/.bashrc', verbose)
        print output

    # check if user is root (should not be!)
    if os.geteuid() == 0:
       print 'Looks like you are root. Please run this script without sudo. Exit program\n'
       sys.exit(2)

    # check OS
    print 'Check which OS is running... '
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    print '  '+os_running+' ('+platform.platform()+')'

    # check installation packages
    print 'Check which Python is running ... '
    print '  '+sys.executable

    # get path of the toolbox
    status, output = sct.run('echo $SCT_DIR', verbose)
    path_sct = output
    if complete_test:
        print (status, output), '\n'

    # fetch version of the toolbox
    print 'Fetch version of the Spinal Cord Toolbox... '
    with open (path_sct+"/version.txt", "r") as myfile:
        version_sct = myfile.read().replace('\n', '')
    print "  version: "+version_sct

    # check pillow
    print_line("Check if pillow is installed ")
    try:
        import PIL
        print_ok()
    except ImportError:
        print_fail()
        print '  pillow is not installed! Please install it via miniconda (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/)'
        install_software = 1

    # check numpy
    print_line('Check if numpy is installed ')
    try:
        import numpy
        print_ok()
    except ImportError:
        print_fail()
        print '  numpy is not installed! Please install it via miniconda (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/)'
        install_software = 1

    # check scipy
    print_line('Check if scipy is installed ')
    try:
        import scipy
        print_ok()
    except ImportError:
        print_fail()
        print '  scipy is not installed! Please install it via miniconda (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/)'
        install_software = 1

    # check sympy
    print_line('Check if sympy is installed ')
    try:
        import sympy
        print_ok()
    except ImportError:
        print_fail()
        print '  sympy is not installed! Please install it via miniconda (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/)'
        install_software = 1

    # check matplotlib
    print_line('Check if matplotlib is installed ')
    try:
        import matplotlib
        print_ok()
    except ImportError:
        print_fail()
        print '  matplotlib is not installed! Please install it via miniconda (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/)'
        install_software = 1

    # check nibabel
    print_line('Check if nibabel is installed ')
    try:
        import nibabel
        print_ok()
    except ImportError:
        print_fail()
        print '  nibabel is not installed! See instructions (https://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/)'
        install_software = 1

    # check if FSL is declared
    print_line('Check if FSL is declared ')
    cmd = 'which fsl'
    status, output = commands.getstatusoutput(cmd)
    if output:
        print_ok()
        path_fsl = output[:-7]
        print '  '+path_fsl
    else:
        print_fail()
        print '  FSL is not working!'
        e = 1

    if complete_test:
        print '>> '+cmd
        print (status, output), '\n'

    # check if FSL is installed
    if not fsl_is_working:
        print_line('Check if FSL is installed ')
        # check first under /usr for faster search
        (status, output) = commands.getstatusoutput('find /usr -name "flirt" -type f -print -quit 2>/dev/null')
        if output:
            print_ok()
            path_fsl = output[:-10]
            print '  '+path_fsl
        else:
            # some users might have installed it under /home, so check it...
            (status, output) = commands.getstatusoutput('find /home -name "flirt" -type f -print -quit 2>/dev/null')
            if output:
                print_ok()
                path_fsl = output[:-10]
                print '  '+path_fsl
            else:
                print_fail()
                print '  FSL does not seem to be installed! Install it from: http://fsl.fmrib.ox.ac.uk/'
                fsl_is_installed = 0
                install_software = 1

    # check ANTs
    print_line('Check which ANTs is running ')
    # (status, output) = commands.getstatusoutput('command -v sct_antsRegistration >/dev/null 2>&1 || { echo >&2 "nope";}')
    cmd = 'which sct_antsRegistration'
    status, output = commands.getstatusoutput(cmd)
    if output:
        print_ok()
        path_ants = output[:-20]
        print '  '+path_ants
    else:
        print_warning()
        print '  ANTs is not declared.'
        e = 1
    if complete_test:
        print '>> '+cmd
        print (status, output), '\n'

    # check if ANTs is compatible with OS
    print_line('Check ANTs compatibility with OS ')
    cmd = 'sct_antsRegistration'
    status, output = commands.getstatusoutput(cmd)
    if status in [0, 256]:
        print_ok()
    else:
        print_fail()
        e = 1
    if complete_test:
        print '>> '+cmd
        print (status, output), '\n'

    # check sct_c3d
    print_line('Check which sct_c3d is running ')
    # (status, output) = commands.getstatusoutput('command -v sct_c3d >/dev/null 2>&1 || { echo >&2 "nope";}')
    status, output = commands.getstatusoutput('which sct_c3d')
    if output:
        print_ok()
        path_sct_c3d = output[:-7]
        print '  '+path_sct_c3d
    else:
        print_warning()
        print '  sct_c3d is not installed or not declared.'
        install_software = 1
    if complete_test:
        print (status, output), '\n'

    # check sct_c3d compatibility with OS
    print_line('Check sct_c3d compatibility with OS ')
    (status, output) = commands.getstatusoutput('sct_c3d -h')
    if status in [0, 256]:
        print_ok()
    else:
        print_fail()
        install_software = 1
    if complete_test:
        print (status, output), '\n'

    # check PropSeg compatibility with OS
    print_line('Check PropSeg compatibility with OS ')
    (status, output) = commands.getstatusoutput('sct_propseg')
    if status in [0, 256]:
        print_ok()
    else:
        print_fail()
        e = 1
    if complete_test:
        print (status, output), '\n'

    # Check ANTs integrity
    print_line('Check integrity of ANTs output ')
    (status, output) = commands.getstatusoutput('isct_test_ants -v 0')
    if status in [0]:
        print_ok()
    else:
        print_fail()
        e = 1
    if complete_test:
        print (status, output), '\n'

    print
    
    # close log file
    if create_log_file:
        sys.stdout = orig_stdout
        handle_log.close()
        print "File generated: "+file_log+'\n'

    sys.exit(e + install_software)
    

# print without carriage return
# ==========================================================================================
def print_line(string):
    import sys
    sys.stdout.write(string + make_dot_lines(string))
    sys.stdout.flush()


# fill line with dots
# ==========================================================================================
def make_dot_lines(string):
    if len(string) < 52:
        dot_lines = '.'*(52 - len(string))
        return dot_lines
    else: return ''


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
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Check the installation and environment variables of the toolbox and its dependences.

USAGE
  """+os.path.basename(__file__)+"""

OPTIONAL ARGUMENTS
  -c                complete test.
  -l                generate log file.
  -h                print help.

EXAMPLE
  """+os.path.basename(__file__)+""" -l\n"""

    # exit program
    sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()