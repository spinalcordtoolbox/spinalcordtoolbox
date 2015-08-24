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
import platform
import getopt
import importlib

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
    # isct_c3d_is_installed = 1
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
        if os.path.isfile('~/.bash_profile'):
            (status, output) = sct.run('more ~/.bash_profile', verbose)
            print output
        if os.path.isfile('~/.bashrc'):
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

    # check RAM
    print 'Check RAM... '
    sct.checkRAM(os_running)

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

    version_requirements = get_version_requirements()

    # loop across python packages
    for i in version_requirements:
        if i == 'scikit-image':
            module = 'skimage'
        elif i == 'pillow':
            module = 'PIL'
        else:
            module = i
        print_line('Check if '+i+' ('+version_requirements.get(i)+') is installed')
        try:
            module = importlib.import_module(module)
            version = get_package_version(i)
            if check_package_version(version, version_requirements, i):
                print_ok()
            else:
                print_warning()
                print '  Detected version: '+version+'. Required version: '+version_requirements[i]
        except ImportError:
            print_fail()
            print '  '+i+' is not installed!'
            install_software = 1


    # check nibabel
    print_line('Check if nibabel is installed ')
    try:
        import nibabel
        # nibabel_version = get_package_version("nibabel")
        # if check_package_version(nibabel_version, version_requirements, "nibabel"):
        print_ok()
        # else:
        #    print_warning()
        #    print "nibabel version: "+nibabel_version+" detected. SCT requires version "+version_requirements["nibabel"]
    except ImportError:
        print_fail()
        print '  nibabel is not installed!'
        install_software = 1

    # check if git is installed
    print_line('Check if git is installed ')
    cmd = 'which git'
    status, output = commands.getstatusoutput(cmd)
    if output:
        print_ok()
    else:
        print_fail()
        print '  git is not installed.'
        print '  - To install it: http://git-scm.com/book/en/v1/Getting-Started-Installing-Git'
        e = 1

    if complete_test:
        print '>> '+cmd
        print (status, output), '\n'

    # check if ANTs is compatible with OS
    print_line('Check ANTs compatibility with OS ')
    cmd = 'isct_antsRegistration'
    status, output = commands.getstatusoutput(cmd)
    if status in [0, 256]:
        print_ok()
    else:
        print_fail()
        e = 1
    if complete_test:
        print '>> '+cmd
        print (status, output), '\n'

    # check isct_c3d compatibility with OS
    print_line('Check c3d compatibility with OS ')
    (status, output) = commands.getstatusoutput('isct_c3d -h')
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
    cmd_ants_test = 'isct_test_ants'
    # if not complete_test:
    #     cmd_ants_test += ' -v 0'
    (status, output) = commands.getstatusoutput(cmd_ants_test)
    if status in [0]:
        print_ok()
    else:
        print_fail()
        print output
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


def get_version_requirements():
    status, path_sct = sct.run('echo $SCT_DIR', 0)
    file = open(path_sct+"/install/requirements/requirementsConda.txt")
    dict = {}
    while True:
        line = file.readline()
        if line == "":
            break  # OH GOD HELP
        arg = line.split("==")
        dict[arg[0]] = arg[1].rstrip("\n")
    file.close()
    return dict


def get_package_version(package_name):
    cmd = "conda list "+package_name
    output = commands.getoutput(cmd)
    while True:
        line = output.split("\n")
        for i in line:
            if i.find(package_name) != -1:
                vers = i.split(' ')
                vers[:] = (value for value in vers if value != "")
                return vers[1]
        raise Exception("Could not find package: "+package_name)


def check_package_version(installed, required, package_name):
    if package_name in required:
        if required[package_name] == installed:
            return True
        return False


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
