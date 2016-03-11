#!/usr/bin/env python
#########################################################################################
#
# Check the installation and environment variables of the toolbox and its dependencies.
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
from msct_parser import Parser


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
    install_software = 0
    e = 0
    restart_terminal = 0
    create_log_file = param.create_log_file
    file_log = 'sct_check_dependencies.log'
    complete_test = param.complete_test
    os_running = 'not identified'
    print

    # Check input parameters
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    if '-c' in arguments:
        complete_test = 1
    if '-log' in arguments:
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

    # check OS
    print 'Check which OS is running... '
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    print '.. '+os_running+' ('+platform.platform()+')'

    # Check number of CPU cores
    print 'Check number of CPU cores...'
    from multiprocessing import cpu_count
    print '.. Available: ' + str(cpu_count())
    status, output = sct.run('echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0)
    print '.. Used by SCT: ' + output

    # check RAM
    print 'Check RAM... '
    sct.checkRAM(os_running)

    # check installation packages
    print 'Check which Python is running...'
    print '.. '+sys.executable

    # get path of the toolbox
    print 'Check SCT path...'
    path_sct = os.getenv("SCT_DIR")
    if path_sct is None :
        raise EnvironmentError("SCT_DIR, which is the path to the "
                               "Spinalcordtoolbox install needs to be set")
    print ('.. {0}'.format(path_sct))

    # fetch version of the toolbox
    print 'Check SCT version... '
    with open (path_sct+"/version.txt", "r") as myfile:
        version_sct = myfile.read().replace('\n', '')
    print ".. "+version_sct

    # loop across python packages -- CONDA
    version_requirements = get_version_requirements()
    for i in version_requirements:
        if i == 'scikit-image':
            module = 'skimage'
        else:
            module = i
        print_line('Check if '+i+' ('+version_requirements.get(i)+') is installed')
        try:
            module = importlib.import_module(module)
            # get version
            version = module.__version__
            # check if version matches requirements
            if check_package_version(version, version_requirements, i):
                print_ok()
            else:
                print_warning()
                print '  Detected version: '+version+'. Required version: '+version_requirements[i]
        except ImportError:
            print_fail()
            install_software = 1

    # loop across python packages -- PIP
    version_requirements_pip = get_version_requirements_pip()
    for i in version_requirements_pip:
        module = i
        print_line('Check if '+i+' ('+version_requirements_pip.get(i)+') is installed')
        try:
            module = importlib.import_module(module)
            # get version
            version = module.__version__
            # check if version matches requirements
            if check_package_version(version, version_requirements_pip, i):
                print_ok()
            else:
                print_warning()
                print '  Detected version: '+version+'. Required version: '+version_requirements_pip[i]
        except ImportError:
            print_fail()
            install_software = 1

    # CHECK EXTERNAL MODULES:
    # Check if ornlm is installed
    print_line('Check if ornlm is installed')
#    sys.path.append(path_sct + '/external/denoise/ornlm')  # append to PYTHONPATH
    try:
        importlib.import_module('ornlm')
        print_ok()
    except ImportError:
        print_fail()
        install_software = 1

    # Check if dipy is installed
    print_line('Check if dipy is installed')
    try:
        importlib.import_module('dipy')
        print_ok()
    except ImportError:
        print_fail()
        install_software = 1

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

    # close log file
    if create_log_file:
        sys.stdout = orig_stdout
        handle_log.close()
        print "File generated: "+file_log+'\n'

    print ''
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

def get_version_requirements_pip():
    status, path_sct = sct.run('echo $SCT_DIR', 0)
    file = open(path_sct+"/install/requirements/requirementsPip.txt")
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


# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Check the installation and environment variables of the'
                                 ' toolbox and its dependencies.')
    parser.add_option(name="-c",
                      description="Complete test.",
                      mandatory=False)
    parser.add_option(name="-log",
                      description="Generate log file.",
                      mandatory=False)
    parser.add_option(name="-l",
                      type_value=None,
                      description="Generate log file.",
                      deprecated_by="-log",
                      mandatory=False)
    return parser


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
