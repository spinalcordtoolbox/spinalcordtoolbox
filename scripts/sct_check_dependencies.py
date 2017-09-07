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

# TODO: if fail, run with log and display message to send to sourceforge.
# TODO: check chmod of binaries
# TODO: find another way to create log file. E.g. sct.print(). For color as well.
# TODO: manage .cshrc files
# TODO: add linux distrib when checking OS


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.create_log_file = 0
        self.complete_test = 0


import sys

import os
import commands
import platform
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
    dipy_version = '0.10.0dev'

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
        handle_log = sct.ForkStdoutToFile(file_log)

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
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    print 'OS: ' + os_running + ' (' + platform.platform() + ')'

    # Check number of CPU cores
    from multiprocessing import cpu_count
    status, output = sct.run('echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0)
    print 'CPU cores: Available: ' + str(cpu_count()) + ', Used by SCT: ' + output

    # check RAM
    sct.checkRAM(os_running, 0)

    # get path of the toolbox
    path_sct = os.path.dirname(os.path.dirname(__file__))
    if path_sct is None:
        raise EnvironmentError("SCT_DIR, which is the path SCT install needs to be set")
    print ('SCT path: {0}'.format(path_sct))

    # fetch SCT version
    install_type, sct_commit, sct_branch, version_sct = sct.get_sct_version()
    print 'Installation type: git'
    print '  version: ' + version_sct
    print '  commit: ' + sct_commit
    print '  branch: ' + sct_branch

    # check if Python path is within SCT path
    print_line('Check Python path')
    path_python = sys.executable
    if path_sct in path_python:
        print_ok()
    else:
        print_fail()
        print '  Python path: ' + path_python

    # check if data folder is empty
    print_line('Check if data are installed')
    if os.listdir(path_sct + "/data"):
        print_ok()
    else:
        print_fail()

    # loop across python packages -- CONDA
    version_requirements = get_version_requirements()
    for i in version_requirements:
        # need to adapt import name and module name in specific cases
        if i == 'scikit-image':
            module = 'skimage'
        elif i == 'scikit-learn':
            module = 'sklearn'
        elif i == 'pyqt':
            module = 'PyQt4'
        else:
            module = i
        print_line('Check if ' + i + ' (' + version_requirements.get(i) + ') is installed')
        try:
            module = importlib.import_module(module)
            # get version
            try:
                version = module.__version__
            except:
                try:
                    version = module.__VERSION__
                except:
                    # skip if module doesn't have __version__ nor __VERSION__ (e.g., xlutils)
                    version = version_requirements[i]
            # check if version matches requirements
            if check_package_version(version, version_requirements, i):
                print_ok()
            else:
                print_warning()
                print '  Detected version: ' + version + '. Required version: ' + version_requirements[i]
        except ImportError:
            print_fail()
            install_software = 1

    # loop across python packages -- PIP
    version_requirements_pip = get_version_requirements_pip()
    for i in version_requirements_pip:
        module = i
        print_line('Check if ' + i + ' (' + version_requirements_pip.get(i) + ') is installed')
        try:
            module = importlib.import_module(module)
            # get version
            version = module.__version__
            # check if version matches requirements
            if check_package_version(version, version_requirements_pip, i):
                print_ok()
            else:
                print_warning()
                print '  Detected version: ' + version + '. Required version: ' + version_requirements_pip[i]
        except ImportError:
            print_fail()
            install_software = 1

    # CHECK DEPENDENT MODULES (installed by nibabel/dipy):
    print_line('Check if numpy is installed')
    try:
        importlib.import_module('numpy')
        print_ok()
    except ImportError:
        print_fail()
        install_software = 1
    print_line('Check if scipy is installed')
    try:
        importlib.import_module('scipy')
        print_ok()
    except ImportError:
        print_fail()
        install_software = 1
    print_line('Check if spinalcordtoolbox is installed')
    try:
        importlib.import_module('spinalcordtoolbox')
        print_ok()
    except ImportError:
        print_fail()
        install_software = 1

    # CHECK EXTERNAL MODULES:

    # Check if dipy is installed
    # print_line('Check if dipy ('+dipy_version+') is installed')
    # try:
    #     module = importlib.import_module('dipy')
    #     if module.__version__ == dipy_version:
    #         print_ok()
    #     else:
    #         print_warning()
    #         print '  Detected version: '+version+'. Required version: '+dipy_version
    # except ImportError:
    #     print_fail()
    #     install_software = 1

    # Check ANTs integrity
    print_line('Check ANTs compatibility with OS ')
    cmd = 'isct_test_ants'
    # here, cannot use commands.getstatusoutput because status is wrong (because of launcher)
    # status = os.system(cmd+" &> /dev/null")
    # status, output = sct.run(cmd, 0)
    # import subprocess
    # process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # status = subprocess.call(cmd, shell=True)
    # status = process.returncode
    (status, output) = commands.getstatusoutput(cmd)
    # from subprocess import call
    # status, output = call(cmd)
    # print status
    # print output
    # if status in [0, 256]:
    if status == 0:
        print_ok()
    else:
        print_fail()
        print output
        e = 1
    if complete_test:
        print '>> ' + cmd
        print (status, output), '\n'

    # check if ANTs is compatible with OS
    # print_line('Check ANTs compatibility with OS ')
    # cmd = 'isct_antsRegistration'
    # status, output = commands.getstatusoutput(cmd)
    # if status in [0, 256]:
    #     print_ok()
    # else:
    #     print_fail()
    #     e = 1
    # if complete_test:
    #     print '>> '+cmd
    #     print (status, output), '\n'

    # check PropSeg compatibility with OS
    print_line('Check PropSeg compatibility with OS ')
    (status, output) = commands.getstatusoutput('isct_propseg')
    if status in [0, 256]:
        print_ok()
    else:
        print_fail()
        print output
        e = 1
    if complete_test:
        print (status, output), '\n'

    # check if figure can be opened (in case running SCT via ssh connection)
    print_line('Check if figure can be opened')
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib.pyplot as plt
            plt.figure()
            plt.close()
            print_ok()
    except:
        print_fail()
        print sys.exc_info()

    print ''
    sys.exit(e + install_software)


# print without carriage return
# ==========================================================================================
def print_line(string):
    sys.stdout.write(string + make_dot_lines(string))
    sys.stdout.flush()


# fill line with dots
# ==========================================================================================
def make_dot_lines(string):
    if len(string) < 52:
        dot_lines = '.' * (52 - len(string))
        return dot_lines
    else:
        return ''


def print_ok():
    print "[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]"


def print_warning():
    print "[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]"


def print_fail():
    print "[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]"


def add_bash_profile(string):
    from os.path import expanduser
    home = expanduser("~")
    with open(home + "/.bash_profile", "a") as file_bash:
        file_bash.write("\n" + string)


def get_version_requirements():
    status, path_sct = sct.run('echo $SCT_DIR', 0)
    file = open(path_sct + "/install/requirements/requirementsConda.txt")
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
    file = open(path_sct + "/install/requirements/requirementsPip.txt")
    dict = {}
    while True:
        line = file.readline()
        if line == "":
            break  # OH GOD HELP
        arg = line.split("==")
        arg[0] = arg[0].split('[')[0]
        dict[arg[0]] = arg[1].rstrip("\n")
    file.close()
    return dict


def get_package_version(package_name):
    cmd = "conda list " + package_name
    output = commands.getoutput(cmd)
    while True:
        line = output.split("\n")
        for i in line:
            if i.find(package_name) != -1:
                vers = i.split(' ')
                vers[:] = (value for value in vers if value != "")
                return vers[1]
        raise Exception("Could not find package: " + package_name)


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
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    # call main function
    main()
