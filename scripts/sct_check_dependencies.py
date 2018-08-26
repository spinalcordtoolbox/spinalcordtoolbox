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

from __future__ import print_function, absolute_import

# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.create_log_file = 0
        self.complete_test = 0

import argparse

import sys
import io
import os
import platform
import importlib

import sct_utils as sct


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def resolve_module(framework_name):
    """This function will resolve the framework name
    to the module name in cases where it is different.

    :param framework_name: the name of the framework.
    :return: the tuple (module name, supress stderr).
    """
    # Framework name : (module name, suppress stderr)
    modules_map = {
        'futures': ('concurrent.futures', False),
        'scikit-image': ('skimage', False),
        'scikit-learn': ('sklearn', False),
        'pyqt': ('PyQt4', False),
        'Keras': ('keras', True),
        'futures': ("concurrent.futures", False),
        'opencv': ('cv2', False),
        'mkl-service': (None, False),
        'pytest-cov': ('pytest_cov', False),
    }

    try:
        return modules_map[framework_name]
    except KeyError:
        return (framework_name, False)


def module_import(module_name, suppress_stderr=False):
    """Import a module using importlib.

    :param module_name: the name of the module.
    :param suppress_stderr: if the stderr should be suppressed.
    :return: the imported module.
    """
    if suppress_stderr:
        original_stderr = sys.stderr
        if sys.hexversion < 0x03000000:
            sys.stderr = io.BytesIO()
        else:
            sys.stderr = io.TextIOWrapper(io.BytesIO(), sys.stderr.encoding)
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            sys.stderr = original_stderr
            raise
        else:
            sys.stderr = original_stderr

    else:
        module = importlib.import_module(module_name)

    return module


# MAIN
# ==========================================================================================
def main():
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
    print("SCT info:")
    print("- version: {}".format(sct.__version__))
    print("- path: {0}".format(path_sct))

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
    arguments = parser.parse_args()
    if arguments.complete:
        complete_test = 1
    if arguments.generate_log:
        create_log_file = 1

    # use variable "verbose" when calling sct.run for more clarity
    verbose = complete_test

    # redirect to log file
    if create_log_file:
        handle_log = sct.ForkStdoutToFile(file_log)

    # complete test
    if complete_test:
        print(sct.run('date', verbose))
        print(sct.run('whoami', verbose))
        print(sct.run('pwd', verbose))
        bash_profile = os.path.expanduser("~/.bash_profile")
        if os.path.isfile(bash_profile):
            with io.open(bash_profile, "r") as f:
                print(f.read())
        bashrc = os.path.expanduser("~/.bashrc")
        if os.path.isfile(bashrc):
            with io.open(bashrc, "r") as f:
                print(f.read())

    # check OS
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'

    print('OS: ' + os_running + ' (' + platform.platform() + ')')

    # Check number of CPU cores
    from multiprocessing import cpu_count
    output = int(os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0))
    print('CPU cores: Available: {}, Used by SCT: {}'.format(cpu_count(), output))

    # check RAM
    sct.checkRAM(os_running, 0)

    # check if Python path is within SCT path
    print_line('Check Python executable')
    path_python = sys.executable
    if path_sct in path_python:
        print_ok()
        print('  Using bundled python {} at {}'.format(sys.version, path_python))
    else:
        print_warning()
        print('  Using system python which is unsupported: {}'.format(path_python))

    # check if data folder is empty
    print_line('Check if data are installed')
    if os.listdir(os.path.join(path_sct, "data")):
        print_ok()
    else:
        print_fail()

    # loop across python packages -- CONDA
    version_requirements = get_version_requirements()
    for i in version_requirements:
        module_name, suppress_stderr = resolve_module(i)
        if module_name is None:
            continue
        print_line('Check if %s (%s) is installed' % (i, version_requirements.get(i)))
        try:
            module = module_import(module_name, suppress_stderr)
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
                print('  Detected version: ' + version + '. Required version: ' + version_requirements[i])
        except ImportError:
            print_fail()
            install_software = 1

    # loop across python packages -- PIP
    version_requirements_pip = get_version_requirements_pip()
    for i in version_requirements_pip:

        module_name, suppress_stderr = resolve_module(i)

        print_line('Check if ' + i + ' (' + version_requirements_pip.get(i) + ') is installed')
        try:
            module = module_import(module_name, suppress_stderr)
            if module_name in ("raven",):
                version = module.VERSION
            else:
                try:
                    version = module.__version__
                except AttributeError:
                    # Futures package as no embedded version info
                    version = version_requirements_pip[i]

            # check if version matches requirements
            if check_package_version(version, version_requirements_pip, i):
                print_ok()
            else:
                print_warning()
                print('  Detected version: ' + version + '. Required version: ' + version_requirements_pip[i])
        except ImportError:
            print_fail()
            install_software = 1

    # CHECK DEPENDENT MODULES (installed by nibabel/dipy):
    sys.stdout.write('Check if numpy is installed')
    sys.stdout.flush()
    try:
        np = importlib.import_module('numpy')
        sys.stdout.write(' ({})'.format(np.__version__).ljust(25, '.'))
        print_ok()
    except ImportError:
        sys.stdout.write(' (........................')
        print_fail()
        install_software = 1
    sys.stdout.write('Check if scipy is installed')
    sys.stdout.flush()
    try:
        sp = importlib.import_module('scipy')
        sys.stdout.write(' ({})'.format(sp.__version__).ljust(25, '.'))
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
    #         print('  Detected version: '+version+'. Required version: '+dipy_version)
    # except ImportError:
    #     print_fail()
    #     install_software = 1

    # Check ANTs integrity
    print_line('Check ANTs compatibility with OS ')
    cmd = 'isct_test_ants'
    status, output = sct.run(cmd, verbose=0, raise_exception=False)
    if status == 0:
        print_ok()
    else:
        print_fail()
        print(output)
        e = 1
    if complete_test:
        print('>> ' + cmd)
        print((status, output), '\n')

    # check if ANTs is compatible with OS
    # print_line('Check ANTs compatibility with OS ')
    # cmd = 'isct_antsRegistration'
    # status, output = sct.run(cmd)
    # if status in [0, 256]:
    #     print_ok()
    # else:
    #     print_fail()
    #     e = 1
    # if complete_test:
    #     print('>> '+cmd)
    #     print((status, output), '\n')

    # check PropSeg compatibility with OS
    print_line('Check PropSeg compatibility with OS ')
    status, output = sct.run('isct_propseg', verbose=0, raise_exception=False)
    if status in (0, 1):
        print_ok()
    else:
        print_fail()
        print(output)
        e = 1
    if complete_test:
        print((status, output), '\n')

    # check if figure can be opened (in case running SCT via ssh connection)
    print_line('Check if figure can be opened')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import matplotlib.pyplot as plt
        try:
            plt.figure()
            plt.close()
            print_ok()
        except Exception:
            print("Please try manually setting environment variable $MPLBACKEND=agg")
            print_fail()

    print('')
    sys.exit(e + install_software)


# print without carriage return
# ==========================================================================================
def print_line(string):
    sys.stdout.write(string.ljust(52, '.'))
    sys.stdout.flush()


def print_ok():
    print("[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]")


def print_warning():
    print("[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]")


def print_fail():
    print("[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]")


def add_bash_profile(string):
    bash_profile = os.path.expanduser("~/bash_profile")
    with io.open(bash_profile, "a") as file_bash:
        file_bash.write("\n" + string)


def get_version_requirements():
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
    file = open(os.path.join(path_sct, "install", "requirements", "requirementsConda.txt"))
    dict = {}
    while True:
        line = file.readline().rstrip()
        if line == "":
            break  # OH GOD HELP
        arg = line.split("==")
        if len(arg) == 1:
            dict[arg[0]] = None
        else:
            dict[arg[0]] = arg[1]
    file.close()
    return dict


def get_version_requirements_pip():
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
    file = open(os.path.join(path_sct, "install", "requirements", "requirementsSetup.txt"))
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


def check_package_version(installed, required, package_name):
    if package_name in required:
        if required[package_name] is None:
            return True
        if required[package_name] == installed:
            return True
        return False


# ==========================================================================================
def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Check the installation and environment variables of the'
        ' toolbox and its dependencies.',
    )

    parser.add_argument("--complete", "-c",
                        help="Complete test.",
                        action="store_true",
                        )

    parser.add_argument("--generate-log", "-log", "-l",
                        help="Generate log file.",
                        action="store_true",
                        )

    return parser


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    # call main function
    main()
