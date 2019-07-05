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

from __future__ import print_function, absolute_import

import argparse

import sys
import io
import os
import re
import platform
import importlib

import sct_utils as sct


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.create_log_file = 0
        self.complete_test = 0


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
        'pyqt5': ('PyQt5', False),
        'Keras': ('keras', True),
        'futures': ("concurrent.futures", False),
        'opencv': ('cv2', False),
        'mkl-service': (None, False),
        'pytest-cov': ('pytest_cov', False),
        'urllib3[secure]': ('urllib3', False),
        'git+https://github.com/jcohenadad/nipy.git': ('nipy', False),
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
    print("SCT info:")
    print("- version: {}".format(sct.__version__))
    print("- path: {0}".format(sct.__sct_dir__))

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

    # use variable "verbose" when calling sct.run for more clarity
    verbose = complete_test

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
    if platform_running.find('darwin') != -1:
        os_running = 'osx'
    elif platform_running.find('linux') != -1:
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
    if sct.__sct_dir__ in path_python:
        print_ok()
        print('  Using bundled python {} at {}'.format(sys.version, path_python))
    else:
        print_warning()
        print('  Using system python which is unsupported: {}'.format(path_python))

    # check if data folder is empty
    print_line('Check if data are installed')
    if os.path.isdir(sct.__data_dir__):
        print_ok()
    else:
        print_fail()

    for dep_pkg, dep_ver_spec in get_dependencies():
        if dep_ver_spec is None:
            print_line('Check if %s is installed' % (dep_pkg))
        else:
            print_line('Check if %s (%s) is installed' % (dep_pkg, dep_ver_spec))

        try:
            module_name, suppress_stderr = resolve_module(dep_pkg)
            module = module_import(module_name, suppress_stderr)
            version = getattr(module, "__version__", getattr(module, "__VERSION__", None))

            if dep_ver_spec is None and version is not None:
                ver_pip_setup = dict(get_dependencies(os.path.join(sct.__sct_dir__, "requirements.txt"))).get(dep_pkg, None)
                if ver_pip_setup is None:
                    print_ok(more=(" (%s)" % version))
                elif ver_pip_setup is not None and version.startswith(ver_pip_setup):
                    print_ok(more=(" (%s)" % version))
                else:
                    print_warning(more=(" (%s != %s reference version))" % (version, ver_pip_setup)))

            elif dep_ver_spec == version:
                print_ok()
            else:
                print_warning(more=(" (%s != %s mandated version))" % (version, dep_ver_spec)))

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
    status, output = sct.run('isct_propseg', verbose=0, raise_exception=False, is_sct_binary=True)
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
            print_fail()

    print('')
    sys.exit(e + install_software)


# print without carriage return
# ==========================================================================================
def print_line(string):
    sys.stdout.write(string.ljust(52, '.'))
    sys.stdout.flush()


def print_ok(more=None):
    print("[{}OK{}]{}".format(bcolors.OKGREEN, bcolors.ENDC, more if more is not None else ""))


def print_warning(more=None):
    print("[{}WARNING{}]{}".format(bcolors.WARNING, bcolors.ENDC, more if more is not None else ""))


def print_fail(more=None):
    print("[{}FAIL{}]{}".format(bcolors.FAIL, bcolors.ENDC, more if more is not None else ""))


def add_bash_profile(string):
    bash_profile = os.path.expanduser("~/bash_profile")
    with io.open(bash_profile, "a") as file_bash:
        file_bash.write("\n" + string)


def get_dependencies(requirements_txt=None):
    if requirements_txt is None:
        requirements_txt = os.path.join(sct.__sct_dir__, "requirements.txt")

    out = list()
    with io.open(requirements_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split('#')[0].strip()
            # check if conditions (separated by ";")
            if ';' in line:
                line, condition = [x.strip() for x in line.split(';')]
                # check if conditions apply
                if not _test_condition(condition):
                    break
            m = re.match("^(?P<pkg>\S+?)(==?(?P<ver>\S+))?\s*(#.*)?$", line)
            if m is None:
                print("WARNING: Invalid requirements.txt line: %s", line)
                continue
            pkg = m.group("pkg")
            try:
                ver = m.group("ver")
            except IndexError:
                ver = None
            out.append((pkg, ver))

    return out


def _test_condition(condition):
    """Test condition formatted in requirements"""
    # Define Environment markers (https://www.python.org/dev/peps/pep-0508/#environment-markers)
    os_name = os.name
    platform_machine = platform.machine()
    platform_release = platform.release()
    platform_system = platform.system()
    platform_version = platform.version()
    python_full_version = platform.python_version()
    platform_python_implementation = platform.python_implementation()
    python_version = platform.python_version()[:3]
    sys_platform = sys.platform
    # Test condition
    return eval(condition)


# ==========================================================================================
def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Check the installation and environment variables of the'
        ' toolbox and its dependencies.',
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-c",
        "--complete",
        help="Complete test.",
        action="store_true")

    return parser


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    # call main function
    main()
