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
# TODO: find another way to create log file. E.g. print(). For color as well.
# TODO: manage .cshrc files

import sys
import io
import os
import platform
import importlib
import warnings
import psutil
import traceback

import requirements

from spinalcordtoolbox.utils.shell import SCTArgumentParser
from spinalcordtoolbox.utils.sys import sct_dir_local_path, init_sct, run_proc, __version__, __sct_dir__, __data_dir__, set_loglevel


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


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


def resolve_module(framework_name):
    """This function will resolve the framework name
    to the module name in cases where it is different.

    :param framework_name: the name of the framework.
    :return: the tuple (module name, supress stderr).
    """
    # Framework name : (module name, suppress stderr)
    modules_map = {
        'futures': ('concurrent.futures', False),
        'requirements-parser': ('requirements', False),
        'scikit-image': ('skimage', False),
        'scikit-learn': ('sklearn', False),
        'pyqt5': ('PyQt5.QtCore', False),  # Importing Qt instead PyQt5 to be able to catch this issue #2523
        'Keras': ('keras', True),
        'futures': ("concurrent.futures", False),
        'opencv': ('cv2', False),
        'mkl-service': (None, False),
        'pytest-cov': ('pytest_cov', False),
        'urllib3[secure]': ('urllib3', False),
        'pytest-xdist': ('xdist', False),
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


def get_version(module):
    """
    Get module version. This function is required due to some exceptions in fetching module versions.
    :param module: the module to get version from
    :return: string: the version of the module
    """
    if module.__name__ == 'PyQt5.QtCore':
        # Unfortunately importing PyQt5.Qt makes sklearn import crash on Ubuntu 14.04 (corresponding to Debian's jessie)
        # so we don't display the version for this distros.
        # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2522#issuecomment-559310454
        if 'jessie' in platform.platform():
            version = None
        else:
            from PyQt5.Qt import PYQT_VERSION_STR
            version = PYQT_VERSION_STR
    else:
        version = getattr(module, "__version__", getattr(module, "__VERSION__", None))
    return version


def print_line(string):
    """print without carriage return"""
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
        requirements_txt = sct_dir_local_path("requirements.txt")

    requirements_txt = open(requirements_txt, "r", encoding="utf-8")

    # workaround for https://github.com/davidfischer/requirements-parser/issues/39
    warnings.filterwarnings(action='ignore', module='requirements')

    for req in requirements.parse(requirements_txt):
        if ';' in req.line:  # handle environment markers; TODO: move this upstream into requirements-parser
            condition = req.line.split(';', 1)[-1].strip()
            if not _test_condition(condition):
                continue
        pkg = req.name
        # TODO: just return req directly and make sure caller can deal with fancier specs
        ver = dict(req.specs).get("==", None)
        yield pkg, ver


def get_parser():
    parser = SCTArgumentParser(
        description='Check the installation and environment variables of the toolbox and its dependencies.'
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-complete',
        help="Complete test.",
        action="store_true")
    optional.add_argument(
        "-short",
        help="Short test. Only shows SCT version, CPU cores and RAM available.",
        action="store_true")

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = complete_test = arguments.complete
    set_loglevel(verbose=verbose)

    print("SCT info:")
    print("- version: {}".format(__version__))
    print("- path: {0}".format(__sct_dir__))

    # initialization
    install_software = 0
    e = 0
    os_running = 'not identified'

    # complete test
    if complete_test:
        print(run_proc('date', verbose))
        print(run_proc('whoami', verbose))
        print(run_proc('pwd', verbose))
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
    print('CPU cores: Available: {}, Used by ITK functions: {}'.format(psutil.cpu_count(), int(os.getenv('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0))))

    ram = psutil.virtual_memory()
    factor_MB = 1024 * 1024
    print('RAM: Total: {}MB, Used: {}MB, Available: {}MB'.format(ram.total // factor_MB, ram.used // factor_MB, ram.available // factor_MB))

    if arguments.short:
        sys.exit()

    # check if Python path is within SCT path
    print_line('Check Python executable')
    path_python = sys.executable
    if __sct_dir__ in path_python:
        print_ok()
        print('  Using bundled python {} at {}'.format(sys.version, path_python))
    else:
        print_warning()
        print('  Using system python which is unsupported: {}'.format(path_python))

    # check if data folder is empty
    print_line('Check if data are installed')
    if os.path.isdir(__data_dir__):
        print_ok()
    else:
        print_fail()

    # Import matplotlib.pyplot here (before PyQt can be imported) in order to mitigate a libgcc error
    # See also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3511#issuecomment-912167649
    import matplotlib.pyplot as plt

    for dep_pkg, dep_ver_spec in get_dependencies():
        if dep_ver_spec is None:
            print_line('Check if %s is installed' % (dep_pkg))
        else:
            print_line('Check if %s (%s) is installed' % (dep_pkg, dep_ver_spec))

        try:
            module_name, suppress_stderr = resolve_module(dep_pkg)
            module = module_import(module_name, suppress_stderr)
            version = get_version(module)

            if dep_ver_spec is not None and version is not None and dep_ver_spec != version:
                print_warning(more=(" (%s != %s mandated version))" % (version, dep_ver_spec)))
            elif version is not None:
                print_ok(more=(" (%s)" % version))
            else:
                print_ok()

        except Exception as err:
            print_fail()
            print(f"An error occured while importing module {dep_pkg} -> {err}")
            print(f"Full traceback: {traceback.format_exc()}")
            install_software = 1

    print_line('Check if spinalcordtoolbox is installed')
    try:
        importlib.import_module('spinalcordtoolbox')
        print_ok()
    except ImportError:
        print_fail("Unable to import spinalcordtoolbox module.")
        install_software = 1

    # Check ANTs integrity
    print_line('Check ANTs compatibility with OS ')
    cmd = 'isct_test_ants'
    status, output = run_proc(cmd, verbose=0, raise_exception=False)
    if status == 0:
        print_ok()
    else:
        print_fail()
        print(output)
        e = 1
    if complete_test:
        print('>> ' + cmd)
        print((status, output), '\n')

    # check PropSeg compatibility with OS
    print_line('Check PropSeg compatibility with OS ')
    status, output = run_proc('isct_propseg', verbose=0, raise_exception=False, is_sct_binary=True)
    if status in (0, 1):
        print_ok()
    else:
        print_fail()
        print(output)
        e = 1
    if complete_test:
        print((status, output), '\n')

    print_line('Check if figure can be opened with matplotlib')
    try:
        import matplotlib
        # If matplotlib is using a GUI backend, the default 'show()` function will be overridden
        # See: https://github.com/matplotlib/matplotlib/issues/20281#issuecomment-846467732
        fig = plt.figure()  # NB: `plt` was imported earlier in the script to avoid a libgcc error
        if getattr(fig.canvas.manager.show, "__func__", None) != matplotlib.backend_bases.FigureManagerBase.show:
            print_ok(f" (Using GUI backend: '{matplotlib.get_backend()}')")
        else:
            print_fail(f" (Using non-GUI backend '{matplotlib.get_backend()}')")
    except Exception as err:
        print_fail()
        print(err)

    print_line('Check if figure can be opened with PyQt')
    if sys.platform == "linux" and 'DISPLAY' not in os.environ:
        print_fail(" ($DISPLAY not set on X11-supporting system)")
    else:
        try:
            from PyQt5.QtWidgets import QApplication, QLabel
            _ = QApplication([])
            label = QLabel('Hello World!')
            label.show()
            label.close()
            print_ok()
        except Exception as err:
            print_fail()
            print(err)

    # Check version of FSLeyes
    print_line('Check FSLeyes version')
    cmd = 'fsleyes --version'
    status, output = run_proc(cmd, verbose=0, raise_exception=False)
    # Exit code 0 - command has run successfully
    if status == 0:
        # Fetch only version number (full output of 'fsleyes --version' is 'fsleyes/FSLeyes version 0.34.2')
        fsleyes_version = output.split()[2]
        print_ok(more=(" (%s)" % fsleyes_version))
    # Exit code 126 - Command invoked cannot execute (permission problem or command is not an executable)
    elif status == 126:
        print('Command not executable. Please check permissions of fsleyes command.')
    # Exit code 127 - Command not found (possible problem with $PATH)
    elif status == 127:
        print('Command not found. If you installed FSLeyes as part of FSL package, please check that FSL is included '
              'in $PATH variable. If you installed FSLeyes using conda environment, make sure that the environment is '
              'activated. If you do not have FSLeyes installed, consider its installation to easily visualize '
              'processing outputs and/or to use SCT within FSLeyes. More info at: '
              'https://spinalcordtoolbox.com/en/latest/user_section/fsleyes.html')
    # All other exit codes
    else:
        print(f'Exit code {status} occurred. Please report this issue on SCT GitHub: '
              f'https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues')
        if complete_test:
            print(output)

    print('')
    sys.exit(e + install_software)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
