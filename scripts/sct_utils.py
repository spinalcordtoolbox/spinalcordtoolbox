#!/usr/bin/env python
#########################################################################################
#
# Module containing several useful functions.
#
# PLEASE!! SORT FUNCTIONS ALPHABETICALLY!
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-07-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import print_function, division, absolute_import

import sys, io, os, re, time, datetime, platform
import errno
import logging
import shutil
import subprocess
import tempfile
import pathlib

import numpy as np

# backwards compat
import spinalcordtoolbox.utils as utils


logger = logging.getLogger(__name__)

if os.getenv('SENTRY_DSN', None):
    # do no import if Sentry is not set (i.e., if variable SENTRY_DSN is not defined)
    import raven

if sys.hexversion < 0x03030000:
    import pipes
    def list2cmdline(lst):
        return " ".join(pipes.quote(x) for x in lst)
else:
    import shlex
    def list2cmdline(lst):
        return " ".join(shlex.quote(x) for x in lst)


from spinalcordtoolbox import __version__, __sct_dir__, __data_dir__
from spinalcordtoolbox.utils import check_exe, sct_dir_local_path


def init_sct(log_level=1, update=False):
    """
    Initialize the sct for typical terminal usage
    :param log_level: int: 0: warning, 1: info, 2: debug.
    :param update: Bool: If True, only update logging log level. Otherwise, set logging + Sentry.
    :return:
    """
    dict_log_levels = {0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}

    def _format_wrap(old_format):
        def _format(record):
            res = old_format(record)
            if record.levelno >= logging.ERROR:
                res = "\x1B[31;1m{}\x1B[0m".format(res)
            elif record.levelno >= logging.WARNING:
                res = "\x1B[33m{}\x1B[0m".format(res)
            else:
                pass
            return res
        return _format

    # Set logging level for logger and increase level for global config (to avoid logging when calling child functions)
    logger.setLevel(getattr(logging, dict_log_levels[log_level]))
    logging.root.setLevel(getattr(logging, dict_log_levels[log_level]))

    if not update:
        # Initialize logging
        hdlr = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter()
        fmt.format = _format_wrap(fmt.format)
        hdlr.setFormatter(fmt)
        logging.root.addHandler(hdlr)

        # Sentry config
        init_error_client()
        if os.environ.get("SCT_TIMER", None) is not None:
            add_elapsed_time_counter()

        # Display SCT version
        logger.info('\n--\nSpinal Cord Toolbox ({})\n'.format(__version__))


def add_elapsed_time_counter():
    """
    """
    import atexit
    class Timer():
        def __init__(self):
            self._t0 = time.time()
        def atexit(self):
            print("Elapsed time: %.3f seconds" % (time.time()-self._t0))
    t = Timer()
    atexit.register(t.atexit)


def init_error_client():
    """ Send traceback to neuropoly servers

    :return:
    """
    if os.getenv('SENTRY_DSN'):
        logger.debug('Configuring sentry report')
        try:
            client = raven.Client(
             release=__version__,
             processors=(
              'raven.processors.RemoveStackLocalsProcessor',
              'raven.processors.SanitizePasswordsProcessor'),
            )
            server_log_handler(client)
            traceback_to_server(client)

            old_exitfunc = sys.exitfunc
            def exitfunc():
                sent_something = False
                try:
                    # implementation-specific
                    import atexit
                    for handler, args, kw in atexit._exithandlers:
                        if handler.__module__.startswith("raven."):
                            sent_something = True
                except:
                    pass
                old_exitfunc()
                if sent_something:
                    print("Note: you can opt out of Sentry reporting by editing the file ${SCT_DIR}/bin/sct_launcher and delete the line starting with \"export SENTRY_DSN\"")

            sys.exitfunc = exitfunc
        except raven.exceptions.InvalidDsn:
            # This could happen if sct staff change the dsn
            logger.debug('Sentry DSN not valid anymore, not reporting errors')


def traceback_to_server(client):
    """
        Send all traceback children of Exception to sentry
    """

    def excepthook(exctype, value, traceback):
        if issubclass(exctype, Exception):
            client.captureException(exc_info=(exctype, value, traceback))
        sys.__excepthook__(exctype, value, traceback)

    sys.excepthook = excepthook


def server_log_handler(client):
    """ Adds sentry log handler to the logger

    :return: the sentry handler
    """
    from raven.handlers.logging import SentryHandler

    sh = SentryHandler(client=client, level=logging.ERROR)

    # Don't send Sentry events for command-line usage errors
    old_emit = sh.emit
    def emit(self, record):
        if record.message.startswith("Command-line usage error:"):
            return
        return old_emit(record)

    sh.emit = lambda x: emit(sh, x)


    fmt = ("[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)d | "
            "%(message)s")
    formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")
    formatter.converter = time.gmtime
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return sh


# define class color
class bcolors(object):
    normal = '\033[0m'
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    magenta = '\033[95m'
    cyan = '\033[96m'
    bold = '\033[1m'
    underline = '\033[4m'
    @classmethod
    def colors(cls):
        return [v for k, v in cls.__dict__.items() if not k.startswith("_") and k is not "colors"]


def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension.

    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii

    Examples:

    - add_suffix(t2.nii, _mean) -> t2_mean.nii
    - add_suffix(t2.nii.gz, a) -> t2a.nii.gz
    """

    return utils.add_suffix(fname=fname, suffix=suffix)


#=======================================================================================================================
# run
#=======================================================================================================================
# Run UNIX command
def run_old(cmd, verbose=1):
    if verbose:
        printv(bcolors.blue + cmd + bcolors.normal)
    status, output = run(cmd)
    if status != 0:
        printv('\nERROR! \n' + output + '\nExit program.\n', 1, 'error')
    else:
        return status, output

def which_sct_binaries():
    """
    :return name of the sct binaries to use on this platform
    """

    if sys.platform.startswith("linux"):
        return "binaries_linux"
    else:
        return "binaries_osx"


def run(cmd, verbose=1, raise_exception=True, cwd=None, env=None, is_sct_binary=False):
    # if verbose == 2:
    #     printv(sys._getframe().f_back.f_code.co_name, 1, 'process')

    if cwd is None:
        cwd = os.getcwd()

    if env is None:
        env = os.environ

    if sys.hexversion < 0x03000000 and isinstance(cmd, unicode):
        cmd = str(cmd)


    if is_sct_binary:
        name = cmd[0] if isinstance(cmd, list) else cmd.split(" ", 1)[0]
        path = None
        #binaries_location_default = os.path.expanduser("~/.cache/spinalcordtoolbox-{}/bin".format(__version__)
        binaries_location_default = sct_dir_local_path("bin")
        for directory in (
         #binaries_location_default,
         sct_dir_local_path("bin"),
         ):
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                path = candidate
        if path is None:
            run(["sct_download_data", "-d", which_sct_binaries(), "-o", binaries_location_default])
            path = os.path.join(binaries_location_default, name)

        if isinstance(cmd, list):
            cmd[0] = path
        elif isinstance(cmd, str):
            rem = cmd.split(" ", 1)[1:]
            cmd = path if len(rem) == 0 else "{} {}".format(path, rem[0])

    if isinstance(cmd, str):
        cmdline = cmd
    else:
        cmdline = list2cmdline(cmd)


    if verbose:
        printv("%s # in %s" % (cmdline, cwd), 1, 'code')

    shell = isinstance(cmd, str)

    process = subprocess.Popen(cmd, shell=shell, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    output_final = ''
    while True:
        # Watch out for deadlock!!!
        output = process.stdout.readline().decode("utf-8")
        if output == '' and process.poll() is not None:
            break
        if output:
            if verbose == 2:
                printv(output.strip())
            output_final += output.strip() + '\n'

    status = process.returncode
    output = output_final.rstrip()

    # process.stdin.close()
    # process.stdout.close()
    # process.terminate()

    if status != 0 and raise_exception:
        raise RunError(output)

    return status, output


def display_open(file):
    """Print the syntax to open a file based on the platform."""
    if sys.platform == 'linux':
        printv('\nDone! To view results, type:')
        printv('xdg-open ' + file + '\n', verbose=1, type='info')
    elif sys.platform == 'darwin':
        printv('\nDone! To view results, type:')
        printv('open ' + file + '\n', verbose=1, type='info')
    else:
        printv('\nDone! To view results, open the following file:')
        printv(file + '\n', verbose=1, type='info')


def display_viewer_syntax(files, colormaps=[], minmax=[], opacities=[], mode='', verbose=1):
    """
    Print the syntax to open a viewer and display images for QC. To use default values, enter empty string: ''
    Parameters
    ----------
    files [list:string]: list of NIFTI file names
    colormaps [list:string]: list of colormaps associated with each file. Available colour maps: see dict_fsleyes
    minmax [list:string]: list of min,max brightness scale associated with each file. Separate with comma.
    opacities [list:string]: list of opacity associated with each file. Between 0 and 1.

    Returns
    -------
    None

    Example
    -------
    sct.display_viewer_syntax([file1, file2, file3])
    sct.display_viewer_syntax([file1, file2], colormaps=['gray', 'red'], minmax=['', '0,1'], opacities=['', '0.7'])
    """
    list_viewer = ['fsleyes', 'fslview_deprecated', 'fslview']  # list of known viewers. Can add more.
    dict_fslview = {'gray': 'Greyscale', 'red-yellow': 'Red-Yellow', 'blue-lightblue': 'Blue-Lightblue', 'red': 'Red',
                    'green': 'Green', 'random': 'Random-Rainbow', 'hsv': 'hsv', 'subcortical': 'MGH-Subcortical'}
    dict_fsleyes = {'gray': 'greyscale', 'red-yellow': 'red-yellow', 'blue-lightblue': 'blue-lightblue', 'red': 'red',
                    'green': 'green', 'random': 'random', 'hsv': 'hsv', 'subcortical': 'subcortical'}
    selected_viewer = None

    # find viewer
    exe_viewers = [viewer for viewer in list_viewer if check_exe(viewer)]
    if exe_viewers:
        selected_viewer = exe_viewers[0]
    else:
        return

    # loop across files and build syntax
    cmd = selected_viewer
    # add mode (only supported by fslview for the moment)
    if mode and selected_viewer in ['fslview', 'fslview_deprecated']:
        cmd += ' -m ' + mode
    for i in range(len(files)):
        # add viewer-specific options
        if selected_viewer in ['fslview', 'fslview_deprecated']:
            cmd += ' ' + files[i]
            if colormaps:
                if colormaps[i]:
                    cmd += ' -l ' + dict_fslview[colormaps[i]]
            if minmax:
                if minmax[i]:
                    cmd += ' -b ' + minmax[i]  # a,b
            if opacities:
                if opacities[i]:
                    cmd += ' -t ' + opacities[i]
        if selected_viewer in ['fsleyes']:
            cmd += ' ' + files[i]
            if colormaps:
                if colormaps[i]:
                    cmd += ' -cm ' + dict_fsleyes[colormaps[i]]
            if minmax:
                if minmax[i]:
                    cmd += ' -dr ' + ' '.join(minmax[i].split(','))  # a b
            if opacities:
                if opacities[i]:
                    cmd += ' -a ' + str(float(opacities[i]) * 100)  # in percentage
    cmd += ' &'
    # display
    if verbose:
        printv('\nDone! To view results, type:')
        printv(cmd + '\n', verbose=1, type='info')


def mkdir(path, verbose=1):
    """Create a folder, like os.mkdir
    """
    try:
        printv("mkdir %s" % (path), verbose=verbose, type="code")
        os.mkdir(path)
    except Exception as e:
        raise

def rm(path, verbose=1):
    """Remove a file, almost like os.remove
    """
    try:
        printv("rm %s" % (path), verbose=verbose, type="code")
        os.remove(path)
    except Exception as e:
        raise

def mv(src, dst, verbose=1):
    """Move a file from src to dst, almost like os.rename
    """
    try:
        printv("mv %s %s" % (src, dst), verbose=verbose, type="code")
        os.rename(src, dst)
    except Exception as e:
        raise

def copy(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise Exception("Couldn't find %s in %s (contents: %s)" \
         % (os.path.basename(src), folder, contents))
    try:
        printv("cp %s %s" % (src, dst), verbose=verbose, type="code")
        shutil.copy(src, dst)
    except Exception as e:
        if sys.hexversion < 0x03000000:
            if isinstance(e, shutil.Error) and "same file" in str(e):
                return
        else:
            if isinstance(e, shutil.SameFileError):
                return
        raise # Must be another error

def rmtree(folder, verbose=1):
    """Recursively remove folder, almost like shutil.rmtree
    """
    try:
        printv("rm -rf %s" % (folder), verbose=verbose, type="code")
        shutil.rmtree(folder, ignore_errors=True)
    except Exception as e:
        raise # Must be another error


#=======================================================================================================================
# check RAM usage
# work only on Mac OSX
#=======================================================================================================================
def checkRAM(os, verbose=1):
    if (os == 'linux'):
        status, output = run('grep MemTotal /proc/meminfo', 0)
        printv('RAM: ' + output)
        ram_split = output.split()
        ram_total = float(ram_split[1])
        status, output = run('free -m', 0)
        printv(output)
        return ram_total / 1024

    elif (os == 'osx'):
        status, output = run('hostinfo | grep memory', 0)
        printv('RAM: ' + output)
        ram_split = output.split(' ')
        ram_total = float(ram_split[3])

        # Get process info
        ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0].decode(sys.stdout.encoding)
        vm = subprocess.Popen(['vm_stat'], stdout=subprocess.PIPE).communicate()[0].decode(sys.stdout.encoding)

        # Iterate processes
        processLines = ps.split('\n')
        sep = re.compile(r'[\s]+')
        rssTotal = 0  # kB
        for row in range(1, len(processLines)):
            rowText = processLines[row].strip()
            rowElements = sep.split(rowText)
            try:
                rss = float(rowElements[0]) * 1024
            except:
                rss = 0  # ignore...
            rssTotal += rss

        # Process vm_stat
        vmLines = vm.split('\n')
        sep = re.compile(r':[\s]+')
        vmStats = {}
        for row in range(1, len(vmLines) - 2):
            rowText = vmLines[row].strip()
            rowElements = sep.split(rowText)
            vmStats[(rowElements[0])] = int(rowElements[1].strip(r'\.')) * 4096

        if verbose:
            printv('  Wired Memory:\t\t%d MB' % (vmStats["Pages wired down"] / 1024 / 1024))
            printv('  Active Memory:\t%d MB' % (vmStats["Pages active"] / 1024 / 1024))
            printv('  Inactive Memory:\t%d MB' % (vmStats["Pages inactive"] / 1024 / 1024))
            printv('  Free Memory:\t\t%d MB' % (vmStats["Pages free"] / 1024 / 1024))
            # printv('Real Mem Total (ps):\t%.3f MB' % ( rssTotal/1024/1024 ))

        return ram_total


def extract_fname(fpath):
    """
    Split a full path into a parent folder component, filename stem and extension.

    Note: for .nii.gz the extension is understandably .nii.gz, not .gz
    (``os.path.splitext()`` would want to do the latter, hence the special case).
    """
    parent, filename = os.path.split(fpath)
    if filename.endswith(".nii.gz"):
        stem, ext = filename[:-7], ".nii.gz"
    else:
        stem, ext = os.path.splitext(filename)
    return parent, stem, ext


#=======================================================================================================================
# get_absolute_path
#=======================================================================================================================
# Return the absolute path of a file or a directory
def get_absolute_path(fname):
    if os.path.isfile(fname) or os.path.isdir(fname):
        return os.path.realpath(fname)
    else:
        printv('\nERROR: ' + fname + ' does not exist. Exit program.\n', 1, 'error')

#=======================================================================================================================
# check_file_exist:  Check existence of a file or path
#=======================================================================================================================


def check_file_exist(fname, verbose=1):
    if fname[0] == '-':
        # fname should be a warping field that will be inverted, ignore the "-"
        fname_to_test = fname[1:]
    else:
        fname_to_test = fname
    if os.path.isfile(fname_to_test):
        if verbose:
            printv('  OK: ' + fname, verbose, 'normal')
        return True
    else:
        printv('\nERROR: The file ' + fname + ' does not exist. Exit program.\n', 1, 'error')
        return False


#=======================================================================================================================
# check_folder_exist:  Check existence of a folder.
#   Does not create it. If you want to create a folder, use create_folder
#=======================================================================================================================
def check_folder_exist(fname, verbose=1):
    if os.path.isdir(fname):
        printv('  OK: ' + fname, verbose, 'normal')
        return True
    else:
        printv('\nWarning: The directory ' + str(fname) + ' does not exist.\n', 1, 'warning')
        return False


#=======================================================================================================================
# check_write_permission:  Check existence of a folder.
#   Does not create it. If you want to create a folder, use create_folder
#=======================================================================================================================
def check_write_permission(fname, verbose=1):
    if os.path.isdir(fname):
        if os.path.isdir(fname):
            return os.access(fname, os.W_OK)
        else:
            printv('\nERROR: The directory ' + fname + ' does not exist. Exit program.\n', 1, 'error')
    else:
        path_fname, file_fname, ext_fname = extract_fname(os.path.abspath(fname))
        return os.access(path_fname, os.W_OK)


#=======================================================================================================================
# create_folder:  create folder (check if exists before creating it)
#   output: 0 -> folder created
#           1 -> folder already exist
#           2 -> permission denied
#=======================================================================================================================
def create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            return 0
        except OSError as e:
            if e.errno != errno.EEXIST:
                return 2
    else:
        return 1

#=======================================================================================================================
# check_dim
#=======================================================================================================================

def check_dim(fname, dim_lst=[3]):
    """
    Check if input dimension matches the input dimension requirements specified in the dim list.
    Example: to check if an image is 2D or 3D: check_dim(my_file, dim_lst=[2, 3])
    :param fname:
    :return: True or False
    """
    from spinalcordtoolbox.image import Image
    dim = Image(fname).hdr['dim'][:4]

    if not dim[0] in dim_lst:
        printv('\nERROR: File ' + fname + ' has {} dimensions. Authorized dimensions are: {}. '
                'Exit program.\n'.format(dim[0], dim_lst), 1, 'error')
        sys.exit(2)
    else:
        return True


#=======================================================================================================================
# find_file_within_folder
#=======================================================================================================================
def find_file_within_folder(fname, directory, seek_type='file'):
    """Find file (or part of file, e.g. 'my_file*.txt') within folder tree recursively - fname and directory must be
    strings
    seek_type: 'file' or 'dir' to look for either a file or a directory respectively."""
    import fnmatch

    all_path = []
    for root, dirs, files in os.walk(directory):
        if seek_type == 'dir':
            for folder in dirs:
                if fnmatch.fnmatch(folder, fname):
                    all_path.append(os.path.join(root, folder))
        else:
            for file in files:
                if fnmatch.fnmatch(file, fname):
                    all_path.append(os.path.join(root, file))
    return all_path


def tmp_create(basename=None, verbose=1):
    """Create temporary folder and return its path
    """
    prefix = "sct-%s-" % datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
    if basename:
        prefix += "%s-" % basename
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    printv('\nCreate temporary folder (%s)...' % tmpdir, verbose)
    return tmpdir


class TempFolder(object):
    """This class will create a temporary folder."""

    def __init__(self, verbose=0):
        self.path_tmp = tmp_create(verbose=verbose)
        self.previous_path = None

    def chdir(self):
        """This method will change the working directory to the temporary folder."""
        self.previous_path = os.getcwd()
        os.chdir(self.path_tmp)

    def chdir_undo(self):
        """This method will return to the previous working directory, the directory
        that was the state before calling the chdir() method."""
        if self.previous_path is not None:
            os.chdir(self.previous_path)

    def get_path(self):
        """Return the temporary folder path."""
        return self.path_tmp

    def copy_from(self, filename):
        """This method will copy a specified file to the temporary folder.

        :param filename: The filename to copy into the folder.
        """
        file_fname = os.path.basename(filename)
        copy(filename, self.path_tmp)
        return self.path_tmp + '/' + file_fname

    def cleanup(self):
        """Remove the created folder and its contents."""
        rmtree(self.path_tmp)


#=======================================================================================================================
# copy a nifti file to (temporary) folder and convert to .nii or .nii.gz
#=======================================================================================================================
def tmp_copy_nifti(fname, path_tmp, fname_out='data.nii', verbose=0):
    # tmp_copy_nifti('input.nii', path_tmp, 'raw.nii')
    path_fname, file_fname, ext_fname = extract_fname(fname)
    path_fname_out, file_fname_out, ext_fname_out = extract_fname(fname_out)

    run('cp ' + fname + ' ' + path_tmp + file_fname_out + ext_fname)
    if ext_fname_out == '.nii':
        run('fslchfiletype NIFTI ' + path_tmp + file_fname_out, 0)
    elif ext_fname_out == '.nii.gz':
        run('fslchfiletype NIFTI_GZ ' + path_tmp + file_fname_out, 0)


#=======================================================================================================================
# generate_output_file
#=======================================================================================================================
def generate_output_file(fname_in, fname_out, squeeze_data=True, verbose=1):
    """
    Copy fname_in to fname_out with a few convenient checks: make sure input file exists, if fname_out exists send a
    warning, if input and output NIFTI format are different (nii vs. nii.gz) convert by unzipping or zipping, and
    display nice message at the end.
    :param fname_in:
    :param fname_out:
    :param verbose:
    :return: fname_out
    """
    from sct_convert import convert
    path_in, file_in, ext_in = extract_fname(fname_in)
    path_out, file_out, ext_out = extract_fname(fname_out)
    # create output path (ignore if it already exists)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    # if input image does not exist, give error
    if not os.path.isfile(fname_in):
        printv('  ERROR: File ' + fname_in + ' is not a regular file. Exit program.', 1, 'error')
        sys.exit(2)
    # if input and output fnames are the same, do nothing and exit function
    if fname_in == fname_out:
        printv('  WARNING: fname_in and fname_out are the same. Do nothing.', verbose, 'warning')
        printv('  File created: ' + os.path.join(path_out, file_out + ext_out))
        return os.path.join(path_out, file_out + ext_out)
    # if fname_out already exists in nii or nii.gz format
    if os.path.isfile(os.path.join(path_out, file_out + ext_out)):
        printv('  WARNING: File ' + os.path.join(path_out, file_out + ext_out) + ' already exists. Deleting it...', 1, 'warning')
        os.remove(os.path.join(path_out, file_out + ext_out))
    if ext_in != ext_out:
        # Generate output file
        '''
        # TRY TO UNCOMMENT THIS LINES AND RUN IT IN AN OTHER STATION THAN EVANS (testing of sct_label_vertebrae and sct_smooth_spinalcord never stops with this lines on evans)
        if ext_in == '.nii.gz' and ext_out == '.nii':  # added to resolve issue #728
            run('gunzip -f ' + fname_in)
            os.rename(os.path.join(path_in, file_in + '.nii'), fname_out)
        else:
        '''
        convert(fname_in, fname_out, squeeze_data=squeeze_data, verbose=0)
    else:
        # Generate output file without changing the extension
        shutil.move(fname_in, fname_out)

    # # Move file to output folder (keep the same extension as input)
    # shutil.move(fname_in, path_out+file_out+ext_in)
    # # convert to nii (only if necessary)
    # if ext_out == '.nii' and ext_in != '.nii':
    #     convert(os.path.join(path_out, file_out+ext_in), os.path.join(path_out, file_out+ext_out))
    #     os.remove(os.path.join(path_out, file_out+ext_in))  # remove nii.gz file
    # # convert to nii.gz (only if necessary)
    # if ext_out == '.nii.gz' and ext_in != '.nii.gz':
    #     convert(os.path.join(path_out, file_out+ext_in), os.path.join(path_out, file_out+ext_out))
    #     os.remove(os.path.join(path_out, file_out+ext_in))  # remove nii file
    # display message
    printv('  File created: ' + os.path.join(path_out, file_out + ext_out), verbose)
    # if verbose:
    #     printv('  File created: '+ os.path.join(path_out, file_out+ext_out))
    return os.path.join(path_out, file_out + ext_out)


#=======================================================================================================================
# check_if_installed
#=======================================================================================================================
# check if dependant software is installed
def check_if_installed(cmd, name_software):
    status, output = run(cmd)
    if status != 0:
        printv('\nERROR: ' + name_software + ' is not installed.\nExit program.\n')
        sys.exit(2)


#=======================================================================================================================
# check_if_same_space
#=======================================================================================================================
# check if two images are in the same space and same orientation
def check_if_same_space(fname_1, fname_2):
    from spinalcordtoolbox.image import Image

    im_1 = Image(fname_1)
    im_2 = Image(fname_2)
    q1 = im_1.hdr.get_qform()
    q2 = im_2.hdr.get_qform()

    dec = int(np.abs(np.round(np.log10(np.min(np.abs(q1[np.nonzero(q1)]))))) + 1)
    dec = 4 if dec > 4 else dec
    return np.all(np.around(q1, dec) == np.around(q2, dec))


def printv(string, verbose=1, type='normal'):
    """
    Enables to print color-coded messages, depending on verbose status. Only use in command-line programs (e.g.,
    sct_propseg).
    """

    colors = {'normal': bcolors.normal, 'info': bcolors.green, 'warning': bcolors.yellow, 'error': bcolors.red,
              'code': bcolors.blue, 'bold': bcolors.bold, 'process': bcolors.magenta}

    if verbose:
        # The try/except is there in case stdout does not have isatty field (it did happen to me)
        try:
            # Print color only if the output is the terminal
            if sys.stdout.isatty():
                color = colors.get(type, bcolors.normal)
                print(color + string + bcolors.normal)
            else:
                print(string)
        except Exception as e:
            print(string)

#=======================================================================================================================
# sign
#=======================================================================================================================
# Get the sign of a number. Returns 1 if x>=0 and -1 if x<0
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


#=======================================================================================================================
# delete_nifti: delete nifti file(s)
#=======================================================================================================================
def delete_nifti(fname_in):
    # extract input file extension
    path_in, file_in, ext_in = extract_fname(fname_in)
    # delete nifti if exist
    if os.path.isfile(os.path.join(path_in, file_in + '.nii')):
        os.system('rm ' + os.path.join(path_in, file_in + '.nii'))
    # delete nifti if exist
    if os.path.isfile(os.path.join(path_in, file_in + '.nii.gz')):
        os.system('rm ' + os.path.join(path_in, file_in + '.nii.gz'))


def get_interpolation(program, interp):
    """
    Get syntax on interpolation field depending on program. Supported programs: ants, flirt, WarpImageMultiTransform
    :param program:
    :param interp:
    :return:
    """
    # TODO: check if field and program exists
    interp_program = ''
    # FLIRT
    if program == 'flirt':
        if interp == 'nn':
            interp_program = ' -interp nearestneighbour'
        elif interp == 'linear':
            interp_program = ' -interp trilinear'
        elif interp == 'spline':
            interp_program = ' -interp spline'
    # ANTs
    elif program == 'ants' or program == 'ants_affine' or program == 'isct_antsApplyTransforms' \
            or program == 'isct_antsSliceRegularizedRegistration' or program == 'isct_antsRegistration':
        if interp == 'nn':
            interp_program = ' -n NearestNeighbor'
        elif interp == 'linear':
            interp_program = ' -n Linear'
        elif interp == 'spline':
            interp_program = ' -n BSpline[3]'
    # check if not assigned
    if interp_program == '':
        printv('WARNING (' + os.path.basename(__file__) + '): interp_program not assigned. Using linear for ants_affine.', 1, 'warning')
        interp_program = ' -n Linear'
    # return
    return interp_program.strip().split()


#=======================================================================================================================
# template file dictionary
#=======================================================================================================================
def template_dict(template):
    """
    Dictionary of file names for template
    :param template:
    :return: dict_template
    """
    if template == 'PAM50':
        dict_template = {'t2': 'template/PAM50_t2.nii.gz',
                         't1': 'template/PAM50_t1.nii.gz'}
    return dict_template


class UnsupportedOs(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Os(object):
    '''Work out which platform we are running on'''

    def __init__(self):
        import os
        if os.name != 'posix':
            raise UnsupportedOs('We only support macOS/Linux')
        import platform
        self.os = platform.system().lower()
        self.arch = platform.machine()
        self.applever = ''

        if self.os == 'darwin':
            self.os = 'osx'
            self.vendor = 'apple'
            self.version = Version(platform.release())
            (self.applever, _, _) = platform.mac_ver()
            if self.arch == 'Power Macintosh':
                raise UnsupportedOs('We do not support PowerPC')
            self.glibc = ''
            self.bits = ''
        elif self.os == 'linux':
            if hasattr(platform, 'linux_distribution'):
                # We have a modern python (>2.4)
                (self.vendor, version, _) = platform.linux_distribution(full_distribution_name=0)
            else:
                (self.vendor, version, _) = platform.dist()
            self.vendor = self.vendor.lower()
            self.version = Version(version)
            self.glibc = platform.libc_ver()
            if self.arch == 'x86_64':
                self.bits = '64'
            else:
                self.bits = '32'
                # raise UnsupportedOs("We no longer support 32 bit Linux. If you must use 32 bit Linux then try building from our sources.")
        else:
            raise UnsupportedOs("We do not support this OS.")


class Version(object):
    def __init__(self, version_sct):
        self.version_sct = version_sct

        if not isinstance(version_sct, basestring):
            printv(version_sct)
            raise Exception('Version is not a string.')

        # detect beta, if it exist
        version_sct_beta = version_sct.split('_')
        try:
            self.beta = version_sct_beta[1]
            version_sct_main = version_sct_beta[0]
            self.isbeta = True
        except IndexError:
            self.beta = ""
            version_sct_main = version_sct
            self.isbeta = False

        version_sct_split = version_sct_main.split('.')

        for v in version_sct_split:
            if not v.isdigit():
                raise ValueError('Bad version string.')
        self.major = int(version_sct_split[0])
        try:
            self.minor = int(version_sct_split[1])
        except IndexError:
            self.minor = 0
        try:
            self.patch = int(version_sct_split[2])
        except IndexError:
            self.patch = 0
        try:
            self.hotfix = int(version_sct_split[3])
        except IndexError:
            self.hotfix = 0

    def __repr__(self):
        return "Version(%s,%s,%s,%s,%r)" % (self.major, self.minor, self.patch, self.hotfix, self.beta)

    def __str__(self):
        result = str(self.major) + "." + str(self.minor)
        if self.patch != 0:
            result = result + "." + str(self.patch)
        if self.hotfix != 0:
            result = result + "." + str(self.hotfix)
        if self.beta != "":
            result = result + "_" + self.beta
        return result

    def __ge__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self > other or self == other:
            return True
        return False

    def __le__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self < other or self == other:
            return True
        return False

    def __cmp__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.__lt__(other):
            return -1
        if self.__gt__(other):
            return 1
        return 0

    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major < other.major:
            return True
        if self.major > other.major:
            return False
        if self.minor < other.minor:
            return True
        if self.minor > other.minor:
            return False
        if self.patch < other.patch:
            return True
        if self.patch > other.patch:
            return False
        if self.hotfix < other.hotfix:
            return True
        if self.hotfix > other.hotfix:
            return False
        if self.isbeta and not other.isbeta:
            return True
        if not self.isbeta and other.isbeta:
            return False
        # major, minor and patch all match so this is not less than
        return False

    def __gt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major > other.major:
            return True
        if self.major < other.major:
            return False
        if self.minor > other.minor:
            return True
        if self.minor < other.minor:
            return False
        if self.patch > other.patch:
            return True
        if self.patch < other.patch:
            return False
        if self.hotfix > other.hotfix:
            return True
        if self.hotfix < other.hotfix:
            return False
        if not self.isbeta and other.isbeta:
            return True
        if self.isbeta and not other.isbeta:
            return False
        # major, minor and patch all match so this is not less than
        return False

    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major == other.major and self.minor == other.minor and self.patch == other.patch and self.hotfix == other.hotfix and self.beta == other.beta:
            return True
        return False

    def __ne__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.__eq__(other):
            return False
        return True

    def isLessThan_MajorMinor(self, other):
        if self.major < other.major:
            return True
        if self.major > other.major:
            return False
        if self.minor < other.minor:
            return True
        else:
            return False

    def isGreaterOrEqualThan_MajorMinor(self, other):
        if self.major > other.major:
            return True
        if self.major < other.major:
            return False
        if self.minor >= other.minor:
            return True
        else:
            return False

    def isEqualTo_MajorMinor(self, other):
        return self.major == other.major and self.minor == other.minor

    def isLessPatchThan_MajorMinor(self, other):
        if self.isEqualTo_MajorMinor(other):
            if self.patch < other.patch:
                return True
        return False

    def getFolderName(self):
        result = str(self.major) + "." + str(self.minor)
        if self.patch != 0:
            result = result + "." + str(self.patch)
        if self.hotfix != 0:
            result = result + "." + str(self.hotfix)
        result = result + "_" + self.beta
        return result


class MsgUser(object):
    # TODO: check if should be removed
    __debug = False
    __quiet = False

    @classmethod
    def debugOn(cls):
        cls.__debug = True

    @classmethod
    def debugOff(cls):
        cls.__debug = False

    @classmethod
    def quietOn(cls):
        cls.__quiet = True

    @classmethod
    def quietOff(cls):
        cls.__quiet = False

    @classmethod
    def isquiet(cls):
        return cls.__quiet

    @classmethod
    def isdebug(cls):
        return cls.__debug

    @classmethod
    def debug(cls, message, newline=True):
        if cls.__debug:
            from sys import stderr
            mess = str(message)
            if newline:
                mess += "\n"
            stderr.write(mess)

    @classmethod
    def message(cls, msg):
        if cls.__quiet:
            return
        printv(msg)

    @classmethod
    def question(cls, msg):
        printv(msg,)

    @classmethod
    def skipped(cls, msg):
        if cls.__quiet:
            return
        printv("".join((bcolors.magenta, "[Skipped] ", bcolors.normal, msg)))

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        log.info("".join((bcolors.green, "[OK] ", bcolors.normal, msg)))

    @classmethod
    def failed(cls, msg):
        log.error("".join((bcolors.red, "[FAILED] ", bcolors.normal, msg)))

    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        log.warning("".join((bcolors.yellow, bcolors.bold, "[Warning]", bcolors.normal, " ", msg)))


class Error(Exception):
    """
    The sct Basic error class
    """
    pass


class RunError(Error):
    """
    sct runtime error
    """
    pass


def cache_signature(input_files=[], input_data=[], input_params={}):
    """
    Create a signature to be used for caching purposes

    :param input_files: paths of input files (that can influence output)
    :param input_data: input data (that can influence output)
    :param input_params: input parameters (that can influence output)

    Notes:

    - Use this with cache_valid (to validate caching assumptions)
      and cache_save (to store them)

    - Using this system, the outputs are not checked, only the
      inputs and parameters (to regenerate the outputs in case
      these change).
      If the outputs have been modified directly, then they may
      be reused.
      To prevent that, the next step would be to record the outputs
      signature in the cache file, so as to also verify them prior
      to taking a shortcut.

    """
    import hashlib
    h = hashlib.md5()
    for path in input_files:
        with io.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
    for data in input_data:
        h.update(str(type(data)))
        try:
            h.update(data)
        except:
            h.update(str(data))
    for k, v in sorted(input_params.items()):
        h.update(str(type(k)).encode('utf-8'))
        h.update(str(k).encode('utf-8'))
        h.update(str(type(v)).encode('utf-8'))
        h.update(str(v).encode('utf-8'))

    return "# Cache file generated by SCT\nDEPENDENCIES_SIG={}\n".format(h.hexdigest()).encode()


def cache_valid(cachefile, sig_expected):
    """
    Verify that the cachefile contains the right signature
    """
    if not os.path.exists(cachefile):
        return False
    with io.open(cachefile, "rb") as f:
        sig_measured = f.read()
    return sig_measured == sig_expected


def cache_save(cachefile, sig):
    """
    Save signature to cachefile

    :param cachefile: path to cache file to be saved
    :param sig: cache signature created with cache_signature()
    """
    with io.open(cachefile, "wb") as f:
        f.write(sig)

