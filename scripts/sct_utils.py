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

import errno
import sys
import time

import os
import commands
import subprocess
import re
from sys import stdout
import logging

import glob
import shutil


# TODO: under run(): add a flag "ignore error" for isct_ComposeMultiTransform
# TODO: check if user has bash or t-schell for fsloutput definition

"""
Basic logging setup for the sct
set SCT_LOG_LEVEL and SCT_LOG_FORMAT in ~/.sctrc to change the sct log
format and level
"""


log = logging.getLogger('sct')
log.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
nh = logging.NullHandler()
log.addHandler(nh)
LOG_LEVEL = os.getenv('SCT_LOG_LEVEL')
LOG_FORMAT = os.getenv('SCT_LOG_FORMAT')
if not LOG_FORMAT:
    LOG_FORMAT = None


def start_stream_logger():
    """ Log to terminal, by default the formating is like a print() call

    :return: 
    """

    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler.setFormatter(formatter)

    if LOG_LEVEL in logging._levelNames:
        stream_handler.setLevel(LOG_LEVEL)
        log.addHandler(stream_handler)
    elif LOG_LEVEL == 'DISABLE':
        stream_handler.setLevel(sys.maxint)
    else:
        stream_handler.setLevel(logging.INFO)
        log.addHandler(stream_handler)


def pause_stream_logger():
    """ Pause the log to Terminal
    
    :return: 
    """
    log.removeHandler(stream_handler)


class NoColorFormatter(logging.Formatter):
    """
    Formater removing terminal specific colors from outputs
    """
    def format(self, record):
        for color in bcolors.colors():
            record.msg = record.msg.replace(color, "")
        return super(NoColorFormatter, self).format(record)


def add_file_handler_to_logger(filename="{}.log".format(__file__), mode='a', log_format=None, log_level=None):
    """ Convenience fct to add a file handler to the sct log
        Will remove colors from prints
    :param filename: 
    :param mode: 
    :param log_format: 
    :param log_level: 
    :return: the file handler 
    """
    log.debug('Adding file handler {}'.format(filename))
    fh = logging.FileHandler(filename=filename, mode=mode)

    if log_format is None:
        formatter = NoColorFormatter(LOG_FORMAT)  # sct.printv() emulator)
    else:
        formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)

    if log_level:
        fh.setLevel(log_level)
    else:
        fh.setLevel(logging.INFO)
    log.addHandler(fh)
    return fh


def remove_handler(handler):
    """ Remore any handler from logs
    
    :param handler: 
    :return: 
    """
    log.debug("Removing log handler {} ".format(handler))
    log.removeHandler(handler)

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


#=======================================================================================================================
# add suffix
#=======================================================================================================================
def add_suffix(fname, suffix):
    """
    Add suffix between end of file name and extension on a nii or nii.gz file.
    :param fname: absolute or relative file name. Example: t2.nii
    :param suffix: suffix. Example: _mean
    :return: file name with suffix. Example: t2_mean.nii
    """
    # get index of extension. Here, we search from the end to avoid issue with folders that have ".nii" in their name.
    ind_nii = fname.rfind('.nii')
    # in case no extension was found (i.e. only prefix was specified by user)
    if ind_nii == -1:
        return fname[:len(fname)] + suffix
    else:
        # return file name with suffix
        return fname[:ind_nii] + suffix + fname[ind_nii:]


#=======================================================================================================================
# run
#=======================================================================================================================
# Run UNIX command
def run_old(cmd, verbose=1):
    if verbose:
        printv(bcolors.blue + cmd + bcolors.normal)
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        printv('\nERROR! \n' + output + '\nExit program.\n', 1, 'error')
    else:
        return status, output


def run(cmd, verbose=1, error_exit='error', raise_exception=False):
    # if verbose == 2:
    #     printv(sys._getframe().f_back.f_code.co_name, 1, 'process')
    if verbose:
        printv(cmd, 1, 'code')
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_final = ''
    while True:
        # Watch out for deadlock!!!
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            if verbose == 2:
                printv(output.strip())
            output_final += output.strip() + '\n'
    status_output = process.returncode
    # process.stdin.close()
    # process.stdout.close()
    # process.terminate()

    # need to remove the last \n character in the output -> return output_final[0:-1]
    if status_output:
        # from inspect import stack
        printv(output_final[0:-1], 1, error_exit)
        # in case error_exit is not error (immediate exit), the line below can be run
        return status_output, output_final[0:-1]
        # printv('\nERROR in '+stack()[1][1]+'\n', 1, 'error')  # printv(name of parent function)
        # sys.exit()
        if raise_exception:
            raise Exception(output_final[0:-1])
    else:
        # no need to output process.returncode (because different from 0)
        return status_output, output_final[0:-1]


# =======================================================================================================================
# Get SCT version
# =======================================================================================================================
def get_sct_version():

    sct_commit = ''
    sct_branch = ''

    # get path of SCT
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)

    # fetch true commit number and branch
    path_curr = os.path.abspath(os.curdir)
    os.chdir(path_sct)
    # first, make sure there is a .git folder
    if os.path.isdir('.git'):
        install_type = 'git'
        sct_commit = commands.getoutput('git rev-parse HEAD')
        sct_branch = commands.getoutput('git branch | grep \*').strip('* ')
        if not (sct_commit.isalnum()):
            sct_commit = 'unknown'
            sct_branch = 'unknown'
        # print '  branch: ' + sct_branch
    else:
        install_type = 'package'
    # fetch version
    with open(path_sct + '/version.txt', 'r') as myfile:
        version_sct = myfile.read().replace('\n', '')
        # print '  version: ' + version_sct

    # go back to previous dir
    os.chdir(path_curr)
    return install_type, sct_commit, sct_branch, version_sct

#
#
#     # check if there is a .git repos
#     if [-e ${SCT_DIR} /.git]; then
#     # retrieve commit
#     SCT_COMMIT = `git - -git - dir =${SCT_DIR} /.git
#     rev - parse
#     HEAD
#     `
#     # retrieve branch
#     SCT_BRANCH = `git - -git - dir =${SCT_DIR} /.git
#     branch | grep \ * | awk
#     '{print $2}'
#     `
#     echo
#     "Spinal Cord Toolbox ($SCT_BRANCH/$SCT_COMMIT)"
#
# else
# echo
# "Spinal Cord Toolbox (version: $SCT_VERSION)"
# fi


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
        ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0]
        vm = subprocess.Popen(['vm_stat'], stdout=subprocess.PIPE).communicate()[0]

        # Iterate processes
        processLines = ps.split('\n')
        sep = re.compile('[\s]+')
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
        sep = re.compile(':[\s]+')
        vmStats = {}
        for row in range(1, len(vmLines) - 2):
            rowText = vmLines[row].strip()
            rowElements = sep.split(rowText)
            vmStats[(rowElements[0])] = int(rowElements[1].strip('\.')) * 4096

        if verbose:
            printv('  Wired Memory:\t\t%d MB' % (vmStats["Pages wired down"] / 1024 / 1024))
            printv('  Active Memory:\t%d MB' % (vmStats["Pages active"] / 1024 / 1024))
            printv('  Inactive Memory:\t%d MB' % (vmStats["Pages inactive"] / 1024 / 1024))
            printv('  Free Memory:\t\t%d MB' % (vmStats["Pages free"] / 1024 / 1024))
            # printv('Real Mem Total (ps):\t%.3f MB' % ( rssTotal/1024/1024 ))

        return ram_total


class Timer:
    def __init__(self, number_of_iteration=1):
        self.start_timer = 0
        self.time_list = []
        self.total_number_of_iteration = number_of_iteration
        self.number_of_iteration_done = 0
        self.is_started = False

    def start(self):
        self.start_timer = time.time()
        self.is_started = True

    def add_iteration(self, num_iteration_done=1):
        self.number_of_iteration_done += num_iteration_done
        self.time_list.append(time.time() - self.start_timer)
        remaining_iterations = self.total_number_of_iteration - self.number_of_iteration_done
        time_one_iteration = self.time_list[-1] / self.number_of_iteration_done
        remaining_time = remaining_iterations * time_one_iteration
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        log.info('\rRemaining time: {:0>2}:{:0>2}:{:05.2f} ({}/{})                      '.format(int(hours), int(minutes), seconds, self.number_of_iteration_done, self.total_number_of_iteration))
        if log.handlers:
            [h.flush() for h in log.handlers]

    def iterations_done(self, total_num_iterations_done):
        if total_num_iterations_done != 0:
            self.number_of_iteration_done = total_num_iterations_done
            self.time_list.append(time.time() - self.start_timer)
            remaining_iterations = self.total_number_of_iteration - self.number_of_iteration_done
            time_one_iteration = self.time_list[-1] / self.number_of_iteration_done
            remaining_time = remaining_iterations * time_one_iteration
            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)
            log.info('\rRemaining time: {:0>2}:{:0>2}:{:05.2f} ({}/{})                      '.format(int(hours), int(minutes), seconds, self.number_of_iteration_done, self.total_number_of_iteration))
            if log.handlers:
                [h.flush() for h in log.handlers]

    def stop(self):
        self.time_list.append(time.time() - self.start_timer)
        hours, rem = divmod(self.time_list[-1], 3600)
        minutes, seconds = divmod(rem, 60)
        printv('Total time: {:0>2}:{:0>2}:{:05.2f}                      '.format(int(hours), int(minutes), seconds))
        self.is_started = False

    def printRemainingTime(self):
        remaining_iterations = self.total_number_of_iteration - self.number_of_iteration_done
        time_one_iteration = self.time_list[-1] / self.number_of_iteration_done
        remaining_time = remaining_iterations * time_one_iteration
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.is_started:
            log.info('\rRemaining time: {:0>2}:{:0>2}:{:05.2f} ({}/{})                      '.format(int(hours), int(minutes), seconds, self.number_of_iteration_done, self.total_number_of_iteration))
            if log.handlers:
                [h.flush() for h in log.handlers]
        else:
            printv('Total time: {:0>2}:{:0>2}:{:05.2f}                      '.format(int(hours), int(minutes), seconds))

    def printTotalTime(self):
        hours, rem = divmod(self.time_list[-1], 3600)
        minutes, seconds = divmod(rem, 60)
        if self.is_started:
            log.info('\rRemaining time: {:0>2}:{:0>2}:{:05.2f}                      '.format(int(hours), int(minutes), seconds))
            if log.handlers:
                [h.flush() for h in log.handlers]
        else:
            printv('Total time: {:0>2}:{:0>2}:{:05.2f}                      '.format(int(hours), int(minutes), seconds))

class ForkStdoutToFile(object):
    """Use to redirect stdout to file
    Default mode is to send stdout to file AND to terminal

    """
    def __init__(self, filename="{}.log".format(__file__), to_file_only=False):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a")
        self.filename = filename
        self.to_file_only = False
        sys.stdout = self

    def __del__(self):
        self.pause()
        self.close()

    def pause(self):
        sys.stdout = self.terminal

    def restart(self):
        sys.stdout = self

    def write(self, message):
        if not self.to_file_only:
            self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        if not self.to_file_only:
            self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

    def read(self):
        with open(self.filename, "r") as fp:
            fp.read()

    # def send_email(self, email, passwd_from=None, subject="file_log", attachment=True):
    #     if attachment:
    #         filename = self.filename
    #     else:
    #         filename = None
    #     send_email(email, passwd_from=passwd_from, subject=subject, message=self.read(), filename=filename)

#=======================================================================================================================
# extract_fname
#=======================================================================================================================
# Extract path, file and extension
def extract_fname(fname):
    # extract path
    path_fname = os.path.dirname(fname) + '/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname, '')
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname) - 4]
        ext_fname = ".nii.gz"
    return path_fname, file_fname, ext_fname


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
        except OSError, e:
            if e.errno != errno.EEXIST:
                return 2
    else:
        return 1

#=======================================================================================================================
# check_if_3d
#=======================================================================================================================


def check_if_3d(fname):
    """
    Check if input volume is 3d or less.
    :param fname:
    :return: True or False
    """
    from msct_image import Image
    nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
    if not nt == 1:
        printv('\nERROR: ' + fname + ' is not a 3D volume. Exit program.\n', 1, 'error')
    else:
        return True

#=======================================================================================================================
# check_if_rpi:  check if data are in RPI orientation
#=======================================================================================================================


def check_if_rpi(fname):
    from sct_image import get_orientation_3d
    if not get_orientation_3d(fname, filename=True) == 'RPI':
        printv('\nERROR: ' + fname + ' is not in RPI orientation. Use sct_image -setorient to reorient your data. Exit program.\n', 1, 'error')


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


#=======================================================================================================================
# create temporary folder and return path of tmp dir
#=======================================================================================================================
def tmp_create(verbose=1):
    printv('\nCreate temporary folder...', verbose)
    import time
    import random
    path_tmp = slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S") + '_' + str(random.randint(1, 1000000)), 1)
    # create directory
    try:
        os.makedirs(path_tmp)
    except OSError:
        if not os.path.isdir(path_tmp):
            raise
    return path_tmp


class TempFolder(object):
    """This class will create a temporary folder."""

    def __init__(self, verbose=0):
        self.path_tmp = tmp_create(verbose)
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
        shutil.copy(filename, self.path_tmp)

    def cleanup(self):
        """Remove the created folder and its contents."""
        shutil.rmtree(self.path_tmp, ignore_errors=True)


def delete_tmp_files_and_folders(path=''):
    """
    This function removes all files that starts with 'tmp.' in the path specified as input. If no path are provided,
    the current path is selected. The function removes files and directories recursively and handles Exceptions and
    errors by ignoring them.
    Args:
        path: directory in which temporary files and folders must be removed

    Returns:

    """
    if not path:
        path = os.getcwd()
    pattern = os.path.join(path, 'tmp.*')

    for item in glob.glob(pattern):
        try:
            if os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
            elif os.path.isfile(item):
                os.remove(item)
        except:  # in case an exception is raised (e.g., on Windows, if the file is in use)
            continue


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
def generate_output_file(fname_in, fname_out, verbose=1):
    """
    Generate output file. Only works for images (e.g., nifti, nifti_gz)
    :param fname_in:
    :param fname_out:
    :param verbose:
    :return: fname_out
    """
    # import stuff
    import shutil  # for moving files
    path_in, file_in, ext_in = extract_fname(fname_in)
    path_out, file_out, ext_out = extract_fname(fname_out)
    # if input image does not exist, give error
    if not os.path.isfile(fname_in):
        printv('  ERROR: File ' + fname_in + ' does not exist. Exit program.', 1, 'error')
        sys.exit(2)
    # if input and output fnames are the same, do nothing and exit function
    if fname_in == fname_out:
        printv('  WARNING: fname_in and fname_out are the same. Do nothing.', verbose, 'warning')
        printv('  File created: ' + path_out + file_out + ext_out)
        return path_out + file_out + ext_out
    # if fname_out already exists in nii or nii.gz format
    if os.path.isfile(path_out + file_out + ext_out):
        printv('  WARNING: File ' + path_out + file_out + ext_out + ' already exists. Deleting it...', 1, 'warning')
        os.remove(path_out + file_out + ext_out)
    if ext_in != ext_out:
        # Generate output file
        '''
        # TRY TO UNCOMMENT THIS LINES AND RUN IT IN AN OTHER STATION THAN EVANS (testing of sct_label_vertebrae and sct_smooth_spinalcord never stops with this lines on evans)
        if ext_in == '.nii.gz' and ext_out == '.nii':  # added to resolve issue #728
            run('gunzip -f ' + fname_in)
            os.rename(path_in + file_in + '.nii', fname_out)
        else:
        '''
        from sct_convert import convert
        convert(fname_in, fname_out)
    else:
        # Generate output file without changing the extension
        shutil.move(fname_in, fname_out)

    # # Move file to output folder (keep the same extension as input)
    # shutil.move(fname_in, path_out+file_out+ext_in)
    # # convert to nii (only if necessary)
    # if ext_out == '.nii' and ext_in != '.nii':
    #     convert(path_out+file_out+ext_in, path_out+file_out+ext_out)
    #     os.remove(path_out+file_out+ext_in)  # remove nii.gz file
    # # convert to nii.gz (only if necessary)
    # if ext_out == '.nii.gz' and ext_in != '.nii.gz':
    #     convert(path_out+file_out+ext_in, path_out+file_out+ext_out)
    #     os.remove(path_out+file_out+ext_in)  # remove nii file
    # display message
    printv('  File created: ' + path_out + file_out + ext_out, verbose)
    # if verbose:
    #     printv('  File created: '+path_out+file_out+ext_out)
    return path_out + file_out + ext_out


#=======================================================================================================================
# check_if_installed
#=======================================================================================================================
# check if dependant software is installed
def check_if_installed(cmd, name_software):
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        printv('\nERROR: ' + name_software + ' is not installed.\nExit program.\n')
        sys.exit(2)


#=======================================================================================================================
# check_if_same_space
#=======================================================================================================================
# check if two images are in the same space and same orientation
def check_if_same_space(fname_1, fname_2):
    from msct_image import Image
    from numpy import min, nonzero, all, around
    from numpy import abs as np_abs
    from numpy import log10 as np_log10

    im_1 = Image(fname_1)
    im_2 = Image(fname_2)
    q1 = im_1.hdr.get_qform()
    q2 = im_2.hdr.get_qform()

    dec = int(np_abs(round(np_log10(min(np_abs(q1[nonzero(q1)]))))) + 1)
    dec = 4 if dec > 4 else dec
    return all(around(q1, dec) == around(q2, dec))


def printv(string, verbose=1, type='normal'):
    """enables to print (color coded messages, depending on verbose status) 
    """

    colors = {'normal': bcolors.normal, 'info': bcolors.green, 'warning': bcolors.yellow, 'error': bcolors.red,
              'code': bcolors.blue, 'bold': bcolors.bold, 'process': bcolors.magenta}

    if verbose:
        # Print color only if the output is the terminal
        # Note jcohen: i added a try/except in case stdout does not have isatty field (it did happen to me)
        try:
            if sys.stdout.isatty():
                color = colors.get(type, bcolors.normal)
                log.info('{0}{1}{2}'.format(color, string, bcolors.normal))

            else:
                log.info(string)
        except Exception as e:
            log.info(string)

    if type == 'error':
        from inspect import stack
        import traceback

        frame, filename, line_number, function_name, lines, index = stack()[1]
        if sys.stdout.isatty():
            log.error('\n' + bcolors.red + filename + traceback.format_exc() + bcolors.normal)
        else:
            log.error('\n' + filename + traceback.format_exc())
        sys.exit(2)


#=======================================================================================================================
# send email
#=======================================================================================================================
def send_email(addr_to='', addr_from='spinalcordtoolbox@gmail.com', passwd_from='', subject='', message='', filename=None, html=False):
    import smtplib
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    from email.MIMEBase import MIMEBase
    from email import encoders

    msg = MIMEMultipart()

    msg['From'] = addr_from
    msg['To'] = addr_to
    msg['Subject'] = subject  # "SUBJECT OF THE EMAIL"
    body = message  # "TEXT YOU WANT TO SEND"

    # body in html format for monospaced formatting
    body_html = """
    <html><pre style="font: monospace"><body>
    """+body+"""
    </body></pre></html>
    """

    # # We must choose the body charset manually
    # for body_charset in 'US-ASCII', 'ISO-8859-1', 'UTF-8':
    #     try:
    #         body.encode(body_charset)
    #     except UnicodeError:
    #         pass
    #     else:
    #         break

    # msg.set_charset("utf-8")

    if html:
        msg.attach(MIMEText(body_html, 'html'))
    else:
        msg.attach(MIMEText(body, 'plain'))

    # msg.attach(MIMEText(body.encode(body_charset), 'plain', body_charset))

    # filename = "NAME OF THE FILE WITH ITS EXTENSION"
    if filename:
        attachment = open(filename, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

    # send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(addr_from, passwd_from)
    text = msg.as_string()
    server.sendmail(addr_from, addr_to, text)
    server.quit()


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
# slash_at_the_end: make sure there is (or not) a slash at the end of path name
#=======================================================================================================================
def slash_at_the_end(path, slash=0):
    if slash == 0:
        if path[-1:] == '/':
            path = path[:-1]
    if slash == 1:
        if not path[-1:] == '/':
            path = path + '/'
    return path


#=======================================================================================================================
# delete_nifti: delete nifti file(s)
#=======================================================================================================================
def delete_nifti(fname_in):
    # extract input file extension
    path_in, file_in, ext_in = extract_fname(fname_in)
    # delete nifti if exist
    if os.path.isfile(path_in + file_in + '.nii'):
        os.system('rm ' + path_in + file_in + '.nii')
    # delete nifti if exist
    if os.path.isfile(path_in + file_in + '.nii.gz'):
        os.system('rm ' + path_in + file_in + '.nii.gz')


#=======================================================================================================================
# get_interpolation: get correct interpolation field depending on program used. Supported programs: ants, flirt, WarpImageMultiTransform
#=======================================================================================================================
def get_interpolation(program, interp):
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
    elif program == 'ants' or program == 'ants_affine' or program == 'isct_antsApplyTransforms' or program == 'isct_antsSliceRegularizedRegistration':
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
    return interp_program


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
            raise UnsupportedOs('We only support OS X/Linux')
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
