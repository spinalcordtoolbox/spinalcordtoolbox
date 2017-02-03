#!/usr/bin/env python
###############################################################################
#
# Module containing several useful functions.
#
# PLEASE!! SORT FUNCTIONS ALPHABETICALLY!
#
# -----------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-07-01
#
# About the license: see the file LICENSE.TXT
###############################################################################

import email
import errno
import fnmatch
import glob
import inspect
import os
import platform
import random
import re
import shutil
import smtplib
import subprocess
import sys
import time
import traceback

import numpy as np

import msct_image
import sct_convert
import sct_image


# TODO: under run(): add a flag "ignore error" for isct_ComposeMultiTransform
# TODO: check if user has bash or t-schell for fsloutput definition


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


def run(cmd, verbose=1, error_exit='error', raise_exception=False):
    if verbose:
        printv(cmd, 1, 'code')
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_final = ''
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            if verbose == 2:
                print output.strip()
            output_final += output.strip() + '\n'
    status_output = process.returncode
    # need to remove the last \n character in the output -> return output_final[0:-1]
    if status_output:
        printv(output_final[0:-1], 1, error_exit)
        if raise_exception:
            raise Exception(output_final[0:-1])
    else:
        # no need to output process.returncode (because different from 0)
        return status_output, output_final[0:-1]


def checkRAM(os, verbose=1):
    if (os == 'linux'):
        status, output = run('grep MemTotal /proc/meminfo', 0)
        print 'RAM: ' + output
        ram_split = output.split()
        ram_total = float(ram_split[1])
        status, output = run('free -m', 0)
        print output
        return ram_total / 1024

    elif (os == 'osx'):
        status, output = run('hostinfo | grep memory', 0)
        print 'RAM: ' + output
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
                rss = 0
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
            print '  Wired Memory:\t\t%d MB' % (vmStats["Pages wired down"] / 1024 / 1024)
            print '  Active Memory:\t%d MB' % (vmStats["Pages active"] / 1024 / 1024)
            print '  Inactive Memory:\t%d MB' % (vmStats["Pages inactive"] / 1024 / 1024)
            print '  Free Memory:\t\t%d MB' % (vmStats["Pages free"] / 1024 / 1024)
        return ram_total


class Timer(object):
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
        sys.stdout.write('\rRemaining time: {:0>2}:{:0>2}:{:05.2f} ({}/{})'.format(
            int(hours), int(minutes), seconds, self.number_of_iteration_done, self.total_number_of_iteration))
        sys.stdout.flush()

    def iterations_done(self, total_num_iterations_done):
        if total_num_iterations_done != 0:
            self.number_of_iteration_done = total_num_iterations_done
            self.time_list.append(time.time() - self.start_timer)
            remaining_iterations = self.total_number_of_iteration - self.number_of_iteration_done
            time_one_iteration = self.time_list[-1] / self.number_of_iteration_done
            remaining_time = remaining_iterations * time_one_iteration
            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)
            sys.stdout.write('\rRemaining time: {:0>2}:{:0>2}:{:05.2f} ({}/{})'.format(
                int(hours), int(minutes), seconds, self.number_of_iteration_done, self.total_number_of_iteration))
            sys.stdout.flush()

    def stop(self):
        self.time_list.append(time.time() - self.start_timer)
        hours, rem = divmod(self.time_list[-1], 3600)
        minutes, seconds = divmod(rem, 60)
        printv('Total time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
        self.is_started = False

    def printRemainingTime(self):
        remaining_iterations = self.total_number_of_iteration - self.number_of_iteration_done
        time_one_iteration = self.time_list[-1] / self.number_of_iteration_done
        remaining_time = remaining_iterations * time_one_iteration
        hours, rem = divmod(remaining_time, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.is_started:
            sys.stdout.write('\rRemaining time: {:0>2}:{:0>2}:{:05.2f} ({}/{})'.format(
                int(hours), int(minutes), seconds, self.number_of_iteration_done, self.total_number_of_iteration))
            sys.stdout.flush()
        else:
            printv('Total time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

    def printTotalTime(self):
        hours, rem = divmod(self.time_list[-1], 3600)
        minutes, seconds = divmod(rem, 60)
        if self.is_started:
            sys.stdout.write('\rRemaining time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
            sys.stdout.flush()
        else:
            printv('Total time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))


def extract_fname(fname):
    """Return the path, base file name and extension"""
    abspath = os.path.abspath(fname)
    path, filename = os.path.split(abspath)
    basename, extension = os.path.splitext(filename)
    if extension == '.gz':
        basename, extension = os.path.splitext(basename)
        extension += '.gz'

    return path+'/', basename, extension


def get_absolute_path(fname):
    """Return the absolute path of a file or a directory"""
    if os.path.isfile(fname) or os.path.isdir(fname):
        return os.path.realpath(fname)
    else:
        printv('\nERROR: ' + fname + ' does not exist. Exit program.\n', 1, 'error')


def check_file_exist(fname, verbose=1):
    if fname[0] == '-':
        # fname should be a warping field that will be inverted, ignore the "-"
        fname_to_test = fname[1:]
    else:
        fname_to_test = fname
    if os.path.isfile(fname_to_test):
        if verbose:
            printv('  OK: '+fname_to_test, verbose, 'normal')
        return True
    else:
        printv('\nERROR: The file ' + os.path.abspath(os.curdir) + '/' + fname_to_test + ' does not exist. Exit program.\n', 1, 'error')
        return False


def check_folder_exist(fname, verbose=1):
    if os.path.isdir(fname):
        printv('  OK: ' + fname, verbose, 'normal')
        return True
    else:
        printv('\nWarning: The directory ' + str(fname) + ' does not exist.\n', 1, 'warning')
        return False


def check_write_permission(fname, verbose=1):
    if os.path.isdir(fname):
        if os.path.isdir(fname):
            return os.access(fname, os.W_OK)
        else:
            printv('\nERROR: The directory ' + fname + ' does not exist. Exit program.\n', 1, 'error')
    else:
        path_fname, file_fname, ext_fname = extract_fname(os.path.abspath(fname))
        return os.access(path_fname, os.W_OK)


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


def check_if_3d(fname):
    """
    Check if input volume is 3d or less.
    :param fname:
    :return: True or False
    """

    nx, ny, nz, nt, px, py, pz, pt = msct_image.Image(fname).dim
    if not nt == 1:
        printv('\nERROR: ' + fname + ' is not a 3D volume. Exit program.\n', 1, 'error')
    else:
        return True


def check_if_rpi(fname):
    if not sct_image.get_orientation_3d(fname, filename=True) == 'RPI':
        printv('\nERROR: ' + fname +
               ' is not in RPI orientation. Use sct_image -setorient to reorient your data. Exit program.\n', 1,
               'error')


def find_file_within_folder(fname, directory, seek_type='file'):
    """Find file (or part of file, e.g. 'my_file*.txt') within folder tree recursively - fname and directory must be
    strings
    seek_type: 'file' or 'dir' to look for either a file or a directory respectively."""

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


def tmp_create(verbose=1):
    printv('\nCreate temporary folder...', verbose)
    path_tmp = slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S") + '_' + str(random.randint(1, 1000000)), 1)
    try:
        os.makedirs(path_tmp)
    except OSError:
        if not os.path.isdir(path_tmp):
            raise
    return path_tmp


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


def tmp_copy_nifti(fname,path_tmp,fname_out='data.nii',verbose=0):
    """Copy a nifti file to (temporary) folder and convert to .nii or .nii.gz"""
    path_fname, file_fname, ext_fname = extract_fname(fname)
    path_fname_out, file_fname_out, ext_fname_out = extract_fname(fname_out)

    shutil.copy(fname, os.path.join(path_tmp, file_fname + ext_fname))
    if ext_fname_out == '.nii':
        run('fslchfiletype NIFTI ' + path_tmp + file_fname_out, 0)
    elif ext_fname_out == '.nii.gz':
        run('fslchfiletype NIFTI_GZ ' + path_tmp + file_fname_out, 0)


def generate_output_file(fname_in, fname_out, verbose=1):
    """Generate output file. Only works for images (e.g., nifti, nifti_gz)
    :param fname_in:
    :param fname_out:
    :param verbose:
    :return: fname_out
    """
    path_in, file_in, ext_in = extract_fname(fname_in)
    path_out, file_out, ext_out = extract_fname(fname_out)
    # if input image does not exist, give error
    if not os.path.isfile(fname_in):
        printv('  ERROR: File ' + fname_in + ' does not exist. Exit program.', 1, 'error')
        sys.exit(2)
    # if input and output fnames are the same, do nothing and exit function
    if fname_in == fname_out:
        printv('  WARNING: fname_in and fname_out are the same. Do nothing.', verbose, 'warning')
        printv('  File created: {0}{1}'.format(os.path.join(path_out, file_out), ext_out))
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
        sct_convert.convert(fname_in, fname_out)
    else:
        # Generate output file without changing the extension
        shutil.move(fname_in, fname_out)

    printv('  File created: ' + os.path.join(path_out + file_out) + ext_out, verbose)
    return path_out + file_out + ext_out


def check_if_same_space(fname_1, fname_2):
    """check if two images are in the same space and same orientation"""

    im_1 = msct_image.Image(fname_1)
    im_2 = msct_image.Image(fname_2)
    q1 = im_1.hdr.get_qform()
    q2 = im_2.hdr.get_qform()

    dec = int(np.abs(round(np.log10(min(np.abs(q1[np.nonzero(q1)]))))) + 1)
    dec = 4 if dec > 4 else dec
    return np.array_equal(np.around(q1, dec), np.around(q2, dec))


def printv(string, verbose=1, type='normal'):
    """enables to print color coded messages, depending on verbose status """

    colors = {
        'normal': bcolors.normal,
        'info': bcolors.green,
        'warning': bcolors.yellow,
        'error': bcolors.red,
        'code': bcolors.blue,
        'bold': bcolors.bold,
        'process': bcolors.magenta
    }

    if verbose:
        # Print color only if the output is the terminal
        if sys.stdout.isatty():
            color = colors.get(type, bcolors.normal)
            print(color + string + bcolors.normal)
        else:
            print(string)

    if type == 'error':
        frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
        if sys.stdout.isatty():
            print('\n' + bcolors.red + filename + traceback.format_exc() + bcolors.normal)
        else:
            print('\n' + filename + traceback.format_exc())
        sys.exit(2)


def send_email(addr_to, addr_from='spinalcordtoolbox@gmail.com', passwd_from='', subject='', message='', filename=None):

    msg = email.MIMEMultipart.MIMEMultipart()

    msg['From'] = addr_from
    msg['To'] = addr_to
    msg['Subject'] = subject  # "SUBJECT OF THE EMAIL"
    body = message  # "TEXT YOU WANT TO SEND"

    msg.attach(email.MIMEText.MIMEText(body, 'plain'))

    # filename = "NAME OF THE FILE WITH ITS EXTENSION"
    if filename:
        attachment = open(filename, "rb")
        part = email.MIMEBase.MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        email.encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(addr_from, passwd_from)
    text = msg.as_string()
    server.sendmail(addr_from, addr_to, text)
    server.quit()


def sign(x):
    """Get the sign of a number. Returns 1 if x>=0 and -1 if x<0"""
    if x >= 0:
        return 1
    else:
        return -1


def slash_at_the_end(path, slash=0):
    """make sure there is (or not) a slash at the end of path name"""
    if slash == 0:
        if path[-1:] == '/':
            path = path[:-1]
    if slash == 1:
        if not path[-1:] == '/':
            path = path + '/'
    return path


def delete_nifti(fname_in):
    # extract input file extension
    path_in, file_in, ext_in = extract_fname(fname_in)
    # delete nifti if exist
    if os.path.isfile(path_in + file_in + '.nii'):
        os.system('rm ' + path_in + file_in + '.nii')
    # delete nifti if exist
    if os.path.isfile(path_in + file_in + '.nii.gz'):
        os.system('rm ' + path_in + file_in + '.nii.gz')


def get_interpolation(program, interp):
    """get correct interpolation field depending on program used. Supported programs: ants, flirt, WarpImageMultiTransform"""
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
        printv(
            'WARNING (' + os.path.basename(__file__) + '): interp_program not assigned. Using linear for ants_affine.',
            1, 'warning')
        interp_program = ' -n Linear'
    # return
    return interp_program


def template_dict(template):
    """Dictionary of file names for template
    :param template:
    :return: dict_template
    """
    if template == 'PAM50':
        dict_template = {'t2': 'template/PAM50_t2.nii.gz', 't1': 'template/PAM50_t1.nii.gz'}
    return dict_template


class UnsupportedOs(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Os(object):
    '''Work out which platform we are running on'''

    def __init__(self):
        if os.name != 'posix': raise UnsupportedOs('We only support OS X/Linux')
        self.os = platform.system().lower()
        self.arch = platform.machine()
        self.applever = ''

        if self.os == 'darwin':
            self.os = 'osx'
            self.vendor = 'apple'
            self.version = Version(platform.release())
            (self.applever, _, _) = platform.mac_ver()
            if self.arch == 'Power Macintosh': raise UnsupportedOs('We do not support PowerPC')
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
            print version_sct
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


class ___MsgUser(object):
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
            mess = str(message)
            if newline:
                mess += "\n"
            sys.stderr.write(mess)

    @classmethod
    def message(cls, msg):
        if cls.__quiet:
            return
        print msg

    @classmethod
    def question(cls, msg):
        print msg,

    @classmethod
    def skipped(cls, msg):
        if cls.__quiet:
            return
        print "".join((bcolors.magenta, "[Skipped] ", bcolors.normal, msg))

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        print "".join((bcolors.green, "[OK] ", bcolors.normal, msg))

    @classmethod
    def failed(cls, msg):
        print "".join((bcolors.red, "[FAILED] ", bcolors.normal, msg))

    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        print "".join((bcolors.yellow, bcolors.bold, "[Warning]", bcolors.normal, " ", msg))
