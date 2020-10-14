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

import sys
import io
import os
import logging
import shutil
import pathlib

from spinalcordtoolbox.utils import check_exe, tmp_create

logger = logging.getLogger(__name__)

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


def mv(src, dst, verbose=1):
    """Move a file from src to dst, almost like os.rename
    """
    printv("mv %s %s" % (src, dst), verbose=verbose, type="code")
    os.rename(src, dst)


def copy(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise Exception("Couldn't find %s in %s (contents: %s)"
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
        raise  # Must be another error


def rmtree(folder, verbose=1):
    """Recursively remove folder, almost like shutil.rmtree
    """
    printv("rm -rf %s" % (folder), verbose=verbose, type="code")
    shutil.rmtree(folder, ignore_errors=True)


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


def get_absolute_path(fname):
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
            printv('  OK: ' + fname, verbose, 'normal')
        return True
    else:
        printv('\nERROR: The file ' + fname + ' does not exist. Exit program.\n', 1, 'error')
        return False


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

# FIXME: chdir()


class TempFolder(object):
    """This class will create a temporary folder."""

    def __init__(self, verbose=0):
        self.path_tmp = tmp_create()
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
            utils.run_proc('gunzip -f ' + fname_in)
            os.rename(os.path.join(path_in, file_in + '.nii'), fname_out)
        else:
        '''
        convert(fname_in, fname_out, squeeze_data=squeeze_data, verbose=0)
    else:
        # Generate output file without changing the extension
        shutil.move(fname_in, fname_out)

    printv('  File created: ' + os.path.join(path_out, file_out + ext_out), verbose)
    return os.path.join(path_out, file_out + ext_out)


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
