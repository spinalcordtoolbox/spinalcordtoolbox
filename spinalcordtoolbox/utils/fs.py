#!/usr/bin/env python
# -*- coding: utf-8
# Filesystem related helpers and utilities

import sys
import io
import os
import shutil
import tempfile
import datetime
import logging

from .sys import printv

logger = logging.getLogger(__name__)


def tmp_create(basename=None):
    """Create temporary folder and return its path
    """
    prefix = "sct-%s-" % datetime.datetime.now().strftime("%Y%m%d%H%M%S.%f")
    if basename:
        prefix += "%s-" % basename
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    logger.info("Creating temporary folder (%s)" % tmpdir)
    return tmpdir


# Modified from http://shallowsky.com/blog/programming/python-tee.html
class Tee:
    def __init__(self, _fd1, _fd2):
        self.fd1 = _fd1
        self.fd2 = _fd2

    # This is breaking pytest for test_sct_run_batch.py somehow.
    # I think it is ok to omit this, allowing the fd objects to close themselves
    # this prevents closing an fd in use elsewhere.
    # def __del__(self):
    #     self.close()

    def close(self):
        if self.fd1 != sys.__stdout__ and self.fd1 != sys.__stderr__:
            self.fd1.close()
        if self.fd2 != sys.__stdout__ and self.fd2 != sys.__stderr__:
            self.fd2.close()

    def write(self, text):
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self):
        self.fd1.flush()
        self.fd2.flush()


def copy_helper(src, dst, verbose=1):
    """Copy src to dst, almost like shutil.copy
    If src and dst are the same files, don't crash.
    """
    if not os.path.isfile(src):
        folder = os.path.dirname(src)
        contents = os.listdir(folder)
        raise ValueError(f"Couldn't find {os.path.basename(src)} in {folder} (contents: {contents})")

    try:
        logger.info(f"cp {src} {dst}")
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
