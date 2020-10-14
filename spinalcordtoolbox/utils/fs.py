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
import subprocess

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


def abspath(fname):
    """
    Get absolute path of input file name or path. Deals with tilde.

    '~/code/bla' ------------------> '/usr/bob/code/bla'
    '~/code/bla/pouf.txt' ---------> '/usr/bob/code/bla/pouf.txt'
    '/usr/bob/code/bla' -----------> '/usr/bob/code/bla'
    '/usr/bob/code/bla/pouf.txt' --> '/usr/bob/code/bla/pouf.txt'

    :param fname:
    :return:
    """
    return os.path.abspath(os.path.expanduser(fname))


def check_exe(name):
    """
    Ensure that a program exists

    :param name: str: name or path to program
    :return: path of the program or None
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(name)
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, name)
            if is_exe(exe_file):
                return exe_file

    return None

def sct_dir_local_path(*args):
    """Construct a directory path relative to __sct_dir__"""
    return os.path.join(__sct_dir__, *args)


def sct_test_path(*args):
    """Construct a directory path relative to the sct testing data. Consults the
    SCT_TESTING_DATA environment variable, if unset, paths are relative to the
    current directory."""

    test_path = os.environ.get('SCT_TESTING_DATA', '')
    return os.path.join(test_path, 'sct_testing_data', *args)

def _version_string():
    install_type, sct_commit, sct_branch, version_sct = _git_info()
    if install_type == "package":
        return version_sct
    return "{install_type}-{sct_branch}-{sct_commit}".format(**locals())


def _git_info(commit_env='SCT_COMMIT', branch_env='SCT_BRANCH'):

    sct_commit = os.getenv(commit_env, "unknown")
    sct_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and os.path.isdir(os.path.join(__sct_dir__, ".git")):
        sct_commit = __get_commit() or sct_commit
        sct_branch = __get_branch() or sct_branch

    if sct_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    with io.open(os.path.join(__sct_dir__, 'spinalcordtoolbox', 'version.txt'), 'r') as f:
        version_sct = f.read().rstrip()

    return install_type, sct_commit, sct_branch, version_sct

def __get_branch():
    """
    Fallback if for some reason the value vas no set by sct_launcher
    :return:
    """

    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def __get_commit(path_to_git_folder=None):
    """
    :return: git commit ID, with trailing '*' if modified
    """
    if path_to_git_folder is None:
        path_to_git_folder = __sct_dir__
    else:
        path_to_git_folder = abspath(path_to_git_folder)

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def __get_git_origin(path_to_git_folder=None):
    """
    :return: git origin url if available
    """
    if path_to_git_folder is None:
        path_to_git_folder = __sct_dir__
    else:
        path_to_git_folder = abspath(path_to_git_folder)

    p = subprocess.Popen(["git", "remote", "get-url", "origin"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        origin = output.decode().strip()
    else:
        origin = "?!?"

    return origin

__sct_dir__ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__version__ = _version_string()
__data_dir__ = os.path.join(__sct_dir__, 'data')
__deepseg_dir__ = os.path.join(__data_dir__, 'deepseg_models')