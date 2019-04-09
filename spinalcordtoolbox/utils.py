#!/usr/bin/env python
# -*- coding: utf-8
# Collection of useful functions

from __future__ import absolute_import

import sys, io, os, re
import time, datetime
import platform
import logging
import subprocess


# TODO: add test

def check_exe(name):
    """
    Ensure that a program exists
    :type name: string
    :param name: name or path to program
    :return: path of the program or None
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(name)
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, name)
            if is_exe(exe_file):
                return exe_file

    return None


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


def __get_commit():
    """
    :return: git commit ID, with trailing '*' if modified
    """
    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=__sct_dir__)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"): # ignore ignored files, they can't hurt
               continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit

def _git_info(commit_env='SCT_COMMIT',branch_env='SCT_BRANCH'):

    sct_commit = os.getenv(commit_env, "unknown")
    sct_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and os.path.isdir(os.path.join(__sct_dir__, ".git")):
        sct_commit = __get_commit() or sct_commit
        sct_branch = __get_branch() or sct_branch

    if sct_commit is not 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    with io.open(os.path.join(__sct_dir__, 'version.txt'), 'r') as f:
        version_sct = f.read().rstrip()

    return install_type, sct_commit, sct_branch, version_sct

def _version_string():
    install_type, sct_commit, sct_branch, version_sct = _git_info()
    if install_type == "package":
        return version_sct
    else:
        return "{install_type}-{sct_branch}-{sct_commit}".format(**locals())


__sct_dir__ = os.getenv("SCT_DIR", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
__version__ = _version_string()
__data_dir__ = os.getenv("SCT_DATA_DIR", os.path.join(__sct_dir__, 'data'))

logger = logging.getLogger('sct')

def parse_num_list(str_num):
    """
    Parse numbers in string based on delimiter: , or :
    Examples:
      '' -> []
      '1,2,3' -> [1, 2, 3]
      '1:3,4' -> [1, 2, 3, 4]
      '1,1:4' -> [1, 2, 3, 4]
    :param str_num: string
    :return: list of ints
    """
    list_num = list()

    if not str_num:
        return list_num

    elements = str_num.split(",")
    for element in elements:
        m = re.match(r"^\d+$", element)
        if m is not None:
            val = int(element)
            if val not in list_num:
                list_num.append(val)
            continue
        m = re.match(r"^(?P<first>\d+):(?P<last>\d+)$", element)
        if m is not None:
            a = int(m.group("first"))
            b = int(m.group("last"))
            list_num += [ x for x in range(a, b+1) if x not in list_num ]
            continue
        raise ValueError("unexpected group element {} group spec {}".format(element, str_num))

    return list_num


def parse_num_list_inv(list_int):
    """
    Take a list of numbers and output a string that reduce this list based on delimiter: ; or :
    Note: we use ; instead of , for compatibility with csv format.
    Examples:
      [] -> ''
      [1, 2, 3] --> '1:3'
      [1, 2, 3, 5] -> '1:3;5'
    :param list_int: list of ints
    :return: str_num: string
    """
    # deal with empty list
    if not list_int or list_int is None:
        return ''
    # Sort list in increasing number
    list_int = sorted(list_int)
    # initialize string
    str_num = str(list_int[0])
    colon_is_present = False
    # Loop across list elements and build string iteratively
    for i in range(1, len(list_int)):
        # if previous element is the previous integer: I(i-1) = I(i)-1
        if list_int[i] == list_int[i-1] + 1:
            # if ":" already there, update the last chars (based on the number of digits)
            if colon_is_present:
                str_num = str_num[:-len(str(list_int[i-1]))] + str(list_int[i])
            # if not, add it along with the new int value
            else:
                str_num += ':' + str(list_int[i])
                colon_is_present = True
        # I(i-1) != I(i)-1
        else:
            str_num += ';' + str(list_int[i])
            colon_is_present = False

    return str_num
