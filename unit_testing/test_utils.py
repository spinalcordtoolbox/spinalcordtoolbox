#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for utils

from __future__ import print_function, absolute_import, division

import os
import shutil
import tempfile
import zipfile

from spinalcordtoolbox import utils


def test_abspath():
    # absolute dir
    path_temp = tempfile.mkdtemp(prefix='sct_test_abspath_')
    assert utils.abspath(path_temp) == path_temp
    # absolute file name
    open(os.path.join(path_temp, 'pouf1.txt'), "w+").close()
    assert utils.abspath(os.path.join(path_temp, 'pouf1.txt')) == os.path.join(path_temp, 'pouf1.txt')
    # relative dir
    currentDir = os.getcwd()
    os.chdir(os.path.join(path_temp, '../'))
    # note: here we use realpath(path_temp) intead of abspath(path_temp) because on OSX, tempfile.mkdtemp creates a
    # folder under the alias /var --> /private/var
    assert utils.abspath(os.path.join(os.path.basename(path_temp), 'pouf1.txt')) == \
           os.path.realpath(os.path.join(path_temp, 'pouf1.txt'))
    os.chdir(currentDir)


def test_parse_num_list_inv():
    assert utils.parse_num_list_inv([1, 2, 3, 5, 6, 9]) == '1:3;5:6;9'
    assert utils.parse_num_list_inv([3, 2, 1, 5]) == '1:3;5'
    assert utils.parse_num_list_inv([]) == ''

