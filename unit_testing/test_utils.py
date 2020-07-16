#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for utils

from __future__ import print_function, absolute_import, division

import os
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


def test_add_suffix():
    assert utils.add_suffix('t2.nii', '_mean') == 't2_mean.nii'
    assert utils.add_suffix('t2.nii.gz', 'a') == 't2a.nii.gz'
    assert utils.add_suffix('var/lib/usr/t2.nii.gz', 'sfx') == 'var/lib/usr/t2sfx.nii.gz'
    assert utils.add_suffix('var/lib.version.3/usr/t2.nii.gz', 'sfx') == 'var/lib.version.3/usr/t2sfx.nii.gz'


def test_splitext():
    assert utils.splitext('image.nii') == ('image', '.nii')
    assert utils.splitext('image.nii.gz') == ('image', '.nii.gz')
    assert utils.splitext('folder/image.nii.gz') == ('folder/image', '.nii.gz')
    assert utils.splitext('nice.image.nii.gz') == ('nice.image', '.nii.gz')
    assert utils.splitext('nice.folder/image.nii.gz') == ('nice.folder/image', '.nii.gz')
    assert utils.splitext('image.tar.gz') == ('image', '.tar.gz')


def test_parse_num_list_inv():
    assert utils.parse_num_list_inv([1, 2, 3, 5, 6, 9]) == '1:3;5:6;9'
    assert utils.parse_num_list_inv([3, 2, 1, 5]) == '1:3;5'
    assert utils.parse_num_list_inv([]) == ''


def test_zipdir_dir_abspath():
    path_temp = tempfile.mkdtemp(prefix='sct_test_zipdir_')
    open(os.path.join(path_temp, 'pouf1.txt'), "w+").close()
    os.makedirs(os.path.join(path_temp, 'dir'))
    open(os.path.join(path_temp, 'dir', 'pouf2.txt'), "w+").close()
    # specifying absolute path
    path_extract = tempfile.mkdtemp(prefix='sct_test_zipdir_')
    fname_zip = utils.zipdir(path_temp, os.path.join(path_extract, 'tmp.zip'))
    with zipfile.ZipFile(fname_zip, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(path_extract)
    assert os.path.isfile(os.path.join(path_extract, 'pouf1.txt'))
    assert os.path.isfile(os.path.join(path_extract, 'dir', 'pouf2.txt'))


def test_zipdir_dir_relativepath():
    path_temp = tempfile.mkdtemp(prefix='sct_test_zipdir_')
    open(os.path.join(path_temp, 'pouf1.txt'), "w+").close()
    os.makedirs(os.path.join(path_temp, 'dir'))
    open(os.path.join(path_temp, 'dir', 'pouf2.txt'), "w+").close()
    # specifying relative path
    currentDir = os.getcwd()
    os.chdir(os.path.join(path_temp, '../'))
    fname_zip = utils.zipdir(os.path.basename(path_temp), 'tmp.zip')
    path_extract = tempfile.mkdtemp(prefix='sct_test_zipdir_')
    with zipfile.ZipFile(fname_zip, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(path_extract)
    assert os.path.isfile(os.path.join(path_extract, 'pouf1.txt'))
    assert os.path.isfile(os.path.join(path_extract, 'dir', 'pouf2.txt'))
    os.chdir(currentDir)
