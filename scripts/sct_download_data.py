#!/usr/bin/env python
##############################################################################
#
# Download data using http.
#
# ----------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
###############################################################################

from __future__ import absolute_import

import os
import sys
from shutil import rmtree
from distutils.dir_util import copy_tree

from spinalcordtoolbox.utils import download_data, unzip

from msct_parser import Parser
import sct_utils as sct


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('''Download binaries from the web.''')
    parser.add_option(
        name="-d",
        type_value="multiple_choice",
        description="Name of the dataset.",
        mandatory=True,
        example=[
            'sct_example_data',
            'sct_testing_data',
            'course_hawaii17',
            'course_paris18',
            'course_london19',
            'course_beijing19',
            'PAM50',
            'MNI-Poly-AMU',
            'gm_model',
            'optic_models',
            'pmj_models',
            'binaries_debian',
            'binaries_centos',
            'binaries_osx',
            'deepseg_gm_models',
            'deepseg_sc_models',
            'deepseg_lesion_models',
            'c2c3_disc_models'
        ])
    parser.add_option(
        name="-v",
        type_value="multiple_choice",
        description="Verbose. 0: nothing. 1: basic. 2: extended.",
        mandatory=False,
        default_value=1,
        example=['0', '1', '2'])
    parser.add_option(
        name="-o",
        type_value="folder_creation",
        description="path to save the downloaded data",
        mandatory=False)
    parser.add_option(
        name="-h",
        type_value=None,
        description="Display this help",
        mandatory=False)
    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    # initialization
    # note: mirror servers are listed in order of priority
    dict_url = {
        'sct_example_data': ['https://osf.io/kjcgs/?action=download',
                             'https://www.neuro.polymtl.ca/_media/downloads/sct/20180525_sct_example_data.zip'],
        'sct_testing_data': ['https://osf.io/yutrj/?action=download',
                             'https://www.neuro.polymtl.ca/_media/downloads/sct/20191031_sct_testing_data.zip'],
        'PAM50': ['https://osf.io/u79sr/download',
                  'https://www.neuro.polymtl.ca/_media/downloads/sct/20191029_PAM50.zip'],
        'MNI-Poly-AMU': ['https://osf.io/sh6h4/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170310_MNI-Poly-AMU.zip'],
        'gm_model': ['https://osf.io/ugscu/?action=download',
                     'https://www.neuro.polymtl.ca/_media/downloads/sct/20160922_gm_model.zip'],
        'optic_models': ['https://osf.io/g4fwn/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170413_optic_models.zip'],
        'pmj_models': ['https://osf.io/4gufr/?action=download',
                       'https://www.neuro.polymtl.ca/_media/downloads/sct/20170922_pmj_models.zip'],
        'binaries_debian': ['https://osf.io/bt58d/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20190930_sct_binaries_linux.tar.gz'],
        'binaries_centos': ['https://osf.io/8kpt4/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20190930_sct_binaries_linux_centos6.tar.gz'],
        'binaries_osx': ['https://osf.io/msjb5/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20190930_sct_binaries_osx.tar.gz'],
        'course_hawaii17': 'https://osf.io/6exht/?action=download',
        'course_paris18': ['https://osf.io/9bmn5/?action=download',
                           'https://www.neuro.polymtl.ca/_media/downloads/sct/20180612_sct_course-paris18.zip'],
        'course_london19': ['https://osf.io/4q3u7/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20190121_sct_course-london19.zip'],
        'course_beijing19': ['https://osf.io/ef4xz/?action=download',
                             'https://www.neuro.polymtl.ca/_media/downloads/sct/20190802_sct_course-beijing19.zip'],
        'deepseg_gm_models': ['https://osf.io/b9y4x/?action=download',
                              'https://www.neuro.polymtl.ca/_media/downloads/sct/20180205_deepseg_gm_models.zip'],
        'deepseg_sc_models': ['https://osf.io/avf97/?action=download',
                              'https://www.neuro.polymtl.ca/_media/downloads/sct/20180610_deepseg_sc_models.zip'],
        'deepseg_lesion_models': ['https://osf.io/eg7v9/?action=download',
                              'https://www.neuro.polymtl.ca/_media/downloads/sct/20180613_deepseg_lesion_models.zip'],
        'c2c3_disc_models': ['https://osf.io/t97ap/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20190117_c2c3_disc_models.zip']
    }

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    data_name = arguments['-d']
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    dest_folder = arguments.get('-o', os.path.abspath(os.curdir))

    # Download data
    url = dict_url[data_name]
    tmp_file = download_data(url)

    # unzip
    dest_tmp_folder = sct.tmp_create()
    unzip(tmp_file, dest_tmp_folder)
    extracted_files_paths = []
    # Get the name of the extracted files and directories
    extracted_files = os.listdir(dest_tmp_folder)
    for extracted_file in extracted_files:
        extracted_files_paths.append(os.path.join(os.path.abspath(dest_tmp_folder), extracted_file))

    # Check if files and folder already exists
    sct.printv('\nCheck if folder already exists on the destination path...', verbose)
    for data_extracted_name in extracted_files:
        fullpath_dest = os.path.join(dest_folder, data_extracted_name)
        if os.path.isdir(fullpath_dest):
            sct.printv("Folder {} already exists. Removing it...".format(data_extracted_name), 1, 'warning')
            rmtree(fullpath_dest)

    # Destination path
    # for source_path in extracted_files_paths:
        # Copy the content of source to destination (and create destination folder)
    copy_tree(dest_tmp_folder, dest_folder)

    sct.printv("\nRemove temporary folders...", verbose)
    try:
        rmtree(os.path.split(tmp_file)[0])
        rmtree(dest_tmp_folder)
    except Exception as error:
        print("Cannot remove temp folder: " + repr(error))

    sct.printv('Done!\n', verbose)
    return 0


if __name__ == "__main__":
    sct.init_sct()
    res = main()
    raise SystemExit(res)
