#!/usr/bin/env python
#########################################################################################
#
# Download data using http.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import shutil
import sys
import time
import zipfile
from os import path, remove, rename

import msct_parser
import requests
import sct_utils as sct


def get_parser():
    # parser initialisation
    parser = msct_parser.Parser(__file__)
    parser.usage.set_description('''Download dataset from the web.''')
    parser.add_option(name="-d",
                      type_value="multiple_choice",
                      description="Name of the dataset.",
                      mandatory=True,
                      example=['sct_example_data', 'sct_testing_data', 'PAM50', 'MNI-Poly-AMU', 'gm_model'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      example=['0', '1', '2'])
    parser.add_option(name="-h",
                      type_value=None,
                      description="Display this help",
                      mandatory=False)
    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    # initialization
    verbose = 1
    dict_url = {'sct_example_data': 'https://osf.io/feuef/?action=download',
                'sct_testing_data': 'https://osf.io/uqcz5/?action=download',
                'PAM50': 'https://osf.io/gdwn6/?action=download',
                'MNI-Poly-AMU': 'https://osf.io/b26vh/?action=download',
                'gm_model': 'https://osf.io/ugscu/?action=download'}
    tmp_file = 'tmp.data.zip'

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    data_name = arguments['-d']
    if '-v' in arguments:
        verbose = int(arguments['-v'])

    # Download data
    url = dict_url[data_name]
    try:
        download_from_url(url, tmp_file)
    except(KeyboardInterrupt):
        sct.printv('\nERROR: User canceled process.', 1, 'error')

    # Check if folder already exists
    sct.printv('Check if folder already exists...', verbose)
    if path.isdir(data_name):
        sct.printv('.. WARNING: Folder '+data_name+' already exists. Removing it...', 1, 'warning')
        shutil.rmtree(data_name, ignore_errors=True)

    # unzip
    sct.printv('Unzip dataset...', verbose)
    try:
        zf = zipfile.ZipFile(tmp_file)
        zf.extractall()
    except (zipfile.BadZipfile):
        sct.printv('\nERROR: ZIP package corrupted. Please try downloading again.', verbose, 'error')

    # if downloaded from GitHub, need to remove the "-master" suffix
    if 'master.zip' in url:
        sct.printv('Rename folder...', verbose)
        rename(data_name+'-master', data_name)

    # remove zip file
    sct.printv('Remove temporary file...', verbose)
    remove(tmp_file)

    # display stuff
    sct.printv('Done! Folder created: '+data_name+'\n', verbose, 'info')


def download_from_url(url, local):
    """
    Simple downloading with progress indicator
    :param url:
    :param local:
    :return:
    """
    with open(local, 'wb') as local_file:
        for i in range(3):
            try:
                time.sleep(0.5)
                response = requests.get(url, stream=True, timeout=10)
                if response.ok:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, local_file)
            except requests.exceptions.ConnectionError:
                pass


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    main()
