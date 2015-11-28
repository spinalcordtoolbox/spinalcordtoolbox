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

import sys
from os import remove, rename, path
from urllib import urlretrieve
import zipfile
from sct_utils import run, printv, check_folder_exist
from msct_parser import Parser


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)
    parser.usage.set_description('''Download dataset from the web.''')
    parser.add_option(name="-d",
                      type_value="multiple_choice",
                      description="Name of the dataset.",
                      mandatory=True,
                      example=['sct_example_data', 'sct_testing_data', 'PAM50'])
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


# MAIN
# ==========================================================================================
def main(args=None):

    # initialization
    verbose = 1
    dict_url = {'sct_example_data': 'https://github.com/neuropoly/sct_example_data/archive/master.zip',
                'sct_testing_data': 'https://github.com/neuropoly/sct_testing_data/archive/master.zip',
                'PAM50': 'https://dl.dropboxusercontent.com/u/20592661/sct/PAM50.zip'}
    tmp_file = 'tmp.data.zip'

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    data_name = arguments["-d"]
    if '-v' in arguments:
        verbose = int(arguments['-v'])

    # Download data
    url = dict_url[data_name]
    printv('\nDownload dataset from: '+url, verbose)
    urlretrieve(url, tmp_file)

    # Check if folder already exists
    printv('Check if folder already exists...', verbose)
    if path.isdir(data_name):
        printv('.. WARNING: Folder '+data_name+' already exists. Removing it...', 1, 'warning')
        run('rm -rf '+data_name, 0)

    # unzip
    printv('Unzip dataset...', verbose)
    zf = zipfile.ZipFile(tmp_file)
    zf.extractall()

    # if downloaded from GitHub, need to remove the "-master" suffix
    if 'master.zip' in url:
        printv('Rename folder...', verbose)
        rename(data_name+'-master', data_name)

    # remove zip file
    printv('Remove temporary file...', verbose)
    remove(tmp_file)

    # display stuff
    printv('Done! Folder created: '+data_name+'\n', verbose, 'info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()
