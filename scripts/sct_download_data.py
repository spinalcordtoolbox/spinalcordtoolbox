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
# import time
# from urllib import urlretrieve
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
    download_from_url(url, tmp_file)
    # try:
    #     printv('\nDownload data from: '+url, verbose)
    #     urlretrieve(url, tmp_file)
    #     # Allow time for data to download/save:
    #     print "hola1"
    #     time.sleep(0.5)
    #     print "hola2"
    # except:
    #     printv("ERROR: Download Failed.", verbose, 'error')

    # Check if folder already exists
    printv('Check if folder already exists...', verbose)
    if path.isdir(data_name):
        printv('.. WARNING: Folder '+data_name+' already exists. Removing it...', 1, 'warning')
        run('rm -rf '+data_name, 0)

    # unzip
    printv('Unzip dataset...', verbose)
    try:
        zf = zipfile.ZipFile(tmp_file)
        zf.extractall()
    except (zipfile.BadZipfile):
        printv('ERROR: ZIP package corrupted. Please try downloading again.', verbose, 'error')

    # if downloaded from GitHub, need to remove the "-master" suffix
    if 'master.zip' in url:
        printv('Rename folder...', verbose)
        rename(data_name+'-master', data_name)

    # remove zip file
    printv('Remove temporary file...', verbose)
    remove(tmp_file)

    # display stuff
    printv('Done! Folder created: '+data_name+'\n', verbose, 'info')



def download_from_url(url, local):
    """
    Simple downloading with progress indicator, by Cees Timmerman, 16mar12.
    :param url:
    :param local:
    :return:
    """
    import urllib2

    try:
        u = urllib2.urlopen(url)
    except:
        printv('ERROR: Cannot open URL.', 1, 'error')
    h = u.info()
    totalSize = int(h["Content-Length"])

    print "Downloading %s bytes..." % totalSize,
    fp = open(local, 'wb')

    blockSize = 8192 #100000 # urllib.urlretrieve uses 8192
    count = 0
    while True:
        chunk = u.read(blockSize)
        if not chunk: break
        fp.write(chunk)
        count += 1
        if totalSize > 0:
            percent = int(count * blockSize * 100 / totalSize)
            if percent > 100: percent = 100
            print "%2d%%" % percent,
            if percent < 100:
                print "\b\b\b\b\b",  # Erase "NN% "
            else:
                print "Done."

    fp.flush()
    fp.close()
    if not totalSize:
        print



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()
