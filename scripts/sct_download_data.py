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

import cgi
import os
import sys
import tarfile
import tempfile
import zipfile
import shutil

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

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
            'sct_example_data', 'sct_testing_data', 'PAM50', 'MNI-Poly-AMU',
            'gm_model', 'optic_models', 'binaries_debian', 'binaries_centos', 'binaries_osx', 'course_hawaii17'
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
    dict_url = {
        'sct_example_data': 'https://osf.io/4nnk3/?action=download',
        'sct_testing_data': 'https://osf.io/uqcz5/?action=download',
        'PAM50': 'https://osf.io/gdwn6/?action=download',
        'MNI-Poly-AMU': 'https://osf.io/sh6h4/?action=download',
        'gm_model': 'https://osf.io/ugscu/?action=download',
        'optic_models': 'https://osf.io/g4fwn/?action=download',
        'binaries_debian': 'https://osf.io/a83jr/?action=download',
        'binaries_centos': 'https://osf.io/sgy6x/?action=download',
        'binaries_osx': 'https://osf.io/rtzey/?action=download',
        'course_hawaii17': 'https://osf.io/6exht/?action=download'
    }

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    data_name = arguments['-d']
    verbose = int(arguments['-v'])
    dest_folder = sct.slash_at_the_end(arguments.get('-o', os.path.abspath(os.curdir)), 1)

    # Download data
    url = dict_url[data_name]
    try:
        tmp_file = download_data(url, verbose)
    except (KeyboardInterrupt):
        sct.printv('\nERROR: User canceled process.\n', 1, 'error')

    # Check if folder already exists
    sct.printv('\nCheck if folder already exists...', verbose)
    if os.path.isdir(data_name):
        sct.printv('WARNING: Folder ' + data_name + ' already exists. Removing it...', 1, 'warning')
        shutil.rmtree(data_name, ignore_errors=True)

    # unzip
    unzip(tmp_file, dest_folder, verbose)

    sct.printv('\nRemove temporary file...', verbose)
    os.remove(tmp_file)

    sct.printv('Done!\n', verbose)


def unzip(compressed, dest_folder, verbose):
    """Extract compressed file to the dest_folder"""
    sct.printv('\nUnzip data to: %s' % dest_folder, verbose)
    if compressed.endswith('zip'):
        try:
            zf = zipfile.ZipFile(compressed)
            zf.extractall(dest_folder)
            return
        except (zipfile.BadZipfile):
            sct.printv(
                'ERROR: ZIP package corrupted. Please try downloading again.',
                verbose, 'error')
    elif compressed.endswith('tar.gz'):
        try:
            tar = tarfile.open(compressed)
            tar.extractall(path=dest_folder)
            return
        except tarfile.TarError:
            sct.printv('ERROR: ZIP package corrupted. Please try again.',
                   verbose, 'error')
    else:
        sct.printv('ERROR: The file %s is of wrong format' % compressed, verbose,
               'error')


def download_data(url, verbose):
    """Download the binaries from a URL and return the destination filename

    Retry downloading if either server or connection errors occur on a SSL
    connection
    """

    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retry))
    response = session.get(url, stream=True)

    _, content = cgi.parse_header(response.headers['Content-Disposition'])
    tmp_path = os.path.join(tempfile.mkdtemp(), content['filename'])
    sct.printv('\nDownloading %s...' % content['filename'], verbose)

    with open(tmp_path, 'wb') as tmp_file:
        total = int(response.headers.get('content-length', 1))
        dl = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
                if verbose > 0:
                    dl += len(chunk)
                    done = min(int(20 * dl / total), 20)
                    sys.stdout.write("\r[%s%s] Total: %s MB" % ('=' * done, ' ' * (20 - done), "{:,}".format(total/1000)))
                    sys.stdout.flush()

    sct.printv('\nDownload complete', verbose=verbose)
    return tmp_path


if __name__ == "__main__":
    main()
