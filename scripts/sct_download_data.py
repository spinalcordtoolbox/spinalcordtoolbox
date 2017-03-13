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

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry

from msct_parser import Parser
from sct_utils import printv


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
            'gm_model', 'binaries_debian', 'binaries_centos', 'binaries_osx'
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
        'sct_example_data': 'https://osf.io/feuef/?action=download',
        'sct_testing_data': 'https://osf.io/uqcz5/?action=download',
        'PAM50': 'https://osf.io/gdwn6/?action=download',
        'MNI-Poly-AMU': 'https://osf.io/sh6h4/?action=download',
        'gm_model': 'https://osf.io/ugscu/?action=download',
        'binaries_debian': 'https://osf.io/2pztn/?action=download',
        'binaries_centos': 'https://osf.io/4wbgt/?action=download',
        'binaries_osx': 'https://osf.io/ceg8p/?action=download'
    }

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    data_name = arguments['-d']
    verbose = int(arguments['-v'])
    dest_folder = arguments.get('-o', os.path.abspath(os.curdir))

    # Download data
    url = dict_url[data_name]
    try:
        tmp_file = download_data(url, verbose)
    except (KeyboardInterrupt):
        printv('\nERROR: User canceled process.\n', 1, 'error')

    unzip(tmp_file, dest_folder, verbose)

    printv('Remove temporary file...\n', verbose)
    os.remove(tmp_file)

    printv('Done! Folder created: %s\n' % dest_folder, verbose, 'info')


def unzip(compressed, dest_folder, verbose):
    """Extract compressed file to the dest_folder"""
    printv('Copy binaries to %s\n' % dest_folder, verbose)
    printv('Unzip dataset...\n', verbose)
    if compressed.endswith('zip'):
        try:
            zf = zipfile.ZipFile(compressed)
            zf.extractall(dest_folder)
            return
        except (zipfile.BadZipfile):
            printv(
                'ERROR: ZIP package corrupted. Please try downloading again.',
                verbose, 'error')
    elif compressed.endswith('tar.gz'):
        try:
            tar = tarfile.open(compressed)
            tar.extractall(path=dest_folder)
            return
        except tarfile.TarError:
            printv('ERROR: ZIP package corrupted. Please try again.',
                   verbose, 'error')
    else:
        printv('ERROR: The file %s is of wrong format' % compressed, verbose,
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
    printv('Downloading %s\n' % content['filename'], verbose)

    with open(tmp_path, 'wb') as tmp_file:
        total = int(response.headers.get('content-length', 1))
        dl = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp_file.write(chunk)
                if verbose > 1:
                    dl += len(chunk)
                    done = min(int(20 * dl / total), 20)
                    sys.stdout.write("\r[%s%s]" % ('=' * done,
                                                   ' ' * (20-done)))
                    sys.stdout.flush()

    printv('\nDownload complete %s' % content['filename'], verbose=verbose)
    return tmp_path


if __name__ == "__main__":
    main()
