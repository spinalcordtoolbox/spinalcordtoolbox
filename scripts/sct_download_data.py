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
from tqdm import tqdm

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
            'gm_model', 'optic_models', 'pmj_models', 'binaries_debian',
            'binaries_centos', 'binaries_osx', 'course_hawaii17',
            'deepseg_gm_models', 'deepseg_sc_models'
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
        'sct_example_data': ['https://osf.io/4nnk3/?action=download',
                             'https://www.neuro.polymtl.ca/_media/downloads/sct/20170208_sct_example_data.zip'],
        'sct_testing_data': ['https://osf.io/z8gaj/?action=download',
                             'https://www.neuro.polymtl.ca/_media/downloads/sct/20180125_sct_testing_data.zip'],
        'PAM50': ['https://osf.io/gdwn6/?action=download',
                  'https://www.neuro.polymtl.ca/_media/downloads/sct/20170101_PAM50.zip'],
        'MNI-Poly-AMU': ['https://osf.io/sh6h4/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170310_MNI-Poly-AMU.zip'],
        'gm_model': ['https://osf.io/ugscu/?action=download',
                     'https://www.neuro.polymtl.ca/_media/downloads/sct/20160922_gm_model.zip'],
        'optic_models': ['https://osf.io/g4fwn/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170413_optic_models.zip'],
        'pmj_models': ['https://osf.io/4gufr/?action=download',
                       'https://www.neuro.polymtl.ca/_media/downloads/sct/20170922_pmj_models.zip'],
        'binaries_debian': ['https://osf.io/2egh5/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20170915_sct_binaries_linux.tar.gz'],
        'binaries_centos': ['https://osf.io/qngj2/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20170915_sct_binaries_linux_centos6.tar.gz'],
        'binaries_osx': ['https://osf.io/hsa5r/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170915_sct_binaries_osx.tar.gz'],
        'course_hawaii17': 'https://osf.io/6exht/?action=download',
        'deepseg_gm_models': ['https://osf.io/b9y4x/?action=download',
                              'https://www.neuro.polymtl.ca/_media/downloads/sct/20180205_deepseg_gm_models.zip'],
        'deepseg_sc_models': ['https://osf.io/86phg/?action=download',
                              'https://www.neuro.polymtl.ca/_media/downloads/sct/20180125_deepseg_sc_models.zip']
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
            sct.printv('ERROR: ZIP package corrupted. Please try again.', verbose, 'error')
    else:
        sct.printv('ERROR: The file %s is of wrong format' % compressed, verbose, 'error')


def download_data(urls, verbose):
    """Download the binaries from a URL and return the destination filename

    Retry downloading if either server or connection errors occur on a SSL
    connection
    urls: list of several urls (mirror servers) or single url (string)
    """

    # if urls is not a list, make it one
    if not isinstance(urls, (list, tuple)):
        urls = [urls]

    # loop through URLs
    for url in urls:
        try:
            sct.printv('\nTrying URL: %s' % url, verbose)
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
            session = requests.Session()
            session.mount('https://', HTTPAdapter(max_retries=retry))
            response = session.get(url, stream=True)

            _, content = cgi.parse_header(response.headers['Content-Disposition'])
            tmp_path = os.path.join(tempfile.mkdtemp(), content['filename'])
            sct.printv('Downloading %s...' % content['filename'], verbose)

            with open(tmp_path, 'wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))
                tqdm_bar = tqdm(total=total, unit='B', unit_scale=True,
                                desc="Status", ascii=True)

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        if verbose > 0:
                            dl_chunk = len(chunk)
                            tqdm_bar.update(dl_chunk)

                tqdm_bar.close()
            return tmp_path

        except requests.RequestException as err:
            sct.printv(err.message, type='warning')
    else:
        sct.printv('\nDownload error', type='error')


if __name__ == "__main__":
    sct.start_stream_logger()
    main()
