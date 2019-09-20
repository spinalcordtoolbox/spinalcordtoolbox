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

import cgi
import os
import sys
import tarfile
import tempfile
import zipfile
from shutil import rmtree, move, copytree

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
        'sct_testing_data': ['https://osf.io/z8gaj/?action=download',
                             'https://www.neuro.polymtl.ca/_media/downloads/sct/20180125_sct_testing_data.zip'],
        'PAM50': ['https://osf.io/kc3jx/?action=download',
                  'https://www.neuro.polymtl.ca/_media/downloads/sct/20181214_PAM50.zip'],
        'MNI-Poly-AMU': ['https://osf.io/sh6h4/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170310_MNI-Poly-AMU.zip'],
        'gm_model': ['https://osf.io/ugscu/?action=download',
                     'https://www.neuro.polymtl.ca/_media/downloads/sct/20160922_gm_model.zip'],
        'optic_models': ['https://osf.io/g4fwn/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20170413_optic_models.zip'],
        'pmj_models': ['https://osf.io/4gufr/?action=download',
                       'https://www.neuro.polymtl.ca/_media/downloads/sct/20170922_pmj_models.zip'],
        'binaries_debian': ['https://osf.io/z72vn/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20181204_sct_binaries_linux.tar.gz'],
        'binaries_centos': ['https://osf.io/97ybd/?action=download',
                            'https://www.neuro.polymtl.ca/_media/downloads/sct/20181204_sct_binaries_linux_centos6.tar.gz'],
        'binaries_osx': ['https://osf.io/zjv4c/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20181204_sct_binaries_osx.tar.gz'],
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
    tmp_file = download_data(url, verbose)

    # unzip
    dest_tmp_folder = sct.tmp_create()
    unzip(tmp_file, dest_tmp_folder, verbose)
    extracted_files_paths = []
    # Get the name of the extracted files and directories
    extracted_files = os.listdir(dest_tmp_folder)
    for extracted_file in extracted_files:
        extracted_files_paths.append(os.path.join(os.path.abspath(dest_tmp_folder), extracted_file))

    # Check if files and folder already exists
    sct.printv('\nCheck if files or folder already exists on the destination path...', verbose)
    for data_extracted_name in extracted_files:
        fullpath_dest = os.path.join(dest_folder, data_extracted_name)
        if os.path.isdir(fullpath_dest):
            sct.printv("Folder {} already exists. Removing it...".format(data_extracted_name), 1, 'warning')
            rmtree(fullpath_dest)
        elif os.path.isfile(fullpath_dest):
            sct.printv("File {} already exists. Removing it...".format(data_extracted_name), 1, 'warning')
            os.remove(fullpath_dest)

    # Destination path
    for source_path in extracted_files_paths:
        # Move the content of source to destination
        move(source_path, dest_folder, copy_function=copytree)

    sct.printv('\nRemove temporary files...', verbose)
    os.remove(tmp_file)
    rmtree(dest_tmp_folder)

    sct.printv('Done!\n', verbose)
    return 0


def unzip(compressed, dest_folder, verbose):
    """Extract compressed file to the dest_folder. Can handle .zip, .tar.gz."""
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

            if "Content-Disposition" in response.headers:
                _, content = cgi.parse_header(response.headers['Content-Disposition'])
                filename = content["filename"]
            else:
                sct.printv("Unexpected: link doesn't provide a filename", type="warning")
                continue

            tmp_path = os.path.join(tempfile.mkdtemp(), filename)
            sct.printv('Downloading %s...' % filename, verbose)

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

        except Exception as e:
            sct.printv("Link download error, trying next mirror (error was: %s)" % e, type='warning')
    else:
        sct.printv('\nDownload error', type='error')


if __name__ == "__main__":
    sct.init_sct()
    res = main()
    raise SystemExit(res)
