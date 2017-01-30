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

import httplib
import os
import shutil
import sys
import tarfile
import time
import urllib2
import zipfile

import requests

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
        'MNI-Poly-AMU': 'https://osf.io/b26vh/?action=download',
        'gm_model': 'https://osf.io/ugscu/?action=download',
        'binaries_debian': 'https://osf.io/2pztn/?action=download',
        'binaries_centos': 'https://osf.io/4wbgt/?action=download',
        'binaries_osx': 'https://osf.io/ceg8p/?action=download'
    }
    tmp_file = 'tmp.data.zip'

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    data_name = arguments['-d']
    verbose = int(arguments['-v'])
    dest_folder = arguments.get('-o', os.curdir)

    # Download data
    url = dict_url[data_name]
    try:
        # download_from_url(url, tmp_file)
        tmp_file = download_data(url, verbose)
    except (KeyboardInterrupt):
        printv('\nERROR: User canceled process.', 1, 'error')

    unzip(tmp_file, dest_folder, verbose)

    printv('Remove temporary file...', verbose)
    os.remove(tmp_file)

    printv('Done! Folder created: ' + dest_folder + '\n', verbose, 'info')


def unzip(compressed, dest_folder, verbose):
    """Extract compressed file to the dest_folder"""
    printv('Unzip dataset...', verbose)
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
    """Download the binaries from a URL and return the destination filename"""
    response = requests.get(url, stream=True)
    import re
    import tempfile

    filename = re.findall('filename="?([\w\.]+)"?',
                          response.headers['Content-Disposition'])
    tmp_path = os.path.join(tempfile.mkdtemp(), filename[0])

    with open(tmp_path, 'wb') as tmp_file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                tmp_file.write(chunk)

    printv('Download complete %s' % filename, verbose=verbose)
    return tmp_path


def download_from_url(url, local):
    """
    Simple downloading with progress indicator, by Cees Timmerman, 16mar12.
    :param url:
    :param local:
    :return:
    """
    keep_connecting = True
    i_trial = 1
    max_trials = 3

    print 'Reaching URL: ' + url
    while keep_connecting:
        try:
            u = urllib2.urlopen(url)
        except urllib2.HTTPError, e:
            printv('\nHTTPError = ' + str(e.code), 1, 'error')
        except urllib2.URLError, e:
            printv('\nURLError = ' + str(e.reason), 1, 'error')
        except httplib.HTTPException, e:
            printv('\nHTTPException', 1, 'error')
        except (KeyboardInterrupt):
            printv('\nERROR: User canceled process.', 1, 'error')
        except Exception:
            import traceback
            printv('\nERROR: Cannot open URL: ' + traceback.format_exc(), 1,
                   'error')
        h = u.info()
        try:
            totalSize = int(h["Content-Length"])
            keep_connecting = False
        except:
            # if URL was badly reached (issue #895):
            # send warning message
            printv(
                '\nWARNING: URL cannot be reached. Trying again (maximum trials: '
                + str(max_trials) + ').', 1, 'warning')
            # pause for 0.5s
            time.sleep(0.5)
            # iterate i_trial and try again
            i_trial += 1
            # if i_trial exceeds max_trials, exit with error
            if i_trial > max_trials:
                printv(
                    '\nERROR: Maximum number of trials reached. Try again later.',
                    1, 'error')
                keep_connecting = False

    print "Downloading %s bytes..." % totalSize,
    fp = open(local, 'wb')

    blockSize = 8192
    count = 0
    while True:
        chunk = u.read(blockSize)
        if not chunk:
            break
        fp.write(chunk)
        count += 1
        if totalSize > 0:
            percent = int(count * blockSize * 100 / totalSize)
            if percent > 100:
                percent = 100
            print "%2d%%" % percent,
            if percent < 100:
                print "\b\b\b\b\b",
            else:
                print "Done."

    fp.flush()
    fp.close()
    if not totalSize:
        print


if __name__ == "__main__":
    main()
