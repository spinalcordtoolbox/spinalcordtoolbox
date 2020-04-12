#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with data download and installation from the Internet.

import os
import shutil
import distutils.dir_util
import logging
import cgi
import tempfile
import tarfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from tqdm import tqdm

import spinalcordtoolbox as sct
import spinalcordtoolbox.utils


logger = logging.getLogger(__name__)

# Dictionary containing list of URLs for data names. Mirror servers are listed in order of priority.
DICT_URL = {
    'sct_example_data': ['https://osf.io/kjcgs/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20180525_sct_example_data.zip'],
    'sct_testing_data': ['https://osf.io/yutrj/?action=download',
                         'https://www.neuro.polymtl.ca/_media/downloads/sct/20191031_sct_testing_data.zip'],
    'PAM50': ['https://osf.io/u79sr/?pid=6zbyf/?action=download',
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


def download_data(urls):
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
            logger.info('Trying URL: %s' % url)
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
            session = requests.Session()
            session.mount('https://', HTTPAdapter(max_retries=retry))
            response = session.get(url, stream=True)

            if "Content-Disposition" in response.headers:
                _, content = cgi.parse_header(response.headers['Content-Disposition'])
                filename = content["filename"]
            else:
                logger.warning("Unexpected: link doesn't provide a filename")
                continue

            tmp_path = os.path.join(tempfile.mkdtemp(), filename)
            logger.info('Downloading: %s' % filename)

            with open(tmp_path, 'wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))
                tqdm_bar = tqdm(total=total, unit='B', unit_scale=True, desc="Status", ascii=False, position=0)

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        dl_chunk = len(chunk)
                        tqdm_bar.update(dl_chunk)

                tqdm_bar.close()
            return tmp_path

        except Exception as e:
            logger.warning("Link download error, trying next mirror (error was: %s)" % e)
    else:
        logger.error('Download error')


def unzip(compressed, dest_folder):
    """
    Extract compressed file to the dest_folder. Can handle .zip, .tar.gz.
    """
    logger.info('Unzip data to: %s' % dest_folder)
    if compressed.endswith('zip'):
        try:
            zf = zipfile.ZipFile(compressed)
            zf.extractall(dest_folder)
            return
        except (zipfile.BadZipfile):
            logger.error("ZIP package corrupted. Please try downloading again.")
    elif compressed.endswith('tar.gz'):
        try:
            tar = tarfile.open(compressed)
            tar.extractall(path=dest_folder)
            return
        except tarfile.TarError:
            logger.error("ZIP package corrupted. Please try again.")
    else:
        logger.error("The file %s is of wrong format" % compressed)


def install_data(url, dest_folder):
    """
    Download data from a URL and install in the appropriate folder. Deals with multiple mirrors, retry download,
    check if data already exist.
    :param url: string or list or strings. URL or list of URLs (if mirrors).
    :param dest_folder:
    :return:
    """

    # Download data
    tmp_file = download_data(url)

    # unzip
    dest_tmp_folder = sct.utils.tmp_create()
    unzip(tmp_file, dest_tmp_folder)
    extracted_files_paths = []
    # Get the name of the extracted files and directories
    extracted_files = os.listdir(dest_tmp_folder)
    for extracted_file in extracted_files:
        extracted_files_paths.append(os.path.join(os.path.abspath(dest_tmp_folder), extracted_file))

    # Check if files and folder already exists
    logger.info("Destination folder: {}".format(dest_folder))
    logger.info("Checking if folder already exists...")
    for data_extracted_name in extracted_files:
        fullpath_dest = os.path.join(dest_folder, data_extracted_name)
        if os.path.isdir(fullpath_dest):
            logger.warning("Folder {} already exists. Removing it...".format(data_extracted_name))
            shutil.rmtree(fullpath_dest)

    # Destination path
    # for source_path in extracted_files_paths:
        # Copy the content of source to destination (and create destination folder)
    distutils.dir_util.copy_tree(dest_tmp_folder, dest_folder)

    logger.info("Removing temporary folders...")
    try:
        shutil.rmtree(os.path.split(tmp_file)[0])
        shutil.rmtree(dest_tmp_folder)
    except Exception as error:
        logger.error("Cannot remove temp folder: " + repr(error))
