#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with data download and installation from the Internet.

import os
import shutil
import distutils.dir_util
import logging
import cgi
import tempfile
import urllib.parse
import tarfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from spinalcordtoolbox.utils import sct_progress_bar

import spinalcordtoolbox as sct
import spinalcordtoolbox.utils


logger = logging.getLogger(__name__)


def download_data(urls):
    """Download the binaries from a URL and return the destination filename

    Retry downloading if either server or connection errors occur on a SSL
    connection
    urls: list of several urls (mirror servers) or single url (string)
    """

    # make sure urls becomes a list, in case user inputs a str
    if isinstance(urls, str):
        urls = [urls]

    # loop through URLs
    exceptions = []
    for url in urls:
        try:
            logger.info('Trying URL: %s' % url)
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
            session = requests.Session()
            session.mount('https://', HTTPAdapter(max_retries=retry))
            response = session.get(url, stream=True)
            response.raise_for_status()

            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if "Content-Disposition" in response.headers:
                _, content = cgi.parse_header(response.headers['Content-Disposition'])
                filename = content["filename"]

            # protect against directory traversal
            filename = os.path.basename(filename)
            if not filename:
                # this handles cases where you're loading something like an index page
                # instead of a specific file. e.g. https://osf.io/ugscu/?action=view.
                raise ValueError("Unable to determine target filename for URL: %s" % (url,))

            tmp_path = os.path.join(tempfile.mkdtemp(), filename)

            logger.info('Downloading: %s' % filename)

            with open(tmp_path, 'wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))
                sct_bar = sct_progress_bar(total=total, unit='B', unit_scale=True, desc="Status", ascii=False, position=0)

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        dl_chunk = len(chunk)
                        sct_bar.update(dl_chunk)

                sct_bar.close()
            return tmp_path

        except Exception as e:
            logger.warning("Link download error, trying next mirror (error was: %s)" % e)
            exceptions.append(e)
    else:
        raise Exception('Download error', exceptions)


def unzip(compressed, dest_folder):
    """
    Extract compressed file to the dest_folder. Can handle .zip, .tar.gz.
    """
    logger.info('Unzip data to: %s' % dest_folder)

    formats = {'.zip': zipfile.ZipFile,
               '.tar.gz': tarfile.open,
               '.tgz': tarfile.open}
    for format, open in formats.items():
        if compressed.lower().endswith(format):
            break
    else:
        raise TypeError('ERROR: The file %s is of wrong format' % (compressed,))

    try:
        open(compressed).extractall(dest_folder)
    except:
        sct.printv('ERROR: ZIP package corrupted. Please try downloading again.', verbose, 'error')
        raise


def install_data(url, dest_folder, keep=False):
    """
    Download a data bundle from a URL and install in the destination folder.

    :param url: URL or sequence thereof (if mirrors).
    :param dest_folder: destination directory for the data (to be created).
    :param keep: whether to keep existing data in the destination folder.
    :return: None

    Notes:

    - The function tries to be smart about the data contents.

      Examples:

      a. If the archive only contains a `README.md`, and the destination folder is `${dst}`,
         `${dst}/README.md` will be created.

         Note: an archive not containing a single folder is commonly known as a "bomb" because
         it puts files anywhere in the current working directory.
         https://en.wikipedia.org/wiki/Tar_(computing)#Tarbomb

      b. If the archive contains a `${dir}/README.md`, and the destination folder is `${dst}`,
         `${dst}/README.md` will be created.

         Note: typically the package will be called `${basename}-${revision}.zip` and contain
         a root folder named `${basename}-${revision}/` under which all the other files will
         be located.
         The right thing to do in this case is to take the files from there and install them
         in `${dst}`.

    - Uses `download_data()` to retrieve the data.
    - Uses `unzip()` to extract the bundle.
    """

    if not keep and os.path.exists(dest_folder):
        logger.warning("Removing existing destination folder “%s”", dest_folder)
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

    tmp_file = download_data(url)

    extraction_folder = sct.utils.tmp_create()

    unzip(tmp_file, extraction_folder)

    # Identify whether we have a proper archive or a tarbomb
    with os.scandir(extraction_folder) as it:
        has_dir = False
        nb_entries = 0
        for entry in it:
            if entry.name in ("__MACOSX",):
                continue
            nb_entries += 1
            if entry.is_dir():
                has_dir = True

    if nb_entries == 1 and has_dir:
        # tarball with single-directory -> go under
        with os.scandir(extraction_folder) as it:
            for entry in it:
                if entry.name in ("__MACOSX",):
                    continue
                bundle_folder = entry.path
    else:
        # bomb scenario -> stay here
        bundle_folder = extraction_folder

    # Copy over
    for cwd, ds, fs in os.walk(bundle_folder):
        ds.sort()
        fs.sort()
        ds[:] = [ d for d in ds if d not in ("__MACOSX",) ]
        for d in ds:
            srcpath = os.path.join(cwd, d)
            relpath = os.path.relpath(srcpath, bundle_folder)
            dstpath = os.path.join(dest_folder, relpath)
            if os.path.exists(dstpath):
                # lazy -- we assume existing is a directory, otherwise it will crash safely
                logger.debug("- d- %s", relpath)
            else:
                logger.debug("- d+ %s", relpath)
                os.makedirs(dstpath)

        for f in fs:
            srcpath = os.path.join(cwd, f)
            relpath = os.path.relpath(srcpath, bundle_folder)
            dstpath = os.path.join(dest_folder, relpath)
            if os.path.exists(dstpath):
                logger.debug("- f! %s", relpath)
                logger.warning("Updating existing “%s”", dstpath)
                os.unlink(dstpath)
            else:
                logger.debug("- f+ %s", relpath)
            shutil.copy(srcpath, dstpath)

    logger.info("Removing temporary folders...")
    shutil.rmtree(os.path.dirname(tmp_file))
    shutil.rmtree(extraction_folder)
