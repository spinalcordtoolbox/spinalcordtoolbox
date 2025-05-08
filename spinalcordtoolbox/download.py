"""
Utilities for downloading

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import shutil
import logging
import cgi
import tempfile
import urllib.parse
import tarfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from spinalcordtoolbox.utils.fs import tmp_create
from spinalcordtoolbox.utils.sys import __bin_dir__, __sct_dir__, sct_progress_bar, stylize

logger = logging.getLogger(__name__)


# Dictionary containing list of URLs and locations for datasets.
# Mirror servers are listed in order of decreasing priority.
# If exists, favour release artifact straight from github
# For the location field, this is where the dataset will be
# downloaded to (relative to the repo) if a location isn't passed by
# the user.
DATASET_DICT = {
    "sct_example_data": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_example_data/releases/download/r20180525/20180525_sct_example_data.zip",
            "https://osf.io/kjcgs/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "sct_example_data"),
        "download_type": "Datasets",
        "default": False,
    },
    "sct_testing_data": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_testing_data/releases/download/r20250506/sct_testing_data-r20250506.zip",
            "https://osf.io/verw8/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "sct_testing_data"),
        "download_type": "Datasets",
        "default": False,
    },
    "sct_course_data": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/archive/refs/tags/r20250310.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "sct_course_data"),
        "download_type": "SCT Course Files",
        "default": False,
    },
    "manual-correction": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/manual-correction/archive/refs/tags/r20231101.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "manual-correction"),
        "download_type": "SCT Course Files",
        "default": False,
    },
    "PAM50": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/PAM50/releases/download/r20250422/PAM50-r20250422.zip",
            "https://osf.io/q3m4t/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "PAM50"),
        "download_type": "Templates",
        "default": True,
    },
    "MNI-Poly-AMU": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/MNI-Poly-AMU/releases/download/r20170310/20170310_MNI-Poly-AMU.zip",
            "https://osf.io/sh6h4/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "MNI-Poly-AMU"),
        "download_type": "Templates",
        "default": False,
    },
    "template-dog": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/template-dog/releases/download/r20240709/templatedog_r20240709.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "template-dog"),
        "download_type": "Templates",
        "default": False,
    },
    "binaries_linux": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/spinalcordtoolbox-binaries/releases/download/r20221109/spinalcordtoolbox-binaries_linux.tar.gz",
        ],
        "default_location": __bin_dir__,
        "download_type": "Binaries",
        "default": False,  # Technically 'True', but whether to download by default is platform-dependent
    },
    "binaries_osx": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/spinalcordtoolbox-binaries/releases/download/r20221018/spinalcordtoolbox-binaries_osx.tar.gz",
        ],
        "default_location": __bin_dir__,
        "download_type": "Binaries",
        "default": False,  # Technically 'True', but whether to download by default is platform-dependent
    },
    "binaries_win": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/spinalcordtoolbox-binaries/releases/download/r20221018/spinalcordtoolbox-binaries_windows.tar.gz",
        ],
        "default_location": __bin_dir__,
        "download_type": "Binaries",
        "default": False,  # Technically 'True', but whether to download by default is platform-dependent
    },
    "deepseg_gm_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/deepseg_gm_models/releases/download/r20180205/20220325_deepseg_gm_models_onnx.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepseg_gm_models"),
        "download_type": "Models",
        "default": True,
    },
    "deepseg_sc_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/deepseg_sc_models/releases/download/r20180610/20220325_deepseg_sc_models_onnx.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepseg_sc_models"),
        "download_type": "Models",
        "default": True,
    },
    "deepseg_lesion_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/deepseg_lesion_models/releases/download/r20180613/20220325_deepseg_lesion_models_onnx.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepseg_lesion_models"),
        "download_type": "Models",
        "default": True,
    },
    "exvivo_template": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/exvivo-template/archive/refs/tags/r20210317.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "exvivo_template"),
        "download_type": "Templates",
        "default": False,
    },
    "deepreg_models": {
        "mirrors": [
            "https://github.com/ivadomed/multimodal-registration/releases/download/r20220512/models.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepreg_models"),
        "download_type": "Models",
        "default": True,
    },
    "PAM50_normalized_metrics": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/PAM50-normalized-metrics/archive/refs/tags/r20250321.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "PAM50_normalized_metrics"),
        "download_type": "Templates",
        "default": True,
    },
}


def download_data(urls):
    """Download the binaries from a URL and return the destination filename

    Retry downloading if either server or connection errors occur on a SSL
    connection
    urls: list of several urls (mirror servers) or single url (string)
    :return: tmp_path: the path to the tmp directory where the filename was downloaded to
    :return: url_used: the URL that was successfully used to download the bundle
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
                filename = content.get("filename", filename)  # Fall-back on original 'filename' if header is malformed

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
                sct_bar = sct_progress_bar(total=total, unit='B', unit_scale=True, desc="Status", position=0)

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        dl_chunk = len(chunk)
                        sct_bar.update(dl_chunk)

                sct_bar.close()
            return tmp_path, url

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
    except Exception:
        logger.error('ERROR: ZIP package corrupted. Please try downloading again.')
        raise


def install_data(url, dest_folder, keep=False, dirs_to_preserve=()):
    """
    Download a data bundle from a URL and install in the destination folder.

    :param url: URL or sequence thereof (if mirrors).
    :param dest_folder: destination directory for the data (to be created).
    :param keep: whether to keep existing data in the destination folder.
    :param dirs_to_preserve: Tuple of substrings where, if found in subdirectory name, the dir will be preserved.
    :return: url_used: the URL that was successfully used to download the bundle

    .. note::
        The function tries to be smart about the data contents.

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
        logger.warning("Removing existing destination folder '%s'", dest_folder)
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder, exist_ok=True)

    tmp_file, url_used = download_data(url)

    extraction_folder = tmp_create(basename="install-data")

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
                elif any((s in entry.name) for s in dirs_to_preserve):
                    bundle_folder = extraction_folder
                else:
                    bundle_folder = entry.path
    else:
        # bomb scenario -> stay here
        bundle_folder = extraction_folder

    # Copy over
    logger.info(f"Copying data to: {dest_folder}")
    for cwd, ds, fs in os.walk(bundle_folder):
        ds.sort()
        fs.sort()
        ds[:] = [d for d in ds if d not in ("__MACOSX",)]
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
                logger.warning("Updating existing '%s'", dstpath)
                os.unlink(dstpath)
            else:
                logger.debug("- f+ %s", relpath)
            shutil.copy(srcpath, dstpath)

    logger.info("Removing temporary folders...")
    shutil.rmtree(os.path.dirname(tmp_file))
    shutil.rmtree(extraction_folder)

    return url_used


def install_named_dataset(dataset_name, dest_folder=None, keep=False):
    """
    A light wrapper for the 'install_data' function to allow downstream consumers to download
    datasets using only the dataset's name (i.e. without needing to access DATASET_DICT fields).
    """
    if dataset_name not in DATASET_DICT.keys():
        # This `lambda` accounts for capitals (A, a, B, b), see https://stackoverflow.com/a/10269828
        raise ValueError(f"Dataset '{dataset_name}' is not contained in list of datasets. Choose from:\n\n "
                         f"{sorted(list(DATASET_DICT.keys()), key=str.casefold)}")

    urls = DATASET_DICT[dataset_name]["mirrors"]
    if dest_folder is None:
        dest_folder = DATASET_DICT[dataset_name]["default_location"]

    install_data(urls, dest_folder, keep)


def default_datasets():
    return [name_model for name_model, value in DATASET_DICT.items() if value['default']]


def install_default_datasets(keep=False):
    """
    Download all default datasets and install them under SCT installation dir.

    :return: None
    """
    for name_model in default_datasets():
        install_named_dataset(name_model, keep=keep)


def is_installed(dataset_name):
    """
    :param dataset_name: Dataset name
    :returns: Whether the dataset is installed in $SCT_DIR.
    :rtype: bool
    """
    path_dataset = DATASET_DICT[dataset_name]['default_location']
    return os.path.exists(path_dataset) and len(os.listdir(path_dataset)) > 0


def list_datasets():
    """
    :returns: A table listing the downloadable datasets
    :rtype: str
    """
    color = {True: 'LightGreen', False: 'LightRed'}
    table = f"{'DATASET NAME':<30s}{'TYPE':<20s}\n"
    table += f"{'-' * 50}\n"
    sorted_datasets = sorted(DATASET_DICT,
                             key=lambda k: DATASET_DICT[k]['download_type'] + k)
    for dataset_name in sorted_datasets:
        download_type = DATASET_DICT[dataset_name]['download_type']
        dataset_status = dataset_name.ljust(30)
        if download_type != "Binaries":
            dataset_status = stylize(dataset_status, color[is_installed(dataset_name)])
        table += f"{dataset_status}{download_type:<20s}\n"

    table += '\nLegend: {} | {} (in the $SCT_DIR/data folder)\n\n'.format(
            stylize("installed", color[True]),
            stylize("not installed", color[False]))

    return table
