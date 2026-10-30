import shutil
import logging
import cgi
import tempfile
import urllib.parse
import tarfile
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
from pathlib import Path
import argparse
import textwrap

from ivadomed import utils as imed_utils
from ivadomed.keywords import IgnoredFolderKW


DICT_URL = {
    "data_example_spinegeneric": {
        "url": ["https://github.com/ivadomed/data_example_spinegeneric/archive/r20200825.zip"],
        "description": """10 randomly picked subject from
            `Spine Generic <https://github.com/spine-generic/data-multi-subject>`_.
            Used for Tutorial and example in Ivadomed."""},
    "data_testing": {
        "url": ["https://github.com/ivadomed/data-testing/archive/r20240130.zip"],
        "description": "Data Used for integration/unit test in Ivadomed."},
    "data_multi_testing": {
        "url": ["https://github.com/MotionCorrect/data_multi-sessions-contrasts/archive/refs/tags/v2022-01-06.zip"],
        "description": "Large Data Used for multi-session contrasts integration/unit test in Ivadomed."},
    "t2_tumor": {
        "url": ["https://github.com/ivadomed/t2_tumor/archive/r20200621.zip"],
        "description": "Cord tumor segmentation model, trained on T2-weighted contrast."},
    "t2star_sc": {
        "url": ["https://github.com/ivadomed/t2star_sc/archive/r20200622.zip"],
        "description": "spinal cord segmentation model, trained on T2-star contrast."},
    "mice_uqueensland_gm": {
        "url": ["https://github.com/ivadomed/mice_uqueensland_gm/archive/r20200622.zip"],
        "description": """Gray matter segmentation model on mouse MRI. Data from University of
            Queensland."""},
    "mice_uqueensland_sc": {
        "url": ["https://github.com/ivadomed/mice_uqueensland_sc/archive/r20200622.zip"],
        "description": "Cord segmentation model on mouse MRI. Data from University of Queensland."},
    "findcord_tumor": {
        "url": ["https://github.com/ivadomed/findcord_tumor/archive/r20200621.zip"],
        "description": "Cord localisation model, trained on T2-weighted images with tumor."},
    "model_find_disc_t1": {
        "url": ["https://github.com/ivadomed/model_find_disc_t1/archive/r20201013.zip"],
        "description": "Intervertebral disc detection model trained on T1-weighted images."},
    "model_find_disc_t2": {
        "url": ["https://github.com/ivadomed/model_find_disc_t2/archive/r20200928.zip"],
        "description": "Intervertebral disc detection model trained on T2-weighted images."},
    "data_functional_testing": {
        "url": ["https://github.com/ivadomed/data_functional_testing/archive/r20211002.zip"],
        "description": "Data used for functional testing in Ivadomed."},
    "data_axondeepseg_sem": {
        "url": ["https://github.com/axondeepseg/data_axondeepseg_sem/archive/r20211130.zip"],
        "description": """SEM dataset for AxonDeepSeg. 10 rat spinal cord samples with axon and myelin
            manual segmentation labels. Used for microscopy tutorial in ivadomed."""},
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True,
                        choices=(sorted(DICT_URL)),
                        help="Data to download", metavar=imed_utils.Metavar.str)
    parser.add_argument("-k", "--keep", action="store_true",
                        help="Keep existing data in destination directory")
    parser.add_argument("-o", "--output", required=False,
                        help="Output Folder.", metavar=imed_utils.Metavar.file)
    return parser


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def download_data(urls):
    """
    Function used to download the data form github or other mirrors
    Args:
        urls (list): List of urls to try.

    Returns:
        downloaded folder path
    """
    if isinstance(urls, str):
        urls = [urls]

    exceptions = []

    for url in urls:
        try:
            logger.info('Trying URL: %s' % url)
            retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
            session = requests.Session()
            session.mount('https://', HTTPAdapter(max_retries=retry))
            response = session.get(url, stream=True)
            response.raise_for_status()

            filename = Path(urllib.parse.urlparse(url).path).name
            if "Content-Disposition" in response.headers:
                _, content = cgi.parse_header(response.headers['Content-Disposition'])
                filename = content["filename"]

            # protect against directory traversal
            filename = Path(filename).name
            if not filename:
                # this handles cases where you're loading something like an index page
                # instead of a specific file. e.g. https://osf.io/ugscu/?action=view.
                raise ValueError("Unable to determine target filename for URL: %s" % (url,))

            tmp_path = Path(tempfile.mkdtemp(), filename)

            logger.info('Downloading: %s' % filename)

            with tmp_path.open(mode='wb') as tmp_file:
                total = int(response.headers.get('content-length', 1))

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)

            return str(tmp_path)

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
        logger.error("ERROR: ZIP package corrupted. Please try downloading again.")
        raise


def _format_bundles():
    def format_bundle(name, values):
        return f'`{name} <{values["url"]}>`_ : {values["description"]}'
    return str.join("\n", ["* %s" % format_bundle(name, values) for name, values in DICT_URL.items()])


def install_data(url, dest_folder, keep=False):
    """
    Download a data bundle from an URL and install it in the destination folder.

    Usage example ::

        ivadomed_download_data -d data_testing -o ivado_testing_data


    Existing data bundles:

{BUNDLES}

    .. note::
        The function tries to be smart about the data contents.
        Examples:


        a. If the archive only contains a `README.md`, and the destination folder is `${{dst}}`,
        `${{dst}}/README.md` will be created.
        Note: an archive not containing a single folder is commonly known as a "bomb" because
        it puts files anywhere in the current working directory.( see `Tarbomb
        <https://en.wikipedia.org/wiki/Tar_(computing)#Tarbomb>`_)


        b. If the archive contains a `${{dir}}/README.md`, and the destination folder is `${{dst}}`,
        `${{dst}}/README.md` will be created.
        Note: typically the package will be called `${{basename}}-${{revision}}.zip` and contain
        a root folder named `${{basename}}-${{revision}}/` under which all the other files will
        be located.
        The right thing to do in this case is to take the files from there and install them
        in `${{dst}}`.
        - Uses `download_data()` to retrieve the data.
        - Uses `unzip()` to extract the bundle.

    Args:
        url (string): URL or sequence thereof (if mirrors). For this package there is a dictionnary
            listing existing data bundle with their url. Type ivadomed_download_data -h to see possible value. Flag ``-d``
        dest_folder (string): destination directory for the data (to be created). If not used the output folder
            will be the name of the data bundle. Flag ``-o``, ``--output``
        keep (bool): whether to keep existing data in the destination folder (if it exists). Flag ``-k``, ``--keep``
    """

    if not keep and Path(dest_folder).exists():
        logger.warning("Removing existing destination folder “%s”", dest_folder)
        shutil.rmtree(dest_folder)
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    tmp_file = download_data(url)

    extraction_folder = tempfile.mkdtemp()

    unzip(tmp_file, extraction_folder)

    # Identify whether we have a proper archive or a tarbomb
    has_dir = False
    nb_entries = 0
    for path_object in Path(extraction_folder).iterdir():
        if path_object.name in (IgnoredFolderKW.MACOSX,):
            continue
        nb_entries += 1
        if path_object.is_dir():
            has_dir = True

    if nb_entries == 1 and has_dir:
        # tarball with single-directory -> go under
        for path_object in Path(extraction_folder).iterdir():
            if path_object.name in (IgnoredFolderKW.MACOSX,):
                continue
            bundle_folder = path_object
    else:
        # bomb scenario -> stay here
        bundle_folder: Path = Path(extraction_folder)

    # Copy over
    for path_object in bundle_folder.glob("**/*"):
        if path_object.is_dir():
            if path_object.name not in (IgnoredFolderKW.MACOSX,):
                relpath = path_object.relative_to(bundle_folder)
                dstpath = Path(dest_folder, relpath)
                if dstpath.exists():
                    logger.debug("- d- %s", str(relpath))
                else:
                    logger.debug("- d+ %s", relpath)
                    dstpath.mkdir(parents=True)
        if path_object.is_file():
            relpath = path_object.relative_to(bundle_folder)
            dstpath = Path(dest_folder, relpath)
            if dstpath.exists():
                logger.debug("- f! %s", relpath)
                logger.warning("Updating existing “%s”", dstpath)
                dstpath.unlink()
            else:
                logger.debug("- f+ %s", relpath)
            shutil.copy(str(path_object), str(dstpath))

    logger.info("Removing temporary folders...")
    logger.info("Folder Created: {}".format(dest_folder))
    shutil.rmtree(str(Path(tmp_file).parent))
    shutil.rmtree(extraction_folder)


# This line allows to format the `install_data()` docstrings, because this formatting
# cannot be done in the function directly.
# `create_string()` is a custom function that converts our dict into a string
# which is easier to add in the documentation.
install_data.__doc__ = install_data.__doc__.format(BUNDLES=textwrap.indent(_format_bundles(), ' '*6))


def main(args=None):
    imed_utils.init_ivadomed()

    # Dictionary containing list of URLs for data names.
    # Mirror servers are listed in order of decreasing priority.
    # If exists, favour release artifact straight from github

    parser = get_parser()
    arguments = imed_utils.get_arguments(parser, args)

    data_name = arguments.d

    if arguments.output is None:
        dest_folder = str(Path(Path.cwd().absolute(), data_name))
    else:
        dest_folder = arguments.output

    url = DICT_URL[data_name]["url"]
    install_data(url, dest_folder, keep=bool(arguments.keep))
    return 0


if __name__ == '__main__':
    main()
