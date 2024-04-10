"""
Quality Control report generator

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

from contextlib import contextmanager
import datetime
import importlib.resources
from importlib.abc import Traversable
import json
import logging
import math
import os
from pathlib import Path
import string
from typing import Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patheffects as path_effects
import numpy as np
import portalocker
from scipy.ndimage import center_of_mass
import skimage.exposure

from spinalcordtoolbox.image import Image
import spinalcordtoolbox.reports
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils.shell import display_open
from spinalcordtoolbox.utils.sys import __version__, list2cmdline

logger = logging.getLogger(__name__)


@contextmanager
def locked_file(path: Path):
    """
    Open and lock a file for reading and/or writing.

    Any other process that tries to lock the file will wait until this lock is released.
    """
    # Make sure the file exists before trying to lock it
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    # NB: We use 'r+' because it allows us to open an existing file for
    # locking *without* immediately truncating the existing contents prior
    # to opening. We can then use this file to overwrite the contents later.
    file = path.open('r+', encoding='utf-8')
    portalocker.lock(file, portalocker.LOCK_EX)
    try:
        # Let the caller use the open, locked file
        yield file
    finally:
        # Safely release the lock
        file.flush()
        os.fsync(file.fileno())
        portalocker.unlock(file)
        file.close()


@contextmanager
def create_qc_entry(
    path_input: Path,
    path_qc: Path,
    command: str,
    cmdline: str,
    plane: str,
    dataset: Optional[str],
    subject: Optional[str],
):
    """
    Generate a new QC report entry.

    This context manager yields a tuple of two paths, to be used for the QC report images:
    1. the path to `background_img.png`, and
    2. the path to `overlay_img.png`.

    The body of the `with` block should create these two image files.
    When the `with` block exits, the QC report is updated, with proper file synchronization.
    """
    if plane not in ['Axial', 'Sagittal']:
        raise ValueError(f'Invalid plane: {plane!r}')

    logger.info('\n*** Generating Quality Control (QC) html report ***')
    mod_date = datetime.datetime.now()

    # Not quite following BIDS convention, we derive the value of
    # dataset, subject, contrast from the input file by splitting it as
    # {dataset}/{subject}/{contrast}/filename
    if dataset is None:
        dataset = path_input.parent.parent.parent.name
    if subject is None:
        subject = path_input.parent.parent.name
    contrast = path_input.parent.name
    timestamp = mod_date.strftime('%Y_%m_%d_%H%M%S.%f')

    # Make sure the image directory exists
    path_img = path_qc / dataset / subject / contrast / command / timestamp
    path_img.mkdir(parents=True, exist_ok=True)

    # Ask the caller to generate the image files for the entry
    imgs_to_generate = {
        'path_background_img': path_img / 'background_img.png',
        'path_overlay_img':  path_img / 'overlay_img.png',
    }
    yield imgs_to_generate
    # Double-check that the images were generated during the 'with:' block
    for img_type, path in imgs_to_generate.items():
        if not path.exists():
            raise FileNotFoundError(f"Required QC image '{img_type}' was not found at the expected path: '{path}')")

    # We lock `index.html` so that we halt any other processes *before*
    # they have a chance to generate or read any .json files. This ensures
    # that the last process to write to `index.html` has read in all of the
    # available .json files, preventing:
    # https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3701#discussion_r816300380
    path_index_html = path_qc / 'index.html'
    with locked_file(path_index_html) as file_index_html:

        # Create a json file for the new QC report entry
        path_json = path_qc / '_json'
        path_json.mkdir(parents=True, exist_ok=True)
        path_result = path_json / f'qc_{timestamp}.json'
        with path_result.open('w') as file_result:
            json.dump({
                'cwd': str(Path.cwd()),
                'cmdline': cmdline,
                'command': command,
                'sct_version': __version__,
                'dataset': dataset,
                'subject': subject,
                'contrast': contrast,
                'fname_in': path_input.name,
                'plane': plane,
                'background_img': str(imgs_to_generate['path_background_img'].relative_to(path_qc)),
                'overlay_img': str(imgs_to_generate['path_overlay_img'].relative_to(path_qc)),
                'moddate': mod_date.strftime("%Y-%m-%d %H:%M:%S"),
                'qc': '',
            }, file_result, indent=1)

        # Collect all existing QC report entries
        json_data = []
        for path in sorted(path_json.glob('*.json')):
            with path.open() as file:
                json_data.append(json.load(file))

        # Insert the QC report entries into index.html
        path_assets = importlib.resources.files(spinalcordtoolbox.reports) / 'assets'
        template = string.Template((path_assets / 'index.html').read_text(encoding='utf-8'))
        # Empty the HTML file before writing, to make sure there's no leftover junk at the end
        file_index_html.truncate()
        file_index_html.write(template.substitute(sct_json_data=json.dumps(json_data)))

        # Copy any missing assets
        path_qc.mkdir(parents=True, exist_ok=True)
        update_files(path_assets / '_assets', path_qc)

    logger.info('Successfully generated the QC results in %s', str(path_result))
    display_open(file=str(path_index_html), message="To see the results in a browser")


def update_files(resource: Traversable, destination: Path):
    """
    Make sure that an up-to-date copy of `resource` exists at `destination`,
    by creating or updating files and directories recursively.
    """
    path = destination / resource.name
    if resource.is_dir():
        path.mkdir(exist_ok=True)
        for sub_resource in resource.iterdir():
            copy_missing(sub_resource, path)
    elif resource.is_file():
        new_content = resource.read_bytes()
        old_content = path.read_bytes() if path.is_file() else None
        if new_content != old_content:
            path.write_bytes(new_content)
    else:
        # Some weird kind of resource? Ignore it
        pass


def sct_register_multimodal(
    fname_input: str,
    fname_output: str,
    fname_seg: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
):
    """
    Generate a QC report for sct_register_multimodal.
    """
    command = 'sct_register_multimodal'
    cmdline = [command]
    cmdline.extend(argv)

    # Axial orientation, switch between two input images
    with create_qc_entry(
        path_input=Path(fname_input).absolute(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
    ) as imgs_to_generate:

        # Resample images slice by slice
        p_resample = 0.6
        logger.info('Resample images to %fx%f mm', p_resample, p_resample)
        img_input = Image(fname_input).change_orientation('SAL')
        img_input = resample_nib(
            image=img_input,
            new_size=[img_input.dim[4], p_resample, p_resample],
            new_size_type='mm',
            interpolation='spline',
        )
        img_output = resample_nib(
            image=Image(fname_output).change_orientation('SAL'),
            image_dest=img_input,
            interpolation='spline',
        )
        img_seg = resample_nib(
            image=Image(fname_seg).change_orientation('SAL'),
            image_dest=img_input,
            interpolation='linear',
        )
        img_seg.data = (img_seg.data > 0.5) * 1

        # Each slice is centered on the segmentation
        logger.info('Find the center of each slice')
        centers = np.array([center_of_mass(slice) for slice in img_seg.data])
        inf_nan_fill(centers[:, 0])
        inf_nan_fill(centers[:, 1])

        # Generate the first QC report image
        img = equalize_histogram(mosaic(img_input, centers))

        # For QC reports, axial mosaics will often have smaller height than width
        # (e.g. WxH = 20x3 slice images). So, we want to reduce the fig height to match this.
        # `size_fig` is in inches. So, dpi=300 --> 1500px, dpi=100 --> 500px, etc.
        size_fig = [5, 5 * img.shape[0] / img.shape[1]]

        fig = Figure()
        fig.set_size_inches(*size_fig, forward=True)
        FigureCanvas(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_background_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=300)

        # Generate the second QC report image
        img = equalize_histogram(mosaic(img_output, centers))
        fig = Figure()
        fig.set_size_inches(*size_fig, forward=True)
        FigureCanvas(fig)
        ax = fig.add_axes((0, 0, 1, 1), label='0')
        ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_overlay_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=300)


def inf_nan_fill(A: np.ndarray):
    """
    Interpolate inf and NaN values with neighboring values in a 1D array, in-place.
    If only inf and NaNs, fills the array with zeros.
    """
    valid = np.isfinite(A)
    invalid = ~valid
    if np.all(invalid):
        A.fill(0)
    elif np.any(invalid):
        A[invalid] = np.interp(
            np.nonzero(invalid)[0],
            np.nonzero(valid)[0],
            A[valid])


def mosaic(img: Image, centers: np.ndarray, radius: tuple[int, int] = (15, 15)):
    """
    Arrange the slices of `img` into a grid of images.

    Each slice is centered at the approximate coordinates given in `centers`,
    and cropped to `radius` pixels in each direction (horizontal, vertical).

    If `img` has N slices, then `centers` should have shape (N, 2).
    """
    # Fit as many slices as possible in each row of 600 pixels
    num_col = math.floor(600 / (2*radius[0]))

    # Center and crop each axial slice
    cropped = []
    for center, slice in zip(centers.astype(int), img.data):
        # Add a margin before cropping, in case the center is too close to the edge
        cropped.append(np.pad(slice, radius)[
            center[0]:center[0] + 2*radius[0],
            center[1]:center[1] + 2*radius[1],
        ])

    # Pad the list with empty arrays, to get complete rows of num_col
    empty = np.zeros((2*radius[0], 2*radius[1]))
    cropped.extend([empty] * (-len(cropped) % num_col))

    # Arrange the images into a grid
    return np.block([cropped[i:i+num_col] for i in range(0, len(cropped), num_col)])


def add_orientation_labels(ax: Axes):
    """
    Add orientation labels (A, P, L, R) to a figure, yellow with a black outline.
    """
    for x, y, letter in [
        (12, 6, 'A'),
        (12, 28, 'P'),
        (0, 18, 'L'),
        (24, 18, 'R'),
    ]:
        ax.text(x, y, letter, color='yellow', size=4).set_path_effects([
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal(),
        ])


def equalize_histogram(img: np.ndarray):
    """
    Perform histogram equalization using CLAHE.

    Notes:
    - Image value range is preserved
    - Workaround for adapthist artifact by padding (#1664)
    """
    winsize = 16
    min_, max_ = img.min(), img.max()
    b = (np.float32(img) - min_) / (max_ - min_)
    b[b >= 1] = 1  # 1+eps numerical error may happen (#1691)

    h, w = b.shape
    h1 = (h + (winsize - 1)) // winsize * winsize
    w1 = (w + (winsize - 1)) // winsize * winsize
    if h != h1 or w != w1:
        b1 = np.zeros((h1, w1), dtype=b.dtype)
        b1[:h, :w] = b
        b = b1
    c = skimage.exposure.equalize_adapthist(b, kernel_size=(winsize, winsize))
    if h != h1 or w != w1:
        c = c[:h, :w]

    return np.array(c * (max_ - min_) + min_, dtype=img.dtype)
