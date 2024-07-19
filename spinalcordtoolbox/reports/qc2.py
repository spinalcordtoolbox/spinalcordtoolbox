"""
Quality Control report generator

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

from contextlib import contextmanager
import datetime
from hashlib import md5
import importlib.resources
from importlib.abc import Traversable
import json
import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import center_of_mass
import skimage.exposure

from spinalcordtoolbox.image import Image
import spinalcordtoolbox.reports
from spinalcordtoolbox.reports.assets._assets.py import refresh_qc_entries
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils.shell import display_open
from spinalcordtoolbox.utils.sys import __version__, list2cmdline, LazyLoader
from spinalcordtoolbox.utils.fs import mutex

mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")
mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_axes = LazyLoader("mpl_axes", globals(), "matplotlib.axes")
mpl_cm = LazyLoader("mpl_cm", globals(), "matplotlib.cm")
mpl_colors = LazyLoader("mpl_colors", globals(), "matplotlib.colors")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_patheffects = LazyLoader("mpl_patheffects", globals(), "matplotlib.patheffects")

logger = logging.getLogger(__name__)


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

    # Use mutex to ensure that we're only generating shared QC assets using one process at a time
    realpath = path_qc.resolve()
    with mutex(f"sct_qc-{realpath.name}-{md5(str(realpath).encode('utf-8')).hexdigest()}"):
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

        # Copy any missing QC assets
        path_assets = importlib.resources.files(spinalcordtoolbox.reports) / 'assets'
        path_qc.mkdir(parents=True, exist_ok=True)
        update_files(path_assets / '_assets', path_qc)

        # Inject the JSON QC entries into the index.html file
        path_index_html = refresh_qc_entries.main(path_qc)

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
            update_files(sub_resource, path)
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

        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
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
        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1), label='0')
        ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_overlay_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=300)


def sct_deepseg(
    fname_input: str,
    fname_seg: str,
    fname_seg2: Optional[str],
    species: str,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
):
    """
    Generate a QC report for sct_deepseg, with varied colormaps depending on the type of segmentation.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    command = 'sct_deepseg'
    cmdline = [command]
    cmdline.extend(argv)

    # Axial orientation, switch between one anat image and 1-2 seg images
    with create_qc_entry(
        path_input=Path(fname_input).absolute(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
    ) as imgs_to_generate:
        # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
        #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.
        # Resample images slice by slice
        p_resample = {
            'human': 0.6, 'mouse': 0.1,
        }[species]
        logger.info('Resample images to %fx%f vox', p_resample, p_resample)
        img_input = Image(fname_input).change_orientation('SAL')
        img_input = resample_nib(
            image=img_input,
            new_size=[img_input.dim[4], p_resample, p_resample],
            new_size_type='mm',
            interpolation='spline',
        )
        img_seg_sc = resample_nib(
            image=Image(fname_seg).change_orientation('SAL'),
            image_dest=img_input,
            interpolation='linear',
        )
        img_seg_lesion = resample_nib(
            image=Image(fname_seg2).change_orientation('SAL'),
            image_dest=img_input,
            interpolation='linear',
        ) if fname_seg2 else None

        # Each slice is centered on the segmentation
        logger.info('Find the center of each slice')
        centers = np.array([center_of_mass(slice) for slice in img_seg_sc.data])
        inf_nan_fill(centers[:, 0])
        inf_nan_fill(centers[:, 1])

        # Generate the first QC report image
        img = equalize_histogram(mosaic(img_input, centers))

        # For QC reports, axial mosaics will often have smaller height than width
        # (e.g. WxH = 20x3 slice images). So, we want to reduce the fig height to match this.
        # `size_fig` is in inches. So, dpi=300 --> 1500px, dpi=100 --> 500px, etc.
        size_fig = [5, 5 * img.shape[0] / img.shape[1]]

        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
        add_orientation_labels(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_background_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=300)

        # Generate the second QC report image
        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        colormaps = [mpl_colors.ListedColormap(["#ff0000"]),  # Red for first image
                     mpl_colors.ListedColormap(["#00ffff"])]  # Cyan for second
        for i, image in enumerate([img_seg_sc, img_seg_lesion]):
            if not image:
                continue
            img = mosaic(image, centers)
            img = np.ma.masked_less_equal(img, 0)
            img.set_fill_value(0)
            ax.imshow(img,
                      cmap=colormaps[i],
                      norm=mpl_colors.Normalize(vmin=0.5, vmax=1),
                      # img==1 -> opaque, but soft regions -> more transparent as value decreases
                      alpha=(img / img.max()),  # scale to [0, 1]
                      interpolation='none',
                      aspect=1.0)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        img_path = str(imgs_to_generate['path_overlay_img'])
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path, format='png', transparent=True, dpi=300)


def sct_analyze_lesion(
    fname_input: str,
    tissue_bridges_plotting_data,
    fname_sc,
    measure_pd,
    angles,
    argv: Sequence[str],
    path_qc: str,
    dataset: Optional[str],
    subject: Optional[str],
):
    """
    Generate a QC report for sct_analyze_lesion, specifically highlighting the tissue bridge widths.
    """
    command = 'sct_analyze_lesion'
    cmdline = [command]
    cmdline.extend(argv)

    # Axial orientation, switch between one anat image and 1-2 seg images
    with create_qc_entry(
        path_input=Path(fname_input).absolute(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=dataset,
        subject=subject,
    ) as imgs_to_generate:
        # Load the spinal cord segmentation mask
        im_sc = Image(fname_sc)
        im_sc.change_orientation("RPI")
        im_sc_data = im_sc.data

        # Load the labeled lesion mask
        im_lesion = Image(fname_input)
        im_lesion.change_orientation("RPI")
        im_lesion_data = im_lesion.data
        p_lst = im_lesion.dim[4:7]
        label_lst = [label for label in np.unique(im_lesion.data) if label]

        # Get the total number of lesions; this will represent the number of rows in the figure. For example, if we have
        # 2 lesions, we will have two rows. One row per lesion.
        num_of_lesions = len(label_lst)
        # Get the minimum sagittal slice with lesion. For example, if a lesion cover slices 7,8,9, get 7
        min_sag_slice = min([min(tissue_bridges_plotting_data[lesion_idx].keys()) for
                             lesion_idx in tissue_bridges_plotting_data])[0]
        # Get the maximum sagittal slice with lesion. For example, if a lesion cover slices 7,8,9, get 9
        max_sag_slice = max([max(tissue_bridges_plotting_data[lesion_idx].keys()) for
                             lesion_idx in tissue_bridges_plotting_data])[0]
        # Get the number of slices containing the lesion. For example, if a lesion cover slices 7,8,9, get 3
        num_of_sag_slices = max_sag_slice - min_sag_slice + 1

        # Get the midsagittal slice of the spinal cord
        # Note: as the midsagittal slice is the same for all lesions, we can pick the first lesion ([0]) to get it
        mid_sagittal_sc_slice = measure_pd.loc[0, 'midsagittal_spinal_cord_slice']

        #  Create a figure
        #  The figure has one row per lesion and one column per sagittal slice containing the lesion
        fig, axes = mpl_plt.subplots(num_of_lesions,
                                     num_of_sag_slices,
                                     figsize=(num_of_sag_slices * 5, num_of_lesions * 5))

        # Force axes to be a 2-dimensional array (to avoid indexing issues if we have only a single lesion or a single
        # sagittal slice)
        axes = np.atleast_2d(axes)

        # Loop across lesions
        for idx_row, lesion_label in enumerate(label_lst):
            # NOTE: As the 'label_lesion()' function has been called at the beginning of the script, im_lesion_data is
            # now "labeled" meaning that different lesions have different values, e.g., 1, 2, 3
            # As we are looping across lesions, we get the lesion mask for the current lesion label
            im_label_data_cur = im_lesion_data == lesion_label
            # Restrict the lesion mask to the spinal cord mask (from anatomical level, it does not make sense to have
            # lesion outside the spinal cord mask)
            boolean_mask = (im_sc_data.data != 0)  # Nonzero -> True | Zero -> False; we use this in case of soft SC
            im_label_data_cur = im_label_data_cur * boolean_mask

            # Loop across sagittal slices
            for idx_col, sagittal_slice in enumerate(range(min_sag_slice, max_sag_slice + 1)):
                # Get spinal cord and lesion masks data for the selected sagittal slice
                slice_sc = im_sc_data[sagittal_slice]
                slice_lesion = im_label_data_cur[sagittal_slice]

                # Plot spinal cord and lesion masks
                axes[idx_row, idx_col].imshow(np.swapaxes(slice_sc, 1, 0),
                                              cmap='gray', origin="lower")
                axes[idx_row, idx_col].imshow(np.swapaxes(slice_lesion, 1, 0),
                                              cmap='jet', alpha=0.8, interpolation='nearest', origin="lower")

                # Add title for each column
                if idx_row == 0:
                    if sagittal_slice == mid_sagittal_sc_slice:
                        axes[idx_row, idx_col].set_title(f'Midsagittal slice\n'
                                                         f'Sagittal slice #{sagittal_slice}')
                    else:
                        axes[idx_row, idx_col].set_title(f'Sagittal slice #{sagittal_slice}')

                # Add title to each row, (i.e., y-axis)
                if idx_col == 0:
                    axes[idx_row, idx_col].set_ylabel(f'Lesion #{lesion_label}\n'
                                                      f'Inferior-Superior')
                else:
                    axes[idx_row, idx_col].set_ylabel('Inferior-Superior')

                # Add x-axis label
                axes[idx_row, idx_col].set_xlabel('Posterior-Anterior')

                # Crop the slice around the lesion (to zoom in)
                axes[idx_row, idx_col].set_xlim(np.min(np.where(im_label_data_cur)[1]) - 20,
                                                np.max(np.where(im_label_data_cur)[1]) + 20)
                axes[idx_row, idx_col].set_ylim(np.min(np.where(im_label_data_cur)[2]) - 20,
                                                np.max(np.where(im_label_data_cur)[2]) + 20)

                # --------------------------------------
                # Add text for tissue bridges
                # --------------------------------------
                # Check if the [idx_row][sagittal_slice, 'dorsal'] key exists in the tissue_bridges_plotting_data
                # If the key exists, it means that we have tissue bridges for the current lesion and sagittal slice
                if (idx_row in tissue_bridges_plotting_data) and \
                        ((sagittal_slice, 'dorsal') in tissue_bridges_plotting_data[idx_row]):
                    _plot_dorsal_tissue_bridge(tissue_bridges_plotting_data, idx_row, idx_col, sagittal_slice, p_lst,
                                               angles, axes)
                    _plot_ventral_tissue_bridge(tissue_bridges_plotting_data, idx_row, idx_col, sagittal_slice, p_lst,
                                                angles, axes)

        # tight layout
        mpl_plt.tight_layout()
        for fname in imgs_to_generate.values():
            mpl_plt.savefig(fname)
        mpl_plt.close()


def _plot_dorsal_tissue_bridge(tissue_bridges_plotting_data, idx_row, idx_col, sagittal_slice, p_lst, angles, axes):
    """
    Add text for the dorsal bridge
    """
    # lesion_indices_dorsal_bridge: ndarray of the indices of the lesion mask
    # Note: we use [0] because .values returns a numpy array
    lesion_indices_dorsal_bridge = \
        tissue_bridges_plotting_data[idx_row][sagittal_slice, 'dorsal']['lesion_indices'].values[0]
    # Get the posterior/dorsal tip of the lesion (the first element in the lesion_indices)
    x_dorsal = lesion_indices_dorsal_bridge[0]
    # dorsal_bridge_width: the width of the tissue bridge
    dorsal_bridge_width = float(
        tissue_bridges_plotting_data[idx_row][sagittal_slice, 'dorsal']['dorsal_bridge_width'])
    # y_dorsal: the axial slice with the minimum dorsal tissue bridge width
    y_dorsal = int(tissue_bridges_plotting_data[idx_row][sagittal_slice, 'dorsal']['axial_slice'])

    # Add text with the width of the tissue in mm above each bridge
    dorsal_bridge_width_mm = float(dorsal_bridge_width * p_lst[1]) * np.cos(angles[y_dorsal])
    axes[idx_row, idx_col].text(x_dorsal - 3,
                                y_dorsal,
                                f'Dorsal bridge\n{np.round(dorsal_bridge_width_mm, 2)} mm',
                                color='red', fontsize=12, ha='right', va='bottom')


def _plot_ventral_tissue_bridge(tissue_bridges_plotting_data, idx_row, idx_col, sagittal_slice, p_lst, angles, axes):
    """
    Add text for the ventral bridge
    """
    # lesion_indices_ventral_bridge: ndarray of the indices of the lesion mask
    # Note: we use [0] because .values returns a numpy array
    lesion_indices_ventral_bridge = \
        tissue_bridges_plotting_data[idx_row][sagittal_slice, 'ventral']['lesion_indices'].values[0]
    # Get the anterior/ventral tip of the lesion (the last element in the lesion_indices)
    x_ventral = lesion_indices_ventral_bridge[-1]
    # ventral_bridge_width: the width of the tissue bridge
    ventral_bridge_width = float(
        tissue_bridges_plotting_data[idx_row][sagittal_slice, 'ventral']['ventral_bridge_width'])
    # y_ventral: the axial slice with the minimum dorsal tissue bridge width
    y_ventral = int(tissue_bridges_plotting_data[idx_row][sagittal_slice, 'ventral']['axial_slice'])

    # Add text with the width of the tissue in mm above each bridge
    ventral_bridge_width_mm = float(ventral_bridge_width * p_lst[1]) * np.cos(angles[y_ventral])
    axes[idx_row, idx_col].text(x_ventral + 3,
                                y_ventral,
                                f'Ventral bridge\n{np.round(ventral_bridge_width_mm, 2)} mm',
                                color='red', fontsize=12, ha='left', va='bottom')


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


def add_orientation_labels(ax: mpl_axes.Axes):
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
            mpl_patheffects.Stroke(linewidth=1, foreground='black'),
            mpl_patheffects.Normal(),
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
