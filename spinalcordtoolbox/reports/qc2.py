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
import itertools as it
import json
import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import center_of_mass
import skimage.exposure

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.image import Image, rpi_slice_to_orig_orientation
import spinalcordtoolbox.reports
from spinalcordtoolbox.reports.assets._assets.py import refresh_qc_entries
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils.shell import display_open
from spinalcordtoolbox.utils.sys import __version__, list2cmdline, LazyLoader
from spinalcordtoolbox.utils.fs import mutex

pd = LazyLoader("pd", globals(), "pandas")
mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")
mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_axes = LazyLoader("mpl_axes", globals(), "matplotlib.axes")
mpl_cm = LazyLoader("mpl_cm", globals(), "matplotlib.cm")
mpl_colors = LazyLoader("mpl_colors", globals(), "matplotlib.colors")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_patheffects = LazyLoader("mpl_patheffects", globals(), "matplotlib.patheffects")
mpl_collections = LazyLoader("mpl_collections", globals(), "matplotlib.collections")
mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")


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
    image_extension: str = 'png',
):
    """
    Generate a new QC report entry.

    This context manager yields a dict of two paths, to be used for the QC report images:
    'path_background_img': the path to `background_img.{image_extension}`, and
    'path_overlay_img': the path to `overlay_img.{image_extension}`.

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
        'path_background_img': path_img / f'background_img.{image_extension}',
        'path_overlay_img':  path_img / f'overlay_img.{image_extension}',
    }
    yield imgs_to_generate
    # Double-check that the images were generated during the 'with:' block
    for img_type, path in imgs_to_generate.items():
        if not path.exists():
            raise FileNotFoundError(f"Required QC image '{img_type}' was not found at the expected path: '{path}'")

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
    if resource.name in ['__pycache__']:
        return  # skip this one
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
    plane: Optional[str],
):
    """
    Generate a QC report for sct_deepseg, based on which task was used.
    """
    command = 'sct_deepseg'
    cmdline = [command]
    cmdline.extend(argv)

    with create_qc_entry(
        path_input=Path(fname_input).absolute(),
        path_qc=Path(path_qc),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane=plane,
        dataset=dataset,
        subject=subject,
    ) as imgs_to_generate:
        # Custom QC to handle multiclass segmentation outside the spinal cord
        if "seg_spinal_rootlets_t2w" in argv:
            sct_deepseg_spinal_rootlets_t2w(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species,
                radius=(23, 23))
        elif "totalspineseg" in argv:
            sct_deepseg_spinal_rootlets_t2w(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species,
                radius=(40, 40), outline=False)
        # Non-rootlets, axial/sagittal DeepSeg QC report
        elif plane == 'Axial':
            sct_deepseg_axial(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species)
        else:
            assert plane == 'Sagittal', (f"`plane` must be either 'Axial' "
                                         f"or 'Sagittal', but got {plane}")
            sct_deepseg_sagittal(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species)


def sct_deepseg_axial(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
):
    """
    Generate a QC report for sct_deepseg, with varied colormaps depending on the type of segmentation.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.
    # Resample images slice by slice
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]
    img_input = Image(fname_input).change_orientation('SAL')
    img_seg_sc = Image(fname_seg_sc).change_orientation('SAL')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('SAL') if fname_seg_lesion else None

    # Resample images slice by slice
    logger.info('Resample images to %fx%f vox', p_resample, p_resample)
    img_input = resample_nib(
        image=img_input,
        new_size=[img_input.dim[4], p_resample, p_resample],
        new_size_type='mm',
        interpolation='spline',
    )
    img_seg_sc = resample_nib(
        image=img_seg_sc,
        image_dest=img_input,
        interpolation='linear',
    )
    img_seg_sc.data = (img_seg_sc.data > 0.5) * 1
    img_seg_lesion = resample_nib(
        image=img_seg_lesion,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_seg_lesion else None
    if fname_seg_lesion:
        img_seg_lesion.data = (img_seg_lesion.data > 0.5) * 1

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
                  alpha=1.0,
                  interpolation='none',
                  aspect=1.0)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)


def sct_deepseg_spinal_rootlets_t2w(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    radius: Sequence[int],
    outline: bool = True
):
    """
    Generate a QC report for `sct_deepseg -task seg_spinal_rootlets_t2w`.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.

    # Load the input images
    img_input = Image(fname_input).change_orientation('SAL')
    img_seg_sc = Image(fname_seg_sc).change_orientation('SAL')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('SAL') if fname_seg_lesion else None

    # - Normally, we would apply isotropic resampling to the image to a specific mm resolution (based on the species).
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]
    #   Choosing a fixed resolution allows us to crop the image around the spinal cord at a fixed radius that matches the chosen resolution,
    #   while also handling anisotropic images (so that they display correctly on an isotropic grid).
    # - However, we cannot apply resampling here because rootlets labels are often small (~1vox wide), and so resampling might
    #   corrupt the labels and cause them to be displayed unfaithfully.
    # - So, instead of resampling the image to fit the default crop radius, we scale the crop radius to suit the original resolution.
    p_original = (img_seg_sc.dim[5], img_seg_sc.dim[6])  # Image may be anisotropic, so use both resolutions (H,W)
    p_ratio = tuple(p_resample / p for p in p_original)
    radius = tuple(int(r * p) for r, p in zip(radius, p_ratio))
    # - One problem with this, however, is that if the crop radius ends up being smaller than the default, the QC will in turn be smaller as well.
    #   So, to ensure that the QC is still readable, we scale up by an integer factor whenever the p_ratio is < 1
    scale = int(math.ceil(1 / max(p_ratio)))  # e.g. 0.8mm human => p_ratio == 0.6/0.8 == 0.75; scale == 1/p_ratio == 1/0.75 == 1.33 => 2x scale
    # - One other problem is that for anisotropic images, the aspect ratio won't be 1:1 between width/height.
    #   So, we use `aspect` to adjust the image via imshow, and `radius` to know where to place the text in x/y coords
    aspect = p_ratio[1] / p_ratio[0]

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    centerline_param = ParamCenterline(algo_fitting="optic", contrast="t2")
    img_centerline, _, _, _ = get_centerline(img_input, param=centerline_param)
    centers = np.array([center_of_mass(slice) for slice in img_centerline.data])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius, scale))

    # For QC reports, axial mosaics will often have smaller height than width
    # (e.g. WxH = 20x3 slice images). So, we want to reduce the fig height to match this.
    # `size_fig` is in inches. So, dpi=300 --> 1500px, dpi=100 --> 500px, etc.
    size_fig = [5, 5 * (img.shape[0] / img.shape[1]) * aspect]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=aspect)
    add_orientation_labels(ax, radius=tuple(r*scale for r in radius))
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
    # get available labels
    img = np.rint(np.ma.masked_where(img_seg_sc.data < 1, img_seg_sc.data))
    labels = np.unique(img[np.where(~img.mask)]).astype(int)
    colormaps = [mpl_colors.ListedColormap(assign_label_colors_by_groups(labels))]
    for i, image in enumerate([img_seg_sc, img_seg_lesion]):
        if not image:
            continue
        img = mosaic(image, centers, radius, scale)
        img = np.ma.masked_less_equal(img, 0)
        img.set_fill_value(0)
        ax.imshow(img,
                  cmap=colormaps[i],
                  norm=None,
                  alpha=1.0,
                  interpolation='none',
                  aspect=aspect)
        if outline:
            # linewidth 0.5 is too thick, 0.25 is too thin
            plot_outlines(img, ax=ax, facecolor='none', edgecolor='black', linewidth=0.3)
        add_segmentation_labels(ax, img, colors=colormaps[i].colors, radius=tuple(r*scale for r in radius))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)


def sct_deepseg_sagittal(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
):
    """
    Generate a QC report for sct_deepseg, with varied colormaps depending on the type of segmentation.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.
    # Resample images slice by slice
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]
    img_input = Image(fname_input).change_orientation('RSP')
    img_seg_sc = Image(fname_seg_sc).change_orientation('RSP')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('RSP') if fname_seg_lesion else None

    # if the image has more then 30 slices then we resample it to 30 slices on the R-L direction
    if img_input.dim[0] > 30:
        R_L_resolution = img_input.dim[4] * img_input.dim[0] / 30
    else:
        R_L_resolution = img_input.dim[4]

    # Resample images slice by slice
    logger.info('Resample images to %fx%fx%f vox', R_L_resolution, p_resample, p_resample)
    img_input = resample_nib(
        image=img_input,
        new_size=[R_L_resolution, p_resample, p_resample],
        new_size_type='mm',
        interpolation='spline',
    )
    img_seg_sc = resample_nib(
        image=img_seg_sc,
        image_dest=img_input,
        interpolation='linear',
    )
    img_seg_sc.data = (img_seg_sc.data > 0.5) * 1
    img_seg_lesion = resample_nib(
        image=img_seg_lesion,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_seg_lesion else None
    if fname_seg_lesion:
        img_seg_lesion.data = (img_seg_lesion.data > 0.5) * 1

    # The radius is set to display the entire slice
    radius = (img_input.dim[1]//2, img_input.dim[2]//2)

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    centers = np.array([radius] * img_input.data.shape[0])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius=radius))

    # For QC reports, axial mosaics will often have smaller height than width
    # (e.g. WxH = 20x3 slice images). So, we want to reduce the fig height to match this.
    # `size_fig` is in inches. So, dpi=300 --> 1500px, dpi=100 --> 500px, etc.
    size_fig = [5, 5 * img.shape[0] / img.shape[1]]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=1.0)
    add_orientation_labels(ax, radius=radius, letters=['S', 'I', 'P', 'A'])
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
        img = mosaic(image, centers, radius=radius)
        img = np.ma.masked_less_equal(img, 0)
        img.set_fill_value(0)
        ax.imshow(img,
                  cmap=colormaps[i],
                  norm=mpl_colors.Normalize(vmin=0.5, vmax=1),
                  # img==1 -> opaque, but soft regions -> more transparent as value decreases
                  alpha=1.0,
                  interpolation='none',
                  aspect=1.0)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)


def sct_analyze_lesion(
    fname_input: str,
    fname_label: str,
    fname_sc: str,
    measure_pd: pd.DataFrame,
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
        plane='Sagittal',
        dataset=dataset,
        subject=subject,
    ) as imgs_to_generate:
        # Load the spinal cord segmentation mask
        im_sc = Image(fname_sc)
        im_sc.change_orientation("RPI")
        im_sc_data = im_sc.data

        # Load the labeled lesion mask
        im_lesion = Image(fname_label)
        # Store the original orientation of the lesion mask before reorienting it to RPI
        orig_orientation = im_lesion.orientation
        im_lesion.change_orientation("RPI")
        im_lesion_data = im_lesion.data
        label_lst = [label for label in np.unique(im_lesion.data) if label]

        # Get the total number of lesions; this will represent the number of rows in the figure. For example, if we have
        # 2 lesions, we will have two rows. One row per lesion.
        num_of_lesions = len(label_lst)
        # Get the sagittal lesion slices
        sagittal_lesion_slices = np.unique(np.where(im_lesion_data)[0])
        # Get the minimum sagittal slice with lesion. For example, if a lesion cover slices 7,8,9, get 7
        min_sag_slice = min(sagittal_lesion_slices)
        # Get the maximum sagittal slice with lesion. For example, if a lesion cover slices 7,8,9, get 9
        max_sag_slice = max(sagittal_lesion_slices)
        # Get the number of slices containing the lesion. For example, if a lesion cover slices 7,8,9, get 3
        num_of_sag_slices = max_sag_slice - min_sag_slice + 1

        #  Create a figure
        #  The figure has one row per lesion and one column per sagittal slice containing the lesion
        fig, axes = mpl_plt.subplots(num_of_lesions,
                                     num_of_sag_slices,
                                     figsize=(num_of_sag_slices * 5, num_of_lesions * 5))

        # Force axes to be a 2-dimensional array (to avoid indexing issues if we have only a single lesion or a single
        # sagittal slice)
        axes = np.asanyarray(axes).reshape((num_of_lesions, num_of_sag_slices))

        # Loop across lesions
        for idx_row, lesion_label in enumerate(label_lst):
            # NOTE: As the 'label_lesion()' function has been called at the beginning of the script, im_lesion_data is
            # now "labeled" meaning that different lesions have different values, e.g., 1, 2, 3
            # As we are looping across lesions, we get the lesion mask for the current lesion label
            im_label_data_cur = im_lesion_data == lesion_label
            # Restrict the lesion mask to the spinal cord mask (from anatomical level, it does not make sense to have
            # lesion outside the spinal cord mask)
            # Note for `im_sc_data != 0`: Nonzero -> True | Zero -> False; we use this in case of soft SC
            im_label_data_cur = np.logical_and(im_lesion_data == lesion_label, im_sc_data != 0)

            # Loop across sagittal slices
            for idx_col, sagittal_slice in enumerate(range(min_sag_slice, max_sag_slice + 1)):
                # Get spinal cord and lesion masks data for the selected sagittal slice
                slice_sc = im_sc_data[sagittal_slice]
                slice_lesion = im_label_data_cur[sagittal_slice]

                # Convert the sagittal slice to the original orientation
                # '0' because of the R-L direction (first in RPI)
                sagittal_slice = rpi_slice_to_orig_orientation(im_lesion.dim, orig_orientation,
                                                               sagittal_slice, 0)

                # Plot spinal cord and lesion masks
                axes[idx_row, idx_col].imshow(np.swapaxes(slice_sc, 1, 0),
                                              cmap='gray', origin="lower")
                axes[idx_row, idx_col].imshow(np.swapaxes(slice_lesion, 1, 0),
                                              cmap='jet', alpha=0.8, interpolation='nearest', origin="lower")

                # Add title for each column
                if idx_row == 0:
                    axes[idx_row, idx_col].set_title(f'Sagittal slice #{sagittal_slice}')

                # Add title to each row, (i.e., y-axis)
                if idx_col == 0:
                    axes[idx_row, idx_col].set_ylabel(f'Lesion #{lesion_label}\n'
                                                      f'Inferior-Superior')
                else:
                    axes[idx_row, idx_col].set_ylabel('Inferior-Superior')

                # Add x-axis label
                axes[idx_row, idx_col].set_xlabel('Anterior-Posterior')

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
                if idx_row < len(measure_pd):
                    col_name_dorsal = f"slice_{int(sagittal_slice)}_dorsal_bridge_width [mm]"
                    dorsal_bridge_width_mm = measure_pd[col_name_dorsal][idx_row]
                    if col_name_dorsal in measure_pd.columns and not pd.isna(dorsal_bridge_width_mm):
                        axes[idx_row, idx_col].text(min(np.where(slice_lesion)[0]) - 3,
                                                    min(np.where(slice_lesion)[1]),
                                                    f'Dorsal bridge\n{np.round(dorsal_bridge_width_mm, 2)} mm',
                                                    color='red', fontsize=12, ha='left', va='bottom')

                    col_name_ventral = f"slice_{int(sagittal_slice)}_ventral_bridge_width [mm]"
                    ventral_bridge_width_mm = measure_pd[col_name_ventral][idx_row]
                    if col_name_ventral in measure_pd.columns and not pd.isna(ventral_bridge_width_mm):
                        axes[idx_row, idx_col].text(max(np.where(slice_lesion)[0]) + 3,
                                                    min(np.where(slice_lesion)[1]),
                                                    f'Ventral bridge\n{np.round(ventral_bridge_width_mm, 2)} mm',
                                                    color='red', fontsize=12, ha='right', va='bottom')

                # Swap x-axis to anterior-posterior (from the current posterior-anterior), so that ventral tissue
                # bridges are on the left and dorsal tissue bridges on the right
                # Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4608#issue-2482923134
                axes[idx_row, idx_col].invert_xaxis()

        # tight layout
        mpl_plt.tight_layout()
        for fname in imgs_to_generate.values():
            mpl_plt.savefig(fname)


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


def mosaic(img: Image, centers: np.ndarray, radius: tuple[int, int] = (15, 15), scale: int = 1):
    """
    Arrange the slices of `img` into a grid of images.

    Each slice is centered at the approximate coordinates given in `centers`,
    and cropped to `radius` pixels in each direction (horizontal, vertical).

    If `img` has N slices, then `centers` should have shape (N, 2).
    """
    # Fit as many slices as possible in each row of 600 pixels
    num_col = math.floor(600 / (2*radius[0]*scale))

    # Center and crop each axial slice
    cropped = []
    for center, slice in zip(centers.astype(int), img.data):
        # Add a margin before cropping, in case the center is too close to the edge
        # Also, use Kronecker product to scale each block in multiples
        cropped.append(np.kron(np.pad(slice, [[r] for r in radius])[
            center[0]:center[0] + 2*radius[0],
            center[1]:center[1] + 2*radius[1],
        ], np.ones((scale, scale))))

    # Pad the list with empty arrays, to get complete rows of num_col
    empty = np.zeros((2*radius[0]*scale, 2*radius[1]*scale))
    cropped.extend([empty] * (-len(cropped) % num_col))

    # Arrange the images into a grid
    return np.block([cropped[i:i+num_col] for i in range(0, len(cropped), num_col)])


def add_orientation_labels(ax: mpl_axes.Axes, radius: tuple[int, int] = (15, 15), letters: tuple[str, str, str, str] = ('A', 'P', 'L', 'R')):
    """
    Add orientation labels (A, P, L, R) to a figure, yellow with a black outline.
    """
    # Ensure that letter locations are determined as a function of the bounding box. For a 15,15 radius (30x30):
    #    A                    [12,  6]
    # L     R   -->  [0, 17]            [24, 17]
    #    P                    [12, 28]
    for letter, x, y, in [
        (letters[0], radius[1] - 3,   6),
        (letters[1], radius[1] - 3,   radius[0]*2 - 2),
        (letters[2], 0,               radius[0] + 2),
        (letters[3], radius[1]*2 - 6, radius[0] + 2)
    ]:
        ax.text(x, y, letter, color='yellow', size=4).set_path_effects([
            mpl_patheffects.Stroke(linewidth=1, foreground='black'),
            mpl_patheffects.Normal(),
        ])


def add_segmentation_labels(ax: mpl_axes.Axes, seg_mosaic: np.ndarray, colors: list[str],
                            radius: tuple[int, int] = (15, 15)):
    """
    Add labels corresponding to the value of the segmentation for each slice in the mosaic.
    """
    # Fetch mosaic shape properties
    bbox = [2*radius[0], 2*radius[1]]
    grid_shape = [s // bb for s, bb in zip(seg_mosaic.shape, bbox)]
    # Fetch set of labels in the mosaic (including labels in between min/max)
    labels = [float(val) for val in range(int(np.unique(seg_mosaic).min()),
                                          int(np.unique(seg_mosaic).max())+1)]
    # Iterate over each sub-array in the mosaic
    for row in range(grid_shape[0]):
        for col in range(grid_shape[1]):
            # Fetch sub-array from mosaic
            extents = (slice(row*bbox[0], (row+1)*bbox[0]),
                       slice(col*bbox[1], (col+1)*bbox[1]))
            arr = seg_mosaic[extents]
            # Check for nonzero labels, then draw text for each label found
            labels_in_arr = [v for v in np.unique(arr) if v]
            for idx_pos, l_arr in enumerate(labels_in_arr, start=1):
                lr_shift = -4 * (len(str(int(l_arr))) - 1)
                y, x = (extents[0].stop - 6*idx_pos + 3,  # Shift each subsequent label up in case there are >1
                        extents[1].stop - 6 + lr_shift)   # Shift labels left if double/triple digit
                color = colors[0] if len(colors) == 1 else colors[labels.index(l_arr)]
                ax.text(x, y, str(int(l_arr)), color=color, size=4).set_path_effects([
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


def plot_outlines(bool_img: np.ndarray, ax: mpl_axes.Axes, **kwargs):
    """
    Draw the outlines of a 2D binary Numpy array with Matplotlib.

    kwargs are forwarded to `matplotlib.collections.PolyCollection` for styling.
    """
    # To see where each (row, column) index shows up in the graphic
    # (left, right, top, bottom), see:
    # https://matplotlib.org/stable/users/explain/artists/imshow_extent.html#default-extent

    # Add a frame of zeros around the image to account for cells on the border
    padded = np.pad(bool_img, 1)

    # horizontal_edge[r, c] is True when there should be a visible horizontal
    # outline between bool_img[r, c+1] and bool_img[r+1, c+1]
    horizontal_edge = (padded[:-1, :] != padded[1:, :])

    # corner[r, c] is True when a run of horizontal edges starts or stops at
    # the upper left corner of bool_img[r, c]
    corner = (horizontal_edge[:, :-1] != horizontal_edge[:, 1:])

    # List the corners in order: first by row (from top to bottom), then by
    # column within each row (from left to right). These will be all the
    # polygon vertices we need, and the two endpoints of each run of horizontal
    # edges appearing side by side in the list
    vertices = sorted(map(tuple, np.argwhere(corner)))

    # Given a vertex, we want to quickly travel to its horizontal neighbour
    # i^1 == i+1 when i is even (0->1, 2->3, ...)
    # i^1 == i-1 when i is odd (1->0, 3->2, ...)
    horizontal_move = {v: vertices[i ^ 1] for i, v in enumerate(vertices)}

    # We also want to quickly travel to vertical neighbours. We can do the same
    # thing as above, with a different sorting order: by column, then by row
    vertices.sort(key=lambda v: v[::-1])
    vertical_move = {v: vertices[i ^ 1] for i, v in enumerate(vertices)}

    # Now build each polygon by listing its vertices in order. We remove the
    # vertices from horizontal_move as we visit them
    polys = []
    while horizontal_move:
        polygon = []
        # Start at an arbitrary unvisited vertex
        v1 = next(iter(horizontal_move))
        while v1 in horizontal_move:
            # Keep alternating horizontal and vertical moves, until we revisit
            # the polygon's first vertex
            polygon.append(v1)
            v2 = horizontal_move.pop(v1)  # remove one endpoint
            polygon.append(v2)
            del horizontal_move[v2]  # remove the other endpoint
            v1 = vertical_move[v2]
        # Convert (row, column) coordinates to (x, y)
        polys.append(np.array(polygon)[:, ::-1] - 0.5)

    # Draw it
    ax.add_collection(mpl_collections.PolyCollection(polys, **kwargs))


def assign_label_colors_by_groups(labels):
    """
    This function handles label colors when labels may not be in a single continuous group:

        - Normal vertebral labels:  [2, 3, 4, 5, 6, ...]
        - Edge case, TotalSegmentator labels: [31, 32, 33, 200, 201, 217, 218, 219]

    We assume that the subgroups of labels (1: [31, 32, 33, ...], 2: [200, 201], 3: [217, 218, 219, ...])
    should each be assigned their own distinct colormap, as to group them semantically.
    """
    # Arrange colormaps for max contrast between colormaps, and max contrast between colors in colormaps
    # Put reds first because SC labels tend to be ~1, so they will usually be first in a list of labels
    distinct_colormaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']
    colormap_sampling = [0.25, 0.5, 0.75, 0.5]  # light -> medium -> dark -> medium -> (repeat)

    # Split labels into subgroups --> we split the groups wherever the difference between labels is > 1
    start_end = [0, len(labels)]
    for idx, (prev, curr) in enumerate(zip(labels, labels[1:]), start=1):
        if curr - prev > 1:
            start_end.insert(len(start_end) - 1, idx)
    label_groups = [labels[start:end] for start, end in zip(start_end, start_end[1:])]

    # Handle the usual case: A single continuous group (likely vertebral labels)
    # Contrast ratios against #000000 taken from https://webaim.org/resources/contrastchecker/
    labels_color = [
        "#ffffff",  # White       (21.00:1)
        "#F28C28",  # Orange      ( 8.55:1)
        "#0096FF",  # Blue        ( 6.80:1)
        "#ffee00",  # Yellow      (17.48:1)
        "#ff0000",  # Red         ( 5.25:1)
        "#50ff30",  # Green       (15.68:1)
        "#F749FD",  # Magenta     ( 7.32:1)
    ]
    if len(label_groups) == 1:
        # repeat high-contrast colors until we have enough to cover the range of labels
        n_colors = labels.max() - labels.min() + 1
        color_list = list(it.islice(it.cycle(labels_color), n_colors))

    # Handle the edge case: Multiple continuous groups
    else:
        # Initialize a list by repeating the color black (#000000) to fill in the gaps between colors.
        # We do this because matplotlib applies colormaps by scaling both the data and the colormap to [0, 1].
        # Without filling in the gaps between groups, the colormap would be scaled incorrectly relative to the data.
        # ((Note that, if done right, the #000000 color should never be assigned to our label values.))
        color_list = ['#000000'] * (labels.max() - labels.min() + 1)
        # Assign a colormap to each group of labels (while sampling the colormap at different points)
        for i, label_group in enumerate(label_groups):
            colormap = mpl_plt.get_cmap(distinct_colormaps[i % len(distinct_colormaps)])
            sampled_colors = [mpl_colors.to_hex(c) for c in [colormap(n) for n in colormap_sampling]]
            # Then, assign a color to each label within the group
            for j, label in enumerate(label_group):
                color_list[label - labels.min()] = sampled_colors[j % len(sampled_colors)]

    return color_list
