"""
Quality Control report generator

Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

from contextlib import contextmanager
import datetime
from hashlib import md5
import importlib.resources
import itertools as it
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import center_of_mass
import skimage.exposure

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image, rpi_slice_to_orig_orientation
import spinalcordtoolbox.reports
from spinalcordtoolbox.reports.assets.py import refresh_qc_entries
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


logger = logging.getLogger(__name__)

# Clarify some constants that will control the width of the output images
# (the height is automatically adjusted based on the aspect ratio of the image)
# Notes:
#   - This shouldn't ever need to be changed, since "target width" is related
#     to the design of the QC report interface. If things need to be scaled,
#     it should be done to the array itself before it is written to the .
#   - `matplotlib` uses inches/DPI to define the canvas. But, given we want
#     a fixed output size, we can choose some arbitrary values for both.
#     Presumably, choosing a different DPI value (and thus a different canvas
#     size in inches) wouldn't change the output image at all. (Is this true?)
TARGET_WIDTH_PIXL = 1500
DPI = 300
TARGET_WIDTH_INCH = TARGET_WIDTH_PIXL / DPI


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
                'path': str(Path.cwd()),
                'cmdline': cmdline,
                'command': command,
                'sctVersion': __version__,
                'dataset': dataset,
                'subject': subject,
                'contrast': contrast,
                'inputFile': path_input.name,
                'plane': plane,
                'backgroundImage': str(imgs_to_generate['path_background_img'].relative_to(path_qc)),
                'overlayImage': str(imgs_to_generate['path_overlay_img'].relative_to(path_qc)),
                'date': mod_date.strftime("%Y-%m-%d %H:%M:%S"),
                'rank': '',
                'qc': '',
            }, file_result, indent=1)

        # Copy any missing QC assets
        path_qc.mkdir(parents=True, exist_ok=True)
        path_assets = importlib.resources.files(spinalcordtoolbox.reports) / 'assets'
        shutil.copytree(path_assets, path_qc, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '__init__.py'))
        # Inject the JSON QC entries into the index.html file
        path_index_html = refresh_qc_entries.main(path_qc)

    logger.info('Successfully generated the QC results in %s', str(path_result))
    display_open(file=str(path_index_html), message="To see the results in a browser")


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

        # Fix the width to a specific size, and vary the height based on how many rows there are.
        size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img.shape[0] / img.shape[1]]

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
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

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
        fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


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
    fname_qc_seg: Optional[str],
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
        if "rootlets" in argv:
            sct_deepseg_spinal_rootlets(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species,
                radius=(23, 23), base_scaling=2.5)  # standard upscale to see rootlets detail
        elif "totalspineseg" in argv:
            sct_deepseg_spinal_rootlets(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species,
                radius=(40, 40), base_scaling=0.5)  # downscale to get "big picture" view of all slices
        # Non-rootlets, axial/sagittal DeepSeg QC report
        elif plane == 'Axial':
            sct_deepseg_axial(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species, fname_qc_seg)
        else:
            assert plane == 'Sagittal', (f"`plane` must be either 'Axial' "
                                         f"or 'Sagittal', but got {plane}")
            sct_deepseg_sagittal(
                imgs_to_generate, fname_input, fname_seg, fname_seg2, species, fname_qc_seg)


def sct_deepseg_axial(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    fname_qc_seg: Optional[str],
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
    img_qc_seg = Image(fname_qc_seg).change_orientation('SAL') if fname_qc_seg else None

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
    img_qc_seg = resample_nib(
        image=img_qc_seg,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_qc_seg else None

    # Using -qc-seg mask if available, we remove slices which are empty (except for the first 3 and last 3 slices just around the segmented slices)
    for img_to_crop in [img_input, img_seg_sc, img_seg_lesion, img_qc_seg]:
        if fname_qc_seg and img_to_crop:
            crop_with_mask(img_to_crop, img_qc_seg, pad=3)

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    # Use the -qc-seg mask if available, otherwise use the spinal cord mask
    img_centers = img_qc_seg if fname_qc_seg else img_seg_sc
    centers = np.array([center_of_mass(slice) for slice in img_centers.data])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # If -qc-seg is available, use it to generate the radius
    radius = get_max_axial_radius(img_qc_seg) if fname_qc_seg else (15, 15)

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius))

    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img.shape[0] / img.shape[1]]

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
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

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
        img = mosaic(image, centers, radius)
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
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def sct_deepseg_spinal_rootlets(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    radius: Sequence[int],
    base_scaling: float = 2.5,  # scale up mosaic slices to make them more readable by default
    outline: bool = True
):
    """
    Generate a QC report for `sct_deepseg rootlets`.

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
    #   So, to ensure that the QC is still readable, we scale up whenever the p_ratio is < 1
    scale = max((1 / ratio) for ratio in p_ratio)  # e.g. 0.8mm human => p_ratio == 0.6/0.8 == 0.75; scale == 1/p_ratio == 1/0.75 == 1.33
    # - Note: `mosaic()` already has a base scaling factor of 2.5 (to help make the QC readable).
    #          Since resolution-based scaling would overwrite this, we need to preserve the base scaling factor.
    # - Note2: For `totalspineseg`, we actually _don't_ want to upscale by 2.5, so that's why `base_scaling` has
    #          been parametrized, allowing per-QC customization.
    scale *= base_scaling
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

    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * (img.shape[0] / img.shape[1]) * aspect]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=aspect)
    add_orientation_labels(ax, radius=tuple(r for r in radius))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_background_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

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

        # only display outlines and segmentation labels if scale is large enough to accommodate
        # (in practice, this saves them from being added to the tiny totalspineseg QC)
        if scale >= 1.0:
            add_segmentation_labels(ax, img, colors=colormaps[i].colors, radius=tuple(r for r in radius))
            if outline:
                # linewidth 0.5 is too thick, 0.25 is too thin
                plot_outlines(img, ax=ax, facecolor='none', edgecolor='black', linewidth=0.3)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


def sct_deepseg_sagittal(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    fname_qc_seg: Optional[str],
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
    img_qc_seg = Image(fname_qc_seg).change_orientation('RSP') if fname_qc_seg else None

    # Resample images slice by slice
    R_L_resolution = img_input.dim[4]
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
    img_qc_seg = resample_nib(
        image=img_qc_seg,
        image_dest=img_input,
        interpolation='linear',
    ) if fname_qc_seg else None

    # Using -qc-seg mask if available, we remove slices which are empty (except for the first 2 and last 2 slices just around the segmented slices)
    # If -qc-seg mask isn't available, but image would create a mosaic that's too large, keep only the center 30 sagittal slices (using segmnetation as a reference)
    for img_to_crop in [img_input, img_seg_sc, img_seg_lesion, img_qc_seg]:
        if img_to_crop is None:
            continue  # don't crop missing images
        if fname_qc_seg:
            crop_with_mask(img_to_crop, img_qc_seg, pad=2)
        elif img_input.dim[0] > 30:
            crop_with_mask(img_to_crop, img_seg_sc, max_slices=30)
            if img_to_crop == img_input:  # display a message only once
                logger.warning("Source image is too large to display in a sagittal mosaic. Applying automatic cropping around segmentation.\n"
                               "Please consider using `sct_deepseg -qc-seg` option to customize the crop. You can use `sct_create_mask` to create a suitable mask to pass to "
                               "`-qc-seg`. If this message still occurs after cropping, please consider resampling your image to a lower resolution using `sct_resample`.")

    logger.info('Find the center of each slice')
    # Use the -qc-seg mask if available to get crop radius (as well as the center of mass) for each slice
    if fname_qc_seg:
        radius = get_max_sagittal_radius(img_qc_seg)
        centers = np.array([center_of_mass(slice) for slice in img_qc_seg.data])
        inf_nan_fill(centers[:, 0])
        inf_nan_fill(centers[:, 1])
    # otherwise, if -qc-seg isn't provided, display the full sagittal slice and use the center of the uncropped image
    else:
        radius = (img_input.dim[1] // 2, img_input.dim[2] // 2)
        centers = np.array([radius] * img_input.data.shape[0])

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius=radius))

    # Fix the width to a specific size, and vary the height based on how many rows there are.
    size_fig = [TARGET_WIDTH_INCH, TARGET_WIDTH_INCH * img.shape[0] / img.shape[1]]

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
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)

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
    fig.savefig(img_path, format='png', transparent=True, dpi=DPI)


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
        # TODO: This report breaks the assumption that the width is fixed in pixel size
        #       The figure could be very wide horizontally, which wouldn't work with the "new QC"
        #       See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4887
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


def mosaic(img: Image, centers: np.ndarray, radius: tuple[int, int] = (15, 15), scale: float = 2.5):
    """
    Arrange the slices of `img` into a grid of images.

    Each slice is centered at the approximate coordinates given in `centers`,
    and cropped to `radius` pixels in each direction (horizontal, vertical).

    If `img` has N slices, then `centers` should have shape (N, 2).
    """
    # Note: This function used to hardcode a max row width of 600 pixels
    #       In practice, because the canvas size is fixed to 1500 pixels, this
    #       resulted in a permanent upscaling by 2.5x when saving the image.
    #       To make things clearer, we now use a variable.
    max_row_width = TARGET_WIDTH_PIXL / scale

    # Fit as many slices as possible in each row
    num_col = math.floor(max_row_width / (2*radius[0]))

    # Center and crop each axial slice
    cropped = []
    for center, slice in zip(centers.astype(int), img.data):
        # If the `center` coordinate is close to the edge, then move it away from the edge to capture more of the image
        # In other words, make sure the `center` coordinate is at least `radius` pixels away from the edge
        for i in [0, 1]:
            center[i] = min(slice.shape[i] - radius[i], center[i])  # Check far edge first
            center[i] = max(radius[i],                  center[i])  # Then check 0 edge last
        # Add a margin before cropping, in case the center is still too close to one of the edges
        cropped.append(np.pad(slice, [[r] for r in radius])[
            center[0]:center[0] + 2*radius[0],
            center[1]:center[1] + 2*radius[1],
        ])

    # Pad the list with empty arrays, to get complete rows of num_col
    empty = np.zeros((2*radius[0], 2*radius[1]))
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


def plot_outlines(img: np.ndarray, ax: mpl_axes.Axes, **kwargs):
    """
    Draw the outlines of every equal-value area of a 2D Numpy array with Matplotlib.

    kwargs are forwarded to `matplotlib.collections.PolyCollection` for styling.
    """
    # To see where each (row, column) index shows up in the graphic
    # (left, right, top, bottom), see:
    # https://matplotlib.org/stable/users/explain/artists/imshow_extent.html#default-extent

    # Add a frame of zeros around the image to account for cells on the border
    padded = np.pad(img, 1)

    # First we compute the endpoints of every horizontal segment of the outline

    # edge[r, c] is True when there should be a visible horizontal outline
    # between img[r, c+1] and img[r+1, c+1]
    edge = (padded[:-1, :] != padded[1:, :])
    # run[r, c] is True when a run of horizontal edges starts or stops at the
    # upper left corner of img[r, c]
    run = (edge[:, :-1] != edge[:, 1:])
    # Get the coordinates from the big boolean array. They are sorted so that
    # the horizontal segments start at even vertices and stop at the next (odd)
    # vertex: v0--v1, v2--v3, v4--v5, ...
    vertices = sorted(map(tuple, np.argwhere(run)))
    # Given a vertex, we want to quickly travel to its horizontal neighbour
    # i^1 == i+1 when i is even (v0->v1, v2->v3, ...)
    # i^1 == i-1 when i is odd (v1->v0, v3->v2, ...)
    horizontal_segment = {v: vertices[i ^ 1] for i, v in enumerate(vertices)}

    # Second, we compute endpoints for the vertical segments

    # edge[r, c] is True when there should be a visible vertical outline
    # between img[r+1, c] and img[r+1, c+1]
    edge = (padded[:, :-1] != padded[:, 1:])
    # run[r, c] is True when a run of vertical edges starts or stops at the
    # upper left corner of img[r, c]
    run = (edge[:-1, :] != edge[1:, :])
    # Get the coordinates from the big boolean array. They are sorted so that
    # the vertical segments start at even vertices and stop at the next (odd)
    # vertex: v0--v1, v2--v3, v4--v5, ...
    vertices = sorted(map(tuple, np.argwhere(run)), key=lambda v: v[::-1])
    # Given a vertex, we want to quickly travel to its vertical neighbour
    # i^1 == i+1 when i is even (v0->v1, v2->v3, ...)
    # i^1 == i-1 when i is odd (v1->v0, v3->v2, ...)
    vertical_segment = {v: vertices[i ^ 1] for i, v in enumerate(vertices)}

    # Now we need to collect the horizontal and vertical segments into a list
    # of polygons. We may need some open-ended polygons, which start and end
    # at vertices which are in the middle of a perpendicular segment. And we
    # may need some closed polygons, which loop back to their starting vertex.

    # The open-ended polygons:
    open_polygons = []
    open_vertices = set(horizontal_segment).symmetric_difference(vertical_segment)
    while open_vertices:
        polygon = []
        vertex = open_vertices.pop()
        # Build the polygon by listing its vertices in order. We remove
        # vertices from horizontal_segment and vertical_segment as we trace
        # over them with the polygon.
        while True:
            polygon.append(vertex)
            if vertex in horizontal_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = horizontal_segment.pop(vertex)
                del horizontal_segment[vertex]
            elif vertex in vertical_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = vertical_segment.pop(vertex)
                del vertical_segment[vertex]
            else:
                # We have reached the end of the current open-ended polygon.
                open_vertices.remove(vertex)
                # Convert (row, column) coordinates to (x, y)
                open_polygons.append(np.array(polygon)[:, ::-1] - 0.5)
                break

    # The closed polygons:
    closed_polygons = []
    while horizontal_segment:
        polygon = []
        # Remove both endpoints of an arbitrary segment.
        vertex, other = horizontal_segment.popitem()
        del horizontal_segment[other]
        while True:
            polygon.append(vertex)
            if vertex in horizontal_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = horizontal_segment.pop(vertex)
                del horizontal_segment[vertex]
            elif vertex in vertical_segment:
                # Move to the other endpoint of the segment, and make sure to
                # remove both endpoints from the dictionary.
                vertex = vertical_segment.pop(vertex)
                del vertical_segment[vertex]
            else:
                # We have reached the end of the current closed polygon.
                # Convert (row, column) coordinates to (x, y)
                closed_polygons.append(np.array(polygon)[:, ::-1] - 0.5)
                break
    assert not vertical_segment

    # Draw the outline
    ax.add_collection(mpl_collections.PolyCollection(
        open_polygons, closed=False, **kwargs))
    ax.add_collection(mpl_collections.PolyCollection(
        closed_polygons, closed=True, **kwargs))


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


def crop_with_mask(img_to_crop, img_ref, pad=3, max_slices=None):
    """
    Crop array along a specific axis based on nonzero slices in the reference image.

    Use `pad` if you want to pad around the mask (no matter how big the mask is).

    Use `max_slices` to pad only until that amount of slices is reached (overrides `pad`). If
    the segmentation spans more slices than `max_slices`, then no padding will occur. Instead,
    all slices containing the segmentation will be used (to preserve the segmentation).
    """
    # QC images are reoriented to SAL (axial) or RSP (sagittal) such that axis=0 is always the slice index
    axis = 0
    # get extents of segmentation used for cropping
    first_slice = min(np.where(img_ref.data)[axis])
    last_slice = max(np.where(img_ref.data)[axis])
    # if `max_slices` is specified, then override `pad`
    if max_slices is not None:
        # use `max(0, ...)` to avoid cropping the segmentation if it would exceed `max_slices`
        pad_total = max(0, max_slices - (last_slice - first_slice + 1))
        l_pad = math.floor(pad_total / 2)
        r_pad = math.ceil(pad_total / 2)
    # otherwise, use the provided value as-is
    else:
        l_pad = r_pad = pad
    # pad (but make sure the index slices are within the image bounds)
    start_slice = max(first_slice - l_pad, 0)
    stop_slice = min(last_slice + r_pad, img_ref.data.shape[axis] - 1)
    # crop the image at the specified axis
    cropper = ImageCropper(img_in=img_to_crop)
    cropper.bbox = BoundingBox(xmin=start_slice, xmax=stop_slice,
                               ymin=0, ymax=img_to_crop.data.shape[1]-1,
                               zmin=0, zmax=img_to_crop.data.shape[2]-1)
    img_cropped = cropper.crop()
    # since `ImageCropper` returns a copy of the image, we need to update the original image
    # we could instead just return the new copy, but that would require refactoring the qc
    # function to no longer iterate over the images in-place, which would be a larger change
    img_to_crop.data = img_cropped.data
    img_to_crop.hdr = img_cropped.hdr


def get_max_axial_radius(img):
    """
    Determine the maximum slicewise width/height of the nonzero voxels in img.

    Input images should be SAL, such that each SI axial slice is composed of [1] -> AP and [2] -> LR.
    """
    # In Axial plane, the radius is the maximum width/height of the spinal cord mask dilated by 20% or 15, whichever is larger.
    dilation = 1.2
    default = 15
    heights = [np.max(np.where(slice)[0]) - np.min(np.where(slice)[0]) if np.sum(slice) > 0 else 0 for slice in img.data]
    widths = [np.max(np.where(slice)[1]) - np.min(np.where(slice)[1]) if np.sum(slice) > 0 else 0 for slice in img.data]
    heights = [(h * dilation)//2 for h in heights]
    widths = [(w * dilation)//2 for w in widths]
    return max(default, max(heights)), max(default, max(widths))


def get_max_sagittal_radius(img):
    """
    Determine the maximum slicewise width/height of the nonzero voxels for sagittal images.

    Input images should be RSP, such that each LR sagittal slice is composed of [1] -> SI and [2] -> PA.
    """
    # In Sagittal plane, the radius is the maximum width of the spinal cord mask dilated by 20% or 1/2 of the image width, whichever is larger.
    # The height is always the entirety of the image height (for example, to view possible lesions in the brain stem)
    widths = [np.max(np.where(slice)[1]) - np.min(np.where(slice)[1]) if np.sum(slice) > 0 else 0 for slice in img.data]
    widths = [w//2 + 0.1*w//2 for w in widths]
    height = np.floor(img.data.shape[2]/2).astype(int)
    return height, max(np.floor(img.data.shape[1]/4).astype(int), int(max(widths)))
