# :)

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
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
import sys
import numpy as np
import skimage.exposure
from spinalcordtoolbox.utils.shell import SCTArgumentParser
from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image, rpi_slice_to_orig_orientation
import spinalcordtoolbox.reports
from spinalcordtoolbox.reports.assets.py import refresh_qc_entries
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils.shell import display_open
from spinalcordtoolbox.utils.sys import __version__, init_sct, list2cmdline, LazyLoader, __sct_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import mutex
from spinalcordtoolbox.aggregate_slicewise import Metric


pd = LazyLoader("pd", globals(), "pandas")
mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")
mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_axes = LazyLoader("mpl_axes", globals(), "matplotlib.axes")
mpl_cm = LazyLoader("mpl_cm", globals(), "matplotlib.cm")
mpl_colors = LazyLoader("mpl_colors", globals(), "matplotlib.colors")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_patheffects = LazyLoader("mpl_patheffects", globals(), "matplotlib.patheffects")
mpl_collections = LazyLoader("mpl_collections", globals(), "matplotlib.collections")
nib_orientations = LazyLoader("nib_orientations", globals(), "nibabel.orientations")
ndimage = LazyLoader("ndimage", globals(), "scipy.ndimage")

logger = logging.getLogger(__name__)

TARGET_WIDTH_PIXL = 1060
DPI = 300
TARGET_WIDTH_INCH = TARGET_WIDTH_PIXL / DPI

def reorient_grid(
    shape: tuple[int, int, int],  # image data array shape
    affine: np.ndarray,  # image affine, of shape (4, 4)
    orientation: str,  # for example, "SAL"
) -> tuple[
    tuple[int, int, int],  # reoriented shape
    np.ndarray,  # reoriented affine, of shape (4, 4)
]:
    """
    Reorient a nifti sampling grid to the given orientation.
    """
    opposite = dict(zip("RASLPI", "LPIRAS"))  # SCT and nibabel use opposite conventions
    ornt = nib_orientations.ornt_transform(
        nib_orientations.io_orientation(affine),
        nib_orientations.axcodes2ornt(tuple(opposite[char] for char in orientation))
    )

    new_shape = [0, 0, 0]
    for axis, new_axis in enumerate(ornt[:, 0]):
        new_shape[int(new_axis)] = shape[axis]

    new_affine = np.dot(affine, nib_orientations.inv_ornt_aff(ornt, shape))

    return tuple(new_shape), new_affine


def rescale_grid(
    shape: tuple[int, int, int],  # image data array shape
    affine: np.ndarray,  # image affine, of shape (4, 4)
    resolution: list[float | None],  # desired resolution for each axis
) -> tuple[
    tuple[int, int, int],  # rescaled shape
    np.ndarray,  # rescaled affine, of shape (4, 4)
]:
    """
    Rescale a nifti sampling grid to the given resolution for each axis.

    No rescaling is done for an axis if the requested resolution is `None`.
    """
    # Make a copy of the input before modifying it
    shape = list(shape)
    affine = np.array(affine)

    old_resolution = np.linalg.norm(affine[:, :3], axis=0)

    for axis, (old_res, new_res) in enumerate(zip(old_resolution, resolution)):
        if new_res is not None:
            affine[:, axis] *= new_res / old_res
            # Round up to make sure we cover the original FOV entirely
            shape[axis] = math.ceil(shape[axis] * old_res / new_res)

    return tuple(shape), affine


@dataclass(frozen=True)
class SlicingSpec:
    """
    Specs for extracting one or more 2D slices in the same way from multiple 3D images.

    affine:
        A numpy array of shape=(4, 4) and dtype=np.float64, as returned by
        `nib.Nifti1Header.get_best_affine()`. This matrix gives the conversion
        between voxel coordinates (i, j, k), which are non-negative integers,
        and physical coordinates (x, y, z), which are floating point numbers:

            [ x ]   [ * * * | * ]   [ i ]
            [ y ]   [ * * * | * ]   [ j ]
            [ z ] = [ * * * | * ] @ [ k ]
            [---]   [-------+---]   [---]
            [ 1 ]   [ 0 0 0 | 1 ]   [ 1 ]

        The first column gives the step size and direction (in physical space)
        for taking a 1-voxel step from i to i+1. The second and third columns
        are the same for j to j+1, or k to k+1. The last column gives the
        physical coordinates of the (0, 0, 0) voxel. The last row is fixed.

        The 2D slices to be extracted are all parallel to the (j, k) plane.

    offsets:
        An ordered dict of slice labels and slice offsets for the 2D slices
        to be extracted. Each slice offset is a numpy array of shape=(3,)
        and dtype=np.float64, and gives voxel coordinates (i, j, k) for a
        corner of a slice to be extracted. Even though they are in voxel space,
        these coordinates are allowed to be negative or fractional.

        The insertion order of items in the dict controls the position of each
        slice in the generated mosaic.

    shape:
        The (height, width) size of each 2D slice to be extracted. The height
        is measured in voxels in the j direction, and the width is measured
        in voxels in the k direction.

    axis_labels:
        Two pairs of strings used to display orientation labels:

            ((up == -j direction, down == +j direction),
             (left == -k direction, right == +k direction))

        Used for the top-left tile in a mosaic.
    """
    # ----------------------------------------------------------------
    # Dataclass fields
    # ----------------------------------------------------------------
    # The default @dataclass __init__ method, used internally by the
    # SlicingSpec factory functions, always assigns a value to these.

    affine: np.ndarray  # shape=(4, 4)
    offsets: dict[str, np.ndarray]  # each shape=(3,), insertion order is important
    shape: tuple[int, int]  # 2D shape of all output slices
    axis_labels: tuple[tuple[str, str], tuple[str, str]]  # ((top, bottom), (left, right))

    # ----------------------------------------------------------------
    # Factory functions
    # ----------------------------------------------------------------
    # This is the normal way for users of this class to create
    # instances. They are @staticmethod, so they are called as:
    # SlicingSpec.full_axial(...) etc.

    @staticmethod
    def full_axial(img: Image, p_resample: float) -> 'SlicingSpec':
        """
        A slicing spec to extract full axial slices.

        The field of view is given by `img`.
        The number of slices is the same as `img` in the S-I axis,
        but the other two axes are resampled to `p_resample` mm.
        The slice labels correspond to the original orientation of `img`,
        but the output slices are always ordered and oriented in "SAL".
        """
        return SlicingSpec.full_oriented(img, "SAL", p_resample)

    @staticmethod
    def full_sagittal(img: Image, p_resample: float) -> 'SlicingSpec':
        """
        A slicing spec to extract full sagittal slices.

        The field of view is given by `img`.
        The number of slices is the same as `img` in the R-L axis,
        but the other two axes are resampled to `p_resample` mm.
        The slice labels correspond to the original orientation of `img`,
        but the output slices are always ordered and oriented in "RSP".
        """
        return SlicingSpec.full_oriented(img, "RSP", p_resample)

    @staticmethod
    def full_oriented(img: Image, orientation: str, p_resample: float) -> 'SlicingSpec':
        """
        A slicing spec to extract full slices in a given orientation.

        The meaning of the `orientation` argument is that:
        - The first axis is perpendicular to the 2D slices produced.
        - The second axis is the "up" direction for the 2D slices produced.
        - The third axis is the "left" direction for the 2D slices produced.

        Each output slice is resampled to `p_resample` mm.

        The slice labels match the original orientation of `img`,
        but their ordering matches the requested orientation.
        """
        # For axis labels.
        opposite = dict(zip("RASLPI", "LPIRAS"))

        # Reorient and rescale.
        shape, affine = img.data.shape, img.affine
        shape, affine = reorient_grid(shape, affine, orientation)
        shape, affine = rescale_grid(shape, affine, [None, p_resample, p_resample])

        # Split up the 3D image shape into a number of slices and a 2D slice shape.
        num_slices, shape = shape[0], shape[1:3]

        # The slice labels in the original orientation of the input image.
        slice_labels = [str(i) for i in range(num_slices)]
        if opposite[orientation[0]] in img.orientation:
            slice_labels.reverse()

        return SlicingSpec(
            affine=affine,
            offsets={
                label: np.array([i, 0, 0], dtype=np.float64)
                for i, label in enumerate(slice_labels)
            },
            shape=shape,
            axis_labels=tuple(
                (letter, opposite[letter])
                for letter in orientation[1:3]
            ),
        )

    # ----------------------------------------------------------------
    # Transformer methods
    # ----------------------------------------------------------------
    # These methods can be called on an existing SlicingSpec, to
    # produce a new SlicingSpec. For example, they can be used to
    # center and crop slices around a segmentation, or to drop some
    # slices entirely.

    def center_patches(
        self: 'SlicingSpec', seg: Image, shape: tuple[int, int] = (30, 30)
    ) -> 'SlicingSpec':
        """
        A re-centered and cropped slicing spec, meant for axial views.

        Each slice is individually centered around the center of mass of the
        given segmentation before being cropped (in both axes).
        """
        # Recenter each slice around its center of mass, and interpolate missing
        # slices. We use a single (N, 3) array so that we can inf_nan_fill.
        offsets = np.array(list(self.offsets.values()))
        slices_seg = self.get_slices(seg, order=1)
        for offset, slice_seg in zip(offsets, slices_seg.values()):
            offset[1:3] += ndimage.center_of_mass(slice_seg)
        inf_nan_fill(offsets[:, 1])
        inf_nan_fill(offsets[:, 2])

        # Take into account the size of the cropping rectangle.
        offsets[:, 1:3] -= (np.array(shape) - 1) / 2

        return SlicingSpec(
            affine=self.affine,
            offsets=dict(zip(self.offsets.keys(), offsets)),
            shape=shape,  # new shape
            axis_labels=self.axis_labels,
        )

    def center_columns(self: 'SlicingSpec', seg: Image) -> 'SlicingSpec':
        """
        A re-centered and cropped slicing spec, meant for sagittal views.

        Each slice is individually centered in the last axis around the center
        of mass of the given segmentation. It is also cropped in the last axis
        to the larger of:
        - 110% of the maximum width of the segmentation, or
        - 50% of the total image.

        Any slices that are more than 2 slices away from the segmentation are
        also dropped.
        """
        # Check which slices contain the segmentation, and compute their width.
        slices_seg = self.get_slices(seg, order=1)
        width = {}
        for i, slice_seg in enumerate(slices_seg.values()):
            [columns] = slice_seg.any(axis=0).nonzero()
            if columns.size != 0:
                width[i] = int(columns.max()) - int(columns.min()) + 1
        if not width:
            raise ValueError("The mask image is empty. Cannot crop using an empty mask. Check the input (e.g. '-qc-seg').")

        # Compute the target slice width.
        shape = (
            self.shape[0],  # the full image height
            math.ceil(max(
                1.1 * max(width.values()),  # 110% of the segmentation width
                0.5 * self.shape[1],  # 50% of the full image width
            )),
        )

        # Drop slices that are far from the segmentation.
        slice_min = max(0, min(width.keys()) - 2)
        slice_max = min(len(slices_seg) - 1, max(width.keys()) + 2)
        offsets_dict = dict(list(self.offsets.items())[slice_min:slice_max+1])

        # Re-center each remaining slice.
        for slice_label, offset in offsets_dict.items():
            offset[2] += ndimage.center_of_mass(slices_seg[slice_label])[1]

        # Take into account the cropping width.
        offsets_array = np.array(list(offsets_dict.values()))
        inf_nan_fill(offsets_array[:, 2])
        offsets_array[:, 2] -= (shape[1] - 1) / 2
        offsets_dict = dict(zip(offsets_dict.keys(), offsets_array))

        return SlicingSpec(
            affine=self.affine,
            offsets=offsets_dict,  # new offsets
            shape=shape,  # new shape
            axis_labels=self.axis_labels,
        )

    def composite_sagittal(self: 'SlicingSpec', seg: Image) -> 'SlicingSpec':
        """
        Construct a composite 2D sagittal image out of axial slices (one
        row from each axial slice), similarly to `sct_flatten_sagittal`.
        (For each axial slice, the center of mass is used to identify the
        midsagittal row. The individual rows can then be stitched together
        to build a composite midsagittal 2D image.)
        """
        # Recenter each slice around its center of mass, and interpolate missing
        # slices. We use a single (N, 3) array so that we can inf_nan_fill.
        offsets = np.array(list(self.offsets.values()))
        slices_seg = self.get_slices(seg, order=1)
        for offset, slice_seg in zip(offsets, slices_seg.values()):
            offset[2] += ndimage.center_of_mass(slice_seg)[1]
        inf_nan_fill(offsets[:, 2])

        return SlicingSpec(
            affine=self.affine,
            offsets=dict(zip(self.offsets.keys(), offsets)),
            shape=(self.shape[0], 1),  # crop to a single line
            axis_labels=self.axis_labels,
        )

    # ----------------------------------------------------------------
    # Consumer methods
    # ----------------------------------------------------------------
    # These methods are called on an existing SlicingSpec to produce
    # some other result.

    def get_slices(
        self: 'SlicingSpec',
        img: Image,  # 3D image to sample from
        order: int,  # interpolation order, can be [0, 1, 2, 3, 4, 5]
    ) -> dict[str, np.ndarray]:  # shape=self.shape
        """
        Get several slices of the same shape from a single image by resampling.

        The offsets are measured in output voxels, and can be fractional.
        """
        # To fill regions outside of the image with zeros,
        # while avoiding sharp cutoffs around the edges.
        mode = "grid-constant"

        voxel_to_voxel = np.linalg.inv(img.affine).dot(self.affine)
        matrix = voxel_to_voxel[:3, :3]
        origin = voxel_to_voxel[:3, 3]

        if order > 1:
            # Doing this prefilter step outside of the loop is crucial for performance
            data_filtered = ndimage.spline_filter(img.data, order=order, mode=mode)
        else:
            data_filtered = img.data.astype(np.float64)
        return {
            slice_label: ndimage.affine_transform(
                data_filtered,
                matrix=matrix,
                offset=(origin + matrix.dot(offset)),
                output_shape=(1, *self.shape),
                order=order,
                mode=mode,
                prefilter=False,
            )[0]
            for slice_label, offset in self.offsets.items()
        }

    def get_mosaic(self: 'SlicingSpec', **kwargs) -> 'Mosaic':
        """
        Get a mosaic that can hold the result of `self.get_slices`.
        """
        return Mosaic(self.shape, list(self.offsets.keys()), self.axis_labels, **kwargs)


class Mosaic:
    """
    Convenience methods for rendering several 2D slices into a grid of images.

    The caller is expected to manipulate `self.fig` and/or `self.ax` directly
    before calling `self.save(path)`.
    """
    canvas: np.ndarray  # The data array for the entire mosaic.
    tiles: dict[str, tuple[slice, slice]]  # The canvas coordinates for each slice label.
    axis_labels: tuple[tuple[str, str], tuple[str, str]]  # ((top, bottom), (left, right))

    fig: 'mpl_figure.Figure'
    ax: 'mpl_axes.Axes'

    def __init__(
        self,
        shape: tuple[int, int],  # (height, width) of each tile
        slice_labels: list[str],
        axis_labels: tuple[tuple[str, str], tuple[str, str]],  # ((top, bottom), (left, right))
        *,  # keyword-only arguments after this
        rect: tuple[float, float, float, float] = (0, 0, 1, 1),  # passed to Figure.add_axes()
        scale: float = 2.5,
    ):
        # Make a canvas of the right size to hold all the tiles.
        num_col = max(math.floor(TARGET_WIDTH_PIXL / scale / shape[1]), 1)
        num_row = math.ceil(len(slice_labels) / num_col)
        self.canvas = np.zeros((num_row * shape[0], num_col * shape[1]))

        # Compute the canvas coordinates for each slice label.
        self.tiles = dict(zip(slice_labels, [
            (slice(x, x + shape[0]), slice(y, y + shape[1]))
            for x in range(0, self.canvas.shape[0], shape[0])
            for y in range(0, self.canvas.shape[1], shape[1])
        ], strict=False))

        # Save the axis labels for displaying later.
        self.axis_labels = axis_labels

        # Initialize the Figure and the Axes for rendering the mosaic later.
        self.fig = mpl_figure.Figure(
            # figsize is (width, height) but canvas shape is (height, width)
            figsize=(self.canvas.shape[1] * scale / DPI, self.canvas.shape[0] * scale / DPI),
            dpi=DPI,
        )
        mpl_backend_agg.FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes(rect)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

    def insert_slices(self, slices: dict[str, np.ndarray]):
        """
        Insert 2D slices into the canvas based on their slice labels.

        Note that this doesn't render them to `self.ax`, so that the caller
        can post-process `self.canvas` (for example, with `equalize_histogram`)
        and has full control of `self.ax.imshow(self.canvas, ...)`.
        """
        for slice_label, data in slices.items():
            self.canvas[self.tiles[slice_label]] = data

    def add_labels(self):
        """
        Add axis labels and slice labels to the mosaic figure.

        Axis labels are added to the first tile, and slice labels are added to
        the other tiles.
        """
        # Rendering options for all the labels.
        text_args = dict(
            color='yellow',
            fontsize=4,
            path_effects=[
                mpl_patheffects.Stroke(linewidth=1, foreground='black'),
                mpl_patheffects.Normal(),
            ],
        )
        for tile_num, (slice_label, coords) in enumerate(self.tiles.items()):
            # Get real coordinates from the `slice` objects.
            top, bottom, _ = coords[0].indices(self.canvas.shape[0])
            left, right, _ = coords[1].indices(self.canvas.shape[1])
            if tile_num == 0:
                # Axis labels for the first tile.
                ((top_label, bottom_label), (left_label, right_label)) = self.axis_labels
                self.ax.text((left + right) / 2, top, top_label, ha='center', va='top', **text_args)
                self.ax.text((left + right) / 2, bottom, bottom_label, ha='center', va='bottom', **text_args)
                self.ax.text(left, (top + bottom) / 2, left_label, ha='left', va='center', **text_args)
                self.ax.text(right, (top + bottom) / 2, right_label, ha='right', va='center', **text_args)
            else:
                # Slice labels for the other tiles.
                self.ax.text(left, top, slice_label, ha='left', va='top', **text_args)

    def save(self, path: Path):
        """Save the final figure."""
        self.fig.savefig(str(path), format='png', transparent=True)


def sct_deepseb(
    fname_input,
    fname_seg,
    cmin,
    cmax,
    cbar,
    fname_output,
    with_seg = False,
):
    """
    asdas
    """
    img_input = Image(fname_input)
    img_seg = Image(fname_seg)

    # Center the axial slices and create mosaice
    slicing_spec = SlicingSpec.full_axial(img_input, p_resample=None).center_patches(img_seg, shape=(30, 30))

    mosaic = slicing_spec.get_mosaic()
    mosaic.insert_slices(slicing_spec.get_slices(img_input, order=2)) # 2 for quadratic interpolation, 0 is for nearest-neighbor
    mosaic.ax.imshow(mosaic.canvas, cmap=cbar, vmin=cmin, vmax=cmax, interpolation='none')
    if with_seg:
        mosaic.insert_slices(slicing_spec.get_slices(img_seg, order=0)) #  1 is for linear
        mosaic.ax.imshow(mosaic.canvas, cmap = mpl_colors.ListedColormap(["#000000", "#00ffff"]), alpha = 1.0, norm = mpl_colors.Normalize(vmin=0, vmax=1), interpolation='none')
    mosaic.add_labels()
    # This shows the input image with colormap and vmin/vmax custom
    mosaic.save(fname_output)
    

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


def plot_outlines(img: np.ndarray, ax: 'mpl_axes.Axes', **kwargs):
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

def get_parser():
    parser = SCTArgumentParser(
        description="TChasfasdasdasdasdasdasdasdasdsaasfasdfasdfasdfasdfasdfasdfasdfasdfasdfasfasdfasdfsadf",
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        help="Input image. Example: `infile.nii.gz`",
    )
    mandatory.add_argument(
        '-s',
        help="Segmentation image. Example: `infile_seg.nii.gz`",
    )
    mandatory.add_argument(
        '-o',
        help="Output image filename (end in .png)",
    )
    mandatory.add_argument(
        '-cmin',
        type=float,
        help="Colorbar min.",
    )
    mandatory.add_argument(
        '-cmax',
        type=float,
        help=" Colorbar max",
    )
    mandatory.add_argument(
        '-cbar',
        type=str,
        help="Colorbar type.",
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser



def main(argv: Sequence[str]):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    sct_deepseb(
        fname_input=arguments.i,
        fname_seg=arguments.s,
        cmin=arguments.cmin,
        cmax=arguments.cmax,
        cbar=arguments.cbar,
        fname_output=arguments.o,
        )
    sct_deepseb(
        fname_input=arguments.i,
        fname_seg=arguments.s,
        cmin=arguments.cmin,
        cmax=arguments.cmax,
        cbar=arguments.cbar,
        fname_output=f"{arguments.o.removesuffix('.png')}_with_seg.png",
        with_seg=True,
        )
    

if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])